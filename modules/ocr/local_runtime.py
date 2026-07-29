from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from modules.ocr.selection import is_local_ocr_engine
from modules.utils.exceptions import LocalServiceSetupError, OperationCancelledError
from modules.utils.llama_cpp_runtime import (
    DEFAULT_LLAMA_CPP_IMAGE,
    DEFAULT_MANAGED_RUNTIME_STOP_TIMEOUT_SEC,
    inspect_llama_cpp_runtime,
    resolve_docker_compose_command,
)

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_HUNYUAN_N_GPU_LAYERS = "80"
LATE_START_STOP_GRACE_SEC = 3.0
LATE_START_STOP_POLL_SEC = 0.25
PADDLEOCR_IMAGE_REPOSITORY = (
    "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/"
    "paddlepaddle/paddleocr-genai-vllm-server"
)
PADDLEOCR_IMAGE_DIGEST = (
    "sha256:d0d32c04a2119613d25a0a4c292e165ccc107954b74580613cf59e378037f8f5"
)
PADDLEOCR_IMAGE_REF = f"{PADDLEOCR_IMAGE_REPOSITORY}@{PADDLEOCR_IMAGE_DIGEST}"
PADDLEOCR_RUNTIME_FINGERPRINT_LABEL = (
    "com.comictranslate.paddleocr-runtime-fingerprint"
)
OCRPreflightProbeResult = Literal["healthy", "unavailable", "not_managed"]
OCRHealthState = Literal["healthy", "loading", "unavailable"]

_ENGINE_CONFIG = {
    "HunyuanOCR": {
        "compose_file": ROOT_DIR / "hunyuanocr_docker_files" / "docker-compose.yaml",
        "managed_url": "http://127.0.0.1:28080/v1",
        "health_url": "http://127.0.0.1:28080/health",
        "settings_page_name": "HunyuanOCR Settings",
        "container_name": "hunyuanocr-local-server",
        "container_names": ["hunyuanocr-local-server"],
        "uses_llama_cpp": True,
    },
    "MangaLMM": {
        "compose_file": ROOT_DIR / "mangalmm_docker_files" / "docker-compose.yaml",
        "managed_url": "http://127.0.0.1:28081/v1",
        "health_url": "http://127.0.0.1:28081/health",
        "settings_page_name": "MangaLMM Settings",
        "container_name": "mangalmm-local-server",
        "container_names": ["mangalmm-local-server"],
        "uses_llama_cpp": True,
    },
    "PaddleOCR VL": {
        "compose_file": ROOT_DIR / "paddleocr_vl_docker_files" / "docker-compose.yaml",
        "managed_url": "http://127.0.0.1:28118/layout-parsing",
        "health_url": "http://127.0.0.1:28118/docs",
        "health_urls": [
            "http://127.0.0.1:28118/docs",
            "http://127.0.0.1:18000/v1/models",
        ],
        "settings_page_name": "PaddleOCR VL Settings",
        "container_name": "paddleocr-server",
        "container_names": ["paddleocr-vllm", "paddleocr-server"],
        "uses_llama_cpp": False,
    },
}


def _normalize_url(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    path = parsed.path.rstrip("/")
    if not path:
        path = "/"
    return parsed._replace(path=path, params="", query="", fragment="").geturl()


class LocalOCRRuntimeManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._compose_command: tuple[str, ...] | None = None
        self._active_engine: str | None = None
        self._managed_start_attempted_engine: str | None = None
        self._readiness_cache: set[tuple[str, str, str]] = set()
        self._startup_cancel_checker: Callable[[], bool] | None = None

    def validate_engine(self, engine_key: str, settings_page: Any) -> None:
        if not is_local_ocr_engine(engine_key):
            return
        server_url = self._resolve_server_url(engine_key, settings_page)
        if not server_url:
            raise self._build_setup_error(engine_key, "Server URL is empty.")
        if not self.should_manage_engine(engine_key, settings_page):
            return
        compose_file = self._config_for(engine_key)["compose_file"]
        if not compose_file.is_file():
            raise self._build_setup_error(
                engine_key,
                f"Bundled Docker compose file was not found: {compose_file}",
            )
        self._resolve_compose_command(engine_key)

    def should_manage_engine(self, engine_key: str, settings_page: Any) -> bool:
        if not is_local_ocr_engine(engine_key):
            return False
        config = self._config_for(engine_key)
        server_url = self._resolve_server_url(engine_key, settings_page)
        return _normalize_url(server_url) == _normalize_url(config["managed_url"])

    def preflight_cache_key(self, engine_key: str, settings_page: Any) -> str | None:
        if not self.should_manage_engine(engine_key, settings_page):
            return None
        return f"{engine_key}|{_normalize_url(self._resolve_server_url(engine_key, settings_page))}"

    def get_ocr_cache_identity(
        self,
        engine_key: str,
        settings_page: Any,
    ) -> dict[str, Any] | None:
        """Return a trustworthy identity without starting the managed runtime."""

        if engine_key != "PaddleOCR VL":
            return None
        if not self.should_manage_engine(engine_key, settings_page):
            return None
        try:
            expected_names = self._managed_container_names(engine_key)
            present_names = self._present_managed_container_names(engine_key)
            if (
                len(present_names) != len(expected_names)
                or not self._paddle_containers_match_contract(present_names)
            ):
                return None
            contract = self._paddle_runtime_contract()
            image_id = self._inspect_docker_image_id(PADDLEOCR_IMAGE_REF)
        except OperationCancelledError:
            raise
        except Exception:
            logger.warning(
                "Persistent PaddleOCR-VL cache is disabled for this run because "
                "the managed runtime identity could not be resolved.",
                exc_info=True,
            )
            return None
        if not image_id:
            logger.info(
                "Persistent PaddleOCR-VL cache is disabled until the pinned "
                "managed image is installed locally."
            )
            return None
        return {
            "identity_schema_version": 1,
            "managed": True,
            "engine": engine_key,
            "endpoint": _normalize_url(
                self._resolve_server_url(engine_key, settings_page)
            ),
            "model_name": "PaddleOCR-VL-1.6-0.9B",
            "image_ref": PADDLEOCR_IMAGE_REF,
            "image_digest": PADDLEOCR_IMAGE_DIGEST,
            "image_id": image_id,
            "compose_sha256": contract["compose_sha256"],
            "command_sha256": contract["command_sha256"],
            "vllm_config_sha256": contract["vllm_config_sha256"],
            "pipeline_config_sha256": contract["pipeline_config_sha256"],
            "runtime_fingerprint": contract["runtime_fingerprint"],
        }

    def probe_managed_engine(
        self,
        engine_key: str,
        settings_page: Any,
        *,
        timeout_sec: int = 2,
    ) -> OCRPreflightProbeResult:
        if not self.should_manage_engine(engine_key, settings_page):
            return "not_managed"
        config = self._config_for(engine_key)
        health_urls = self._config_health_urls(config)
        if self._wait_for_health(
            health_urls,
            timeout_sec=timeout_sec,
            progress_callback=None,
            cancel_checker=None,
            engine_key=engine_key,
            step_key="health_probe",
            message=f"{engine_key} 상태를 확인하는 중...",
        ):
            return "healthy"
        return "unavailable"

    def ensure_engine(
        self,
        engine_key: str,
        settings_page: Any,
        *,
        timeout_sec: int = 300,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> None:
        with self._lock:
            self._startup_cancel_checker = cancel_checker
            if not is_local_ocr_engine(engine_key) or not self.should_manage_engine(engine_key, settings_page):
                self._deactivate_active_engine()
                return

            cache_key = self._readiness_cache_key(engine_key, settings_page, managed=True)
            if self._is_cancelled(cancel_checker):
                self._readiness_cache.discard(cache_key)
                raise OperationCancelledError(f"Cancelled while preparing {engine_key} runtime.")
            if self._active_engine and self._active_engine != engine_key:
                self._stop_engine(self._active_engine)
                self._active_engine = None
                self._managed_start_attempted_engine = None
                self._readiness_cache.clear()
            if cache_key in self._readiness_cache:
                self._active_engine = engine_key
                self._managed_start_attempted_engine = engine_key
                self._emit_readiness_cache_hit(progress_callback, engine_key)
                return

            try:
                self._ensure_engine_uncached(
                    engine_key,
                    settings_page,
                    timeout_sec=timeout_sec,
                    progress_callback=progress_callback,
                    cancel_checker=cancel_checker,
                )
            except (LocalServiceSetupError, OperationCancelledError):
                self._readiness_cache.discard(cache_key)
                raise
            else:
                self._readiness_cache.add(cache_key)

    def _ensure_engine_uncached(
        self,
        engine_key: str,
        settings_page: Any,
        *,
        timeout_sec: int,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        cancel_checker: Callable[[], bool] | None,
    ) -> None:
        self.validate_engine(engine_key, settings_page)
        config = self._config_for(engine_key)
        health_urls = self._config_health_urls(config)
        present_containers = (
            self._present_managed_container_names(engine_key)
            if engine_key == "PaddleOCR VL"
            else []
        )
        if (
            engine_key == "PaddleOCR VL"
            and present_containers
            and (
                len(present_containers)
                != len(self._managed_container_names(engine_key))
                or not self._paddle_containers_match_contract(present_containers)
            )
        ):
            self._managed_start_attempted_engine = engine_key
            self._emit_progress(
                progress_callback,
                engine_key,
                status="starting",
                step_key="container_recreate",
                message=f"{engine_key} 런타임 구성이 변경되어 컨테이너를 다시 만드는 중...",
                detail="docker compose up -d --force-recreate",
            )
            self._run_compose(
                engine_key,
                "up",
                "-d",
                "--force-recreate",
                step_name="force-recreate",
            )
            self._active_engine = engine_key
            if not self._wait_for_health(
                health_urls,
                timeout_sec=timeout_sec,
                progress_callback=progress_callback,
                cancel_checker=cancel_checker,
                engine_key=engine_key,
                step_key="health_wait",
                message=f"{engine_key} health 기다리는 중...",
            ):
                raise self._build_setup_error(
                    engine_key,
                    "Recreated managed containers did not become healthy.",
                )
            self._emit_progress(
                progress_callback,
                engine_key,
                status="completed",
                step_key="container_recreate",
                message=f"{engine_key} 컨테이너 재생성이 완료되었습니다.",
            )
            return

        existing_containers = list(present_containers)
        initial_state = self._probe_health_state(health_urls)
        if initial_state == "healthy":
            self._active_engine = engine_key
            self._managed_start_attempted_engine = engine_key
            self._emit_progress(
                progress_callback,
                engine_key,
                status="completed",
                step_key="health_probe",
                message=f"기존 {engine_key} 런타임을 재사용합니다.",
            )
            return
        if initial_state == "loading":
            self._active_engine = engine_key
            self._managed_start_attempted_engine = engine_key
            self._emit_progress(
                progress_callback,
                engine_key,
                status="waiting_health",
                step_key="health_probe",
                message=f"{engine_key} 모델 로딩이 끝날 때까지 기다리는 중...",
                detail=f"Waiting for {', '.join(health_urls)}",
            )
            if self._wait_for_health(
                health_urls,
                timeout_sec=min(timeout_sec, 300),
                progress_callback=progress_callback,
                cancel_checker=cancel_checker,
                engine_key=engine_key,
                step_key="health_probe",
                message=f"{engine_key} 상태를 확인하는 중...",
            ):
                self._active_engine = engine_key
                self._emit_progress(
                    progress_callback,
                    engine_key,
                    status="completed",
                    step_key="health_probe",
                    message=f"기존 {engine_key} 런타임을 재사용합니다.",
                )
                return

        if not existing_containers:
            existing_containers = self._existing_managed_container_names(engine_key)
        if existing_containers:
            self._managed_start_attempted_engine = engine_key
            self._emit_progress(
                progress_callback,
                engine_key,
                status="starting",
                step_key="container_start",
                message=f"기존 {engine_key} 컨테이너를 다시 시작하는 중...",
                detail="docker start " + " ".join(existing_containers),
            )
            self._start_existing_managed_containers(engine_key, existing_containers)
            self._active_engine = engine_key
            self._emit_progress(
                progress_callback,
                engine_key,
                status="completed",
                step_key="container_start",
                message=f"기존 {engine_key} 컨테이너 시작 명령을 보냈습니다.",
            )
            if self._wait_for_health(
                health_urls,
                timeout_sec=timeout_sec,
                progress_callback=progress_callback,
                cancel_checker=cancel_checker,
                engine_key=engine_key,
                step_key="health_wait",
                message=f"{engine_key} health 기다리는 중...",
            ):
                self._active_engine = engine_key
                self._emit_progress(
                    progress_callback,
                    engine_key,
                    status="completed",
                    step_key="health_wait",
                    message=f"{engine_key} health 확인이 완료되었습니다.",
                    existing_container_reused=True,
                )
                self._log_runtime_metadata(engine_key)
                return
            raise self._build_setup_error(
                engine_key,
                (
                    f"Existing managed containers did not become healthy after docker start: {existing_containers}. "
                    "Stop/remove the stale containers or check Docker logs before retrying."
                ),
            )

        self._emit_progress(
            progress_callback,
            engine_key,
            status="starting",
            step_key="compose_up",
            message=f"{engine_key} 컨테이너를 시작하는 중...",
            detail="docker compose up -d",
        )
        self._managed_start_attempted_engine = engine_key
        self._run_compose(engine_key, "up", "-d", step_name="up")
        self._active_engine = engine_key
        self._emit_progress(
            progress_callback,
            engine_key,
            status="completed",
            step_key="compose_up",
            message=f"{engine_key} 컨테이너 시작 명령을 보냈습니다.",
        )
        if not self._wait_for_health(
            health_urls,
            timeout_sec=timeout_sec,
            progress_callback=progress_callback,
            cancel_checker=cancel_checker,
            engine_key=engine_key,
            step_key="health_wait",
            message=f"{engine_key} health 기다리는 중...",
        ):
            raise self._build_setup_error(
                engine_key,
                (
                    f"Timed out while waiting for {engine_key} at {', '.join(health_urls)} "
                    f"after docker compose up -d of {config['compose_file'].name}."
                ),
            )
        self._active_engine = engine_key
        self._emit_progress(
            progress_callback,
            engine_key,
            status="completed",
            step_key="health_wait",
            message=f"{engine_key} health 확인이 완료되었습니다.",
        )
        self._log_runtime_metadata(engine_key)

    def shutdown(self) -> None:
        with self._lock:
            self._readiness_cache.clear()
            self._deactivate_active_engine()

    def _deactivate_active_engine(self) -> None:
        self._readiness_cache.clear()
        engine_key = self._active_engine or self._managed_start_attempted_engine
        if not engine_key:
            return
        self._stop_engine(engine_key)
        self._active_engine = None
        self._managed_start_attempted_engine = None

    def _stop_engine(self, engine_key: str) -> None:
        watch_for_late_start = bool(
            self._managed_start_attempted_engine == engine_key
            and self._active_engine != engine_key
        )
        self._run_compose(
            engine_key,
            "stop",
            "--timeout",
            str(DEFAULT_MANAGED_RUNTIME_STOP_TIMEOUT_SEC),
            step_name="stop",
        )
        deadline = time.monotonic() + (
            LATE_START_STOP_GRACE_SEC if watch_for_late_start else 0.0
        )
        while True:
            running = self._running_managed_container_names(engine_key)
            if running and watch_for_late_start:
                self._run_compose(
                    engine_key,
                    "stop",
                    "--timeout",
                    str(DEFAULT_MANAGED_RUNTIME_STOP_TIMEOUT_SEC),
                    step_name="stop",
                )
                running = self._running_managed_container_names(engine_key)
            if time.monotonic() >= deadline:
                if running:
                    raise self._build_setup_error(
                        engine_key,
                        f"Managed OCR containers are still running after stop: {running}",
                    )
                return
            time.sleep(LATE_START_STOP_POLL_SEC)

    def _run_compose(self, engine_key: str, *compose_args: str, step_name: str) -> None:
        config = self._config_for(engine_key)
        compose_file = config["compose_file"]
        env = self._build_env(engine_key)
        command = [
            *self._resolve_compose_command(engine_key),
            "-f",
            str(compose_file),
            *compose_args,
        ]
        try:
            from modules.utils.llama_cpp_runtime import run_docker_command

            run_docker_command(
                command,
                cwd=compose_file.parent,
                env=env,
                timeout_sec=(
                    DEFAULT_MANAGED_RUNTIME_STOP_TIMEOUT_SEC + 15.0
                    if step_name == "stop"
                    else 600.0
                ),
                cancel_checker=(
                    None if step_name == "stop" else self._startup_cancel_checker
                ),
            )
            return
        except RuntimeError as exc:
            detail = str(exc).strip()
        requested_image = env.get("LLAMA_CPP_IMAGE", "") if config.get("uses_llama_cpp") else ""
        extra = f"Docker compose {step_name} failed.\n{detail}"
        if requested_image:
            extra = f"{extra}\nRequested image: {requested_image}"
        raise self._build_setup_error(engine_key, extra)

    def _resolve_compose_command(self, engine_key: str) -> tuple[str, ...]:
        if self._compose_command is not None:
            return self._compose_command
        try:
            self._compose_command = resolve_docker_compose_command(
                cancel_checker=self._startup_cancel_checker,
            )
            return self._compose_command
        except RuntimeError as exc:
            raise self._build_setup_error(
                engine_key,
                "Docker Compose is not available. Install Docker Desktop or docker-compose and try again.",
            ) from exc

    def _build_env(self, engine_key: str) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("LLAMA_CPP_IMAGE", DEFAULT_LLAMA_CPP_IMAGE)
        if engine_key == "HunyuanOCR":
            env.setdefault("LLAMA_N_GPU_LAYERS", DEFAULT_HUNYUAN_N_GPU_LAYERS)
        if engine_key == "PaddleOCR VL":
            env["PADDLEOCR_RUNTIME_FINGERPRINT"] = str(
                self._paddle_runtime_contract()["runtime_fingerprint"]
            )
        return env

    def _resolve_server_url(self, engine_key: str, settings_page: Any) -> str:
        if engine_key == "HunyuanOCR":
            return str(settings_page.get_hunyuan_ocr_settings().get("server_url", "")).strip()
        if engine_key == "MangaLMM":
            return str(settings_page.get_mangalmm_ocr_settings().get("server_url", "")).strip()
        if engine_key == "PaddleOCR VL":
            return str(settings_page.get_paddleocr_vl_settings().get("server_url", "")).strip()
        return ""

    def _readiness_cache_key(
        self,
        engine_key: str,
        settings_page: Any,
        *,
        managed: bool,
    ) -> tuple[str, str, str]:
        mode = "managed" if managed else "unmanaged"
        return (engine_key, _normalize_url(self._resolve_server_url(engine_key, settings_page)), mode)

    def _emit_readiness_cache_hit(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        engine_key: str,
    ) -> None:
        self._emit_progress(
            progress_callback,
            engine_key,
            status="completed",
            step_key="readiness_cache",
            message=f"기존 {engine_key} 런타임을 재사용합니다.",
            readiness_cache_hit=True,
        )

    def _config_for(self, engine_key: str) -> dict[str, Any]:
        try:
            return _ENGINE_CONFIG[engine_key]
        except KeyError as exc:
            raise self._build_setup_error(engine_key, f"Unsupported local OCR engine: {engine_key}") from exc

    def _log_runtime_metadata(self, engine_key: str) -> None:
        config = self._config_for(engine_key)
        if not config.get("uses_llama_cpp"):
            return
        try:
            runtime = inspect_llama_cpp_runtime(
                image_ref=self._build_env(engine_key).get("LLAMA_CPP_IMAGE"),
                container_name=str(config.get("container_name") or ""),
                cancel_checker=self._startup_cancel_checker,
            )
        except OperationCancelledError:
            raise
        except Exception:
            logger.warning("Failed to inspect llama.cpp runtime metadata for %s.", engine_key, exc_info=True)
            return
        logger.info(
            "%s runtime ready: image=%s digest=%s version=%s",
            engine_key,
            runtime.get("llama_cpp_image", ""),
            runtime.get("llama_cpp_digest", ""),
            runtime.get("llama_cpp_version", ""),
        )

    def _managed_container_names(self, engine_key: str) -> list[str]:
        config = self._config_for(engine_key)
        names = config.get("container_names") or [config.get("container_name")]
        return [str(name).strip() for name in names if str(name or "").strip()]

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _paddle_runtime_contract(self) -> dict[str, str]:
        compose_file = Path(self._config_for("PaddleOCR VL")["compose_file"])
        vllm_config = compose_file.parent / "vllm_config.yml"
        pipeline_config = compose_file.parent / "pipeline_conf.yaml"
        for path in (compose_file, vllm_config, pipeline_config):
            if not path.is_file():
                raise FileNotFoundError(path)

        try:
            import yaml

            compose_payload = yaml.safe_load(
                compose_file.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to parse PaddleOCR-VL compose contract: {compose_file}"
            ) from exc
        services = (
            compose_payload.get("services", {})
            if isinstance(compose_payload, dict)
            else {}
        )
        commands = {
            str(name): str(config.get("command", "") or "")
            for name, config in sorted(services.items())
            if isinstance(config, dict)
        }
        command_sha256 = hashlib.sha256(
            json.dumps(
                commands,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        contract = {
            "compose_sha256": self._sha256_file(compose_file),
            "command_sha256": command_sha256,
            "vllm_config_sha256": self._sha256_file(vllm_config),
            "pipeline_config_sha256": self._sha256_file(pipeline_config),
        }
        contract["runtime_fingerprint"] = hashlib.sha256(
            json.dumps(
                {
                    **contract,
                    "image_ref": PADDLEOCR_IMAGE_REF,
                    "model_name": "PaddleOCR-VL-1.6-0.9B",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return contract

    def _inspect_docker_image_id(self, image_ref: str) -> str:
        from modules.utils.llama_cpp_runtime import run_docker_command

        completed = run_docker_command(
            ["docker", "image", "inspect", image_ref, "--format", "{{.Id}}"],
            check=False,
            timeout_sec=15.0,
            cancel_checker=self._startup_cancel_checker,
        )
        if getattr(completed, "returncode", 1) != 0:
            return ""
        return str(getattr(completed, "stdout", "") or "").strip()

    def _paddle_containers_match_contract(
        self,
        container_names: list[str],
    ) -> bool:
        try:
            contract = self._paddle_runtime_contract()
            expected_image_id = self._inspect_docker_image_id(PADDLEOCR_IMAGE_REF)
        except OperationCancelledError:
            raise
        except Exception:
            logger.warning(
                "Failed to resolve the PaddleOCR-VL runtime contract; stale "
                "containers will be recreated.",
                exc_info=True,
            )
            return False
        if not expected_image_id:
            return False

        from modules.utils.llama_cpp_runtime import run_docker_command

        expected_fingerprint = str(contract["runtime_fingerprint"])
        for name in container_names:
            completed = run_docker_command(
                [
                    "docker",
                    "inspect",
                    "--format",
                    (
                        "{{index .Config.Labels "
                        f"\"{PADDLEOCR_RUNTIME_FINGERPRINT_LABEL}\""
                        "}}|{{.Image}}"
                    ),
                    name,
                ],
                check=False,
                timeout_sec=15.0,
                cancel_checker=self._startup_cancel_checker,
            )
            if getattr(completed, "returncode", 1) != 0:
                return False
            fingerprint, separator, image_id = str(
                getattr(completed, "stdout", "") or ""
            ).strip().partition("|")
            if (
                not separator
                or fingerprint != expected_fingerprint
                or image_id != expected_image_id
            ):
                return False
        return True

    def _existing_managed_container_names(self, engine_key: str) -> list[str]:
        names = self._managed_container_names(engine_key)
        existing = self._present_managed_container_names(engine_key)
        return existing if len(existing) == len(names) else []

    def _present_managed_container_names(self, engine_key: str) -> list[str]:
        names = self._managed_container_names(engine_key)
        existing: list[str] = []
        for name in names:
            try:
                from modules.utils.llama_cpp_runtime import run_docker_command

                completed = run_docker_command(
                    ["docker", "inspect", "--format", "{{.Name}}", name],
                    check=False,
                    cancel_checker=self._startup_cancel_checker,
                )
            except OperationCancelledError:
                raise
            except Exception:
                continue
            if getattr(completed, "returncode", 1) == 0:
                existing.append(name)
        return existing

    def _start_existing_managed_containers(self, engine_key: str, container_names: list[str]) -> None:
        if not container_names:
            return
        try:
            from modules.utils.llama_cpp_runtime import run_docker_command

            run_docker_command(
                ["docker", "start", *container_names],
                cancel_checker=self._startup_cancel_checker,
            )
        except RuntimeError as exc:
            raise self._build_setup_error(
                engine_key,
                f"Failed to start existing managed containers {container_names}.\n{str(exc).strip()}",
            ) from exc

    def _running_managed_container_names(self, engine_key: str) -> list[str]:
        running: list[str] = []
        for name in self._managed_container_names(engine_key):
            try:
                from modules.utils.llama_cpp_runtime import run_docker_command

                completed = run_docker_command(
                    [
                        "docker",
                        "inspect",
                        "--format",
                        "{{.State.Running}}",
                        name,
                    ],
                    check=False,
                    timeout_sec=15.0,
                )
            except Exception as exc:
                raise self._build_setup_error(
                    engine_key,
                    f"Failed to verify stopped OCR container {name}: {exc}",
                ) from exc
            state = (completed.stdout or "").strip().lower()
            if completed.returncode == 0 and state == "true":
                running.append(name)
                continue
            if completed.returncode == 0 and state == "false":
                continue
            detail = (
                (completed.stderr or "") + "\n" + (completed.stdout or "")
            ).strip()
            normalized_detail = detail.lower()
            if completed.returncode != 0 and (
                "no such object" in normalized_detail
                or "no such container" in normalized_detail
            ):
                continue
            if completed.returncode == 0:
                detail = f"Unexpected docker inspect state: {state or '<empty>'}"
            raise self._build_setup_error(
                engine_key,
                (
                    f"Docker could not verify whether OCR container {name} stopped."
                    + (f"\n{detail}" if detail else "")
                ),
            )
        return running

    def _build_setup_error(self, engine_key: str, detail: str) -> LocalServiceSetupError:
        config = _ENGINE_CONFIG.get(engine_key, {})
        expected_url = config.get("managed_url") or ""
        compose_file = config.get("compose_file")
        settings_page_name = config.get("settings_page_name", "PaddleOCR VL Settings")
        extra = detail.strip()
        if expected_url:
            extra = f"{extra}\nRequested engine: {engine_key}\nExpected endpoint: {expected_url}"
        if compose_file:
            extra = f"{extra}\nCompose file: {compose_file}"
        return LocalServiceSetupError(
            extra.strip(),
            service_name=engine_key,
            settings_page_name=settings_page_name,
        )

    def _wait_for_health(
        self,
        urls: str | list[str] | tuple[str, ...],
        *,
        timeout_sec: int,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_checker: Callable[[], bool] | None = None,
        engine_key: str,
        step_key: str,
        message: str,
    ) -> bool:
        health_urls = self._normalize_health_urls(urls)
        if not health_urls:
            return False
        deadline = time.monotonic() + max(timeout_sec, 1)
        started = time.monotonic()
        while time.monotonic() < deadline:
            if self._is_cancelled(cancel_checker):
                raise OperationCancelledError(f"Cancelled while waiting for {engine_key} runtime.")
            if self._probe_health_state(health_urls) == "healthy":
                return True
            self._emit_progress(
                progress_callback,
                engine_key,
                status="waiting_health",
                step_key=step_key,
                message=message,
                elapsed_sec=time.monotonic() - started,
                detail=f"Waiting for {', '.join(health_urls)}",
            )
            time.sleep(1)
        return False

    @staticmethod
    def _normalize_health_urls(
        urls: str | list[str] | tuple[str, ...],
    ) -> tuple[str, ...]:
        candidates = [urls] if isinstance(urls, str) else list(urls)
        return tuple(
            dict.fromkeys(
                str(url).strip()
                for url in candidates
                if str(url).strip()
            )
        )

    @classmethod
    def _config_health_urls(cls, config: dict[str, Any]) -> tuple[str, ...]:
        configured = config.get("health_urls")
        if configured is None:
            configured = str(config.get("health_url") or "")
        return cls._normalize_health_urls(configured)

    @staticmethod
    def _probe_single_health_state(url: str) -> OCRHealthState:
        try:
            with urlopen(url, timeout=2) as response:
                status = int(getattr(response, "status", 200))
        except HTTPError as exc:
            if int(exc.code) == 503:
                return "loading"
            return "unavailable"
        except (URLError, OSError, ValueError):
            return "unavailable"
        if status == 503:
            return "loading"
        if 200 <= status < 400:
            return "healthy"
        return "unavailable"

    @classmethod
    def _probe_health_state(
        cls,
        urls: str | list[str] | tuple[str, ...],
    ) -> OCRHealthState:
        health_urls = cls._normalize_health_urls(urls)
        if not health_urls:
            return "unavailable"
        states = [cls._probe_single_health_state(url) for url in health_urls]
        if all(state == "healthy" for state in states):
            return "healthy"
        if any(state in {"healthy", "loading"} for state in states):
            return "loading"
        return "unavailable"

    def _emit_progress(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        engine_key: str,
        **payload: Any,
    ) -> None:
        if progress_callback is None:
            return
        event = {
            "phase": "ocr_startup",
            "service": "paddleocr_vl" if engine_key == "PaddleOCR VL" else engine_key.lower(),
            "status": payload.pop("status", "running"),
            "step_key": payload.pop("step_key", "health_wait"),
            "message": payload.pop("message", f"{engine_key} 준비 중..."),
            "detail": payload.pop("detail", ""),
        }
        event.update(payload)
        try:
            progress_callback(event)
        except Exception:
            logger.debug("Failed to emit OCR runtime progress event.", exc_info=True)

    @staticmethod
    def _is_cancelled(cancel_checker: Callable[[], bool] | None) -> bool:
        try:
            return bool(cancel_checker and cancel_checker())
        except Exception:
            return False
