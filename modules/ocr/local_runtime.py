from __future__ import annotations

import base64
import binascii
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
from modules.ocr.paddle_llamacpp_runtime_contract import (
    DEFAULT_PADDLE_LAYOUT_IMAGE,
    DEFAULT_PADDLE_LLAMA_CPP_IMAGE,
    DEFAULT_PADDLE_LLAMA_MODEL_VOLUME,
    DEFAULT_PADDLE_LLAMA_READY_MANIFEST,
    PADDLE_LLAMA_MMPROJ_NAME,
    PADDLE_LLAMA_MODEL_ALIAS,
    PADDLE_LLAMA_MODEL_NAME,
    PADDLE_LLAMA_MODEL_SPECS,
    PADDLE_LLAMA_RUNTIME_PREPARATION_VERSION,
    PADDLE_RUNTIME_FINGERPRINT_LABEL,
    PaddleLlamaRuntimeContract,
    PaddleLlamaRuntimeContractError,
    build_paddle_llama_runtime_contract,
    validate_paddle_llama_volume_name,
)
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
PADDLEOCR_LAYOUT_IMAGE_REF = DEFAULT_PADDLE_LAYOUT_IMAGE
PADDLEOCR_LAYOUT_IMAGE_DIGEST = PADDLEOCR_LAYOUT_IMAGE_REF.rsplit("@", 1)[-1]
PADDLEOCR_LLAMA_CPP_IMAGE_REF = DEFAULT_PADDLE_LLAMA_CPP_IMAGE
PADDLEOCR_LLAMA_CPP_IMAGE_DIGEST = PADDLEOCR_LLAMA_CPP_IMAGE_REF.rsplit("@", 1)[
    -1
]
# Compatibility aliases used by existing cache/runtime callers.
PADDLEOCR_IMAGE_REF = PADDLEOCR_LAYOUT_IMAGE_REF
PADDLEOCR_IMAGE_DIGEST = PADDLEOCR_LAYOUT_IMAGE_DIGEST
PADDLEOCR_RUNTIME_FINGERPRINT_LABEL = PADDLE_RUNTIME_FINGERPRINT_LABEL
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
            "http://127.0.0.1:18000/health",
        ],
        "settings_page_name": "PaddleOCR VL Settings",
        "container_name": "paddleocr-llamacpp",
        "container_names": ["paddleocr-llamacpp", "paddleocr-server"],
        "uses_llama_cpp": True,
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
        self._paddle_runtime_contract_cache: PaddleLlamaRuntimeContract | None = None
        self._paddle_idle_released = False

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
        if engine_key == "PaddleOCR VL":
            try:
                self._ensure_paddle_runtime_images()
                self._paddle_runtime_contract(force_refresh=True)
            except (PaddleLlamaRuntimeContractError, OSError) as exc:
                raise self._build_setup_error(
                    engine_key,
                    (
                        f"Prepared PaddleOCR llama.cpp runtime validation failed: {exc}\n"
                        "Run scripts/prepare_paddleocr_llamacpp_runtime.ps1 "
                        "in Prepare or Verify mode."
                    )
                ) from exc

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
        except OperationCancelledError:
            raise
        except Exception:
            logger.warning(
                "Persistent PaddleOCR-VL cache is disabled for this run because "
                "the managed runtime identity could not be resolved.",
                exc_info=True,
            )
            return None
        if not contract.llama_image_id or not contract.layout_image_id:
            logger.info(
                "Persistent PaddleOCR-VL cache is disabled until the pinned "
                "managed images are installed locally."
            )
            return None
        return {
            "identity_schema_version": 2,
            "managed": True,
            "engine": engine_key,
            "backend": "llama.cpp",
            "endpoint": _normalize_url(
                self._resolve_server_url(engine_key, settings_page)
            ),
            "model_name": PADDLE_LLAMA_MODEL_ALIAS,
            "model_file": PADDLE_LLAMA_MODEL_NAME,
            "model_sha256": str(
                PADDLE_LLAMA_MODEL_SPECS[PADDLE_LLAMA_MODEL_NAME]["sha256"]
            ),
            "mmproj_file": PADDLE_LLAMA_MMPROJ_NAME,
            "mmproj_sha256": str(
                PADDLE_LLAMA_MODEL_SPECS[PADDLE_LLAMA_MMPROJ_NAME]["sha256"]
            ),
            "model_volume": contract.volume_name,
            "ready_manifest_sha256": contract.ready_manifest_sha256,
            "llama_image_ref": contract.llama_image_ref,
            "llama_image_digest": PADDLEOCR_LLAMA_CPP_IMAGE_DIGEST,
            "llama_image_id": contract.llama_image_id,
            "layout_image_ref": contract.layout_image_ref,
            "layout_image_digest": PADDLEOCR_LAYOUT_IMAGE_DIGEST,
            "layout_image_id": contract.layout_image_id,
            "compose_sha256": contract.compose_file_sha256,
            "command_sha256": contract.command_sha256,
            "pipeline_config_sha256": contract.pipeline_config_sha256,
            "runtime_fingerprint": contract.fingerprint,
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
                cache_is_healthy = (
                    engine_key != "PaddleOCR VL"
                    or self._probe_health_state(
                        self._config_health_urls(self._config_for(engine_key))
                    )
                    == "healthy"
                )
                if cache_is_healthy:
                    self._active_engine = engine_key
                    self._managed_start_attempted_engine = engine_key
                    if engine_key == "PaddleOCR VL":
                        self._paddle_idle_released = False
                    self._emit_readiness_cache_hit(progress_callback, engine_key)
                    return
                self._readiness_cache.discard(cache_key)
                self._paddle_idle_released = False

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
                if engine_key == "PaddleOCR VL":
                    self._paddle_idle_released = False

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
            self._paddle_idle_released = False

    def release_for_handoff(self) -> None:
        """Release OCR GPU residency while preserving a reusable llama server."""

        with self._lock:
            if self._active_engine not in (None, "PaddleOCR VL"):
                self._deactivate_active_engine()
                return
            if self._active_engine is None:
                running = self._running_managed_container_names("PaddleOCR VL")
                if not running:
                    return
                expected = self._managed_container_names("PaddleOCR VL")
                if (
                    len(running) != len(expected)
                    or not self._paddle_containers_match_contract(running)
                ):
                    self._managed_start_attempted_engine = "PaddleOCR VL"
                    self._deactivate_active_engine()
                    return
                self._active_engine = "PaddleOCR VL"
                self._managed_start_attempted_engine = "PaddleOCR VL"
            if self._paddle_idle_released:
                return
            if self._wait_for_paddle_llama_sleep():
                self._paddle_idle_released = True
                logger.info(
                    "PaddleOCR llama.cpp entered idle sleep; containers remain "
                    "available for the next OCR stage."
                )
                return
            logger.warning(
                "PaddleOCR llama.cpp did not confirm idle sleep; falling back "
                "to the normal managed stop before the next GPU stage."
            )
            self._deactivate_active_engine()
            self._paddle_idle_released = False

    def _deactivate_active_engine(self) -> None:
        self._readiness_cache.clear()
        engine_key = self._active_engine or self._managed_start_attempted_engine
        if not engine_key:
            return
        self._stop_engine(engine_key)
        self._active_engine = None
        self._managed_start_attempted_engine = None
        if engine_key == "PaddleOCR VL":
            self._paddle_idle_released = False

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

    def _wait_for_paddle_llama_sleep(self) -> bool:
        try:
            contract = self._paddle_runtime_contract()
            idle_seconds = int(
                contract.runtime_options[
                    "PADDLEOCR_LLAMA_SLEEP_IDLE_SECONDS"
                ]
            )
        except Exception:
            logger.warning(
                "Unable to resolve the PaddleOCR llama.cpp sleep contract.",
                exc_info=True,
            )
            return False

        from modules.utils.llama_cpp_runtime import run_docker_command

        deadline = time.monotonic() + idle_seconds + 15.0
        while time.monotonic() < deadline:
            if self._is_cancelled(self._startup_cancel_checker):
                raise OperationCancelledError(
                    "Cancelled while waiting for PaddleOCR llama.cpp to sleep."
                )
            completed = run_docker_command(
                [
                    "docker",
                    "logs",
                    "--tail",
                    "240",
                    "paddleocr-llamacpp",
                ],
                check=False,
                timeout_sec=10.0,
                cancel_checker=self._startup_cancel_checker,
            )
            if completed.returncode != 0:
                return False
            log_text = (
                (completed.stdout or "") + "\n" + (completed.stderr or "")
            )
            sleep_index = max(
                log_text.rfind("entering sleeping state"),
                log_text.rfind("server is entering sleeping state"),
            )
            activity_index = max(
                log_text.rfind("exiting sleeping state"),
                log_text.rfind("processing task"),
                log_text.rfind("stop processing"),
            )
            if sleep_index >= 0 and sleep_index > activity_index:
                return True
            time.sleep(0.25)
        return False

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
        requested_image = ""
        if config.get("uses_llama_cpp"):
            requested_image = env.get(
                (
                    "PADDLEOCR_LLAMA_CPP_IMAGE"
                    if engine_key == "PaddleOCR VL"
                    else "LLAMA_CPP_IMAGE"
                ),
                "",
            )
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
            env.update(self._paddle_runtime_contract().compose_environment())
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
            image_env_key = (
                "PADDLEOCR_LLAMA_CPP_IMAGE"
                if engine_key == "PaddleOCR VL"
                else "LLAMA_CPP_IMAGE"
            )
            runtime = inspect_llama_cpp_runtime(
                image_ref=self._build_env(engine_key).get(image_env_key),
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

    def _paddle_runtime_contract(
        self,
        *,
        force_refresh: bool = False,
    ) -> PaddleLlamaRuntimeContract:
        if self._paddle_runtime_contract_cache is not None and not force_refresh:
            return self._paddle_runtime_contract_cache

        compose_file = Path(self._config_for("PaddleOCR VL")["compose_file"])
        pipeline_config = compose_file.parent / "pipeline_conf.yaml"
        for path in (compose_file, pipeline_config):
            if not path.is_file():
                raise FileNotFoundError(path)
        volume_name = validate_paddle_llama_volume_name(
            os.environ.get(
                "PADDLEOCR_LLAMA_MODEL_VOLUME",
                DEFAULT_PADDLE_LLAMA_MODEL_VOLUME,
            )
        )
        (
            manifest_bytes,
            manifest_sha256,
            observed_file_bytes,
        ) = self._probe_paddle_model_volume(
            volume_name=volume_name,
            image_ref=PADDLEOCR_LLAMA_CPP_IMAGE_REF,
        )
        llama_image_id = self._inspect_docker_image_id(
            PADDLEOCR_LLAMA_CPP_IMAGE_REF
        )
        layout_image_id = self._inspect_docker_image_id(
            PADDLEOCR_LAYOUT_IMAGE_REF
        )
        if not llama_image_id or not layout_image_id:
            raise PaddleLlamaRuntimeContractError(
                "Pinned PaddleOCR runtime images are not installed."
            )
        contract = build_paddle_llama_runtime_contract(
            manifest_bytes=manifest_bytes,
            manifest_sha256=manifest_sha256,
            observed_file_bytes=observed_file_bytes,
            volume_name=volume_name,
            llama_image_ref=PADDLEOCR_LLAMA_CPP_IMAGE_REF,
            llama_image_id=llama_image_id,
            layout_image_ref=PADDLEOCR_LAYOUT_IMAGE_REF,
            layout_image_id=layout_image_id,
            compose_file=compose_file,
            pipeline_config_file=pipeline_config,
            environment=os.environ,
        )
        self._paddle_runtime_contract_cache = contract
        return contract

    def _ensure_paddle_runtime_images(self) -> None:
        for image_ref in (
            PADDLEOCR_LLAMA_CPP_IMAGE_REF,
            PADDLEOCR_LAYOUT_IMAGE_REF,
        ):
            if self._inspect_docker_image_id(image_ref):
                continue
            from modules.utils.llama_cpp_runtime import run_docker_command

            try:
                run_docker_command(
                    ["docker", "pull", image_ref],
                    cancel_checker=self._startup_cancel_checker,
                )
            except RuntimeError as exc:
                raise self._build_setup_error(
                    "PaddleOCR VL",
                    f"Unable to load the pinned PaddleOCR image: {image_ref}\n{exc}"
                ) from exc
            if not self._inspect_docker_image_id(image_ref):
                raise self._build_setup_error(
                    "PaddleOCR VL",
                    f"Docker returned no image ID for the pinned PaddleOCR image: {image_ref}"
                )

    def _probe_paddle_model_volume(
        self,
        *,
        volume_name: str,
        image_ref: str,
    ) -> tuple[bytes, str, dict[str, int]]:
        from modules.utils.llama_cpp_runtime import run_docker_command

        volume_inspection = run_docker_command(
            [
                "docker",
                "volume",
                "inspect",
                "--format",
                "{{json .Labels}}",
                volume_name,
            ],
            check=False,
            cancel_checker=self._startup_cancel_checker,
        )
        if volume_inspection.returncode != 0:
            raise self._build_setup_error(
                "PaddleOCR VL",
                (
                    "Prepared PaddleOCR llama.cpp model volume does not exist: "
                    f"{volume_name}\n"
                    "Run scripts/prepare_paddleocr_llamacpp_runtime.ps1 before "
                    "starting the managed endpoint."
                )
            )
        try:
            volume_labels = json.loads(
                (volume_inspection.stdout or "").strip() or "{}"
            )
        except json.JSONDecodeError as exc:
            raise self._build_setup_error(
                "PaddleOCR VL",
                f"Unable to parse Docker labels for PaddleOCR volume: {volume_name}"
            ) from exc
        expected_labels = {
            "comic-translate.runtime": "PaddleOCR-VL-llama.cpp",
            "comic-translate.preparation-version": str(
                PADDLE_LLAMA_RUNTIME_PREPARATION_VERSION
            ),
        }
        if not isinstance(volume_labels, dict) or any(
            str(volume_labels.get(key, "")) != expected
            for key, expected in expected_labels.items()
        ):
            raise self._build_setup_error(
                "PaddleOCR VL",
                (
                    "PaddleOCR llama.cpp volume labels do not match the "
                    f"preparation contract: {volume_name}\n"
                    f"Expected labels: {expected_labels}\n"
                    f"Actual labels: {volume_labels}"
                )
            )

        shell_script = r'''
set -eu
manifest_path="/models/$READY_MANIFEST"
model_path="/models/$MODEL_FILE"
mmproj_path="/models/$MMPROJ_FILE"
test -f "$manifest_path"
test -f "$model_path"
test -f "$mmproj_path"
printf 'manifest_sha256=%s\n' "$(sha256sum "$manifest_path" | cut -d ' ' -f 1)"
printf 'manifest_base64='
base64 -w 0 "$manifest_path"
printf '\nmodel_bytes=%s\n' "$(stat -c %s "$model_path")"
printf 'mmproj_bytes=%s\n' "$(stat -c %s "$mmproj_path")"
'''.strip()
        completed = run_docker_command(
            [
                "docker",
                "run",
                "--rm",
                "--pull",
                "never",
                "-e",
                f"READY_MANIFEST={DEFAULT_PADDLE_LLAMA_READY_MANIFEST}",
                "-e",
                f"MODEL_FILE={PADDLE_LLAMA_MODEL_NAME}",
                "-e",
                f"MMPROJ_FILE={PADDLE_LLAMA_MMPROJ_NAME}",
                "--mount",
                f"type=volume,source={volume_name},target=/models,readonly",
                "--entrypoint",
                "/bin/sh",
                image_ref,
                "-ec",
                shell_script,
            ],
            check=False,
            cancel_checker=self._startup_cancel_checker,
        )
        if completed.returncode != 0:
            detail = (
                (completed.stderr or "") + "\n" + (completed.stdout or "")
            ).strip()
            raise self._build_setup_error(
                "PaddleOCR VL",
                (
                    "Prepared PaddleOCR llama.cpp model volume is incomplete: "
                    f"{volume_name}\n{detail}\n"
                    "Run scripts/prepare_paddleocr_llamacpp_runtime.ps1 in "
                    "Prepare or Verify mode."
                )
            )

        values: dict[str, str] = {}
        for line in (completed.stdout or "").splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key.strip()] = value.strip()
        try:
            manifest_bytes = base64.b64decode(
                values["manifest_base64"],
                validate=True,
            )
            manifest_sha256 = values["manifest_sha256"].lower()
            observed_file_bytes = {
                PADDLE_LLAMA_MODEL_NAME: int(values["model_bytes"]),
                PADDLE_LLAMA_MMPROJ_NAME: int(values["mmproj_bytes"]),
            }
        except (KeyError, ValueError, binascii.Error) as exc:
            raise self._build_setup_error(
                "PaddleOCR VL",
                "Unable to parse the prepared PaddleOCR llama.cpp volume "
                f"probe output: {completed.stdout}"
            ) from exc
        return manifest_bytes, manifest_sha256, observed_file_bytes

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
        except OperationCancelledError:
            raise
        except Exception:
            logger.warning(
                "Failed to resolve the PaddleOCR-VL runtime contract; stale "
                "containers will be recreated.",
                exc_info=True,
            )
            return False

        from modules.utils.llama_cpp_runtime import run_docker_command

        expected_fingerprint = contract.fingerprint
        expected_image_ids = {
            "paddleocr-llamacpp": contract.llama_image_id,
            "paddleocr-server": contract.layout_image_id,
        }
        for name in container_names:
            expected_image_id = expected_image_ids.get(name)
            if not expected_image_id:
                return False
            completed = run_docker_command(
                [
                    "docker",
                    "inspect",
                    "--format",
                    (
                        "{{index .Config.Labels "
                        f"\"{PADDLEOCR_RUNTIME_FINGERPRINT_LABEL}\""
                        "}}|{{.Image}}|"
                        "{{index .Config.Labels "
                        "\"desktop.docker.io/wsl-distro\"}}"
                    ),
                    name,
                ],
                check=False,
                timeout_sec=15.0,
                cancel_checker=self._startup_cancel_checker,
            )
            if getattr(completed, "returncode", 1) != 0:
                return False
            parts = str(
                getattr(completed, "stdout", "") or ""
            ).strip().split("|", 2)
            if len(parts) != 3:
                return False
            fingerprint, image_id, wsl_distro = parts
            if (
                fingerprint != expected_fingerprint
                or image_id != expected_image_id
            ):
                return False
            if os.name == "nt" and wsl_distro.strip():
                logger.info(
                    "PaddleOCR container %s was created by WSL Compose (%s); "
                    "Windows will recreate it so Docker Desktop can manage the "
                    "Compose application without invoking wsl.",
                    name,
                    wsl_distro.strip(),
                )
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
