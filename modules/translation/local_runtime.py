from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from modules.translation.llm.custom_local_gemma import (
    DEFAULT_GEMMA_LOCAL_ENDPOINT,
    DEFAULT_GEMMA_LOCAL_MODEL,
)
from modules.translation.gemma_runtime_contract import (
    DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
    DEFAULT_GEMMA_MODEL_VOLUME,
    DEFAULT_GEMMA_READY_MANIFEST,
    GEMMA_RUNTIME_PREPARATION_VERSION,
    GemmaRuntimeContract,
    GemmaRuntimeContractError,
    build_gemma_runtime_contract,
    container_contract_mismatch_reasons,
    validate_gemma_model_name,
    validate_gemma_volume_name,
)
from modules.utils.exceptions import (
    LocalServiceConnectionError,
    LocalServiceResponseError,
    LocalServiceSetupError,
    OperationCancelledError,
)
from modules.utils.llama_cpp_runtime import (
    DEFAULT_MANAGED_RUNTIME_STOP_TIMEOUT_SEC,
    inspect_llama_cpp_runtime,
    resolve_docker_compose_command,
    run_docker_command,
)

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_GEMMA_HEALTH_URL = "http://127.0.0.1:18080/health"
DEFAULT_GEMMA_MODELS_URL = "http://127.0.0.1:18080/v1/models"
DEFAULT_GEMMA_SETTINGS_PAGE = "Gemma Local Server Settings"
DEFAULT_GEMMA_STARTUP_TIMEOUT_SEC = 420
LATE_START_STOP_GRACE_SEC = 3.0
LATE_START_STOP_POLL_SEC = 0.25

_RUNTIME_CONFIG = {
    "compose_file": ROOT_DIR / "docker-compose.yaml",
    "managed_url": DEFAULT_GEMMA_LOCAL_ENDPOINT,
    "health_url": DEFAULT_GEMMA_HEALTH_URL,
    "models_url": DEFAULT_GEMMA_MODELS_URL,
    "settings_page_name": DEFAULT_GEMMA_SETTINGS_PAGE,
    "container_name": "gemma-local-server",
}


def _normalize_url(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    path = parsed.path.rstrip("/")
    if not path:
        path = "/"
    return parsed._replace(path=path, params="", query="", fragment="").geturl()


def _strip_v1_path(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3] or "/"
    if not path:
        path = "/"
    return parsed._replace(path=path, params="", query="", fragment="").geturl().rstrip("/")


def _derive_probe_urls(api_base_url: str) -> list[str]:
    base = _normalize_url(api_base_url)
    stripped = _strip_v1_path(base)
    candidates = [
        f"{stripped}/health",
        f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models",
    ]
    seen: set[str] = set()
    normalized: list[str] = []
    for item in candidates:
        url = item.rstrip("/")
        if url not in seen:
            seen.add(url)
            normalized.append(url)
    return normalized


class LocalGemmaRuntimeManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._compose_command: tuple[str, ...] | None = None
        self._managed_active = False
        self._managed_start_attempted = False
        self._active_contract: GemmaRuntimeContract | None = None
        self._readiness_cache: set[tuple[str, str, str, str]] = set()
        self._startup_cancel_checker: Callable[[], bool] | None = None

    def validate_server(self, settings_page: Any) -> None:
        api_base_url, _ = self._resolve_credentials(settings_page)
        if not api_base_url:
            raise self._build_setup_error("Endpoint URL is empty.")
        if not self.should_manage_server(settings_page):
            return
        compose_file = _RUNTIME_CONFIG["compose_file"]
        if not compose_file.is_file():
            raise self._build_setup_error(
                f"Bundled Docker compose file was not found: {compose_file}",
            )
        self._resolve_compose_command()

    def should_manage_server(self, settings_page: Any) -> bool:
        api_base_url, _ = self._resolve_credentials(settings_page)
        return _normalize_url(api_base_url) == _normalize_url(_RUNTIME_CONFIG["managed_url"])

    def get_translation_cache_identity(
        self,
        settings_page: Any,
    ) -> dict[str, Any] | None:
        """Resolve the managed runtime identity without starting its container."""

        with self._lock:
            api_base_url, model_name = self._resolve_credentials(settings_page)
            if not api_base_url or not self.should_manage_server(settings_page):
                return None
            contract = self._load_runtime_contract(model_name)
            return {
                "managed": True,
                "api_base_url": _normalize_url(api_base_url),
                "model_name": contract.model_name,
                "model_sha256": contract.model_sha256,
                "runtime_fingerprint": contract.fingerprint,
                "runtime_image_ref": contract.image_ref,
                "runtime_image_id": contract.image_id,
                "runtime_command_sha256": contract.command_sha256,
                "runtime_compose_sha256": contract.compose_file_sha256,
                "runtime_manifest_sha256": contract.ready_manifest_sha256,
                "runtime_preparation_version": contract.preparation_version,
                "runtime_options": dict(contract.runtime_options),
            }

    def ensure_server(
        self,
        settings_page: Any,
        *,
        timeout_sec: int = DEFAULT_GEMMA_STARTUP_TIMEOUT_SEC,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> None:
        with self._lock:
            self._startup_cancel_checker = cancel_checker
            api_base_url, model_name = self._resolve_credentials(settings_page)
            if not api_base_url:
                raise self._build_setup_error("Endpoint URL is empty.")

            managed = self.should_manage_server(settings_page)
            if self._is_cancelled(cancel_checker):
                raise OperationCancelledError(
                    "Cancelled while preparing Gemma runtime."
                )
            if managed:
                self.validate_server(settings_page)
            runtime_contract = (
                self._load_runtime_contract(model_name)
                if managed
                else None
            )
            cache_key = self._readiness_cache_key(
                api_base_url,
                model_name,
                managed,
                runtime_contract,
            )
            if self._is_cancelled(cancel_checker):
                self._readiness_cache.discard(cache_key)
                raise OperationCancelledError("Cancelled while preparing Gemma runtime.")

            if cache_key in self._readiness_cache:
                if managed:
                    assert runtime_contract is not None
                    container_state = self._inspect_managed_container_state(
                        runtime_contract
                    )
                    if (
                        container_state["exists"]
                        and container_state["matches"]
                        and container_state["running"]
                        and self._probe_url(_RUNTIME_CONFIG["health_url"])
                    ):
                        self._active_contract = runtime_contract
                        self._managed_active = True
                        self._managed_start_attempted = True
                        self._emit_readiness_cache_hit(
                            progress_callback,
                            managed=True,
                        )
                        return
                    self._readiness_cache.discard(cache_key)
                else:
                    self._emit_readiness_cache_hit(
                        progress_callback,
                        managed=False,
                    )
                    return

            try:
                self._ensure_server_uncached(
                    settings_page,
                    api_base_url=api_base_url,
                    model_name=model_name,
                    managed=managed,
                    runtime_contract=runtime_contract,
                    timeout_sec=timeout_sec,
                    progress_callback=progress_callback,
                    cancel_checker=cancel_checker,
                )
            except (
                LocalServiceConnectionError,
                LocalServiceResponseError,
                LocalServiceSetupError,
                OperationCancelledError,
            ):
                self._readiness_cache.discard(cache_key)
                raise
            else:
                self._readiness_cache.add(cache_key)

    def _ensure_server_uncached(
        self,
        settings_page: Any,
        *,
        api_base_url: str,
        model_name: str,
        managed: bool,
        runtime_contract: GemmaRuntimeContract | None,
        timeout_sec: int,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        cancel_checker: Callable[[], bool] | None,
    ) -> None:
        if managed:
            if runtime_contract is None:
                raise self._build_setup_error(
                    "Gemma runtime contract was not loaded for the managed endpoint."
                )
            self._active_contract = runtime_contract

            self._emit_progress(
                progress_callback,
                status="starting",
                step_key="health_probe",
                message="Gemma 상태를 확인하는 중...",
                detail=f"Endpoint: {_RUNTIME_CONFIG['managed_url']}",
            )
            container_state = self._inspect_managed_container_state(runtime_contract)
            if container_state["exists"] and container_state["matches"] and container_state["running"]:
                self._managed_active = True
                self._managed_start_attempted = True
                self._emit_progress(
                    progress_callback,
                    status="completed",
                    step_key="health_probe",
                    message="이미 실행 중인 Gemma 런타임을 재사용합니다.",
                )
            elif container_state["exists"] and container_state["matches"]:
                self._managed_start_attempted = True
                self._emit_progress(
                    progress_callback,
                    status="starting",
                    step_key="container_start",
                    message="준비된 Gemma 컨테이너를 다시 시작하는 중...",
                    detail=f"docker start {_RUNTIME_CONFIG['container_name']}",
                )
                self._start_managed_container()
                self._managed_active = True
                self._emit_progress(
                    progress_callback,
                    status="completed",
                    step_key="container_start",
                    message="준비된 Gemma 컨테이너 시작 명령을 보냈습니다.",
                )
            else:
                self._managed_start_attempted = True
                recreate = bool(container_state["exists"])
                compose_args = ["up", "-d"]
                step_key = "compose_up"
                message = "Gemma 컨테이너를 시작하는 중..."
                if recreate:
                    compose_args.append("--force-recreate")
                    step_key = "compose_recreate"
                    message = "Gemma 런타임 fingerprint가 달라 컨테이너를 재생성하는 중..."
                self._emit_progress(
                    progress_callback,
                    status="starting",
                    step_key=step_key,
                    message=message,
                    detail=", ".join(container_state.get("mismatch_reasons", [])),
                )
                self._run_compose(
                    *compose_args,
                    step_name="recreate" if recreate else "up",
                    runtime_contract=runtime_contract,
                )
                self._managed_active = True
                self._assert_managed_container_contract(runtime_contract)
                self._emit_progress(
                    progress_callback,
                    status="completed",
                    step_key=step_key,
                    message="Gemma 컨테이너 시작 명령을 보냈습니다.",
                )

            if not self._wait_for_any_probe(
                [_RUNTIME_CONFIG["health_url"], _RUNTIME_CONFIG["models_url"]],
                timeout_sec=timeout_sec,
                progress_callback=progress_callback,
                cancel_checker=cancel_checker,
                step_key="health_wait",
                message="Gemma health 기다리는 중...",
            ):
                raise self._build_setup_error(
                    (
                        f"Timed out while waiting for Gemma at {_RUNTIME_CONFIG['health_url']} "
                        f"after docker compose up -d of {_RUNTIME_CONFIG['compose_file'].name}."
                    ),
                )
            self._managed_active = True
            self._emit_progress(
                progress_callback,
                status="completed",
                step_key="health_wait",
                message="Gemma health 확인이 완료되었습니다.",
            )
            self._validate_model_with_progress(api_base_url, model_name, progress_callback)
            self._prewarm_chat_completion_with_progress(
                api_base_url,
                model_name,
                progress_callback,
                cancel_checker=cancel_checker,
            )
            self._log_runtime_metadata()
            return

        self._emit_progress(
            progress_callback,
            status="starting",
            step_key="health_probe",
            message="Gemma endpoint에 연결을 확인하는 중...",
        )
        if not self._wait_for_any_probe(
            _derive_probe_urls(api_base_url),
            timeout_sec=5,
            progress_callback=progress_callback,
            cancel_checker=cancel_checker,
            step_key="health_probe",
            message="Gemma endpoint에 연결을 확인하는 중...",
        ):
            raise self._build_connection_error(
                f"Unable to reach local Gemma server at {api_base_url}.",
            )
        self._emit_progress(
            progress_callback,
            status="completed",
            step_key="health_probe",
            message="Gemma endpoint 연결이 확인되었습니다.",
        )
        self._validate_model_with_progress(api_base_url, model_name, progress_callback)
        self._prewarm_chat_completion_with_progress(
            api_base_url,
            model_name,
            progress_callback,
            cancel_checker=cancel_checker,
        )

    def shutdown(self) -> dict[str, str | bool]:
        with self._lock:
            self._readiness_cache.clear()
            gpu_release_expected = bool(
                self._managed_active or self._managed_start_attempted
            )
            if not gpu_release_expected:
                self._active_contract = None
                return {
                    "runtime_state": "stopped",
                    "gpu_release_expected": False,
                }
            self._stop_managed_container()
            self._managed_active = False
            self._managed_start_attempted = False
            self._active_contract = None
            return {
                "runtime_state": "stopped",
                "gpu_release_expected": True,
            }

    def _resolve_credentials(self, settings_page: Any) -> tuple[str, str]:
        creds = settings_page.get_credentials("Custom Local Server(Gemma)")
        api_base_url = str((creds or {}).get("api_url", "")).strip().rstrip("/")
        model_name = str((creds or {}).get("model", "")).strip() or DEFAULT_GEMMA_LOCAL_MODEL
        return api_base_url, model_name

    @staticmethod
    def _readiness_cache_key(
        api_base_url: str,
        model_name: str,
        managed: bool,
        runtime_contract: GemmaRuntimeContract | None = None,
    ) -> tuple[str, str, str, str]:
        mode = "managed" if managed else "unmanaged"
        fingerprint = runtime_contract.fingerprint if runtime_contract is not None else ""
        return (
            _normalize_url(api_base_url),
            str(model_name or "").strip(),
            mode,
            fingerprint,
        )

    def _emit_readiness_cache_hit(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        *,
        managed: bool,
    ) -> None:
        message = "이미 실행 중인 Gemma 런타임을 재사용합니다." if managed else "Gemma endpoint 연결이 확인되었습니다."
        self._emit_progress(
            progress_callback,
            status="completed",
            step_key="readiness_cache",
            message=message,
            readiness_cache_hit=True,
        )

    def _resolve_compose_command(self) -> tuple[str, ...]:
        if self._compose_command is not None:
            return self._compose_command
        try:
            self._compose_command = resolve_docker_compose_command(
                cancel_checker=self._startup_cancel_checker,
            )
            return self._compose_command
        except RuntimeError as exc:
            raise self._build_setup_error(
                "Docker Compose is not available. Install Docker Desktop or docker-compose and try again.",
            ) from exc

    def _build_env(
        self,
        model_name: str | None = None,
        runtime_contract: GemmaRuntimeContract | None = None,
    ) -> dict[str, str]:
        env = dict(os.environ)
        contract = runtime_contract or self._active_contract
        if contract is not None:
            env.update(contract.compose_environment())
        else:
            env.setdefault("LLAMA_CPP_IMAGE", DEFAULT_GEMMA_LLAMA_CPP_IMAGE)
            env.setdefault("GEMMA_MODEL_VOLUME", DEFAULT_GEMMA_MODEL_VOLUME)
            model_file = Path(str(model_name or DEFAULT_GEMMA_LOCAL_MODEL)).name
            env["LLAMA_MODEL_FILE"] = model_file
        return env

    def _run_compose(
        self,
        *compose_args: str,
        step_name: str,
        model_name: str | None = None,
        runtime_contract: GemmaRuntimeContract | None = None,
    ) -> None:
        compose_file = Path(_RUNTIME_CONFIG["compose_file"])
        env = self._build_env(model_name, runtime_contract)
        command = [*self._resolve_compose_command(), "-f", str(compose_file), *compose_args]
        try:
            run_docker_command(
                command,
                cwd=compose_file.parent,
                env=env,
                cancel_checker=self._startup_cancel_checker,
            )
            return
        except RuntimeError as exc:
            detail = str(exc).strip()
        requested_image = env.get("LLAMA_CPP_IMAGE", "")
        extra = f"Docker compose {step_name} failed.\n{detail}"
        if requested_image:
            extra = f"{extra}\nRequested image: {requested_image}"
        raise self._build_setup_error(extra)

    def _load_runtime_contract(self, model_name: str) -> GemmaRuntimeContract:
        image_ref = DEFAULT_GEMMA_LLAMA_CPP_IMAGE
        volume_name = str(
            os.environ.get("GEMMA_MODEL_VOLUME", DEFAULT_GEMMA_MODEL_VOLUME)
            or DEFAULT_GEMMA_MODEL_VOLUME
        ).strip()
        try:
            volume_name = validate_gemma_volume_name(volume_name)
            model_file = validate_gemma_model_name(
                str(model_name or DEFAULT_GEMMA_LOCAL_MODEL)
            )
        except GemmaRuntimeContractError as exc:
            raise self._build_setup_error(str(exc)) from exc
        image_id = self._ensure_runtime_image_id(image_ref)
        manifest_bytes, manifest_sha256, observed_model_bytes = self._probe_model_volume(
            volume_name=volume_name,
            model_name=model_file,
            image_ref=image_ref,
        )
        try:
            return build_gemma_runtime_contract(
                manifest_bytes=manifest_bytes,
                manifest_sha256=manifest_sha256,
                observed_model_bytes=observed_model_bytes,
                volume_name=volume_name,
                model_name=model_file,
                image_ref=image_ref,
                image_id=image_id,
                compose_file=_RUNTIME_CONFIG["compose_file"],
                environment=os.environ,
            )
        except (GemmaRuntimeContractError, OSError) as exc:
            raise self._build_setup_error(
                (
                    f"Prepared Gemma runtime validation failed: {exc}\n"
                    "Run scripts/prepare_gemma_runtime.ps1 in Prepare mode, "
                    "or use Verify mode to recompute the model hashes."
                )
            ) from exc

    def _ensure_runtime_image_id(self, image_ref: str) -> str:
        inspect_command = [
            "docker",
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            image_ref,
        ]
        completed = run_docker_command(
            inspect_command,
            check=False,
            cancel_checker=self._startup_cancel_checker,
        )
        image_id = (completed.stdout or "").strip()
        if completed.returncode == 0 and image_id:
            return image_id

        try:
            run_docker_command(
                ["docker", "pull", image_ref],
                cancel_checker=self._startup_cancel_checker,
            )
            completed = run_docker_command(
                inspect_command,
                cancel_checker=self._startup_cancel_checker,
            )
        except RuntimeError as exc:
            raise self._build_setup_error(
                f"Unable to load the pinned Gemma runtime image: {image_ref}\n{exc}"
            ) from exc
        image_id = (completed.stdout or "").strip()
        if not image_id:
            raise self._build_setup_error(
                f"Docker returned no image ID for the pinned Gemma runtime image: {image_ref}"
            )
        return image_id

    def _probe_model_volume(
        self,
        *,
        volume_name: str,
        model_name: str,
        image_ref: str,
    ) -> tuple[bytes, str, int]:
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
                (
                    f"Prepared Gemma model volume does not exist: {volume_name}\n"
                    "Run scripts/prepare_gemma_runtime.ps1 before starting "
                    "the managed endpoint."
                )
            )
        try:
            volume_labels = json.loads(
                (volume_inspection.stdout or "").strip() or "{}"
            )
        except json.JSONDecodeError as exc:
            raise self._build_setup_error(
                f"Unable to parse Docker labels for Gemma volume: {volume_name}"
            ) from exc
        expected_volume_labels = {
            "comic-translate.runtime": "Gemma",
            "comic-translate.preparation-version": str(
                GEMMA_RUNTIME_PREPARATION_VERSION
            ),
        }
        if not isinstance(volume_labels, dict) or any(
            str(volume_labels.get(key, "")) != expected
            for key, expected in expected_volume_labels.items()
        ):
            raise self._build_setup_error(
                (
                    f"Gemma volume labels do not match the preparation contract: "
                    f"{volume_name}\n"
                    f"Expected labels: {expected_volume_labels}\n"
                    f"Actual labels: {volume_labels}"
                )
            )

        shell_script = r'''
set -eu
manifest_path="/models/$READY_MANIFEST"
model_path="/models/$MODEL_FILE"
test -f "$manifest_path"
test -f "$model_path"
printf 'manifest_sha256=%s\n' "$(sha256sum "$manifest_path" | cut -d ' ' -f 1)"
printf 'manifest_base64='
base64 -w 0 "$manifest_path"
printf '\nmodel_bytes=%s\n' "$(stat -c %s "$model_path")"
'''.strip()
        command = [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "-e",
            f"READY_MANIFEST={DEFAULT_GEMMA_READY_MANIFEST}",
            "-e",
            f"MODEL_FILE={model_name}",
            "--mount",
            f"type=volume,source={volume_name},target=/models,readonly",
            "--entrypoint",
            "/bin/sh",
            image_ref,
            "-ec",
            shell_script,
        ]
        completed = run_docker_command(
            command,
            check=False,
            cancel_checker=self._startup_cancel_checker,
        )
        if completed.returncode != 0:
            detail = ((completed.stderr or "") + "\n" + (completed.stdout or "")).strip()
            raise self._build_setup_error(
                (
                    f"Prepared Gemma model volume is unavailable or incomplete: {volume_name}\n"
                    f"Configured model: {model_name}\n{detail}\n"
                    "Run scripts/prepare_gemma_runtime.ps1 before starting the managed endpoint."
                )
            )

        values: dict[str, str] = {}
        for line in (completed.stdout or "").splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key.strip()] = value.strip()
        try:
            manifest_bytes = base64.b64decode(values["manifest_base64"], validate=True)
            manifest_sha256 = values["manifest_sha256"].lower()
            observed_model_bytes = int(values["model_bytes"])
        except (KeyError, ValueError, binascii.Error) as exc:
            raise self._build_setup_error(
                f"Unable to parse the prepared Gemma volume probe output: {completed.stdout}"
            ) from exc
        return manifest_bytes, manifest_sha256, observed_model_bytes

    def _inspect_managed_container_state(
        self,
        runtime_contract: GemmaRuntimeContract,
    ) -> dict[str, Any]:
        completed = run_docker_command(
            ["docker", "inspect", str(_RUNTIME_CONFIG["container_name"])],
            check=False,
            cancel_checker=self._startup_cancel_checker,
        )
        if completed.returncode != 0:
            return {
                "exists": False,
                "running": False,
                "matches": False,
                "mismatch_reasons": ["container-missing"],
            }
        try:
            payload = json.loads(completed.stdout or "[]")
            inspection = payload[0]
        except (json.JSONDecodeError, IndexError, TypeError) as exc:
            raise self._build_setup_error(
                f"Unable to parse Docker inspection for {_RUNTIME_CONFIG['container_name']}."
            ) from exc
        mismatch_reasons = container_contract_mismatch_reasons(
            inspection,
            runtime_contract,
        )
        state = inspection.get("State")
        running = bool(state.get("Running")) if isinstance(state, dict) else False
        return {
            "exists": True,
            "running": running,
            "matches": not mismatch_reasons,
            "mismatch_reasons": mismatch_reasons,
        }

    def _assert_managed_container_contract(
        self,
        runtime_contract: GemmaRuntimeContract,
    ) -> None:
        state = self._inspect_managed_container_state(runtime_contract)
        if state["exists"] and state["matches"]:
            return
        raise self._build_setup_error(
            (
                "Gemma container was created but does not match the prepared runtime contract.\n"
                f"Mismatches: {', '.join(state.get('mismatch_reasons', []))}"
            )
        )

    def _start_managed_container(self) -> None:
        try:
            run_docker_command(
                ["docker", "start", str(_RUNTIME_CONFIG["container_name"])],
                cancel_checker=self._startup_cancel_checker,
            )
        except RuntimeError as exc:
            raise self._build_setup_error(
                f"Failed to start the prepared Gemma container.\n{exc}"
            ) from exc

    def _stop_managed_container(self) -> None:
        watch_for_late_start = bool(
            self._managed_start_attempted and not self._managed_active
        )
        try:
            run_docker_command(
                [
                    "docker",
                    "stop",
                    "--timeout",
                    str(DEFAULT_MANAGED_RUNTIME_STOP_TIMEOUT_SEC),
                    str(_RUNTIME_CONFIG["container_name"]),
                ],
                check=False,
                timeout_sec=DEFAULT_MANAGED_RUNTIME_STOP_TIMEOUT_SEC + 15.0,
            )
        except RuntimeError as exc:
            raise self._build_setup_error(
                f"Failed to stop the managed Gemma container.\n{exc}"
            ) from exc

        deadline = time.monotonic() + (
            LATE_START_STOP_GRACE_SEC if watch_for_late_start else 0.0
        )
        while True:
            running = self._inspect_managed_container_running()
            if running:
                try:
                    run_docker_command(
                        [
                            "docker",
                            "stop",
                            "--timeout",
                            str(DEFAULT_MANAGED_RUNTIME_STOP_TIMEOUT_SEC),
                            str(_RUNTIME_CONFIG["container_name"]),
                        ],
                        timeout_sec=DEFAULT_MANAGED_RUNTIME_STOP_TIMEOUT_SEC + 15.0,
                    )
                except RuntimeError as exc:
                    raise self._build_setup_error(
                        f"Failed to stop the managed Gemma container.\n{exc}"
                    ) from exc
                running = self._inspect_managed_container_running()
            if time.monotonic() >= deadline:
                if running:
                    raise self._build_setup_error(
                        "Docker reported that the managed Gemma container is still running after stop."
                    )
                return
            time.sleep(LATE_START_STOP_POLL_SEC)

    def _inspect_managed_container_running(self) -> bool:
        try:
            inspection = run_docker_command(
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{.State.Running}}",
                    str(_RUNTIME_CONFIG["container_name"]),
                ],
                check=False,
                timeout_sec=15.0,
            )
        except RuntimeError as exc:
            raise self._build_setup_error(
                f"Failed to verify the managed Gemma container state.\n{exc}"
            ) from exc
        state = (inspection.stdout or "").strip().lower()
        if inspection.returncode == 0 and state == "true":
            return True
        if inspection.returncode == 0 and state == "false":
            return False
        detail = (
            (inspection.stderr or "") + "\n" + (inspection.stdout or "")
        ).strip()
        normalized_detail = detail.lower()
        if inspection.returncode != 0 and (
            "no such object" in normalized_detail
            or "no such container" in normalized_detail
        ):
            return False
        if inspection.returncode == 0:
            detail = f"Unexpected docker inspect state: {state or '<empty>'}"
        raise self._build_setup_error(
            "Docker could not verify whether the managed Gemma container stopped."
            + (f"\n{detail}" if detail else "")
        )

    def _wait_for_any_probe(
        self,
        urls: list[str],
        *,
        timeout_sec: int,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_checker: Callable[[], bool] | None = None,
        step_key: str = "health_wait",
        message: str = "Gemma health 기다리는 중...",
    ) -> bool:
        deadline = time.monotonic() + max(timeout_sec, 1)
        started = time.monotonic()
        while time.monotonic() < deadline:
            if self._is_cancelled(cancel_checker):
                raise OperationCancelledError("Cancelled while waiting for Gemma runtime.")
            for url in urls:
                if self._probe_url(url):
                    return True
            self._emit_progress(
                progress_callback,
                status="waiting_health",
                step_key=step_key,
                message=message,
                elapsed_sec=time.monotonic() - started,
                detail=f"Waiting for {urls[0]}",
            )
            time.sleep(1)
        return False

    @staticmethod
    def _probe_url(url: str) -> bool:
        try:
            with urlopen(url, timeout=2) as response:
                return getattr(response, "status", 200) < 500
        except (URLError, OSError, ValueError):
            return False

    def _validate_model_with_progress(
        self,
        api_base_url: str,
        expected_model: str,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._emit_progress(
            progress_callback,
            status="starting",
            step_key="models_check",
            message="Gemma 모델 목록을 확인하는 중...",
        )
        self._validate_loaded_model(api_base_url, expected_model)
        self._emit_progress(
            progress_callback,
            status="completed",
            step_key="models_check",
            message=f"Gemma 모델 확인 완료: {expected_model}",
        )

    def _validate_loaded_model(self, api_base_url: str, expected_model: str) -> None:
        base = _normalize_url(api_base_url)
        models_url = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"
        try:
            with urlopen(models_url, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return

        model_ids: list[str] = []
        for entry in payload.get("data", []):
            if isinstance(entry, dict):
                model_id = str(entry.get("id", "")).strip()
                if model_id:
                    model_ids.append(model_id)
        expected_model_name = Path(str(expected_model or "")).name
        loaded_model_names = {Path(model_id).name for model_id in model_ids}
        if (
            model_ids
            and expected_model
            and expected_model not in model_ids
            and expected_model_name not in loaded_model_names
        ):
            raise LocalServiceResponseError(
                (
                    "Gemma server is reachable but loaded models do not match the configured model.\n"
                    f"Configured model: {expected_model}\nAvailable models: {', '.join(model_ids)}"
                ),
                service_name="Gemma",
                settings_page_name=_RUNTIME_CONFIG["settings_page_name"],
            )

    def _prewarm_chat_completion_with_progress(
        self,
        api_base_url: str,
        model_name: str,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> None:
        self._emit_progress(
            progress_callback,
            status="starting",
            step_key="chat_prewarm",
            message="Gemma 첫 번역 요청을 예열하는 중...",
        )
        self._prewarm_chat_completion(
            api_base_url,
            model_name,
            cancel_checker=cancel_checker,
        )
        self._emit_progress(
            progress_callback,
            status="completed",
            step_key="chat_prewarm",
            message="Gemma 첫 번역 요청 예열이 완료되었습니다.",
        )

    def _prewarm_chat_completion(
        self,
        api_base_url: str,
        model_name: str,
        *,
        timeout_sec: int = 90,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> None:
        base = _normalize_url(api_base_url)
        chat_url = f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Return exactly one JSON object."}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "{\"translation\":\"ok\"}"}],
                },
            ],
            "temperature": 0.0,
            "max_completion_tokens": 32,
            "response_format": {"type": "json_object"},
        }
        request = Request(
            chat_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        def request_chat_completion() -> None:
            with urlopen(request, timeout=timeout_sec) as response:
                response.read()

        try:
            self._run_interruptible_io(
                request_chat_completion,
                timeout_sec=timeout_sec,
                cancel_checker=cancel_checker,
                cancel_message="Cancelled while prewarming Gemma chat completion.",
            )
        except OperationCancelledError:
            raise
        except Exception as exc:
            raise self._build_connection_error(
                f"Gemma chat prewarm failed at {chat_url}: {exc}"
            ) from exc

    def _log_runtime_metadata(self) -> None:
        try:
            image_ref = (
                self._active_contract.image_ref
                if self._active_contract is not None
                else DEFAULT_GEMMA_LLAMA_CPP_IMAGE
            )
            runtime = inspect_llama_cpp_runtime(
                image_ref=image_ref,
                container_name=str(_RUNTIME_CONFIG["container_name"]),
                cancel_checker=self._startup_cancel_checker,
            )
        except OperationCancelledError:
            raise
        except Exception:
            logger.warning("Failed to inspect llama.cpp runtime metadata for Gemma.", exc_info=True)
            return
        logger.info(
            "Gemma runtime ready: image=%s digest=%s version=%s",
            runtime.get("llama_cpp_image", ""),
            runtime.get("llama_cpp_digest", ""),
            runtime.get("llama_cpp_version", ""),
        )

    @staticmethod
    def _run_interruptible_io(
        operation: Callable[[], None],
        *,
        timeout_sec: float,
        cancel_checker: Callable[[], bool] | None,
        cancel_message: str,
    ) -> None:
        results: queue.Queue[BaseException | None] = queue.Queue(maxsize=1)

        def runner() -> None:
            try:
                operation()
            except BaseException as exc:
                results.put(exc)
            else:
                results.put(None)

        worker = threading.Thread(
            target=runner,
            name="ct-gemma-prewarm-io",
            daemon=True,
        )
        worker.start()
        deadline = time.monotonic() + max(0.1, float(timeout_sec))
        while True:
            if LocalGemmaRuntimeManager._is_cancelled(cancel_checker):
                raise OperationCancelledError(cancel_message)
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(
                    f"Gemma prewarm timed out after {float(timeout_sec):.1f}s."
                )
            try:
                result = results.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue
            if result is not None:
                raise result
            return

    def _emit_progress(self, progress_callback: Callable[[dict[str, Any]], None] | None, **payload: Any) -> None:
        if progress_callback is None:
            return
        event = {
            "phase": "gemma_startup",
            "service": "gemma",
            "status": payload.pop("status", "running"),
            "step_key": payload.pop("step_key", "health_wait"),
            "message": payload.pop("message", "Gemma health 기다리는 중..."),
            "detail": payload.pop("detail", ""),
        }
        event.update(payload)
        try:
            progress_callback(event)
        except Exception:
            logger.debug("Failed to emit Gemma progress event.", exc_info=True)

    @staticmethod
    def _is_cancelled(cancel_checker: Callable[[], bool] | None) -> bool:
        try:
            return bool(cancel_checker and cancel_checker())
        except Exception:
            return False

    def _build_setup_error(self, detail: str) -> LocalServiceSetupError:
        extra = detail.strip()
        extra = f"{extra}\nRequested engine: Gemma\nExpected endpoint: {_RUNTIME_CONFIG['managed_url']}"
        extra = f"{extra}\nCompose file: {_RUNTIME_CONFIG['compose_file']}"
        return LocalServiceSetupError(
            extra.strip(),
            service_name="Gemma",
            settings_page_name=_RUNTIME_CONFIG["settings_page_name"],
        )

    def _build_connection_error(self, detail: str) -> LocalServiceConnectionError:
        return LocalServiceConnectionError(
            detail.strip(),
            service_name="Gemma",
            settings_page_name=_RUNTIME_CONFIG["settings_page_name"],
        )
