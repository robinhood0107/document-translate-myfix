from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import queue
import threading
import time
from contextlib import nullcontext
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
from modules.utils.local_llama_router import (
    DEFAULT_GEMMA_ROUTER_MODEL,
    LocalLlamaRouterCoordinator,
    RouterModelMaterial,
    RouterPair,
    RouterRuntimeSpec,
)
from modules.utils.local_llama_router.contracts import (
    ROUTER_GEMMA_HOST_PORT,
    router_pair_for_engine_key,
)
from modules.utils.llama_cpp_runtime import (
    DEFAULT_MANAGED_RUNTIME_STOP_TIMEOUT_SEC,
    inspect_llama_cpp_runtime,
    resolve_docker_compose_command,
    run_docker_command,
)
from modules.utils.windows_installation import active_windows_install_state

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_GEMMA_HEALTH_URL = "http://127.0.0.1:18080/health"
DEFAULT_GEMMA_MODELS_URL = "http://127.0.0.1:18080/v1/models"
DEFAULT_GEMMA_SETTINGS_PAGE = "Gemma Local Server Settings"
DEFAULT_GEMMA_STARTUP_TIMEOUT_SEC = 420
LATE_START_STOP_GRACE_SEC = 3.0
LATE_START_STOP_POLL_SEC = 0.25

# 프리페치 전용 컨테이너. 이름을 고정해 임의 이름 컨테이너가 생기지 않게 한다.
GEMMA_PAGE_CACHE_PREFETCH_CONTAINER = "comic-translate-gemma-cache-warm"
# 크기를 알아내는 컨테이너. 프리페치와 이름을 나눠 서로 정리를 방해하지 않게 한다.
GEMMA_MODEL_SIZE_PROBE_CONTAINER = "comic-translate-gemma-cache-warm-size"

class _GemmaVolumeNotProvisioned(RuntimeError):
    """준비 볼륨이 아예 없거나 계약된 파일이 빠져 있다.

    준비 스크립트를 돌리면 해결되는 상태다. 볼륨 라벨 불일치처럼 사람의 판단을
    요구하는 상태와 구별하려고 따로 둔다.
    """


_RUNTIME_CONFIG = {
    "compose_file": ROOT_DIR / "docker-compose.yaml",
    "managed_url": DEFAULT_GEMMA_LOCAL_ENDPOINT,
    "health_url": DEFAULT_GEMMA_HEALTH_URL,
    "models_url": DEFAULT_GEMMA_MODELS_URL,
    "settings_page_name": DEFAULT_GEMMA_SETTINGS_PAGE,
    "container_name": "gemma-local-server",
}


def _available_page_cache_bytes(
    *,
    cancel_checker: Callable[[], bool] | None = None,
) -> int:
    """모델을 담을 수 있는 여유 메모리(바이트). 알 수 없으면 0.

    호스트가 아니라 **Docker 가 도는 리눅스 쪽** 여유를 본다. Windows 에서 모델
    페이지 캐시는 호스트 RAM 이 아니라 WSL VM 안에 잡히므로, 호스트 여유를 보면
    엉뚱한 판단을 한다.
    """

    completed = run_docker_command(
        [
            "docker",
            "run",
            "--rm",
            "--name",
            GEMMA_MODEL_SIZE_PROBE_CONTAINER + "-mem",
            "--pull",
            "never",
            "--entrypoint",
            "/bin/sh",
            DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
            "-ec",
            "awk '/^MemAvailable:/ {print $2}' /proc/meminfo",
        ],
        check=False,
        cancel_checker=cancel_checker,
    )
    if completed.returncode != 0:
        return 0
    try:
        # /proc/meminfo 는 kB 단위다.
        return int(str(completed.stdout or "").strip()) * 1024
    except (TypeError, ValueError):
        return 0


def _volume_model_size_bytes(
    *,
    volume_name: str,
    model_name: str,
    image_ref: str,
    cancel_checker: Callable[[], bool] | None = None,
) -> int:
    """볼륨 안 모델 파일 크기(바이트). 알 수 없으면 0."""

    from modules.utils.llama_cpp_runtime import remove_named_container

    remove_named_container(GEMMA_MODEL_SIZE_PROBE_CONTAINER)
    completed = run_docker_command(
        [
            "docker",
            "run",
            "--name",
            GEMMA_MODEL_SIZE_PROBE_CONTAINER,
            "--rm",
            "--pull",
            "never",
            "-e",
            f"MODEL_FILE={model_name}",
            "--mount",
            f"type=volume,source={volume_name},target=/models,readonly",
            "--entrypoint",
            "/bin/sh",
            image_ref,
            "-ec",
            'stat -c %s "/models/$MODEL_FILE"',
        ],
        check=False,
        cancel_checker=cancel_checker,
    )
    if completed.returncode != 0:
        return 0
    try:
        return int(str(completed.stdout or "").strip())
    except (TypeError, ValueError):
        return 0


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
    def __init__(
        self,
        *,
        router_coordinator: LocalLlamaRouterCoordinator | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._compose_command: tuple[str, ...] | None = None
        self._managed_active = False
        self._managed_start_attempted = False
        self._active_contract: GemmaRuntimeContract | None = None
        self._runtime_contract_cache: GemmaRuntimeContract | None = None
        self._runtime_contract_cache_key: tuple[str, str, str, str] | None = None
        self._readiness_cache: set[tuple[str, str, str, str]] = set()
        self._startup_cancel_checker: Callable[[], bool] | None = None
        # Keep the current startup callback available to nested read-only
        # probes without threading it through every helper signature.
        self._startup_progress_callback: Callable[[dict[str, Any]], None] | None = None
        self._router_coordinator = router_coordinator
        self._router_spec: RouterRuntimeSpec | None = None
        self._router_pair: RouterPair | None = None

    def set_router_spec(self, spec: RouterRuntimeSpec | None) -> None:
        self._router_spec = spec
        if spec is None:
            self._router_pair = None

    @staticmethod
    def router_credentials(settings_page: Any) -> tuple[str, str]:
        """Read the configured endpoint without legacy URL normalization."""

        creds = settings_page.get_credentials("Custom Local Server(Gemma)")
        api_base_url = str((creds or {}).get("api_url", "")).strip()
        model_name = (
            str((creds or {}).get("model", "")).strip()
            or DEFAULT_GEMMA_LOCAL_MODEL
        )
        return api_base_url, model_name

    def router_model_material(
        self,
        settings_page: Any,
    ) -> tuple[RouterModelMaterial, str]:
        """Return only prepared Gemma evidence; never start its old container."""

        _endpoint, model_name = self.router_credentials(settings_page)
        if model_name != DEFAULT_GEMMA_ROUTER_MODEL:
            raise self._build_setup_error(
                "Router requires the product-default Gemma model alias.",
            )
        contract = self._load_runtime_contract(model_name)
        material = RouterModelMaterial(
            alias=contract.model_name,
            model_file=contract.model_name,
            model_sha256=contract.model_sha256,
            volume_name=contract.volume_name,
            ready_manifest_sha256=contract.ready_manifest_sha256,
            source_fingerprint=contract.fingerprint,
            runtime_options=dict(contract.runtime_options),
            preparation_version=contract.preparation_version,
        )
        return material, contract.image_ref

    def _router_pair_for_server(self, settings_page: Any) -> RouterPair | None:
        coordinator = self._router_coordinator
        spec = self._router_spec
        if coordinator is None or spec is None:
            return None
        endpoint, model = self.router_credentials(settings_page)
        pair = coordinator.current_pair_for_gemma(endpoint, model)
        return pair if pair == spec.pair else None

    def router_is_active(self) -> bool:
        pair = self._router_pair
        coordinator = self._router_coordinator
        return bool(
            pair is not None
            and coordinator is not None
            and coordinator.snapshot().pair == pair.kind.value
        )

    def release_separate_server_for_router(self) -> bool:
        """Router보다 먼저 이 제품의 separate-server Gemma 컨테이너를 정지한다.

        Router는 18080을 publish하고, separate-server Gemma 컨테이너도 같은 포트를
        쥔다. 제품 컨테이너만, 그리고 Docker가 실행 중이라고 보고할 때만 정지하므로
        제품 소유가 아닌 listener는 여전히 Router adapter의 명시적 ownership 오류로
        드러난다.
        """

        with self._lock:
            if self._router_pair is not None:
                return False
            if not self._inspect_managed_container_running():
                return False
            self._stop_managed_container()
            self._managed_active = False
            self._managed_start_attempted = False
            self._readiness_cache.clear()
            logger.info(
                "Stopped the separate-server Gemma container so the Router can bind port 18080."
            )
            return True

    def _release_stale_router_gemma_port(
        self,
        api_base_url: str,
        *,
        cancel_checker: Callable[[], bool] | None,
    ) -> None:
        """Gemma 호스트 포트를 쥔 남은 Router 컨테이너를 회수한다.

        Router는 OCR 포트와 함께 18080도 publish하므로, 이전 프로세스가 남긴
        컨테이너는 separate-server Gemma compose의 바인딩을 영구히 막는다. 정확한
        기본 Gemma 포트만, 그리고 Router 소유 컨테이너에서만 회수한다.
        """

        coordinator = self._router_coordinator
        if coordinator is None or self._router_pair is not None:
            return
        # 모든 pair가 하나의 공유 Gemma 호스트 포트를 publish하므로, 어느 pair의
        # 포트를 회수해도 다른 pair가 남긴 Gemma listener까지 함께 풀린다.
        pair = router_pair_for_engine_key("PaddleOCR VL")
        if pair is None:
            return
        try:
            configured_port = urlparse(str(api_base_url or "").strip()).port
        except ValueError:
            return
        if configured_port != ROUTER_GEMMA_HOST_PORT:
            return
        try:
            released = coordinator.release_owned_pair_ports(
                pair,
                cancel_checker=cancel_checker,
            )
        except OperationCancelledError:
            raise
        except Exception as exc:
            raise self._build_setup_error(str(exc)) from exc
        if released:
            logger.info(
                "Released leftover Router container(s) %s so the separate-server "
                "Gemma runtime can bind port %s.",
                ", ".join(released),
                pair.gemma_port,
            )

    def _finish_router_for_selection_change(
        self,
        *,
        resource_arbiter: Any | None,
        runtime_service: str,
        cancel_checker: Callable[[], bool] | None,
    ) -> None:
        """Terminally release the owned Router before a custom route starts."""

        coordinator = self._router_coordinator
        pair = self._router_pair
        if coordinator is None or pair is None:
            return
        snapshot = coordinator.snapshot()
        if snapshot.pair == pair.kind.value:
            service = runtime_service or "gemma"
            if str(getattr(snapshot, "loaded_model", "") or "") == pair.ocr_alias:
                service = (
                    "paddleocr_vl"
                    if pair.ocr_engine_key == "PaddleOCR VL"
                    else "paddleocr_vl_spotting"
                )
            coordinator.finish(
                arbiter=resource_arbiter,
                service=service,
                stop_container=True,
                cancel_checker=cancel_checker,
            )
        self._router_pair = None
        self._readiness_cache.clear()

    def router_inference_lease(self, settings_page: Any) -> Any:
        """Return the Gemma Router request lease without holding manager state.

        Standalone Gemma and custom endpoints continue through their existing
        separate-server paths.  Once the shared pair is selected, a missing
        Gemma load is a setup failure rather than permission to issue an
        unprotected HTTP request to the Router port.
        """

        with self._lock:
            pair = self._router_pair_for_server(settings_page)
            if pair is None:
                return nullcontext()
            coordinator = self._router_coordinator
            if coordinator is None or self._router_pair != pair:
                raise self._build_setup_error(
                    "Router inference was requested before the matching Gemma model lease was loaded."
                )
            return coordinator.inference_lease(
                pair=pair,
                model_alias=DEFAULT_GEMMA_ROUTER_MODEL,
            )

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
            router_pair = self._router_pair_for_server(settings_page)
            if router_pair is not None and self._router_coordinator is not None:
                snapshot = self._router_coordinator.snapshot()
                if snapshot.fingerprint:
                    return {
                        "managed": True,
                        "router": True,
                        "router_pair": router_pair.kind.value,
                        "router_fingerprint": snapshot.fingerprint,
                        "router_model_generation": snapshot.model_generation,
                        "api_base_url": self.router_credentials(settings_page)[0],
                        "model_name": DEFAULT_GEMMA_ROUTER_MODEL,
                    }
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

    def prefetch_model_into_page_cache(
        self,
        settings_page: Any,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """모델 파일을 순차로 읽어 호스트 페이지 캐시에 올린다. GPU 는 쓰지 않는다.

        llama.cpp 는 mmap 으로 GGUF 를 읽는다. 즉 적재 시간은 페이지 폴트가 디스크를
        때리느냐 캐시에 맞느냐로 갈린다. 실측(13.58 GB, Docker Desktop WSL VM):

        * 캐시 미적중 순차 읽기 7.99초 (1,825 MB/s)
        * 캐시 적중 순차 읽기 0.72~0.88초 (약 20 GB/s)
        * 실제 Gemma 적재는 첫 실행 43.95초, 이후 4.53~12.28초

        적재가 순차 읽기보다 5배 넘게 느린 이유는 mmap 페이지 폴트의 접근 패턴이
        순차가 아니기 때문이다. 그래서 순차 읽기로 캐시를 먼저 채우면 폴트가 전부
        캐시 적중이 되어 첫 실행 적재가 재실행 수준으로 내려간다.

        이 작업은 디스크에서 RAM 으로만 옮기므로 OCR sweep 과 겹쳐도 VRAM 을 다투지
        않는다. 기존 볼륨 프로브와 같은 기법을 쓴다 — 핀된 런타임 이미지,
        ``--pull never``, 읽기 전용 볼륨 마운트.
        """

        with self._lock:
            self._startup_cancel_checker = cancel_checker
            try:
                _endpoint, configured_model = self.router_credentials(settings_page)
                model_name = validate_gemma_model_name(
                    str(configured_model or DEFAULT_GEMMA_LOCAL_MODEL)
                )
                volume_name = self._configured_volume_name()
                image_ref = DEFAULT_GEMMA_LLAMA_CPP_IMAGE
            except Exception as exc:
                logger.info("Gemma 페이지 캐시 프리페치를 건너뜁니다: %s", exc)
                return {"performed": False, "reason": "contract-unavailable"}

            headroom = _available_page_cache_bytes(cancel_checker=cancel_checker)
            model_bytes = _volume_model_size_bytes(
                volume_name=volume_name,
                model_name=model_name,
                image_ref=image_ref,
                cancel_checker=cancel_checker,
            )
            if model_bytes and headroom and headroom < model_bytes:
                # 여유보다 큰 파일을 읽어 캐시를 채우면 다른 단계가 쓰던 캐시를
                # 밀어낸다. 이득보다 손해가 크다.
                logger.info(
                    "Gemma 페이지 캐시 프리페치를 건너뜁니다: 여유 %.1f GB < 모델 %.1f GB",
                    headroom / 1e9,
                    model_bytes / 1e9,
                )
                return {
                    "performed": False,
                    "reason": "insufficient-memory",
                    "available_bytes": headroom,
                    "model_bytes": model_bytes,
                }

            from modules.utils.llama_cpp_runtime import remove_named_container

            remove_named_container(GEMMA_PAGE_CACHE_PREFETCH_CONTAINER)
            started_at = time.perf_counter()
            completed = run_docker_command(
                [
                    "docker",
                    "run",
                    "--name",
                    GEMMA_PAGE_CACHE_PREFETCH_CONTAINER,
                    "--rm",
                    "--pull",
                    "never",
                    "-e",
                    f"MODEL_FILE={model_name}",
                    "--mount",
                    f"type=volume,source={volume_name},target=/models,readonly",
                    "--entrypoint",
                    "/bin/sh",
                    image_ref,
                    "-ec",
                    'test -f "/models/$MODEL_FILE" && '
                    'cat "/models/$MODEL_FILE" > /dev/null',
                ],
                check=False,
                cancel_checker=cancel_checker,
            )
            elapsed_sec = time.perf_counter() - started_at
            if completed.returncode != 0:
                # 프리페치는 최적화일 뿐이다. 실패해도 적재는 그대로 된다.
                logger.info(
                    "Gemma 페이지 캐시 프리페치가 실패했습니다(코드 %s). 계속 진행합니다.",
                    completed.returncode,
                )
                return {
                    "performed": False,
                    "reason": "docker-failed",
                    "elapsed_sec": elapsed_sec,
                }
            logger.info(
                "Gemma 페이지 캐시 프리페치 완료: %.2f초, %.1f GB",
                elapsed_sec,
                (model_bytes or 0) / 1e9,
            )
            return {
                "performed": True,
                "elapsed_sec": elapsed_sec,
                "model_bytes": model_bytes,
            }

    def _configured_volume_name(self) -> str:
        volume_name = str(
            os.environ.get("GEMMA_MODEL_VOLUME", DEFAULT_GEMMA_MODEL_VOLUME)
            or DEFAULT_GEMMA_MODEL_VOLUME
        ).strip()
        return validate_gemma_volume_name(volume_name)

    def ensure_server(
        self,
        settings_page: Any,
        *,
        timeout_sec: int = DEFAULT_GEMMA_STARTUP_TIMEOUT_SEC,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_checker: Callable[[], bool] | None = None,
        resource_arbiter: Any | None = None,
        runtime_service: str = "",
    ) -> None:
        with self._lock:
            self._startup_cancel_checker = cancel_checker
            self._startup_progress_callback = progress_callback
            router_pair = self._router_pair_for_server(settings_page)
            if router_pair is not None:
                coordinator = self._router_coordinator
                spec = self._router_spec
                if coordinator is None or spec is None:
                    raise self._build_setup_error("Router state is unavailable for Gemma.")
                try:
                    coordinator.load(
                        spec,
                        DEFAULT_GEMMA_ROUTER_MODEL,
                        arbiter=resource_arbiter,
                        service=runtime_service or "gemma",
                        cancel_checker=cancel_checker,
                    )
                except OperationCancelledError:
                    # Do not translate a user cancellation into a Gemma setup
                    # error after the coordinator has cleaned up its owned
                    # Router container.
                    raise
                except Exception as exc:
                    raise self._build_setup_error(str(exc)) from exc
                self._router_pair = router_pair
                self._readiness_cache.clear()
                return
            # Query strings, fragments, a different host/path/port, or a
            # non-default model are all user-managed routes.  A Router that
            # was active for the previous exact default route must be stopped
            # before this manager can touch the separate-server path.
            self._finish_router_for_selection_change(
                resource_arbiter=resource_arbiter,
                runtime_service=runtime_service or "gemma",
                cancel_checker=cancel_checker,
            )
            api_base_url, model_name = self._resolve_credentials(settings_page)
            if not api_base_url:
                raise self._build_setup_error("Endpoint URL is empty.")

            self._release_stale_router_gemma_port(
                api_base_url,
                cancel_checker=cancel_checker,
            )

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

    def shutdown(
        self,
        *,
        resource_arbiter: Any | None = None,
        runtime_service: str = "",
        cancel_checker: Callable[[], bool] | None = None,
        allow_foreign_owner_teardown: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            if self.router_is_active() and self._router_coordinator is not None:
                evidence = self._router_coordinator.finish(
                    arbiter=resource_arbiter,
                    service=runtime_service or "gemma",
                    stop_container=True,
                    cancel_checker=cancel_checker,
                    allow_foreign_owner_teardown=allow_foreign_owner_teardown,
                )
                snapshot = self._router_coordinator.snapshot()
                self._router_pair = None
                self._readiness_cache.clear()
                return {
                    "runtime_state": "stopped",
                    "gpu_release_expected": False,
                    "router_release_evidence": (
                        evidence.vram if evidence is not None else {"required": False}
                    ),
                    "router_model_generation": snapshot.model_generation,
                }
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

    def _load_runtime_contract(
        self,
        model_name: str,
    ) -> GemmaRuntimeContract:
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
        install_state = active_windows_install_state()
        sealed_image = install_state.get("llama_image")
        sealed_image_id = ""
        if (
            isinstance(sealed_image, dict)
            and str(sealed_image.get("ref") or "") == image_ref
        ):
            sealed_image_id = str(sealed_image.get("id") or "").strip()
        image_id = sealed_image_id or self._ensure_runtime_image_id(image_ref)
        cache_key = (image_ref, image_id, volume_name, model_file)
        if (
            self._runtime_contract_cache is not None
            and self._runtime_contract_cache_key == cache_key
        ):
            return self._runtime_contract_cache
        try:
            (
                manifest_bytes,
                manifest_sha256,
                observed_model_bytes,
            ) = self._probe_model_volume(
                volume_name=volume_name,
                model_name=model_file,
                image_ref=image_ref,
            )
        except _GemmaVolumeNotProvisioned as exc:
            raise self._build_setup_error(
                f"{exc}\nRun the matching setup BAT and start Comic Translate again."
            ) from exc

        def build(contract_image_ref: str, contract_image_id: str) -> GemmaRuntimeContract:
            return build_gemma_runtime_contract(
                manifest_bytes=manifest_bytes,
                manifest_sha256=manifest_sha256,
                observed_model_bytes=observed_model_bytes,
                volume_name=volume_name,
                model_name=model_file,
                image_ref=contract_image_ref,
                image_id=contract_image_id,
                compose_file=_RUNTIME_CONFIG["compose_file"],
                environment=os.environ,
            )

        try:
            contract = build(image_ref, image_id)
        except (GemmaRuntimeContractError, OSError) as exc:
            raise self._build_setup_error(
                (
                    f"Prepared Gemma runtime validation failed: {exc}\n"
                    "Run the matching setup BAT and start Comic Translate again."
                )
            ) from exc
        self._runtime_contract_cache = contract
        self._runtime_contract_cache_key = cache_key
        return contract

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
        raise self._build_setup_error(
            f"The setup-sealed Gemma runtime image is missing: {image_ref}"
        )

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
            raise _GemmaVolumeNotProvisioned(
                f"Prepared Gemma model volume does not exist: {volume_name}"
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
        from modules.utils.llama_cpp_runtime import remove_named_container

        remove_named_container("comic-translate-gemma-volume-probe")
        command = [
            "docker",
            "run",
            "--name",
            "comic-translate-gemma-volume-probe",
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
            raise _GemmaVolumeNotProvisioned(
                f"Prepared Gemma model volume is unavailable or incomplete: {volume_name}\n"
                f"Configured model: {model_name}\n{detail}"
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
