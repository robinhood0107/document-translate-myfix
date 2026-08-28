from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from modules.ocr.selection import is_local_ocr_engine
from modules.ocr.managed_backend_policy import (
    MANAGED_LOCAL_INFERENCE_BACKEND,
    sanitize_managed_runtime_environment,
)
from modules.ocr.paddle_crop.runtime import (
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
from modules.ocr.paddle_crop.transport import (
    DEFAULT_PADDLE_DIRECT_SERVER_URL,
    direct_transport_identity,
)
from modules.ocr.hunyuan_llamacpp_runtime_contract import (
    DEFAULT_HUNYUAN_OCR_LLAMA_CPP_IMAGE,
    DEFAULT_HUNYUAN_OCR_MODEL_VOLUME,
    DEFAULT_HUNYUAN_OCR_READY_MANIFEST,
    DEFAULT_HUNYUAN_OCR_RUNTIME_OPTIONS,
    HUNYUAN_OCR_MMPROJ_NAME,
    HUNYUAN_OCR_MODEL_NAME,
    HUNYUAN_OCR_MODEL_SPECS,
    HUNYUAN_OCR_RUNTIME_PREPARATION_VERSION,
    HunyuanOCRRuntimeContract,
    HunyuanOCRRuntimeContractError,
    build_hunyuan_ocr_runtime_contract,
    resolve_hunyuan_ocr_runtime_options,
    validate_hunyuan_ocr_volume_name,
)
from modules.ocr.mangalmm_full_page.runtime import (
    DEFAULT_MANGALMM_LLAMA_CPP_IMAGE,
    DEFAULT_MANGALMM_MODEL_VOLUME,
    DEFAULT_MANGALMM_READY_MANIFEST,
    DEFAULT_MANGALMM_RUNTIME_OPTIONS,
    MANGALMM_MMPROJ_NAME,
    MANGALMM_MODEL_NAME,
    MANGALMM_MODEL_SPECS,
    MANGALMM_RUNTIME_FINGERPRINT_LABEL,
    MANGALMM_RUNTIME_PREPARATION_VERSION,
    MangaLMMRuntimeContract,
    MangaLMMRuntimeContractError,
    build_mangalmm_runtime_contract,
    resolve_mangalmm_runtime_options,
    validate_mangalmm_volume_name,
)
from modules.ocr.paddle_spotting.runtime import (
    DEFAULT_PADDLE_SPOTTING_LLAMA_CPP_IMAGE,
    DEFAULT_PADDLE_SPOTTING_MODEL_VOLUME,
    DEFAULT_PADDLE_SPOTTING_READY_MANIFEST,
    PADDLE_SPOTTING_IMAGE_MAX_PIXELS,
    PADDLE_SPOTTING_MMPROJ_NAME,
    PADDLE_SPOTTING_MODEL_ALIAS,
    PADDLE_SPOTTING_MODEL_NAME,
    PADDLE_SPOTTING_MODEL_SPECS,
    PADDLE_SPOTTING_RUNTIME_FINGERPRINT_LABEL,
    PADDLE_SPOTTING_RUNTIME_PREPARATION_VERSION,
    PaddleSpottingRuntimeContract,
    PaddleSpottingRuntimeContractError,
    build_paddle_spotting_runtime_contract,
    validate_paddle_spotting_volume_name,
)
from modules.utils.exceptions import LocalServiceSetupError, OperationCancelledError
from modules.utils.local_llama_router import (
    LocalLlamaRouterCoordinator,
    RouterModelMaterial,
    RouterPair,
    RouterPairKind,
    RouterRuntimeSpec,
)
from modules.utils.local_llama_router.contracts import router_pair_for_engine_key
from modules.utils.llama_cpp_runtime import (
    DEFAULT_LLAMA_CPP_IMAGE,
    DEFAULT_MANAGED_RUNTIME_STOP_TIMEOUT_SEC,
    inspect_llama_cpp_runtime,
    resolve_docker_compose_command,
)
from modules.utils.managed_runtime_repair import (
    describe_image_identity_drift,
    is_image_identity_only_drift,
)

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]


class _ManagedVolumeNotProvisioned(RuntimeError):
    """준비 볼륨이 아예 없거나 계약된 파일이 빠져 있다.

    준비 스크립트를 돌리면 해결되는 상태다. 볼륨 라벨 불일치처럼 사람의 판단을
    요구하는 상태와 구별하려고 따로 둔다.
    """

LATE_START_STOP_GRACE_SEC = 3.0
LATE_START_STOP_POLL_SEC = 0.25
PADDLEOCR_LLAMA_CPP_IMAGE_REF = DEFAULT_PADDLE_LLAMA_CPP_IMAGE
# 캐시 식별자로 쓰는 이미지 토큰이다. digest로 고정된 참조면 digest만, 태그
# 참조면 태그 전체가 남는다. 어느 쪽이든 이미지를 바꾸면 값이 함께 바뀌므로
# 캐시 무효화 목적에는 그대로 성립한다.
PADDLEOCR_LLAMA_CPP_IMAGE_DIGEST = PADDLEOCR_LLAMA_CPP_IMAGE_REF.rsplit("@", 1)[
    -1
]
# Compatibility aliases used by existing cache/runtime callers now identify
# the only active managed Paddle crop image.
PADDLEOCR_IMAGE_REF = PADDLEOCR_LLAMA_CPP_IMAGE_REF
PADDLEOCR_IMAGE_DIGEST = PADDLEOCR_LLAMA_CPP_IMAGE_DIGEST
PADDLEOCR_RUNTIME_FINGERPRINT_LABEL = PADDLE_RUNTIME_FINGERPRINT_LABEL
HUNYUAN_OCR_LLAMA_CPP_IMAGE_REF = DEFAULT_HUNYUAN_OCR_LLAMA_CPP_IMAGE
MANGALMM_LLAMA_CPP_IMAGE_REF = DEFAULT_MANGALMM_LLAMA_CPP_IMAGE
MANGALMM_LLAMA_CPP_IMAGE_DIGEST = MANGALMM_LLAMA_CPP_IMAGE_REF.rsplit("@", 1)[
    -1
]
PADDLEOCR_SPOTTING_LLAMA_CPP_IMAGE_REF = (
    DEFAULT_PADDLE_SPOTTING_LLAMA_CPP_IMAGE
)
PADDLEOCR_SPOTTING_LLAMA_CPP_IMAGE_DIGEST = (
    PADDLEOCR_SPOTTING_LLAMA_CPP_IMAGE_REF.rsplit("@", 1)[-1]
)
# 관리형 OCR 엔진별 Arbiter lease 이름. 스테이지 스케줄러는 같은 이름을 명시적으로
# 넘기고, 이 표는 Router의 load와 release가 함께 참조하는 기본값이다. 양쪽이 서로
# 다른 이름을 쓰면 lease가 교착되므로 한 곳에서만 정의한다.
_OCR_RUNTIME_SERVICE_NAMES = {
    "PaddleOCR VL": "paddleocr_vl",
    "PaddleOCR VL Spotting": "paddleocr_vl_spotting",
    "HunyuanOCR": "hunyuanocr",
    "MangaLMM": "mangalmm",
}

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
        "managed_backend": MANAGED_LOCAL_INFERENCE_BACKEND,
        "uses_llama_cpp": True,
    },
    "MangaLMM": {
        "compose_file": ROOT_DIR / "mangalmm_docker_files" / "docker-compose.yaml",
        "managed_url": "http://127.0.0.1:28081/v1",
        "health_url": "http://127.0.0.1:28081/health",
        "settings_page_name": "MangaLMM Settings",
        "container_name": "mangalmm-local-server",
        "container_names": ["mangalmm-local-server"],
        "managed_backend": MANAGED_LOCAL_INFERENCE_BACKEND,
        "uses_llama_cpp": True,
    },
    "PaddleOCR VL": {
        "compose_file": ROOT_DIR / "paddleocr_vl_docker_files" / "docker-compose.yaml",
        "managed_url": DEFAULT_PADDLE_DIRECT_SERVER_URL,
        "health_url": "http://127.0.0.1:18000/health",
        "settings_page_name": "PaddleOCR VL Settings",
        "container_name": "paddleocr-llamacpp",
        "container_names": ["paddleocr-llamacpp"],
        "managed_backend": MANAGED_LOCAL_INFERENCE_BACKEND,
        "uses_llama_cpp": True,
    },
    "PaddleOCR VL Spotting": {
        "compose_file": (
            ROOT_DIR
            / "paddleocr_vl_spotting_docker_files"
            / "docker-compose.yaml"
        ),
        "managed_url": (
            "http://127.0.0.1:18002/v1/chat/completions"
        ),
        "health_url": "http://127.0.0.1:18002/health",
        "settings_page_name": "PaddleOCR VL Spotting Settings",
        "container_name": "paddleocr-spotting-llamacpp",
        "container_names": ["paddleocr-spotting-llamacpp"],
        "managed_backend": MANAGED_LOCAL_INFERENCE_BACKEND,
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
    def __init__(
        self,
        *,
        router_coordinator: LocalLlamaRouterCoordinator | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._compose_command: tuple[str, ...] | None = None
        self._active_engine: str | None = None
        self._managed_start_attempted_engine: str | None = None
        self._readiness_cache: set[tuple[str, str, str]] = set()
        self._startup_cancel_checker: Callable[[], bool] | None = None
        # 자가복구는 계약을 읽는 깊은 곳에서 일어난다. 진행 콜백을 그 경로 전체로
        # 인자로 흘리는 대신 현재 기동의 콜백을 여기에 둔다.
        self._startup_progress_callback: Callable[[dict[str, Any]], None] | None = None
        self._paddle_runtime_contract_cache: PaddleLlamaRuntimeContract | None = None
        self._paddle_spotting_runtime_contract_cache: (
            PaddleSpottingRuntimeContract | None
        ) = None
        self._hunyuan_ocr_runtime_contract_cache: (
            HunyuanOCRRuntimeContract | None
        ) = None
        self._mangalmm_runtime_contract_cache: MangaLMMRuntimeContract | None = None
        self._paddle_idle_released = False
        self._warned_legacy_backend_environment: tuple[str, ...] = ()
        self._router_coordinator = router_coordinator
        self._router_gemma_manager: Any | None = None
        self._router_pair: RouterPair | None = None
        self._router_spec: RouterRuntimeSpec | None = None

    def set_router_gemma_manager(self, manager: Any | None) -> None:
        """Inject the companion manager only for the controller-owned pair."""

        self._router_gemma_manager = manager

    def router_pair_for_engine(
        self,
        engine_key: str,
        settings_page: Any,
    ) -> RouterPair | None:
        """기본 OCR + Gemma 조합이 완전할 때만 Router pair를 반환한다."""

        coordinator = self._router_coordinator
        gemma_manager = self._router_gemma_manager
        if coordinator is None or gemma_manager is None:
            return None
        if not self._is_gemma_translator_selected(settings_page):
            return None
        credentials = getattr(gemma_manager, "router_credentials", None)
        if not callable(credentials):
            return None
        try:
            gemma_endpoint, gemma_model = credentials(settings_page)
        except Exception:
            return None
        pair = coordinator.classify_pair(
            engine_key,
            self._resolve_server_url(engine_key, settings_page),
            gemma_endpoint,
            gemma_model,
        )
        if pair is None or not self._router_preset_matches_runtime_options(pair):
            return None
        return pair

    def _router_preset_matches_runtime_options(self, pair: RouterPair) -> bool:
        """정적 preset이 사용자가 조정한 옵션을 버릴 상황이면 Router를 거부한다.

        Router는 모델을 Compose 환경변수가 아니라 fingerprint된 preset 파일로
        구성한다. 따라서 separate-server 경로가 조정 가능한 런타임 옵션을 노출하는
        엔진은, 그 옵션이 고정된 기본값을 유지하는 동안에만 Router로 보낼 수 있다.
        그렇지 않으면 조정값이 조용히 무시되는데, 이는 기동 비용만 달라지는 것이
        아니라 OCR 동작 자체를 바꾸므로 separate-server 경로가 계속 담당한다.
        """

        if pair.kind is RouterPairKind.MANGALMM:
            try:
                resolved = resolve_mangalmm_runtime_options(os.environ)
            except MangaLMMRuntimeContractError:
                return False
            return resolved == DEFAULT_MANGALMM_RUNTIME_OPTIONS
        if pair.kind is RouterPairKind.HUNYUAN:
            try:
                resolved = resolve_hunyuan_ocr_runtime_options(os.environ)
            except HunyuanOCRRuntimeContractError:
                return False
            return resolved == DEFAULT_HUNYUAN_OCR_RUNTIME_OPTIONS
        return True

    def router_is_active(self) -> bool:
        pair = self._router_pair
        coordinator = self._router_coordinator
        if pair is None or coordinator is None:
            return False
        return coordinator.snapshot().pair == pair.kind.value

    def router_inference_lease(
        self,
        engine_key: str,
        settings_page: Any,
    ) -> Any:
        """Return an HTTP-only Router lease, or a no-op for non-Router paths.

        A default Paddle + Gemma combination is never allowed to silently fall
        through to a raw endpoint after the Router has been selected.  The
        returned coordinator lease checks the loaded alias again immediately
        before the request; it does not retain this manager lock during HTTP.
        """

        with self._lock:
            pair = self.router_pair_for_engine(engine_key, settings_page)
            if pair is None:
                return nullcontext()
            coordinator = self._router_coordinator
            if coordinator is None or self._router_pair != pair:
                raise self._build_setup_error(
                    engine_key,
                    "Router inference was requested before the matching OCR model lease was loaded.",
                )
            return coordinator.inference_lease(
                pair=pair,
                model_alias=pair.ocr_alias,
            )

    def _release_separate_server_for_router(self, engine_key: str) -> bool:
        """Router보다 먼저 이 제품의 separate-server OCR 컨테이너를 정지한다.

        두 경로는 같은 OCR 호스트 포트를 publish하므로, 이전 프로세스가 남긴
        separate-server 컨테이너는 Router가 그 포트를 바인딩하는 것을 영구히 막는다.
        요청된 엔진의 제품 번들 컨테이너만 정지하며, 제품 소유가 아닌 listener는
        adapter의 ownership 오류로 남는다.
        """

        if not self._running_managed_container_names(engine_key):
            return False
        self._stop_engine(engine_key)
        if self._active_engine == engine_key:
            self._active_engine = None
        if self._managed_start_attempted_engine == engine_key:
            self._managed_start_attempted_engine = None
        self._readiness_cache.clear()
        if engine_key == "PaddleOCR VL":
            self._paddle_idle_released = False
        logger.info(
            "Stopped the separate-server %s container so the Router can bind its port.",
            engine_key,
        )
        return True

    def _release_stale_router_ports_for_engine(
        self,
        engine_key: str,
        server_url: str,
        *,
        cancel_checker: Callable[[], bool] | None,
    ) -> None:
        """separate-server 경로 이전에 남은 Router 컨테이너의 포트를 회수한다.

        Router는 OCR 포트와 18080을 함께 publish하므로, 이전 프로세스가 남긴
        컨테이너는 separate-server compose의 바인딩을 영구히 실패시킨다. 회수는
        Router 편입 엔진의 정확한 관리형 포트로만 제한하며, custom 포트는 건드리지
        않는다.
        """

        coordinator = self._router_coordinator
        if coordinator is None:
            return
        pair = router_pair_for_engine_key(engine_key)
        if pair is None:
            return
        try:
            configured_port = urlparse(str(server_url or "").strip()).port
        except ValueError:
            return
        if configured_port != pair.ocr_port:
            return
        try:
            released = coordinator.release_owned_pair_ports(
                pair,
                cancel_checker=cancel_checker,
            )
        except OperationCancelledError:
            raise
        except Exception as exc:
            raise self._build_setup_error(engine_key, str(exc)) from exc
        if released:
            logger.info(
                "Released leftover Router container(s) %s so the separate-server "
                "%s runtime can bind port %s.",
                ", ".join(released),
                engine_key,
                pair.ocr_port,
            )

    def _is_gemma_translator_selected(self, settings_page: Any) -> bool:
        getter = getattr(settings_page, "get_tool_selection", None)
        if not callable(getter):
            return False
        try:
            selected = str(getter("translator") or "").strip()
        except Exception:
            return False
        accepted = {"Custom Local Server(Gemma)", "Custom Local Server"}
        ui = getattr(settings_page, "ui", None)
        translator = getattr(ui, "tr", None)
        if callable(translator):
            for key in tuple(accepted):
                try:
                    accepted.add(str(translator(key)))
                except Exception:
                    continue
        return selected in accepted

    def _router_runtime_spec(
        self,
        engine_key: str,
        settings_page: Any,
        pair: RouterPair,
    ) -> RouterRuntimeSpec:
        # 모델 alias는 반드시 pair 정의에서 가져온다. 라우터는 pair alias로
        # 모델을 적재하고, GPU 귀속 검증은 worker의 --alias를 contract의
        # ocr_model.alias와 대조한다. 둘이 어긋나면 컨테이너는 떠도 귀속 증거를
        # 찾지 못해 기동이 실패한다.
        gemma_manager = self._router_gemma_manager
        material_getter = getattr(gemma_manager, "router_model_material", None)
        if not callable(material_getter):
            raise self._build_setup_error(
                engine_key,
                "Router Gemma runtime material is unavailable.",
            )
        if engine_key == "PaddleOCR VL":
            self._ensure_paddle_runtime_images()
            contract = self._paddle_runtime_contract(force_refresh=True)
            material = RouterModelMaterial(
                alias=pair.ocr_alias,
                model_file=PADDLE_LLAMA_MODEL_NAME,
                model_sha256=str(
                    PADDLE_LLAMA_MODEL_SPECS[PADDLE_LLAMA_MODEL_NAME]["sha256"]
                ),
                mmproj_file=PADDLE_LLAMA_MMPROJ_NAME,
                mmproj_sha256=str(
                    PADDLE_LLAMA_MODEL_SPECS[PADDLE_LLAMA_MMPROJ_NAME]["sha256"]
                ),
                volume_name=contract.volume_name,
                ready_manifest_sha256=contract.ready_manifest_sha256,
                source_fingerprint=contract.fingerprint,
                runtime_options=dict(contract.runtime_options),
                preparation_version=contract.preparation_version,
            )
        elif engine_key == "PaddleOCR VL Spotting":
            self._ensure_paddle_spotting_runtime_image()
            contract = self._paddle_spotting_runtime_contract(force_refresh=True)
            material = RouterModelMaterial(
                alias=pair.ocr_alias,
                model_file=PADDLE_SPOTTING_MODEL_NAME,
                model_sha256=str(
                    PADDLE_SPOTTING_MODEL_SPECS[PADDLE_SPOTTING_MODEL_NAME]["sha256"]
                ),
                mmproj_file=PADDLE_SPOTTING_MMPROJ_NAME,
                mmproj_sha256=str(
                    PADDLE_SPOTTING_MODEL_SPECS[
                        PADDLE_SPOTTING_MMPROJ_NAME
                    ]["sha256"]
                ),
                volume_name=contract.volume_name,
                ready_manifest_sha256=contract.ready_manifest_sha256,
                source_fingerprint=contract.fingerprint,
                runtime_options=dict(contract.runtime_options),
                preparation_version=contract.preparation_version,
            )
        elif engine_key == "HunyuanOCR":
            self._ensure_hunyuan_ocr_runtime_image()
            contract = self._hunyuan_ocr_runtime_contract(force_refresh=True)
            material = RouterModelMaterial(
                alias=pair.ocr_alias,
                model_file=HUNYUAN_OCR_MODEL_NAME,
                model_sha256=str(
                    HUNYUAN_OCR_MODEL_SPECS[HUNYUAN_OCR_MODEL_NAME]["sha256"]
                ),
                mmproj_file=HUNYUAN_OCR_MMPROJ_NAME,
                mmproj_sha256=str(
                    HUNYUAN_OCR_MODEL_SPECS[HUNYUAN_OCR_MMPROJ_NAME]["sha256"]
                ),
                volume_name=contract.volume_name,
                ready_manifest_sha256=contract.ready_manifest_sha256,
                source_fingerprint=contract.fingerprint,
                runtime_options=dict(contract.runtime_options),
                preparation_version=contract.preparation_version,
            )
        elif engine_key == "MangaLMM":
            self._ensure_mangalmm_runtime_image()
            contract = self._mangalmm_runtime_contract(force_refresh=True)
            material = RouterModelMaterial(
                alias=pair.ocr_alias,
                model_file=MANGALMM_MODEL_NAME,
                model_sha256=str(
                    MANGALMM_MODEL_SPECS[MANGALMM_MODEL_NAME]["sha256"]
                ),
                mmproj_file=MANGALMM_MMPROJ_NAME,
                mmproj_sha256=str(
                    MANGALMM_MODEL_SPECS[MANGALMM_MMPROJ_NAME]["sha256"]
                ),
                volume_name=contract.volume_name,
                ready_manifest_sha256=contract.ready_manifest_sha256,
                source_fingerprint=contract.fingerprint,
                runtime_options=dict(contract.runtime_options),
                preparation_version=contract.preparation_version,
            )
        else:
            raise self._build_setup_error(
                engine_key,
                "Router는 PaddleOCR-VL 두 경로와 HunyuanOCR, MangaLMM만 지원합니다.",
            )
        gemma_material, gemma_image_ref = material_getter(settings_page)
        if str(contract.llama_image_ref) != str(gemma_image_ref):
            raise self._build_setup_error(
                engine_key,
                "Router requires one identical pinned llama.cpp image for OCR and Gemma.",
            )
        return RouterRuntimeSpec(
            pair=pair,
            ocr_model=material,
            gemma_model=gemma_material,
            image_ref=contract.llama_image_ref,
        )

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
        if engine_key == "PaddleOCR VL Spotting":
            try:
                self._ensure_paddle_spotting_runtime_image()
                self._paddle_spotting_runtime_contract(
                    force_refresh=True
                )
            except (
                PaddleSpottingRuntimeContractError,
                OSError,
            ) as exc:
                raise self._build_setup_error(
                    engine_key,
                    (
                        "Prepared PaddleOCR-VL Spotting runtime validation "
                        f"failed: {exc}\n"
                        "Run scripts/"
                        "prepare_paddleocr_spotting_llamacpp_runtime.ps1 "
                        "in Prepare or Verify mode."
                    ),
                ) from exc
        if engine_key == "MangaLMM":
            try:
                self._ensure_mangalmm_runtime_image()
                self._mangalmm_runtime_contract(force_refresh=True)
            except (MangaLMMRuntimeContractError, OSError) as exc:
                raise self._build_setup_error(
                    engine_key,
                    (
                        f"Prepared MangaLMM runtime validation failed: {exc}\n"
                        "Run scripts/prepare_mangalmm_llamacpp_runtime.ps1 "
                        "in Prepare or Verify mode."
                    )
                ) from exc

    def should_manage_engine(self, engine_key: str, settings_page: Any) -> bool:
        if not is_local_ocr_engine(engine_key):
            return False
        config = self._config_for(engine_key)
        server_url = self._resolve_server_url(engine_key, settings_page)
        return _normalize_url(server_url) == _normalize_url(config["managed_url"])

    def get_ocr_cache_identity(
        self,
        engine_key: str,
        settings_page: Any,
    ) -> dict[str, Any] | None:
        """Return a trustworthy identity without starting the managed runtime."""

        if engine_key not in {
            "PaddleOCR VL",
            "PaddleOCR VL Spotting",
        }:
            return None
        router_pair = self.router_pair_for_engine(engine_key, settings_page)
        if router_pair is not None and self._router_coordinator is not None:
            router_snapshot = self._router_coordinator.snapshot()
            if (
                router_snapshot.pair != router_pair.kind.value
                or not router_snapshot.fingerprint
            ):
                # Do not declare a cache identity before the owned Router
                # contract has been independently built.  Starting a runtime
                # is safer than reusing an identity whose model generation is
                # unknown.
                return None
            return {
                "identity_schema_version": 4,
                "managed": True,
                "router": True,
                "router_pair": router_pair.kind.value,
                "router_fingerprint": router_snapshot.fingerprint,
                "router_model_generation": router_snapshot.model_generation,
                "engine": engine_key,
                "endpoint": self._resolve_server_url(engine_key, settings_page),
                "model_name": router_pair.ocr_alias,
            }
        if not self.should_manage_engine(engine_key, settings_page):
            return None
        try:
            expected_names = self._managed_container_names(engine_key)
            present_names = self._present_managed_container_names(engine_key)
            if (
                len(present_names) != len(expected_names)
                or not self._managed_containers_match_contract(
                    engine_key,
                    present_names,
                )
            ):
                return None
            contract = (
                self._paddle_runtime_contract()
                if engine_key == "PaddleOCR VL"
                else self._paddle_spotting_runtime_contract()
            )
        except OperationCancelledError:
            raise
        except Exception:
            logger.warning(
                "Persistent PaddleOCR-VL cache is disabled for this run because "
                "the managed runtime identity could not be resolved.",
                exc_info=True,
            )
            return None
        if not contract.llama_image_id:
            logger.info(
                "Persistent PaddleOCR-VL cache is disabled until the pinned "
                "managed images are installed locally."
            )
            return None
        if engine_key == "PaddleOCR VL Spotting":
            return {
                "identity_schema_version": 1,
                "managed": True,
                "engine": engine_key,
                "strategy": "paddle_spotting_full_page",
                "backend": "llama.cpp",
                "endpoint": _normalize_url(
                    self._resolve_server_url(engine_key, settings_page)
                ),
                "model_name": PADDLE_SPOTTING_MODEL_ALIAS,
                "model_file": PADDLE_SPOTTING_MODEL_NAME,
                "model_sha256": str(
                    PADDLE_SPOTTING_MODEL_SPECS[
                        PADDLE_SPOTTING_MODEL_NAME
                    ]["sha256"]
                ),
                "mmproj_file": PADDLE_SPOTTING_MMPROJ_NAME,
                "mmproj_sha256": str(
                    PADDLE_SPOTTING_MODEL_SPECS[
                        PADDLE_SPOTTING_MMPROJ_NAME
                    ]["sha256"]
                ),
                "model_volume": contract.volume_name,
                "ready_manifest_sha256": (
                    contract.ready_manifest_sha256
                ),
                "llama_image_ref": contract.llama_image_ref,
                "llama_image_digest": (
                    PADDLEOCR_SPOTTING_LLAMA_CPP_IMAGE_DIGEST
                ),
                "llama_image_id": contract.llama_image_id,
                "compose_sha256": contract.compose_file_sha256,
                "command_sha256": contract.command_sha256,
                "runtime_fingerprint": contract.fingerprint,
                "prompt": "Spotting:",
                "special_tokens": True,
                "clip.vision.image_max_pixels": (
                    PADDLE_SPOTTING_IMAGE_MAX_PIXELS
                ),
            }
        return {
            "identity_schema_version": 3,
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
            "compose_sha256": contract.compose_file_sha256,
            "command_sha256": contract.command_sha256,
            "transport": direct_transport_identity(),
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
        resource_arbiter: Any | None = None,
        runtime_service: str = "",
    ) -> None:
        with self._lock:
            self._startup_cancel_checker = cancel_checker
            self._startup_progress_callback = progress_callback
            router_pair = self.router_pair_for_engine(engine_key, settings_page)
            if router_pair is not None:
                self._ensure_router_engine(
                    engine_key,
                    settings_page,
                    router_pair,
                    resource_arbiter=resource_arbiter,
                    runtime_service=runtime_service,
                    cancel_checker=cancel_checker,
                )
                return
            # The configured endpoint/model is no longer the exact product
            # Router contract.  Do not let a stale owned Router keep the
            # default ports occupied while the user moves to a custom or
            # separate-server path.
            if self._router_pair is not None:
                self._router_finish(
                    stop_container=True,
                    resource_arbiter=resource_arbiter,
                    runtime_service=self._router_owner_service(
                        runtime_service or self._router_owned_service_default()
                    ),
                    cancel_checker=cancel_checker,
                )
            if not is_local_ocr_engine(engine_key) or not self.should_manage_engine(engine_key, settings_page):
                self._deactivate_active_engine()
                return

            # 이전 프로세스의 Router 컨테이너가 이 엔진의 포트를 계속 publish
            # 하고 있으므로, Compose가 바인딩을 시도하기 전에 회수한다.
            self._release_stale_router_ports_for_engine(
                engine_key,
                self._resolve_server_url(engine_key, settings_page),
                cancel_checker=cancel_checker,
            )

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
                    engine_key
                    not in {
                        "PaddleOCR VL",
                        "PaddleOCR VL Spotting",
                        "MangaLMM",
                    }
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

    def prepare_engine_container(
        self,
        engine_key: str,
        settings_page: Any,
        *,
        resource_arbiter: Any | None = None,
        runtime_service: str = "",
        cancel_checker: Callable[[], bool] | None = None,
    ) -> bool:
        """모델은 올리지 않고 Router 컨테이너만 미리 띄운다.

        Router v2 는 `--models-max 1 --no-models-autoload` 로 뜬다. 즉 컨테이너 기동
        자체는 어떤 모델도 적재하지 않는다. 그래서 이 작업은 검출 sweep 과 겹쳐도
        모델 적재 baseline 을 오염시키지 않는다 — PR #242 가 예열을 검출 뒤로 미룬
        이유는 검출기의 ONNX 세션이 **모델 적재** baseline 에 섞이는 것이었고, 여기서는
        모델을 적재하지 않는다.

        컨테이너 기동은 이미지 인출·컨테이너 생성·서버 부팅을 포함해 실측 6~8초다.
        그만큼을 검출 뒤에서 앞으로 당긴다.

        준비를 실제로 수행했으면 True. Router 경로가 아니거나 실패하면 False —
        이것은 최적화일 뿐이므로 실패가 배치를 멈춰서는 안 된다.
        """

        with self._lock:
            coordinator = self._router_coordinator
            if coordinator is None:
                return False
            try:
                pair = self.router_pair_for_engine(engine_key, settings_page)
            except Exception:
                return False
            if pair is None:
                return False
            if self._router_pair is not None:
                # 이미 어떤 쌍을 소유하고 있다. 여기서 쌍 전환을 벌이면 예열이 아니라
                # 본 작업이 된다. 준비는 건너뛰고 정식 경로에 맡긴다.
                return False
            try:
                self._release_separate_server_for_router(engine_key)
                release_gemma = getattr(
                    self._router_gemma_manager,
                    "release_separate_server_for_router",
                    None,
                )
                if callable(release_gemma):
                    release_gemma()
                spec = self._router_runtime_spec(engine_key, settings_page, pair)
                coordinator.prepare(
                    spec,
                    arbiter=resource_arbiter,
                    service=runtime_service or self._router_service_name(engine_key),
                    cancel_checker=cancel_checker,
                )
            except OperationCancelledError:
                raise
            except Exception:
                logger.info(
                    "Router 컨테이너 사전 기동을 건너뜁니다. 정식 경로가 처리합니다.",
                    exc_info=True,
                )
                return False
            return True

    def _ensure_router_engine(
        self,
        engine_key: str,
        settings_page: Any,
        pair: RouterPair,
        *,
        resource_arbiter: Any | None,
        runtime_service: str,
        cancel_checker: Callable[[], bool] | None,
    ) -> None:
        coordinator = self._router_coordinator
        if coordinator is None:
            raise self._build_setup_error(
                engine_key,
                "Router coordinator was not injected into the managed OCR runtime.",
            )
        if self._router_pair is not None and self._router_pair != pair:
            # A Crop <-> Spotting change must terminally release the exact
            # owned Router pair before the second pair can reserve 18080.
            # `_router_finish` verifies ownership and never touches a foreign
            # container or direct server.
            self._router_finish(
                stop_container=True,
                resource_arbiter=resource_arbiter,
                runtime_service=self._router_owner_service(
                    runtime_service or self._router_owned_service_default()
                ),
                cancel_checker=cancel_checker,
            )
        elif self._active_engine is not None and self._active_engine != engine_key:
            # This only stops a separate runtime the manager itself started.
            # A foreign process at a default port remains the adapter's
            # explicit ownership error and is never guessed/stopped here.
            self._deactivate_active_engine()
        # separate-server 컨테이너는 Router와 같은 호스트 포트를 publish하므로,
        # Router가 그 포트를 확보하기 전에 이 제품의 OCR·Gemma 컨테이너를 먼저
        # 정지한다. 제품 소유가 아닌 listener는 adapter의 명시적 ownership 오류로
        # 남는다.
        try:
            self._release_separate_server_for_router(engine_key)
        except OperationCancelledError:
            raise
        except Exception as exc:
            raise self._build_setup_error(engine_key, str(exc)) from exc
        release_gemma = getattr(
            self._router_gemma_manager,
            "release_separate_server_for_router",
            None,
        )
        if callable(release_gemma):
            try:
                release_gemma()
            except OperationCancelledError:
                raise
            except Exception as exc:
                raise self._build_setup_error(engine_key, str(exc)) from exc
        spec = self._router_runtime_spec(engine_key, settings_page, pair)
        service = runtime_service or self._router_service_name(engine_key)
        try:
            coordinator.load(
                spec,
                pair.ocr_alias,
                arbiter=resource_arbiter,
                service=service,
                cancel_checker=cancel_checker,
            )
        except OperationCancelledError:
            # The coordinator already ran its owned-container cleanup through
            # the Arbiter. Preserve cancellation so the stage does not record
            # a user stop as an OCR setup failure.
            raise
        except Exception as exc:
            raise self._build_setup_error(engine_key, str(exc)) from exc
        self._router_pair = pair
        self._router_spec = spec
        setter = getattr(self._router_gemma_manager, "set_router_spec", None)
        if callable(setter):
            setter(spec)
        self._active_engine = engine_key
        self._managed_start_attempted_engine = engine_key
        self._paddle_idle_released = False
        self._readiness_cache.clear()

    def _router_finish(
        self,
        *,
        stop_container: bool,
        resource_arbiter: Any | None,
        runtime_service: str,
        cancel_checker: Callable[[], bool] | None = None,
        allow_foreign_owner_teardown: bool = False,
    ) -> dict[str, Any]:
        coordinator = self._router_coordinator
        if coordinator is None or self._router_pair is None:
            return {"runtime_state": "stopped", "gpu_release_expected": False}
        evidence = coordinator.finish(
            arbiter=resource_arbiter,
            service=runtime_service or self._router_owned_service_default(),
            stop_container=stop_container,
            cancel_checker=cancel_checker,
            allow_foreign_owner_teardown=allow_foreign_owner_teardown,
        )
        snapshot = coordinator.snapshot()
        if stop_container:
            self._router_pair = None
            self._router_spec = None
            clear_spec = getattr(self._router_gemma_manager, "set_router_spec", None)
            if callable(clear_spec):
                clear_spec(None)
        self._active_engine = None
        self._managed_start_attempted_engine = None
        self._readiness_cache.clear()
        return {
            "runtime_state": (
                "stopped" if stop_container else "router_container_ready"
            ),
            # The coordinator already ran the stricter Router-specific 30 s
            # gate.  The legacy stage gate must not replace it with its former
            # sleep-residue heuristic.
            "gpu_release_expected": False,
            "router_release_evidence": (
                evidence.vram if evidence is not None else {"required": False}
            ),
            "router_model_generation": snapshot.model_generation,
        }

    @staticmethod
    def _router_service_name(engine_key: str, fallback: str = "ocr") -> str:
        """Router의 load와 release가 공유해야 하는 단일 GPU lease 이름.

        load가 잡은 lease 이름과 release가 요청하는 이름이 다르면 Arbiter가
        교착되므로, 양방향이 각자 기본값을 정하지 않고 여기서 함께 해석한다.
        """

        return _OCR_RUNTIME_SERVICE_NAMES.get(
            str(engine_key or ""),
            str(fallback or "ocr"),
        )

    def _router_owner_service(self, fallback: str) -> str:
        """Return the Arbiter service that owns the currently selected pair."""

        coordinator = self._router_coordinator
        spec = self._router_spec
        if coordinator is not None and spec is not None:
            loaded_model = str(
                getattr(coordinator.snapshot(), "loaded_model", "") or ""
            )
            if loaded_model == spec.gemma_model.alias:
                return "gemma"
            if loaded_model == spec.ocr_model.alias:
                return self._router_service_name(
                    spec.pair.ocr_engine_key,
                    fallback,
                )
        return self._router_service_name(self._active_engine or "", fallback)

    def _router_owned_service_default(self) -> str:
        """이 매니저가 현재 소유한 pair의 lease 이름."""

        pair = self._router_pair
        if pair is None:
            return "ocr"
        return self._router_service_name(pair.ocr_engine_key)

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
        contract_managed = engine_key in {
            "PaddleOCR VL",
            "PaddleOCR VL Spotting",
            "MangaLMM",
        }
        present_containers = (
            self._present_managed_container_names(engine_key)
            if contract_managed
            else []
        )
        if (
            contract_managed
            and present_containers
            and (
                len(present_containers)
                != len(self._managed_container_names(engine_key))
                or not self._managed_containers_match_contract(
                    engine_key,
                    present_containers,
                )
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

    def shutdown(
        self,
        *,
        resource_arbiter: Any | None = None,
        runtime_service: str = "",
        cancel_checker: Callable[[], bool] | None = None,
        allow_foreign_owner_teardown: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            if self.router_is_active():
                return self._router_finish(
                    stop_container=True,
                    resource_arbiter=resource_arbiter,
                    runtime_service=runtime_service or self._router_owned_service_default(),
                    cancel_checker=cancel_checker,
                    allow_foreign_owner_teardown=allow_foreign_owner_teardown,
                )
            gpu_release_expected = bool(
                self._active_engine is not None
                and not self._paddle_idle_released
            )
            self._readiness_cache.clear()
            self._deactivate_active_engine()
            self._paddle_idle_released = False
            return {
                "runtime_state": "stopped",
                "gpu_release_expected": gpu_release_expected,
            }

    def release_for_handoff(
        self,
        *,
        resource_arbiter: Any | None = None,
        runtime_service: str = "",
        cancel_checker: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Release OCR GPU residency while preserving a reusable llama server.

        The stage scheduler must distinguish a confirmed sleeping llama.cpp
        process from a fallback managed stop.  Returning that fact keeps its
        GPU lease and telemetry state honest without exposing implementation
        details of the runtime manager.
        """

        with self._lock:
            if self.router_is_active():
                return self._router_finish(
                    stop_container=False,
                    resource_arbiter=resource_arbiter,
                    runtime_service=runtime_service or self._router_owned_service_default(),
                    cancel_checker=cancel_checker,
                )
            if self._active_engine not in (None, "PaddleOCR VL"):
                self._deactivate_active_engine()
                return {
                    "runtime_state": "stopped",
                    "gpu_release_expected": True,
                }
            if self._active_engine is None:
                running = self._running_managed_container_names("PaddleOCR VL")
                if not running:
                    return {
                        "runtime_state": "stopped",
                        "gpu_release_expected": False,
                    }
                expected = self._managed_container_names("PaddleOCR VL")
                if (
                    len(running) != len(expected)
                    or not self._paddle_containers_match_contract(running)
                ):
                    self._managed_start_attempted_engine = "PaddleOCR VL"
                    self._deactivate_active_engine()
                    return {
                        "runtime_state": "stopped",
                        "gpu_release_expected": True,
                    }
                self._active_engine = "PaddleOCR VL"
                self._managed_start_attempted_engine = "PaddleOCR VL"
            if self._paddle_idle_released:
                return {
                    "runtime_state": "sleeping",
                    "gpu_release_expected": False,
                }
            if self._wait_for_paddle_llama_sleep():
                self._paddle_idle_released = True
                logger.info(
                    "PaddleOCR llama.cpp entered idle sleep; containers remain "
                    "available for the next OCR stage."
                )
                return {
                    "runtime_state": "sleeping",
                    "gpu_release_expected": True,
                }
            logger.warning(
                "PaddleOCR llama.cpp did not confirm idle sleep; falling back "
                "to the normal managed stop before the next GPU stage."
            )
            self._deactivate_active_engine()
            self._paddle_idle_released = False
            return {
                "runtime_state": "stopped",
                "gpu_release_expected": True,
            }

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
            image_key = {
                "PaddleOCR VL": "PADDLEOCR_LLAMA_CPP_IMAGE",
                "PaddleOCR VL Spotting": (
                    "PADDLEOCR_SPOTTING_LLAMA_CPP_IMAGE"
                ),
                "HunyuanOCR": "HUNYUAN_OCR_LLAMA_CPP_IMAGE",
                "MangaLMM": "MANGALMM_LLAMA_CPP_IMAGE",
            }.get(engine_key, "LLAMA_CPP_IMAGE")
            requested_image = env.get(image_key, "")
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
        env, ignored = sanitize_managed_runtime_environment(os.environ)
        ignored_keys = tuple(sorted(ignored))
        if (
            ignored_keys
            and ignored_keys != self._warned_legacy_backend_environment
        ):
            logger.warning(
                "Ignoring retired managed vLLM environment controls and "
                "using llama.cpp: %s",
                ", ".join(ignored_keys),
            )
            self._warned_legacy_backend_environment = ignored_keys
        env.setdefault("LLAMA_CPP_IMAGE", DEFAULT_LLAMA_CPP_IMAGE)
        if engine_key == "HunyuanOCR":
            env.update(
                self._hunyuan_ocr_runtime_contract().compose_environment()
            )
        if engine_key == "PaddleOCR VL":
            env.update(self._paddle_runtime_contract().compose_environment())
        if engine_key == "PaddleOCR VL Spotting":
            env.update(
                self._paddle_spotting_runtime_contract()
                .compose_environment()
            )
        if engine_key == "MangaLMM":
            env.update(self._mangalmm_runtime_contract().compose_environment())
        return env

    def _resolve_server_url(self, engine_key: str, settings_page: Any) -> str:
        if engine_key == "HunyuanOCR":
            return str(settings_page.get_hunyuan_ocr_settings().get("server_url", "")).strip()
        if engine_key == "MangaLMM":
            return str(settings_page.get_mangalmm_ocr_settings().get("server_url", "")).strip()
        if engine_key == "PaddleOCR VL":
            return str(settings_page.get_paddleocr_vl_settings().get("server_url", "")).strip()
        if engine_key == "PaddleOCR VL Spotting":
            return str(
                settings_page
                .get_paddleocr_vl_spotting_settings()
                .get("server_url", "")
            ).strip()
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
            image_env_key = {
                "PaddleOCR VL": "PADDLEOCR_LLAMA_CPP_IMAGE",
                "PaddleOCR VL Spotting": (
                    "PADDLEOCR_SPOTTING_LLAMA_CPP_IMAGE"
                ),
                "HunyuanOCR": "HUNYUAN_OCR_LLAMA_CPP_IMAGE",
                "MangaLMM": "MANGALMM_LLAMA_CPP_IMAGE",
            }.get(engine_key, "LLAMA_CPP_IMAGE")
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

    def _contract_or_reseal(
        self,
        *,
        engine_key: str,
        volume_name: str,
        manifest_bytes: bytes,
        image_ref: str,
        image_id: str,
        build: Callable[[str, str], Any],
        allow_repair: bool,
        retry: Callable[[], Any],
    ) -> Any:
        """계약을 세우되, 어긋난 것이 이미지 identity 뿐이면 한 번 다시 봉인한다.

        모델 해시 불일치처럼 실제로 볼륨을 신뢰할 수 없는 상태는 복구하지 않고
        그대로 올린다. 신뢰할 수 없는 볼륨에 유효 도장을 찍어서는 안 된다.
        """

        try:
            return build(image_ref, image_id)
        except Exception:
            if not allow_repair or not is_image_identity_only_drift(
                manifest_bytes,
                current_image_id=image_id,
                revalidate=build,
            ):
                raise
        self._repair_managed_runtime_volume(
            engine_key=engine_key,
            volume_name=volume_name,
            detail=describe_image_identity_drift(
                manifest_bytes,
                current_image_id=image_id,
                runtime_label=engine_key,
            ),
        )
        return retry()

    def _repair_managed_runtime_volume(
        self,
        *,
        engine_key: str,
        volume_name: str,
        detail: str,
        message: str = "",
    ) -> None:
        """Fail before page work; setup is the only provisioning authority."""

        raise self._build_setup_error(
            engine_key,
            (
                f"{detail}\n"
                "The application does not repair or download managed runtimes. "
                "Run the matching setup BAT and start Comic Translate again."
            ),
        )

    def _paddle_runtime_contract(
        self,
        *,
        force_refresh: bool = False,
        allow_repair: bool = True,
    ) -> PaddleLlamaRuntimeContract:
        if self._paddle_runtime_contract_cache is not None and not force_refresh:
            return self._paddle_runtime_contract_cache

        compose_file = Path(self._config_for("PaddleOCR VL")["compose_file"])
        if not compose_file.is_file():
            raise FileNotFoundError(compose_file)
        volume_name = validate_paddle_llama_volume_name(
            os.environ.get(
                "PADDLEOCR_LLAMA_MODEL_VOLUME",
                DEFAULT_PADDLE_LLAMA_MODEL_VOLUME,
            )
        )
        try:
            (
                manifest_bytes,
                manifest_sha256,
                observed_file_bytes,
            ) = self._probe_paddle_model_volume(
                volume_name=volume_name,
                image_ref=PADDLEOCR_LLAMA_CPP_IMAGE_REF,
            )
        except _ManagedVolumeNotProvisioned as exc:
            if not allow_repair:
                raise self._build_setup_error("PaddleOCR VL", str(exc)) from exc
            # 볼륨 자체가 없거나 비었다. 준비 스크립트가 원본을 찾아 채운다.
            self._repair_managed_runtime_volume(
                engine_key="PaddleOCR VL",
                volume_name=volume_name,
                detail=str(exc),
                message=(
                    "PaddleOCR VL 런타임 볼륨을 준비하는 중... "
                    "모델을 내려받아야 하면 오래 걸립니다."
                ),
            )
            return self._paddle_runtime_contract(force_refresh=True, allow_repair=False)
        llama_image_id = self._inspect_docker_image_id(
            PADDLEOCR_LLAMA_CPP_IMAGE_REF
        )
        if not llama_image_id:
            raise PaddleLlamaRuntimeContractError(
                "Pinned PaddleOCR llama.cpp image is not installed."
            )
        def build(image_ref: str, image_id: str) -> PaddleLlamaRuntimeContract:
            return build_paddle_llama_runtime_contract(
                manifest_bytes=manifest_bytes,
                manifest_sha256=manifest_sha256,
                observed_file_bytes=observed_file_bytes,
                volume_name=volume_name,
                llama_image_ref=image_ref,
                llama_image_id=image_id,
                compose_file=compose_file,
                environment=os.environ,
            )

        contract = self._contract_or_reseal(
            engine_key="PaddleOCR VL",
            volume_name=volume_name,
            manifest_bytes=manifest_bytes,
            image_ref=PADDLEOCR_LLAMA_CPP_IMAGE_REF,
            image_id=llama_image_id,
            build=build,
            allow_repair=allow_repair,
            retry=lambda: self._paddle_runtime_contract(
                force_refresh=True,
                allow_repair=False,
            ),
        )
        self._paddle_runtime_contract_cache = contract
        return contract

    def _ensure_paddle_runtime_images(self) -> None:
        for image_ref in (PADDLEOCR_LLAMA_CPP_IMAGE_REF,):
            if self._inspect_docker_image_id(image_ref):
                continue
            raise self._build_setup_error(
                "PaddleOCR VL",
                f"The setup-sealed PaddleOCR image is missing: {image_ref}",
            )

    def _paddle_spotting_runtime_contract(
        self,
        *,
        force_refresh: bool = False,
        allow_repair: bool = True,
    ) -> PaddleSpottingRuntimeContract:
        if (
            self._paddle_spotting_runtime_contract_cache is not None
            and not force_refresh
        ):
            return self._paddle_spotting_runtime_contract_cache

        compose_file = Path(
            self._config_for("PaddleOCR VL Spotting")["compose_file"]
        )
        if not compose_file.is_file():
            raise FileNotFoundError(compose_file)
        volume_name = validate_paddle_spotting_volume_name(
            os.environ.get(
                "PADDLEOCR_SPOTTING_MODEL_VOLUME",
                DEFAULT_PADDLE_SPOTTING_MODEL_VOLUME,
            )
        )
        try:
            (
                manifest_bytes,
                manifest_sha256,
                observed_file_bytes,
            ) = self._probe_paddle_spotting_model_volume(
                volume_name=volume_name,
                image_ref=PADDLEOCR_SPOTTING_LLAMA_CPP_IMAGE_REF,
            )
        except _ManagedVolumeNotProvisioned as exc:
            if not allow_repair:
                raise self._build_setup_error("PaddleOCR VL Spotting", str(exc)) from exc
            # 볼륨 자체가 없거나 비었다. 준비 스크립트가 원본을 찾아 채운다.
            self._repair_managed_runtime_volume(
                engine_key="PaddleOCR VL Spotting",
                volume_name=volume_name,
                detail=str(exc),
                message=(
                    "PaddleOCR VL Spotting 런타임 볼륨을 준비하는 중... "
                    "모델을 내려받아야 하면 오래 걸립니다."
                ),
            )
            return self._paddle_spotting_runtime_contract(force_refresh=True, allow_repair=False)
        llama_image_id = self._inspect_docker_image_id(
            PADDLEOCR_SPOTTING_LLAMA_CPP_IMAGE_REF
        )
        if not llama_image_id:
            raise PaddleSpottingRuntimeContractError(
                "Pinned PaddleOCR-VL Spotting llama.cpp image is not "
                "installed."
            )
        def build(image_ref: str, image_id: str) -> PaddleSpottingRuntimeContract:
            return build_paddle_spotting_runtime_contract(
                manifest_bytes=manifest_bytes,
                manifest_sha256=manifest_sha256,
                observed_file_bytes=observed_file_bytes,
                volume_name=volume_name,
                llama_image_ref=image_ref,
                llama_image_id=image_id,
                compose_file=compose_file,
                environment=os.environ,
            )

        contract = self._contract_or_reseal(
            engine_key="PaddleOCR VL Spotting",
            volume_name=volume_name,
            manifest_bytes=manifest_bytes,
            image_ref=PADDLEOCR_SPOTTING_LLAMA_CPP_IMAGE_REF,
            image_id=llama_image_id,
            build=build,
            allow_repair=allow_repair,
            retry=lambda: self._paddle_spotting_runtime_contract(
                force_refresh=True,
                allow_repair=False,
            ),
        )
        self._paddle_spotting_runtime_contract_cache = contract
        return contract

    def _ensure_paddle_spotting_runtime_image(self) -> None:
        image_ref = PADDLEOCR_SPOTTING_LLAMA_CPP_IMAGE_REF
        if self._inspect_docker_image_id(image_ref):
            return
        raise self._build_setup_error(
            "PaddleOCR VL Spotting",
            f"The setup-sealed PaddleOCR-VL Spotting image is missing: {image_ref}",
        )

    def _mangalmm_runtime_contract(
        self,
        *,
        force_refresh: bool = False,
        allow_repair: bool = True,
    ) -> MangaLMMRuntimeContract:
        if (
            self._mangalmm_runtime_contract_cache is not None
            and not force_refresh
        ):
            return self._mangalmm_runtime_contract_cache

        compose_file = Path(self._config_for("MangaLMM")["compose_file"])
        if not compose_file.is_file():
            raise FileNotFoundError(compose_file)
        volume_name = validate_mangalmm_volume_name(
            os.environ.get(
                "MANGALMM_MODEL_VOLUME",
                DEFAULT_MANGALMM_MODEL_VOLUME,
            )
        )
        try:
            (
                manifest_bytes,
                manifest_sha256,
                observed_file_bytes,
            ) = self._probe_mangalmm_model_volume(
                volume_name=volume_name,
                image_ref=MANGALMM_LLAMA_CPP_IMAGE_REF,
            )
        except _ManagedVolumeNotProvisioned as exc:
            if not allow_repair:
                raise self._build_setup_error("MangaLMM", str(exc)) from exc
            # 볼륨 자체가 없거나 비었다. 준비 스크립트가 원본을 찾아 채운다.
            self._repair_managed_runtime_volume(
                engine_key="MangaLMM",
                volume_name=volume_name,
                detail=str(exc),
                message=(
                    "MangaLMM 런타임 볼륨을 준비하는 중... "
                    "모델을 내려받아야 하면 오래 걸립니다."
                ),
            )
            return self._mangalmm_runtime_contract(force_refresh=True, allow_repair=False)
        llama_image_id = self._inspect_docker_image_id(
            MANGALMM_LLAMA_CPP_IMAGE_REF
        )
        if not llama_image_id:
            raise MangaLMMRuntimeContractError(
                "Pinned MangaLMM llama.cpp image is not installed."
            )
        def build(image_ref: str, image_id: str) -> MangaLMMRuntimeContract:
            return build_mangalmm_runtime_contract(
                manifest_bytes=manifest_bytes,
                manifest_sha256=manifest_sha256,
                observed_file_bytes=observed_file_bytes,
                volume_name=volume_name,
                llama_image_ref=image_ref,
                llama_image_id=image_id,
                compose_file=compose_file,
                environment=os.environ,
            )

        contract = self._contract_or_reseal(
            engine_key="MangaLMM",
            volume_name=volume_name,
            manifest_bytes=manifest_bytes,
            image_ref=MANGALMM_LLAMA_CPP_IMAGE_REF,
            image_id=llama_image_id,
            build=build,
            allow_repair=allow_repair,
            retry=lambda: self._mangalmm_runtime_contract(
                force_refresh=True,
                allow_repair=False,
            ),
        )
        self._mangalmm_runtime_contract_cache = contract
        return contract

    def _ensure_mangalmm_runtime_image(self) -> None:
        image_ref = MANGALMM_LLAMA_CPP_IMAGE_REF
        if self._inspect_docker_image_id(image_ref):
            return
        raise self._build_setup_error(
            "MangaLMM",
            f"The setup-sealed MangaLMM image is missing: {image_ref}",
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
            raise _ManagedVolumeNotProvisioned(
                "Prepared PaddleOCR llama.cpp model volume does not exist: "
                f"{volume_name}"
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
        from modules.utils.llama_cpp_runtime import remove_named_container

        remove_named_container("comic-translate-paddleocr-vl-volume-probe")
        completed = run_docker_command(
            [
                "docker",
                "run",
                "--name",
                "comic-translate-paddleocr-vl-volume-probe",
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
            raise _ManagedVolumeNotProvisioned(
                "Prepared PaddleOCR llama.cpp model volume is incomplete: "
                f"{volume_name}\n{detail}"
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

    def _probe_mangalmm_model_volume(
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
            raise _ManagedVolumeNotProvisioned(
                "Prepared MangaLMM model volume does not exist: "
                f"{volume_name}"
            )
        try:
            volume_labels = json.loads(
                (volume_inspection.stdout or "").strip() or "{}"
            )
        except json.JSONDecodeError as exc:
            raise self._build_setup_error(
                "MangaLMM",
                f"Unable to parse Docker labels for MangaLMM volume: {volume_name}",
            ) from exc
        expected_labels = {
            "comic-translate.runtime": "MangaLMM-llama.cpp",
            "comic-translate.preparation-version": str(
                MANGALMM_RUNTIME_PREPARATION_VERSION
            ),
        }
        if not isinstance(volume_labels, dict) or any(
            str(volume_labels.get(key, "")) != expected
            for key, expected in expected_labels.items()
        ):
            raise self._build_setup_error(
                "MangaLMM",
                (
                    "MangaLMM volume labels do not match the preparation "
                    f"contract: {volume_name}\n"
                    f"Expected labels: {expected_labels}\n"
                    f"Actual labels: {volume_labels}"
                ),
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
        from modules.utils.llama_cpp_runtime import remove_named_container

        remove_named_container("comic-translate-mangalmm-volume-probe")
        completed = run_docker_command(
            [
                "docker",
                "run",
                "--name",
                "comic-translate-mangalmm-volume-probe",
                "--rm",
                "--pull",
                "never",
                "-e",
                f"READY_MANIFEST={DEFAULT_MANGALMM_READY_MANIFEST}",
                "-e",
                f"MODEL_FILE={MANGALMM_MODEL_NAME}",
                "-e",
                f"MMPROJ_FILE={MANGALMM_MMPROJ_NAME}",
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
            raise _ManagedVolumeNotProvisioned(
                "Prepared MangaLMM model volume is incomplete: "
                f"{volume_name}\n{detail}"
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
                MANGALMM_MODEL_NAME: int(values["model_bytes"]),
                MANGALMM_MMPROJ_NAME: int(values["mmproj_bytes"]),
            }
        except (KeyError, ValueError, binascii.Error) as exc:
            raise self._build_setup_error(
                "MangaLMM",
                "Unable to parse the prepared MangaLMM volume probe output: "
                f"{completed.stdout}",
            ) from exc
        return manifest_bytes, manifest_sha256, observed_file_bytes

    def _hunyuan_ocr_runtime_contract(
        self,
        *,
        force_refresh: bool = False,
        allow_repair: bool = True,
    ) -> HunyuanOCRRuntimeContract:
        if (
            self._hunyuan_ocr_runtime_contract_cache is not None
            and not force_refresh
        ):
            return self._hunyuan_ocr_runtime_contract_cache

        compose_file = Path(self._config_for("HunyuanOCR")["compose_file"])
        if not compose_file.is_file():
            raise FileNotFoundError(compose_file)
        volume_name = validate_hunyuan_ocr_volume_name(
            os.environ.get(
                "HUNYUAN_OCR_MODEL_VOLUME",
                DEFAULT_HUNYUAN_OCR_MODEL_VOLUME,
            )
        )
        try:
            (
                manifest_bytes,
                manifest_sha256,
                observed_file_bytes,
            ) = self._probe_hunyuan_ocr_model_volume(
                volume_name=volume_name,
                image_ref=HUNYUAN_OCR_LLAMA_CPP_IMAGE_REF,
            )
        except _ManagedVolumeNotProvisioned as exc:
            if not allow_repair:
                raise self._build_setup_error("HunyuanOCR", str(exc)) from exc
            # 볼륨 자체가 없거나 비었다. 준비 스크립트가 원본을 찾아 채운다.
            self._repair_managed_runtime_volume(
                engine_key="HunyuanOCR",
                volume_name=volume_name,
                detail=str(exc),
                message=(
                    "HunyuanOCR 런타임 볼륨을 준비하는 중... "
                    "모델을 내려받아야 하면 오래 걸립니다."
                ),
            )
            return self._hunyuan_ocr_runtime_contract(force_refresh=True, allow_repair=False)
        llama_image_id = self._inspect_docker_image_id(
            HUNYUAN_OCR_LLAMA_CPP_IMAGE_REF
        )
        if not llama_image_id:
            raise HunyuanOCRRuntimeContractError(
                "고정된 HunyuanOCR llama.cpp 이미지가 설치되어 있지 않습니다."
            )
        def build(image_ref: str, image_id: str) -> HunyuanOCRRuntimeContract:
            return build_hunyuan_ocr_runtime_contract(
                manifest_bytes=manifest_bytes,
                manifest_sha256=manifest_sha256,
                observed_file_bytes=observed_file_bytes,
                volume_name=volume_name,
                llama_image_ref=image_ref,
                llama_image_id=image_id,
                compose_file=compose_file,
                environment=os.environ,
            )

        contract = self._contract_or_reseal(
            engine_key="HunyuanOCR",
            volume_name=volume_name,
            manifest_bytes=manifest_bytes,
            image_ref=HUNYUAN_OCR_LLAMA_CPP_IMAGE_REF,
            image_id=llama_image_id,
            build=build,
            allow_repair=allow_repair,
            retry=lambda: self._hunyuan_ocr_runtime_contract(
                force_refresh=True,
                allow_repair=False,
            ),
        )
        self._hunyuan_ocr_runtime_contract_cache = contract
        return contract

    def _ensure_hunyuan_ocr_runtime_image(self) -> None:
        image_ref = HUNYUAN_OCR_LLAMA_CPP_IMAGE_REF
        if self._inspect_docker_image_id(image_ref):
            return
        raise self._build_setup_error(
            "HunyuanOCR",
            f"The setup-sealed HunyuanOCR image is missing: {image_ref}",
        )

    def _probe_hunyuan_ocr_model_volume(
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
            raise _ManagedVolumeNotProvisioned(
                f"준비된 HunyuanOCR 모델 volume이 없습니다: {volume_name}"
            )
        try:
            volume_labels = json.loads(
                (volume_inspection.stdout or "").strip() or "{}"
            )
        except json.JSONDecodeError as exc:
            raise self._build_setup_error(
                "HunyuanOCR",
                f"HunyuanOCR volume의 Docker label을 파싱할 수 없습니다: {volume_name}",
            ) from exc
        expected_labels = {
            "comic-translate.runtime": "HunyuanOCR-llama.cpp",
            "comic-translate.preparation-version": str(
                HUNYUAN_OCR_RUNTIME_PREPARATION_VERSION
            ),
        }
        if not isinstance(volume_labels, dict) or any(
            str(volume_labels.get(key, "")) != expected
            for key, expected in expected_labels.items()
        ):
            raise self._build_setup_error(
                "HunyuanOCR",
                (
                    "HunyuanOCR volume label이 준비 계약과 다릅니다: "
                    f"{volume_name}\n"
                    f"기대한 label: {expected_labels}\n"
                    f"실제 label: {volume_labels}"
                ),
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
        from modules.utils.llama_cpp_runtime import remove_named_container

        remove_named_container("comic-translate-hunyuanocr-volume-probe")
        completed = run_docker_command(
            [
                "docker",
                "run",
                "--name",
                "comic-translate-hunyuanocr-volume-probe",
                "--rm",
                "--pull",
                "never",
                "-e",
                f"READY_MANIFEST={DEFAULT_HUNYUAN_OCR_READY_MANIFEST}",
                "-e",
                f"MODEL_FILE={HUNYUAN_OCR_MODEL_NAME}",
                "-e",
                f"MMPROJ_FILE={HUNYUAN_OCR_MMPROJ_NAME}",
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
            raise _ManagedVolumeNotProvisioned(
                "준비된 HunyuanOCR 모델 volume이 불완전합니다: "
                f"{volume_name}\n{detail}"
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
                HUNYUAN_OCR_MODEL_NAME: int(values["model_bytes"]),
                HUNYUAN_OCR_MMPROJ_NAME: int(values["mmproj_bytes"]),
            }
        except (KeyError, ValueError, binascii.Error) as exc:
            raise self._build_setup_error(
                "HunyuanOCR",
                "준비된 HunyuanOCR volume 프로브 출력을 파싱할 수 없습니다: "
                f"{completed.stdout}",
            ) from exc
        return manifest_bytes, manifest_sha256, observed_file_bytes

    def _probe_paddle_spotting_model_volume(
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
            raise _ManagedVolumeNotProvisioned(
                "Prepared PaddleOCR-VL Spotting model volume does not exist: "
                f"{volume_name}"
            )
        try:
            volume_labels = json.loads(
                (volume_inspection.stdout or "").strip() or "{}"
            )
        except json.JSONDecodeError as exc:
            raise self._build_setup_error(
                "PaddleOCR VL Spotting",
                (
                    "Unable to parse Docker labels for PaddleOCR-VL "
                    f"Spotting volume: {volume_name}"
                ),
            ) from exc
        expected_labels = {
            "comic-translate.runtime": (
                "PaddleOCR-VL-Spotting-llama.cpp"
            ),
            "comic-translate.preparation-version": str(
                PADDLE_SPOTTING_RUNTIME_PREPARATION_VERSION
            ),
        }
        if not isinstance(volume_labels, dict) or any(
            str(volume_labels.get(key, "")) != expected
            for key, expected in expected_labels.items()
        ):
            raise self._build_setup_error(
                "PaddleOCR VL Spotting",
                (
                    "PaddleOCR-VL Spotting volume labels do not match "
                    f"the preparation contract: {volume_name}\n"
                    f"Expected labels: {expected_labels}\n"
                    f"Actual labels: {volume_labels}"
                ),
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
        from modules.utils.llama_cpp_runtime import remove_named_container

        remove_named_container("comic-translate-paddleocr-vl-spotting-volume-probe")
        completed = run_docker_command(
            [
                "docker",
                "run",
                "--name",
                "comic-translate-paddleocr-vl-spotting-volume-probe",
                "--rm",
                "--pull",
                "never",
                "-e",
                (
                    "READY_MANIFEST="
                    f"{DEFAULT_PADDLE_SPOTTING_READY_MANIFEST}"
                ),
                "-e",
                f"MODEL_FILE={PADDLE_SPOTTING_MODEL_NAME}",
                "-e",
                f"MMPROJ_FILE={PADDLE_SPOTTING_MMPROJ_NAME}",
                "--mount",
                (
                    f"type=volume,source={volume_name},"
                    "target=/models,readonly"
                ),
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
                (completed.stderr or "")
                + "\n"
                + (completed.stdout or "")
            ).strip()
            raise _ManagedVolumeNotProvisioned(
                "Prepared PaddleOCR-VL Spotting model volume is incomplete: "
                f"{volume_name}\n{detail}"
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
                PADDLE_SPOTTING_MODEL_NAME: int(values["model_bytes"]),
                PADDLE_SPOTTING_MMPROJ_NAME: int(
                    values["mmproj_bytes"]
                ),
            }
        except (KeyError, ValueError, binascii.Error) as exc:
            raise self._build_setup_error(
                "PaddleOCR VL Spotting",
                (
                    "Unable to parse the prepared PaddleOCR-VL "
                    f"Spotting volume probe output: {completed.stdout}"
                ),
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

    def _mangalmm_containers_match_contract(
        self,
        container_names: list[str],
    ) -> bool:
        try:
            contract = self._mangalmm_runtime_contract()
        except OperationCancelledError:
            raise
        except Exception:
            logger.warning(
                "Failed to resolve the MangaLMM runtime contract; stale "
                "containers will be recreated.",
                exc_info=True,
            )
            return False

        from modules.utils.llama_cpp_runtime import run_docker_command

        if container_names != ["mangalmm-local-server"]:
            return False
        completed = run_docker_command(
            [
                "docker",
                "inspect",
                "--format",
                (
                    "{{index .Config.Labels "
                    f"\"{MANGALMM_RUNTIME_FINGERPRINT_LABEL}\""
                    "}}|{{.Image}}|"
                    "{{index .Config.Labels "
                    "\"desktop.docker.io/wsl-distro\"}}"
                ),
                "mangalmm-local-server",
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
            fingerprint != contract.fingerprint
            or image_id != contract.llama_image_id
        ):
            return False
        if os.name == "nt" and wsl_distro.strip():
            logger.info(
                "MangaLMM container was created by WSL Compose (%s); Windows "
                "will recreate it so Docker Desktop can manage the Compose "
                "application without invoking wsl.",
                wsl_distro.strip(),
            )
            return False
        return True

    def _paddle_spotting_containers_match_contract(
        self,
        container_names: list[str],
    ) -> bool:
        try:
            contract = self._paddle_spotting_runtime_contract()
        except OperationCancelledError:
            raise
        except Exception:
            logger.warning(
                "Failed to resolve the PaddleOCR-VL Spotting runtime "
                "contract; stale containers will be recreated.",
                exc_info=True,
            )
            return False

        from modules.utils.llama_cpp_runtime import run_docker_command

        if container_names != ["paddleocr-spotting-llamacpp"]:
            return False
        completed = run_docker_command(
            [
                "docker",
                "inspect",
                "--format",
                (
                    "{{index .Config.Labels "
                    f"\"{PADDLE_SPOTTING_RUNTIME_FINGERPRINT_LABEL}\""
                    "}}|{{.Image}}|"
                    "{{index .Config.Labels "
                    "\"desktop.docker.io/wsl-distro\"}}"
                ),
                "paddleocr-spotting-llamacpp",
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
            fingerprint != contract.fingerprint
            or image_id != contract.llama_image_id
        ):
            return False
        if os.name == "nt" and wsl_distro.strip():
            logger.info(
                "PaddleOCR-VL Spotting container was created by WSL "
                "Compose (%s); Windows will recreate it so Docker "
                "Desktop can manage the Compose application without "
                "invoking wsl.",
                wsl_distro.strip(),
            )
            return False
        return True

    def _managed_containers_match_contract(
        self,
        engine_key: str,
        container_names: list[str],
    ) -> bool:
        if engine_key == "PaddleOCR VL":
            return self._paddle_containers_match_contract(container_names)
        if engine_key == "PaddleOCR VL Spotting":
            return self._paddle_spotting_containers_match_contract(
                container_names
            )
        if engine_key == "MangaLMM":
            return self._mangalmm_containers_match_contract(container_names)
        return False

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
        service = {
            "PaddleOCR VL": "paddleocr_vl",
            "PaddleOCR VL Spotting": "paddleocr_vl_spotting",
            "HunyuanOCR": "hunyuanocr",
            "MangaLMM": "mangalmm",
        }.get(engine_key, engine_key.lower().replace(" ", "_"))
        event = {
            "phase": "ocr_startup",
            "service": service,
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
