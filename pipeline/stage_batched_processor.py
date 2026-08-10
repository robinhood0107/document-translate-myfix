from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import imkit as imk
import numpy as np
from PySide6.QtCore import QCoreApplication

from app.path_materialization import ensure_path_materialized
from app.projects.stage_checkpoints import (
    apply_translation_checkpoint,
    build_detection_fingerprint,
    build_detection_identity,
    build_inpaint_fingerprint,
    build_inpaint_identity,
    build_project_ocr_fingerprint,
    build_project_ocr_identity,
    build_render_fingerprint,
    build_render_identity,
    build_skipped_stage_fingerprint,
    build_translation_fingerprint,
    build_translation_identity,
    decoded_image_sha256,
    detection_structure_signature,
    lookup_inpaint_checkpoint,
    lookup_detection_checkpoint,
    lookup_ocr_checkpoint,
    lookup_render_checkpoint,
    lookup_translation_checkpoint,
    materialize_render_checkpoint_output,
    open_project_stage_checkpoint_store,
    project_checkpoint_page_key,
    record_inpaint_checkpoint,
    record_detection_checkpoint,
    record_ocr_checkpoint,
    record_render_checkpoint,
    record_translation_checkpoint,
    registered_inpainter_model_identity,
    resolve_font_identity,
    restore_inpaint_block_state,
    snapshot_project_render_blocks,
    snapshot_project_translations,
)
from app.ui.messages import Messages
from modules.detection.processor import TextBlockDetector
from modules.ocr.factory import OCRFactory
from modules.ocr.local_runtime import LocalOCRRuntimeManager
from modules.ocr.paddle_crop.engine import PaddleOCRVLEngine
from modules.ocr.persistent_cache import (
    OCRPersistentResultCache,
    canonical_sha256,
    snapshot_raw_ocr_result,
)
from modules.ocr.common.result_contract import (
    canonicalize_exact_duplicate_blocks,
    finalize_ocr_processing_contracts,
    select_translate_inpaint_blocks,
)
from modules.ocr.selection import (
    STAGE_BATCHED_WORKFLOW_MODE,
    resolve_stage_batched_ocr_policy,
)
from modules.rendering.render import (
    _register_render_fallback_system_fonts,
    _render_font_has_real_glyph,
    get_best_render_area,
    resolve_render_glyph_fallback_font_family,
    resolve_render_symbol_fallback_font_family,
    should_use_strict_render_symbols,
)
from modules.translation.local_runtime import LocalGemmaRuntimeManager
from modules.translation.processor import Translator
from modules.utils.correction_dictionary import (
    apply_ocr_result_dictionary,
    apply_translation_result_dictionary,
)
from modules.utils.automatic_output import (
    DEFAULT_OUTPUT_ARCHIVE_IMAGE_FORMAT,
    DEFAULT_OUTPUT_IMAGE_FORMAT,
    build_archive_page_file_name,
    build_archive_staging_dir,
    build_output_file_name,
    is_single_archive_mode,
    reserve_unique_path,
    resolve_individual_output_format,
)
from modules.utils.device import resolve_device
from modules.utils.export_paths import (
    build_export_timestamp,
    export_run_root,
    resolve_export_directory,
)
from modules.utils.exceptions import OperationCancelledError
from modules.utils.stage_sweep_eta import StageSweepEtaEstimator
from modules.utils.gpu_handoff import (
    DEFAULT_MANAGED_SLEEPING_RELEASE_RATIO,
    DEFAULT_MANAGED_SLEEPING_RESIDUAL_MB,
    gpu_release_enforcement_enabled,
    DEFAULT_VRAM_RELEASE_EXPECTED_RATIO,
    wait_for_global_vram_release,
)
from modules.utils.gpu_metrics import query_cuda_handoff_metrics
from modules.utils.image_utils import (
    generate_mask,
    release_protected_mask_for_explicit_additions,
)
from modules.utils.inpaint_cleanup import apply_duplicate_bubble_inner_fill, refine_bubble_residue_inpaint
from modules.utils.inpaint_composite import (
    composite_with_edit_mask,
    count_changed_outside_edit_mask,
)
from modules.inpainting.runtime_contract import inpaint_outside_mask_message
from modules.utils.language_utils import get_language_code, language_codes
from modules.utils.ocr_debug import (
    all_empty_blocks_are_rejected,
    drop_embedded_ui_ocr_blocks,
    drop_rejected_empty_ocr_blocks,
    split_inpaint_protected_ocr_blocks,
)
from modules.utils.ocr_quality import summarize_ocr_quality
from modules.utils.pipeline_config import get_config, get_inpainter_runtime, inpaint_map
from modules.utils.render_style_policy import VERTICAL_ALIGNMENT_CENTER
from pipeline.inpaint_cleanup_job import (
    InpaintCleanupInput,
    run_inpaint_cleanup,
)
from modules.utils.run_report import build_run_report, write_run_report
from modules.utils.text_normalization import RENDER_NORMALIZABLE_GLYPHS
from modules.utils.textblock import ensure_text_block_id, sort_blk_list
from modules.utils.translator_utils import get_raw_text, get_raw_translation

from .batch_processor import BatchProcessor
from .render_pool import QtRenderPool
from .render_worker import RenderJobInput, RenderJobResult, run_render_job
from .runtime_resource_arbiter import (
    RuntimeModelState,
    RuntimeResourceArbiter,
)

logger = logging.getLogger(__name__)


@dataclass
class StagePageContext:
    image_path: str
    image_name: str
    source_lang: str
    target_lang: str
    directory: str = ""
    archive_bname: str = ""
    export_token: str = ""
    export_root: str = ""
    image: Any | None = None
    blk_list: list[Any] = field(default_factory=list)
    precomputed_mask_details: dict[str, Any] | None = None
    detector_key: str = ""
    detector_engine: str = ""
    detector_device: str = ""
    source_decoded_sha256: str = ""
    project_checkpoint_page_key: str = ""
    detection_fingerprint: str = ""
    detection_structure_signature: str = ""
    detection_checkpoint_status: str = "disabled"
    project_ocr_fingerprint: str = ""
    project_ocr_identity: dict[str, Any] = field(default_factory=dict)
    project_ocr_hit: Any | None = None
    project_ocr_checkpoint_status: str = "disabled"
    project_translation_snapshot: list[dict[str, Any]] = field(
        default_factory=list
    )
    project_render_blocks: list[Any] = field(default_factory=list)
    project_viewer_state: dict[str, Any] = field(default_factory=dict)
    project_translation_identity: dict[str, Any] = field(default_factory=dict)
    project_translation_fingerprint: str = ""
    project_translation_checkpoint_status: str = "disabled"
    project_inpaint_identity: dict[str, Any] = field(default_factory=dict)
    project_inpaint_fingerprint: str = ""
    project_inpaint_checkpoint_status: str = "disabled"
    project_inpaint_artifact_sha256: str = ""
    project_render_identity: dict[str, Any] = field(default_factory=dict)
    project_render_fingerprint: str = ""
    project_render_checkpoint_status: str = "disabled"
    page_ocr_metrics: dict[str, int] = field(default_factory=dict)
    ocr_canonicalization_summary: dict[str, Any] = field(
        default_factory=dict
    )
    ocr_processing_summary: dict[str, Any] = field(
        default_factory=dict
    )
    translation_blocks: list[Any] = field(default_factory=list)
    page_translation_metrics: dict[str, int | float] = field(default_factory=dict)
    paddleocr_cache_plan: Any | None = None
    paddleocr_cache_engine: Any | None = None
    raw_mask: Any | None = None
    mask: Any | None = None
    mask_details: dict[str, Any] = field(default_factory=dict)
    patches: list[dict[str, Any]] = field(default_factory=list)
    inpaint_input_img: Any | None = None
    cleanup_stats: dict[str, Any] = field(
        default_factory=lambda: {"applied": False, "component_count": 0, "block_count": 0}
    )
    inpaint_diagnostics: dict[str, Any] = field(default_factory=dict)
    no_text_detected: bool = False
    failed_stage: str = ""
    failed_reason: str = ""
    # 이 페이지가 실제로 남긴 파일. 배치 끝의 정합성 검사가 이 값을 본다.
    output_path: str = ""
    # 폴백으로 저장했다면 무엇으로 저장했는지. 빈 문자열이면 정상 렌더다.
    output_fallback_kind: str = ""
    # 페이지가 끝난 뒤 큰 배열을 놓아주기 전에 옮겨 담는 값들. 스윕이 끝난 다음
    # 집계하려고 이미지 전체를 붙들고 있을 이유는 없다.
    mask_pixel_count: int = 0
    released_buffer_bytes: int = 0

    def release_page_buffers(self) -> int:
        """이 페이지의 전체 해상도 배열을 놓아준다. 놓아준 바이트 수를 돌려준다.

        스테이지 배치 파이프라인은 실행 내내 모든 페이지의 컨텍스트를 리스트로
        들고 있다. 페이지당 원본, 인페인팅 결과, 마스크 두 장, 패치가 전부
        살아 있으면 4K 페이지 기준 60 MiB 를 넘고, 수백 페이지에서는 프로세스가
        수 GB 짜리 24 MiB 할당조차 실패하는 지경이 된다. 렌더까지 끝난 페이지의
        픽셀 데이터는 아무도 다시 보지 않으므로 그 자리에서 놓아준다.

        나중에 필요한 집계값(마스크 픽셀 수)은 놓아주기 전에 옮겨 담는다.
        """

        released = 0
        if self.mask is not None:
            try:
                self.mask_pixel_count = int(np.count_nonzero(self.mask))
            except Exception:
                self.mask_pixel_count = 0
        for name in (
            "image",
            "inpaint_input_img",
            "raw_mask",
            "mask",
            "project_ocr_hit",
            "precomputed_mask_details",
        ):
            value = getattr(self, name, None)
            released += _approximate_buffer_bytes(value)
            setattr(self, name, None)
        released += _approximate_buffer_bytes(self.patches)
        self.patches = []
        # `mask_details` 는 raw/final 마스크를 그대로 다시 참조한다. 비우지 않으면
        # 위에서 놓아준 배열이 그대로 살아남는다.
        released += _approximate_buffer_bytes(self.mask_details)
        self.mask_details = {}
        self.released_buffer_bytes += released
        return released


def _approximate_buffer_bytes(value: Any) -> int:
    """중첩 컨테이너 안 numpy 배열이 차지하는 바이트 수의 근사치.

    진단 로그용이다. 정확할 필요는 없고, 순환 참조에서 멈추기만 하면 된다.
    """

    return _approximate_buffer_bytes_inner(value, set())


def _approximate_buffer_bytes_inner(value: Any, seen: set[int]) -> int:
    if value is None:
        return 0
    marker = id(value)
    if marker in seen:
        return 0
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (str, bytes, int, float, bool)):
        return 0
    seen.add(marker)
    if isinstance(value, dict):
        return sum(
            _approximate_buffer_bytes_inner(item, seen) for item in value.values()
        )
    if isinstance(value, (list, tuple, set)):
        return sum(_approximate_buffer_bytes_inner(item, seen) for item in value)
    # 체크포인트 히트처럼 배열을 필드로 들고 있는 작은 객체.
    slots = getattr(value, "__dict__", None)
    if isinstance(slots, dict):
        return sum(
            _approximate_buffer_bytes_inner(item, seen) for item in slots.values()
        )
    return 0


@dataclass
class _PendingRenderJob:
    """전용 렌더 워커에 제출된 작업 하나를 추적한다 (파이프라인 스레드 전용 상태)."""

    future: Future
    ctx: StagePageContext
    index: int
    total_images: int
    file_on_display: bool
    output_root: str
    started_monotonic: float


class StageBatchedProcessor(BatchProcessor):
    # 각 sweep 은 emit_progress 에 자기 이름을 직접 넘긴다. 이 표는 그 이름을 잃어버린
    # 호출이 있을 때의 안전망일 뿐이며, 로그 라벨의 근거로 쓰이지 않는다. 숫자 step 에서
    # 이름을 되찾는 방식이 오해를 만들었다. step 3 은 마스크 생성과 LaMa 통과를 모두
    # 하는 단계인데 레거시 표의 3 은 `pre-inpaint-setup` 이라, "인페인트 준비" 가 366 줄
    # 찍혔다.
    STAGE_NAMES_BY_STEP = {
        0: 'detect-all',
        2: 'ocr-all',
        3: 'inpaint-all',
        7: 'translate-all',
        9: 'render-all',
        10: 'save-and-finish',
    }

    # sweep 이 실제로 도는 순서. 숫자 step 은 레거시 호환용 식별자라서 정렬해도 실행
    # 순서가 되지 않는다. 번역(7)이 인페인팅(3)보다 먼저 돌기 때문이다.
    STAGE_SWEEP_ORDER = (
        'detect-all',
        'ocr-all',
        'translate-all',
        'inpaint-all',
        'render-all',
        'save-and-finish',
    )

    # 사용자에게 보일 단계 이름. 내부 sweep 이름을 그대로 쓰면 무슨 일이 일어나는지
    # 알 수 없다.
    STAGE_LABELS = {
        'detect-all': '텍스트 영역 검출',
        'ocr-all': '텍스트 인식(OCR)',
        'inpaint-all': '원본 텍스트 제거(인페인팅)',
        'translate-all': '번역',
        'render-all': '번역문 렌더링',
        'save-and-finish': '저장 및 마무리',
    }

    def _stage_eta_estimator(self, total: int):
        """이 실행의 (단계 x 페이지) 추정기. 페이지 수가 바뀌면 새로 만든다."""

        estimator = getattr(self, "_stage_eta", None)
        if estimator is None or estimator.page_total != max(int(total), 0):
            estimator = StageSweepEtaEstimator(
                page_total=total,
                stage_order=self.STAGE_SWEEP_ORDER,
            )
            estimator.start_run(time.monotonic())
            # 지난 실행들에서 측정한 단계 속도로 시작한다. 내장 사전 비중은 짐작이라
            # 첫 실행에서 남은 시간이 튄다. 이력이 있으면 첫 페이지부터 맞는다.
            tracker = getattr(self.main_page, "_automatic_progress_tracker", None)
            reader = getattr(tracker, "read_stage_rates", None)
            if callable(reader):
                try:
                    estimator.seed_from_history(reader())
                except Exception:
                    logger.debug("Could not seed the stage ETA model.", exc_info=True)
            # 컨테이너 기동·모델 적재처럼 페이지 수와 무관한 고정 비용도 이력에서
            # 채운다. 이게 없으면 아직 시작하지 않은 단계의 시작 비용만큼 남은
            # 시간을 계속 과소평가하고, 그 단계로 넘어가는 순간 위로 튄다.
            startup_reader = getattr(tracker, "read_stage_startups", None)
            if callable(startup_reader):
                try:
                    estimator.seed_startup_from_history(startup_reader())
                except Exception:
                    logger.debug(
                        "Could not seed the stage startup costs.",
                        exc_info=True,
                    )
            self._stage_eta = estimator
        return estimator

    def _persist_stage_rates(self) -> None:
        """이번 실행에서 측정한 단계 속도를 다음 실행을 위해 남긴다."""

        estimator = getattr(self, "_stage_eta", None)
        if estimator is None:
            return
        tracker = getattr(self.main_page, "_automatic_progress_tracker", None)
        writer = getattr(tracker, "record_stage_rates", None)
        if callable(writer):
            try:
                writer(estimator.measured_per_page_by_stage())
            except Exception:
                logger.debug("Could not persist the stage ETA model.", exc_info=True)
        startup_writer = getattr(tracker, "record_stage_startups", None)
        if callable(startup_writer):
            try:
                startup_writer(estimator.measured_startup_by_stage())
            except Exception:
                logger.debug(
                    "Could not persist the stage startup costs.",
                    exc_info=True,
                )

    def observe_progress(
        self,
        stage_name: str,
        index: int,
        total: int,
        step: int,
        steps: int,
    ) -> dict[str, float | None]:
        """단계 sweep 모델로 진행률과 남은 시간을 계산한다.

        레거시 외삽은 첫 sweep 의 마지막 페이지를 실행 종료로 착각해
        `overall=99.8% eta=00:00:02` 를 냈다. 여기서는 남은 sweep 이 모두 계산에
        들어간다.
        """

        estimator = self._stage_eta_estimator(total)
        estimator.observe(stage_name, index, time.monotonic())
        return {
            "progress_fraction": estimator.progress_fraction(),
            "eta_seconds": estimator.remaining_seconds(),
            # 파이프라인 전 단계의 남은 시간과 상태. UI 가 마우스를 올렸을 때
            # 보여준다.
            "eta_by_stage": [
                {**row, "label": self.STAGE_LABELS.get(row["stage"], row["stage"])}
                for row in estimator.remaining_by_stage()
            ],
        }

    def describe_progress(self, stage_name: str, index: int, total: int) -> str:
        label = self.STAGE_LABELS.get(stage_name, stage_name)
        return f"{label}: {index + 1}/{total} 페이지"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._timestamp: str = ""
        self._prewarm_executor: ThreadPoolExecutor | None = None
        self._prewarm_jobs: dict[str, Future] = {}
        # 페이지 캐시 프리페치는 GPU 를 쓰지 않고 디스크만 읽는다. 런타임 명령 실행기
        # (max_workers=1)를 8초 동안 붙들면 다음 런타임 명령이 그만큼 밀리므로,
        # 별도 실행기에 둔다.
        self._page_cache_executor: ThreadPoolExecutor | None = None
        self._gemma_prefetch_job: Future | None = None
        self._ocr_container_prewarm_job: Future | None = None
        self._prewarm_cancel_event = threading.Event()
        self._runtime_resource_arbiter_instance = RuntimeResourceArbiter()
        self._inpainter_runtime_lease_held = False
        self._runtime_gpu_start_baselines: dict[str, dict[str, Any]] = {}
        self._runtime_gpu_release_required: set[str] = set()
        self._runtime_progress_lock = threading.RLock()
        self._runtime_progress_started: dict[tuple[str, str], float] = {}
        self._paddleocr_cache_store: OCRPersistentResultCache | None = None
        self._paddleocr_cache_identity: dict[str, Any] | None = None
        self._project_checkpoint_store = None
        self._project_checkpoint_page_keys: list[str] = []
        # Phase 3a: 인페인팅 sweep 뒤에 렌더(약 300ms/page)를 숨기는 전용 단일
        # 워커. 렌더 워커는 순수 함수이므로 별도 실행기에 둔다 — 공유
        # QThreadPool/prewarm 실행기와 절대 섞지 않는다.
        self._render_executor: QtRenderPool | None = None
        self._render_cancel_event = threading.Event()
        self._pending_render_jobs: list[_PendingRenderJob] = []
        self._released_page_buffer_bytes = 0
        self._render_context_cache: tuple[Any, str] | None = None

    def _stage_tr(self, text: str) -> str:
        return QCoreApplication.translate("StageBatchedProcessor", text)

    def _prewarm_progress(self, **payload: Any) -> None:
        payload.setdefault("phase", "runtime_prewarm")
        payload.setdefault("service", "batch")
        payload.setdefault("status", "running")
        payload["runtime_prewarm"] = True
        self._report_runtime_progress(**payload)

    def _raise_if_cancelled(self) -> None:
        checker = getattr(self.main_page, "is_current_task_cancelled", None)
        try:
            cancelled = bool(checker()) if callable(checker) else bool(self._is_cancelled())
        except Exception:
            cancelled = bool(self._is_cancelled())
        if cancelled:
            # 렌더 워커(별도 스레드)가 취소를 즉시 보도록 함께 신호한다 —
            # 그러지 않으면 teardown 때까지 진행 중인 렌더 작업이 계속 돈다.
            render_cancel_event = getattr(self, "_render_cancel_event", None)
            if render_cancel_event is not None:
                render_cancel_event.set()
            raise OperationCancelledError("Automatic translation was cancelled.")

    def _ensure_prewarm_executor(self) -> ThreadPoolExecutor:
        if self._prewarm_executor is None:
            self._prewarm_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="ct-runtime-command",
            )
        return self._prewarm_executor

    def _runtime_resource_arbiter(self) -> RuntimeResourceArbiter:
        arbiter = getattr(
            self,
            "_runtime_resource_arbiter_instance",
            None,
        )
        if not isinstance(arbiter, RuntimeResourceArbiter):
            arbiter = RuntimeResourceArbiter()
            self._runtime_resource_arbiter_instance = arbiter
        return arbiter

    def _runtime_gpu_baselines(self) -> dict[str, dict[str, Any]]:
        baselines = getattr(self, "_runtime_gpu_start_baselines", None)
        if not isinstance(baselines, dict):
            baselines = {}
            self._runtime_gpu_start_baselines = baselines
        return baselines

    def _runtime_gpu_release_services(self) -> set[str]:
        required = getattr(self, "_runtime_gpu_release_required", None)
        if not isinstance(required, set):
            required = set()
            self._runtime_gpu_release_required = required
        return required

    def _capture_runtime_gpu_start_baseline(self, service: str) -> None:
        """Remember driver-total memory before a managed GPU model loads."""

        self._runtime_gpu_baselines()[service] = query_cuda_handoff_metrics()

    def _verify_managed_runtime_gpu_release(
        self,
        service: str,
        release_report: Any,
        *,
        before: dict[str, Any],
    ) -> dict[str, Any]:
        """관리형 Docker 모델의 GPU 반환을 측정해 게이트 결과로 돌려준다.

        측정과 기록은 항상 수행한다. 결과를 실패로 처리할지는
        `gpu_handoff.gpu_release_enforcement_enabled()`가 정하며 기본값은
        처리하지 않는 것이다. 드라이버 보고 노이즈 하나로 폴더 전체 작업이
        중단되는 편이 미확인 잔여 메모리보다 나쁘다.
        """

        report = release_report if isinstance(release_report, dict) else {}
        release_required = self._runtime_gpu_release_services()
        if bool(report.get("gpu_release_expected", False)):
            release_required.add(service)
        sleeping_runtime = str(report.get("runtime_state") or "") == "sleeping"
        gate = wait_for_global_vram_release(
            before,
            gpu_release_expected=service in release_required,
            driver_baseline=self._runtime_gpu_baselines().get(service),
            expected_drop_ratio=(
                DEFAULT_MANAGED_SLEEPING_RELEASE_RATIO
                if sleeping_runtime
                else DEFAULT_VRAM_RELEASE_EXPECTED_RATIO
            ),
            residual_allowance_mb=(
                DEFAULT_MANAGED_SLEEPING_RESIDUAL_MB
                if sleeping_runtime
                else 0.0
            ),
        )
        self._record_runtime_performance(
            service=service,
            operation="vram_release_gate",
            elapsed_ms=float(gate.get("elapsed_sec", 0.0) or 0.0) * 1000.0,
            outcome=str(gate.get("status") or "unknown"),
        )
        if bool(gate.get("observed", False)):
            release_required.discard(service)
            self._runtime_gpu_baselines().pop(service, None)
        return gate

    def _managed_runtime_stale_cleanup(
        self,
        service: str,
        runtime_manager: Any,
    ) -> None:
        """Stop a cancelled model load without bypassing its VRAM gate."""

        before = query_cuda_handoff_metrics()
        if self._router_runtime_is_active(runtime_manager):
            release_report = runtime_manager.shutdown(
                resource_arbiter=self._runtime_resource_arbiter(),
                runtime_service=service,
                # 이미 취소된 적재를 되돌리는 정리다. 취소 검사기를 넘기면 정리
                # 자체가 즉시 취소되어 컨테이너가 남는다.
                cancel_checker=self._cleanup_cancel_checker,
            )
        else:
            release_report = runtime_manager.shutdown()
        gate = self._verify_managed_runtime_gpu_release(
            service,
            release_report,
            before=before,
        )
        self._handle_unconfirmed_gpu_release(service, gate)

    def _handle_unconfirmed_gpu_release(
        self,
        service: str,
        gate: dict[str, Any],
    ) -> None:
        """GPU 반환이 확인되지 않았을 때의 단일 처리 지점.

        기본값은 경고만 남기고 계속 진행하는 것이다. 회귀 조사에서 다시 강제하려면
        `COMIC_TRANSLATE_ENFORCE_GPU_RELEASE=1`을 설정한다.
        """

        if not bool(gate.get("required")) or bool(gate.get("observed")):
            return
        if gpu_release_enforcement_enabled():
            raise RuntimeError(
                QCoreApplication.translate(
                    "StageBatchedProcessor",
                    "Managed runtime GPU release was not confirmed.",
                )
            )
        logger.warning(
            "%s 런타임의 GPU 반환을 확인하지 못했습니다(status=%s). 실행은 계속합니다.",
            service,
            gate.get("status", "unknown"),
        )

    @staticmethod
    def _router_runtime_is_active(runtime_manager: Any) -> bool:
        if not isinstance(
            runtime_manager,
            (LocalOCRRuntimeManager, LocalGemmaRuntimeManager),
        ):
            return False
        checker = getattr(runtime_manager, "router_is_active", None)
        if not callable(checker):
            return False
        try:
            return bool(checker())
        except Exception:
            return False

    def _prewarm_cancel_checker(self) -> bool:
        cancel_event = getattr(self, "_prewarm_cancel_event", None)
        if cancel_event is not None and cancel_event.is_set():
            return True
        checker = getattr(self.main_page, "is_current_task_cancelled", None)
        try:
            return bool(checker()) if callable(checker) else bool(self._is_cancelled())
        except Exception:
            return bool(self._is_cancelled())

    @staticmethod
    def _cleanup_cancel_checker() -> bool:
        """런타임 정리는 절대 취소되지 않는다.

        정리는 실행이 끝났거나 취소된 뒤에 돈다. 그 시점에는 일반 취소 검사기가
        이미 True이고(`_shutdown_prewarm_executor`가 취소 이벤트를 설정한 직후에
        정리가 시작된다), 그것을 그대로 넘기면 컨테이너 정지가 자기 자신을 취소해
        `Router terminal cleanup failed: Router operation cancelled` 로 끝난다.
        그러면 컨테이너와 GPU가 그대로 남는다. 정리에는 취소가 없어야 한다.
        """

        return False

    def _reset_prewarm_lifecycle(self) -> None:
        cancel_event = getattr(self, "_prewarm_cancel_event", None)
        if cancel_event is None:
            self._prewarm_cancel_event = threading.Event()
        else:
            cancel_event.clear()
        self._runtime_resource_arbiter().reset()
        self._inpainter_runtime_lease_held = False
        self._runtime_gpu_baselines().clear()
        self._runtime_gpu_release_services().clear()
        with self._runtime_progress_lock:
            self._runtime_progress_started.clear()

    def _runtime_progress_callback(self) -> Callable[[dict[str, Any]], None]:
        def observe(event: dict[str, Any]) -> None:
            payload = dict(event or {})
            service = str(payload.get("service") or "runtime").strip().lower()
            step_key = str(payload.get("step_key") or "runtime").strip().lower()
            status = str(payload.get("status") or "running").strip().lower()
            key = (service, step_key)
            now = time.perf_counter()
            runtime_lock = getattr(self, "_runtime_progress_lock", None)
            if runtime_lock is None:
                runtime_lock = threading.RLock()
                self._runtime_progress_lock = runtime_lock
            if not hasattr(self, "_runtime_progress_started"):
                self._runtime_progress_started = {}
            with runtime_lock:
                if status in {"starting", "running", "waiting", "waiting_health"}:
                    progress_started_now = key not in self._runtime_progress_started
                    self._runtime_progress_started.setdefault(key, now)
                elif status in {"completed", "ready", "failed", "cancelled"}:
                    progress_started_now = False
                    started = self._runtime_progress_started.pop(key, None)
                else:
                    progress_started_now = False
                    started = None
            if status in {"starting", "running", "waiting", "waiting_health"}:
                if progress_started_now and step_key in {
                    "container_start",
                    "container_recreate",
                    "compose_up",
                    "compose_recreate",
                }:
                    self._record_runtime_transition(
                        service=service,
                        to_state="process_starting",
                    )
                elif progress_started_now and step_key in {
                    "health_wait",
                    "model_validation",
                    "prewarm",
                }:
                    self._record_runtime_transition(
                        service=service,
                        to_state="model_loading",
                    )
            elif status in {"completed", "ready", "failed", "cancelled"}:
                elapsed_ms = (
                    max(0.0, now - started) * 1000.0
                    if started is not None
                    else 0.0
                )
                outcome = (
                    "completed"
                    if status in {"completed", "ready"}
                    else status
                )
                self._record_runtime_performance(
                    service=service,
                    operation=step_key,
                    elapsed_ms=elapsed_ms,
                    outcome=outcome,
                )
                if status in {"completed", "ready"}:
                    if step_key in {
                        "container_start",
                        "container_recreate",
                        "compose_up",
                        "compose_recreate",
                    }:
                        to_state = "process_ready"
                    elif step_key in {
                        "health_wait",
                        "health_probe",
                        "model_validation",
                        "prewarm",
                    }:
                        to_state = "model_ready"
                    else:
                        to_state = "process_ready"
                else:
                    to_state = "stopped"
                self._record_runtime_transition(
                    service=service,
                    to_state=to_state,
                    elapsed_ms=elapsed_ms,
                    outcome=outcome,
                )

            callback = getattr(
                self.main_page,
                "report_runtime_progress",
                None,
            )
            if callable(callback):
                callback(payload)

        return observe

    def _start_prewarm(
        self,
        key: str,
        label: str,
        service: str,
        fn: Callable[[], None],
        *,
        stale_cleanup: Callable[[], None] | None = None,
    ) -> None:
        if key in self._prewarm_jobs:
            return
        self._raise_if_cancelled()
        self._prewarm_progress(
            service=service,
            status="starting",
            step_key=f"{key}_prewarm",
            message=self._stage_tr("{label} Docker 예열을 시작합니다.").format(label=label),
        )

        arbiter = self._runtime_resource_arbiter()
        token = arbiter.token(service)

        def runner() -> None:
            if self._prewarm_cancel_checker():
                raise OperationCancelledError(f"{label} prewarm was cancelled before startup.")
            started_at = time.perf_counter()
            outcome = "completed"
            try:
                with arbiter.model_start(
                    token,
                    cancel_checker=self._prewarm_cancel_checker,
                    stale_cleanup=stale_cleanup,
                ):
                    self._capture_runtime_gpu_start_baseline(service)
                    fn()
                if self._prewarm_cancel_checker():
                    with arbiter.model_release(service):
                        if callable(stale_cleanup):
                            stale_cleanup()
                    raise OperationCancelledError(
                        f"{label} prewarm was cancelled after startup."
                    )
            except OperationCancelledError:
                outcome = "cancelled"
                raise
            except Exception:
                outcome = "failed"
                raise
            finally:
                elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                self._record_runtime_performance(
                    service=service,
                    operation="start",
                    elapsed_ms=elapsed_ms,
                    outcome=outcome,
                )
            self._prewarm_progress(
                service=service,
                status="ready",
                step_key=f"{key}_prewarm_ready",
                message=self._stage_tr("{label} Docker 예열이 완료되었습니다.").format(label=label),
            )

        self._prewarm_jobs[key] = self._ensure_prewarm_executor().submit(runner)

    def _run_runtime_fallback(
        self,
        *,
        service: str,
        fallback: Callable[[], None],
        stale_cleanup: Callable[[], None] | None = None,
    ) -> None:
        started_at = time.perf_counter()
        outcome = "completed"
        arbiter = self._runtime_resource_arbiter()
        token = arbiter.token(service)
        try:
            with arbiter.model_start(
                token,
                cancel_checker=self._prewarm_cancel_checker,
                stale_cleanup=stale_cleanup,
            ):
                self._capture_runtime_gpu_start_baseline(service)
                fallback()
            if self._prewarm_cancel_checker():
                with arbiter.model_release(service):
                    if callable(stale_cleanup):
                        stale_cleanup()
                raise OperationCancelledError(
                    f"{service} startup was cancelled after model load."
                )
        except OperationCancelledError:
            outcome = "cancelled"
            raise
        except Exception:
            outcome = "failed"
            raise
        finally:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            self._record_runtime_performance(
                service=service,
                operation="start",
                elapsed_ms=elapsed_ms,
                outcome=outcome,
            )

    def _await_prewarm_or_run(
        self,
        key: str,
        label: str,
        service: str,
        fallback: Callable[[], None],
        *,
        stale_cleanup: Callable[[], None] | None = None,
    ) -> None:
        self._raise_if_cancelled()
        job = self._prewarm_jobs.pop(key, None)
        if job is None:
            self._run_runtime_fallback(
                service=service,
                fallback=fallback,
                stale_cleanup=stale_cleanup,
            )
            self._raise_if_cancelled()
            return
        wait_started_at = time.perf_counter()
        wait_outcome = "completed"
        prewarm_error: Exception | None = None
        try:
            job.result()
        except OperationCancelledError:
            wait_outcome = "cancelled"
            raise
        except Exception as exc:
            wait_outcome = "failed"
            prewarm_error = exc
        finally:
            self._record_runtime_performance(
                service=service,
                operation="wait",
                elapsed_ms=(time.perf_counter() - wait_started_at) * 1000.0,
                outcome=wait_outcome,
            )
        if prewarm_error is not None:
            self._raise_if_cancelled()
            logger.warning(
                "%s prewarm failed; falling back to synchronous startup: %s",
                label,
                prewarm_error,
            )
            self._prewarm_progress(
                service=service,
                status="running",
                runtime_prewarm_status="failed",
                step_key=f"{key}_prewarm_failed",
                message=self._stage_tr(
                    "{label} 예열 실패. 해당 단계에서 다시 준비합니다."
                ).format(label=label),
                detail=str(prewarm_error),
            )
            self._run_runtime_fallback(
                service=service,
                fallback=fallback,
                stale_cleanup=stale_cleanup,
            )
        self._raise_if_cancelled()

    def _shutdown_prewarm_executor(self) -> None:
        executor = self._prewarm_executor
        self._prewarm_executor = None
        cancel_event = getattr(self, "_prewarm_cancel_event", None)
        if cancel_event is not None:
            cancel_event.set()
        self._runtime_resource_arbiter().cancel_generation()
        jobs = list(self._prewarm_jobs.values())
        # 컨테이너 사전 기동도 같은 실행기에 있다. 취소 대상에서 빠지면 teardown 이
        # 그 작업이 끝날 때까지 기다린다.
        container_job = getattr(self, "_ocr_container_prewarm_job", None)
        if container_job is not None:
            jobs.append(container_job)
            self._ocr_container_prewarm_job = None
        for job in jobs:
            job.cancel()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        self._prewarm_jobs.clear()

    def _shutdown_managed_runtimes(
        self,
        *,
        context: str = "batch cleanup",
        raise_on_failure: bool = False,
        preserve_sleeping_paddle: bool = False,
        ocr_service: str = "ocr",
    ) -> None:
        runtime_managers = (
            (
                "OCR",
                getattr(self.main_page, "local_ocr_runtime_manager", None),
                LocalOCRRuntimeManager,
            ),
            (
                "Gemma",
                getattr(self.main_page, "local_translation_runtime_manager", None),
                LocalGemmaRuntimeManager,
            ),
        )
        failures: list[Exception] = []
        for label, runtime_manager, manager_type in runtime_managers:
            if not isinstance(runtime_manager, manager_type):
                continue
            try:
                self._shutdown_runtime_with_retry(
                    label,
                    runtime_manager,
                    context=context,
                    raise_on_failure=True,
                    release_for_handoff=(
                        preserve_sleeping_paddle and label == "OCR"
                    ),
                    service=(ocr_service if label == "OCR" else "gemma"),
                    allow_foreign_owner_teardown=bool(failures),
                    cancellable=False,
                )
            except Exception as exc:
                failures.append(exc)
        if raise_on_failure and failures:
            raise failures[0]

    def _shutdown_runtime_with_retry(
        self,
        label: str,
        runtime_manager: Any,
        *,
        context: str,
        raise_on_failure: bool,
        release_for_handoff: bool = False,
        service: str = "",
        allow_foreign_owner_teardown: bool = False,
        cancellable: bool = True,
    ) -> None:
        service = str(service or label or "runtime").strip().lower() or "runtime"
        # 실행 도중의 핸드오프는 사용자가 취소하면 멈춰야 한다. 반대로 종료 정리는
        # 취소 신호가 이미 켜진 뒤에 돌기 때문에, 같은 검사기를 쓰면 정리가 스스로를
        # 취소하고 컨테이너와 GPU 를 남긴다.
        cancel_checker = (
            self._prewarm_cancel_checker
            if cancellable
            else self._cleanup_cancel_checker
        )
        self._record_runtime_transition(
            service=service,
            to_state="releasing",
        )
        self._sample_performance_resources(f"{service}_release_start")
        last_error: Exception | None = None
        release_before = query_cuda_handoff_metrics()
        for attempt in range(2):
            started_at = time.perf_counter()
            release = getattr(runtime_manager, "release_for_handoff", None)
            used_handoff_release = (
                release_for_handoff
                and label == "OCR"
                and callable(release)
            )
            target_state = (
                RuntimeModelState.SLEEPING
                if used_handoff_release
                else RuntimeModelState.STOPPED
            )
            try:
                router_active = self._router_runtime_is_active(runtime_manager)
                with self._runtime_resource_arbiter().model_release(
                    service,
                    target_state=target_state,
                    allow_foreign_owner_teardown=allow_foreign_owner_teardown,
                ) as release_context:
                    if used_handoff_release:
                        if router_active:
                            release_report = release(
                                resource_arbiter=self._runtime_resource_arbiter(),
                                runtime_service=service,
                                cancel_checker=cancel_checker,
                            )
                        else:
                            release_report = release()
                        if (
                            isinstance(release_report, dict)
                            and str(release_report.get("runtime_state") or "")
                            == "stopped"
                        ):
                            release_context.target_state = (
                                RuntimeModelState.STOPPED
                            )
                    else:
                        if router_active:
                            release_report = runtime_manager.shutdown(
                                resource_arbiter=self._runtime_resource_arbiter(),
                                runtime_service=service,
                                cancel_checker=cancel_checker,
                                allow_foreign_owner_teardown=(
                                    allow_foreign_owner_teardown
                                ),
                            )
                        else:
                            release_report = runtime_manager.shutdown()
                    gate = self._verify_managed_runtime_gpu_release(
                        service,
                        release_report,
                        before=release_before,
                    )
                    self._handle_unconfirmed_gpu_release(service, gate)
                elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                completed_state = release_context.target_state.value
                self._record_runtime_performance(
                    service=service,
                    operation="release",
                    elapsed_ms=elapsed_ms,
                )
                self._record_runtime_transition(
                    service=service,
                    to_state=completed_state,
                    elapsed_ms=elapsed_ms,
                )
                self._sample_performance_resources(
                    f"{service}_release_end"
                )
                return
            except Exception as exc:
                self._record_runtime_performance(
                    service=service,
                    operation="release",
                    elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
                    outcome="failed",
                )
                last_error = exc
                logger.warning(
                    "Failed to stop managed %s runtime during %s%s.",
                    label,
                    context,
                    "; retrying once" if attempt == 0 else "",
                    exc_info=True,
                )
        self._record_runtime_transition(
            service=service,
            to_state="release_failed",
            outcome="failed",
        )
        if raise_on_failure and last_error is not None:
            raise last_error

    def _start_ocr_prewarm(
        self,
        policy: dict[str, Any],
        *,
        cache_miss_confirmed: bool = False,
    ) -> None:
        runtime_manager = getattr(self.main_page, "local_ocr_runtime_manager", None)
        if not isinstance(runtime_manager, LocalOCRRuntimeManager):
            return
        engine_key = str(policy["primary_ocr_engine"])
        # A non-empty project cache may contain a page-local OCR hit that can be
        # known only after detection restores the exact ordered blocks. Defer in
        # that case to preserve the all-hit zero-runtime contract. A brand-new,
        # empty sidecar cannot contain a hit, so keep the cold-path overlap.
        project_store = getattr(self, "_project_checkpoint_store", None)
        if (
            not cache_miss_confirmed
            and engine_key in {"PaddleOCR VL", "PaddleOCR VL Spotting"}
            and project_store is not None
        ):
            try:
                page_keys = list(
                    getattr(self, "_project_checkpoint_page_keys", []) or []
                )
                if page_keys and all(
                    project_store.has_stage_record(page_key, "ocr")
                    or project_store.has_stage_record(
                        page_key,
                        "detection",
                    )
                    for page_key in page_keys
                ):
                    return
                if not page_keys and project_store.has_stage_records("ocr"):
                    return
            except Exception:
                logger.warning(
                    "Project checkpoint prewarm probe failed open; OCR startup "
                    "will be resolved after cache lookup.",
                    exc_info=True,
                )
                return
        settings_page = self.main_page.settings_page
        if (
            not cache_miss_confirmed
            and engine_key == "PaddleOCR VL"
            and bool(
                settings_page.get_paddleocr_vl_settings().get(
                    "persistent_cache_enabled",
                    True,
                )
            )
        ):
            store = self._get_paddleocr_cache_store()
            stats = store.stats()
            if bool(stats.get("enabled", False)) and int(
                stats.get("item_count", 0) or 0
            ) > 0:
                return
        service = self._ocr_runtime_service_name(engine_key)
        self._start_prewarm(
            "ocr",
            "OCR",
            service,
            lambda: runtime_manager.ensure_engine(
                engine_key,
                settings_page,
                progress_callback=self._runtime_progress_callback(),
                cancel_checker=self._prewarm_cancel_checker,
                resource_arbiter=self._runtime_resource_arbiter(),
                runtime_service=service,
            ),
            stale_cleanup=lambda: self._managed_runtime_stale_cleanup(
                service,
                runtime_manager,
            ),
        )

    def _plan_detected_page_ocr_prewarm(
        self,
        ctx: StagePageContext,
        policy: dict[str, Any],
        *,
        index: int,
        total_images: int,
    ) -> None:
        """Start Paddle only after a detected page proves a cache miss.

        Unrelated rows in the persistent OCR database do not prove that the
        current folder is cached. Planning each page as detection completes
        preserves all-hit zero-runtime behavior while overlapping the first
        real miss with detection of the remaining pages.
        """

        if (
            ctx.failed_stage
            or ctx.no_text_detected
            or not ctx.blk_list
            or "ocr" in getattr(self, "_prewarm_jobs", {})
        ):
            return
        engine_key = str(policy.get("primary_ocr_engine", ""))
        if engine_key not in {"PaddleOCR VL", "PaddleOCR VL Spotting"}:
            return
        runtime_manager = getattr(
            self.main_page,
            "local_ocr_runtime_manager",
            None,
        )
        if not isinstance(runtime_manager, LocalOCRRuntimeManager):
            return

        settings_page = self.main_page.settings_page
        paddle_settings = (
            settings_page.get_paddleocr_vl_settings()
            if engine_key == "PaddleOCR VL"
            else settings_page.get_paddleocr_vl_spotting_settings()
        )
        persistent_cache_requested = bool(
            engine_key == "PaddleOCR VL"
            and paddle_settings.get("persistent_cache_enabled", True)
        )
        project_cache_requested = (
            getattr(self, "_project_checkpoint_store", None) is not None
        )
        if not persistent_cache_requested and not project_cache_requested:
            self._start_ocr_prewarm(
                policy,
                cache_miss_confirmed=True,
            )
            return

        self._canonicalize_ocr_inputs([ctx])
        runtime_identity = dict(
            getattr(self, "_paddleocr_cache_identity", None) or {}
        ) or None
        if runtime_identity is None:
            runtime_identity = runtime_manager.get_ocr_cache_identity(
                engine_key,
                settings_page,
            )
        if runtime_identity is None:
            self._emit_benchmark_event(
                "ocr_prewarm_decision",
                image_path=ctx.image_path,
                image_index=index,
                total_images=total_images,
                decision="start",
                reason="runtime_identity_unavailable",
            )
            self._start_ocr_prewarm(
                policy,
                cache_miss_confirmed=True,
            )
            return
        self._paddleocr_cache_identity = dict(runtime_identity)

        if project_cache_requested:
            self._prepare_project_ocr_hits(
                [ctx],
                policy,
                runtime_identity,
            )
            if ctx.project_ocr_hit is not None:
                self._emit_benchmark_event(
                    "ocr_prewarm_decision",
                    image_path=ctx.image_path,
                    image_index=index,
                    total_images=total_images,
                    decision="defer",
                    reason="project_checkpoint_hit",
                )
                return

        requires_runtime = True
        if persistent_cache_requested:
            requires_runtime = self._prepare_paddleocr_cache_plans(
                [ctx],
                policy,
                runtime_identity,
            )
        if ctx.failed_stage:
            self._emit_benchmark_event(
                "ocr_prewarm_decision",
                image_path=ctx.image_path,
                image_index=index,
                total_images=total_images,
                decision="defer",
                reason="cache_plan_failed",
            )
            return
        self._emit_benchmark_event(
            "ocr_prewarm_decision",
            image_path=ctx.image_path,
            image_index=index,
            total_images=total_images,
            decision="start" if requires_runtime else "defer",
            reason=(
                "persistent_cache_miss"
                if requires_runtime and persistent_cache_requested
                else (
                    "project_checkpoint_miss"
                    if requires_runtime
                    else "persistent_cache_hit"
                )
            ),
        )
        if requires_runtime:
            self._start_ocr_prewarm(
                policy,
                cache_miss_confirmed=True,
            )

    def _await_ocr_runtime(self, policy: dict[str, Any]) -> None:
        runtime_manager = getattr(self.main_page, "local_ocr_runtime_manager", None)
        if not isinstance(runtime_manager, LocalOCRRuntimeManager):
            return
        settings_page = self.main_page.settings_page
        engine_key = str(policy["primary_ocr_engine"])
        service = self._ocr_runtime_service_name(engine_key)
        self._await_prewarm_or_run(
            "ocr",
            "OCR",
            service,
            lambda: runtime_manager.ensure_engine(
                engine_key,
                settings_page,
                progress_callback=self._runtime_progress_callback(),
                cancel_checker=self._prewarm_cancel_checker,
                resource_arbiter=self._runtime_resource_arbiter(),
                runtime_service=service,
            ),
            stale_cleanup=lambda: self._managed_runtime_stale_cleanup(
                service,
                runtime_manager,
            ),
        )
        self._sample_performance_resources(f"{service}_model_ready")

    @staticmethod
    def _ocr_runtime_service_name(engine_key: str) -> str:
        return {
            "PaddleOCR VL": "paddleocr_vl",
            "PaddleOCR VL Spotting": "paddleocr_vl_spotting",
            "HunyuanOCR": "hunyuanocr",
            "MangaLMM": "mangalmm",
        }.get(str(engine_key), str(engine_key).lower().replace(" ", "_"))

    def _start_ocr_container_prewarm(self, policy: dict[str, Any]) -> None:
        """OCR Router 컨테이너만 검출 sweep 과 겹쳐 미리 띄운다.

        모델 적재는 지금처럼 검출이 끝난 뒤에 한다. PR #242 가 예열을 검출 뒤로 미룬
        이유는 검출기의 ONNX 세션이 **모델 적재** baseline 에 섞이는 것이었는데,
        Router v2 는 `--no-models-autoload` 로 떠서 컨테이너 기동만으로는 어떤 모델도
        올리지 않는다. 그래서 이 부분만 앞으로 당길 수 있다.
        """

        if getattr(self, "_ocr_container_prewarm_job", None) is not None:
            return
        runtime_manager = getattr(self.main_page, "local_ocr_runtime_manager", None)
        if not isinstance(runtime_manager, LocalOCRRuntimeManager):
            return
        prepare = getattr(runtime_manager, "prepare_engine_container", None)
        if not callable(prepare):
            return
        engine_key = str(policy.get("primary_ocr_engine", "") or "")
        if not engine_key:
            return
        settings_page = self.main_page.settings_page
        service = self._ocr_runtime_service_name(engine_key)

        def runner() -> bool:
            if self._prewarm_cancel_checker():
                return False
            try:
                return bool(
                    prepare(
                        engine_key,
                        settings_page,
                        resource_arbiter=self._runtime_resource_arbiter(),
                        runtime_service=service,
                        cancel_checker=self._prewarm_cancel_checker,
                    )
                )
            except OperationCancelledError:
                return False
            except Exception:
                # 컨테이너 사전 기동은 최적화일 뿐이다. 정식 경로가 처리한다.
                logger.info(
                    "OCR Router 컨테이너 사전 기동이 예외로 끝났습니다. 계속 진행합니다.",
                    exc_info=True,
                )
                return False

        self._ocr_container_prewarm_job = self._ensure_prewarm_executor().submit(runner)

    def _await_ocr_container_prewarm(self) -> None:
        """컨테이너 기동이 끝난 뒤에 모델 적재로 넘어간다.

        같은 코디네이터를 두 스레드가 동시에 만지지 않게 한다. 여기까지 오면 검출
        sweep 이 이미 지났으므로 대개 즉시 반환한다.
        """

        job = getattr(self, "_ocr_container_prewarm_job", None)
        if job is None:
            return
        self._ocr_container_prewarm_job = None
        try:
            prepared = bool(job.result())
        except Exception:
            logger.info(
                "OCR Router 컨테이너 사전 기동 결과를 읽지 못했습니다.",
                exc_info=True,
            )
            return
        if prepared:
            self._emit_benchmark_event("ocr_container_prewarm", prepared=True)

    def _start_gemma_page_cache_prefetch(self) -> None:
        """Gemma GGUF 를 호스트 페이지 캐시로 미리 끌어올린다.

        OCR sweep(실측 71초)과 겹쳐 돌린다. 디스크에서 RAM 으로만 옮기므로 OCR 이 쥔
        VRAM 과 다투지 않는다. 실측으로 캐시 미적중 읽기가 8초이고, 이후 Gemma 적재가
        44초에서 재실행 수준(약 5초)으로 내려간다.
        """

        if getattr(self, "_gemma_prefetch_job", None) is not None:
            return
        runtime_manager = getattr(
            self.main_page,
            "local_translation_runtime_manager",
            None,
        )
        if not isinstance(runtime_manager, LocalGemmaRuntimeManager):
            return
        settings_page = self.main_page.settings_page

        def runner() -> dict[str, Any]:
            if self._prewarm_cancel_checker():
                return {"performed": False, "reason": "cancelled"}
            try:
                return runtime_manager.prefetch_model_into_page_cache(
                    settings_page,
                    cancel_checker=self._prewarm_cancel_checker,
                )
            except Exception:
                # 프리페치는 최적화일 뿐이다. 실패가 배치를 멈춰서는 안 된다.
                logger.info(
                    "Gemma 페이지 캐시 프리페치가 예외로 끝났습니다. 계속 진행합니다.",
                    exc_info=True,
                )
                return {"performed": False, "reason": "exception"}

        if getattr(self, "_page_cache_executor", None) is None:
            self._page_cache_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="ct-page-cache",
            )
        self._gemma_prefetch_job = self._page_cache_executor.submit(runner)

    def _await_gemma_page_cache_prefetch(self) -> None:
        """프리페치가 끝난 뒤에 Gemma 적재를 시작한다.

        같은 파일을 프리페치와 mmap 이 동시에 읽으면 디스크를 두 배로 두드린다.
        여기까지 오면 OCR sweep 이 이미 지났으므로 대개 즉시 반환한다.
        """

        job = getattr(self, "_gemma_prefetch_job", None)
        if job is None:
            return
        self._gemma_prefetch_job = None
        try:
            result = job.result()
        except Exception:
            logger.info("Gemma 페이지 캐시 프리페치 결과를 읽지 못했습니다.", exc_info=True)
            return
        if isinstance(result, dict) and result.get("performed"):
            self._emit_benchmark_event(
                "gemma_page_cache_prefetch",
                prefetch_elapsed_sec=float(result.get("elapsed_sec", 0.0) or 0.0),
                prefetch_model_bytes=int(result.get("model_bytes", 0) or 0),
            )

    def _shutdown_page_cache_executor(self) -> None:
        # 정리는 어떤 상황에서도 실패하면 안 된다. 생성자를 거치지 않은 객체에서도
        # 안전하도록 속성 존재를 가정하지 않는다.
        executor = getattr(self, "_page_cache_executor", None)
        self._page_cache_executor = None
        self._gemma_prefetch_job = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _render_worker_count() -> int:
        """렌더 워커 수. 기본 1이며 ``CT_RENDER_WORKERS`` 로만 바꾼다.

        렌더를 융합한 근거는 "렌더 약 300ms 가 인페인팅 1.28s 뒤에 숨는다" 였다.
        실측 366장 실행에서 렌더는 페이지당 **2.31초**였고, 워커가 하나뿐이라
        14.1분의 직렬 작업이 됐다. 즉 그 전제는 더 이상 성립하지 않는다.

        그렇다고 기본값을 올리지는 않는다. 렌더가 Qt 스레드를 벗어나면
        `scene.render()` 가 **예외도 경고도 없이** 빈 이미지를 만든다
        (`pipeline/render_pool.py` 참고). 워커를 늘렸을 때 같은 종류의 조용한
        실패가 없다는 것은 렌더 결과 픽셀로 확인해야 하며, 그 확인 전에는
        기본값을 바꾸지 않는다. 이 환경변수는 그 A/B 를 할 수 있게 열어둔 것이다.
        """

        raw = str(os.environ.get("CT_RENDER_WORKERS", "") or "").strip()
        if not raw:
            return 1
        try:
            value = int(raw)
        except ValueError:
            logger.warning("Ignoring a non-numeric CT_RENDER_WORKERS: %r", raw)
            return 1
        if value < 1 or value > 8:
            logger.warning("Ignoring an out-of-range CT_RENDER_WORKERS: %d", value)
            return 1
        return value

    def _ensure_render_executor(self) -> QtRenderPool:
        if self._render_executor is None:
            # plain Python 스레드가 아니라 Qt 스레드여야 한다. 이유는
            # `pipeline/render_pool.py` 의 모듈 주석 참고 — 디스패처 없는
            # 스레드에서는 `scene.render()` 가 조용히 빈 이미지를 만든다.
            workers = self._render_worker_count()
            if workers > 1:
                logger.info("Running the render sweep with %d workers.", workers)
            self._render_executor = QtRenderPool(max_workers=workers)
        return self._render_executor

    def _shutdown_render_executor(self) -> None:
        # 정리는 어떤 상황에서도 실패하면 안 된다. 생성자를 거치지 않은 객체에서도
        # 안전하도록 속성 존재를 가정하지 않는다.
        try:
            cancel_event = getattr(self, "_render_cancel_event", None)
            if cancel_event is not None:
                cancel_event.set()
            executor = getattr(self, "_render_executor", None)
            self._render_executor = None
            self._pending_render_jobs = []
            if executor is not None:
                executor.shutdown(wait=True)
        except Exception:
            logger.warning(
                "렌더 워커 실행기 정리 중 예외가 발생했지만 무시합니다.",
                exc_info=True,
            )

    def _warm_render_font_caches(self, render_settings: Any) -> None:
        """렌더 워커가 폰트 캐시(`modules/rendering/render.py`의 lru_cache 4개)를
        처음 만지는 순간이 파이프라인 스레드와 겹치지 않도록, sweep 시작 전에
        파이프라인 스레드에서 1회 워밍한다. 정확한 캐시 키 적중이 목적이 아니라
        Qt 폰트 서브시스템(QFontDatabase/QRawFont)의 최초 접근 경쟁 제거가
        목적이다."""
        try:
            _register_render_fallback_system_fonts()
            resolve_render_symbol_fallback_font_family()
            resolve_render_glyph_fallback_font_family(tuple(sorted(RENDER_NORMALIZABLE_GLYPHS)))
            font_family = str(getattr(render_settings, "font_family", "") or "")
            if font_family:
                _render_font_has_real_glyph(font_family, "A")
        except Exception:
            logger.debug("렌더 폰트 캐시 워밍이 예외로 끝났습니다. 계속 진행합니다.", exc_info=True)

    def _reserve_render_output_path(
        self,
        ctx: StagePageContext,
        *,
        export_settings: dict[str, Any],
        page_index: int,
        total_pages: int,
    ) -> tuple[str, str, str]:
        """`_write_final_render_export`의 경로 계산·예약을 파이프라인 스레드에서
        미리 끝낸다. 개별 이미지 모드의 `reserve_unique_path` 존재 검사(TOCTOU
        지점)가 여기서 끝나므로, 렌더 워커는 반환된 정확한 경로에 값만 쓴다."""
        page_base_name = os.path.splitext(os.path.basename(ctx.image_path))[0]
        series_dir = self.main_page.get_reserved_automatic_output_series_dir(
            ctx.directory,
            anchor_path=(
                self.main_page.image_files[0] if self.main_page.image_files else ctx.image_path
            ),
        )
        if is_single_archive_mode(export_settings):
            staging_dir = build_archive_staging_dir(series_dir, ctx.export_token)
            os.makedirs(staging_dir, exist_ok=True)
            output_format = str(
                export_settings.get(
                    "resolved_automatic_output_archive_image_format",
                    DEFAULT_OUTPUT_ARCHIVE_IMAGE_FORMAT,
                )
            )
            output_path = os.path.join(
                staging_dir,
                build_archive_page_file_name(page_index, total_pages, page_base_name, output_format),
            )
            return output_path, series_dir, output_format
        os.makedirs(series_dir, exist_ok=True)
        candidate = os.path.join(
            series_dir,
            build_output_file_name(page_base_name, "translated", ctx.image_path, export_settings),
        )
        output_path = reserve_unique_path(candidate)
        requested = str(
            export_settings.get("resolved_automatic_output_image_format", DEFAULT_OUTPUT_IMAGE_FORMAT)
        )
        output_format = resolve_individual_output_format(ctx.image_path, requested)
        return output_path, series_dir, output_format

    def _start_gemma_prewarm(self) -> None:
        runtime_manager = getattr(self.main_page, "local_translation_runtime_manager", None)
        if not isinstance(runtime_manager, LocalGemmaRuntimeManager):
            return
        settings_page = self.main_page.settings_page
        self._start_prewarm(
            "gemma",
            "Gemma",
            "gemma",
            lambda: runtime_manager.ensure_server(
                settings_page,
                progress_callback=self._runtime_progress_callback(),
                cancel_checker=self._prewarm_cancel_checker,
                resource_arbiter=self._runtime_resource_arbiter(),
                runtime_service="gemma",
            ),
            stale_cleanup=lambda: self._managed_runtime_stale_cleanup(
                "gemma",
                runtime_manager,
            ),
        )

    def _await_gemma_runtime(self) -> None:
        runtime_manager = getattr(self.main_page, "local_translation_runtime_manager", None)
        if not isinstance(runtime_manager, LocalGemmaRuntimeManager):
            return
        settings_page = self.main_page.settings_page
        self._await_prewarm_or_run(
            "gemma",
            "Gemma",
            "gemma",
            lambda: runtime_manager.ensure_server(
                settings_page,
                progress_callback=self._runtime_progress_callback(),
                cancel_checker=self._prewarm_cancel_checker,
                resource_arbiter=self._runtime_resource_arbiter(),
                runtime_service="gemma",
            ),
            stale_cleanup=lambda: self._managed_runtime_stale_cleanup(
                "gemma",
                runtime_manager,
            ),
        )
        self._sample_performance_resources("gemma_model_ready")

    def _set_current_image(self, image_path: str) -> None:
        try:
            self.main_page.curr_img_idx = self.main_page.image_files.index(image_path)
        except ValueError:
            pass

    def _source_lang_english(self, source_lang: str) -> str:
        return self.main_page.lang_mapping.get(source_lang, source_lang)

    def _load_page_contexts(self, image_list: list[str]) -> list[StagePageContext]:
        pages: list[StagePageContext] = []
        self._timestamp = build_export_timestamp()
        self._export_run_tokens = {}
        for image_path in image_list:
            state = self._ensure_page_state(image_path)
            existing_blocks = list(state.get("blk_list", []) or [])
            project_translation_snapshot = snapshot_project_translations(
                existing_blocks
            )
            try:
                project_render_blocks = snapshot_project_render_blocks(
                    existing_blocks
                )
                project_viewer_state = copy.deepcopy(
                    dict(state.get("viewer_state", {}) or {})
                )
            except Exception:
                logger.warning(
                    "Unable to snapshot the existing viewer state for %s; "
                    "render checkpoint lookup is disabled for this page.",
                    os.path.basename(image_path),
                    exc_info=True,
                )
                project_render_blocks = []
                project_viewer_state = {}
            source_lang = str(state.get("source_lang", self.main_page.s_combo.currentText()))
            target_lang = str(state.get("target_lang", self.main_page.t_combo.currentText()))
            directory, archive_bname = resolve_export_directory(
                image_path,
                archive_info=self.main_page.file_handler.archive_info,
                source_records=getattr(self.main_page, "export_source_by_path", {}),
                project_file=getattr(self.main_page, "project_file", None),
                temp_dir=getattr(self.main_page, "temp_dir", None),
            )
            export_token = self._resolve_export_token(directory, self._timestamp, archive_bname)
            export_root = export_run_root(directory, export_token, archive_bname)
            pages.append(
                StagePageContext(
                    image_path=image_path,
                    image_name=os.path.basename(image_path),
                    source_lang=source_lang,
                    target_lang=target_lang,
                    directory=directory,
                    archive_bname=archive_bname,
                    export_token=export_token,
                    export_root=export_root,
                    project_translation_snapshot=project_translation_snapshot,
                    project_render_blocks=project_render_blocks,
                    project_viewer_state=project_viewer_state,
                )
            )
        return pages

    def _ensure_stage_policy(self, pages: list[StagePageContext]) -> dict[str, Any]:
        if not pages:
            raise RuntimeError("No pages selected for stage-batched processing.")
        source_lang_english = self._source_lang_english(pages[0].source_lang)
        policy = resolve_stage_batched_ocr_policy(
            STAGE_BATCHED_WORKFLOW_MODE,
            self.main_page.settings_page.get_tool_selection("ocr"),
            source_lang_english,
            self.main_page.settings_page.get_tool_selection("translator"),
        )
        if not policy.stage_batched_supported or policy.requires_sidecar_collection:
            reason = policy.unsupported_reason or "selector_or_sidecar_route_is_not_promoted"
            raise RuntimeError(
                f"Stage-Batched Pipeline is not supported for this OCR/translator combination: {reason}"
            )
        for ctx in pages[1:]:
            if self._source_lang_english(ctx.source_lang) != source_lang_english:
                raise RuntimeError("Stage-Batched Pipeline currently requires a single shared source language.")
            if ctx.target_lang != pages[0].target_lang:
                raise RuntimeError("Stage-Batched Pipeline currently requires a single shared target language.")
        return policy.to_dict()

    def _mark_page_failed(
        self,
        ctx: StagePageContext,
        *,
        index: int,
        total_images: int,
        stage: str,
        reason: str,
        detail: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        ctx.failed_stage = stage
        ctx.failed_reason = reason
        self.main_page.image_ctrl.update_processing_summary(
            ctx.image_path,
            {"last_failure_reason": reason},
        )
        self.main_page.image_ctrl.mark_processing_stage(ctx.image_path, stage, "failed", reason=reason)
        self._emit_benchmark_event(
            "page_failed",
            image_path=ctx.image_path,
            image_index=index,
            total_images=total_images,
            failed_stage=stage,
            reason=reason,
            **(extra or {}),
        )
        self.main_page.image_skipped.emit(ctx.image_path, stage, detail or reason)
        # 실패한 페이지는 이후 스테이지가 모두 건너뛴다. 픽셀 데이터를 계속 붙들고
        # 있으면 한 페이지의 실패가 남은 배치 전체의 메모리를 갉아먹는다.
        self._release_page_buffers(ctx)

    def _detect_all(
        self,
        pages: list[StagePageContext],
        policy: dict[str, Any] | None = None,
    ) -> None:
        total_images = len(pages)
        settings_page = self.main_page.settings_page
        detector = self.block_detection.block_detector_cache
        checkpoint_store = getattr(self, "_project_checkpoint_store", None)

        for index, ctx in enumerate(pages):
            self._raise_if_cancelled()
            self._set_current_image(ctx.image_path)
            self.emit_progress(index, total_images, 0, 10, True, stage_name='detect-all')
            self._start_page_summary(ctx.image_path, ctx.source_lang, ctx.target_lang)
            self._log_page_start(index, total_images, ctx.image_path)
            self.main_page.image_ctrl.update_processing_summary(
                ctx.image_path,
                {"export_root": ctx.export_root},
            )
            self._emit_benchmark_event(
                "page_start",
                image_path=ctx.image_path,
                image_index=index,
                total_images=total_images,
                source_lang=ctx.source_lang,
                target_lang=ctx.target_lang,
            )
            self._emit_benchmark_event(
                "detect_start",
                image_path=ctx.image_path,
                image_index=index,
                total_images=total_images,
            )

            with self._measure_performance(
                stage="detect",
                operation="image_decode",
            ):
                ctx.image = self.main_page.image_ctrl.load_image(ctx.image_path)
                if ctx.image is None:
                    ensure_path_materialized(ctx.image_path)
                    ctx.image = imk.read_image(ctx.image_path)

            source_lang_english = self._source_lang_english(ctx.source_lang)
            detection_hit = None
            detection_identity = None
            if checkpoint_store is not None:
                with self._measure_performance(
                    stage="detect",
                    operation="decoded_hash",
                    workload={
                        "page_pixel_count": int(ctx.image.shape[0])
                        * int(ctx.image.shape[1]),
                    },
                ):
                    ctx.source_decoded_sha256 = decoded_image_sha256(ctx.image)
                ctx.project_checkpoint_page_key = project_checkpoint_page_key(
                    self.main_page,
                    ctx.image_path,
                )
                try:
                    detection_identity = build_detection_identity(
                        settings_page,
                        source_lang_english=source_lang_english,
                    )
                    if detection_identity is not None:
                        ctx.detection_fingerprint = build_detection_fingerprint(
                            source_sha256=ctx.source_decoded_sha256,
                            identity=detection_identity,
                        )
                        detection_hit = lookup_detection_checkpoint(
                            checkpoint_store,
                            page_key=ctx.project_checkpoint_page_key,
                            fingerprint=ctx.detection_fingerprint,
                            source_sha256=ctx.source_decoded_sha256,
                            identity=detection_identity,
                        )
                except Exception:
                    logger.warning(
                        "Detection checkpoint lookup failed open for %s.",
                        ctx.image_name,
                        exc_info=True,
                    )
                    detection_identity = None

            if detection_hit is not None:
                ctx.blk_list = detection_hit.blocks
                ctx.precomputed_mask_details = (
                    detection_hit.precomputed_mask_details
                )
                ctx.detector_key = str(detection_identity.get("detector", ""))
                ctx.detector_engine = str(
                    detection_identity.get("engine", "")
                )
                ctx.detector_device = str(
                    detection_identity.get("device", "")
                )
                ctx.detection_checkpoint_status = "hit"
            else:
                if detector is None:
                    detector = TextBlockDetector(settings_page)
                    self.block_detection.block_detector_cache = detector
                with self._measure_performance(
                    stage="detect",
                    operation="model_inference",
                    workload={
                        "page_pixel_count": int(ctx.image.shape[0])
                        * int(ctx.image.shape[1]),
                    },
                    service="detector",
                ):
                    blk_list = detector.detect(ctx.image)
                self._raise_if_cancelled()
                ctx.precomputed_mask_details = detector.last_mask_details
                ctx.detector_key = (
                    detector.detector
                    or settings_page.get_tool_selection("detector")
                    or "RT-DETR-v2"
                )
                ctx.detector_engine = detector.last_engine_name or ""
                ctx.detector_device = (
                    detector.last_device
                    or resolve_device(
                        settings_page.is_gpu_enabled(),
                        backend="onnx",
                    )
                )
                rtl = source_lang_english == "Japanese"
                if blk_list:
                    get_best_render_area(blk_list, ctx.image)
                    ctx.blk_list = sort_blk_list(blk_list, rtl)
                else:
                    ctx.blk_list = []
                ctx.detection_checkpoint_status = (
                    "miss"
                    if checkpoint_store is not None
                    and detection_identity is not None
                    else "disabled"
                )
                if checkpoint_store is not None and detection_identity is not None:
                    try:
                        record_detection_checkpoint(
                            checkpoint_store,
                            page_key=ctx.project_checkpoint_page_key,
                            fingerprint=ctx.detection_fingerprint,
                            source_sha256=ctx.source_decoded_sha256,
                            identity=detection_identity,
                            blocks=ctx.blk_list,
                            precomputed_mask_details=ctx.precomputed_mask_details,
                        )
                    except Exception:
                        logger.warning(
                            "Detection checkpoint write failed open for %s.",
                            ctx.image_name,
                            exc_info=True,
                        )

            if checkpoint_store is not None:
                ctx.detection_structure_signature = (
                    detection_structure_signature(ctx.blk_list)
                )

            if ctx.blk_list:
                self._persist_detect_state(
                    ctx.image_path,
                    ctx.blk_list,
                    ctx.detector_key,
                    ctx.detector_engine,
                    ctx.image,
                )
                self._emit_benchmark_event(
                    "detect_end",
                    image_path=ctx.image_path,
                    image_index=index,
                    total_images=total_images,
                    block_count=len(ctx.blk_list or []),
                    detector_key=ctx.detector_key,
                    detector_engine=ctx.detector_engine,
                    cache_status=ctx.detection_checkpoint_status,
                    project_checkpoint_status=ctx.detection_checkpoint_status,
                )
                export_settings = self._effective_export_settings(settings_page)
                detector_overlay_path = self._write_detector_overlay_debug_image(
                    export_root=ctx.export_root,
                    archive_bname=ctx.archive_bname,
                    image_path=ctx.image_path,
                    image=ctx.image,
                    blk_list=ctx.blk_list,
                    export_settings=export_settings,
                )
                self._maybe_emit_preview_image(
                    index=index,
                    total=total_images,
                    image_path=ctx.image_path,
                    stage_key="detector_overlay",
                    stage_label="텍스트 감지",
                    export_settings=export_settings,
                    preferred_path=detector_overlay_path,
                )
                continue

            state = self._ensure_page_state(ctx.image_path)
            state["blk_list"] = []
            state.setdefault("viewer_state", {})["rectangles"] = []
            ctx.no_text_detected = True
            self.main_page.image_ctrl.mark_processing_stage(
                ctx.image_path,
                "detect",
                "completed",
                reason="no_text_detected",
                block_count=0,
            )
            self._write_inpaint_debug_exports(
                export_root=ctx.export_root,
                archive_bname=ctx.archive_bname,
                image_path=ctx.image_path,
                image=ctx.image,
                blk_list=[],
                export_settings=self._effective_export_settings(settings_page),
                raw_mask=None,
                final_mask=None,
                detector_key=ctx.detector_key,
                detector_engine=ctx.detector_engine,
                detector_device=ctx.detector_device,
                inpainter_key=settings_page.get_tool_selection("inpainter"),
                hd_strategy=settings_page.get_hd_strategy_settings().get("strategy", "Resize"),
                cleanup_stats={"applied": False, "component_count": 0, "block_count": 0},
                mask_details={
                    "mask_refiner": settings_page.get_mask_refiner_settings().get("mask_refiner", "ctd"),
                    "mask_inpaint_mode": settings_page.get_mask_refiner_settings().get("mask_inpaint_mode", ""),
                },
                inpainter_backend=get_inpainter_runtime(settings_page)["backend"],
            )
            ctx.page_ocr_metrics = self._ocr_quality_metrics(None)
            self._emit_benchmark_event(
                "detect_end",
                image_path=ctx.image_path,
                image_index=index,
                total_images=total_images,
                block_count=0,
                detector_key=ctx.detector_key,
                detector_engine=ctx.detector_engine,
                cache_status=ctx.detection_checkpoint_status,
                project_checkpoint_status=ctx.detection_checkpoint_status,
                skip_reason="no_text_detected",
                **ctx.page_ocr_metrics,
            )
            self._raise_if_cancelled()

    def _run_primary_ocr(self, ctx: StagePageContext, policy: dict[str, Any]) -> dict[str, Any]:
        settings_page = self.main_page.settings_page
        source_lang_english = self._source_lang_english(ctx.source_lang)
        source_lang_code = language_codes.get(source_lang_english, "en")
        for blk in ctx.blk_list:
            blk.source_lang = source_lang_code
        device = resolve_device(settings_page.is_gpu_enabled())
        engine_key = str(policy["primary_ocr_engine"])
        cache_key = (
            self.cache_manager._get_ocr_cache_key(
                ctx.image,
                ctx.source_lang,
                engine_key,
                device,
            )
            if engine_key
            not in {"PaddleOCR VL", "PaddleOCR VL Spotting"}
            else None
        )
        cache_status = "miss"
        attempt_count = 0
        page_profile: dict[str, Any] = {}
        engine_name = engine_key
        records = []
        raw_results: dict[str, dict[str, Any]] = {}
        project_checkpoint_hit = ctx.project_ocr_hit

        def snapshot_raw_results() -> None:
            nonlocal raw_results
            raw_results = {
                ensure_text_block_id(block): snapshot_raw_ocr_result(block)
                for block in ctx.blk_list
            }

        def attach_cancel_checker(engine) -> None:
            setter = getattr(engine, "set_cancel_checker", None)
            if callable(setter):
                setter(getattr(self.main_page, "is_current_task_cancelled", None))

        paddle_plan = ctx.paddleocr_cache_plan
        paddle_engine = ctx.paddleocr_cache_engine
        if project_checkpoint_hit is not None:
            raw_results = copy.deepcopy(project_checkpoint_hit.raw_results)
            page_profile = dict(project_checkpoint_hit.page_profile or {})
            page_profile["project_checkpoint"] = {
                "status": "hit",
                "inference_count": 0,
                "http_request_count": 0,
            }
            apply_ocr_result_dictionary(
                ctx.blk_list,
                settings_page.get_ocr_result_dictionary_rules(),
            )
            attempt_count = max(1, int(project_checkpoint_hit.attempt_count))
            engine_name = (
                project_checkpoint_hit.engine_name
                or "PaddleOCRVLEngine"
            )
            cache_status = "hit"
        elif (
            isinstance(paddle_engine, PaddleOCRVLEngine)
            and paddle_plan is not None
        ):
            attach_cancel_checker(paddle_engine)
            paddle_engine.process_persistent_cache_plan(paddle_plan)
            records = paddle_engine.build_persistent_cache_records(paddle_plan)
            page_profile = dict(paddle_engine.last_page_profile or {})
            snapshot_raw_results()
            apply_ocr_result_dictionary(
                ctx.blk_list,
                settings_page.get_ocr_result_dictionary_rules(),
            )
            attempt_count = 1
            engine_name = paddle_engine.__class__.__name__
            if paddle_plan.lookup_disabled:
                cache_status = "disabled"
            elif paddle_plan.all_hit:
                cache_status = "hit"
            elif paddle_plan.hit_count:
                cache_status = "partial"
            else:
                cache_status = "refreshed"
        elif (
            engine_key
            not in {"PaddleOCR VL", "PaddleOCR VL Spotting"}
            and self.cache_manager._can_serve_all_blocks_from_ocr_cache(
                cache_key,
                ctx.blk_list,
            )
        ):
            self.cache_manager._apply_cached_ocr_to_blocks(cache_key, ctx.blk_list)
            snapshot_raw_results()
            apply_ocr_result_dictionary(ctx.blk_list, settings_page.get_ocr_result_dictionary_rules())
            cache_status = "hit"
            attempt_count = 1
        else:
            engine = OCRFactory.create_engine(
                settings_page,
                source_lang_english,
                engine_key,
                selected_ocr_mode=policy["normalized_ocr_mode"],
            )
            attach_cancel_checker(engine)
            engine.process_image(ctx.image, ctx.blk_list)
            page_profile = dict(getattr(engine, "last_page_profile", {}) or {})
            snapshot_raw_results()
            apply_ocr_result_dictionary(ctx.blk_list, settings_page.get_ocr_result_dictionary_rules())
            if engine_key not in {
                "PaddleOCR VL",
                "PaddleOCR VL Spotting",
            }:
                self.cache_manager._cache_ocr_results(cache_key, ctx.blk_list)
            cache_status = "refreshed"
            attempt_count = 1
            engine_name = engine.__class__.__name__
            records = []

        quality = summarize_ocr_quality(ctx.blk_list)
        if (
            project_checkpoint_hit is None
            and engine_key != "PaddleOCR VL Spotting"
            and quality.get("low_quality", False)
            and not all_empty_blocks_are_rejected(ctx.blk_list)
        ):
            attempt_count += 1
            for blk in ctx.blk_list:
                blk.text = ""
                blk.texts = []
                blk.ocr_regions = []
            if (
                isinstance(paddle_engine, PaddleOCRVLEngine)
                and self._paddleocr_cache_identity is not None
                and self._paddleocr_cache_store is not None
            ):
                if paddle_plan is not None and not paddle_plan.requires_runtime:
                    self._await_ocr_runtime(policy)
                retry_plan = paddle_engine.prepare_persistent_cache(
                    ctx.image,
                    ctx.blk_list,
                    self._paddleocr_cache_store,
                    self._paddleocr_cache_identity,
                    lookup=False,
                )
                paddle_engine.process_persistent_cache_plan(retry_plan)
                records = paddle_engine.build_persistent_cache_records(retry_plan)
                ctx.paddleocr_cache_plan = retry_plan
                page_profile = dict(paddle_engine.last_page_profile or {})
                engine = paddle_engine
            else:
                engine = OCRFactory.create_engine(
                    settings_page,
                    source_lang_english,
                    engine_key,
                    selected_ocr_mode=policy["normalized_ocr_mode"],
                )
                attach_cancel_checker(engine)
                engine.process_image(ctx.image, ctx.blk_list)
                page_profile = dict(getattr(engine, "last_page_profile", {}) or {})
            snapshot_raw_results()
            apply_ocr_result_dictionary(ctx.blk_list, settings_page.get_ocr_result_dictionary_rules())
            if engine_key not in {
                "PaddleOCR VL",
                "PaddleOCR VL Spotting",
            }:
                self.cache_manager._cache_ocr_results(cache_key, ctx.blk_list)
            cache_status = "refreshed"
            quality = summarize_ocr_quality(ctx.blk_list)
            engine_name = engine.__class__.__name__

        if records and self._paddleocr_cache_store is not None:
            self._paddleocr_cache_store.store_records(records)

        ctx.blk_list, rejected_empty_blocks = drop_rejected_empty_ocr_blocks(ctx.blk_list)
        if rejected_empty_blocks:
            logger.info(
                "Dropped %d rejected empty OCR block(s) before stage-batched inpaint for %s.",
                len(rejected_empty_blocks),
                ctx.image_name,
            )
            quality = summarize_ocr_quality(ctx.blk_list)
            page_profile = dict(page_profile or {})
            page_profile["rejected_empty_dropped_block_count"] = len(rejected_empty_blocks)

        ctx.blk_list, embedded_ui_blocks = drop_embedded_ui_ocr_blocks(ctx.blk_list, ctx.image.shape)
        if embedded_ui_blocks:
            logger.info(
                "Dropped %d embedded UI OCR block(s) before stage-batched inpaint for %s.",
                len(embedded_ui_blocks),
                ctx.image_name,
            )
            quality = summarize_ocr_quality(ctx.blk_list)
            page_profile = dict(page_profile or {})
            page_profile["embedded_ui_dropped_block_count"] = len(embedded_ui_blocks)

        routed_blocks = [
            *ctx.blk_list,
            *rejected_empty_blocks,
            *embedded_ui_blocks,
        ]
        ctx.ocr_processing_summary = finalize_ocr_processing_contracts(
            routed_blocks
        )
        page_profile = dict(page_profile or {})
        page_profile["ocr_processing_contract"] = dict(
            ctx.ocr_processing_summary
        )

        canonicalization = dict(ctx.ocr_canonicalization_summary or {})
        if canonicalization:
            page_profile = dict(page_profile or {})
            page_profile["exact_duplicate_canonicalization"] = {
                "input_block_count": int(
                    canonicalization.get("input_block_count", 0) or 0
                ),
                "canonical_block_count": int(
                    canonicalization.get("canonical_block_count", 0) or 0
                ),
                "duplicate_alias_count": int(
                    canonicalization.get("duplicate_alias_count", 0) or 0
                ),
            }

        metrics = self._ocr_quality_metrics(quality)
        retained_ids = {
            str(getattr(block, "block_id", "") or "")
            for block in ctx.blk_list
        }
        raw_results = {
            block_id: payload
            for block_id, payload in raw_results.items()
            if block_id in retained_ids
        }
        return {
            "quality": quality,
            "metrics": metrics,
            "cache_status": cache_status,
            "attempt_count": attempt_count,
            "page_profile": page_profile,
            "engine_name": engine_name,
            "raw_results": raw_results,
        }

    def _prepare_project_ocr_hits(
        self,
        pages: list[StagePageContext],
        policy: dict[str, Any],
        runtime_identity: dict[str, Any],
    ) -> None:
        store = getattr(self, "_project_checkpoint_store", None)
        engine_key = str(policy.get("primary_ocr_engine", ""))
        if store is None or engine_key not in {
            "PaddleOCR VL",
            "PaddleOCR VL Spotting",
        }:
            return
        if engine_key == "PaddleOCR VL":
            paddle_settings = (
                self.main_page.settings_page.get_paddleocr_vl_settings()
            )
        else:
            paddle_settings = (
                self.main_page.settings_page
                .get_paddleocr_vl_spotting_settings()
            )
        for ctx in pages:
            if (
                ctx.failed_stage
                or ctx.no_text_detected
                or not ctx.detection_fingerprint
                or (
                    ctx.project_ocr_checkpoint_status in {"hit", "miss"}
                    and bool(ctx.project_ocr_identity)
                )
            ):
                continue
            try:
                identity = build_project_ocr_identity(
                    detection_fingerprint=ctx.detection_fingerprint,
                    runtime_identity=runtime_identity,
                    policy=policy,
                    paddle_settings=paddle_settings,
                    source_lang_english=self._source_lang_english(ctx.source_lang),
                )
                fingerprint = build_project_ocr_fingerprint(identity)
                ctx.project_ocr_identity = identity
                ctx.project_ocr_fingerprint = fingerprint
                hit = lookup_ocr_checkpoint(
                    store,
                    page_key=ctx.project_checkpoint_page_key,
                    fingerprint=fingerprint,
                    identity=identity,
                    detection_blocks=ctx.blk_list,
                )
            except Exception:
                logger.warning(
                    "OCR project checkpoint lookup failed open for %s.",
                    ctx.image_name,
                    exc_info=True,
                )
                ctx.project_ocr_checkpoint_status = "disabled"
                continue
            if hit is None:
                ctx.project_ocr_checkpoint_status = "miss"
                continue
            ctx.project_ocr_hit = hit
            ctx.project_ocr_checkpoint_status = "hit"
            ctx.blk_list = hit.blocks

    def _record_project_ocr_result(
        self,
        ctx: StagePageContext,
        result: dict[str, Any],
    ) -> None:
        if (
            ctx.project_ocr_checkpoint_status == "hit"
            or not ctx.project_ocr_fingerprint
            or not ctx.project_ocr_identity
        ):
            return
        try:
            recorded = record_ocr_checkpoint(
                getattr(self, "_project_checkpoint_store", None),
                page_key=ctx.project_checkpoint_page_key,
                fingerprint=ctx.project_ocr_fingerprint,
                identity=ctx.project_ocr_identity,
                blocks=ctx.blk_list,
                raw_results=result.get("raw_results") or {},
                attempt_count=int(result.get("attempt_count", 0) or 0),
                engine_name=str(result.get("engine_name", "") or ""),
                page_profile=result.get("page_profile") or {},
            )
        except Exception:
            logger.warning(
                "OCR project checkpoint write failed open for %s.",
                ctx.image_name,
                exc_info=True,
            )
            ctx.project_ocr_checkpoint_status = "disabled"
            return
        if recorded:
            ctx.project_ocr_checkpoint_status = "stored"

    @staticmethod
    def _canonicalize_ocr_inputs(pages: list[StagePageContext]) -> None:
        for ctx in pages:
            if ctx.failed_stage or ctx.no_text_detected or not ctx.blk_list:
                continue
            canonical_blocks, summary = canonicalize_exact_duplicate_blocks(
                ctx.blk_list,
                source_identity=(
                    ctx.source_decoded_sha256
                    or ctx.project_checkpoint_page_key
                ),
            )
            ctx.blk_list = canonical_blocks
            ctx.ocr_canonicalization_summary = summary
            duplicate_count = int(summary.get("duplicate_alias_count", 0) or 0)
            if duplicate_count:
                logger.info(
                    "Canonicalized %d exact detector duplicate(s) before OCR for %s.",
                    duplicate_count,
                    ctx.image_name,
                )

    def _ocr_all(self, pages: list[StagePageContext], policy: dict[str, Any]) -> None:
        total_images = len(pages)
        settings_page = self.main_page.settings_page
        runtime_manager = getattr(self.main_page, "local_ocr_runtime_manager", None)
        engine_key = str(policy["primary_ocr_engine"])
        self._canonicalize_ocr_inputs(pages)
        paddle_settings = (
            settings_page.get_paddleocr_vl_settings()
            if engine_key == "PaddleOCR VL"
            else (
                settings_page.get_paddleocr_vl_spotting_settings()
                if engine_key == "PaddleOCR VL Spotting"
                else {}
            )
        )
        persistent_cache_requested = (
            engine_key == "PaddleOCR VL"
            and bool(
                paddle_settings.get(
                    "persistent_cache_enabled",
                    True,
                )
            )
        )
        cache_identity_required = bool(
            persistent_cache_requested
            or getattr(self, "_project_checkpoint_store", None) is not None
        )
        runtime_identity = dict(
            getattr(self, "_paddleocr_cache_identity", None) or {}
        ) or None
        if (
            cache_identity_required
            and engine_key
            in {"PaddleOCR VL", "PaddleOCR VL Spotting"}
            and isinstance(runtime_manager, LocalOCRRuntimeManager)
        ):
            runtime_identity = runtime_manager.get_ocr_cache_identity(
                engine_key,
                settings_page,
            )
            if runtime_identity is not None:
                self._prepare_project_ocr_hits(
                    pages,
                    policy,
                    runtime_identity,
                )
        def has_project_miss() -> bool:
            return any(
                not ctx.failed_stage
                and not ctx.no_text_detected
                and ctx.project_ocr_hit is None
                and bool(ctx.blk_list)
                for ctx in pages
            )

        requires_runtime = has_project_miss()
        if (
            persistent_cache_requested
            and runtime_identity is not None
            and requires_runtime
        ):
            requires_runtime = self._prepare_paddleocr_cache_plans(
                pages,
                policy,
                runtime_identity,
            )
        if requires_runtime:
            self._await_ocr_runtime(policy)
        if (
            cache_identity_required
            and engine_key
            in {"PaddleOCR VL", "PaddleOCR VL Spotting"}
            and runtime_identity is None
            and isinstance(runtime_manager, LocalOCRRuntimeManager)
            and has_project_miss()
        ):
            runtime_identity = runtime_manager.get_ocr_cache_identity(
                engine_key,
                settings_page,
            )
            if runtime_identity is not None:
                self._prepare_project_ocr_hits(
                    pages,
                    policy,
                    runtime_identity,
                )
                if persistent_cache_requested and has_project_miss():
                    self._prepare_paddleocr_cache_plans(
                        pages,
                        policy,
                        runtime_identity,
                    )

        for index, ctx in enumerate(pages):
            self._raise_if_cancelled()
            if ctx.failed_stage:
                continue
            self._set_current_image(ctx.image_path)
            self.emit_progress(index, total_images, 2, 10, False, stage_name='ocr-all')
            if ctx.no_text_detected:
                self.main_page.image_ctrl.mark_processing_stage(
                    ctx.image_path,
                    "ocr",
                    "skipped",
                    reason="no_text_detected",
                )
                self._emit_benchmark_event(
                    "ocr_end",
                    image_path=ctx.image_path,
                    image_index=index,
                    total_images=total_images,
                    block_count=0,
                    project_checkpoint_status=(
                        ctx.project_ocr_checkpoint_status
                    ),
                    skip_reason="no_text_detected",
                    **ctx.page_ocr_metrics,
                )
                continue
            self._emit_benchmark_event(
                "ocr_start",
                image_path=ctx.image_path,
                image_index=index,
                total_images=total_images,
                block_count=len(ctx.blk_list or []),
            )
            try:
                result = self._run_primary_ocr(ctx, policy)
                self._raise_if_cancelled()
                quality = result["quality"]
                self._log_ocr_quality(ctx.image_path, quality, int(result["attempt_count"]))
                if not ctx.blk_list or int(quality.get("non_empty", 0) or 0) <= 0:
                    ctx.blk_list = []
                    self._record_project_ocr_result(ctx, result)
                    ctx.no_text_detected = True
                    state = self._ensure_page_state(ctx.image_path)
                    state["blk_list"] = []
                    state.setdefault("viewer_state", {})["rectangles"] = []
                    self.main_page.image_ctrl.mark_processing_stage(
                        ctx.image_path,
                        "ocr",
                        "completed",
                        reason="no_text_detected",
                        cache_status=result["cache_status"],
                        attempt_count=int(result["attempt_count"]),
                        quality=quality,
                    )
                    ctx.page_ocr_metrics = dict(result["metrics"] or {})
                    self._emit_benchmark_event(
                        "ocr_end",
                        image_path=ctx.image_path,
                        image_index=index,
                        total_images=total_images,
                        block_count=0,
                        ocr_model=str(policy["primary_ocr_engine"]),
                        ocr_engine=result["engine_name"],
                        cache_status=result["cache_status"],
                        attempt_count=int(result["attempt_count"]),
                        ocr_page_profile=result["page_profile"],
                        project_checkpoint_status=(
                            ctx.project_ocr_checkpoint_status
                        ),
                        skip_reason="no_text_detected",
                        **ctx.page_ocr_metrics,
                    )
                    continue
                if quality.get("low_quality", False):
                    raise RuntimeError(quality.get("reason") or "OCR quality too low after retry.")
                self._record_project_ocr_result(ctx, result)
                device = resolve_device(settings_page.is_gpu_enabled())
                self._persist_ocr_state(
                    ctx.image_path,
                    ctx.blk_list,
                    settings_page.get_tool_selection("ocr"),
                    result["engine_name"],
                    device,
                    quality,
                    result["cache_status"],
                    int(result["attempt_count"]),
                )
                ctx.page_ocr_metrics = dict(result["metrics"] or {})
                self._emit_benchmark_event(
                    "ocr_end",
                    image_path=ctx.image_path,
                    image_index=index,
                    total_images=total_images,
                    block_count=len(ctx.blk_list or []),
                    ocr_model=str(policy["primary_ocr_engine"]),
                    ocr_engine=result["engine_name"],
                    cache_status=result["cache_status"],
                    attempt_count=int(result["attempt_count"]),
                    ocr_page_profile=result["page_profile"],
                    project_checkpoint_status=(
                        ctx.project_ocr_checkpoint_status
                    ),
                    **ctx.page_ocr_metrics,
                )
            except OperationCancelledError:
                raise
            except Exception as exc:
                self._mark_page_failed(
                    ctx,
                    index=index,
                    total_images=total_images,
                    stage="ocr",
                    reason=str(exc),
                    extra=dict(ctx.page_ocr_metrics or {}),
                )

        if isinstance(runtime_manager, LocalOCRRuntimeManager):
            self._shutdown_runtime_with_retry(
                "OCR",
                runtime_manager,
                context="OCR-to-translate handoff",
                raise_on_failure=True,
                release_for_handoff=True,
                service=self._ocr_runtime_service_name(engine_key),
            )
        self._raise_if_cancelled()

    def _prepare_paddleocr_cache_plans(
        self,
        pages: list[StagePageContext],
        policy: dict[str, Any],
        runtime_identity: dict[str, Any],
    ) -> bool:
        settings_page = self.main_page.settings_page
        store = self._get_paddleocr_cache_store()
        self._paddleocr_cache_store = store
        self._paddleocr_cache_identity = dict(runtime_identity)

        requires_runtime = False
        total_images = len(pages)
        for index, ctx in enumerate(pages):
            self._raise_if_cancelled()
            if (
                ctx.failed_stage
                or ctx.no_text_detected
                or ctx.project_ocr_hit is not None
            ):
                continue
            if (
                isinstance(ctx.paddleocr_cache_engine, PaddleOCRVLEngine)
                and ctx.paddleocr_cache_plan is not None
            ):
                requires_runtime = requires_runtime or bool(
                    ctx.paddleocr_cache_plan.requires_runtime
                )
                continue
            try:
                source_lang_english = self._source_lang_english(ctx.source_lang)
                source_lang_code = language_codes.get(source_lang_english, "en")
                for blk in ctx.blk_list:
                    blk.source_lang = source_lang_code
                engine = OCRFactory.create_engine(
                    settings_page,
                    source_lang_english,
                    "PaddleOCR VL",
                    selected_ocr_mode=policy["normalized_ocr_mode"],
                )
                if not isinstance(engine, PaddleOCRVLEngine):
                    raise RuntimeError(
                        "PaddleOCR-VL persistent cache resolved an unexpected OCR engine."
                    )
                setter = getattr(engine, "set_cancel_checker", None)
                if callable(setter):
                    setter(
                        getattr(
                            self.main_page,
                            "is_current_task_cancelled",
                            None,
                        )
                    )
                plan = engine.prepare_persistent_cache(
                    ctx.image,
                    ctx.blk_list,
                    store,
                    runtime_identity,
                )
                ctx.paddleocr_cache_engine = engine
                ctx.paddleocr_cache_plan = plan
                requires_runtime = requires_runtime or plan.requires_runtime
            except OperationCancelledError:
                raise
            except Exception as exc:
                self._mark_page_failed(
                    ctx,
                    index=index,
                    total_images=total_images,
                    stage="ocr",
                    reason=str(exc),
                    extra=dict(ctx.page_ocr_metrics or {}),
                )
        return requires_runtime

    def _get_paddleocr_cache_store(self) -> OCRPersistentResultCache:
        settings_page = self.main_page.settings_page
        cache_settings = settings_page.get_paddleocr_vl_settings()
        cache_limit = int(cache_settings.get("persistent_cache_limit", 50_000))
        store = getattr(
            self.main_page,
            "paddleocr_persistent_result_cache",
            None,
        )
        if not isinstance(store, OCRPersistentResultCache):
            store = OCRPersistentResultCache(result_cache_limit=cache_limit)
            self.main_page.paddleocr_persistent_result_cache = store
        else:
            store.configure_limit(cache_limit)
        return store

    def _ensure_inpainter(self):
        settings_page = self.main_page.settings_page
        runtime = get_inpainter_runtime(settings_page)
        uses_cuda = str(runtime.get("device", "") or "").lower().startswith(
            "cuda"
        )
        if uses_cuda and not bool(
            getattr(self, "_inpainter_runtime_lease_held", False)
        ):
            self._runtime_resource_arbiter().acquire_external_model(
                "inpainter"
            )
            self._inpainter_runtime_lease_held = True
        self.inpainting._ensure_inpainter()
        if uses_cuda:
            self._runtime_resource_arbiter().mark_external_model_ready(
                "inpainter"
            )
        return runtime

    def _release_inpainter_runtime_lease(
        self,
        *,
        release_succeeded: bool,
    ) -> None:
        if not bool(getattr(self, "_inpainter_runtime_lease_held", False)):
            return
        self._runtime_resource_arbiter().release_external_model(
            "inpainter",
            release_succeeded=release_succeeded,
        )
        self._inpainter_runtime_lease_held = False

    def _inpaint_all(self, pages: list[StagePageContext]) -> None:
        active_pages = [
            ctx
            for ctx in pages
            if not ctx.failed_stage and not ctx.no_text_detected
        ]
        runtime_loaded = False
        try:
            runtime_loaded = self._inpaint_pages(pages)
        except BaseException:
            if runtime_loaded or active_pages:
                try:
                    self._release_inpainter_before_render(
                        pages,
                        handoff_outcome="aborted",
                    )
                except Exception:
                    logger.warning(
                        "Failed to release inpainter resources after an "
                        "aborted inpaint stage.",
                        exc_info=True,
                    )
            raise
        if runtime_loaded:
            self._release_inpainter_before_render(pages)
        else:
            self._inpainter_release_gate = {
                "required": False,
                "observed": True,
                "status": "not-loaded",
                "elapsed_sec": 0.0,
            }
            self._emit_benchmark_event(
                "inpainter_release",
                handoff_policy="after-release",
                handoff_outcome="not-loaded",
                inpainter_release={"cached_inpainter_key": ""},
                inpainter_gpu_release_expected=False,
                inpainter_vram_release_required=False,
                inpainter_vram_release_observed=True,
                inpainter_vram_release_status="not-loaded",
                inpainter_vram_release_elapsed_sec=0.0,
            )
            self._release_inpainter_runtime_lease(
                release_succeeded=True,
            )

    def _finish_inpaint_page(
        self,
        ctx: StagePageContext,
        *,
        index: int,
        total_images: int,
        runtime: dict[str, Any],
        hd_strategy: str,
        export_settings: dict[str, Any],
    ) -> None:
        settings_page = self.main_page.settings_page
        ctx.patches = self.inpainting.get_inpainted_patches(
            ctx.mask,
            ctx.inpaint_input_img,
        )
        self.main_page.patches_processed.emit(ctx.patches, ctx.image_path)
        self.main_page.image_ctrl.update_processing_summary(
            ctx.image_path,
            {
                "inpainter": settings_page.get_tool_selection("inpainter"),
                "hd_strategy": hd_strategy,
                "cleanup_applied": bool(
                    ctx.cleanup_stats.get("applied", False)
                ),
                "cleanup_component_count": int(
                    ctx.cleanup_stats.get("component_count", 0) or 0
                ),
                "cleanup_block_count": int(
                    ctx.cleanup_stats.get("block_count", 0) or 0
                ),
                "inpaint_project_checkpoint_status": (
                    ctx.project_inpaint_checkpoint_status
                ),
                "inpaint_runtime_diagnostics": dict(
                    ctx.inpaint_diagnostics or {}
                ),
            },
        )
        cleaned_output_path = self._write_inpainted_debug_image(
            export_root=ctx.export_root,
            archive_bname=ctx.archive_bname,
            image_path=ctx.image_path,
            cleaned_image=ctx.inpaint_input_img,
            export_settings=export_settings,
        )
        self.main_page.image_ctrl.update_processing_summary(
            ctx.image_path,
            {"cleaned_image_path": cleaned_output_path},
        )
        debug_paths = self._write_inpaint_debug_exports(
            export_root=ctx.export_root,
            archive_bname=ctx.archive_bname,
            image_path=ctx.image_path,
            image=ctx.image,
            blk_list=ctx.blk_list,
            export_settings=export_settings,
            raw_mask=ctx.raw_mask,
            final_mask=ctx.mask,
            detector_key=ctx.detector_key,
            detector_engine=ctx.detector_engine,
            detector_device=ctx.detector_device,
            inpainter_key=str(runtime.get("key", "") or ""),
            hd_strategy=hd_strategy,
            cleanup_stats=ctx.cleanup_stats,
            mask_details=ctx.mask_details,
            inpainter_backend=str(runtime.get("backend", "") or ""),
            inpaint_runtime_diagnostics=ctx.inpaint_diagnostics,
        )
        for stage_key, stage_label, preferred_path in (
            ("raw_mask", "원본 마스크", debug_paths.get("raw_mask", "")),
            (
                "mask_overlay",
                "마스크 오버레이",
                debug_paths.get("mask_overlay", ""),
            ),
            (
                "cleanup_delta",
                "정리 마스크 변화량",
                debug_paths.get("cleanup_delta", ""),
            ),
            (
                "inpainted_image",
                "인페인트 결과",
                cleaned_output_path,
            ),
        ):
            self._maybe_emit_preview_image(
                index=index,
                total=total_images,
                image_path=ctx.image_path,
                stage_key=stage_key,
                stage_label=stage_label,
                export_settings=export_settings,
                preferred_path=preferred_path,
            )
        processing_action_skipped = (
            str(ctx.inpaint_diagnostics.get("status", "") or "")
            == "processing_action_skipped"
        )
        stage_kwargs: dict[str, Any] = {
            "patch_count": len(ctx.patches or []),
            "cache_status": ctx.project_inpaint_checkpoint_status,
        }
        if processing_action_skipped:
            stage_kwargs["reason"] = "no_translate_inpaint_blocks"
        self.main_page.image_ctrl.mark_processing_stage(
            ctx.image_path,
            "inpaint",
            "skipped" if processing_action_skipped else "completed",
            **stage_kwargs,
        )
        self._emit_benchmark_event(
            "inpaint_end",
            image_path=ctx.image_path,
            image_index=index,
            total_images=total_images,
            block_count=len(ctx.blk_list or []),
            patch_count=len(ctx.patches or []),
            project_checkpoint_status=ctx.project_inpaint_checkpoint_status,
            skip_reason=(
                "no_translate_inpaint_blocks"
                if processing_action_skipped
                else ""
            ),
            inpaint_runtime_diagnostics=dict(
                ctx.inpaint_diagnostics or {}
            ),
        )

    def _inpaint_pages(self, pages: list[StagePageContext]) -> bool:
        total_images = len(pages)
        settings_page = self.main_page.settings_page
        export_settings = self._effective_export_settings(settings_page)
        hd_strategy_settings = settings_page.get_hd_strategy_settings()
        hd_strategy = settings_page.ui.value_mappings.get(
            hd_strategy_settings.get("strategy", ""),
            hd_strategy_settings.get("strategy", ""),
        )
        runtime = get_inpainter_runtime(settings_page)
        model_identity = registered_inpainter_model_identity(
            str(runtime.get("key", "") or ""),
            str(runtime.get("backend", "") or ""),
        )
        mask_settings = settings_page.get_mask_refiner_settings()
        config = get_config(settings_page)
        checkpoint_store = getattr(self, "_project_checkpoint_store", None)
        pending: list[tuple[int, StagePageContext, list[Any]]] = []

        for index, ctx in enumerate(pages):
            self._raise_if_cancelled()
            self._drain_render_futures(block=False)
            if ctx.failed_stage:
                continue
            self._set_current_image(ctx.image_path)
            self.emit_progress(index, total_images, 3, 10, False, stage_name='inpaint-all')
            if ctx.no_text_detected:
                ctx.inpaint_input_img = ctx.image
                ctx.patches = []
                ctx.project_inpaint_checkpoint_status = "skipped"
                if (
                    checkpoint_store is not None
                    and ctx.source_decoded_sha256
                    and ctx.detection_fingerprint
                ):
                    ctx.project_translation_fingerprint = (
                        build_skipped_stage_fingerprint(
                            stage="translation",
                            source_sha256=ctx.source_decoded_sha256,
                            detection_fingerprint=ctx.detection_fingerprint,
                            reason="no_text_detected",
                        )
                    )
                    ctx.project_translation_checkpoint_status = "skipped"
                    ctx.project_inpaint_fingerprint = (
                        build_skipped_stage_fingerprint(
                            stage="inpaint",
                            source_sha256=ctx.source_decoded_sha256,
                            detection_fingerprint=ctx.detection_fingerprint,
                            reason="no_text_detected",
                        )
                    )
                    ctx.project_inpaint_artifact_sha256 = (
                        decoded_image_sha256(ctx.image)
                    )
                self.main_page.image_ctrl.mark_processing_stage(
                    ctx.image_path,
                    "inpaint",
                    "skipped",
                    reason="no_text_detected",
                    patch_count=0,
                )
                self._emit_benchmark_event(
                    "inpaint_end",
                    image_path=ctx.image_path,
                    image_index=index,
                    total_images=total_images,
                    block_count=0,
                    patch_count=0,
                    skip_reason="no_text_detected",
                    project_checkpoint_status="skipped",
                )
                # 인페인팅은 건너뛰어도 **출력은 반드시 나가야 한다.** 이 분기가
                # 렌더 제출 없이 넘어가는 바람에, 텍스트가 없다고 판정된 페이지는
                # 정상 경로로 파일을 남기지 못했다. 실측 366장에서 15장이 여기로
                # 빠져 배치 끝 폴백이 대신 저장했고, 실패가 아닌데도 실패 경로로
                # 처리됐다. 글자가 없으면 렌더할 텍스트가 없을 뿐, 페이지 자체는
                # 그대로 내보내면 된다.
                self._submit_or_inline_render(
                    ctx,
                    index=index,
                    total_images=total_images,
                    export_settings=export_settings,
                )
                continue
            self._emit_benchmark_event(
                "inpaint_start",
                image_path=ctx.image_path,
                image_index=index,
                total_images=total_images,
                block_count=len(ctx.blk_list or []),
            )
            try:
                inpaint_blocks, protected_blocks = (
                    split_inpaint_protected_ocr_blocks(ctx.blk_list)
                )
                page_state = self._ensure_page_state(ctx.image_path)
                brush_strokes = list(
                    page_state.get("brush_strokes", []) or []
                )

                if not inpaint_blocks and not brush_strokes:
                    zero_mask = np.zeros(
                        ctx.image.shape[:2],
                        dtype=np.uint8,
                    )
                    ctx.inpaint_input_img = np.ascontiguousarray(
                        ctx.image.copy(),
                        dtype=np.uint8,
                    )
                    ctx.raw_mask = zero_mask.copy()
                    ctx.mask = zero_mask
                    protected_reasons = [
                        str(
                            getattr(
                                block,
                                "_inpaint_protected_reason",
                                "",
                            )
                            or ""
                        )
                        for block in protected_blocks
                    ]
                    ctx.mask_details = {
                        "raw_mask": ctx.raw_mask,
                        "final_mask": ctx.mask,
                        "processing_action_skipped": True,
                        "inpaint_protected_block_count": len(
                            protected_blocks
                        ),
                        "inpaint_protected_reasons": protected_reasons,
                    }
                    ctx.cleanup_stats = {
                        "applied": False,
                        "component_count": 0,
                        "block_count": 0,
                        "reason": "no_translate_inpaint_blocks",
                    }
                    ctx.inpaint_diagnostics = {
                        "status": "processing_action_skipped",
                        "reason": "no_translate_inpaint_blocks",
                        "inference_call_count": 0,
                        "cpu_fallback_used": False,
                    }
                    ctx.project_inpaint_checkpoint_status = "skipped"
                    if (
                        checkpoint_store is not None
                        and ctx.source_decoded_sha256
                        and ctx.detection_fingerprint
                    ):
                        ctx.project_inpaint_fingerprint = (
                            build_skipped_stage_fingerprint(
                                stage="inpaint",
                                source_sha256=ctx.source_decoded_sha256,
                                detection_fingerprint=(
                                    ctx.detection_fingerprint
                                ),
                                reason="no_translate_inpaint_blocks",
                            )
                        )
                    ctx.project_inpaint_artifact_sha256 = (
                        decoded_image_sha256(ctx.inpaint_input_img)
                    )
                    self._finish_inpaint_page(
                        ctx,
                        index=index,
                        total_images=total_images,
                        runtime=runtime,
                        hd_strategy=hd_strategy,
                        export_settings=export_settings,
                    )
                    self._submit_or_inline_render(
                        ctx,
                        index=index,
                        total_images=total_images,
                        export_settings=export_settings,
                    )
                    continue

                project_hit = None
                if (
                    checkpoint_store is not None
                    and ctx.project_checkpoint_page_key
                    and ctx.source_decoded_sha256
                    and ctx.detection_fingerprint
                    and ctx.project_ocr_fingerprint
                ):
                    try:
                        ctx.project_inpaint_identity = build_inpaint_identity(
                            source_sha256=ctx.source_decoded_sha256,
                            detection_fingerprint=ctx.detection_fingerprint,
                            ocr_fingerprint=ctx.project_ocr_fingerprint,
                            blocks=inpaint_blocks,
                            brush_strokes=brush_strokes,
                            runtime=runtime,
                            model_identity=model_identity,
                            hd_strategy=hd_strategy_settings,
                            mask_settings=mask_settings,
                        )
                        ctx.project_inpaint_fingerprint = (
                            build_inpaint_fingerprint(
                                ctx.project_inpaint_identity
                            )
                        )
                        project_hit = lookup_inpaint_checkpoint(
                            checkpoint_store,
                            page_key=ctx.project_checkpoint_page_key,
                            fingerprint=ctx.project_inpaint_fingerprint,
                            identity=ctx.project_inpaint_identity,
                            source_shape=tuple(
                                int(item) for item in ctx.image.shape
                            ),
                            current_blocks=ctx.blk_list,
                        )
                    except Exception:
                        logger.warning(
                            "Inpaint checkpoint lookup failed open for %s.",
                            ctx.image_name,
                            exc_info=True,
                        )
                        ctx.project_inpaint_identity = {}
                        ctx.project_inpaint_fingerprint = ""
                        project_hit = None
                if project_hit is not None:
                    restore_inpaint_block_state(
                        ctx.blk_list,
                        project_hit.block_states,
                    )
                    ctx.inpaint_input_img = project_hit.cleaned_image
                    ctx.raw_mask = project_hit.raw_mask
                    ctx.mask = project_hit.final_mask
                    ctx.mask_details = {
                        "raw_mask": ctx.raw_mask,
                        "final_mask": ctx.mask,
                        "project_checkpoint_restored": True,
                    }
                    if protected_blocks:
                        ctx.mask_details[
                            "inpaint_protected_block_count"
                        ] = len(protected_blocks)
                        ctx.mask_details[
                            "inpaint_protected_reasons"
                        ] = [
                            getattr(
                                block,
                                "_inpaint_protected_reason",
                                "",
                            )
                            for block in protected_blocks
                        ]
                    ctx.cleanup_stats = project_hit.cleanup_stats
                    ctx.inpaint_diagnostics = {
                        "status": "project_checkpoint_hit",
                        "inference_call_count": 0,
                        "cpu_fallback_used": False,
                    }
                    ctx.project_inpaint_artifact_sha256 = (
                        project_hit.cleaned_decoded_sha256
                    )
                    ctx.project_inpaint_checkpoint_status = "hit"
                    self._finish_inpaint_page(
                        ctx,
                        index=index,
                        total_images=total_images,
                        runtime=runtime,
                        hd_strategy=hd_strategy,
                        export_settings=export_settings,
                    )
                    self._submit_or_inline_render(
                        ctx,
                        index=index,
                        total_images=total_images,
                        export_settings=export_settings,
                    )
                    continue

                with self._measure_performance(
                    stage="inpaint",
                    operation="mask_generation",
                    workload={
                        "block_count": len(inpaint_blocks),
                        "page_pixel_count": int(ctx.image.shape[0])
                        * int(ctx.image.shape[1]),
                    },
                ):
                    ctx.mask_details = generate_mask(
                        ctx.image,
                        inpaint_blocks,
                        settings=mask_settings,
                        return_details=True,
                        precomputed_mask_details=ctx.precomputed_mask_details,
                    )
                if protected_blocks:
                    ctx.mask_details["inpaint_protected_block_count"] = len(
                        protected_blocks
                    )
                    ctx.mask_details["inpaint_protected_reasons"] = [
                        getattr(block, "_inpaint_protected_reason", "")
                        for block in protected_blocks
                    ]
                ctx.mask = np.ascontiguousarray(
                    ctx.mask_details["final_mask"]
                )
                ctx.raw_mask = np.ascontiguousarray(
                    ctx.mask_details["raw_mask"]
                )
                if brush_strokes:
                    automatic_mask = np.ascontiguousarray(ctx.mask.copy())
                    merged_mask = (
                        self.inpainting._generate_mask_from_saved_strokes(
                            brush_strokes,
                            ctx.image,
                            base_mask=ctx.mask,
                        )
                    )
                    if merged_mask is not None:
                        ctx.mask = np.ascontiguousarray(
                            merged_mask,
                            dtype=np.uint8,
                        )
                        ctx.mask_details["final_mask"] = ctx.mask
                        (
                            ctx.mask_details["protected_corner_mask"],
                            released_protected_pixels,
                        ) = release_protected_mask_for_explicit_additions(
                            ctx.mask_details.get("protected_corner_mask"),
                            automatic_mask,
                            ctx.mask,
                            ctx.image.shape,
                        )
                        ctx.mask_details[
                            "project_brush_strokes_applied"
                        ] = True
                        ctx.mask_details[
                            "protected_corner_brush_override_pixel_count"
                        ] = int(released_protected_pixels)

                ctx.project_inpaint_checkpoint_status = (
                    "miss" if checkpoint_store is not None else "disabled"
                )
                pending.append((index, ctx, inpaint_blocks))
            except OperationCancelledError:
                raise
            except Exception as exc:
                detail = (
                    f"{type(exc).__name__}: {exc}\n\n"
                    f"{traceback.format_exc()}"
                )
                self._mark_page_failed(
                    ctx,
                    index=index,
                    total_images=total_images,
                    stage="inpaint",
                    reason=str(exc),
                    detail=detail,
                    extra={
                        **ctx.page_ocr_metrics,
                        **ctx.page_translation_metrics,
                    },
                )

        runtime_loaded = False
        if pending:
            self._raise_if_cancelled()
            self._drain_render_futures(block=False)
            with self._measure_performance(
                stage="inpaint",
                operation="model_load",
                workload={"page_count": len(pending)},
                service="inpainter",
            ):
                self._ensure_inpainter()
            self._sample_performance_resources("inpainter_model_ready")
            runtime_loaded = True
            refreshed_model_identity = registered_inpainter_model_identity(
                str(runtime.get("key", "") or ""),
                str(runtime.get("backend", "") or ""),
            )
            if refreshed_model_identity != model_identity:
                model_identity = refreshed_model_identity
                for _index, ctx, inpaint_blocks in pending:
                    if not ctx.project_inpaint_identity:
                        continue
                    page_state = self._ensure_page_state(ctx.image_path)
                    ctx.project_inpaint_identity = build_inpaint_identity(
                        source_sha256=ctx.source_decoded_sha256,
                        detection_fingerprint=ctx.detection_fingerprint,
                        ocr_fingerprint=ctx.project_ocr_fingerprint,
                        blocks=inpaint_blocks,
                        brush_strokes=list(
                            page_state.get("brush_strokes", []) or []
                        ),
                        runtime=runtime,
                        model_identity=model_identity,
                        hd_strategy=hd_strategy_settings,
                        mask_settings=mask_settings,
                    )
                    ctx.project_inpaint_fingerprint = (
                        build_inpaint_fingerprint(
                            ctx.project_inpaint_identity
                        )
                    )

        for index, ctx, inpaint_blocks in pending:
            if ctx.failed_stage:
                continue
            try:
                self._raise_if_cancelled()
                self._drain_render_futures(block=False)
                with self._measure_performance(
                    stage="inpaint",
                    operation="model_forward",
                    workload={
                        "block_count": len(inpaint_blocks),
                        "mask_pixel_count": int(np.count_nonzero(ctx.mask)),
                    },
                    service="inpainter",
                ):
                    ctx.inpaint_input_img = self.inpainting.inpaint_with_blocks(
                        ctx.image,
                        ctx.mask,
                        inpaint_blocks,
                        config=config,
                        protected_corner_mask=ctx.mask_details.get(
                            "protected_corner_mask"
                        ),
                    )
                self._sample_performance_resources("inpainter_forward_end")
                ctx.inpaint_diagnostics = dict(
                    getattr(
                        self.inpainting,
                        "last_inpaint_diagnostics",
                        {},
                    )
                    or {}
                )
                ctx.inpaint_diagnostics["model_identity"] = copy.deepcopy(
                    model_identity
                )
                self._raise_if_cancelled()
                with self._measure_performance(
                    stage="inpaint",
                    operation="cleanup_and_composite",
                    workload={
                        "block_count": len(inpaint_blocks),
                        "mask_pixel_count": int(np.count_nonzero(ctx.mask)),
                    },
                ):
                    cleanup = run_inpaint_cleanup(
                        InpaintCleanupInput(
                            image=ctx.image,
                            inpaint_input_img=ctx.inpaint_input_img,
                            mask=ctx.mask,
                            mask_details=ctx.mask_details,
                            inpaint_blocks=inpaint_blocks,
                            config=config,
                            page_label=f"{index + 1}/{total_images}",
                            inpaint_edit_mask=getattr(
                                self.inpainting,
                                "last_inpaint_edit_mask",
                                None,
                            ),
                        )
                    )
                    ctx.inpaint_input_img = cleanup.inpaint_input_img
                    ctx.mask = cleanup.mask
                    ctx.cleanup_stats = cleanup.cleanup_stats
                    outside_before_restore = cleanup.outside_before_restore
                    outside_after_restore = cleanup.outside_after_restore
                    # 진행 보고는 파이프라인 스레드에 남긴다. 계산과 달리 이건
                    # 시그널을 건드린다.
                    self._report_residue_cleanup(
                        index=index,
                        total=total_images,
                        image_path=ctx.image_path,
                        cleanup_stats=ctx.cleanup_stats,
                    )
                ctx.inpaint_diagnostics[
                    "outside_mask_changed_before_restore"
                ] = int(outside_before_restore)
                ctx.inpaint_diagnostics[
                    "outside_mask_changed_pixel_count"
                ] = int(outside_after_restore)
                if outside_after_restore:
                    raise RuntimeError(inpaint_outside_mask_message())
                ctx.mask_details[
                    "inpaint_runtime_diagnostics"
                ] = copy.deepcopy(ctx.inpaint_diagnostics)
                self._raise_if_cancelled()
                ctx.inpaint_input_img = np.ascontiguousarray(
                    ctx.inpaint_input_img,
                    dtype=np.uint8,
                )
                ctx.mask = np.ascontiguousarray(ctx.mask, dtype=np.uint8)
                ctx.project_inpaint_artifact_sha256 = decoded_image_sha256(
                    ctx.inpaint_input_img
                )
                if ctx.project_inpaint_identity:
                    try:
                        stored = record_inpaint_checkpoint(
                            checkpoint_store,
                            page_key=ctx.project_checkpoint_page_key,
                            fingerprint=ctx.project_inpaint_fingerprint,
                            identity=ctx.project_inpaint_identity,
                            blocks=ctx.blk_list,
                            cleaned_image=ctx.inpaint_input_img,
                            raw_mask=ctx.raw_mask,
                            final_mask=ctx.mask,
                            cleanup_stats=ctx.cleanup_stats,
                            cleaned_decoded_sha256=(
                                ctx.project_inpaint_artifact_sha256
                            ),
                        )
                    except Exception:
                        logger.warning(
                            "Inpaint checkpoint publication failed open for "
                            "%s.",
                            ctx.image_name,
                            exc_info=True,
                        )
                        stored = False
                    ctx.project_inpaint_checkpoint_status = (
                        "refreshed" if stored else "miss"
                    )
                self._finish_inpaint_page(
                    ctx,
                    index=index,
                    total_images=total_images,
                    runtime=runtime,
                    hd_strategy=hd_strategy,
                    export_settings=export_settings,
                )
                self._submit_or_inline_render(
                    ctx,
                    index=index,
                    total_images=total_images,
                    export_settings=export_settings,
                )
            except OperationCancelledError:
                raise
            except Exception as exc:
                detail = (
                    f"{type(exc).__name__}: {exc}\n\n"
                    f"{traceback.format_exc()}"
                )
                self._mark_page_failed(
                    ctx,
                    index=index,
                    total_images=total_images,
                    stage="inpaint",
                    reason=str(exc),
                    detail=detail,
                    extra={
                        **ctx.page_ocr_metrics,
                        **ctx.page_translation_metrics,
                    },
                )
        return runtime_loaded

    def _release_gemma_before_inpainter(self) -> None:
        """인페인팅 전에 Router 컨테이너를 확실히 내린다.

        번역 단계는 정상 종료 시 이미 Gemma 를 정지한다. 이 함수는 그 뒤에도 남아
        있는 경우를 위한 확실한 마감이다. 컨테이너가 열려 있으면 아무 모델이 없어도
        CUDA 컨텍스트로 약 278 MiB 를 붙들고, 그만큼 LaMa 가 못 쓴다. 예전 순서에서는
        그 상태로 인페인팅 sweep 전체(실측 467초)를 지났다.
        """

        runtime_manager = getattr(
            self.main_page, "local_translation_runtime_manager", None
        )
        if not isinstance(runtime_manager, LocalGemmaRuntimeManager):
            return
        try:
            self._shutdown_runtime_with_retry(
                "Gemma",
                runtime_manager,
                context="translation-to-inpaint handoff",
                raise_on_failure=False,
                service="gemma",
            )
        except Exception:
            # 인페인팅을 막지 않는다. 남은 컨텍스트는 VRAM 여유를 줄일 뿐이고,
            # 실제로 부족하면 인페인터 적재가 그 사실을 드러낸다.
            logger.warning(
                "Could not stop the Gemma runtime before inpainting; continuing.",
                exc_info=True,
            )

    def _release_inpainter_before_render(
        self,
        pages: list[StagePageContext],
        *,
        handoff_outcome: str = "completed",
    ) -> None:
        """인페인팅이 끝나면 LaMa 를 내린다. 다음은 렌더다.

        예전에는 여기서 Gemma 예열을 시작했다. 순서가 번역 → 인페인팅으로 바뀌면서
        이 지점 다음은 번역이 아니라 렌더가 되었고, 그 훅은 도달할 수 없게 되었다.
        두 호출부 모두 `start_gemma=False` 를 넘기고 있었으므로 매개변수째 걷어낸다.
        """
        try:
            report = self.inpainting.release_inpainter_resources()
        except BaseException:
            self._release_inpainter_runtime_lease(
                release_succeeded=False,
            )
            raise
        gate = dict(report.get("vram_release_gate") or {})
        self._inpainter_release_gate = gate
        self._emit_benchmark_event(
            "inpainter_release",
            handoff_policy="after-release",
            handoff_outcome=handoff_outcome,
            inpainter_release=report,
            inpainter_gpu_release_expected=bool(report.get("gpu_release_expected")),
            inpainter_vram_release_required=bool(gate.get("required")),
            inpainter_vram_release_observed=gate.get("observed"),
            inpainter_vram_release_status=str(gate.get("status") or ""),
            inpainter_vram_release_elapsed_sec=float(gate.get("elapsed_sec", 0.0) or 0.0),
        )
        release_succeeded = not bool(
            gate.get("required") and not gate.get("observed")
        )
        self._release_inpainter_runtime_lease(
            release_succeeded=release_succeeded,
        )
        if not release_succeeded:
            if gpu_release_enforcement_enabled():
                raise RuntimeError(
                    QCoreApplication.translate(
                        "StageBatchedProcessor",
                        "Gemma could not start because inpainter VRAM release was not confirmed."
                    )
                )
            logger.warning(
                "인페인터 VRAM 반환을 확인하지 못했습니다(status=%s). 렌더 단계를 계속합니다.",
                gate.get("status", "unknown"),
            )

    def _build_project_translation_identity(
        self,
        ctx: StagePageContext,
        translator: Translator,
        *,
        extra_context: str,
        runtime_manager: Any,
    ) -> dict[str, Any] | None:
        checkpoint_store = getattr(self, "_project_checkpoint_store", None)
        if (
            checkpoint_store is None
            or not ctx.project_ocr_fingerprint
            or not translator.uses_persistent_translation_memory
            or not isinstance(runtime_manager, LocalGemmaRuntimeManager)
        ):
            return None
        try:
            runtime_identity = runtime_manager.get_translation_cache_identity(
                self.main_page.settings_page
            )
            if (
                not isinstance(runtime_identity, dict)
                or not runtime_identity.get("model_sha256")
                or not runtime_identity.get("runtime_fingerprint")
            ):
                return None
            engine = translator.engine
            engine_contract = {
                "contract_version": 1,
                "model": str(getattr(engine, "model", "") or ""),
                "chunk_size": int(
                    getattr(engine, "chunk_size", 0) or 0
                ),
                "request_mode": str(
                    getattr(engine, "request_mode", "") or ""
                ),
                "prompt_profile": str(
                    getattr(engine, "prompt_profile", "") or ""
                ),
                "response_format_mode": str(
                    getattr(engine, "response_format_mode", "") or ""
                ),
                "response_schema_mode": str(
                    getattr(engine, "response_schema_mode", "") or ""
                ),
                "think_briefly_prompt": bool(
                    getattr(engine, "think_briefly_prompt", False)
                ),
                "contextual_merge_input": bool(
                    getattr(engine, "contextual_merge_input", False)
                ),
                "temperature": float(
                    getattr(engine, "temperature", 0.0) or 0.0
                ),
                "top_k": int(getattr(engine, "top_k", 0) or 0),
                "top_p": float(
                    getattr(engine, "top_p", 0.0) or 0.0
                ),
                "min_p": float(
                    getattr(engine, "min_p", 0.0) or 0.0
                ),
                "max_tokens": int(
                    getattr(engine, "max_tokens", 0) or 0
                ),
            }
            store = getattr(self.main_page, "translation_memory_store", None)
            tm_revision_getter = getattr(store, "get_tm_revision", None)
            tm_revision = (
                int(tm_revision_getter())
                if callable(tm_revision_getter)
                else 0
            )
            settings_page = self.main_page.settings_page
            engine_contract["translation_memory"] = {
                **dict(
                    settings_page.get_translation_memory_settings() or {}
                ),
                "tm_revision": tm_revision,
            }
            engine_contract["ocr_processing_contract"] = {
                "schema_version": int(
                    ctx.ocr_processing_summary.get("schema_version", 0)
                    or 0
                ),
                "selected_blocks": [
                    {
                        "block_id": str(
                            getattr(block, "block_id", "") or ""
                        ),
                        "semantic_role": str(
                            getattr(block, "semantic_role", "") or ""
                        ),
                        "processing_action": str(
                            getattr(block, "processing_action", "") or ""
                        ),
                    }
                    for block in ctx.translation_blocks
                ],
            }
            return build_translation_identity(
                ocr_fingerprint=ctx.project_ocr_fingerprint,
                source_lang=ctx.source_lang,
                target_lang=ctx.target_lang,
                extra_context=extra_context,
                translator_key=translator.translator_key,
                translator_engine=engine.__class__.__name__,
                translator_settings=engine_contract,
                runtime_identity=runtime_identity,
                dictionary_fingerprint=canonical_sha256(
                    settings_page.get_translation_result_dictionary_rules()
                    or []
                ),
            )
        except Exception:
            logger.warning(
                "Translation checkpoint identity resolution failed open for "
                "%s.",
                ctx.image_name,
                exc_info=True,
            )
            return None

    @staticmethod
    def _translation_snapshot_for_blocks(
        snapshot: list[dict[str, Any]] | None,
        blocks: list[Any],
    ) -> list[dict[str, Any]]:
        by_id = {
            str(item.get("block_id", "") or ""): item
            for item in list(snapshot or [])
            if isinstance(item, dict)
            and str(item.get("block_id", "") or "")
        }
        selected: list[dict[str, Any]] = []
        for block in blocks:
            block_id = str(getattr(block, "block_id", "") or "")
            item = by_id.get(block_id)
            if item is None:
                return []
            selected.append(copy.deepcopy(item))
        return selected

    def _translate_all(self, pages: list[StagePageContext]) -> None:
        total_images = len(pages)
        settings_page = self.main_page.settings_page
        extra_context = settings_page.get_llm_settings()["extra_context"]
        translator_key = settings_page.get_tool_selection("translator")
        runtime_manager = getattr(
            self.main_page,
            "local_translation_runtime_manager",
            None,
        )

        prepared_translators: dict[int, Translator] = {}
        project_hits: dict[int, Any] = {}
        gemma_runtime_required = False
        planning_started_at = time.perf_counter()
        for ctx in pages:
            if ctx.failed_stage or ctx.no_text_detected:
                continue
            ctx.translation_blocks = select_translate_inpaint_blocks(
                ctx.blk_list
            )
            if not ctx.translation_blocks:
                ctx.project_translation_checkpoint_status = "skipped"
                ctx.project_translation_fingerprint = (
                    build_skipped_stage_fingerprint(
                        stage="translation",
                        source_sha256=ctx.source_decoded_sha256,
                        detection_fingerprint=ctx.detection_fingerprint,
                        reason="no_translate_inpaint_blocks",
                    )
                )
                continue
            translator = Translator(
                self.main_page,
                ctx.source_lang,
                ctx.target_lang,
            )
            prepared_translators[id(ctx)] = translator
            identity = self._build_project_translation_identity(
                ctx,
                translator,
                extra_context=extra_context,
                runtime_manager=runtime_manager,
            )
            if identity is not None:
                try:
                    ctx.project_translation_identity = identity
                    ctx.project_translation_fingerprint = (
                        build_translation_fingerprint(identity)
                    )
                    project_hit = lookup_translation_checkpoint(
                        getattr(self, "_project_checkpoint_store", None),
                        page_key=ctx.project_checkpoint_page_key,
                        fingerprint=ctx.project_translation_fingerprint,
                        identity=identity,
                        current_blocks=ctx.translation_blocks,
                        project_snapshot=(
                            self._translation_snapshot_for_blocks(
                                ctx.project_translation_snapshot,
                                ctx.translation_blocks,
                            )
                        ),
                    )
                except Exception:
                    logger.warning(
                        "Translation checkpoint lookup failed open for %s.",
                        ctx.image_name,
                        exc_info=True,
                    )
                    ctx.project_translation_identity = {}
                    ctx.project_translation_fingerprint = ""
                    project_hit = None
                if project_hit is not None:
                    apply_translation_checkpoint(
                        ctx.translation_blocks,
                        project_hit,
                    )
                    ctx.project_translation_checkpoint_status = "hit"
                    project_hits[id(ctx)] = project_hit
                    continue
                ctx.project_translation_checkpoint_status = "miss"
            if translator.uses_persistent_translation_memory:
                gemma_runtime_required = (
                    translator.prepare_translation(
                        ctx.translation_blocks,
                        extra_context,
                    )
                    or gemma_runtime_required
                )

        self._record_performance_detail(
            stage="translate",
            operation="plan",
            elapsed_ms=(time.perf_counter() - planning_started_at) * 1000.0,
            workload={
                "page_count": total_images,
                "translator_count": len(prepared_translators),
                "project_hit_count": len(project_hits),
            },
        )

        gemma_runtime_started = False
        if gemma_runtime_required:
            gate = dict(getattr(self, "_inpainter_release_gate", {}) or {})
            if gate.get("required") and not gate.get("observed"):
                if gpu_release_enforcement_enabled():
                    raise RuntimeError(
                        QCoreApplication.translate(
                            "StageBatchedProcessor",
                            "Gemma could not start because inpainter VRAM release was not confirmed."
                        )
                    )
                # 인페인터 VRAM 확인에 실패했다고 번역 단계를 포기하지 않는다.
                # 실제로 메모리가 부족하면 Gemma 적재가 그 사실을 드러낸다.
                logger.warning(
                    "인페인터 VRAM 반환을 확인하지 못했습니다(status=%s). "
                    "Gemma 기동을 계속합니다.",
                    gate.get("status", "unknown"),
                )
            # 프리페치가 아직 돌고 있으면 끝내고 적재한다. 같은 파일을 둘이 동시에
            # 읽으면 디스크를 두 배로 두드린다.
            self._await_gemma_page_cache_prefetch()
            self._start_gemma_prewarm()
            self._await_gemma_runtime()
            gemma_runtime_started = True

        for index, ctx in enumerate(pages):
            self._raise_if_cancelled()
            if ctx.failed_stage:
                continue
            self._set_current_image(ctx.image_path)
            self.emit_progress(index, total_images, 7, 10, False, stage_name='translate-all')
            if ctx.no_text_detected:
                self.main_page.image_ctrl.mark_processing_stage(
                    ctx.image_path,
                    "translation",
                    "skipped",
                    reason="no_text_detected",
                )
                self._emit_benchmark_event(
                    "translate_end",
                    image_path=ctx.image_path,
                    image_index=index,
                    total_images=total_images,
                    block_count=0,
                    translator_key=translator_key,
                    skip_reason="no_text_detected",
                )
                continue
            if not ctx.translation_blocks:
                self.main_page.image_ctrl.mark_processing_stage(
                    ctx.image_path,
                    "translation",
                    "skipped",
                    reason="no_translate_inpaint_blocks",
                )
                self._emit_benchmark_event(
                    "translate_end",
                    image_path=ctx.image_path,
                    image_index=index,
                    total_images=total_images,
                    block_count=0,
                    preserved_or_review_block_count=len(
                        ctx.blk_list or []
                    ),
                    translator_key=translator_key,
                    skip_reason="no_translate_inpaint_blocks",
                    project_checkpoint_status="skipped",
                )
                continue
            self._report_runtime_progress(
                phase="pipeline",
                service="gemma",
                status="running",
                step_key="translation",
                stage_name="translation",
                message=f"{index + 1}/{total_images} 페이지 Gemma 번역 중...",
                page_index=index,
                page_total=total_images,
                image_name=ctx.image_name,
            )
            self._emit_benchmark_event(
                "translate_start",
                image_path=ctx.image_path,
                image_index=index,
                total_images=total_images,
                block_count=len(ctx.translation_blocks),
                preserved_or_review_block_count=(
                    len(ctx.blk_list or []) - len(ctx.translation_blocks)
                ),
                translator_key=translator_key,
            )
            translator = prepared_translators[id(ctx)]
            if id(ctx) in project_hits:
                try:
                    self._persist_translation_state(
                        ctx.image_path,
                        ctx.blk_list,
                        translator_key,
                        translator.engine.__class__.__name__,
                        "project-checkpoint",
                    )
                    self._emit_benchmark_event(
                        "translate_end",
                        image_path=ctx.image_path,
                        image_index=index,
                        total_images=total_images,
                        block_count=len(ctx.translation_blocks),
                        preserved_or_review_block_count=(
                            len(ctx.blk_list or [])
                            - len(ctx.translation_blocks)
                        ),
                        translator_key=translator_key,
                        translator_engine=translator.engine.__class__.__name__,
                        cache_status="project-checkpoint",
                        project_checkpoint_status="hit",
                    )
                    raw_text_obj = json.loads(
                        get_raw_text(ctx.translation_blocks)
                    )
                    translated_text_obj = json.loads(
                        get_raw_translation(ctx.translation_blocks)
                    )
                    if not raw_text_obj or not translated_text_obj:
                        raise RuntimeError(
                            "Translation checkpoint returned empty JSON."
                        )
                    continue
                except Exception as exc:
                    self._mark_page_failed(
                        ctx,
                        index=index,
                        total_images=total_images,
                        stage="translation",
                        reason=str(exc),
                        extra={
                            **ctx.page_ocr_metrics,
                            **ctx.page_translation_metrics,
                        },
                    )
                    continue
            if translator.uses_persistent_translation_memory:
                # The factory may share one engine across page translators, and
                # the SQLite/runtime identity can change after folder preflight.
                # Refresh this page's plan immediately before use.
                with self._measure_performance(
                    stage="translate",
                    operation="refresh_plan",
                    workload={
                        "block_count": len(ctx.translation_blocks),
                    },
                ):
                    runtime_required_now = translator.prepare_translation(
                        ctx.translation_blocks,
                        extra_context,
                    )
                if runtime_required_now and not gemma_runtime_started:
                    gate = dict(getattr(self, "_inpainter_release_gate", {}) or {})
                    if gate.get("required") and not gate.get("observed"):
                        if gpu_release_enforcement_enabled():
                            raise RuntimeError(
                                QCoreApplication.translate(
                                    "StageBatchedProcessor",
                                    "Gemma could not start because inpainter VRAM release was not confirmed."
                                )
                            )
                        logger.warning(
                            "인페인터 VRAM 반환을 확인하지 못했습니다(status=%s). "
                            "Gemma 기동을 계속합니다.",
                            gate.get("status", "unknown"),
                        )
                    self._start_gemma_prewarm()
                    self._await_gemma_runtime()
                    gemma_runtime_started = True
            try:
                with self._measure_performance(
                    stage="translate",
                    operation="inference_and_cache",
                    workload={
                        "block_count": len(ctx.translation_blocks),
                    },
                ):
                    _, translation_cache_status = translator.translate_with_cache_manager(
                        ctx.translation_blocks,
                        ctx.image,
                        extra_context,
                        self.cache_manager,
                    )
                self._sample_performance_resources("gemma_request_end")
                self._raise_if_cancelled()
                with self._measure_performance(
                    stage="translate",
                    operation="dictionary_and_sanitizer",
                    workload={
                        "block_count": len(ctx.translation_blocks),
                    },
                ):
                    apply_translation_result_dictionary(
                        ctx.translation_blocks,
                        settings_page.get_translation_result_dictionary_rules(),
                    )
                ctx.page_translation_metrics = self._translation_benchmark_metrics(translator)
                self._persist_translation_state(
                    ctx.image_path,
                    ctx.blk_list,
                    translator_key,
                    translator.engine.__class__.__name__,
                    translation_cache_status,
                )
                if ctx.project_translation_identity:
                    try:
                        stored = record_translation_checkpoint(
                            getattr(
                                self,
                                "_project_checkpoint_store",
                                None,
                            ),
                            page_key=ctx.project_checkpoint_page_key,
                            fingerprint=(
                                ctx.project_translation_fingerprint
                            ),
                            identity=ctx.project_translation_identity,
                            blocks=ctx.translation_blocks,
                        )
                    except Exception:
                        logger.warning(
                            "Translation checkpoint publication failed open "
                            "for %s.",
                            ctx.image_name,
                            exc_info=True,
                        )
                        stored = False
                    ctx.project_translation_checkpoint_status = (
                        "refreshed" if stored else "miss"
                    )
                self._emit_benchmark_event(
                    "translate_end",
                    image_path=ctx.image_path,
                    image_index=index,
                    total_images=total_images,
                    block_count=len(ctx.translation_blocks),
                    preserved_or_review_block_count=(
                        len(ctx.blk_list or [])
                        - len(ctx.translation_blocks)
                    ),
                    translator_key=translator_key,
                    translator_engine=translator.engine.__class__.__name__,
                    cache_status=translation_cache_status,
                    project_checkpoint_status=(
                        ctx.project_translation_checkpoint_status
                    ),
                    **ctx.page_translation_metrics,
                )
                raw_text_obj = json.loads(
                    get_raw_text(ctx.translation_blocks)
                )
                translated_text_obj = json.loads(
                    get_raw_translation(ctx.translation_blocks)
                )
                if (not raw_text_obj) or (not translated_text_obj):
                    raise RuntimeError("Translator returned empty JSON.")
                self._raise_if_cancelled()
            except OperationCancelledError:
                raise
            except Exception as exc:
                self._mark_page_failed(
                    ctx,
                    index=index,
                    total_images=total_images,
                    stage="translation",
                    reason=str(exc),
                    extra={**ctx.page_ocr_metrics, **ctx.page_translation_metrics},
                )

        if (
            gemma_runtime_started
            and isinstance(runtime_manager, LocalGemmaRuntimeManager)
        ):
            self._shutdown_runtime_with_retry(
                "Gemma",
                runtime_manager,
                context="translation-to-inpaint handoff",
                raise_on_failure=True,
                service="gemma",
            )
        self._raise_if_cancelled()

    @staticmethod
    def _render_settings_checkpoint_mapping(render_settings: Any) -> dict[str, Any]:
        values = dict(vars(render_settings))
        direction = values.get("direction")
        values["direction"] = int(
            getattr(direction, "value", direction or 0)
        )
        return values

    def _prepare_render_checkpoint(
        self,
        ctx: StagePageContext,
        *,
        render_settings: Any,
        export_settings: dict[str, Any],
        target_language_code: str,
    ) -> tuple[Any | None, str]:
        checkpoint_store = getattr(self, "_project_checkpoint_store", None)
        if (
            checkpoint_store is None
            or not ctx.project_translation_fingerprint
            or not ctx.project_inpaint_fingerprint
            or not ctx.project_inpaint_artifact_sha256
        ):
            ctx.project_render_checkpoint_status = "disabled"
            return None, ""
        try:
            anchor = (
                self.main_page.image_files[0]
                if self.main_page.image_files
                else ctx.image_path
            )
            output_base_root = (
                self.main_page.get_automatic_output_series_dir(
                    ctx.directory,
                    anchor_path=anchor,
                )
            )
            ctx.project_render_identity = build_render_identity(
                source_sha256=ctx.source_decoded_sha256,
                translation_fingerprint=(
                    ctx.project_translation_fingerprint
                ),
                inpaint_fingerprint=ctx.project_inpaint_fingerprint,
                inpaint_artifact_sha256=(
                    ctx.project_inpaint_artifact_sha256
                ),
                blocks=ctx.blk_list,
                render_settings=self._render_settings_checkpoint_mapping(
                    render_settings
                ),
                export_settings=export_settings,
                font_identity=resolve_font_identity(
                    self.main_page,
                    str(render_settings.font_family or ""),
                ),
                target_language_code=target_language_code,
                output_base_root=output_base_root,
            )
            ctx.project_render_fingerprint = build_render_fingerprint(
                ctx.project_render_identity
            )
            hit = lookup_render_checkpoint(
                checkpoint_store,
                page_key=ctx.project_checkpoint_page_key,
                fingerprint=ctx.project_render_fingerprint,
                identity=ctx.project_render_identity,
                project_blocks=ctx.project_render_blocks,
                project_viewer_state=ctx.project_viewer_state,
                current_output_base_root=output_base_root,
            )
            ctx.project_render_checkpoint_status = (
                "hit" if hit is not None else "miss"
            )
            return hit, output_base_root
        except Exception:
            logger.warning(
                "Render checkpoint lookup failed open for %s.",
                ctx.image_name,
                exc_info=True,
            )
            ctx.project_render_checkpoint_status = "miss"
            return None, ""

    def _restore_render_project_state(
        self,
        ctx: StagePageContext,
    ) -> None:
        ctx.blk_list = [
            block.deep_copy() for block in ctx.project_render_blocks
        ]
        page_state = self._ensure_page_state(ctx.image_path)
        viewer_state = page_state.setdefault("viewer_state", {})
        viewer_state["text_items_state"] = copy.deepcopy(
            ctx.project_viewer_state.get("text_items_state", [])
        )
        viewer_state["push_to_stack"] = True
        page_state["blk_list"] = ctx.blk_list
        self.main_page.image_ctrl.mark_processing_stage(
            ctx.image_path,
            "render",
            "completed",
            cache_status="project-checkpoint",
            text_item_count=len(viewer_state["text_items_state"]),
        )
        self.main_page.image_ctrl.mark_processing_stage(
            ctx.image_path,
            "pipeline",
            "completed",
        )
        self.main_page.render_state_ready.emit(ctx.image_path)

    def _lazy_render_context(self, ctx: StagePageContext) -> tuple[Any, str]:
        """렌더에 필요한 값들을 배치당 1회만 계산한다.

        `_inpaint_pages`가 페이지마다 즉시 렌더를 제출하므로, 이 계산은
        `main_page.render_settings()`를 실제로 요구하는 첫 페이지에서만
        일어난다 — 렌더까지 가지 않는 호출부(단위 테스트의 최소 더블 포함)는
        이 표면을 갖출 필요가 없다. 폰트 캐시 워밍도 여기, 첫 렌더 제출보다
        먼저 파이프라인 스레드에서 1회 끝낸다."""
        cached = getattr(self, "_render_context_cache", None)
        if cached is not None:
            return cached
        render_settings = self.main_page.render_settings()
        target_lang_en = self.main_page.lang_mapping.get(ctx.target_lang, ctx.target_lang)
        trg_lng_cd = get_language_code(target_lang_en)
        self._warm_render_font_caches(render_settings)
        cached = (render_settings, trg_lng_cd)
        self._render_context_cache = cached
        return cached

    def _submit_or_inline_render(
        self,
        ctx: StagePageContext,
        *,
        index: int,
        total_images: int,
        export_settings: dict[str, Any],
    ) -> None:
        """페이지 하나의 인페인팅이 막 끝났다. 렌더 체크포인트가 있으면 그
        자리에서(저렴하므로) 처리하고, 없으면 전용 단일 워커에 넘긴 뒤 곧바로
        반환한다 — 파이프라인 스레드는 다음 페이지 인페인팅으로 계속 간다."""
        if ctx.failed_stage:
            return
        self._raise_if_cancelled()
        render_settings, trg_lng_cd = self._lazy_render_context(ctx)
        self._set_current_image(ctx.image_path)
        self._write_json_exports(
            ctx.directory,
            ctx.export_token,
            ctx.archive_bname,
            ctx.image_path,
            ctx.image,
            ctx.blk_list,
            self._ensure_page_state(ctx.image_path),
            ctx.source_lang,
            export_settings,
        )
        self._emit_benchmark_event(
            "render_start",
            image_path=ctx.image_path,
            image_index=index,
            total_images=total_images,
            block_count=len(ctx.blk_list or []),
        )
        try:
            render_hit, _output_base_root = self._prepare_render_checkpoint(
                ctx,
                render_settings=render_settings,
                export_settings=export_settings,
                target_language_code=trg_lng_cd,
            )
            if render_hit is not None:
                try:
                    final_output_path = materialize_render_checkpoint_output(render_hit)
                except (OSError, ValueError):
                    logger.warning(
                        "Render checkpoint output materialization failed "
                        "open for %s; the page will be rendered normally.",
                        ctx.image_name,
                        exc_info=True,
                    )
                    ctx.project_render_checkpoint_status = "miss"
                    render_hit = None
                if render_hit is not None:
                    self._restore_render_project_state(ctx)
                    self._finish_render_checkpoint_hit(
                        ctx,
                        index=index,
                        total_images=total_images,
                        final_output_path=final_output_path,
                        final_output_root=render_hit.output_root,
                        output_materialized=not render_hit.output_exists,
                    )
                    return

            strict_render_symbols = should_use_strict_render_symbols(trg_lng_cd)
            output_path, output_root, output_format = self._reserve_render_output_path(
                ctx,
                export_settings=export_settings,
                page_index=index,
                total_pages=total_images,
            )
            page_state = self._ensure_page_state(ctx.image_path)
            file_on_display = (
                0 <= self.main_page.curr_img_idx < len(self.main_page.image_files)
                and self.main_page.image_files[self.main_page.curr_img_idx] == ctx.image_path
            )
            alignment = self.main_page.button_to_alignment.get(
                1,
                self.main_page.button_to_alignment[render_settings.alignment_id],
            )
            vertical_alignment = self.main_page.button_to_vertical_alignment.get(
                1,
                VERTICAL_ALIGNMENT_CENTER,
            )
            job = RenderJobInput(
                image_path=ctx.image_path,
                image=ctx.image,
                inpaint_input_img=ctx.inpaint_input_img,
                mask=ctx.mask,
                patches=ctx.patches,
                blk_list=ctx.blk_list,
                translation_blocks=ctx.translation_blocks,
                no_text_detected=ctx.no_text_detected,
                trg_lng_cd=trg_lng_cd,
                render_settings=render_settings,
                strict_render_symbols=strict_render_symbols,
                alignment=alignment,
                vertical_alignment=vertical_alignment,
                viewer_state=dict(page_state.get("viewer_state", {})),
                output_path=output_path,
                output_format=output_format,
                is_cancelled=self._render_cancel_event.is_set,
                submitted_monotonic=time.monotonic(),
            )
            self._raise_if_cancelled()
            future = self._ensure_render_executor().submit(run_render_job, job)
            self._pending_render_jobs.append(
                _PendingRenderJob(
                    future=future,
                    ctx=ctx,
                    index=index,
                    total_images=total_images,
                    file_on_display=file_on_display,
                    output_root=output_root,
                    started_monotonic=time.monotonic(),
                )
            )
        except OperationCancelledError:
            raise
        except Exception as exc:
            self._mark_page_failed(
                ctx,
                index=index,
                total_images=total_images,
                stage="render",
                reason=str(exc),
                extra={**ctx.page_ocr_metrics, **ctx.page_translation_metrics},
            )

    def _finish_render_checkpoint_hit(
        self,
        ctx: StagePageContext,
        *,
        index: int,
        total_images: int,
        final_output_path: str,
        final_output_root: str,
        output_materialized: bool,
    ) -> None:
        ctx.output_path = final_output_path
        self.main_page.image_ctrl.update_processing_summary(
            ctx.image_path,
            {
                "translated_image_path": final_output_path,
                "translated_page_image_path": final_output_path,
                "export_root": final_output_root,
                "render_project_checkpoint_status": "hit",
                "render_output_materialized": output_materialized,
            },
        )
        self._emit_benchmark_event(
            "render_end",
            image_path=ctx.image_path,
            image_index=index,
            total_images=total_images,
            block_count=len(ctx.blk_list or []),
            translated_image_path=final_output_path,
            project_checkpoint_status="hit",
            render_skipped=True,
            output_materialized=output_materialized,
        )
        self._emit_benchmark_event(
            "page_done",
            image_path=ctx.image_path,
            image_index=index,
            total_images=total_images,
            block_count=len(ctx.blk_list or []),
            patch_count=len(ctx.patches or []),
            project_checkpoint_status="hit",
        )
        self._log_page_done(index, total_images, ctx.image_path, preview_path=final_output_path)
        self.emit_progress(index, total_images, 9, 10, False, stage_name='render-all')
        self._raise_if_cancelled()

    def _finish_render_page_bookkeeping(
        self,
        pending: _PendingRenderJob,
        *,
        result: RenderJobResult | None,
        exc: BaseException | None,
    ) -> None:
        """렌더 워커가 끝낸 작업 하나의 비순수 후처리를 파이프라인 스레드에서
        수행한다 — 구 `_render_all`이 하던 시그널 발신·page_state 기록·체크포인트
        기록이 전부 여기 모인다."""
        ctx = pending.ctx
        index = pending.index
        total_images = pending.total_images
        if exc is not None:
            self._mark_page_failed(
                ctx,
                index=index,
                total_images=total_images,
                stage="render",
                reason=str(exc),
                extra={**ctx.page_ocr_metrics, **ctx.page_translation_metrics},
            )
            return

        assert result is not None
        page_state = self._ensure_page_state(ctx.image_path)
        page_state.setdefault("viewer_state", {}).update(result.viewer_state)
        page_state["blk_list"] = ctx.blk_list
        if pending.file_on_display:
            for translation, font_size, blk in result.blk_rendered_events:
                self.main_page.blk_rendered.emit(translation, font_size, blk, ctx.image_path)
        self.main_page.image_ctrl.mark_processing_stage(
            ctx.image_path,
            "render",
            "completed",
            text_item_count=len(result.viewer_state.get("text_items_state", [])),
        )
        self.main_page.image_ctrl.mark_processing_stage(ctx.image_path, "pipeline", "completed")
        self.main_page.render_state_ready.emit(ctx.image_path)

        ctx.inpaint_input_img = result.inpaint_input_img
        ctx.mask = result.mask
        ctx.patches = result.patches
        if result.restore_applied:
            ctx.cleanup_stats = dict(ctx.cleanup_stats or {})
            ctx.cleanup_stats["render_restore"] = result.restore_stats
            self.main_page.patches_processed.emit(ctx.patches, ctx.image_path)
            self.main_page.image_ctrl.update_processing_summary(
                ctx.image_path,
                {
                    "render_restore_block_count": int(result.restore_stats.get("block_count", 0) or 0),
                    "render_restore_pixel_count": int(result.restore_stats.get("pixel_count", 0) or 0),
                },
            )

        # 렌더는 전용 워커에서 돈다. 파이프라인 스레드는 제출만 하고 떠나므로,
        # 워커가 돌려준 실측값을 여기서 텔레메트리에 넣지 않으면 렌더 비용이
        # 어디에도 남지 않는다. 실제로 그래서 렌더가 페이지당 2.31초를 쓰는데도
        # 리포트에서는 보이지 않았다.
        self._record_performance_detail(
            stage="render",
            operation="worker",
            elapsed_ms=float(result.worker_seconds) * 1000.0,
        )
        self._record_performance_detail(
            stage="render",
            operation="queue_wait",
            elapsed_ms=float(result.queue_wait_seconds) * 1000.0,
        )
        final_output_path = result.final_output_path
        final_output_root = pending.output_root
        ctx.output_path = final_output_path
        if ctx.project_render_identity:
            try:
                stored = record_render_checkpoint(
                    getattr(self, "_project_checkpoint_store", None),
                    page_key=ctx.project_checkpoint_page_key,
                    fingerprint=ctx.project_render_fingerprint,
                    identity=ctx.project_render_identity,
                    blocks=ctx.blk_list,
                    viewer_state=page_state.get("viewer_state", {}),
                    output_path=final_output_path,
                    output_root=final_output_root,
                )
            except Exception:
                logger.warning(
                    "Render checkpoint publication failed open for %s.",
                    ctx.image_name,
                    exc_info=True,
                )
                stored = False
            ctx.project_render_checkpoint_status = "refreshed" if stored else "miss"

        self.main_page.image_ctrl.update_processing_summary(
            ctx.image_path,
            {
                "translated_image_path": final_output_path,
                "translated_page_image_path": final_output_path,
                "export_root": final_output_root,
                "render_project_checkpoint_status": ctx.project_render_checkpoint_status,
                **({"skip_reason": "no_text_detected"} if ctx.no_text_detected else {}),
            },
        )
        self._emit_benchmark_event(
            "render_end",
            image_path=ctx.image_path,
            image_index=index,
            total_images=total_images,
            block_count=len(ctx.blk_list or []),
            translated_image_path=final_output_path,
            project_checkpoint_status=ctx.project_render_checkpoint_status,
        )
        self._emit_benchmark_event(
            "page_done",
            image_path=ctx.image_path,
            image_index=index,
            total_images=total_images,
            block_count=len(ctx.blk_list or []),
            patch_count=len(ctx.patches or []),
            **({"skip_reason": "no_text_detected"} if ctx.no_text_detected else {}),
        )
        self._log_page_done(index, total_images, ctx.image_path, preview_path=final_output_path)
        self.emit_progress(index, total_images, 9, 10, False, stage_name='render-all')
        self._release_page_buffers(ctx)

    def _release_page_buffers(self, ctx: StagePageContext) -> None:
        """끝난 페이지의 전체 해상도 배열을 놓아주고 그 사실을 기록한다."""

        released = ctx.release_page_buffers()
        if released <= 0:
            return
        self._released_page_buffer_bytes += released
        logger.debug(
            "Released %.1f MiB of page buffers for %s (run total %.1f MiB).",
            released / (1024 * 1024),
            ctx.image_name,
            self._released_page_buffer_bytes / (1024 * 1024),
        )

    def _resolve_render_future(self, pending: _PendingRenderJob) -> None:
        try:
            result = pending.future.result()
        except OperationCancelledError:
            raise
        except Exception as exc:
            self._finish_render_page_bookkeeping(pending, result=None, exc=exc)
            return
        self._finish_render_page_bookkeeping(pending, result=result, exc=None)

    def _drain_render_futures(self, *, block: bool) -> None:
        """제출된 렌더 작업 중 끝난 것을 후처리한다.

        `block=False`는 인페인팅 루프의 기존 취소 체크 지점마다 얹혀 논블로킹
        으로 흘려보낸다. `block=True`(`_render_all`의 최종 드레인)는
        `modules/ocr/paddle_crop/engine.py:320-326` 관용구 그대로: 매 틱마다
        취소를 확인하며 남은 작업이 없어질 때까지 기다린다."""
        # `object.__new__`로 `__init__`을 거치지 않고 만들어진 인스턴스(단위
        # 테스트의 최소 더블)에서도 안전하도록 속성 존재를 가정하지 않는다.
        if not getattr(self, "_pending_render_jobs", None):
            return
        pending_by_future = {p.future: p for p in self._pending_render_jobs}
        if not block:
            done, _ = wait(set(pending_by_future), timeout=0)
            for future in done:
                pending = pending_by_future[future]
                self._pending_render_jobs.remove(pending)
                self._resolve_render_future(pending)
            return

        remaining = set(pending_by_future)
        while remaining:
            self._raise_if_cancelled()
            done, remaining = wait(remaining, timeout=0.2, return_when=FIRST_COMPLETED)
            for future in done:
                pending = pending_by_future[future]
                self._pending_render_jobs.remove(pending)
                self._resolve_render_future(pending)
        self._raise_if_cancelled()

    def _fallback_page_image(self, ctx: StagePageContext):
        """폴백 저장에 쓸 최선의 이미지. 없으면 ``None``.

        인페인팅까지 갔다면 글자가 지워진 결과가 원본보다 낫다. 그마저 없으면
        원본이다. 버퍼는 이미 놓아준 뒤일 수 있으므로 그때는 디스크에서 다시
        읽는다. 폴백 대상은 소수라 다시 읽는 비용이 메모리를 붙들고 있는 것보다
        훨씬 싸다.
        """

        for candidate, kind in (
            (ctx.inpaint_input_img, "inpainted"),
            (ctx.image, "source"),
        ):
            if candidate is not None:
                return candidate, kind
        try:
            reloaded = self.main_page.image_ctrl.load_image(ctx.image_path)
            if reloaded is None:
                reloaded = imk.read_image(ctx.image_path)
        except Exception:
            logger.warning(
                "Could not reload %s for the fallback export.",
                ctx.image_name,
                exc_info=True,
            )
            return None, ""
        if reloaded is None:
            return None, ""
        return reloaded, "source-reloaded"

    def _write_fallback_export(
        self,
        ctx: StagePageContext,
        *,
        index: int,
        total_images: int,
        export_settings: dict[str, Any],
    ) -> bool:
        """렌더까지 가지 못한 페이지도 파일 하나를 반드시 남긴다.

        한 스테이지의 실패로 페이지가 출력에서 통째로 사라지던 동작을 막는다.
        실측으로 366장 입력에서 347장만 나온 적이 있고, 사라진 19장에는 아무런
        흔적도 남지 않았다. 품질이 떨어지는 것과 결과가 없는 것은 전혀 다르다.

        정상 렌더와 똑같은 내보내기 경로를 쓴다. 아카이브 모드에서 페이지 순서와
        파일명 규칙이 어긋나면 안 되기 때문이다.
        """

        image, kind = self._fallback_page_image(ctx)
        if image is None:
            logger.error(
                "No image available for the fallback export of %s.",
                ctx.image_name,
            )
            return False
        try:
            output_path, output_root = self._write_final_render_export(
                ctx.directory,
                ctx.export_token,
                ctx.image_path,
                image,
                # 폴백은 번역 텍스트를 얹지 않는다. 얹을 만한 상태였다면 정상
                # 렌더가 이미 성공했을 것이다.
                [],
                {},
                export_settings,
                page_index=index,
                total_pages=total_images,
            )
        except Exception:
            logger.error(
                "Fallback export failed for %s.",
                ctx.image_name,
                exc_info=True,
            )
            return False
        ctx.output_path = output_path
        ctx.output_fallback_kind = kind
        self.main_page.image_ctrl.update_processing_summary(
            ctx.image_path,
            {
                "translated_image_path": output_path,
                "translated_page_image_path": output_path,
                "export_root": output_root,
                "output_fallback_kind": kind,
                "output_fallback_reason": ctx.failed_reason or ctx.failed_stage,
            },
        )
        self._emit_benchmark_event(
            "page_output_fallback",
            image_path=ctx.image_path,
            image_index=index,
            total_images=total_images,
            fallback_kind=kind,
            failed_stage=ctx.failed_stage,
            reason=ctx.failed_reason,
            translated_image_path=output_path,
        )
        logger.warning(
            "Exported %s from its %s image: %s",
            ctx.image_name,
            kind,
            self._fallback_cause(ctx),
        )
        return True

    @staticmethod
    def _fallback_cause(ctx: StagePageContext) -> str:
        """폴백을 하게 된 이유를 사실대로 적는다.

        스테이지 실패가 기록돼 있으면 그것이 이유다. 그렇지 않은데도 출력이 없다면
        원인은 다른 곳이다 — 실측으로 366장 중 15장이 인페인팅까지 끝나고도 렌더
        결과가 기록되지 않은 채 조용히 빠졌고, 실패로 표시되지도 않았다. 그때
        "unknown stage failed" 라고 적으면 없는 실패를 지어내는 셈이 된다.
        """

        if ctx.failed_stage:
            return (
                f"the {ctx.failed_stage} stage failed: "
                f"{ctx.failed_reason or '(no reason recorded)'}"
            )
        return "no rendered output was recorded for it"

    def _reconcile_page_outputs(
        self,
        pages: list[StagePageContext],
        *,
        export_settings: dict[str, Any],
    ) -> dict[str, Any]:
        """모든 페이지가 파일을 남겼는지 확인하고, 없으면 폴백으로 채운다.

        배치가 끝나는 시점에 입력 수와 출력 수가 같아야 한다. 어긋나면 조용히
        넘어가지 않고 배치 리포트에 올린다.
        """

        total_images = len(pages)
        fallbacks: list[dict[str, str]] = []
        unrecoverable: list[str] = []
        for index, ctx in enumerate(pages):
            # 내부 기록만 믿지 않고 파일이 실제로 있는지 본다. 어느 한 경로가
            # 출력 기록을 빠뜨려도(실측으로 그런 페이지가 15장 있었다) 여기서
            # 잡힌다. 디스크에 있는 파일이 유일한 진실이다.
            if ctx.output_path and os.path.exists(ctx.output_path):
                continue
            if ctx.output_path:
                logger.warning(
                    "%s recorded an output that is not on disk: %s",
                    ctx.image_name,
                    ctx.output_path,
                )
                ctx.output_path = ""
            if self._write_fallback_export(
                ctx,
                index=index,
                total_images=total_images,
                export_settings=export_settings,
            ):
                fallbacks.append(
                    {
                        "image_name": ctx.image_name,
                        "kind": ctx.output_fallback_kind,
                        "failed_stage": ctx.failed_stage,
                        "reason": ctx.failed_reason,
                        "cause": self._fallback_cause(ctx),
                    }
                )
            else:
                unrecoverable.append(ctx.image_name)
            self._release_page_buffers(ctx)

        produced = sum(1 for ctx in pages if ctx.output_path)
        summary = {
            "input_count": total_images,
            "output_count": produced,
            "fallback_count": len(fallbacks),
            "fallbacks": fallbacks,
            "missing": unrecoverable,
        }
        self._emit_benchmark_event(
            "batch_output_reconciled",
            total_images=total_images,
            output_count=produced,
            fallback_count=len(fallbacks),
            missing_count=len(unrecoverable),
        )
        if unrecoverable:
            logger.error(
                "Batch produced %d outputs for %d input pages. Missing: %s",
                produced,
                total_images,
                ", ".join(unrecoverable),
            )
            report = getattr(self.main_page, "batch_report_ctrl", None)
            register = getattr(report, "register_preflight_error", None)
            if callable(register):
                try:
                    register(
                        self._stage_tr("출력 페이지 수가 입력과 다릅니다."),
                        self._stage_tr(
                            "입력 {input}장 중 {output}장만 저장되었습니다. "
                            "누락: {missing}"
                        )
                        .replace("{input}", str(total_images))
                        .replace("{output}", str(produced))
                        .replace("{missing}", ", ".join(unrecoverable)),
                    )
                except Exception:
                    logger.debug(
                        "Could not register the output reconciliation error.",
                        exc_info=True,
                    )
        elif fallbacks:
            logger.warning(
                "Batch produced all %d outputs, but %d came from a fallback image.",
                total_images,
                len(fallbacks),
            )
        return summary

    def _run_started_wall_text(self) -> str:
        """이번 실행이 시작된 벽시계 시각. 사용자가 버튼을 누른 시점이다."""

        tracker = getattr(self.main_page, "_automatic_progress_tracker", None)
        started = getattr(tracker, "run_started_wall", None)
        if isinstance(started, str) and started:
            return started
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _click_to_now_seconds(self) -> float:
        """사용자가 '모두 번역'을 누른 순간부터 지금까지의 실측 경과 시간.

        파이프라인 자체 시작 시각이 아니라 클릭 시점을 기준으로 한다. 런타임
        준비(컨테이너 기동, 모델 적재)도 사용자가 기다린 시간이기 때문이다.
        """

        tracker = getattr(self.main_page, "_automatic_progress_tracker", None)
        started = getattr(tracker, "run_started_at", None)
        if isinstance(started, (int, float)):
            return max(time.monotonic() - float(started), 0.0)
        run_started = getattr(self, "_run_started_at", None)
        if isinstance(run_started, (int, float)):
            return max(time.monotonic() - float(run_started), 0.0)
        return 0.0

    def _page_outcomes(self, pages: list[StagePageContext]) -> list[dict[str, Any]]:
        return [
            {
                "image_name": ctx.image_name,
                "output_path": ctx.output_path,
                "fallback_kind": ctx.output_fallback_kind,
                "failed_stage": ctx.failed_stage,
                "failed_reason": ctx.failed_reason,
                "no_text_detected": bool(ctx.no_text_detected),
                "block_count": len(ctx.blk_list or []),
            }
            for ctx in pages
        ]

    def _write_run_report(
        self,
        pages: list[StagePageContext],
        *,
        output_summary: dict[str, Any],
    ) -> str:
        """이번 실행의 실측 소요시간과 페이지별 결과를 파일로 남긴다."""

        try:
            telemetry = self._performance_telemetry().snapshot()
        except Exception:
            logger.debug("Could not snapshot the run telemetry.", exc_info=True)
            telemetry = {}
        report = build_run_report(
            telemetry=telemetry,
            total_wall_sec=self._click_to_now_seconds(),
            page_outcomes=self._page_outcomes(pages),
            output_summary=output_summary,
            started_at_local=self._run_started_wall_text(),
        )
        try:
            from modules.utils.paths import get_log_dir

            log_dir = get_log_dir("runs")
        except Exception:
            logger.debug("Could not resolve the run report directory.", exc_info=True)
            return ""
        path = write_run_report(report, log_dir=log_dir)
        if path:
            logger.info(
                "Run finished in %s for %d page(s); report at %s",
                report.get("total_wall_text", "?"),
                int(report.get("page_count", 0) or 0),
                path,
            )
        return path

    def _render_all(self, pages: list[StagePageContext]) -> None:
        # 대부분의 렌더는 이미 인페인팅 sweep 동안 전용 워커에서 끝나 있다
        # (`_submit_or_inline_render`가 `_inpaint_pages`에서 페이지별로 제출).
        # 여기 남는 시간이 곧 **융합이 숨기지 못한 렌더 잔량**이다. 이상적으로는
        # 마지막 한두 페이지분이지만, 렌더가 인페인팅보다 느려지면 큐가 쌓여
        # 이 값이 커진다. 그때는 인페인팅을 더 줄여도 전체 시간이 줄지 않는다.
        pending = len(getattr(self, "_pending_render_jobs", []) or [])
        started_at = time.monotonic()
        try:
            self._drain_render_futures(block=True)
        finally:
            elapsed = max(0.0, time.monotonic() - started_at)
            self._record_performance_detail(
                stage="render",
                operation="tail_drain",
                elapsed_ms=elapsed * 1000.0,
                workload={"pending_at_drain": pending},
            )
            if elapsed > 1.0:
                logger.info(
                    "Render tail drain took %.1fs with %d job(s) still queued; "
                    "that is render time the inpaint sweep could not hide.",
                    elapsed,
                    pending,
                )

    def batch_process(self, selected_paths: list[str] | None = None):
        image_list = selected_paths if selected_paths is not None else self.main_page.image_files
        total_images = len(image_list)
        benchmark_stage_ceiling = self._benchmark_stage_ceiling()
        reset_output_reservations = getattr(self.main_page, "reset_automatic_output_reservations", None)
        if callable(reset_output_reservations):
            reset_output_reservations()
        self._run_started_at = time.monotonic()
        self._page_started_at = None
        self._progress_image_path = None
        self._recent_page_durations.clear()
        self._released_page_buffer_bytes = 0
        self._paddleocr_cache_store = None
        self._paddleocr_cache_identity = None
        self._render_context_cache = None
        self._emit_benchmark_event("batch_run_start", total_images=total_images)
        self._record_performance_workload(
            "pipeline",
            page_count=total_images,
            workflow_mode="stage_batched",
        )
        self._reset_prewarm_lifecycle()
        pdf_preflight = getattr(
            self.main_page.file_handler, "preflight_for_processing", None
        )
        pdf_count = (
            pdf_preflight(
                image_list,
                should_cancel=lambda: bool(
                    getattr(self.main_page, "is_current_task_cancelled", lambda: False)()
                ),
            )
            if callable(pdf_preflight)
            else 0
        )
        if pdf_count:
            logger.info(
                "Validated %d PDF-backed pages before stage-batched processing.",
                pdf_count,
            )
            warnings = self.main_page.file_handler.get_pdf_import_warnings(image_list)
            if warnings:
                pages = ", ".join(str(item["page_number"]) for item in warnings)
                sizes = "; ".join(
                    "{page_number}: {requested_width}×{requested_height} → "
                    "{applied_width}×{applied_height}".format(**item)
                    for item in warnings
                )
                self.main_page.batch_report_ctrl.register_preflight_warning(
                    QCoreApplication.translate(
                        "PdfImport", "PDF import memory limit applied"
                    ),
                    QCoreApplication.translate(
                        "PdfImport", "Pages: {pages}. Requested/applied sizes: {sizes}."
                    ).replace("{pages}", pages).replace("{sizes}", sizes),
                )
        try:
            with self._measure_performance(
                stage="pipeline",
                operation="pre_materialize",
                workload={"page_count": total_images},
            ):
                if self.main_page.file_handler.should_pre_materialize(image_list):
                    self.main_page.file_handler.pre_materialize(image_list)
        except Exception:
            logger.debug("Stage-batched pre-materialization failed; continuing lazily.", exc_info=True)

        with self._measure_performance(
            stage="pipeline",
            operation="prepare_work_context",
            workload={"page_count": total_images},
            node_id="pipeline.prepare",
        ):
            pages = self._load_page_contexts(image_list)
            policy = self._ensure_stage_policy(pages)
            self._project_checkpoint_store = open_project_stage_checkpoint_store(
                self.main_page,
                initialize=True,
            )
        self._project_checkpoint_page_keys = (
            [
                project_checkpoint_page_key(self.main_page, image_path)
                for image_path in image_list
            ]
            if self._project_checkpoint_store is not None
            else []
        )
        batch_completed = False
        try:
            self._raise_if_cancelled()
            with self._measure_performance(
                stage="pipeline",
                operation="runtime_startup_preflight",
                node_id="pipeline.runtime_preflight",
                dependencies=("pipeline.prepare",),
            ):
                self._shutdown_managed_runtimes(
                    context="batch startup preflight",
                    raise_on_failure=True,
                    preserve_sleeping_paddle=True,
                    ocr_service=self._ocr_runtime_service_name(
                        str(policy.get("primary_ocr_engine", ""))
                    ),
                )
            self._raise_if_cancelled()
            # 모델은 올리지 않고 OCR Router 컨테이너만 미리 띄운다. 검출 sweep 과
            # 겹치며, `--no-models-autoload` 라서 모델 적재 baseline 을 오염시키지
            # 않는다. 컨테이너 기동 6~8초를 검출 뒤에서 앞으로 당긴다.
            self._start_ocr_container_prewarm(policy)
            with self._measure_performance(
                stage="detect",
                operation="stage_window",
                workload={"page_count": total_images},
                node_id="stage.detect",
                dependencies=("pipeline.runtime_preflight",),
                service="detector",
            ):
                self._detect_all(pages, policy)
            source_pixels = 0
            for ctx in pages:
                image = getattr(ctx, "image", None)
                if image is None:
                    continue
                image_array = np.asarray(image)
                if image_array.ndim >= 2:
                    source_pixels += int(image_array.shape[0]) * int(
                        image_array.shape[1]
                    )
            detected_blocks = sum(
                len(getattr(ctx, "blk_list", []) or []) for ctx in pages
            )
            self._record_performance_workload(
                "detect",
                page_count=total_images,
                source_megapixels=source_pixels / 1_000_000.0,
                detected_block_count=detected_blocks,
            )
            self._sample_performance_resources("detect_stage_end")
            self._raise_if_cancelled()
            # GPU detection can keep an ONNX session resident.  Delay model
            # prewarm until it completes so the model-start baseline and the
            # exclusive runtime handoff are not contaminated by concurrent
            # detector inference.
            # 컨테이너 기동이 끝난 뒤에 모델 적재로 넘어간다. 같은 코디네이터를 두
            # 스레드가 동시에 만지지 않게 한다.
            self._await_ocr_container_prewarm()
            self._start_ocr_prewarm(policy)
            # Gemma GGUF 를 페이지 캐시로 끌어올리는 작업을 여기서 띄운다. OCR sweep
            # 과 겹치며, 디스크에서 RAM 으로만 옮기므로 OCR 이 쥔 VRAM 과 다투지
            # 않는다. 번역 단계에서 mmap 이 전부 캐시 적중이 되어 첫 실행 적재가
            # 44초에서 재실행 수준으로 내려간다.
            self._start_gemma_page_cache_prefetch()
            self._raise_if_cancelled()
            with self._measure_performance(
                stage="ocr",
                operation="stage_window",
                workload={
                    "page_count": total_images,
                    "detected_block_count": detected_blocks,
                },
                node_id="stage.ocr",
                dependencies=("stage.detect",),
                service=self._ocr_runtime_service_name(
                    str(policy.get("primary_ocr_engine", ""))
                ),
            ):
                self._ocr_all(pages, policy)
            ocr_blocks = sum(
                len(getattr(ctx, "blk_list", []) or []) for ctx in pages
            )
            self._record_performance_workload(
                "ocr",
                page_count=total_images,
                block_count=ocr_blocks,
                runtime_required=bool(ocr_blocks),
            )
            self._sample_performance_resources("ocr_stage_end")
            self._raise_if_cancelled()
            if benchmark_stage_ceiling == "ocr":
                self._complete_ocr_stage_ceiling(pages)
                self._emit_benchmark_event(
                    "batch_run_done",
                    total_images=total_images,
                    stage_ceiling="ocr",
                )
                batch_completed = True
                return
            for ctx in pages:
                if getattr(ctx, "failed_stage", "") or getattr(
                    ctx,
                    "no_text_detected",
                    False,
                ):
                    ctx.translation_blocks = []
                    continue
                ctx.translation_blocks = select_translate_inpaint_blocks(
                    getattr(ctx, "blk_list", []) or []
                )
            translation_blocks = sum(
                len(getattr(ctx, "translation_blocks", []) or [])
                for ctx in pages
            )
            source_character_count = 0
            for ctx in pages:
                for block in getattr(ctx, "translation_blocks", []) or []:
                    block_text = getattr(block, "text", "") or ""
                    if isinstance(block_text, (list, tuple)):
                        source_character_count += sum(
                            len(str(item or "")) for item in block_text
                        )
                    else:
                        source_character_count += len(str(block_text))
            with self._measure_performance(
                stage="translate",
                operation="stage_window",
                workload={
                    "page_count": total_images,
                    "block_count": translation_blocks,
                    "source_character_count": source_character_count,
                },
                node_id="stage.translate",
                dependencies=("stage.ocr",),
                service="gemma",
            ):
                self._translate_all(pages)
            self._record_performance_workload(
                "translate",
                page_count=total_images,
                block_count=translation_blocks,
                source_character_count=source_character_count,
            )
            self._sample_performance_resources("translate_stage_end")
            self._raise_if_cancelled()
            # 번역이 끝나면 Router 컨테이너를 완전히 정지한다. 예전에는 OCR sweep
            # 뒤에도 컨테이너를 살려둔 채 인페인팅 sweep 전체(실측 467초)를 지나서,
            # 아무 모델도 없는 컨테이너가 CUDA 컨텍스트로 약 278 MiB 를 붙들고
            # 있었다. LaMa 가 그만큼 못 쓴다.
            self._release_gemma_before_inpainter()
            # Phase 3a: 인페인팅+렌더 융합 sweep 동안 페이지별 자동저장을
            # 억제하고 sweep 종료 시 1회로 합친다 (finally 블록에서 해제).
            project_ctrl = getattr(self.main_page, "project_ctrl", None)
            if project_ctrl is not None:
                project_ctrl.begin_batch_autosave_deferral()
            with self._measure_performance(
                stage="inpaint",
                operation="stage_window",
                workload={"page_count": total_images, "block_count": ocr_blocks},
                node_id="stage.inpaint",
                dependencies=("stage.translate",),
                service="inpainter",
            ):
                self._inpaint_all(pages)
            # 마스크는 페이지가 끝나는 즉시 놓아준다. 집계는 놓아주기 전에 옮겨
            # 담아둔 값을 쓰고, 아직 살아 있는 페이지만 그 자리에서 센다.
            mask_pixels = sum(
                int(ctx.mask_pixel_count)
                if getattr(ctx, "mask", None) is None
                else int(np.count_nonzero(ctx.mask))
                for ctx in pages
            )
            inpaint_roi_count = sum(
                len(
                    list(
                        dict(getattr(ctx, "inpaint_diagnostics", {}) or {}).get(
                            "model_call_diagnostics",
                            [],
                        )
                        or []
                    )
                )
                for ctx in pages
            )
            self._record_performance_workload(
                "inpaint",
                page_count=total_images,
                mask_pixel_count=mask_pixels,
                roi_count=inpaint_roi_count,
            )
            self._sample_performance_resources("inpaint_stage_end")
            self._raise_if_cancelled()
            with self._measure_performance(
                stage="render",
                operation="stage_window",
                workload={
                    "page_count": total_images,
                    "block_count": ocr_blocks,
                    "source_megapixels": source_pixels / 1_000_000.0,
                },
                node_id="stage.render",
                dependencies=("stage.inpaint",),
                service="cpu_render",
            ):
                self._render_all(pages)
            self._record_performance_workload(
                "render",
                page_count=total_images,
                block_count=ocr_blocks,
                source_megapixels=source_pixels / 1_000_000.0,
            )
            self._sample_performance_resources("render_stage_end")
            # 한 스테이지의 실패로 페이지가 출력에서 사라지지 않게, 아직 파일을
            # 남기지 못한 페이지를 여기서 마지막으로 채운다.
            output_summary = self._reconcile_page_outputs(
                pages,
                export_settings=self._effective_export_settings(
                    self.main_page.settings_page
                ),
            )
            self._emit_benchmark_event(
                "batch_run_done",
                total_images=total_images,
                output_count=int(output_summary.get("output_count", 0)),
                fallback_count=int(output_summary.get("fallback_count", 0)),
            )
            self._write_run_report(pages, output_summary=output_summary)
            batch_completed = True
        except OperationCancelledError:
            self._emit_benchmark_event("batch_run_cancelled", total_images=total_images)
            return
        except Exception as exc:
            self._emit_benchmark_event(
                "batch_run_failed",
                total_images=total_images,
                reason=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            self._persist_stage_rates()
            self._shutdown_page_cache_executor()
            self._shutdown_render_executor()
            try:
                project_ctrl = getattr(self.main_page, "project_ctrl", None)
                if project_ctrl is not None:
                    project_ctrl.end_batch_autosave_deferral(
                        pages[-1].image_path if pages else ""
                    )
            except Exception:
                logger.warning(
                    "Sweep 종료 자동저장 트리거 중 예외가 발생했지만 무시합니다.",
                    exc_info=True,
                )
            try:
                self._shutdown_prewarm_executor()
            finally:
                self._shutdown_managed_runtimes(
                    preserve_sleeping_paddle=batch_completed,
                    ocr_service=self._ocr_runtime_service_name(
                        str(policy.get("primary_ocr_engine", ""))
                    ),
                )
            self._progress_image_path = None

    def _complete_ocr_stage_ceiling(self, pages: list[StagePageContext]) -> None:
        total_images = len(pages)
        for index, ctx in enumerate(pages):
            if ctx.failed_stage:
                continue
            self.main_page.image_ctrl.update_processing_summary(
                ctx.image_path,
                {"benchmark_stage_ceiling": "ocr"},
            )
            self._emit_benchmark_event(
                "page_done",
                image_path=ctx.image_path,
                image_index=index,
                total_images=total_images,
                block_count=len(ctx.blk_list or []),
                patch_count=0,
                stage_ceiling="ocr",
                **ctx.page_ocr_metrics,
            )
            self._log_page_done(index, total_images, ctx.image_path)
            self.emit_progress(index, total_images, 10, 10, False, stage_name='save-and-finish')
