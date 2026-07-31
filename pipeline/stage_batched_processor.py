from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import imkit as imk
import numpy as np
from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QColor

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
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.text_item import OutlineInfo, OutlineType
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
    PROCESSING_ACTION_TRANSLATE_INPAINT,
    canonicalize_exact_duplicate_blocks,
    finalize_ocr_processing_contract,
    finalize_ocr_processing_contracts,
    select_translate_inpaint_blocks,
)
from modules.ocr.selection import (
    STAGE_BATCHED_WORKFLOW_MODE,
    resolve_stage_batched_ocr_policy,
)
from modules.rendering.render import (
    apply_strict_render_viewer_state_guard,
    build_duplicate_bubble_render_key,
    build_render_rects_for_block,
    build_text_item_layout_geometry,
    describe_auto_render_review_status_gate,
    describe_render_text_markup,
    describe_render_text_sanitization,
    describe_text_free_large_mask_gate,
    describe_text_free_render_mask_gate,
    describe_text_free_render_translation_gate,
    describe_text_free_underfill_gate,
    get_best_render_area,
    get_render_fit_clearance_for_block,
    is_vertical_block,
    pyside_word_wrap,
    refit_detected_bubble_text_if_underfilled,
    register_duplicate_bubble_render_key,
    resolve_text_free_manga_layout,
    select_blocks_for_original_restore_after_render,
    should_skip_short_render_translation,
    should_use_strict_render_symbols,
)
from modules.translation.local_runtime import LocalGemmaRuntimeManager
from modules.translation.processor import Translator
from modules.utils.correction_dictionary import (
    apply_ocr_result_dictionary,
    apply_translation_result_dictionary,
)
from modules.utils.device import resolve_device
from modules.utils.export_paths import (
    build_export_timestamp,
    export_run_root,
    resolve_export_directory,
)
from modules.utils.exceptions import OperationCancelledError
from modules.utils.image_utils import generate_mask, restore_original_for_block_masks
from modules.utils.inpaint_cleanup import apply_duplicate_bubble_inner_fill, refine_bubble_residue_inpaint
from modules.utils.inpaint_composite import (
    composite_with_edit_mask,
    count_changed_outside_edit_mask,
)
from modules.inpainting.runtime_contract import inpaint_outside_mask_message
from modules.utils.language_utils import get_language_code, is_no_space_lang, language_codes
from modules.utils.ocr_debug import (
    all_empty_blocks_are_rejected,
    drop_embedded_ui_ocr_blocks,
    drop_rejected_empty_ocr_blocks,
    is_block_ocr_empty,
    is_bubble_panel_text_candidate,
    is_embedded_ui_panel_layout_review_candidate,
    split_inpaint_protected_ocr_blocks,
)
from modules.utils.ocr_quality import summarize_ocr_quality
from modules.utils.pipeline_config import get_config, get_inpainter_runtime, inpaint_map
from modules.utils.render_style_policy import (
    VERTICAL_ALIGNMENT_CENTER,
    resolve_render_text_color,
)
from modules.utils.textblock import ensure_text_block_id, sort_blk_list
from modules.utils.translator_utils import (
    format_translations,
    get_raw_text,
    get_raw_translation,
)

from .batch_processor import BatchProcessor

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


class StageBatchedProcessor(BatchProcessor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._timestamp: str = ""
        self._prewarm_executor: ThreadPoolExecutor | None = None
        self._prewarm_jobs: dict[str, Future] = {}
        self._prewarm_cancel_event = threading.Event()
        self._paddleocr_cache_store: OCRPersistentResultCache | None = None
        self._paddleocr_cache_identity: dict[str, Any] | None = None
        self._project_checkpoint_store = None
        self._project_checkpoint_page_keys: list[str] = []

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
            raise OperationCancelledError("Automatic translation was cancelled.")

    def _ensure_prewarm_executor(self) -> ThreadPoolExecutor:
        if self._prewarm_executor is None:
            self._prewarm_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ct-stage-prewarm")
        return self._prewarm_executor

    def _prewarm_cancel_checker(self) -> bool:
        cancel_event = getattr(self, "_prewarm_cancel_event", None)
        if cancel_event is not None and cancel_event.is_set():
            return True
        checker = getattr(self.main_page, "is_current_task_cancelled", None)
        try:
            return bool(checker()) if callable(checker) else bool(self._is_cancelled())
        except Exception:
            return bool(self._is_cancelled())

    def _reset_prewarm_lifecycle(self) -> None:
        cancel_event = getattr(self, "_prewarm_cancel_event", None)
        if cancel_event is None:
            self._prewarm_cancel_event = threading.Event()
        else:
            cancel_event.clear()

    def _start_prewarm(self, key: str, label: str, service: str, fn: Callable[[], None]) -> None:
        if key in self._prewarm_jobs:
            return
        self._raise_if_cancelled()
        self._prewarm_progress(
            service=service,
            status="starting",
            step_key=f"{key}_prewarm",
            message=self._stage_tr("{label} Docker 예열을 시작합니다.").format(label=label),
        )

        def runner() -> None:
            if self._prewarm_cancel_checker():
                raise OperationCancelledError(f"{label} prewarm was cancelled before startup.")
            started_at = time.perf_counter()
            outcome = "completed"
            try:
                fn()
            except OperationCancelledError:
                outcome = "cancelled"
                raise
            except Exception:
                outcome = "failed"
                raise
            finally:
                self._record_runtime_performance(
                    service=service,
                    operation="start",
                    elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
                    outcome=outcome,
                )
            if self._prewarm_cancel_checker():
                raise OperationCancelledError(f"{label} prewarm was cancelled after startup.")
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
    ) -> None:
        started_at = time.perf_counter()
        outcome = "completed"
        try:
            fallback()
        except OperationCancelledError:
            outcome = "cancelled"
            raise
        except Exception:
            outcome = "failed"
            raise
        finally:
            self._record_runtime_performance(
                service=service,
                operation="start",
                elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
                outcome=outcome,
            )

    def _await_prewarm_or_run(
        self,
        key: str,
        label: str,
        service: str,
        fallback: Callable[[], None],
    ) -> None:
        self._raise_if_cancelled()
        job = self._prewarm_jobs.pop(key, None)
        if job is None:
            self._run_runtime_fallback(
                service=service,
                fallback=fallback,
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
            )
        self._raise_if_cancelled()

    def _shutdown_prewarm_executor(self) -> None:
        executor = self._prewarm_executor
        self._prewarm_executor = None
        cancel_event = getattr(self, "_prewarm_cancel_event", None)
        if cancel_event is not None:
            cancel_event.set()
        jobs = list(self._prewarm_jobs.values())
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
                )
            except Exception as exc:
                failures.append(exc)
        if raise_on_failure and failures:
            raise failures[0]

    @staticmethod
    def _shutdown_runtime_with_retry(
        label: str,
        runtime_manager: Any,
        *,
        context: str,
        raise_on_failure: bool,
        release_for_handoff: bool = False,
    ) -> None:
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                release = getattr(runtime_manager, "release_for_handoff", None)
                if release_for_handoff and label == "OCR" and callable(release):
                    release()
                else:
                    runtime_manager.shutdown()
                return
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Failed to stop managed %s runtime during %s%s.",
                    label,
                    context,
                    "; retrying once" if attempt == 0 else "",
                    exc_info=True,
                )
        if raise_on_failure and last_error is not None:
            raise last_error

    def _start_ocr_prewarm(self, policy: dict[str, Any]) -> None:
        runtime_manager = getattr(self.main_page, "local_ocr_runtime_manager", None)
        if not isinstance(runtime_manager, LocalOCRRuntimeManager):
            return
        # A non-empty project cache may contain a page-local OCR hit that can be
        # known only after detection restores the exact ordered blocks. Defer in
        # that case to preserve the all-hit zero-runtime contract. A brand-new,
        # empty sidecar cannot contain a hit, so keep the cold-path overlap.
        project_store = getattr(self, "_project_checkpoint_store", None)
        if project_store is not None:
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
        engine_key = str(policy["primary_ocr_engine"])
        if (
            engine_key == "PaddleOCR VL"
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
                progress_callback=getattr(self.main_page, "report_runtime_progress", None),
                cancel_checker=self._prewarm_cancel_checker,
            ),
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
                progress_callback=getattr(self.main_page, "report_runtime_progress", None),
                cancel_checker=self._prewarm_cancel_checker,
            ),
        )

    @staticmethod
    def _ocr_runtime_service_name(engine_key: str) -> str:
        return {
            "PaddleOCR VL": "paddleocr_vl",
            "PaddleOCR VL Spotting": "paddleocr_vl_spotting",
            "HunyuanOCR": "hunyuanocr",
            "MangaLMM": "mangalmm",
        }.get(str(engine_key), str(engine_key).lower().replace(" ", "_"))

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
                progress_callback=getattr(self.main_page, "report_runtime_progress", None),
                cancel_checker=self._prewarm_cancel_checker,
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
                progress_callback=getattr(self.main_page, "report_runtime_progress", None),
                cancel_checker=self._prewarm_cancel_checker,
            ),
        )

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

    def _detect_all(self, pages: list[StagePageContext]) -> None:
        total_images = len(pages)
        settings_page = self.main_page.settings_page
        detector = self.block_detection.block_detector_cache
        checkpoint_store = getattr(self, "_project_checkpoint_store", None)

        for index, ctx in enumerate(pages):
            self._raise_if_cancelled()
            self._set_current_image(ctx.image_path)
            self.emit_progress(index, total_images, 0, 10, True)
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

            ctx.image = self.main_page.image_ctrl.load_image(ctx.image_path)
            if ctx.image is None:
                ensure_path_materialized(ctx.image_path)
                ctx.image = imk.read_image(ctx.image_path)

            source_lang_english = self._source_lang_english(ctx.source_lang)
            detection_hit = None
            detection_identity = None
            if checkpoint_store is not None:
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
        self._paddleocr_cache_store = None
        self._paddleocr_cache_identity = None
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
        runtime_identity = None
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
            self.emit_progress(index, total_images, 2, 10, False)
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
                context="OCR-to-inpaint handoff",
                raise_on_failure=True,
                release_for_handoff=True,
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
        self.inpainting._ensure_inpainter()
        return runtime

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
                    self._release_inpainter_before_gemma(
                        pages,
                        start_gemma=False,
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
            self._release_inpainter_before_gemma(
                pages,
                start_gemma=False,
            )
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
            if ctx.failed_stage:
                continue
            self._set_current_image(ctx.image_path)
            self.emit_progress(index, total_images, 3, 10, False)
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
                    continue

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
                        ctx.mask_details[
                            "project_brush_strokes_applied"
                        ] = True

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
            self._ensure_inpainter()
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
                ctx.inpaint_input_img = self.inpainting.inpaint_with_blocks(
                    ctx.image,
                    ctx.mask,
                    inpaint_blocks,
                    config=config,
                )
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
                ctx.inpaint_input_img = imk.convert_scale_abs(
                    ctx.inpaint_input_img
                )
                inpaint_edit_mask = getattr(
                    self.inpainting,
                    "last_inpaint_edit_mask",
                    None,
                )
                if inpaint_edit_mask is not None:
                    ctx.mask = np.where(
                        (ctx.mask > 0) | (inpaint_edit_mask > 0),
                        255,
                        0,
                    ).astype(np.uint8)
                (
                    ctx.inpaint_input_img,
                    ctx.mask,
                    ctx.cleanup_stats,
                ) = refine_bubble_residue_inpaint(
                    ctx.inpaint_input_img,
                    ctx.mask,
                    inpaint_blocks,
                    self.inpainting.inpainter_cache,
                    config,
                )
                (
                    ctx.inpaint_input_img,
                    ctx.mask,
                    ctx.cleanup_stats,
                ) = apply_duplicate_bubble_inner_fill(
                    ctx.inpaint_input_img,
                    ctx.mask,
                    ctx.mask_details,
                    ctx.cleanup_stats,
                )
                outside_before_restore = count_changed_outside_edit_mask(
                    ctx.image,
                    ctx.inpaint_input_img,
                    ctx.mask,
                )
                ctx.inpaint_input_img = composite_with_edit_mask(
                    ctx.image,
                    ctx.inpaint_input_img,
                    ctx.mask,
                )
                outside_after_restore = count_changed_outside_edit_mask(
                    ctx.image,
                    ctx.inpaint_input_img,
                    ctx.mask,
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

    def _release_inpainter_before_gemma(
        self,
        pages: list[StagePageContext],
        *,
        start_gemma: bool = True,
        handoff_outcome: str = "completed",
    ) -> None:
        report = self.inpainting.release_inpainter_resources()
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
        if start_gemma and gate.get("required") and not gate.get("observed"):
            raise RuntimeError(
                QCoreApplication.translate(
                    "StageBatchedProcessor",
                    "Gemma could not start because inpainter VRAM release was not confirmed."
                )
            )
        if not start_gemma:
            return
        self._raise_if_cancelled()
        for ctx in pages:
            if ctx.failed_stage or ctx.no_text_detected:
                ctx.translation_blocks = []
                continue
            ctx.translation_blocks = select_translate_inpaint_blocks(
                ctx.blk_list
            )
        if any(ctx.translation_blocks for ctx in pages):
            self._start_gemma_prewarm()

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

        gemma_runtime_started = False
        if gemma_runtime_required:
            gate = dict(getattr(self, "_inpainter_release_gate", {}) or {})
            if gate.get("required") and not gate.get("observed"):
                raise RuntimeError(
                    QCoreApplication.translate(
                        "StageBatchedProcessor",
                        "Gemma could not start because inpainter VRAM release was not confirmed."
                    )
                )
            self._start_gemma_prewarm()
            self._await_gemma_runtime()
            gemma_runtime_started = True

        for index, ctx in enumerate(pages):
            self._raise_if_cancelled()
            if ctx.failed_stage:
                continue
            self._set_current_image(ctx.image_path)
            self.emit_progress(index, total_images, 7, 10, False)
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
                runtime_required_now = translator.prepare_translation(
                    ctx.translation_blocks,
                    extra_context,
                )
                if runtime_required_now and not gemma_runtime_started:
                    gate = dict(getattr(self, "_inpainter_release_gate", {}) or {})
                    if gate.get("required") and not gate.get("observed"):
                        raise RuntimeError(
                            QCoreApplication.translate(
                                "StageBatchedProcessor",
                                "Gemma could not start because inpainter VRAM release was not confirmed."
                            )
                        )
                    self._start_gemma_prewarm()
                    self._await_gemma_runtime()
                    gemma_runtime_started = True
            try:
                _, translation_cache_status = translator.translate_with_cache_manager(
                    ctx.translation_blocks,
                    ctx.image,
                    extra_context,
                    self.cache_manager,
                )
                self._raise_if_cancelled()
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
                context="translation-to-render handoff",
                raise_on_failure=True,
            )
        self._raise_if_cancelled()

    def _render_page_text_items(
        self,
        ctx: StagePageContext,
        *,
        render_settings,
        trg_lng_cd: str,
    ) -> None:
        font = render_settings.font_family
        setting_font_color = QColor(render_settings.color)
        file_on_display = None
        if 0 <= self.main_page.curr_img_idx < len(self.main_page.image_files):
            file_on_display = self.main_page.image_files[self.main_page.curr_img_idx]

        text_items_state: list[dict[str, Any]] = []
        alignment = self.main_page.button_to_alignment.get(
            1,
            self.main_page.button_to_alignment[render_settings.alignment_id],
        )
        vertical_alignment = self.main_page.button_to_vertical_alignment.get(
            1,
            VERTICAL_ALIGNMENT_CENTER,
        )
        strict_render_symbols = should_use_strict_render_symbols(trg_lng_cd)
        seen_bubble_render_keys: set[tuple[tuple[int, int, int, int], str]] = set()
        for blk in ctx.blk_list:
            if is_block_ocr_empty(blk):
                continue
            finalize_ocr_processing_contract(blk)
            if (
                getattr(blk, "processing_action", "")
                != PROCESSING_ACTION_TRANSLATE_INPAINT
            ):
                blk._render_skip_reason = (
                    "processing_action_"
                    + str(
                        getattr(blk, "processing_action", "")
                        or "review"
                    )
                )
                continue
            x1, y1, block_width, block_height = blk.xywh
            translation_raw = blk.translation
            if should_skip_short_render_translation(blk, translation_raw):
                continue
            render_normalization = describe_render_text_sanitization(
                translation_raw,
                font,
                block_index=getattr(blk, "_debug_block_index", None),
                image_path=ctx.image_path,
                strict_symbols=strict_render_symbols,
            )
            translation = render_normalization.text
            blk._render_translation_raw = str(translation_raw or "")
            blk._render_text = str(translation or "")
            blk._render_normalization_applied = bool(
                render_normalization.normalization_applied
            )
            blk._render_normalization_reasons = list(render_normalization.reasons)
            blk._render_normalization_replacements = list(
                render_normalization.replacements
            )
            if should_skip_short_render_translation(blk, translation):
                continue
            gate_decision = describe_text_free_render_translation_gate(
                blk,
                translation,
                target_lang_code=trg_lng_cd,
            )
            if not gate_decision.render:
                blk._text_fit_status = gate_decision.status
                blk._render_skip_reason = gate_decision.status
                blk._render_normalization_reasons = sorted(
                    set(getattr(blk, "_render_normalization_reasons", []) or [])
                    .union(gate_decision.reasons)
                )
                continue
            mask_gate_decision = describe_text_free_render_mask_gate(
                blk,
                target_lang_code=trg_lng_cd,
            )
            if not mask_gate_decision.render:
                blk._text_fit_status = mask_gate_decision.status
                blk._render_skip_reason = mask_gate_decision.status
                blk.mask_decision = "review"
                blk.mask_reject_reason = mask_gate_decision.status
                blk._render_normalization_reasons = sorted(
                    set(getattr(blk, "_render_normalization_reasons", []) or [])
                    .union(mask_gate_decision.reasons)
                )
                continue
            duplicate_key = build_duplicate_bubble_render_key(blk)
            duplicate_gate = register_duplicate_bubble_render_key(
                blk,
                duplicate_key,
                seen_bubble_render_keys,
            )
            if not duplicate_gate.render:
                blk._text_fit_status = duplicate_gate.status
                blk._render_skip_reason = duplicate_gate.status
                blk._render_normalization_reasons = sorted(
                    set(getattr(blk, "_render_normalization_reasons", []) or [])
                    .union(duplicate_gate.reasons)
                )
                continue
            if duplicate_gate.reasons:
                blk._render_normalization_reasons = sorted(
                    set(getattr(blk, "_render_normalization_reasons", []) or [])
                    .union(duplicate_gate.reasons)
                )
            vertical = is_vertical_block(blk, trg_lng_cd)
            text_to_wrap = translation
            source_rect, block_anchor = build_render_rects_for_block(blk)
            block_width = int(source_rect[2])
            block_height = int(source_rect[3])
            block_alignment = alignment
            block_vertical_alignment = vertical_alignment
            layout_policy = resolve_text_free_manga_layout(
                blk,
                source_rect,
                target_lang_code=trg_lng_cd,
            )
            wrap_width = block_width
            if layout_policy.enabled:
                block_alignment = layout_policy.alignment
                block_vertical_alignment = layout_policy.vertical_alignment
                wrap_width = min(block_width, int(layout_policy.wrap_width))
            fit_clearance = get_render_fit_clearance_for_block(
                blk,
                render_settings.outline_width,
                auto_max_font_profile=getattr(render_settings, "auto_max_font_profile", "current"),
            )
            translation, font_size, rendered_width, rendered_height = pyside_word_wrap(
                text_to_wrap,
                font,
                wrap_width,
                block_height,
                float(render_settings.line_spacing),
                float(render_settings.outline_width),
                render_settings.bold,
                render_settings.italic,
                render_settings.underline,
                block_alignment,
                render_settings.direction,
                render_settings.max_font_size,
                render_settings.min_font_size,
                vertical,
                fit_clearance=fit_clearance,
                return_metrics=True,
            )
            translation, font_size, rendered_width, rendered_height = (
                refit_detected_bubble_text_if_underfilled(
                    blk,
                    text_to_wrap,
                    font,
                    wrap_width,
                    block_height,
                    float(render_settings.line_spacing),
                    float(render_settings.outline_width),
                    render_settings.bold,
                    render_settings.italic,
                    render_settings.underline,
                    block_alignment,
                    render_settings.direction,
                    render_settings.max_font_size,
                    render_settings.min_font_size,
                    vertical,
                    fit_clearance,
                    translation,
                    font_size,
                    rendered_width,
                    rendered_height,
                    auto_max_font_size=getattr(render_settings, "auto_max_font_size", True),
                    auto_max_font_profile=getattr(render_settings, "auto_max_font_profile", "current"),
                )
            )
            blk._text_fit_status = (
                "needs_review"
                if rendered_width > wrap_width or rendered_height > block_height
                else "fit"
            )
            if layout_policy.enabled and blk._text_fit_status != "fit":
                blk._text_fit_status = "needs_review_text_free_layout"
            underfill_gate = describe_text_free_underfill_gate(
                blk,
                source_rect=source_rect,
                rendered_width=rendered_width,
                rendered_height=rendered_height,
                target_lang_code=trg_lng_cd,
            )
            if blk._text_fit_status == "fit" and not underfill_gate.render:
                blk._text_fit_status = underfill_gate.status
                blk._render_normalization_reasons = sorted(
                    set(getattr(blk, "_render_normalization_reasons", []) or [])
                    .union(underfill_gate.reasons)
                )
            large_mask_gate = describe_text_free_large_mask_gate(
                blk,
                source_rect=source_rect,
                target_lang_code=trg_lng_cd,
            )
            if blk._text_fit_status == "fit" and not large_mask_gate.render:
                blk._text_fit_status = large_mask_gate.status
                blk.mask_decision = "review"
                blk.mask_reject_reason = large_mask_gate.status
                blk._render_normalization_reasons = sorted(
                    set(getattr(blk, "_render_normalization_reasons", []) or [])
                    .union(large_mask_gate.reasons)
                )
            if is_embedded_ui_panel_layout_review_candidate(blk) and not is_bubble_panel_text_candidate(blk):
                blk._text_fit_status = "needs_review_embedded_ui_panel_layout"
                blk._render_normalization_reasons = sorted(
                    set(getattr(blk, "_render_normalization_reasons", []) or [])
                    .union({"needs_review_embedded_ui_panel_layout"})
                )
            review_status_gate = describe_auto_render_review_status_gate(
                getattr(blk, "_text_fit_status", "fit")
            )
            if not review_status_gate.render:
                blk._render_skip_reason = review_status_gate.status
                blk._render_normalization_reasons = sorted(
                    set(getattr(blk, "_render_normalization_reasons", []) or [])
                    .union(review_status_gate.reasons)
                )
                continue
            blk._text_fit_metrics = {
                "rendered_width": float(rendered_width),
                "rendered_height": float(rendered_height),
                "box_width": float(wrap_width),
                "box_height": float(block_height),
                "item_width": float(block_width),
                "font_size": float(font_size),
            }
            if is_no_space_lang(trg_lng_cd):
                translation = translation.replace(" ", "")
            font_color = resolve_render_text_color(
                blk.font_color,
                setting_font_color,
                render_settings.force_font_color,
                render_settings.smart_global_apply_all,
            )
            render_markup = describe_render_text_markup(
                translation,
                font_family=font,
                font_size=font_size,
                text_color=font_color,
                alignment=block_alignment,
                line_spacing=float(render_settings.line_spacing),
                bold=render_settings.bold,
                italic=render_settings.italic,
                underline=render_settings.underline,
                direction=render_settings.direction,
                strict_symbols=strict_render_symbols,
            )
            blk._render_text = str(translation or "")
            blk._render_html = str(render_markup.html_text if render_markup.html_applied else translation)
            blk._render_html_applied = bool(render_markup.html_applied)
            blk._render_fallback_font_family = str(render_markup.fallback_font_family or "")
            blk._render_normalization_applied = bool(
                render_normalization.normalization_applied or render_markup.html_applied
            )
            blk._render_normalization_reasons = sorted(
                set(render_normalization.reasons)
                .union(render_markup.reasons)
                .union(layout_policy.reasons)
            )
            blk._render_normalization_replacements = list(
                render_normalization.replacements
            ) + list(render_markup.replacements)
            blk._render_centered_layout = bool(layout_policy.enabled)
            blk._render_layout_reasons = list(layout_policy.reasons)
            position, item_width, item_height = build_text_item_layout_geometry(
                source_rect,
                rendered_height,
                block_vertical_alignment,
            )
            outline_color = QColor(render_settings.outline_color) if render_settings.outline else None
            text_props = TextItemProperties(
                text=blk._render_html,
                font_family=font,
                font_size=font_size,
                text_color=font_color,
                alignment=block_alignment,
                line_spacing=float(render_settings.line_spacing),
                outline_color=outline_color,
                outline_width=float(render_settings.outline_width),
                bold=render_settings.bold,
                italic=render_settings.italic,
                underline=render_settings.underline,
                position=position,
                rotation=blk.angle,
                scale=1.0,
                transform_origin=blk.tr_origin_point,
                width=item_width,
                height=item_height,
                direction=render_settings.direction,
                vertical=vertical,
                vertical_alignment=block_vertical_alignment,
                source_rect=source_rect,
                block_anchor=block_anchor,
                selection_outlines=[
                    OutlineInfo(
                        0,
                        len(translation),
                        outline_color,
                        float(render_settings.outline_width),
                        OutlineType.Full_Document,
                    )
                ] if render_settings.outline else [],
            )
            text_item_state = text_props.to_dict()
            text_item_state["translation_raw"] = str(translation_raw or "")
            text_item_state["render_text"] = str(translation or "")
            text_item_state["render_html_applied"] = bool(render_markup.html_applied)
            text_item_state["render_fallback_font_family"] = str(
                render_markup.fallback_font_family or ""
            )
            text_item_state["render_area_source"] = str(
                getattr(blk, "_render_area_source", "text_bbox") or "text_bbox"
            )
            text_item_state["render_source_xyxy"] = list(
                getattr(blk, "_render_area_xyxy", []) or []
            )
            text_item_state["render_anchor_xyxy"] = list(
                getattr(blk, "_render_original_xyxy", []) or []
            )
            text_item_state["render_bubble_xyxy"] = list(
                getattr(blk, "_render_bubble_xyxy", []) or []
            )
            text_item_state["render_normalization_applied"] = bool(
                blk._render_normalization_applied
            )
            text_item_state["render_normalization_reasons"] = list(
                blk._render_normalization_reasons
            )
            text_item_state["render_centered_layout"] = bool(layout_policy.enabled)
            text_item_state["render_layout_reasons"] = list(layout_policy.reasons)
            text_item_state["text_fit_status"] = str(
                getattr(blk, "_text_fit_status", "fit") or "fit"
            )
            text_item_state["text_fit_metrics"] = dict(
                getattr(blk, "_text_fit_metrics", {}) or {}
            )
            text_items_state.append(text_item_state)
            if ctx.image_path == file_on_display:
                self.main_page.blk_rendered.emit(translation, font_size, blk, ctx.image_path)

        page_state = self._ensure_page_state(ctx.image_path)
        page_state.setdefault("viewer_state", {}).update({"text_items_state": text_items_state, "push_to_stack": True})
        page_state["blk_list"] = ctx.blk_list
        self.main_page.image_ctrl.mark_processing_stage(
            ctx.image_path,
            "render",
            "completed",
            text_item_count=len(text_items_state),
        )
        self.main_page.image_ctrl.mark_processing_stage(ctx.image_path, "pipeline", "completed")
        self.main_page.render_state_ready.emit(ctx.image_path)

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

    def _render_all(self, pages: list[StagePageContext]) -> None:
        total_images = len(pages)
        settings_page = self.main_page.settings_page
        export_settings = self._effective_export_settings(settings_page)
        target_lang_en = self.main_page.lang_mapping.get(pages[0].target_lang, pages[0].target_lang) if pages else ""
        trg_lng_cd = get_language_code(target_lang_en)
        render_settings = self.main_page.render_settings()
        for index, ctx in enumerate(pages):
            self._raise_if_cancelled()
            if ctx.failed_stage:
                continue
            self._set_current_image(ctx.image_path)
            self.emit_progress(index, total_images, 9, 10, False)
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
                render_hit, _output_base_root = (
                    self._prepare_render_checkpoint(
                        ctx,
                        render_settings=render_settings,
                        export_settings=export_settings,
                        target_language_code=trg_lng_cd,
                    )
                )
                if render_hit is not None:
                    try:
                        final_output_path = (
                            materialize_render_checkpoint_output(render_hit)
                        )
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
                        final_output_root = render_hit.output_root
                        self.main_page.image_ctrl.update_processing_summary(
                            ctx.image_path,
                            {
                                "translated_image_path": final_output_path,
                                "translated_page_image_path": (
                                    final_output_path
                                ),
                                "export_root": final_output_root,
                                "render_project_checkpoint_status": "hit",
                                "render_output_materialized": (
                                    not render_hit.output_exists
                                ),
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
                            output_materialized=not render_hit.output_exists,
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
                        self._log_page_done(
                            index,
                            total_images,
                            ctx.image_path,
                            preview_path=final_output_path,
                        )
                        self._raise_if_cancelled()
                        continue

                canonical_translations = [
                    (
                        getattr(block, "translation", ""),
                        copy.deepcopy(getattr(block, "rich_text", "")),
                    )
                    for block in ctx.blk_list
                ]
                if not ctx.no_text_detected:
                    try:
                        format_translations(
                            ctx.translation_blocks,
                            trg_lng_cd,
                            upper_case=render_settings.upper_case,
                        )
                        self._raise_if_cancelled()
                        get_best_render_area(
                            ctx.translation_blocks,
                            ctx.image,
                            ctx.inpaint_input_img,
                            auto_max_font_profile=getattr(
                                render_settings,
                                "auto_max_font_profile",
                                "current",
                            ),
                        )
                        self._render_page_text_items(
                            ctx,
                            render_settings=render_settings,
                            trg_lng_cd=trg_lng_cd,
                        )
                    finally:
                        for block, (translation, rich_text) in zip(
                            ctx.blk_list,
                            canonical_translations,
                        ):
                            block.translation = translation
                            block.rich_text = rich_text
                else:
                    self._render_page_text_items(
                        ctx,
                        render_settings=render_settings,
                        trg_lng_cd=trg_lng_cd,
                    )
                restore_blocks = select_blocks_for_original_restore_after_render(ctx.blk_list)
                if restore_blocks and ctx.inpaint_input_img is not None and ctx.mask is not None:
                    ctx.inpaint_input_img, ctx.mask, restore_stats = restore_original_for_block_masks(
                        ctx.image,
                        ctx.inpaint_input_img,
                        ctx.mask,
                        restore_blocks,
                    )
                    if restore_stats.get("applied"):
                        ctx.cleanup_stats = dict(ctx.cleanup_stats or {})
                        ctx.cleanup_stats["render_restore"] = restore_stats
                        ctx.patches = self.inpainting.get_inpainted_patches(ctx.mask, ctx.inpaint_input_img)
                        self.main_page.patches_processed.emit(ctx.patches, ctx.image_path)
                        self.main_page.image_ctrl.update_processing_summary(
                            ctx.image_path,
                            {
                                "render_restore_block_count": int(restore_stats.get("block_count", 0) or 0),
                                "render_restore_pixel_count": int(restore_stats.get("pixel_count", 0) or 0),
                            },
                        )
                self._raise_if_cancelled()
                page_state = self._ensure_page_state(ctx.image_path)
                final_output_path, final_output_root = self._write_final_render_export(
                    ctx.directory,
                    ctx.export_token,
                    ctx.image_path,
                    ctx.image,
                    ctx.patches,
                    page_state.get("viewer_state", {}),
                    export_settings,
                    page_index=index,
                    total_pages=total_images,
                    strict_render_symbols=should_use_strict_render_symbols(trg_lng_cd),
                )
                if ctx.project_render_identity:
                    try:
                        stored = record_render_checkpoint(
                            getattr(
                                self,
                                "_project_checkpoint_store",
                                None,
                            ),
                            page_key=ctx.project_checkpoint_page_key,
                            fingerprint=ctx.project_render_fingerprint,
                            identity=ctx.project_render_identity,
                            blocks=ctx.blk_list,
                            viewer_state=page_state.get(
                                "viewer_state",
                                {},
                            ),
                            output_path=final_output_path,
                            output_root=final_output_root,
                        )
                    except Exception:
                        logger.warning(
                            "Render checkpoint publication failed open for "
                            "%s.",
                            ctx.image_name,
                            exc_info=True,
                        )
                        stored = False
                    ctx.project_render_checkpoint_status = (
                        "refreshed" if stored else "miss"
                    )
                self.main_page.image_ctrl.update_processing_summary(
                    ctx.image_path,
                    {
                        "translated_image_path": final_output_path,
                        "translated_page_image_path": final_output_path,
                        "export_root": final_output_root,
                        "render_project_checkpoint_status": (
                            ctx.project_render_checkpoint_status
                        ),
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
                    project_checkpoint_status=(
                        ctx.project_render_checkpoint_status
                    ),
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
                self._raise_if_cancelled()
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
        self._emit_benchmark_event("batch_run_start", total_images=total_images)
        self._reset_prewarm_lifecycle()
        try:
            if self.main_page.file_handler.should_pre_materialize(image_list):
                self.main_page.file_handler.pre_materialize(image_list)
        except Exception:
            logger.debug("Stage-batched pre-materialization failed; continuing lazily.", exc_info=True)

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
            self._shutdown_managed_runtimes(
                context="batch startup preflight",
                raise_on_failure=True,
                preserve_sleeping_paddle=True,
            )
            self._raise_if_cancelled()
            self._start_ocr_prewarm(policy)
            self._raise_if_cancelled()
            self._detect_all(pages)
            self._sample_performance_resources("detect_stage_end")
            self._raise_if_cancelled()
            self._ocr_all(pages, policy)
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
            self._inpaint_all(pages)
            self._sample_performance_resources("inpaint_stage_end")
            self._raise_if_cancelled()
            self._translate_all(pages)
            self._sample_performance_resources("translate_stage_end")
            self._raise_if_cancelled()
            self._render_all(pages)
            self._sample_performance_resources("render_stage_end")
            self._emit_benchmark_event("batch_run_done", total_images=total_images)
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
            try:
                self._shutdown_prewarm_executor()
            finally:
                self._shutdown_managed_runtimes(
                    preserve_sleeping_paddle=batch_completed,
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
            self.emit_progress(index, total_images, 10, 10, False)
