from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import imkit as imk
from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QColor

from app.path_materialization import ensure_path_materialized
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.text_item import OutlineInfo, OutlineType
from app.ui.messages import Messages
from modules.detection.processor import TextBlockDetector
from modules.ocr.factory import OCRFactory
from modules.ocr.local_runtime import LocalOCRRuntimeManager
from modules.ocr.selection import (
    STAGE_BATCHED_WORKFLOW_MODE,
    resolve_stage_batched_ocr_policy,
)
from modules.rendering.render import (
    describe_render_text_markup,
    describe_render_text_sanitization,
    get_best_render_area,
    is_vertical_block,
    pyside_word_wrap,
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
    reserve_export_run_token,
    resolve_export_directory,
)
from modules.utils.image_utils import generate_mask
from modules.utils.inpaint_cleanup import refine_bubble_residue_inpaint
from modules.utils.language_utils import get_language_code, is_no_space_lang, language_codes
from modules.utils.ocr_quality import summarize_ocr_quality
from modules.utils.pipeline_config import get_config, get_inpainter_runtime, inpaint_map
from modules.utils.render_style_policy import (
    VERTICAL_ALIGNMENT_TOP,
    build_rect_tuple,
    resolve_render_text_color,
)
from modules.utils.textblock import sort_blk_list
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
    page_ocr_metrics: dict[str, int] = field(default_factory=dict)
    page_translation_metrics: dict[str, int] = field(default_factory=dict)
    raw_mask: Any | None = None
    mask: Any | None = None
    mask_details: dict[str, Any] = field(default_factory=dict)
    patches: list[dict[str, Any]] = field(default_factory=list)
    inpaint_input_img: Any | None = None
    cleanup_stats: dict[str, Any] = field(
        default_factory=lambda: {"applied": False, "component_count": 0, "block_count": 0}
    )
    failed_stage: str = ""
    failed_reason: str = ""


class StageBatchedProcessor(BatchProcessor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._timestamp: str = ""
        self._prewarm_executor: ThreadPoolExecutor | None = None
        self._prewarm_jobs: dict[str, Future] = {}

    def _stage_tr(self, text: str) -> str:
        return QCoreApplication.translate("StageBatchedProcessor", text)

    def _prewarm_progress(self, **payload: Any) -> None:
        payload.setdefault("phase", "runtime_prewarm")
        payload.setdefault("service", "batch")
        payload.setdefault("status", "running")
        payload["runtime_prewarm"] = True
        self._report_runtime_progress(**payload)

    def _ensure_prewarm_executor(self) -> ThreadPoolExecutor:
        if self._prewarm_executor is None:
            self._prewarm_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ct-stage-prewarm")
        return self._prewarm_executor

    def _start_prewarm(self, key: str, label: str, service: str, fn: Callable[[], None]) -> None:
        if key in self._prewarm_jobs:
            return
        self._prewarm_progress(
            service=service,
            status="starting",
            step_key=f"{key}_prewarm",
            message=self._stage_tr("{label} Docker 예열을 시작합니다.").format(label=label),
        )

        def runner() -> None:
            fn()
            self._prewarm_progress(
                service=service,
                status="ready",
                step_key=f"{key}_prewarm_ready",
                message=self._stage_tr("{label} Docker 예열이 완료되었습니다.").format(label=label),
            )

        self._prewarm_jobs[key] = self._ensure_prewarm_executor().submit(runner)

    def _await_prewarm_or_run(
        self,
        key: str,
        label: str,
        service: str,
        fallback: Callable[[], None],
    ) -> None:
        job = self._prewarm_jobs.pop(key, None)
        if job is None:
            fallback()
            return
        try:
            job.result()
        except Exception as exc:
            logger.warning("%s prewarm failed; falling back to synchronous startup: %s", label, exc)
            self._prewarm_progress(
                service=service,
                status="running",
                runtime_prewarm_status="failed",
                step_key=f"{key}_prewarm_failed",
                message=self._stage_tr("{label} 예열 실패. 해당 단계에서 다시 준비합니다.").format(label=label),
                detail=str(exc),
            )
            fallback()

    def _shutdown_prewarm_executor(self) -> None:
        executor = self._prewarm_executor
        self._prewarm_executor = None
        self._prewarm_jobs.clear()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _start_ocr_prewarm(self, policy: dict[str, Any]) -> None:
        runtime_manager = getattr(self.main_page, "local_ocr_runtime_manager", None)
        if not isinstance(runtime_manager, LocalOCRRuntimeManager):
            return
        settings_page = self.main_page.settings_page
        engine_key = str(policy["primary_ocr_engine"])
        service = "paddleocr_vl" if "paddle" in engine_key.lower() else "hunyuanocr" if "hunyuan" in engine_key.lower() else engine_key.lower()
        self._start_prewarm(
            "ocr",
            "OCR",
            service,
            lambda: runtime_manager.ensure_engine(
                engine_key,
                settings_page,
                progress_callback=getattr(self.main_page, "report_runtime_progress", None),
                cancel_checker=getattr(self.main_page, "is_current_task_cancelled", None),
            ),
        )

    def _await_ocr_runtime(self, policy: dict[str, Any]) -> None:
        runtime_manager = getattr(self.main_page, "local_ocr_runtime_manager", None)
        if not isinstance(runtime_manager, LocalOCRRuntimeManager):
            return
        settings_page = self.main_page.settings_page
        engine_key = str(policy["primary_ocr_engine"])
        service = "paddleocr_vl" if "paddle" in engine_key.lower() else "hunyuanocr" if "hunyuan" in engine_key.lower() else engine_key.lower()
        self._await_prewarm_or_run(
            "ocr",
            "OCR",
            service,
            lambda: runtime_manager.ensure_engine(
                engine_key,
                settings_page,
                progress_callback=getattr(self.main_page, "report_runtime_progress", None),
                cancel_checker=getattr(self.main_page, "is_current_task_cancelled", None),
            ),
        )

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
                cancel_checker=getattr(self.main_page, "is_current_task_cancelled", None),
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
                cancel_checker=getattr(self.main_page, "is_current_task_cancelled", None),
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
            source_lang = str(state.get("source_lang", self.main_page.s_combo.currentText()))
            target_lang = str(state.get("target_lang", self.main_page.t_combo.currentText()))
            directory, archive_bname = resolve_export_directory(
                image_path,
                archive_info=self.main_page.file_handler.archive_info,
                source_records=getattr(self.main_page, "export_source_by_path", {}),
                project_file=getattr(self.main_page, "project_file", None),
                temp_dir=getattr(self.main_page, "temp_dir", None),
            )
            export_token = self._resolve_export_token(directory, self._timestamp)
            export_root = os.path.join(directory, f"comic_translate_{export_token}")
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
        self.main_page.image_skipped.emit(ctx.image_path, stage, reason)

    def _detect_all(self, pages: list[StagePageContext]) -> None:
        total_images = len(pages)
        settings_page = self.main_page.settings_page
        detector = self.block_detection.block_detector_cache
        if detector is None:
            detector = TextBlockDetector(settings_page)
            self.block_detection.block_detector_cache = detector

        for index, ctx in enumerate(pages):
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

            blk_list = detector.detect(ctx.image)
            ctx.precomputed_mask_details = detector.last_mask_details
            ctx.detector_key = detector.detector or settings_page.get_tool_selection("detector") or "RT-DETR-v2"
            ctx.detector_engine = detector.last_engine_name or ""
            ctx.detector_device = detector.last_device or resolve_device(settings_page.is_gpu_enabled(), backend="onnx")
            rtl = self._source_lang_english(ctx.source_lang) == "Japanese"

            if blk_list:
                get_best_render_area(blk_list, ctx.image)
                ctx.blk_list = sort_blk_list(blk_list, rtl)
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
            self._mark_page_failed(
                ctx,
                index=index,
                total_images=total_images,
                stage="detect",
                reason="No text blocks detected.",
                extra=self._ocr_quality_metrics(None),
            )

    def _run_primary_ocr(self, ctx: StagePageContext, policy: dict[str, Any]) -> dict[str, Any]:
        settings_page = self.main_page.settings_page
        source_lang_english = self._source_lang_english(ctx.source_lang)
        source_lang_code = language_codes.get(source_lang_english, "en")
        for blk in ctx.blk_list:
            blk.source_lang = source_lang_code
        device = resolve_device(settings_page.is_gpu_enabled())
        engine_key = str(policy["primary_ocr_engine"])
        cache_key = self.cache_manager._get_ocr_cache_key(ctx.image, ctx.source_lang, engine_key, device)
        cache_status = "miss"
        attempt_count = 0
        page_profile: dict[str, Any] = {}
        engine_name = engine_key

        if self.cache_manager._can_serve_all_blocks_from_ocr_cache(cache_key, ctx.blk_list):
            self.cache_manager._apply_cached_ocr_to_blocks(cache_key, ctx.blk_list)
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
            engine.process_image(ctx.image, ctx.blk_list)
            page_profile = dict(getattr(engine, "last_page_profile", {}) or {})
            apply_ocr_result_dictionary(ctx.blk_list, settings_page.get_ocr_result_dictionary_rules())
            self.cache_manager._cache_ocr_results(cache_key, ctx.blk_list)
            cache_status = "refreshed"
            attempt_count = 1
            engine_name = engine.__class__.__name__

        quality = summarize_ocr_quality(ctx.blk_list)
        if quality.get("low_quality", False):
            attempt_count += 1
            for blk in ctx.blk_list:
                blk.text = ""
                blk.texts = []
                blk.ocr_regions = []
            engine = OCRFactory.create_engine(
                settings_page,
                source_lang_english,
                engine_key,
                selected_ocr_mode=policy["normalized_ocr_mode"],
            )
            engine.process_image(ctx.image, ctx.blk_list)
            page_profile = dict(getattr(engine, "last_page_profile", {}) or {})
            apply_ocr_result_dictionary(ctx.blk_list, settings_page.get_ocr_result_dictionary_rules())
            self.cache_manager._cache_ocr_results(cache_key, ctx.blk_list)
            cache_status = "refreshed"
            quality = summarize_ocr_quality(ctx.blk_list)
            engine_name = engine.__class__.__name__

        metrics = self._ocr_quality_metrics(quality)
        return {
            "quality": quality,
            "metrics": metrics,
            "cache_status": cache_status,
            "attempt_count": attempt_count,
            "page_profile": page_profile,
            "engine_name": engine_name,
        }

    def _ocr_all(self, pages: list[StagePageContext], policy: dict[str, Any]) -> None:
        total_images = len(pages)
        settings_page = self.main_page.settings_page
        runtime_manager = getattr(self.main_page, "local_ocr_runtime_manager", None)
        self._await_ocr_runtime(policy)

        for index, ctx in enumerate(pages):
            if ctx.failed_stage:
                continue
            self._set_current_image(ctx.image_path)
            self.emit_progress(index, total_images, 2, 10, False)
            self._emit_benchmark_event(
                "ocr_start",
                image_path=ctx.image_path,
                image_index=index,
                total_images=total_images,
                block_count=len(ctx.blk_list or []),
            )
            try:
                result = self._run_primary_ocr(ctx, policy)
                quality = result["quality"]
                self._log_ocr_quality(ctx.image_path, quality, int(result["attempt_count"]))
                if quality.get("low_quality", False):
                    raise RuntimeError(quality.get("reason") or "OCR quality too low after retry.")
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
                    **ctx.page_ocr_metrics,
                )
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
            runtime_manager.shutdown()

    def _ensure_inpainter(self):
        settings_page = self.main_page.settings_page
        runtime = get_inpainter_runtime(settings_page)
        inpainter_key = runtime["key"]
        inpainter_backend = runtime["backend"]
        if self.inpainting.inpainter_cache is None or self.inpainting.cached_inpainter_key != inpainter_key:
            device = resolve_device(settings_page.is_gpu_enabled(), backend=inpainter_backend)
            inpainter_class = inpaint_map[inpainter_key]
            self.inpainting.inpainter_cache = inpainter_class(
                device,
                backend=inpainter_backend,
                runtime_device=runtime.get("device", device),
                inpaint_size=runtime.get("inpaint_size"),
                precision=runtime.get("precision"),
            )
            self.inpainting.cached_inpainter_key = inpainter_key
        return runtime

    def _inpaint_all(self, pages: list[StagePageContext]) -> None:
        total_images = len(pages)
        settings_page = self.main_page.settings_page
        export_settings = self._effective_export_settings(settings_page)
        hd_strategy_settings = settings_page.get_hd_strategy_settings()
        hd_strategy = settings_page.ui.value_mappings.get(
            hd_strategy_settings.get("strategy", ""),
            hd_strategy_settings.get("strategy", ""),
        )
        runtime = self._ensure_inpainter()
        config = get_config(settings_page)
        self._start_gemma_prewarm()

        for index, ctx in enumerate(pages):
            if ctx.failed_stage:
                continue
            self._set_current_image(ctx.image_path)
            self.emit_progress(index, total_images, 3, 10, False)
            self._emit_benchmark_event(
                "inpaint_start",
                image_path=ctx.image_path,
                image_index=index,
                total_images=total_images,
                block_count=len(ctx.blk_list or []),
            )
            try:
                ctx.mask_details = generate_mask(
                    ctx.image,
                    ctx.blk_list,
                    settings=settings_page.get_mask_refiner_settings(),
                    return_details=True,
                    precomputed_mask_details=ctx.precomputed_mask_details,
                )
                ctx.mask = ctx.mask_details["final_mask"]
                ctx.raw_mask = ctx.mask_details["raw_mask"]
                ctx.inpaint_input_img = self.inpainting.inpaint_with_blocks(ctx.image, ctx.mask, ctx.blk_list, config=config)
                ctx.inpaint_input_img = imk.convert_scale_abs(ctx.inpaint_input_img)
                ctx.inpaint_input_img, ctx.mask, ctx.cleanup_stats = refine_bubble_residue_inpaint(
                    ctx.inpaint_input_img,
                    ctx.mask,
                    ctx.blk_list,
                    self.inpainting.inpainter_cache,
                    config,
                )
                ctx.patches = self.inpainting.get_inpainted_patches(ctx.mask, ctx.inpaint_input_img)
                self.main_page.patches_processed.emit(ctx.patches, ctx.image_path)
                self.main_page.image_ctrl.update_processing_summary(
                    ctx.image_path,
                    {
                        "inpainter": settings_page.get_tool_selection("inpainter"),
                        "hd_strategy": hd_strategy,
                        "cleanup_applied": bool(ctx.cleanup_stats.get("applied", False)),
                        "cleanup_component_count": int(ctx.cleanup_stats.get("component_count", 0) or 0),
                        "cleanup_block_count": int(ctx.cleanup_stats.get("block_count", 0) or 0),
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
                    inpainter_key=runtime["key"],
                    hd_strategy=hd_strategy,
                    cleanup_stats=ctx.cleanup_stats,
                    mask_details=ctx.mask_details,
                    inpainter_backend=runtime["backend"],
                )
                self._maybe_emit_preview_image(
                    index=index,
                    total=total_images,
                    image_path=ctx.image_path,
                    stage_key="raw_mask",
                    stage_label="원본 마스크",
                    export_settings=export_settings,
                    preferred_path=debug_paths.get("raw_mask", ""),
                )
                self._maybe_emit_preview_image(
                    index=index,
                    total=total_images,
                    image_path=ctx.image_path,
                    stage_key="mask_overlay",
                    stage_label="마스크 오버레이",
                    export_settings=export_settings,
                    preferred_path=debug_paths.get("mask_overlay", ""),
                )
                self._maybe_emit_preview_image(
                    index=index,
                    total=total_images,
                    image_path=ctx.image_path,
                    stage_key="cleanup_delta",
                    stage_label="정리 마스크 변화량",
                    export_settings=export_settings,
                    preferred_path=debug_paths.get("cleanup_delta", ""),
                )
                self._maybe_emit_preview_image(
                    index=index,
                    total=total_images,
                    image_path=ctx.image_path,
                    stage_key="inpainted_image",
                    stage_label="인페인트 결과",
                    export_settings=export_settings,
                    preferred_path=cleaned_output_path,
                )
                self.main_page.image_ctrl.mark_processing_stage(
                    ctx.image_path,
                    "inpaint",
                    "completed",
                    patch_count=len(ctx.patches or []),
                )
                self._emit_benchmark_event(
                    "inpaint_end",
                    image_path=ctx.image_path,
                    image_index=index,
                    total_images=total_images,
                    block_count=len(ctx.blk_list or []),
                    patch_count=len(ctx.patches or []),
                )
            except Exception as exc:
                self._mark_page_failed(
                    ctx,
                    index=index,
                    total_images=total_images,
                    stage="inpaint",
                    reason=str(exc),
                    extra={**ctx.page_ocr_metrics, **ctx.page_translation_metrics},
                )

    def _translate_all(self, pages: list[StagePageContext]) -> None:
        total_images = len(pages)
        settings_page = self.main_page.settings_page
        extra_context = settings_page.get_llm_settings()["extra_context"]
        translator_key = settings_page.get_tool_selection("translator")
        runtime_manager = getattr(self.main_page, "local_translation_runtime_manager", None)
        self._await_gemma_runtime()

        for index, ctx in enumerate(pages):
            if ctx.failed_stage:
                continue
            self._set_current_image(ctx.image_path)
            self.emit_progress(index, total_images, 7, 10, False)
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
                block_count=len(ctx.blk_list or []),
                translator_key=translator_key,
            )
            translation_cache_key = self.cache_manager._get_translation_cache_key(
                ctx.image, ctx.source_lang, ctx.target_lang, translator_key, extra_context
            )
            try:
                translator = Translator(self.main_page, ctx.source_lang, ctx.target_lang)
                translation_cache_status = "miss"
                if self.cache_manager._can_serve_all_blocks_from_translation_cache(translation_cache_key, ctx.blk_list):
                    self.cache_manager._apply_cached_translations_to_blocks(translation_cache_key, ctx.blk_list)
                    apply_translation_result_dictionary(
                        ctx.blk_list,
                        settings_page.get_translation_result_dictionary_rules(),
                    )
                    translation_cache_status = "hit"
                else:
                    translator.translate(ctx.blk_list, ctx.image, extra_context)
                    apply_translation_result_dictionary(
                        ctx.blk_list,
                        settings_page.get_translation_result_dictionary_rules(),
                    )
                    self.cache_manager._cache_translation_results(translation_cache_key, ctx.blk_list)
                    translation_cache_status = "refreshed"
                ctx.page_translation_metrics = self._translation_benchmark_metrics(translator)
                self._persist_translation_state(
                    ctx.image_path,
                    ctx.blk_list,
                    translator_key,
                    translator.engine.__class__.__name__,
                    translation_cache_status,
                )
                self._emit_benchmark_event(
                    "translate_end",
                    image_path=ctx.image_path,
                    image_index=index,
                    total_images=total_images,
                    block_count=len(ctx.blk_list or []),
                    translator_key=translator_key,
                    translator_engine=translator.engine.__class__.__name__,
                    cache_status=translation_cache_status,
                    **ctx.page_translation_metrics,
                )
                raw_text_obj = json.loads(get_raw_text(ctx.blk_list))
                translated_text_obj = json.loads(get_raw_translation(ctx.blk_list))
                if (not raw_text_obj) or (not translated_text_obj):
                    raise RuntimeError("Translator returned empty JSON.")
            except Exception as exc:
                self._mark_page_failed(
                    ctx,
                    index=index,
                    total_images=total_images,
                    stage="translation",
                    reason=str(exc),
                    extra={**ctx.page_ocr_metrics, **ctx.page_translation_metrics},
                )

        if isinstance(runtime_manager, LocalGemmaRuntimeManager):
            runtime_manager.shutdown()

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
        for blk in ctx.blk_list:
            x1, y1, block_width, block_height = blk.xywh
            translation_raw = blk.translation
            if not translation_raw or len(translation_raw) == 1:
                continue
            render_normalization = describe_render_text_sanitization(
                translation_raw,
                font,
                block_index=getattr(blk, "_debug_block_index", None),
                image_path=ctx.image_path,
            )
            translation = render_normalization.text
            if not translation or len(translation) == 1:
                continue
            vertical = is_vertical_block(blk, trg_lng_cd)
            translation, font_size, rendered_width, rendered_height = pyside_word_wrap(
                translation,
                font,
                block_width,
                block_height,
                float(render_settings.line_spacing),
                float(render_settings.outline_width),
                render_settings.bold,
                render_settings.italic,
                render_settings.underline,
                self.main_page.button_to_alignment[render_settings.alignment_id],
                render_settings.direction,
                render_settings.max_font_size,
                render_settings.min_font_size,
                vertical,
                return_metrics=True,
            )
            if is_no_space_lang(trg_lng_cd):
                translation = translation.replace(" ", "")
            render_markup = describe_render_text_markup(translation)
            font_color = resolve_render_text_color(
                blk.font_color,
                setting_font_color,
                render_settings.force_font_color,
                render_settings.smart_global_apply_all,
            )
            source_rect = build_rect_tuple(x1, y1, block_width, block_height)
            outline_color = QColor(render_settings.outline_color) if render_settings.outline else None
            text_props = TextItemProperties(
                text=render_markup.html_text if render_markup.html_applied else translation,
                font_family=font,
                font_size=font_size,
                text_color=font_color,
                alignment=self.main_page.button_to_alignment[render_settings.alignment_id],
                line_spacing=float(render_settings.line_spacing),
                outline_color=outline_color,
                outline_width=float(render_settings.outline_width),
                bold=render_settings.bold,
                italic=render_settings.italic,
                underline=render_settings.underline,
                position=(x1, y1),
                rotation=blk.angle,
                scale=1.0,
                transform_origin=blk.tr_origin_point,
                width=rendered_width,
                height=rendered_height,
                direction=render_settings.direction,
                vertical=vertical,
                vertical_alignment=self.main_page.button_to_vertical_alignment.get(
                    render_settings.vertical_alignment_id,
                    VERTICAL_ALIGNMENT_TOP,
                ),
                source_rect=source_rect,
                block_anchor=source_rect,
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

    def _render_all(self, pages: list[StagePageContext]) -> None:
        total_images = len(pages)
        settings_page = self.main_page.settings_page
        export_settings = self._effective_export_settings(settings_page)
        target_lang_en = self.main_page.lang_mapping.get(pages[0].target_lang, pages[0].target_lang) if pages else ""
        trg_lng_cd = get_language_code(target_lang_en)
        render_settings = self.main_page.render_settings()
        for index, ctx in enumerate(pages):
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
                format_translations(ctx.blk_list, trg_lng_cd, upper_case=render_settings.upper_case)
                get_best_render_area(ctx.blk_list, ctx.image, ctx.inpaint_input_img)
                self._render_page_text_items(ctx, render_settings=render_settings, trg_lng_cd=trg_lng_cd)
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
                )
                self.main_page.image_ctrl.update_processing_summary(
                    ctx.image_path,
                    {
                        "translated_image_path": final_output_path,
                        "export_root": final_output_root,
                    },
                )
                self._emit_benchmark_event(
                    "render_end",
                    image_path=ctx.image_path,
                    image_index=index,
                    total_images=total_images,
                    block_count=len(ctx.blk_list or []),
                    translated_image_path=final_output_path,
                )
                self._emit_benchmark_event(
                    "page_done",
                    image_path=ctx.image_path,
                    image_index=index,
                    total_images=total_images,
                    block_count=len(ctx.blk_list or []),
                    patch_count=len(ctx.patches or []),
                )
                self._log_page_done(index, total_images, ctx.image_path, preview_path=final_output_path)
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
        self._run_started_at = time.monotonic()
        self._page_started_at = None
        self._progress_image_path = None
        self._recent_page_durations.clear()
        self._emit_benchmark_event("batch_run_start", total_images=total_images)
        try:
            if self.main_page.file_handler.should_pre_materialize(image_list):
                self.main_page.file_handler.pre_materialize(image_list)
        except Exception:
            logger.debug("Stage-batched pre-materialization failed; continuing lazily.", exc_info=True)

        pages = self._load_page_contexts(image_list)
        policy = self._ensure_stage_policy(pages)
        try:
            self._start_ocr_prewarm(policy)
            self._detect_all(pages)
            self._ocr_all(pages, policy)
            self._inpaint_all(pages)
            self._translate_all(pages)
            self._render_all(pages)
            self._emit_benchmark_event("batch_run_done", total_images=total_images)
        finally:
            self._shutdown_prewarm_executor()
            self._progress_image_path = None
