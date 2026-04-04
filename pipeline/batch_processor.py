from __future__ import annotations

import os
import json
import requests
import logging
import traceback
import imkit as imk
import time
from typing import TYPE_CHECKING
from datetime import datetime
from typing import List
from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QColor

from modules.detection.processor import TextBlockDetector
from modules.translation.processor import Translator
from modules.utils.textblock import sort_blk_list
from modules.utils.pipeline_config import inpaint_map, get_config
from modules.utils.image_utils import generate_mask
from modules.utils.language_utils import get_language_code, is_no_space_lang
from modules.utils.ocr_quality import summarize_ocr_quality
from modules.utils.ocr_debug import export_ocr_debug_artifacts
from modules.utils.render_style_policy import (
    VERTICAL_ALIGNMENT_TOP,
    build_rect_tuple,
    resolve_render_text_color,
)
from modules.utils.translator_utils import get_raw_translation, get_raw_text, format_translations
from modules.rendering.render import get_best_render_area, pyside_word_wrap, is_vertical_block
from modules.utils.device import resolve_device
from modules.utils.exceptions import InsufficientCreditsException
from app.path_materialization import ensure_path_materialized
from app.ui.canvas.text_item import OutlineInfo, OutlineType
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.messages import Messages
from .cache_manager import CacheManager
from .block_detection import BlockDetectionHandler
from .inpainting import InpaintingHandler
from .ocr_handler import OCRHandler

if TYPE_CHECKING:
    from controller import ComicTranslate

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Handles batch processing of comic translation."""
    
    def __init__(
            self, 
            main_page: ComicTranslate, 
            cache_manager: CacheManager, 
            block_detection_handler: BlockDetectionHandler, 
            inpainting_handler: InpaintingHandler, 
            ocr_handler: OCRHandler 
        ):
        
        self.main_page = main_page
        self.cache_manager = cache_manager
        # Use shared handlers from the main pipeline
        self.block_detection = block_detection_handler
        self.inpainting = inpainting_handler
        self.ocr_handler = ocr_handler

    def skip_save(self, directory, timestamp, base_name, extension, archive_bname, image):
        logger.info("Skipping fallback translated image save for '%s'.", base_name)

    def emit_progress(self, index, total, step, steps, change_name):
        """Wrapper around main_page.progress_update.emit that logs a human-readable stage."""
        stage_map = {
            0: 'start-image',
            1: 'text-block-detection',
            2: 'ocr-processing',
            3: 'pre-inpaint-setup',
            4: 'generate-mask',
            5: 'inpainting',
            7: 'translation',
            9: 'text-rendering-prepare',
            10: 'save-and-finish',
        }
        stage_name = stage_map.get(step, f'stage-{step}')
        logger.info(f"Progress: image_index={index}/{total} step={step}/{steps} ({stage_name}) change_name={change_name}")
        self.main_page.progress_update.emit(index, total, step, steps, change_name)

    def log_skipped_image(self, directory, timestamp, image_path, reason="", full_traceback=""):
        # Deprecated: skip details are captured by batch reporting/UI signals.
        return

    def _is_cancelled(self) -> bool:
        worker = getattr(self.main_page, "current_worker", None)
        return bool(worker and worker.is_cancelled)

    def _current_run_type(self) -> str:
        return str(getattr(self.main_page, "_current_batch_run_type", "batch") or "batch")

    def _should_force_json_exports(self) -> bool:
        return self._current_run_type() == "one_page_auto"

    def _effective_export_settings(self, settings_page) -> dict:
        export_settings = dict(settings_page.get_export_settings())
        if self._should_force_json_exports():
            export_settings["export_raw_text"] = True
            export_settings["export_translated_text"] = True
        return export_settings

    def _write_json_exports(
        self,
        directory: str,
        timestamp: str,
        archive_bname: str,
        image_path: str,
        image,
        blk_list,
        page_state: dict,
        source_lang: str,
        export_settings: dict,
    ) -> None:
        page_base_name = os.path.splitext(os.path.basename(image_path))[0]
        blocks = list(blk_list or [])

        if export_settings.get("export_raw_text", False):
            path = os.path.join(directory, f"comic_translate_{timestamp}", "raw_texts", archive_bname)
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, f"{page_base_name}_raw.json"), "w", encoding="UTF-8") as file:
                file.write(get_raw_text(blocks))

        if export_settings.get("export_translated_text", False):
            path = os.path.join(directory, f"comic_translate_{timestamp}", "translated_texts", archive_bname)
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, f"{page_base_name}_translated.json"), "w", encoding="UTF-8") as file:
                file.write(get_raw_translation(blocks))

        if export_settings.get("export_raw_text", False) or export_settings.get("export_translated_text", False):
            ocr_summary = page_state.get("processing_summary", {})
            debug_path = os.path.join(
                directory,
                f"comic_translate_{timestamp}",
                "ocr_debugs",
                archive_bname,
            )
            export_ocr_debug_artifacts(
                debug_path,
                page_base_name,
                image,
                blocks,
                ocr_summary.get("ocr_engine", ""),
                page_state.get("source_lang", source_lang),
            )

    def _ensure_page_state(self, image_path: str) -> dict:
        return self.main_page.image_ctrl.ensure_page_state(image_path)

    def _serialize_rectangles_from_blocks(self, blk_list) -> list[dict]:
        rects: list[dict] = []
        for blk in blk_list or []:
            x1, y1, x2, y2 = blk.xyxy
            rects.append(
                {
                    "rect": (float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                    "rotation": float(getattr(blk, "angle", 0)),
                    "transform_origin": tuple(blk.tr_origin_point)
                    if getattr(blk, "tr_origin_point", None)
                    else (0.0, 0.0),
                }
            )
        return rects

    def _start_page_summary(self, image_path: str, source_lang: str, target_lang: str) -> None:
        state = self._ensure_page_state(image_path)
        summary = self.main_page.image_ctrl.reset_processing_summary(
            image_path, run_type=self._current_run_type()
        )
        self.main_page.image_ctrl.update_processing_summary(
            image_path,
            {
                "source_lang": source_lang,
                "target_lang": target_lang,
                "device": resolve_device(self.main_page.settings_page.is_gpu_enabled()),
                "stage_status": summary.get("stage_status", {}),
            },
        )
        state["source_lang"] = source_lang
        state["target_lang"] = target_lang

    def _persist_detect_state(
        self,
        image_path: str,
        blk_list,
        detector_key: str,
        detector_engine: str,
        image,
    ) -> None:
        state = self._ensure_page_state(image_path)
        state["blk_list"] = blk_list
        viewer_state = state.setdefault("viewer_state", {})
        viewer_state["rectangles"] = self._serialize_rectangles_from_blocks(blk_list)
        self.main_page.image_ctrl.update_processing_summary(
            image_path,
            {
                "detector_key": detector_key,
                "detector_engine": detector_engine,
                "block_count": len(blk_list or []),
                "image_shape": list(getattr(image, "shape", []) or []),
            },
        )
        self.main_page.image_ctrl.mark_processing_stage(
            image_path,
            "detect",
            "completed",
            block_count=len(blk_list or []),
        )

    def _persist_ocr_state(
        self,
        image_path: str,
        blk_list,
        ocr_key: str,
        ocr_engine: str,
        device: str,
        quality: dict,
        cache_status: str,
        attempt_count: int,
    ) -> None:
        state = self._ensure_page_state(image_path)
        state["blk_list"] = blk_list
        self.main_page.image_ctrl.update_processing_summary(
            image_path,
            {
                "ocr_key": ocr_key,
                "ocr_engine": ocr_engine,
                "device": device,
                "block_count": len(blk_list or []),
                "ocr_quality_counts": {
                    "non_empty": quality.get("non_empty", 0),
                    "empty": quality.get("empty", 0),
                    "single_char_like": quality.get("single_char_like", 0),
                },
            },
        )
        self.main_page.image_ctrl.mark_processing_stage(
            image_path,
            "ocr",
            "completed",
            cache_status=cache_status,
            attempt_count=attempt_count,
            quality=quality,
        )

    def _persist_translation_state(
        self,
        image_path: str,
        blk_list,
        translator_key: str,
        translator_engine: str,
        cache_status: str,
    ) -> None:
        state = self._ensure_page_state(image_path)
        state["blk_list"] = blk_list
        self.main_page.image_ctrl.update_processing_summary(
            image_path,
            {
                "translator_key": translator_key,
                "translator_engine": translator_engine,
                "block_count": len(blk_list or []),
            },
        )
        self.main_page.image_ctrl.mark_processing_stage(
            image_path,
            "translation",
            "completed",
            cache_status=cache_status,
        )

    def _log_ocr_quality(self, image_path: str, quality: dict, attempt: int) -> None:
        logger.info(
            "ocr quality summary: image=%s attempt=%d blocks=%d non_empty=%d empty=%d single_char_like=%d low_quality=%s reason=%s",
            os.path.basename(image_path),
            attempt,
            quality.get("block_count", 0),
            quality.get("non_empty", 0),
            quality.get("empty", 0),
            quality.get("single_char_like", 0),
            quality.get("low_quality", False),
            quality.get("reason", ""),
        )

    def batch_process(self, selected_paths: List[str] = None):
        timestamp = datetime.now().strftime("%b-%d-%Y_%I-%M-%S%p")
        image_list = selected_paths if selected_paths is not None else self.main_page.image_files
        total_images = len(image_list)
        try:
            if self.main_page.file_handler.should_pre_materialize(image_list):
                count = self.main_page.file_handler.pre_materialize(image_list)
                logger.info("Batch pre-materialized %d paths before full-run processing.", count)
        except Exception:
            logger.debug("Batch pre-materialization failed; continuing lazily.", exc_info=True)

        for index, image_path in enumerate(image_list):
            if self._is_cancelled():
                return

            file_on_display = None
            if 0 <= self.main_page.curr_img_idx < len(self.main_page.image_files):
                file_on_display = self.main_page.image_files[self.main_page.curr_img_idx]

            # index, step, total_steps, change_name
            self.emit_progress(index, total_images, 0, 10, True)

            settings_page = self.main_page.settings_page
            export_settings = self._effective_export_settings(settings_page)
            page_state = self._ensure_page_state(image_path)
            source_lang = page_state['source_lang']
            target_lang = page_state['target_lang']
            self._start_page_summary(image_path, source_lang, target_lang)

            target_lang_en = self.main_page.lang_mapping.get(target_lang, None)
            trg_lng_cd = get_language_code(target_lang_en)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0].strip()
            extension = os.path.splitext(image_path)[1]
            directory = os.path.dirname(image_path)

            archive_bname = ""
            for archive in self.main_page.file_handler.archive_info:
                images = archive['extracted_images']
                archive_path = archive['archive_path']

                for img_pth in images:
                    if img_pth == image_path:
                        directory = os.path.dirname(archive_path)
                        archive_bname = os.path.splitext(os.path.basename(archive_path))[0].strip()

            image = self.main_page.image_ctrl.load_image(image_path)
            if image is None:
                ensure_path_materialized(image_path)
                image = imk.read_image(image_path)

            # skip UI-skipped images
            if page_state.get('skip', False):
                self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                self.log_skipped_image(directory, timestamp, image_path, "User-skipped")
                continue

            # Text Block Detection
            self.emit_progress(index, total_images, 1, 10, False)
            if self._is_cancelled():
                return

            # Use the shared block detector from the handler
            if self.block_detection.block_detector_cache is None:
                self.block_detection.block_detector_cache = TextBlockDetector(settings_page)
            
            blk_list = self.block_detection.block_detector_cache.detect(image)
            detector_key = settings_page.get_tool_selection('detector') or 'RT-DETR-v2'
            detector_engine = self.block_detection.block_detector_cache.last_engine_name or ""
            source_lang_english = self.main_page.lang_mapping.get(source_lang, source_lang)
            rtl = source_lang_english == 'Japanese'
            if blk_list:
                get_best_render_area(blk_list, image)
                blk_list = sort_blk_list(blk_list, rtl)
                self._persist_detect_state(
                    image_path,
                    blk_list,
                    detector_key,
                    detector_engine,
                    image,
                )

            self.emit_progress(index, total_images, 2, 10, False)
            if self._is_cancelled():
                return

            if blk_list:
                # Get ocr cache key for batch processing
                ocr_model = settings_page.get_tool_selection('ocr')
                device = resolve_device(settings_page.is_gpu_enabled())
                cache_key = self.cache_manager._get_ocr_cache_key(image, source_lang, ocr_model, device)
                # Use the shared OCR processor from the handler
                self.ocr_handler.ocr.initialize(self.main_page, source_lang)
                try:
                    cache_status = "miss"
                    attempt_count = 0
                    if self.cache_manager._can_serve_all_blocks_from_ocr_cache(cache_key, blk_list):
                        cache_status = "hit"
                        logger.info("ocr cache hit: using cached OCR for %d blocks", len(blk_list))
                        self.cache_manager._apply_cached_ocr_to_blocks(cache_key, blk_list)
                        attempt_count = 1
                    else:
                        logger.info("ocr cache miss: running OCR for %d blocks", len(blk_list))
                        self.ocr_handler.ocr.process(image, blk_list)
                        self.cache_manager._cache_ocr_results(cache_key, blk_list)
                        cache_status = "refreshed"
                        attempt_count = 1

                    quality = summarize_ocr_quality(blk_list)
                    self._log_ocr_quality(image_path, quality, attempt_count)

                    if quality.get("low_quality", False):
                        attempt_count += 1
                        logger.info(
                            "ocr quality gate triggered retry for %s: %s",
                            os.path.basename(image_path),
                            quality.get("reason", ""),
                        )
                        for blk in blk_list:
                            blk.text = ""
                        self.ocr_handler.ocr.process(image, blk_list)
                        self.cache_manager._cache_ocr_results(cache_key, blk_list)
                        quality = summarize_ocr_quality(blk_list)
                        self._log_ocr_quality(image_path, quality, attempt_count)
                        cache_status = "refreshed"

                    if quality.get("low_quality", False):
                        err_msg = quality.get("reason") or self.main_page.tr("OCR quality too low after retry.")
                        self.main_page.image_ctrl.update_processing_summary(
                            image_path,
                            {
                                "ocr_quality_counts": {
                                    "non_empty": quality.get("non_empty", 0),
                                    "empty": quality.get("empty", 0),
                                    "single_char_like": quality.get("single_char_like", 0),
                                },
                                "last_failure_reason": err_msg,
                            },
                        )
                        self.main_page.image_ctrl.mark_processing_stage(
                            image_path,
                            "ocr",
                            "failed",
                            reason=err_msg,
                            quality=quality,
                            cache_status=cache_status,
                            attempt_count=attempt_count,
                        )
                        raise RuntimeError(err_msg)

                    self._persist_ocr_state(
                        image_path,
                        blk_list,
                        ocr_model,
                        self.ocr_handler.ocr.last_engine_name or "",
                        device,
                        quality,
                        cache_status,
                        attempt_count,
                    )
                    
                except InsufficientCreditsException:
                    raise
                except Exception as e:
                    # if it's a connection/network error, give a short message
                    if isinstance(e, requests.exceptions.ConnectionError):
                        err_msg = QCoreApplication.translate("Messages", "Unable to connect to the server.\nPlease check your internet connection.")
                    # if it's an HTTPError, try to pull the "error_description" field
                    elif isinstance(e, requests.exceptions.HTTPError):
                        status_code = e.response.status_code if e.response is not None else 500
                        if status_code >= 500:
                            err_msg = Messages.get_server_error_text(status_code, context='ocr')
                        else:
                            try:
                                err_json = e.response.json()
                                if "detail" in err_json and isinstance(err_json["detail"], dict):
                                    err_msg = err_json["detail"].get("error_description", str(e))
                                else:
                                    err_msg = err_json.get("error_description", str(e))
                            except Exception:
                                err_msg = str(e)
                    else:
                        err_msg = str(e)

                    logger.exception(f"OCR processing failed: {err_msg}")
                    self.main_page.image_ctrl.mark_processing_stage(
                        image_path,
                        "ocr",
                        "failed",
                        reason=err_msg,
                    )
                    reason = f"OCR: {err_msg}"
                    full_traceback = traceback.format_exc()
                    if self._should_force_json_exports():
                        self._write_json_exports(
                            directory,
                            timestamp,
                            archive_bname,
                            image_path,
                            image,
                            blk_list,
                            page_state,
                            source_lang,
                            export_settings,
                        )
                    self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                    self.main_page.image_skipped.emit(image_path, "OCR", err_msg)
                    self.log_skipped_image(directory, timestamp, image_path, reason, full_traceback)
                    continue
            else:
                page_state = self._ensure_page_state(image_path)
                page_state["blk_list"] = []
                page_state.setdefault("viewer_state", {})["rectangles"] = []
                self.main_page.image_ctrl.mark_processing_stage(
                    image_path,
                    "detect",
                    "failed",
                    reason="No text blocks detected.",
                )
                if self._should_force_json_exports():
                    self._write_json_exports(
                        directory,
                        timestamp,
                        archive_bname,
                        image_path,
                        image,
                        [],
                        page_state,
                        source_lang,
                        export_settings,
                    )
                self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                self.main_page.image_skipped.emit(image_path, "Text Blocks", "")
                self.log_skipped_image(directory, timestamp, image_path, "No text blocks detected")
                continue

            self.emit_progress(index, total_images, 3, 10, False)
            if self._is_cancelled():
                return

            # Clean Image of text

            # Use the shared inpainter from the handler
            if self.inpainting.inpainter_cache is None or self.inpainting.cached_inpainter_key != settings_page.get_tool_selection('inpainter'):
                backend = 'onnx'
                device = resolve_device(
                    settings_page.is_gpu_enabled(),
                    backend=backend
                )
                inpainter_key = settings_page.get_tool_selection('inpainter')
                InpainterClass = inpaint_map[inpainter_key]
                logger.info("pre-inpaint: initializing inpainter '%s' on device %s", inpainter_key, device)
                t0 = time.time()
                self.inpainting.inpainter_cache = InpainterClass(device, backend=backend)
                self.inpainting.cached_inpainter_key = inpainter_key
                t1 = time.time()
                logger.info("pre-inpaint: inpainter initialized in %.2fs", t1 - t0)

            config = get_config(settings_page)
            logger.info("pre-inpaint: generating mask (blk_list=%d blocks)", len(blk_list))
            t0 = time.time()
            mask = generate_mask(image, blk_list)
            t1 = time.time()
            logger.info("pre-inpaint: mask generated in %.2fs (mask shape=%s)", t1 - t0, getattr(mask, 'shape', None))

            self.emit_progress(index, total_images, 4, 10, False)
            if self._is_cancelled():
                return

            inpaint_input_img = self.inpainting.inpainter_cache(image, mask, config)
            inpaint_input_img = imk.convert_scale_abs(inpaint_input_img)

            # Saving cleaned image
            patches = self.inpainting.get_inpainted_patches(mask, inpaint_input_img)
            self.main_page.patches_processed.emit(patches, image_path)

            # inpaint_input_img is already in RGB format

            if export_settings['export_inpainted_image']:
                path = os.path.join(directory, f"comic_translate_{timestamp}", "cleaned_images", archive_bname)
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                imk.write_image(os.path.join(path, f"{base_name}_cleaned{extension}"), inpaint_input_img)
            self.main_page.image_ctrl.mark_processing_stage(
                image_path,
                "inpaint",
                "completed",
                patch_count=len(patches or []),
            )

            self.emit_progress(index, total_images, 5, 10, False)
            if self._is_cancelled():
                return

            # Get Translations/ Export if selected
            extra_context = settings_page.get_llm_settings()['extra_context']
            translator_key = settings_page.get_tool_selection('translator')
            translator = Translator(self.main_page, source_lang, target_lang)
            
            # Get translation cache key for batch processing
            translation_cache_key = self.cache_manager._get_translation_cache_key(
                image, source_lang, target_lang, translator_key, extra_context
            )
            
            try:
                translation_cache_status = "miss"
                if self.cache_manager._can_serve_all_blocks_from_translation_cache(translation_cache_key, blk_list):
                    self.cache_manager._apply_cached_translations_to_blocks(translation_cache_key, blk_list)
                    translation_cache_status = "hit"
                    logger.info("Using cached translation results for all %d blocks", len(blk_list))
                else:
                    translator.translate(blk_list, image, extra_context)
                    # Cache the translation results for potential future use
                    self.cache_manager._cache_translation_results(translation_cache_key, blk_list)
                    translation_cache_status = "refreshed"
                    logger.info("Translation completed and cached for %d blocks", len(blk_list))
                self._persist_translation_state(
                    image_path,
                    blk_list,
                    translator_key,
                    translator.engine.__class__.__name__,
                    translation_cache_status,
                )
            except InsufficientCreditsException:
                raise
            except Exception as e:
                # if it's a connection/network error, give a short message
                if isinstance(e, requests.exceptions.ConnectionError):
                    err_msg = QCoreApplication.translate("Messages", "Unable to connect to the server.\nPlease check your internet connection.")
                # if it's an HTTPError, try to pull the "error_description" field
                elif isinstance(e, requests.exceptions.HTTPError):
                    status_code = e.response.status_code if e.response is not None else 500
                    if status_code >= 500:
                        err_msg = Messages.get_server_error_text(status_code, context='translation')
                    else:
                        try:
                            err_json = e.response.json()
                            if "detail" in err_json and isinstance(err_json["detail"], dict):
                                err_msg = err_json["detail"].get("error_description", str(e))
                            else:
                                err_msg = err_json.get("error_description", str(e))
                        except Exception:
                            err_msg = str(e)
                else:
                    err_msg = str(e)

                logger.exception(f"Translation failed: {err_msg}")
                self.main_page.image_ctrl.mark_processing_stage(
                    image_path,
                    "translation",
                    "failed",
                    reason=err_msg,
                )
                reason = f"Translator: {err_msg}"
                full_traceback = traceback.format_exc()
                if self._should_force_json_exports():
                    self._write_json_exports(
                        directory,
                        timestamp,
                        archive_bname,
                        image_path,
                        image,
                        blk_list,
                        page_state,
                        source_lang,
                        export_settings,
                    )
                self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                self.main_page.image_skipped.emit(image_path, "Translator", err_msg)
                self.log_skipped_image(directory, timestamp, image_path, reason, full_traceback)
                continue

            if self._is_cancelled():
                return

            entire_raw_text = get_raw_text(blk_list)
            entire_translated_text = get_raw_translation(blk_list)

            # Parse JSON strings and check if they're empty objects or invalid
            try:
                raw_text_obj = json.loads(entire_raw_text)
                translated_text_obj = json.loads(entire_translated_text)
                
                if (not raw_text_obj) or (not translated_text_obj):
                    self.main_page.image_ctrl.mark_processing_stage(
                        image_path,
                        "translation",
                        "failed",
                        reason="Translator returned empty JSON.",
                    )
                    if self._should_force_json_exports():
                        self._write_json_exports(
                            directory,
                            timestamp,
                            archive_bname,
                            image_path,
                            image,
                            blk_list,
                            page_state,
                            source_lang,
                            export_settings,
                        )
                    self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                    self.main_page.image_skipped.emit(image_path, "Translator", "")
                    self.log_skipped_image(directory, timestamp, image_path, "Translator: empty JSON")
                    continue
            except json.JSONDecodeError as e:
                # Handle invalid JSON
                error_message = str(e)
                reason = f"Translator: JSONDecodeError: {error_message}"
                logger.exception(reason)
                self.main_page.image_ctrl.mark_processing_stage(
                    image_path,
                    "translation",
                    "failed",
                    reason=error_message,
                )
                full_traceback = traceback.format_exc()
                if self._should_force_json_exports():
                    self._write_json_exports(
                        directory,
                        timestamp,
                        archive_bname,
                        image_path,
                        image,
                        blk_list,
                        page_state,
                        source_lang,
                        export_settings,
                    )
                self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                self.main_page.image_skipped.emit(image_path, "Translator", error_message)
                self.log_skipped_image(directory, timestamp, image_path, reason, full_traceback)
                continue

            self._write_json_exports(
                directory,
                timestamp,
                archive_bname,
                image_path,
                image,
                blk_list,
                page_state,
                source_lang,
                export_settings,
            )

            self.emit_progress(index, total_images, 7, 10, False)
            if self._is_cancelled():
                return

            # Text Rendering
            render_settings = self.main_page.render_settings()
            upper_case = render_settings.upper_case
            outline = render_settings.outline
            format_translations(blk_list, trg_lng_cd, upper_case=upper_case)
            get_best_render_area(blk_list, image, inpaint_input_img)

            font = render_settings.font_family
            setting_font_color = QColor(render_settings.color)

            max_font_size = render_settings.max_font_size
            min_font_size = render_settings.min_font_size
            line_spacing = float(render_settings.line_spacing) 
            outline_width = float(render_settings.outline_width)
            outline_color = QColor(render_settings.outline_color) if outline else None
            bold = render_settings.bold
            italic = render_settings.italic
            underline = render_settings.underline
            alignment_id = render_settings.alignment_id
            alignment = self.main_page.button_to_alignment[alignment_id]
            vertical_alignment = self.main_page.button_to_vertical_alignment.get(
                render_settings.vertical_alignment_id,
                VERTICAL_ALIGNMENT_TOP,
            )
            direction = render_settings.direction
                
            text_items_state = []
            for blk in blk_list:
                x1, y1, block_width, block_height = blk.xywh

                translation = blk.translation
                if not translation or len(translation) == 1:
                    continue
                
                # Determine if this block should use vertical rendering
                vertical = is_vertical_block(blk, trg_lng_cd)

                translation, font_size, rendered_width, rendered_height = pyside_word_wrap(
                    translation, 
                    font, 
                    block_width, 
                    block_height,
                    line_spacing, 
                    outline_width, 
                    bold, 
                    italic, 
                    underline,
                    alignment, 
                    direction, 
                    max_font_size, 
                    min_font_size,
                    vertical,
                    return_metrics=True
                )
                
                # Display text if on current page  
                if image_path == file_on_display:
                    self.main_page.blk_rendered.emit(translation, font_size, blk, image_path)

                # Language-specific formatting for state storage
                if is_no_space_lang(trg_lng_cd):
                    translation = translation.replace(' ', '')

                # Smart Color Override
                font_color = resolve_render_text_color(
                    blk.font_color,
                    setting_font_color,
                    render_settings.force_font_color,
                    render_settings.smart_global_apply_all,
                )
                source_rect = build_rect_tuple(x1, y1, block_width, block_height)

                # Use TextItemProperties for consistent text item creation
                text_props = TextItemProperties(
                    text=translation,
                    font_family=font,
                    font_size=font_size,
                    text_color=font_color,
                    alignment=alignment,
                    line_spacing=line_spacing,
                    outline_color=outline_color,
                    outline_width=outline_width,
                    bold=bold,
                    italic=italic,
                    underline=underline,
                    position=(x1, y1),
                    rotation=blk.angle,
                    scale=1.0,
                    transform_origin=blk.tr_origin_point,
                    width=rendered_width,
                    height=rendered_height,
                    direction=direction,
                    vertical=vertical,
                    vertical_alignment=vertical_alignment,
                    source_rect=source_rect,
                    block_anchor=source_rect,
                    selection_outlines=[
                        OutlineInfo(0, len(translation), 
                        outline_color, 
                        outline_width, 
                        OutlineType.Full_Document)
                    ] if outline else [],
                )
                text_items_state.append(text_props.to_dict())

            page_state = self._ensure_page_state(image_path)
            page_state['viewer_state'].update({
                'text_items_state': text_items_state
            })
            page_state['viewer_state'].update({
                'push_to_stack': True
            })
            self.main_page.image_ctrl.mark_processing_stage(
                image_path,
                "render",
                "completed",
                text_item_count=len(text_items_state),
            )
            
            self.emit_progress(index, total_images, 9, 10, False)
            if self._is_cancelled():
                return

            # Saving blocks with texts to history
            page_state.update({
                'blk_list': blk_list
            })
            self.main_page.image_ctrl.mark_processing_stage(
                image_path,
                "pipeline",
                "completed",
            )

            # Notify UI that this page's render state is finalized.
            # This enables a deterministic refresh when the user navigates to this page
            # during processing and misses live blk_rendered events.
            self.main_page.render_state_ready.emit(image_path)

            if image_path == file_on_display:
                self.main_page.blk_list = blk_list.copy()

            self.emit_progress(index, total_images, 10, 10, False)
