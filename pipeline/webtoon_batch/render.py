from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, List

import imkit as imk
from PySide6.QtGui import QColor

from app.path_materialization import ensure_path_materialized
from app.ui.canvas.save_renderer import ImageSaveRenderer
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.text_item import OutlineInfo, OutlineType
from modules.rendering.render import (
    build_render_rects_for_block,
    build_text_item_layout_geometry,
    describe_render_text_markup,
    describe_render_text_sanitization,
    get_best_render_area,
    get_render_fit_clearance_for_block,
    is_vertical_block,
    pyside_word_wrap,
)
from modules.utils.export_paths import export_run_root, reserve_export_run_token, resolve_export_directory
from modules.utils.automatic_output import (
    build_archive_page_file_name,
    build_archive_staging_dir,
    build_output_file_name,
    is_single_archive_mode,
    write_archive_image,
    write_output_image,
)
from modules.utils.language_utils import get_language_code, is_no_space_lang
from modules.utils.ocr_debug import export_ocr_debug_artifacts
from modules.utils.inpaint_debug import (
    build_inpaint_debug_metadata,
    export_inpaint_debug_artifacts,
)
from modules.utils.render_style_policy import (
    VERTICAL_ALIGNMENT_TOP,
    resolve_render_text_color,
)
from modules.utils.textblock import TextBlock
from modules.utils.translator_utils import format_translations, get_raw_text, get_raw_translation

if TYPE_CHECKING:
    from .processor import WebtoonBatchProcessor

logger = logging.getLogger(__name__)


class RenderMixin:
    def _effective_export_settings(self: WebtoonBatchProcessor) -> dict:
        return dict(self.main_page.get_resolved_export_settings())

    def _resolve_export_token(
        self: WebtoonBatchProcessor, directory: str, base_timestamp: str
    ) -> str:
        cache = getattr(self, "_export_run_tokens", None)
        if cache is None:
            cache = {}
            self._export_run_tokens = cache
        return reserve_export_run_token(directory, base_timestamp, cache)

    def _prepare_page_blocks_for_render(
        self: WebtoonBatchProcessor,
        image_path: str,
        blocks: List[TextBlock],
        has_patches: bool,
    ) -> List[TextBlock]:
        page_state = self.main_page.image_states[image_path]
        if not blocks:
            page_state["blk_list"] = []
            page_state["skip_render"] = not has_patches
            return []

        render_settings = self.main_page.render_settings()
        target_lang = page_state["target_lang"]
        target_lang_en = self.main_page.lang_mapping.get(target_lang, target_lang)
        target_lang_code = get_language_code(target_lang_en)

        format_translations(
            blocks, target_lang_code, upper_case=render_settings.upper_case
        )
        if is_no_space_lang(target_lang_code):
            for block in blocks:
                if block.translation:
                    block.translation = block.translation.replace(" ", "")

        page_state["blk_list"] = blocks
        page_state["skip_render"] = False
        return blocks

    def _store_page_text_items(
        self: WebtoonBatchProcessor,
        page_index: int,
        image_path: str,
        blocks: List[TextBlock],
        image_shape: tuple,
    ) -> None:
        page_state = self.main_page.image_states[image_path]
        viewer_state = page_state.setdefault("viewer_state", {})
        viewer_state["text_items_state"] = []
        viewer_state["push_to_stack"] = True

        if not blocks:
            return

        render_settings = self.main_page.render_settings()
        font = render_settings.font_family
        base_font_color = QColor(render_settings.color)
        max_font_size = render_settings.max_font_size
        min_font_size = render_settings.min_font_size
        line_spacing = float(render_settings.line_spacing)
        outline_width = float(render_settings.outline_width)
        outline = render_settings.outline
        outline_color = QColor(render_settings.outline_color) if outline else None
        bold = render_settings.bold
        italic = render_settings.italic
        underline = render_settings.underline
        alignment = self.main_page.button_to_alignment[render_settings.alignment_id]
        vertical_alignment = self.main_page.button_to_vertical_alignment.get(
            render_settings.vertical_alignment_id,
            VERTICAL_ALIGNMENT_TOP,
        )
        direction = render_settings.direction

        target_lang = page_state["target_lang"]
        target_lang_en = self.main_page.lang_mapping.get(target_lang, target_lang)
        target_lang_code = get_language_code(target_lang_en)

        virtual_img = SimpleNamespace(shape=image_shape)
        # Avoid clipping seam-owned overflow blocks (needed for page-spanning text).
        in_bounds_blocks = [
            block
            for block in blocks
            if float(block.xyxy[1]) >= 0 and float(block.xyxy[3]) <= float(image_shape[0])
        ]
        if in_bounds_blocks:
            get_best_render_area(in_bounds_blocks, virtual_img)

        should_emit_live = False
        webtoon_manager = getattr(self.main_page.image_viewer, "webtoon_manager", None)
        if self.main_page.webtoon_mode and webtoon_manager:
            should_emit_live = page_index in webtoon_manager.loaded_pages
        page_scene_offset = self._get_page_scene_offset(page_index)

        for block in blocks:
            x1, y1, x2, y2 = [float(v) for v in block.xyxy]
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)

            translation_raw = block.translation
            if not translation_raw:
                continue
            render_normalization = describe_render_text_sanitization(
                translation_raw,
                font,
                block_index=getattr(block, "_debug_block_index", None),
                image_path=image_path,
            )
            translation = render_normalization.text
            block._render_translation_raw = str(translation_raw or "")
            block._render_text = str(translation or "")
            block._render_normalization_applied = bool(
                render_normalization.normalization_applied
            )
            block._render_normalization_reasons = list(render_normalization.reasons)
            block._render_normalization_replacements = list(
                render_normalization.replacements
            )
            if not translation:
                continue

            vertical = is_vertical_block(block, target_lang_code)
            (
                wrapped_translation,
                font_size,
                rendered_width,
                rendered_height,
            ) = pyside_word_wrap(
                translation,
                font,
                width,
                height,
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
                fit_clearance=get_render_fit_clearance_for_block(
                    block,
                    outline_width,
                ),
                return_metrics=True,
            )

            if is_no_space_lang(target_lang_code):
                wrapped_translation = wrapped_translation.replace(" ", "")
            font_color = resolve_render_text_color(
                block.font_color,
                base_font_color,
                render_settings.force_font_color,
                render_settings.smart_global_apply_all,
            )
            render_markup = describe_render_text_markup(
                wrapped_translation,
                font_family=font,
                font_size=font_size,
                text_color=font_color,
                alignment=alignment,
                line_spacing=line_spacing,
                bold=bold,
                italic=italic,
                underline=underline,
                direction=direction,
            )
            block._render_text = str(wrapped_translation or "")
            block._render_html = str(
                render_markup.html_text if render_markup.html_applied else wrapped_translation or ""
            )
            block._render_html_applied = bool(render_markup.html_applied)
            block._render_fallback_font_family = str(
                render_markup.fallback_font_family or ""
            )
            block._render_normalization_applied = bool(
                render_normalization.normalization_applied
                or render_markup.html_applied
            )
            block._render_normalization_reasons = sorted(
                set(render_normalization.reasons).union(render_markup.reasons)
            )
            block._render_normalization_replacements = list(
                render_normalization.replacements
            ) + list(render_markup.replacements)

            source_rect, block_anchor = build_render_rects_for_block(block)
            position, item_width, item_height = build_text_item_layout_geometry(
                source_rect,
                rendered_height,
                vertical_alignment,
            )
            text_props = TextItemProperties(
                text=block._render_html,
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
                position=position,
                rotation=block.angle,
                scale=1.0,
                transform_origin=block.tr_origin_point if block.tr_origin_point else (0, 0),
                width=item_width,
                height=item_height,
                direction=direction,
                vertical=vertical,
                vertical_alignment=vertical_alignment,
                source_rect=source_rect,
                block_anchor=block_anchor,
                selection_outlines=[
                    OutlineInfo(
                        0,
                        len(wrapped_translation),
                        outline_color,
                        outline_width,
                        OutlineType.Full_Document,
                    )
                ]
                if outline
                else [],
            )
            text_item_state = text_props.to_dict()
            text_item_state["translation_raw"] = str(translation_raw or "")
            text_item_state["render_text"] = str(wrapped_translation or "")
            text_item_state["render_html_applied"] = bool(
                render_markup.html_applied
            )
            text_item_state["render_fallback_font_family"] = str(
                render_markup.fallback_font_family or ""
            )
            text_item_state["render_area_source"] = str(
                getattr(block, "_render_area_source", "text_bbox") or "text_bbox"
            )
            text_item_state["render_source_xyxy"] = list(
                getattr(block, "_render_area_xyxy", []) or []
            )
            text_item_state["render_anchor_xyxy"] = list(
                getattr(block, "_render_original_xyxy", []) or []
            )
            text_item_state["render_bubble_xyxy"] = list(
                getattr(block, "_render_bubble_xyxy", []) or []
            )
            text_item_state["render_normalization_applied"] = bool(
                block._render_normalization_applied
            )
            text_item_state["render_normalization_reasons"] = list(
                block._render_normalization_reasons
            )
            viewer_state["text_items_state"].append(text_item_state)

            if should_emit_live:
                render_block = block.deep_copy()
                render_block.translation = wrapped_translation
                render_block._render_html = block._render_html
                render_block.xyxy = list(render_block.xyxy)
                render_block.xyxy[1] += page_scene_offset
                render_block.xyxy[3] += page_scene_offset
                if render_block.bubble_xyxy is not None:
                    render_block.bubble_xyxy = list(render_block.bubble_xyxy)
                    render_block.bubble_xyxy[1] += page_scene_offset
                    render_block.bubble_xyxy[3] += page_scene_offset
                self.main_page.blk_rendered.emit(
                    wrapped_translation, font_size, render_block, image_path
                )

    def _save_final_rendered_page(
        self: WebtoonBatchProcessor, page_idx: int, image_path: str, timestamp: str
    ):
        """
        Handle per-page exports once page results are finalized.
        """
        logger.info(
            "Starting final render process for page %s at path: %s", page_idx, image_path
        )

        ensure_path_materialized(image_path)
        image = imk.read_image(image_path)
        if image is None:
            logger.error("Failed to load physical image for rendering: %s", image_path)
            return

        base_name = os.path.splitext(os.path.basename(image_path))[0].strip()
        extension = os.path.splitext(image_path)[1]
        directory, archive_bname = resolve_export_directory(
            image_path,
            archive_info=self.main_page.file_handler.archive_info,
            source_records=getattr(self.main_page, "export_source_by_path", {}),
            project_file=getattr(self.main_page, "project_file", None),
            temp_dir=getattr(self.main_page, "temp_dir", None),
        )

        export_settings = self._effective_export_settings()
        export_token = self._resolve_export_token(directory, timestamp)
        export_root = export_run_root(directory, export_token)
        self.main_page.image_ctrl.update_processing_summary(
            image_path,
            {
                "export_root": export_root,
                "export_settings": {
                    "export_raw_text": bool(export_settings.get("export_raw_text", False)),
                    "export_translated_text": bool(export_settings.get("export_translated_text", False)),
                    "export_inpainted_image": bool(export_settings.get("export_inpainted_image", False)),
                    "export_detector_overlay": bool(export_settings.get("export_detector_overlay", False)),
                    "export_raw_mask": bool(export_settings.get("export_raw_mask", False)),
                    "export_mask_overlay": bool(export_settings.get("export_mask_overlay", False)),
                    "export_cleanup_mask_delta": bool(export_settings.get("export_cleanup_mask_delta", False)),
                    "export_debug_metadata": bool(export_settings.get("export_debug_metadata", False)),
                },
            },
        )

        if self.main_page.image_states[image_path].get("skip_render"):
            logger.info("Skipping final render for page %s, copying original.", page_idx)
            reason = "No text blocks detected or processed successfully."
            self.skip_save(directory, export_token, base_name, extension, archive_bname, image)
            self.log_skipped_image(directory, export_token, image_path, reason)
            return

        if export_settings["export_inpainted_image"]:
            renderer = ImageSaveRenderer(image)
            patches = self.final_patches_for_save.get(image_path, [])
            renderer.apply_patches(patches)
            cleaned_image_rgb = renderer.render_to_image()
            path = os.path.join(export_root, "inpainted_images", archive_bname)
            os.makedirs(path, exist_ok=True)
            cleaned_output_path = os.path.join(
                path,
                build_output_file_name(
                    base_name,
                    "cleaned",
                    image_path,
                    export_settings,
                ),
            )
            write_output_image(
                cleaned_output_path,
                cleaned_image_rgb,
                source_path=image_path,
                resolved_settings=export_settings,
            )
            self.main_page.image_ctrl.update_processing_summary(
                image_path,
                {"cleaned_image_path": cleaned_output_path},
            )
        else:
            self.main_page.image_ctrl.update_processing_summary(
                image_path,
                {"cleaned_image_path": ""},
            )

        blk_list = self.main_page.image_states[image_path].get("blk_list", [])

        if export_settings["export_raw_text"] and blk_list:
            path = os.path.join(
                directory, f"comic_translate_{export_token}", "raw_texts", archive_bname
            )
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            raw_text = get_raw_text(blk_list)
            with open(os.path.join(path, f"{base_name}_raw.json"), "w", encoding="UTF-8") as f:
                f.write(raw_text)

        if export_settings["export_translated_text"] and blk_list:
            path = os.path.join(
                directory,
                f"comic_translate_{export_token}",
                "translated_texts",
                archive_bname,
            )
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            translated_text = get_raw_translation(blk_list)
            with open(
                os.path.join(path, f"{base_name}_translated.json"),
                "w",
                encoding="UTF-8",
            ) as f:
                f.write(translated_text)

        if (export_settings["export_raw_text"] or export_settings["export_translated_text"]) and blk_list:
            summary = self.main_page.image_states[image_path].get("processing_summary", {})
            path = os.path.join(
                directory,
                f"comic_translate_{export_token}",
                "ocr_debugs",
                archive_bname,
            )
            export_ocr_debug_artifacts(
                path,
                base_name,
                image,
                blk_list,
                summary.get("ocr_engine", ""),
                self.main_page.image_states[image_path].get("source_lang", ""),
            )

        summary = self.main_page.image_states[image_path].get("processing_summary", {})
        debug_state = self.main_page.image_states[image_path].get("inpaint_debug_state") or {}
        strategy_settings = self.main_page.settings_page.get_hd_strategy_settings()
        hd_strategy = self.main_page.settings_page.ui.value_mappings.get(
            strategy_settings.get("strategy", ""),
            strategy_settings.get("strategy", ""),
        )
        detector_cache = getattr(self.block_detection, "block_detector_cache", None)
        detector_key = summary.get("detector_key") or self.main_page.settings_page.get_tool_selection("detector")
        detector_engine = summary.get("detector_engine") or getattr(detector_cache, "last_engine_name", "") or ""
        detector_device = summary.get("device") or getattr(detector_cache, "last_device", "") or ""
        cleanup_stats = debug_state.get("cleanup_stats") or {"applied": False, "component_count": 0, "block_count": 0}
        raw_mask = debug_state.get("raw_mask")
        final_mask = debug_state.get("final_mask")
        cleanup_delta = None
        if final_mask is not None:
            import numpy as _np
            final_arr = _np.asarray(final_mask)
            if final_arr.ndim == 3:
                final_arr = final_arr[:, :, 0]
            if raw_mask is None:
                raw_arr = _np.zeros_like(final_arr, dtype=_np.uint8)
            else:
                raw_arr = _np.asarray(raw_mask)
                if raw_arr.ndim == 3:
                    raw_arr = raw_arr[:, :, 0]
            cleanup_delta = _np.where((final_arr > 0) & (raw_arr <= 0), 255, 0).astype(_np.uint8)
        mask_details = debug_state.get("mask_details") or {}
        debug_metadata = build_inpaint_debug_metadata(
            image_path=image_path,
            run_type=str(getattr(self.main_page, "_current_batch_run_type", "batch") or "batch"),
            detector_key=detector_key or "",
            detector_engine=detector_engine,
            device=detector_device,
            inpainter=self.main_page.settings_page.get_tool_selection("inpainter"),
            hd_strategy=hd_strategy,
            blocks=debug_state.get("mask_blocks") or blk_list,
            raw_mask=raw_mask,
            cleanup_delta=cleanup_delta,
            cleanup_stats=cleanup_stats,
            mask_refiner=str(mask_details.get("mask_refiner", "legacy_bbox") or "legacy_bbox"),
            protect_mask_applied=bool(mask_details.get("keep_existing_lines", False)),
            protect_mask=mask_details.get("protect_mask"),
            refiner_backend=str(mask_details.get("refiner_backend", "legacy") or "legacy"),
            refiner_device=str(mask_details.get("refiner_device", "cpu") or "cpu"),
            inpainter_backend=str(debug_state.get("inpainter_backend", "unknown") or "unknown"),
            legacy_base_mask=mask_details.get("legacy_base_mask"),
            hard_box_rescue_mask=mask_details.get("hard_box_rescue_mask"),
            hard_box_applied_count=int(mask_details.get("hard_box_applied_count", 0) or 0),
            hard_box_reason_totals=dict(mask_details.get("hard_box_reason_totals", {}) or {}),
        )
        export_inpaint_debug_artifacts(
            export_root=export_root,
            archive_bname=archive_bname,
            page_base_name=base_name,
            image=image,
            blocks=debug_state.get("mask_blocks") or blk_list,
            export_settings=export_settings,
            raw_mask=raw_mask,
            mask_overlay_mask=mask_details.get("final_mask", final_mask),
            cleanup_delta=cleanup_delta,
            metadata=debug_metadata,
        )

        renderer = ImageSaveRenderer(image)
        patches = self.final_patches_for_save.get(image_path, [])
        renderer.apply_patches(patches)
        viewer_state = self.main_page.image_states[image_path].get("viewer_state", {})
        renderer.add_state_to_image(viewer_state, page_idx, self.main_page)
        translated_dir = self.main_page.get_automatic_output_series_dir(
            directory,
            anchor_path=self.main_page.image_files[0] if self.main_page.image_files else image_path,
        )
        translated_image_rgb = renderer.render_to_image()
        if is_single_archive_mode(export_settings):
            staging_dir = build_archive_staging_dir(translated_dir, export_token)
            os.makedirs(staging_dir, exist_ok=True)
            output_path = os.path.join(
                staging_dir,
                build_archive_page_file_name(
                    page_idx,
                    len(self.page_paths),
                    base_name,
                    str(
                        export_settings.get(
                            "resolved_automatic_output_archive_image_format",
                            "png",
                        )
                    ),
                ),
            )
            write_archive_image(
                output_path,
                translated_image_rgb,
                resolved_settings=export_settings,
            )
        else:
            os.makedirs(translated_dir, exist_ok=True)
            output_path = os.path.join(
                translated_dir,
                build_output_file_name(
                    base_name,
                    "translated",
                    image_path,
                    export_settings,
                ),
            )
            write_output_image(
                output_path,
                translated_image_rgb,
                source_path=image_path,
                resolved_settings=export_settings,
            )
        logger.info("Saved final translated image to %s", output_path)
        self.main_page.image_ctrl.update_processing_summary(
            image_path,
            {
                "translated_image_path": output_path,
                "export_root": translated_dir,
            },
        )
