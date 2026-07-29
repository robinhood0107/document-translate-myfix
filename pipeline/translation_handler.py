from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from modules.translation.processor import Translator
from modules.utils.correction_dictionary import apply_translation_result_dictionary
from modules.utils.translator_utils import set_upper_case
from pipeline.webtoon_utils import filter_and_convert_visible_blocks, restore_original_block_coordinates
from .cache_manager import CacheManager

if TYPE_CHECKING:
    from controller import ComicTranslate
    from .main_pipeline import ComicTranslatePipeline

logger = logging.getLogger(__name__)


class TranslationHandler:
    """Handles translation processing with caching support."""
    
    def __init__(
            self, 
            main_page: ComicTranslate, 
            cache_manager: CacheManager, 
            pipeline: ComicTranslatePipeline,
        ):
        
        self.main_page = main_page
        self.cache_manager = cache_manager
        self.pipeline = pipeline

    def _current_file_path(self) -> str | None:
        if 0 <= self.main_page.curr_img_idx < len(self.main_page.image_files):
            return self.main_page.image_files[self.main_page.curr_img_idx]
        return None

    def _persist_current_page_translation_state(
        self,
        blk_list,
        translator_key: str,
        translator_engine: str,
        cache_status: str,
    ) -> None:
        current_file = self._current_file_path()
        if not current_file:
            return
        state = self.main_page.image_ctrl.ensure_page_state(current_file)
        state["blk_list"] = self.main_page.blk_list.copy()
        self.main_page.image_ctrl.update_processing_summary(
            current_file,
            {
                "translator_key": translator_key,
                "translator_engine": translator_engine,
                "block_count": len(blk_list or []),
            },
        )
        self.main_page.image_ctrl.mark_processing_stage(
            current_file,
            "translation",
            "completed",
            cache_status=cache_status,
        )

    def _apply_translation_corrections(self, blocks) -> None:
        apply_translation_result_dictionary(
            blocks,
            self.main_page.settings_page.get_translation_result_dictionary_rules(),
        )

    def translate_image(self, single_block=False):
        source_lang = self.main_page.s_combo.currentText()
        target_lang = self.main_page.t_combo.currentText()
        if self.main_page.image_viewer.hasPhoto() and self.main_page.blk_list:
            settings_page = self.main_page.settings_page
            image = self.main_page.image_viewer.get_image_array()
            extra_context = settings_page.get_llm_settings()['extra_context']
            translator_key = settings_page.get_tool_selection('translator')

            upper_case = settings_page.ui.uppercase_checkbox.isChecked()

            translator = Translator(self.main_page, source_lang, target_lang)

            if single_block:
                blk = self.pipeline.get_selected_block()
                if blk is None:
                    return

                if getattr(blk, "translation", "") and blk.translation.strip():
                    return

                legacy_cache_key = None
                if not translator.uses_persistent_translation_memory:
                    legacy_cache_key = self.cache_manager._get_translation_cache_key(
                        image,
                        source_lang,
                        target_lang,
                        translator_key,
                        extra_context,
                    )
                    cached_translation = (
                        self.cache_manager._get_cached_translation_for_block(
                            legacy_cache_key,
                            blk,
                        )
                        if self.cache_manager._is_translation_cached(legacy_cache_key)
                        else None
                    )
                    if cached_translation is not None:
                        blk.translation = cached_translation
                        self._apply_translation_corrections([blk])
                        set_upper_case([blk], upper_case)
                        self._persist_current_page_translation_state(
                            self.main_page.blk_list,
                            translator_key,
                            translator.engine.__class__.__name__,
                            "hit",
                        )
                        return

                selected_index = next(
                    (
                        index
                        for index, current in enumerate(self.main_page.blk_list)
                        if current is blk
                    ),
                    None,
                )
                if selected_index is None:
                    working_blocks = [blk.deep_copy()]
                    selected_index = 0
                else:
                    working_blocks = [
                        current.deep_copy()
                        for current in self.main_page.blk_list
                    ]

                translator.translate(
                    working_blocks,
                    image,
                    extra_context,
                    requested_indices=[selected_index],
                )
                if legacy_cache_key is not None:
                    original_blocks = (
                        self.main_page.blk_list
                        if len(working_blocks) == len(self.main_page.blk_list)
                        else [blk]
                    )
                    self.cache_manager._cache_translation_results(
                        legacy_cache_key,
                        original_blocks,
                        working_blocks,
                    )
                translated = working_blocks[selected_index]
                blk.translation = translated.translation
                guard_metadata = getattr(
                    translated,
                    "_translation_repetition_guard",
                    None,
                )
                if guard_metadata is not None:
                    setattr(blk, "_translation_repetition_guard", guard_metadata)
                elif hasattr(blk, "_translation_repetition_guard"):
                    delattr(blk, "_translation_repetition_guard")
                self._apply_translation_corrections([blk])
                set_upper_case([blk], upper_case)
                self._persist_current_page_translation_state(
                    self.main_page.blk_list,
                    translator_key,
                    translator.engine.__class__.__name__,
                    (
                        translator.translation_cache_status
                        if translator.uses_persistent_translation_memory
                        else "refreshed"
                    ),
                )
            else:
                _, cache_status = translator.translate_with_cache_manager(
                    self.main_page.blk_list,
                    image,
                    extra_context,
                    self.cache_manager,
                )
                self._apply_translation_corrections(self.main_page.blk_list)
                set_upper_case(self.main_page.blk_list, upper_case)
                self._persist_current_page_translation_state(
                    self.main_page.blk_list,
                    translator_key,
                    translator.engine.__class__.__name__,
                    cache_status,
                )

    def translate_webtoon_visible_area(self, single_block=False):
        """Perform translation on the visible area in webtoon mode."""
        source_lang = self.main_page.s_combo.currentText()
        target_lang = self.main_page.t_combo.currentText()
        
        if not (self.main_page.image_viewer.hasPhoto() and 
                self.main_page.webtoon_mode):
            logger.warning("translate_webtoon_visible_area called but not in webtoon mode")
            return
        
        # Get the visible area image and mapping data
        visible_image, mappings = self.main_page.image_viewer.get_visible_area_image()
        if visible_image is None or not mappings:
            logger.warning("No visible area found for translation")
            return
        
        # Filter blocks to only those in the visible area and convert coordinates
        visible_blocks = filter_and_convert_visible_blocks(
            self.main_page, self.pipeline, mappings, single_block
        )
        if not visible_blocks:
            logger.info("No blocks found in visible area")
            return
        
        # Perform translation on the visible image with filtered blocks
        settings_page = self.main_page.settings_page
        extra_context = settings_page.get_llm_settings()['extra_context']
        upper_case = settings_page.ui.uppercase_checkbox.isChecked()
        
        translator = Translator(self.main_page, source_lang, target_lang)
        translator.translate(visible_blocks, visible_image, extra_context)
        self._apply_translation_corrections(visible_blocks)
        
        # Translation is set, now restore original coordinates
        restore_original_block_coordinates(visible_blocks)
        
        # Apply upper case if needed
        set_upper_case(visible_blocks, upper_case)
        
        logger.info(f"Translation completed for {len(visible_blocks)} blocks in visible area")
