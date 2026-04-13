import logging
import os
from modules.ocr.processor import OCRProcessor
from modules.utils.correction_dictionary import apply_ocr_result_dictionary
from modules.utils.device import resolve_device
from modules.utils.ocr_quality import summarize_ocr_quality
from pipeline.webtoon_utils import filter_and_convert_visible_blocks, restore_original_block_coordinates

logger = logging.getLogger(__name__)


class OCRHandler:
    """Handles OCR processing with caching support."""
    
    def __init__(self, main_page, cache_manager, pipeline):
        self.main_page = main_page
        self.cache_manager = cache_manager
        self.pipeline = pipeline
        self.ocr = OCRProcessor()

    def _current_file_path(self) -> str | None:
        if 0 <= self.main_page.curr_img_idx < len(self.main_page.image_files):
            return self.main_page.image_files[self.main_page.curr_img_idx]
        return None

    def _persist_current_page_ocr_state(
        self,
        blk_list,
        cache_status: str,
    ) -> None:
        current_file = self._current_file_path()
        if not current_file:
            return
        state = self.main_page.image_ctrl.ensure_page_state(current_file)
        state["blk_list"] = blk_list.copy()
        quality = summarize_ocr_quality(blk_list)
        logger.info(
            "ocr quality summary: image=%s blocks=%d non_empty=%d empty=%d single_char_like=%d cache=%s",
            os.path.basename(current_file),
            quality.get("block_count", 0),
            quality.get("non_empty", 0),
            quality.get("empty", 0),
            quality.get("single_char_like", 0),
            cache_status,
        )
        self.main_page.image_ctrl.update_processing_summary(
            current_file,
            {
                "ocr_key": self.ocr.ocr_key,
                "ocr_engine": self.ocr.last_engine_name or "",
                "device": self.ocr.last_device or "",
                "block_count": len(blk_list or []),
                "ocr_quality_counts": {
                    "non_empty": quality.get("non_empty", 0),
                    "empty": quality.get("empty", 0),
                    "single_char_like": quality.get("single_char_like", 0),
                },
            },
        )
        self.main_page.image_ctrl.mark_processing_stage(
            current_file,
            "ocr",
            "completed",
            cache_status=cache_status,
            quality=quality,
        )

    def _apply_ocr_corrections(self, blocks) -> None:
        apply_ocr_result_dictionary(
            blocks,
            self.main_page.settings_page.get_ocr_result_dictionary_rules(),
        )

    def OCR_image(self, single_block: bool = False):
        source_lang = self.main_page.s_combo.currentText()
        if self.main_page.image_viewer.hasPhoto() and self.main_page.image_viewer.rectangles:
            image = self.main_page.image_viewer.get_image_array()
            ocr_model = self.main_page.settings_page.get_tool_selection('ocr')
            device = resolve_device(
                self.main_page.settings_page.is_gpu_enabled()
            )
            cache_key = self.cache_manager._get_ocr_cache_key(image, source_lang, ocr_model, device)
            
            if single_block:
                blk = self.pipeline.get_selected_block()
                if blk is None:
                    return
                
                # Check if block already has text to avoid redundant processing
                if hasattr(blk, 'text') and blk.text and blk.text.strip():
                    return
                
                # Check if we have cached results for this image/model/language
                if self.cache_manager._is_ocr_cached(cache_key):
                    # Check if block exists in cache (even if text is empty)
                    payload = self.cache_manager._get_cached_ocr_payload_for_block(cache_key, blk)
                    if payload is not None:  # Block was processed before (even if text is empty)
                        self.cache_manager._apply_cached_ocr_payload_to_block(blk, payload)
                        self._apply_ocr_corrections([blk])
                        logger.info(f"Using cached OCR result for block: '{blk.text}'")
                        self._persist_current_page_ocr_state(self.main_page.blk_list, "hit")
                        return
                    else:
                        logger.info("Block not found in cache, processing single block...")
                        # Process just this single block
                        self.ocr.initialize(self.main_page, source_lang)
                        single_block_list = [blk]
                        self.ocr.process(image, single_block_list)
                        self._apply_ocr_corrections(single_block_list)
                        
                        # Update the cache with this new result using the cache manager's method
                        self.cache_manager.update_ocr_cache_for_block(cache_key, blk)
                        
                        logger.info(f"Processed single block and updated cache: '{blk.text}'")
                        self._persist_current_page_ocr_state(self.main_page.blk_list, "refreshed")
                else:
                    # Run OCR on all blocks and cache the results
                    logger.info("No cached OCR results found, running OCR on entire page...")
                    self.ocr.initialize(self.main_page, source_lang)
                    # Create a mapping between original blocks and their copies
                    original_to_copy = {}
                    all_blocks_copy = []
                    
                    for original_blk in self.main_page.blk_list:
                        copy_blk = original_blk.deep_copy()
                        all_blocks_copy.append(copy_blk)
                        # Use the original block's ID as the key for mapping
                        original_id = self.cache_manager._get_block_id(original_blk)
                        original_to_copy[original_id] = copy_blk
                    
                    if all_blocks_copy:  
                        self.ocr.process(image, all_blocks_copy)
                        self._apply_ocr_corrections(all_blocks_copy)
                        # Cache using the original blocks to maintain consistent IDs
                        self.cache_manager._cache_ocr_results(cache_key, self.main_page.blk_list, all_blocks_copy)
                        payload = self.cache_manager._get_cached_ocr_payload_for_block(cache_key, blk)
                        if payload is not None:
                            self.cache_manager._apply_cached_ocr_payload_to_block(blk, payload)
                        logger.info(f"Cached OCR results and extracted text for block: {blk.text}")
                        self._persist_current_page_ocr_state(self.main_page.blk_list, "refreshed")
            else:
                # For full page OCR, check if we can use cached results
                if self.cache_manager._can_serve_all_blocks_from_ocr_cache(cache_key, self.main_page.blk_list):
                    # All blocks can be served from cache
                    self.cache_manager._apply_cached_ocr_to_blocks(cache_key, self.main_page.blk_list)
                    self._apply_ocr_corrections(self.main_page.blk_list)
                    logger.info(f"Using cached OCR results for all {len(self.main_page.blk_list)} blocks")
                    self._persist_current_page_ocr_state(self.main_page.blk_list, "hit")
                else:
                    # Need to run OCR and cache results
                    self.ocr.initialize(self.main_page, source_lang)
                    if self.main_page.blk_list:  
                        self.ocr.process(image, self.main_page.blk_list)
                        self._apply_ocr_corrections(self.main_page.blk_list)
                        self.cache_manager._cache_ocr_results(cache_key, self.main_page.blk_list)
                        logger.info("OCR completed and cached for %d blocks", len(self.main_page.blk_list))
                        self._persist_current_page_ocr_state(self.main_page.blk_list, "refreshed")

    def OCR_webtoon_visible_area(self, single_block: bool = False):
        """Perform OCR on the visible area in webtoon mode."""
        source_lang = self.main_page.s_combo.currentText()
        
        if not (self.main_page.image_viewer.hasPhoto() and 
                self.main_page.webtoon_mode):
            logger.warning("OCR_webtoon_visible_area called but not in webtoon mode")
            return
        
        # Get the visible area image and mapping data
        visible_image, mappings = self.main_page.image_viewer.get_visible_area_image()
        if visible_image is None or not mappings:
            logger.warning("No visible area found for OCR")
            return
        
        # Filter blocks to only those in the visible area and convert coordinates
        visible_blocks = filter_and_convert_visible_blocks(
            self.main_page, self.pipeline, mappings, single_block
        )
        if not visible_blocks:
            logger.info("No blocks found in visible area")
            return
        
        # Perform OCR on the visible image with filtered blocks
        self.ocr.initialize(self.main_page, source_lang)
        self.ocr.process(visible_image, visible_blocks)
        self._apply_ocr_corrections(visible_blocks)
        
        # The OCR text is already set on the blocks, just restore coordinates
        restore_original_block_coordinates(visible_blocks)
        
        logger.info(f"OCR completed for {len(visible_blocks)} blocks in visible area")
