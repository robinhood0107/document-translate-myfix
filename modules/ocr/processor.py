import logging
import numpy as np
from typing import Any

from .local_runtime import LocalOCRRuntimeManager
from .selection import normalize_ocr_mode, resolve_ocr_engine
from ..utils.textblock import TextBlock
from ..utils.device import resolve_device
from ..utils.language_utils import language_codes
from .factory import OCRFactory

logger = logging.getLogger(__name__)


class OCRProcessor:
    """
    Processor for OCR operations using various engines.
    
    Uses a factory pattern to create and utilize the appropriate OCR engine
    based on settings and language.
    """
    
    def __init__(self):
        self.main_page = None
        self.settings = None
        self.source_lang = None
        self.source_lang_english = None
        self.last_engine_name = None
        self.last_device = None
        
    def initialize(self, main_page: Any, source_lang: str) -> None:
        """
        Initialize the OCR processor with settings and language.
        
        Args:
            main_page: The main application page with settings
            source_lang: The source language for OCR
        """
        self.main_page = main_page
        self.settings = main_page.settings_page
        self.source_lang = source_lang
        self.source_lang_english = self._get_english_lang(source_lang)
        self.ocr_mode = normalize_ocr_mode(self.settings.get_tool_selection('ocr'))
        self.ocr_key = resolve_ocr_engine(self.ocr_mode, self.source_lang_english)
        self.last_device = resolve_device(self.settings.is_gpu_enabled())
        runtime_manager = getattr(self.main_page, "local_ocr_runtime_manager", None)
        if isinstance(runtime_manager, LocalOCRRuntimeManager):
            runtime_manager.ensure_engine(
                self.ocr_key,
                self.settings,
                progress_callback=getattr(self.main_page, "report_runtime_progress", None),
                cancel_checker=getattr(self.main_page, "is_current_task_cancelled", None),
            )
        try:
            self.last_engine_name = OCRFactory.create_engine(
                self.settings,
                self.source_lang_english,
                self.ocr_key,
            ).__class__.__name__
        except Exception:
            self.last_engine_name = None
        
    def _get_english_lang(self, translated_lang: str) -> str:
        return self.main_page.lang_mapping.get(translated_lang, translated_lang)

    def process(self, img: np.ndarray, blk_list: list[TextBlock]) -> list[TextBlock]:
        """
        Process image with appropriate OCR engine.
        
        Args:
            img: Input image as numpy array
            blk_list: List of TextBlock objects to update with OCR text
            
        Returns:
            Updated list of TextBlock objects with recognized text
        """

        self._set_source_language(blk_list)
        engine = OCRFactory.create_engine(self.settings, self.source_lang_english, self.ocr_key)
        self.last_engine_name = engine.__class__.__name__
        logger.info(
            "ocr self-check: selected_mode=%s resolved_key=%s resolved_engine=%s source_lang=%s device=%s blocks=%d",
            self.ocr_mode,
            self.ocr_key,
            self.last_engine_name,
            self.source_lang_english,
            self.last_device,
            len(blk_list or []),
        )
        return engine.process_image(img, blk_list)
            
    def _set_source_language(self, blk_list: list[TextBlock]) -> None:
        source_lang_code = language_codes.get(self.source_lang_english, 'en')
        for blk in blk_list:
            blk.source_lang = source_lang_code

