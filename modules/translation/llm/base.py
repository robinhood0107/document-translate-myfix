from typing import Any
import numpy as np
from abc import abstractmethod
import base64
import logging
import imkit as imk

from ..base import LLMTranslation
from ...ocr.selection import resolve_ocr_engine
from ...utils.textblock import TextBlock
from ...utils.translator_utils import (
    build_translation_input_json,
    get_raw_text,
    normalize_text_for_translation,
    set_texts_from_json,
)

logger = logging.getLogger(__name__)


class BaseLLMTranslation(LLMTranslation):
    """Base class for LLM-based translation engines with shared functionality."""
    
    def __init__(self):
        self.settings = None
        self.source_lang = None
        self.target_lang = None
        self.api_key = None
        self.api_url = None
        self.model = None
        self.img_as_llm_input = False
        self.temperature = None
        self.top_p = None
        self.max_tokens = None
        self.timeout = 30  
        self.debug_log_raw_response = False
        self.debug_log_response_json = False
        self.translation_mode_label = self.__class__.__name__
        self.last_translation_input_raw_json = ""
        self.last_translation_input_normalized_json = ""
    
    def initialize(self, settings: Any, source_lang: str, target_lang: str, **kwargs) -> None:
        """
        Initialize the LLM translation engine.
        
        Args:
            settings: Settings object with credentials
            source_lang: Source language name
            target_lang: Target language name
            **kwargs: Engine-specific initialization parameters
        """
        llm_settings = settings.get_llm_settings()
        self.settings = settings
        self.source_lang = source_lang
        self.target_lang = target_lang
        # Keep image input opt-in. The UI default is unchecked, so missing or
        # older settings should not silently enable image uploads.
        self.img_as_llm_input = llm_settings.get('image_input_enabled', False)
        self.temperature = 1.0
        self.top_p = 0.95
        self.max_tokens = 5000
        
    def translate(self, blk_list: list[TextBlock], image: np.ndarray, extra_context: str) -> list[TextBlock]:
        """
        Translate text blocks using LLM.
        
        Args:
            blk_list: List of TextBlock objects to translate
            image: Image as numpy array
            extra_context: Additional context information for translation
            
        Returns:
            List of updated TextBlock objects with translations
        """
        _, translation_input_json = self._build_translation_input_payloads(blk_list)
        system_prompt = self.get_system_prompt(self.source_lang, self.target_lang)
        user_prompt = f"{extra_context}\nMake the translation sound as natural as possible.\nTranslate this:\n{translation_input_json}"
        
        entire_translated_text = self._perform_translation(user_prompt, system_prompt, image)
        if self.debug_log_raw_response:
            logger.info(
                "translation raw content (%s): %s",
                self.translation_mode_label,
                entire_translated_text,
            )
        try:
            updated_count = set_texts_from_json(blk_list, entire_translated_text)
        except Exception:
            logger.exception(
                "translation response parse failed (%s). raw_content=%s",
                self.translation_mode_label,
                entire_translated_text,
            )
            raise
        logger.info(
            "translation parsed successfully (%s): updated_blocks=%d total_blocks=%d",
            self.translation_mode_label,
            updated_count,
            len(blk_list),
        )
            
        return blk_list

    def _resolved_ocr_engine(self) -> str:
        if self.settings is None or not hasattr(self.settings, "get_tool_selection"):
            return ""
        try:
            ocr_mode = self.settings.get_tool_selection("ocr")
        except Exception:
            logger.exception("translation input OCR mode lookup failed (%s)", self.translation_mode_label)
            return ""
        return resolve_ocr_engine(ocr_mode, self.source_lang)

    def _build_translation_input_payloads(self, blk_list: list[TextBlock]) -> tuple[str, str]:
        source_lang_code = self.get_language_code(self.source_lang) or (self.source_lang or "")
        ocr_engine = self._resolved_ocr_engine()
        raw_json = get_raw_text(blk_list)
        normalized_json = build_translation_input_json(
            blk_list,
            source_lang_code,
            ocr_engine=ocr_engine,
        )
        self.last_translation_input_raw_json = raw_json
        self.last_translation_input_normalized_json = normalized_json

        changed_blocks = sum(
            1
            for blk in blk_list
            if normalize_text_for_translation(
                getattr(blk, "text", ""),
                source_lang_code,
                ocr_engine=ocr_engine,
            )
            != str(getattr(blk, "text", "") or "")
        )
        if changed_blocks:
            logger.info(
                "translation input normalized (%s): source_lang=%s ocr_engine=%s changed_blocks=%d total_blocks=%d",
                self.translation_mode_label,
                source_lang_code,
                ocr_engine or "",
                changed_blocks,
                len(blk_list),
            )
            logger.debug("translation input raw (%s): %s", self.translation_mode_label, raw_json)
            logger.debug(
                "translation input normalized (%s): %s",
                self.translation_mode_label,
                normalized_json,
            )

        return raw_json, normalized_json

    @abstractmethod
    def _perform_translation(self, user_prompt: str, system_prompt: str, image: np.ndarray) -> str:
        """
        Perform translation using specific LLM.
        
        Args:
            user_prompt: User prompt for LLM
            system_prompt: System prompt for LLM
            image: Image as numpy array
            
        Returns:
            Translated JSON text
        """
        pass

    def encode_image(self, image: np.ndarray, ext=".jpg"):
        """
        Encode CV2/numpy image directly to base64 string using cv2.imencode.
        
        Args:
            image: Numpy array representing the image
            ext: Extension/format to encode the image as (".png" by default for higher quality)
                
        Returns:
            Tuple of (Base64 encoded string, mime_type)
        """
        # Direct encoding from numpy/cv2 format to bytes
        buffer = imk.encode_image(image, ext.lstrip('.'))
        
        # Convert to base64
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        # Map extension to mime type
        mime_types = {
            ".jpg": "image/jpeg", 
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp"
        }
        mime_type = mime_types.get(ext.lower(), f"image/{ext[1:].lower()}")
        
        return img_str, mime_type
