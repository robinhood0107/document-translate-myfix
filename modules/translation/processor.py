import logging
import numpy as np

from ..utils.textblock import TextBlock
from ..utils.device import resolve_device
from .local_runtime import LocalGemmaRuntimeManager
from .base import LLMTranslation
from .factory import TranslationFactory

logger = logging.getLogger(__name__)


class Translator:
    """
    Main translator class that orchestrates the translation process.
    
    Supports multiple translation engines including:
    - Traditional translators (e.g Google, Microsoft, DeepL, Yandex)
    - LLM-based translators (e.g GPT, Claude, Gemini, Deepseek, Custom Service, Custom Local Server(Gemma))
    """
    
    def __init__(self, main_page, source_lang: str = "", target_lang: str = ""):
        """
        Initialize translator with settings and languages.
        
        Args:
            main_page: Main application page with settings
            source_lang: Source language name (localized)
            target_lang: Target language name (localized)
        """
        self.main_page = main_page
        self.settings = main_page.settings_page
        
        self.translator_key = self._get_translator_key(self.settings.get_tool_selection('translator'))
        
        self.source_lang = source_lang
        self.source_lang_en = self._get_english_lang(main_page, self.source_lang)
        self.target_lang = target_lang
        self.target_lang_en = self._get_english_lang(main_page, self.target_lang)

        if self.translator_key == "Custom Local Server(Gemma)":
            runtime_manager = getattr(main_page, "local_translation_runtime_manager", None)
            if not isinstance(runtime_manager, LocalGemmaRuntimeManager):
                runtime_manager = LocalGemmaRuntimeManager()
                main_page.local_translation_runtime_manager = runtime_manager
            runtime_manager.ensure_server(self.settings)
        
        # Create appropriate engine using factory
        self.engine = TranslationFactory.create_engine(
            self.settings,
            self.source_lang_en,
            self.target_lang_en,
            self.translator_key
        )
        
        # Track engine type for method dispatching
        self.is_llm_engine = isinstance(self.engine, LLMTranslation)
    
    def _get_translator_key(self, localized_translator: str) -> str:
        """
        Map localized translator names to standard keys.
        
        Args:
            localized_translator: Translator name in UI language
            
        Returns:
            Standard translator key
        """
        translator_map = {
            self.settings.ui.tr("Custom Service"): "Custom Service",
            self.settings.ui.tr("Custom Local Server(Gemma)"): "Custom Local Server(Gemma)",
            self.settings.ui.tr("Custom Local Server"): "Custom Local Server(Gemma)",
            self.settings.ui.tr("Deepseek-v3"): "Deepseek-v3",
            self.settings.ui.tr("GPT-4.1"): "GPT-4.1",
            self.settings.ui.tr("GPT-4.1-mini"): "GPT-4.1-mini",
            self.settings.ui.tr("Claude-4.6-Sonnet"): "Claude-4.6-Sonnet",
            self.settings.ui.tr("Claude-4.5-Haiku"): "Claude-4.5-Haiku",
            self.settings.ui.tr("Gemini-3.0-Flash"): "Gemini-3.0-Flash",
            self.settings.ui.tr("Gemini-2.5-Pro"): "Gemini-2.5-Pro",
            self.settings.ui.tr("Microsoft Translator"): "Microsoft Translator",
            self.settings.ui.tr("DeepL"): "DeepL",
            self.settings.ui.tr("Yandex"): "Yandex"
        }
        return translator_map.get(localized_translator, localized_translator)
    
    def _get_english_lang(self, main_page, translated_lang: str) -> str:
        """
        Get English language name from localized language name.
        
        Args:
            main_page: Main application page with language mapping
            translated_lang: Language name in UI language
            
        Returns:
            Language name in English
        """
        return main_page.lang_mapping.get(translated_lang, translated_lang)
    
    def translate(self, blk_list: list[TextBlock], image: np.ndarray = None, extra_context: str = "") -> list[TextBlock]:
        """
        Translate text in text blocks using the configured translation engine.
        
        Args:
            blk_list: List of TextBlock objects to translate
            image: Image as numpy array (for context in LLM translators)
            extra_context: Additional context information for translation
            
        Returns:
            List of updated TextBlock objects with translations
        """
        llm_settings = self.settings.get_llm_settings()
        gpu_enabled = bool(self.settings.is_gpu_enabled())
        logger.info(
            "translation self-check: translator=%s ocr=%s gpu=%s resolved_device=%s image_input_enabled=%s blocks=%d extra_context_len=%d llm_engine=%s",
            self.translator_key,
            self.settings.get_tool_selection('ocr'),
            gpu_enabled,
            resolve_device(gpu_enabled),
            bool(llm_settings.get('image_input_enabled', False)),
            len(blk_list or []),
            len(extra_context or ""),
            self.is_llm_engine,
        )

        if self.is_llm_engine:
            # LLM translators need image and extra context
            return self.engine.translate(blk_list, image, extra_context)
        else:
            # Text-based translators only need the text blocks
            return self.engine.translate(blk_list)
