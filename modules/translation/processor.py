import logging
import numpy as np
from typing import Iterable
from PySide6.QtCore import QCoreApplication

from ..utils.textblock import TextBlock
from ..utils.device import resolve_device
from .local_runtime import LocalGemmaRuntimeManager
from .base import LLMTranslation
from .factory import TranslationFactory
from .translation_memory import (
    DEFAULT_RESULT_CACHE_LIMIT,
    DEFAULT_TM_CANDIDATE_LIMIT,
    TranslationMemoryStore,
)

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

        self._local_gemma_runtime_manager: LocalGemmaRuntimeManager | None = None
        if self.translator_key == "Custom Local Server(Gemma)":
            runtime_manager = getattr(main_page, "local_translation_runtime_manager", None)
            if not isinstance(runtime_manager, LocalGemmaRuntimeManager):
                runtime_manager = LocalGemmaRuntimeManager()
                main_page.local_translation_runtime_manager = runtime_manager
            self._local_gemma_runtime_manager = runtime_manager
        
        # Create appropriate engine using factory
        self.engine = TranslationFactory.create_engine(
            self.settings,
            self.source_lang_en,
            self.target_lang_en,
            self.translator_key
        )
        
        # Track engine type for method dispatching
        self.is_llm_engine = isinstance(self.engine, LLMTranslation)
        self._configure_local_gemma_engine()

    def _translation_memory_settings(self) -> dict:
        getter = getattr(self.settings, "get_translation_memory_settings", None)
        if callable(getter):
            try:
                values = dict(getter() or {})
            except Exception:
                logger.warning(
                    "Unable to read translation-memory settings; using safe defaults.",
                    exc_info=True,
                )
                values = {}
        else:
            values = {}
        values.setdefault("persistent_cache_enabled", True)
        values.setdefault("exact_tm_enabled", True)
        values.setdefault("result_cache_limit", DEFAULT_RESULT_CACHE_LIMIT)
        values.setdefault("candidate_limit", DEFAULT_TM_CANDIDATE_LIMIT)
        return values

    def _configure_local_gemma_engine(self) -> None:
        runtime_manager = self._local_gemma_runtime_manager
        if runtime_manager is None:
            return
        configure_runtime = getattr(self.engine, "configure_runtime_hooks", None)
        configure_memory = getattr(self.engine, "configure_translation_memory", None)
        if not callable(configure_runtime) or not callable(configure_memory):
            return

        memory_settings = self._translation_memory_settings()
        store = getattr(self.main_page, "translation_memory_store", None)
        if not isinstance(store, TranslationMemoryStore):
            store = TranslationMemoryStore(
                result_cache_limit=int(memory_settings["result_cache_limit"]),
                candidate_limit=int(memory_settings["candidate_limit"]),
            )
            self.main_page.translation_memory_store = store
        else:
            store.configure_limits(
                result_cache_limit=int(memory_settings["result_cache_limit"]),
                candidate_limit=int(memory_settings["candidate_limit"]),
            )

        configure_memory(store, memory_settings)
        configure_runtime(
            ensure_runtime=lambda: runtime_manager.ensure_server(
                self.settings,
                progress_callback=getattr(
                    self.main_page,
                    "report_runtime_progress",
                    None,
                ),
                cancel_checker=getattr(
                    self.main_page,
                    "is_current_task_cancelled",
                    None,
                ),
            ),
            runtime_identity_provider=lambda: runtime_manager.get_translation_cache_identity(
                self.settings
            ),
        )
    
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
    
    def prepare_translation(
        self,
        blk_list: list[TextBlock],
        extra_context: str = "",
        *,
        requested_indices: Iterable[int] | None = None,
    ) -> bool:
        """Resolve persistent hits and report whether local Gemma is still needed."""

        prepare = getattr(self.engine, "prepare_translation", None)
        if self._local_gemma_runtime_manager is None or not callable(prepare):
            return False
        return bool(
            prepare(
                blk_list,
                extra_context,
                requested_indices=requested_indices,
            )
        )

    @property
    def translation_cache_status(self) -> str:
        return str(
            getattr(self.engine, "translation_cache_status", "refreshed")
            or "refreshed"
        )

    @property
    def uses_persistent_translation_memory(self) -> bool:
        return self._local_gemma_runtime_manager is not None

    def translate_with_cache_manager(
        self,
        blk_list: list[TextBlock],
        image: np.ndarray,
        extra_context: str,
        cache_manager,
    ) -> tuple[list[TextBlock], str]:
        """Translate a full block list while preserving the legacy non-Gemma cache."""

        if self.uses_persistent_translation_memory or cache_manager is None:
            translated = self.translate(blk_list, image, extra_context)
            return translated, self.translation_cache_status

        cache_key = cache_manager._get_translation_cache_key(
            image,
            self.source_lang,
            self.target_lang,
            self.translator_key,
            extra_context,
        )
        if cache_manager._can_serve_all_blocks_from_translation_cache(
            cache_key,
            blk_list,
        ):
            cache_manager._apply_cached_translations_to_blocks(cache_key, blk_list)
            return blk_list, "hit"

        translated = self.translate(blk_list, image, extra_context)
        # Keep the legacy cache useful for remote/traditional translators, but
        # store the raw translation before user dictionary substitution so both
        # hit and miss paths apply current rules exactly once.
        cache_manager._cache_translation_results(cache_key, blk_list)
        return translated, "refreshed"

    def _report_translation_memory_warning(self) -> None:
        store = getattr(self.main_page, "translation_memory_store", None)
        reason = (
            store.disabled_reason
            if isinstance(store, TranslationMemoryStore)
            else ""
        )
        if not reason:
            return
        warned_reasons = getattr(
            self.main_page,
            "_translation_memory_warned_reasons",
            None,
        )
        if not isinstance(warned_reasons, set):
            warned_reasons = set()
            self.main_page._translation_memory_warned_reasons = warned_reasons
        if reason in warned_reasons:
            return
        warned_reasons.add(reason)
        callback = getattr(self.main_page, "report_runtime_progress", None)
        if callable(callback):
            try:
                callback(
                    {
                        "phase": "translation",
                        "service": "gemma",
                        "status": "warning",
                        "step_key": "translation_memory",
                        "stage_name": "translation",
                        "message": QCoreApplication.translate(
                            "Translator",
                            "Persistent translation cache is unavailable, so caching is disabled for this task while normal translation continues.",
                        ),
                        "detail": reason,
                        "translation_memory_disabled": True,
                    }
                )
            except Exception:
                logger.warning(
                    "Unable to report the translation-memory fail-open warning.",
                    exc_info=True,
                )

    def translate(
        self,
        blk_list: list[TextBlock],
        image: np.ndarray = None,
        extra_context: str = "",
        *,
        requested_indices: Iterable[int] | None = None,
    ) -> list[TextBlock]:
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
            if self._local_gemma_runtime_manager is not None:
                try:
                    return self.engine.translate(
                        blk_list,
                        image,
                        extra_context,
                        requested_indices=requested_indices,
                    )
                finally:
                    self._report_translation_memory_warning()
            return self.engine.translate(blk_list, image, extra_context)
        else:
            # Text-based translators only need the text blocks
            return self.engine.translate(blk_list)
