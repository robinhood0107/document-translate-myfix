from typing import Any

from .gpt import GPTTranslation


class CustomServiceTranslation(GPTTranslation):
    """Translation engine for authenticated OpenAI-compatible services."""

    def __init__(self):
        super().__init__()

    def initialize(
        self,
        settings: Any,
        source_lang: str,
        target_lang: str,
        translator_key: str,
        **kwargs,
    ) -> None:
        # Call BaseLLMTranslation.initialize directly so OpenAI credentials are
        # not loaded for this custom service path.
        super(GPTTranslation, self).initialize(settings, source_lang, target_lang, **kwargs)

        credentials = settings.get_credentials(settings.ui.tr("Custom Service"))
        self.api_key = credentials.get("api_key", "").strip()
        self.model = credentials.get("model", "").strip()
        self.api_base_url = credentials.get("api_url", "").strip().rstrip("/")
        self.timeout = 120
        self.translation_mode_label = "Custom Service"
