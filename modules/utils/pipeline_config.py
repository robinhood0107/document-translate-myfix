from __future__ import annotations

from PySide6.QtCore import QCoreApplication
from typing import TYPE_CHECKING
from modules.inpainting.lama import LaMa
from modules.inpainting.mi_gan import MIGAN
from modules.inpainting.aot import AOT
from modules.inpainting.schema import Config
from app.ui.messages import Messages
from app.ui.settings.settings_page import SettingsPage

if TYPE_CHECKING:
    from controller import ComicTranslate

inpaint_map = {
    "LaMa": LaMa,
    "MI-GAN": MIGAN,
    "AOT": AOT,
}

FIELD_LABELS = {
    "api_key": "API Key",
    "api_url": "Endpoint URL",
    "model": "Model",
    "api_key_ocr": "OCR API Key",
    "endpoint": "Endpoint URL",
    "api_key_translator": "Translator API Key",
    "region_translator": "Translator Region",
    "folder_id": "Folder ID",
}

OCR_REQUIREMENTS = {
    "Microsoft OCR": ("Microsoft Azure", ("api_key_ocr", "endpoint")),
    "Google Cloud Vision": ("Google Cloud", ("api_key",)),
    "Gemini-2.0-Flash": ("Google Gemini", ("api_key",)),
    "GPT-4.1-mini": ("Open AI GPT", ("api_key",)),
}

TRANSLATOR_REQUIREMENTS = {
    "Custom": ("Custom", ("api_key", "api_url", "model")),
    "GPT-4.1": ("Open AI GPT", ("api_key",)),
    "GPT-4.1-mini": ("Open AI GPT", ("api_key",)),
    "Claude-4.5-Sonnet": ("Anthropic Claude", ("api_key",)),
    "Claude-4.5-Haiku": ("Anthropic Claude", ("api_key",)),
    "Gemini-2.5-Pro": ("Google Gemini", ("api_key",)),
    "Gemini-3.0-Flash": ("Google Gemini", ("api_key",)),
    "Deepseek-v3": ("Deepseek", ("api_key",)),
    "Microsoft Translator": ("Microsoft Azure", ("api_key_translator", "region_translator")),
    "DeepL": ("DeepL", ("api_key",)),
    "Yandex": ("Yandex", ("api_key", "folder_id")),
}

def get_config(settings_page: SettingsPage):
    strategy_settings = settings_page.get_hd_strategy_settings()
    if strategy_settings['strategy'] == settings_page.ui.tr("Resize"):
        config = Config(hd_strategy="Resize", hd_strategy_resize_limit = strategy_settings['resize_limit'])
    elif strategy_settings['strategy'] == settings_page.ui.tr("Crop"):
        config = Config(hd_strategy="Crop", hd_strategy_crop_margin = strategy_settings['crop_margin'],
                        hd_strategy_crop_trigger_size = strategy_settings['crop_trigger_size'])
    else:
        config = Config(hd_strategy="Original")

    return config


def _missing_fields(creds: dict, required_fields: tuple[str, ...]) -> list[str]:
    return [field for field in required_fields if not (creds.get(field) or "").strip()]


def _show_missing_credentials(main: ComicTranslate, provider_name: str, missing_fields: list[str]) -> None:
    field_text = ", ".join(FIELD_LABELS.get(field, field) for field in missing_fields)
    Messages.show_missing_credentials_error(main, provider_name, field_text)

def validate_ocr(main: ComicTranslate):
    """Ensure the selected OCR tool has the credentials it needs."""
    settings_page = main.settings_page
    settings = settings_page.get_all_settings()
    ocr_tool = settings['tools']['ocr']

    if not ocr_tool:
        Messages.show_missing_tool_error(main, QCoreApplication.translate("Messages", "Text Recognition model"))
        return False

    normalized_tool = settings_page.ui.value_mappings.get(ocr_tool, ocr_tool)
    provider = OCR_REQUIREMENTS.get(normalized_tool)
    if provider is None:
        return True

    provider_name, required_fields = provider
    creds = settings_page.get_credentials(provider_name)
    missing_fields = _missing_fields(creds, required_fields)
    if missing_fields:
        _show_missing_credentials(main, provider_name, missing_fields)
        return False

    return True


def validate_translator(main: ComicTranslate, target_lang: str):
    """Ensure the selected translator has the credentials it needs."""
    settings_page = main.settings_page
    settings = settings_page.get_all_settings()
    translator_tool = settings['tools']['translator']

    if not translator_tool:
        Messages.show_missing_tool_error(main, QCoreApplication.translate("Messages", "Translator"))
        return False

    normalized_tool = settings_page.ui.value_mappings.get(translator_tool, translator_tool)
    provider = TRANSLATOR_REQUIREMENTS.get(normalized_tool)
    if provider is None:
        return True

    provider_name, required_fields = provider
    creds = settings_page.get_credentials(provider_name)
    missing_fields = _missing_fields(creds, required_fields)
    if missing_fields:
        if normalized_tool == "Custom":
            Messages.show_custom_not_configured_error(main)
        else:
            _show_missing_credentials(main, provider_name, missing_fields)
        return False

    return True

def font_selected(main: ComicTranslate):
    if not main.render_settings().font_family:
        Messages.select_font_error(main)
        return False
    return True

def validate_settings(main: ComicTranslate, target_lang: str):
    if not validate_ocr(main):
        return False
    if not validate_translator(main, target_lang):
        return False
    if not font_selected(main):
        return False
    
    return True
