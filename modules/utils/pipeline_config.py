from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QCoreApplication

from modules.ocr.selection import resolve_ocr_engine
from modules.ocr.local_runtime import LocalOCRRuntimeManager
from modules.inpainting.aot import AOT
from modules.inpainting.lama_variants import LaMaLarge512px, LaMaMPE
from modules.inpainting.mi_gan import MIGAN
from modules.inpainting.schema import Config
from modules.utils.exceptions import LocalServiceSetupError
from modules.utils.inpainting_runtime import (
    inpainter_backend_for,
    inpainter_default_settings,
    normalize_inpainter_key,
)
from app.ui.messages import Messages
from app.ui.settings.settings_page import SettingsPage

if TYPE_CHECKING:
    from controller import ComicTranslate

inpaint_map = {
    "AOT": AOT,
    "lama_large_512px": LaMaLarge512px,
    "lama_mpe": LaMaMPE,
    "MI-GAN": MIGAN,
    # Backward compatibility for older saved settings or stale project metadata.
    "LaMa": LaMaLarge512px,
}

FIELD_LABELS = {
    "api_key": "API Key",
    "api_url": "Endpoint URL",
    "server_url": "Server URL",
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
    "Custom Service": ("Custom Service", ("api_key", "api_url", "model")),
    "Custom Local Server(Gemma)": ("Custom Local Server(Gemma)", ("api_url", "model")),
    "GPT-4.1": ("Open AI GPT", ("api_key",)),
    "GPT-4.1-mini": ("Open AI GPT", ("api_key",)),
    "Claude-4.6-Sonnet": ("Anthropic Claude", ("api_key",)),
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
        return Config(hd_strategy="Resize", hd_strategy_resize_limit=strategy_settings['resize_limit'])
    if strategy_settings['strategy'] == settings_page.ui.tr("Crop"):
        return Config(
            hd_strategy="Crop",
            hd_strategy_crop_margin=strategy_settings['crop_margin'],
            hd_strategy_crop_trigger_size=strategy_settings['crop_trigger_size'],
        )
    return Config(hd_strategy="Original")


def get_inpainter_runtime(settings_page: SettingsPage, inpainter_key: str | None = None) -> dict:
    normalized = normalize_inpainter_key(inpainter_key or settings_page.get_tool_selection("inpainter"))
    defaults = inpainter_default_settings(normalized)
    runtime_settings = settings_page.get_inpainter_runtime_settings(normalized)
    merged = dict(defaults)
    merged.update(runtime_settings)
    merged["key"] = normalized
    merged["backend"] = str(merged.get("backend") or inpainter_backend_for(normalized))
    return merged


def _missing_fields(creds: dict, required_fields: tuple[str, ...]) -> list[str]:
    return [field for field in required_fields if not (creds.get(field) or "").strip()]


def _show_missing_credentials(main: ComicTranslate, provider_name: str, missing_fields: list[str]) -> None:
    field_text = ", ".join(FIELD_LABELS.get(field, field) for field in missing_fields)
    if hasattr(main, "batch_report_ctrl"):
        main.batch_report_ctrl.register_preflight_error(
            QCoreApplication.translate("Messages", "Missing credentials for {provider}").format(provider=provider_name),
            QCoreApplication.translate("Messages", "Required fields: {fields}").format(fields=field_text),
        )
    Messages.show_missing_credentials_error(main, provider_name, field_text)


def validate_ocr(main: ComicTranslate, source_lang: str | None = None):
    settings_page = main.settings_page
    ocr_tool = settings_page.get_tool_selection("ocr")

    if not ocr_tool:
        if hasattr(main, "batch_report_ctrl"):
            main.batch_report_ctrl.register_preflight_error(
                QCoreApplication.translate("Messages", "Missing OCR tool"),
                QCoreApplication.translate("Messages", "No Text Recognition model selected."),
            )
        Messages.show_missing_tool_error(main, QCoreApplication.translate("Messages", "Text Recognition model"))
        return False

    source_lang = source_lang or main.s_combo.currentText()
    source_lang_english = main.lang_mapping.get(source_lang, source_lang)
    normalized_tool = resolve_ocr_engine(ocr_tool, source_lang_english)
    if normalized_tool in {"PaddleOCR VL", "HunyuanOCR"}:
        local_service_configs = {
            "PaddleOCR VL": (
                "PaddleOCR VL",
                settings_page.ui.tr("PaddleOCR VL Settings"),
                settings_page.get_paddleocr_vl_settings(),
            ),
            "HunyuanOCR": (
                "HunyuanOCR",
                settings_page.ui.tr("HunyuanOCR Settings"),
                settings_page.get_hunyuan_ocr_settings(),
            ),
        }
        service_name, settings_page_name, service_settings = local_service_configs[normalized_tool]
        if not service_settings.get("server_url", "").strip():
            if hasattr(main, "batch_report_ctrl"):
                main.batch_report_ctrl.register_preflight_error(
                    QCoreApplication.translate("Messages", "{service} settings missing").format(service=service_name),
                    QCoreApplication.translate("Messages", "Required fields: {fields}").format(fields=FIELD_LABELS["server_url"]),
                )
            Messages.show_missing_local_service_config_error(
                main,
                service_name,
                FIELD_LABELS["server_url"],
                settings_page_name=settings_page_name,
            )
            return False
        runtime_manager = getattr(main, "local_ocr_runtime_manager", None)
        if not isinstance(runtime_manager, LocalOCRRuntimeManager):
            runtime_manager = LocalOCRRuntimeManager()
            main.local_ocr_runtime_manager = runtime_manager
        try:
            runtime_manager.validate_engine(normalized_tool, settings_page)
        except LocalServiceSetupError as exc:
            if hasattr(main, "batch_report_ctrl"):
                main.batch_report_ctrl.register_preflight_error(
                    QCoreApplication.translate("Messages", "{service} runtime setup failed").format(service=service_name),
                    str(exc),
                )
            Messages.show_local_service_error(
                main,
                details=str(exc),
                service_name=service_name,
                settings_page_name=settings_page_name,
                error_kind="setup",
            )
            return False
        return True

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
    settings_page = main.settings_page
    settings = settings_page.get_all_settings()
    translator_tool = settings['tools']['translator']

    if not translator_tool:
        if hasattr(main, "batch_report_ctrl"):
            main.batch_report_ctrl.register_preflight_error(
                QCoreApplication.translate("Messages", "Missing translator"),
                QCoreApplication.translate("Messages", "No Translator selected."),
            )
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
        if normalized_tool == "Custom Service":
            Messages.show_custom_service_not_configured_error(main)
        elif normalized_tool == "Custom Local Server(Gemma)":
            Messages.show_custom_local_gemma_not_configured_error(main)
        else:
            _show_missing_credentials(main, provider_name, missing_fields)
        return False

    return True


def font_selected(main: ComicTranslate):
    if not main.render_settings().font_family:
        if hasattr(main, "batch_report_ctrl"):
            main.batch_report_ctrl.register_preflight_error(
                QCoreApplication.translate("Messages", "No font selected"),
                QCoreApplication.translate("Messages", "Go to Settings > Text Rendering > Font to select or import one."),
            )
        Messages.select_font_error(main)
        return False
    return True


def validate_settings(main: ComicTranslate, target_lang: str, source_lang: str | None = None):
    if not validate_ocr(main, source_lang=source_lang):
        return False
    if not validate_translator(main, target_lang):
        return False
    if not font_selected(main):
        return False
    return True
