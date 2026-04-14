import logging
import os
import shutil
import json
from dataclasses import asdict, is_dataclass
from typing import Mapping

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import QSettings, QTimer, Qt, Signal
from PySide6.QtGui import QFont, QFontDatabase

from modules.ocr.selection import (
    OCR_DEFAULT_LABEL,
    OCR_MODE_BEST_LOCAL,
    OCR_MODE_BEST_LOCAL_PLUS,
    OCR_MODE_DEFAULT,
    OCR_MODE_GEMINI,
    OCR_MODE_GOOGLE,
    OCR_MODE_HUNYUAN,
    OCR_MODE_MANGALMM,
    OCR_MODE_MICROSOFT,
    OCR_MODE_PADDLE_VL,
    OCR_OPTIMAL_LABEL,
    OCR_OPTIMAL_PLUS_LABEL,
    normalize_ocr_mode,
)
from .settings_ui import SettingsPageUI
from .gemma_local_server_page import GemmaLocalServerPage
from .hunyuan_ocr_page import HunyuanOCRPage
from .mangalmm_ocr_page import MangaLMMOCRPage
from app.update_checker import UpdateChecker
from app.shortcuts import get_default_shortcuts
from modules.utils.device import is_gpu_available
from modules.utils.paths import get_default_project_autosave_dir, get_user_data_dir
from modules.utils.notification_sound import (
    SYSTEM_SOUND_MODE,
    play_completion_sound,
)
from modules.utils.inpainting_runtime import (
    inpainter_default_settings,
    normalize_inpainter_key,
    normalized_mask_refiner_settings,
)
from modules.utils.mask_inpaint_mode import (
    DEFAULT_MASK_INPAINT_MODE,
)
from modules.utils.automatic_output import (
    DEFAULT_OUTPUT_ARCHIVE_COMPRESSION_LEVEL,
    DEFAULT_OUTPUT_ARCHIVE_FORMAT,
    DEFAULT_OUTPUT_ARCHIVE_IMAGE_FORMAT,
    DEFAULT_OUTPUT_IMAGE_FORMAT,
    DEFAULT_OUTPUT_TARGET,
    normalize_global_output_settings,
    normalize_project_output_preferences,
    resolve_automatic_output_settings,
)


logger = logging.getLogger(__name__)


class SettingsPage(QtWidgets.QWidget):
    theme_changed = Signal(str)
    font_imported = Signal(str)

    TOOL_CREDENTIAL_SERVICE_MAP = {
        "Custom Service": "Custom Service",
        "Custom Local Server(Gemma)": "Custom Local Server(Gemma)",
        "Custom Local Server": "Custom Local Server(Gemma)",
        "Custom": "Custom",
        "GPT-4.1": "Open AI GPT",
        "GPT-4.1-mini": "Open AI GPT",
        "Claude-4.6-Sonnet": "Anthropic Claude",
        "Claude-4.5-Haiku": "Anthropic Claude",
        "Gemini-2.0-Flash": "Google Gemini",
        "Gemini-2.5-Pro": "Google Gemini",
        "Gemini-3.0-Flash": "Google Gemini",
        "Deepseek-v3": "Deepseek",
        "Microsoft OCR": "Microsoft Azure",
        "Microsoft Translator": "Microsoft Azure",
        "Google Cloud Vision": "Google Cloud",
        "DeepL": "DeepL",
        "Yandex": "Yandex",
    }
    CREDENTIAL_FIELDS = {
        "Custom Service": ("api_key", "api_url", "model"),
        "Custom Local Server(Gemma)": ("api_url", "model"),
        "Custom Local Server": ("api_url", "model"),
        "Custom": ("api_key", "api_url", "model"),
        "Microsoft Azure": ("api_key_ocr", "endpoint", "api_key_translator", "region_translator"),
        "Yandex": ("api_key", "folder_id"),
        "Open AI GPT": ("api_key",),
        "Anthropic Claude": ("api_key",),
        "Google Gemini": ("api_key",),
        "Google Cloud": ("api_key",),
        "Deepseek": ("api_key",),
        "DeepL": ("api_key",),
    }

    def __init__(self, parent=None):
        super(SettingsPage, self).__init__(parent)

        self.ui = SettingsPageUI(self)
        self._loading_settings = False
        self._is_background_check = False
        self._current_language = None
        self._settings_save_timer = QTimer(self)
        self._settings_save_timer.setSingleShot(True)
        self._settings_save_timer.setInterval(250)
        self._settings_save_timer.timeout.connect(self._flush_scheduled_settings_save)

        self.update_checker = UpdateChecker()
        self.update_checker.update_available.connect(self.on_update_available)
        self.update_checker.up_to_date.connect(self.on_up_to_date)
        self.update_checker.error_occurred.connect(self.on_update_error)
        self.update_checker.download_progress.connect(self.on_download_progress)
        self.update_checker.download_finished.connect(self.on_download_finished)
        self.update_dialog = None

        self._setup_connections()

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.ui)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def _setup_connections(self):
        self.ui.theme_combo.currentTextChanged.connect(self.on_theme_changed)
        self.ui.lang_combo.currentTextChanged.connect(self.on_language_changed)
        self.ui.font_browser.sig_files_changed.connect(self.import_font)
        self.ui.check_update_button.clicked.connect(self.check_for_updates)
        self.ui.shortcuts_page.shortcut_changed.connect(self.on_shortcut_changed)
        self.ui.translator_combo.currentTextChanged.connect(self._sync_extra_context_limit)
        self.ui.raw_text_checkbox.stateChanged.connect(self._save_settings_if_not_loading)
        self.ui.translated_text_checkbox.stateChanged.connect(self._save_settings_if_not_loading)
        self.ui.inpainted_image_checkbox.stateChanged.connect(self._save_settings_if_not_loading)
        self.ui.detector_overlay_checkbox.stateChanged.connect(self._save_settings_if_not_loading)
        self.ui.raw_mask_checkbox.stateChanged.connect(self._save_settings_if_not_loading)
        self.ui.mask_overlay_checkbox.stateChanged.connect(self._save_settings_if_not_loading)
        self.ui.cleanup_mask_delta_checkbox.stateChanged.connect(self._save_settings_if_not_loading)
        self.ui.debug_metadata_checkbox.stateChanged.connect(self._save_settings_if_not_loading)
        self.ui.automatic_output_target_combo.currentIndexChanged.connect(self._save_settings_if_not_loading)
        self.ui.automatic_output_image_format_combo.currentIndexChanged.connect(self._save_settings_if_not_loading)
        self.ui.automatic_output_archive_format_combo.currentIndexChanged.connect(self._save_settings_if_not_loading)
        self.ui.automatic_output_archive_image_format_combo.currentIndexChanged.connect(self._save_settings_if_not_loading)
        self.ui.automatic_output_archive_level_spinbox.valueChanged.connect(self._save_settings_if_not_loading)
        self.ui.user_dictionaries_page.changed.connect(self._save_settings_if_not_loading)
        self.ui.notifications_page.changed.connect(self._save_settings_if_not_loading)
        self.ui.notifications_page.test_requested.connect(self.play_test_completion_sound)
        self._connect_live_save_signals()

    def _save_settings_if_not_loading(self, *_args):
        if self._loading_settings:
            return
        self._settings_save_timer.start()

    def _flush_scheduled_settings_save(self):
        if self._loading_settings:
            return
        self.save_settings()

    def _connect_live_save_signals(self):
        text_changed_widgets = [
            self.ui.extra_context,
            self.ui.project_autosave_folder_input,
            self.ui.paddleocr_vl_server_url_input,
            self.ui.hunyuan_ocr_server_url_input,
            self.ui.mangalmm_ocr_server_url_input,
        ]
        for widget in text_changed_widgets:
            signal = getattr(widget, "textChanged", None)
            if signal is not None:
                signal.connect(self._save_settings_if_not_loading)

        combo_widgets = [
            self.ui.theme_combo,
            self.ui.translator_combo,
            self.ui.ocr_combo,
            self.ui.detector_combo,
            self.ui.inpainter_combo,
            self.ui.inpaint_strategy_combo,
            self.ui.inpainter_size_combo,
            self.ui.inpainter_device_combo,
            self.ui.inpainter_precision_combo,
        ]
        for widget in combo_widgets:
            widget.currentTextChanged.connect(self._save_settings_if_not_loading)

        checkbox_widgets = [
            self.ui.use_gpu_checkbox,
            self.ui.image_checkbox,
            self.ui.uppercase_checkbox,
            self.ui.save_keys_checkbox,
            self.ui.paddleocr_vl_prettify_checkbox,
            self.ui.paddleocr_vl_visualize_checkbox,
            self.ui.hunyuan_ocr_raw_response_logging_checkbox,
            self.ui.mangalmm_ocr_raw_response_logging_checkbox,
            self.ui.mangalmm_ocr_safe_resize_checkbox,
            self.ui.gemma_raw_response_logging_checkbox,
        ]
        for widget in checkbox_widgets:
            widget.stateChanged.connect(self._save_settings_if_not_loading)

        spin_widgets = [
            self.ui.resize_spinbox,
            self.ui.crop_margin_spinbox,
            self.ui.crop_trigger_spinbox,
            self.ui.min_font_spinbox,
            self.ui.max_font_spinbox,
            self.ui.project_autosave_interval_spinbox,
            self.ui.paddleocr_vl_max_new_tokens_spinbox,
            self.ui.paddleocr_vl_parallel_workers_spinbox,
            self.ui.hunyuan_ocr_max_completion_tokens_spinbox,
            self.ui.hunyuan_ocr_parallel_workers_spinbox,
            self.ui.hunyuan_ocr_request_timeout_spinbox,
            self.ui.mangalmm_ocr_max_completion_tokens_spinbox,
            self.ui.mangalmm_ocr_parallel_workers_spinbox,
            self.ui.mangalmm_ocr_request_timeout_spinbox,
            self.ui.mangalmm_ocr_max_pixels_spinbox,
            self.ui.mangalmm_ocr_max_long_side_spinbox,
            self.ui.gemma_chunk_size_spinbox,
            self.ui.gemma_max_completion_tokens_spinbox,
            self.ui.gemma_request_timeout_spinbox,
            self.ui.gemma_temperature_spinbox,
            self.ui.gemma_top_k_spinbox,
            self.ui.gemma_top_p_spinbox,
            self.ui.gemma_min_p_spinbox,
        ]
        for widget in spin_widgets:
            widget.valueChanged.connect(self._save_settings_if_not_loading)

        for widget in self.ui.credential_widgets.values():
            signal = getattr(widget, "textChanged", None)
            if signal is not None:
                signal.connect(self._save_settings_if_not_loading)

    def _sync_extra_context_limit(self, translator: str | None = None):
        raw_service = translator if translator is not None else self.ui.translator_combo.currentText()
        normalized = self.ui.value_mappings.get(raw_service, raw_service)
        self.ui.llms_page.set_extra_context_unlimited(normalized == "Custom")

    def on_theme_changed(self, theme: str):
        self.theme_changed.emit(theme)

    def get_language(self):
        return self.ui.lang_combo.currentText()

    def get_theme(self):
        return self.ui.theme_combo.currentText()

    def get_tool_selection(self, tool_type):
        tool_combos = {
            "translator": self.ui.translator_combo,
            "ocr": self.ui.ocr_combo,
            "inpainter": self.ui.inpainter_combo,
            "detector": self.ui.detector_combo,
        }
        combo = tool_combos[tool_type]
        if tool_type == "ocr":
            current_data = combo.currentData()
            if isinstance(current_data, str) and current_data.strip():
                return self._normalize_ocr_mode_value(current_data)
            return self._normalize_ocr_mode_value(combo.currentText())
        if tool_type == "inpainter":
            return "lama_large_512px"
        if tool_type == "detector":
            return "RT-DETR-v2"
        return combo.currentText()

    def get_mask_inpaint_mode(self) -> str:
        return DEFAULT_MASK_INPAINT_MODE

    def get_tool_display_text(self, tool_type: str) -> str:
        tool_combos = {
            "translator": self.ui.translator_combo,
            "ocr": self.ui.ocr_combo,
            "inpainter": self.ui.inpainter_combo,
            "detector": self.ui.detector_combo,
        }
        return tool_combos[tool_type].currentText()

    def get_ocr_mode_label(self, mode_key: str | None = None) -> str:
        normalized = self._normalize_ocr_mode_value(mode_key or self.get_tool_selection("ocr"))
        index = self.ui.ocr_combo.findData(normalized)
        if index != -1:
            return self.ui.ocr_combo.itemText(index)
        return normalized

    def _normalized_ocr_aliases(self) -> dict[str, str]:
        return {
            self.ui.tr("Default"): OCR_MODE_DEFAULT,
            self.ui.tr(OCR_DEFAULT_LABEL): OCR_MODE_DEFAULT,
            self.ui.tr("Optimal (HunyuanOCR / PaddleOCR VL)"): OCR_MODE_BEST_LOCAL,
            self.ui.tr(OCR_OPTIMAL_LABEL): OCR_MODE_BEST_LOCAL,
            self.ui.tr("Optimal+ (HunyuanOCR / MangaLMM / PaddleOCR VL)"): OCR_MODE_BEST_LOCAL_PLUS,
            self.ui.tr(OCR_OPTIMAL_PLUS_LABEL): OCR_MODE_BEST_LOCAL_PLUS,
            self.ui.tr("Microsoft OCR"): OCR_MODE_MICROSOFT,
            self.ui.tr("Google Cloud Vision"): OCR_MODE_GOOGLE,
            self.ui.tr("Gemini-2.0-Flash"): OCR_MODE_GEMINI,
            self.ui.tr("PaddleOCR VL"): OCR_MODE_PADDLE_VL,
            self.ui.tr("HunyuanOCR"): OCR_MODE_HUNYUAN,
            self.ui.tr("MangaLMM"): OCR_MODE_MANGALMM,
        }

    def _normalize_ocr_mode_value(self, raw_value: str | None) -> str:
        raw = str(raw_value or "").strip()
        if not raw:
            return OCR_MODE_DEFAULT
        return normalize_ocr_mode(self._normalized_ocr_aliases().get(raw, raw))

    def _set_ocr_mode(self, raw_value: str | None) -> None:
        normalized = self._normalize_ocr_mode_value(raw_value)
        index = self.ui.ocr_combo.findData(normalized)
        if index != -1:
            self.ui.ocr_combo.setCurrentIndex(index)
            return
        fallback = self.ui.ocr_combo.findData(OCR_MODE_DEFAULT)
        self.ui.ocr_combo.setCurrentIndex(fallback if fallback != -1 else 0)

    def is_gpu_enabled(self):
        if not is_gpu_available():
            return False
        return self.ui.use_gpu_checkbox.isChecked()

    def get_llm_settings(self):
        return {
            "extra_context": self.ui.extra_context.toPlainText(),
            "image_input_enabled": self.ui.image_checkbox.isChecked(),
        }

    def get_paddleocr_vl_settings(self):
        server_url = self.ui.paddleocr_vl_server_url_input.text().strip()
        if not server_url:
            server_url = self.ui.paddleocr_vl_page.DEFAULT_SERVER_URL

        return {
            "server_url": server_url,
            "prettify_markdown": self.ui.paddleocr_vl_prettify_checkbox.isChecked(),
            "visualize": self.ui.paddleocr_vl_visualize_checkbox.isChecked(),
            "max_new_tokens": int(self.ui.paddleocr_vl_max_new_tokens_spinbox.value()),
            "parallel_workers": int(self.ui.paddleocr_vl_parallel_workers_spinbox.value()),
        }

    def get_gemma_local_server_settings(self):
        return {
            "chunk_size": int(self.ui.gemma_chunk_size_spinbox.value()),
            "max_completion_tokens": int(self.ui.gemma_max_completion_tokens_spinbox.value()),
            "request_timeout_sec": int(self.ui.gemma_request_timeout_spinbox.value()),
            "temperature": float(self.ui.gemma_temperature_spinbox.value()),
            "top_k": int(self.ui.gemma_top_k_spinbox.value()),
            "top_p": float(self.ui.gemma_top_p_spinbox.value()),
            "min_p": float(self.ui.gemma_min_p_spinbox.value()),
            "raw_response_logging": self.ui.gemma_raw_response_logging_checkbox.isChecked(),
        }

    def get_hunyuan_ocr_settings(self):
        server_url = self.ui.hunyuan_ocr_server_url_input.text().strip()
        if not server_url:
            server_url = self.ui.hunyuan_ocr_page.DEFAULT_SERVER_URL

        return {
            "server_url": server_url,
            "max_completion_tokens": int(self.ui.hunyuan_ocr_max_completion_tokens_spinbox.value()),
            "parallel_workers": int(self.ui.hunyuan_ocr_parallel_workers_spinbox.value()),
            "request_timeout_sec": int(self.ui.hunyuan_ocr_request_timeout_spinbox.value()),
            "raw_response_logging": self.ui.hunyuan_ocr_raw_response_logging_checkbox.isChecked(),
        }

    def get_mangalmm_ocr_settings(self):
        server_url = self.ui.mangalmm_ocr_server_url_input.text().strip()
        if not server_url:
            server_url = self.ui.mangalmm_ocr_page.DEFAULT_SERVER_URL

        return {
            "server_url": server_url,
            "max_completion_tokens": int(self.ui.mangalmm_ocr_max_completion_tokens_spinbox.value()),
            "parallel_workers": int(self.ui.mangalmm_ocr_parallel_workers_spinbox.value()),
            "request_timeout_sec": int(self.ui.mangalmm_ocr_request_timeout_spinbox.value()),
            "raw_response_logging": self.ui.mangalmm_ocr_raw_response_logging_checkbox.isChecked(),
            "safe_resize": self.ui.mangalmm_ocr_safe_resize_checkbox.isChecked(),
            "max_pixels": int(self.ui.mangalmm_ocr_max_pixels_spinbox.value()),
            "max_long_side": int(self.ui.mangalmm_ocr_max_long_side_spinbox.value()),
        }

    def get_ocr_generic_settings(self):
        settings = {
            "manga_expansion_percentage": 7,
            "crop_padding_ratio": 0.05,
            "ppocr_retry_crop_ratio_x": 0.06,
            "ppocr_retry_crop_ratio_y": 0.10,
        }
        benchmark_overlay = getattr(self, "_benchmark_ocr_generic_settings", None)
        if isinstance(benchmark_overlay, dict):
            settings.update(benchmark_overlay)
        return settings

    def get_mask_refiner_settings(self):
        return normalized_mask_refiner_settings(
            {
                "mask_refiner": "legacy_bbox",
                "mask_inpaint_mode": self.get_mask_inpaint_mode(),
                "keep_existing_lines": False,
            }
        )

    def get_inpainter_runtime_settings(self, inpainter_key: str | None = None):
        normalized = normalize_inpainter_key(inpainter_key or self.get_tool_selection("inpainter"))
        defaults = inpainter_default_settings(normalized)
        runtime = {
            "backend": defaults.get("backend", "torch"),
            "device": self.ui.inpainter_device_combo.currentText() or defaults.get("device", "cuda"),
            "inpaint_size": int(self.ui.inpainter_size_combo.currentText() or defaults.get("inpaint_size", 2048)),
            "precision": self.ui.inpainter_precision_combo.currentText() or defaults.get("precision", "fp32"),
        }
        if normalized != "lama_large_512px":
            runtime["precision"] = defaults.get("precision", "fp32")
        return runtime

    def get_export_settings(self):
        owner = self.window()
        title_bar = getattr(owner, "title_bar", None)
        settings = QSettings("ComicLabs", "ComicTranslate")
        settings.beginGroup("export")
        persisted_autosave_enabled = settings.value("project_autosave_enabled", False, type=bool)
        settings.endGroup()
        if title_bar is not None:
            autosave_enabled = bool(title_bar.autosave_switch.isChecked())
        else:
            autosave_enabled = bool(persisted_autosave_enabled)
        autosave_folder = self.ui.project_autosave_folder_input.text().strip()
        if not autosave_folder:
            autosave_folder = get_default_project_autosave_dir()
        auto_export_source_txt = bool(
            getattr(owner, "auto_export_source_txt_checkbox", None)
            and owner.auto_export_source_txt_checkbox.isChecked()
        )
        auto_export_source_md = bool(
            getattr(owner, "auto_export_source_md_checkbox", None)
            and owner.auto_export_source_md_checkbox.isChecked()
        )
        auto_export_translation_txt = bool(
            getattr(owner, "auto_export_translation_txt_checkbox", None)
            and owner.auto_export_translation_txt_checkbox.isChecked()
        )
        auto_export_translation_md = bool(
            getattr(owner, "auto_export_translation_md_checkbox", None)
            and owner.auto_export_translation_md_checkbox.isChecked()
        )
        return {
            "export_raw_text": self.ui.raw_text_checkbox.isChecked(),
            "export_translated_text": self.ui.translated_text_checkbox.isChecked(),
            "export_inpainted_image": self.ui.inpainted_image_checkbox.isChecked(),
            "export_detector_overlay": self.ui.detector_overlay_checkbox.isChecked(),
            "export_raw_mask": self.ui.raw_mask_checkbox.isChecked(),
            "export_mask_overlay": self.ui.mask_overlay_checkbox.isChecked(),
            "export_cleanup_mask_delta": self.ui.cleanup_mask_delta_checkbox.isChecked(),
            "export_debug_metadata": self.ui.debug_metadata_checkbox.isChecked(),
            "automatic_output_target": str(
                self.ui.automatic_output_target_combo.currentData() or DEFAULT_OUTPUT_TARGET
            ),
            "automatic_output_image_format": str(
                self.ui.automatic_output_image_format_combo.currentData() or DEFAULT_OUTPUT_IMAGE_FORMAT
            ),
            "automatic_output_archive_format": str(
                self.ui.automatic_output_archive_format_combo.currentData() or DEFAULT_OUTPUT_ARCHIVE_FORMAT
            ),
            "automatic_output_archive_image_format": str(
                self.ui.automatic_output_archive_image_format_combo.currentData()
                or DEFAULT_OUTPUT_ARCHIVE_IMAGE_FORMAT
            ),
            "automatic_output_archive_compression_level": int(
                self.ui.automatic_output_archive_level_spinbox.value()
            ),
            "project_autosave_enabled": autosave_enabled,
            "project_autosave_interval_min": int(self.ui.project_autosave_interval_spinbox.value()),
            "project_autosave_folder": autosave_folder,
            "auto_export_source_txt": auto_export_source_txt,
            "auto_export_source_md": auto_export_source_md,
            "auto_export_translation_txt": auto_export_translation_txt,
            "auto_export_translation_md": auto_export_translation_md,
        }

    def get_resolved_automatic_output_settings(
        self,
        project_preferences: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        project = normalize_project_output_preferences(project_preferences)
        return resolve_automatic_output_settings(self.get_export_settings(), project)

    def get_dictionary_settings(self) -> dict[str, list[dict]]:
        return {
            "ocr_substitutions": self.ui.user_dictionaries_page.get_ocr_rules(),
            "translation_substitutions": self.ui.user_dictionaries_page.get_translation_rules(),
        }

    def get_notification_settings(self) -> dict[str, object]:
        return self.ui.notifications_page.get_notification_settings()

    def get_ocr_result_dictionary_rules(self) -> list[dict]:
        return self.get_dictionary_settings()["ocr_substitutions"]

    def get_translation_result_dictionary_rules(self) -> list[dict]:
        return self.get_dictionary_settings()["translation_substitutions"]

    def _normalize_service_name(self, raw_service: str) -> str:
        normalized = self.ui.value_mappings.get(raw_service, raw_service)
        return self.TOOL_CREDENTIAL_SERVICE_MAP.get(normalized, normalized)

    def _resolve_legacy_custom_translator(self) -> str:
        settings = QSettings("ComicLabs", "ComicTranslate")
        legacy_api_key = settings.value("credentials/Custom Service_api_key", "", type=str).strip()
        if not legacy_api_key:
            legacy_api_key = settings.value("credentials/Custom_api_key", "", type=str).strip()
        return "Custom Service" if legacy_api_key else "Custom Local Server(Gemma)"

    def _load_credential_value(
        self,
        settings: QSettings,
        service_name: str,
        field: str,
        save_keys: bool,
    ) -> str:
        if service_name == "Custom Local Server(Gemma)" and field in ("api_url", "model"):
            default_value = (
                GemmaLocalServerPage.DEFAULT_ENDPOINT_URL
                if field == "api_url"
                else GemmaLocalServerPage.DEFAULT_MODEL
            )
            if not save_keys:
                return default_value

            value = settings.value(f"{service_name}_{field}", "", type=str)
            if value:
                return value

            legacy_value = settings.value(f"Custom Local Server_{field}", "", type=str)
            if legacy_value:
                return legacy_value
            return settings.value(f"Custom_{field}", default_value, type=str)

        if not save_keys:
            return ""

        value = settings.value(f"{service_name}_{field}", "", type=str)
        if value:
            return value

        if service_name == "Custom Service" and field in ("api_key", "api_url", "model"):
            return settings.value(f"Custom_{field}", "", type=str)
        return ""

    def get_credentials(self, service: str = ""):
        save_keys = self.ui.save_keys_checkbox.isChecked()

        def _text_or_none(widget_key):
            widget = self.ui.credential_widgets.get(widget_key)
            return widget.text() if widget is not None else None

        if service:
            normalized = self._normalize_service_name(service)
            creds = {"save_key": save_keys}
            for field in self.CREDENTIAL_FIELDS.get(normalized, ("api_key",)):
                creds[field] = _text_or_none(f"{normalized}_{field}")
            return creds

        return {service_name: self.get_credentials(service_name) for service_name in self.ui.credential_services}

    def get_hd_strategy_settings(self):
        strategy = self.ui.inpaint_strategy_combo.currentText()
        settings = {"strategy": strategy}

        if strategy == self.ui.tr("Resize"):
            settings["resize_limit"] = self.ui.resize_spinbox.value()
        elif strategy == self.ui.tr("Crop"):
            settings["crop_margin"] = self.ui.crop_margin_spinbox.value()
            settings["crop_trigger_size"] = self.ui.crop_trigger_spinbox.value()

        return settings

    def get_all_settings(self):
        return {
            "language": self.get_language(),
            "theme": self.get_theme(),
            "tools": {
                "translator": self.get_tool_selection("translator"),
                "ocr": self.get_tool_selection("ocr"),
                "detector": self.get_tool_selection("detector"),
                "inpainter": self.get_tool_selection("inpainter"),
                "use_gpu": self.is_gpu_enabled(),
                "hd_strategy": self.get_hd_strategy_settings(),
                "mask_refiner_settings": self.get_mask_refiner_settings(),
                "inpainter_runtime": self.get_inpainter_runtime_settings(),
            },
            "paddleocr_vl": self.get_paddleocr_vl_settings(),
            "hunyuan_ocr": self.get_hunyuan_ocr_settings(),
            "mangalmm_ocr": self.get_mangalmm_ocr_settings(),
            "gemma_local_server": self.get_gemma_local_server_settings(),
            "llm": self.get_llm_settings(),
            "export": self.get_export_settings(),
            "notifications": self.get_notification_settings(),
            "shortcuts": self.ui.shortcuts_page.get_shortcuts(),
            "credentials": self.get_credentials(),
            "save_keys": self.ui.save_keys_checkbox.isChecked(),
        }

    def on_shortcut_changed(self, shortcut_id: str, sequence: str) -> None:
        if self._loading_settings:
            return

        settings = QSettings("ComicLabs", "ComicTranslate")
        settings.beginGroup("shortcuts")
        settings.setValue(shortcut_id, sequence)
        settings.endGroup()

        owner = self.window()
        shortcut_ctrl = getattr(owner, "shortcut_ctrl", None)
        if shortcut_ctrl is not None:
            shortcut_ctrl.apply_shortcuts()

    def import_font(self, file_paths: list[str]):
        file_paths = [
            file_path
            for file_path in file_paths
            if file_path.lower().endswith((".ttf", ".ttc", ".otf", ".woff", ".woff2"))
        ]

        user_font_dir = os.path.join(get_user_data_dir(), "fonts")
        if not os.path.exists(user_font_dir):
            os.makedirs(user_font_dir, exist_ok=True)

        if file_paths:
            loaded_families = []
            for src in file_paths:
                dst = os.path.join(user_font_dir, os.path.basename(src))
                if os.path.normcase(src) != os.path.normcase(dst):
                    shutil.copy(src, dst)
                font_family = self.add_font_family(dst)
                if font_family:
                    loaded_families.append(font_family)

            if loaded_families:
                self.font_imported.emit(loaded_families[0])

    def select_color(self, outline=False):
        default_color = QtGui.QColor("#000000") if not outline else QtGui.QColor("#FFFFFF")
        color_dialog = QtWidgets.QColorDialog()
        color_dialog.setCurrentColor(default_color)

        if color_dialog.exec() == QtWidgets.QDialog.Accepted:
            color = color_dialog.selectedColor()
            if color.isValid():
                button = self.ui.color_button if not outline else self.ui.outline_color_button
                button.setStyleSheet(
                    f"background-color: {color.name()}; border: none; border-radius: 5px;"
                )
                button.setProperty("selected_color", color.name())

    def save_settings(self):
        settings = QSettings("ComicLabs", "ComicTranslate")
        all_settings = self.get_all_settings()

        def process_group(group_key, group_value, settings_obj: QSettings):
            if is_dataclass(group_value):
                group_value = asdict(group_value)
            if isinstance(group_value, dict):
                settings_obj.beginGroup(group_key)
                for sub_key, sub_value in group_value.items():
                    process_group(sub_key, sub_value, settings_obj)
                settings_obj.endGroup()
            else:
                mapped_value = self.ui.value_mappings.get(group_value, group_value)
                settings_obj.setValue(group_key, mapped_value)

        for key, value in all_settings.items():
            process_group(key, value, settings)

        settings.beginGroup("tools")
        settings.beginGroup("mask_refiner_settings")
        for stale_key in (
            "ctd_detect_size",
            "ctd_det_rearrange_max_batches",
            "ctd_device",
            "ctd_font_size_multiplier",
            "ctd_font_size_max",
            "ctd_font_size_min",
            "ctd_mask_dilate_size",
            "keep_existing_lines",
        ):
            settings.remove(stale_key)
        settings.endGroup()
        settings.endGroup()

        settings.beginGroup("export")
        settings.remove("auto_save")
        settings.remove("archive_save_as")
        settings.remove("automatic_output_format")
        settings.remove("automatic_output_preset")
        settings.remove("automatic_output_png_compression_level")
        settings.remove("automatic_output_jpg_quality")
        settings.remove("automatic_output_webp_quality")
        settings.endGroup()

        dictionaries = self.get_dictionary_settings()
        settings.beginGroup("dictionaries")
        settings.setValue(
            "ocr_substitutions_json",
            json.dumps(dictionaries["ocr_substitutions"], ensure_ascii=False),
        )
        settings.setValue(
            "translation_substitutions_json",
            json.dumps(dictionaries["translation_substitutions"], ensure_ascii=False),
        )
        settings.endGroup()

        credentials = self.get_credentials()
        save_keys = self.ui.save_keys_checkbox.isChecked()
        settings.beginGroup("credentials")
        settings.setValue("save_keys", save_keys)
        if save_keys:
            for service, cred in credentials.items():
                translated_service = self._normalize_service_name(service)
                for field in self.CREDENTIAL_FIELDS.get(translated_service, ("api_key",)):
                    settings.setValue(f"{translated_service}_{field}", cred.get(field, ""))
        else:
            settings.remove("")
        settings.endGroup()

    def load_settings(self):
        self._loading_settings = True
        settings = QSettings("ComicLabs", "ComicTranslate")

        language = settings.value("language", "English")
        translated_language = self.ui.reverse_mappings.get(language, language)
        lang_index = self.ui.lang_combo.findText(translated_language)
        self.ui.lang_combo.setCurrentIndex(lang_index if lang_index != -1 else 0)

        theme = settings.value("theme", "Dark")
        translated_theme = self.ui.reverse_mappings.get(theme, theme)
        theme_index = self.ui.theme_combo.findText(translated_theme)
        self.ui.theme_combo.setCurrentIndex(theme_index if theme_index != -1 else 0)
        self.theme_changed.emit(translated_theme)

        translator = settings.value("tools/translator", "Custom Local Server(Gemma)", type=str)
        if translator == "Custom":
            translator = self._resolve_legacy_custom_translator()
        elif translator == "Custom Local Server":
            translator = "Custom Local Server(Gemma)"
        elif translator == "Claude-4.5-Sonnet":
            translator = "Claude-4.6-Sonnet"

        settings.beginGroup("tools")
        translated_translator = self.ui.reverse_mappings.get(translator, translator)
        if self.ui.translator_combo.findText(translated_translator) != -1:
            self.ui.translator_combo.setCurrentIndex(self.ui.translator_combo.findText(translated_translator))
        else:
            self.ui.translator_combo.setCurrentIndex(-1)
        self._sync_extra_context_limit(translated_translator)

        ocr = settings.value("ocr", OCR_MODE_PADDLE_VL, type=str)
        self._set_ocr_mode(ocr)

        inpainter = "lama_large_512px"
        translated_inpainter = self.ui.reverse_mappings.get(inpainter, inpainter)
        if self.ui.inpainter_combo.findText(translated_inpainter) != -1:
            self.ui.inpainter_combo.setCurrentIndex(self.ui.inpainter_combo.findText(translated_inpainter))
        else:
            self.ui.inpainter_combo.setCurrentIndex(0)

        detector = "RT-DETR-v2"
        translated_detector = self.ui.reverse_mappings.get(detector, detector)
        if self.ui.detector_combo.findText(translated_detector) != -1:
            self.ui.detector_combo.setCurrentIndex(self.ui.detector_combo.findText(translated_detector))
        else:
            self.ui.detector_combo.setCurrentIndex(0)

        self.ui.tools_page._update_inpainter_runtime_widgets(self.ui.inpainter_combo.currentIndex())

        if is_gpu_available():
            self.ui.use_gpu_checkbox.setChecked(settings.value("use_gpu", True, type=bool))
        else:
            self.ui.use_gpu_checkbox.setChecked(False)

        settings.beginGroup("hd_strategy")
        strategy = settings.value("strategy", "Resize")
        translated_strategy = self.ui.reverse_mappings.get(strategy, strategy)
        if self.ui.inpaint_strategy_combo.findText(translated_strategy) != -1:
            self.ui.inpaint_strategy_combo.setCurrentIndex(self.ui.inpaint_strategy_combo.findText(translated_strategy))
        else:
            self.ui.inpaint_strategy_combo.setCurrentIndex(0)

        if strategy == "Resize":
            self.ui.resize_spinbox.setValue(settings.value("resize_limit", 960, type=int))
        elif strategy == "Crop":
            self.ui.crop_margin_spinbox.setValue(settings.value("crop_margin", 512, type=int))
            self.ui.crop_trigger_spinbox.setValue(settings.value("crop_trigger_size", 512, type=int))
        settings.endGroup()

        runtime_defaults = inpainter_default_settings(inpainter)
        settings.beginGroup("inpainter_runtime")
        self.ui.inpainter_size_combo.setCurrentText(str(settings.value("inpaint_size", runtime_defaults.get("inpaint_size", 2048), type=int)))
        self.ui.inpainter_device_combo.setCurrentText(settings.value("device", runtime_defaults.get("device", "cuda"), type=str))
        self.ui.inpainter_precision_combo.setCurrentText(settings.value("precision", runtime_defaults.get("precision", "fp32"), type=str))
        settings.endGroup()
        settings.endGroup()

        settings.beginGroup("paddleocr_vl")
        self.ui.paddleocr_vl_server_url_input.setText(
            settings.value(
                "server_url",
                self.ui.paddleocr_vl_page.DEFAULT_SERVER_URL,
                type=str,
            )
        )
        self.ui.paddleocr_vl_prettify_checkbox.setChecked(
            settings.value("prettify_markdown", False, type=bool)
        )
        self.ui.paddleocr_vl_visualize_checkbox.setChecked(
            settings.value("visualize", False, type=bool)
        )
        self.ui.paddleocr_vl_max_new_tokens_spinbox.setValue(
            settings.value(
                "max_new_tokens",
                self.ui.paddleocr_vl_page.DEFAULT_MAX_NEW_TOKENS,
                type=int,
            )
        )
        self.ui.paddleocr_vl_parallel_workers_spinbox.setValue(
            settings.value(
                "parallel_workers",
                self.ui.paddleocr_vl_page.DEFAULT_PARALLEL_WORKERS,
                type=int,
            )
        )
        settings.endGroup()

        settings.beginGroup("hunyuan_ocr")
        self.ui.hunyuan_ocr_server_url_input.setText(
            settings.value(
                "server_url",
                HunyuanOCRPage.DEFAULT_SERVER_URL,
                type=str,
            )
        )
        self.ui.hunyuan_ocr_max_completion_tokens_spinbox.setValue(
            settings.value(
                "max_completion_tokens",
                HunyuanOCRPage.DEFAULT_MAX_COMPLETION_TOKENS,
                type=int,
            )
        )
        self.ui.hunyuan_ocr_parallel_workers_spinbox.setValue(
            settings.value(
                "parallel_workers",
                HunyuanOCRPage.DEFAULT_PARALLEL_WORKERS,
                type=int,
            )
        )
        self.ui.hunyuan_ocr_request_timeout_spinbox.setValue(
            settings.value(
                "request_timeout_sec",
                HunyuanOCRPage.DEFAULT_REQUEST_TIMEOUT_SEC,
                type=int,
            )
        )
        self.ui.hunyuan_ocr_raw_response_logging_checkbox.setChecked(
            settings.value("raw_response_logging", False, type=bool)
        )
        settings.endGroup()

        settings.beginGroup("mangalmm_ocr")
        self.ui.mangalmm_ocr_server_url_input.setText(
            settings.value(
                "server_url",
                MangaLMMOCRPage.DEFAULT_SERVER_URL,
                type=str,
            )
        )
        self.ui.mangalmm_ocr_max_completion_tokens_spinbox.setValue(
            settings.value(
                "max_completion_tokens",
                MangaLMMOCRPage.DEFAULT_MAX_COMPLETION_TOKENS,
                type=int,
            )
        )
        self.ui.mangalmm_ocr_parallel_workers_spinbox.setValue(
            settings.value(
                "parallel_workers",
                MangaLMMOCRPage.DEFAULT_PARALLEL_WORKERS,
                type=int,
            )
        )
        self.ui.mangalmm_ocr_request_timeout_spinbox.setValue(
            settings.value(
                "request_timeout_sec",
                MangaLMMOCRPage.DEFAULT_REQUEST_TIMEOUT_SEC,
                type=int,
            )
        )
        self.ui.mangalmm_ocr_raw_response_logging_checkbox.setChecked(
            settings.value("raw_response_logging", False, type=bool)
        )
        self.ui.mangalmm_ocr_safe_resize_checkbox.setChecked(
            settings.value(
                "safe_resize",
                MangaLMMOCRPage.DEFAULT_SAFE_RESIZE,
                type=bool,
            )
        )
        self.ui.mangalmm_ocr_max_pixels_spinbox.setValue(
            settings.value(
                "max_pixels",
                MangaLMMOCRPage.DEFAULT_MAX_PIXELS,
                type=int,
            )
        )
        self.ui.mangalmm_ocr_max_long_side_spinbox.setValue(
            settings.value(
                "max_long_side",
                MangaLMMOCRPage.DEFAULT_MAX_LONG_SIDE,
                type=int,
            )
        )
        settings.endGroup()

        settings.beginGroup("gemma_local_server")
        self.ui.gemma_chunk_size_spinbox.setValue(
            settings.value(
                "chunk_size",
                GemmaLocalServerPage.DEFAULT_CHUNK_SIZE,
                type=int,
            )
        )
        self.ui.gemma_max_completion_tokens_spinbox.setValue(
            settings.value(
                "max_completion_tokens",
                GemmaLocalServerPage.DEFAULT_MAX_COMPLETION_TOKENS,
                type=int,
            )
        )
        self.ui.gemma_request_timeout_spinbox.setValue(
            settings.value(
                "request_timeout_sec",
                GemmaLocalServerPage.DEFAULT_REQUEST_TIMEOUT_SEC,
                type=int,
            )
        )
        self.ui.gemma_temperature_spinbox.setValue(
            settings.value("temperature", GemmaLocalServerPage.DEFAULT_TEMPERATURE, type=float)
        )
        self.ui.gemma_top_k_spinbox.setValue(
            settings.value("top_k", GemmaLocalServerPage.DEFAULT_TOP_K, type=int)
        )
        self.ui.gemma_top_p_spinbox.setValue(
            settings.value("top_p", GemmaLocalServerPage.DEFAULT_TOP_P, type=float)
        )
        self.ui.gemma_min_p_spinbox.setValue(
            settings.value("min_p", GemmaLocalServerPage.DEFAULT_MIN_P, type=float)
        )
        self.ui.gemma_raw_response_logging_checkbox.setChecked(
            settings.value("raw_response_logging", False, type=bool)
        )
        settings.endGroup()

        settings.beginGroup("llm")
        self.ui.extra_context.setPlainText(settings.value("extra_context", ""))
        self.ui.image_checkbox.setChecked(settings.value("image_input_enabled", False, type=bool))
        settings.endGroup()

        settings.beginGroup("export")
        self.ui.raw_text_checkbox.setChecked(settings.value("export_raw_text", False, type=bool))
        self.ui.translated_text_checkbox.setChecked(
            settings.value("export_translated_text", False, type=bool)
        )
        self.ui.inpainted_image_checkbox.setChecked(
            settings.value("export_inpainted_image", False, type=bool)
        )
        self.ui.detector_overlay_checkbox.setChecked(
            settings.value("export_detector_overlay", False, type=bool)
        )
        self.ui.raw_mask_checkbox.setChecked(
            settings.value("export_raw_mask", False, type=bool)
        )
        self.ui.mask_overlay_checkbox.setChecked(
            settings.value("export_mask_overlay", False, type=bool)
        )
        self.ui.cleanup_mask_delta_checkbox.setChecked(
            settings.value("export_cleanup_mask_delta", False, type=bool)
        )
        self.ui.debug_metadata_checkbox.setChecked(
            settings.value("export_debug_metadata", False, type=bool)
        )
        normalized_output_settings = normalize_global_output_settings(
            {
                "automatic_output_target": settings.value(
                    "automatic_output_target",
                    DEFAULT_OUTPUT_TARGET,
                    type=str,
                ),
                "automatic_output_image_format": settings.value(
                    "automatic_output_image_format",
                    settings.value("automatic_output_format", DEFAULT_OUTPUT_IMAGE_FORMAT, type=str),
                    type=str,
                ),
                "automatic_output_archive_format": settings.value(
                    "automatic_output_archive_format",
                    DEFAULT_OUTPUT_ARCHIVE_FORMAT,
                    type=str,
                ),
                "automatic_output_archive_image_format": settings.value(
                    "automatic_output_archive_image_format",
                    DEFAULT_OUTPUT_ARCHIVE_IMAGE_FORMAT,
                    type=str,
                ),
                "automatic_output_archive_compression_level": settings.value(
                    "automatic_output_archive_compression_level",
                    DEFAULT_OUTPUT_ARCHIVE_COMPRESSION_LEVEL,
                    type=int,
                ),
            }
        )
        target_index = self.ui.automatic_output_target_combo.findData(
            normalized_output_settings["automatic_output_target"]
        )
        self.ui.automatic_output_target_combo.setCurrentIndex(max(target_index, 0))
        image_format_index = self.ui.automatic_output_image_format_combo.findData(
            normalized_output_settings["automatic_output_image_format"]
        )
        self.ui.automatic_output_image_format_combo.setCurrentIndex(max(image_format_index, 0))
        archive_format_index = self.ui.automatic_output_archive_format_combo.findData(
            normalized_output_settings["automatic_output_archive_format"]
        )
        self.ui.automatic_output_archive_format_combo.setCurrentIndex(max(archive_format_index, 0))
        archive_image_format_index = self.ui.automatic_output_archive_image_format_combo.findData(
            normalized_output_settings["automatic_output_archive_image_format"]
        )
        self.ui.automatic_output_archive_image_format_combo.setCurrentIndex(
            max(archive_image_format_index, 0)
        )
        self.ui.automatic_output_archive_level_spinbox.setValue(
            int(normalized_output_settings["automatic_output_archive_compression_level"])
        )
        autosave_enabled = settings.value("project_autosave_enabled", False, type=bool)
        owner = self.parent()
        title_bar = getattr(owner, "title_bar", None)
        if title_bar is not None:
            title_bar.set_autosave_checked(bool(autosave_enabled))
        self.ui.project_autosave_interval_spinbox.setValue(
            settings.value("project_autosave_interval_min", 3, type=int)
        )
        self.ui.project_autosave_folder_input.setText(
            settings.value("project_autosave_folder", get_default_project_autosave_dir(), type=str)
        )
        owner = self.window()
        auto_export_source_txt = settings.value("auto_export_source_txt", False, type=bool)
        auto_export_source_md = settings.value("auto_export_source_md", False, type=bool)
        auto_export_translation_txt = settings.value("auto_export_translation_txt", False, type=bool)
        auto_export_translation_md = settings.value("auto_export_translation_md", False, type=bool)
        if getattr(owner, "auto_export_source_txt_checkbox", None) is not None:
            owner.auto_export_source_txt_checkbox.setChecked(bool(auto_export_source_txt))
        if getattr(owner, "auto_export_source_md_checkbox", None) is not None:
            owner.auto_export_source_md_checkbox.setChecked(bool(auto_export_source_md))
        if getattr(owner, "auto_export_translation_txt_checkbox", None) is not None:
            owner.auto_export_translation_txt_checkbox.setChecked(bool(auto_export_translation_txt))
        if getattr(owner, "auto_export_translation_md_checkbox", None) is not None:
            owner.auto_export_translation_md_checkbox.setChecked(bool(auto_export_translation_md))
        settings.endGroup()

        settings.beginGroup("dictionaries")
        ocr_rules_json = settings.value("ocr_substitutions_json", "[]", type=str)
        translation_rules_json = settings.value("translation_substitutions_json", "[]", type=str)
        try:
            ocr_rules = json.loads(ocr_rules_json or "[]")
        except Exception:
            ocr_rules = []
        try:
            translation_rules = json.loads(translation_rules_json or "[]")
        except Exception:
            translation_rules = []
        self.ui.user_dictionaries_page.load_rules(ocr_rules, translation_rules)
        settings.endGroup()

        settings.beginGroup("notifications")
        self.ui.notifications_page.load_settings(
            enable_completion_sound=settings.value("enable_completion_sound", True, type=bool),
            completion_sound_mode=settings.value("completion_sound_mode", SYSTEM_SOUND_MODE, type=str),
            completion_sound_file=settings.value("completion_sound_file", "", type=str),
        )
        settings.endGroup()

        settings.beginGroup("shortcuts")
        default_shortcuts = get_default_shortcuts()
        shortcut_values = {}
        for shortcut_id, default_value in default_shortcuts.items():
            shortcut_values[shortcut_id] = settings.value(shortcut_id, default_value, type=str)
        settings.endGroup()
        self.ui.shortcuts_page.load_shortcuts(shortcut_values)
        owner = self.window()
        shortcut_ctrl = getattr(owner, "shortcut_ctrl", None)
        if shortcut_ctrl is not None:
            shortcut_ctrl.apply_shortcuts()

        settings.beginGroup("credentials")
        save_keys = settings.value("save_keys", False, type=bool)
        self.ui.save_keys_checkbox.setChecked(save_keys)
        for service in self.ui.credential_services:
            translated_service = self._normalize_service_name(service)
            for field in self.CREDENTIAL_FIELDS.get(translated_service, ("api_key",)):
                widget = self.ui.credential_widgets.get(f"{translated_service}_{field}")
                if widget is not None:
                    widget.setText(
                        self._load_credential_value(
                            settings,
                            translated_service,
                            field,
                            save_keys,
                        )
                    )
        settings.endGroup()

        self._current_language = self.ui.lang_combo.currentText()
        self._loading_settings = False
        owner = self.window()
        if owner is not None and hasattr(owner, "refresh_automatic_output_controls"):
            try:
                owner.refresh_automatic_output_controls()
            except Exception:
                logger.debug("Failed to refresh automatic output controls after loading settings.", exc_info=True)

    def on_language_changed(self, new_language):
        if not self._loading_settings:
            self.show_restart_dialog(new_language)

    def play_test_completion_sound(self) -> None:
        settings = self.get_notification_settings()
        play_completion_sound(
            str(settings.get("completion_sound_mode") or SYSTEM_SOUND_MODE),
            str(settings.get("completion_sound_file") or ""),
        )

    def _show_message_box(self, icon: QtWidgets.QMessageBox.Icon, title: str, text: str):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(icon)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        ok_btn = msg_box.addButton(self.tr("OK"), QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        msg_box.setDefaultButton(ok_btn)
        msg_box.exec()

    def _ask_yes_no(self, title: str, text: str, default_yes: bool = False) -> bool:
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Question)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        yes_btn = msg_box.addButton(self.tr("Yes"), QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        no_btn = msg_box.addButton(self.tr("No"), QtWidgets.QMessageBox.ButtonRole.RejectRole)
        msg_box.setDefaultButton(yes_btn if default_yes else no_btn)
        msg_box.exec()
        return msg_box.clickedButton() == yes_btn

    def show_restart_dialog(self, new_language):
        from modules.utils.common_utils import restart_application

        response = self._ask_yes_no(
            self.tr("Restart Required"),
            self.tr("The application needs to restart for the language changes to take effect.\nRestart now?"),
            default_yes=True,
        )

        if response:
            self.save_settings()
            self._current_language = new_language
            restart_application()
        else:
            self._loading_settings = True
            self.ui.lang_combo.setCurrentText(self._current_language)
            self._loading_settings = False

    def get_min_font_size(self):
        return int(self.ui.min_font_spinbox.value())

    def get_max_font_size(self):
        return int(self.ui.max_font_spinbox.value())

    def add_font_family(self, font_input: str) -> QFont:
        if os.path.splitext(font_input)[1].lower() in [".ttf", ".ttc", ".otf", ".woff", ".woff2"]:
            font_id = QFontDatabase.addApplicationFont(font_input)
            if font_id != -1:
                font_families = QFontDatabase.applicationFontFamilies(font_id)
                if font_families:
                    return font_families[0]

        return font_input

    def check_for_updates(self, is_background=False):
        self._is_background_check = is_background
        if not is_background:
            self.ui.check_update_button.setEnabled(False)
            self.ui.check_update_button.setText(self.tr("Checking..."))
        self.update_checker.check_for_updates()

    def on_update_available(self, version, release_url, download_url):
        if not self._is_background_check:
            self.ui.check_update_button.setEnabled(True)
            self.ui.check_update_button.setText(self.tr("Check for Updates"))

        settings = QSettings("ComicLabs", "ComicTranslate")
        ignored_version = settings.value("updates/ignored_version", "")

        if self._is_background_check and version == ignored_version:
            return

        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle(self.tr("Update Available"))
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setTextInteractionFlags(Qt.TextBrowserInteraction)
        msg_box.setText(self.tr("A new version {version} is available.").format(version=version))
        link_text = self.tr("Release Notes")
        msg_box.setInformativeText(f'<a href="{release_url}" style="color: #4da6ff;">{link_text}</a>')

        download_btn = msg_box.addButton(self.tr("Yes"), QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        msg_box.addButton(self.tr("No"), QtWidgets.QMessageBox.ButtonRole.RejectRole)

        dotted_ask_btn = None
        if self._is_background_check:
            dotted_ask_btn = msg_box.addButton(
                self.tr("Skip This Version"),
                QtWidgets.QMessageBox.ButtonRole.ApplyRole,
            )

        msg_box.setDefaultButton(download_btn)
        msg_box.exec()

        if msg_box.clickedButton() == download_btn:
            self.start_download(download_url)
        elif dotted_ask_btn and msg_box.clickedButton() == dotted_ask_btn:
            settings.setValue("updates/ignored_version", version)

    def on_up_to_date(self):
        if self._is_background_check:
            return

        self.ui.check_update_button.setEnabled(True)
        self.ui.check_update_button.setText(self.tr("Check for Updates"))
        self._show_message_box(
            QtWidgets.QMessageBox.Icon.Information,
            self.tr("Up to Date"),
            self.tr("You are using the latest version."),
        )

    def on_update_error(self, message):
        if self._is_background_check:
            logger.error(f"Background update check failed: {message}")
            return

        self.ui.check_update_button.setEnabled(True)
        self.ui.check_update_button.setText(self.tr("Check for Updates"))
        if self.update_dialog:
            self.update_dialog.close()

        self._show_message_box(
            QtWidgets.QMessageBox.Icon.Warning,
            self.tr("Update Error"),
            message,
        )

    def start_download(self, url):
        self.update_dialog = QtWidgets.QProgressDialog(
            self.tr("Downloading update..."),
            self.tr("Cancel"),
            0,
            100,
            self,
        )
        self.update_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.update_dialog.show()

        filename = url.split("/")[-1]
        self.update_checker.download_installer(url, filename)

    def on_download_progress(self, percent):
        if self.update_dialog:
            self.update_dialog.setValue(percent)

    def on_download_finished(self, file_path):
        if self.update_dialog:
            self.update_dialog.close()

        if self._ask_yes_no(
            self.tr("Download Complete"),
            self.tr("Installer downloaded to {path}. Run it now?").format(path=file_path),
            default_yes=True,
        ):
            self.update_checker.run_installer(file_path)

    def shutdown(self):
        if getattr(self, "_is_shutting_down", False):
            return
        self._is_shutting_down = True

        try:
            self.update_checker.shutdown()
        except Exception:
            pass

        if self.update_dialog:
            try:
                self.update_dialog.close()
            except Exception:
                pass
