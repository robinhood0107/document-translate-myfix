import logging
import os
import shutil
from dataclasses import asdict, is_dataclass

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import QSettings, QTimer, Qt, Signal
from PySide6.QtGui import QFont, QFontDatabase

from .settings_ui import SettingsPageUI
from app.update_checker import UpdateChecker
from modules.utils.device import is_gpu_available
from modules.utils.paths import get_default_project_autosave_dir, get_user_data_dir


logger = logging.getLogger(__name__)


class SettingsPage(QtWidgets.QWidget):
    theme_changed = Signal(str)
    font_imported = Signal(str)

    TOOL_CREDENTIAL_SERVICE_MAP = {
        "Custom Service": "Custom Service",
        "Custom Local Server": "Custom Local Server",
        "Custom": "Custom",
        "GPT-4.1": "Open AI GPT",
        "GPT-4.1-mini": "Open AI GPT",
        "Claude-4.5-Sonnet": "Anthropic Claude",
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
        self.ui.raw_text_checkbox.stateChanged.connect(self._save_settings_if_not_loading)
        self.ui.translated_text_checkbox.stateChanged.connect(self._save_settings_if_not_loading)
        self.ui.inpainted_image_checkbox.stateChanged.connect(self._save_settings_if_not_loading)
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
        ]
        for widget in spin_widgets:
            widget.valueChanged.connect(self._save_settings_if_not_loading)

        for widget in self.ui.credential_widgets.values():
            signal = getattr(widget, "textChanged", None)
            if signal is not None:
                signal.connect(self._save_settings_if_not_loading)

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
        return tool_combos[tool_type].currentText()

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
        return {
            "export_raw_text": self.ui.raw_text_checkbox.isChecked(),
            "export_translated_text": self.ui.translated_text_checkbox.isChecked(),
            "export_inpainted_image": self.ui.inpainted_image_checkbox.isChecked(),
            "project_autosave_enabled": autosave_enabled,
            "project_autosave_interval_min": int(self.ui.project_autosave_interval_spinbox.value()),
            "project_autosave_folder": autosave_folder,
        }

    def _normalize_service_name(self, raw_service: str) -> str:
        normalized = self.ui.value_mappings.get(raw_service, raw_service)
        return self.TOOL_CREDENTIAL_SERVICE_MAP.get(normalized, normalized)

    def _resolve_legacy_custom_translator(self) -> str:
        settings = QSettings("ComicLabs", "ComicTranslate")
        legacy_api_key = settings.value("credentials/Custom Service_api_key", "", type=str).strip()
        if not legacy_api_key:
            legacy_api_key = settings.value("credentials/Custom_api_key", "", type=str).strip()
        return "Custom Service" if legacy_api_key else "Custom Local Server"

    def _load_credential_value(
        self,
        settings: QSettings,
        service_name: str,
        field: str,
        save_keys: bool,
    ) -> str:
        if not save_keys:
            return ""

        value = settings.value(f"{service_name}_{field}", "", type=str)
        if value:
            return value

        if service_name == "Custom Service" and field in ("api_key", "api_url", "model"):
            return settings.value(f"Custom_{field}", "", type=str)
        if service_name == "Custom Local Server" and field in ("api_url", "model"):
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
            },
            "paddleocr_vl": self.get_paddleocr_vl_settings(),
            "llm": self.get_llm_settings(),
            "export": self.get_export_settings(),
            "credentials": self.get_credentials(),
            "save_keys": self.ui.save_keys_checkbox.isChecked(),
        }

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

        settings.beginGroup("export")
        settings.remove("auto_save")
        settings.remove("archive_save_as")
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
        self.ui.lang_combo.setCurrentText(translated_language)

        theme = settings.value("theme", "Dark")
        translated_theme = self.ui.reverse_mappings.get(theme, theme)
        self.ui.theme_combo.setCurrentText(translated_theme)
        self.theme_changed.emit(translated_theme)

        translator = settings.value("tools/translator", "Gemini-2.5-Pro", type=str)
        if translator == "Custom":
            translator = self._resolve_legacy_custom_translator()

        settings.beginGroup("tools")
        translated_translator = self.ui.reverse_mappings.get(translator, translator)
        if self.ui.translator_combo.findText(translated_translator) != -1:
            self.ui.translator_combo.setCurrentText(translated_translator)
        else:
            self.ui.translator_combo.setCurrentIndex(-1)

        ocr = settings.value("ocr", "Default")
        translated_ocr = self.ui.reverse_mappings.get(ocr, ocr)
        if self.ui.ocr_combo.findText(translated_ocr) != -1:
            self.ui.ocr_combo.setCurrentText(translated_ocr)
        else:
            self.ui.ocr_combo.setCurrentIndex(-1)

        inpainter = settings.value("inpainter", "AOT")
        translated_inpainter = self.ui.reverse_mappings.get(inpainter, inpainter)
        if self.ui.inpainter_combo.findText(translated_inpainter) != -1:
            self.ui.inpainter_combo.setCurrentText(translated_inpainter)
        else:
            self.ui.inpainter_combo.setCurrentIndex(-1)

        detector = settings.value("detector", "RT-DETR-v2")
        translated_detector = self.ui.reverse_mappings.get(detector, detector)
        if self.ui.detector_combo.findText(translated_detector) != -1:
            self.ui.detector_combo.setCurrentText(translated_detector)
        else:
            self.ui.detector_combo.setCurrentIndex(-1)

        if is_gpu_available():
            self.ui.use_gpu_checkbox.setChecked(settings.value("use_gpu", False, type=bool))
        else:
            self.ui.use_gpu_checkbox.setChecked(False)

        settings.beginGroup("hd_strategy")
        strategy = settings.value("strategy", "Resize")
        translated_strategy = self.ui.reverse_mappings.get(strategy, strategy)
        if self.ui.inpaint_strategy_combo.findText(translated_strategy) != -1:
            self.ui.inpaint_strategy_combo.setCurrentText(translated_strategy)
        else:
            self.ui.inpaint_strategy_combo.setCurrentIndex(0)

        if strategy == "Resize":
            self.ui.resize_spinbox.setValue(settings.value("resize_limit", 960, type=int))
        elif strategy == "Crop":
            self.ui.crop_margin_spinbox.setValue(settings.value("crop_margin", 512, type=int))
            self.ui.crop_trigger_spinbox.setValue(settings.value("crop_trigger_size", 512, type=int))
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
            settings.value("max_new_tokens", 256, type=int)
        )
        self.ui.paddleocr_vl_parallel_workers_spinbox.setValue(
            settings.value("parallel_workers", 2, type=int)
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
        settings.endGroup()

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

    def on_language_changed(self, new_language):
        if not self._loading_settings:
            self.show_restart_dialog(new_language)

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
