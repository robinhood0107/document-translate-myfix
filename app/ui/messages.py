from .dayu_widgets.message import MMessage
from PySide6.QtCore import QCoreApplication, Qt
from PySide6 import QtWidgets

class Messages:

    @staticmethod
    def _show_passive(parent, level: str, text: str, *, duration=None, closable=True, source: str = "generic"):
        router = getattr(parent, "route_passive_message", None)
        if callable(router):
            handled = router(
                level,
                text,
                duration=duration,
                closable=closable,
                source=source,
            )
            if handled is not None:
                return handled
        fallback = {
            "info": MMessage.info,
            "success": MMessage.success,
            "warning": MMessage.warning,
            "error": MMessage.error,
        }.get(level, MMessage.info)
        return fallback(
            text=text,
            parent=parent,
            duration=duration,
            closable=closable,
        )

    @staticmethod
    def show_info(parent, text: str, *, duration=None, closable=True, source: str = "generic"):
        return Messages._show_passive(parent, "info", text, duration=duration, closable=closable, source=source)

    @staticmethod
    def show_success(parent, text: str, *, duration=None, closable=True, source: str = "generic"):
        return Messages._show_passive(parent, "success", text, duration=duration, closable=closable, source=source)

    @staticmethod
    def show_warning(parent, text: str, *, duration=None, closable=True, source: str = "generic"):
        return Messages._show_passive(parent, "warning", text, duration=duration, closable=closable, source=source)

    @staticmethod
    def show_error(parent, text: str, *, duration=None, closable=True, source: str = "generic"):
        return Messages._show_passive(parent, "error", text, duration=duration, closable=closable, source=source)

    @staticmethod
    def show_translation_complete(parent):
        return Messages.show_success(
            parent,
            QCoreApplication.translate(
                "Messages",
                "Comic has been Translated!"
            ),
            duration=None,
            closable=True,
            source="batch",
        )

    @staticmethod
    def select_font_error(parent):
        return Messages.show_error(
            parent,
            QCoreApplication.translate(
                "Messages", 
                "No Font selected.\nGo to Settings > Text Rendering > Font to select or import one "
            ),
            duration=None,
            closable=True,
        )

    @staticmethod
    def show_missing_credentials_error(parent, provider_name: str, fields_text: str = ""):
        details = (
            QCoreApplication.translate(
                "Messages",
                "Required fields: {fields}"
            ).format(fields=fields_text)
            if fields_text
            else QCoreApplication.translate("Messages", "Please fill in the required credential fields.")
        )
        return Messages.show_error(
            parent,
            QCoreApplication.translate(
                "Messages",
                "Missing credentials for {provider}.\nConfigure them in Settings > Credentials.\n{details}"
            ).format(provider=provider_name, details=details),
            duration=None,
            closable=True,
        )

    @staticmethod
    def show_missing_local_service_config_error(
        parent,
        service_name: str,
        fields_text: str = "",
        settings_page_name: str | None = None,
    ):
        details = (
            QCoreApplication.translate(
                "Messages",
                "Required fields: {fields}"
            ).format(fields=fields_text)
            if fields_text
            else QCoreApplication.translate("Messages", "Please fill in the required settings fields.")
        )
        page_name = settings_page_name or QCoreApplication.translate("Messages", "PaddleOCR VL Settings")
        return Messages.show_error(
            parent,
            QCoreApplication.translate(
                "Messages",
                "Missing settings for {service}.\nConfigure them in Settings > {settings_page}.\n{details}"
            ).format(
                service=service_name,
                settings_page=page_name,
                details=details,
            ),
            duration=None,
            closable=True,
        )

    @staticmethod
    def show_translator_language_not_supported(parent):
        return Messages.show_error(
            parent,
            QCoreApplication.translate(
                "Messages",
                "The translator does not support the selected target language. Please choose a different language or tool."
            ),
            duration=None,
            closable=True,
        )

    @staticmethod
    def show_missing_tool_error(parent, tool_name):
        return Messages.show_error(
            parent,
            QCoreApplication.translate(
                "Messages",
                "No {} selected. Please select a {} in Settings > Tools."
            ).format(tool_name, tool_name),
            duration=None,
            closable=True,
        )

    @staticmethod
    def show_custom_service_not_configured_error(parent):
        return Messages.show_error(
            parent,
            QCoreApplication.translate(
                "Messages",
                "Custom Service requires an OpenAI-compatible API configuration.\n"
                "Please set API Key, Endpoint URL, and Model in Settings > Credentials."
            ),
            duration=None,
            closable=True,
        )

    @staticmethod
    def show_custom_local_gemma_not_configured_error(parent):
        return Messages.show_error(
            parent,
            QCoreApplication.translate(
                "Messages",
                "Custom Local Server(Gemma) requires your local Gemma endpoint and model.\n"
                "Please set Endpoint URL and Model in Settings > Credentials."
            ),
            duration=None,
            closable=True,
        )

    @staticmethod
    def show_error_with_copy(parent, title: str, text: str, detailed_text: str | None = None):
        """
        Show a critical error dialog where the main text is selectable and the
        full details (traceback) are placed in the Details pane. A Copy button
        is provided to copy the full details to the clipboard.

        Args:
            parent: parent widget
            title: dialog window title
            text: short error text shown in the main area
            detailed_text: optional long text (traceback) shown in Details
        """
        msg = QtWidgets.QMessageBox(parent)
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(text)
        if detailed_text:
            msg.setDetailedText(detailed_text)

        # Allow selecting the main text
        try:
            msg.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        except Exception:
            pass

        copy_btn = msg.addButton(QCoreApplication.translate("Messages", "Copy"), QtWidgets.QMessageBox.ButtonRole.ActionRole)
        ok_btn = msg.addButton(QCoreApplication.translate("Messages", "OK"), QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        msg.addButton(QCoreApplication.translate("Messages", "Close"), QtWidgets.QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(ok_btn)
        msg.exec()

        if msg.clickedButton() == copy_btn:
            try:
                QtWidgets.QApplication.clipboard().setText(detailed_text or text)
            except Exception:
                pass

    @staticmethod
    def get_server_error_text(status_code: int = 500, context: str = None) -> str:
        """
        Return the localized error string for a given HTTP status code.
        Thread-safe — does not touch the UI. Use this from worker threads,
        then pass the result through a signal so the UI can display it.

        Args:
            status_code: HTTP status code (500, 501, 502, 503, 504)
            context: optional context ('translation', 'ocr', or None for generic)
        """
        messages = {
            500: QCoreApplication.translate("Messages", "We encountered an unexpected server error.\nPlease try again in a few moments."),
            502: QCoreApplication.translate("Messages", "The external service provider is having trouble.\nPlease try again later."),
            503: QCoreApplication.translate("Messages", "The server is currently busy or under maintenance.\nPlease try again shortly."),
            504: QCoreApplication.translate("Messages", "The server took too long to respond.\nPlease check your connection or try again later."),
        }

        if status_code == 501:
            if context == 'ocr':
                return QCoreApplication.translate("Messages", "The selected text recognition tool is not supported.\nPlease select a different tool in Settings.")
            elif context == 'translation':
                return QCoreApplication.translate("Messages", "The selected translator is not supported.\nPlease select a different tool in Settings.")
            else:
                return QCoreApplication.translate("Messages", "The selected tool is not supported.\nPlease select a different tool in Settings.")

        return messages.get(status_code, messages[500])

    @staticmethod
    def show_server_error(parent, status_code: int = 500, context: str = None):
        """
        Show a user-friendly error for 5xx server issues.
        
        Args:
            parent: parent widget
            status_code: HTTP status code
            context: optional context ('translation', 'ocr', or None for generic)
        """
        text = Messages.get_server_error_text(status_code, context)
        return Messages.show_error(
            parent,
            text,
            duration=None,
            closable=True,
            source="local_service",
        )

    @staticmethod
    def show_network_error(parent):
        """
        Show a user-friendly error for network/connectivity issues.
        """
        return Messages.show_error(
            parent,
            QCoreApplication.translate(
                "Messages", 
                "Unable to connect to the server.\nPlease check your internet connection."
            ),
            duration=None,
            closable=True,
            source="network",
        )

    @staticmethod
    def show_local_service_error(
        parent,
        details: str = None,
        *,
        service_name: str = "PaddleOCR VL",
        settings_page_name: str | None = None,
        error_kind: str = "connection",
    ):
        """
        Show a user-friendly error when a required local OCR service is unavailable.
        """
        page_name = settings_page_name or QCoreApplication.translate("Messages", "PaddleOCR VL Settings")
        if error_kind == "response":
            text = QCoreApplication.translate(
                "Messages",
                "The local {service} service returned an invalid response.\nCheck Settings > {settings_page} and review the local service logs."
            ).format(service=service_name, settings_page=page_name)
        elif error_kind == "setup":
            text = QCoreApplication.translate(
                "Messages",
                "Unable to prepare the local {service} runtime.\nCheck Settings > {settings_page} and make sure Docker is available."
            ).format(service=service_name, settings_page=page_name)
        else:
            text = QCoreApplication.translate(
                "Messages",
                "Unable to reach the local {service} service.\nCheck Settings > {settings_page} and make sure the local service is running."
            ).format(service=service_name, settings_page=page_name)

        if details and details.strip() != text.strip():
            text = f"{text}\n{details}"
        return Messages.show_error(
            parent,
            text,
            duration=None,
            closable=True,
            source="local_service",
        )

    @staticmethod
    def confirm_automatic_run(
        parent,
        *,
        run_label: str,
        page_count: int,
        source_lang: str,
        target_lang: str,
        ocr_mode_label: str,
        resolved_ocr_label: str | None = None,
    ) -> bool:
        msg = QtWidgets.QMessageBox(parent)
        msg.setIcon(QtWidgets.QMessageBox.Question)
        msg.setWindowTitle(
            QCoreApplication.translate("Messages", "Confirm Automatic Processing")
        )

        lines = [
            QCoreApplication.translate(
                "Messages",
                "Review the automatic processing settings before starting.",
            ),
            "",
            QCoreApplication.translate("Messages", "Run: {run_label}").format(
                run_label=run_label
            ),
            QCoreApplication.translate("Messages", "Pages: {page_count}").format(
                page_count=page_count
            ),
            QCoreApplication.translate("Messages", "Source Language: {source_lang}").format(
                source_lang=source_lang
            ),
            QCoreApplication.translate("Messages", "Target Language: {target_lang}").format(
                target_lang=target_lang
            ),
            QCoreApplication.translate("Messages", "Text Recognition Mode: {ocr_mode}").format(
                ocr_mode=ocr_mode_label
            ),
        ]
        if resolved_ocr_label:
            lines.append(
                QCoreApplication.translate(
                    "Messages",
                    "Resolved Text Recognition: {ocr_engine}",
                ).format(ocr_engine=resolved_ocr_label)
            )
        msg.setText("\n".join(lines))
        try:
            msg.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        except Exception:
            pass
        start_btn = msg.addButton(
            QCoreApplication.translate("Messages", "Start"),
            QtWidgets.QMessageBox.ButtonRole.AcceptRole,
        )
        msg.addButton(
            QCoreApplication.translate("Messages", "Cancel"),
            QtWidgets.QMessageBox.ButtonRole.RejectRole,
        )
        msg.setDefaultButton(start_btn)
        msg.exec()
        return msg.clickedButton() == start_btn

    @staticmethod
    def get_content_flagged_text(details: str = None, context: str = "Operation") -> str:
        """
        Build the standardized content-flagged error text.
        """
        if context == "OCR":
            msg = QCoreApplication.translate(
                "Messages",
                "Text Recognition blocked: The AI provider flagged this content.\nPlease try a different Text Recognition tool."
            )
        elif context in ("Translator", "Translation"):
            msg = QCoreApplication.translate(
                "Messages",
                "Translation blocked: The AI provider flagged this content.\nPlease try a different translator."
            )
        else:
            msg = QCoreApplication.translate(
                "Messages",
                "Operation blocked: The AI provider flagged this content.\nPlease try a different tool."
            )
        
        return msg

    @staticmethod
    def show_content_flagged_error(parent, details: str = None, context: str = "Operation", duration=None, closable=True):
        """
        Show a friendly error when content is blocked by safety filters.
        """
        msg_text = Messages.get_content_flagged_text(details=details, context=context)
        return Messages.show_error(
            parent,
            msg_text,
            duration=duration,
            closable=closable,
            source="content_filter",
        )

    @staticmethod
    def show_batch_skipped_summary(parent, skipped_count: int):
        """
        Show a persistent summary when a batch finished with skipped images.
        """
        text = QCoreApplication.translate(
            "Messages",
            "{0} image(s) were skipped in this batch.\nOpen Batch Report to see all skipped images and reasons."
        ).format(skipped_count)
        return Messages.show_warning(
            parent,
            text,
            duration=None,
            closable=True,
            source="batch",
        )
