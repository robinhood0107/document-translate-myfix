from PySide6 import QtCore, QtWidgets

from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.line_edit import MLineEdit
from .utils import set_label_width


class CredentialsPage(QtWidgets.QWidget):
    def __init__(self, services: list[str], value_mappings: dict[str, str], parent=None):
        super().__init__(parent)
        self.services = services
        self.value_mappings = value_mappings
        self.credential_widgets: dict[str, MLineEdit] = {}

        main_layout = QtWidgets.QVBoxLayout(self)
        content_layout = QtWidgets.QVBoxLayout()

        self.save_keys_checkbox = MCheckBox(self.tr("Save Keys"))

        info_label = MLabel(
            self.tr(
                "Configure provider API keys or custom endpoints here.\n"
                "Use Custom Service for authenticated OpenAI-compatible providers.\n"
                "Use Custom Local Server for OpenAI-compatible local or keyless endpoints."
            )
        ).secondary()
        info_label.setWordWrap(True)

        content_layout.addWidget(info_label)
        content_layout.addSpacing(10)
        content_layout.addWidget(self.save_keys_checkbox)
        content_layout.addSpacing(20)

        for service_label in self.services:
            service_layout = QtWidgets.QVBoxLayout()
            service_header = MLabel(service_label).strong()
            service_header.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
            service_layout.addWidget(service_header)

            normalized = self.value_mappings.get(service_label, service_label)

            if normalized == "Microsoft Azure":
                self._add_microsoft_azure_fields(service_layout)
            elif normalized == "Custom Service":
                self._add_custom_service_fields(service_layout)
            elif normalized == "Custom Local Server":
                self._add_custom_local_server_fields(service_layout)
            elif normalized == "Yandex":
                self._add_yandex_fields(service_layout)
            else:
                self._add_standard_api_key_field(service_layout, normalized)

            content_layout.addLayout(service_layout)
            content_layout.addSpacing(20)

        content_layout.addStretch(1)
        main_layout.addLayout(content_layout)

    def _build_line_input(
        self,
        label_text: str,
        widget_key: str,
        *,
        password: bool = False,
    ) -> MLineEdit:
        line_input = MLineEdit()
        if password:
            line_input.setEchoMode(QtWidgets.QLineEdit.Password)
        line_input.setFixedWidth(400)
        prefix = MLabel(label_text).border()
        set_label_width(prefix)
        prefix.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        line_input.set_prefix_widget(prefix)
        self.credential_widgets[widget_key] = line_input
        return line_input

    def _add_standard_api_key_field(self, layout: QtWidgets.QVBoxLayout, normalized: str) -> None:
        layout.addWidget(
            self._build_line_input(
                self.tr("API Key"),
                f"{normalized}_api_key",
                password=True,
            )
        )

    def _add_custom_service_fields(self, layout: QtWidgets.QVBoxLayout) -> None:
        layout.addWidget(
            self._build_line_input(
                self.tr("API Key"),
                "Custom Service_api_key",
                password=True,
            )
        )
        layout.addWidget(
            self._build_line_input(
                self.tr("Endpoint URL"),
                "Custom Service_api_url",
            )
        )
        layout.addWidget(
            self._build_line_input(
                self.tr("Model"),
                "Custom Service_model",
            )
        )

    def _add_custom_local_server_fields(self, layout: QtWidgets.QVBoxLayout) -> None:
        layout.addWidget(
            self._build_line_input(
                self.tr("Endpoint URL"),
                "Custom Local Server_api_url",
            )
        )
        layout.addWidget(
            self._build_line_input(
                self.tr("Model"),
                "Custom Local Server_model",
            )
        )

    def _add_yandex_fields(self, layout: QtWidgets.QVBoxLayout) -> None:
        layout.addWidget(
            self._build_line_input(
                self.tr("Secret Key"),
                "Yandex_api_key",
                password=True,
            )
        )
        layout.addWidget(
            self._build_line_input(
                self.tr("Folder ID"),
                "Yandex_folder_id",
            )
        )

    def _add_microsoft_azure_fields(self, layout: QtWidgets.QVBoxLayout) -> None:
        ocr_label = MLabel(self.tr("OCR")).secondary()
        layout.addWidget(ocr_label)
        layout.addWidget(
            self._build_line_input(
                self.tr("API Key"),
                "Microsoft Azure_api_key_ocr",
                password=True,
            )
        )
        layout.addWidget(
            self._build_line_input(
                self.tr("Endpoint URL"),
                "Microsoft Azure_endpoint",
            )
        )

        translate_label = MLabel(self.tr("Translate")).secondary()
        layout.addWidget(translate_label)
        layout.addWidget(
            self._build_line_input(
                self.tr("API Key"),
                "Microsoft Azure_api_key_translator",
                password=True,
            )
        )
        layout.addWidget(
            self._build_line_input(
                self.tr("Region"),
                "Microsoft Azure_region_translator",
            )
        )
