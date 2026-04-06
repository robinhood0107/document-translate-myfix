from PySide6 import QtCore, QtWidgets

from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.line_edit import MLineEdit
from ..dayu_widgets.spin_box import MSpinBox
from modules.ocr.ocr_paddle_VL import (
    DEFAULT_PADDLEOCR_VL_PARALLEL_WORKERS,
    DEFAULT_PADDLEOCR_VL_REQUEST_TIMEOUT_SEC,
    DEFAULT_PADDLEOCR_VL_SERVER_URL,
)


class PaddleOCRVLPage(QtWidgets.QWidget):
    DEFAULT_SERVER_URL = DEFAULT_PADDLEOCR_VL_SERVER_URL
    DEFAULT_PARALLEL_WORKERS = DEFAULT_PADDLEOCR_VL_PARALLEL_WORKERS
    DEFAULT_REQUEST_TIMEOUT_SEC = DEFAULT_PADDLEOCR_VL_REQUEST_TIMEOUT_SEC

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title_label = MLabel(self.tr("PaddleOCR VL Settings")).h3()
        layout.addWidget(title_label)

        note = MLabel(
            self.tr(
                "Connect Comic Translate to the official PaddleOCR genai_server runtime.\n"
                "This OCR engine sends cropped text regions to the OpenAI-compatible /v1/chat/completions endpoint.\n"
                "Use the direct genai_server service for RT-DETR-v2 block OCR instead of the legacy /layout-parsing pipeline."
            )
        ).secondary()
        note.setWordWrap(True)
        note.setTextFormat(QtCore.Qt.PlainText)
        layout.addWidget(note)

        server_group = QtWidgets.QVBoxLayout()
        server_label = MLabel(self.tr("Server URL")).strong()
        self.server_url_input = MLineEdit()
        self.server_url_input.setMinimumWidth(360)
        self.server_url_input.setMaximumWidth(560)
        self.server_url_input.setPlaceholderText(self.DEFAULT_SERVER_URL)
        server_group.addWidget(server_label)
        server_group.addWidget(self.server_url_input)
        layout.addLayout(server_group)

        tuning_label = MLabel(self.tr("Paddle OCR Tuning")).h4()
        layout.addWidget(tuning_label)

        workers_layout = QtWidgets.QHBoxLayout()
        workers_label = MLabel(self.tr("Parallel Workers"))
        self.parallel_workers_spinbox = MSpinBox().small()
        self.parallel_workers_spinbox.setRange(1, 8)
        self.parallel_workers_spinbox.setValue(self.DEFAULT_PARALLEL_WORKERS)
        self.parallel_workers_spinbox.setFixedWidth(90)
        workers_layout.addWidget(workers_label)
        workers_layout.addWidget(self.parallel_workers_spinbox)
        workers_layout.addStretch(1)
        layout.addLayout(workers_layout)

        timeout_layout = QtWidgets.QHBoxLayout()
        timeout_label = MLabel(self.tr("Request Timeout (sec)"))
        self.request_timeout_spinbox = MSpinBox().small()
        self.request_timeout_spinbox.setRange(15, 600)
        self.request_timeout_spinbox.setSingleStep(15)
        self.request_timeout_spinbox.setValue(self.DEFAULT_REQUEST_TIMEOUT_SEC)
        self.request_timeout_spinbox.setFixedWidth(90)
        timeout_layout.addWidget(timeout_label)
        timeout_layout.addWidget(self.request_timeout_spinbox)
        timeout_layout.addStretch(1)
        layout.addLayout(timeout_layout)

        self.raw_response_logging_checkbox = MCheckBox(self.tr("Raw Response Log"))
        layout.addWidget(self.raw_response_logging_checkbox)

        tip = MLabel(
            self.tr(
                "Recommended starting values for the official PaddleOCR-VL genai_server:\n"
                "- Parallel Workers: 2\n"
                "- Request Timeout: 60 seconds\n"
                "- Keep server-side backend tuning at the official defaults until direct /v1 validation passes."
            )
        ).secondary()
        tip.setWordWrap(True)
        tip.setTextFormat(QtCore.Qt.PlainText)
        layout.addWidget(tip)

        layout.addStretch(1)
