from PySide6 import QtCore, QtWidgets

from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.line_edit import MLineEdit
from ..dayu_widgets.spin_box import MSpinBox
from modules.ocr.hunyuan_ocr import (
    DEFAULT_HUNYUAN_MAX_COMPLETION_TOKENS,
    DEFAULT_HUNYUAN_PARALLEL_WORKERS,
    DEFAULT_HUNYUAN_REQUEST_TIMEOUT_SEC,
    DEFAULT_HUNYUAN_SERVER_URL,
)


class HunyuanOCRPage(QtWidgets.QWidget):
    DEFAULT_SERVER_URL = DEFAULT_HUNYUAN_SERVER_URL
    DEFAULT_MAX_COMPLETION_TOKENS = DEFAULT_HUNYUAN_MAX_COMPLETION_TOKENS
    DEFAULT_PARALLEL_WORKERS = DEFAULT_HUNYUAN_PARALLEL_WORKERS
    DEFAULT_REQUEST_TIMEOUT_SEC = DEFAULT_HUNYUAN_REQUEST_TIMEOUT_SEC

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title_label = MLabel(self.tr("HunyuanOCR Settings")).h3()
        layout.addWidget(title_label)

        note = MLabel(
            self.tr(
                "Connect Comic Translate to your local HunyuanOCR llama.cpp server.\n"
                "This OCR engine sends cropped text regions to the OpenAI-compatible /chat/completions endpoint.\n"
                "Keep the default localhost URL if you want Comic Translate to start the bundled Docker runtime on demand.\n"
                "Start the server with both the HunyuanOCR GGUF model and the matching mmproj file."
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

        tuning_label = MLabel(self.tr("Hunyuan OCR Tuning")).h4()
        layout.addWidget(tuning_label)

        max_tokens_layout = QtWidgets.QHBoxLayout()
        max_tokens_label = MLabel(self.tr("Max Completion Tokens"))
        self.max_completion_tokens_spinbox = MSpinBox().small()
        self.max_completion_tokens_spinbox.setRange(64, 2048)
        self.max_completion_tokens_spinbox.setSingleStep(64)
        self.max_completion_tokens_spinbox.setValue(self.DEFAULT_MAX_COMPLETION_TOKENS)
        self.max_completion_tokens_spinbox.setFixedWidth(90)
        max_tokens_layout.addWidget(max_tokens_label)
        max_tokens_layout.addWidget(self.max_completion_tokens_spinbox)
        max_tokens_layout.addStretch(1)
        layout.addLayout(max_tokens_layout)

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
                "Recommended starting values for the included HunyuanOCR Docker setup:\n"
                "- Max Completion Tokens: 256\n"
                "- Parallel Workers: 2\n"
                "- Request Timeout: 60 seconds"
            )
        ).secondary()
        tip.setWordWrap(True)
        tip.setTextFormat(QtCore.Qt.PlainText)
        layout.addWidget(tip)

        layout.addStretch(1)
