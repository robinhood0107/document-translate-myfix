from PySide6 import QtCore, QtWidgets

from ..dayu_widgets.label import MLabel
from ..dayu_widgets.line_edit import MLineEdit
from ..dayu_widgets.spin_box import MSpinBox


class PaddleOCRVLSpottingPage(QtWidgets.QWidget):
    DEFAULT_SERVER_URL = (
        "http://127.0.0.1:18002/v1/chat/completions"
    )
    DEFAULT_MAX_COMPLETION_TOKENS = 3000
    DEFAULT_REQUEST_TIMEOUT_SEC = 360
    OFFICIAL_IMAGE_MAX_PIXELS = 1_605_632

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        layout.addWidget(
            MLabel(self.tr("PaddleOCR VL Spotting Settings")).h3()
        )
        note = MLabel(
            self.tr(
                "This is a separate full-page Spotting route, not the detector "
                "crop OCR route.\n"
                "It uses the official Spotting: prompt, special location tokens, "
                "a dedicated projector with 1,605,632 maximum image pixels, and "
                "a dedicated named volume.\n"
                "Detector geometry remains authoritative; ambiguous or unmatched "
                "native regions are left for review without a hidden crop fallback.\n"
                "Keep the default localhost URL to use the bundled managed "
                "llama.cpp runtime."
            )
        ).secondary()
        note.setWordWrap(True)
        note.setTextFormat(QtCore.Qt.PlainText)
        layout.addWidget(note)

        server_layout = QtWidgets.QVBoxLayout()
        server_layout.addWidget(MLabel(self.tr("Server URL")).strong())
        self.server_url_input = MLineEdit()
        self.server_url_input.setMinimumWidth(360)
        self.server_url_input.setMaximumWidth(560)
        self.server_url_input.setPlaceholderText(self.DEFAULT_SERVER_URL)
        server_layout.addWidget(self.server_url_input)
        layout.addLayout(server_layout)

        token_layout = QtWidgets.QHBoxLayout()
        token_layout.addWidget(MLabel(self.tr("Max Completion Tokens")))
        self.max_completion_tokens_spinbox = MSpinBox().small()
        self.max_completion_tokens_spinbox.setRange(512, 4096)
        self.max_completion_tokens_spinbox.setSingleStep(64)
        self.max_completion_tokens_spinbox.setValue(
            self.DEFAULT_MAX_COMPLETION_TOKENS
        )
        self.max_completion_tokens_spinbox.setFixedWidth(90)
        token_layout.addWidget(self.max_completion_tokens_spinbox)
        token_layout.addStretch(1)
        layout.addLayout(token_layout)

        timeout_layout = QtWidgets.QHBoxLayout()
        timeout_layout.addWidget(MLabel(self.tr("Request Timeout (sec)")))
        self.request_timeout_spinbox = MSpinBox().small()
        self.request_timeout_spinbox.setRange(30, 600)
        self.request_timeout_spinbox.setSingleStep(15)
        self.request_timeout_spinbox.setValue(
            self.DEFAULT_REQUEST_TIMEOUT_SEC
        )
        self.request_timeout_spinbox.setFixedWidth(90)
        timeout_layout.addWidget(self.request_timeout_spinbox)
        timeout_layout.addStretch(1)
        layout.addLayout(timeout_layout)

        fixed_contract = MLabel(
            self.tr(
                "The Spotting pixel budget and special-token mode are fixed by "
                "the official model contract and cannot be changed here. The "
                "existing crop OCR route keeps its original 1,003,520-pixel "
                "projector unchanged."
            )
        ).secondary()
        fixed_contract.setWordWrap(True)
        fixed_contract.setTextFormat(QtCore.Qt.PlainText)
        layout.addWidget(fixed_contract)
        layout.addStretch(1)
