from PySide6 import QtCore, QtWidgets

from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.line_edit import MLineEdit
from ..dayu_widgets.spin_box import MSpinBox
from modules.ocr.mangalmm_ocr import (
    DEFAULT_MANGALMM_MAX_COMPLETION_TOKENS,
    DEFAULT_MANGALMM_MAX_LONG_SIDE,
    DEFAULT_MANGALMM_MAX_PIXELS,
    DEFAULT_MANGALMM_PARALLEL_WORKERS,
    DEFAULT_MANGALMM_REQUEST_TIMEOUT_SEC,
    DEFAULT_MANGALMM_SAFE_RESIZE,
    DEFAULT_MANGALMM_SERVER_URL,
)


class MangaLMMOCRPage(QtWidgets.QWidget):
    DEFAULT_SERVER_URL = DEFAULT_MANGALMM_SERVER_URL
    DEFAULT_MAX_COMPLETION_TOKENS = DEFAULT_MANGALMM_MAX_COMPLETION_TOKENS
    DEFAULT_PARALLEL_WORKERS = DEFAULT_MANGALMM_PARALLEL_WORKERS
    DEFAULT_REQUEST_TIMEOUT_SEC = DEFAULT_MANGALMM_REQUEST_TIMEOUT_SEC
    DEFAULT_SAFE_RESIZE = DEFAULT_MANGALMM_SAFE_RESIZE
    DEFAULT_MAX_PIXELS = DEFAULT_MANGALMM_MAX_PIXELS
    DEFAULT_MAX_LONG_SIDE = DEFAULT_MANGALMM_MAX_LONG_SIDE

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title_label = MLabel(self.tr("MangaLMM Settings")).h3()
        layout.addWidget(title_label)

        note = MLabel(
            self.tr(
                "Connect Comic Translate to your local MangaLMM llama.cpp server.\n"
                "This OCR engine sends cropped text regions to the OpenAI-compatible /chat/completions endpoint.\n"
                "MangaLMM is used as block-crop OCR only, not as full-page spotting inside the app.\n"
                "Keep the default localhost URL if you want Comic Translate to reuse the bundled Docker runtime."
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

        tuning_label = MLabel(self.tr("MangaLMM OCR Tuning")).h4()
        layout.addWidget(tuning_label)

        max_tokens_layout = QtWidgets.QHBoxLayout()
        max_tokens_layout.addWidget(MLabel(self.tr("Max Completion Tokens")))
        self.max_completion_tokens_spinbox = MSpinBox().small()
        self.max_completion_tokens_spinbox.setRange(64, 2048)
        self.max_completion_tokens_spinbox.setSingleStep(64)
        self.max_completion_tokens_spinbox.setValue(self.DEFAULT_MAX_COMPLETION_TOKENS)
        self.max_completion_tokens_spinbox.setFixedWidth(90)
        max_tokens_layout.addWidget(self.max_completion_tokens_spinbox)
        max_tokens_layout.addStretch(1)
        layout.addLayout(max_tokens_layout)

        workers_layout = QtWidgets.QHBoxLayout()
        workers_layout.addWidget(MLabel(self.tr("Parallel Workers")))
        self.parallel_workers_spinbox = MSpinBox().small()
        self.parallel_workers_spinbox.setRange(1, 8)
        self.parallel_workers_spinbox.setValue(self.DEFAULT_PARALLEL_WORKERS)
        self.parallel_workers_spinbox.setFixedWidth(90)
        workers_layout.addWidget(self.parallel_workers_spinbox)
        workers_layout.addStretch(1)
        layout.addLayout(workers_layout)

        timeout_layout = QtWidgets.QHBoxLayout()
        timeout_layout.addWidget(MLabel(self.tr("Request Timeout (sec)")))
        self.request_timeout_spinbox = MSpinBox().small()
        self.request_timeout_spinbox.setRange(15, 600)
        self.request_timeout_spinbox.setSingleStep(15)
        self.request_timeout_spinbox.setValue(self.DEFAULT_REQUEST_TIMEOUT_SEC)
        self.request_timeout_spinbox.setFixedWidth(90)
        timeout_layout.addWidget(self.request_timeout_spinbox)
        timeout_layout.addStretch(1)
        layout.addLayout(timeout_layout)

        self.raw_response_logging_checkbox = MCheckBox(self.tr("Raw Response Log"))
        layout.addWidget(self.raw_response_logging_checkbox)

        resize_label = MLabel(self.tr("Safe Resize")).h4()
        layout.addWidget(resize_label)

        self.safe_resize_checkbox = MCheckBox(self.tr("Enable Safe Resize"))
        self.safe_resize_checkbox.setChecked(self.DEFAULT_SAFE_RESIZE)
        layout.addWidget(self.safe_resize_checkbox)

        max_pixels_layout = QtWidgets.QHBoxLayout()
        max_pixels_layout.addWidget(MLabel(self.tr("Max Pixels")))
        self.max_pixels_spinbox = MSpinBox().small()
        self.max_pixels_spinbox.setRange(100000, 4000000)
        self.max_pixels_spinbox.setSingleStep(100000)
        self.max_pixels_spinbox.setValue(self.DEFAULT_MAX_PIXELS)
        self.max_pixels_spinbox.setFixedWidth(110)
        max_pixels_layout.addWidget(self.max_pixels_spinbox)
        max_pixels_layout.addStretch(1)
        layout.addLayout(max_pixels_layout)

        max_long_side_layout = QtWidgets.QHBoxLayout()
        max_long_side_layout.addWidget(MLabel(self.tr("Max Long Side")))
        self.max_long_side_spinbox = MSpinBox().small()
        self.max_long_side_spinbox.setRange(512, 4096)
        self.max_long_side_spinbox.setSingleStep(64)
        self.max_long_side_spinbox.setValue(self.DEFAULT_MAX_LONG_SIDE)
        self.max_long_side_spinbox.setFixedWidth(110)
        max_long_side_layout.addWidget(self.max_long_side_spinbox)
        max_long_side_layout.addStretch(1)
        layout.addLayout(max_long_side_layout)

        tip = MLabel(
            self.tr(
                "Recommended values for the bundled MangaLMM runtime:\n"
                "- ctx-size 4096: enough for block OCR while keeping VRAM safer\n"
                "- Max Completion Tokens: 256\n"
                "- Parallel Workers: 1\n"
                "- Request Timeout: 60 seconds\n"
                "- Safe Resize: on\n"
                "- Max Pixels / Max Long Side: 1200000 / 1280\n"
                "Reasoning:\n"
                "- Keep the request deterministic with temperature 0 and top_k 1 internally.\n"
                "- Large crops are resized only when needed, and OCR region boxes are mapped back to original coordinates.\n"
                "- Workers 1 is the safest default when Gemma and MangaLMM stay resident on the same GPU."
            )
        ).secondary()
        tip.setWordWrap(True)
        tip.setTextFormat(QtCore.Qt.PlainText)
        layout.addWidget(tip)

        layout.addStretch(1)
