from PySide6 import QtCore, QtWidgets

from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.spin_box import MSpinBox
from modules.translation.llm.custom_local_gemma import (
    DEFAULT_GEMMA_CHUNK_SIZE,
    DEFAULT_GEMMA_LOCAL_ENDPOINT,
    DEFAULT_GEMMA_LOCAL_MODEL,
    DEFAULT_GEMMA_MAX_COMPLETION_TOKENS,
    DEFAULT_GEMMA_REQUEST_TIMEOUT_SEC,
    DEFAULT_GEMMA_TRANSLATION_MIN_P,
    DEFAULT_GEMMA_TRANSLATION_TEMPERATURE,
    DEFAULT_GEMMA_TRANSLATION_TOP_K,
    DEFAULT_GEMMA_TRANSLATION_TOP_P,
)


class GemmaLocalServerPage(QtWidgets.QWidget):
    DEFAULT_ENDPOINT_URL = DEFAULT_GEMMA_LOCAL_ENDPOINT
    DEFAULT_MODEL = DEFAULT_GEMMA_LOCAL_MODEL
    DEFAULT_CHUNK_SIZE = DEFAULT_GEMMA_CHUNK_SIZE
    DEFAULT_MAX_COMPLETION_TOKENS = DEFAULT_GEMMA_MAX_COMPLETION_TOKENS
    DEFAULT_REQUEST_TIMEOUT_SEC = DEFAULT_GEMMA_REQUEST_TIMEOUT_SEC
    DEFAULT_TEMPERATURE = DEFAULT_GEMMA_TRANSLATION_TEMPERATURE
    DEFAULT_TOP_K = DEFAULT_GEMMA_TRANSLATION_TOP_K
    DEFAULT_TOP_P = DEFAULT_GEMMA_TRANSLATION_TOP_P
    DEFAULT_MIN_P = DEFAULT_GEMMA_TRANSLATION_MIN_P

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title_label = MLabel(self.tr("Gemma Local Server Settings")).h3()
        layout.addWidget(title_label)

        note = MLabel(
            self.tr(
                "Comic Translate can reuse your local Gemma Docker server for translation.\n"
                "1. Keep the existing Gemma container running if it is already healthy.\n"
                "2. In Settings > Credentials, use Endpoint URL `http://127.0.0.1:18080/v1`.\n"
                "3. Set Model to the exact GGUF filename in `testmodel/` (recommended: `gemma-4-26B-IQ4_NL.gguf`).\n"
                "Automatic translation reuses an existing Gemma runtime first and only runs `docker compose up -d` when needed.\n"
                "If responses are truncated, lower Chunk Size or Max Completion Tokens before recreating the container."
            )
        ).secondary()
        note.setWordWrap(True)
        note.setTextFormat(QtCore.Qt.PlainText)
        layout.addWidget(note)

        perf_label = MLabel(self.tr("Gemma Translation Tuning")).h4()
        layout.addWidget(perf_label)

        chunk_layout = QtWidgets.QHBoxLayout()
        chunk_label = MLabel(self.tr("Chunk Size"))
        self.chunk_size_spinbox = MSpinBox().small()
        self.chunk_size_spinbox.setRange(1, 8)
        self.chunk_size_spinbox.setValue(self.DEFAULT_CHUNK_SIZE)
        self.chunk_size_spinbox.setFixedWidth(90)
        chunk_layout.addWidget(chunk_label)
        chunk_layout.addWidget(self.chunk_size_spinbox)
        chunk_layout.addStretch(1)
        layout.addLayout(chunk_layout)

        max_tokens_layout = QtWidgets.QHBoxLayout()
        max_tokens_label = MLabel(self.tr("Max Completion Tokens"))
        self.max_completion_tokens_spinbox = MSpinBox().small()
        self.max_completion_tokens_spinbox.setRange(128, 2048)
        self.max_completion_tokens_spinbox.setSingleStep(64)
        self.max_completion_tokens_spinbox.setValue(self.DEFAULT_MAX_COMPLETION_TOKENS)
        self.max_completion_tokens_spinbox.setFixedWidth(90)
        max_tokens_layout.addWidget(max_tokens_label)
        max_tokens_layout.addWidget(self.max_completion_tokens_spinbox)
        max_tokens_layout.addStretch(1)
        layout.addLayout(max_tokens_layout)

        timeout_layout = QtWidgets.QHBoxLayout()
        timeout_label = MLabel(self.tr("Request Timeout (sec)"))
        self.request_timeout_spinbox = MSpinBox().small()
        self.request_timeout_spinbox.setRange(30, 600)
        self.request_timeout_spinbox.setSingleStep(30)
        self.request_timeout_spinbox.setValue(self.DEFAULT_REQUEST_TIMEOUT_SEC)
        self.request_timeout_spinbox.setFixedWidth(90)
        timeout_layout.addWidget(timeout_label)
        timeout_layout.addWidget(self.request_timeout_spinbox)
        timeout_layout.addStretch(1)
        layout.addLayout(timeout_layout)

        advanced_label = MLabel(self.tr("Advanced Sampler Settings")).h4()
        layout.addWidget(advanced_label)

        self.temperature_spinbox = self._build_double_spinbox(0.0, 2.0, 0.05, self.DEFAULT_TEMPERATURE)
        self.top_k_spinbox = MSpinBox().small()
        self.top_k_spinbox.setRange(1, 512)
        self.top_k_spinbox.setValue(self.DEFAULT_TOP_K)
        self.top_k_spinbox.setFixedWidth(90)
        self.top_p_spinbox = self._build_double_spinbox(0.0, 1.0, 0.01, self.DEFAULT_TOP_P)
        self.min_p_spinbox = self._build_double_spinbox(0.0, 1.0, 0.01, self.DEFAULT_MIN_P)

        for label_text, widget in [
            (self.tr("Temperature"), self.temperature_spinbox),
            (self.tr("Top K"), self.top_k_spinbox),
            (self.tr("Top P"), self.top_p_spinbox),
            (self.tr("Min P"), self.min_p_spinbox),
        ]:
            row = QtWidgets.QHBoxLayout()
            row.addWidget(MLabel(label_text))
            row.addWidget(widget)
            row.addStretch(1)
            layout.addLayout(row)

        self.raw_response_logging_checkbox = MCheckBox(self.tr("Raw Response Log"))
        layout.addWidget(self.raw_response_logging_checkbox)

        tip = MLabel(
            self.tr(
                "Promoted winner defaults for the bundled Gemma runtime:\n"
                "- Chunk Size: 6\n"
                "- Max Completion Tokens: 512\n"
                "- Request Timeout: 180 seconds\n"
                "- Temperature: 0.7\n"
                "- Top K / Top P / Min P: 64 / 0.95 / 0.0"
            )
        ).secondary()
        tip.setWordWrap(True)
        tip.setTextFormat(QtCore.Qt.PlainText)
        layout.addWidget(tip)

        layout.addStretch(1)

    def _build_double_spinbox(self, minimum: float, maximum: float, step: float, value: float) -> QtWidgets.QDoubleSpinBox:
        spinbox = QtWidgets.QDoubleSpinBox(self)
        spinbox.setDecimals(2)
        spinbox.setRange(minimum, maximum)
        spinbox.setSingleStep(step)
        spinbox.setValue(value)
        spinbox.setFixedWidth(90)
        return spinbox
