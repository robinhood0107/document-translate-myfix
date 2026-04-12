from PySide6 import QtCore, QtWidgets

from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.spin_box import MSpinBox
from modules.translation.llm.custom_local_gemma import (
    DEFAULT_GEMMA_CHUNK_SIZE,
    DEFAULT_GEMMA_MAX_COMPLETION_TOKENS,
    DEFAULT_GEMMA_REQUEST_TIMEOUT_SEC,
    DEFAULT_GEMMA_LOCAL_ENDPOINT,
    DEFAULT_GEMMA_LOCAL_MODEL,
)


class GemmaLocalServerPage(QtWidgets.QWidget):
    DEFAULT_ENDPOINT_URL = DEFAULT_GEMMA_LOCAL_ENDPOINT
    DEFAULT_MODEL = DEFAULT_GEMMA_LOCAL_MODEL
    DEFAULT_CHUNK_SIZE = DEFAULT_GEMMA_CHUNK_SIZE
    DEFAULT_MAX_COMPLETION_TOKENS = DEFAULT_GEMMA_MAX_COMPLETION_TOKENS
    DEFAULT_REQUEST_TIMEOUT_SEC = DEFAULT_GEMMA_REQUEST_TIMEOUT_SEC

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title_label = MLabel(self.tr("Gemma Local Server Settings")).h3()
        layout.addWidget(title_label)

        note = MLabel(
            self.tr(
                "Comic Translate can use your local Gemma Docker server for translation.\n"
                "1. Run `docker compose pull --policy always` and then `docker compose up -d --force-recreate` in the repository root.\n"
                "2. In Settings > Credentials, use Endpoint URL `http://127.0.0.1:18080/v1`.\n"
                "3. Set Model to the exact GGUF filename in `testmodel/` (recommended: `gemma-4-26B-IQ4_NL.gguf`).\n"
                "If responses are truncated, lower Chunk Size or increase LLAMA_CTX_SIZE in docker-compose.yaml."
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

        self.raw_response_logging_checkbox = MCheckBox(self.tr("Raw Response Log"))
        layout.addWidget(self.raw_response_logging_checkbox)

        tip = MLabel(
            self.tr(
                "Recommended defaults for the included Gemma Docker setup:\n"
                "- Chunk Size: 6\n"
                "- Max Completion Tokens: 512\n"
                "- Request Timeout: 180 seconds"
            )
        ).secondary()
        tip.setWordWrap(True)
        tip.setTextFormat(QtCore.Qt.PlainText)
        layout.addWidget(tip)

        layout.addStretch(1)
