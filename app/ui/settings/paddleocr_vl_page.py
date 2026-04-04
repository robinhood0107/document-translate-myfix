from PySide6 import QtCore, QtWidgets

from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.line_edit import MLineEdit
from ..dayu_widgets.spin_box import MSpinBox


class PaddleOCRVLPage(QtWidgets.QWidget):
    DEFAULT_SERVER_URL = "http://127.0.0.1:28118/layout-parsing"

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title_label = MLabel(self.tr("PaddleOCR VL Settings")).h3()
        layout.addWidget(title_label)

        note = MLabel(
            self.tr(
                "Connect Comic Translate to your local PaddleOCR VL Docker service.\n"
                "These settings control the /layout-parsing OCR requests."
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

        toggles_group = QtWidgets.QVBoxLayout()
        toggles_label = MLabel(self.tr("Response Options")).h4()
        toggles_group.addWidget(toggles_label)
        self.prettify_markdown_checkbox = MCheckBox(self.tr("Prettify Markdown"))
        self.visualize_checkbox = MCheckBox(self.tr("Visualize"))
        toggles_group.addWidget(self.prettify_markdown_checkbox)
        toggles_group.addWidget(self.visualize_checkbox)
        layout.addLayout(toggles_group)

        perf_label = MLabel(self.tr("Performance")).h4()
        layout.addWidget(perf_label)

        perf_note = MLabel(
            self.tr(
                "For your RTX 4070 SUPER, 256 tokens and 2 parallel workers are the recommended fastest defaults."
            )
        ).secondary()
        perf_note.setWordWrap(True)
        perf_note.setTextFormat(QtCore.Qt.PlainText)
        layout.addWidget(perf_note)

        max_tokens_layout = QtWidgets.QHBoxLayout()
        max_tokens_label = MLabel(self.tr("Max New Tokens"))
        self.max_new_tokens_spinbox = MSpinBox().small()
        self.max_new_tokens_spinbox.setRange(64, 2048)
        self.max_new_tokens_spinbox.setSingleStep(64)
        self.max_new_tokens_spinbox.setValue(256)
        self.max_new_tokens_spinbox.setFixedWidth(90)
        max_tokens_layout.addWidget(max_tokens_label)
        max_tokens_layout.addWidget(self.max_new_tokens_spinbox)
        max_tokens_layout.addStretch(1)
        layout.addLayout(max_tokens_layout)

        workers_layout = QtWidgets.QHBoxLayout()
        workers_label = MLabel(self.tr("Parallel Workers"))
        self.parallel_workers_spinbox = MSpinBox().small()
        self.parallel_workers_spinbox.setRange(1, 8)
        self.parallel_workers_spinbox.setValue(2)
        self.parallel_workers_spinbox.setFixedWidth(90)
        workers_layout.addWidget(workers_label)
        workers_layout.addWidget(self.parallel_workers_spinbox)
        workers_layout.addStretch(1)
        layout.addLayout(workers_layout)

        layout.addStretch(1)
