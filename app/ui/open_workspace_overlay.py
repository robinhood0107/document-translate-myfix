from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class OpenWorkspaceOverlay(QtWidgets.QFrame):
    """Embedded, non-window progress surface for workspace opening."""

    close_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("openWorkspaceOverlay")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setWindowFlags(QtCore.Qt.WindowType.Widget)
        self.hide()

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.addStretch(1)

        card = QtWidgets.QFrame(self)
        card.setObjectName("openWorkspaceCard")
        card.setMinimumWidth(440)
        card.setMaximumWidth(620)
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(12)

        self.title_label = QtWidgets.QLabel(self.tr("Opening workspace"), card)
        self.title_label.setObjectName("openWorkspaceTitle")
        self.message_label = QtWidgets.QLabel("", card)
        self.message_label.setObjectName("openWorkspaceMessage")
        self.message_label.setWordWrap(True)
        self.progress_bar = QtWidgets.QProgressBar(card)
        self.progress_bar.setObjectName("openWorkspaceProgress")
        self.progress_bar.setTextVisible(True)
        self.detail_label = QtWidgets.QLabel("", card)
        self.detail_label.setObjectName("openWorkspaceDetail")
        self.detail_label.setWordWrap(True)
        self.close_button = QtWidgets.QPushButton(self.tr("Close"), card)
        self.close_button.setObjectName("openWorkspaceClose")
        self.close_button.clicked.connect(self.close_requested)
        self.close_button.hide()

        layout.addWidget(self.title_label)
        layout.addWidget(self.message_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.detail_label)
        layout.addWidget(self.close_button, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        outer.addWidget(card, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        outer.addStretch(1)

        self.setStyleSheet(
            """
            QFrame#openWorkspaceOverlay {
                background-color: rgba(10, 12, 16, 92);
            }
            QFrame#openWorkspaceCard {
                background-color: rgba(44, 46, 52, 250);
                border: 1px solid rgba(255, 255, 255, 30);
                border-radius: 14px;
            }
            QLabel#openWorkspaceTitle {
                color: #fbfcfd;
                font-size: 16px;
                font-weight: 700;
            }
            QLabel#openWorkspaceMessage {
                color: #eef2f5;
                font-size: 13px;
                font-weight: 600;
            }
            QLabel#openWorkspaceDetail {
                color: #aeb6c2;
                font-size: 11px;
            }
            QProgressBar#openWorkspaceProgress {
                min-height: 12px;
                border: none;
                border-radius: 6px;
                background: rgba(255, 255, 255, 36);
                color: #eef2f5;
                text-align: center;
            }
            QProgressBar#openWorkspaceProgress::chunk {
                border-radius: 6px;
                background: #5f8cf5;
            }
            QPushButton#openWorkspaceClose {
                min-height: 34px;
                padding: 0 16px;
                border-radius: 9px;
                color: #f7f8f9;
                background: rgba(255, 255, 255, 22);
                border: 1px solid rgba(255, 255, 255, 34);
            }
            """
        )

    def start(self, message: str) -> None:
        self.title_label.setText(self.tr("Opening workspace"))
        self.message_label.setText(message)
        self.detail_label.clear()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        self.close_button.hide()
        self.show()
        self.raise_()

    def update_progress(self, event: dict) -> None:
        message = str(event.get("message") or "").strip()
        detail = str(event.get("detail") or "").strip()
        current = int(event.get("current") or 0)
        total = int(event.get("total") or 0)
        if message:
            self.message_label.setText(message)
        self.detail_label.setText(detail)
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(max(0, min(current, total)))
            self.progress_bar.setFormat("%v/%m")
        else:
            self.progress_bar.setRange(0, 0)

    def show_failure(self, message: str, detail: str = "") -> None:
        self.title_label.setText(self.tr("Could not open workspace"))
        self.message_label.setText(message)
        self.detail_label.setText(detail)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.close_button.show()
        self.show()
        self.raise_()
