from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets


def _panel_tr(text: str) -> str:
    return QtCore.QCoreApplication.translate("PipelineStatusPanel", text)


class PipelineInteractionOverlay(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.hide()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor(10, 12, 16, 128))

        card_width = min(max(420, self.width() // 2), 760)
        card_height = min(max(120, self.height() // 7), 180)
        card_rect = QtCore.QRect(0, 0, card_width, card_height)
        card_rect.moveCenter(self.rect().center())
        card_rect.translate(0, -self.height() // 6)

        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 26), 1))
        painter.setBrush(QtGui.QColor(255, 255, 255, 14))
        painter.drawRoundedRect(card_rect, 18, 18)

        super().paintEvent(event)


class _PreviewLabel(QtWidgets.QLabel):
    clicked = QtCore.Signal()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class PipelineCompletionOverlay(QtWidgets.QWidget):
    close_requested = QtCore.Signal()
    open_output_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._preview_path = ""
        self._preview_pixmap = QtGui.QPixmap()
        self._output_root = ""
        self.hide()
        self._build_ui()

    def _build_ui(self) -> None:
        self.setObjectName("pipelineCompletionOverlay")
        self.setStyleSheet(
            """
            QWidget#pipelineCompletionOverlay QFrame#completionCard {
                background-color: rgba(42, 44, 50, 246);
                border: 1px solid rgba(255, 255, 255, 28);
                border-radius: 18px;
            }
            QWidget#pipelineCompletionOverlay QLabel {
                color: #f1f3f5;
            }
            QWidget#pipelineCompletionOverlay QPushButton {
                background-color: rgba(255, 255, 255, 22);
                border: 1px solid rgba(255, 255, 255, 34);
                border-radius: 10px;
                color: #f6f7f8;
                font-weight: 600;
                min-height: 36px;
                padding: 0 14px;
            }
            QWidget#pipelineCompletionOverlay QPushButton:hover {
                background-color: rgba(255, 255, 255, 32);
            }
            QWidget#pipelineCompletionOverlay QPushButton#completionPrimaryButton {
                background-color: rgba(95, 140, 245, 220);
                border-color: rgba(95, 140, 245, 240);
            }
            QWidget#pipelineCompletionOverlay QPushButton#completionPrimaryButton:hover {
                background-color: rgba(112, 155, 252, 236);
            }
            """
        )

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(28, 28, 28, 28)
        root.addStretch(1)

        self.card = QtWidgets.QFrame(self)
        self.card.setObjectName("completionCard")
        card_layout = QtWidgets.QVBoxLayout(self.card)
        card_layout.setContentsMargins(22, 22, 22, 22)
        card_layout.setSpacing(16)

        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        self.state_label = QtWidgets.QLabel(_panel_tr("Done"))
        header_font = QtGui.QFont(QtWidgets.QApplication.font())
        header_font.setBold(True)
        header_font.setPointSize(max(header_font.pointSize(), 11) + 2)
        self.state_label.setFont(header_font)
        self.message_label = QtWidgets.QLabel("")
        self.message_label.setWordWrap(True)
        self.close_button = QtWidgets.QPushButton(_panel_tr("Close"))
        self.close_button.clicked.connect(self.close_requested.emit)
        header_layout.addWidget(self.state_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.close_button)

        self.preview_label = _PreviewLabel(self.card)
        self.preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(420, 260)
        self.preview_label.setStyleSheet(
            "background-color: rgba(20, 22, 26, 222); border: 1px solid rgba(255, 255, 255, 22); border-radius: 14px;"
        )
        self.preview_label.setText(_panel_tr("No Preview"))
        self.preview_label.clicked.connect(self._open_preview_path)

        action_layout = QtWidgets.QHBoxLayout()
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(10)
        action_layout.addWidget(self.message_label, 1)
        self.open_output_button = QtWidgets.QPushButton(_panel_tr("Open Output"))
        self.open_output_button.setObjectName("completionPrimaryButton")
        self.open_output_button.setVisible(False)
        self.open_output_button.clicked.connect(self.open_output_requested.emit)
        action_layout.addWidget(self.open_output_button)

        card_layout.addLayout(header_layout)
        card_layout.addWidget(self.preview_label, 1)
        card_layout.addLayout(action_layout)

        root.addWidget(self.card, 1)
        root.addStretch(1)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(6, 8, 10, 168))
        super().paintEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_preview_pixmap()

    def set_output_root(self, output_root: str | None) -> None:
        self._output_root = str(output_root or "").strip()
        self.open_output_button.setVisible(bool(self._output_root and os.path.exists(self._output_root)))

    def show_preview(self, preview_path: str | None, *, message: str = "", output_root: str | None = None) -> None:
        self.set_output_root(output_root)
        self.message_label.setText(message or "")
        self._set_preview_path(preview_path)
        self.show()
        self.raise_()

    def clear_preview(self) -> None:
        self.hide()
        self._preview_path = ""
        self._preview_pixmap = QtGui.QPixmap()
        self.open_output_button.setVisible(False)
        self.preview_label.clear()
        self.preview_label.setText(_panel_tr("No Preview"))
        self.message_label.clear()
        self._output_root = ""

    def _set_preview_path(self, preview_path: str | None) -> None:
        path = str(preview_path or "").strip()
        self._preview_path = path
        self._preview_pixmap = QtGui.QPixmap()
        if path and os.path.isfile(path):
            pixmap = QtGui.QPixmap(path)
            if not pixmap.isNull():
                self._preview_pixmap = pixmap
        self._refresh_preview_pixmap()

    def _refresh_preview_pixmap(self) -> None:
        if self._preview_pixmap.isNull():
            self.preview_label.clear()
            self.preview_label.setText(_panel_tr("No Preview"))
            return
        target = self.preview_label.size() - QtCore.QSize(20, 20)
        if target.width() <= 0 or target.height() <= 0:
            return
        scaled = self._preview_pixmap.scaled(
            target,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    def _open_preview_path(self) -> None:
        if not self._preview_path:
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self._preview_path))


class PipelineStatusPanel(QtWidgets.QFrame):
    cancel_requested = QtCore.Signal()
    retry_requested = QtCore.Signal()
    open_settings_requested = QtCore.Signal()
    report_requested = QtCore.Signal()
    open_output_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(None)
        self._anchor_widget = parent
        self.setWindowFlags(
            QtCore.Qt.WindowType.Window
            | QtCore.Qt.WindowType.WindowTitleHint
            | QtCore.Qt.WindowType.WindowSystemMenuHint
            | QtCore.Qt.WindowType.WindowMinimizeButtonHint
            | QtCore.Qt.WindowType.WindowMaximizeButtonHint
            | QtCore.Qt.WindowType.WindowCloseButtonHint
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setObjectName("pipelineStatusPanel")
        self.setWindowTitle(self.tr("Pipeline Status"))
        self.setMinimumSize(580, 420)
        self.resize(760, 540)
        self.setFont(QtGui.QFont(QtWidgets.QApplication.font()))
        self.setStyleSheet(
            """
            QFrame#pipelineStatusPanel {
                background-color: rgba(44, 46, 52, 246);
                border: 1px solid rgba(255, 255, 255, 26);
                border-radius: 16px;
            }
            QFrame#pipelineStatusPanel QLabel {
                color: #eef2f5;
            }
            QFrame#pipelineStatusPanel QLabel#pipelineTitle {
                color: #fbfcfd;
                font-size: 15px;
                font-weight: 700;
            }
            QFrame#pipelineStatusPanel QLabel#pipelineState {
                color: #d8dee9;
                font-size: 13px;
                font-weight: 600;
            }
            QFrame#pipelineStatusPanel QLabel#summaryLabel {
                color: #b5bcc7;
                font-weight: 600;
            }
            QFrame#pipelineStatusPanel QLabel#summaryValue {
                color: #f5f7fa;
            }
            QFrame#pipelineStatusPanel QPlainTextEdit {
                background-color: rgba(16, 18, 22, 236);
                border: 1px solid rgba(255, 255, 255, 22);
                border-radius: 12px;
                color: #f5f7fa;
                selection-background-color: rgba(95, 140, 245, 164);
                padding: 10px;
            }
            QFrame#pipelineStatusPanel QPushButton {
                background-color: rgba(255, 255, 255, 20);
                border: 1px solid rgba(255, 255, 255, 30);
                border-radius: 10px;
                color: #f7f8f9;
                font-weight: 600;
                min-height: 36px;
                padding: 0 14px;
            }
            QFrame#pipelineStatusPanel QPushButton:hover {
                background-color: rgba(255, 255, 255, 30);
            }
            QFrame#pipelineStatusPanel QPushButton#dangerAction {
                background-color: rgba(194, 94, 94, 224);
                border-color: rgba(207, 106, 106, 238);
            }
            QFrame#pipelineStatusPanel QPushButton#dangerAction:hover {
                background-color: rgba(212, 110, 110, 236);
            }
            QFrame#pipelineStatusPanel QPushButton#primaryAction {
                background-color: rgba(95, 140, 245, 220);
                border-color: rgba(95, 140, 245, 240);
            }
            QFrame#pipelineStatusPanel QPushButton#primaryAction:hover {
                background-color: rgba(112, 155, 252, 236);
            }
            QFrame#pipelineStatusPanel QPushButton#accentAction {
                background-color: rgba(108, 181, 124, 210);
                border-color: rgba(118, 194, 136, 230);
            }
            QFrame#pipelineStatusPanel QPushButton#accentAction:hover {
                background-color: rgba(122, 199, 142, 226);
            }
            """
        )

        self._anchor_rect = QtCore.QRect()
        self._drag_offset = QtCore.QPoint()
        self._drag_active = False
        self._user_positioned = False
        self._positioning_from_anchor = False
        self._pipeline_active = False
        self._preview_path = ""
        self._preview_pixmap = QtGui.QPixmap()
        self._output_root = ""
        self._hide_timer = QtCore.QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._hide_if_idle)
        self.hide()
        self._build_ui()
        self._apply_state("idle")

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        self.header_widget = QtWidgets.QFrame(self)
        self.header_widget.installEventFilter(self)
        header_layout = QtWidgets.QHBoxLayout(self.header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)

        self.title_label = QtWidgets.QLabel(self.tr("Pipeline Status"))
        self.title_label.setObjectName("pipelineTitle")
        self.state_label = QtWidgets.QLabel(self.tr("Idle"))
        self.state_label.setObjectName("pipelineState")

        self.expand_button = QtWidgets.QToolButton()
        self.expand_button.setCheckable(True)
        self.expand_button.setChecked(True)
        self.expand_button.setArrowType(QtCore.Qt.ArrowType.DownArrow)
        self.expand_button.setToolTip(self.tr("Show details"))
        self.expand_button.clicked.connect(self._toggle_details)

        self.minimize_button = QtWidgets.QToolButton()
        self.minimize_button.setText("−")
        self.minimize_button.setToolTip(self.tr("Minimize"))
        self.minimize_button.clicked.connect(self.toggle_minimized)

        header_layout.addWidget(self.title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.state_label)
        header_layout.addSpacing(6)
        header_layout.addWidget(self.expand_button)
        header_layout.addWidget(self.minimize_button)

        self.body_widget = QtWidgets.QWidget(self)
        body_layout = QtWidgets.QVBoxLayout(self.body_widget)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(12)

        summary_layout = QtWidgets.QHBoxLayout()
        summary_layout.setSpacing(14)

        left_col = QtWidgets.QGridLayout()
        left_col.setHorizontalSpacing(12)
        left_col.setVerticalSpacing(8)
        self.service_value = self._add_summary_row(left_col, 0, self.tr("Service"))
        self.progress_value = self._add_summary_row(left_col, 1, self.tr("Progress"))
        self.file_value = self._add_summary_row(left_col, 2, self.tr("File"), wrap=True)
        self.eta_value = self._add_summary_row(left_col, 3, self.tr("ETA"))
        self.message_value = self._add_summary_row(left_col, 4, self.tr("Message"), wrap=True)
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left_col)

        self.preview_label = _PreviewLabel(self)
        self.preview_label.setMinimumSize(280, 180)
        self.preview_label.setMaximumWidth(340)
        self.preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setText(self.tr("No Preview"))
        self.preview_label.setStyleSheet(
            "background-color: rgba(18, 20, 24, 222); border: 1px solid rgba(255, 255, 255, 20); border-radius: 12px;"
        )
        self.preview_label.clicked.connect(self._open_preview_path)

        summary_layout.addWidget(left_widget, 1)
        summary_layout.addWidget(self.preview_label, 0)

        self.details_view = QtWidgets.QPlainTextEdit(self)
        self.details_view.setReadOnly(True)
        self.details_view.setMaximumBlockCount(400)
        self.details_view.setMinimumHeight(230)
        self.details_view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        detail_font = QtGui.QFont(QtWidgets.QApplication.font())
        detail_font.setPointSize(max(detail_font.pointSize(), 10) + 1)
        self.details_view.setFont(detail_font)

        action_layout = QtWidgets.QHBoxLayout()
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(8)
        self.cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
        self.cancel_button.setObjectName("dangerAction")
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        self.report_button = QtWidgets.QPushButton(self.tr("Report"))
        self.retry_button = QtWidgets.QPushButton(self.tr("Retry"))
        self.retry_button.setObjectName("primaryAction")
        self.retry_button.clicked.connect(self.retry_requested.emit)
        self.settings_button = QtWidgets.QPushButton(self.tr("Settings"))
        self.open_output_button = QtWidgets.QPushButton(self.tr("Open Output"))
        self.open_output_button.setObjectName("accentAction")
        self.open_output_button.clicked.connect(self.open_output_requested.emit)
        self.close_button = QtWidgets.QPushButton(self.tr("Close"))
        self.report_button.clicked.connect(self.report_requested.emit)
        self.settings_button.clicked.connect(self.open_settings_requested.emit)
        self.close_button.clicked.connect(self.close_panel)
        self._action_buttons = [
            self.cancel_button,
            self.report_button,
            self.retry_button,
            self.settings_button,
            self.open_output_button,
            self.close_button,
        ]
        for button in self._action_buttons:
            action_layout.addWidget(button)
        action_layout.addStretch(1)

        body_layout.addLayout(summary_layout)
        body_layout.addWidget(self.details_view, 1)
        body_layout.addLayout(action_layout)

        root.addWidget(self.header_widget)
        root.addWidget(self.body_widget, 1)

    def _add_summary_row(
        self,
        layout: QtWidgets.QGridLayout,
        row: int,
        label_text: str,
        *,
        wrap: bool = False,
    ) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(label_text)
        label.setObjectName("summaryLabel")
        value = QtWidgets.QLabel("-")
        value.setObjectName("summaryValue")
        value.setWordWrap(wrap)
        layout.addWidget(label, row, 0)
        layout.addWidget(value, row, 1)
        return value

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is self.header_widget:
            if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                mouse_event = event
                if (
                    mouse_event.button() == QtCore.Qt.MouseButton.LeftButton
                    and not self.isMaximized()
                    and not self.isMinimized()
                ):
                    self._drag_active = True
                    self._drag_offset = mouse_event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                    self._user_positioned = True
                    return True
            elif event.type() == QtCore.QEvent.Type.MouseMove and self._drag_active:
                mouse_event = event
                if not self.isMaximized() and not self.isMinimized():
                    self.move(mouse_event.globalPosition().toPoint() - self._drag_offset)
                    return True
            elif event.type() == QtCore.QEvent.Type.MouseButtonRelease:
                self._drag_active = False
                return True
        return super().eventFilter(watched, event)

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # type: ignore[override]
        if not self._user_positioned:
            self._position_from_anchor()
        super().showEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_preview_pixmap()

    def moveEvent(self, event: QtGui.QMoveEvent) -> None:  # type: ignore[override]
        super().moveEvent(event)
        if self.isVisible() and not self._positioning_from_anchor and not self.isMinimized():
            self._user_positioned = True

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        if self._pipeline_active:
            self.showMinimized()
            event.ignore()
            return
        self.hide()
        event.ignore()

    def set_allowed_area(self, rect: QtCore.QRect) -> None:
        self.set_anchor_rect(rect)

    def set_anchor_rect(self, rect: QtCore.QRect) -> None:
        self._anchor_rect = QtCore.QRect(rect)
        if not self._user_positioned:
            self._position_from_anchor()

    def _position_from_anchor(self) -> None:
        if self._anchor_rect.isEmpty():
            return
        width = max(self.minimumWidth(), 760)
        height = max(self.minimumHeight(), 540)
        width = min(width, max(560, self._anchor_rect.width() // 2))
        height = min(height, max(420, self._anchor_rect.height() - 64))
        x = self._anchor_rect.left() + 24
        y = self._anchor_rect.bottom() - height - 24
        if y < self._anchor_rect.top() + 24:
            y = self._anchor_rect.top() + 24
        self._positioning_from_anchor = True
        self.setGeometry(x, y, width, height)
        self._positioning_from_anchor = False

    def _toggle_details(self, checked: bool) -> None:
        self.details_view.setVisible(checked)
        self.expand_button.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow
        )

    def toggle_minimized(self) -> None:
        if self.isMinimized():
            self.showNormal()
        else:
            self.showMinimized()

    def set_minimized(self, minimized: bool) -> None:
        if minimized:
            self.showMinimized()
        elif self.isMinimized():
            self.showNormal()

    def close_panel(self) -> None:
        if self._pipeline_active:
            self.showMinimized()
            return
        self.hide()

    def _hide_if_idle(self) -> None:
        if not self._pipeline_active:
            self.hide()

    def _append_log(self, text: str) -> None:
        if not text:
            return
        self.details_view.appendPlainText(text)
        bar = self.details_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _refresh_preview_pixmap(self) -> None:
        if self._preview_pixmap.isNull():
            self.preview_label.clear()
            self.preview_label.setText(self.tr("No Preview"))
            return
        target = self.preview_label.size() - QtCore.QSize(16, 16)
        if target.width() <= 0 or target.height() <= 0:
            return
        scaled = self._preview_pixmap.scaled(
            target,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    def _set_preview_path(self, preview_path: str | None) -> None:
        path = str(preview_path or "").strip()
        self._preview_path = path
        self._preview_pixmap = QtGui.QPixmap()
        if path and os.path.isfile(path):
            pixmap = QtGui.QPixmap(path)
            if not pixmap.isNull():
                self._preview_pixmap = pixmap
        self._refresh_preview_pixmap()

    def _open_preview_path(self) -> None:
        if not self._preview_path:
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self._preview_path))

    def set_output_root(self, output_root: str | None) -> None:
        self._output_root = str(output_root or "").strip()

    def has_output_root(self) -> bool:
        return bool(self._output_root and os.path.exists(self._output_root))

    def update_event(self, event: dict) -> None:
        phase = str(event.get("phase") or "")
        status = str(event.get("status") or "")
        page_total = int(event.get("page_total") or 0)
        page_index = event.get("page_index")
        service_key = str(event.get("service") or "-")
        service_map = {
            "batch": self.tr("Automatic"),
            "gemma": "Gemma",
            "paddleocr_vl": "PaddleOCR VL",
            "hunyuanocr": "HunyuanOCR",
        }

        if not self.isVisible():
            self.show()
        if not self.isMinimized():
            self.raise_()

        self.service_value.setText(service_map.get(service_key, service_key))
        if page_total > 0 and page_index is not None:
            self.progress_value.setText(f"{int(page_index) + 1}/{page_total}")
        else:
            self.progress_value.setText("-")
        self.file_value.setText(str(event.get("image_name") or "-"))
        self.eta_value.setText(str(event.get("eta_text") or self.tr("Calculating")))
        self.message_value.setText(str(event.get("message") or "-"))
        self._set_preview_path(event.get("preview_path"))

        detail = str(event.get("detail") or "")
        log_line = str(event.get("message") or detail or "")
        if log_line:
            self._append_log(log_line)

        if phase == "done" or status == "completed":
            self._pipeline_active = False
            self._apply_state("done")
        elif status == "failed" or phase == "error":
            self._pipeline_active = False
            self._apply_state("failed")
        elif status == "cancelled":
            self._pipeline_active = False
            self._apply_state("cancelled")
        else:
            self._pipeline_active = True
            self._apply_state("running")

    def show_passive_message(
        self,
        level: str,
        text: str,
        *,
        duration: int | None = None,
        closable: bool = True,
        source: str = "generic",
    ) -> None:
        self.show()
        if not self.isMinimized():
            self.raise_()
        self.title_label.setText(self.tr("Status"))
        level_map = {
            "info": self.tr("Info"),
            "success": self.tr("Success"),
            "warning": self.tr("Warning"),
            "error": self.tr("Error"),
        }
        source_map = {
            "generic": "-",
            "batch": self.tr("Automatic"),
            "txt_md": "TXT/MD",
            "download": self.tr("Download"),
            "local_service": self.tr("Local Service"),
            "network": self.tr("Network"),
            "content_filter": self.tr("Content Filter"),
            "batch_report": self.tr("Batch Report"),
            "pipeline": self.tr("Pipeline"),
        }
        self.state_label.setText(level_map.get(level, level.title()))
        self.message_value.setText(text)
        self.service_value.setText(source_map.get(source, source))
        self.progress_value.setText("-")
        self.file_value.setText("-")
        self.eta_value.setText("-")
        self._append_log(text)
        self.close_button.setVisible(bool(closable))
        if not self._pipeline_active and duration:
            self._hide_timer.start(int(duration) * 1000)
        else:
            self._hide_timer.stop()

    def show_download_message(self, text: str) -> None:
        self.show_passive_message("info", text, duration=None, closable=True, source="download")

    def clear_download_message(self) -> None:
        if self._pipeline_active:
            return
        self.message_value.setText("-")
        self.service_value.setText("-")

    def _apply_state(self, state: str) -> None:
        state_map = {
            "idle": self.tr("Idle"),
            "running": self.tr("Running"),
            "done": self.tr("Done"),
            "failed": self.tr("Failed"),
            "cancelled": self.tr("Cancelled"),
        }
        self.title_label.setText(self.tr("Pipeline Status") if state in {"idle", "running", "done", "failed", "cancelled"} else self.tr("Status"))
        self.state_label.setText(state_map.get(state, state))
        self.cancel_button.setVisible(state == "running")
        self.report_button.setVisible(state in {"running", "done", "failed", "cancelled"})
        self.retry_button.setVisible(state == "failed")
        self.settings_button.setVisible(state == "failed")
        self.open_output_button.setVisible(state == "done" and self.has_output_root())
        self.close_button.setVisible(state in {"done", "failed", "cancelled"})
