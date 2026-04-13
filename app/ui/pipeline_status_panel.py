from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets


class PipelineInteractionOverlay(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.hide()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(10, 10, 10, 38))
        super().paintEvent(event)


class _PreviewLabel(QtWidgets.QLabel):
    clicked = QtCore.Signal()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class PipelineStatusPanel(QtWidgets.QFrame):
    cancel_requested = QtCore.Signal()
    retry_requested = QtCore.Signal()
    open_settings_requested = QtCore.Signal()
    report_requested = QtCore.Signal()
    open_output_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pipelineStatusPanel")
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            """
            QFrame#pipelineStatusPanel {
                background-color: rgba(36, 36, 36, 235);
                border: 1px solid #585858;
                border-radius: 8px;
            }
            QFrame#pipelineStatusPanel QLabel {
                color: #f0f0f0;
            }
            QFrame#pipelineStatusPanel QPlainTextEdit {
                background-color: rgba(18, 18, 18, 210);
                border: 1px solid #4d4d4d;
                border-radius: 6px;
                color: #e8e8e8;
            }
            """
        )

        self._allowed_area = QtCore.QRect()
        self._drag_offset = QtCore.QPoint()
        self._drag_active = False
        self._user_positioned = False
        self._minimized = False
        self._normal_geometry = QtCore.QRect()
        self._allowed_max_width = 16777215
        self._allowed_max_height = 16777215
        self._pipeline_active = False
        self._preview_path = ""
        self._output_root = ""
        self._hide_timer = QtCore.QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._hide_if_idle)
        self.hide()
        self._build_ui()
        self._apply_state("idle")

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self.header_widget = QtWidgets.QFrame(self)
        self.header_widget.installEventFilter(self)
        header_layout = QtWidgets.QHBoxLayout(self.header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)

        self.title_label = QtWidgets.QLabel(self.tr("Pipeline Status"))
        title_font = self.title_label.font()
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.state_label = QtWidgets.QLabel(self.tr("Idle"))

        self.expand_button = QtWidgets.QToolButton()
        self.expand_button.setCheckable(True)
        self.expand_button.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.expand_button.setToolTip(self.tr("Show details"))
        self.expand_button.clicked.connect(self._toggle_details)

        self.minimize_button = QtWidgets.QToolButton()
        self.minimize_button.setText("−")
        self.minimize_button.setToolTip(self.tr("Minimize"))
        self.minimize_button.clicked.connect(self.toggle_minimized)

        header_layout.addWidget(self.title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.state_label)
        header_layout.addWidget(self.expand_button)
        header_layout.addWidget(self.minimize_button)

        self.body_widget = QtWidgets.QWidget(self)
        body_layout = QtWidgets.QVBoxLayout(self.body_widget)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(8)

        summary_layout = QtWidgets.QHBoxLayout()
        summary_layout.setSpacing(10)

        left_col = QtWidgets.QGridLayout()
        left_col.setHorizontalSpacing(10)
        left_col.setVerticalSpacing(4)
        self.service_value = self._add_summary_row(left_col, 0, self.tr("Service"))
        self.progress_value = self._add_summary_row(left_col, 1, self.tr("Progress"))
        self.file_value = self._add_summary_row(left_col, 2, self.tr("File"))
        self.eta_value = self._add_summary_row(left_col, 3, self.tr("ETA"))
        self.message_value = self._add_summary_row(left_col, 4, self.tr("Message"), wrap=True)
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left_col)

        self.preview_label = _PreviewLabel(self)
        self.preview_label.setFixedSize(180, 120)
        self.preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setText(self.tr("No Preview"))
        self.preview_label.setStyleSheet(
            "background-color: rgba(10, 10, 10, 190); border: 1px solid #505050; border-radius: 6px;"
        )
        self.preview_label.clicked.connect(self._open_preview_path)

        summary_layout.addWidget(left_widget, 1)
        summary_layout.addWidget(self.preview_label, 0)

        self.details_view = QtWidgets.QPlainTextEdit(self)
        self.details_view.setReadOnly(True)
        self.details_view.setMaximumBlockCount(200)
        self.details_view.hide()

        action_layout = QtWidgets.QHBoxLayout()
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(6)
        self.cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        self.report_button = QtWidgets.QPushButton(self.tr("Report"))
        self.report_button.clicked.connect(self.report_requested.emit)
        self.retry_button = QtWidgets.QPushButton(self.tr("Retry"))
        self.retry_button.clicked.connect(self.retry_requested.emit)
        self.settings_button = QtWidgets.QPushButton(self.tr("Settings"))
        self.settings_button.clicked.connect(self.open_settings_requested.emit)
        self.open_output_button = QtWidgets.QPushButton(self.tr("Open Output"))
        self.open_output_button.clicked.connect(self.open_output_requested.emit)
        self.close_button = QtWidgets.QPushButton(self.tr("Close"))
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
        self.size_grip = QtWidgets.QSizeGrip(self.body_widget)
        action_layout.addWidget(self.size_grip, 0, QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignBottom)

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
        value = QtWidgets.QLabel("-")
        value.setWordWrap(wrap)
        layout.addWidget(label, row, 0)
        layout.addWidget(value, row, 1)
        return value

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is self.header_widget:
            if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                mouse_event = event
                if mouse_event.button() == QtCore.Qt.MouseButton.LeftButton:
                    self._drag_active = True
                    self._drag_offset = mouse_event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                    self._user_positioned = True
                    return True
            elif event.type() == QtCore.QEvent.Type.MouseMove and self._drag_active:
                mouse_event = event
                self.move(mouse_event.globalPosition().toPoint() - self._drag_offset)
                self._clamp_to_allowed_area()
                return True
            elif event.type() == QtCore.QEvent.Type.MouseButtonRelease:
                self._drag_active = False
                return True
        return super().eventFilter(watched, event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._clamp_to_allowed_area()

    def set_allowed_area(self, rect: QtCore.QRect) -> None:
        self._allowed_area = QtCore.QRect(rect)
        if self._allowed_area.isEmpty():
            return
        max_width = max(320, self._allowed_area.width() // 4)
        max_height = max(180, self._allowed_area.height() // 4)
        self._allowed_max_width = max_width
        self._allowed_max_height = max_height
        self.setMaximumSize(max_width, max_height)
        if not self.isVisible() or not self._user_positioned:
            width = min(max_width, 520)
            height = min(max_height, 260)
            x = self._allowed_area.left() + 16
            y = self._allowed_area.bottom() - height - 16
            self.setGeometry(x, y, width, height)
            self._normal_geometry = self.geometry()
        else:
            self._clamp_to_allowed_area()

    def _clamp_to_allowed_area(self) -> None:
        if self._allowed_area.isEmpty():
            return
        rect = self.geometry()
        width = min(rect.width(), self.maximumWidth())
        height = min(rect.height(), self.maximumHeight())
        x = min(max(rect.x(), self._allowed_area.left()), self._allowed_area.right() - width)
        y = min(max(rect.y(), self._allowed_area.top()), self._allowed_area.bottom() - height)
        if rect.x() != x or rect.y() != y or rect.width() != width or rect.height() != height:
            self.setGeometry(x, y, width, height)

    def _toggle_details(self, checked: bool) -> None:
        self.details_view.setVisible(checked and not self._minimized)
        self.expand_button.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow
        )

    def toggle_minimized(self) -> None:
        self.set_minimized(not self._minimized)

    def set_minimized(self, minimized: bool) -> None:
        if minimized == self._minimized:
            return
        self._minimized = minimized
        if minimized:
            self._normal_geometry = self.geometry()
            self.body_widget.hide()
            self.setFixedHeight(self.header_widget.sizeHint().height() + 20)
            self.minimize_button.setText("+")
        else:
            self.setMinimumHeight(0)
            self.setMaximumHeight(self._allowed_max_height)
            self.setMaximumWidth(self._allowed_max_width)
            self.body_widget.show()
            self.setGeometry(self._normal_geometry)
            self.details_view.setVisible(self.expand_button.isChecked())
            self.minimize_button.setText("−")
        self._clamp_to_allowed_area()

    def close_panel(self) -> None:
        if self._pipeline_active:
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

    def _set_preview_path(self, preview_path: str | None) -> None:
        path = str(preview_path or "").strip()
        self._preview_path = path
        if not path or not os.path.isfile(path):
            self.preview_label.clear()
            self.preview_label.setText(self.tr("No Preview"))
            return
        pixmap = QtGui.QPixmap(path)
        if pixmap.isNull():
            self.preview_label.clear()
            self.preview_label.setText(self.tr("No Preview"))
            return
        scaled = pixmap.scaled(
            self.preview_label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

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

        self.show()
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
        self.state_label.setText(state_map.get(state, state))
        self.cancel_button.setVisible(state == "running")
        self.report_button.setVisible(state in {"running", "done", "failed", "cancelled"})
        self.retry_button.setVisible(state == "failed")
        self.settings_button.setVisible(state == "failed")
        self.open_output_button.setVisible(state == "done" and self.has_output_root())
        self.close_button.setVisible(state in {"done", "failed", "cancelled"})
