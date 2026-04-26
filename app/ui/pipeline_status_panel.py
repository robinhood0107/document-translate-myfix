from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets


def _panel_tr(text: str) -> str:
    return QtCore.QCoreApplication.translate("PipelineStatusPanel", text)


class PipelineInteractionOverlay(QtWidgets.QWidget):
    DIM_COLOR = QtGui.QColor(10, 12, 16, 31)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        self._clear_rects: list[QtCore.QRect] = []
        self.hide()

    def set_clear_rects(self, rects: list[QtCore.QRect]) -> None:
        self._clear_rects = [QtCore.QRect(rect) for rect in rects if rect and not rect.isEmpty()]
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        painter = QtGui.QPainter(self)
        if self._clear_rects:
            region = QtGui.QRegion(self.rect())
            for rect in self._clear_rects:
                region -= QtGui.QRegion(rect.adjusted(-4, -4, 4, 4))
            painter.setClipRegion(region)
        painter.fillRect(self.rect(), self.DIM_COLOR)


class _PreviewLabel(QtWidgets.QLabel):
    clicked = QtCore.Signal()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class PipelineStatusPanel(QtWidgets.QFrame):
    EMBEDDED_MODE = "embedded"
    WINDOW_MODE = "window"

    cancel_requested = QtCore.Signal()
    pause_requested = QtCore.Signal()
    retry_requested = QtCore.Signal()
    open_settings_requested = QtCore.Signal()
    report_requested = QtCore.Signal()
    open_output_requested = QtCore.Signal()
    open_series_board_requested = QtCore.Signal()
    open_current_series_item_requested = QtCore.Signal()
    panel_hidden = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._host_widget = parent
        self._display_mode = self.EMBEDDED_MODE
        self._window_flags = (
            QtCore.Qt.WindowType.Window
            | QtCore.Qt.WindowType.WindowTitleHint
            | QtCore.Qt.WindowType.WindowSystemMenuHint
            | QtCore.Qt.WindowType.WindowMinimizeButtonHint
            | QtCore.Qt.WindowType.WindowMaximizeButtonHint
            | QtCore.Qt.WindowType.WindowCloseButtonHint
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
        self._details_visible = True
        self._current_state = "idle"
        self._series_queue_pause_visible = False
        self._series_queue_pause_requested = False
        self._window_geometry = QtCore.QRect()
        self._embedded_geometry = QtCore.QRect()
        self._resize_margin = 8
        self._resize_active = False
        self._resize_edges = set()
        self._resize_start_global = QtCore.QPoint()
        self._resize_start_geometry = QtCore.QRect()

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setObjectName("pipelineStatusPanel")
        self.setWindowTitle(self.tr("Pipeline Status"))
        self.setMinimumSize(680, 420)
        self.resize(860, 560)
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
                color: #f5f7fa;
                font-size: 13px;
                font-weight: 700;
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
            QFrame#pipelineStatusPanel QPushButton#modeAction {
                min-height: 28px;
                padding: 0 10px;
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
            QFrame#pipelineStatusPanel QFrame#previewFrame {
                background-color: rgba(18, 20, 24, 222);
                border: 1px solid rgba(255, 255, 255, 20);
                border-radius: 12px;
            }
            """
        )

        self._hide_timer = QtCore.QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._hide_if_idle)
        self.hide()

        self._build_ui()
        self._update_mode_button()
        self._update_left_column_width()
        self._set_display_mode_flags(self.EMBEDDED_MODE, initial=True)
        self._apply_state("idle")

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 12, 12)
        root.setSpacing(12)

        self.header_widget = QtWidgets.QFrame(self)
        self.header_widget.installEventFilter(self)
        header_layout = QtWidgets.QHBoxLayout(self.header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        self.title_label = QtWidgets.QLabel(self.tr("Pipeline Status"))
        self.title_label.setObjectName("pipelineTitle")

        self.state_label = QtWidgets.QLabel(self.tr("Idle"))
        self.state_label.setObjectName("pipelineState")

        self.mode_button = QtWidgets.QPushButton(self)
        self.mode_button.setObjectName("modeAction")
        self.mode_button.clicked.connect(self.toggle_display_mode)

        self.logs_button = QtWidgets.QPushButton(self.tr("Logs"))
        self.logs_button.setObjectName("modeAction")
        self.logs_button.setCheckable(True)
        self.logs_button.setChecked(True)
        self.logs_button.clicked.connect(self._toggle_details)

        header_layout.addWidget(self.title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.state_label)
        header_layout.addWidget(self.mode_button)
        header_layout.addWidget(self.logs_button)

        self.body_widget = QtWidgets.QWidget(self)
        body_layout = QtWidgets.QHBoxLayout(self.body_widget)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(14)

        self.left_panel = QtWidgets.QWidget(self.body_widget)
        left_layout = QtWidgets.QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        summary_grid = QtWidgets.QGridLayout()
        summary_grid.setHorizontalSpacing(12)
        summary_grid.setVerticalSpacing(8)
        self.service_value = self._add_summary_row(summary_grid, 0, self.tr("Service"))
        self.progress_value = self._add_summary_row(summary_grid, 1, self.tr("Progress"))
        self.file_value = self._add_summary_row(summary_grid, 2, self.tr("File"), wrap=True)
        self.eta_value = self._add_summary_row(summary_grid, 3, self.tr("ETA"))
        self.message_value = self._add_summary_row(summary_grid, 4, self.tr("Message"), wrap=True)

        self.details_view = QtWidgets.QPlainTextEdit(self)
        self.details_view.setReadOnly(True)
        self.details_view.setMaximumBlockCount(400)
        self.details_view.setMinimumHeight(220)
        self.details_view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        detail_font = QtGui.QFont(QtWidgets.QApplication.font())
        detail_font.setPointSize(max(detail_font.pointSize(), 10) + 1)
        self.details_view.setFont(detail_font)

        action_layout = QtWidgets.QHBoxLayout()
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(8)

        self.pause_button = QtWidgets.QPushButton(self.tr("Pause"))
        self.pause_button.clicked.connect(self.pause_requested.emit)

        self.series_board_button = QtWidgets.QPushButton(self.tr("Series Board"))
        self.series_board_button.clicked.connect(self.open_series_board_requested.emit)

        self.current_item_button = QtWidgets.QPushButton(self.tr("Current Item"))
        self.current_item_button.clicked.connect(self.open_current_series_item_requested.emit)

        self.cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
        self.cancel_button.setObjectName("dangerAction")
        self.cancel_button.clicked.connect(self.cancel_requested.emit)

        self.report_button = QtWidgets.QPushButton(self.tr("Report"))
        self.report_button.clicked.connect(self.report_requested.emit)

        self.retry_button = QtWidgets.QPushButton(self.tr("Retry"))
        self.retry_button.setObjectName("primaryAction")
        self.retry_button.clicked.connect(self.retry_requested.emit)

        self.settings_button = QtWidgets.QPushButton(self.tr("Settings"))
        self.settings_button.clicked.connect(self.open_settings_requested.emit)

        self.open_output_button = QtWidgets.QPushButton(self.tr("Open Output"))
        self.open_output_button.setObjectName("accentAction")
        self.open_output_button.clicked.connect(self.open_output_requested.emit)

        self.close_button = QtWidgets.QPushButton(self.tr("Close"))
        self.close_button.clicked.connect(self.close_panel)

        self._action_buttons = [
            self.pause_button,
            self.series_board_button,
            self.current_item_button,
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

        left_layout.addLayout(summary_grid)
        left_layout.addWidget(self.details_view, 1)
        left_layout.addLayout(action_layout)

        self.preview_frame = QtWidgets.QFrame(self.body_widget)
        self.preview_frame.setObjectName("previewFrame")
        preview_layout = QtWidgets.QVBoxLayout(self.preview_frame)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        preview_layout.setSpacing(0)

        self.preview_label = _PreviewLabel(self.preview_frame)
        self.preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.preview_label.setText(self.tr("No Preview"))
        self.preview_label.clicked.connect(self._open_preview_path)
        preview_layout.addWidget(self.preview_label, 1)

        body_layout.addWidget(self.left_panel, 0)
        body_layout.addWidget(self.preview_frame, 1)

        grip_row = QtWidgets.QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 0, 0)
        grip_row.addStretch(1)
        self.size_grip = QtWidgets.QSizeGrip(self)
        grip_row.addWidget(self.size_grip, 0, QtCore.Qt.AlignmentFlag.AlignRight)

        root.addWidget(self.header_widget)
        root.addWidget(self.body_widget, 1)
        root.addLayout(grip_row)

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

    def set_series_queue_pause_visible(self, visible: bool, *, pause_requested: bool = False) -> None:
        self._series_queue_pause_visible = bool(visible)
        self._series_queue_pause_requested = bool(pause_requested)
        self.pause_button.setText(
            self.tr("Pause Requested")
            if self._series_queue_pause_requested
            else self.tr("Pause")
        )
        self.pause_button.setEnabled(
            self._series_queue_pause_visible and not self._series_queue_pause_requested
        )
        self.series_board_button.setVisible(self._series_queue_pause_visible)
        self.current_item_button.setVisible(self._series_queue_pause_visible)
        self._apply_state(self._current_state)

    def display_mode(self) -> str:
        return self._display_mode

    def prepare_for_new_run(self) -> None:
        self.cancel_auto_close()
        self._user_positioned = False
        self.set_logs_visible(True)
        self.set_display_mode(self.EMBEDDED_MODE, reposition=True)
        self.show()
        self.raise_()

    def set_display_mode(self, mode: str, *, reposition: bool = False) -> None:
        mode = str(mode or self.EMBEDDED_MODE).strip().lower()
        if mode not in {self.EMBEDDED_MODE, self.WINDOW_MODE}:
            mode = self.EMBEDDED_MODE

        if mode == self._display_mode and not reposition:
            return

        was_visible = self.isVisible()
        current_global_geometry = self._current_global_geometry()
        if self._display_mode == self.EMBEDDED_MODE:
            self._embedded_geometry = QtCore.QRect(self.geometry())
        else:
            self._window_geometry = QtCore.QRect(self.frameGeometry())

        self._display_mode = mode
        self._set_display_mode_flags(mode)

        if mode == self.EMBEDDED_MODE:
            local_geometry = self._host_local_rect_from_global(current_global_geometry)
            local_geometry = self._clamp_embedded_geometry(local_geometry)
            if reposition or local_geometry.isEmpty():
                self._position_from_anchor()
            else:
                self._positioning_from_anchor = True
                self.setGeometry(local_geometry)
                self._positioning_from_anchor = False
        else:
            if reposition or current_global_geometry.isEmpty():
                self._position_from_anchor()
            else:
                self._positioning_from_anchor = True
                self.setGeometry(current_global_geometry)
                self._positioning_from_anchor = False

        if was_visible:
            self.show()
        self._update_mode_button()
        self._update_left_column_width()
        self._refresh_preview_pixmap()
        self.raise_()

    def toggle_display_mode(self) -> None:
        target = self.WINDOW_MODE if self._display_mode == self.EMBEDDED_MODE else self.EMBEDDED_MODE
        self.set_display_mode(target)

    def _set_display_mode_flags(self, mode: str, *, initial: bool = False) -> None:
        if mode == self.WINDOW_MODE:
            self.setParent(None)
            self.setWindowFlags(self._window_flags)
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_QuitOnClose, False)
            self.size_grip.hide()
        else:
            if self.parentWidget() is not self._host_widget:
                self.setParent(self._host_widget)
            self.setWindowFlags(QtCore.Qt.WindowType.Widget)
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_QuitOnClose, False)
            self.size_grip.show()
        if not initial:
            self.show()

    def _update_mode_button(self) -> None:
        if self._display_mode == self.EMBEDDED_MODE:
            self.mode_button.setText(self.tr("Window"))
            self.mode_button.setToolTip(self.tr("Switch to window mode"))
        else:
            self.mode_button.setText(self.tr("Embed"))
            self.mode_button.setToolTip(self.tr("Switch to embedded mode"))

    def set_logs_visible(self, visible: bool) -> None:
        self._details_visible = bool(visible)
        self.details_view.setVisible(self._details_visible)
        blocker = QtCore.QSignalBlocker(self.logs_button)
        self.logs_button.setChecked(self._details_visible)
        del blocker
        self.logs_button.setToolTip(self.tr("Hide logs") if self._details_visible else self.tr("Show logs"))
        self._update_left_column_width()
        self._refresh_preview_pixmap()

    def _toggle_details(self, checked: bool) -> None:
        self.set_logs_visible(bool(checked))

    def _update_left_column_width(self) -> None:
        total_width = max(self.width(), self.minimumWidth())
        target = 340 if self._details_visible else 240
        target = min(target, max(220, total_width // 3))
        self.left_panel.setFixedWidth(target)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is self.header_widget and self._display_mode == self.EMBEDDED_MODE:
            if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                mouse_event = event
                if self._resize_edges_for_pos(self.mapFromGlobal(mouse_event.globalPosition().toPoint())):
                    return False
                if mouse_event.button() == QtCore.Qt.MouseButton.LeftButton:
                    self._drag_active = True
                    self._drag_offset = mouse_event.globalPosition().toPoint() - self.mapToGlobal(self.rect().topLeft())
                    self._user_positioned = True
                    return True
            elif event.type() == QtCore.QEvent.Type.MouseMove and self._drag_active:
                mouse_event = event
                global_top_left = mouse_event.globalPosition().toPoint() - self._drag_offset
                target_rect = QtCore.QRect(
                    self._host_local_point_from_global(global_top_left),
                    self.size(),
                )
                target_rect = self._clamp_embedded_geometry(target_rect)
                self.move(target_rect.topLeft())
                return True
            elif event.type() == QtCore.QEvent.Type.MouseButtonRelease:
                self._drag_active = False
                return True
        return super().eventFilter(watched, event)

    def _resize_edges_for_pos(self, pos: QtCore.QPoint) -> set[str]:
        if self._display_mode != self.EMBEDDED_MODE:
            return set()
        rect = self.rect()
        margin = self._resize_margin
        edges: set[str] = set()
        if pos.x() <= rect.left() + margin:
            edges.add("left")
        elif pos.x() >= rect.right() - margin:
            edges.add("right")
        if pos.y() <= rect.top() + margin:
            edges.add("top")
        elif pos.y() >= rect.bottom() - margin:
            edges.add("bottom")
        return edges

    def _cursor_for_edges(self, edges: set[str]) -> QtCore.Qt.CursorShape:
        if {"left", "top"}.issubset(edges) or {"right", "bottom"}.issubset(edges):
            return QtCore.Qt.CursorShape.SizeFDiagCursor
        if {"right", "top"}.issubset(edges) or {"left", "bottom"}.issubset(edges):
            return QtCore.Qt.CursorShape.SizeBDiagCursor
        if "left" in edges or "right" in edges:
            return QtCore.Qt.CursorShape.SizeHorCursor
        if "top" in edges or "bottom" in edges:
            return QtCore.Qt.CursorShape.SizeVerCursor
        return QtCore.Qt.CursorShape.ArrowCursor

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._display_mode == self.EMBEDDED_MODE:
            edges = self._resize_edges_for_pos(event.position().toPoint())
            if edges:
                self._resize_active = True
                self._resize_edges = edges
                self._resize_start_global = event.globalPosition().toPoint()
                self._resize_start_geometry = QtCore.QRect(self.geometry())
                self._user_positioned = True
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if self._resize_active and self._display_mode == self.EMBEDDED_MODE:
            delta = event.globalPosition().toPoint() - self._resize_start_global
            rect = QtCore.QRect(self._resize_start_geometry)
            if "left" in self._resize_edges:
                rect.setLeft(rect.left() + delta.x())
            if "right" in self._resize_edges:
                rect.setRight(rect.right() + delta.x())
            if "top" in self._resize_edges:
                rect.setTop(rect.top() + delta.y())
            if "bottom" in self._resize_edges:
                rect.setBottom(rect.bottom() + delta.y())
            self.setGeometry(self._clamp_embedded_geometry(rect))
            event.accept()
            return
        edges = self._resize_edges_for_pos(event.position().toPoint())
        self.setCursor(QtGui.QCursor(self._cursor_for_edges(edges)))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if self._resize_active and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._resize_active = False
            self._resize_edges = set()
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:  # type: ignore[override]
        if not self._resize_active:
            self.unsetCursor()
        super().leaveEvent(event)

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # type: ignore[override]
        if not self._user_positioned:
            self._position_from_anchor()
        super().showEvent(event)

    def hideEvent(self, event: QtGui.QHideEvent) -> None:  # type: ignore[override]
        super().hideEvent(event)
        self.panel_hidden.emit()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if (
            self._display_mode == self.EMBEDDED_MODE
            and self.isVisible()
            and not self._positioning_from_anchor
            and event.oldSize().isValid()
        ):
            self._user_positioned = True
            self.setGeometry(self._clamp_embedded_geometry(self.geometry()))
        self._update_left_column_width()
        self._refresh_preview_pixmap()

    def moveEvent(self, event: QtGui.QMoveEvent) -> None:  # type: ignore[override]
        super().moveEvent(event)
        if self.isVisible() and not self._positioning_from_anchor and not self.isMinimized():
            self._user_positioned = True

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        if self._pipeline_active and self._display_mode == self.WINDOW_MODE:
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

    def _default_geometry_for_anchor(self, anchor: QtCore.QRect) -> QtCore.QRect:
        if anchor.isEmpty():
            return QtCore.QRect()

        available_width = max(520, anchor.width() - 48)
        available_height = max(360, anchor.height() - 48)
        width = min(available_width, max(820, min(980, int(anchor.width() * 0.58))))
        height = min(available_height, max(520, min(720, int(anchor.height() * 0.68))))
        x = anchor.left() + 24
        y = anchor.bottom() - height - 24
        if y < anchor.top() + 24:
            y = anchor.top() + 24
        return QtCore.QRect(x, y, width, height)

    def _position_from_anchor(self) -> None:
        anchor = self._anchor_rect if self._display_mode == self.EMBEDDED_MODE else self._global_anchor_rect()
        target = self._default_geometry_for_anchor(anchor)
        if target.isEmpty():
            return
        if self._display_mode == self.EMBEDDED_MODE:
            target = self._clamp_embedded_geometry(target)
        self._positioning_from_anchor = True
        self.setGeometry(target)
        self._positioning_from_anchor = False

    def _global_anchor_rect(self) -> QtCore.QRect:
        if self._host_widget is None or self._anchor_rect.isEmpty():
            return QtCore.QRect(self._anchor_rect)
        top_left = self._host_widget.mapToGlobal(self._anchor_rect.topLeft())
        return QtCore.QRect(top_left, self._anchor_rect.size())

    def _host_local_point_from_global(self, point: QtCore.QPoint) -> QtCore.QPoint:
        if self._host_widget is None:
            return QtCore.QPoint(point)
        return self._host_widget.mapFromGlobal(point)

    def _host_local_rect_from_global(self, rect: QtCore.QRect) -> QtCore.QRect:
        if rect.isEmpty():
            return QtCore.QRect()
        return QtCore.QRect(self._host_local_point_from_global(rect.topLeft()), rect.size())

    def _current_global_geometry(self) -> QtCore.QRect:
        if self._display_mode == self.WINDOW_MODE or self.isWindow():
            return QtCore.QRect(self.frameGeometry())
        if self._host_widget is None:
            return QtCore.QRect(self.geometry())
        return QtCore.QRect(self.mapToGlobal(self.rect().topLeft()), self.size())

    def _clamp_embedded_geometry(self, rect: QtCore.QRect) -> QtCore.QRect:
        if self._anchor_rect.isEmpty():
            return QtCore.QRect(rect)
        anchor = self._anchor_rect
        margin = 16
        max_width = max(self.minimumWidth(), anchor.width() - margin * 2)
        max_height = max(self.minimumHeight(), anchor.height() - margin * 2)
        width = min(max(rect.width(), self.minimumWidth()), max_width)
        height = min(max(rect.height(), self.minimumHeight()), max_height)
        min_x = anchor.left() + margin
        min_y = anchor.top() + margin
        max_x = max(min_x, anchor.right() - width - margin + 1)
        max_y = max(min_y, anchor.bottom() - height - margin + 1)
        x = min(max(rect.x(), min_x), max_x)
        y = min(max(rect.y(), min_y), max_y)
        return QtCore.QRect(x, y, width, height)

    def set_minimized(self, minimized: bool) -> None:
        if self._display_mode == self.WINDOW_MODE:
            if minimized:
                self.showMinimized()
            elif self.isMinimized():
                self.showNormal()
        elif not minimized:
            self.set_logs_visible(True)

    def close_panel(self) -> None:
        if self._pipeline_active and self._display_mode == self.WINDOW_MODE:
            self.showMinimized()
            return
        self.hide()

    def schedule_auto_close(self, milliseconds: int) -> None:
        if self._pipeline_active:
            return
        self._hide_timer.start(max(0, int(milliseconds)))

    def cancel_auto_close(self) -> None:
        if self._hide_timer.isActive():
            self._hide_timer.stop()

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
        target = self.preview_label.contentsRect().size() - QtCore.QSize(8, 8)
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
        if self._current_state == "done":
            self.open_output_button.setVisible(self.has_output_root())

    def has_output_root(self) -> bool:
        return bool(self._output_root and os.path.exists(self._output_root))

    def update_event(self, event: dict) -> None:
        phase = str(event.get("phase") or "")
        status = str(event.get("status") or "")
        panel_state = str(event.get("panel_state") or "").strip().lower()
        page_total = int(event.get("page_total") or 0)
        page_index = event.get("page_index")
        service_key = str(event.get("service") or "-")
        service_map = {
            "batch": self.tr("Automatic"),
            "gemma": "Gemma",
            "paddleocr_vl": "PaddleOCR VL",
            "paddleocr-vl": "PaddleOCR VL",
            "hunyuanocr": "HunyuanOCR",
            "mangalmm": "MangaLMM",
        }

        self.cancel_auto_close()
        if not self.isVisible():
            self.show()
        if self._display_mode == self.WINDOW_MODE:
            if not self.isMinimized():
                self.raise_()
        else:
            self.raise_()

        self.service_value.setText(service_map.get(service_key, service_key))
        if page_total > 0 and page_index is not None:
            self.progress_value.setText(f"{int(page_index) + 1}/{page_total}")
        else:
            self.progress_value.setText("-")
        self.file_value.setText(str(event.get("image_name") or "-"))
        self.eta_value.setText(str(event.get("eta_text") or self.tr("Calculating")))
        self.message_value.setText(str(event.get("message") or "-"))
        if "preview_path" in event:
            preview_path = str(event.get("preview_path") or "").strip()
            if preview_path or bool(event.get("clear_preview", False)):
                self._set_preview_path(preview_path)

        detail = str(event.get("detail") or "")
        log_line = str(event.get("message") or detail or "")
        if log_line:
            self._append_log(log_line)

        if phase == "done" or panel_state == "done":
            self._pipeline_active = False
            self._apply_state("done")
            auto_hide_ms = int(event.get("auto_hide_ms") or 0)
            if auto_hide_ms > 0:
                self.schedule_auto_close(auto_hide_ms)
        elif panel_state == "failed" or status == "failed" or phase == "error":
            self._pipeline_active = False
            self._apply_state("failed")
        elif panel_state == "cancelled" or status == "cancelled":
            self._pipeline_active = False
            self._apply_state("cancelled")
        elif panel_state == "cancelling" or status == "cancelling":
            self._pipeline_active = True
            self._apply_state("cancelling")
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
        self.cancel_auto_close()
        self.show()
        if self._display_mode == self.WINDOW_MODE and not self.isMinimized():
            self.raise_()
        elif self._display_mode == self.EMBEDDED_MODE:
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
            self.schedule_auto_close(int(duration) * 1000)

    def show_download_message(self, text: str) -> None:
        self.show_passive_message("info", text, duration=None, closable=True, source="download")

    def clear_download_message(self) -> None:
        if self._pipeline_active:
            return
        self.message_value.setText("-")
        self.service_value.setText("-")

    def _apply_state(self, state: str) -> None:
        self._current_state = state
        state_map = {
            "idle": self.tr("Idle"),
            "running": self.tr("Running"),
            "cancelling": self.tr("Running"),
            "done": self.tr("Done"),
            "failed": self.tr("Failed"),
            "cancelled": self.tr("Cancelled"),
        }
        self.title_label.setText(
            self.tr("Pipeline Status")
            if state in {"idle", "running", "cancelling", "done", "failed", "cancelled"}
            else self.tr("Status")
        )
        self.state_label.setText(state_map.get(state, state))
        self.pause_button.setVisible(state == "running" and self._series_queue_pause_visible)
        self.series_board_button.setVisible(state == "running" and self._series_queue_pause_visible)
        self.current_item_button.setVisible(state == "running" and self._series_queue_pause_visible)
        self.cancel_button.setVisible(state in {"running", "cancelling"})
        self.cancel_button.setEnabled(state == "running")
        self.report_button.setVisible(state in {"running", "cancelling", "done", "failed", "cancelled"})
        self.retry_button.setVisible(state == "failed")
        self.settings_button.setVisible(state == "failed")
        self.open_output_button.setVisible(state == "done" and self.has_output_root())
        self.close_button.setVisible(state in {"done", "failed", "cancelled"})
