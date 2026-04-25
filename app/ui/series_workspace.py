from __future__ import annotations

import hashlib
import os
import shutil
import tempfile

from PySide6 import QtCore, QtGui, QtWidgets

from modules.utils.archives import list_archive_image_entries, materialize_archive_entry

from .dayu_widgets import dayu_theme
from .dayu_widgets.tool_button import MToolButton
from .settings.series_page import SeriesPage


_DIRECT_PREVIEW_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".psd"}
_ARCHIVE_PREVIEW_EXTS = {".pdf", ".epub", ".zip", ".rar", ".7z", ".tar", ".cbz", ".cbr", ".cb7", ".cbt"}


def _hex_to_rgba(value: str, alpha: float) -> str:
    color = QtGui.QColor(value)
    color.setAlphaF(max(0.0, min(1.0, alpha)))
    return color.name(QtGui.QColor.NameFormat.HexArgb)


def _safe_mtime(path: str) -> float:
    try:
        return float(os.path.getmtime(path))
    except OSError:
        return 0.0


class _SeriesItemPreviewPopup(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.WindowType.ToolTip | QtCore.Qt.WindowType.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setObjectName("seriesItemPreviewPopup")
        self.setWindowFlag(QtCore.Qt.WindowType.NoDropShadowWindowHint, False)
        self._pixmap_cache: dict[str, QtGui.QPixmap | None] = {}
        self._preview_temp_dir = tempfile.mkdtemp(prefix="ct_series_preview_")

        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setColor(QtGui.QColor(0, 0, 0, 180))
        shadow.setOffset(0, 10)
        self.setGraphicsEffect(shadow)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.thumbnail_label = QtWidgets.QLabel(self)
        self.thumbnail_label.setObjectName("seriesItemPreviewThumbnail")
        self.thumbnail_label.setFixedSize(272, 168)
        self.thumbnail_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        meta_row = QtWidgets.QHBoxLayout()
        meta_row.setContentsMargins(0, 0, 0, 0)
        meta_row.setSpacing(8)

        self.type_chip = QtWidgets.QLabel(self)
        self.type_chip.setObjectName("seriesItemPreviewChip")
        self.type_chip.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.path_chip = QtWidgets.QLabel(self)
        self.path_chip.setObjectName("seriesItemPreviewPath")
        self.path_chip.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

        meta_row.addWidget(self.type_chip, 0)
        meta_row.addWidget(self.path_chip, 1)

        self.title_label = QtWidgets.QLabel(self)
        self.title_label.setObjectName("seriesItemPreviewTitle")
        self.title_label.setWordWrap(True)

        self.subtitle_label = QtWidgets.QLabel(self)
        self.subtitle_label.setObjectName("seriesItemPreviewSubtitle")
        self.subtitle_label.setWordWrap(True)

        layout.addWidget(self.thumbnail_label)
        layout.addLayout(meta_row)
        layout.addWidget(self.title_label)
        layout.addWidget(self.subtitle_label)

        self._apply_theme_styles()

    def _apply_theme_styles(self) -> None:
        accent = dayu_theme.primary_color or dayu_theme.yellow or "#fadb14"
        text = dayu_theme.primary_text_color or "#f5f5f5"
        sub_text = dayu_theme.secondary_text_color or "#b6b6b6"
        panel = dayu_theme.background_in_color or "#252a31"
        border = _hex_to_rgba(accent, 0.22)
        chip_fill = _hex_to_rgba(accent, 0.16)
        chip_border = _hex_to_rgba(accent, 0.34)
        path_fill = _hex_to_rgba(text, 0.06)
        self.setStyleSheet(
            f"""
            QFrame#seriesItemPreviewPopup {{
                background: {panel};
                border: 1px solid {border};
                border-radius: 18px;
            }}
            QLabel#seriesItemPreviewThumbnail {{
                background: {dayu_theme.background_color or "#1f242b"};
                border: 1px solid {_hex_to_rgba(text, 0.08)};
                border-radius: 14px;
                color: {sub_text};
            }}
            QLabel#seriesItemPreviewChip {{
                padding: 4px 10px;
                border-radius: 999px;
                background: {chip_fill};
                border: 1px solid {chip_border};
                color: {text};
                font-weight: 700;
            }}
            QLabel#seriesItemPreviewPath {{
                padding: 4px 10px;
                border-radius: 999px;
                background: {path_fill};
                color: {sub_text};
            }}
            QLabel#seriesItemPreviewTitle {{
                color: {text};
                font-size: 14px;
                font-weight: 700;
            }}
            QLabel#seriesItemPreviewSubtitle {{
                color: {sub_text};
                font-size: 12px;
            }}
            """
        )

    def closeEvent(self, event):  # type: ignore[override]
        try:
            shutil.rmtree(self._preview_temp_dir, ignore_errors=True)
        finally:
            super().closeEvent(event)

    def hide_preview(self) -> None:
        self.hide()

    def shutdown(self) -> None:
        self.hide()
        try:
            shutil.rmtree(self._preview_temp_dir, ignore_errors=True)
        except Exception:
            pass
        self._pixmap_cache.clear()

    def show_preview(self, payload: dict[str, object], global_pos: QtCore.QPoint) -> None:
        display_name = str(payload.get("display_name") or "")
        source_path = str(payload.get("source_origin_path") or "")
        relpath = str(payload.get("source_origin_relpath") or "")
        ext = os.path.splitext(source_path)[1].lstrip(".").upper() or "FILE"
        subtitle = relpath or source_path or self.tr("Series item")

        self.type_chip.setText(ext)
        self.path_chip.setText(self.tr("Preview"))
        self.title_label.setText(display_name or os.path.basename(source_path) or self.tr("Series item"))
        self.subtitle_label.setText(subtitle)
        pixmap = self._load_preview_pixmap(source_path)
        if pixmap is None or pixmap.isNull():
            pixmap = self._build_placeholder_pixmap(ext, display_name)
        self.thumbnail_label.setPixmap(self._rounded_scaled_pixmap(pixmap, self.thumbnail_label.size(), 14))
        self.adjustSize()
        self.move(self._bounded_position(global_pos + QtCore.QPoint(22, 18)))
        self.show()
        self.raise_()

    def _bounded_position(self, proposed: QtCore.QPoint) -> QtCore.QPoint:
        screen = QtGui.QGuiApplication.screenAt(proposed) or QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            return proposed
        available = screen.availableGeometry()
        x = min(proposed.x(), available.right() - self.width() - 12)
        y = min(proposed.y(), available.bottom() - self.height() - 12)
        x = max(available.left() + 12, x)
        y = max(available.top() + 12, y)
        return QtCore.QPoint(x, y)

    def _rounded_scaled_pixmap(
        self,
        pixmap: QtGui.QPixmap,
        target_size: QtCore.QSize,
        radius: int,
    ) -> QtGui.QPixmap:
        scaled = pixmap.scaled(
            target_size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        rounded = QtGui.QPixmap(target_size)
        rounded.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(rounded)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(rounded.rect()), radius, radius)
        painter.setClipPath(path)
        x = (target_size.width() - scaled.width()) // 2
        y = (target_size.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        painter.end()
        return rounded

    def _build_placeholder_pixmap(self, ext: str, title: str) -> QtGui.QPixmap:
        size = QtCore.QSize(272, 168)
        pixmap = QtGui.QPixmap(size)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        gradient = QtGui.QLinearGradient(0, 0, size.width(), size.height())
        gradient.setColorAt(0.0, QtGui.QColor(_hex_to_rgba(dayu_theme.primary_color or "#fadb14", 0.24)))
        gradient.setColorAt(1.0, QtGui.QColor(dayu_theme.background_out_color or "#3a3f47"))
        painter.setBrush(QtGui.QBrush(gradient))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QtCore.QRectF(0, 0, size.width(), size.height()), 14, 14)

        badge_rect = QtCore.QRectF(16, 16, 90, 38)
        painter.setBrush(QtGui.QColor(_hex_to_rgba(dayu_theme.background_color or "#171a1f", 0.92)))
        painter.drawRoundedRect(badge_rect, 12, 12)
        painter.setPen(QtGui.QColor(dayu_theme.primary_text_color or "#f5f5f5"))
        badge_font = QtGui.QFont(self.font())
        badge_font.setBold(True)
        badge_font.setPointSize(max(10, badge_font.pointSize()))
        painter.setFont(badge_font)
        painter.drawText(badge_rect, QtCore.Qt.AlignmentFlag.AlignCenter, ext)

        title_font = QtGui.QFont(self.font())
        title_font.setBold(True)
        title_font.setPointSize(max(14, title_font.pointSize() + 2))
        painter.setFont(title_font)
        painter.drawText(
            QtCore.QRectF(18, 72, size.width() - 36, 34),
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
            self.tr("Preview Unavailable"),
        )
        painter.setPen(QtGui.QColor(dayu_theme.secondary_text_color or "#c0c0c0"))
        body_font = QtGui.QFont(self.font())
        body_font.setPointSize(max(10, body_font.pointSize()))
        painter.setFont(body_font)
        painter.drawText(
            QtCore.QRectF(18, 108, size.width() - 36, 42),
            int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
            | int(QtCore.Qt.TextFlag.TextWordWrap),
            title or self.tr("This item does not have an image preview, so a file card is shown instead."),
        )
        painter.end()
        return pixmap

    def _archive_preview_path(self, source_path: str) -> str:
        ext = ".png"
        digest = hashlib.sha1(source_path.encode("utf-8")).hexdigest()
        return os.path.join(self._preview_temp_dir, f"{digest}{ext}")

    def _load_preview_pixmap(self, source_path: str) -> QtGui.QPixmap | None:
        clean_path = os.path.normpath(os.path.abspath(source_path or ""))
        if not clean_path:
            return None
        if clean_path in self._pixmap_cache:
            return self._pixmap_cache[clean_path]

        pixmap: QtGui.QPixmap | None = None
        ext = os.path.splitext(clean_path)[1].lower()
        if ext in _DIRECT_PREVIEW_EXTS and os.path.exists(clean_path):
            loaded = QtGui.QPixmap(clean_path)
            pixmap = loaded if not loaded.isNull() else None
        elif ext in _ARCHIVE_PREVIEW_EXTS and os.path.exists(clean_path):
            preview_path = self._archive_preview_path(clean_path)
            if not os.path.exists(preview_path):
                try:
                    if ext == ".pdf":
                        entry = {"kind": "pdf_page", "page_index": 0, "ext": ".png"}
                    else:
                        entries = list_archive_image_entries(clean_path)
                        entry = entries[0] if entries else None
                    if entry is not None:
                        materialize_archive_entry(clean_path, entry, preview_path)
                except Exception:
                    preview_path = ""
            if preview_path and os.path.exists(preview_path):
                loaded = QtGui.QPixmap(preview_path)
                pixmap = loaded if not loaded.isNull() else None
        self._pixmap_cache[clean_path] = pixmap
        return pixmap


class _SeriesQueueTable(QtWidgets.QTableWidget):
    order_changed = QtCore.Signal(list)
    open_requested = QtCore.Signal(str)
    remove_requested = QtCore.Signal(str)
    queue_index_requested = QtCore.Signal(str, int)
    hover_preview_requested = QtCore.Signal(dict, QtCore.QPoint)
    hover_preview_hidden = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._suppress_item_changed = False
        self._queue_running = False
        self._active_item_id = ""
        self._lock_reason = ""
        self._hovered_item_id = ""
        self.setColumnCount(6)
        self.setHorizontalHeaderLabels(
            [
                self.tr("No."),
                self.tr("Project"),
                self.tr("Type"),
                self.tr("Folder"),
                self.tr("Status"),
                "",
            ]
        )
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.viewport().installEventFilter(self)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(40)
        self.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
            | QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
        )
        header = self.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)

        self.itemChanged.connect(self._on_item_changed)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.apply_theme_styles()

    def apply_theme_styles(self) -> None:
        text = dayu_theme.primary_text_color or "#d9d9d9"
        sub_text = dayu_theme.secondary_text_color or "#a6a6a6"
        panel = dayu_theme.background_color or "#323232"
        panel_alt = dayu_theme.background_in_color or "#3a3a3a"
        header = dayu_theme.background_out_color or "#494949"
        border = dayu_theme.divider_color or "#262626"
        accent = dayu_theme.primary_color or dayu_theme.yellow or "#fadb14"
        accent_soft = _hex_to_rgba(accent, 0.18)
        handle = _hex_to_rgba(text, 0.22)
        danger_soft = _hex_to_rgba("#ff7875", 0.18)
        self.setStyleSheet(
            f"""
            QTableWidget {{
                background: {panel};
                border: 1px solid {border};
                border-radius: 12px;
                alternate-background-color: {panel_alt};
                color: {text};
                selection-background-color: {accent_soft};
                selection-color: {text};
                outline: none;
            }}
            QTableWidget::item {{
                background: transparent;
                border: none;
                border-bottom: 1px solid {border};
                padding: 7px 10px;
            }}
            QTableWidget::item:selected {{
                background: {accent_soft};
                color: {text};
            }}
            QHeaderView::section {{
                background: {header};
                color: {text};
                border: none;
                border-bottom: 1px solid {border};
                padding: 9px 10px;
                font-weight: 600;
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 12px;
                margin: 8px 0 8px 0;
            }}
            QScrollBar::handle:vertical {{
                background: {handle};
                border-radius: 6px;
                min-height: 28px;
            }}
            QToolButton#seriesRemoveButton {{
                min-width: 28px;
                min-height: 28px;
                max-width: 28px;
                max-height: 28px;
                border: none;
                border-radius: 10px;
                background: transparent;
                padding: 0;
            }}
            QToolButton#seriesRemoveButton:hover {{
                background: {danger_soft};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
            """
        )

    def set_interaction_state(
        self,
        *,
        queue_running: bool,
        active_item_id: str | None = None,
        lock_reason: str = "",
    ) -> None:
        self._queue_running = bool(queue_running)
        self._active_item_id = str(active_item_id or "")
        self._lock_reason = str(lock_reason or "")
        if self._queue_running:
            self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.NoDragDrop)
            self.setDragEnabled(False)
            self.setAcceptDrops(False)
            self.setDropIndicatorShown(False)
            self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        else:
            self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
            self.setDragEnabled(True)
            self.setAcceptDrops(True)
            self.setDropIndicatorShown(True)
            self.setEditTriggers(
                QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
                | QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            )
        self.viewport().setToolTip(self._lock_reason if self._queue_running else "")

    def _queue_item_flags(self) -> QtCore.Qt.ItemFlag:
        flags = QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
        if not self._queue_running:
            flags |= QtCore.Qt.ItemFlag.ItemIsEditable
        return flags

    def _name_item_flags(self) -> QtCore.Qt.ItemFlag:
        flags = QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
        if not self._queue_running:
            flags |= QtCore.Qt.ItemFlag.ItemIsDragEnabled
        return flags

    def _status_label(self, status: str) -> str:
        normalized = str(status or "pending").strip().lower()
        mapping = {
            "pending": self.tr("Pending"),
            "running": self.tr("Running"),
            "done": self.tr("Done"),
            "failed": self.tr("Failed"),
            "skipped": self.tr("Skipped"),
        }
        return mapping.get(normalized, normalized or self.tr("Pending"))

    def _apply_row_style(self, row: int, *, item_id: str, status: str) -> None:
        is_running = str(status).strip().lower() == "running" or item_id == self._active_item_id
        muted_color = QtGui.QColor(dayu_theme.secondary_text_color or "#a6a6a6")
        accent = QtGui.QColor(dayu_theme.primary_color or dayu_theme.yellow or "#fadb14")
        running_fg = QtGui.QColor(dayu_theme.background_color or "#323232")
        running_bg = QtGui.QColor(accent)

        for column in range(5):
            table_item = self.item(row, column)
            if table_item is None:
                continue
            font = table_item.font()
            font.setBold(is_running)
            table_item.setFont(font)
            if is_running:
                table_item.setForeground(QtGui.QBrush(running_fg))
                table_item.setBackground(QtGui.QBrush(running_bg))
                table_item.setToolTip(self.tr("Currently translating this project in the queue."))
            elif self._queue_running:
                table_item.setForeground(QtGui.QBrush(muted_color))
                table_item.setBackground(QtGui.QBrush())
                if self._lock_reason:
                    table_item.setToolTip(self._lock_reason)
            else:
                table_item.setForeground(QtGui.QBrush())
                table_item.setBackground(QtGui.QBrush())
                table_item.setToolTip("")

    def _row_payload(self, row: int) -> dict[str, object]:
        queue_item = self.item(row, 0)
        if queue_item is None:
            return {}
        payload = queue_item.data(QtCore.Qt.ItemDataRole.UserRole + 1)
        return dict(payload or {})

    def set_series_items(self, items: list[dict[str, object]]) -> None:
        self._suppress_item_changed = True
        self.setRowCount(len(items))
        for row, item in enumerate(items):
            item_id = str(item.get("series_item_id") or "")
            queue_text = f"{int(item.get('queue_index', row + 1) or (row + 1)):02d}"
            status = str(item.get("status") or "pending")
            source_path = str(item.get("source_origin_path") or "")
            payload = {
                "series_item_id": item_id,
                "display_name": str(item.get("display_name") or ""),
                "source_origin_path": source_path,
                "source_origin_relpath": str(item.get("source_origin_relpath") or ""),
                "source_kind": str(item.get("source_kind") or ""),
                "status": status,
                "queue_index": int(item.get("queue_index", row + 1) or (row + 1)),
                "modified_ts": _safe_mtime(source_path),
            }

            queue_item = QtWidgets.QTableWidgetItem(queue_text)
            queue_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            queue_item.setData(QtCore.Qt.ItemDataRole.UserRole, item_id)
            queue_item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, payload)
            queue_item.setFlags(self._queue_item_flags())

            name_item = QtWidgets.QTableWidgetItem(str(item.get("display_name") or ""))
            name_item.setData(QtCore.Qt.ItemDataRole.UserRole, item_id)
            name_item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, payload)
            name_item.setToolTip(source_path)
            name_item.setFlags(self._name_item_flags())

            source_kind = str(item.get("source_kind") or "")
            source_kind_label = (
                self.tr("Project File")
                if source_kind == "ctpr_import"
                else self.tr("Source File")
            )
            type_item = QtWidgets.QTableWidgetItem(source_kind_label)
            type_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)

            folder_item = QtWidgets.QTableWidgetItem(str(item.get("source_origin_relpath") or ""))
            folder_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)

            status_item = QtWidgets.QTableWidgetItem(self._status_label(status))
            status_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
            status_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

            remove_button = MToolButton(self).svg("trash_line.svg").small().icon_only()
            remove_button.setObjectName("seriesRemoveButton")
            remove_button.setToolTip(
                self._lock_reason if self._queue_running else self.tr("Remove from series")
            )
            remove_button.setEnabled(not self._queue_running)
            remove_button.clicked.connect(lambda _=False, target=item_id: self.remove_requested.emit(target))

            self.setItem(row, 0, queue_item)
            self.setItem(row, 1, name_item)
            self.setItem(row, 2, type_item)
            self.setItem(row, 3, folder_item)
            self.setItem(row, 4, status_item)
            self.setCellWidget(row, 5, remove_button)
            self._apply_row_style(row, item_id=item_id, status=status)
        self._suppress_item_changed = False

    def ordered_item_ids(self) -> list[str]:
        ordered = []
        for row in range(self.rowCount()):
            queue_item = self.item(row, 0)
            item_id = queue_item.data(QtCore.Qt.ItemDataRole.UserRole) if queue_item is not None else ""
            if item_id:
                ordered.append(str(item_id))
        return ordered

    def selected_item_id(self) -> str:
        row = self.currentRow()
        if row < 0:
            return ""
        item = self.item(row, 0)
        if item is None:
            return ""
        return str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")

    def dropEvent(self, event):  # type: ignore[override]
        if self._queue_running:
            event.ignore()
            return
        super().dropEvent(event)
        self._renumber_rows()
        self.order_changed.emit(self.ordered_item_ids())

    def _renumber_rows(self) -> None:
        self._suppress_item_changed = True
        for row in range(self.rowCount()):
            item = self.item(row, 0)
            if item is not None:
                item.setText(f"{row + 1:02d}")
        self._suppress_item_changed = False

    def _move_row(self, source_row: int, target_row: int) -> None:
        cells = []
        widgets = []
        for column in range(self.columnCount()):
            item = self.item(source_row, column)
            cells.append(item.clone() if item is not None else None)
            widgets.append(self.cellWidget(source_row, column))

        self.removeRow(source_row)
        self.insertRow(target_row)
        for column, cloned in enumerate(cells):
            if cloned is not None:
                self.setItem(target_row, column, cloned)
        for column, widget in enumerate(widgets):
            if widget is not None:
                self.setCellWidget(target_row, column, widget)
        self._renumber_rows()

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._suppress_item_changed or item is None or item.column() != 0:
            return
        if self._queue_running:
            item.setText(f"{item.row() + 1:02d}")
            return
        item_id = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
        try:
            requested_index = max(1, int(item.text()))
        except (TypeError, ValueError):
            requested_index = item.row() + 1
        self.queue_index_requested.emit(item_id, requested_index)

    def _on_item_double_clicked(self, item: QtWidgets.QTableWidgetItem) -> None:
        if item is None or self._queue_running:
            return
        item_id = ""
        queue_item = self.item(item.row(), 0)
        if queue_item is not None:
            item_id = str(queue_item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
        if item_id:
            self.open_requested.emit(item_id)

    def sorted_item_ids(self, mode: str) -> list[str]:
        rows = []
        for row in range(self.rowCount()):
            payload = self._row_payload(row)
            if payload:
                rows.append(payload)
        if mode == "name_asc":
            rows.sort(key=lambda item: (str(item.get("display_name") or "").casefold(), str(item.get("source_origin_path") or "").casefold()))
        elif mode == "name_desc":
            rows.sort(key=lambda item: (str(item.get("display_name") or "").casefold(), str(item.get("source_origin_path") or "").casefold()), reverse=True)
        elif mode == "date_desc":
            rows.sort(key=lambda item: (float(item.get("modified_ts") or 0.0), str(item.get("display_name") or "").casefold()), reverse=True)
        elif mode == "date_asc":
            rows.sort(key=lambda item: (float(item.get("modified_ts") or 0.0), str(item.get("display_name") or "").casefold()))
        return [str(item.get("series_item_id") or "") for item in rows if str(item.get("series_item_id") or "").strip()]

    def eventFilter(self, watched, event):  # type: ignore[override]
        if watched is self.viewport():
            event_type = event.type()
            if event_type == QtCore.QEvent.Type.MouseMove:
                mouse_event = event
                row = self.rowAt(mouse_event.pos().y())
                if row < 0:
                    if self._hovered_item_id:
                        self._hovered_item_id = ""
                        self.hover_preview_hidden.emit()
                    return super().eventFilter(watched, event)
                payload = self._row_payload(row)
                item_id = str(payload.get("series_item_id") or "")
                if not item_id:
                    return super().eventFilter(watched, event)
                self._hovered_item_id = item_id
                self.hover_preview_requested.emit(payload, self.viewport().mapToGlobal(mouse_event.pos()))
            elif event_type in {
                QtCore.QEvent.Type.Leave,
                QtCore.QEvent.Type.Hide,
                QtCore.QEvent.Type.WindowDeactivate,
            }:
                if self._hovered_item_id:
                    self._hovered_item_id = ""
                    self.hover_preview_hidden.emit()
        return super().eventFilter(watched, event)

    def closeEvent(self, event):  # type: ignore[override]
        try:
            viewport = self.viewport()
            if viewport is not None:
                viewport.removeEventFilter(self)
        except Exception:
            pass
        super().closeEvent(event)


class _SeriesQuickSettings(QtWidgets.QWidget):
    changed = QtCore.Signal()
    auto_translate_requested = QtCore.Signal()
    open_series_settings_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lock_reason = ""

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header = QtWidgets.QLabel(self.tr("Global Queue Settings"))
        header.setObjectName("seriesQuickHeader")
        header.setWordWrap(True)
        note = QtWidgets.QLabel(
            self.tr(
                "These controls apply to queue execution. Open a child project to edit detailed page-level settings."
            )
        )
        note.setWordWrap(True)
        note.setObjectName("seriesQuickNote")

        self.source_lang_combo = QtWidgets.QComboBox()
        self.target_lang_combo = QtWidgets.QComboBox()
        self.ocr_combo = QtWidgets.QComboBox()
        self.translator_combo = QtWidgets.QComboBox()
        self.workflow_combo = QtWidgets.QComboBox()
        self.use_gpu_checkbox = QtWidgets.QCheckBox(self.tr("Use GPU"))
        self.use_gpu_checkbox.setChecked(True)

        form = QtWidgets.QFormLayout()
        form.addRow(self.tr("Source language:"), self.source_lang_combo)
        form.addRow(self.tr("Target language:"), self.target_lang_combo)
        form.addRow(self.tr("OCR:"), self.ocr_combo)
        form.addRow(self.tr("Translator:"), self.translator_combo)
        form.addRow(self.tr("Workflow mode:"), self.workflow_combo)
        form.addRow("", self.use_gpu_checkbox)

        self.render_summary_label = QtWidgets.QLabel(self.tr("Render: --"))
        self.render_summary_label.setObjectName("seriesQuickNote")
        self.render_summary_label.setWordWrap(True)
        self.export_summary_label = QtWidgets.QLabel(self.tr("Export: --"))
        self.export_summary_label.setObjectName("seriesQuickNote")
        self.export_summary_label.setWordWrap(True)

        self._series_settings_tooltip = self.tr(
            "Edit queue, pipeline, render, export, and debug defaults for this series."
        )
        self.series_settings_button = QtWidgets.QPushButton(self.tr("Series Design / Global Settings…"))
        self.series_settings_button.setToolTip(self._series_settings_tooltip)
        self.auto_translate_button = QtWidgets.QPushButton(self.tr("Translate in Queue Order"))
        self.auto_translate_button.setObjectName("seriesAutoTranslateButton")

        layout.addWidget(header)
        layout.addWidget(note)
        layout.addLayout(form)
        layout.addWidget(self.render_summary_label)
        layout.addWidget(self.export_summary_label)
        layout.addWidget(self.series_settings_button)
        layout.addWidget(self.auto_translate_button)
        layout.addStretch(1)

        for combo in (
            self.source_lang_combo,
            self.target_lang_combo,
            self.ocr_combo,
            self.translator_combo,
            self.workflow_combo,
        ):
            combo.currentIndexChanged.connect(self.changed)
        self.use_gpu_checkbox.stateChanged.connect(self.changed)
        self.series_settings_button.clicked.connect(self.open_series_settings_requested)
        self.auto_translate_button.clicked.connect(self.auto_translate_requested)

    def set_locked(self, locked: bool, reason: str = "") -> None:
        self._lock_reason = str(reason or "")
        widgets = (
            self.source_lang_combo,
            self.target_lang_combo,
            self.ocr_combo,
            self.translator_combo,
            self.workflow_combo,
            self.use_gpu_checkbox,
            self.series_settings_button,
        )
        for widget in widgets:
            widget.setEnabled(not locked)
            if widget is self.series_settings_button:
                widget.setToolTip(self._lock_reason if locked else self._series_settings_tooltip)
            else:
                widget.setToolTip(self._lock_reason if locked else "")

    def set_options(self, *, languages, ocr_modes, translators, workflow_modes) -> None:
        self._populate_combo(self.source_lang_combo, languages)
        self._populate_combo(self.target_lang_combo, languages)
        self._populate_combo(self.ocr_combo, ocr_modes)
        self._populate_combo(self.translator_combo, translators)
        self._populate_combo(self.workflow_combo, workflow_modes)

    def _populate_combo(self, combo: QtWidgets.QComboBox, options: list[tuple[str, str]]) -> None:
        current = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for value, label in options:
            combo.addItem(label, value)
        if current:
            idx = combo.findData(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def set_values(self, values: dict[str, object]) -> None:
        blockers = [
            QtCore.QSignalBlocker(self.source_lang_combo),
            QtCore.QSignalBlocker(self.target_lang_combo),
            QtCore.QSignalBlocker(self.ocr_combo),
            QtCore.QSignalBlocker(self.translator_combo),
            QtCore.QSignalBlocker(self.workflow_combo),
            QtCore.QSignalBlocker(self.use_gpu_checkbox),
        ]
        self._set_combo_value(self.source_lang_combo, str(values.get("source_language") or ""))
        self._set_combo_value(self.target_lang_combo, str(values.get("target_language") or ""))
        self._set_combo_value(self.ocr_combo, str(values.get("ocr") or ""))
        self._set_combo_value(self.translator_combo, str(values.get("translator") or ""))
        self._set_combo_value(self.workflow_combo, str(values.get("workflow_mode") or ""))
        self.use_gpu_checkbox.setChecked(bool(values.get("use_gpu", True)))
        self._update_summaries(values)
        del blockers

    def values(self) -> dict[str, object]:
        return {
            "source_language": str(self.source_lang_combo.currentData() or ""),
            "target_language": str(self.target_lang_combo.currentData() or ""),
            "ocr": str(self.ocr_combo.currentData() or ""),
            "translator": str(self.translator_combo.currentData() or ""),
            "workflow_mode": str(self.workflow_combo.currentData() or ""),
            "use_gpu": self.use_gpu_checkbox.isChecked(),
        }

    def _set_combo_value(self, combo: QtWidgets.QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index < 0 and combo.count() > 0:
            index = 0
        if index >= 0:
            combo.setCurrentIndex(index)

    def _update_summaries(self, values: dict[str, object]) -> None:
        render = values.get("render_settings")
        render = render if isinstance(render, dict) else {}
        font = str(render.get("font_family") or "--")
        max_font = render.get("max_font_size", "--")
        line_spacing = str(render.get("line_spacing") or "--")
        align_labels = [self.tr("left"), self.tr("center"), self.tr("right")]
        align_id = int(render.get("alignment_id", 1) or 1)
        align = align_labels[align_id] if 0 <= align_id < len(align_labels) else self.tr("center")
        outline = (
            self.tr("outline {width}").format(width=str(render.get("outline_width") or "1.0"))
            if bool(render.get("outline", False))
            else self.tr("outline off")
        )
        self.render_summary_label.setText(
            self.tr("Render: {font} / max {max_font} / line {line_spacing} / {align} / {outline}").format(
                font=font,
                max_font=max_font,
                line_spacing=line_spacing,
                align=align,
                outline=outline,
            )
        )

        export = values.get("export_settings")
        export = export if isinstance(export, dict) else {}
        target = str(export.get("automatic_output_target") or "--")
        debug_keys = (
            "export_inpainted_image",
            "export_detector_overlay",
            "export_raw_mask",
            "export_mask_overlay",
            "export_cleanup_mask_delta",
            "export_debug_metadata",
        )
        debug_count = sum(1 for key in debug_keys if bool(export.get(key, False)))
        self.export_summary_label.setText(
            self.tr("Export: {target} / debug {count} enabled").format(
                target=target,
                count=debug_count,
            )
        )


class SeriesSettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Series Design / Global Settings"))
        self.setObjectName("seriesSettingsDialog")
        self.resize(940, 860)
        self.setMinimumSize(820, 640)
        self._output_options: dict[str, list[tuple[str, str]]] = {}

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        title = QtWidgets.QLabel(self.tr("Series Design / Global Settings"), self)
        title.setObjectName("seriesSettingsTitle")
        subtitle = QtWidgets.QLabel(
            self.tr(
                "These defaults are applied when the series queue runs. Child projects keep their own page-level edits."
            ),
            self,
        )
        subtitle.setObjectName("seriesSettingsSubtitle")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)

        self.tabs = QtWidgets.QTabWidget(self)
        self.tabs.setObjectName("seriesSettingsTabs")
        self.tabs.setDocumentMode(True)
        root.addWidget(self.tabs, 1)

        self._build_queue_tab()

        self._build_pipeline_tab()
        self._build_render_tab()
        self._build_export_tab()

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self._apply_style()

    def _build_queue_tab(self) -> None:
        tab, layout = self._make_scroll_tab()
        self.queue_page = SeriesPage(tab)
        layout.addWidget(self.queue_page)
        layout.addStretch(1)
        self.tabs.addTab(tab, self.tr("Queue"))

    def _build_pipeline_tab(self) -> None:
        tab, layout = self._make_scroll_tab()

        self.source_lang_combo = QtWidgets.QComboBox(tab)
        self.target_lang_combo = QtWidgets.QComboBox(tab)
        self.ocr_combo = QtWidgets.QComboBox(tab)
        self.translator_combo = QtWidgets.QComboBox(tab)
        self.workflow_combo = QtWidgets.QComboBox(tab)
        self.use_gpu_checkbox = QtWidgets.QCheckBox(self.tr("Use GPU"), tab)
        self.use_gpu_checkbox.setChecked(True)

        language_form = self._make_rows_layout(
            self._make_field_row(self.tr("Source language:"), self.source_lang_combo),
            self._make_field_row(self.tr("Target language:"), self.target_lang_combo),
        )
        self.pipeline_language_group = self._make_section(
            self.tr("Languages"),
            self.tr("Set the source and target language for every queued child project."),
            language_form,
        )

        runtime_form = self._make_rows_layout(
            self._make_field_row(self.tr("OCR:"), self.ocr_combo),
            self._make_field_row(self.tr("Translator:"), self.translator_combo),
            self._make_field_row(self.tr("Workflow mode:"), self.workflow_combo),
            self._make_check_row(self.use_gpu_checkbox),
        )
        self.pipeline_runtime_group = self._make_section(
            self.tr("Pipeline Runtime"),
            self.tr("Choose the queue workflow and the runtime services it should use."),
            runtime_form,
        )
        layout.addWidget(self.pipeline_language_group)
        layout.addWidget(self.pipeline_runtime_group)
        layout.addStretch(1)
        self.tabs.addTab(tab, self.tr("Pipeline"))

    def _build_render_tab(self) -> None:
        tab, layout = self._make_scroll_tab()

        self.font_combo = QtWidgets.QComboBox(tab)
        self.font_combo.setEditable(True)
        self.min_font_spin = QtWidgets.QSpinBox(tab)
        self.min_font_spin.setRange(1, 300)
        self.max_font_spin = QtWidgets.QSpinBox(tab)
        self.max_font_spin.setRange(1, 300)
        self.line_spacing_combo = QtWidgets.QComboBox(tab)
        self.line_spacing_combo.setEditable(True)
        self.line_spacing_combo.addItems(["1.0", "1.1", "1.2", "1.3", "1.4", "1.5"])
        self.text_color_button = self._make_color_button("#000000")
        self.force_color_checkbox = QtWidgets.QCheckBox(self.tr("Use Selected Color"), tab)
        self.horizontal_alignment_combo = QtWidgets.QComboBox(tab)
        self.horizontal_alignment_combo.addItem(self.tr("Left"), 0)
        self.horizontal_alignment_combo.addItem(self.tr("Center"), 1)
        self.horizontal_alignment_combo.addItem(self.tr("Right"), 2)
        self.vertical_alignment_combo = QtWidgets.QComboBox(tab)
        self.vertical_alignment_combo.addItem(self.tr("Top"), 0)
        self.vertical_alignment_combo.addItem(self.tr("Center"), 1)
        self.vertical_alignment_combo.addItem(self.tr("Bottom"), 2)
        self.bold_checkbox = QtWidgets.QCheckBox(self.tr("Bold"), tab)
        self.italic_checkbox = QtWidgets.QCheckBox(self.tr("Italic"), tab)
        self.underline_checkbox = QtWidgets.QCheckBox(self.tr("Underline"), tab)
        self.uppercase_checkbox = QtWidgets.QCheckBox(self.tr("Uppercase"), tab)
        self.outline_checkbox = QtWidgets.QCheckBox(self.tr("Outline"), tab)
        self.outline_color_button = self._make_color_button("#ffffff")
        self.outline_width_combo = QtWidgets.QComboBox(tab)
        self.outline_width_combo.setEditable(True)
        self.outline_width_combo.addItems(["1.0", "1.15", "1.3", "1.4", "1.5", "2.0", "3.0"])

        style_row = QtWidgets.QWidget(tab)
        style_layout = QtWidgets.QHBoxLayout(style_row)
        style_layout.setContentsMargins(0, 0, 0, 0)
        style_layout.addWidget(self.bold_checkbox)
        style_layout.addWidget(self.italic_checkbox)
        style_layout.addWidget(self.underline_checkbox)
        style_layout.addWidget(self.uppercase_checkbox)
        style_layout.addStretch(1)

        color_row = QtWidgets.QWidget(tab)
        color_layout = QtWidgets.QHBoxLayout(color_row)
        color_layout.setContentsMargins(0, 0, 0, 0)
        color_layout.addWidget(self.text_color_button)
        color_layout.addWidget(self.force_color_checkbox)
        color_layout.addStretch(1)

        outline_row = QtWidgets.QWidget(tab)
        outline_layout = QtWidgets.QHBoxLayout(outline_row)
        outline_layout.setContentsMargins(0, 0, 0, 0)
        outline_layout.addWidget(self.outline_checkbox)
        outline_layout.addWidget(self.outline_color_button)
        outline_layout.addWidget(self.outline_width_combo)
        outline_layout.addStretch(1)

        typography_form = self._make_rows_layout(
            self._make_field_row(self.tr("Font:"), self.font_combo),
            self._make_field_row(self.tr("Min font size:"), self.min_font_spin),
            self._make_field_row(self.tr("Max font size:"), self.max_font_spin),
            self._make_field_row(self.tr("Line spacing:"), self.line_spacing_combo),
        )
        self.render_typography_group = self._make_section(
            self.tr("Typography"),
            self.tr("Control the base font and automatic font-fit limits for generated text."),
            typography_form,
        )

        color_form = self._make_rows_layout(
            self._make_field_row(self.tr("Text color:"), color_row),
        )
        self.render_color_group = self._make_section(
            self.tr("Color"),
            self.tr("Use a fixed text color when the queue renders translated pages."),
            color_form,
        )

        alignment_form = self._make_rows_layout(
            self._make_field_row(self.tr("Horizontal:"), self.horizontal_alignment_combo),
            self._make_field_row(self.tr("Vertical:"), self.vertical_alignment_combo),
        )
        self.render_alignment_group = self._make_section(
            self.tr("Alignment"),
            self.tr("Align text inside each detected text box."),
            alignment_form,
        )

        style_form = self._make_rows_layout(
            self._make_field_row(self.tr("Style:"), style_row),
            self._make_field_row(self.tr("Outline:"), outline_row),
        )
        self.render_style_group = self._make_section(
            self.tr("Style and Outline"),
            self.tr("Apply emphasis and outline defaults across the series queue."),
            style_form,
        )
        layout.addWidget(self.render_typography_group)
        layout.addWidget(self.render_color_group)
        layout.addWidget(self.render_alignment_group)
        layout.addWidget(self.render_style_group)
        layout.addStretch(1)
        self.tabs.addTab(tab, self.tr("Render"))

    def _build_export_tab(self) -> None:
        tab, layout = self._make_scroll_tab()

        self.output_target_combo = QtWidgets.QComboBox(tab)
        self.output_image_format_combo = QtWidgets.QComboBox(tab)
        self.output_archive_format_combo = QtWidgets.QComboBox(tab)
        self.output_archive_image_format_combo = QtWidgets.QComboBox(tab)
        self.output_archive_level_spin = QtWidgets.QSpinBox(tab)
        self.output_archive_level_spin.setRange(0, 9)
        form = self._make_rows_layout(
            self._make_field_row(self.tr("Output target:"), self.output_target_combo),
            self._make_field_row(self.tr("Image format:"), self.output_image_format_combo),
            self._make_field_row(self.tr("Archive format:"), self.output_archive_format_combo),
            self._make_field_row(self.tr("Archive image format:"), self.output_archive_image_format_combo),
            self._make_field_row(self.tr("Archive compression:"), self.output_archive_level_spin),
        )
        self.export_output_group = self._make_section(
            self.tr("Final Output"),
            self.tr("Choose where the completed translated result is saved."),
            form,
        )

        self.export_raw_text_checkbox = QtWidgets.QCheckBox(self.tr("Raw source text"), tab)
        self.export_translated_text_checkbox = QtWidgets.QCheckBox(self.tr("Translated text"), tab)
        text_exports_layout = self._make_rows_layout(
            self._make_check_row(self.export_raw_text_checkbox),
            self._make_check_row(self.export_translated_text_checkbox),
        )
        self.export_text_group = self._make_section(
            self.tr("Text Exports"),
            self.tr("Write OCR and translation text files next to the queue output."),
            text_exports_layout,
        )

        self.export_inpainted_image_checkbox = QtWidgets.QCheckBox(self.tr("Inpainted image"), tab)
        self.export_detector_overlay_checkbox = QtWidgets.QCheckBox(self.tr("Detector overlay"), tab)
        self.export_raw_mask_checkbox = QtWidgets.QCheckBox(self.tr("Raw inpaint mask"), tab)
        self.export_mask_overlay_checkbox = QtWidgets.QCheckBox(self.tr("Mask overlay"), tab)
        self.export_cleanup_mask_delta_checkbox = QtWidgets.QCheckBox(self.tr("Cleanup mask delta"), tab)
        self.export_debug_metadata_checkbox = QtWidgets.QCheckBox(self.tr("Debug metadata"), tab)
        debug_layout = QtWidgets.QVBoxLayout()
        debug_layout.setContentsMargins(0, 0, 0, 0)
        debug_layout.setSpacing(8)
        for checkbox in (
            self.export_inpainted_image_checkbox,
            self.export_detector_overlay_checkbox,
            self.export_raw_mask_checkbox,
            self.export_mask_overlay_checkbox,
            self.export_cleanup_mask_delta_checkbox,
            self.export_debug_metadata_checkbox,
        ):
            debug_layout.addWidget(self._make_check_row(checkbox))
        self.export_debug_group = self._make_section(
            self.tr("Debug Artifacts"),
            self.tr(
                "Only checked debug artifacts are created. When unchecked, the status panel logs that preview generation is disabled."
            ),
            debug_layout,
        )
        layout.addWidget(self.export_output_group)
        layout.addWidget(self.export_text_group)
        layout.addWidget(self.export_debug_group)
        layout.addStretch(1)
        self.tabs.addTab(tab, self.tr("Export / Debug"))

    def _make_scroll_tab(self) -> tuple[QtWidgets.QScrollArea, QtWidgets.QVBoxLayout]:
        content = QtWidgets.QWidget(self)
        content.setObjectName("seriesSettingsTabContent")
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(22)
        scroll = QtWidgets.QScrollArea(self)
        scroll.setObjectName("seriesSettingsScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        return scroll, layout

    def _make_rows_layout(self, *rows: QtWidgets.QWidget) -> QtWidgets.QVBoxLayout:
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        for row in rows:
            layout.addWidget(row)
        return layout

    def _make_field_row(self, label_text: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        row = QtWidgets.QFrame(self)
        row.setObjectName("seriesSettingsFieldRow")
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(12)
        label = QtWidgets.QLabel(label_text, row)
        label.setObjectName("seriesSettingsFieldLabel")
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        label.setFixedWidth(132)
        widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        row_layout.addWidget(label, 0)
        row_layout.addWidget(widget, 1)
        return row

    def _make_check_row(self, checkbox: QtWidgets.QCheckBox) -> QtWidgets.QWidget:
        row = QtWidgets.QFrame(self)
        row.setObjectName("seriesSettingsCheckRow")
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(8, 2, 8, 2)
        row_layout.setSpacing(0)
        checkbox.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        row_layout.addWidget(checkbox, 1)
        return row

    def _make_section(
        self,
        title: str,
        note: str,
        child_layout: QtWidgets.QLayout,
    ) -> QtWidgets.QFrame:
        section = QtWidgets.QFrame(self)
        section.setObjectName("seriesSettingsSection")
        section.setProperty("section_title", title)
        layout = QtWidgets.QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QtWidgets.QFrame(section)
        header.setObjectName("seriesSettingsSectionHeader")
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(14, 10, 14, 6)
        header_layout.setSpacing(10)
        accent_bar = QtWidgets.QFrame(header)
        accent_bar.setObjectName("seriesSettingsSectionAccent")
        accent_bar.setFixedSize(4, 18)
        title_label = QtWidgets.QLabel(title, header)
        title_label.setObjectName("seriesSettingsSectionTitle")
        header_layout.addWidget(accent_bar, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        header_layout.addWidget(title_label, 1)
        layout.addWidget(header)

        if note:
            note_label = QtWidgets.QLabel(note, section)
            note_label.setObjectName("seriesSettingsSectionNote")
            note_label.setWordWrap(True)
            layout.addWidget(note_label)

        body = QtWidgets.QWidget(section)
        body.setObjectName("seriesSettingsSectionBody")
        body_layout = QtWidgets.QVBoxLayout(body)
        body_layout.setContentsMargins(14, 8, 14, 14)
        body_layout.setSpacing(0)
        body_layout.addLayout(child_layout)
        layout.addWidget(body)
        return section

    def _configure_form(self, form: QtWidgets.QFormLayout) -> None:
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

    def _apply_style(self) -> None:
        accent = dayu_theme.primary_color or dayu_theme.yellow or "#fadb14"
        text = dayu_theme.primary_text_color or "#d9d9d9"
        sub_text = dayu_theme.secondary_text_color or "#a6a6a6"
        window = dayu_theme.background_color or "#323232"
        panel = dayu_theme.background_in_color or "#3a3a3a"
        panel_alt = dayu_theme.background_out_color or "#494949"
        border = dayu_theme.divider_color or "#262626"
        section_bg = dayu_theme.background_out_color or "#4a4a4a"
        row_bg = dayu_theme.background_in_color or "#383838"
        row_border = _hex_to_rgba(accent, 0.32)
        input_bg = dayu_theme.background_color or "#303030"
        accent_soft = _hex_to_rgba(accent, 0.16)
        hover = _hex_to_rgba(text, 0.08)
        self.setStyleSheet(
            f"""
            QDialog#seriesSettingsDialog {{
                background: {window};
                color: {text};
                font-size: 14px;
            }}
            QLabel#seriesSettingsTitle {{
                color: {text};
                font-size: 22px;
                font-weight: 700;
            }}
            QLabel#seriesSettingsSubtitle {{
                color: {sub_text};
                font-size: 14px;
                padding-bottom: 2px;
            }}
            QTabWidget#seriesSettingsTabs::pane {{
                border: none;
                background: transparent;
                top: 0;
            }}
            QTabBar::tab {{
                min-width: 104px;
                min-height: 46px;
                padding: 0 18px;
                margin-right: 6px;
                border: 1px solid {border};
                border-bottom: none;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
                background: {panel_alt};
                color: {sub_text};
                font-size: 14px;
            }}
            QTabBar::tab:selected {{
                background: {section_bg};
                color: {accent};
                font-weight: 700;
            }}
            QTabBar::tab:hover {{
                background: {hover};
                color: {text};
            }}
            QScrollArea#seriesSettingsScroll,
            QWidget#seriesSettingsTabContent {{
                background: transparent;
            }}
            QFrame#seriesSettingsSection {{
                background: {section_bg};
                border: 1px solid {border};
                border-radius: 7px;
            }}
            QFrame#seriesSettingsSectionHeader,
            QWidget#seriesSettingsSectionBody {{
                background: transparent;
                border: none;
            }}
            QFrame#seriesSettingsSectionAccent {{
                background: {accent};
                border-radius: 2px;
            }}
            QLabel#seriesSettingsSectionTitle {{
                color: {accent};
                background: transparent;
                font-size: 16px;
                font-weight: 700;
            }}
            QLabel#seriesSettingsSectionNote {{
                color: {sub_text};
                background: transparent;
                font-size: 13px;
                padding: 0 14px 4px 28px;
                border: none;
            }}
            QLabel {{
                color: {text};
            }}
            QFrame#seriesSettingsFieldRow {{
                background: {row_bg};
                border-left: 2px solid {row_border};
                border-radius: 4px;
                min-height: 39px;
            }}
            QLabel#seriesSettingsFieldLabel {{
                color: {text};
                background: transparent;
                font-size: 14px;
                font-weight: 700;
                padding: 9px 8px;
                border: none;
            }}
            QFrame#seriesSettingsCheckRow {{
                background: {row_bg};
                border-left: 2px solid {row_border};
                border-radius: 4px;
                min-height: 31px;
            }}
            QComboBox, QSpinBox {{
                min-height: 34px;
                border-radius: 6px;
                border: 1px solid {border};
                background: {input_bg};
                color: {text};
                font-size: 14px;
                padding: 0 8px;
                selection-background-color: {accent_soft};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 26px;
            }}
            QComboBox QAbstractItemView {{
                background: {panel};
                color: {text};
                border: 1px solid {border};
                selection-background-color: {accent_soft};
                selection-color: {text};
            }}
            QCheckBox {{
                color: {text};
                font-size: 14px;
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 14px;
                height: 14px;
            }}
            QPushButton {{
                min-height: 30px;
                border-radius: 6px;
                border: 1px solid {border};
                background: {panel_alt};
                color: {text};
                padding: 0 14px;
            }}
            QPushButton:hover {{
                background: {hover};
            }}
            QDialogButtonBox QPushButton {{
                min-width: 82px;
            }}
            """
        )

    def _make_color_button(self, color: str) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(self)
        button.setFixedSize(34, 28)
        self._set_button_color(button, color)
        button.clicked.connect(lambda _checked=False, b=button: self._choose_button_color(b))
        return button

    def _choose_button_color(self, button: QtWidgets.QPushButton) -> None:
        current = QtGui.QColor(str(button.property("selected_color") or "#000000"))
        color = QtWidgets.QColorDialog.getColor(current, self, self.tr("Select Color"))
        if color.isValid():
            self._set_button_color(button, color.name())

    def _set_button_color(self, button: QtWidgets.QPushButton, color: str) -> None:
        value = str(color or "#000000")
        button.setProperty("selected_color", value)
        button.setStyleSheet(f"background-color: {value}; border: 1px solid #555; border-radius: 5px;")

    def configure_options(
        self,
        *,
        languages: list[tuple[str, str]],
        ocr_modes: list[tuple[str, str]],
        translators: list[tuple[str, str]],
        workflow_modes: list[tuple[str, str]],
        fonts: list[str],
        output_options: dict[str, list[tuple[str, str]]],
    ) -> None:
        self._populate_combo(self.source_lang_combo, languages)
        self._populate_combo(self.target_lang_combo, languages)
        self._populate_combo(self.ocr_combo, ocr_modes)
        self._populate_combo(self.translator_combo, translators)
        self._populate_combo(self.workflow_combo, workflow_modes)
        self.font_combo.clear()
        for font in fonts:
            if font:
                self.font_combo.addItem(str(font), str(font))
        self._output_options = dict(output_options or {})
        self._populate_combo(self.output_target_combo, self._output_options.get("automatic_output_target", []))
        self._populate_combo(self.output_image_format_combo, self._output_options.get("automatic_output_image_format", []))
        self._populate_combo(self.output_archive_format_combo, self._output_options.get("automatic_output_archive_format", []))
        self._populate_combo(self.output_archive_image_format_combo, self._output_options.get("automatic_output_archive_image_format", []))

    def set_payload(self, series_settings: dict[str, object], global_settings: dict[str, object]) -> None:
        self.queue_page.set_settings(series_settings)
        self._set_combo_value(self.source_lang_combo, str(global_settings.get("source_language") or ""))
        self._set_combo_value(self.target_lang_combo, str(global_settings.get("target_language") or ""))
        self._set_combo_value(self.ocr_combo, str(global_settings.get("ocr") or ""))
        self._set_combo_value(self.translator_combo, str(global_settings.get("translator") or ""))
        self._set_combo_value(self.workflow_combo, str(global_settings.get("workflow_mode") or ""))
        self.use_gpu_checkbox.setChecked(bool(global_settings.get("use_gpu", True)))
        self._set_render_values(global_settings.get("render_settings") if isinstance(global_settings.get("render_settings"), dict) else {})
        self._set_export_values(global_settings.get("export_settings") if isinstance(global_settings.get("export_settings"), dict) else {})

    def payload(self) -> tuple[dict[str, object], dict[str, object]]:
        return self.queue_page.get_settings(), {
            "source_language": str(self.source_lang_combo.currentData() or ""),
            "target_language": str(self.target_lang_combo.currentData() or ""),
            "ocr": str(self.ocr_combo.currentData() or ""),
            "translator": str(self.translator_combo.currentData() or ""),
            "workflow_mode": str(self.workflow_combo.currentData() or ""),
            "use_gpu": self.use_gpu_checkbox.isChecked(),
            "render_settings": self._render_values(),
            "export_settings": self._export_values(),
        }

    def _set_render_values(self, values: dict[str, object]) -> None:
        font = str(values.get("font_family") or "")
        if font and self.font_combo.findText(font) < 0:
            self.font_combo.addItem(font, font)
        if font:
            self.font_combo.setCurrentText(font)
        self.min_font_spin.setValue(max(1, int(values.get("min_font_size", 5) or 5)))
        self.max_font_spin.setValue(max(1, int(values.get("max_font_size", 40) or 40)))
        self.line_spacing_combo.setCurrentText(str(values.get("line_spacing") or "1.0"))
        self._set_button_color(self.text_color_button, str(values.get("color") or "#000000"))
        self.force_color_checkbox.setChecked(bool(values.get("force_font_color", False)))
        self._set_combo_value(self.horizontal_alignment_combo, int(values.get("alignment_id", 1) or 1))
        self._set_combo_value(self.vertical_alignment_combo, int(values.get("vertical_alignment_id", 0) or 0))
        self.bold_checkbox.setChecked(bool(values.get("bold", False)))
        self.italic_checkbox.setChecked(bool(values.get("italic", False)))
        self.underline_checkbox.setChecked(bool(values.get("underline", False)))
        self.uppercase_checkbox.setChecked(bool(values.get("upper_case", False)))
        self.outline_checkbox.setChecked(bool(values.get("outline", True)))
        self._set_button_color(self.outline_color_button, str(values.get("outline_color") or "#ffffff"))
        self.outline_width_combo.setCurrentText(str(values.get("outline_width") or "1.0"))

    def _render_values(self) -> dict[str, object]:
        return {
            "alignment_id": int(self.horizontal_alignment_combo.currentData() or 0),
            "vertical_alignment_id": int(self.vertical_alignment_combo.currentData() or 0),
            "font_family": str(self.font_combo.currentText() or ""),
            "min_font_size": int(self.min_font_spin.value()),
            "max_font_size": int(self.max_font_spin.value()),
            "color": str(self.text_color_button.property("selected_color") or "#000000"),
            "force_font_color": self.force_color_checkbox.isChecked(),
            "upper_case": self.uppercase_checkbox.isChecked(),
            "outline": self.outline_checkbox.isChecked(),
            "outline_color": str(self.outline_color_button.property("selected_color") or "#ffffff"),
            "outline_width": str(self.outline_width_combo.currentText() or "1.0"),
            "bold": self.bold_checkbox.isChecked(),
            "italic": self.italic_checkbox.isChecked(),
            "underline": self.underline_checkbox.isChecked(),
            "line_spacing": str(self.line_spacing_combo.currentText() or "1.0"),
        }

    def _set_export_values(self, values: dict[str, object]) -> None:
        self.export_raw_text_checkbox.setChecked(bool(values.get("export_raw_text", False)))
        self.export_translated_text_checkbox.setChecked(bool(values.get("export_translated_text", False)))
        self.export_inpainted_image_checkbox.setChecked(bool(values.get("export_inpainted_image", False)))
        self.export_detector_overlay_checkbox.setChecked(bool(values.get("export_detector_overlay", False)))
        self.export_raw_mask_checkbox.setChecked(bool(values.get("export_raw_mask", False)))
        self.export_mask_overlay_checkbox.setChecked(bool(values.get("export_mask_overlay", False)))
        self.export_cleanup_mask_delta_checkbox.setChecked(bool(values.get("export_cleanup_mask_delta", False)))
        self.export_debug_metadata_checkbox.setChecked(bool(values.get("export_debug_metadata", False)))
        self._set_combo_value(self.output_target_combo, str(values.get("automatic_output_target") or ""))
        self._set_combo_value(self.output_image_format_combo, str(values.get("automatic_output_image_format") or ""))
        self._set_combo_value(self.output_archive_format_combo, str(values.get("automatic_output_archive_format") or ""))
        self._set_combo_value(self.output_archive_image_format_combo, str(values.get("automatic_output_archive_image_format") or ""))
        self.output_archive_level_spin.setValue(max(0, min(9, int(values.get("automatic_output_archive_compression_level", 6) or 6))))

    def _export_values(self) -> dict[str, object]:
        return {
            "export_raw_text": self.export_raw_text_checkbox.isChecked(),
            "export_translated_text": self.export_translated_text_checkbox.isChecked(),
            "export_inpainted_image": self.export_inpainted_image_checkbox.isChecked(),
            "export_detector_overlay": self.export_detector_overlay_checkbox.isChecked(),
            "export_raw_mask": self.export_raw_mask_checkbox.isChecked(),
            "export_mask_overlay": self.export_mask_overlay_checkbox.isChecked(),
            "export_cleanup_mask_delta": self.export_cleanup_mask_delta_checkbox.isChecked(),
            "export_debug_metadata": self.export_debug_metadata_checkbox.isChecked(),
            "automatic_output_target": str(self.output_target_combo.currentData() or ""),
            "automatic_output_image_format": str(self.output_image_format_combo.currentData() or ""),
            "automatic_output_archive_format": str(self.output_archive_format_combo.currentData() or ""),
            "automatic_output_archive_image_format": str(self.output_archive_image_format_combo.currentData() or ""),
            "automatic_output_archive_compression_level": int(self.output_archive_level_spin.value()),
        }

    def _populate_combo(self, combo: QtWidgets.QComboBox, options: list[tuple[str, str]]) -> None:
        combo.clear()
        for value, label in options or []:
            combo.addItem(str(label), value)

    def _set_combo_value(self, combo: QtWidgets.QComboBox, value: object) -> None:
        index = combo.findData(value)
        if index < 0 and isinstance(value, str):
            index = combo.findText(value)
        if index < 0 and combo.count() > 0:
            index = 0
        if index >= 0:
            combo.setCurrentIndex(index)


class _SeriesStatusPanel(QtWidgets.QWidget):
    pause_requested = QtCore.Signal()
    resume_requested = QtCore.Signal()
    open_failed_requested = QtCore.Signal()
    open_current_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header = QtWidgets.QLabel(self.tr("Queue Status"))
        header.setObjectName("seriesStatusHeader")
        note = QtWidgets.QLabel(
            self.tr(
                "Monitor the current queue execution and control safe pause/resume behavior here."
            )
        )
        note.setWordWrap(True)
        note.setObjectName("seriesStatusNote")

        self.state_value = QtWidgets.QLabel("—")
        self.current_item_value = QtWidgets.QLabel("—")
        self.current_item_value.setWordWrap(True)
        self.next_item_value = QtWidgets.QLabel("—")
        self.next_item_value.setWordWrap(True)
        self.failed_item_value = QtWidgets.QLabel("—")
        self.failed_item_value.setWordWrap(True)
        self.retry_remaining_value = QtWidgets.QLabel("—")
        self.last_run_value = QtWidgets.QLabel("—")
        self.last_run_value.setWordWrap(True)

        form = QtWidgets.QFormLayout()
        form.addRow(self.tr("State:"), self.state_value)
        form.addRow(self.tr("Current item:"), self.current_item_value)
        form.addRow(self.tr("Next item:"), self.next_item_value)
        form.addRow(self.tr("Last failed item:"), self.failed_item_value)
        form.addRow(self.tr("Retries left:"), self.retry_remaining_value)
        form.addRow(self.tr("Last run time:"), self.last_run_value)

        button_row = QtWidgets.QHBoxLayout()
        self.pause_button = QtWidgets.QPushButton(self.tr("Pause"))
        self.resume_button = QtWidgets.QPushButton(self.tr("Resume"))
        self.open_current_button = QtWidgets.QPushButton(self.tr("Open Current Item"))
        self.open_failed_button = QtWidgets.QPushButton(self.tr("Open Failed Item"))
        button_row.addWidget(self.pause_button)
        button_row.addWidget(self.resume_button)
        button_row.addWidget(self.open_current_button)
        button_row.addWidget(self.open_failed_button)

        layout.addWidget(header)
        layout.addWidget(note)
        layout.addLayout(form)
        layout.addLayout(button_row)

        self.pause_button.clicked.connect(self.pause_requested.emit)
        self.resume_button.clicked.connect(self.resume_requested.emit)
        self.open_current_button.clicked.connect(self.open_current_requested.emit)
        self.open_failed_button.clicked.connect(self.open_failed_requested.emit)

    def _state_label(self, queue_state: str) -> str:
        mapping = {
            "idle": self.tr("Idle"),
            "running": self.tr("Running"),
            "paused": self.tr("Paused"),
        }
        return mapping.get(str(queue_state or "").strip().lower(), self.tr("Idle"))

    def _item_label(self, item_map: dict[str, dict[str, object]], item_id: str | None) -> str:
        if not item_id:
            return "—"
        item = item_map.get(str(item_id))
        if item is None:
            return "—"
        queue_index = int(item.get("queue_index", 0) or 0)
        display_name = str(item.get("display_name") or item_id)
        return self.tr("#{index:02d} · {name}").format(index=queue_index, name=display_name)

    def set_runtime(self, queue_runtime: dict[str, object], items: list[dict[str, object]]) -> None:
        runtime = dict(queue_runtime or {})
        item_map = {
            str(item.get("series_item_id") or ""): item
            for item in items
            if str(item.get("series_item_id") or "").strip()
        }
        queue_state = str(runtime.get("queue_state") or "idle").strip().lower()
        pause_requested = bool(runtime.get("pause_requested", False))
        active_item_id = str(runtime.get("active_item_id") or "").strip() or None
        failed_item_id = str(runtime.get("failed_item_id") or "").strip() or None
        pending_ids = [
            str(item_id)
            for item_id in list(runtime.get("pending_item_ids") or [])
            if str(item_id or "").strip()
        ]
        retry_remaining_by_item = dict(runtime.get("retry_remaining_by_item") or {})
        last_run_finished_at = str(runtime.get("last_run_finished_at") or "").strip() or None
        last_run_started_at = str(runtime.get("last_run_started_at") or "").strip() or None

        self.state_value.setText(
            self.tr("{state} (pause requested)").format(state=self._state_label(queue_state))
            if queue_state == "running" and pause_requested
            else self._state_label(queue_state)
        )
        self.current_item_value.setText(self._item_label(item_map, active_item_id))
        next_item_id = pending_ids[0] if pending_ids else None
        self.next_item_value.setText(self._item_label(item_map, next_item_id))
        self.failed_item_value.setText(self._item_label(item_map, failed_item_id))
        retry_target = active_item_id or failed_item_id or ""
        retries_left = retry_remaining_by_item.get(retry_target, 0)
        self.retry_remaining_value.setText(str(int(retries_left or 0)))
        self.last_run_value.setText(last_run_finished_at or last_run_started_at or "—")

        self.pause_button.setVisible(queue_state == "running")
        self.pause_button.setEnabled(queue_state == "running" and not pause_requested)
        self.pause_button.setText(self.tr("Pause") if not pause_requested else self.tr("Pause Requested"))
        self.resume_button.setVisible(queue_state == "paused")
        self.resume_button.setEnabled(queue_state == "paused" and bool(pending_ids))
        self.open_current_button.setVisible(queue_state == "running")
        self.open_current_button.setEnabled(bool(active_item_id))
        self.open_failed_button.setEnabled(bool(failed_item_id) and queue_state != "running")


class _SeriesRunSummaryPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header = QtWidgets.QLabel(self.tr("Last Queue Run"))
        header.setObjectName("seriesSummaryHeader")

        self.done_value = QtWidgets.QLabel("0")
        self.failed_value = QtWidgets.QLabel("0")
        self.skipped_value = QtWidgets.QLabel("0")
        self.duration_value = QtWidgets.QLabel("—")
        self.window_value = QtWidgets.QLabel("—")
        self.window_value.setWordWrap(True)

        form = QtWidgets.QFormLayout()
        form.addRow(self.tr("Done:"), self.done_value)
        form.addRow(self.tr("Failed:"), self.failed_value)
        form.addRow(self.tr("Skipped:"), self.skipped_value)
        form.addRow(self.tr("Total time:"), self.duration_value)
        form.addRow(self.tr("Started / Finished:"), self.window_value)

        layout.addWidget(header)
        layout.addLayout(form)

    def set_summary(self, summary: dict[str, object] | None) -> None:
        payload = dict(summary or {})
        done_count = int(payload.get("done_count", 0) or 0)
        failed_count = int(payload.get("failed_count", 0) or 0)
        skipped_count = int(payload.get("skipped_count", 0) or 0)
        duration = payload.get("duration_sec")
        started_at = str(payload.get("started_at") or "").strip() or None
        finished_at = str(payload.get("finished_at") or "").strip() or None

        self.done_value.setText(str(done_count))
        self.failed_value.setText(str(failed_count))
        self.skipped_value.setText(str(skipped_count))
        self.duration_value.setText(
            self.tr("{seconds} sec").format(seconds=int(duration))
            if duration not in (None, "")
            else "—"
        )
        if started_at or finished_at:
            self.window_value.setText(
                self.tr("{started} → {finished}").format(
                    started=started_at or "—",
                    finished=finished_at or "—",
                )
            )
        else:
            self.window_value.setText("—")


class SeriesTreeJumpDialog(QtWidgets.QDialog):
    def __init__(self, items: list[dict[str, object]], parent=None):
        super().__init__(parent)
        self._selected_item_id = ""
        self.setWindowTitle(self.tr("Tree Jump"))
        self.resize(520, 620)
        self.setMinimumSize(460, 500)
        self.setSizeGripEnabled(True)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowMaximizeButtonHint, True)

        accent = dayu_theme.primary_color or dayu_theme.yellow or "#fadb14"
        text = dayu_theme.primary_text_color or "#d9d9d9"
        sub_text = dayu_theme.secondary_text_color or "#a6a6a6"
        window = dayu_theme.background_color or "#323232"
        panel = dayu_theme.background_in_color or "#3a3a3a"
        header = dayu_theme.background_out_color or "#494949"
        border = dayu_theme.divider_color or "#262626"
        accent_soft = _hex_to_rgba(accent, 0.18)
        button_hover = _hex_to_rgba(text, 0.08)
        button_fill = _hex_to_rgba(text, 0.04)
        self.setStyleSheet(
            f"""
            QDialog {{
                background: {window};
                color: {text};
            }}
            QLabel {{
                background: transparent;
                color: {sub_text};
            }}
            QTreeWidget {{
                background: {panel};
                color: {text};
                border: 1px solid {border};
                border-radius: 12px;
                alternate-background-color: {header};
                selection-background-color: {accent_soft};
                selection-color: {text};
            }}
            QTreeWidget::item {{
                height: 28px;
            }}
            QHeaderView::section {{
                background: {header};
                color: {text};
                border: none;
                border-bottom: 1px solid {border};
                padding: 8px 10px;
                font-weight: 600;
            }}
            QPushButton {{
                min-height: 34px;
                padding: 0 16px;
                border-radius: 10px;
                border: 1px solid {border};
                background: {button_fill};
                color: {text};
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: {button_hover};
            }}
            """
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        note = QtWidgets.QLabel(
            self.tr("Select a series item from the original folder structure or choose the board view.")
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.tree = QtWidgets.QTreeWidget(self)
        self.tree.setHeaderLabels([self.tr("Location")])
        layout.addWidget(self.tree, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        board_item = QtWidgets.QTreeWidgetItem([self.tr("Series Board")])
        board_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, "__board__")
        self.tree.addTopLevelItem(board_item)

        nodes = {(): board_item}
        for item in items:
            rel = str(item.get("source_origin_relpath") or item.get("display_name") or "")
            item_id = str(item.get("series_item_id") or "")
            parts = [part for part in rel.replace("\\", "/").split("/") if part]
            parent_item = board_item
            prefix = []
            for part in parts[:-1]:
                prefix.append(part)
                key = tuple(prefix)
                node = nodes.get(key)
                if node is None:
                    node = QtWidgets.QTreeWidgetItem([part])
                    nodes[key] = node
                    parent_item.addChild(node)
                parent_item = node
            leaf_label = parts[-1] if parts else str(item.get("display_name") or item_id)
            leaf = QtWidgets.QTreeWidgetItem([leaf_label])
            leaf.setData(0, QtCore.Qt.ItemDataRole.UserRole, item_id)
            parent_item.addChild(leaf)

        self.tree.expandToDepth(1)
        self.tree.itemDoubleClicked.connect(lambda *_: self.accept())

    def selected_target(self) -> str:
        current = self.tree.currentItem()
        if current is None:
            return ""
        return str(current.data(0, QtCore.Qt.ItemDataRole.UserRole) or "")


class SeriesWorkspace(QtWidgets.QWidget):
    open_item_requested = QtCore.Signal(str)
    remove_item_requested = QtCore.Signal(str)
    reorder_requested = QtCore.Signal(list)
    queue_index_requested = QtCore.Signal(str, int)
    add_files_requested = QtCore.Signal()
    add_folder_requested = QtCore.Signal()
    back_requested = QtCore.Signal()
    forward_requested = QtCore.Signal()
    tree_jump_requested = QtCore.Signal()
    auto_translate_requested = QtCore.Signal()
    pause_requested = QtCore.Signal()
    resume_requested = QtCore.Signal()
    open_failed_item_requested = QtCore.Signal()
    open_current_item_requested = QtCore.Signal()
    open_series_settings_requested = QtCore.Signal()
    global_settings_changed = QtCore.Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue_running = False
        self._can_back = False
        self._can_forward = False
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        top_row = QtWidgets.QHBoxLayout()
        self.back_button = QtWidgets.QToolButton()
        self.back_button.setText("←")
        self.back_button.setToolTip(self.tr("Back"))
        self.forward_button = QtWidgets.QToolButton()
        self.forward_button.setText("→")
        self.forward_button.setToolTip(self.tr("Forward"))
        self.tree_button = QtWidgets.QToolButton()
        self.tree_button.setText(self.tr("Tree"))
        self.tree_button.setToolTip(self.tr("Tree Jump"))
        self.scope_badge = QtWidgets.QLabel(self.tr("Series Project"))
        self.scope_badge.setObjectName("seriesScopeBadge")
        self.title_label = QtWidgets.QLabel("")
        self.title_label.setObjectName("seriesTitleLabel")
        self.title_label.setWordWrap(True)
        top_row.addWidget(self.back_button)
        top_row.addWidget(self.forward_button)
        top_row.addWidget(self.tree_button)
        top_row.addSpacing(10)
        top_row.addWidget(self.scope_badge, 0)
        top_row.addWidget(self.title_label, 1)
        layout.addLayout(top_row)

        badge_row = QtWidgets.QHBoxLayout()
        self.recovery_badge = QtWidgets.QLabel(self.tr("Recovered Snapshot"))
        self.recovery_badge.setObjectName("seriesRecoveryBadge")
        self.recovery_badge.hide()
        self.unsynced_badge = QtWidgets.QLabel(self.tr("Child Changes Not Synced"))
        self.unsynced_badge.setObjectName("seriesUnsyncedBadge")
        self.unsynced_badge.hide()
        badge_row.addWidget(self.recovery_badge, 0)
        badge_row.addWidget(self.unsynced_badge, 0)
        badge_row.addStretch(1)
        layout.addLayout(badge_row)

        self.body_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        self.body_splitter.setChildrenCollapsible(False)
        self.body_splitter.setHandleWidth(10)
        layout.addWidget(self.body_splitter, 1)

        left_panel = QtWidgets.QWidget(self)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        action_row = QtWidgets.QHBoxLayout()
        self.open_button = QtWidgets.QPushButton(self.tr("Open Selected"))
        self.add_files_button = QtWidgets.QPushButton(self.tr("Add Files"))
        self.add_folder_button = QtWidgets.QPushButton(self.tr("Add Folder"))
        action_row.addWidget(self.open_button)
        action_row.addWidget(self.add_files_button)
        action_row.addWidget(self.add_folder_button)
        action_row.addStretch(1)
        self.sort_label = QtWidgets.QLabel(self.tr("Sort"))
        self.sort_label.setObjectName("seriesQuickNote")
        self.sort_combo = QtWidgets.QComboBox(self)
        self.sort_combo.addItem(self.tr("Manual Queue"), "manual")
        self.sort_combo.addItem(self.tr("Name (A-Z)"), "name_asc")
        self.sort_combo.addItem(self.tr("Name (Z-A)"), "name_desc")
        self.sort_combo.addItem(self.tr("Date (Newest First)"), "date_desc")
        self.sort_combo.addItem(self.tr("Date (Oldest First)"), "date_asc")
        self.sort_combo.setMinimumWidth(180)
        action_row.addWidget(self.sort_label)
        action_row.addWidget(self.sort_combo)
        left_layout.addLayout(action_row)

        self.queue_notice = QtWidgets.QLabel("")
        self.queue_notice.setObjectName("seriesQueueNotice")
        self.queue_notice.setWordWrap(True)
        self.queue_notice.hide()
        left_layout.addWidget(self.queue_notice)

        self.queue_table = _SeriesQueueTable(self)
        left_layout.addWidget(self.queue_table, 1)
        self.body_splitter.addWidget(left_panel)

        self.quick_settings = _SeriesQuickSettings(self)
        self.status_panel = _SeriesStatusPanel(self)
        self.summary_panel = _SeriesRunSummaryPanel(self)
        quick_frame = QtWidgets.QFrame(self)
        quick_frame.setObjectName("seriesQuickFrame")
        quick_frame_layout = QtWidgets.QVBoxLayout(quick_frame)
        quick_frame_layout.setContentsMargins(12, 12, 12, 12)
        quick_frame_layout.addWidget(self.status_panel)
        quick_frame_layout.addWidget(self.summary_panel)
        quick_frame_layout.addWidget(self.quick_settings)
        quick_frame.setMinimumWidth(320)
        self.body_splitter.addWidget(quick_frame)
        self.body_splitter.setStretchFactor(0, 1)
        self.body_splitter.setStretchFactor(1, 0)
        self.body_splitter.setSizes([860, 360])

        self._hover_preview_popup = _SeriesItemPreviewPopup(self)
        self._hover_preview_timer = QtCore.QTimer(self)
        self._hover_preview_timer.setSingleShot(True)
        self._hover_preview_timer.setInterval(140)
        self._pending_preview_payload: dict[str, object] = {}
        self._pending_preview_pos = QtCore.QPoint()

        self.back_button.clicked.connect(self.back_requested)
        self.forward_button.clicked.connect(self.forward_requested)
        self.tree_button.clicked.connect(self.tree_jump_requested)
        self.open_button.clicked.connect(self._emit_open_selected)
        self.add_files_button.clicked.connect(self.add_files_requested)
        self.add_folder_button.clicked.connect(self.add_folder_requested)
        self.queue_table.open_requested.connect(self.open_item_requested)
        self.queue_table.remove_requested.connect(self.remove_item_requested)
        self.queue_table.order_changed.connect(self.reorder_requested)
        self.queue_table.queue_index_requested.connect(self.queue_index_requested)
        self.queue_table.hover_preview_requested.connect(self._queue_hover_requested)
        self.queue_table.hover_preview_hidden.connect(self._hide_hover_preview)
        self.quick_settings.auto_translate_requested.connect(self.auto_translate_requested)
        self.status_panel.pause_requested.connect(self.pause_requested)
        self.status_panel.resume_requested.connect(self.resume_requested)
        self.status_panel.open_failed_requested.connect(self.open_failed_item_requested)
        self.status_panel.open_current_requested.connect(self.open_current_item_requested)
        self.quick_settings.open_series_settings_requested.connect(self.open_series_settings_requested)
        self.quick_settings.changed.connect(self._emit_global_settings_changed)
        self.sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        self._hover_preview_timer.timeout.connect(self._show_pending_hover_preview)
        self._apply_theme_styles()

    def _apply_theme_styles(self) -> None:
        accent = dayu_theme.primary_color or dayu_theme.yellow or "#fadb14"
        text = dayu_theme.primary_text_color or "#d9d9d9"
        sub_text = dayu_theme.secondary_text_color or "#a6a6a6"
        window = dayu_theme.background_color or "#323232"
        panel = dayu_theme.background_in_color or "#3a3a3a"
        header = dayu_theme.background_out_color or "#494949"
        border = dayu_theme.divider_color or "#262626"
        accent_soft = _hex_to_rgba(accent, 0.16)
        button_fill = _hex_to_rgba(text, 0.04)
        button_hover = _hex_to_rgba(text, 0.08)
        badge_fill = _hex_to_rgba(accent, 0.14)
        badge_border = _hex_to_rgba(accent, 0.34)
        warning_fill = _hex_to_rgba("#f5a623", 0.14)
        warning_border = _hex_to_rgba("#f5a623", 0.30)
        recovery_fill = _hex_to_rgba("#6fb1fc", 0.16)
        recovery_border = _hex_to_rgba("#6fb1fc", 0.34)
        self.setStyleSheet(
            f"""
            QWidget {{
                color: {text};
            }}
            QLabel {{
                background: transparent;
                border: none;
            }}
            QToolButton, QPushButton {{
                min-height: 34px;
                padding: 0 14px;
                border-radius: 10px;
                border: 1px solid {border};
                background: {button_fill};
                color: {text};
                font-weight: 600;
            }}
            QToolButton:hover, QPushButton:hover {{
                background: {button_hover};
            }}
            QComboBox {{
                min-height: 32px;
                border-radius: 10px;
                border: 1px solid {border};
                background: {window};
                color: {text};
                padding: 0 10px;
                selection-background-color: {accent_soft};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 28px;
                background: transparent;
            }}
            QComboBox QAbstractItemView {{
                background: {panel};
                color: {text};
                border: 1px solid {border};
                selection-background-color: {accent_soft};
                selection-color: {text};
            }}
            QCheckBox {{
                color: {text};
            }}
            QLabel#seriesScopeBadge {{
                padding: 5px 12px;
                border-radius: 999px;
                background: {badge_fill};
                border: 1px solid {badge_border};
                color: {text};
                font-weight: 700;
            }}
            QLabel#seriesTitleLabel {{
                font-size: 18px;
                font-weight: 700;
                color: {text};
            }}
            QLabel#seriesQueueNotice {{
                padding: 10px 12px;
                border-radius: 10px;
                background: {warning_fill};
                border: 1px solid {warning_border};
                color: {text};
            }}
            QLabel#seriesRecoveryBadge {{
                padding: 5px 10px;
                border-radius: 999px;
                background: {recovery_fill};
                border: 1px solid {recovery_border};
                color: {text};
                font-weight: 600;
            }}
            QLabel#seriesUnsyncedBadge {{
                padding: 5px 10px;
                border-radius: 999px;
                background: {warning_fill};
                border: 1px solid {warning_border};
                color: {text};
                font-weight: 600;
            }}
            QFrame#seriesQuickFrame {{
                background: {panel};
                border: 1px solid {border};
                border-radius: 14px;
            }}
            QLabel#seriesQuickHeader,
            QLabel#seriesStatusHeader,
            QLabel#seriesSummaryHeader {{
                font-size: 14px;
                font-weight: 700;
                color: {text};
            }}
            QLabel#seriesQuickNote,
            QLabel#seriesStatusNote {{
                color: {sub_text};
            }}
            QLabel#seriesPreviewHelper {{
                color: {sub_text};
            }}
            QPushButton#seriesAutoTranslateButton {{
                background: {accent};
                border: 1px solid {accent};
                color: #111111;
            }}
            QPushButton#seriesAutoTranslateButton:hover {{
                background: {dayu_theme.primary_4 or accent};
                border: 1px solid {dayu_theme.primary_4 or accent};
            }}
            QSplitter::handle {{
                background: transparent;
            }}
            QSplitter::handle:horizontal {{
                width: 10px;
                margin: 6px 0;
                image: none;
                border-left: 1px solid {border};
                border-right: 1px solid {border};
            }}
            """
        )
        self.queue_table.apply_theme_styles()

    def _set_sort_mode(self, mode: str) -> None:
        index = self.sort_combo.findData(mode)
        if index < 0:
            index = 0
        blocker = QtCore.QSignalBlocker(self.sort_combo)
        self.sort_combo.setCurrentIndex(index)
        del blocker

    def configure_options(self, *, languages, ocr_modes, translators, workflow_modes) -> None:
        self.quick_settings.set_options(
            languages=languages,
            ocr_modes=ocr_modes,
            translators=translators,
            workflow_modes=workflow_modes,
        )

    def set_global_settings(self, values: dict[str, object]) -> None:
        self.quick_settings.set_values(values)

    def global_settings(self) -> dict[str, object]:
        return self.quick_settings.values()

    def set_navigation_state(self, *, can_back: bool, can_forward: bool) -> None:
        self._can_back = bool(can_back)
        self._can_forward = bool(can_forward)
        self._apply_navigation_state()

    def _queue_lock_reason(self) -> str:
        return self.tr(
            "Queue changes are locked while automatic translation is running.\n"
            "The current running item stays fixed, and you can change the queue after the run finishes."
        )

    def _apply_navigation_state(self) -> None:
        self.back_button.setEnabled(self._can_back and not self._queue_running)
        self.forward_button.setEnabled(self._can_forward and not self._queue_running)
        self.tree_button.setEnabled(not self._queue_running)
        if self._queue_running:
            self.back_button.setToolTip(self._queue_lock_reason())
            self.forward_button.setToolTip(self._queue_lock_reason())
            self.tree_button.setToolTip(self._queue_lock_reason())
        else:
            self.back_button.setToolTip(self.tr("Back"))
            self.forward_button.setToolTip(self.tr("Forward"))
            self.tree_button.setToolTip(self.tr("Tree Jump"))

    def set_series_state(
        self,
        *,
        series_file: str,
        items: list[dict[str, object]],
        queue_running: bool = False,
        active_item_id: str | None = None,
        queue_runtime: dict[str, object] | None = None,
        child_unsynced_dirty: bool = False,
        recovery_loaded: bool = False,
    ) -> None:
        self._queue_running = bool(queue_running)
        queue_state = str((queue_runtime or {}).get("queue_state") or "idle").strip().lower()
        self.title_label.setText(series_file)
        self.scope_badge.setText(self.tr("Series Project"))
        lock_reason = self._queue_lock_reason()
        self._hide_hover_preview()
        self.queue_table.set_interaction_state(
            queue_running=queue_running,
            active_item_id=active_item_id,
            lock_reason=lock_reason,
        )
        self.queue_table.set_series_items(items)
        self.status_panel.set_runtime(queue_runtime or {}, items)
        self.summary_panel.set_summary(
            dict((queue_runtime or {}).get("last_run_summary") or {})
        )
        self.recovery_badge.setVisible(bool(recovery_loaded))
        self.unsynced_badge.setVisible(bool(child_unsynced_dirty))
        controls_locked = bool(queue_running)
        self.open_button.setEnabled(bool(items) and not controls_locked)
        self.add_files_button.setEnabled(not controls_locked)
        self.add_folder_button.setEnabled(not controls_locked)
        self.sort_combo.setEnabled(not controls_locked and len(items) > 1)
        self.sort_combo.setToolTip(lock_reason if controls_locked else self.tr("Apply a quick queue sort by name or modified date."))
        self.open_button.setToolTip(lock_reason if controls_locked else self.tr("Open the selected child project."))
        self.add_files_button.setToolTip(lock_reason if controls_locked else self.tr("Add supported files to this series."))
        self.add_folder_button.setToolTip(lock_reason if controls_locked else self.tr("Scan and add a folder to this series."))
        self.quick_settings.set_locked(controls_locked, lock_reason)
        self.quick_settings.auto_translate_button.setEnabled(
            bool(items) and not queue_running and queue_state != "paused"
        )
        self.quick_settings.auto_translate_button.setToolTip(
            self.tr("Run automatic translation in queue order.")
            if not queue_running and queue_state != "paused"
            else self.tr("Resume the paused queue from the queue status panel.")
            if queue_state == "paused"
            else self.tr("Automatic translation is already running.")
        )
        self.queue_notice.setVisible(controls_locked)
        self.queue_notice.setText(
            self.tr(
                "Queue changes are locked while automatic translation is running.\n"
                "The current running item stays fixed, and you can change the queue after the run finishes."
            )
            if controls_locked
            else ""
        )
        self._apply_navigation_state()

    def prompt_tree_jump(self, items: list[dict[str, object]]) -> str:
        dialog = SeriesTreeJumpDialog(items, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return ""
        return dialog.selected_target()

    def _emit_open_selected(self) -> None:
        item_id = self.queue_table.selected_item_id()
        if item_id:
            self.open_item_requested.emit(item_id)

    def _on_sort_changed(self) -> None:
        mode = str(self.sort_combo.currentData() or "manual")
        if mode == "manual":
            return
        ordered_ids = self.queue_table.sorted_item_ids(mode)
        if ordered_ids:
            self.reorder_requested.emit(ordered_ids)
        self._set_sort_mode("manual")

    def _queue_hover_requested(self, payload: dict[str, object], global_pos: QtCore.QPoint) -> None:
        self._pending_preview_payload = dict(payload or {})
        self._pending_preview_pos = QtCore.QPoint(global_pos)
        if self._hover_preview_popup.isVisible():
            self._hover_preview_popup.show_preview(self._pending_preview_payload, self._pending_preview_pos)
            return
        self._hover_preview_timer.start()

    def _show_pending_hover_preview(self) -> None:
        if not self._pending_preview_payload:
            return
        self._hover_preview_popup.show_preview(self._pending_preview_payload, self._pending_preview_pos)

    def _hide_hover_preview(self) -> None:
        self._hover_preview_timer.stop()
        self._pending_preview_payload = {}
        self._hover_preview_popup.hide_preview()

    def hideEvent(self, event):  # type: ignore[override]
        self._hide_hover_preview()
        super().hideEvent(event)

    def closeEvent(self, event):  # type: ignore[override]
        self._hide_hover_preview()
        try:
            self.queue_table.hover_preview_requested.disconnect(self._queue_hover_requested)
        except Exception:
            pass
        try:
            self.queue_table.hover_preview_hidden.disconnect(self._hide_hover_preview)
        except Exception:
            pass
        try:
            self._hover_preview_popup.shutdown()
            self._hover_preview_popup.deleteLater()
        except Exception:
            pass
        super().closeEvent(event)

    def _emit_global_settings_changed(self) -> None:
        self.global_settings_changed.emit(self.quick_settings.values())
