from __future__ import annotations

import json
import os
from dataclasses import dataclass

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

from .dayu_widgets.browser import MClickBrowserFilePushButton
from .dayu_widgets.menu import MMenu


@dataclass(slots=True)
class PageListItemData:
    file_path: str
    file_name: str
    display_name: str
    skipped: bool = False
    thumbnail: QtGui.QPixmap | None = None
    modified_at: float = 0.0
    source_path: str = ""
    source_kind: str = "file"


class PageListModel(QtCore.QAbstractListModel):
    FilePathRole = Qt.ItemDataRole.UserRole
    FileNameRole = Qt.ItemDataRole.UserRole + 1
    ThumbnailRole = Qt.ItemDataRole.UserRole + 2
    SkippedRole = Qt.ItemDataRole.UserRole + 3
    ModifiedAtRole = Qt.ItemDataRole.UserRole + 4
    SourcePathRole = Qt.ItemDataRole.UserRole + 5
    SourceKindRole = Qt.ItemDataRole.UserRole + 6

    MIME_TYPE = "application/x-comic-translate-page-paths"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._items: list[PageListItemData] = []

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._items)

    def data(self, index: QtCore.QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = index.row()
        if not (0 <= row < len(self._items)):
            return None
        item = self._items[row]
        if role == Qt.ItemDataRole.DisplayRole:
            return item.display_name
        if role == Qt.ItemDataRole.ToolTipRole:
            return item.file_name
        if role == self.FilePathRole:
            return item.file_path
        if role == self.FileNameRole:
            return item.file_name
        if role == self.ThumbnailRole:
            return item.thumbnail
        if role == self.SkippedRole:
            return item.skipped
        if role == self.ModifiedAtRole:
            return item.modified_at
        if role == self.SourcePathRole:
            return item.source_path
        if role == self.SourceKindRole:
            return item.source_kind
        return None

    def flags(self, index: QtCore.QModelIndex) -> Qt.ItemFlag:
        default_flags = (
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsDragEnabled
            | Qt.ItemFlag.ItemIsDropEnabled
        )
        if not index.isValid():
            return Qt.ItemFlag.ItemIsDropEnabled
        return default_flags

    def supportedDropActions(self) -> Qt.DropAction:
        return Qt.DropAction.MoveAction

    def mimeTypes(self) -> list[str]:
        return [self.MIME_TYPE]

    def mimeData(self, indexes: list[QtCore.QModelIndex]) -> QtCore.QMimeData:
        mime = QtCore.QMimeData()
        file_paths = []
        seen: set[str] = set()
        for index in indexes:
            if not index.isValid():
                continue
            file_path = self.data(index, self.FilePathRole)
            if isinstance(file_path, str) and file_path and file_path not in seen:
                file_paths.append(file_path)
                seen.add(file_path)
        mime.setData(self.MIME_TYPE, json.dumps(file_paths).encode("utf-8"))
        return mime

    def set_items(self, items: list[PageListItemData]) -> None:
        self.beginResetModel()
        self._items = list(items)
        self.endResetModel()

    def clear(self) -> None:
        self.set_items([])

    def items(self) -> list[PageListItemData]:
        return list(self._items)

    def file_paths(self) -> list[str]:
        return [item.file_path for item in self._items]

    def file_path_at(self, row: int) -> str | None:
        if 0 <= row < len(self._items):
            return self._items[row].file_path
        return None

    def row_for_path(self, file_path: str) -> int:
        for row, item in enumerate(self._items):
            if item.file_path == file_path:
                return row
        return -1

    def set_thumbnail(self, file_path: str, pixmap: QtGui.QPixmap | None) -> None:
        row = self.row_for_path(file_path)
        if row < 0:
            return
        self._items[row].thumbnail = pixmap
        index = self.index(row, 0)
        self.dataChanged.emit(index, index, [self.ThumbnailRole])

    def clear_thumbnail(self, file_path: str) -> None:
        self.set_thumbnail(file_path, None)

    def set_skipped(self, file_path: str, skipped: bool) -> None:
        row = self.row_for_path(file_path)
        if row < 0 or self._items[row].skipped == skipped:
            return
        self._items[row].skipped = skipped
        index = self.index(row, 0)
        self.dataChanged.emit(index, index, [self.SkippedRole])

    def reorder_paths(self, moving_paths: list[str], target_row: int) -> bool:
        if not moving_paths:
            return False

        moving_set = set(moving_paths)
        current_paths = self.file_paths()
        moving_items = [item for item in self._items if item.file_path in moving_set]
        if not moving_items:
            return False

        remaining = [item for item in self._items if item.file_path not in moving_set]
        adjusted_target = max(0, min(target_row, len(self._items)))
        for item in self._items[:target_row]:
            if item.file_path in moving_set:
                adjusted_target -= 1
        adjusted_target = max(0, min(adjusted_target, len(remaining)))

        reordered = remaining[:adjusted_target] + moving_items + remaining[adjusted_target:]
        new_paths = [item.file_path for item in reordered]
        if new_paths == current_paths:
            return False

        self.beginResetModel()
        self._items = reordered
        self.endResetModel()
        return True


class PageListDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._thumbnail_size = QtCore.QSize(35, 50)
        self._horizontal_padding = 8
        self._vertical_padding = 6

    def sizeHint(
        self,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> QtCore.QSize:
        height = self._thumbnail_size.height() + (self._vertical_padding * 2) + 4
        return QtCore.QSize(option.rect.width(), height)

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        rect = option.rect.adjusted(4, 2, -4, -2)
        palette = option.palette
        selected = bool(option.state & QtWidgets.QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QtWidgets.QStyle.StateFlag.State_MouseOver)
        skipped = bool(index.data(PageListModel.SkippedRole))

        bg_color = palette.base().color().lighter(112 if hovered else 104)
        border_color = palette.mid().color()
        if selected:
            bg_color = palette.highlight().color().lighter(120)
            border_color = palette.highlight().color()

        painter.setPen(QtGui.QPen(border_color, 1))
        painter.setBrush(bg_color)
        painter.drawRoundedRect(rect, 8, 8)

        thumb_rect = QtCore.QRect(
            rect.left() + self._horizontal_padding,
            rect.top() + self._vertical_padding,
            self._thumbnail_size.width(),
            self._thumbnail_size.height(),
        )
        painter.setPen(QtGui.QPen(palette.mid().color(), 1))
        painter.setBrush(palette.alternateBase())
        painter.drawRoundedRect(thumb_rect, 4, 4)

        pixmap = index.data(PageListModel.ThumbnailRole)
        if isinstance(pixmap, QtGui.QPixmap) and not pixmap.isNull():
            scaled = pixmap.scaled(
                self._thumbnail_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            draw_rect = QtCore.QRect(
                thumb_rect.left() + (thumb_rect.width() - scaled.width()) // 2,
                thumb_rect.top() + (thumb_rect.height() - scaled.height()) // 2,
                scaled.width(),
                scaled.height(),
            )
            clip = QtGui.QPainterPath()
            clip.addRoundedRect(QtCore.QRectF(thumb_rect), 4, 4)
            painter.setClipPath(clip)
            painter.drawPixmap(draw_rect, scaled)
            painter.setClipping(False)

        text_rect = QtCore.QRect(
            thumb_rect.right() + 8,
            rect.top() + self._vertical_padding,
            rect.width() - thumb_rect.width() - (self._horizontal_padding * 3),
            thumb_rect.height(),
        )
        title = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        font = QtGui.QFont(option.font)
        font.setBold(selected)
        font.setStrikeOut(skipped)
        painter.setFont(font)

        text_color = palette.text().color()
        if skipped:
            text_color = palette.mid().color()
        elif selected:
            text_color = palette.highlightedText().color()
        painter.setPen(text_color)

        metrics = QtGui.QFontMetrics(font)
        title = metrics.elidedText(title, Qt.TextElideMode.ElideMiddle, text_rect.width())
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            title,
        )

        painter.restore()


class _PageListContentView(QtWidgets.QListView):
    def __init__(self, owner: "PageListView") -> None:
        super().__init__(owner)
        self._owner = owner
        self.setSpacing(4)
        self.setAlternatingRowColors(False)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setAutoScroll(True)
        self.setAutoScrollMargin(36)
        self.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setUniformItemSizes(True)
        self.setMouseTracking(True)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)

    def startDrag(self, supportedActions: Qt.DropAction) -> None:
        indexes = self.selectionModel().selectedRows()
        if not indexes:
            return

        mime_data = self.model().mimeData(indexes)
        drag = QtGui.QDrag(self)
        drag.setMimeData(mime_data)
        drag.setPixmap(self._build_drag_pixmap(indexes))
        drag.setHotSpot(QtCore.QPoint(16, 16))
        drag.exec(Qt.DropAction.MoveAction)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        model = self.model()
        if not isinstance(model, PageListModel):
            super().dropEvent(event)
            return

        if event.source() is self and event.mimeData().hasFormat(PageListModel.MIME_TYPE):
            try:
                moving_paths = json.loads(
                    bytes(event.mimeData().data(PageListModel.MIME_TYPE)).decode("utf-8")
                )
            except Exception:
                moving_paths = []

            target_row = self._target_row_from_event(event)
            if model.reorder_paths(moving_paths, target_row):
                self._owner.clear_active_sort()
                self._owner.restore_selection(moving_paths, moving_paths[0] if moving_paths else None)
                self._owner.order_changed.emit(model.file_paths())
            event.acceptProposedAction()
            return

        super().dropEvent(event)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        index = self.indexAt(event.pos())
        if index.isValid() and not self.selectionModel().isSelected(index):
            self.selectionModel().setCurrentIndex(
                index,
                QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
                | QtCore.QItemSelectionModel.SelectionFlag.Rows,
            )
        self._owner.show_context_menu(event.globalPos())

    def _target_row_from_event(self, event: QtGui.QDropEvent) -> int:
        pos = event.position().toPoint()
        target_index = self.indexAt(pos)
        if not target_index.isValid():
            return self.model().rowCount()

        rect = self.visualRect(target_index)
        if pos.y() > rect.center().y():
            return target_index.row() + 1
        return target_index.row()

    def _build_drag_pixmap(self, indexes: list[QtCore.QModelIndex]) -> QtGui.QPixmap:
        title = str(indexes[0].data(Qt.ItemDataRole.DisplayRole) or "")
        count = len(indexes)
        label = title if count == 1 else self.tr("{count} pages").format(count=count)

        metrics = QtGui.QFontMetrics(self.font())
        width = min(260, max(140, metrics.horizontalAdvance(label) + 36))
        pixmap = QtGui.QPixmap(width, 34)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        base = self.palette().window().color().darker(110)
        painter.setBrush(base)
        painter.setPen(QtGui.QPen(self.palette().highlight().color(), 1))
        painter.drawRoundedRect(pixmap.rect().adjusted(1, 1, -1, -1), 8, 8)
        painter.setPen(self.palette().windowText().color())
        painter.drawText(
            pixmap.rect().adjusted(12, 0, -12, 0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            metrics.elidedText(label, Qt.TextElideMode.ElideRight, width - 24),
        )
        painter.end()
        return pixmap


class PageListView(QtWidgets.QWidget):
    currentItemChanged = Signal(object, object)
    del_img = Signal(list)
    toggle_skip_img = Signal(list, bool)
    translate_imgs = Signal(list)
    selection_changed = Signal(list)
    order_changed = Signal(list)
    sort_requested = Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumWidth(100)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.insert_browser = MClickBrowserFilePushButton(multiple=True)
        self.insert_browser.set_dayu_filters(
            [
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
                ".bmp",
                ".zip",
                ".cbz",
                ".cbr",
                ".cb7",
                ".cbt",
                ".pdf",
                ".epub",
            ]
        )

        self._model = PageListModel(self)
        self._delegate = PageListDelegate(self)
        self._list_view = _PageListContentView(self)
        self._list_view.setModel(self._model)
        self._list_view.setItemDelegate(self._delegate)

        self._active_sort_key: str | None = None
        self._name_direction = "asc"
        self._date_direction = "desc"

        self._name_sort_button = QtWidgets.QPushButton(self.tr("Name"))
        self._name_sort_button.setCheckable(True)
        self._name_sort_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._name_sort_button.clicked.connect(self._request_name_sort)

        self._date_sort_button = QtWidgets.QPushButton(self.tr("Date"))
        self._date_sort_button.setCheckable(True)
        self._date_sort_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._date_sort_button.clicked.connect(self._request_date_sort)

        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self._name_sort_button)
        controls_layout.addWidget(self._date_sort_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._list_view, 1)
        layout.addLayout(controls_layout)

        selection_model = self._list_view.selectionModel()
        selection_model.selectionChanged.connect(self._on_selection_changed)
        selection_model.currentChanged.connect(self.currentItemChanged.emit)
        self._update_sort_buttons()

    def _direction_icon(self, direction: str) -> QtGui.QIcon:
        style = self.style()
        if direction == "asc":
            return style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowUp)
        return style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowDown)

    def _update_sort_buttons(self) -> None:
        self._name_sort_button.setIcon(self._direction_icon(self._name_direction))
        self._date_sort_button.setIcon(self._direction_icon("asc" if self._date_direction == "oldest" else "desc"))
        self._name_sort_button.setToolTip(
            self.tr("Sort by Name")
            + "\n"
            + (
                self.tr("Name: A to Z")
                if self._name_direction == "asc"
                else self.tr("Name: Z to A")
            )
        )
        self._date_sort_button.setToolTip(
            self.tr("Sort by Date")
            + "\n"
            + (
                self.tr("Date: Newest First")
                if self._date_direction == "desc"
                else self.tr("Date: Oldest First")
            )
        )
        self._name_sort_button.setChecked(self._active_sort_key == "name")
        self._date_sort_button.setChecked(self._active_sort_key == "date")

    def _request_name_sort(self) -> None:
        if self._active_sort_key == "name":
            self._name_direction = "desc" if self._name_direction == "asc" else "asc"
        self._active_sort_key = "name"
        self._update_sort_buttons()
        self.sort_requested.emit("name", self._name_direction)

    def _request_date_sort(self) -> None:
        if self._active_sort_key == "date":
            self._date_direction = "oldest" if self._date_direction == "desc" else "desc"
        self._active_sort_key = "date"
        self._update_sort_buttons()
        self.sort_requested.emit("date", self._date_direction)

    def clear_active_sort(self) -> None:
        self._active_sort_key = None
        self._update_sort_buttons()

    def model(self) -> PageListModel:
        return self._model

    def content_view(self) -> QtWidgets.QListView:
        return self._list_view

    def set_page_items(self, items: list[PageListItemData]) -> None:
        self._model.set_items(items)

    def clear(self) -> None:
        self._model.clear()

    def count(self) -> int:
        return self._model.rowCount()

    def setCurrentRow(self, row: int) -> None:
        if not (0 <= row < self.count()):
            self._list_view.clearSelection()
            self._list_view.setCurrentIndex(QtCore.QModelIndex())
            return
        index = self._model.index(row, 0)
        selection_model = self._list_view.selectionModel()
        selection_model.setCurrentIndex(
            index,
            QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
            | QtCore.QItemSelectionModel.SelectionFlag.Rows,
        )
        self._list_view.scrollTo(index, QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter)

    def currentRow(self) -> int:
        current_index = self._list_view.currentIndex()
        return current_index.row() if current_index.isValid() else -1

    def selected_file_paths(self) -> list[str]:
        paths: list[str] = []
        for index in self._list_view.selectionModel().selectedRows():
            path = index.data(PageListModel.FilePathRole)
            if isinstance(path, str) and path:
                paths.append(path)
        return paths

    def restore_selection(self, file_paths: list[str], current_path: str | None = None) -> None:
        selection_model = self._list_view.selectionModel()
        selection_model.clearSelection()
        for file_path in file_paths:
            row = self._model.row_for_path(file_path)
            if row < 0:
                continue
            index = self._model.index(row, 0)
            selection_model.select(
                index,
                QtCore.QItemSelectionModel.SelectionFlag.Select
                | QtCore.QItemSelectionModel.SelectionFlag.Rows,
            )

        target_path = current_path or (file_paths[0] if file_paths else None)
        if target_path:
            row = self._model.row_for_path(target_path)
            if row >= 0:
                index = self._model.index(row, 0)
                selection_model.setCurrentIndex(
                    index,
                    QtCore.QItemSelectionModel.SelectionFlag.Current
                    | QtCore.QItemSelectionModel.SelectionFlag.Select
                    | QtCore.QItemSelectionModel.SelectionFlag.Rows,
                )
                self._list_view.scrollTo(index, QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter)

    def set_thumbnail(self, file_path: str, pixmap: QtGui.QPixmap | None) -> None:
        self._model.set_thumbnail(file_path, pixmap)

    def clear_thumbnail(self, file_path: str) -> None:
        self._model.clear_thumbnail(file_path)

    def set_path_skipped(self, file_path: str, skipped: bool) -> None:
        self._model.set_skipped(file_path, skipped)

    def set_selected_paths(self, file_paths: list[str], current_path: str | None = None) -> None:
        self.restore_selection(file_paths, current_path)

    def show_context_menu(self, global_pos: QtCore.QPoint) -> None:
        menu = MMenu(parent=self)
        insert = menu.addAction(self.tr("Insert"))
        delete_act = menu.addAction(self.tr("Delete"))

        selected_paths = self.selected_file_paths()
        selected_indexes = self._list_view.selectionModel().selectedRows()
        all_skipped = bool(selected_indexes) and all(
            bool(index.data(PageListModel.SkippedRole)) for index in selected_indexes
        )

        skip_action = menu.addAction(self.tr("Unskip") if all_skipped else self.tr("Skip"))
        translate_act = menu.addAction(self.tr("Translate"))

        insert.triggered.connect(self.insert_browser.clicked)
        delete_act.triggered.connect(lambda: self.del_img.emit(selected_paths))
        skip_action.triggered.connect(lambda: self.toggle_skip_img.emit(selected_paths, not all_skipped))
        translate_act.triggered.connect(lambda: self.translate_imgs.emit(selected_paths))

        menu.exec_(global_pos)

    def _on_selection_changed(self, selected, deselected) -> None:
        selected_indices = [index.row() for index in self._list_view.selectionModel().selectedRows()]
        self.selection_changed.emit(selected_indices)
