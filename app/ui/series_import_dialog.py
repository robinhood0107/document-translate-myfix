from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets


class _SeriesImportTable(QtWidgets.QTableWidget):
    order_changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels(
            [
                self.tr("Use"),
                self.tr("Queue"),
                self.tr("Name"),
                self.tr("Type"),
                self.tr("Folder"),
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
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(42)
        self.verticalHeader().setMinimumSectionSize(42)
        self.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)

        header = self.horizontalHeader()
        header.setStretchLastSection(False)
        header.setHighlightSections(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Stretch)

        self.setColumnWidth(0, 64)
        self.setColumnWidth(1, 72)
        self.setColumnWidth(3, 88)

        self.setStyleSheet(
            """
            QTableWidget {
                background: #181c22;
                border: 1px solid rgba(255, 255, 255, 0.10);
                border-radius: 14px;
                alternate-background-color: #1f242c;
                color: #eef2f7;
                selection-background-color: rgba(79, 141, 255, 0.20);
                selection-color: #ffffff;
                outline: none;
            }
            QTableWidget::item {
                padding: 8px 10px;
                border: none;
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            }
            QTableWidget::item:selected {
                background: rgba(79, 141, 255, 0.22);
                color: #ffffff;
            }
            QHeaderView::section {
                background: #242b35;
                color: #dce7f5;
                border: none;
                border-bottom: 1px solid rgba(255, 255, 255, 0.12);
                padding: 10px 12px;
                font-weight: 600;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 12px;
                margin: 8px 0 8px 0;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 0.18);
                border-radius: 6px;
                min-height: 28px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            """
        )

    def dropEvent(self, event):  # type: ignore[override]
        super().dropEvent(event)
        self.order_changed.emit()


class SeriesImportDialog(QtWidgets.QDialog):
    def __init__(self, root_dir: str, paths: list[str], parent=None):
        super().__init__(parent)
        self._root_dir = os.path.normpath(os.path.abspath(root_dir or ""))
        self._paths = [os.path.normpath(os.path.abspath(path)) for path in paths]
        self._suppress_item_changed = False

        self.setWindowTitle(self.tr("Create Series Project"))
        self.resize(1080, 700)
        self.setMinimumSize(900, 620)
        self.setModal(True)
        self.setStyleSheet(
            """
            QDialog {
                background: #10141a;
                color: #eef2f7;
            }
            QLabel#seriesImportHeroTitle {
                font-size: 22px;
                font-weight: 700;
                color: #f5f7fb;
            }
            QLabel#seriesImportHeroNote {
                font-size: 13px;
                color: #b9c4d3;
            }
            QLabel#seriesImportSectionTitle {
                font-size: 12px;
                font-weight: 700;
                color: #d8e1ed;
                letter-spacing: 0.3px;
            }
            QLabel#seriesImportRootPath {
                padding: 10px 12px;
                border-radius: 10px;
                background: rgba(255, 255, 255, 0.05);
                color: #e4ebf5;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
            QLabel#seriesImportChip {
                padding: 6px 12px;
                border-radius: 999px;
                background: rgba(79, 141, 255, 0.16);
                color: #ddebff;
                border: 1px solid rgba(79, 141, 255, 0.34);
                font-weight: 600;
            }
            QFrame#seriesImportHero,
            QFrame#seriesImportBoard,
            QFrame#seriesImportFooter {
                background: #151a20;
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 16px;
            }
            QPushButton {
                min-height: 36px;
                padding: 0 16px;
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.10);
                background: rgba(255, 255, 255, 0.06);
                color: #eef2f7;
                font-weight: 600;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.10);
            }
            QPushButton#seriesImportPrimary {
                background: #4f8dff;
                border: 1px solid #6ea2ff;
                color: white;
            }
            QPushButton#seriesImportPrimary:hover {
                background: #6a9eff;
            }
            QPushButton#seriesImportGhost {
                background: transparent;
            }
            """
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        hero = QtWidgets.QFrame(self)
        hero.setObjectName("seriesImportHero")
        hero_layout = QtWidgets.QVBoxLayout(hero)
        hero_layout.setContentsMargins(18, 18, 18, 18)
        hero_layout.setSpacing(10)

        hero_top = QtWidgets.QHBoxLayout()
        hero_top.setContentsMargins(0, 0, 0, 0)
        hero_top.setSpacing(12)

        title_block = QtWidgets.QVBoxLayout()
        title_block.setContentsMargins(0, 0, 0, 0)
        title_block.setSpacing(6)

        title_label = QtWidgets.QLabel(self.tr("Create Series Project"))
        title_label.setObjectName("seriesImportHeroTitle")
        note = QtWidgets.QLabel(
            self.tr(
                "Choose the files to embed into the new `.seriesctpr`, then arrange the reading queue with drag-and-drop or precise queue numbers."
            )
        )
        note.setObjectName("seriesImportHeroNote")
        note.setWordWrap(True)
        title_block.addWidget(title_label)
        title_block.addWidget(note)

        self.summary_chip = QtWidgets.QLabel("")
        self.summary_chip.setObjectName("seriesImportChip")
        self.summary_chip.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        hero_top.addLayout(title_block, 1)
        hero_top.addWidget(self.summary_chip, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        hero_layout.addLayout(hero_top)

        root_title = QtWidgets.QLabel(self.tr("Series Root Folder"))
        root_title.setObjectName("seriesImportSectionTitle")
        self.root_path_label = QtWidgets.QLabel(self._root_dir or "—")
        self.root_path_label.setObjectName("seriesImportRootPath")
        self.root_path_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.root_path_label.setWordWrap(True)
        self.root_path_label.setToolTip(self._root_dir or "")
        hero_layout.addWidget(root_title)
        hero_layout.addWidget(self.root_path_label)

        layout.addWidget(hero)

        board = QtWidgets.QFrame(self)
        board.setObjectName("seriesImportBoard")
        board_layout = QtWidgets.QVBoxLayout(board)
        board_layout.setContentsMargins(16, 14, 16, 14)
        board_layout.setSpacing(12)

        board_header = QtWidgets.QHBoxLayout()
        board_header.setContentsMargins(0, 0, 0, 0)
        board_header.setSpacing(12)

        board_title_wrap = QtWidgets.QVBoxLayout()
        board_title_wrap.setContentsMargins(0, 0, 0, 0)
        board_title_wrap.setSpacing(4)

        board_title = QtWidgets.QLabel(self.tr("Queue Preview"))
        board_title.setObjectName("seriesImportSectionTitle")
        self.helper_label = QtWidgets.QLabel(
            self.tr(
                "Checked rows will be embedded. Drag rows to reorder them, or edit the queue number for exact placement."
            )
        )
        self.helper_label.setObjectName("seriesImportHeroNote")
        self.helper_label.setWordWrap(True)
        board_title_wrap.addWidget(board_title)
        board_title_wrap.addWidget(self.helper_label)

        self.selected_label = QtWidgets.QLabel("")
        self.selected_label.setObjectName("seriesImportChip")
        self.selected_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        board_header.addLayout(board_title_wrap, 1)
        board_header.addWidget(self.selected_label, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        board_layout.addLayout(board_header)

        self.table = _SeriesImportTable(self)
        board_layout.addWidget(self.table, 1)
        layout.addWidget(board, 1)

        footer = QtWidgets.QFrame(self)
        footer.setObjectName("seriesImportFooter")
        footer_layout = QtWidgets.QHBoxLayout(footer)
        footer_layout.setContentsMargins(16, 12, 16, 12)
        footer_layout.setSpacing(10)

        footer_note = QtWidgets.QLabel(
            self.tr(
                "Only checked files will be added. You can still reorder or remove queue items later from the series board."
            )
        )
        footer_note.setObjectName("seriesImportHeroNote")
        footer_note.setWordWrap(True)

        self.select_all_button = QtWidgets.QPushButton(self.tr("Select All"))
        self.select_all_button.setObjectName("seriesImportGhost")
        self.clear_all_button = QtWidgets.QPushButton(self.tr("Clear All"))
        self.clear_all_button.setObjectName("seriesImportGhost")
        self.cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
        self.cancel_button.setObjectName("seriesImportGhost")
        self.create_button = QtWidgets.QPushButton(self.tr("Create Series"))
        self.create_button.setObjectName("seriesImportPrimary")
        self.create_button.setDefault(True)

        footer_layout.addWidget(footer_note, 1)
        footer_layout.addWidget(self.select_all_button)
        footer_layout.addWidget(self.clear_all_button)
        footer_layout.addSpacing(6)
        footer_layout.addWidget(self.cancel_button)
        footer_layout.addWidget(self.create_button)
        layout.addWidget(footer)

        self.cancel_button.clicked.connect(self.reject)
        self.create_button.clicked.connect(self._accept_if_valid)
        self.select_all_button.clicked.connect(lambda: self._set_all_checked(True))
        self.clear_all_button.clicked.connect(lambda: self._set_all_checked(False))
        self.table.order_changed.connect(self._sync_after_drag_drop)
        self.table.itemChanged.connect(self._on_item_changed)

        self._populate_table()

    def _build_entries(self) -> list[dict[str, object]]:
        entries = []
        for index, path in enumerate(self._paths, start=1):
            rel_dir = ""
            try:
                rel_path = os.path.relpath(path, self._root_dir)
                rel_dir = os.path.dirname(rel_path)
            except ValueError:
                rel_dir = os.path.dirname(path)
            entries.append(
                {
                    "queue_index": index,
                    "path": path,
                    "name": os.path.basename(path),
                    "type": os.path.splitext(path)[1].lstrip(".").upper(),
                    "folder": rel_dir.replace("\\", "/"),
                    "include": True,
                }
            )
        return entries

    def _decorate_row(self, row: int) -> None:
        queue_item = self.table.item(row, 1)
        name_item = self.table.item(row, 2)
        type_item = self.table.item(row, 3)
        folder_item = self.table.item(row, 4)
        if queue_item is not None:
            queue_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            font = queue_item.font()
            font.setBold(True)
            queue_item.setFont(font)
            queue_item.setForeground(QtGui.QBrush(QtGui.QColor("#dcebff")))
        if name_item is not None:
            font = name_item.font()
            font.setBold(True)
            name_item.setFont(font)
        if type_item is not None:
            type_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            type_item.setForeground(QtGui.QBrush(QtGui.QColor("#8fd7ff")))
        if folder_item is not None:
            folder_item.setForeground(QtGui.QBrush(QtGui.QColor("#8f9aad")))

    def _populate_table(self) -> None:
        entries = self._build_entries()
        self._suppress_item_changed = True
        self.table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            include_item = QtWidgets.QTableWidgetItem()
            include_item.setFlags(
                QtCore.Qt.ItemFlag.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsSelectable
                | QtCore.Qt.ItemFlag.ItemIsUserCheckable
            )
            include_item.setCheckState(QtCore.Qt.CheckState.Checked)
            include_item.setData(QtCore.Qt.ItemDataRole.UserRole, entry["path"])
            include_item.setToolTip(str(entry["path"]))

            queue_item = QtWidgets.QTableWidgetItem(f"{int(entry['queue_index']):02d}")
            queue_item.setData(QtCore.Qt.ItemDataRole.UserRole, entry["path"])

            name_item = QtWidgets.QTableWidgetItem(str(entry["name"]))
            name_item.setData(QtCore.Qt.ItemDataRole.UserRole, entry["path"])
            name_item.setFlags(
                QtCore.Qt.ItemFlag.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsSelectable
                | QtCore.Qt.ItemFlag.ItemIsDragEnabled
            )
            name_item.setToolTip(str(entry["path"]))

            type_item = QtWidgets.QTableWidgetItem(str(entry["type"]))
            type_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)

            folder_item = QtWidgets.QTableWidgetItem(str(entry["folder"] or self.tr("Root Folder")))
            folder_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
            folder_item.setToolTip(str(entry["folder"]))

            self.table.setItem(row, 0, include_item)
            self.table.setItem(row, 1, queue_item)
            self.table.setItem(row, 2, name_item)
            self.table.setItem(row, 3, type_item)
            self.table.setItem(row, 4, folder_item)
            self._decorate_row(row)
        self._suppress_item_changed = False
        self._refresh_summary()

    def _set_all_checked(self, checked: bool) -> None:
        self._suppress_item_changed = True
        target = QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None:
                item.setCheckState(target)
        self._suppress_item_changed = False
        self._refresh_summary()

    def _sync_after_drag_drop(self) -> None:
        self._suppress_item_changed = True
        for row in range(self.table.rowCount()):
            queue_item = self.table.item(row, 1)
            if queue_item is not None:
                queue_item.setText(f"{row + 1:02d}")
            self._decorate_row(row)
        self._suppress_item_changed = False
        self._refresh_summary()

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._suppress_item_changed or item is None:
            return
        if item.column() == 1:
            try:
                requested_index = max(1, int(item.text()))
            except (TypeError, ValueError):
                requested_index = item.row() + 1
            requested_row = min(self.table.rowCount(), requested_index) - 1
            current_row = item.row()
            if requested_row != current_row:
                self._move_row(current_row, requested_row)
            else:
                item.setText(f"{current_row + 1:02d}")
        self._refresh_summary()

    def _move_row(self, source_row: int, target_row: int) -> None:
        source_data = []
        for column in range(self.table.columnCount()):
            original = self.table.item(source_row, column)
            source_data.append(original.clone() if original is not None else None)

        self.table.removeRow(source_row)
        self.table.insertRow(target_row)
        for column, cloned in enumerate(source_data):
            if cloned is not None:
                self.table.setItem(target_row, column, cloned)
        self._sync_after_drag_drop()

    def _refresh_summary(self) -> None:
        total = self.table.rowCount()
        selected = len(self.selected_paths())
        folders = {
            str(self.table.item(row, 4).text() or "")
            for row in range(total)
            if self.table.item(row, 4) is not None
        }
        self.summary_chip.setText(
            self.tr("{total} files found").format(total=total)
        )
        self.selected_label.setText(
            self.tr("{selected} selected · {folders} folders").format(
                selected=selected,
                folders=len(folders),
            )
        )
        self.create_button.setEnabled(selected > 0)

    def selected_paths(self) -> list[str]:
        selected = []
        for row in range(self.table.rowCount()):
            include_item = self.table.item(row, 0)
            path = include_item.data(QtCore.Qt.ItemDataRole.UserRole) if include_item is not None else ""
            if include_item is not None and include_item.checkState() == QtCore.Qt.CheckState.Checked and path:
                selected.append(str(path))
        return selected

    def _accept_if_valid(self) -> None:
        if not self.selected_paths():
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Create Series Project"),
                self.tr("Select at least one file to include in the series."),
            )
            return
        self.accept()
