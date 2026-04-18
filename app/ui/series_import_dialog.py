from __future__ import annotations

import os

from PySide6 import QtCore, QtWidgets


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
        self.verticalHeader().setVisible(False)
        header = self.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Stretch)

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
        self.resize(1040, 640)

        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            self.tr(
                "Select the files to include in the new `.seriesctpr`, then reorder the queue by drag-and-drop or by editing the queue number."
            )
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.summary_label = QtWidgets.QLabel("")
        layout.addWidget(self.summary_label)

        self.table = _SeriesImportTable(self)
        layout.addWidget(self.table, 1)

        button_row = QtWidgets.QHBoxLayout()
        self.select_all_button = QtWidgets.QPushButton(self.tr("Select All"))
        self.clear_all_button = QtWidgets.QPushButton(self.tr("Clear All"))
        self.cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
        self.create_button = QtWidgets.QPushButton(self.tr("Create Series"))
        self.create_button.setDefault(True)
        button_row.addWidget(self.select_all_button)
        button_row.addWidget(self.clear_all_button)
        button_row.addStretch(1)
        button_row.addWidget(self.cancel_button)
        button_row.addWidget(self.create_button)
        layout.addLayout(button_row)

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

            queue_item = QtWidgets.QTableWidgetItem(str(entry["queue_index"]))
            queue_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            queue_item.setData(QtCore.Qt.ItemDataRole.UserRole, entry["path"])

            name_item = QtWidgets.QTableWidgetItem(str(entry["name"]))
            name_item.setData(QtCore.Qt.ItemDataRole.UserRole, entry["path"])
            name_item.setFlags(
                QtCore.Qt.ItemFlag.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsSelectable
                | QtCore.Qt.ItemFlag.ItemIsDragEnabled
            )

            type_item = QtWidgets.QTableWidgetItem(str(entry["type"]))
            type_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)

            folder_item = QtWidgets.QTableWidgetItem(str(entry["folder"]))
            folder_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)

            self.table.setItem(row, 0, include_item)
            self.table.setItem(row, 1, queue_item)
            self.table.setItem(row, 2, name_item)
            self.table.setItem(row, 3, type_item)
            self.table.setItem(row, 4, folder_item)
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
                queue_item.setText(str(row + 1))
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
                item.setText(str(current_row + 1))
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
        self.summary_label.setText(
            self.tr("{selected} / {total} items selected").format(
                selected=selected,
                total=total,
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
