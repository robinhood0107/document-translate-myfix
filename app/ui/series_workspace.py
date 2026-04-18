from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class _SeriesQueueTable(QtWidgets.QTableWidget):
    order_changed = QtCore.Signal(list)
    open_requested = QtCore.Signal(str)
    remove_requested = QtCore.Signal(str)
    queue_index_requested = QtCore.Signal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._suppress_item_changed = False
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
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(40)
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

    def set_series_items(self, items: list[dict[str, object]]) -> None:
        self._suppress_item_changed = True
        self.setRowCount(len(items))
        for row, item in enumerate(items):
            item_id = str(item.get("series_item_id") or "")
            queue_text = f"{int(item.get('queue_index', row + 1) or (row + 1)):02d}"

            queue_item = QtWidgets.QTableWidgetItem(queue_text)
            queue_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            queue_item.setData(QtCore.Qt.ItemDataRole.UserRole, item_id)

            name_item = QtWidgets.QTableWidgetItem(str(item.get("display_name") or ""))
            name_item.setData(QtCore.Qt.ItemDataRole.UserRole, item_id)
            name_item.setToolTip(str(item.get("source_origin_path") or ""))

            source_kind = str(item.get("source_kind") or "")
            source_kind_label = self.tr("Project") if source_kind == "ctpr_import" else self.tr("Source")
            type_item = QtWidgets.QTableWidgetItem(source_kind_label)
            type_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)

            folder_item = QtWidgets.QTableWidgetItem(str(item.get("source_origin_relpath") or ""))
            folder_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)

            status_item = QtWidgets.QTableWidgetItem(str(item.get("status") or "pending"))
            status_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
            status_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

            remove_button = QtWidgets.QToolButton(self)
            remove_button.setText("×")
            remove_button.setToolTip(self.tr("Remove from series"))
            remove_button.clicked.connect(lambda _=False, target=item_id: self.remove_requested.emit(target))

            self.setItem(row, 0, queue_item)
            self.setItem(row, 1, name_item)
            self.setItem(row, 2, type_item)
            self.setItem(row, 3, folder_item)
            self.setItem(row, 4, status_item)
            self.setCellWidget(row, 5, remove_button)
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
        item_id = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
        try:
            requested_index = max(1, int(item.text()))
        except (TypeError, ValueError):
            requested_index = item.row() + 1
        self.queue_index_requested.emit(item_id, requested_index)

    def _on_item_double_clicked(self, item: QtWidgets.QTableWidgetItem) -> None:
        if item is None:
            return
        item_id = ""
        queue_item = self.item(item.row(), 0)
        if queue_item is not None:
            item_id = str(queue_item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
        if item_id:
            self.open_requested.emit(item_id)


class _SeriesQuickSettings(QtWidgets.QWidget):
    changed = QtCore.Signal()
    auto_translate_requested = QtCore.Signal()
    open_series_settings_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

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

        self.series_settings_button = QtWidgets.QPushButton(self.tr("Series Settings…"))
        self.auto_translate_button = QtWidgets.QPushButton(self.tr("대기열대로 자동번역"))
        self.auto_translate_button.setObjectName("seriesAutoTranslateButton")

        layout.addWidget(header)
        layout.addWidget(note)
        layout.addLayout(form)
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


class SeriesTreeJumpDialog(QtWidgets.QDialog):
    def __init__(self, items: list[dict[str, object]], parent=None):
        super().__init__(parent)
        self._selected_item_id = ""
        self.setWindowTitle(self.tr("Tree Jump"))
        self.resize(480, 560)

        layout = QtWidgets.QVBoxLayout(self)
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
    open_series_settings_requested = QtCore.Signal()
    global_settings_changed = QtCore.Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
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

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(12)
        layout.addLayout(body, 1)

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
        left_layout.addLayout(action_row)

        self.queue_table = _SeriesQueueTable(self)
        left_layout.addWidget(self.queue_table, 1)
        body.addWidget(left_panel, 1)

        self.quick_settings = _SeriesQuickSettings(self)
        quick_frame = QtWidgets.QFrame(self)
        quick_frame.setObjectName("seriesQuickFrame")
        quick_frame_layout = QtWidgets.QVBoxLayout(quick_frame)
        quick_frame_layout.setContentsMargins(12, 12, 12, 12)
        quick_frame_layout.addWidget(self.quick_settings)
        quick_frame.setMinimumWidth(320)
        quick_frame.setMaximumWidth(420)
        body.addWidget(quick_frame, 0)

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
        self.quick_settings.auto_translate_requested.connect(self.auto_translate_requested)
        self.quick_settings.open_series_settings_requested.connect(self.open_series_settings_requested)
        self.quick_settings.changed.connect(self._emit_global_settings_changed)

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
        self.back_button.setEnabled(can_back)
        self.forward_button.setEnabled(can_forward)

    def set_series_state(
        self,
        *,
        series_file: str,
        items: list[dict[str, object]],
        queue_running: bool = False,
    ) -> None:
        self.title_label.setText(series_file)
        self.scope_badge.setText(self.tr("Series Project"))
        self.queue_table.set_series_items(items)
        self.quick_settings.auto_translate_button.setEnabled(bool(items) and not queue_running)

    def prompt_tree_jump(self, items: list[dict[str, object]]) -> str:
        dialog = SeriesTreeJumpDialog(items, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return ""
        return dialog.selected_target()

    def _emit_open_selected(self) -> None:
        item_id = self.queue_table.selected_item_id()
        if item_id:
            self.open_item_requested.emit(item_id)

    def _emit_global_settings_changed(self) -> None:
        self.global_settings_changed.emit(self.quick_settings.values())
