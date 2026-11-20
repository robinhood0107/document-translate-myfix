from PySide6 import QtCore, QtWidgets

from ..dayu_widgets.label import MLabel
from ..dayu_widgets.spin_box import MSpinBox
from modules.utils.paths import get_default_project_autosave_dir


class ProjectPage(QtWidgets.QWidget):
    glossary_changed = QtCore.Signal()
    open_checkpoint_folder_requested = QtCore.Signal()
    clean_checkpoint_cache_requested = QtCore.Signal()
    force_checkpoint_recompute_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        self._glossary_editor_enabled = True
        self._glossary_validation_error = False

        auto_recover_label = MLabel(self.tr("Auto-Recover")).h4()
        auto_recover_note = MLabel(
            self.tr(
                "Auto-Recover saves recovery snapshots in the background so work can be restored after a crash.\n"
                "These snapshots are not your main project file; use Save/Auto-Save for normal project saves."
            )
        ).secondary()
        auto_recover_note.setTextFormat(QtCore.Qt.PlainText)
        auto_recover_note.setWordWrap(True)

        interval_layout = QtWidgets.QHBoxLayout()
        interval_label = MLabel(self.tr("Create recovery snapshot every (minutes):"))
        self.project_autosave_interval_spinbox = MSpinBox().small()
        self.project_autosave_interval_spinbox.setMinimum(1)
        self.project_autosave_interval_spinbox.setMaximum(120)
        self.project_autosave_interval_spinbox.setValue(3)
        self.project_autosave_interval_spinbox.setFixedWidth(70)
        interval_layout.addWidget(interval_label)
        interval_layout.addWidget(self.project_autosave_interval_spinbox)
        interval_layout.addStretch(1)

        autosave_label = MLabel(self.tr("Auto-Save Project Files")).h4()
        autosave_note = MLabel(
            self.tr(
                "These are normal .ctpr or .seriesctpr project files saved while Auto-Save is enabled.\n"
                "This folder is separate from Auto-Recover snapshots."
            )
        ).secondary()
        autosave_note.setTextFormat(QtCore.Qt.PlainText)
        autosave_note.setWordWrap(True)

        autosave_folder_layout = QtWidgets.QHBoxLayout()
        autosave_folder_label = MLabel(self.tr("Auto-Save folder:"))
        self.project_autosave_folder_input = QtWidgets.QLineEdit()
        self.project_autosave_folder_input.setMinimumWidth(280)
        self.project_autosave_folder_input.setMaximumWidth(460)
        self.project_autosave_folder_input.setPlaceholderText(
            self.tr("Select a folder for auto-saved project files")
        )
        self.project_autosave_folder_input.setText(get_default_project_autosave_dir())
        browse_button = QtWidgets.QPushButton(self.tr("Browse"))
        browse_button.clicked.connect(self._choose_autosave_folder)
        reset_button = QtWidgets.QPushButton(self.tr("Reset"))
        reset_button.clicked.connect(self._reset_autosave_folder_to_default)
        autosave_folder_layout.addWidget(autosave_folder_label)
        autosave_folder_layout.addWidget(self.project_autosave_folder_input, 1)
        autosave_folder_layout.addWidget(browse_button)
        autosave_folder_layout.addWidget(reset_button)
        autosave_folder_layout.addStretch(1)

        checkpoint_label = MLabel(self.tr("Project Stage Checkpoints")).h4()
        checkpoint_note = MLabel(
            self.tr(
                "Preview: save reusable stage manifests and large artifacts beside "
                "each .ctpr project. Missing or damaged cache data is ignored and "
                "recalculated."
            )
        ).secondary()
        checkpoint_note.setTextFormat(QtCore.Qt.PlainText)
        checkpoint_note.setWordWrap(True)
        self.project_checkpoint_enabled_checkbox = QtWidgets.QCheckBox(
            self.tr("Enable project stage checkpoints")
        )
        self.project_checkpoint_enabled_checkbox.setChecked(False)

        checkpoint_actions = QtWidgets.QHBoxLayout()
        self.open_checkpoint_folder_button = QtWidgets.QPushButton(
            self.tr("Open Cache Folder")
        )
        self.clean_checkpoint_cache_button = QtWidgets.QPushButton(
            self.tr("Clean Unused Cache")
        )
        self.force_checkpoint_recompute_button = QtWidgets.QPushButton(
            self.tr("Force Stage Recalculation")
        )
        self.open_checkpoint_folder_button.clicked.connect(
            self.open_checkpoint_folder_requested.emit
        )
        self.clean_checkpoint_cache_button.clicked.connect(
            self.clean_checkpoint_cache_requested.emit
        )
        self.force_checkpoint_recompute_button.clicked.connect(
            self.force_checkpoint_recompute_requested.emit
        )
        checkpoint_actions.addWidget(self.open_checkpoint_folder_button)
        checkpoint_actions.addWidget(self.clean_checkpoint_cache_button)
        checkpoint_actions.addWidget(self.force_checkpoint_recompute_button)
        checkpoint_actions.addStretch(1)

        glossary_label = MLabel(self.tr("Project Glossary")).h4()
        glossary_note = MLabel(
            self.tr(
                "Preferred translations are sent to LLM translators as guidance. "
                "They do not replace translated text automatically."
            )
        ).secondary()
        glossary_note.setWordWrap(True)
        self.project_glossary_table = QtWidgets.QTableWidget(0, 2, self)
        self.project_glossary_table.setAccessibleName(glossary_label.text())
        self.project_glossary_table.setHorizontalHeaderLabels(
            [self.tr("Source term"), self.tr("Preferred translation")]
        )
        self.project_glossary_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.project_glossary_table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.project_glossary_table.verticalHeader().setVisible(False)
        self.project_glossary_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.project_glossary_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.project_glossary_table.itemChanged.connect(
            lambda *_args: self.glossary_changed.emit()
        )
        self.glossary_status_label = MLabel("").secondary()
        self.glossary_status_label.setWordWrap(True)
        self.glossary_status_label.setAccessibleName(glossary_label.text())
        self.glossary_status_label.hide()
        glossary_buttons = QtWidgets.QHBoxLayout()
        self.add_glossary_entry_button = QtWidgets.QPushButton(self.tr("Add"))
        self.delete_glossary_entry_button = QtWidgets.QPushButton(self.tr("Delete"))
        self.add_glossary_entry_button.clicked.connect(self.add_glossary_entry)
        self.delete_glossary_entry_button.clicked.connect(self.delete_glossary_entries)
        glossary_buttons.addWidget(self.add_glossary_entry_button)
        glossary_buttons.addWidget(self.delete_glossary_entry_button)
        glossary_buttons.addStretch(1)

        layout.addWidget(autosave_label)
        layout.addWidget(autosave_note)
        layout.addLayout(autosave_folder_layout)
        layout.addSpacing(12)
        layout.addWidget(auto_recover_label)
        layout.addWidget(auto_recover_note)
        layout.addLayout(interval_layout)
        layout.addSpacing(12)
        layout.addWidget(checkpoint_label)
        layout.addWidget(checkpoint_note)
        layout.addWidget(self.project_checkpoint_enabled_checkbox)
        layout.addLayout(checkpoint_actions)
        layout.addSpacing(12)
        layout.addWidget(glossary_label)
        layout.addWidget(glossary_note)
        layout.addWidget(self.glossary_status_label)
        layout.addWidget(self.project_glossary_table)
        layout.addLayout(glossary_buttons)
        layout.addStretch(1)

    def glossary_entries(self) -> list[dict[str, str]]:
        entries = []
        for row in range(self.project_glossary_table.rowCount()):
            source_item = self.project_glossary_table.item(row, 0)
            translation_item = self.project_glossary_table.item(row, 1)
            source = source_item.text() if source_item else ""
            translation = translation_item.text() if translation_item else ""
            if source or translation:
                entries.append({"source": source, "translation": translation})
        return entries

    def set_glossary_entries(self, entries: list[dict[str, str]]) -> None:
        table = self.project_glossary_table
        self._glossary_validation_error = False
        table.blockSignals(True)
        try:
            table.setRowCount(0)
            for entry in entries:
                row = table.rowCount()
                table.insertRow(row)
                table.setItem(row, 0, QtWidgets.QTableWidgetItem(entry["source"]))
                table.setItem(
                    row,
                    1,
                    QtWidgets.QTableWidgetItem(entry["translation"]),
                )
        finally:
            table.blockSignals(False)
        self._refresh_glossary_status()

    def set_glossary_validation_error(self, invalid: bool) -> None:
        self._glossary_validation_error = bool(invalid)
        self._refresh_glossary_status()

    def has_unpersisted_glossary_draft(self) -> bool:
        return self._glossary_validation_error

    def set_glossary_editor_enabled(self, enabled: bool) -> None:
        self._glossary_editor_enabled = bool(enabled)
        self.project_glossary_table.setEnabled(self._glossary_editor_enabled)
        self.add_glossary_entry_button.setEnabled(self._glossary_editor_enabled)
        self.delete_glossary_entry_button.setEnabled(self._glossary_editor_enabled)
        self._refresh_glossary_status()

    def _refresh_glossary_status(self) -> None:
        if not self._glossary_editor_enabled:
            message = self.tr("Open a series item to edit its project glossary.")
        elif self._glossary_validation_error:
            message = self.tr(
                "The whole glossary table is pending and is not saved until every row is "
                "complete. Complete every row to save it; otherwise, saving keeps the last "
                "valid table and discards these edits."
            )
        else:
            self.glossary_status_label.clear()
            self.glossary_status_label.hide()
            return
        self.glossary_status_label.setText(message)
        self.glossary_status_label.show()

    def add_glossary_entry(self) -> None:
        row = self.project_glossary_table.rowCount()
        self.project_glossary_table.insertRow(row)
        self.project_glossary_table.setItem(row, 0, QtWidgets.QTableWidgetItem())
        self.project_glossary_table.setItem(row, 1, QtWidgets.QTableWidgetItem())
        self.glossary_changed.emit()

    def delete_glossary_entries(self) -> None:
        rows = sorted(
            {index.row() for index in self.project_glossary_table.selectedIndexes()},
            reverse=True,
        )
        if not rows:
            return
        for row in rows:
            self.project_glossary_table.removeRow(row)
        self.glossary_changed.emit()

    def _choose_autosave_folder(self):
        current_value = self.project_autosave_folder_input.text().strip()
        initial_dir = current_value or get_default_project_autosave_dir()
        selected_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Choose Auto-Save Folder"),
            initial_dir,
        )
        if selected_folder:
            self.project_autosave_folder_input.setText(selected_folder)

    def _reset_autosave_folder_to_default(self):
        self.project_autosave_folder_input.setText(get_default_project_autosave_dir())
