from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from modules.translation.translation_memory import (
    DEFAULT_RESULT_CACHE_LIMIT,
    DEFAULT_TM_CANDIDATE_LIMIT,
    TranslationMemoryStore,
)

from ..dayu_widgets.label import MLabel
from ..dayu_widgets.push_button import MPushButton


class CorrectionDictionaryTable(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, title: str, description: str, parent=None):
        super().__init__(parent)
        self._loading = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        self.setObjectName("correctionDictionaryCard")
        self.setStyleSheet(
            """
            QWidget#correctionDictionaryCard {
                background: rgba(255, 255, 255, 0.035);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 14px;
            }
            QWidget#correctionDictionaryCard QLabel {
                background: transparent;
                border: none;
            }
            QTableWidget {
                background-color: #e8edf3;
                alternate-background-color: #dfe6ee;
                color: #111827;
                border: 1px solid #c7d1dc;
                border-radius: 12px;
                gridline-color: transparent;
                selection-background-color: #d6e4f5;
                selection-color: #111827;
                outline: none;
            }
            QTableWidget::item {
                color: #111827;
                padding: 6px 8px;
            }
            QTableWidget::item:selected {
                color: #111827;
            }
            QHeaderView::section {
                background-color: #d6dee8;
                color: #334155;
                border: none;
                border-bottom: 1px solid #c7d1dc;
                padding: 8px 10px;
                font-weight: 600;
            }
            QTableCornerButton::section {
                background-color: #d6dee8;
                border: none;
                border-bottom: 1px solid #c7d1dc;
            }
            """
        )
        title_label = MLabel(title).h4()
        desc_label = MLabel(description).secondary()
        desc_label.setWordWrap(True)

        self.table = QtWidgets.QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels(
            [
                self.tr("Keyword"),
                self.tr("Substitution"),
                self.tr("Use regex"),
                self.tr("Case sensitive"),
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            0,
            QtWidgets.QHeaderView.ResizeMode.Stretch,
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1,
            QtWidgets.QHeaderView.ResizeMode.Stretch,
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2,
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents,
        )
        self.table.horizontalHeader().setSectionResizeMode(
            3,
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents,
        )
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(38)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.table.itemChanged.connect(self._on_item_changed)

        buttons = QtWidgets.QHBoxLayout()
        self.new_button = MPushButton(self.tr("New")).small()
        self.delete_button = MPushButton(self.tr("Delete")).small()
        self.new_button.clicked.connect(self.add_rule)
        self.delete_button.clicked.connect(self.delete_selected_rows)
        buttons.addWidget(self.new_button)
        buttons.addWidget(self.delete_button)
        buttons.addStretch(1)

        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        layout.addWidget(self.table)
        layout.addLayout(buttons)

    def _make_text_item(self, text: str) -> QtWidgets.QTableWidgetItem:
        item = QtWidgets.QTableWidgetItem(text)
        item.setForeground(QtGui.QBrush(QtGui.QColor("#111827")))
        item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsEditable
        )
        return item

    def _make_checkbox_item(self, checked: bool) -> QtWidgets.QTableWidgetItem:
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsUserCheckable
        )
        item.setCheckState(
            QtCore.Qt.CheckState.Checked
            if checked
            else QtCore.Qt.CheckState.Unchecked
        )
        return item

    def add_rule(
        self,
        keyword: str = "",
        sub: str = "",
        use_reg: bool = False,
        case_sens: bool = True,
        *,
        emit_changed: bool = True,
    ) -> None:
        self._loading = True
        try:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, self._make_text_item(keyword))
            self.table.setItem(row, 1, self._make_text_item(sub))
            self.table.setItem(row, 2, self._make_checkbox_item(use_reg))
            self.table.setItem(row, 3, self._make_checkbox_item(case_sens))
        finally:
            self._loading = False
        if emit_changed:
            self.changed.emit()

    def delete_selected_rows(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        self._loading = True
        try:
            for row in rows:
                self.table.removeRow(row)
        finally:
            self._loading = False
        self.changed.emit()

    def rules(self) -> list[dict]:
        result: list[dict] = []
        for row in range(self.table.rowCount()):
            keyword_item = self.table.item(row, 0)
            sub_item = self.table.item(row, 1)
            regex_item = self.table.item(row, 2)
            case_item = self.table.item(row, 3)
            result.append(
                {
                    "keyword": keyword_item.text() if keyword_item else "",
                    "sub": sub_item.text() if sub_item else "",
                    "use_reg": bool(
                        regex_item
                        and regex_item.checkState() == QtCore.Qt.CheckState.Checked
                    ),
                    "case_sens": bool(
                        case_item
                        and case_item.checkState() == QtCore.Qt.CheckState.Checked
                    ),
                }
            )
        return result

    def load_rules(self, rules: list[dict]) -> None:
        self._loading = True
        try:
            self.table.setRowCount(0)
            for rule in rules or []:
                self.add_rule(
                    keyword=str(rule.get("keyword", "") or ""),
                    sub=str(rule.get("sub", "") or ""),
                    use_reg=bool(rule.get("use_reg", False)),
                    case_sens=bool(rule.get("case_sens", True)),
                    emit_changed=False,
                )
        finally:
            self._loading = False

    def _on_item_changed(self, _item: QtWidgets.QTableWidgetItem) -> None:
        if self._loading:
            return
        self.changed.emit()


class TranslationMemoryPanel(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._store = TranslationMemoryStore()
        self.destroyed.connect(lambda *_args: self._store.close())

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        self.setObjectName("translationMemoryCard")
        self.setStyleSheet(
            """
            QWidget#translationMemoryCard {
                background: rgba(255, 255, 255, 0.035);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 14px;
            }
            QWidget#translationMemoryCard QLabel {
                background: transparent;
                border: none;
            }
            QTableWidget {
                background-color: #e8edf3;
                alternate-background-color: #dfe6ee;
                color: #111827;
                border: 1px solid #c7d1dc;
                border-radius: 12px;
                gridline-color: transparent;
                selection-background-color: #d6e4f5;
                selection-color: #111827;
                outline: none;
            }
            QTableWidget::item {
                color: #111827;
                padding: 6px 8px;
            }
            QHeaderView::section {
                background-color: #d6dee8;
                color: #334155;
                border: none;
                border-bottom: 1px solid #c7d1dc;
                padding: 8px 10px;
                font-weight: 600;
            }
            """
        )

        title = MLabel(self.tr("Exact Translation Memory")).h4()
        description = MLabel(
            self.tr(
                "Persistent result-cache text and translation-memory entries are sensitive local user data. "
                "Only explicitly approved exact source-to-translation pairs can bypass Gemma across contexts."
            )
        ).secondary()
        description.setWordWrap(True)

        toggles = QtWidgets.QGridLayout()
        self.persistent_cache_checkbox = QtWidgets.QCheckBox(
            self.tr("Enable persistent block result cache")
        )
        self.exact_tm_checkbox = QtWidgets.QCheckBox(
            self.tr(
                "Enable exact translation memory and collect unapproved candidates"
            )
        )
        self.persistent_cache_checkbox.setChecked(True)
        self.exact_tm_checkbox.setChecked(True)

        self.result_cache_limit_spinbox = QtWidgets.QSpinBox(self)
        self.result_cache_limit_spinbox.setRange(1_000, 500_000)
        self.result_cache_limit_spinbox.setSingleStep(1_000)
        self.result_cache_limit_spinbox.setValue(DEFAULT_RESULT_CACHE_LIMIT)
        self.candidate_limit_spinbox = QtWidgets.QSpinBox(self)
        self.candidate_limit_spinbox.setRange(100, 50_000)
        self.candidate_limit_spinbox.setSingleStep(100)
        self.candidate_limit_spinbox.setValue(DEFAULT_TM_CANDIDATE_LIMIT)

        toggles.addWidget(self.persistent_cache_checkbox, 0, 0, 1, 2)
        toggles.addWidget(self.exact_tm_checkbox, 1, 0, 1, 2)
        toggles.addWidget(self._label(self.tr("Result cache retention")), 2, 0)
        toggles.addWidget(self.result_cache_limit_spinbox, 2, 1)
        toggles.addWidget(self._label(self.tr("Unapproved candidate retention")), 3, 0)
        toggles.addWidget(self.candidate_limit_spinbox, 3, 1)
        toggles.setColumnStretch(0, 1)

        self.status_label = MLabel("").secondary()
        self.status_label.setWordWrap(True)

        self.table = QtWidgets.QTableWidget(0, 5, self)
        self.table.setHorizontalHeaderLabels(
            [
                self.tr("Source text"),
                self.tr("Translation"),
                self.tr("Source language"),
                self.tr("Target language"),
                self.tr("Approved"),
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            0,
            QtWidgets.QHeaderView.ResizeMode.Stretch,
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1,
            QtWidgets.QHeaderView.ResizeMode.Stretch,
        )
        for column in (2, 3, 4):
            self.table.horizontalHeader().setSectionResizeMode(
                column,
                QtWidgets.QHeaderView.ResizeMode.ResizeToContents,
            )
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.setMinimumHeight(240)

        buttons = QtWidgets.QHBoxLayout()
        self.refresh_button = MPushButton(self.tr("Refresh")).small()
        self.approve_button = MPushButton(self.tr("Approve Selected")).small()
        self.unapprove_button = MPushButton(self.tr("Unapprove Selected")).small()
        self.delete_button = MPushButton(self.tr("Delete Selected")).small()
        self.import_button = MPushButton(self.tr("Import")).small()
        self.export_button = MPushButton(self.tr("Export")).small()
        self.clear_cache_button = MPushButton(self.tr("Clear Result Cache")).small()
        for button in (
            self.refresh_button,
            self.approve_button,
            self.unapprove_button,
            self.delete_button,
            self.import_button,
            self.export_button,
            self.clear_cache_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)

        self.persistent_cache_checkbox.toggled.connect(
            lambda _checked: self.changed.emit()
        )
        self.exact_tm_checkbox.toggled.connect(
            lambda _checked: self.changed.emit()
        )
        self.result_cache_limit_spinbox.valueChanged.connect(self._limits_changed)
        self.candidate_limit_spinbox.valueChanged.connect(self._limits_changed)
        self.refresh_button.clicked.connect(self.refresh)
        self.approve_button.clicked.connect(lambda: self._set_selected_approved(True))
        self.unapprove_button.clicked.connect(lambda: self._set_selected_approved(False))
        self.delete_button.clicked.connect(self._delete_selected)
        self.import_button.clicked.connect(self._import_entries)
        self.export_button.clicked.connect(self._export_entries)
        self.clear_cache_button.clicked.connect(self._clear_result_cache)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addLayout(toggles)
        layout.addWidget(self.status_label)
        layout.addWidget(self.table)
        layout.addLayout(buttons)
        self.refresh()

    @staticmethod
    def _label(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        return label

    def get_settings(self) -> dict[str, object]:
        return {
            "persistent_cache_enabled": self.persistent_cache_checkbox.isChecked(),
            "exact_tm_enabled": self.exact_tm_checkbox.isChecked(),
            "result_cache_limit": int(self.result_cache_limit_spinbox.value()),
            "candidate_limit": int(self.candidate_limit_spinbox.value()),
        }

    def load_settings(self, settings: dict | None) -> None:
        values = dict(settings or {})
        blockers = [
            QtCore.QSignalBlocker(self.persistent_cache_checkbox),
            QtCore.QSignalBlocker(self.exact_tm_checkbox),
            QtCore.QSignalBlocker(self.result_cache_limit_spinbox),
            QtCore.QSignalBlocker(self.candidate_limit_spinbox),
        ]
        try:
            self.persistent_cache_checkbox.setChecked(
                bool(values.get("persistent_cache_enabled", True))
            )
            self.exact_tm_checkbox.setChecked(
                bool(values.get("exact_tm_enabled", True))
            )
            self.result_cache_limit_spinbox.setValue(
                int(values.get("result_cache_limit", DEFAULT_RESULT_CACHE_LIMIT))
            )
            self.candidate_limit_spinbox.setValue(
                int(values.get("candidate_limit", DEFAULT_TM_CANDIDATE_LIMIT))
            )
        finally:
            del blockers
        self._configure_store_limits()

    def _configure_store_limits(self) -> None:
        self._store.configure_limits(
            result_cache_limit=self.result_cache_limit_spinbox.value(),
            candidate_limit=self.candidate_limit_spinbox.value(),
        )

    def _limits_changed(self, _value: int) -> None:
        self._configure_store_limits()
        self.changed.emit()

    def _selected_entry_ids(self) -> list[int]:
        ids: list[int] = []
        for index in self.table.selectionModel().selectedRows():
            item = self.table.item(index.row(), 0)
            if item is None:
                continue
            entry_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if entry_id is not None:
                ids.append(int(entry_id))
        return ids

    def refresh(self) -> None:
        entries = self._store.list_tm_entries()
        self.table.setRowCount(0)
        for entry in entries:
            row = self.table.rowCount()
            self.table.insertRow(row)
            values = [
                entry["source_text"],
                entry["translation"],
                entry["source_lang"],
                entry["target_lang"],
                self.tr("Yes") if entry["approved"] else self.tr("No"),
            ]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                item.setForeground(QtGui.QBrush(QtGui.QColor("#111827")))
                if column == 0:
                    item.setData(
                        QtCore.Qt.ItemDataRole.UserRole,
                        int(entry["id"]),
                    )
                self.table.setItem(row, column, item)

        stats = self._store.stats()
        if stats["disabled"]:
            self.status_label.setText(
                self.tr(
                    "Translation memory is unavailable for this run. The database was left unchanged. Reason: {0}"
                ).format(stats["disabled_reason"])
            )
            return
        self.status_label.setText(
            self.tr(
                "Result cache: {0} entries · Approved TM: {1} · Candidates: {2} · Showing latest {3}"
            ).format(
                stats["result_cache_entries"],
                stats["approved_tm_entries"],
                stats["candidate_tm_entries"],
                len(entries),
            )
        )

    def _set_selected_approved(self, approved: bool) -> None:
        entry_ids = self._selected_entry_ids()
        if not entry_ids:
            return
        try:
            self._store.set_approved(entry_ids, approved)
        except Exception as exc:
            self._show_operation_failure(exc)
            return
        self.refresh()

    def _delete_selected(self) -> None:
        entry_ids = self._selected_entry_ids()
        if not entry_ids:
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            self.tr("Delete Translation Memory Entries"),
            self.tr("Delete the selected translation-memory entries?"),
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            self._store.delete_tm_entries(entry_ids)
        except Exception as exc:
            self._show_operation_failure(exc)
            return
        self.refresh()

    def _import_entries(self) -> None:
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Import Exact Translation Memory"),
            "",
            self.tr("JSON files (*.json)"),
        )
        if not file_name:
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            self.tr("Import Exact Translation Memory"),
            self.tr(
                "Approved entries in this file will be trusted and may bypass Gemma. Import this translation-memory file?"
            ),
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            imported = self._store.import_tm(file_name)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Import Failed"),
                str(exc),
            )
            return
        QtWidgets.QMessageBox.information(
            self,
            self.tr("Import Complete"),
            self.tr("Imported {0} translation-memory entries.").format(imported),
        )
        self.refresh()

    def _export_entries(self) -> None:
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            self.tr("Export Exact Translation Memory"),
            "comic-translate-exact-tm.json",
            self.tr("JSON files (*.json)"),
        )
        if not file_name:
            return
        try:
            exported = self._store.export_tm(file_name)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Export Failed"),
                str(exc),
            )
            return
        QtWidgets.QMessageBox.information(
            self,
            self.tr("Export Complete"),
            self.tr("Exported {0} translation-memory entries.").format(exported),
        )

    def _clear_result_cache(self) -> None:
        answer = QtWidgets.QMessageBox.question(
            self,
            self.tr("Clear Result Cache"),
            self.tr(
                "Clear all persistent block-result cache entries? Approved and candidate translation-memory entries will be kept."
            ),
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            removed = self._store.clear_result_cache()
        except Exception as exc:
            self._show_operation_failure(exc)
            return
        QtWidgets.QMessageBox.information(
            self,
            self.tr("Result Cache Cleared"),
            self.tr("Removed {0} result-cache entries.").format(removed),
        )
        self.refresh()

    def _show_operation_failure(self, exc: BaseException) -> None:
        QtWidgets.QMessageBox.warning(
            self,
            self.tr("Translation Memory Operation Failed"),
            str(exc),
        )


class UserDictionariesPage(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        intro = MLabel(
            self.tr(
                "Correction dictionaries rewrite OCR and translation results before they are saved to the project."
            )
        ).secondary()
        intro.setWordWrap(True)

        self.ocr_dictionary_table = CorrectionDictionaryTable(
            self.tr("OCR Result Dictionary"),
            self.tr(
                "Apply these substitutions immediately after OCR returns text, before the source text is stored."
            ),
            parent=self,
        )
        self.translation_dictionary_table = CorrectionDictionaryTable(
            self.tr("Translation Result Dictionary"),
            self.tr(
                "Apply these substitutions immediately after translation or TXT/MD import returns text, before the translation is stored."
            ),
            parent=self,
        )
        self.translation_memory_panel = TranslationMemoryPanel(parent=self)

        self.ocr_dictionary_table.changed.connect(self.changed.emit)
        self.translation_dictionary_table.changed.connect(self.changed.emit)
        self.translation_memory_panel.changed.connect(self.changed.emit)

        layout.addWidget(intro)
        layout.addWidget(self.ocr_dictionary_table)
        layout.addWidget(self.translation_dictionary_table)
        layout.addWidget(self.translation_memory_panel)
        layout.addStretch(1)

    def get_ocr_rules(self) -> list[dict]:
        return self.ocr_dictionary_table.rules()

    def get_translation_rules(self) -> list[dict]:
        return self.translation_dictionary_table.rules()

    def get_translation_memory_settings(self) -> dict[str, object]:
        return self.translation_memory_panel.get_settings()

    def load_rules(
        self,
        ocr_rules: list[dict] | None,
        translation_rules: list[dict] | None,
    ) -> None:
        self.ocr_dictionary_table.load_rules(ocr_rules or [])
        self.translation_dictionary_table.load_rules(translation_rules or [])

    def load_translation_memory_settings(self, settings: dict | None) -> None:
        self.translation_memory_panel.load_settings(settings)
