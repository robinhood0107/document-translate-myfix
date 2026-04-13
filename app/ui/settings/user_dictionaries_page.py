from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

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
                background-color: #f8fafc;
                alternate-background-color: #f1f5f9;
                color: #111827;
                border: 1px solid #d6dde8;
                border-radius: 12px;
                gridline-color: transparent;
                selection-background-color: #dbeafe;
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
                background-color: #eef2f7;
                color: #334155;
                border: none;
                border-bottom: 1px solid #d6dde8;
                padding: 8px 10px;
                font-weight: 600;
            }
            QTableCornerButton::section {
                background-color: #eef2f7;
                border: none;
                border-bottom: 1px solid #d6dde8;
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

        self.ocr_dictionary_table.changed.connect(self.changed.emit)
        self.translation_dictionary_table.changed.connect(self.changed.emit)

        layout.addWidget(intro)
        layout.addWidget(self.ocr_dictionary_table)
        layout.addWidget(self.translation_dictionary_table)
        layout.addStretch(1)

    def get_ocr_rules(self) -> list[dict]:
        return self.ocr_dictionary_table.rules()

    def get_translation_rules(self) -> list[dict]:
        return self.translation_dictionary_table.rules()

    def load_rules(
        self,
        ocr_rules: list[dict] | None,
        translation_rules: list[dict] | None,
    ) -> None:
        self.ocr_dictionary_table.load_rules(ocr_rules or [])
        self.translation_dictionary_table.load_rules(translation_rules or [])
