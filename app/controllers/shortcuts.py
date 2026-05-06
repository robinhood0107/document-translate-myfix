from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QSettings

from app.shortcuts import get_default_shortcuts, get_shortcut_definitions

if TYPE_CHECKING:
    from controller import ComicTranslate


class ShortcutController:
    SETTINGS_GROUP = "shortcuts"

    def __init__(self, main: "ComicTranslate"):
        self.main = main
        self._shortcuts: dict[str, QtGui.QShortcut] = {}
        self._register_shortcuts()

    def _register_shortcuts(self) -> None:
        for definition in get_shortcut_definitions():
            shortcut = QtGui.QShortcut(self.main)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(
                lambda shortcut_id=definition.id: self._activate_shortcut(shortcut_id)
            )
            self._shortcuts[definition.id] = shortcut

        self.apply_shortcuts()

    def apply_shortcuts(self) -> None:
        current_shortcuts = self.get_current_shortcuts()
        for shortcut_id, shortcut in self._shortcuts.items():
            shortcut.setKey(QtGui.QKeySequence(current_shortcuts.get(shortcut_id, "")))

    def get_current_shortcuts(self) -> dict[str, str]:
        shortcuts = get_default_shortcuts()
        settings = QSettings("ComicLabs", "ComicTranslate")
        settings.beginGroup(self.SETTINGS_GROUP)
        for definition in get_shortcut_definitions():
            shortcuts[definition.id] = settings.value(
                definition.id,
                shortcuts[definition.id],
                type=str,
            )
        settings.endGroup()
        return shortcuts

    def _activate_shortcut(self, shortcut_id: str) -> None:
        handlers = {
            "undo": self._undo,
            "redo": self._redo,
            "save_project": self._save_project,
            "delete_selected_box": self._delete_selected_box,
            "restore_text_blocks": self._restore_text_blocks,
        }
        handler = handlers.get(shortcut_id)
        if handler is not None:
            handler()

    def _workspace_is_active(self) -> bool:
        try:
            return self.main._center_stack.currentWidget() is self.main.main_content_widget
        except Exception:
            return False

    def _is_text_input_focused(self) -> bool:
        focus_widget = QtWidgets.QApplication.focusWidget()
        editable_types = (
            QtWidgets.QLineEdit,
            QtWidgets.QTextEdit,
            QtWidgets.QPlainTextEdit,
            QtWidgets.QAbstractSpinBox,
            QtWidgets.QKeySequenceEdit,
        )
        return isinstance(focus_widget, editable_types)

    def _try_focused_undo(self) -> bool:
        focus_widget = QtWidgets.QApplication.focusWidget()
        if isinstance(focus_widget, (QtWidgets.QLineEdit, QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit)):
            focus_widget.undo()
            return True
        if isinstance(focus_widget, QtWidgets.QAbstractSpinBox):
            line_edit = focus_widget.lineEdit()
            if line_edit is not None:
                line_edit.undo()
                return True
        text_item = getattr(self.main, "curr_tblock_item", None)
        if text_item is not None and getattr(text_item, "editing_mode", False):
            document = text_item.document()
            if document is not None:
                document.undo()
                return True
        return False

    def _try_focused_redo(self) -> bool:
        focus_widget = QtWidgets.QApplication.focusWidget()
        if isinstance(focus_widget, (QtWidgets.QLineEdit, QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit)):
            focus_widget.redo()
            return True
        if isinstance(focus_widget, QtWidgets.QAbstractSpinBox):
            line_edit = focus_widget.lineEdit()
            if line_edit is not None:
                line_edit.redo()
                return True
        text_item = getattr(self.main, "curr_tblock_item", None)
        if text_item is not None and getattr(text_item, "editing_mode", False):
            document = text_item.document()
            if document is not None:
                document.redo()
                return True
        return False

    def _undo(self) -> None:
        if not self._workspace_is_active():
            return
        if self._try_focused_undo():
            return
        stack = self.main.undo_group.activeStack()
        if stack is not None and stack.canUndo():
            self.main.undo_group.undo()

    def _redo(self) -> None:
        if not self._workspace_is_active():
            return
        if self._try_focused_redo():
            return
        stack = self.main.undo_group.activeStack()
        if stack is not None and stack.canRedo():
            self.main.undo_group.redo()

    def _save_project(self) -> None:
        if not self._workspace_is_active():
            return
        try:
            self.main.text_ctrl._commit_pending_text_command()
        except Exception:
            pass
        self.main.project_ctrl.thread_save_project()

    def _delete_selected_box(self) -> None:
        if not self._workspace_is_active() or self._is_text_input_focused():
            return
        self.main.delete_selected_box()

    def _restore_text_blocks(self) -> None:
        if not self._workspace_is_active() or self._is_text_input_focused():
            return
        self.main.restore_text_blocks()
