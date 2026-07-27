from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets
from PySide6.QtGui import QUndoCommand, QUndoGroup, QUndoStack

from controller import ComicTranslate


class _Button:
    def __init__(self) -> None:
        self.enabled_values: list[bool] = []

    def setEnabled(self, value: bool) -> None:
        self.enabled_values.append(bool(value))


class _ButtonGroup:
    def __init__(self) -> None:
        self._buttons = [_Button(), _Button()]

    def buttons(self):
        return self._buttons


class _UndoToolGroup:
    def __init__(self) -> None:
        self.button_group = _ButtonGroup()

    def get_button_group(self):
        return self.button_group


class UndoStackShutdownTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _controller(self):
        controller = ComicTranslate.__new__(ComicTranslate)
        controller.undo_group = QUndoGroup()
        controller.undo_stacks = {}
        controller.undo_tool_group = _UndoToolGroup()
        controller._is_shutting_down = False
        controller._dirty_revision = 0
        controller.project_ctrl = SimpleNamespace(
            notify_project_dirty_revision_changed=lambda: None
        )
        controller.image_files = ["page-a.png"]
        controller.image_states = {"page-a.png": {}}
        controller.image_ctrl = SimpleNamespace(
            ensure_page_state=mock.Mock(return_value={})
        )
        controller.refresh_render_dirty_ui = mock.Mock()
        controller._current_image_path = lambda: "page-a.png"
        controller._update_window_modified = mock.Mock()
        return controller

    def test_clear_undo_stacks_disconnects_late_stack_signals(self) -> None:
        controller = self._controller()
        controller.mark_render_dirty = mock.Mock()
        stack = QUndoStack()

        ComicTranslate.register_undo_stack_for_path(controller, "page-a.png", stack)
        controller.mark_render_dirty.reset_mock()
        controller._dirty_revision = 0

        ComicTranslate.clear_undo_stacks(controller)
        stack.push(QUndoCommand("late change after clear"))

        self.assertEqual(controller.undo_stacks, {})
        controller.mark_render_dirty.assert_not_called()
        self.assertEqual(controller._dirty_revision, 0)

    def test_shutdown_guard_ignores_late_undo_callbacks(self) -> None:
        controller = self._controller()
        controller._is_shutting_down = True
        controller.image_ctrl.ensure_page_state.side_effect = AssertionError(
            "shutdown guard should skip page-state access"
        )

        ComicTranslate.update_undo_redo_actions(controller)
        ComicTranslate.mark_render_dirty(controller, "page-a.png")
        ComicTranslate.mark_render_dirty_for_paths(controller, ["page-a.png"])

        controller.image_ctrl.ensure_page_state.assert_not_called()

    def test_shutdown_stops_both_local_runtime_managers(self) -> None:
        controller = self._controller()
        controller.batch_report_ctrl = SimpleNamespace(shutdown=mock.Mock())
        controller.cancel_current_task = mock.Mock()
        controller.threadpool = SimpleNamespace(clear=mock.Mock(), waitForDone=mock.Mock())
        controller.clear_undo_stacks = mock.Mock()
        controller.settings_page = SimpleNamespace(shutdown=mock.Mock())
        controller.pipeline_status_panel = SimpleNamespace(hide=mock.Mock())
        controller.set_pipeline_overlay_active = mock.Mock()
        controller.local_ocr_runtime_manager = SimpleNamespace(shutdown=mock.Mock())
        controller.local_translation_runtime_manager = SimpleNamespace(shutdown=mock.Mock())

        ComicTranslate.shutdown(controller)

        controller.local_ocr_runtime_manager.shutdown.assert_called_once_with()
        controller.local_translation_runtime_manager.shutdown.assert_called_once_with()

    def test_shutdown_still_stops_gemma_when_ocr_shutdown_fails(self) -> None:
        controller = self._controller()
        controller.batch_report_ctrl = SimpleNamespace(shutdown=mock.Mock())
        controller.cancel_current_task = mock.Mock()
        controller.threadpool = SimpleNamespace(clear=mock.Mock(), waitForDone=mock.Mock())
        controller.clear_undo_stacks = mock.Mock()
        controller.settings_page = SimpleNamespace(shutdown=mock.Mock())
        controller.pipeline_status_panel = SimpleNamespace(hide=mock.Mock())
        controller.set_pipeline_overlay_active = mock.Mock()
        controller.local_ocr_runtime_manager = SimpleNamespace(
            shutdown=mock.Mock(side_effect=RuntimeError("stop failed"))
        )
        controller.local_translation_runtime_manager = SimpleNamespace(shutdown=mock.Mock())

        ComicTranslate.shutdown(controller)

        controller.local_translation_runtime_manager.shutdown.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
