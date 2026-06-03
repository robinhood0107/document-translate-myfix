from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtGui, QtWidgets

from app.controllers.image import ImageStateController
from app.ui.canvas.image_viewer import ImageViewer


class _DummyCombo:
    def __init__(self, value: str) -> None:
        self.value = value
        self.blocked: list[bool] = []

    def blockSignals(self, blocked: bool) -> None:
        self.blocked.append(bool(blocked))

    def currentText(self) -> str:
        return self.value

    def setCurrentText(self, value: str) -> None:
        self.value = value


class _DummyViewport:
    def update(self) -> None:
        pass


class _DummyViewer:
    def __init__(self) -> None:
        self.loaded_states: list[dict] = []
        self.loaded_strokes: list[list] = []
        self.text_items: list = []
        self._viewport = _DummyViewport()

    def setUpdatesEnabled(self, _enabled: bool) -> None:
        pass

    def display_image_array(self, _rgb_image, fit: bool = False) -> None:
        self.fit = fit

    def load_state(self, state: dict) -> None:
        self.loaded_states.append(state)

    def load_brush_strokes(self, strokes: list) -> None:
        self.loaded_strokes.append(strokes)

    def viewport(self) -> _DummyViewport:
        return self._viewport


class ViewerStateRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.parent = QtWidgets.QWidget()
        self.viewer = ImageViewer(self.parent)
        self.viewer.display_image_array(np.zeros((20, 20, 3), dtype=np.uint8), fit=False)
        self.addCleanup(self.viewer.deleteLater)
        self.addCleanup(self.parent.deleteLater)

    def test_image_viewer_load_state_tolerates_empty_state(self) -> None:
        self.viewer.load_state({})
        self.viewer.load_state({"push_to_stack": False})
        self.viewer.load_state({"text_items_state": []})

    def test_image_viewer_load_state_preserves_transform_when_invalid(self) -> None:
        expected = QtGui.QTransform()
        expected.scale(2.0, 2.0)
        self.viewer.setTransform(expected)

        self.viewer.load_state({"transform": [1.0, 2.0], "rectangles": []})

        self.assertEqual(self.viewer.transform().m11(), expected.m11())
        self.assertEqual(self.viewer.transform().m22(), expected.m22())

    def test_image_viewer_load_state_skips_malformed_rectangles(self) -> None:
        self.viewer.load_state(
            {
                "rectangles": [
                    {},
                    {"rect": [1, 2, 3]},
                    {"rect": [1, 2, 10, 12]},
                ]
            }
        )

        self.assertEqual(len(self.viewer.rectangles), 1)

    def test_image_controller_load_image_state_handles_partial_viewer_state(self) -> None:
        file_path = "page.png"
        main = mock.Mock()
        main.image_data = {file_path: np.zeros((8, 8, 3), dtype=np.uint8)}
        main.image_states = {
            file_path: {
                "viewer_state": {"push_to_stack": False},
                "source_lang": "English",
                "target_lang": "Korean",
            }
        }
        main.image_viewer = _DummyViewer()
        main.s_combo = _DummyCombo("Japanese")
        main.t_combo = _DummyCombo("English")
        main.text_ctrl = mock.Mock()

        controller = ImageStateController.__new__(ImageStateController)
        controller.main = main
        controller.load_patch_state = mock.Mock()

        controller.load_image_state(file_path)

        self.assertEqual(main.blk_list, [])
        self.assertEqual(main.s_combo.currentText(), "English")
        self.assertEqual(main.t_combo.currentText(), "Korean")
        self.assertEqual(main.image_viewer.loaded_states, [{"push_to_stack": False}])
        self.assertEqual(main.image_viewer.loaded_strokes, [[]])
        controller.load_patch_state.assert_called_once_with(file_path)
        main.text_ctrl.clear_text_edits.assert_called_once()


if __name__ == "__main__":
    unittest.main()
