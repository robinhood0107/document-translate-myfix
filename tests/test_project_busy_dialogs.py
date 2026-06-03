from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

from app.ui.messages import Messages


class ProjectBusyDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_busy_dialog_is_window_modal_progress_without_cancel(self) -> None:
        parent = QtWidgets.QWidget()
        self.addCleanup(parent.deleteLater)

        dialog = Messages.show_busy(parent, "Creating project file...", title="Project File")
        self.addCleanup(Messages.close_busy, dialog)

        self.assertEqual(dialog.windowTitle(), "Project File")
        self.assertEqual(dialog.labelText(), "Creating project file...")
        self.assertEqual(dialog.minimum(), 0)
        self.assertEqual(dialog.maximum(), 0)
        self.assertEqual(dialog.windowModality(), QtCore.Qt.WindowModality.ApplicationModal)

    def test_busy_dialog_can_force_close_immediately(self) -> None:
        parent = QtWidgets.QWidget()
        self.addCleanup(parent.deleteLater)

        dialog = Messages.show_busy(parent, "Scanning series folder...", title="Create Series Project")
        self.assertTrue(dialog.isVisible())

        Messages.close_busy(dialog, force=True)
        QtWidgets.QApplication.processEvents()

        self.assertFalse(dialog.isVisible())


if __name__ == "__main__":
    unittest.main()
