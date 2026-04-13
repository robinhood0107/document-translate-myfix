from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

from app.controllers.projects import ProjectController
from app.ui.main_window.window import ComicTranslateUI


class ProjectNavigationRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_show_home_returns_to_startup_home_after_workspace_is_initialized(self) -> None:
        window = ComicTranslateUI()
        self.addCleanup(window.deleteLater)

        window.show_main_page()
        self.assertIs(window._center_stack.currentWidget(), window.main_content_widget)
        self.assertFalse(window.home_nav_button.isChecked())
        self.assertFalse(window.settings_nav_button.isChecked())

        window.show_home()

        self.assertIs(window._center_stack.currentWidget(), window.startup_home)
        self.assertTrue(window.home_nav_button.isChecked())
        self.assertFalse(window.settings_nav_button.isChecked())

    def test_show_settings_marks_only_settings_button_checked(self) -> None:
        window = ComicTranslateUI()
        self.addCleanup(window.deleteLater)

        window.show_settings_page()

        self.assertIs(window._center_stack.currentWidget(), window.settings_page)
        self.assertFalse(window.home_nav_button.isChecked())
        self.assertTrue(window.settings_nav_button.isChecked())


class RecentProjectOrderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
        QtCore.QSettings.setPath(
            QtCore.QSettings.Format.IniFormat,
            QtCore.QSettings.Scope.UserScope,
            self._temp_dir.name,
        )
        settings = QtCore.QSettings("ComicLabs", "ComicTranslate")
        settings.clear()
        settings.sync()

    def tearDown(self) -> None:
        settings = QtCore.QSettings("ComicLabs", "ComicTranslate")
        settings.clear()
        settings.sync()
        self._temp_dir.cleanup()

    def test_recent_projects_are_ordered_by_open_recency_not_file_mtime(self) -> None:
        older = os.path.join(self._temp_dir.name, "older.ctpr")
        newer = os.path.join(self._temp_dir.name, "newer.ctpr")
        for path in (older, newer):
            with open(path, "wb") as handle:
                handle.write(b"test")
        os.utime(older, (200.0, 200.0))
        os.utime(newer, (100.0, 100.0))

        controller = ProjectController.__new__(ProjectController)
        controller.add_recent_project(older)
        controller.add_recent_project(newer)

        entries = controller.get_recent_projects()

        self.assertEqual([entry["path"] for entry in entries], [newer, older])
        self.assertGreater(entries[0]["opened_at"], entries[1]["opened_at"])
        self.assertLess(entries[0]["mtime"], entries[1]["mtime"])

    def test_missing_opened_at_preserves_existing_settings_order(self) -> None:
        first = os.path.join(self._temp_dir.name, "first.ctpr")
        second = os.path.join(self._temp_dir.name, "second.ctpr")
        for path in (first, second):
            with open(path, "wb") as handle:
                handle.write(b"test")

        settings = QtCore.QSettings("ComicLabs", "ComicTranslate")
        settings.beginGroup("recent_projects")
        settings.setValue("paths", [first, second])
        settings.setValue("mtimes", [1.0, 2.0])
        settings.setValue("pinned", [False, False])
        settings.endGroup()
        settings.sync()

        controller = ProjectController.__new__(ProjectController)
        entries = controller.get_recent_projects()

        self.assertEqual([entry["path"] for entry in entries], [first, second])


if __name__ == "__main__":
    unittest.main()
