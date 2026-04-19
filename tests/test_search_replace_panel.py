from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from app.ui.search_replace_panel import SearchReplacePanel


class SearchReplacePanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.panel = SearchReplacePanel()
        self.addCleanup(self.panel.deleteLater)

    def test_clear_find_resets_without_error_status(self) -> None:
        self.panel.find_input.setPlainText("hello")
        QtWidgets.QApplication.processEvents()

        self.panel._clear_find()
        QtWidgets.QApplication.processEvents()

        self.assertEqual(self.panel.find_input.text(), "")
        self.assertEqual(self.panel.summary_label.text(), "0 results")
        self.assertEqual(self.panel.status_label.text(), "Ready")
        self.assertFalse(self.panel._live_timer.isActive())

    def test_blank_query_does_not_schedule_live_search(self) -> None:
        self.panel.find_input.setPlainText("")
        QtWidgets.QApplication.processEvents()

        self.panel._schedule_live_search()

        self.assertEqual(self.panel.summary_label.text(), "0 results")
        self.assertEqual(self.panel.status_label.text(), "Ready")
        self.assertFalse(self.panel._live_timer.isActive())

    def test_hide_event_stops_live_timer(self) -> None:
        self.panel.show()
        QtWidgets.QApplication.processEvents()
        self.panel.find_input.setPlainText("hello")
        QtWidgets.QApplication.processEvents()

        self.panel._schedule_live_search()
        self.assertTrue(self.panel._live_timer.isActive())

        self.panel.hide()
        QtWidgets.QApplication.processEvents()

        self.assertFalse(self.panel._live_timer.isActive())


if __name__ == "__main__":
    unittest.main()
