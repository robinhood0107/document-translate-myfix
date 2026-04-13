from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from app.ui.list_view import PageListItemData, PageListView


class PageListRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.widget = PageListView()
        self.widget.set_page_items(
            [
                PageListItemData("c.png", "c.png", "c.png"),
                PageListItemData("a.png", "a.png", "a.png"),
                PageListItemData("b.png", "b.png", "b.png"),
            ]
        )
        self.addCleanup(self.widget.deleteLater)

    def test_restore_selection_returns_selected_paths(self) -> None:
        self.widget.restore_selection(["a.png", "b.png"], "b.png")
        QtWidgets.QApplication.processEvents()

        self.assertEqual(self.widget.currentRow(), 2)
        self.assertEqual(set(self.widget.selected_file_paths()), {"a.png", "b.png"})

    def test_model_reorder_paths_updates_order(self) -> None:
        moved = self.widget.model().reorder_paths(["a.png"], 0)
        self.assertTrue(moved)
        self.assertEqual(self.widget.model().file_paths(), ["a.png", "c.png", "b.png"])

    def test_sort_buttons_toggle_directions(self) -> None:
        requested: list[tuple[str, str]] = []
        self.widget.sort_requested.connect(lambda key, direction: requested.append((key, direction)))

        self.widget._name_sort_button.click()
        self.widget._name_sort_button.click()
        self.widget._date_sort_button.click()
        self.widget._date_sort_button.click()

        self.assertEqual(
            requested,
            [
                ("name", "asc"),
                ("name", "desc"),
                ("date", "desc"),
                ("date", "oldest"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
