from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

from app.ui.canvas.text_item import TextBlockItem
from modules.utils.render_style_policy import (
    VERTICAL_ALIGNMENT_BOTTOM,
    VERTICAL_ALIGNMENT_CENTER,
    VERTICAL_ALIGNMENT_TOP,
    compute_vertical_aligned_y,
)


class TextAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_vertical_alignment_uses_source_rect_height(self) -> None:
        self.assertEqual(compute_vertical_aligned_y(10, 100, 40, VERTICAL_ALIGNMENT_TOP), 10)
        self.assertEqual(compute_vertical_aligned_y(10, 100, 40, VERTICAL_ALIGNMENT_CENTER), 40)
        self.assertEqual(compute_vertical_aligned_y(10, 100, 40, VERTICAL_ALIGNMENT_BOTTOM), 70)

    def test_text_item_repositions_for_vertical_alignment(self) -> None:
        item = TextBlockItem(
            "hello",
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
            source_rect=(20, 30, 160, 180),
        )
        item.set_text("hello", 80)
        item.set_vertical_alignment(VERTICAL_ALIGNMENT_BOTTOM)
        bottom_y = item.pos().y()

        item.set_vertical_alignment(VERTICAL_ALIGNMENT_TOP)
        top_y = item.pos().y()

        self.assertGreater(bottom_y, top_y)
        self.assertEqual(item.pos().x(), 20)

    def test_horizontal_alignment_is_stored_in_document_blocks(self) -> None:
        item = TextBlockItem(
            "hello",
            alignment=QtCore.Qt.AlignmentFlag.AlignRight,
            source_rect=(0, 0, 120, 60),
        )
        item.set_text("hello", 120)

        self.assertEqual(
            item.document().firstBlock().blockFormat().alignment(),
            QtCore.Qt.AlignmentFlag.AlignRight,
        )


if __name__ == "__main__":
    unittest.main()
