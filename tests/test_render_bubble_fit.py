from __future__ import annotations

import os
import unittest

import numpy as np
from PySide6 import QtCore, QtWidgets

from modules.rendering.render import (
    build_render_rects_for_block,
    build_text_item_layout_geometry,
    get_best_render_area,
    get_render_fit_clearance_for_block,
    pyside_word_wrap,
)
from modules.utils.textblock import TextBlock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _block(*, xyxy, text_class="text_bubble", bubble_xyxy=None) -> TextBlock:
    return TextBlock(
        text_bbox=np.asarray(xyxy, dtype=np.int32),
        bubble_bbox=np.asarray(bubble_xyxy, dtype=np.int32) if bubble_xyxy is not None else None,
        text_class=text_class,
        text="demo",
        translation="demo",
        source_lang="ko",
    )


class RenderBubbleFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_detected_bubble_area_is_used_for_safe_text_bubble(self) -> None:
        image = np.zeros((2400, 1700, 3), dtype=np.uint8)
        block = _block(
            xyxy=[1037, 80, 1557, 308],
            bubble_xyxy=[919, 17, 1693, 366],
        )

        get_best_render_area([block], image)

        self.assertEqual(block._render_area_source, "detected_bubble")
        self.assertNotEqual(block.xyxy.tolist(), [1037, 80, 1557, 308])
        source_rect, block_anchor = build_render_rects_for_block(block)
        self.assertEqual(block_anchor, (1037.0, 80.0, 520.0, 228.0))
        self.assertGreater(source_rect[2], 520.0)
        self.assertGreater(source_rect[3], 228.0)

    def test_repeated_bubble_fit_preserves_original_anchor(self) -> None:
        image = np.zeros((2400, 1700, 3), dtype=np.uint8)
        block = _block(
            xyxy=[1037, 80, 1557, 308],
            bubble_xyxy=[919, 17, 1693, 366],
        )

        get_best_render_area([block], image)
        first_source, first_anchor = build_render_rects_for_block(block)
        get_best_render_area([block], image)
        second_source, second_anchor = build_render_rects_for_block(block)

        self.assertEqual(first_anchor, (1037.0, 80.0, 520.0, 228.0))
        self.assertEqual(second_anchor, first_anchor)
        self.assertEqual(second_source, first_source)

    def test_free_text_and_invalid_bubbles_keep_original_bbox(self) -> None:
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        free = _block(
            xyxy=[100, 100, 180, 160],
            text_class="text_free",
            bubble_xyxy=[50, 50, 250, 220],
        )
        mismatch = _block(
            xyxy=[400, 400, 450, 450],
            bubble_xyxy=[50, 50, 250, 220],
        )

        get_best_render_area([free, mismatch], image)

        self.assertEqual(free._render_area_source, "text_bbox")
        self.assertEqual(free.xyxy.tolist(), [100, 100, 180, 160])
        self.assertEqual(mismatch._render_area_source, "text_bbox")
        self.assertEqual(mismatch.xyxy.tolist(), [400, 400, 450, 450])

    def test_korean_wrap_uses_qt_metrics_and_respects_max_cap(self) -> None:
        text = (
            "세상에... 나, 난 정말 이렇게 빨리 마주칠 줄 몰랐어! "
            "별일 없다면, 내 눈앞의 이 꼬마애가 「조직」이 계속 쫓던 "
            "배신자이자, 마가 입에 달고 살던 「보스」야!"
        )

        wrapped_30, font_30, width_30, height_30 = pyside_word_wrap(
            text,
            "Ownglyph gumama3",
            635,
            286,
            1.0,
            3.0,
            False,
            False,
            False,
            QtCore.Qt.AlignmentFlag.AlignCenter,
            QtCore.Qt.LayoutDirection.LeftToRight,
            30,
            5,
            False,
            return_metrics=True,
        )
        wrapped_60, font_60, width_60, height_60 = pyside_word_wrap(
            text,
            "Ownglyph gumama3",
            635,
            286,
            1.0,
            3.0,
            False,
            False,
            False,
            QtCore.Qt.AlignmentFlag.AlignCenter,
            QtCore.Qt.LayoutDirection.LeftToRight,
            60,
            5,
            False,
            return_metrics=True,
        )

        self.assertLessEqual(font_30, 30)
        self.assertLessEqual(width_30, 635)
        self.assertLessEqual(height_30, 286)
        self.assertIn("\n", wrapped_30)
        self.assertFalse(any(line[:1] in ".,!?)]}」』" for line in wrapped_30.splitlines()))
        self.assertLessEqual(font_60, 60)
        self.assertLessEqual(width_60, 635)
        self.assertLessEqual(height_60, 286)
        self.assertGreaterEqual(font_60, font_30)

    def test_text_item_geometry_keeps_layout_width_for_center_alignment(self) -> None:
        position, width, height = build_text_item_layout_geometry(
            (100, 50, 600, 300),
            rendered_height=180,
            vertical_alignment="center",
        )

        self.assertEqual(position, (100.0, 110.0))
        self.assertEqual(width, 600.0)
        self.assertEqual(height, 180)

    def test_detected_bubble_fit_clearance_reduces_border_touch_risk(self) -> None:
        image = np.zeros((2400, 1700, 3), dtype=np.uint8)
        block = _block(
            xyxy=[1037, 80, 1557, 308],
            bubble_xyxy=[919, 17, 1693, 366],
        )
        get_best_render_area([block], image)
        source_rect, _anchor = build_render_rects_for_block(block)
        clearance = get_render_fit_clearance_for_block(block, 3.0)

        text = (
            "세상에... 나, 난 정말 이렇게 빨리 마주칠 줄 몰랐어! "
            "별일 없다면, 내 눈앞의 이 꼬마애가 「조직」이 계속 쫓던 "
            "배신자이자, 마가 입에 달고 살던 「보스」야!"
        )
        _wrapped_open, font_open, _width_open, height_open = pyside_word_wrap(
            text,
            "Ownglyph gumama3",
            int(source_rect[2]),
            int(source_rect[3]),
            1.0,
            3.0,
            False,
            False,
            False,
            QtCore.Qt.AlignmentFlag.AlignCenter,
            QtCore.Qt.LayoutDirection.LeftToRight,
            60,
            5,
            False,
            return_metrics=True,
        )
        wrapped_safe, font_safe, width_safe, height_safe = pyside_word_wrap(
            text,
            "Ownglyph gumama3",
            int(source_rect[2]),
            int(source_rect[3]),
            1.0,
            3.0,
            False,
            False,
            False,
            QtCore.Qt.AlignmentFlag.AlignCenter,
            QtCore.Qt.LayoutDirection.LeftToRight,
            60,
            5,
            False,
            fit_clearance=clearance,
            return_metrics=True,
        )

        self.assertGreater(clearance, 0.0)
        self.assertLess(font_safe, font_open)
        self.assertLess(height_safe, height_open)
        self.assertLessEqual(width_safe, source_rect[2] - (clearance * 2))
        self.assertLessEqual(height_safe, source_rect[3] - (clearance * 2))
        self.assertIn("\n", wrapped_safe)

    def test_free_text_has_no_detected_bubble_fit_clearance(self) -> None:
        free = _block(
            xyxy=[100, 100, 180, 160],
            text_class="text_free",
            bubble_xyxy=[50, 50, 250, 220],
        )

        self.assertEqual(get_render_fit_clearance_for_block(free, 3.0), 0.0)
