from __future__ import annotations

import os
import unittest

import numpy as np
from PySide6 import QtCore, QtWidgets

import modules.rendering.render as render_module
from modules.rendering.render import (
    AUTO_MAX_FONT_PROFILE_CURRENT,
    AUTO_MAX_FONT_PROFILE_STRONG,
    DETECTED_BUBBLE_DYNAMIC_FONT_CAP,
    build_render_rects_for_block,
    build_text_item_layout_geometry,
    get_dynamic_bubble_font_cap,
    build_duplicate_bubble_render_key,
    describe_text_free_render_translation_gate,
    get_best_render_area,
    get_render_fit_clearance_for_block,
    resolve_text_free_manga_layout,
    pyside_word_wrap,
    refit_detected_bubble_text_if_underfilled,
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

    def test_render_rect_prefers_detected_bubble_area_after_later_bbox_adjustment(self) -> None:
        block = _block(
            xyxy=[1037, 80, 1557, 308],
            bubble_xyxy=[919, 17, 1693, 366],
        )
        block._render_original_xyxy = [1037, 80, 1557, 308]
        block._render_area_source = "detected_bubble"
        block._render_area_xyxy = [980, 70, 1620, 340]
        block.xyxy[:] = [1042, 85, 1552, 303]

        source_rect, block_anchor = build_render_rects_for_block(block)

        self.assertEqual(source_rect, (980.0, 70.0, 640.0, 270.0))
        self.assertEqual(block_anchor, (1037.0, 80.0, 520.0, 228.0))

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

    def test_overlapping_detected_bubble_areas_keep_original_bboxes(self) -> None:
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        top = _block(
            xyxy=[140, 120, 220, 165],
            bubble_xyxy=[80, 80, 320, 260],
        )
        bottom = _block(
            xyxy=[150, 180, 230, 225],
            bubble_xyxy=[80, 80, 320, 260],
        )

        get_best_render_area([top, bottom], image)

        self.assertEqual(top._render_area_source, "text_bbox")
        self.assertEqual(bottom._render_area_source, "text_bbox")
        self.assertEqual(top.xyxy.tolist(), [140, 120, 220, 165])
        self.assertEqual(bottom.xyxy.tolist(), [150, 180, 230, 225])

    def test_non_overlapping_detected_bubbles_can_expand_independently(self) -> None:
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        left = _block(
            xyxy=[90, 100, 150, 145],
            bubble_xyxy=[40, 60, 210, 210],
        )
        right = _block(
            xyxy=[320, 100, 380, 145],
            bubble_xyxy=[270, 60, 460, 210],
        )

        get_best_render_area([left, right], image)

        self.assertEqual(left._render_area_source, "detected_bubble")
        self.assertEqual(right._render_area_source, "detected_bubble")

    def test_detected_bubble_area_does_not_cover_other_text_bbox(self) -> None:
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        bubble = _block(
            xyxy=[120, 120, 180, 165],
            bubble_xyxy=[80, 80, 340, 260],
        )
        nearby_caption = _block(
            xyxy=[265, 120, 335, 165],
            text_class="text_free",
            bubble_xyxy=None,
        )

        get_best_render_area([bubble, nearby_caption], image)

        self.assertEqual(bubble._render_area_source, "text_bbox")
        self.assertEqual(bubble.xyxy.tolist(), [120, 120, 180, 165])
        self.assertEqual(nearby_caption._render_area_source, "text_bbox")

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

    def test_vertical_japanese_text_free_uses_centered_manga_paragraph_width(self) -> None:
        block = TextBlock(
            text_bbox=np.asarray([100, 100, 700, 420], dtype=np.int32),
            text_class="text_free",
            text="demo",
            translation="한국어 문장이 너무 길게 한 줄로 뭉치면 안 된다",
            source_lang="ja",
            direction="vertical",
        )

        policy = resolve_text_free_manga_layout(
            block,
            (100.0, 100.0, 600.0, 320.0),
            target_lang_code="ko",
        )

        self.assertTrue(policy.enabled)
        self.assertEqual(policy.vertical_alignment, "center")
        self.assertEqual(policy.alignment, QtCore.Qt.AlignmentFlag.AlignCenter)
        self.assertLess(policy.wrap_width, 600)
        self.assertGreaterEqual(policy.item_width, 600)
        self.assertIn("render_centered_layout", policy.reasons)

    def test_short_text_free_overexpanded_translation_is_reviewed_not_rendered(self) -> None:
        block = TextBlock(
            text_bbox=np.asarray([4, 2133, 81, 2847], dtype=np.int32),
            text_class="text_free",
            text="卷之十",
            translation=(
                "흠정사고전서 권12, 8, 9, 10, 11, 12, 13, 14, 15, "
                "16, 17, 18, 19, 20, 22, 23, 24"
            ),
            source_lang="ja",
            direction="vertical",
        )

        decision = describe_text_free_render_translation_gate(
            block,
            block.translation,
            target_lang_code="ko",
        )

        self.assertFalse(decision.render)
        self.assertEqual(decision.status, "needs_review_text_free_translation")
        self.assertIn("text_free_translation_overexpanded", decision.reasons)

    def test_normal_text_free_translation_still_renders(self) -> None:
        block = TextBlock(
            text_bbox=np.asarray([121, 99, 210, 517], dtype=np.int32),
            text_class="text_free",
            text="イきかけなんだろ！？",
            translation="가기 직전이었잖아!?",
            source_lang="ja",
            direction="vertical",
        )

        decision = describe_text_free_render_translation_gate(
            block,
            block.translation,
            target_lang_code="ko",
        )

        self.assertTrue(decision.render)
        self.assertEqual(decision.status, "ok")

    def test_text_free_marker_only_translation_is_skipped_not_rendered(self) -> None:
        block = TextBlock(
            text_bbox=np.asarray([441, 268, 535, 319], dtype=np.int32),
            text_class="text_free",
            text="ーーー",
            translation="ーー",
            source_lang="ja",
            direction="horizontal",
        )

        decision = describe_text_free_render_translation_gate(
            block,
            block.translation,
            target_lang_code="ko",
        )

        self.assertFalse(decision.render)
        self.assertEqual(decision.status, "skipped_text_free_marker_only")
        self.assertIn("text_free_marker_only", decision.reasons)

    def test_embedded_ui_panel_review_status_is_not_auto_rendered(self) -> None:
        self.assertTrue(hasattr(render_module, "describe_auto_render_review_status_gate"))
        decision = render_module.describe_auto_render_review_status_gate(
            "needs_review_embedded_ui_panel_layout"
        )

        self.assertFalse(decision.render)
        self.assertEqual(decision.status, "needs_review_embedded_ui_panel_layout")
        self.assertIn("render_skipped_review_gate", decision.reasons)

    def test_text_free_layout_review_status_is_diagnostic_not_auto_skip(self) -> None:
        self.assertTrue(hasattr(render_module, "describe_auto_render_review_status_gate"))
        decision = render_module.describe_auto_render_review_status_gate(
            "needs_review_text_free_layout"
        )

        self.assertTrue(decision.render)
        self.assertEqual(decision.status, "ok")

    def test_duplicate_bubble_render_key_matches_same_bubble_same_source(self) -> None:
        first = _block(
            xyxy=[1630, 2570, 1900, 2760],
            bubble_xyxy=[1604, 2525, 1943, 2948],
            text_class="text_bubble",
        )
        second = _block(
            xyxy=[1618, 2582, 1910, 2775],
            bubble_xyxy=[1607, 2528, 1945, 2949],
            text_class="text_bubble",
        )
        first.text = "の愛アなりすまし用記憶アクセスオプションってのン自然憶レ自記よ"
        second.text = " の愛ア なりすまし用記憶アクセスオプションってのン自然憶レ自記よ "

        self.assertEqual(
            build_duplicate_bubble_render_key(first),
            build_duplicate_bubble_render_key(second),
        )

    def test_duplicate_bubble_render_key_ignores_text_free(self) -> None:
        free = _block(
            xyxy=[100, 100, 180, 160],
            bubble_xyxy=[50, 50, 250, 220],
            text_class="text_free",
        )

        self.assertIsNone(build_duplicate_bubble_render_key(free))

    def test_duplicate_same_bubble_source_does_not_block_detected_bubble_area(self) -> None:
        image = np.zeros((3035, 2150, 3), dtype=np.uint8)
        first = _block(
            xyxy=[1606, 2547, 1681, 2727],
            bubble_xyxy=[1604, 2525, 1943, 2948],
            text_class="text_bubble",
        )
        second = _block(
            xyxy=[1697, 2552, 1916, 2917],
            bubble_xyxy=[1604, 2525, 1943, 2948],
            text_class="text_bubble",
        )
        first.text = "の愛アなりすまし用記憶アクセスオプションってのン自然憶レ自記よ"
        second.text = first.text

        get_best_render_area([first, second], image)

        self.assertEqual(first._render_area_source, "detected_bubble")
        source_rect, block_anchor = build_render_rects_for_block(first)
        self.assertEqual(block_anchor, (1606.0, 2547.0, 75.0, 180.0))
        self.assertGreater(source_rect[2], 75.0)
        self.assertGreater(source_rect[3], 180.0)

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

    def test_detected_bubble_fit_profile_changes_clearance(self) -> None:
        image = np.zeros((800, 800, 3), dtype=np.uint8)
        block = _block(
            xyxy=[260, 250, 340, 300],
            bubble_xyxy=[120, 100, 520, 460],
        )
        get_best_render_area([block], image)

        self.assertEqual(
            get_render_fit_clearance_for_block(
                block,
                0.0,
                auto_max_font_profile=AUTO_MAX_FONT_PROFILE_CURRENT,
            ),
            8.0,
        )
        self.assertEqual(
            get_render_fit_clearance_for_block(
                block,
                0.0,
                auto_max_font_profile=AUTO_MAX_FONT_PROFILE_STRONG,
            ),
            7.0,
        )

    def test_strong_profile_expands_bubble_area_more_than_current(self) -> None:
        image = np.zeros((800, 800, 3), dtype=np.uint8)
        current = _block(
            xyxy=[260, 250, 340, 300],
            bubble_xyxy=[120, 100, 520, 460],
        )
        strong = _block(
            xyxy=[260, 250, 340, 300],
            bubble_xyxy=[120, 100, 520, 460],
        )

        get_best_render_area(
            [current],
            image,
            auto_max_font_profile=AUTO_MAX_FONT_PROFILE_CURRENT,
        )
        get_best_render_area(
            [strong],
            image,
            auto_max_font_profile=AUTO_MAX_FONT_PROFILE_STRONG,
        )
        current_rect, _ = build_render_rects_for_block(current)
        strong_rect, _ = build_render_rects_for_block(strong)

        self.assertGreater(strong_rect[2], current_rect[2])
        self.assertGreater(strong_rect[3], current_rect[3])

    def test_dynamic_bubble_font_cap_expands_underfilled_detected_bubble(self) -> None:
        image = np.zeros((800, 800, 3), dtype=np.uint8)
        block = _block(
            xyxy=[260, 250, 340, 300],
            bubble_xyxy=[120, 100, 520, 460],
        )
        get_best_render_area([block], image)

        cap = get_dynamic_bubble_font_cap(
            block,
            configured_max_font_size=60,
            rendered_width=120,
            rendered_height=80,
            vertical=False,
        )

        self.assertGreater(cap, 60)
        self.assertLessEqual(cap, DETECTED_BUBBLE_DYNAMIC_FONT_CAP)

    def test_dynamic_bubble_font_cap_uses_wide_bubble_for_short_underfilled_text(self) -> None:
        block = _block(
            xyxy=[260, 150, 340, 210],
            bubble_xyxy=[50, 100, 750, 280],
        )
        block._render_area_source = "detected_bubble"
        block._render_area_xyxy = [50, 100, 750, 280]

        cap = get_dynamic_bubble_font_cap(
            block,
            configured_max_font_size=40,
            rendered_width=60,
            rendered_height=50,
            vertical=False,
            final_font_size=40,
        )

        self.assertGreaterEqual(cap, 110)
        self.assertLessEqual(cap, DETECTED_BUBBLE_DYNAMIC_FONT_CAP)

    def test_strong_profile_uses_larger_dynamic_font_cap(self) -> None:
        block = _block(
            xyxy=[260, 150, 340, 210],
            bubble_xyxy=[50, 100, 750, 280],
        )
        block._render_area_source = "detected_bubble"
        block._render_area_xyxy = [50, 100, 750, 280]

        current_cap = get_dynamic_bubble_font_cap(
            block,
            configured_max_font_size=40,
            rendered_width=60,
            rendered_height=50,
            vertical=False,
            final_font_size=40,
            auto_max_font_profile=AUTO_MAX_FONT_PROFILE_CURRENT,
        )
        strong_cap = get_dynamic_bubble_font_cap(
            block,
            configured_max_font_size=40,
            rendered_width=60,
            rendered_height=50,
            vertical=False,
            final_font_size=40,
            auto_max_font_profile=AUTO_MAX_FONT_PROFILE_STRONG,
        )

        self.assertEqual(current_cap, 160)
        self.assertEqual(strong_cap, 190)
        self.assertGreater(strong_cap, current_cap)

    def test_auto_max_detected_bubble_still_expands_when_initial_text_is_not_underfill(self) -> None:
        block = _block(
            xyxy=[260, 150, 340, 210],
            bubble_xyxy=[50, 100, 650, 380],
        )
        block._render_area_source = "detected_bubble"
        block._render_area_xyxy = [50, 100, 650, 380]

        cap = get_dynamic_bubble_font_cap(
            block,
            configured_max_font_size=40,
            rendered_width=260,
            rendered_height=160,
            vertical=False,
            final_font_size=40,
        )

        self.assertGreater(cap, 40)

    def test_dynamic_bubble_font_cap_does_not_change_text_free_or_vertical_blocks(self) -> None:
        free = _block(
            xyxy=[100, 100, 180, 160],
            text_class="text_free",
            bubble_xyxy=[50, 50, 250, 220],
        )
        free._render_area_source = "detected_bubble"
        free._render_area_xyxy = [50, 50, 250, 220]

        vertical = _block(
            xyxy=[100, 100, 180, 160],
            bubble_xyxy=[50, 50, 250, 220],
        )
        vertical._render_area_source = "detected_bubble"
        vertical._render_area_xyxy = [50, 50, 250, 220]

        self.assertEqual(
            get_dynamic_bubble_font_cap(
                free,
                configured_max_font_size=60,
                rendered_width=50,
                rendered_height=40,
                vertical=False,
            ),
            60,
        )
        self.assertEqual(
            get_dynamic_bubble_font_cap(
                vertical,
                configured_max_font_size=60,
                rendered_width=50,
                rendered_height=40,
                vertical=True,
            ),
            60,
        )

    def test_refit_detected_bubble_text_increases_font_when_cap_was_limiting(self) -> None:
        image = np.zeros((800, 800, 3), dtype=np.uint8)
        block = _block(
            xyxy=[260, 250, 340, 300],
            bubble_xyxy=[120, 100, 520, 460],
        )
        get_best_render_area([block], image)
        source_rect, _anchor = build_render_rects_for_block(block)
        text = "괜찮아"
        clearance = get_render_fit_clearance_for_block(block, 2.0)
        wrapped, font_size, rendered_width, rendered_height = pyside_word_wrap(
            text,
            "Ownglyph gumama3",
            int(source_rect[2]),
            int(source_rect[3]),
            1.0,
            2.0,
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

        refit_text, refit_size, refit_width, refit_height = refit_detected_bubble_text_if_underfilled(
            block,
            text,
            "Ownglyph gumama3",
            int(source_rect[2]),
            int(source_rect[3]),
            1.0,
            2.0,
            False,
            False,
            False,
            QtCore.Qt.AlignmentFlag.AlignCenter,
            QtCore.Qt.LayoutDirection.LeftToRight,
            60,
            5,
            False,
            clearance,
            wrapped,
            font_size,
            rendered_width,
            rendered_height,
        )

        self.assertEqual(refit_text.replace("\n", ""), text)
        self.assertGreater(refit_size, font_size)
        self.assertLessEqual(refit_width, source_rect[2])
        self.assertLessEqual(refit_height, source_rect[3])

    def test_refit_detected_bubble_text_can_disable_auto_font_cap(self) -> None:
        image = np.zeros((800, 800, 3), dtype=np.uint8)
        block = _block(
            xyxy=[260, 250, 340, 300],
            bubble_xyxy=[120, 100, 520, 460],
        )
        get_best_render_area([block], image)
        source_rect, _anchor = build_render_rects_for_block(block)
        text = "괜찮아"
        clearance = get_render_fit_clearance_for_block(block, 2.0)
        wrapped, font_size, rendered_width, rendered_height = pyside_word_wrap(
            text,
            "Ownglyph gumama3",
            int(source_rect[2]),
            int(source_rect[3]),
            1.0,
            2.0,
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

        refit_text, refit_size, refit_width, refit_height = refit_detected_bubble_text_if_underfilled(
            block,
            text,
            "Ownglyph gumama3",
            int(source_rect[2]),
            int(source_rect[3]),
            1.0,
            2.0,
            False,
            False,
            False,
            QtCore.Qt.AlignmentFlag.AlignCenter,
            QtCore.Qt.LayoutDirection.LeftToRight,
            60,
            5,
            False,
            clearance,
            wrapped,
            font_size,
            rendered_width,
            rendered_height,
            auto_max_font_size=False,
        )

        self.assertEqual(refit_text, wrapped)
        self.assertEqual(refit_size, font_size)
        self.assertEqual(refit_width, rendered_width)
        self.assertEqual(refit_height, rendered_height)

    def test_korean_wrap_does_not_shrink_below_readable_floor(self) -> None:
        wrapped, font_size, width, height = pyside_word_wrap(
            "매끈매끈하고 매끄러워요... ♥",
            "Ownglyph gumama3",
            40,
            24,
            1.0,
            2.0,
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

        self.assertGreaterEqual(font_size, 12)
        self.assertTrue(wrapped)
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)
