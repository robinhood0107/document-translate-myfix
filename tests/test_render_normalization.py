from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui, QtWidgets

import imkit as imk

from app.ui.canvas.text_item import TextBlockItem
from modules.rendering.render import (
    TextRenderingSettings,
    apply_strict_render_state_guard,
    block_needs_original_restore_after_render,
    describe_auto_render_review_status_gate,
    describe_render_text_markup,
    describe_render_text_sanitization,
    describe_text_free_large_mask_gate,
    describe_text_free_render_mask_gate,
    describe_text_free_underfill_gate,
    select_blocks_for_original_restore_after_render,
    should_skip_short_render_translation,
)
from modules.utils.textblock import TextBlock
from modules.rendering.rich_text import (
    repair_render_html_style,
    should_use_rich_text,
)


class RenderNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_quotes_are_preserved_when_dedicated_fallback_font_exists(self) -> None:
        with mock.patch(
            "modules.rendering.render.resolve_render_symbol_fallback_font_family",
            return_value="FallbackFont",
        ), mock.patch(
            "modules.rendering.render._render_font_supports",
            side_effect=lambda _metrics, ch: ch not in {"「", "」"},
        ):
            result = describe_render_text_sanitization(
                '코스네임 「니지카와 사키」',
                "StubFont",
            )

        self.assertEqual(result.text, '코스네임 「니지카와 사키」')
        self.assertFalse(result.normalization_applied)

    def test_text_rendering_settings_keeps_auto_max_defaults_for_product_ui(self) -> None:
        settings = TextRenderingSettings(
            alignment_id=1,
            vertical_alignment_id=1,
            font_family="StubFont",
            min_font_size=5,
            max_font_size=40,
            color="#000000",
            force_font_color=False,
            smart_global_apply_all=False,
            upper_case=False,
            outline=False,
            outline_color="#FFFFFF",
            outline_width="0",
            bold=False,
            italic=False,
            underline=False,
            line_spacing="1.0",
            direction=QtCore.Qt.LayoutDirection.LeftToRight,
        )

        self.assertTrue(settings.auto_max_font_size)
        self.assertEqual(settings.auto_max_font_profile, "current")

    def test_quotes_fallback_to_ascii_without_dedicated_fallback_font(self) -> None:
        with mock.patch(
            "modules.rendering.render.resolve_render_symbol_fallback_font_family",
            return_value="",
        ), mock.patch(
            "modules.rendering.render._render_font_supports",
            side_effect=lambda _metrics, ch: ch not in {"「", "」"},
        ):
            result = describe_render_text_sanitization(
                '코스네임 「니지카와 사키」',
                "StubFont",
            )

        self.assertEqual(result.text, '코스네임 "니지카와 사키"')
        self.assertTrue(result.normalization_applied)
        self.assertIn("quote-to-ascii", result.reasons)

    def test_markup_wraps_quotes_and_hearts_with_dedicated_font(self) -> None:
        with mock.patch(
            "modules.rendering.render.resolve_render_symbol_fallback_font_family",
            return_value="FallbackFont",
        ):
            result = describe_render_text_markup('「나」랑 보낼래요♥')

        self.assertTrue(result.html_applied)
        self.assertEqual(result.text, '「나」랑 보낼래요♥')
        self.assertIn("symbol-fallback-font", result.reasons)
        self.assertIn("font-family:'FallbackFont';", result.html_text)
        self.assertIn("<span", result.html_text)

    def test_decorative_noise_is_removed_but_render_chars_are_preserved(self) -> None:
        with mock.patch(
            "modules.rendering.render.resolve_render_symbol_fallback_font_family",
            return_value="FallbackFont",
        ), mock.patch(
            "modules.rendering.render._render_font_supports",
            return_value=True,
        ):
            result = describe_render_text_sanitization(
                '⌒「테스트」♥︸',
                "StubFont",
            )

        self.assertEqual(result.text, '「테스트」♥')
        self.assertIn("decorative-noise", result.reasons)
        self.assertNotIn("quote-to-ascii", result.reasons)

    def test_strict_korean_render_drops_decorative_symbols_before_rendering(self) -> None:
        with mock.patch(
            "modules.rendering.render.resolve_render_symbol_fallback_font_family",
            return_value="FallbackFont",
        ), mock.patch(
            "modules.rendering.render._render_font_supports",
            return_value=True,
        ):
            result = describe_render_text_sanitization(
                "가・게・해・줘♡ ↘11 🍁",
                "StubFont",
                strict_symbols=True,
            )

        self.assertEqual(result.text, "가게해줘 11")
        self.assertTrue(result.normalization_applied)
        self.assertIn("render_sanitized_symbols", result.reasons)

    def test_strict_markup_does_not_revive_hearts_with_fallback_font(self) -> None:
        with mock.patch(
            "modules.rendering.render.resolve_render_symbol_fallback_font_family",
            return_value="FallbackFont",
        ):
            result = describe_render_text_markup(
                "좋아♡",
                font_family="StubFont",
                font_size=30,
                strict_symbols=True,
            )

        self.assertEqual(result.text, "좋아")
        self.assertNotIn("♡", result.html_text)
        self.assertNotIn("♥", result.html_text)
        self.assertNotIn("symbol-fallback-font", result.reasons)

    def test_strict_render_state_guard_removes_forbidden_symbols_before_export(self) -> None:
        state = {
            "text": "좋아♥",
            "render_text": "좋아♥",
            "font_family": "StubFont",
            "render_html_applied": True,
            "render_normalization_reasons": [],
            "render_normalization_replacements": [],
        }

        changed = apply_strict_render_state_guard(
            state,
            block_index=2,
            image_path="example.png",
        )

        self.assertTrue(changed)
        self.assertEqual(state["text"], "좋아")
        self.assertEqual(state["render_text"], "좋아")
        self.assertFalse(state["render_html_applied"])
        self.assertTrue(state["render_forbidden_symbol_guard"])
        self.assertIn("render_forbidden_symbol_guard", state["render_normalization_reasons"])

    def test_text_free_render_mask_gate_blocks_render_without_matching_erase_mask(self) -> None:
        block = TextBlock(
            text_bbox=QtCore.QRect(0, 0, 100, 260).getCoords(),
            text_class="text_free",
            text="身体の反応がいい",
            translation="몸 반응이 좋아",
        )
        block.block_final_mask_pixel_count = 0

        decision = describe_text_free_render_mask_gate(block, target_lang_code="ko")

        self.assertFalse(decision.render)
        self.assertEqual(decision.status, "needs_review_text_free_mask")
        self.assertIn("render_without_erase_mask", decision.reasons)

    def test_text_free_render_mask_gate_allows_large_text_when_any_erase_mask_exists(self) -> None:
        block = TextBlock(
            text_bbox=QtCore.QRect(0, 0, 185, 389).getCoords(),
            text_class="text_free",
            text="女として気持ちよくなる方法",
            translation="여자로서 기분 좋아지는 법",
        )
        block.block_final_mask_pixel_count = 6119
        block.block_mask_iou = 0.06
        block.block_mask_span_coverage = 1.0

        decision = describe_text_free_render_mask_gate(block, target_lang_code="ko")

        self.assertTrue(decision.render)
        self.assertEqual(decision.status, "ok")

    def test_text_free_underfill_gate_blocks_tiny_render_on_large_free_text(self) -> None:
        block = TextBlock(
            text_bbox=QtCore.QRect(0, 0, 120, 260).getCoords(),
            text_class="text_free",
            text="身体の反応がいい",
            translation="몸 반응이 좋아",
        )

        decision = describe_text_free_underfill_gate(
            block,
            source_rect=(0.0, 0.0, 120.0, 260.0),
            rendered_width=28.0,
            rendered_height=44.0,
            target_lang_code="ko",
        )

        self.assertFalse(decision.render)
        self.assertEqual(decision.status, "needs_review_text_free_underfilled")
        self.assertIn("text_free_underfilled", decision.reasons)

    def test_text_free_large_mask_gate_is_diagnostic_not_a_render_blocker(self) -> None:
        block = TextBlock(
            text_bbox=QtCore.QRect(0, 0, 148, 274).getCoords(),
            text_class="text_free",
            text="大きな縦書き文字",
            translation="큰 세로 글자",
        )
        block.block_final_mask_pixel_count = 43672
        block.block_mask_iou = 0.73
        block.block_mask_span_coverage = 1.0
        block.block_mask_source = "ctd_refined"

        decision = describe_text_free_large_mask_gate(
            block,
            source_rect=(0.0, 0.0, 148.0, 274.0),
            target_lang_code="ko",
        )

        self.assertTrue(decision.render)
        self.assertEqual(decision.status, "ok")

    def test_text_free_large_mask_gate_allows_moderate_free_erase(self) -> None:
        block = TextBlock(
            text_bbox=QtCore.QRect(0, 0, 174, 199).getCoords(),
            text_class="text_free",
            text="普通の自由文字",
            translation="보통 자유 텍스트",
        )
        block.block_final_mask_pixel_count = 37344
        block.block_mask_iou = 0.68
        block.block_mask_span_coverage = 1.0
        block.block_mask_source = "ctd_refined"

        decision = describe_text_free_large_mask_gate(
            block,
            source_rect=(0.0, 0.0, 174.0, 199.0),
            target_lang_code="ko",
        )

        self.assertTrue(decision.render)
        self.assertEqual(decision.status, "ok")

    def test_text_free_large_mask_review_status_is_diagnostic_not_auto_skip(self) -> None:
        decision = describe_auto_render_review_status_gate(
            "needs_review_text_free_large_mask"
        )

        self.assertTrue(decision.render)
        self.assertEqual(decision.status, "ok")

    def test_single_character_text_bubble_translation_is_renderable(self) -> None:
        block = TextBlock(
            text_bbox=QtCore.QRect(0, 0, 80, 140).getCoords(),
            text_class="text_bubble",
            text="おっと",
            translation="엇",
        )

        self.assertFalse(should_skip_short_render_translation(block, "엇"))

    def test_single_character_text_free_translation_still_skips(self) -> None:
        block = TextBlock(
            text_bbox=QtCore.QRect(0, 0, 80, 140).getCoords(),
            text_class="text_free",
            text="指",
            translation="손",
        )

        self.assertTrue(should_skip_short_render_translation(block, "손"))

    def test_render_skip_with_mask_requires_original_restore(self) -> None:
        block = TextBlock(
            text_bbox=QtCore.QRect(0, 0, 80, 80).getCoords(),
            text_class="text_free",
            text="身体の反応がいい",
            translation="몸 반응이 좋아",
        )
        block.block_final_mask_pixel_count = 25
        block._render_skip_reason = "needs_review_text_free_mask"
        block._render_translation_raw = "몸 반응이 좋아"
        block._render_text = ""

        self.assertTrue(block_needs_original_restore_after_render(block))

    def test_bubble_panel_empty_fragment_does_not_restore_group_with_rendered_member(self) -> None:
        empty_fragment = TextBlock(
            text_bbox=QtCore.QRect(10, 10, 120, 180).getCoords(),
            text_class="text_bubble",
            text="broken",
            translation="",
        )
        rendered_fragment = TextBlock(
            text_bbox=QtCore.QRect(10, 10, 120, 180).getCoords(),
            text_class="text_bubble",
            text="broken duplicate",
            translation="번역 있음",
        )
        for block in (empty_fragment, rendered_fragment):
            block.block_final_mask_pixel_count = 120
            block.bubble_panel_text_candidate = True
            block.bubble_panel_group_id = "bubble_panel_1"
            block.mask_reject_reason = "bubble_panel_text_candidate"
        empty_fragment._render_translation_raw = ""
        empty_fragment._render_text = ""
        rendered_fragment._render_translation_raw = "번역 있음"
        rendered_fragment._render_text = "번역 있음"
        rendered_fragment._text_fit_status = "fit"

        restore_blocks = select_blocks_for_original_restore_after_render(
            [empty_fragment, rendered_fragment]
        )

        self.assertEqual(restore_blocks, [])
        self.assertEqual(
            empty_fragment._render_restore_suppressed_reason,
            "bubble_panel_group_rendered",
        )

    def test_unsafe_control_chars_are_removed_before_rendering(self) -> None:
        with mock.patch(
            "modules.rendering.render.resolve_render_symbol_fallback_font_family",
            return_value="FallbackFont",
        ), mock.patch(
            "modules.rendering.render._render_font_supports",
            return_value=True,
        ):
            result = describe_render_text_sanitization(
                "안\u200b녕\u2066�\ufffc\ue000\t끝",
                "StubFont",
            )

        self.assertEqual(result.text, "안녕 끝")
        self.assertTrue(result.normalization_applied)
        self.assertIn("unsafe-control", result.reasons)

    def test_severe_repetition_is_collapsed_before_rendering(self) -> None:
        with mock.patch(
            "modules.rendering.render.resolve_render_symbol_fallback_font_family",
            return_value="FallbackFont",
        ), mock.patch(
            "modules.rendering.render._render_font_supports",
            return_value=True,
        ):
            result = describe_render_text_sanitization("으" * 80, "StubFont")

        self.assertEqual(result.text, "으으으으...")
        self.assertTrue(result.normalization_applied)
        self.assertIn("severe-repetition", result.reasons)

    def test_multichar_sfx_repetition_is_not_collapsed_before_rendering(self) -> None:
        text = "으아" * 30
        with mock.patch(
            "modules.rendering.render.resolve_render_symbol_fallback_font_family",
            return_value="FallbackFont",
        ), mock.patch(
            "modules.rendering.render._render_font_supports",
            return_value=True,
        ):
            result = describe_render_text_sanitization(text, "StubFont")

        self.assertEqual(result.text, text)
        self.assertFalse(result.normalization_applied)

    def test_sample_japan_101_known_gemma_runaway_collapses_before_rendering(self) -> None:
        sample_path = Path(__file__).resolve().parents[1] / "Sample" / "japan" / "101.png"
        self.assertTrue(sample_path.exists())
        image = imk.read_image(str(sample_path))
        self.assertIsNotNone(image)
        self.assertEqual(tuple(image.shape[:2]), (2885, 2014))

        with mock.patch(
            "modules.rendering.render.resolve_render_symbol_fallback_font_family",
            return_value="FallbackFont",
        ), mock.patch(
            "modules.rendering.render._render_font_supports",
            return_value=True,
        ):
            result = describe_render_text_sanitization(
                "으" * 152,
                "StubFont",
                block_index=1,
                image_path=str(sample_path),
            )

        self.assertEqual(result.text, "으으으으...")
        self.assertTrue(result.normalization_applied)
        self.assertIn("severe-repetition", result.reasons)

    def test_styled_markup_keeps_base_font_size_with_fallback_spans(self) -> None:
        with mock.patch(
            "modules.rendering.render.resolve_render_symbol_fallback_font_family",
            return_value="Malgun Gothic",
        ):
            result = describe_render_text_markup(
                '「조직」이 계속 했던\n내 얘기가 아니야.',
                font_family="Ownglyph gumama3",
                font_size=30,
                text_color=QtGui.QColor("#111111"),
                alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
                line_spacing=1.0,
            )

        self.assertTrue(result.html_applied)
        self.assertIn("font-family:'Ownglyph gumama3';", result.html_text)
        self.assertIn("font-size:30pt;", result.html_text)
        self.assertIn("font-family:'Malgun Gothic';", result.html_text)
        self.assertNotIn("11.25pt", result.html_text)

        doc = QtGui.QTextDocument()
        doc.setHtml(result.html_text)
        cursor = QtGui.QTextCursor(doc)
        cursor.setPosition(2)
        self.assertEqual(cursor.charFormat().fontPointSize(), 30)
        self.assertEqual(
            doc.firstBlock().blockFormat().alignment(),
            QtCore.Qt.AlignmentFlag.AlignCenter,
        )

    def test_styled_markup_wraps_unsupported_korean_glyphs_with_fallback_font(self) -> None:
        support_call_count = {}

        def supports(metrics, ch):
            del metrics
            support_call_count[ch] = support_call_count.get(ch, 0) + 1
            if ch == "큥":
                return support_call_count[ch] != 1
            return True

        with mock.patch(
            "modules.rendering.render.resolve_render_symbol_fallback_font_family",
            return_value="Malgun Gothic",
        ), mock.patch(
            "modules.rendering.render._render_font_supports",
            side_effect=supports,
        ), mock.patch(
            "modules.rendering.render._render_font_has_real_glyph",
            return_value=True,
        ):
            result = describe_render_text_markup(
                "배 안쪽이 큥큥거려",
                font_family="Ownglyph gumama3",
                font_size=30,
                text_color=QtGui.QColor("#111111"),
                alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
            )

        self.assertTrue(result.html_applied)
        self.assertIn("glyph-fallback-font", result.reasons)
        self.assertIn("<span style=\"font-family:'Malgun Gothic';\">큥</span>", result.html_text)
        self.assertEqual(result.html_text.count("font-family:'Malgun Gothic';"), 2)

    def test_styled_markup_uses_raw_glyph_index_when_metrics_overreports_support(self) -> None:
        def has_real_glyph(font_family, ch):
            if font_family == "Ownglyph gumama3" and ch == "큥":
                return False
            return True

        with mock.patch(
            "modules.rendering.render.resolve_render_symbol_fallback_font_family",
            return_value="Malgun Gothic",
        ), mock.patch(
            "modules.rendering.render._render_font_supports",
            return_value=True,
        ), mock.patch(
            "modules.rendering.render._render_font_has_real_glyph",
            side_effect=has_real_glyph,
        ):
            result = describe_render_text_markup(
                "배 안쪽이 큥큥거려",
                font_family="Ownglyph gumama3",
                font_size=30,
                text_color=QtGui.QColor("#111111"),
                alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
            )

        self.assertTrue(result.html_applied)
        self.assertIn("glyph-fallback-font", result.reasons)
        self.assertIn("<span style=\"font-family:'Malgun Gothic';\">큥</span>", result.html_text)
        self.assertEqual(result.html_text.count("font-family:'Malgun Gothic';"), 2)

    def test_styled_markup_checks_effective_app_font_when_font_family_is_empty(self) -> None:
        previous_font = QtWidgets.QApplication.font()
        QtWidgets.QApplication.setFont(QtGui.QFont("Ownglyph gumama3", 25))

        def has_real_glyph(font_family, ch):
            if font_family == "Ownglyph gumama3" and ch == "큥":
                return False
            return True

        try:
            with mock.patch(
                "modules.rendering.render.resolve_render_symbol_fallback_font_family",
                return_value="Malgun Gothic",
            ), mock.patch(
                "modules.rendering.render._render_font_supports",
                return_value=True,
            ), mock.patch(
                "modules.rendering.render._render_font_has_real_glyph",
                side_effect=has_real_glyph,
            ):
                result = describe_render_text_markup(
                    "배\n안쪽이\n큥큥거려",
                    font_family="",
                    font_size=25,
                    text_color=QtGui.QColor("#111111"),
                    alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
                )
        finally:
            QtWidgets.QApplication.setFont(previous_font)

        self.assertTrue(result.html_applied)
        self.assertIn("font-family:'Ownglyph gumama3';", result.html_text)
        self.assertIn("glyph-fallback-font", result.reasons)
        self.assertEqual(result.html_text.count("font-family:'Malgun Gothic';"), 2)

    def test_styled_markup_uses_glyph_fallback_when_symbol_fallback_is_unavailable(self) -> None:
        def has_real_glyph(font_family, ch):
            if font_family == "Ownglyph gumama3" and ch == "큥":
                return False
            return True

        with mock.patch(
            "modules.rendering.render.resolve_render_symbol_fallback_font_family",
            return_value="",
        ), mock.patch(
            "modules.rendering.render.resolve_render_glyph_fallback_font_family",
            return_value="Malgun Gothic",
        ), mock.patch(
            "modules.rendering.render._render_font_supports",
            return_value=True,
        ), mock.patch(
            "modules.rendering.render._render_font_has_real_glyph",
            side_effect=has_real_glyph,
        ):
            result = describe_render_text_markup(
                "배\n안쪽이\n큥큥거려",
                font_family="Ownglyph gumama3",
                font_size=25,
                text_color=QtGui.QColor("#111111"),
                alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
            )

        self.assertTrue(result.html_applied)
        self.assertIn("glyph-fallback-font", result.reasons)
        self.assertEqual(result.html_text.count("font-family:'Malgun Gothic';"), 2)

    def test_glyph_fallback_resolver_registers_system_fonts_when_database_is_sparse(self) -> None:
        from modules.rendering import render

        render.resolve_render_glyph_fallback_font_family.cache_clear()
        database = mock.Mock()
        database.families.side_effect = [
            ["Ownglyph gumama3"],
            ["Ownglyph gumama3", "Malgun Gothic"],
        ]

        with mock.patch(
            "modules.rendering.render.QFontDatabase",
            return_value=database,
        ), mock.patch(
            "modules.rendering.render._register_render_fallback_system_fonts",
            return_value=("Malgun Gothic",),
        ) as register_fonts, mock.patch(
            "modules.rendering.render._render_font_supports",
            return_value=True,
        ), mock.patch(
            "modules.rendering.render._render_font_has_real_glyph",
            return_value=True,
        ):
            family = render.resolve_render_glyph_fallback_font_family(("큥",))

        self.assertEqual(family, "Malgun Gothic")
        register_fonts.assert_called_once()
        render.resolve_render_glyph_fallback_font_family.cache_clear()

    def test_broken_qt_body_font_size_is_repaired_from_item_style(self) -> None:
        broken = (
            '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" '
            '"http://www.w3.org/TR/REC-html40/strict.dtd">'
            "<html><head><meta name=\"qrichtext\" content=\"1\" /></head>"
            "<body style=\" font-family:'Malgun Gothic'; font-size:11.25pt; font-weight:400;\">"
            "<p>「조직」이 계속 했던</p></body></html>"
        )

        repaired = repair_render_html_style(
            broken,
            font_family="Ownglyph gumama3",
            font_size=30,
            text_color=QtGui.QColor("#000000"),
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
            line_spacing=1.0,
        )

        self.assertIn("font-family:'Ownglyph gumama3';", repaired)
        self.assertIn("font-size:30pt;", repaired)
        self.assertNotIn("11.25pt", repaired)

        item = TextBlockItem(
            font_family="Ownglyph gumama3",
            font_size=30,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )
        item.set_text(repaired, 240)
        cursor = QtGui.QTextCursor(item.document())
        cursor.setPosition(2)
        self.assertEqual(cursor.charFormat().fontPointSize(), 30)

    def test_plain_angle_brackets_are_not_treated_as_html(self) -> None:
        self.assertFalse(should_use_rich_text("번역문에 <tag> 문자열이 있음"))

        item = TextBlockItem(font_family="Ownglyph gumama3", font_size=30)
        item.set_text("번역문에 <tag> 문자열이 있음", 240)

        self.assertEqual(item.toPlainText(), "번역문에 <tag> 문자열이 있음")

    def test_generic_font_family_span_is_treated_as_render_html(self) -> None:
        html = "<span style=\"font-family:'Malgun Gothic';\">큥</span>"

        self.assertTrue(should_use_rich_text(html))

        item = TextBlockItem(font_family="Ownglyph gumama3", font_size=30)
        item.set_text(html, 120)
        self.assertEqual(item.toPlainText(), "큥")


if __name__ == "__main__":
    unittest.main()
