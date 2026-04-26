from __future__ import annotations

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui, QtWidgets

from app.ui.canvas.text_item import TextBlockItem
from modules.rendering.render import (
    describe_render_text_markup,
    describe_render_text_sanitization,
)
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


if __name__ == "__main__":
    unittest.main()
