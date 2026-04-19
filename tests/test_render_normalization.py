from __future__ import annotations

import unittest
from unittest import mock

from modules.rendering.render import (
    describe_render_text_markup,
    describe_render_text_sanitization,
)


class RenderNormalizationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
