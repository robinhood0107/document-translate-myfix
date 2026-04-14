from __future__ import annotations

import unittest
from unittest import mock

from modules.rendering.render import describe_render_text_sanitization


class RenderNormalizationTests(unittest.TestCase):
    def test_quotes_fallback_to_ascii_when_font_lacks_support(self) -> None:
        with mock.patch(
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

    def test_heart_falls_back_to_empty_when_font_lacks_support(self) -> None:
        with mock.patch(
            "modules.rendering.render._render_font_supports",
            side_effect=lambda _metrics, ch: ch != "♥",
        ):
            result = describe_render_text_sanitization(
                "아저씨랑 즐거운 시간 보낼래요♥",
                "StubFont",
            )

        self.assertEqual(result.text, "아저씨랑 즐거운 시간 보낼래요")
        self.assertTrue(result.normalization_applied)
        self.assertIn("heart-dropped", result.reasons)

    def test_decorative_noise_is_removed_but_render_chars_are_preserved_when_supported(self) -> None:
        with mock.patch(
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
