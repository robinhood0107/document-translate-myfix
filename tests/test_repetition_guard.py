from __future__ import annotations

import unittest

from modules.utils.repetition_guard import (
    analyze_repetition,
    guard_severe_repetition,
    is_severe_repetition,
)


class RepetitionGuardTests(unittest.TestCase):
    def test_severe_repetition_requires_long_text_and_long_identical_run(self) -> None:
        self.assertTrue(is_severe_repetition("으" * 40))
        self.assertTrue(is_severe_repetition(("으" * 16) + ("가나다라마바사아자차카타파하" * 2)))

    def test_short_sfx_repetition_is_preserved(self) -> None:
        for text in ("오오오", "으으", "ㅋㅋㅋ", "푸르르르르", "하아하아하아하아"):
            result = guard_severe_repetition(text)
            self.assertFalse(result.changed, text)
            self.assertEqual(result.text, text)

    def test_multichar_repetition_without_long_identical_run_is_preserved(self) -> None:
        text = "으아" * 30
        analysis = analyze_repetition(text)

        self.assertGreaterEqual(analysis.comparable_length, 40)
        self.assertLess(analysis.longest_run_length, 16)
        self.assertFalse(analysis.severe)
        self.assertEqual(guard_severe_repetition(text).text, text)

    def test_punctuation_is_ignored_for_detection_but_output_is_shortened(self) -> None:
        text = "으!" * 40
        result = guard_severe_repetition(text)

        self.assertTrue(result.changed)
        self.assertEqual(result.text, "으으으으...")
        self.assertEqual(result.analysis.longest_run_char, "으")
        self.assertEqual(result.analysis.longest_run_length, 40)

    def test_punctuation_separated_run_keeps_surrounding_text_when_collapsed(self) -> None:
        text = "시작" + ("으!" * 40) + "끝"
        result = guard_severe_repetition(text)

        self.assertTrue(result.changed)
        self.assertEqual(result.text, "시작으으으으...끝")

    def test_long_raw_run_keeps_surrounding_text_when_collapsed(self) -> None:
        text = "시작" + ("으" * 40) + "끝"
        result = guard_severe_repetition(text)

        self.assertTrue(result.changed)
        self.assertEqual(result.text, "시작으으으으...끝")


if __name__ == "__main__":
    unittest.main()
