from __future__ import annotations

import unittest

from modules.ocr.paddleocr_vl_spotting.response_contract import (
    PaddleSpottingResponseContractError,
    parse_paddle_spotting_response,
)


def _line(
    text: str = "テスト",
    coordinates: tuple[int, ...] = (100, 100, 300, 100, 300, 200, 100, 200),
) -> str:
    tokens = "".join(f"<|LOC_{value}|>" for value in coordinates)
    return f"{tokens}{text}"


class PaddleSpottingResponseContractTests(unittest.TestCase):
    def test_accepts_native_lines_and_removes_terminal_token(self) -> None:
        parsed = parse_paddle_spotting_response(
            f"{_line('一行目')}\n{_line('二行目', (400, 300, 600, 300, 600, 400, 400, 400))}</s>"
        )

        self.assertEqual([region.text for region in parsed.regions], ["一行目", "二行目"])
        self.assertEqual(parsed.regions[0].source_line, 1)
        self.assertEqual(
            parsed.regions[0].normalized_points,
            ((100, 100), (300, 100), (300, 200), (100, 200)),
        )
        self.assertEqual(parsed.response_kind, "native_spotting_lines")

    def test_deduplicates_only_exact_text_and_geometry(self) -> None:
        line = _line("同じ")
        parsed = parse_paddle_spotting_response(
            "\n".join(
                (
                    line,
                    line,
                    _line("別文"),
                    _line(
                        "同じ",
                        (100, 210, 300, 210, 300, 310, 100, 310),
                    ),
                )
            )
        )

        self.assertEqual(len(parsed.regions), 3)
        self.assertEqual(parsed.duplicate_region_count, 1)

    def test_rejects_malformed_native_lines_without_partial_acceptance(self) -> None:
        cases = (
            ("", "empty_response"),
            (_line(coordinates=(1, 2, 3, 4, 5, 6)), "invalid_native_line"),
            (
                _line(coordinates=(1, 2, 3, 4, 5, 6, 1001, 8)),
                "coordinate_out_of_range",
            ),
            (
                _line(coordinates=(100, 100, 100, 200, 100, 300, 100, 400)),
                "degenerate_geometry",
            ),
            (
                "<|LOC_100|><|LOC_100|><|LOC_200|><|LOC_100|>"
                "<|LOC_200|><|LOC_200|><|LOC_100|><|LOC_200|>",
                "invalid_native_line",
            ),
            (f"{_line()}\nnot-a-native-spotting-line", "invalid_native_line"),
            (
                _line("本文<|BAD_TOKEN|>"),
                "unexpected_special_token",
            ),
            (
                f"prefix{_line()}",
                "invalid_native_line",
            ),
        )
        for payload, expected_code in cases:
            with self.subTest(expected_code=expected_code), self.assertRaises(
                PaddleSpottingResponseContractError
            ) as raised:
                parse_paddle_spotting_response(payload)
            self.assertEqual(raised.exception.code, expected_code)


if __name__ == "__main__":
    unittest.main()
