from __future__ import annotations

import unittest

from modules.ocr.mangalmm_response_contract import (
    MangaLMMResponseContractError,
    parse_mangalmm_response,
)


class MangaLMMResponseContractTests(unittest.TestCase):
    def test_accepts_complete_array_and_complete_json_fence(self) -> None:
        for payload, expected_kind in (
            (
                '[{"bbox_2d":[1,2,30,40],"text_content":" text "}]',
                "json_array",
            ),
            (
                '```json\n[{"bbox_2d":[1,2,30,40],"text_content":"text"}]\n```',
                "fenced_json_array",
            ),
        ):
            with self.subTest(expected_kind=expected_kind):
                parsed = parse_mangalmm_response(payload)
                self.assertEqual(parsed.response_kind, expected_kind)
                self.assertEqual(len(parsed.regions), 1)
                self.assertEqual(parsed.regions[0]["text_content"], "text")

    def test_preserves_decorative_text_for_later_semantic_routing(self) -> None:
        parsed = parse_mangalmm_response(
            '[{"bbox_2d":[1,2,30,40],"text_content":"⌒テ✺スト︸"}]'
        )

        self.assertEqual(parsed.regions[0]["text_content"], "⌒テ✺スト︸")
        self.assertEqual(
            parsed.regions[0]["raw_text_content"],
            "⌒テ✺スト︸",
        )

    def test_rejects_wrapping_trailing_json_and_top_level_object(self) -> None:
        cases = (
            (
                'prefix [{"bbox_2d":[1,2,30,40],"text_content":"text"}]',
                "invalid_json",
            ),
            (
                '[{"bbox_2d":[1,2,30,40],"text_content":"text"}] trailing',
                "invalid_json",
            ),
            (
                '{"regions":[{"bbox_2d":[1,2,30,40],"text_content":"text"}]}',
                "top_level_not_array",
            ),
        )
        for payload, expected_code in cases:
            with self.subTest(expected_code=expected_code), self.assertRaises(
                MangaLMMResponseContractError
            ) as raised:
                parse_mangalmm_response(payload)
            self.assertEqual(raised.exception.code, expected_code)

    def test_rejects_duplicate_keys_and_partial_region_arrays(self) -> None:
        cases = (
            (
                '[{"bbox_2d":[1,2,30,40],"bbox_2d":[2,3,31,41],'
                '"text_content":"text"}]',
                "duplicate_key",
            ),
            (
                '[{"bbox_2d":[1,2,30,40],"text_content":"text"},'
                '{"bbox_2d":[1,2,30],"text_content":"broken"}]',
                "invalid_bbox",
            ),
        )
        for payload, expected_code in cases:
            with self.subTest(expected_code=expected_code), self.assertRaises(
                MangaLMMResponseContractError
            ) as raised:
                parse_mangalmm_response(payload)
            self.assertEqual(raised.exception.code, expected_code)

    def test_rejects_invalid_bbox_and_text_types(self) -> None:
        cases = (
            (
                '[{"bbox_2d":[30,2,1,40],"text_content":"text"}]',
                "invalid_bbox_order",
            ),
            (
                '[{"bbox_2d":[1,2,30,40],"text_content":12}]',
                "invalid_text_type",
            ),
            (
                '[{"bbox_2d":[1,2,30,40],"text_content":"  "}]',
                "empty_text",
            ),
        )
        for payload, expected_code in cases:
            with self.subTest(expected_code=expected_code), self.assertRaises(
                MangaLMMResponseContractError
            ) as raised:
                parse_mangalmm_response(payload)
            self.assertEqual(raised.exception.code, expected_code)

    def test_rejects_missing_fields_non_object_regions_and_non_finite_bbox(
        self,
    ) -> None:
        cases = (
            (
                '[{"bbox_2d":[1,2,30,40]}]',
                "missing_region_field",
            ),
            (
                '["not-an-object"]',
                "invalid_region_type",
            ),
            (
                '[{"bbox_2d":[true,2,30,40],"text_content":"text"}]',
                "invalid_bbox",
            ),
            (
                '[{"bbox_2d":[1,2,1e999,40],"text_content":"text"}]',
                "invalid_bbox",
            ),
        )
        for payload, expected_code in cases:
            with self.subTest(expected_code=expected_code), self.assertRaises(
                MangaLMMResponseContractError
            ) as raised:
                parse_mangalmm_response(payload)
            self.assertEqual(raised.exception.code, expected_code)

    def test_rejects_incomplete_json_fence(self) -> None:
        with self.assertRaises(MangaLMMResponseContractError) as raised:
            parse_mangalmm_response(
                '```json\n[{"bbox_2d":[1,2,30,40],'
                '"text_content":"text"}]'
            )

        self.assertEqual(raised.exception.code, "invalid_code_fence")


if __name__ == "__main__":
    unittest.main()
