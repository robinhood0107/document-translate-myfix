from __future__ import annotations

import os
import unittest
from pathlib import Path

from modules.utils.textblock import TextBlock
from modules.utils.txt_md_exchange import (
    apply_translation_pages,
    build_exchange_text,
    collect_page_entries,
    find_duplicate_page_names,
    parse_translation_exchange_file,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "이식2" / "format_samples"


class TxtMdExchangeTests(unittest.TestCase):
    def _build_states(self) -> dict[str, dict]:
        page_1 = os.path.join("chapter", "page_001.png")
        page_2 = os.path.join("chapter", "page_002.png")
        return {
            page_1: {
                "blk_list": [
                    TextBlock(text="Hello there.", translation="안녕하세요."),
                    TextBlock(text="This is the second balloon.", translation="이것은 두 번째 말풍선입니다."),
                ]
            },
            page_2: {
                "blk_list": [
                    TextBlock(
                        text=["A", "multiline", "source", "line\nthat continues on the next line."],
                        translation="여러 줄로 이어지는\n번역 예시입니다.",
                    ),
                    TextBlock(text="Final source block.", translation="마지막 번역 블록입니다."),
                ]
            },
        }

    def test_source_export_matches_transplant_samples_for_txt_and_md(self) -> None:
        states = self._build_states()
        ordered_paths = list(states.keys())
        payload = build_exchange_text(collect_page_entries(ordered_paths, states, "source"))

        expected_txt = (SAMPLES_DIR / "sample_source.txt").read_text(encoding="utf-8")
        expected_md = (SAMPLES_DIR / "sample_source.md").read_text(encoding="utf-8")

        self.assertEqual(payload, expected_txt.rstrip("\n"))
        self.assertEqual(payload, expected_md.rstrip("\n"))

    def test_translation_parser_and_apply_support_partial_match_and_dictionary_rules(self) -> None:
        parsed_pages = parse_translation_exchange_file(str(SAMPLES_DIR / "sample_partial_mismatch.txt"))
        page_blocks = {
            "page_001.png": [
                TextBlock(text="src-1", translation="", rich_text="<b>old</b>"),
                TextBlock(text="src-2", translation="", rich_text="<b>old</b>"),
            ],
            "page_002.png": [
                TextBlock(text="src-3", translation="", rich_text="<b>old</b>"),
                TextBlock(text="src-4", translation="", rich_text="<b>old</b>"),
            ],
            "page_003.png": [
                TextBlock(text="src-5", translation="", rich_text="<b>old</b>"),
            ],
        }

        all_matched, result = apply_translation_pages(
            parsed_pages,
            page_blocks,
            translation_rules=[
                {
                    "keyword": "번역",
                    "sub": "TRANS",
                    "use_reg": False,
                    "case_sens": True,
                }
            ],
        )

        self.assertFalse(all_matched)
        self.assertEqual(result["matched_pages"], ["page_001.png", "page_002.png"])
        self.assertEqual(result["missing_pages"], ["page_003.png"])
        self.assertEqual(result["unexpected_pages"], [])
        self.assertEqual(result["unmatched_pages"], ["page_001.png", "page_002.png"])
        self.assertEqual(page_blocks["page_001.png"][0].translation, "첫 번째 블록만 있습니다.")
        self.assertEqual(page_blocks["page_001.png"][1].translation, "")
        self.assertEqual(page_blocks["page_002.png"][0].translation, "첫 번째 TRANS입니다.")
        self.assertEqual(page_blocks["page_002.png"][1].translation, "두 번째 TRANS입니다.")
        self.assertEqual(page_blocks["page_001.png"][0].rich_text, "")
        self.assertEqual(page_blocks["page_002.png"][1].rich_text, "")

    def test_find_duplicate_page_names_blocks_same_basename(self) -> None:
        duplicates = find_duplicate_page_names(
            [
                os.path.join("a", "page_001.png"),
                os.path.join("b", "page_001.png"),
                os.path.join("c", "page_002.png"),
            ]
        )

        self.assertEqual(duplicates, ["page_001.png"])


if __name__ == "__main__":
    unittest.main()
