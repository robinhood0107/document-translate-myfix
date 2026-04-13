from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from modules.ocr.mangalmm_ocr import MangaLMMOCREngine
from modules.utils.textblock import TextBlock


class MangaLMMOCRTests(unittest.TestCase):
    def test_parse_region_payload_salvages_fenced_json(self) -> None:
        engine = MangaLMMOCREngine()
        payload = """```json
        [{"bbox_2d":[10,20,30,40],"text_content":"テスト"}]
        ```"""
        parsed = engine._parse_region_payload(payload)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["text_content"], "テスト")

    def test_map_regions_to_original_bbox_restores_page_coordinates(self) -> None:
        engine = MangaLMMOCREngine()
        regions = [{"bbox_2d": [20, 40, 60, 80], "text_content": "中身"}]
        mapped = engine._map_regions_to_original_bbox(
            regions,
            (100, 200, 300, 400),
            (200, 200),
            0.5,
        )
        self.assertEqual(mapped[0]["bbox_xyxy"], [140, 280, 220, 360])

    def test_process_block_combines_sorted_texts_without_spaces_for_japanese(self) -> None:
        engine = MangaLMMOCREngine()
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        blk = TextBlock(
            text_bbox=np.array([10, 10, 210, 210], dtype=np.int32),
            source_lang="ja",
            direction="vertical",
        )
        with mock.patch.object(
            engine,
            "_request_response_text",
            return_value='[{"bbox_2d":[100,10,140,80],"text_content":"右"},{"bbox_2d":[20,10,60,80],"text_content":"左"}]',
        ):
            engine._process_block(image, blk, (10, 10, 210, 210), "xyxy")
        self.assertEqual(blk.text, "右左")
        self.assertEqual(len(blk.ocr_regions), 2)


if __name__ == "__main__":
    unittest.main()
