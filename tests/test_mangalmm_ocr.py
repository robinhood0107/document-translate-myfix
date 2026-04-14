from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from modules.ocr.mangalmm_ocr import MangaLMMOCREngine, OCRRegion
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

    def test_map_regions_to_page_coords_restores_original_coordinates(self) -> None:
        engine = MangaLMMOCREngine()
        regions = [{"bbox_2d": [20, 40, 60, 80], "text_content": "中身"}]
        mapped = engine._map_regions_to_page_coords(
            regions,
            (100, 200, 300, 400),
            (200, 200),
            0.5,
            "page_tile",
        )
        self.assertEqual(len(mapped), 1)
        self.assertEqual(mapped[0].bbox_xyxy, [140, 280, 220, 360])
        self.assertEqual(mapped[0].unit_bbox_xyxy, [100, 200, 300, 400])
        self.assertEqual(mapped[0].unit_kind, "page_tile")

    def test_build_request_units_splits_large_pages_into_overlapping_tiles(self) -> None:
        engine = MangaLMMOCREngine()
        units = engine._build_request_units((3035, 2150, 3))
        self.assertGreater(len(units), 1)
        self.assertEqual(units[0].bbox_xyxy, (0, 0, 1280, 1280))
        self.assertEqual(units[-1].bbox_xyxy, (870, 1755, 2150, 3035))

    def test_build_rescue_units_uses_text_expansion_overrides(self) -> None:
        engine = MangaLMMOCREngine()
        engine.rescue_min_size = 0
        engine.text_expansion_ratio_x = 0.10
        engine.text_expansion_ratio_y = 0.20
        blk = TextBlock(
            text_bbox=np.array([50, 60, 150, 140], dtype=np.int32),
            bubble_bbox=np.array([40, 50, 160, 150], dtype=np.int32),
            text_class="text_bubble",
            source_lang="ja",
            direction="vertical",
        )
        units = engine._build_rescue_units([blk], (200, 200, 3))
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].bbox_xyxy, (28, 30, 172, 170))

    def test_dedupe_regions_prefers_farther_from_tile_edge(self) -> None:
        engine = MangaLMMOCREngine()
        kept = engine._dedupe_regions(
            [
                OCRRegion(
                    bbox_xyxy=[100, 100, 160, 160],
                    text="セリフ",
                    unit_bbox_xyxy=[0, 0, 180, 180],
                    unit_kind="page_tile",
                    unit_resize_scale=1.0,
                    edge_distance=8.0,
                    normalized_text="セリフ",
                ),
                OCRRegion(
                    bbox_xyxy=[102, 102, 158, 158],
                    text="セリフ",
                    unit_bbox_xyxy=[0, 0, 300, 300],
                    unit_kind="page_tile",
                    unit_resize_scale=1.0,
                    edge_distance=60.0,
                    normalized_text="セリフ",
                ),
            ]
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].edge_distance, 60.0)

    def test_assignment_prefers_precision_box_inside_shared_bubble(self) -> None:
        engine = MangaLMMOCREngine()
        blk_left = TextBlock(
            text_bbox=np.array([40, 40, 90, 120], dtype=np.int32),
            bubble_bbox=np.array([20, 20, 200, 150], dtype=np.int32),
            text_class="text_bubble",
            source_lang="ja",
            direction="vertical",
        )
        blk_right = TextBlock(
            text_bbox=np.array([120, 40, 170, 120], dtype=np.int32),
            bubble_bbox=np.array([20, 20, 200, 150], dtype=np.int32),
            text_class="text_bubble",
            source_lang="ja",
            direction="vertical",
        )
        region = OCRRegion(
            bbox_xyxy=[122, 42, 168, 118],
            text="右",
            unit_bbox_xyxy=[0, 0, 256, 256],
            unit_kind="page_full",
            unit_resize_scale=1.0,
            edge_distance=80.0,
            normalized_text="右",
        )
        assignments = engine._assign_regions_to_blocks([region], [blk_left, blk_right])
        quality = engine._apply_assignments_to_blocks(
            [blk_left, blk_right],
            assignments,
            attempt_count=1,
            success_status="ok",
            empty_status="empty_initial",
        )
        self.assertTrue(quality["non_empty"] >= 1)
        self.assertEqual(blk_left.text, "")
        self.assertEqual(blk_right.text, "右")

    def test_text_free_region_is_not_force_assigned_by_nearest_distance_only(self) -> None:
        engine = MangaLMMOCREngine()
        blk = TextBlock(
            text_bbox=np.array([20, 20, 80, 80], dtype=np.int32),
            text_class="text_free",
            source_lang="ja",
            direction="vertical",
        )
        far_region = OCRRegion(
            bbox_xyxy=[160, 160, 220, 220],
            text="遠い",
            unit_bbox_xyxy=[0, 0, 256, 256],
            unit_kind="page_full",
            unit_resize_scale=1.0,
            edge_distance=20.0,
            normalized_text="遠い",
        )
        assignments = engine._assign_regions_to_blocks([far_region], [blk])
        self.assertEqual(assignments[0], [])

    def test_apply_assignments_sorts_multiple_regions_in_reading_order(self) -> None:
        engine = MangaLMMOCREngine()
        blk = TextBlock(
            text_bbox=np.array([10, 10, 210, 210], dtype=np.int32),
            bubble_bbox=np.array([0, 0, 220, 220], dtype=np.int32),
            text_class="text_bubble",
            source_lang="ja",
            direction="vertical",
        )
        assignments = {
            0: [
                {
                    "region": OCRRegion(
                        bbox_xyxy=[100, 10, 140, 80],
                        text="右",
                        unit_bbox_xyxy=[0, 0, 220, 220],
                        unit_kind="page_full",
                        unit_resize_scale=1.0,
                        edge_distance=70.0,
                        normalized_text="右",
                    ),
                    "metrics": {
                        "ownership_cover": 1.0,
                        "precision_cover": 0.2,
                        "ownership_iou": 0.1,
                        "center_in_ownership": True,
                        "center_in_precision": False,
                        "center_distance_norm": 0.4,
                        "precision_area": 40000,
                    },
                },
                {
                    "region": OCRRegion(
                        bbox_xyxy=[20, 10, 60, 80],
                        text="左",
                        unit_bbox_xyxy=[0, 0, 220, 220],
                        unit_kind="page_full",
                        unit_resize_scale=1.0,
                        edge_distance=70.0,
                        normalized_text="左",
                    ),
                    "metrics": {
                        "ownership_cover": 1.0,
                        "precision_cover": 0.2,
                        "ownership_iou": 0.1,
                        "center_in_ownership": True,
                        "center_in_precision": False,
                        "center_distance_norm": 0.6,
                        "precision_area": 40000,
                    },
                },
            ]
        }
        engine._apply_assignments_to_blocks(
            [blk],
            assignments,
            attempt_count=1,
            success_status="ok",
            empty_status="empty_initial",
        )
        self.assertEqual(blk.text, "右左")
        self.assertEqual(len(blk.ocr_regions), 2)

    def test_analyze_region_payload_salvages_object_wrapper(self) -> None:
        engine = MangaLMMOCREngine()
        payload = '{"regions":[{"bbox_2d":[10,20,30,40],"text_content":"テスト"}]}'
        analysis = engine._analyze_region_payload(payload)
        self.assertEqual(analysis["response_kind"], "json_object_wrapper")
        self.assertEqual(len(analysis["regions"]), 1)

    def test_initialize_reads_sampling_env_overrides(self) -> None:
        class _Settings:
            @staticmethod
            def get_mangalmm_ocr_settings():
                return {}

        engine = MangaLMMOCREngine()
        with mock.patch.dict(
            os.environ,
            {
                "CT_MANGALMM_TEMPERATURE": "0.1",
                "CT_MANGALMM_TOP_K": "32",
                "CT_MANGALMM_TEXT_EXPANSION_RATIO_X": "0.12",
                "CT_MANGALMM_TEXT_EXPANSION_RATIO_Y": "0.14",
            },
            clear=False,
        ):
            engine.initialize(_Settings())
        self.assertAlmostEqual(engine.temperature, 0.1, places=3)
        self.assertEqual(engine.top_k, 32)
        self.assertAlmostEqual(engine.text_expansion_ratio_x, 0.12, places=3)
        self.assertAlmostEqual(engine.text_expansion_ratio_y, 0.14, places=3)

    def test_export_debug_artifact_writes_metadata_and_crop(self) -> None:
        engine = MangaLMMOCREngine()
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        blk = TextBlock(
            text_bbox=np.array([1, 2, 20, 24], dtype=np.int32),
            bubble_bbox=np.array([0, 0, 30, 30], dtype=np.int32),
            source_lang="ja",
            direction="vertical",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            engine.debug_root = Path(os.path.abspath(temp_dir))
            engine.debug_export_limit = 4
            engine._export_debug_artifact(
                blk=blk,
                failure_reason="MangaLMM returned no valid OCR regions.",
                response_kind="plain_text_or_non_json",
                raw_text="テキストだけ",
                crop_bbox=(1, 2, 20, 24),
                crop_source="xyxy",
                resize_scale=1.0,
                crop_image=image,
                request_image=image,
                analysis={"response_kind": "plain_text_or_non_json"},
            )
            exported = list(engine.debug_root.glob("*/meta.json"))
            self.assertEqual(len(exported), 1)
            self.assertTrue((exported[0].parent / "crop.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
