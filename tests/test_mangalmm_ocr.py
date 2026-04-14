from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from modules.ocr.mangalmm_ocr import MangaLMMOCREngine, OCRRegion, ResizePlan
from modules.utils.textblock import TextBlock


def _make_block(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    bubble_bbox: tuple[int, int, int, int] | None = None,
    text_class: str = "text_bubble",
) -> TextBlock:
    kwargs = {
        "text_bbox": np.array([x1, y1, x2, y2], dtype=np.int32),
        "text_class": text_class,
        "source_lang": "ja",
        "direction": "vertical",
    }
    if bubble_bbox is not None:
        kwargs["bubble_bbox"] = np.array(bubble_bbox, dtype=np.int32)
    return TextBlock(**kwargs)


def _make_blocks(
    count: int,
    *,
    width: int,
    height: int,
    columns: int = 5,
    gap_x: int = 40,
    gap_y: int = 30,
    start_x: int = 20,
    start_y: int = 20,
) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    for index in range(count):
        col = index % columns
        row = index // columns
        x1 = start_x + col * (width + gap_x)
        y1 = start_y + row * (height + gap_y)
        x2 = x1 + width
        y2 = y1 + height
        blocks.append(_make_block(x1, y1, x2, y2))
    return blocks


class MangaLMMOCRTests(unittest.TestCase):
    def test_parse_region_payload_salvages_fenced_json(self) -> None:
        engine = MangaLMMOCREngine()
        payload = """```json
        [{"bbox_2d":[10,20,30,40],"text_content":"テスト"}]
        ```"""
        parsed = engine._parse_region_payload(payload)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["text_content"], "テスト")

    def test_build_request_units_returns_single_full_page_unit(self) -> None:
        engine = MangaLMMOCREngine()
        units = engine._build_request_units((3035, 2150, 3))
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].bbox_xyxy, (0, 0, 2150, 3035))
        self.assertEqual(units[0].unit_kind, "page_full")

    def test_select_resize_profile_chooses_dense_for_30_and_standard_for_15_and_9(self) -> None:
        engine = MangaLMMOCREngine()
        page_shape = (3035, 2150, 3)

        dense_profile = engine._select_resize_profile(page_shape, _make_blocks(30, width=160, height=180))
        standard_profile_15 = engine._select_resize_profile(page_shape, _make_blocks(15, width=200, height=300))
        standard_profile_9 = engine._select_resize_profile(page_shape, _make_blocks(9, width=220, height=260))

        self.assertEqual(dense_profile[0], "dense")
        self.assertEqual(standard_profile_15[0], "standard")
        self.assertEqual(standard_profile_9[0], "standard")

    def test_plan_page_request_uses_expected_dense_and_standard_sizes(self) -> None:
        engine = MangaLMMOCREngine()

        dense_plan = engine._plan_page_request((3035, 2150, 3), _make_blocks(30, width=160, height=180))
        standard_plan = engine._plan_page_request((3036, 2150, 3), _make_blocks(15, width=200, height=300))

        self.assertEqual(dense_plan.profile, "dense")
        self.assertEqual(dense_plan.request_shape, (1270, 900))
        self.assertEqual(dense_plan.max_completion_tokens, 768)
        self.assertAlmostEqual(dense_plan.scale_x, 900 / 2150.0)
        self.assertAlmostEqual(dense_plan.scale_y, 1270 / 3035.0)

        self.assertEqual(standard_plan.profile, "standard")
        self.assertEqual(standard_plan.request_shape, (1728, 1224))
        self.assertEqual(standard_plan.max_completion_tokens, 512)
        self.assertAlmostEqual(standard_plan.scale_x, 1224 / 2150.0)
        self.assertAlmostEqual(standard_plan.scale_y, 1728 / 3036.0)

    def test_map_regions_to_page_coords_restores_original_coordinates_with_scale_axes(self) -> None:
        engine = MangaLMMOCREngine()
        resize_plan = ResizePlan(
            profile="standard",
            original_shape=(200, 200),
            request_shape=(100, 100),
            base_scale=0.5,
            scale_x=0.5,
            scale_y=0.5,
            max_completion_tokens=512,
            block_count=1,
            small_block_ratio=0.0,
            text_cover_ratio=0.02,
        )
        regions = [{"bbox_2d": [20, 40, 60, 80], "text_content": "中身"}]
        mapped = engine._map_regions_to_page_coords(
            regions,
            (0, 0, 200, 200),
            (200, 200),
            resize_plan,
            "page_full",
        )

        self.assertEqual(len(mapped), 1)
        self.assertEqual(mapped[0].bbox_xyxy, [40, 80, 120, 160])
        self.assertEqual(mapped[0].bbox_xyxy_float, [40.0, 80.0, 120.0, 160.0])
        self.assertEqual(mapped[0].response_bbox_2d, [20.0, 40.0, 60.0, 80.0])
        self.assertEqual(mapped[0].request_shape, [100, 100])
        self.assertEqual(mapped[0].resize_profile, "standard")
        self.assertAlmostEqual(mapped[0].scale_x, 0.5)
        self.assertAlmostEqual(mapped[0].scale_y, 0.5)

    def test_apply_assignments_preserves_detector_geometry_and_records_region_metadata(self) -> None:
        engine = MangaLMMOCREngine()
        blk = _make_block(120, 40, 170, 120, bubble_bbox=(20, 20, 200, 150))
        original_xyxy = blk.xyxy.copy()
        resize_plan = ResizePlan(
            profile="dense",
            original_shape=(3035, 2150),
            request_shape=(1270, 900),
            base_scale=900 / 2150.0,
            scale_x=900 / 2150.0,
            scale_y=1270 / 3035.0,
            max_completion_tokens=768,
            block_count=30,
            small_block_ratio=0.7,
            text_cover_ratio=0.2,
        )
        assignments = {
            0: [
                {
                    "region": OCRRegion(
                        bbox_xyxy=[122, 42, 168, 118],
                        bbox_xyxy_float=[121.7, 41.6, 168.4, 118.2],
                        text="右",
                        unit_bbox_xyxy=[0, 0, 2150, 3035],
                        unit_kind="page_full",
                        unit_resize_scale=resize_plan.base_scale,
                        edge_distance=42.0,
                        normalized_text="右",
                        response_bbox_2d=[51.0, 18.0, 70.0, 49.0],
                        scale_x=resize_plan.scale_x,
                        scale_y=resize_plan.scale_y,
                        request_shape=[1270, 900],
                        resize_profile="dense",
                    ),
                    "metrics": {
                        "ownership_cover": 1.0,
                        "precision_cover": 0.9,
                        "ownership_iou": 0.6,
                        "center_in_ownership": True,
                        "center_in_precision": True,
                        "center_distance_norm": 0.1,
                        "precision_area": 4000,
                    },
                }
            ]
        }

        engine._apply_assignments_to_blocks(
            [blk],
            assignments,
            attempt_count=1,
            success_status="ok",
            empty_status="empty_initial",
            page_bbox=(0, 0, 2150, 3035),
            resize_plan=resize_plan,
        )

        self.assertEqual(blk.text, "右")
        self.assertEqual(blk.xyxy.tolist(), original_xyxy.tolist())
        self.assertEqual(blk.ocr_crop_bbox, [0, 0, 2150, 3035])
        self.assertAlmostEqual(blk.ocr_resize_scale, resize_plan.base_scale)
        self.assertEqual(len(blk.ocr_regions), 1)
        region = blk.ocr_regions[0]
        self.assertEqual(region["bbox_xyxy"], [122, 42, 168, 118])
        self.assertEqual(region["bbox_xyxy_float"], [121.7, 41.6, 168.4, 118.2])
        self.assertEqual(region["request_shape"], [1270, 900])
        self.assertEqual(region["resize_profile"], "dense")
        self.assertAlmostEqual(region["scale_x"], resize_plan.scale_x)
        self.assertAlmostEqual(region["scale_y"], resize_plan.scale_y)

    def test_process_image_does_not_retry_when_page_returns_empty(self) -> None:
        engine = MangaLMMOCREngine()
        blk = _make_block(20, 20, 80, 80, bubble_bbox=(0, 0, 120, 120))
        image = np.zeros((200, 200, 3), dtype=np.uint8)

        with mock.patch.object(engine, "_request_response_text", return_value="") as request_response_text:
            engine.process_image(image, [blk])

        request_response_text.assert_called_once()
        self.assertEqual(blk.text, "")
        self.assertEqual(blk.ocr_status, "empty_initial")
        self.assertEqual(blk.ocr_crop_bbox, [0, 0, 200, 200])
        self.assertEqual(engine.last_request_metadata["resize_profile"], "standard")
        self.assertEqual(engine.last_request_metadata["region_count"], 0)
        self.assertEqual(engine.last_page_regions, [])


if __name__ == "__main__":
    unittest.main()
