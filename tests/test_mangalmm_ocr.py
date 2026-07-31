from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from modules.ocr.factory import OCRFactory
from modules.ocr.mangalmm_ocr import MangaLMMOCREngine, OCRRegion, ResizePlan
from modules.ocr.selection import OCR_MODE_BEST_LOCAL, OCR_MODE_MANGALMM
from modules.utils.exceptions import LocalServiceResponseError
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


class _FakeSettings:
    class ui:
        @staticmethod
        def tr(value: str) -> str:
            return value

    def __init__(
        self,
        *,
        selected_ocr_mode: str = OCR_MODE_MANGALMM,
        max_completion_tokens: int = 4096,
    ) -> None:
        self._selected_ocr_mode = selected_ocr_mode
        self._max_completion_tokens = max_completion_tokens

    def get_tool_selection(self, tool_type: str) -> str:
        if tool_type == "ocr":
            return self._selected_ocr_mode
        raise KeyError(tool_type)

    def get_mangalmm_ocr_settings(self) -> dict:
        return {
            "server_url": "http://127.0.0.1:28081/v1",
            "max_completion_tokens": self._max_completion_tokens,
            "parallel_workers": 1,
            "request_timeout_sec": 60,
            "raw_response_logging": False,
            "safe_resize": True,
            "max_pixels": 2_116_800,
            "max_long_side": 1728,
            "temperature": 0.1,
            "top_k": 1,
            "top_p": 0.001,
            "min_p": 0.0,
            "repeat_penalty": 1.05,
            "repeat_last_n": 0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        }

    def get_credentials(self, _provider_name: str) -> dict:
        return {}

    def is_gpu_enabled(self) -> bool:
        return False


def _make_resize_plan(
    *,
    profile: str,
    request_shape: tuple[int, int],
    original_shape: tuple[int, int] = (3035, 2150),
    max_completion_tokens: int = 1024,
    block_count: int = 30,
    small_block_ratio: float = 0.7,
    text_cover_ratio: float = 0.2,
) -> ResizePlan:
    request_h, request_w = request_shape
    original_h, original_w = original_shape
    return ResizePlan(
        profile=profile,
        original_shape=original_shape,
        request_shape=request_shape,
        base_scale=request_w / float(original_w),
        scale_x=request_w / float(original_w),
        scale_y=request_h / float(original_h),
        max_completion_tokens=max_completion_tokens,
        block_count=block_count,
        small_block_ratio=small_block_ratio,
        text_cover_ratio=text_cover_ratio,
    )


def _make_region(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    text: str,
) -> OCRRegion:
    return OCRRegion(
        bbox_xyxy=[x1, y1, x2, y2],
        bbox_xyxy_float=[
            float(x1),
            float(y1),
            float(x2),
            float(y2),
        ],
        text=text,
        unit_bbox_xyxy=[0, 0, 200, 200],
        unit_kind="page_full",
        unit_resize_scale=1.0,
        edge_distance=0.0,
        normalized_text=text,
        raw_text=text,
        response_bbox_2d=[
            float(x1),
            float(y1),
            float(x2),
            float(y2),
        ],
    )


class MangaLMMOCRTests(unittest.TestCase):
    def setUp(self) -> None:
        OCRFactory._engines.clear()

    def test_initialize_normalizes_legacy_optimal_plus_to_optimal(self) -> None:
        engine = MangaLMMOCREngine()
        settings = _FakeSettings(selected_ocr_mode=OCR_MODE_BEST_LOCAL)

        engine.initialize(
            settings,
            source_lang_english="Japanese",
            selected_ocr_mode="best_local_plus",
        )

        self.assertEqual(engine.selected_ocr_mode, OCR_MODE_BEST_LOCAL)
        self.assertEqual(engine.contract_mode, "direct_manual")

    def test_initialize_keeps_direct_manual_contract_for_direct_mangalmm(self) -> None:
        engine = MangaLMMOCREngine()
        settings = _FakeSettings(selected_ocr_mode=OCR_MODE_MANGALMM)

        engine.initialize(
            settings,
            source_lang_english="Japanese",
            selected_ocr_mode=OCR_MODE_MANGALMM,
        )

        self.assertEqual(engine.contract_mode, "direct_manual")

    def test_parse_region_payload_accepts_one_complete_fenced_array(self) -> None:
        engine = MangaLMMOCREngine()
        payload = """```json
        [{"bbox_2d":[10,20,30,40],"text_content":"テスト"}]
        ```"""
        parsed = engine._parse_region_payload(payload)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["text_content"], "テスト")

    def test_parse_region_payload_preserves_decorative_text_for_role_routing(
        self,
    ) -> None:
        engine = MangaLMMOCREngine()

        parsed = engine._parse_region_payload(
            '[{"bbox_2d":[10,20,30,40],'
            '"text_content":"⌒テ✺スト︸"}]'
        )

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["text_content"], "⌒テ✺スト︸")
        self.assertEqual(parsed[0]["raw_text_content"], "⌒テ✺スト︸")

    def test_analyze_region_payload_reports_strict_parser_error(self) -> None:
        engine = MangaLMMOCREngine()

        analysis = engine._analyze_region_payload(
            '[{"bbox_2d":[10,20,30,40],'
            '"text_content":"テスト"}] trailing'
        )

        self.assertEqual(analysis["regions"], [])
        self.assertEqual(
            analysis["response_kind"],
            "parser_error:invalid_json",
        )
        self.assertEqual(analysis["parser_error_code"], "invalid_json")
        self.assertIn("complete JSON value", analysis["parser_error"])

    def test_request_attempt_records_parser_diagnostics(self) -> None:
        engine = MangaLMMOCREngine()
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        block = _make_block(20, 20, 80, 80)
        unit = engine._build_request_units(image.shape)[0]
        attempt = engine._build_attempt_specs(image.shape, [block])[0]

        with mock.patch.object(
            engine,
            "_request_response_text",
            return_value=(
                '{"regions":[{"bbox_2d":[10,20,30,40],'
                '"text_content":"テスト"}]}'
            ),
        ):
            result = engine._request_regions_for_attempt(
                image,
                unit,
                attempt,
            )

        self.assertEqual(result["regions"], [])
        self.assertEqual(
            result["metadata"]["response_kind"],
            "parser_error:top_level_not_array",
        )
        self.assertEqual(
            result["metadata"]["parser_error_code"],
            "top_level_not_array",
        )

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

    def test_plan_page_request_uses_manual_limits_in_direct_mode(self) -> None:
        engine = MangaLMMOCREngine()
        settings = _FakeSettings(selected_ocr_mode=OCR_MODE_MANGALMM)
        engine.initialize(
            settings,
            source_lang_english="Japanese",
            selected_ocr_mode=OCR_MODE_MANGALMM,
        )

        dense_plan = engine._plan_page_request((3035, 2150, 3), _make_blocks(30, width=160, height=180))
        standard_plan = engine._plan_page_request((3036, 2150, 3), _make_blocks(15, width=200, height=300))

        self.assertEqual(dense_plan.profile, "dense")
        self.assertEqual(dense_plan.request_shape, (1708, 1204))
        self.assertEqual(dense_plan.max_completion_tokens, 4096)
        self.assertEqual(dense_plan.alignment_factor, 28)
        self.assertLessEqual(
            dense_plan.request_shape[0] * dense_plan.request_shape[1],
            dense_plan.effective_max_pixels,
        )
        self.assertAlmostEqual(dense_plan.scale_x, 1204 / 2150.0)
        self.assertAlmostEqual(dense_plan.scale_y, 1708 / 3035.0)

        self.assertEqual(standard_plan.profile, "standard")
        self.assertEqual(standard_plan.request_shape, (1708, 1204))
        self.assertEqual(standard_plan.max_completion_tokens, 4096)
        self.assertAlmostEqual(standard_plan.scale_x, 1204 / 2150.0)
        self.assertAlmostEqual(standard_plan.scale_y, 1708 / 3036.0)

    def test_official_smart_resize_matches_qwen_factor_and_pixel_contract(self) -> None:
        engine = MangaLMMOCREngine()

        self.assertEqual(
            engine._official_smart_resize(
                1920,
                1360,
                max_pixels=2_116_800,
            ),
            (1708, 1204),
        )
        self.assertEqual(
            engine._official_smart_resize(
                1600,
                1200,
                max_pixels=2_116_800,
            ),
            (1596, 1204),
        )
        tiny_shape = engine._official_smart_resize(
            20,
            20,
            max_pixels=2_116_800,
        )
        self.assertEqual(tiny_shape, (56, 56))
        for height, width in (
            (1708, 1204),
            (1596, 1204),
            tiny_shape,
        ):
            self.assertEqual(height % 28, 0)
            self.assertEqual(width % 28, 0)
            self.assertLessEqual(height * width, 2_116_800)

    def test_official_smart_resize_rejects_unsupported_extreme_aspect_ratio(self) -> None:
        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            MangaLMMOCREngine._official_smart_resize(
                20,
                5000,
                max_pixels=2_116_800,
            )

    def test_plan_page_request_respects_manual_token_limit_in_direct_mode(self) -> None:
        engine = MangaLMMOCREngine()
        settings = _FakeSettings(selected_ocr_mode=OCR_MODE_MANGALMM, max_completion_tokens=320)
        engine.initialize(
            settings,
            source_lang_english="Japanese",
            selected_ocr_mode=OCR_MODE_MANGALMM,
        )

        standard_plan = engine._plan_page_request((3036, 2150, 3), _make_blocks(15, width=200, height=300))

        self.assertEqual(standard_plan.max_completion_tokens, 320)

    def test_build_attempt_specs_adds_one_native_output_recovery(self) -> None:
        engine = MangaLMMOCREngine()
        settings = _FakeSettings(selected_ocr_mode=OCR_MODE_MANGALMM)
        engine.initialize(
            settings,
            source_lang_english="Japanese",
            selected_ocr_mode=OCR_MODE_MANGALMM,
        )

        attempts = engine._build_attempt_specs((3036, 2150, 3), _make_blocks(15, width=200, height=300))

        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0].attempt_kind, "primary")
        self.assertEqual(
            attempts[1].attempt_kind,
            "native_output_recovery",
        )
        self.assertIn(
            "Do not return an empty response",
            attempts[1].prompt_text,
        )
        self.assertEqual(attempts[1].repeat_penalty, 1.15)
        self.assertEqual(attempts[1].repeat_last_n, 4096)

    def test_sparse_recovery_keeps_official_prompt_without_recall_suffix(
        self,
    ) -> None:
        engine = MangaLMMOCREngine()
        settings = _FakeSettings(selected_ocr_mode=OCR_MODE_MANGALMM)
        engine.initialize(
            settings,
            source_lang_english="Japanese",
            selected_ocr_mode=OCR_MODE_MANGALMM,
        )

        attempts = engine._build_attempt_specs(
            (2000, 1430, 3),
            _make_blocks(2, width=120, height=300),
        )

        self.assertEqual(len(attempts), 2)
        self.assertEqual(
            attempts[1].attempt_kind,
            "sparse_native_output_recovery",
        )
        self.assertEqual(attempts[1].prompt_text, attempts[0].prompt_text)
        self.assertNotIn(
            "Output every distinct physical text region",
            attempts[1].prompt_text,
        )
        self.assertEqual(attempts[1].repeat_penalty, 1.15)
        self.assertEqual(attempts[1].repeat_last_n, 4096)

    def test_build_attempt_specs_keeps_one_attempt_without_detector_blocks(
        self,
    ) -> None:
        engine = MangaLMMOCREngine()
        settings = _FakeSettings(selected_ocr_mode=OCR_MODE_MANGALMM)
        engine.initialize(
            settings,
            source_lang_english="Japanese",
            selected_ocr_mode=OCR_MODE_MANGALMM,
        )

        attempts = engine._build_attempt_specs((3036, 2150, 3), [])

        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0].attempt_kind, "primary")

    def test_map_regions_to_page_coords_restores_original_coordinates_with_scale_axes(self) -> None:
        engine = MangaLMMOCREngine()
        resize_plan = ResizePlan(
            profile="standard",
            original_shape=(200, 200),
            request_shape=(100, 100),
            base_scale=0.5,
            scale_x=0.5,
            scale_y=0.5,
            max_completion_tokens=2048,
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
        resize_plan = _make_resize_plan(
            profile="dense",
            request_shape=(1270, 900),
            max_completion_tokens=1024,
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
                        raw_text="⌒右✺",
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
        self.assertEqual(blk.ocr_raw_text, "⌒右✺")
        self.assertEqual(blk.ocr_sanitized_text, "右")
        self.assertEqual(blk.xyxy.tolist(), original_xyxy.tolist())
        self.assertEqual(blk.ocr_crop_bbox, [0, 0, 2150, 3035])
        self.assertAlmostEqual(blk.ocr_resize_scale, resize_plan.base_scale)
        self.assertEqual(len(blk.ocr_regions), 1)
        region = blk.ocr_regions[0]
        self.assertEqual(region["bbox_xyxy"], [122, 42, 168, 118])
        self.assertEqual(region["bbox_xyxy_float"], [121.7, 41.6, 168.4, 118.2])
        self.assertEqual(region["raw_text"], "⌒右✺")
        self.assertEqual(region["request_shape"], [1270, 900])
        self.assertEqual(region["resize_profile"], "dense")
        self.assertAlmostEqual(region["scale_x"], resize_plan.scale_x)
        self.assertAlmostEqual(region["scale_y"], resize_plan.scale_y)

    def test_one_region_covering_multiple_detector_blocks_is_shadow_only(
        self,
    ) -> None:
        engine = MangaLMMOCREngine()
        blocks = [
            _make_block(10, 10, 50, 80, text_class="text_free"),
            _make_block(40, 10, 80, 80, text_class="text_free"),
        ]

        assignments = engine._assign_regions_to_blocks(
            [_make_region(30, 10, 60, 80, "joined")],
            blocks,
        )

        self.assertEqual(assignments, {0: [], 1: []})
        self.assertEqual(len(engine.last_shadow_regions), 1)
        self.assertEqual(
            engine.last_shadow_regions[0]["reason"],
            "one_region_multiple_blocks",
        )
        self.assertEqual(len(engine.last_merge_split_diagnostics), 1)
        self.assertIn(
            "mangalmm_one_region_multiple_blocks",
            blocks[0].merge_split_diagnostics,
        )
        self.assertIn(
            "mangalmm_one_region_multiple_blocks",
            blocks[1].merge_split_diagnostics,
        )

    def test_long_region_covering_separated_blocks_is_explicit_review(
        self,
    ) -> None:
        engine = MangaLMMOCREngine()
        blocks = [
            _make_block(50, 10, 90, 40, text_class="text_free"),
            _make_block(40, 150, 100, 190, text_class="text_free"),
        ]

        assignments = engine._assign_regions_to_blocks(
            [_make_region(40, 5, 100, 195, "merged page text")],
            blocks,
        )

        self.assertEqual(assignments, {0: [], 1: []})
        self.assertEqual(len(engine.last_shadow_regions), 1)
        self.assertEqual(
            engine.last_shadow_regions[0]["reason"],
            "one_region_multiple_blocks_coverage",
        )
        diagnostic = engine.last_merge_split_diagnostics[0]
        self.assertEqual(
            diagnostic["kind"],
            "one_region_multiple_blocks_coverage",
        )
        self.assertEqual(diagnostic["status"], "review")
        self.assertEqual(len(diagnostic["candidate_block_ids"]), 2)
        self.assertIn(
            "mangalmm_one_region_multiple_blocks_coverage",
            blocks[0].merge_split_diagnostics,
        )

    def test_safe_multiple_regions_form_one_ordered_detector_compound(
        self,
    ) -> None:
        engine = MangaLMMOCREngine()
        block = _make_block(
            0,
            0,
            100,
            100,
            text_class="text_bubble",
            bubble_bbox=(0, 0, 100, 100),
        )
        block.direction = "vertical"

        assignments = engine._assign_regions_to_blocks(
            [
                _make_region(10, 50, 30, 90, "left"),
                _make_region(60, 10, 90, 40, "right"),
            ],
            [block],
        )

        self.assertEqual(
            [item["region"].text for item in assignments[0]],
            ["right", "left"],
        )
        self.assertEqual(engine.last_shadow_regions, [])
        self.assertEqual(
            engine.last_merge_split_diagnostics[0]["kind"],
            "multiple_regions_one_block_compound",
        )
        self.assertIn(
            "mangalmm_multiple_regions_one_block_compound",
            block.merge_split_diagnostics,
        )

        resize_plan = _make_resize_plan(
            profile="standard",
            request_shape=(100, 100),
            original_shape=(100, 100),
        )
        engine._apply_assignments_to_blocks(
            [block],
            assignments,
            attempt_count=1,
            success_status="ok",
            empty_status="empty_initial",
            page_bbox=(0, 0, 100, 100),
            resize_plan=resize_plan,
        )
        self.assertEqual(block.texts, ["right", "left"])
        self.assertEqual(block.text, "right\nleft")
        self.assertEqual(len(block.ocr_regions), 2)
        self.assertEqual(block.compound_group_id, block.block_id)

    def test_distinct_free_text_regions_without_bubble_fail_closed(
        self,
    ) -> None:
        engine = MangaLMMOCREngine()
        block = _make_block(0, 0, 100, 100, text_class="text_free")

        assignments = engine._assign_regions_to_blocks(
            [
                _make_region(10, 10, 30, 40, "パル"),
                _make_region(60, 50, 90, 90, "パル"),
            ],
            [block],
        )

        self.assertEqual(assignments, {0: []})
        self.assertEqual(len(engine.last_shadow_regions), 2)
        self.assertTrue(
            all(
                item["reason"]
                == "multiple_regions_one_block_no_bubble"
                for item in engine.last_shadow_regions
            )
        )
        self.assertEqual(
            engine.last_merge_split_diagnostics[0]["decision_reason"],
            "missing_bubble_compound_boundary",
        )

    def test_overlapping_distinct_regions_for_one_block_fail_closed(
        self,
    ) -> None:
        engine = MangaLMMOCREngine()
        block = _make_block(
            0,
            0,
            100,
            100,
            text_class="text_bubble",
            bubble_bbox=(0, 0, 100, 100),
        )

        assignments = engine._assign_regions_to_blocks(
            [
                _make_region(10, 10, 80, 80, "first"),
                _make_region(40, 30, 90, 90, "conflict"),
            ],
            [block],
        )

        self.assertEqual(assignments, {0: []})
        self.assertEqual(len(engine.last_shadow_regions), 2)
        self.assertTrue(
            all(
                item["reason"]
                == "multiple_regions_one_block_overlap"
                for item in engine.last_shadow_regions
            )
        )
        self.assertEqual(
            engine.last_merge_split_diagnostics[0]["status"],
            "review",
        )

    def test_near_duplicate_regions_are_collapsed_before_compounding(
        self,
    ) -> None:
        engine = MangaLMMOCREngine()
        block = _make_block(0, 0, 100, 100, text_class="text_bubble")

        assignments = engine._assign_regions_to_blocks(
            [
                _make_region(10, 10, 50, 80, "same"),
                _make_region(12, 12, 49, 79, "same"),
            ],
            [block],
        )

        self.assertEqual(len(assignments[0]), 1)
        self.assertEqual(assignments[0][0]["region"].text, "same")
        self.assertEqual(engine.last_shadow_regions, [])
        self.assertEqual(
            engine.last_merge_split_diagnostics[0]["kind"],
            "near_duplicate_regions_collapsed",
        )

    def test_exact_duplicate_regions_are_assigned_once(self) -> None:
        engine = MangaLMMOCREngine()
        block = _make_block(0, 0, 100, 100, text_class="text_free")
        region = _make_region(10, 10, 30, 40, "same")

        assignments = engine._assign_regions_to_blocks(
            [region, region],
            [block],
        )

        self.assertEqual(len(assignments[0]), 1)
        self.assertEqual(engine.last_shadow_regions, [])
        self.assertEqual(engine.last_merge_split_diagnostics, [])

    def test_manga_only_region_stays_shadow_candidate(self) -> None:
        engine = MangaLMMOCREngine()
        block = _make_block(0, 0, 20, 20, text_class="text_free")

        assignments = engine._assign_regions_to_blocks(
            [_make_region(120, 120, 150, 160, "shadow")],
            [block],
        )

        self.assertEqual(assignments, {0: []})
        self.assertEqual(
            engine.last_shadow_regions[0]["reason"],
            "manga_only_unmatched",
        )

    def test_single_precision_candidate_wins_over_bubble_only_candidate(
        self,
    ) -> None:
        engine = MangaLMMOCREngine()
        broad_bubble = _make_block(
            120,
            120,
            150,
            150,
            bubble_bbox=(0, 0, 180, 180),
        )
        precise_text = _make_block(20, 20, 60, 80)

        assignments = engine._assign_regions_to_blocks(
            [_make_region(22, 22, 58, 78, "precise")],
            [broad_bubble, precise_text],
        )

        self.assertEqual(assignments[0], [])
        self.assertEqual(len(assignments[1]), 1)
        self.assertEqual(
            assignments[1][0]["region"].text,
            "precise",
        )

    def test_multiple_bubble_only_candidates_are_shadowed_as_ambiguous(
        self,
    ) -> None:
        engine = MangaLMMOCREngine()
        blocks = [
            _make_block(
                5,
                5,
                15,
                15,
                bubble_bbox=(0, 0, 100, 100),
            ),
            _make_block(
                85,
                85,
                95,
                95,
                bubble_bbox=(0, 0, 100, 100),
            ),
        ]

        assignments = engine._assign_regions_to_blocks(
            [_make_region(30, 30, 70, 70, "ambiguous")],
            blocks,
        )

        self.assertEqual(assignments, {0: [], 1: []})
        self.assertEqual(len(engine.last_shadow_regions), 1)
        self.assertEqual(
            engine.last_shadow_regions[0]["reason"],
            "one_region_multiple_blocks",
        )
        self.assertEqual(
            len(
                engine.last_shadow_regions[0][
                    "candidate_block_ids"
                ]
            ),
            2,
        )

    def test_zero_detector_blocks_still_records_manga_only_regions(
        self,
    ) -> None:
        engine = MangaLMMOCREngine()
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        region = _make_region(20, 20, 60, 80, "shadow")
        attempt_payload = {
            "regions": [region],
            "analysis": {
                "response_kind": "json_array",
                "payload_type": "json_array",
            },
            "raw_text": (
                '[{"bbox_2d":[20,20,60,80],'
                '"text_content":"shadow"}]'
            ),
            "crop_image": image,
            "request_image": image,
            "parsed_region_count": 1,
            "mapped_region_count": 1,
            "metadata": {"response_kind": "json_array"},
        }

        with mock.patch.object(
            engine,
            "_request_regions_for_attempt",
            return_value=attempt_payload,
        ) as request_attempt:
            result = engine.process_image(image, [])

        self.assertEqual(result, [])
        self.assertEqual(request_attempt.call_count, 1)
        self.assertEqual(len(engine.last_shadow_regions), 1)
        self.assertEqual(
            engine.last_shadow_regions[0]["reason"],
            "manga_only_unmatched",
        )
        self.assertEqual(
            engine.last_request_metadata["failure_reason"],
            "no_block_match",
        )

    def test_prompt_for_resize_plan_uses_standard_and_dense_variants(self) -> None:
        engine = MangaLMMOCREngine()
        standard_plan = _make_resize_plan(
            profile="standard",
            request_shape=(1728, 1224),
            original_shape=(3036, 2150),
            max_completion_tokens=2048,
            block_count=15,
            small_block_ratio=0.2,
            text_cover_ratio=0.1,
        )
        dense_plan = _make_resize_plan(
            profile="dense",
            request_shape=(1270, 900),
            max_completion_tokens=1024,
        )

        self.assertEqual(
            engine._prompt_for_resize_plan(standard_plan),
            ("standard_grounding", engine.STANDARD_PROMPT),
        )
        self.assertEqual(
            engine._prompt_for_resize_plan(dense_plan),
            ("dense_grounding_json", engine.DENSE_PROMPT),
        )

    def test_request_response_text_sends_image_first_with_selected_prompt(self) -> None:
        engine = MangaLMMOCREngine()
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        response = mock.Mock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": "[]"}}],
        }

        with mock.patch("modules.ocr.mangalmm_ocr.requests.post", return_value=response) as post:
            raw = engine._request_response_text(
                image,
                max_completion_tokens=1024,
                prompt_text=engine.DENSE_PROMPT,
            )

        self.assertEqual(raw, "[]")
        payload = post.call_args.kwargs["json"]
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "image_url")
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(content[1], {"type": "text", "text": engine.DENSE_PROMPT})
        self.assertEqual(payload["max_completion_tokens"], 1024)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["top_k"], 40)
        self.assertEqual(payload["top_p"], 0.95)
        self.assertEqual(payload["min_p"], 0.05)
        self.assertEqual(payload["repeat_penalty"], 1.0)
        self.assertEqual(payload["repeat_last_n"], 64)
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(post.call_args.kwargs["timeout"], 60.0)

    def test_request_response_text_applies_recovery_repeat_overrides(
        self,
    ) -> None:
        engine = MangaLMMOCREngine()
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        response = mock.Mock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": "[]"}}],
        }

        with mock.patch(
            "modules.ocr.mangalmm_ocr.requests.post",
            return_value=response,
        ) as post:
            engine._request_response_text(
                image,
                max_completion_tokens=4096,
                prompt_text=engine.STANDARD_PROMPT,
                repeat_penalty_override=1.05,
                repeat_last_n_override=4096,
            )

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["repeat_penalty"], 1.05)
        self.assertEqual(payload["repeat_last_n"], 4096)

    def test_request_response_text_records_finish_reason_and_usage(
        self,
    ) -> None:
        engine = MangaLMMOCREngine()
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        response = mock.Mock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "[]"},
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
        }

        with mock.patch(
            "modules.ocr.mangalmm_ocr.requests.post",
            return_value=response,
        ):
            engine._request_response_text(
                image,
                max_completion_tokens=4096,
                prompt_text=engine.STANDARD_PROMPT,
            )

        self.assertEqual(
            engine.last_transport_metadata,
            {
                "finish_reason": "stop",
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
        )

    def test_request_response_text_rejects_non_object_response(
        self,
    ) -> None:
        engine = MangaLMMOCREngine()
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        response = mock.Mock()
        response.status_code = 200
        response.json.return_value = []

        with mock.patch(
            "modules.ocr.mangalmm_ocr.requests.post",
            return_value=response,
        ), self.assertRaises(LocalServiceResponseError):
            engine._request_response_text(
                image,
                max_completion_tokens=4096,
                prompt_text=engine.STANDARD_PROMPT,
            )

    def test_request_response_text_rejects_non_object_choice(
        self,
    ) -> None:
        engine = MangaLMMOCREngine()
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        response = mock.Mock()
        response.status_code = 200
        response.json.return_value = {"choices": ["invalid"]}

        with mock.patch(
            "modules.ocr.mangalmm_ocr.requests.post",
            return_value=response,
        ), self.assertRaises(LocalServiceResponseError):
            engine._request_response_text(
                image,
                max_completion_tokens=4096,
                prompt_text=engine.STANDARD_PROMPT,
            )

    def test_runtime_identity_is_hashed_without_exposing_endpoint(
        self,
    ) -> None:
        engine = MangaLMMOCREngine()
        credentials = ":".join(("user", "secret"))
        engine.server_url = f"https://{credentials}@example.test/v1"
        block = _make_block(20, 20, 80, 80)

        with mock.patch.object(
            engine,
            "_request_regions_for_attempt",
            return_value={
                "regions": [],
                "analysis": {
                    "response_kind": "json_array",
                    "payload_type": "json_array",
                },
                "raw_text": "[]",
                "crop_image": np.zeros((16, 16, 3), dtype=np.uint8),
                "request_image": np.zeros((16, 16, 3), dtype=np.uint8),
                "parsed_region_count": 0,
                "mapped_region_count": 0,
                "metadata": {"response_kind": "json_array"},
            },
        ):
            engine.process_image(
                np.zeros((100, 100, 3), dtype=np.uint8),
                [block],
            )

        self.assertEqual(len(block.ocr_runtime_identity), 64)
        self.assertNotIn("secret", block.ocr_runtime_identity)
        self.assertNotIn("example.test", block.ocr_runtime_identity)
        self.assertEqual(
            block.ocr_geometry_provenance[
                "reconciliation_schema_version"
            ],
            2,
        )

    def test_finish_reason_length_rejects_otherwise_valid_json(self) -> None:
        engine = MangaLMMOCREngine()
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        block = _make_block(20, 20, 80, 80)
        unit = engine._build_request_units(image.shape)[0]
        attempt = engine._build_attempt_specs(image.shape, [block])[0]

        def truncated_response(*_args, **_kwargs):
            engine.last_transport_metadata = {
                "finish_reason": "length",
                "prompt_tokens": 100,
                "completion_tokens": 4096,
                "total_tokens": 4196,
            }
            return (
                '[{"bbox_2d":[20,20,80,80],'
                '"text_content":"truncated"}]'
            )

        with mock.patch.object(
            engine,
            "_request_response_text",
            side_effect=truncated_response,
        ):
            result = engine._request_regions_for_attempt(
                image,
                unit,
                attempt,
            )

        self.assertEqual(result["regions"], [])
        self.assertEqual(
            result["metadata"]["parser_error_code"],
            "finish_reason_length",
        )
        self.assertEqual(result["metadata"]["finish_reason"], "length")
        self.assertEqual(result["metadata"]["completion_tokens"], 4096)

    def test_engine_debug_env_is_routed_to_active_sidecar_runtime(self) -> None:
        engine = MangaLMMOCREngine()
        plan = _make_resize_plan(
            profile="standard",
            request_shape=(32, 32),
            original_shape=(32, 32),
        )
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            legacy_root = Path(temporary) / "legacy"
            legacy_root.mkdir()
            active_root = Path(temporary) / "runtime" / "mangalmm-engine"
            active_root.mkdir(parents=True)
            engine.debug_root = legacy_root
            engine._debug_root_from_env = True
            with mock.patch(
                "modules.ocr.mangalmm_ocr."
                "active_debug_runtime_directory",
                return_value=str(active_root),
            ):
                engine._export_debug_artifact(
                    blk=None,
                    failure_reason="test",
                    response_kind="../unsafe",
                    raw_text="raw",
                    crop_bbox=(0, 0, 32, 32),
                    crop_source="page",
                    resize_plan=plan,
                    crop_image=image,
                    request_image=image,
                    analysis={"status": "test"},
                )

            artifacts = list(active_root.iterdir())
            self.assertEqual(len(artifacts), 1)
            self.assertNotIn("..", artifacts[0].name)
            self.assertTrue((artifacts[0] / "meta.json").is_file())
            self.assertTrue((artifacts[0] / "crop.png").is_file())
            self.assertTrue((artifacts[0] / "request.png").is_file())
            self.assertEqual(list(legacy_root.iterdir()), [])

    def test_engine_debug_env_does_not_fall_back_outside_sidecar(self) -> None:
        engine = MangaLMMOCREngine()
        plan = _make_resize_plan(
            profile="standard",
            request_shape=(32, 32),
            original_shape=(32, 32),
        )
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            legacy_root = Path(temporary) / "legacy"
            engine.debug_root = legacy_root
            engine._debug_root_from_env = True
            with mock.patch(
                "modules.ocr.mangalmm_ocr."
                "active_debug_runtime_directory",
                return_value="",
            ):
                engine._export_debug_artifact(
                    blk=None,
                    failure_reason="test",
                    response_kind="missing-sidecar",
                    raw_text="raw",
                    crop_bbox=(0, 0, 32, 32),
                    crop_source="page",
                    resize_plan=plan,
                    crop_image=image,
                    request_image=image,
                    analysis={"status": "test"},
                )

            self.assertFalse(legacy_root.exists())

    def test_process_image_bounded_recovery_text_only_is_failure(self) -> None:
        engine = MangaLMMOCREngine()
        settings = _FakeSettings(selected_ocr_mode=OCR_MODE_MANGALMM)
        engine.initialize(
            settings,
            source_lang_english="Japanese",
            selected_ocr_mode=OCR_MODE_MANGALMM,
        )
        blk = _make_block(20, 20, 80, 80, bubble_bbox=(0, 0, 120, 120))
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        failure_payload = {
            "regions": [],
            "analysis": {"response_kind": "plain_text_or_non_json", "payload_type": "text"},
            "raw_text": "テキストだけ",
            "crop_image": image,
            "request_image": image,
            "parsed_region_count": 0,
            "mapped_region_count": 0,
            "metadata": {"response_kind": "plain_text_or_non_json", "prompt_mode": "standard_grounding"},
        }

        with mock.patch.object(engine, "_request_regions_for_attempt", return_value=failure_payload) as request_attempt:
            engine.process_image(image, [blk])

        self.assertEqual(request_attempt.call_count, 2)
        self.assertEqual(blk.text, "")
        self.assertEqual(blk.ocr_status, "empty_after_retry")
        self.assertEqual(engine.last_request_metadata["final_status"], "failure")
        self.assertEqual(engine.last_request_metadata["retry_count"], 1)

    def test_process_image_retries_clear_detector_coverage_gap_and_uses_better_result(
        self,
    ) -> None:
        engine = MangaLMMOCREngine()
        settings = _FakeSettings(selected_ocr_mode=OCR_MODE_MANGALMM)
        engine.initialize(
            settings,
            source_lang_english="Japanese",
            selected_ocr_mode=OCR_MODE_MANGALMM,
        )
        blocks = [
            _make_block(10, 10, 40, 40),
            _make_block(70, 10, 100, 40),
            _make_block(130, 10, 160, 40),
        ]
        image = np.zeros((200, 200, 3), dtype=np.uint8)

        def payload(regions: list[OCRRegion], label: str) -> dict:
            return {
                "regions": regions,
                "analysis": {
                    "response_kind": "json_array",
                    "payload_type": "json_array",
                },
                "raw_text": label,
                "crop_image": image,
                "request_image": image,
                "parsed_region_count": len(regions),
                "mapped_region_count": len(regions),
                "metadata": {
                    "response_kind": "json_array",
                    "raw_response": label,
                },
            }

        first = [_make_region(10, 10, 40, 40, "first")]
        recovered = [
            _make_region(10, 10, 40, 40, "first"),
            _make_region(70, 10, 100, 40, "second"),
        ]
        with mock.patch.object(
            engine,
            "_request_regions_for_attempt",
            side_effect=[
                payload(first, "primary"),
                payload(recovered, "recovery"),
            ],
        ) as request_attempt:
            engine.process_image(image, blocks)

        self.assertEqual(request_attempt.call_count, 2)
        self.assertEqual([block.text for block in blocks], ["first", "second", ""])
        self.assertEqual(blocks[0].ocr_status, "ok_after_retry")
        self.assertEqual(blocks[2].ocr_status, "empty_after_retry")
        self.assertEqual(
            engine.last_attempt_history[0]["failure_reason"],
            "detector_coverage_gap",
        )
        self.assertEqual(engine.last_request_metadata["matched_block_count"], 2)
        self.assertEqual(engine.last_request_metadata["retry_count"], 1)
        self.assertEqual(engine.last_request_metadata["raw_response"], "recovery")
        self.assertEqual(
            engine.last_request_metadata["detector_delivery_status"],
            "success",
        )

    def test_terminal_low_coverage_is_reported_as_partial_delivery(
        self,
    ) -> None:
        engine = MangaLMMOCREngine()
        blocks = [
            _make_block(index * 20, 10, index * 20 + 10, 30)
            for index in range(10)
        ]
        image = np.zeros((200, 240, 3), dtype=np.uint8)
        empty_payload = {
            "regions": [],
            "analysis": {
                "response_kind": "parser_error:empty_response",
                "payload_type": "invalid",
            },
            "raw_text": "",
            "crop_image": image,
            "request_image": image,
            "parsed_region_count": 0,
            "mapped_region_count": 0,
            "metadata": {
                "response_kind": "parser_error:empty_response",
                "finish_reason": "stop",
            },
        }
        region = _make_region(0, 10, 10, 30, "one")
        partial_payload = {
            "regions": [region],
            "analysis": {
                "response_kind": "json_array",
                "payload_type": "json_array",
            },
            "raw_text": "partial",
            "crop_image": image,
            "request_image": image,
            "parsed_region_count": 1,
            "mapped_region_count": 1,
            "metadata": {
                "response_kind": "json_array",
                "finish_reason": "stop",
                "raw_response": "partial",
            },
        }

        with mock.patch.object(
            engine,
            "_request_regions_for_attempt",
            side_effect=[empty_payload, partial_payload],
        ):
            engine.process_image(image, blocks)

        self.assertEqual(
            engine.last_attempt_history[-1]["detector_delivery_status"],
            "partial",
        )
        self.assertEqual(
            engine.last_attempt_history[-1]["failure_reason"],
            "detector_coverage_gap",
        )
        self.assertEqual(
            engine.last_request_metadata["detector_delivery_status"],
            "partial",
        )
        self.assertEqual(engine.last_request_metadata["final_status"], "success")

    def test_process_image_does_not_retry_at_detector_coverage_threshold(
        self,
    ) -> None:
        engine = MangaLMMOCREngine()
        blocks = [
            _make_block(10, 10, 40, 40),
            _make_block(70, 10, 100, 40),
            _make_block(10, 70, 40, 100),
            _make_block(70, 70, 100, 100),
        ]
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        regions = [
            _make_region(10, 10, 40, 40, "first"),
            _make_region(70, 10, 100, 40, "second"),
        ]
        attempt_payload = {
            "regions": regions,
            "analysis": {
                "response_kind": "json_array",
                "payload_type": "json_array",
            },
            "raw_text": "primary",
            "crop_image": image,
            "request_image": image,
            "parsed_region_count": len(regions),
            "mapped_region_count": len(regions),
            "metadata": {
                "response_kind": "json_array",
                "raw_response": "primary",
            },
        }

        with mock.patch.object(
            engine,
            "_request_regions_for_attempt",
            return_value=attempt_payload,
        ) as request_attempt:
            engine.process_image(image, blocks)

        self.assertEqual(request_attempt.call_count, 1)
        self.assertEqual(engine.last_request_metadata["retry_count"], 0)
        self.assertFalse(engine.last_attempt_history[0]["coverage_gap"])

    def test_process_image_keeps_primary_when_coverage_recovery_is_worse(
        self,
    ) -> None:
        engine = MangaLMMOCREngine()
        blocks = [
            _make_block(10, 10, 40, 40),
            _make_block(70, 10, 100, 40),
            _make_block(130, 10, 160, 40),
        ]
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        first_region = _make_region(10, 10, 40, 40, "primary text")
        primary_payload = {
            "regions": [first_region],
            "analysis": {
                "response_kind": "json_array",
                "payload_type": "json_array",
            },
            "raw_text": "primary",
            "crop_image": image,
            "request_image": image,
            "parsed_region_count": 1,
            "mapped_region_count": 1,
            "metadata": {
                "response_kind": "json_array",
                "raw_response": "primary",
            },
        }
        failed_recovery = {
            "regions": [],
            "analysis": {
                "response_kind": "parser_error:invalid_json",
                "payload_type": "invalid",
            },
            "raw_text": "broken",
            "crop_image": image,
            "request_image": image,
            "parsed_region_count": 0,
            "mapped_region_count": 0,
            "metadata": {
                "response_kind": "parser_error:invalid_json",
                "raw_response": "broken",
            },
        }

        with mock.patch.object(
            engine,
            "_request_regions_for_attempt",
            side_effect=[primary_payload, failed_recovery],
        ):
            engine.process_image(image, blocks)

        self.assertEqual(blocks[0].text, "primary text")
        self.assertEqual(blocks[0].ocr_status, "ok_after_retry")
        self.assertEqual(engine.last_request_metadata["retry_count"], 1)
        self.assertEqual(engine.last_request_metadata["raw_response"], "primary")

    def test_create_cache_key_normalizes_legacy_optimal_plus_value(self) -> None:
        settings = _FakeSettings()

        legacy_key = OCRFactory._create_cache_key(
            "MangaLMM",
            "Japanese",
            settings,
            selected_ocr_mode="best_local_plus",
        )
        optimal_key = OCRFactory._create_cache_key(
            "MangaLMM",
            "Japanese",
            settings,
            selected_ocr_mode=OCR_MODE_BEST_LOCAL,
        )

        self.assertEqual(legacy_key, optimal_key)


if __name__ == "__main__":
    unittest.main()
