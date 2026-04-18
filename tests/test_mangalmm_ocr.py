from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from modules.ocr.factory import OCRFactory
from modules.ocr.mangalmm_ocr import MangaLMMOCREngine, OCRRegion, ResizePlan
from modules.ocr.selection import OCR_MODE_BEST_LOCAL, OCR_MODE_MANGALMM
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
        max_completion_tokens: int = 256,
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

    def test_parse_region_payload_salvages_fenced_json(self) -> None:
        engine = MangaLMMOCREngine()
        payload = """```json
        [{"bbox_2d":[10,20,30,40],"text_content":"テスト"}]
        ```"""
        parsed = engine._parse_region_payload(payload)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["text_content"], "テスト")

    def test_normalize_region_list_strips_decorative_noise_and_drops_empty_regions(self) -> None:
        engine = MangaLMMOCREngine()

        normalized = engine._normalize_region_list(
            [
                {"bbox_2d": [10, 20, 30, 40], "text_content": "⌒テ✺スト︸"},
                {"bbox_2d": [40, 50, 60, 70], "text_content": "⌒✺︸"},
            ]
        )

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["text_content"], "テスト")
        self.assertEqual(normalized[0]["raw_text_content"], "⌒テ✺スト︸")

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
        self.assertEqual(dense_plan.request_shape, (1270, 900))
        self.assertEqual(dense_plan.max_completion_tokens, 256)
        self.assertAlmostEqual(dense_plan.scale_x, 900 / 2150.0)
        self.assertAlmostEqual(dense_plan.scale_y, 1270 / 3035.0)

        self.assertEqual(standard_plan.profile, "standard")
        self.assertEqual(standard_plan.request_shape, (1728, 1224))
        self.assertEqual(standard_plan.max_completion_tokens, 256)
        self.assertAlmostEqual(standard_plan.scale_x, 1224 / 2150.0)
        self.assertAlmostEqual(standard_plan.scale_y, 1728 / 3036.0)

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

    def test_build_attempt_specs_keeps_single_attempt_for_direct_mode(self) -> None:
        engine = MangaLMMOCREngine()
        settings = _FakeSettings(selected_ocr_mode=OCR_MODE_MANGALMM)
        engine.initialize(
            settings,
            source_lang_english="Japanese",
            selected_ocr_mode=OCR_MODE_MANGALMM,
        )

        attempts = engine._build_attempt_specs((3036, 2150, 3), _make_blocks(15, width=200, height=300))

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

    def test_dedupe_assigned_items_for_block_removes_only_near_exact_duplicates(self) -> None:
        engine = MangaLMMOCREngine()
        region_a = OCRRegion(
            bbox_xyxy=[20, 20, 60, 80],
            bbox_xyxy_float=[20.0, 20.0, 60.0, 80.0],
            text="うわ",
            unit_bbox_xyxy=[0, 0, 200, 200],
            unit_kind="page_full",
            unit_resize_scale=1.0,
            edge_distance=20.0,
            normalized_text="うわ",
            response_bbox_2d=[20.0, 20.0, 60.0, 80.0],
        )
        region_b = OCRRegion(
            bbox_xyxy=[21, 21, 61, 81],
            bbox_xyxy_float=[21.0, 21.0, 61.0, 81.0],
            text="うわ",
            unit_bbox_xyxy=[0, 0, 200, 200],
            unit_kind="page_full",
            unit_resize_scale=1.0,
            edge_distance=21.0,
            normalized_text="うわ",
            response_bbox_2d=[21.0, 21.0, 61.0, 81.0],
        )
        region_c = OCRRegion(
            bbox_xyxy=[90, 20, 130, 80],
            bbox_xyxy_float=[90.0, 20.0, 130.0, 80.0],
            text="別",
            unit_bbox_xyxy=[0, 0, 200, 200],
            unit_kind="page_full",
            unit_resize_scale=1.0,
            edge_distance=20.0,
            normalized_text="別",
            response_bbox_2d=[90.0, 20.0, 130.0, 80.0],
        )
        items = [
            {
                "region": region_a,
                "metrics": {
                    "ownership_cover": 0.8,
                    "precision_cover": 0.8,
                    "ownership_iou": 0.5,
                    "center_in_ownership": True,
                    "center_in_precision": True,
                    "center_distance_norm": 0.1,
                    "precision_area": 2400,
                },
            },
            {
                "region": region_b,
                "metrics": {
                    "ownership_cover": 0.9,
                    "precision_cover": 0.9,
                    "ownership_iou": 0.6,
                    "center_in_ownership": True,
                    "center_in_precision": True,
                    "center_distance_norm": 0.05,
                    "precision_area": 2400,
                },
            },
            {
                "region": region_c,
                "metrics": {
                    "ownership_cover": 0.9,
                    "precision_cover": 0.9,
                    "ownership_iou": 0.6,
                    "center_in_ownership": True,
                    "center_in_precision": True,
                    "center_distance_norm": 0.05,
                    "precision_area": 2400,
                },
            },
        ]

        deduped = engine._dedupe_assigned_items_for_block(items)

        self.assertEqual(len(deduped), 2)
        texts = sorted(item["region"].text for item in deduped)
        self.assertEqual(texts, ["うわ", "別"])

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
        self.assertEqual(payload["temperature"], 0.1)
        self.assertEqual(payload["top_k"], 1)
        self.assertEqual(payload["top_p"], 0.001)
        self.assertEqual(payload["min_p"], 0.0)
        self.assertEqual(payload["repeat_penalty"], 1.05)
        self.assertEqual(payload["repeat_last_n"], 0)
        self.assertEqual(post.call_args.kwargs["timeout"], 60.0)

    def test_process_image_single_attempt_text_only_is_failure(self) -> None:
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

        self.assertEqual(request_attempt.call_count, 1)
        self.assertEqual(blk.text, "")
        self.assertEqual(blk.ocr_status, "empty_initial")
        self.assertEqual(engine.last_request_metadata["final_status"], "failure")
        self.assertEqual(engine.last_request_metadata["retry_count"], 0)

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
