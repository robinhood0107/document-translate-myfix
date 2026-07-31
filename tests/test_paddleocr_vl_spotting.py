from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from modules.ocr.paddleocr_vl_spotting.engine import PaddleOCRVLSpottingEngine
from modules.ocr.result_contract import (
    OCR_STRATEGY_PADDLE_SPOTTING,
    PROCESSING_ACTION_REVIEW,
)
from modules.utils.textblock import TextBlock


def _native_line(
    text: str,
    coordinates: tuple[int, ...],
) -> str:
    return "".join(f"<|LOC_{value}|>" for value in coordinates) + text


def _response(content: str, *, finish_reason: str = "stop") -> dict:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


def _block(
    bbox: tuple[int, int, int, int],
    block_id: str,
) -> TextBlock:
    return TextBlock(
        text_bbox=np.asarray(bbox, dtype=np.int32),
        text_class="text_bubble",
        block_id=block_id,
    )


class PaddleOCRVLSpottingEngineTests(unittest.TestCase):
    def test_primary_full_page_request_uses_official_prompt_and_contract(
        self,
    ) -> None:
        engine = PaddleOCRVLSpottingEngine()
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        blocks = [_block((100, 100, 300, 300), "block-a")]
        content = _native_line(
            "よし",
            (100, 100, 300, 100, 300, 300, 100, 300),
        )

        with mock.patch.object(
            engine,
            "_send_request",
            return_value=_response(content),
        ) as send_request:
            result = engine.process_image(image, blocks)

        self.assertIs(result, blocks)
        self.assertEqual(blocks[0].text, "よし")
        self.assertEqual(blocks[0].ocr_strategy, OCR_STRATEGY_PADDLE_SPOTTING)
        self.assertEqual(engine.last_page_profile["attempt_count"], 1)
        payload = send_request.call_args.args[0]
        self.assertEqual(
            payload["messages"][0]["content"][0]["text"],
            "Spotting:",
        )
        self.assertEqual(payload["repeat_penalty"], 1.0)
        self.assertEqual(payload["repeat_last_n"], 64)
        self.assertTrue(
            payload["messages"][0]["content"][1]["image_url"]["url"].startswith(
                "data:image/png;base64,"
            )
        )

    def test_length_response_gets_one_bounded_recovery_attempt(self) -> None:
        engine = PaddleOCRVLSpottingEngine()
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        blocks = [
            _block((100, 100, 300, 300), "block-a"),
            _block((600, 600, 900, 900), "block-b"),
        ]
        first = _native_line(
            "一",
            (100, 100, 300, 100, 300, 300, 100, 300),
        )
        recovered = "\n".join(
            (
                first,
                _native_line(
                    "二",
                    (600, 600, 900, 600, 900, 900, 600, 900),
                ),
            )
        )

        with mock.patch.object(
            engine,
            "_send_request",
            side_effect=(
                _response(first, finish_reason="length"),
                _response(recovered),
            ),
        ) as send_request:
            engine.process_image(image, blocks)

        self.assertEqual(send_request.call_count, 2)
        recovery_payload = send_request.call_args_list[1].args[0]
        self.assertEqual(recovery_payload["repeat_penalty"], 1.15)
        self.assertEqual(recovery_payload["repeat_last_n"], 4096)
        self.assertEqual([block.text for block in blocks], ["一", "二"])
        self.assertEqual(engine.last_page_profile["selected_attempt"], 1)

    def test_unmatched_detector_block_fails_closed_as_review(self) -> None:
        engine = PaddleOCRVLSpottingEngine()
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        blocks = [_block((0, 0, 100, 100), "detector")]
        content = _native_line(
            "outside",
            (700, 700, 900, 700, 900, 900, 700, 900),
        )

        with mock.patch.object(
            engine,
            "_send_request",
            return_value=_response(content),
        ):
            engine.process_image(image, blocks)

        self.assertEqual(blocks[0].text, "")
        self.assertEqual(blocks[0].processing_action, PROCESSING_ACTION_REVIEW)
        self.assertEqual(engine.last_page_profile["unmatched_region_count"], 1)
        self.assertEqual(engine.last_page_profile["unmapped_block_count"], 1)

    def test_ambiguous_native_region_fails_all_involved_blocks_closed(
        self,
    ) -> None:
        engine = PaddleOCRVLSpottingEngine()
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        blocks = [
            _block((100, 100, 300, 300), "top"),
            _block((100, 320, 300, 520), "bottom"),
        ]
        content = _native_line(
            "merged",
            (90, 90, 310, 90, 310, 530, 90, 530),
        )

        with mock.patch.object(
            engine,
            "_send_request",
            return_value=_response(content),
        ):
            engine.process_image(image, blocks)

        self.assertEqual([block.text for block in blocks], ["", ""])
        self.assertTrue(
            all(
                block.processing_action == PROCESSING_ACTION_REVIEW
                for block in blocks
            )
        )
        self.assertEqual(engine.last_page_profile["mapped_block_count"], 0)
        self.assertEqual(engine.last_page_profile["ambiguous_region_count"], 1)

    def test_low_resolution_pages_double_without_changing_aspect_ratio(self) -> None:
        engine = PaddleOCRVLSpottingEngine()
        image = np.zeros((600, 900, 3), dtype=np.uint8)

        request, profile = engine._preprocess_image(image)

        self.assertEqual(request.shape[:2], (1200, 1800))
        self.assertTrue(profile["low_resolution_doubled"])
        self.assertTrue(profile["aspect_ratio_preserved"])
        self.assertEqual(profile["coordinate_space"], "normalized_0_1000")


if __name__ == "__main__":
    unittest.main()
