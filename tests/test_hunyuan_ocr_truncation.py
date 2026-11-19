from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from modules.ocr.hunyuan_ocr import HunyuanOCREngine
from modules.utils.ocr_debug import OCR_EMPTY_REASON_HUNYUAN_TRUNCATED
from modules.utils.textblock import TextBlock
from pipeline.cache_manager import CacheManager


_MISSING = object()


def _response(text: str, finish_reason=_MISSING) -> dict:
    choice = {"message": {"content": text}}
    if finish_reason is not _MISSING:
        choice["finish_reason"] = finish_reason
    return {"choices": [choice]}


def _run_responses(responses: list[dict], *, stale_result: bool = False):
    engine = HunyuanOCREngine()
    engine._image_data_url = mock.Mock(return_value="data:image/jpeg;base64,synthetic")
    engine._crop_image = mock.Mock(
        return_value=np.zeros((8, 8, 3), dtype=np.uint8)
    )
    sender = mock.Mock(side_effect=responses)
    engine._send_request = sender
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    block = TextBlock(
        text_bbox=np.asarray([2, 3, 10, 11], dtype=np.int32),
        block_id="synthetic-block",
    )
    if stale_result:
        block.text = "prior OCR text"
        block.texts = ["prior OCR text"]
        block.ocr_regions = [{"text": "prior OCR text"}]
    before_bbox = block.xyxy.copy()
    engine._process_block(image, block, tuple(block.xyxy))
    return sender, block, before_bbox


class HunyuanOCRTruncationTests(unittest.TestCase):
    def test_ocr_cache_keys_use_full_decoded_image_identity(self) -> None:
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        unchanged = image.copy()
        changed = image.copy()
        changed[1, 1, 0] = 1
        block = TextBlock(
            text_bbox=np.asarray([2, 3, 10, 11], dtype=np.int32),
            text="cached text",
            block_id="synthetic-block",
        )
        cache = CacheManager()
        cache_key = cache._get_ocr_cache_key(image, "English", "Default", "cpu")
        cache._cache_ocr_results(cache_key, [block])

        self.assertEqual(
            cache_key,
            cache._get_ocr_cache_key(unchanged, "English", "Default", "cpu"),
        )
        self.assertTrue(
            cache._can_serve_all_blocks_from_ocr_cache(
                cache._get_ocr_cache_key(unchanged, "English", "Default", "cpu"),
                [block],
            )
        )
        self.assertFalse(
            cache._can_serve_all_blocks_from_ocr_cache(
                cache._get_ocr_cache_key(changed, "English", "Default", "cpu"),
                [block],
            )
        )

        strided_source = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)
        strided = strided_source[::2, ::2]
        self.assertFalse(strided.flags.c_contiguous)
        self.assertEqual(
            cache_key,
            cache._get_ocr_cache_key(strided, "English", "Default", "cpu"),
        )
        self.assertNotEqual(
            cache_key,
            cache._get_ocr_cache_key(
                image.reshape(10, 40, 3),
                "English",
                "Default",
                "cpu",
            ),
        )

    def test_length_response_does_not_publish_partial_text(self) -> None:
        sender, block, before_bbox = _run_responses(
            [_response("partial OCR text", " LENGTH ")],
            stale_result=True,
        )

        self.assertEqual(block.text, "")
        self.assertEqual(block.texts, [])
        self.assertEqual(block.ocr_regions, [])
        self.assertEqual(block.ocr_status, "empty_initial")
        self.assertEqual(
            block.ocr_empty_reason,
            OCR_EMPTY_REASON_HUNYUAN_TRUNCATED,
        )
        self.assertEqual(block.ocr_attempt_count, 1)
        self.assertEqual(block.ocr_raw_text, "")
        self.assertEqual(block.block_id, "synthetic-block")
        np.testing.assert_array_equal(block.xyxy, before_bbox)
        sender.assert_called_once()

    def test_stop_missing_and_null_finish_reason_remain_accepted(self) -> None:
        for finish_reason in ("stop", None, _MISSING):
            with self.subTest(finish_reason=finish_reason):
                sender, block, _ = _run_responses(
                    [_response("완료된 OCR 문장", finish_reason)]
                )

                self.assertEqual(block.text, "완료된 OCR 문장")
                self.assertEqual(block.ocr_status, "ok")
                self.assertEqual(block.ocr_empty_reason, "")
                sender.assert_called_once()

    def test_empty_length_attempt_can_recover_with_existing_retry(self) -> None:
        sender, block, _ = _run_responses(
            [_response("", "length"), _response("복구된 OCR 문장", "stop")]
        )

        self.assertEqual(block.text, "복구된 OCR 문장")
        self.assertEqual(block.ocr_status, "ok")
        self.assertEqual(block.ocr_attempt_count, 2)
        self.assertEqual(sender.call_count, 2)

    def test_truncated_retry_is_not_salvaged_from_prior_response(self) -> None:
        engine = HunyuanOCREngine()
        engine._image_data_url = mock.Mock(return_value="data:image/jpeg;base64,synthetic")
        engine._crop_image = mock.Mock(
            return_value=np.zeros((8, 8, 3), dtype=np.uint8)
        )
        sender = mock.Mock(
            side_effect=[
                _response("descriptive first response", "stop"),
                _response("descriptive partial retry", "length"),
            ]
        )
        engine._send_request = sender
        salvage = mock.Mock(return_value="salvaged first response")
        engine._extract_direct_candidate = mock.Mock(return_value="")
        engine._salvage_text_from_descriptive_output = salvage
        block = TextBlock(
            text_bbox=np.asarray([2, 3, 10, 11], dtype=np.int32),
            block_id="synthetic-block",
        )

        engine._process_block(np.zeros((16, 16, 3), dtype=np.uint8), block, tuple(block.xyxy))

        self.assertEqual(block.text, "")
        self.assertEqual(
            block.ocr_empty_reason,
            OCR_EMPTY_REASON_HUNYUAN_TRUNCATED,
        )
        salvage.assert_not_called()
        self.assertEqual(sender.call_count, 2)

    def test_two_length_attempts_end_as_truncation(self) -> None:
        sender, block, _ = _run_responses(
            [_response("", "length"), _response("partial OCR text", "length")]
        )

        self.assertEqual(block.text, "")
        self.assertEqual(
            block.ocr_empty_reason,
            OCR_EMPTY_REASON_HUNYUAN_TRUNCATED,
        )
        self.assertEqual(block.ocr_attempt_count, 2)
        self.assertEqual(sender.call_count, 2)

    def test_stale_hunyuan_flag_allows_complete_other_model_cache_hit(self) -> None:
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        cache = CacheManager()
        key = cache._get_ocr_cache_key(
            image,
            "Chinese",
            "PaddleOCR VL",
            "cpu",
            resolved_ocr_model="PaddleOCR VL",
        )
        block = TextBlock(
            text_bbox=np.asarray([2, 3, 10, 11], dtype=np.int32),
            block_id="synthetic-block",
        )
        block.text = ""
        block.ocr_status = "empty_initial"
        block.ocr_empty_reason = OCR_EMPTY_REASON_HUNYUAN_TRUNCATED
        before_bbox = block.xyxy.copy()
        block_id = cache._get_block_id(block)
        cache.ocr_cache[key] = {
            block_id: {
                "text": "complete text from selected OCR",
                "texts": ["complete text from selected OCR"],
                "status": "ok",
                "empty_reason": "",
                "attempt_count": 1,
            }
        }

        self.assertTrue(cache._can_serve_all_blocks_from_ocr_cache(key, [block]))
        payload = cache._get_cached_ocr_payload_for_block(key, block)
        self.assertIsNotNone(payload)
        cache._apply_cached_ocr_payload_to_block(block, payload)

        self.assertEqual(block.text, "complete text from selected OCR")
        self.assertEqual(block.ocr_status, "ok")
        self.assertEqual(block.ocr_empty_reason, "")
        self.assertEqual(block.block_id, "synthetic-block")
        np.testing.assert_array_equal(block.xyxy, before_bbox)

    def test_truncated_result_is_not_cached_or_served(self) -> None:
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        cache = CacheManager()
        old_key = cache._get_ocr_cache_key(
            image,
            "Chinese",
            "best_local",
            "cpu",
        )
        key = cache._get_ocr_cache_key(
            image,
            "Chinese",
            "best_local",
            "cpu",
            resolved_ocr_model="HunyuanOCR",
        )
        self.assertNotEqual(old_key, key)
        self.assertEqual(key[-1], cache.HUNYUAN_OCR_CACHE_VERSION)

        sender, block, _ = _run_responses(
            [_response("partial OCR text", "length")]
        )
        block_id = cache._get_block_id(block)
        normal_block = TextBlock(
            text_bbox=block.xyxy.copy(),
            block_id="synthetic-block",
        )
        cache.ocr_cache[old_key] = {
            block_id: {"text": "legacy partial text", "status": "ok"}
        }
        self.assertFalse(
            cache._can_serve_all_blocks_from_ocr_cache(key, [normal_block])
        )
        cache.ocr_cache[key] = {
            block_id: {
                "text": "partial OCR text",
                "status": "empty_initial",
                "empty_reason": OCR_EMPTY_REASON_HUNYUAN_TRUNCATED,
            }
        }
        self.assertFalse(
            cache._can_serve_all_blocks_from_ocr_cache(key, [normal_block])
        )

        cache._cache_ocr_results(key, [block])
        self.assertNotIn(key, cache.ocr_cache)
        cache.ocr_cache[key] = {block_id: {"text": "legacy partial text"}}
        cache.update_ocr_cache_for_block(key, block)
        self.assertNotIn(key, cache.ocr_cache)
        sender.assert_called_once()

        complete_block = TextBlock(
            text_bbox=block.xyxy.copy(),
            text="complete OCR text",
            block_id="synthetic-block",
        )
        complete_block.ocr_status = "ok"
        cache._cache_ocr_results(key, [complete_block])
        self.assertTrue(
            cache._can_serve_all_blocks_from_ocr_cache(key, [complete_block])
        )


if __name__ == "__main__":
    unittest.main()
