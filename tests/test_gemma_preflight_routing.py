from __future__ import annotations

import json
import unittest
from unittest import mock

import numpy as np

from modules.translation.llm.custom_local_gemma import (
    GEMMA_ROUTE_HUGE_SEGMENT_BLOCK,
    GEMMA_ROUTE_RISKY_SINGLE_BLOCK,
    GEMMA_ROUTE_SAFE_FAST_MULTI,
    CustomLocalGemmaTranslation,
)
from modules.utils.textblock import TextBlock


def _block(text: str, index: int = 0) -> TextBlock:
    return TextBlock(
        text_bbox=np.array([0, index * 10, 100, index * 10 + 8]),
        text=text,
    )


def _response(payload: dict[str, str]) -> dict:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps(payload, ensure_ascii=False)},
            }
        ],
        "usage": {},
    }


class GemmaPreflightRoutingTests(unittest.TestCase):
    def _engine(self) -> CustomLocalGemmaTranslation:
        engine = CustomLocalGemmaTranslation()
        engine.source_lang = "English"
        engine.target_lang = "Korean"
        engine.max_tokens = 512
        return engine

    def test_short_blocks_are_packed_and_routed_to_fast_multi(self) -> None:
        engine = self._engine()
        blocks = [_block(f"Short line {index}.", index) for index in range(8)]

        jobs = engine._build_preflight_jobs(blocks)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].route, GEMMA_ROUTE_SAFE_FAST_MULTI)
        self.assertEqual(len(jobs[0].blocks), 8)

    def test_risky_block_is_not_mixed_into_safe_pack(self) -> None:
        engine = self._engine()
        risky_text = "\n".join(f"line {index}" for index in range(10))
        blocks = [
            _block("Short first.", 0),
            _block("Short second.", 1),
            _block(risky_text, 2),
            _block("Short third.", 3),
        ]

        jobs = engine._build_preflight_jobs(blocks)

        self.assertEqual([job.route for job in jobs], [
            GEMMA_ROUTE_SAFE_FAST_MULTI,
            GEMMA_ROUTE_RISKY_SINGLE_BLOCK,
            GEMMA_ROUTE_SAFE_FAST_MULTI,
        ])
        self.assertEqual([len(job.blocks) for job in jobs], [2, 1, 1])

    def test_huge_block_is_routed_to_segment_block(self) -> None:
        engine = self._engine()
        huge_text = ("Paragraph one sentence. " * 45).strip()

        decision = engine._decide_preflight_route([_block(huge_text)])

        self.assertEqual(decision.route, GEMMA_ROUTE_HUGE_SEGMENT_BLOCK)

    def test_safe_route_uses_one_multi_request_without_retry_or_fallback(self) -> None:
        engine = self._engine()
        blocks = [_block("Hello.", 0), _block("How are you?", 1)]
        requests = []

        def fake_request(system_prompt: str, user_prompt: str, *, expected_keys=None) -> dict:
            requests.append((user_prompt, expected_keys))
            return _response({"block_0": "안녕.", "block_1": "잘 지내?"})

        with mock.patch.object(engine, "_request_translation", side_effect=fake_request):
            engine.translate(blocks, np.zeros((1, 1, 3), dtype=np.uint8), "")

        self.assertEqual([block.translation for block in blocks], ["안녕.", "잘 지내?"])
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0][1], ["block_0", "block_1"])
        self.assertEqual(engine.last_benchmark_stats["gemma_retry_count"], 0)
        self.assertEqual(engine.last_benchmark_stats["gemma_fallback_count"], 0)
        self.assertEqual(engine.last_benchmark_stats["safe_fast_multi_chunks"], 1)

    def test_huge_segment_results_are_joined_into_original_block(self) -> None:
        engine = self._engine()
        blocks = [_block("First sentence.\nSecond sentence.\nThird sentence.", 0)]

        with mock.patch.object(
            engine,
            "_split_text_for_segments",
            return_value=["First sentence.", "Second sentence.", "Third sentence."],
        ), mock.patch.object(
            engine,
            "_request_translation",
            side_effect=[
                _response({"translation": "첫 문장."}),
                _response({"translation": "둘째 문장."}),
                _response({"translation": "셋째 문장."}),
            ],
        ):
            engine._translate_huge_segment_block(blocks[0], "")

        self.assertEqual(blocks[0].translation, "첫 문장. 둘째 문장. 셋째 문장.")


if __name__ == "__main__":
    unittest.main()
