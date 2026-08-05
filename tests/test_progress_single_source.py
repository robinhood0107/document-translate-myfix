"""진행률과 남은 시간의 출처는 하나여야 한다.

실제 실행 로그에서 같은 순간에 두 값이 갈렸다.

    pipeline.batch_processor: ... overall=99.8% elapsed=00:10:32 eta=00:00:02
    controller:               ... elapsed=00:10:32 eta=00:23:22 finish_at=19:39

파이프라인은 "남은 페이지 x 페이지당 평균"으로 첫 sweep 의 끝을 실행 종료로 봤고,
트래커는 페이지 완료 표본이 없어 과거 이력 선형 모델로 후퇴했다. 이제 파이프라인이
계산하고 트래커는 그 값을 그대로 쓴다.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.utils.automatic_progress import AutomaticProgressTracker  # noqa: E402
from pipeline.batch_processor import BatchProcessor  # noqa: E402
from pipeline.stage_batched_processor import StageBatchedProcessor  # noqa: E402


PAGES = 366


class StageBatchedProgressTests(unittest.TestCase):
    def _processor(self) -> StageBatchedProcessor:
        processor = object.__new__(StageBatchedProcessor)
        processor._progress_image_path = "001.png"
        processor._run_started_at = 0.0
        processor._page_started_at = 0.0
        processor._recent_page_durations = []
        processor.payloads = []
        processor.main_page = SimpleNamespace(
            progress_update=SimpleNamespace(emit=lambda *_a, **_k: None),
            report_runtime_progress=processor.payloads.append,
        )
        return processor

    def test_finishing_the_detect_sweep_is_not_the_end_of_the_run(self) -> None:
        processor = self._processor()
        clock = [1000.0]

        with mock.patch("pipeline.batch_processor.time.monotonic", lambda: clock[0]), \
             mock.patch("pipeline.stage_batched_processor.time.monotonic", lambda: clock[0]):
            for index in range(PAGES):
                clock[0] += 1.0
                processor.emit_progress(index, PAGES, 0, 10, False)

        # step 0 은 payload 를 보내지 않으므로, 추정기 상태를 직접 확인한다.
        estimator = processor._stage_eta
        self.assertLess(estimator.progress_fraction(), 0.15)
        remaining = estimator.remaining_seconds()
        self.assertIsNotNone(remaining)
        self.assertGreater(remaining, PAGES * 5.0)

    def test_the_payload_carries_the_single_estimate(self) -> None:
        processor = self._processor()
        clock = [1000.0]

        with mock.patch("pipeline.batch_processor.time.monotonic", lambda: clock[0]), \
             mock.patch("pipeline.stage_batched_processor.time.monotonic", lambda: clock[0]):
            for index in range(5):
                clock[0] += 1.0
                processor.emit_progress(index, PAGES, 0, 10, False)
            for index in range(5):
                clock[0] += 2.0
                processor.emit_progress(index, PAGES, 3, 10, False)

        self.assertTrue(processor.payloads)
        payload = processor.payloads[-1]
        self.assertIn("eta_seconds", payload)
        self.assertIn("progress_fraction", payload)
        self.assertIsNotNone(payload["eta_seconds"])

    def test_the_message_names_the_stage_in_product_language(self) -> None:
        processor = self._processor()
        clock = [1000.0]
        with mock.patch("pipeline.batch_processor.time.monotonic", lambda: clock[0]), \
             mock.patch("pipeline.stage_batched_processor.time.monotonic", lambda: clock[0]):
            clock[0] += 1.0
            processor.emit_progress(91, PAGES, 3, 10, False)

        message = processor.payloads[-1]["message"]
        self.assertIn("원본 텍스트 제거", message)
        self.assertIn("92/366", message)
        self.assertNotIn("pre-inpaint-setup", message)


class TrackerUsesSuppliedEstimateTests(unittest.TestCase):
    def _event(self, **extra) -> dict:
        payload = {
            "phase": "pipeline",
            "service": "batch",
            "status": "running",
            "step_key": "inpaint-all",
            "stage_name": "inpaint-all",
            "page_index": 200,
            "page_total": PAGES,
            "image_name": "201.png",
        }
        payload.update(extra)
        return payload

    def test_a_supplied_eta_is_used_verbatim(self) -> None:
        tracker = AutomaticProgressTracker()
        event = tracker.enrich(self._event(eta_seconds=612.0))
        self.assertAlmostEqual(event["eta_sec"], 612.0, delta=1e-6)

    def test_a_supplied_progress_fraction_is_used_verbatim(self) -> None:
        tracker = AutomaticProgressTracker()
        event = tracker.enrich(self._event(progress_fraction=0.42))
        self.assertAlmostEqual(event["overall_progress_percent"], 42.0, delta=1e-6)

    def test_without_a_supplied_value_the_tracker_still_estimates(self) -> None:
        """레거시 페이지별 파이프라인은 값을 싣지 않는다. 그 경로는 그대로 둔다."""

        tracker = AutomaticProgressTracker()
        event = tracker.enrich(self._event())
        self.assertIn("eta_sec", event)
        self.assertIn("overall_progress_percent", event)


class LegacyProcessorKeepsItsOwnModelTests(unittest.TestCase):
    def test_legacy_progress_is_still_page_based(self) -> None:
        processor = object.__new__(BatchProcessor)
        processor._run_started_at = 0.0
        processor._recent_page_durations = [2.0]
        result = processor.observe_progress("ocr-processing", 4, 10, 2, 10)
        # 페이지 10장 x 단계 10개 = 100 칸 중 4*10+2 = 42
        self.assertAlmostEqual(result["progress_fraction"], 0.42, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
