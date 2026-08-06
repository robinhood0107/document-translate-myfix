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


class NonSweepEventsKeepTheEstimateTests(unittest.TestCase):
    """미리보기 알림처럼 sweep 이 아닌 이벤트가 추정을 되돌리면 안 된다.

    실제 로그에서 인페인팅 sweep 이 끝난 직후 미리보기 알림 네 줄에서 남은 시간이
    3분 33초에서 24분 08초로 뛰었다. 그 알림들은 eta 를 싣지 않아 트래커가 자체
    계산으로 되돌아갔기 때문이다.
    """

    def _sweep_event(self, **extra) -> dict:
        payload = {
            "phase": "pipeline",
            "service": "batch",
            "status": "running",
            "step_key": "inpaint-all",
            "stage_name": "inpaint-all",
            "page_index": 365,
            "page_total": PAGES,
            "image_name": "366.png",
            "eta_seconds": 213.0,
            "progress_fraction": 0.714,
        }
        payload.update(extra)
        return payload

    def _preview_notice(self) -> dict:
        return {
            "phase": "pipeline",
            "service": "batch",
            "status": "running",
            "step_key": "preview_raw_mask_disabled",
            "stage_name": "raw_mask",
            "page_index": 0,
            "page_total": PAGES,
            "image_name": "001.png",
        }

    def test_a_preview_notice_holds_the_last_estimate(self) -> None:
        tracker = AutomaticProgressTracker()
        tracker.reset(page_total=PAGES)
        tracker.enrich(self._sweep_event())
        event = tracker.enrich(self._preview_notice())
        self.assertAlmostEqual(event["eta_sec"], 213.0, delta=1e-6)

    def test_a_preview_notice_does_not_rewind_progress(self) -> None:
        tracker = AutomaticProgressTracker()
        tracker.reset(page_total=PAGES)
        tracker.enrich(self._sweep_event())
        event = tracker.enrich(self._preview_notice())
        self.assertAlmostEqual(event["overall_progress_percent"], 71.4, delta=0.1)

    def test_a_new_run_forgets_the_previous_estimate(self) -> None:
        tracker = AutomaticProgressTracker()
        tracker.reset(page_total=PAGES)
        tracker.enrich(self._sweep_event())
        tracker.reset(page_total=PAGES)
        event = tracker.enrich(self._preview_notice())
        self.assertNotAlmostEqual(event.get("eta_sec") or -1.0, 213.0, delta=1e-6)


class StageRateHistoryTests(unittest.TestCase):
    """이력 학습이 없으면 첫 실행의 사전 비중이 그대로 오차가 된다."""

    def test_measured_rates_round_trip_through_the_tracker(self) -> None:
        from modules.utils.stage_sweep_eta import StageSweepEtaEstimator

        tracker = AutomaticProgressTracker()
        # 이력은 중앙값이라, 실제 실행이 남긴 표본이 있으면 방금 쓴 값이 그대로
        # 돌아오지 않는다. 이 테스트의 전제는 "빈 이력"이므로 여기서 만든다.
        tracker.settings.beginGroup(tracker.STAGE_RATE_GROUP)
        for key in list(tracker.settings.childKeys()):
            tracker.settings.remove(key)
        tracker.settings.endGroup()

        measured = {"ocr-all": 0.194, "inpaint-all": 1.276}
        tracker.record_stage_rates(measured)
        stored = tracker.read_stage_rates()
        for name, rate in measured.items():
            self.assertAlmostEqual(stored.get(name, 0.0), rate, delta=1e-6)

        estimator = StageSweepEtaEstimator(page_total=10)
        estimator.start_run(0.0)
        estimator.seed_from_history(stored)
        # 비중은 상대값이다. 인페인팅이 OCR 의 약 6.6배여야 한다.
        self.assertAlmostEqual(
            estimator.stage_weights["inpaint-all"]
            / estimator.stage_weights["ocr-all"],
            1.276 / 0.194,
            delta=0.1,
        )

    def test_seeding_ignores_unusable_samples(self) -> None:
        from modules.utils.stage_sweep_eta import StageSweepEtaEstimator

        estimator = StageSweepEtaEstimator(page_total=5)
        before = dict(estimator.stage_weights)
        estimator.seed_from_history({"ocr-all": 0.0, "inpaint-all": -1.0})
        self.assertEqual(estimator.stage_weights, before)
