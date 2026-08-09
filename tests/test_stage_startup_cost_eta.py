from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.utils.stage_sweep_eta import StageSweepEtaEstimator  # noqa: E402


ORDER = ("detect-all", "ocr-all", "inpaint-all")


def _estimator(pages: int = 10) -> StageSweepEtaEstimator:
    estimator = StageSweepEtaEstimator(page_total=pages, stage_order=ORDER)
    estimator.start_run(0.0)
    return estimator


def _sweep(estimator, stage: str, *, start: float, per_page: float, pages: int) -> float:
    """한 단계가 ``start`` 에 첫 보고를 내고 페이지마다 ``per_page`` 초를 쓴다.

    마지막 **보고 시각**을 돌려준다. 다음 단계와의 공백은 이 시각부터 잰다.
    """

    now = start
    for index in range(pages):
        now = start + per_page * index
        estimator.observe(stage, index, now)
    return now


class StageStartupCaptureTests(unittest.TestCase):
    def test_the_gap_between_stages_is_captured_as_startup_cost(self) -> None:
        # 이전 단계 마지막 페이지와 다음 단계 첫 보고 사이의 공백은 컨테이너
        # 기동과 모델 적재 시간이다. 예전에는 통째로 버려졌다.
        estimator = _estimator()
        end = _sweep(estimator, "detect-all", start=0.0, per_page=1.0, pages=10)

        # 40초 뒤에야 OCR 첫 페이지가 보고된다. 그 사이가 모델 적재다.
        estimator.observe("ocr-all", 0, end + 40.0)

        self.assertAlmostEqual(estimator.stage_startup_estimate("ocr-all"), 40.0)

    def test_a_stage_with_no_gap_records_no_startup_cost(self) -> None:
        estimator = _estimator()
        end = _sweep(estimator, "detect-all", start=0.0, per_page=1.0, pages=10)
        estimator.observe("ocr-all", 0, end)

        self.assertEqual(estimator.stage_startup_estimate("ocr-all"), 0.0)

    def test_the_first_stage_counts_the_gap_from_the_run_start(self) -> None:
        # 실행이 시작되고 첫 단계가 첫 페이지를 보고하기까지의 공백도 모델 적재
        # 시간이다. 첫 단계라고 빼놓을 이유가 없다.
        estimator = _estimator()
        estimator.observe("detect-all", 0, 5.0)

        self.assertAlmostEqual(estimator.stage_startup_estimate("detect-all"), 5.0)


class RemainingSecondsIncludesStartupTests(unittest.TestCase):
    def test_pending_stage_startup_is_counted_in_the_remaining_time(self) -> None:
        # 이게 빠져 있으면 무거운 단계로 넘어가는 순간 남은 시간이 위로 튄다.
        estimator = _estimator(pages=10)
        estimator.seed_startup_from_history({"inpaint-all": 60.0})
        _sweep(estimator, "detect-all", start=0.0, per_page=1.0, pages=10)
        estimator.observe("ocr-all", 0, 10.0)
        estimator.observe("ocr-all", 1, 12.0)

        remaining = estimator.remaining_seconds()

        # ocr 남은 8페이지 x 2초 = 16, inpaint 10페이지 + 시작 60초.
        self.assertIsNotNone(remaining)
        self.assertGreater(remaining, 60.0)

    def test_dropping_the_startup_cost_underestimates_the_remaining_time(self) -> None:
        def remaining_with(startup: dict[str, float]) -> float:
            estimator = _estimator(pages=10)
            estimator.seed_startup_from_history(startup)
            _sweep(estimator, "detect-all", start=0.0, per_page=1.0, pages=10)
            estimator.observe("ocr-all", 0, 10.0)
            estimator.observe("ocr-all", 1, 12.0)
            return estimator.remaining_seconds()

        with_cost = remaining_with({"inpaint-all": 60.0})
        without_cost = remaining_with({})

        self.assertAlmostEqual(with_cost - without_cost, 60.0, places=3)

    def test_a_measured_startup_beats_the_seeded_one(self) -> None:
        estimator = _estimator(pages=10)
        estimator.seed_startup_from_history({"ocr-all": 5.0})
        end = _sweep(estimator, "detect-all", start=0.0, per_page=1.0, pages=10)
        estimator.observe("ocr-all", 0, end + 33.0)

        self.assertAlmostEqual(estimator.stage_startup_estimate("ocr-all"), 33.0)

    def test_startup_of_the_current_stage_is_not_double_counted(self) -> None:
        # 이미 지나간 시작 비용은 남은 시간에 들어가면 안 된다. 현재 단계의
        # 시작 공백이 40초든 0초든 남은 시간은 같아야 한다.
        def remaining_after_gap(gap: float) -> float:
            estimator = _estimator(pages=10)
            end = _sweep(estimator, "detect-all", start=0.0, per_page=1.0, pages=10)
            estimator.observe("ocr-all", 0, end + gap)
            estimator.observe("ocr-all", 1, end + gap + 2.0)
            return estimator.remaining_seconds()

        self.assertAlmostEqual(remaining_after_gap(40.0), remaining_after_gap(0.0))


class StartupPersistenceTests(unittest.TestCase):
    def test_measured_startups_are_exported_for_the_next_run(self) -> None:
        estimator = _estimator()
        end = _sweep(estimator, "detect-all", start=0.0, per_page=1.0, pages=10)
        estimator.observe("ocr-all", 0, end + 12.0)

        self.assertEqual(
            estimator.measured_startup_by_stage(),
            {"ocr-all": 12.0},
        )

    def test_seeding_ignores_junk_values(self) -> None:
        estimator = _estimator()
        estimator.seed_startup_from_history(
            {"ocr-all": 0.0, "inpaint-all": -5.0, "render-all": "x"}
        )

        self.assertEqual(estimator.stage_startup_estimate("ocr-all"), 0.0)
        self.assertEqual(estimator.stage_startup_estimate("inpaint-all"), 0.0)

    def test_an_out_of_band_runtime_swap_adds_to_the_current_stage(self) -> None:
        estimator = _estimator()
        estimator.observe("detect-all", 0, 1.0)
        before = estimator.stage_startup_estimate("detect-all")

        estimator.observe_runtime_swap(9.0)

        self.assertAlmostEqual(
            estimator.stage_startup_estimate("detect-all"), before + 9.0
        )


if __name__ == "__main__":
    unittest.main()
