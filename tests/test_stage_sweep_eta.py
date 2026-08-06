"""stage-batched 남은 시간 추정기 계약.

실제 로그에서 드러난 두 오류를 회귀로 고정한다. 한 단계의 마지막 페이지에서
`99.8% / 2초` 로 끝난 것처럼 보이던 문제와, 같은 순간에 다른 추정이 23분을 낸 문제다.
"""

from __future__ import annotations

import unittest

from modules.utils.stage_sweep_eta import (
    DEFAULT_STAGE_ORDER,
    DEFAULT_STAGE_WEIGHTS,
    StageSweepEtaEstimator,
)


class StageSweepEtaTests(unittest.TestCase):
    def _estimator(self, pages: int = 366) -> StageSweepEtaEstimator:
        estimator = StageSweepEtaEstimator(page_total=pages)
        estimator.start_run(now=0.0)
        return estimator

    def test_no_estimate_before_any_measurement(self) -> None:
        estimator = self._estimator()
        self.assertIsNone(estimator.remaining_seconds())
        self.assertEqual(estimator.progress_fraction(), 0.0)

    def test_finishing_the_first_stage_is_not_the_end_of_the_run(self) -> None:
        """레거시 추정이 여기서 99.8% / 2초를 냈다. 남은 단계가 4개 더 있었다."""

        estimator = self._estimator(pages=366)
        now = 0.0
        for index in range(366):
            now += 1.0
            estimator.observe("detect-all", index, now)

        self.assertLess(estimator.progress_fraction(), 0.15)
        remaining = estimator.remaining_seconds()
        self.assertIsNotNone(remaining)
        # 검출이 페이지당 1초였고 검출 비중이 1.0 이므로, 남은 단계들의 비중 합
        # 만큼이 남아야 한다. 비중은 실측으로 갱신되므로 표에서 계산한다.
        remaining_weight = sum(
            weight
            for name, weight in DEFAULT_STAGE_WEIGHTS.items()
            if name != "detect-all"
        )
        self.assertAlmostEqual(remaining, 366 * remaining_weight, delta=366 * 0.5)

    def test_progress_never_moves_backwards_across_a_stage_boundary(self) -> None:
        estimator = self._estimator(pages=10)
        now = 0.0
        seen: list[float] = []
        for stage in ("detect-all", "ocr-all", "inpaint-all"):
            for index in range(10):
                now += 1.0
                estimator.observe(stage, index, now)
                seen.append(estimator.progress_fraction())
        self.assertEqual(seen, sorted(seen))

    def test_a_measured_stage_replaces_its_prior_weight(self) -> None:
        estimator = self._estimator(pages=4)
        now = 0.0
        for index in range(4):
            now += 2.0
            estimator.observe("detect-all", index, now)
        self.assertAlmostEqual(estimator.stage_per_page_estimate("detect-all"), 2.0, delta=0.2)

        # OCR 은 아직 안 돌았으므로 사전 비중으로 환산된 값이어야 한다.
        ocr_prior = 2.0 * DEFAULT_STAGE_WEIGHTS["ocr-all"]
        self.assertAlmostEqual(
            estimator.stage_per_page_estimate("ocr-all"), ocr_prior, delta=0.2
        )

        # 단계 전환 보고는 시작만 표시한다. 그 구간에는 모델 스왑이 섞여 있어
        # 페이지 속도로 쓰면 안 된다. 따라서 아직 사전 비중이 유지된다.
        now += 10.0
        estimator.observe("ocr-all", 0, now)
        self.assertAlmostEqual(
            estimator.stage_per_page_estimate("ocr-all"), ocr_prior, delta=0.2
        )

        # 두 번째 페이지부터 실측이 사전 비중을 대체한다.
        now += 10.0
        estimator.observe("ocr-all", 1, now)
        self.assertAlmostEqual(estimator.stage_per_page_estimate("ocr-all"), 10.0, delta=1.0)

    def test_estimate_shrinks_monotonically_within_a_stage(self) -> None:
        estimator = self._estimator(pages=20)
        now = 0.0
        estimates: list[float] = []
        for index in range(20):
            now += 1.0
            estimator.observe("detect-all", index, now)
            remaining = estimator.remaining_seconds()
            if remaining is not None:
                estimates.append(remaining)
        self.assertEqual(estimates, sorted(estimates, reverse=True))

    def test_reaching_the_final_stage_end_reports_the_run_complete(self) -> None:
        estimator = self._estimator(pages=5)
        now = 0.0
        for stage in DEFAULT_STAGE_ORDER:
            for index in range(5):
                now += 1.0
                estimator.observe(stage, index, now)
        self.assertAlmostEqual(estimator.progress_fraction(), 1.0, delta=1e-9)
        self.assertAlmostEqual(estimator.remaining_seconds() or 0.0, 0.0, delta=1e-9)

    def test_repeated_reports_for_one_page_do_not_double_count(self) -> None:
        estimator = self._estimator(pages=6)
        now = 0.0
        for _ in range(5):
            now += 1.0
            estimator.observe("detect-all", 0, now)
        self.assertAlmostEqual(
            estimator.completed_units(),
            estimator.stage_weights["detect-all"] * 1.0,
            delta=1e-9,
        )

    def test_an_unknown_stage_is_counted_instead_of_ignored(self) -> None:
        """모르는 단계를 버리면 남은 작업을 과소평가한다."""

        estimator = self._estimator(pages=3)
        now = 0.0
        for index in range(3):
            now += 1.0
            estimator.observe("detect-all", index, now)
        before = estimator.total_units()
        estimator.observe("some-new-stage", 0, now + 1.0)
        self.assertGreater(estimator.total_units(), before)

    def test_runtime_swap_cost_is_recorded(self) -> None:
        estimator = self._estimator(pages=2)
        estimator.observe_runtime_swap(12.5)
        self.assertAlmostEqual(estimator._runtime_swap_sec, 12.5, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
