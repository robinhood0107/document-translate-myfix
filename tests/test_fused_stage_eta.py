from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.utils.stage_sweep_eta import StageSweepEtaEstimator  # noqa: E402


ORDER = ("inpaint-all", "render-all")
PAGES = 100


def _estimator(pages: int = PAGES, order=ORDER) -> StageSweepEtaEstimator:
    estimator = StageSweepEtaEstimator(page_total=pages, stage_order=order)
    estimator.start_run(0.0)
    return estimator


def _fused_sweep(estimator, pages: int, *, inpaint_sec: float, start: float = 0.0):
    """실제 융합 스윕 재현.

    인페인팅이 페이지를 하나 끝낼 때마다 그 직후 렌더가 회수돼 보고된다. 두 단계의
    보고가 교대로 들어오는 것이 이 파이프라인의 정상 동작이다.
    """

    now = start
    for index in range(pages):
        now = start + inpaint_sec * (index + 1)
        estimator.observe("inpaint-all", index, now)
        # 렌더는 인페인팅 뒤에 숨어 거의 즉시 회수된다.
        estimator.observe("render-all", index, now + 0.001)
    return now


class FusedStageObservationTests(unittest.TestCase):
    def test_an_interleaved_stage_does_not_mark_the_other_complete(self) -> None:
        # 예전에는 단계가 바뀔 때마다 직전 단계를 전체 완료로 표시했다. 융합
        # 파이프라인은 두 단계가 교대로 보고하므로, 첫 렌더 보고 하나에
        # 인페인팅이 366/366 으로 찍혀 모델이 통째로 망가졌다.
        estimator = _estimator()

        estimator.observe("inpaint-all", 0, 1.0)
        estimator.observe("render-all", 0, 1.001)
        estimator.observe("inpaint-all", 1, 2.0)

        self.assertEqual(estimator._stages["inpaint-all"].pages_done, 2)
        self.assertEqual(estimator._stages["render-all"].pages_done, 1)
        self.assertIsNone(estimator._stages["inpaint-all"].finished_at)

    def test_a_fused_stage_adds_nothing_to_the_remaining_time(self) -> None:
        # 융합된 단계는 감싸는 단계의 페이스에 맞춰 진행하므로 페이지당 속도가
        # 같게 측정된다. 그걸 합산하면 남은 시간이 두 배가 된다. 실측 사고:
        # 90/100 지점에서 36초가 남았는데 72초로 나왔다.
        estimator = _estimator()
        _fused_sweep(estimator, 20, inpaint_sec=3.6)

        with_render = estimator.remaining_seconds()

        solo = StageSweepEtaEstimator(page_total=PAGES, stage_order=("inpaint-all",))
        solo.start_run(0.0)
        for index in range(20):
            solo.observe("inpaint-all", index, 3.6 * (index + 1))

        self.assertAlmostEqual(with_render, solo.remaining_seconds(), delta=1.0)

    def test_startup_cost_is_recorded_once_not_on_every_re_entry(self) -> None:
        estimator = _estimator()
        estimator.observe("inpaint-all", 0, 10.0)
        first = estimator.stage_startup_estimate("inpaint-all")

        _fused_sweep(estimator, 10, inpaint_sec=3.6, start=10.0)

        self.assertEqual(estimator.stage_startup_estimate("inpaint-all"), first)


class RemainingTimeTests(unittest.TestCase):
    def test_the_estimate_falls_as_the_fused_sweep_progresses(self) -> None:
        # 실측 사고: 인페인팅 47장부터 327장까지 280장이 처리되는 동안 ETA 가
        # 11~12분에 못 박혀 있었다. 아직 시작 안 한 단계를 매번 전체 페이지로
        # 세는 바람에 그 몫이 줄지 않았기 때문이다.
        estimator = _estimator()
        _fused_sweep(estimator, 10, inpaint_sec=3.6)
        early = estimator.remaining_seconds()

        _fused_sweep(estimator, 80, inpaint_sec=3.6)
        late = estimator.remaining_seconds()

        self.assertIsNotNone(early)
        self.assertIsNotNone(late)
        self.assertLess(late, early * 0.5)

    def test_the_estimate_reaches_zero_when_every_stage_is_done(self) -> None:
        estimator = _estimator()
        _fused_sweep(estimator, PAGES, inpaint_sec=1.0)

        self.assertAlmostEqual(estimator.remaining_seconds(), 0.0, places=3)

    def test_the_estimate_tracks_the_real_remaining_time(self) -> None:
        # 90장을 3.6초에 처리했고 10장이 남았다면 남은 시간은 36초 근처여야 한다.
        estimator = _estimator()
        _fused_sweep(estimator, 90, inpaint_sec=3.6)

        remaining = estimator.remaining_seconds()

        self.assertAlmostEqual(remaining, 36.0, delta=8.0)

    def test_a_stage_that_never_runs_still_counts_until_it_does(self) -> None:
        # 아직 시작 안 한 단계는 남은 작업이 맞다. 다만 진행에 따라 줄어야 한다.
        estimator = _estimator(order=("inpaint-all", "render-all", "save-and-finish"))
        _fused_sweep(estimator, 50, inpaint_sec=2.0)

        remaining = estimator.remaining_seconds()

        self.assertIsNotNone(remaining)
        self.assertGreater(remaining, 0.0)


class StageBreakdownTests(unittest.TestCase):
    def test_counted_rows_sum_to_the_remaining_time(self) -> None:
        estimator = _estimator(order=("inpaint-all", "render-all", "save-and-finish"))
        _fused_sweep(estimator, 40, inpaint_sec=2.0)

        rows = estimator.remaining_by_stage()

        self.assertAlmostEqual(
            sum(row["seconds"] for row in rows if row["counted"]),
            estimator.remaining_seconds(),
            places=3,
        )

    def test_every_pipeline_stage_appears_including_finished_and_fused(self) -> None:
        # 남은 단계만 보여주면 지금 파이프라인 어디쯤인지가 오히려 흐려진다.
        estimator = _estimator(
            order=("detect-all", "ocr-all", "inpaint-all", "render-all")
        )
        for index in range(PAGES):
            estimator.observe("detect-all", index, 0.1 * (index + 1))
        _fused_sweep(estimator, 20, inpaint_sec=3.6, start=10.0)

        rows = {row["stage"]: row for row in estimator.remaining_by_stage()}

        self.assertEqual(
            list(rows),
            ["detect-all", "ocr-all", "inpaint-all", "render-all"],
        )
        self.assertEqual(rows["detect-all"]["state"], "done")
        self.assertEqual(rows["inpaint-all"]["state"], "running")
        self.assertEqual(rows["ocr-all"]["state"], "pending")
        self.assertEqual(rows["render-all"]["state"], "fused")

    def test_a_fused_stage_contributes_nothing_even_though_it_is_listed(self) -> None:
        estimator = _estimator()
        _fused_sweep(estimator, 20, inpaint_sec=3.6)

        render = next(
            row for row in estimator.remaining_by_stage() if row["stage"] == "render-all"
        )

        self.assertFalse(render["counted"])
        self.assertEqual(render["seconds"], 0.0)

    def test_a_finished_stage_is_listed_as_done(self) -> None:
        estimator = _estimator(order=("inpaint-all", "render-all"))
        _fused_sweep(estimator, PAGES, inpaint_sec=1.0)

        states = {row["stage"]: row["state"] for row in estimator.remaining_by_stage()}

        self.assertEqual(states["inpaint-all"], "done")

    def test_the_tooltip_lists_the_whole_pipeline_in_order(self) -> None:
        from modules.utils.automatic_progress import format_stage_breakdown

        text = format_stage_breakdown(
            [
                {"stage": "detect-all", "label": "텍스트 영역 검출", "state": "done",
                 "seconds": 0.0, "pages_done": 366, "page_total": 366},
                {"stage": "ocr-all", "label": "텍스트 인식(OCR)", "state": "done",
                 "seconds": 0.0, "pages_done": 366, "page_total": 366},
                {"stage": "translate-all", "label": "번역", "state": "running",
                 "seconds": 900.0, "pages_done": 200, "page_total": 366},
                {"stage": "inpaint-all", "label": "원본 텍스트 제거(인페인팅)",
                 "state": "pending", "seconds": 600.0, "pages_done": 0, "page_total": 366},
                {"stage": "render-all", "label": "번역문 렌더링", "state": "fused",
                 "seconds": 0.0, "pages_done": 0, "page_total": 366},
            ]
        )

        self.assertTrue(text.startswith("<pre"))
        lines = text.splitlines()
        self.assertIn("단계별 남은 시간", lines[0])
        # 실행 순서 그대로. OCR 과 검출도 완료 상태로 남는다.
        self.assertIn("텍스트 영역 검출", lines[1])
        self.assertIn("완료", lines[1])
        self.assertIn("텍스트 인식(OCR)", lines[2])
        self.assertIn("번역", lines[3])
        self.assertIn("00:15:00", lines[3])
        self.assertIn("200/366", lines[3])
        self.assertIn("인페인팅", lines[4])
        self.assertIn("번역문 렌더링", lines[5])
        self.assertIn("인페인팅에 포함", lines[5])

    def test_an_empty_breakdown_produces_no_tooltip(self) -> None:
        # 툴팁이 빈 문자열이면 UI 가 아무것도 띄우지 않는다.
        from modules.utils.automatic_progress import format_stage_breakdown

        for value in (None, [], "junk", [{"seconds": 5.0}], [None, 3]):
            with self.subTest(value=value):
                self.assertEqual(format_stage_breakdown(value), "")

    def test_a_zero_second_stage_is_still_listed(self) -> None:
        # 0초라고 목록에서 빼면 파이프라인 전체가 보이지 않는다.
        from modules.utils.automatic_progress import format_stage_breakdown

        text = format_stage_breakdown(
            [{"label": "번역문 렌더링", "state": "done", "seconds": 0.0}]
        )

        self.assertIn("번역문 렌더링", text)


class TotalEstimateTests(unittest.TestCase):
    def test_the_tracker_reports_elapsed_plus_remaining(self) -> None:
        from modules.utils.automatic_progress import AutomaticProgressTracker

        tracker = AutomaticProgressTracker()
        tracker.reset(page_total=100)

        event = tracker.enrich(
            {
                "phase": "pipeline",
                "status": "running",
                "stage_name": "inpaint-all",
                "elapsed_sec": 600.0,
                "eta_seconds": 300.0,
            }
        )

        self.assertAlmostEqual(event["total_estimate_sec"], 900.0)
        self.assertEqual(event["total_estimate_text"], "00:15:00")

    def test_an_unknown_remaining_time_leaves_the_total_unknown(self) -> None:
        from modules.utils.automatic_progress import AutomaticProgressTracker

        tracker = AutomaticProgressTracker()
        tracker.reset(page_total=100)

        event = tracker.enrich({"phase": "unknown", "elapsed_sec": 30.0})

        self.assertIsNone(event["total_estimate_sec"])


if __name__ == "__main__":
    unittest.main()
