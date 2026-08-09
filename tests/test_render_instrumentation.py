from __future__ import annotations

import inspect
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.utils.run_report import build_run_report  # noqa: E402
from pipeline.render_worker import RenderJobInput, RenderJobResult  # noqa: E402
from pipeline.stage_batched_processor import StageBatchedProcessor  # noqa: E402


class RenderWorkerTimingTests(unittest.TestCase):
    def test_the_result_carries_worker_and_queue_time(self) -> None:
        # 렌더는 전용 워커에서 돈다. 워커가 돌려주지 않으면 그 시간은
        # 어디에도 남지 않는다. 실제로 그래서 페이지당 2.31초가 안 보였다.
        fields = RenderJobResult.__dataclass_fields__
        self.assertIn("worker_seconds", fields)
        self.assertIn("queue_wait_seconds", fields)

    def test_the_job_carries_its_submission_time(self) -> None:
        self.assertIn("submitted_monotonic", RenderJobInput.__dataclass_fields__)

    def test_the_worker_measures_from_start_to_finish(self) -> None:
        source = inspect.getsource(
            sys.modules["pipeline.render_worker"].run_render_job
        )
        self.assertIn("started_monotonic = time.monotonic()", source)
        self.assertIn("queue_wait_seconds", source)
        self.assertIn("worker_seconds=", source)


class RenderTelemetryWiringTests(unittest.TestCase):
    def test_bookkeeping_records_worker_and_queue_wait(self) -> None:
        source = inspect.getsource(
            StageBatchedProcessor._finish_render_page_bookkeeping
        )
        self.assertIn('operation="worker"', source)
        self.assertIn('operation="queue_wait"', source)
        self.assertIn("result.worker_seconds", source)
        self.assertIn("result.queue_wait_seconds", source)

    def test_the_tail_drain_is_measured(self) -> None:
        # 인페인팅이 끝난 뒤 렌더 때문에 더 기다린 시간이 곧 융합이 숨기지 못한
        # 잔량이다. 이 값이 커지면 인페인팅을 더 줄여도 전체가 안 줄어든다.
        source = inspect.getsource(StageBatchedProcessor._render_all)
        self.assertIn('operation="tail_drain"', source)
        self.assertIn("pending_at_drain", source)


class RenderWorkerCountTests(unittest.TestCase):
    def _count(self, value: str | None) -> int:
        env = {} if value is None else {"CT_RENDER_WORKERS": value}
        with mock.patch.dict(os.environ, env, clear=False):
            if value is None:
                os.environ.pop("CT_RENDER_WORKERS", None)
            return StageBatchedProcessor._render_worker_count()

    def test_the_default_stays_one(self) -> None:
        # 워커를 늘렸을 때 빈 이미지가 나오지 않는다는 확인 전에는 기본값을
        # 바꾸지 않는다. 그 실패는 예외도 경고도 없이 조용하다.
        self.assertEqual(self._count(None), 1)

    def test_an_explicit_count_is_honoured(self) -> None:
        self.assertEqual(self._count("3"), 3)

    def test_junk_and_out_of_range_values_fall_back_to_one(self) -> None:
        for value in ("abc", "0", "-2", "99", ""):
            with self.subTest(value=value):
                self.assertEqual(self._count(value), 1)


class OperationReportTests(unittest.TestCase):
    def test_operations_exclude_the_stage_window_and_rank_by_cost(self) -> None:
        report = build_run_report(
            telemetry={
                "stage_details": {
                    "render": {
                        "stage_window": {"wall_ms": 5_000.0, "count": 1},
                        "worker": {"wall_ms": 802_100.0, "count": 347},
                        "queue_wait": {"wall_ms": 12_000.0, "count": 347},
                        "tail_drain": {"wall_ms": 60_000.0, "count": 1},
                    },
                    "translate": {
                        "stage_window": {"wall_ms": 961_500.0, "count": 1},
                        "inference_and_cache": {"wall_ms": 900_000.0, "count": 347},
                    },
                }
            },
            total_wall_sec=2487.0,
            page_outcomes=[],
            output_summary={"input_count": 366, "output_count": 366},
        )

        operations = report["operations"]
        self.assertNotIn(
            "stage_window",
            {row["operation"] for row in operations},
        )
        self.assertEqual(operations[0]["operation"], "inference_and_cache")
        worker = next(r for r in operations if r["operation"] == "worker")
        self.assertAlmostEqual(worker["seconds_each"], 2.3115, places=3)

        drain = next(r for r in operations if r["operation"] == "tail_drain")
        self.assertEqual(drain["seconds"], 60.0)


if __name__ == "__main__":
    unittest.main()
