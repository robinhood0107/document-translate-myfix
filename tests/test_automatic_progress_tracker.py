from __future__ import annotations

import unittest
from unittest.mock import patch

from modules.utils.automatic_progress import (
    AUTOMATIC_PROGRESS_TRANSLATIONS,
    AutomaticProgressTracker,
    STATISTICAL_PIPELINE_FIXED_SEC_PER_RUN,
    STATISTICAL_PIPELINE_SEC_PER_PAGE,
)


class AutomaticProgressTrackerTests(unittest.TestCase):
    def _tracker_without_history(self, *, started_at: float = 1000.0) -> AutomaticProgressTracker:
        with patch("modules.utils.automatic_progress.time.monotonic", return_value=started_at):
            tracker = AutomaticProgressTracker()
            tracker.reset(page_total=100, run_type="batch")
        tracker._read_history = lambda _group, _key: []
        return tracker

    def test_statistical_eta_seeds_fresh_pipeline_runs(self) -> None:
        tracker = self._tracker_without_history()

        with patch("modules.utils.automatic_progress.time.monotonic", return_value=1000.0):
            event = tracker.enrich(
                {
                    "phase": "pipeline",
                    "status": "running",
                    "page_total": 100,
                    "page_index": 0,
                    "step_key": "text-block-detection",
                    "stage_name": "text-block-detection",
                }
            )

        expected = STATISTICAL_PIPELINE_FIXED_SEC_PER_RUN + STATISTICAL_PIPELINE_SEC_PER_PAGE * 100
        self.assertAlmostEqual(event["eta_sec"], expected, delta=0.01)
        self.assertEqual(event["eta_confidence"], AUTOMATIC_PROGRESS_TRANSLATIONS["live_learning"])

    def test_statistical_eta_counts_down_with_elapsed_time(self) -> None:
        tracker = self._tracker_without_history(started_at=500.0)

        with patch("modules.utils.automatic_progress.time.monotonic", return_value=560.0):
            event = tracker.enrich(
                {
                    "phase": "pipeline",
                    "status": "running",
                    "page_total": 100,
                    "page_index": 12,
                    "step_key": "inpainting",
                    "stage_name": "inpainting",
                }
            )

        expected_total = STATISTICAL_PIPELINE_FIXED_SEC_PER_RUN + STATISTICAL_PIPELINE_SEC_PER_PAGE * 100
        self.assertAlmostEqual(event["eta_sec"], expected_total - 60.0, delta=0.01)

    def test_recent_history_still_takes_precedence_over_statistical_seed(self) -> None:
        tracker = self._tracker_without_history()
        tracker._read_history = lambda _group, _key: [{"image_count": 100, "per_page_sec": 2.0}]

        with patch("modules.utils.automatic_progress.time.monotonic", return_value=1000.0):
            event = tracker.enrich(
                {
                    "phase": "pipeline",
                    "status": "running",
                    "page_total": 100,
                    "page_index": 0,
                    "step_key": "text-block-detection",
                    "stage_name": "text-block-detection",
                }
            )

        self.assertEqual(event["eta_sec"], 200.0)
        self.assertEqual(event["eta_confidence"], AUTOMATIC_PROGRESS_TRANSLATIONS["recent_history"])


if __name__ == "__main__":
    unittest.main()
