from __future__ import annotations

import unittest
from unittest.mock import patch

from modules.utils.automatic_progress import (
    AUTOMATIC_PROGRESS_TRANSLATIONS,
    AutomaticProgressTracker,
    _fit_pipeline_eta_model,
    _fit_weighted_linear_eta_model,
)


class AutomaticProgressTrackerTests(unittest.TestCase):
    def _tracker_without_history(self, *, started_at: float = 1000.0) -> AutomaticProgressTracker:
        with patch("modules.utils.automatic_progress.time.monotonic", return_value=started_at):
            tracker = AutomaticProgressTracker()
            tracker.reset(page_total=100, run_type="batch")
        tracker._read_history = lambda _group, _key: []
        return tracker

    def test_weighted_linear_model_learns_equation(self) -> None:
        model = _fit_weighted_linear_eta_model([
            (10.0, 70.0, 1.0),
            (20.0, 120.0, 1.0),
            (30.0, 170.0, 1.0),
        ])

        self.assertIsNotNone(model)
        intercept, sec_per_page = model or (0.0, 0.0)
        self.assertAlmostEqual(intercept, 20.0, delta=0.01)
        self.assertAlmostEqual(sec_per_page, 5.0, delta=0.01)

    def test_learned_eta_seeds_fresh_pipeline_runs(self) -> None:
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

        model, used_history = _fit_pipeline_eta_model([])
        self.assertFalse(used_history)
        intercept, sec_per_page = model or (0.0, 0.0)
        expected = intercept + sec_per_page * 100
        self.assertAlmostEqual(event["eta_sec"], expected, delta=0.01)
        self.assertEqual(event["eta_confidence"], AUTOMATIC_PROGRESS_TRANSLATIONS["live_learning"])

    def test_learned_eta_counts_down_with_elapsed_time(self) -> None:
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

        model, _used_history = _fit_pipeline_eta_model([])
        intercept, sec_per_page = model or (0.0, 0.0)
        expected_total = intercept + sec_per_page * 100
        self.assertAlmostEqual(event["eta_sec"], expected_total - 60.0, delta=0.01)

    def test_recent_history_tunes_learned_equation(self) -> None:
        tracker = self._tracker_without_history()
        fast_history = [
            {"image_count": 80, "elapsed_sec": 160.0, "per_page_sec": 2.0},
            {"image_count": 100, "elapsed_sec": 200.0, "per_page_sec": 2.0},
            {"image_count": 120, "elapsed_sec": 240.0, "per_page_sec": 2.0},
        ]
        tracker._read_history = lambda _group, _key: fast_history

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

        seed_model, _used_seed_history = _fit_pipeline_eta_model([])
        tuned_model, used_history = _fit_pipeline_eta_model(fast_history)
        seed_intercept, seed_sec_per_page = seed_model or (0.0, 0.0)
        tuned_intercept, tuned_sec_per_page = tuned_model or (0.0, 0.0)

        seed_eta = seed_intercept + seed_sec_per_page * 100
        tuned_eta = tuned_intercept + tuned_sec_per_page * 100
        self.assertTrue(used_history)
        self.assertLess(tuned_eta, seed_eta)
        self.assertAlmostEqual(event["eta_sec"], tuned_eta, delta=0.01)
        self.assertEqual(event["eta_confidence"], AUTOMATIC_PROGRESS_TRANSLATIONS["recent_history"])


if __name__ == "__main__":
    unittest.main()
