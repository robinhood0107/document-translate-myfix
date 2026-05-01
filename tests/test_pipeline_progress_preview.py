from __future__ import annotations

import unittest
from types import SimpleNamespace

from pipeline.batch_processor import BatchProcessor


class PipelineProgressPreviewTests(unittest.TestCase):
    def test_running_progress_uses_source_preview_fallback(self) -> None:
        events: list[dict] = []
        processor = BatchProcessor(
            SimpleNamespace(report_runtime_progress=lambda payload: events.append(payload)),
            None,
            None,
            None,
            None,
        )

        processor._report_runtime_progress(
            phase="pipeline",
            service="batch",
            status="running",
            step_key="ocr-processing",
            source_preview_path="/tmp/source.png",
        )

        self.assertEqual(events[0]["preview_path"], "/tmp/source.png")
        self.assertEqual(events[0]["preview_kind"], "source_fallback")

    def test_save_finish_does_not_replace_final_preview_with_source(self) -> None:
        events: list[dict] = []
        processor = BatchProcessor(
            SimpleNamespace(report_runtime_progress=lambda payload: events.append(payload)),
            None,
            None,
            None,
            None,
        )

        processor._report_runtime_progress(
            phase="pipeline",
            service="batch",
            status="running",
            step_key="save-and-finish",
            source_preview_path="/tmp/source.png",
        )

        self.assertNotIn("preview_path", events[0])


if __name__ == "__main__":
    unittest.main()
