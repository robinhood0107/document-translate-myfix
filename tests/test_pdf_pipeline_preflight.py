from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from modules.utils.pdf_pages import PdfImportError
from pipeline.batch_processor import BatchProcessor
from pipeline.stage_batched_processor import StageBatchedProcessor
from pipeline.webtoon_batch.flow import FlowMixin


def _failing_file_handler() -> mock.Mock:
    handler = mock.Mock()
    handler.preflight_for_processing.side_effect = PdfImportError(
        "PDF_PAGE_MATERIALIZATION_FAILED",
        page_index=1,
        detail_code="output_validation_failed",
    )
    return handler


class PdfPipelinePreflightTests(unittest.TestCase):
    def test_normal_pipeline_stops_before_lazy_processing(self) -> None:
        handler = _failing_file_handler()
        processor = object.__new__(BatchProcessor)
        processor.main_page = SimpleNamespace(
            image_files=["page.png"],
            file_handler=handler,
            reset_automatic_output_reservations=mock.Mock(),
        )
        processor._recent_page_durations = []
        processor._emit_benchmark_event = mock.Mock()
        processor._is_cancelled = mock.Mock(return_value=False)

        with self.assertRaises(PdfImportError):
            processor.batch_process()

        handler.should_pre_materialize.assert_not_called()

    def test_webtoon_pipeline_stops_before_lazy_processing(self) -> None:
        handler = _failing_file_handler()
        processor = object.__new__(FlowMixin)
        processor.main_page = SimpleNamespace(
            image_files=["page.png"],
            file_handler=handler,
            is_current_task_cancelled=mock.Mock(return_value=False),
        )
        processor._emit_benchmark_event = mock.Mock()

        with self.assertRaises(PdfImportError):
            processor.webtoon_batch_process()

        handler.should_pre_materialize.assert_not_called()

    def test_stage_batched_pipeline_stops_before_runtime_setup(self) -> None:
        handler = _failing_file_handler()
        processor = object.__new__(StageBatchedProcessor)
        processor.main_page = SimpleNamespace(
            image_files=["page.png"],
            file_handler=handler,
            reset_automatic_output_reservations=mock.Mock(),
            is_current_task_cancelled=mock.Mock(return_value=False),
        )
        processor._recent_page_durations = []
        processor._benchmark_stage_ceiling = mock.Mock(return_value=None)
        processor._emit_benchmark_event = mock.Mock()
        processor._record_performance_workload = mock.Mock()
        processor._reset_prewarm_lifecycle = mock.Mock()
        processor._load_page_contexts = mock.Mock()

        with self.assertRaises(PdfImportError):
            processor.batch_process()

        processor._load_page_contexts.assert_not_called()


if __name__ == "__main__":
    unittest.main()
