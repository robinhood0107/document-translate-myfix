from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from modules.ocr.selection import (
    LEGACY_PAGE_WORKFLOW_MODE,
    STAGE_BATCHED_WORKFLOW_MODE,
)
from pipeline.main_pipeline import ComicTranslatePipeline


class PostPr300PipelineContractTests(unittest.TestCase):
    @staticmethod
    def _pipeline(workflow_mode: str) -> ComicTranslatePipeline:
        page_paths = [f"page-{index:03d}.png" for index in range(93)]
        settings = SimpleNamespace(
            get_workflow_mode=lambda: workflow_mode,
        )
        pipeline = ComicTranslatePipeline.__new__(ComicTranslatePipeline)
        pipeline.main_page = SimpleNamespace(
            settings_page=settings,
            image_files=page_paths,
            s_combo=SimpleNamespace(currentText=lambda: "Japanese"),
            lang_mapping={"Japanese": "Japanese"},
            file_handler=mock.Mock(),
        )
        pipeline.stage_batched_processor = mock.Mock()
        pipeline.stage_batched_processor.batch_process.return_value = "stage"
        pipeline.batch_processor = mock.Mock()
        pipeline.batch_processor.batch_process.return_value = "legacy"
        return pipeline

    def test_stage_batched_selection_cannot_be_overridden_by_aggregate_size(self) -> None:
        pipeline = self._pipeline(STAGE_BATCHED_WORKFLOW_MODE)

        with mock.patch(
            "pipeline.main_pipeline.assert_selected_windows_models_installed"
        ):
            result = pipeline.batch_process()

        self.assertEqual(result, "stage")
        pipeline.stage_batched_processor.batch_process.assert_called_once_with(
            pipeline.main_page.image_files
        )
        pipeline.batch_processor.batch_process.assert_not_called()
        pipeline.main_page.file_handler.image_resource_plan.assert_not_called()

    def test_explicit_legacy_selection_remains_legacy_for_the_same_input(self) -> None:
        pipeline = self._pipeline(LEGACY_PAGE_WORKFLOW_MODE)

        with mock.patch(
            "pipeline.main_pipeline.assert_selected_windows_models_installed"
        ):
            result = pipeline.batch_process()

        self.assertEqual(result, "legacy")
        pipeline.batch_processor.batch_process.assert_called_once_with(
            pipeline.main_page.image_files
        )
        pipeline.stage_batched_processor.batch_process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
