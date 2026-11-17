from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

from app.controllers.batch_report import BatchReportController
from app.controllers.image import ImageStateController
from modules.utils.run_report import build_run_report


class _FakeButton:
    def __init__(self) -> None:
        self.enabled: bool | None = None

    def setEnabled(self, value: bool) -> None:
        self.enabled = bool(value)


class _FakeRadio:
    def isChecked(self) -> bool:
        return True


class _FakeMain:
    def __init__(self) -> None:
        self.batch_report_button = _FakeButton()
        self.retry_failed_button = _FakeButton()
        self.one_page_auto_button = _FakeButton()
        self.automatic_radio = _FakeRadio()
        self._batch_active = False
        self.curr_img_idx = 0
        self.image_files = ["/tmp/page-001.png"]
        self.image_states = {"/tmp/page-001.png": {"skip": False}}
        self.dirty_count = 0

    def mark_project_dirty(self, *, render_dirty: bool = True) -> None:
        self.dirty_count += 1
        self.render_dirty = render_dirty


class BatchReportRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_project_dirty_can_skip_render_dirty_without_changing_default(self) -> None:
        from types import SimpleNamespace
        from unittest import mock
        from controller import ComicTranslate

        main = SimpleNamespace(
            _bump_dirty_revision=mock.Mock(),
            mark_render_dirty=mock.Mock(),
            _update_window_modified=mock.Mock(),
        )
        ComicTranslate.mark_project_dirty(main, render_dirty=False)
        self.assertTrue(main._manual_dirty)
        main.mark_render_dirty.assert_not_called()
        main._bump_dirty_revision.assert_called_once()
        ComicTranslate.mark_project_dirty(main)
        main.mark_render_dirty.assert_called_once()

    def test_preflight_errors_are_preserved_in_project_payload(self) -> None:
        main = _FakeMain()
        ctrl = BatchReportController(main)
        ctrl.start_batch_report(["/tmp/page-001.png"], run_type="batch")
        ctrl.register_preflight_error("HunyuanOCR runtime setup failed", "No such image")
        finalized = ctrl.finalize_batch_report(False)

        self.assertIsNotNone(finalized)
        self.assertEqual(len(finalized["preflight_errors"]), 1)
        self.assertTrue(main.batch_report_button.enabled)
        self.assertEqual(main.dirty_count, 1)
        self.assertFalse(main.render_dirty)

        payload = ctrl.export_latest_report_for_project()
        self.assertIsNotNone(payload)
        self.assertEqual(payload["preflight_errors"][0]["title"], "HunyuanOCR runtime setup failed")
        self.assertEqual(payload["preflight_errors"][0]["details"], "No such image")

        imported_main = _FakeMain()
        imported = BatchReportController(imported_main)
        imported.import_latest_report_from_project(payload)
        reexported = imported.export_latest_report_for_project()
        self.assertIsNotNone(reexported)
        self.assertEqual(reexported["preflight_errors"], payload["preflight_errors"])
        self.assertTrue(imported_main.batch_report_button.enabled)

    def test_preflight_warnings_are_deduplicated_and_preserved(self) -> None:
        main = _FakeMain()
        ctrl = BatchReportController(main)
        ctrl.start_batch_report(["/tmp/page-001.png"], run_type="batch")
        ctrl.register_preflight_warning(
            "PDF import memory limit applied",
            "Pages: 3. Requested/applied sizes: 3: 30000×20000 → 12247×8165.",
        )
        ctrl.register_preflight_warning(
            "PDF import memory limit applied",
            "Pages: 3. Requested/applied sizes: 3: 30000×20000 → 12247×8165.",
        )
        finalized = ctrl.finalize_batch_report(False)

        self.assertIsNotNone(finalized)
        self.assertEqual(len(finalized["preflight_warnings"]), 1)
        payload = ctrl.export_latest_report_for_project()
        imported = BatchReportController(_FakeMain())
        imported.import_latest_report_from_project(payload)
        self.assertEqual(
            imported.export_latest_report_for_project()["preflight_warnings"],
            payload["preflight_warnings"],
        )

    def test_compiled_korean_pdf_warning_is_available(self) -> None:
        translator = QtCore.QTranslator()
        qm_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "resources",
            "translations",
            "compiled",
            "ct_ko.qm",
        )
        self.assertTrue(translator.load(qm_path))
        self._app.installTranslator(translator)
        try:
            translated = QtCore.QCoreApplication.translate(
                "PdfImport", "PDF import memory limit applied"
            )
            statuses = {
                source: QtCore.QCoreApplication.translate("BatchReportController", source)
                for source in (
                    "Completed",
                    "Cancelled",
                    "Failed",
                    "Interrupted",
                    "Paused",
                    "Report failed",
                    "Cleanup failed",
                    "Incomplete",
                    "The run finished, but its report could not be saved. Outputs are preserved.",
                    "The run finished, but some results or cleanup remain unverified. Outputs are preserved.",
                )
            }
        finally:
            self._app.removeTranslator(translator)

        self.assertEqual(translated, "PDF 가져오기 메모리 한도 적용")
        self.assertEqual(
            statuses,
            {
                "Completed": "완료",
                "Cancelled": "취소됨",
                "Failed": "실패",
                "Interrupted": "중단됨",
                "Paused": "일시 중지됨",
                "Report failed": "보고 실패",
                "Cleanup failed": "정리 실패",
                "Incomplete": "미완료",
                "The run finished, but its report could not be saved. Outputs are preserved.": "실행은 끝났지만 보고서를 저장하지 못했습니다. 출력 결과는 보존되어 있습니다.",
                "The run finished, but some results or cleanup remain unverified. Outputs are preserved.": "실행은 끝났지만 일부 결과나 정리 상태를 확인하지 못했습니다. 출력 결과는 보존되어 있습니다.",
            },
        )

    def test_inpaint_skip_preserves_stage_and_boundary_reason(self) -> None:
        main = _FakeMain()
        ctrl = BatchReportController(main)
        ctrl.start_batch_report(["/tmp/page-001.png"], run_type="batch")

        ctrl.register_batch_skip(
            "/tmp/page-001.png",
            "inpaint",
            "IndexError: index 2160 is out of bounds for axis 0 with size 2160\n\nTraceback...",
        )
        finalized = ctrl.finalize_batch_report(False)

        self.assertIsNotNone(finalized)
        entry = finalized["skipped_entries"][0]
        self.assertEqual(entry["stages"], ["inpaint"])
        self.assertIn("index 2160", entry["errors"][0])
        self.assertIn("Inpainting failed", entry["reasons"][0])
        self.assertIn("Mask boundary exceeded image bounds", entry["reasons"][0])

    def test_legacy_translation_skip_reason_maps_to_webtoon_translation(self) -> None:
        main = _FakeMain()
        ctrl = BatchReportController(main)
        ctrl.start_batch_report(["/tmp/page-001.png"], run_type="webtoon_batch")

        ctrl.register_batch_skip("/tmp/page-001.png", "Translation", "translator timed out")
        finalized = ctrl.finalize_batch_report(False)

        self.assertIsNotNone(finalized)
        entry = finalized["skipped_entries"][0]
        self.assertEqual(entry["stages"], ["Translation"])
        self.assertIn("Webtoon translation chunk failed", entry["reasons"][0])
        self.assertNotIn("Page processing failed", entry["reasons"][0])

    def test_retry_preserves_unresolved_pages_and_current_details(self) -> None:
        main = _FakeMain()
        main.image_files = ["a.png", "b.png", "c.png"]
        main.image_states = {path: {"skip": False} for path in main.image_files}
        ctrl = BatchReportController(main)
        ctrl.start_batch_report(main.image_files)
        for path in main.image_files:
            ctrl.register_batch_skip(path, "ocr", "old failure")
        ctrl.finalize_batch_report(False)

        ctrl.start_batch_report(["a.png", "b.png"], run_type="retry_failed")
        ctrl.register_batch_success("a.png")
        ctrl.register_batch_success("b.png")
        ctrl.register_batch_skip("b.png", "render", "review_required: missing render")
        result = ctrl.finalize_batch_report(False)
        self.assertEqual(result["retry_paths"], ["b.png", "c.png"])
        self.assertEqual(result["skipped_entries"][0]["stages"], ["render"])
        self.assertIn("missing render", result["skipped_entries"][0]["errors"][0])
        restored = BatchReportController(main)
        restored.import_latest_report_from_project(ctrl.export_latest_report_for_project())
        self.assertEqual(restored.get_latest_retry_paths(), ["b.png", "c.png"])
        main.image_files = ["c.png"]
        self.assertEqual(restored.get_latest_retry_paths(), ["c.png"])
        restored.clear_latest_report()
        self.assertEqual(restored.get_latest_retry_paths(), [])

    def test_retry_aggregates_observed_page_outcomes_and_round_trips(self) -> None:
        main = _FakeMain()
        main.image_files = ["a.png", "b.png"]
        main.image_states = {path: {"skip": False} for path in main.image_files}
        ctrl = BatchReportController(main)
        ctrl.start_batch_report(main.image_files)
        ctrl.register_batch_success("a.png")
        ctrl.register_batch_skip("b.png", "ocr", "first attempt failed")
        self.assertEqual(ctrl.finalize_batch_report(False)["retry_paths"], ["b.png"])

        ctrl.start_batch_report(["b.png"], run_type="retry_failed")
        ctrl.register_batch_success("b.png")
        report = ctrl.finalize_batch_report(False)

        self.assertEqual(report["total_images"], 2)
        self.assertEqual(report["run_total_images"], 1)
        self.assertEqual(report["completed_count"], 2)
        self.assertEqual(report["unjudged_count"], 0)
        self.assertEqual(report["retry_paths"], [])
        self.assertEqual(
            {item["image_path"]: item["status"] for item in report["page_outcomes"]},
            {"a.png": "completed", "b.png": "completed"},
        )

        restored = BatchReportController(_FakeMain())
        restored.import_latest_report_from_project(ctrl.export_latest_report_for_project())
        self.assertEqual(
            restored.export_latest_report_for_project()["page_outcomes"],
            report["page_outcomes"],
        )

    def test_partial_retry_clears_completed_old_skip_when_another_page_fails(self) -> None:
        main = _FakeMain()
        main.image_files = [
            "C:/ExampleWorkspace/a.png",
            "C:/ExampleWorkspace/b.png",
        ]
        main.image_states = {path: {"skip": False} for path in main.image_files}
        ctrl = BatchReportController(main)
        ctrl.start_batch_report(main.image_files)
        for path in main.image_files:
            ctrl.register_batch_skip(path, "OCR", "first attempt failed")
        ctrl.finalize_batch_report(False)

        ctrl.start_batch_report(main.image_files, run_type="retry_failed")
        ctrl.register_batch_skip(main.image_files[1], "OCR", "retry still failed")
        ctrl.register_pipeline_run_report(
            build_run_report(
                telemetry={},
                total_wall_sec=2.0,
                page_outcomes=[
                    {
                        "source_unit_index": 0,
                        "image_path": main.image_files[0],
                        "image_name": "a.png",
                        "output_path": "output/a.png",
                        "output_status": "measured",
                        "quality_status": "completed",
                    },
                    {
                        "source_unit_index": 1,
                        "image_path": main.image_files[1],
                        "image_name": "b.png",
                        "output_path": None,
                        "output_status": "unmeasured",
                        "failed_stage": "ocr",
                        "quality_status": "failed",
                    },
                ],
                output_summary={
                    "input_count": 2,
                    "output_count": 1,
                    "fallback_count": 0,
                    "fallbacks": [],
                    "missing": ["b.png"],
                    "missing_source_unit_indices": [1],
                },
                processing_status="failed",
                terminal_status="failed",
            )
        )

        report = ctrl.finalize_batch_report(False, processing_status="failed")

        self.assertEqual(report["retry_paths"], [main.image_files[1]])
        self.assertEqual(
            [entry["image_path"] for entry in report["skipped_entries"]],
            [main.image_files[1]],
        )
        self.assertEqual(
            {item["image_path"]: item["status"] for item in report["page_outcomes"]},
            {main.image_files[0]: "completed", main.image_files[1]: "failed"},
        )

    def test_explicit_mismatched_image_path_does_not_fall_back_by_index(self) -> None:
        main = _FakeMain()
        main.image_files = [
            "C:/ExampleWorkspace/a.png",
            "C:/ExampleWorkspace/b.png",
        ]
        main.image_states = {path: {"skip": False} for path in main.image_files}
        ctrl = BatchReportController(main)
        ctrl.start_batch_report(main.image_files)
        ctrl.register_pipeline_run_report(
            build_run_report(
                telemetry={},
                total_wall_sec=1.0,
                page_outcomes=[
                    {
                        "source_unit_index": 0,
                        "image_path": "C:/ExampleWorkspace/other.png",
                        "image_name": "other.png",
                        "output_path": None,
                        "failed_stage": "ocr",
                        "quality_status": "failed",
                    },
                    {
                        "source_unit_index": 1,
                        "image_path": main.image_files[1],
                        "image_name": "b.png",
                        "output_path": "output/b.png",
                        "output_status": "measured",
                        "quality_status": "completed",
                    },
                ],
                output_summary={
                    "input_count": 2,
                    "output_count": 1,
                    "fallback_count": 0,
                    "fallbacks": [],
                    "missing": ["other.png"],
                    "missing_source_unit_indices": [0],
                },
                processing_status="failed",
                terminal_status="failed",
            )
        )

        report = ctrl.finalize_batch_report(False, processing_status="failed")

        self.assertEqual(
            {item["image_path"]: item["status"] for item in report["run_page_outcomes"]},
            {main.image_files[0]: "unjudged", main.image_files[1]: "completed"},
        )

    def test_identityless_legacy_page_can_still_match_by_source_index(self) -> None:
        main = _FakeMain()
        path = "C:/ExampleWorkspace/a.png"
        main.image_files = [path]
        main.image_states = {path: {"skip": False}}
        ctrl = BatchReportController(main)
        ctrl.start_batch_report([path])
        ctrl.register_pipeline_run_report(
            build_run_report(
                telemetry={},
                total_wall_sec=1.0,
                page_outcomes=[
                    {
                        "source_unit_index": 0,
                        "image_name": "a.png",
                        "output_path": "output/a.png",
                        "output_status": "measured",
                        "quality_status": "completed",
                    }
                ],
                output_summary={
                    "input_count": 1,
                    "output_count": 1,
                    "fallback_count": 0,
                    "fallbacks": [],
                    "missing": [],
                },
                processing_status="completed",
                terminal_status="completed",
            )
        )

        report = ctrl.finalize_batch_report(False, processing_status="completed")

        self.assertEqual(report["run_page_outcomes"], [{"image_path": path, "status": "completed"}])

    def test_auto_export_uses_current_results_and_blocks_unresolved_prior_pages(self) -> None:
        main = _FakeMain()
        main.image_files = ["a.png", "b.png"]
        main.image_states = {path: {"skip": False} for path in main.image_files}
        ctrl = BatchReportController(main)
        ctrl.start_batch_report(main.image_files)
        ctrl.register_pipeline_run_report(
            build_run_report(
                telemetry={},
                total_wall_sec=1.0,
                page_outcomes=[
                    {
                        "source_unit_index": 0,
                        "image_path": "a.png",
                        "image_name": "a.png",
                        "output_path": "output/a.png",
                        "quality_status": "completed",
                    },
                    {
                        "source_unit_index": 1,
                        "image_path": "b.png",
                        "image_name": "b.png",
                        "output_path": None,
                        "output_status": "unmeasured",
                        "quality_status": "failed",
                    },
                ],
                output_summary={
                    "input_count": 2,
                    "output_count": 1,
                    "fallback_count": 0,
                    "fallbacks": [],
                    "missing": ["b.png"],
                },
                processing_status="failed",
                terminal_status="failed",
            )
        )
        ctrl.finalize_batch_report(False, processing_status="failed")

        ctrl.start_batch_report(["a.png"])
        ctrl.register_batch_success("a.png")
        ctrl.register_pipeline_run_report(
            build_run_report(
                telemetry={},
                total_wall_sec=1.0,
                page_outcomes=[
                    {
                        "source_unit_index": 0,
                        "image_path": "a.png",
                        "image_name": "a.png",
                        "output_path": "output/a.png",
                        "quality_status": "completed",
                    }
                ],
                output_summary={
                    "input_count": 1,
                    "output_count": 1,
                    "fallback_count": 0,
                    "fallbacks": [],
                    "missing": [],
                },
                processing_status="completed",
                terminal_status="completed",
                cleanup_status="completed",
            )
        )

        self.assertEqual(
            ctrl.auto_export_ineligibility_reason(
                processing_status="completed",
                application_cleanup_status="completed",
                archive_status="not_applicable",
            ),
            "previous_run_has_unresolved_pages",
        )

    def test_legacy_cancel_without_page_outcomes_keeps_unretried_pages_unjudged(self) -> None:
        from types import SimpleNamespace
        from controller import ComicTranslate
        from modules.utils.textblock import TextBlock

        main = _FakeMain()
        main.image_files = ["example_source_a.png", "example_source_b.png"]
        main.image_states = {path: {"skip": False} for path in main.image_files}
        ctrl = BatchReportController(main)
        ctrl.import_latest_report_from_project(
            {
                "was_cancelled": True,
                "paths": list(main.image_files),
                "total_images": 2,
            }
        )

        ctrl.start_batch_report(["example_source_a.png"])
        ctrl.register_batch_success("example_source_a.png")
        ctrl.register_pipeline_run_report(
            build_run_report(
                telemetry=None,
                total_wall_sec=1.0,
                page_outcomes=[{
                    "source_unit_index": 0,
                    "image_name": "example_source_a.png",
                    "output_path": "example_output_a.png",
                    "quality_status": "completed",
                }],
                output_summary={
                    "input_count": 1,
                    "output_count": 1,
                    "fallback_count": 0,
                    "fallbacks": [],
                    "missing": [],
                },
                processing_status="completed",
                terminal_status="completed",
                cleanup_status="completed",
            )
        )

        reason = ctrl.auto_export_ineligibility_reason(
            processing_status="completed",
            application_cleanup_status="completed",
            archive_status="not_applicable",
        )
        with tempfile.TemporaryDirectory() as temporary:
            prior_exchange = Path(temporary) / "example_source.txt"
            prior_exchange.write_bytes(b"previous full exchange")
            exporter = ComicTranslate.__new__(ComicTranslate)
            exporter.settings_page = SimpleNamespace(
                get_export_settings=lambda: {"auto_export_source_txt": True}
            )
            exporter.image_files = ["example_source_a.png"]
            exporter.image_states = {
                "example_source_a.png": {
                    "blk_list": [TextBlock(text="partial current run")]
                }
            }
            exporter._ensure_txt_md_ready = ComicTranslate._ensure_txt_md_ready.__get__(exporter)
            exporter._txt_md_auto_save_path = lambda _target, _suffix: str(prior_exchange)
            exporter._write_txt_md_exchange = ComicTranslate._write_txt_md_exchange.__get__(exporter)
            auto_export_result = ComicTranslate._run_txt_md_auto_exports.__get__(exporter)(
                ["example_source_a.png"],
                attempt=reason is None,
                reason_code=reason,
            )
            report = ctrl.finalize_batch_report(
                False,
                processing_status="completed",
                application_cleanup_status="completed",
                auto_export_result=auto_export_result,
            )
            preserved_exchange = prior_exchange.read_bytes()

        self.assertEqual(reason, "previous_run_has_unresolved_pages")
        self.assertEqual(auto_export_result["status"], "not_attempted")
        self.assertEqual(preserved_exchange, b"previous full exchange")
        self.assertEqual(report["processing_status"], "completed")
        self.assertEqual(report["terminal_status"], "incomplete")
        self.assertEqual(
            report["page_outcomes"],
            [
                {"image_path": "example_source_a.png", "status": "completed"},
                {"image_path": "example_source_b.png", "status": "unjudged"},
            ],
        )

    def test_legacy_cancel_history_does_not_fail_a_run_without_export_selection(self) -> None:
        main = _FakeMain()
        main.image_files = ["example_source_a.png", "example_source_b.png"]
        main.image_states = {path: {"skip": False} for path in main.image_files}
        ctrl = BatchReportController(main)
        ctrl.import_latest_report_from_project(
            {"was_cancelled": True, "paths": list(main.image_files), "total_images": 2}
        )
        ctrl.start_batch_report(["example_source_a.png"])
        ctrl.register_batch_success("example_source_a.png")
        ctrl.register_pipeline_run_report(
            build_run_report(
                telemetry=None,
                total_wall_sec=1.0,
                page_outcomes=[{
                    "source_unit_index": 0,
                    "image_name": "example_source_a.png",
                    "output_path": "example_output_a.png",
                    "quality_status": "completed",
                }],
                output_summary={
                    "input_count": 1,
                    "output_count": 1,
                    "fallback_count": 0,
                    "fallbacks": [],
                    "missing": [],
                },
                processing_status="completed",
                terminal_status="completed",
                cleanup_status="completed",
            )
        )

        report = ctrl.finalize_batch_report(
            False,
            processing_status="completed",
            application_cleanup_status="completed",
            auto_export_result={"status": "not_applicable", "selected_targets": []},
        )

        self.assertEqual(report["processing_status"], "completed")
        self.assertNotEqual(report["terminal_status"], "failed")
        self.assertEqual(report["auto_export_status"], "not_applicable")

    def test_current_unobserved_page_cannot_inherit_previous_success_for_export(self) -> None:
        main = _FakeMain()
        path = main.image_files[0]
        ctrl = BatchReportController(main)
        ctrl.start_batch_report([path])
        ctrl.register_batch_success(path)
        ctrl.register_pipeline_run_report(
            build_run_report(
                telemetry={},
                total_wall_sec=1.0,
                page_outcomes=[
                    {
                        "source_unit_index": 0,
                        "image_path": path,
                        "image_name": "page-001.png",
                        "output_path": "output/page-001.png",
                        "quality_status": "completed",
                    }
                ],
                output_summary={
                    "input_count": 1,
                    "output_count": 1,
                    "fallback_count": 0,
                    "fallbacks": [],
                    "missing": [],
                },
                processing_status="completed",
                terminal_status="completed",
                cleanup_status="completed",
            )
        )
        ctrl.finalize_batch_report(
            False,
            processing_status="completed",
            application_cleanup_status="completed",
        )

        ctrl.start_batch_report([path])
        ctrl.register_batch_success(path)
        self.assertEqual(
            ctrl.auto_export_ineligibility_reason(
                processing_status="completed",
                application_cleanup_status="completed",
                archive_status="not_applicable",
            ),
            "pipeline_outcome_not_observed",
        )

    def test_foreign_explicit_pipeline_identity_blocks_completed_path_fallback(self) -> None:
        main = _FakeMain()
        path = main.image_files[0]
        ctrl = BatchReportController(main)
        ctrl.start_batch_report([path])
        ctrl.register_batch_success(path)
        ctrl.register_pipeline_run_report(
            build_run_report(
                telemetry={},
                total_wall_sec=1.0,
                page_outcomes=[{
                    "source_unit_index": 0,
                    "image_path": "example_foreign_source.png",
                    "image_name": "example_foreign_source.png",
                    "output_path": "example_foreign_output.png",
                    "quality_status": "completed",
                }],
                output_summary={
                    "input_count": 1,
                    "output_count": 1,
                    "fallback_count": 0,
                    "fallbacks": [],
                    "missing": [],
                },
                processing_status="completed",
                terminal_status="completed",
                cleanup_status="completed",
            )
        )

        report = ctrl.finalize_batch_report(
            False,
            processing_status="completed",
            application_cleanup_status="completed",
        )

        self.assertEqual(
            report["run_page_outcomes"],
            [{"image_path": path, "status": "unjudged"}],
        )
        self.assertEqual(report["terminal_status"], "incomplete")

    def test_legacy_saved_output_receipt_builds_pipeline_report_for_selected_export(self) -> None:
        from types import SimpleNamespace
        from controller import ComicTranslate
        from modules.utils.textblock import TextBlock

        main = _FakeMain()
        path = "example_source_page.png"
        main.image_files = [path]
        main.image_states = {path: {"skip": False}}

        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "example_translated.png"
            output_path.write_bytes(b"saved output")
            exchange_path = Path(temporary) / "example_source.txt"
            ctrl = BatchReportController(main)
            ctrl.start_batch_report([path], run_report_log_dir=temporary)
            ctrl.register_batch_output(path, str(output_path))

            reason = ctrl.auto_export_ineligibility_reason(
                processing_status="completed",
                application_cleanup_status="completed",
                archive_status="not_applicable",
            )
            comic = ComicTranslate.__new__(ComicTranslate)
            comic.settings_page = SimpleNamespace(
                get_export_settings=lambda: {"auto_export_source_txt": True}
            )
            comic.image_files = [path]
            comic.image_states = {
                path: {"blk_list": [TextBlock(text="example source text")]}
            }
            comic._ensure_txt_md_ready = ComicTranslate._ensure_txt_md_ready.__get__(comic)
            comic._txt_md_auto_save_path = lambda _target, _suffix: str(exchange_path)
            comic._write_txt_md_exchange = ComicTranslate._write_txt_md_exchange.__get__(comic)
            auto_export_result = ComicTranslate._run_txt_md_auto_exports.__get__(comic)(
                [path],
                attempt=reason is None,
                reason_code=reason,
            )
            report = ctrl.finalize_batch_report(
                False,
                processing_status="completed",
                application_cleanup_status="completed",
                auto_export_result=auto_export_result,
            )
            saved_report = json.loads(
                next(Path(temporary).glob("*.json")).read_text(encoding="utf-8")
            )
            exchange_content = exchange_path.read_text(encoding="utf-8")

        self.assertIsNone(reason)
        self.assertEqual(
            report["run_page_outcomes"],
            [{"image_path": path, "status": "completed"}],
        )
        self.assertEqual(report["auto_export_status"], "completed")
        self.assertEqual(report["terminal_status"], "completed")
        self.assertEqual(saved_report["quality"]["status"], "completed")
        self.assertEqual(saved_report["pipeline_cleanup_status"], "not_applicable")
        self.assertIn("example source text", exchange_content)

    def test_direct_finalize_keeps_synthesized_legacy_pipeline_bookkeeping(self) -> None:
        main = _FakeMain()
        path = "example_source_page.png"
        main.image_files = [path]
        main.image_states = {path: {"skip": False}}

        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "example_translated.png"
            output_path.write_bytes(b"saved output")
            ctrl = BatchReportController(main)
            ctrl.start_batch_report([path], run_report_log_dir=temporary)
            self.assertTrue(ctrl.register_batch_output(path, str(output_path)))

            report = ctrl.finalize_batch_report(
                False,
                processing_status="completed",
                application_cleanup_status="completed",
            )
            saved_report = json.loads(
                next(Path(temporary).glob("*.json")).read_text(encoding="utf-8")
            )

        self.assertEqual(report["processing_status"], "completed")
        self.assertEqual(report["pipeline_cleanup_status"], "not_applicable")
        self.assertEqual(report["cleanup_status"], "completed")
        self.assertEqual(report["terminal_status"], "completed")
        self.assertIsNone(saved_report["telemetry"]["value"])
        self.assertEqual(
            saved_report["telemetry"]["reason_code"],
            "pipeline_telemetry_not_observed",
        )

    def test_partial_failed_legacy_fallback_keeps_observed_telemetry(self) -> None:
        from pipeline.performance_telemetry import PipelinePerformanceTelemetry

        main = _FakeMain()
        paths = ["example_source_a.png", "example_source_b.png"]
        main.image_files = list(paths)
        main.image_states = {path: {"skip": False} for path in paths}

        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "example_output_a.png"
            output_path.write_bytes(b"one saved page")
            ctrl = BatchReportController(main)
            ctrl.start_batch_report(paths, run_report_log_dir=temporary)
            self.assertTrue(ctrl.register_batch_output(paths[0], str(output_path)))
            telemetry = PipelinePerformanceTelemetry(resource_sampling_enabled=False)
            telemetry.record_stage_detail(
                stage="translate",
                operation="previous_run_only",
                elapsed_ms=999.0,
                count=1,
            )
            telemetry.observe_event(
                "batch_run_start",
                payload={"pipeline_mode": "batch", "run_type": "batch"},
            )
            telemetry.record_stage_detail(
                stage="ocr",
                operation="current_partial_run",
                elapsed_ms=7.0,
                count=1,
                outcome="failed",
            )
            telemetry.observe_event("batch_run_failed")
            self.assertTrue(ctrl.register_batch_telemetry(telemetry))

            report = ctrl.finalize_batch_report(
                False,
                processing_status="failed",
                application_cleanup_status="completed",
            )
            saved_report = json.loads(
                next(Path(temporary).glob("*.json")).read_text(encoding="utf-8")
            )

        self.assertEqual(report["terminal_status"], "failed")
        self.assertEqual(
            report["run_page_outcomes"],
            [
                {"image_path": paths[0], "status": "completed"},
                {"image_path": paths[1], "status": "unjudged"},
            ],
        )
        telemetry_data = saved_report["telemetry"]["value"]
        self.assertEqual(telemetry_data["status"], "failed")
        self.assertIn(
            "current_partial_run",
            telemetry_data["stage_details"]["ocr"],
        )
        self.assertNotIn(
            "previous_run_only",
            telemetry_data["stage_details"].get("translate", {}),
        )

    def test_shutdown_records_selected_export_as_not_attempted_or_not_observed(self) -> None:
        from types import SimpleNamespace
        from unittest import mock
        from controller import ComicTranslate

        for settings, expected_status, expected_reason in (
            (
                lambda: {"auto_export_source_txt": True},
                "not_attempted",
                "shutdown",
            ),
            (
                lambda: (_ for _ in ()).throw(OSError("settings unavailable")),
                "not_observed",
                "auto_export_settings_not_observed",
            ),
        ):
            with self.subTest(expected_status=expected_status):
                main = _FakeMain()
                path = main.image_files[0]
                ctrl = BatchReportController(main)
                ctrl.start_batch_report([path])

                comic = ComicTranslate.__new__(ComicTranslate)
                comic._memlogger = None
                comic._batch_pause_requested = False
                comic._batch_cancel_requested = True
                comic._batch_failed = False
                comic._batch_active = True
                comic._current_batch_run_type = "batch"
                comic._is_shutting_down = True
                comic.selected_batch = [path]
                comic.batch_report_ctrl = ctrl
                comic.settings_page = SimpleNamespace(get_export_settings=settings)
                comic._release_batch_model_caches = mock.Mock(return_value=("completed", []))
                comic._finalize_batch_report = ComicTranslate._finalize_batch_report.__get__(comic)
                comic._run_txt_md_auto_exports = ComicTranslate._run_txt_md_auto_exports.__get__(comic)

                with mock.patch("controller.finish_debug_artifact_run"):
                    ComicTranslate.on_batch_process_finished(comic)

                report = ctrl._latest_batch_report
                self.assertEqual(report["processing_status"], "interrupted")
                self.assertEqual(report["auto_export_status"], expected_status)
                self.assertEqual(report["auto_export_targets"], ["source.txt"] if expected_status == "not_attempted" else [])
                self.assertEqual(report["auto_export_reason_code"], expected_reason)

    def test_legacy_batch_save_hook_reports_its_real_output_file(self) -> None:
        from collections import deque
        from types import SimpleNamespace
        from unittest import mock

        import numpy as np
        from PIL import Image
        from controller import ComicTranslate
        from pipeline.batch_processor import BatchProcessor
        from pipeline.performance_telemetry import PipelinePerformanceTelemetry

        with tempfile.TemporaryDirectory() as temporary:
            source_path = Path(temporary) / "example_source.png"
            Image.new("RGB", (8, 8), (1, 2, 3)).save(source_path)
            output_root = Path(temporary) / "output"
            page_state = {
                "source_lang": "Japanese",
                "target_lang": "Korean",
                "skip": False,
            }
            main = _FakeMain()
            main.image_files = [str(source_path)]
            main.image_states = {str(source_path): page_state}
            main.curr_img_idx = 0
            main.lang_mapping = {"Japanese": "Japanese", "Korean": "Korean"}
            main.file_handler = SimpleNamespace(archive_info=None)
            settings_page = SimpleNamespace(
                get_hd_strategy_settings=lambda: {"strategy": "Original"},
                get_tool_selection=lambda _tool: "example-tool",
                get_mask_refiner_settings=lambda: {},
                is_gpu_enabled=lambda: False,
                ui=SimpleNamespace(value_mappings={}),
            )
            main.settings_page = settings_page
            main.image_ctrl = SimpleNamespace(
                load_image=lambda _path: np.zeros((8, 8, 3), dtype=np.uint8),
                mark_processing_stage=mock.Mock(),
                update_processing_summary=mock.Mock(),
            )
            main.render_state_ready = SimpleNamespace(emit=mock.Mock())
            main.get_resolved_export_settings = lambda: {
                "resolved_automatic_output_target": "images"
            }
            main.get_reserved_automatic_output_series_dir = (
                lambda *_args, **_kwargs: str(output_root)
            )
            main.is_current_task_cancelled = lambda: False
            main.emit_memlog = mock.Mock()
            report_ctrl = BatchReportController(main)
            main.batch_report_ctrl = report_ctrl
            main.register_batch_telemetry = (
                ComicTranslate.register_batch_telemetry.__get__(main)
            )
            main.register_batch_output = ComicTranslate.register_batch_output.__get__(main)
            report_ctrl.start_batch_report(
                [str(source_path)],
                run_report_log_dir=temporary,
            )
            processor = BatchProcessor.__new__(BatchProcessor)
            processor.main_page = main
            processor.cache_manager = SimpleNamespace()
            processor.block_detection = SimpleNamespace(
                block_detector_cache=SimpleNamespace(
                    detect=lambda _image: [],
                    last_mask_details=None,
                    detector="example-detector",
                    last_engine_name="",
                    last_device="cpu",
                )
            )
            processor.inpainting = SimpleNamespace()
            processor.ocr_handler = SimpleNamespace()
            processor._recent_page_durations = deque(maxlen=5)
            telemetry = PipelinePerformanceTelemetry(resource_sampling_enabled=False)
            telemetry.record_stage_detail(
                stage="translate",
                operation="previous_run_only",
                elapsed_ms=999.0,
                count=1,
            )
            processor.performance_telemetry = telemetry
            processor.emit_progress = mock.Mock()
            processor._ensure_page_state = mock.Mock(return_value=page_state)
            processor._start_page_summary = mock.Mock()
            processor._log_page_start = mock.Mock()
            processor._log_page_done = mock.Mock()
            processor._write_inpaint_debug_exports = mock.Mock()

            with (
                mock.patch("pipeline.batch_processor.get_inpainter_runtime", return_value={"backend": "test"}),
                mock.patch("pipeline.batch_processor.resolve_device", return_value="cpu"),
            ):
                processor.batch_process([str(source_path)])

            receipt = report_ctrl._current_batch_report["batch_output_paths"][str(source_path)]
            self.assertTrue(Path(receipt).is_file())
            report = report_ctrl.finalize_batch_report(
                False,
                processing_status="completed",
                application_cleanup_status="completed",
            )
            saved_report = json.loads(
                next(Path(temporary).glob("*.json")).read_text(encoding="utf-8")
            )

        telemetry_data = saved_report["telemetry"]["value"]
        self.assertEqual(telemetry_data["status"], "completed")
        self.assertIn("detect", telemetry_data["stages"])
        self.assertNotIn(
            "previous_run_only",
            telemetry_data["stage_details"].get("translate", {}),
        )
        self.assertEqual(report["pipeline_cleanup_status"], "not_applicable")
        self.assertEqual(report["cleanup_status"], "completed")
        self.assertEqual(report["terminal_status"], "completed")

    def test_webtoon_saved_output_hook_reports_its_real_output_file(self) -> None:
        from types import SimpleNamespace
        from unittest import mock

        from PIL import Image
        from pipeline.webtoon_batch.processor import WebtoonBatchProcessor

        with tempfile.TemporaryDirectory() as temporary:
            source_path = Path(temporary) / "example_source.png"
            Image.new("RGB", (8, 8), (1, 2, 3)).save(source_path)
            export_settings = {
                "resolved_automatic_output_target": "images",
                "export_inpainted_image": False,
                "export_raw_text": False,
                "export_translated_text": False,
                "export_ocr_debug": False,
                "export_detector_overlay": False,
                "export_raw_mask": False,
                "export_mask_overlay": False,
                "export_cleanup_mask_delta": False,
                "export_debug_metadata": False,
            }
            main = SimpleNamespace(
                image_files=[str(source_path)],
                image_states={
                    str(source_path): {
                        "skip_render": False,
                        "blk_list": [],
                        "processing_summary": {},
                    }
                },
                file_handler=SimpleNamespace(archive_info=None),
                export_source_by_path={},
                project_file=None,
                temp_dir=temporary,
                image_ctrl=SimpleNamespace(update_processing_summary=mock.Mock()),
                get_reserved_automatic_output_series_dir=lambda *_args, **_kwargs: temporary,
                settings_page=SimpleNamespace(
                    get_hd_strategy_settings=lambda: {"strategy": "Original"},
                    get_tool_selection=lambda _tool: "example-tool",
                    ui=SimpleNamespace(value_mappings={}),
                ),
                register_batch_output=mock.Mock(),
            )
            processor = WebtoonBatchProcessor.__new__(WebtoonBatchProcessor)
            processor.main_page = main
            processor.block_detection = SimpleNamespace(block_detector_cache=None)
            processor.final_patches_for_save = {str(source_path): []}
            processor._export_run_tokens = {}
            processor._effective_export_settings = mock.Mock(return_value=export_settings)

            processor._save_final_rendered_page(0, str(source_path), "20261117")

            main.register_batch_output.assert_called_once()
            self.assertEqual(main.register_batch_output.call_args.args[0], str(source_path))
            self.assertTrue(Path(main.register_batch_output.call_args.args[1]).is_file())

    def test_webtoon_cancel_persists_current_run_telemetry(self) -> None:
        from types import SimpleNamespace
        from unittest import mock

        from controller import ComicTranslate
        from pipeline.performance_telemetry import PipelinePerformanceTelemetry
        from pipeline.webtoon_batch.processor import WebtoonBatchProcessor

        path = "example_source_page.png"
        main = _FakeMain()
        main.image_files = [path]
        main.image_states = {path: {"skip": True}}
        main.file_handler = SimpleNamespace()
        main.is_current_task_cancelled = lambda: True
        main.emit_memlog = mock.Mock()
        report_ctrl = BatchReportController(main)
        main.batch_report_ctrl = report_ctrl
        main.register_batch_telemetry = (
            ComicTranslate.register_batch_telemetry.__get__(main)
        )

        with tempfile.TemporaryDirectory() as temporary:
            report_ctrl.start_batch_report([path], run_report_log_dir=temporary)
            telemetry = PipelinePerformanceTelemetry(resource_sampling_enabled=False)
            telemetry.record_stage_detail(
                stage="translate",
                operation="previous_run_only",
                elapsed_ms=999.0,
                count=1,
            )
            processor = WebtoonBatchProcessor.__new__(WebtoonBatchProcessor)
            processor.main_page = main
            processor.performance_telemetry = telemetry
            processor.last_performance_stats = {}
            processor.final_patches_for_save = {}

            processor.webtoon_batch_process([path])
            report = report_ctrl.finalize_batch_report(
                True,
                processing_status="cancelled",
                application_cleanup_status="completed",
            )
            saved_report = json.loads(
                next(Path(temporary).glob("*.json")).read_text(encoding="utf-8")
            )

        telemetry_data = saved_report["telemetry"]["value"]
        self.assertEqual(report["terminal_status"], "cancelled")
        self.assertEqual(telemetry_data["status"], "cancelled")
        self.assertEqual(telemetry_data["metadata"]["pipeline_mode"], "webtoon")
        self.assertNotIn(
            "previous_run_only",
            telemetry_data["stage_details"].get("translate", {}),
        )

    def test_auto_export_stops_at_first_write_failure_and_preserves_prior_output(self) -> None:
        from controller import ComicTranslate
        from types import SimpleNamespace
        from unittest import mock

        comic = ComicTranslate.__new__(ComicTranslate)
        comic.settings_page = SimpleNamespace(
            get_export_settings=lambda: {
                "auto_export_source_txt": True,
                "auto_export_translation_md": True,
            }
        )
        comic._ensure_txt_md_ready = mock.Mock(return_value=True)
        comic._txt_md_auto_save_path = mock.Mock(side_effect=lambda target, suffix: f"{target}{suffix}")
        comic._write_txt_md_exchange = mock.Mock(side_effect=[None, OSError("write failed")])

        with mock.patch("controller.Messages.show_warning"):
            result = ComicTranslate._run_txt_md_auto_exports(comic, ["page.png"])

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["selected_targets"], ["source.txt", "translation.md"])
        self.assertEqual(result["completed_targets"], ["source.txt"])
        self.assertEqual(result["failed_target"], "translation.md")
        self.assertEqual(result["error"]["phase"], "auto_txt_md_export")
        self.assertEqual(result["error"]["error_type"], "OSError")
        self.assertEqual(comic._write_txt_md_exchange.call_count, 2)

    def test_no_selected_auto_exports_are_not_applicable(self) -> None:
        from controller import ComicTranslate
        from types import SimpleNamespace

        comic = ComicTranslate.__new__(ComicTranslate)
        comic.settings_page = SimpleNamespace(get_export_settings=lambda: {})

        result = ComicTranslate._run_txt_md_auto_exports(comic, ["page.png"])

        self.assertEqual(result["status"], "not_applicable")
        self.assertEqual(result["reason_code"], "no_auto_export_targets")
        self.assertEqual(result["selected_targets"], [])

    def test_selected_auto_exports_are_not_attempted_for_cancelled_run(self) -> None:
        from controller import ComicTranslate
        from types import SimpleNamespace
        from unittest import mock

        comic = ComicTranslate.__new__(ComicTranslate)
        comic.settings_page = SimpleNamespace(
            get_export_settings=lambda: {"auto_export_source_txt": True}
        )
        comic._write_txt_md_exchange = mock.Mock()

        result = ComicTranslate._run_txt_md_auto_exports(
            comic,
            ["page.png"],
            attempt=False,
            reason_code="run_cancelled",
        )

        self.assertEqual(result["status"], "not_attempted")
        self.assertEqual(result["reason_code"], "run_cancelled")
        self.assertEqual(result["selected_targets"], ["source.txt"])
        comic._write_txt_md_exchange.assert_not_called()

    def test_auto_export_failure_is_terminal_and_project_round_trips_phase(self) -> None:
        main = _FakeMain()
        path = main.image_files[0]
        ctrl = BatchReportController(main)
        with tempfile.TemporaryDirectory() as tmp:
            ctrl.start_batch_report([path], run_report_log_dir=tmp)
            ctrl.register_batch_success(path)
            ctrl.register_pipeline_run_report(
                build_run_report(
                    telemetry={},
                    total_wall_sec=1.0,
                    page_outcomes=[
                        {
                            "source_unit_index": 0,
                            "image_path": path,
                            "image_name": "page-001.png",
                            "output_path": "output/page-001.png",
                            "quality_status": "completed",
                        }
                    ],
                    output_summary={
                        "input_count": 1,
                        "output_count": 1,
                        "fallback_count": 0,
                        "fallbacks": [],
                        "missing": [],
                    },
                    processing_status="completed",
                    terminal_status="completed",
                    cleanup_status="completed",
                )
            )
            report = ctrl.finalize_batch_report(
                False,
                processing_status="completed",
                application_cleanup_status="completed",
                auto_export_result={
                    "status": "failed",
                    "selected_targets": ["source.txt", "translation.md"],
                    "completed_targets": ["source.txt"],
                    "failed_target": "translation.md",
                    "reason_code": "auto_export_write_failed",
                    "error": {
                        "phase": "auto_txt_md_export",
                        "error_type": "OSError",
                        "reason_code": "auto_export_write_failed",
                        "message": "write failed",
                    },
                },
            )
            run_report = json.loads(next(Path(tmp).glob("*.json")).read_text(encoding="utf-8"))

        self.assertEqual(report["processing_status"], "completed")
        self.assertEqual(report["terminal_status"], "failed")
        self.assertEqual(report["auto_export_status"], "failed")
        self.assertEqual(report["auto_export_completed_targets"], ["source.txt"])
        self.assertEqual(report["auto_export_failed_target"], "translation.md")
        self.assertEqual(run_report["auto_export_error"]["phase"], "auto_txt_md_export")
        payload = ctrl.export_latest_report_for_project()
        restored = BatchReportController(_FakeMain())
        restored.import_latest_report_from_project(payload)
        exported = restored.export_latest_report_for_project()
        self.assertEqual(exported["auto_export_status"], "failed")
        self.assertEqual(exported["auto_export_failed_target"], "translation.md")
        self.assertEqual(exported["auto_export_error"]["phase"], "auto_txt_md_export")

    def test_cancelled_unjudged_pages_are_retryable_and_legacy_cancel_stays_cancelled(self) -> None:
        main = _FakeMain()
        main.image_files = ["a.png", "b.png"]
        main.image_states = {path: {"skip": False} for path in main.image_files}
        ctrl = BatchReportController(main)
        ctrl.start_batch_report(main.image_files)
        ctrl.register_batch_success("a.png")
        report = ctrl.finalize_batch_report(True)

        self.assertEqual(report["retry_paths"], ["a.png", "b.png"])
        self.assertEqual(report["page_outcomes"], [
            {"image_path": "a.png", "status": "unjudged"},
            {"image_path": "b.png", "status": "unjudged"},
        ])

        restored = BatchReportController(_FakeMain())
        restored.import_latest_report_from_project({"was_cancelled": True})
        self.assertEqual(restored._latest_batch_report["terminal_status"], "cancelled")

    def test_cancel_and_unconfirmed_translation_do_not_resolve_failures(self) -> None:
        main = _FakeMain()
        ctrl = BatchReportController(main)
        path = main.image_files[0]
        ctrl.start_batch_report([path])
        ctrl.register_batch_skip(path, "render", "review_required: missing render")
        ctrl.finalize_batch_report(False)
        main.image_states[path]["blk_list"] = [type("Block", (), {"translation": "text"})()]
        ctrl.start_batch_report([path], run_type="one_page_auto")
        self.assertEqual(ctrl.finalize_batch_report(False)["retry_paths"], [path])
        ctrl.start_batch_report([path], run_type="retry_failed")
        ctrl.register_batch_success(path)
        self.assertEqual(ctrl.finalize_batch_report(True)["retry_paths"], [path])

    def test_failed_terminal_report_keeps_unprocessed_pages_unjudged(self) -> None:
        from controller import ComicTranslate
        from unittest import mock

        main = _FakeMain()
        main.image_files = ["a.png", "b.png"]
        main.image_states = {path: {"skip": False} for path in main.image_files}
        with tempfile.TemporaryDirectory() as tmp:
            ctrl = BatchReportController(main)
            ctrl.start_batch_report(main.image_files, run_report_log_dir=tmp)
            run_report_path = next(Path(tmp).glob("*.json"))
            start_record = json.loads(Path(run_report_path).read_text(encoding="utf-8"))
            self.assertEqual(start_record["terminal_status"], "running")
            ctrl.register_batch_success("a.png")
            comic = ComicTranslate.__new__(ComicTranslate)
            comic._memlogger = None
            comic._batch_pause_requested = False
            comic._batch_cancel_requested = False
            comic._batch_failed = True
            comic._is_shutting_down = False
            comic.selected_batch = main.image_files
            comic.batch_report_ctrl = ctrl
            comic._finish_batch_process_ui = mock.Mock(
                side_effect=lambda **kwargs: ComicTranslate._finalize_batch_report(
                    comic,
                    kwargs["was_cancelled"],
                    processing_status=kwargs["processing_status"],
                    application_cleanup_status="completed",
                    archive_status=kwargs["archive_status"],
                )
            )

            ComicTranslate.on_batch_process_finished(comic)
            report = ctrl._latest_batch_report

            self.assertEqual(report["completed_count"], 1)
            self.assertEqual(report["unjudged_count"], 1)
            self.assertEqual(report["terminal_status"], "failed")
            terminal_record = json.loads(Path(run_report_path).read_text(encoding="utf-8"))
            self.assertEqual(terminal_record["terminal_status"], "failed")
            self.assertEqual(terminal_record["quality"]["unjudged_units"]["value"], 1)
            self.assertEqual(report["retry_paths"], ["b.png"])
            payload = ctrl.export_latest_report_for_project()
            restored = BatchReportController(_FakeMain())
            restored.import_latest_report_from_project(payload)
            self.assertEqual(
                restored.export_latest_report_for_project()["terminal_status"],
                "failed",
            )

    def test_failed_pipeline_quality_cannot_become_completed_in_saved_report(self) -> None:
        from controller import ComicTranslate
        from modules.utils.run_report import build_run_report
        from types import SimpleNamespace
        from unittest import mock

        main = _FakeMain()
        path = main.image_files[0]
        skipped_path = "skip.png"
        main.image_files = [skipped_path, path]
        main.image_states = {skipped_path: {"skip": True}, path: {"skip": False}}
        with tempfile.TemporaryDirectory() as tmp:
            ctrl = BatchReportController(main)
            ctrl.start_batch_report(main.image_files, run_report_log_dir=tmp)
            ctrl.register_batch_success(path)
            ctrl.register_pipeline_run_report(
                build_run_report(
                    telemetry={"status": "completed"},
                    total_wall_sec=1.0,
                    page_outcomes=[{
                        "source_unit_index": 1,
                        "image_path": path,
                        "image_name": os.path.basename(path),
                        "output_path": "output/page.png",
                        "failed_stage": "inpaint",
                        "quality_status": "failed",
                    }],
                    output_summary={
                        "input_count": 1,
                        "output_count": 1,
                        "fallback_count": 1,
                        "fallbacks": [{"source_unit_index": 1, "image_name": os.path.basename(path), "kind": "inpainted"}],
                        "missing": [],
                    },
                    processing_status="completed",
                    terminal_status="completed",
                    report_status="completed",
                    cleanup_status="completed",
                )
            )

            comic = ComicTranslate.__new__(ComicTranslate)
            comic._memlogger = None
            comic.pipeline = SimpleNamespace(release_model_caches=mock.Mock())
            comic.batch_report_ctrl = ctrl
            comic._release_batch_model_caches = ComicTranslate._release_batch_model_caches.__get__(comic)
            comic._finalize_batch_report = ComicTranslate._finalize_batch_report.__get__(comic)
            comic._find_latest_batch_output_root = mock.Mock(return_value="output")
            comic._last_batch_run_type = "batch"
            comic._last_batch_failure_detail = ""
            comic._last_runtime_preview_path = ""
            comic.set_runtime_editing_locked = mock.Mock()
            comic._set_project_navigation_enabled = mock.Mock()
            comic.set_pipeline_overlay_active = mock.Mock()
            comic.progress_bar = SimpleNamespace(setVisible=mock.Mock())
            comic.translate_button = _FakeButton()
            comic.cancel_button = _FakeButton()
            comic.save_as_project_button = _FakeButton()
            comic.webtoon_toggle = _FakeButton()
            comic.pipeline_status_panel = SimpleNamespace(
                set_series_queue_pause_visible=mock.Mock(),
                set_output_root=mock.Mock(),
                update_event=mock.Mock(),
                show=mock.Mock(),
                raise_=mock.Mock(),
            )
            comic.s_combo = SimpleNamespace(currentText=lambda: "Japanese")
            comic.t_combo = SimpleNamespace(currentText=lambda: "Korean")
            comic._automatic_progress_tracker = SimpleNamespace(record_batch_completion=mock.Mock())
            comic._cleanup_intermediate_preview_run = mock.Mock()
            comic.settings_page = SimpleNamespace(
                get_export_settings=lambda: {"auto_export_source_txt": True}
            )
            comic._write_txt_md_exchange = mock.Mock()
            comic.batch_report_ctrl.refresh_action_buttons = mock.Mock()
            comic.series_ctrl = SimpleNamespace(on_batch_process_finished=mock.Mock())

            with mock.patch("controller.finish_debug_artifact_run") as finish_debug:
                ComicTranslate._finish_batch_process_ui(
                    comic,
                    was_cancelled=False,
                    failed=False,
                    total_images=1,
                    completed_batch_paths=[path],
                    processing_status="completed",
                )
            record = json.loads(next(Path(tmp).glob("*.json")).read_text(encoding="utf-8"))

        self.assertEqual(record["terminal_status"], "failed")
        self.assertEqual(record["quality"]["status"], "failed")
        self.assertEqual(record["auto_export_status"], "not_attempted")
        self.assertEqual(record["auto_export_reason_code"], "page_quality_not_completed")
        comic._write_txt_md_exchange.assert_not_called()
        self.assertEqual(
            ctrl._latest_batch_report["run_page_outcomes"],
            [{"image_path": path, "status": "failed"}],
        )
        self.assertEqual(comic.pipeline_status_panel.update_event.call_args.args[0]["step_key"], "batch_failed")
        finish_debug.assert_called_once_with(comic, status="failed")
        comic.series_ctrl.on_batch_process_finished.assert_called_once_with(
            was_cancelled=False,
            failed=True,
            was_paused=False,
            retry_allowed=True,
        )

    def test_application_terminal_wall_time_includes_cleanup_and_archive(self) -> None:
        from modules.utils.run_report import build_run_report
        from unittest import mock

        main = _FakeMain()
        path = main.image_files[0]
        clock = mock.Mock(side_effect=[100.0, 110.0, 120.0])

        def write_start_record(**_kwargs: object) -> str:
            clock()
            return "start-record.json"

        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "app.controllers.batch_report.time.monotonic", clock
        ), mock.patch(
            "app.controllers.batch_report.start_run_report",
            side_effect=write_start_record,
        ):
            ctrl = BatchReportController(main)
            ctrl.start_batch_report([path], run_report_log_dir=tmp)
            ctrl.register_batch_success(path)
            ctrl.register_pipeline_run_report(
                build_run_report(
                    telemetry={"status": "completed"},
                    total_wall_sec=10.0,
                    page_outcomes=[{
                        "source_unit_index": 0,
                        "image_name": os.path.basename(path),
                        "output_path": "output/page.png",
                        "quality_status": "completed",
                    }],
                    output_summary={
                        "input_count": 1,
                        "output_count": 1,
                        "fallback_count": 0,
                        "fallbacks": [],
                        "missing": [],
                    },
                    processing_status="completed",
                    terminal_status="completed",
                    report_status="completed",
                    cleanup_status="completed",
                )
            )

            ctrl.finalize_batch_report(
                False,
                processing_status="completed",
                application_cleanup_status="completed",
                archive_status="completed",
            )
            record = json.loads(next(Path(tmp).glob("*.json")).read_text(encoding="utf-8"))

        self.assertEqual(record["pipeline_wall_sec"], 10.0)
        self.assertEqual(record["total_wall_sec"], 20.0)
        self.assertEqual(record["total_wall_text"], "00:00:20")
        self.assertEqual(record["seconds_per_page"], 20.0)

    def test_worker_return_does_not_infer_completed_pages_or_cleanup(self) -> None:
        from controller import ComicTranslate
        from unittest import mock

        main = _FakeMain()
        ctrl = BatchReportController(main)
        ctrl.start_batch_report(main.image_files)
        comic = ComicTranslate.__new__(ComicTranslate)
        comic._memlogger = None
        comic._batch_pause_requested = False
        comic._batch_cancel_requested = False
        comic._batch_failed = False
        comic._is_shutting_down = False
        comic.selected_batch = main.image_files
        comic.batch_report_ctrl = ctrl
        comic.get_resolved_export_settings = lambda: {}
        comic._finish_batch_process_ui = mock.Mock(
            side_effect=lambda **kwargs: ComicTranslate._finalize_batch_report(
                comic,
                kwargs["was_cancelled"],
                processing_status=kwargs["processing_status"],
                application_cleanup_status="completed",
                archive_status=kwargs["archive_status"],
            )
        )

        ComicTranslate.on_batch_process_finished(comic)

        report = ctrl._latest_batch_report
        self.assertEqual(report["processing_status"], "completed")
        self.assertEqual(report["cleanup_status"], "not_observed")
        self.assertEqual(report["completed_count"], 0)
        self.assertEqual(report["unjudged_count"], 1)
        self.assertEqual(report["terminal_status"], "incomplete")

    def test_shutdown_interruption_overrides_worker_cancellation(self) -> None:
        from modules.utils.run_report import build_run_report

        main = _FakeMain()
        ctrl = BatchReportController(main)
        ctrl.start_batch_report(main.image_files)
        ctrl.register_pipeline_run_report(
            build_run_report(
                telemetry={"status": "cancelled"},
                total_wall_sec=1.0,
                page_outcomes=[{"quality_status": "unjudged"}],
                output_summary={"input_count": 1, "output_count": None},
                processing_status="cancelled",
                terminal_status="cancelled",
                report_status="completed",
                cleanup_status="completed",
            )
        )

        report = ctrl.finalize_batch_report(
            False,
            processing_status="interrupted",
            application_cleanup_status="not_observed",
        )

        self.assertEqual(report["processing_status"], "interrupted")
        self.assertEqual(report["terminal_status"], "interrupted")

    def test_model_cache_release_surfaces_internal_cleanup_failure(self) -> None:
        from unittest import mock

        from modules.detection.factory import DetectionEngineFactory
        from modules.ocr.factory import OCRFactory
        from modules.translation.factory import TranslationFactory
        from pipeline.main_pipeline import ComicTranslatePipeline

        class _FailingCacheOwner:
            @property
            def block_detector_cache(self):
                return object()

            @block_detector_cache.setter
            def block_detector_cache(self, _value) -> None:
                raise RuntimeError("detector cache cleanup failed")

        pipeline = ComicTranslatePipeline.__new__(ComicTranslatePipeline)
        pipeline.block_detection = _FailingCacheOwner()
        pipeline.inpainting = type(
            "InpaintingCacheOwner",
            (),
            {"inpainter_cache": object(), "cached_inpainter_key": "cache-key"},
        )()
        detector_engines = {"synthetic": object()}
        ocr_engines = {"synthetic": object()}
        translation_engines = {"synthetic": object()}

        with (
            mock.patch.object(DetectionEngineFactory, "_engines", detector_engines),
            mock.patch.object(OCRFactory, "_engines", ocr_engines),
            mock.patch.object(TranslationFactory, "_engines", translation_engines),
            self.assertRaisesRegex(RuntimeError, "detector cache cleanup failed"),
        ):
            pipeline.release_model_caches(raise_on_failure=True)

        self.assertEqual(detector_engines, {})
        self.assertEqual(ocr_engines, {})
        self.assertEqual(translation_engines, {})

    def test_application_cleanup_failure_prevents_terminal_success(self) -> None:
        from controller import ComicTranslate
        from modules.utils.run_report import build_run_report
        from types import SimpleNamespace

        main = _FakeMain()
        source_path = "example_source_chapter/page-001.png"
        main.image_files = [source_path]
        main.image_states = {source_path: {"skip": False}}
        report_ctrl = BatchReportController(main)
        with tempfile.TemporaryDirectory() as tmp:
            report_ctrl.start_batch_report([source_path], run_report_log_dir=tmp)
            report_ctrl.register_batch_success(source_path)
            report_ctrl.register_pipeline_run_report(
                build_run_report(
                    telemetry={"status": "completed"},
                    total_wall_sec=1.0,
                    page_outcomes=[
                        {"image_name": "page-001.png", "output_path": "output/page-001.png", "quality_status": "completed"}
                    ],
                    output_summary={
                        "input_count": 1,
                        "output_count": 1,
                        "fallback_count": 0,
                        "fallbacks": [],
                        "missing": [],
                    },
                    terminal_status="completed",
                    processing_status="completed",
                    report_status="completed",
                    cleanup_status="completed",
                )
            )
            comic = ComicTranslate.__new__(ComicTranslate)
            cache_release_calls: list[bool] = []

            def release_model_caches(*, raise_on_failure: bool = False) -> None:
                cache_release_calls.append(raise_on_failure)
                raise RuntimeError("cache release failed")

            comic.pipeline = SimpleNamespace(release_model_caches=release_model_caches)
            comic._memlogger = None
            comic.batch_report_ctrl = report_ctrl

            cleanup_status, cleanup_errors = ComicTranslate._release_batch_model_caches(comic)
            finalized = ComicTranslate._finalize_batch_report(
                comic,
                False,
                processing_status="completed",
                application_cleanup_status=cleanup_status,
                application_cleanup_errors=cleanup_errors,
                archive_status="not_applicable",
            )

            terminal_record = json.loads(next(Path(tmp).glob("*.json")).read_text(encoding="utf-8"))

        self.assertEqual(cleanup_status, "failed")
        self.assertEqual(cache_release_calls, [True])
        self.assertEqual(finalized["processing_status"], "completed")
        self.assertEqual(finalized["cleanup_status"], "failed")
        self.assertEqual(finalized["terminal_status"], "cleanup_failed")
        self.assertEqual(terminal_record["terminal_status"], "cleanup_failed")

    def test_application_cleanup_failure_updates_panel_and_fails_series_item(self) -> None:
        from controller import ComicTranslate
        from types import SimpleNamespace
        from unittest import mock

        main = _FakeMain()
        path = main.image_files[0]
        ctrl = BatchReportController(main)
        ctrl.start_batch_report([path])
        comic = ComicTranslate.__new__(ComicTranslate)
        comic._memlogger = None
        cache_release_calls: list[bool] = []

        def release_model_caches(*, raise_on_failure: bool = False) -> None:
            cache_release_calls.append(raise_on_failure)
            raise RuntimeError("cache release failed")

        comic.pipeline = SimpleNamespace(release_model_caches=release_model_caches)
        comic.batch_report_ctrl = ctrl
        comic.settings_page = SimpleNamespace(get_export_settings=lambda: {})
        comic._release_batch_model_caches = ComicTranslate._release_batch_model_caches.__get__(comic)
        comic._finalize_batch_report = ComicTranslate._finalize_batch_report.__get__(comic)
        comic._find_latest_batch_output_root = mock.Mock(return_value="")
        comic._last_batch_run_type = "batch"
        comic._last_batch_failure_detail = ""
        comic._last_runtime_preview_path = ""
        comic.set_runtime_editing_locked = mock.Mock()
        comic._set_project_navigation_enabled = mock.Mock()
        comic.set_pipeline_overlay_active = mock.Mock()
        comic.progress_bar = SimpleNamespace(setVisible=mock.Mock())
        comic.translate_button = _FakeButton()
        comic.cancel_button = _FakeButton()
        comic.save_as_project_button = _FakeButton()
        comic.webtoon_toggle = _FakeButton()
        comic.pipeline_status_panel = SimpleNamespace(
            set_series_queue_pause_visible=mock.Mock(),
            set_output_root=mock.Mock(),
            update_event=mock.Mock(),
            show=mock.Mock(),
            raise_=mock.Mock(),
        )
        comic._automatic_progress_tracker = SimpleNamespace(record_batch_completion=mock.Mock())
        comic._cleanup_intermediate_preview_run = mock.Mock()
        comic.batch_report_ctrl.refresh_action_buttons = mock.Mock()
        comic.series_ctrl = SimpleNamespace(on_batch_process_finished=mock.Mock())

        with mock.patch("controller.finish_debug_artifact_run") as finish_debug:
            ComicTranslate._finish_batch_process_ui(
                comic,
                was_cancelled=False,
                failed=False,
                total_images=1,
                completed_batch_paths=[path],
                processing_status="completed",
                archive_status="not_applicable",
            )

        event = comic.pipeline_status_panel.update_event.call_args.args[0]
        self.assertEqual(event["step_key"], "batch_cleanup")
        self.assertIn("release_model_caches", event["detail"])
        self.assertEqual(cache_release_calls, [True])
        finish_debug.assert_called_once_with(comic, status="failed")
        comic.series_ctrl.on_batch_process_finished.assert_called_once_with(
            was_cancelled=False,
            failed=True,
            was_paused=False,
            retry_allowed=False,
        )

    def test_report_failure_is_shown_without_blocking_cache_cleanup_or_export(self) -> None:
        from controller import ComicTranslate
        from modules.utils.run_report import build_run_report
        from types import SimpleNamespace
        from unittest import mock

        main = _FakeMain()
        source_path = "example_source_chapter/page-001.png"
        main.image_files = [source_path]
        main.image_states = {source_path: {"skip": False}}
        report_ctrl = BatchReportController(main)
        events: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            report_ctrl.start_batch_report(
                [source_path],
                run_report_log_dir=tmp,
            )
            start_record_path = next(Path(tmp).glob("*.json"))
            report_ctrl.register_batch_success(source_path)
            report_ctrl.register_pipeline_run_report(
                build_run_report(
                    telemetry={"status": "completed"},
                    total_wall_sec=1.0,
                    page_outcomes=[
                        {
                            "source_unit_index": 0,
                            "image_name": "page-001.png",
                            "output_path": "output/page-001.png",
                            "quality_status": "completed",
                        }
                    ],
                    output_summary={
                        "input_count": 1,
                        "output_count": 1,
                        "fallback_count": 0,
                        "fallbacks": [],
                        "missing": [],
                    },
                    terminal_status="completed",
                    processing_status="completed",
                    report_status="completed",
                    cleanup_status="completed",
                )
            )
            comic = ComicTranslate.__new__(ComicTranslate)
            comic.pipeline = SimpleNamespace(
                release_model_caches=lambda *, raise_on_failure=False: events.append("cleanup")
            )
            comic._memlogger = None
            comic.batch_report_ctrl = report_ctrl
            comic.settings_page = SimpleNamespace(get_export_settings=lambda: {})
            comic.selected_batch = [source_path]
            comic._last_batch_run_type = "batch"
            comic._last_batch_failure_detail = ""
            comic._automatic_progress_tracker = SimpleNamespace(
                record_batch_completion=mock.Mock()
            )
            comic.pipeline_status_panel = SimpleNamespace(
                set_series_queue_pause_visible=mock.Mock(),
                set_output_root=mock.Mock(),
                update_event=mock.Mock(),
                show=mock.Mock(),
                raise_=mock.Mock(),
            )
            comic.set_runtime_editing_locked = mock.Mock()
            comic._set_project_navigation_enabled = mock.Mock()
            comic.set_pipeline_overlay_active = mock.Mock()
            comic._find_latest_batch_output_root = mock.Mock(return_value="output")
            comic._cleanup_intermediate_preview_run = mock.Mock()
            def export_before_report(
                _paths: list[str], *, attempt: bool, reason_code: str | None
            ) -> dict[str, object]:
                self.assertTrue(attempt)
                self.assertIsNone(reason_code)
                events.append("export")
                return {
                    "status": "completed",
                    "selected_targets": ["source.txt"],
                    "completed_targets": ["source.txt"],
                }

            comic._run_txt_md_auto_exports = mock.Mock(side_effect=export_before_report)
            comic._finish_batch_process_ui = ComicTranslate._finish_batch_process_ui.__get__(comic)
            comic.progress_bar = SimpleNamespace(setVisible=mock.Mock())
            comic.translate_button = _FakeButton()
            comic.cancel_button = _FakeButton()
            comic.save_as_project_button = _FakeButton()
            comic.webtoon_toggle = _FakeButton()
            comic.series_ctrl = SimpleNamespace(on_batch_process_finished=mock.Mock())
            comic.is_current_task_cancelled = lambda: False

            def report_write_failed(*_args, **_kwargs):
                events.append("report")
                return ""

            with (
                mock.patch("app.controllers.batch_report.write_run_report", side_effect=report_write_failed),
                mock.patch("controller.Messages.show_translation_complete") as show_complete,
            ):
                ComicTranslate._finish_batch_process_ui(
                    comic,
                    was_cancelled=False,
                    failed=False,
                    total_images=1,
                    completed_batch_paths=[source_path],
                    processing_status="completed",
                    archive_status="not_applicable",
                )
            self.assertEqual(
                json.loads(start_record_path.read_text(encoding="utf-8"))["terminal_status"],
                "running",
            )

        finalized = report_ctrl._latest_batch_report
        self.assertEqual(events, ["cleanup", "export", "report"])
        self.assertEqual(finalized["terminal_status"], "report_failed")
        self.assertEqual(finalized["report_status"], "failed")
        self.assertEqual(finalized["auto_export_status"], "completed")
        self.assertIn(
            "Outputs are preserved.",
            comic.pipeline_status_panel.update_event.call_args.args[0]["message"],
        )
        comic._automatic_progress_tracker.record_batch_completion.assert_not_called()
        show_complete.assert_not_called()
        comic.series_ctrl.on_batch_process_finished.assert_called_once_with(
            was_cancelled=False,
            failed=True,
            was_paused=False,
            retry_allowed=False,
        )

    def test_incomplete_run_is_not_reported_done_or_retried_by_series_queue(self) -> None:
        from controller import ComicTranslate
        from types import SimpleNamespace
        from unittest import mock

        panel = SimpleNamespace(
            set_series_queue_pause_visible=mock.Mock(),
            set_output_root=mock.Mock(),
            update_event=mock.Mock(),
            show=mock.Mock(),
            raise_=mock.Mock(),
        )
        comic = SimpleNamespace(
            _release_batch_model_caches=mock.Mock(return_value=("completed", [])),
            _finalize_batch_report=mock.Mock(return_value={
                "terminal_status": "incomplete",
                "skipped_count": 0,
            }),
            _find_latest_batch_output_root=mock.Mock(return_value="output"),
            _last_batch_run_type="batch",
            _last_batch_failure_detail="",
            _last_runtime_preview_path="",
            set_runtime_editing_locked=mock.Mock(),
            _set_project_navigation_enabled=mock.Mock(),
            set_pipeline_overlay_active=mock.Mock(),
            progress_bar=SimpleNamespace(setVisible=mock.Mock()),
            translate_button=_FakeButton(),
            cancel_button=_FakeButton(),
            save_as_project_button=_FakeButton(),
            webtoon_toggle=_FakeButton(),
            selected_batch=[],
            pipeline_status_panel=panel,
            _automatic_progress_tracker=SimpleNamespace(record_batch_completion=mock.Mock()),
            on_runtime_progress_update=mock.Mock(),
            _play_completion_sound_if_enabled=mock.Mock(),
            _cleanup_intermediate_preview_run=mock.Mock(),
            _run_txt_md_auto_exports=mock.Mock(),
            batch_report_ctrl=SimpleNamespace(
                auto_export_ineligibility_reason=mock.Mock(
                    return_value="pipeline_outcome_not_observed"
                ),
                refresh_action_buttons=mock.Mock(),
            ),
            series_ctrl=SimpleNamespace(on_batch_process_finished=mock.Mock()),
        )

        with mock.patch("controller.finish_debug_artifact_run") as finish_debug:
            ComicTranslate._finish_batch_process_ui(
                comic,
                was_cancelled=False,
                failed=False,
                total_images=1,
                completed_batch_paths=["page.png"],
                processing_status="completed",
            )

        self.assertEqual(panel.update_event.call_args.args[0]["step_key"], "incomplete_report")
        finish_debug.assert_called_once_with(comic, status="failed")
        comic.series_ctrl.on_batch_process_finished.assert_called_once_with(
            was_cancelled=False,
            failed=True,
            was_paused=False,
            retry_allowed=False,
        )
        comic._automatic_progress_tracker.record_batch_completion.assert_not_called()

    def test_archive_failure_is_reported_before_terminal_success_can_be_written(self) -> None:
        from controller import ComicTranslate
        from modules.utils.run_report import build_run_report
        from unittest import mock

        main = _FakeMain()
        source_path = "example_source_chapter/page-001.png"
        main.image_files = [source_path]
        main.image_states = {source_path: {"skip": False}}
        report_ctrl = BatchReportController(main)
        with tempfile.TemporaryDirectory() as tmp:
            report_ctrl.start_batch_report([source_path], run_report_log_dir=tmp)
            start_record_path = next(Path(tmp).glob("*.json"))
            report_ctrl.register_batch_success(source_path)
            report_ctrl.register_pipeline_run_report(
                build_run_report(
                    telemetry={"status": "completed"},
                    total_wall_sec=1.0,
                    page_outcomes=[
                        {
                            "source_unit_index": 0,
                            "image_name": "page-001.png",
                            "output_path": "output/page-001.png",
                            "quality_status": "completed",
                        }
                    ],
                    output_summary={
                        "input_count": 1,
                        "output_count": 1,
                        "fallback_count": 0,
                        "fallbacks": [],
                        "missing": [],
                    },
                    terminal_status="completed",
                    processing_status="completed",
                    report_status="completed",
                    cleanup_status="completed",
                )
            )
            comic = ComicTranslate.__new__(ComicTranslate)
            comic.batch_report_ctrl = report_ctrl
            comic.cancel_button = _FakeButton()
            comic._emit_archive_finalization_progress = mock.Mock()
            comic.on_runtime_progress_update = mock.Mock()
            comic._last_batch_failure_detail = ""

            def finish(
                *,
                was_cancelled: bool,
                failed: bool,
                total_images: int,
                completed_batch_paths: list[str],
                archive_path: str = "",
                processing_status: str,
                archive_status: str,
                processing_error: dict[str, object] | None = None,
            ):
                return ComicTranslate._finalize_batch_report(
                    comic,
                    was_cancelled,
                    processing_status=processing_status,
                    processing_error=processing_error,
                    application_cleanup_status="completed",
                    archive_status=archive_status,
                )

            comic._finish_batch_process_ui = mock.Mock(side_effect=finish)
            archive_states = []

            def run_immediately(worker, on_result, on_error, on_finished):
                try:
                    result = worker()
                except Exception as exc:
                    archive_states.append(
                        json.loads(start_record_path.read_text(encoding="utf-8"))["terminal_status"]
                    )
                    on_error((type(exc), exc, "synthetic archive failure"))
                else:
                    on_result(result)
                on_finished()

            comic.run_threaded_immediate = run_immediately
            with mock.patch.object(
                ComicTranslate,
                "_build_single_archive_output",
                side_effect=RuntimeError("synthetic archive failure"),
            ):
                ComicTranslate._start_batch_archive_finalization(
                    comic,
                    completed_batch_paths=[source_path],
                    total_images=1,
                    processing_status="completed",
                )
            terminal_record = json.loads(start_record_path.read_text(encoding="utf-8"))

        self.assertEqual(archive_states, ["running"])
        self.assertEqual(terminal_record["processing_status"], "completed")
        self.assertEqual(terminal_record["archive_status"], "failed")
        self.assertEqual(terminal_record["terminal_status"], "failed")
        self.assertEqual(
            terminal_record["processing_error"]["error_type"], "RuntimeError"
        )
        self.assertEqual(
            terminal_record["processing_error"]["phase"], "archive_finalization"
        )
        self.assertIn(
            "synthetic archive failure",
            terminal_record["processing_error"]["message"],
        )
        self.assertEqual(report_ctrl._latest_batch_report["terminal_status"], "failed")
        restored_main = _FakeMain()
        restored_main.image_files = [source_path]
        restored_main.image_states = {source_path: {"skip": False}}
        restored_report = BatchReportController(restored_main)
        restored_report.import_latest_report_from_project(
            report_ctrl.export_latest_report_for_project()
        )
        self.assertEqual(
            restored_report.export_latest_report_for_project()["processing_error"]["phase"],
            "archive_finalization",
        )

    def test_archive_apply_failure_preserves_exception_type_and_phase(self) -> None:
        from controller import ComicTranslate
        from modules.utils.run_report import build_run_report
        from unittest import mock

        main = _FakeMain()
        source_path = "example_source_chapter/page-001.png"
        main.image_files = [source_path]
        main.image_states = {source_path: {"skip": False}}
        report_ctrl = BatchReportController(main)
        with tempfile.TemporaryDirectory() as tmp:
            report_ctrl.start_batch_report([source_path], run_report_log_dir=tmp)
            report_ctrl.register_batch_success(source_path)
            report_ctrl.register_pipeline_run_report(
                build_run_report(
                    telemetry={"status": "completed"},
                    total_wall_sec=1.0,
                    page_outcomes=[
                        {
                            "source_unit_index": 0,
                            "image_path": source_path,
                            "image_name": "page-001.png",
                            "output_path": "output/page-001.png",
                            "quality_status": "completed",
                        }
                    ],
                    output_summary={
                        "input_count": 1,
                        "output_count": 1,
                        "fallback_count": 0,
                        "fallbacks": [],
                        "missing": [],
                    },
                    terminal_status="completed",
                    processing_status="completed",
                    report_status="completed",
                    cleanup_status="completed",
                )
            )
            comic = ComicTranslate.__new__(ComicTranslate)
            comic.batch_report_ctrl = report_ctrl
            comic.cancel_button = _FakeButton()
            comic._emit_archive_finalization_progress = mock.Mock()
            comic.on_runtime_progress_update = mock.Mock()

            def finish(
                *,
                was_cancelled: bool,
                failed: bool,
                total_images: int,
                completed_batch_paths: list[str],
                archive_path: str = "",
                processing_status: str,
                archive_status: str,
                processing_error: dict[str, object] | None = None,
            ):
                return ComicTranslate._finalize_batch_report(
                    comic,
                    was_cancelled,
                    processing_status=processing_status,
                    processing_error=processing_error,
                    application_cleanup_status="completed",
                    archive_status=archive_status,
                )

            comic._finish_batch_process_ui = mock.Mock(side_effect=finish)

            def run_immediately(worker, on_result, on_error, on_finished) -> None:
                try:
                    result = worker()
                except Exception as exc:
                    on_error((type(exc), exc, "synthetic archive apply failure"))
                else:
                    on_result(result)
                on_finished()

            comic.run_threaded_immediate = run_immediately
            with mock.patch.object(
                ComicTranslate,
                "_build_single_archive_output",
                return_value={"archive_path": "example-output.cbz"},
            ), mock.patch.object(
                ComicTranslate,
                "_apply_single_archive_output_result",
                side_effect=ValueError("synthetic archive apply failure"),
            ):
                ComicTranslate._start_batch_archive_finalization(
                    comic,
                    completed_batch_paths=[source_path],
                    total_images=1,
                    processing_status="completed",
                )

            record = json.loads(next(Path(tmp).glob("*.json")).read_text(encoding="utf-8"))

        self.assertEqual(record["processing_status"], "completed")
        self.assertEqual(record["archive_status"], "failed")
        self.assertEqual(record["terminal_status"], "failed")
        self.assertEqual(record["processing_error"]["error_type"], "ValueError")
        self.assertEqual(record["processing_error"]["phase"], "archive_finalization")
        self.assertIn("synthetic archive apply failure", record["processing_error"]["message"])

    def test_legacy_translation_skip_reason_message_is_not_generic(self) -> None:
        ctrl = object.__new__(ImageStateController)

        message = ctrl._build_skip_message(
            "/tmp/page-001.png",
            "Translation",
            "translator timed out",
        )

        self.assertIn("Could not translate webtoon chunk", message)
        self.assertNotIn("Page processing failed", message)


if __name__ == "__main__":
    unittest.main()
