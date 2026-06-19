from __future__ import annotations

import os
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image as PILImage

from controller import ComicTranslate
from modules.utils.automatic_output import OUTPUT_TARGET_ARCHIVE, OUTPUT_TARGET_IMAGES


class _Toggle:
    def __init__(self, checked: bool = False) -> None:
        self._checked = checked
        self.set_calls: list[bool] = []

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, value: bool) -> None:
        self._checked = value
        self.set_calls.append(value)


class _WidgetState:
    def __init__(self) -> None:
        self.enabled_values: list[bool] = []
        self.visible_values: list[bool] = []

    def setEnabled(self, value: bool) -> None:
        self.enabled_values.append(value)

    def setVisible(self, value: bool) -> None:
        self.visible_values.append(value)


class _ImageCtrl:
    def __init__(self, page_states: dict[str, dict]) -> None:
        self._page_states = page_states
        self.cleared_paths: list[str] | None = None
        self.summary_updates: list[tuple[str, dict]] = []

    def ensure_page_state(self, path: str) -> dict[str, str]:
        return self._page_states[path]

    def clear_page_skip_errors_for_paths(self, paths: list[str]) -> None:
        self.cleared_paths = list(paths)

    def update_processing_summary(self, path: str, patch: dict) -> dict:
        state = self._page_states.setdefault(path, {})
        summary = state.setdefault("processing_summary", {})
        summary.update(patch)
        self.summary_updates.append((path, dict(patch)))
        return summary


class _BatchReportCtrl:
    def __init__(self) -> None:
        self.refresh_calls = 0

    def refresh_action_buttons(self) -> None:
        self.refresh_calls += 1


class _DummyController(SimpleNamespace):
    pass


class ControllerBatchPreflightTests(unittest.TestCase):
    @staticmethod
    def _write_png(path: str, value: int = 0) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image = np.full((4, 4, 3), value, dtype=np.uint8)
        PILImage.fromarray(image).save(path, format="PNG")

    def _build_controller(self) -> _DummyController:
        page_states = {
            "page-a.png": {"target_lang": "English", "source_lang": "Japanese"},
            "page-b.png": {"target_lang": "English", "source_lang": "Japanese"},
        }
        controller = _DummyController()
        controller.image_files = list(page_states.keys())
        controller.image_ctrl = _ImageCtrl(page_states)
        controller._start_batch_report = mock.Mock()
        controller._show_automatic_progress_dialog = mock.Mock()
        controller._set_project_navigation_enabled = mock.Mock()
        controller.set_runtime_editing_locked = mock.Mock()
        controller.batch_mode_selected = mock.Mock()
        controller.run_threaded = mock.Mock()
        controller.default_error_handler = mock.Mock()
        controller.on_batch_process_finished = mock.Mock()
        controller.pipeline = SimpleNamespace(batch_process=mock.Mock(), webtoon_batch_process=mock.Mock())
        controller.pipeline_status_panel = SimpleNamespace(set_series_queue_pause_visible=mock.Mock())
        controller.manual_radio = _Toggle(False)
        controller.automatic_radio = _Toggle(False)
        controller.translate_button = _WidgetState()
        controller.cancel_button = _WidgetState()
        controller.save_as_project_button = _WidgetState()
        controller.webtoon_toggle = _WidgetState()
        controller.progress_bar = _WidgetState()
        controller.batch_report_ctrl = _BatchReportCtrl()
        controller.webtoon_mode = False
        return controller

    def test_start_batch_process_reuses_one_preflight_cache_for_multiple_pages(self) -> None:
        controller = self._build_controller()
        captured_caches: list[dict[str, str]] = []

        def _validate(_main, _target_lang, *, source_lang=None, preflight_cache=None):
            self.assertEqual(source_lang, "Japanese")
            self.assertIsNotNone(preflight_cache)
            captured_caches.append(preflight_cache)
            return True

        with mock.patch("controller.validate_settings", side_effect=_validate):
            result = ComicTranslate._start_batch_process_for_paths(
                controller,
                ["page-a.png", "page-b.png"],
                run_type="batch",
            )

        self.assertTrue(result)
        self.assertEqual(len(captured_caches), 2)
        self.assertIs(captured_caches[0], captured_caches[1])

    def test_one_page_auto_process_uses_batch_entrypoint(self) -> None:
        controller = self._build_controller()
        controller._batch_active = False
        controller.curr_img_idx = 0
        controller._confirm_and_apply_auto_languages = mock.Mock(return_value=True)
        controller._start_batch_process_for_paths = mock.Mock(return_value=True)

        ComicTranslate.start_one_page_auto_process(controller)

        controller._confirm_and_apply_auto_languages.assert_called_once_with(["page-a.png"], "one_page_auto")
        controller._start_batch_process_for_paths.assert_called_once_with(["page-a.png"], run_type="one_page_auto")

    def _build_batch_finished_controller(self, output_target: str) -> _DummyController:
        controller = _DummyController()
        controller._memlogger = None
        controller._batch_cancel_requested = False
        controller._batch_pause_requested = False
        controller._batch_failed = False
        controller.selected_batch = ["page-a.png", "page-b.png"]
        controller._batch_active = True
        controller._current_batch_run_type = "series_queue"
        controller._is_shutting_down = False
        controller._finalize_batch_report = mock.Mock(return_value={"skipped_count": 0})
        controller.get_resolved_export_settings = mock.Mock(
            return_value={"resolved_automatic_output_target": output_target}
        )
        controller._start_batch_archive_finalization = mock.Mock()
        controller._finish_batch_process_ui = mock.Mock()
        return controller

    def test_batch_finished_skips_archive_finalization_for_individual_images(self) -> None:
        controller = self._build_batch_finished_controller(OUTPUT_TARGET_IMAGES)

        ComicTranslate.on_batch_process_finished(controller)

        controller._start_batch_archive_finalization.assert_not_called()
        controller._finish_batch_process_ui.assert_called_once_with(
            was_cancelled=False,
            failed=False,
            total_images=2,
            completed_batch_paths=["page-a.png", "page-b.png"],
            report={"skipped_count": 0},
            was_paused=False,
        )
        self.assertFalse(controller._batch_active)
        self.assertFalse(controller._batch_cancel_requested)
        self.assertFalse(controller._batch_failed)
        self.assertIsNone(controller._current_batch_run_type)

    def test_batch_finished_finalizes_archive_only_for_single_archive_target(self) -> None:
        controller = self._build_batch_finished_controller(OUTPUT_TARGET_ARCHIVE)

        ComicTranslate.on_batch_process_finished(controller)

        controller._finish_batch_process_ui.assert_not_called()
        controller._start_batch_archive_finalization.assert_called_once_with(
            completed_batch_paths=["page-a.png", "page-b.png"],
            total_images=2,
            report={"skipped_count": 0},
        )

    def test_paused_batch_skips_archive_finalization(self) -> None:
        controller = self._build_batch_finished_controller(OUTPUT_TARGET_ARCHIVE)
        controller._batch_pause_requested = True

        ComicTranslate.on_batch_process_finished(controller)

        controller._start_batch_archive_finalization.assert_not_called()
        controller._finish_batch_process_ui.assert_called_once_with(
            was_cancelled=False,
            failed=False,
            total_images=2,
            completed_batch_paths=["page-a.png", "page-b.png"],
            report={"skipped_count": 0},
            was_paused=True,
        )
        self.assertFalse(controller._batch_active)
        self.assertFalse(controller._batch_cancel_requested)
        self.assertFalse(controller._batch_pause_requested)

    def test_finalize_single_archive_output_builds_cbz_from_staging_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            series_dir = os.path.join(temp_dir, "series")
            staging_dir = os.path.join(series_dir, ".archive_staging_test")
            os.makedirs(staging_dir)
            staged_a = os.path.join(staging_dir, "001.png")
            staged_b = os.path.join(staging_dir, "002.png")
            self._write_png(staged_a, 16)
            self._write_png(staged_b, 32)

            page_a = os.path.join(temp_dir, "source-a.png")
            page_b = os.path.join(temp_dir, "source-b.png")
            page_states = {
                page_a: {
                    "processing_summary": {
                        "translated_image_path": staged_a,
                        "export_root": series_dir,
                    }
                },
                page_b: {
                    "processing_summary": {
                        "translated_image_path": staged_b,
                        "export_root": series_dir,
                    }
                },
            }
            controller = _DummyController()
            controller.image_states = page_states
            controller.image_ctrl = _ImageCtrl(page_states)
            controller._txt_md_bundle_name = lambda: "demo"
            controller.temp_dir = temp_dir
            controller._last_runtime_preview_path = ""
            controller.file_handler = SimpleNamespace(archive_info=[])
            controller.export_source_by_path = {}
            controller.project_file = None
            controller.get_resolved_export_settings = lambda: {
                "resolved_automatic_output_target": OUTPUT_TARGET_ARCHIVE,
                "resolved_automatic_output_archive_format": "cbz",
                "resolved_automatic_output_archive_image_format": "png",
                "resolved_automatic_output_archive_compression_level": 6,
            }

            archive_path = ComicTranslate._finalize_single_archive_output(controller, [page_a, page_b])

            self.assertEqual(archive_path, os.path.join(series_dir, "demo_translated.cbz"))
            self.assertTrue(os.path.isfile(archive_path))
            self.assertFalse(os.path.exists(staging_dir))
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(archive.namelist(), ["001_source-a.png", "002_source-b.png"])
            for page_path in (page_a, page_b):
                summary = page_states[page_path]["processing_summary"]
                self.assertEqual(summary["translated_image_path"], archive_path)
                self.assertEqual(summary["export_root"], series_dir)

    def test_finalize_archive_output_saves_next_to_original_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_archive = os.path.join(temp_dir, "example source chapter v01 c01 (E).cbz")
            with open(source_archive, "wb") as handle:
                handle.write(b"placeholder")
            series_dir = os.path.join(temp_dir, "result")
            staging_dir = os.path.join(series_dir, ".archive_staging_test")
            staged_a = os.path.join(staging_dir, "001.png")
            staged_b = os.path.join(staging_dir, "002.png")
            self._write_png(staged_a, 16)
            self._write_png(staged_b, 32)

            page_a = os.path.join(temp_dir, "extract", "001.png")
            page_b = os.path.join(temp_dir, "extract", "002.png")
            page_states = {
                page_a: {"processing_summary": {"translated_image_path": staged_a, "export_root": series_dir}},
                page_b: {"processing_summary": {"translated_image_path": staged_b, "export_root": series_dir}},
            }
            controller = _DummyController()
            controller.image_states = page_states
            controller.image_ctrl = _ImageCtrl(page_states)
            controller._txt_md_bundle_name = lambda: "fallback"
            controller.temp_dir = temp_dir
            controller._last_runtime_preview_path = ""
            controller.file_handler = SimpleNamespace(archive_info=[])
            controller.export_source_by_path = {
                page_a: {"kind": "archive", "source_path": source_archive},
                page_b: {"kind": "archive", "source_path": source_archive},
            }
            controller.project_file = None
            controller.get_resolved_export_settings = lambda: {
                "resolved_automatic_output_target": OUTPUT_TARGET_ARCHIVE,
                "resolved_automatic_output_archive_format": "cbz",
                "resolved_automatic_output_archive_image_format": "png",
                "resolved_automatic_output_archive_compression_level": 6,
            }

            archive_path = ComicTranslate._finalize_single_archive_output(controller, [page_a, page_b])

            expected = os.path.join(temp_dir, "example source chapter v01 c01 (E)_translated.cbz")
            self.assertEqual(archive_path, expected)
            self.assertTrue(os.path.isfile(expected))
            self.assertFalse(os.path.exists(staging_dir))
            with zipfile.ZipFile(expected) as archive:
                self.assertEqual(archive.namelist(), ["001_001.png", "002_002.png"])
            for page_path in (page_a, page_b):
                summary = page_states[page_path]["processing_summary"]
                self.assertEqual(summary["translated_image_path"], expected)
                self.assertEqual(summary["export_root"], temp_dir)

    def test_finalize_archive_output_includes_original_when_no_rendered_page_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            page = os.path.join(temp_dir, "no-text-page.png")
            self._write_png(page, 48)
            page_states = {page: {"processing_summary": {}}}
            controller = _DummyController()
            controller.image_states = page_states
            controller.image_ctrl = _ImageCtrl(page_states)
            controller._txt_md_bundle_name = lambda: "fallback"
            controller.temp_dir = temp_dir
            controller._last_runtime_preview_path = ""
            controller.file_handler = SimpleNamespace(archive_info=[])
            controller.export_source_by_path = {}
            controller.project_file = None
            controller.get_resolved_export_settings = lambda: {
                "resolved_automatic_output_target": OUTPUT_TARGET_ARCHIVE,
                "resolved_automatic_output_archive_format": "cbz",
                "resolved_automatic_output_archive_image_format": "png",
                "resolved_automatic_output_archive_compression_level": 6,
            }

            archive_path = ComicTranslate._finalize_single_archive_output(controller, [page])

            expected = os.path.join(temp_dir, "no-text-page_translated.cbz")
            self.assertEqual(archive_path, expected)
            self.assertTrue(os.path.isfile(expected))
            with zipfile.ZipFile(expected) as archive:
                self.assertEqual(archive.namelist(), ["001_no-text-page.png"])

    def test_build_archive_output_reports_progress_before_applying_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            page = os.path.join(temp_dir, "page.png")
            self._write_png(page, 64)
            page_states = {page: {"processing_summary": {}}}
            controller = _DummyController()
            controller.image_states = page_states
            controller.image_ctrl = _ImageCtrl(page_states)
            controller._txt_md_bundle_name = lambda: "demo"
            controller.temp_dir = temp_dir
            controller._last_runtime_preview_path = ""
            controller.file_handler = SimpleNamespace(archive_info=[])
            controller.export_source_by_path = {}
            controller.project_file = None
            controller.get_resolved_export_settings = lambda: {
                "resolved_automatic_output_target": OUTPUT_TARGET_ARCHIVE,
                "resolved_automatic_output_archive_format": "cbz",
                "resolved_automatic_output_archive_image_format": "png",
                "resolved_automatic_output_archive_compression_level": 6,
            }
            events: list[dict] = []

            result = ComicTranslate._build_single_archive_output(
                controller,
                [page],
                progress_callback=events.append,
            )

            archive_path = os.path.join(temp_dir, "page_translated.cbz")
            self.assertEqual(result["archive_path"], archive_path)
            self.assertTrue(os.path.isfile(archive_path))
            self.assertEqual(page_states[page]["processing_summary"], {})
            self.assertTrue(any("압축" in str(event.get("message", "")) for event in events))

            ComicTranslate._apply_single_archive_output_result(controller, result)

            summary = page_states[page]["processing_summary"]
            self.assertEqual(summary["translated_image_path"], archive_path)
            self.assertEqual(summary["export_root"], temp_dir)


if __name__ == "__main__":
    unittest.main()
