from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from PySide6 import QtCore

from app.controllers.open_workspace import ProjectLoadSnapshot
from app.controllers.series import (
    SeriesController,
    _PreparedSeriesChild,
    _PreparedSeriesManifest,
)
from app.projects.series_state_v1 import create_series_project, load_series_project


class _VisibleStub:
    def __init__(self) -> None:
        self.visible_values: list[bool] = []

    def setVisible(self, value: bool) -> None:
        self.visible_values.append(bool(value))


class _PipelineStatusStub:
    def __init__(self) -> None:
        self.pause_visible_calls: list[tuple[bool, bool]] = []
        self.events: list[dict] = []

    def set_series_queue_pause_visible(self, visible: bool, *, pause_requested: bool) -> None:
        self.pause_visible_calls.append((bool(visible), bool(pause_requested)))

    def update_event(self, event: dict) -> None:
        self.events.append(dict(event))


class _SeriesWorkspaceStub:
    def configure_options(self, **_kwargs) -> None:
        pass

    def set_global_settings(self, _values) -> None:
        pass

    def set_series_state(self, **_kwargs) -> None:
        pass

    def set_navigation_state(self, **_kwargs) -> None:
        pass


class _MainStub(QtCore.QObject):
    def __init__(self) -> None:
        super().__init__()
        self.loading = _VisibleStub()
        self.pipeline_status_panel = _PipelineStatusStub()
        self.series_workspace = _SeriesWorkspaceStub()
        self.batch_pause_requested = False

    def tr(self, text: str) -> str:
        return text

    def request_current_batch_pause(self) -> bool:
        self.batch_pause_requested = True
        return True


class SeriesControllerBusyTests(unittest.TestCase):
    def _series_file(self, temp_dir: str, *, status: str = "pending") -> str:
        series_path = os.path.join(temp_dir, "example.seriesctpr")
        create_series_project(
            series_path,
            root_dir=temp_dir,
            items=[
                {
                    "series_item_id": "item-1",
                    "queue_index": 1,
                    "display_name": "example_child.ctpr",
                    "source_kind": "ctpr_import",
                    "source_origin_path": os.path.join(temp_dir, "example_child.ctpr"),
                    "source_origin_relpath": "example_child.ctpr",
                    "imported_at": "2026-01-01T00:00:00",
                    "updated_at": "2026-01-01T00:00:00",
                    "status": status,
                    "embedded_project_blob_hash": "hash-1",
                    "child_page_count": None,
                }
            ],
            embedded_projects=[
                {
                    "project_hash": "hash-1",
                    "display_name": "example_child.ctpr",
                    "project_size": 7,
                    "project_blob": b"project",
                }
            ],
        )
        return series_path

    def _controller(self, series_path: str) -> SeriesController:
        controller = SeriesController(_MainStub())
        state = load_series_project(series_path)
        controller.series_file = series_path
        controller.series_manifest = dict(state["manifest"])
        controller.series_items = list(state["items"])
        return controller

    def test_manual_status_change_shows_preparing_modal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = self._controller(self._series_file(temp_dir))
            busy_dialog = object()

            with mock.patch("app.controllers.series.Messages.show_busy", return_value=busy_dialog) as show_busy, \
                mock.patch("app.controllers.series.Messages.close_busy") as close_busy, \
                mock.patch.object(controller, "_apply_workspace_state"):
                controller.request_item_status_change("item-1", "done")

            show_busy.assert_called_once()
            self.assertEqual(show_busy.call_args.args[1], "Updating series item status...")
            self.assertEqual(show_busy.call_args.kwargs["title"], "Series Project")
            self.assertEqual(show_busy.call_args.kwargs["minimum_visible_ms"], 300)
            close_busy.assert_called_once_with(busy_dialog)

    def test_queue_auto_translate_shows_preparing_modal_before_first_item(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = self._controller(self._series_file(temp_dir))
            busy_dialog = object()

            with mock.patch("app.controllers.series.Messages.show_busy", return_value=busy_dialog) as show_busy, \
                mock.patch("app.controllers.series.Messages.close_busy") as close_busy, \
                mock.patch.object(controller, "_update_queue_runtime"), \
                mock.patch.object(controller, "_apply_workspace_state"), \
                mock.patch.object(controller, "_run_next_queue_item") as run_next:
                controller.start_queue_translation()

            show_busy.assert_called_once()
            self.assertEqual(show_busy.call_args.args[1], "Preparing automatic translation...")
            self.assertEqual(show_busy.call_args.kwargs["title"], "Series Project")
            self.assertEqual(show_busy.call_args.kwargs["minimum_visible_ms"], 300)
            run_next.assert_called_once()
            close_busy.assert_called_once_with(busy_dialog)

    def test_pause_queue_translation_requests_current_batch_pause(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = self._controller(self._series_file(temp_dir, status="running"))
            controller._queue_active = True
            controller._queue_pending_ids = []
            controller._queue_completed_ids = []
            controller._queue_failed_ids = []
            controller._queue_skipped_ids = []
            controller._queue_retry_remaining = {}

            with mock.patch.object(controller, "_apply_workspace_state"):
                controller.pause_queue_translation()

            self.assertTrue(controller._pause_requested)
            self.assertTrue(controller.main.batch_pause_requested)
            self.assertEqual(controller.main.pipeline_status_panel.pause_visible_calls[-1], (True, True))

    def test_paused_batch_returns_active_item_to_pending_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = self._controller(self._series_file(temp_dir, status="running"))
            controller._queue_active = True
            controller._pause_requested = True
            controller._queue_pending_ids = []
            controller._queue_completed_ids = []
            controller._queue_failed_ids = []
            controller._queue_skipped_ids = []
            controller._queue_retry_remaining = {"item-1": 1}
            controller.active_child_item_id = "item-1"
            controller.is_child_project_active = mock.Mock(return_value=True)
            controller.sync_active_child_to_series = mock.Mock()
            controller._apply_workspace_state = mock.Mock()
            controller._show_board = mock.Mock()

            controller.on_batch_process_finished(
                was_cancelled=False,
                failed=False,
                was_paused=True,
            )

            state = load_series_project(controller.series_file)
            runtime = state["manifest"]["series_queue_runtime"]
            self.assertEqual(state["items"][0]["status"], "pending")
            self.assertEqual(runtime["queue_state"], "paused")
            self.assertEqual(runtime["pending_item_ids"], ["item-1"])
            self.assertIsNone(runtime["active_item_id"])
            self.assertFalse(runtime["pause_requested"])
            self.assertFalse(controller._queue_active)
            self.assertFalse(controller._pause_requested)

    def test_series_manifest_commit_failure_restores_controller_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = self._controller(self._series_file(temp_dir))
            original_file = controller.series_file
            original_items = list(controller.series_items)
            prepared = _PreparedSeriesManifest(controller, {"items": []})
            rollback = prepared.capture_for_commit(controller.main)

            controller.series_file = "replacement.seriesctpr"
            controller.series_items = []
            rollback()

            self.assertEqual(controller.series_file, original_file)
            self.assertEqual(controller.series_items, original_items)

    def test_failed_series_child_staging_cleans_new_temp_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = self._controller(self._series_file(temp_dir))
            old_work = os.path.join(temp_dir, "existing-child")
            new_work = os.path.join(temp_dir, "new-child")
            staged_project = os.path.join(new_work, "staged-project")
            os.makedirs(old_work)
            os.makedirs(staged_project)
            controller.active_child_temp_dir = old_work
            snapshot = ProjectLoadSnapshot(
                SimpleNamespace(temp_dir=staged_project),
                "",
            )
            prepared = _PreparedSeriesChild(
                controller,
                os.path.join(new_work, "child.ctpr"),
                snapshot,
                new_work,
            )

            prepared.cleanup()

            self.assertTrue(os.path.isdir(old_work))
            self.assertFalse(os.path.exists(new_work))


if __name__ == "__main__":
    unittest.main()
