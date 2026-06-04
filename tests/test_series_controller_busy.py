from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from PySide6 import QtCore

from app.controllers.series import SeriesController
from app.projects.series_state_v1 import create_series_project, load_series_project


class _VisibleStub:
    def __init__(self) -> None:
        self.visible_values: list[bool] = []

    def setVisible(self, value: bool) -> None:
        self.visible_values.append(bool(value))


class _PipelineStatusStub:
    def __init__(self) -> None:
        self.pause_visible_calls: list[tuple[bool, bool]] = []

    def set_series_queue_pause_visible(self, visible: bool, *, pause_requested: bool) -> None:
        self.pause_visible_calls.append((bool(visible), bool(pause_requested)))


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

    def tr(self, text: str) -> str:
        return text


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


if __name__ == "__main__":
    unittest.main()
