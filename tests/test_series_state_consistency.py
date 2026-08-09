from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from PySide6 import QtCore

from app.controllers.series import SeriesController
from app.projects.series_state_v1 import create_series_project, load_series_project


class _VisibleStub:
    def setVisible(self, _value: bool) -> None:
        pass


class _PipelineStatusStub:
    def set_series_queue_pause_visible(self, _visible: bool, *, pause_requested: bool) -> None:
        pass

    def update_event(self, _event: dict) -> None:
        pass


class _SeriesWorkspaceStub:
    def configure_options(self, **_kwargs) -> None:
        pass

    def set_global_settings(self, _values) -> None:
        pass

    def set_series_state(self, **_kwargs) -> None:
        pass

    def set_navigation_state(self, **_kwargs) -> None:
        pass


class _ImageControllerStub:
    def __init__(self) -> None:
        self.clear_calls = 0

    def clear_state(self) -> None:
        self.clear_calls += 1


class _MainStub(QtCore.QObject):
    def __init__(self) -> None:
        super().__init__()
        self.loading = _VisibleStub()
        self.pipeline_status_panel = _PipelineStatusStub()
        self.series_workspace = _SeriesWorkspaceStub()
        self.image_ctrl = _ImageControllerStub()
        self.project_file = None
        self.project_kind = "single"
        self.clean_calls = 0

    def tr(self, text: str) -> str:
        return text

    def setWindowTitle(self, _title: str) -> None:
        pass

    def set_project_clean(self) -> None:
        self.clean_calls += 1

    def show_series_page(self) -> None:
        pass

    def refresh_series_breadcrumb(self) -> None:
        pass


def _series_file(temp_dir: str) -> str:
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
                "status": "pending",
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


def _controller(series_path: str) -> SeriesController:
    controller = SeriesController(_MainStub())
    state = load_series_project(series_path)
    controller.series_file = series_path
    controller.series_manifest = dict(state["manifest"])
    controller.series_items = list(state["items"])
    return controller


class GlobalSettingsRestoreTests(unittest.TestCase):
    """시리즈를 닫으면 앱 전역 설정이 원래대로 돌아와야 한다."""

    def test_apply_captures_the_pre_series_snapshot_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = _controller(_series_file(temp_dir))
            snapshots = [{"workflow_mode": "before"}, {"workflow_mode": "later"}]

            with mock.patch.object(
                controller, "_snapshot_main_global_settings", side_effect=snapshots
            ), mock.patch.object(controller, "_apply_global_settings_to_main"):
                controller._apply_series_globals_to_main()
                controller._apply_series_globals_to_main()

            # 두 번째 호출은 이미 시리즈 값이 위젯에 들어간 뒤다. 그때 다시
            # 찍으면 시리즈 값을 "원래 값"으로 오인해 복원이 무의미해진다.
            self.assertEqual(controller._main_globals_snapshot, {"workflow_mode": "before"})

    def test_reset_restores_and_clears_the_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = _controller(_series_file(temp_dir))
            controller._main_globals_snapshot = {"workflow_mode": "before"}

            with mock.patch.object(controller, "_apply_global_settings_to_main") as apply_mock:
                controller.reset_series_context()

            apply_mock.assert_called_once_with({"workflow_mode": "before"})
            self.assertIsNone(controller._main_globals_snapshot)

    def test_reset_without_a_snapshot_touches_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = _controller(_series_file(temp_dir))
            controller._main_globals_snapshot = None

            with mock.patch.object(controller, "_apply_global_settings_to_main") as apply_mock:
                controller.reset_series_context()

            apply_mock.assert_not_called()

    def test_snapshot_failure_does_not_block_opening_the_series(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = _controller(_series_file(temp_dir))

            with mock.patch.object(
                controller, "_snapshot_main_global_settings", side_effect=RuntimeError("boom")
            ), mock.patch.object(controller, "_apply_global_settings_to_main") as apply_mock:
                controller._apply_series_globals_to_main()

            self.assertIsNone(controller._main_globals_snapshot)
            apply_mock.assert_called_once()


class ChildTeardownGuardTests(unittest.TestCase):
    """미반영 자식 변경이 조용히 사라지지 않아야 한다."""

    def _child_active(self, controller: SeriesController, temp_dir: str) -> str:
        work_dir = os.path.join(temp_dir, "work")
        os.makedirs(work_dir, exist_ok=True)
        controller.active_child_item_id = "item-1"
        controller.active_child_project_path = os.path.join(work_dir, "child.ctpr")
        controller.active_child_temp_dir = work_dir
        controller._child_unsynced_dirty = True
        return work_dir

    def test_show_board_syncs_pending_child_changes_first(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = _controller(_series_file(temp_dir))
            work_dir = self._child_active(controller, temp_dir)

            with mock.patch.object(controller, "sync_active_child_to_series") as sync_mock, \
                mock.patch.object(controller, "_apply_workspace_state"):
                controller._show_board(push_history=False)

            sync_mock.assert_called_once()
            self.assertFalse(os.path.isdir(work_dir))

    def test_show_board_keeps_the_work_dir_when_sync_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = _controller(_series_file(temp_dir))
            work_dir = self._child_active(controller, temp_dir)

            with mock.patch.object(
                controller, "sync_active_child_to_series", side_effect=RuntimeError("boom")
            ), mock.patch.object(controller, "_apply_workspace_state"), \
                mock.patch("app.controllers.series.Messages.show_warning") as warn_mock:
                controller._show_board(push_history=False)

            # 작업본을 지우면 반영 못 한 변경이 복구 불가능해진다.
            self.assertTrue(os.path.isdir(work_dir))
            warn_mock.assert_called_once()
            self.assertIn(work_dir, warn_mock.call_args.args[1])

    def test_clean_child_teardown_does_not_sync(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = _controller(_series_file(temp_dir))
            self._child_active(controller, temp_dir)
            controller._child_unsynced_dirty = False

            with mock.patch.object(controller, "sync_active_child_to_series") as sync_mock, \
                mock.patch.object(controller, "_apply_workspace_state"):
                controller._show_board(push_history=False)

            sync_mock.assert_not_called()

    def test_batch_finish_sync_failure_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = _controller(_series_file(temp_dir))
            self._child_active(controller, temp_dir)

            with mock.patch.object(
                controller, "sync_active_child_to_series", side_effect=RuntimeError("boom")
            ), mock.patch("app.controllers.series.Messages.show_warning") as warn_mock:
                controller.on_batch_process_finished(was_cancelled=False, failed=False)

            # 예전에는 아무 흔적 없이 return 했다.
            warn_mock.assert_called_once()


class SeriesChildBranchTests(unittest.TestCase):
    """저장 분기는 project_kind 가 아니라 자식 활성 여부를 봐야 한다."""

    def test_project_controller_distinguishes_board_from_child(self) -> None:
        from app.controllers.projects import ProjectController

        controller = ProjectController.__new__(ProjectController)
        main = mock.Mock()
        main.series_ctrl.is_child_project_active.return_value = True
        controller.main = main
        self.assertTrue(controller._series_child_is_active())

        main.series_ctrl.is_child_project_active.return_value = False
        self.assertFalse(controller._series_child_is_active())

    def test_missing_series_controller_is_not_a_child(self) -> None:
        from app.controllers.projects import ProjectController

        controller = ProjectController.__new__(ProjectController)
        controller.main = object()
        self.assertFalse(controller._series_child_is_active())


if __name__ == "__main__":
    unittest.main()
