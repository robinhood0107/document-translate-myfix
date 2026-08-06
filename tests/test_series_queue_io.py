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


class _ProjectControllerStub:
    def __init__(self) -> None:
        self.save_current_state_calls = 0

    def save_current_state(self) -> None:
        self.save_current_state_calls += 1


class _MainStub(QtCore.QObject):
    def __init__(self) -> None:
        super().__init__()
        self.loading = _VisibleStub()
        self.pipeline_status_panel = _PipelineStatusStub()
        self.series_workspace = _SeriesWorkspaceStub()
        self.project_ctrl = _ProjectControllerStub()
        self.project_file = None
        self.project_kind = "series"

    def tr(self, text: str) -> str:
        return text

    def setWindowTitle(self, _title: str) -> None:
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


class ChildSyncPhaseTests(unittest.TestCase):
    """자식 동기화는 메인 스레드 단계와 워커 단계로 나뉘어야 한다."""

    def _activate_child(self, controller: SeriesController, temp_dir: str) -> str:
        work_dir = os.path.join(temp_dir, "work")
        os.makedirs(work_dir, exist_ok=True)
        child_path = os.path.join(work_dir, "child.ctpr")
        controller.active_child_item_id = "item-1"
        controller.active_child_project_path = child_path
        controller.active_child_temp_dir = work_dir
        return child_path

    def test_prepare_collects_ui_state_on_the_main_thread(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            series_path = _series_file(temp_dir)
            controller = _controller(series_path)
            child_path = self._activate_child(controller, temp_dir)

            payload = controller.prepare_active_child_sync()

            self.assertEqual(controller.main.project_ctrl.save_current_state_calls, 1)
            self.assertEqual(payload, {
                "series_file": series_path,
                "child_project_path": child_path,
                "series_item_id": "item-1",
            })

    def test_prepare_returns_none_without_an_active_child(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = _controller(_series_file(temp_dir))
            self.assertIsNone(controller.prepare_active_child_sync())
            self.assertEqual(controller.main.project_ctrl.save_current_state_calls, 0)

    def test_write_never_mutates_the_global_project_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            series_path = _series_file(temp_dir)
            controller = _controller(series_path)
            child_path = self._activate_child(controller, temp_dir)
            controller.main.project_file = series_path
            payload = controller.prepare_active_child_sync()
            assert payload is not None

            seen: dict[str, object] = {}

            def fake_save(main, file_name, **kwargs):
                # 워커 단계에서 전역이 자식 경로로 바뀌어 있으면 메인 스레드가
                # 잘못된 project_file 을 읽는다.
                seen["project_file_during_write"] = main.project_file
                seen["source_project_file"] = kwargs.get("source_project_file")

            with mock.patch("app.controllers.series.save_state_to_proj_file", fake_save), \
                mock.patch("app.controllers.series.update_series_child_from_file") as embed_mock:
                controller.write_active_child_sync(payload)

            self.assertEqual(seen["project_file_during_write"], series_path)
            self.assertEqual(seen["source_project_file"], child_path)
            self.assertEqual(controller.main.project_file, series_path)
            embed_mock.assert_called_once()
            self.assertEqual(embed_mock.call_args.args[0], series_path)

    def test_write_can_target_a_snapshot_instead_of_the_original(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            series_path = _series_file(temp_dir)
            controller = _controller(series_path)
            self._activate_child(controller, temp_dir)
            payload = controller.prepare_active_child_sync()
            assert payload is not None
            snapshot = os.path.join(temp_dir, "recovery.seriesctpr")

            with mock.patch("app.controllers.series.save_state_to_proj_file"), \
                mock.patch("app.controllers.series.update_series_child_from_file") as embed_mock:
                controller.write_active_child_sync(payload, series_target_file=snapshot)

            self.assertEqual(embed_mock.call_args.args[0], snapshot)

    def test_sync_wrapper_still_runs_all_three_phases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = _controller(_series_file(temp_dir))
            self._activate_child(controller, temp_dir)
            controller._child_unsynced_dirty = True

            with mock.patch("app.controllers.series.save_state_to_proj_file"), \
                mock.patch("app.controllers.series.update_series_child_from_file"), \
                mock.patch.object(controller, "_apply_workspace_state"):
                controller.sync_active_child_to_series()

            self.assertFalse(controller._child_unsynced_dirty)


class QueueRuntimeIoTests(unittest.TestCase):
    """큐 상태 변경마다 시리즈 파일을 다시 읽지 않아야 한다."""

    def test_queue_runtime_update_does_not_reload_the_series_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = _controller(_series_file(temp_dir))

            with mock.patch("app.controllers.series.load_series_project") as load_mock:
                controller._update_queue_runtime(queue_state="running")

            load_mock.assert_not_called()

    def test_queue_runtime_update_keeps_the_manifest_current(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = _controller(_series_file(temp_dir))

            controller._update_queue_runtime(queue_state="running")

            self.assertEqual(
                controller.series_manifest["series_queue_runtime"]["queue_state"],
                "running",
            )
            # 파일에도 반영되어 있어야 한다.
            reloaded = load_series_project(str(controller.series_file))
            self.assertEqual(
                reloaded["manifest"]["series_queue_runtime"]["queue_state"],
                "running",
            )


class QueueRuntimeSentinelTests(unittest.TestCase):
    """생략한 큐 런타임 인자는 기존 값을 보존해야 한다."""

    def test_partial_update_preserves_untouched_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = _controller(_series_file(temp_dir))
            controller._update_queue_runtime(
                queue_state="running",
                pending_item_ids=["item-1"],
                active_item_id="item-1",
            )

            # 컨트롤러와 상태 모듈이 서로 다른 sentinel 을 쓰면, 생략한
            # completed_item_ids 등이 "빈 값이 주어졌다"로 해석돼 통째로
            # 지워지거나 TypeError 가 난다.
            controller._update_queue_runtime(queue_state="paused")

            runtime = controller.series_manifest["series_queue_runtime"]
            self.assertEqual(runtime["queue_state"], "paused")
            self.assertEqual(list(runtime["pending_item_ids"]), ["item-1"])
            self.assertEqual(runtime["active_item_id"], "item-1")

    def test_controller_and_state_module_share_one_sentinel(self) -> None:
        from app.controllers import series as series_module
        from app.projects import series_state_v1

        self.assertIs(series_module._UNSET, series_state_v1._UNSET)


class ChildTempDirLeakTests(unittest.TestCase):
    def test_failed_open_cleans_up_the_previous_work_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = _controller(_series_file(temp_dir))
            controller.main.temp_dir = temp_dir
            old_dir = os.path.join(temp_dir, "old_work")
            os.makedirs(old_dir, exist_ok=True)
            controller.active_child_temp_dir = old_dir
            controller.active_child_item_id = "item-1"

            captured: dict[str, object] = {}

            def fake_run_threaded(worker, on_result, on_error, on_finished, *args):
                captured["on_error"] = on_error

            controller.main.run_threaded = fake_run_threaded
            controller.main.default_error_handler = lambda _e: None

            with mock.patch("app.controllers.series.Messages.show_busy", return_value=object()), \
                mock.patch("app.controllers.series.Messages.close_busy"), \
                mock.patch("app.controllers.series.materialize_series_child_project",
                           return_value=os.path.join(temp_dir, "child.ctpr")), \
                mock.patch.object(controller.main, "image_ctrl", create=True):
                controller._open_item("item-1", push_history=False)
                captured["on_error"]((RuntimeError, RuntimeError("boom"), None))

            # 예전에는 실패 경로가 새 작업본만 지우고 직전 것은 남겨, 열기
            # 실패가 쌓일수록 series_child_* 가 누적됐다.
            self.assertFalse(os.path.isdir(old_dir))


if __name__ == "__main__":
    unittest.main()
