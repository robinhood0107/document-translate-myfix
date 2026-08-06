from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

from app.controllers.series import SeriesController
from app.projects.series_state_v1 import create_series_project, load_series_project
from app.shortcuts import get_default_shortcuts
from app.ui.series_breadcrumb import SeriesBreadcrumbBar


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


class _MainStub(QtCore.QObject):
    def __init__(self) -> None:
        super().__init__()
        self.loading = _VisibleStub()
        self.pipeline_status_panel = _PipelineStatusStub()
        self.series_workspace = _SeriesWorkspaceStub()
        self.breadcrumb_refresh_count = 0

    def tr(self, text: str) -> str:
        return text

    def refresh_series_breadcrumb(self) -> None:
        self.breadcrumb_refresh_count += 1


class SeriesBreadcrumbWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.widget = SeriesBreadcrumbBar()
        self.addCleanup(self.widget.deleteLater)

    def test_context_shows_series_and_child_names(self) -> None:
        self.widget.set_context(
            series_name="example_series.seriesctpr",
            child_name="example source chapter v01 c01 (E)",
            can_back=True,
        )

        # 표시 문자열은 폰트 메트릭에 따라 생략될 수 있으므로, 전체 이름은
        # 툴팁으로만 계약한다.
        self.assertTrue(self.widget.series_button.text())
        self.assertIn("example_series.seriesctpr", self.widget.series_button.toolTip())
        self.assertTrue(self.widget.child_label.text())
        self.assertEqual(
            self.widget.child_label.toolTip(),
            "example source chapter v01 c01 (E)",
        )
        self.assertFalse(self.widget.unsynced_badge.isVisibleTo(self.widget))
        self.assertTrue(self.widget.back_button.isEnabled())

    def test_unsynced_badge_visible_only_when_child_is_dirty(self) -> None:
        self.widget.set_context(series_name="s.seriesctpr", child_name="c", unsynced=False)
        self.assertFalse(self.widget.unsynced_badge.isVisibleTo(self.widget))

        self.widget.set_context(series_name="s.seriesctpr", child_name="c", unsynced=True)
        self.assertTrue(self.widget.unsynced_badge.isVisibleTo(self.widget))

    def test_back_is_disabled_without_history(self) -> None:
        self.widget.set_context(series_name="s.seriesctpr", child_name="c", can_back=False)
        self.assertFalse(self.widget.back_button.isEnabled())

    def test_queue_lock_disables_back_but_never_the_board_link(self) -> None:
        # 큐 실행 중에도 보드는 열려야 한다. `show_board_during_queue` 가
        # 자식 materialization 을 유지한 채 보드 화면만 보여준다.
        self.widget.set_context(
            series_name="s.seriesctpr",
            child_name="c",
            can_back=True,
            locked_reason="locked while running",
        )

        self.assertFalse(self.widget.back_button.isEnabled())
        self.assertEqual(self.widget.back_button.toolTip(), "locked while running")
        self.assertTrue(self.widget.series_button.isEnabled())

    def test_board_and_back_signals_fire_on_click(self) -> None:
        received: list[str] = []
        self.widget.board_requested.connect(lambda: received.append("board"))
        self.widget.back_requested.connect(lambda: received.append("back"))
        self.widget.set_context(series_name="s.seriesctpr", child_name="c", can_back=True)

        self.widget.series_button.click()
        self.widget.back_button.click()

        self.assertEqual(received, ["board", "back"])


class SeriesBreadcrumbStateTests(unittest.TestCase):
    def _series_file(self, temp_dir: str) -> str:
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

    def _controller(self, series_path: str) -> SeriesController:
        controller = SeriesController(_MainStub())
        state = load_series_project(series_path)
        controller.series_file = series_path
        controller.series_manifest = dict(state["manifest"])
        controller.series_items = list(state["items"])
        return controller

    def test_board_context_has_no_breadcrumb(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = self._controller(self._series_file(temp_dir))
            self.assertIsNone(controller.breadcrumb_state())

    def test_child_context_reports_names_and_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            series_path = self._series_file(temp_dir)
            controller = self._controller(series_path)
            controller.active_child_item_id = "item-1"
            controller.active_child_project_path = os.path.join(temp_dir, "child.ctpr")

            state = controller.breadcrumb_state()

            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state["series_name"], "example.seriesctpr")
            self.assertEqual(state["child_name"], "example_child.ctpr")
            self.assertFalse(state["unsynced"])
            self.assertFalse(state["can_back"])
            self.assertEqual(state["locked_reason"], "")

    def test_child_context_propagates_unsynced_history_and_queue_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            series_path = self._series_file(temp_dir)
            controller = self._controller(series_path)
            controller.active_child_item_id = "item-1"
            controller.active_child_project_path = os.path.join(temp_dir, "child.ctpr")
            controller._child_unsynced_dirty = True
            controller.history_back = [{"kind": "board"}]
            controller._queue_active = True

            state = controller.breadcrumb_state()

            assert state is not None
            self.assertTrue(state["unsynced"])
            self.assertTrue(state["can_back"])
            self.assertTrue(str(state["locked_reason"]))

    def test_window_title_refresh_also_refreshes_breadcrumb(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = self._controller(self._series_file(temp_dir))
            controller.main.setWindowTitle = lambda _title: None

            controller._set_series_window_title("example_child.ctpr")

            self.assertEqual(controller.main.breadcrumb_refresh_count, 1)


class SeriesBackShortcutTests(unittest.TestCase):
    def test_series_back_has_a_default_binding(self) -> None:
        self.assertEqual(get_default_shortcuts()["series_back"], "Alt+Left")


if __name__ == "__main__":
    unittest.main()
