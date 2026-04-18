from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets
from shiboken6 import isValid

from app.ui.series_workspace import SeriesWorkspace


class SeriesWorkspaceRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.widget = SeriesWorkspace()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.widget.configure_options(
            languages=[("Japanese", "Japanese"), ("Korean", "Korean")],
            ocr_modes=[("best_local", "Optimal (HunyuanOCR / PaddleOCR VL)")],
            translators=[("gemma_local", "Custom Local Server(Gemma)")],
            workflow_modes=[("stage_batched_pipeline", "Stage-Batched Pipeline (Recommended)")],
        )
        self.widget.set_global_settings(
            {
                "source_language": "Japanese",
                "target_language": "Korean",
                "ocr": "best_local",
                "translator": "gemma_local",
                "workflow_mode": "stage_batched_pipeline",
                "use_gpu": True,
            }
        )
        self._path_a = os.path.join(self.temp_dir.name, "chapter-01.ctpr")
        self._path_b = os.path.join(self.temp_dir.name, "nested", "chapter-02.png")
        os.makedirs(os.path.dirname(self._path_b), exist_ok=True)
        with open(self._path_a, "wb") as fh:
            fh.write(b"ctpr")
        with open(self._path_b, "wb") as fh:
            fh.write(b"png")
        os.utime(self._path_a, (1_700_000_000, 1_700_000_000))
        os.utime(self._path_b, (1_800_000_000, 1_800_000_000))
        self.addCleanup(self.temp_dir.cleanup)
        self.addCleanup(self.widget.deleteLater)

    def _items(self) -> list[dict[str, object]]:
        return [
            {
                "series_item_id": "item-1",
                "queue_index": 1,
                "display_name": "chapter-01.ctpr",
                "source_kind": "ctpr_import",
                "source_origin_relpath": "chapter-01.ctpr",
                "source_origin_path": self._path_a,
                "status": "pending",
            },
            {
                "series_item_id": "item-2",
                "queue_index": 2,
                "display_name": "chapter-02.ctpr",
                "source_kind": "source_file",
                "source_origin_relpath": "nested/chapter-02.png",
                "source_origin_path": self._path_b,
                "status": "running",
            },
        ]

    def test_running_queue_locks_reorder_and_queue_controls(self) -> None:
        self.widget.set_navigation_state(can_back=True, can_forward=True)
        self.widget.set_series_state(
            series_file="demo.seriesctpr",
            items=self._items(),
            queue_running=True,
            active_item_id="item-2",
            queue_runtime={
                "queue_state": "running",
                "active_item_id": "item-2",
                "pending_item_ids": [],
                "failed_item_id": None,
                "retry_remaining_by_item": {"item-2": 1},
                "last_run_summary": {},
            },
        )
        QtWidgets.QApplication.processEvents()

        self.assertEqual(
            self.widget.queue_table.dragDropMode(),
            QtWidgets.QAbstractItemView.DragDropMode.NoDragDrop,
        )
        self.assertFalse(self.widget.open_button.isEnabled())
        self.assertFalse(self.widget.add_files_button.isEnabled())
        self.assertFalse(self.widget.add_folder_button.isEnabled())
        self.assertFalse(self.widget.quick_settings.source_lang_combo.isEnabled())
        self.assertFalse(self.widget.quick_settings.series_settings_button.isEnabled())
        self.assertFalse(self.widget.back_button.isEnabled())
        self.assertFalse(self.widget.forward_button.isEnabled())
        self.assertFalse(self.widget.tree_button.isEnabled())
        self.assertFalse(self.widget.quick_settings.auto_translate_button.isEnabled())
        self.assertFalse(self.widget.queue_notice.isHidden())
        self.assertEqual(self.widget.status_panel.state_value.text(), "Running")
        self.assertEqual(self.widget.status_panel.current_item_value.text(), "#02 · chapter-02.ctpr")
        self.assertFalse(self.widget.status_panel.pause_button.isHidden())
        self.assertTrue(self.widget.status_panel.resume_button.isHidden())
        self.assertEqual(self.widget.queue_table.item(0, 4).text(), "Pending")
        self.assertEqual(self.widget.queue_table.item(1, 4).text(), "Running")
        self.assertTrue(self.widget.queue_table.item(1, 1).font().bold())
        self.assertFalse(self.widget.queue_table.cellWidget(0, 5).isEnabled())

    def test_idle_queue_restores_controls(self) -> None:
        self.widget.set_navigation_state(can_back=True, can_forward=False)
        self.widget.set_series_state(
            series_file="demo.seriesctpr",
            items=self._items(),
            queue_running=False,
            active_item_id="",
            queue_runtime={
                "queue_state": "idle",
                "active_item_id": None,
                "pending_item_ids": [],
                "failed_item_id": None,
                "last_run_summary": {
                    "done_count": 2,
                    "failed_count": 0,
                    "skipped_count": 0,
                    "duration_sec": 42,
                    "started_at": "2026-04-19T10:00:00",
                    "finished_at": "2026-04-19T10:00:42",
                },
            },
        )
        QtWidgets.QApplication.processEvents()

        self.assertEqual(
            self.widget.queue_table.dragDropMode(),
            QtWidgets.QAbstractItemView.DragDropMode.InternalMove,
        )
        self.assertTrue(self.widget.open_button.isEnabled())
        self.assertTrue(self.widget.add_files_button.isEnabled())
        self.assertTrue(self.widget.add_folder_button.isEnabled())
        self.assertTrue(self.widget.quick_settings.source_lang_combo.isEnabled())
        self.assertTrue(self.widget.quick_settings.series_settings_button.isEnabled())
        self.assertTrue(self.widget.back_button.isEnabled())
        self.assertFalse(self.widget.forward_button.isEnabled())
        self.assertTrue(self.widget.tree_button.isEnabled())
        self.assertTrue(self.widget.quick_settings.auto_translate_button.isEnabled())
        self.assertTrue(self.widget.queue_notice.isHidden())
        self.assertEqual(self.widget.status_panel.state_value.text(), "Idle")
        self.assertTrue(self.widget.status_panel.pause_button.isHidden())
        self.assertTrue(self.widget.status_panel.resume_button.isHidden())
        self.assertEqual(self.widget.summary_panel.done_value.text(), "2")
        self.assertEqual(self.widget.summary_panel.duration_value.text(), "42 sec")
        self.assertEqual(self.widget.queue_table.item(0, 4).text(), "Pending")
        self.assertEqual(self.widget.queue_table.item(1, 4).text(), "Running")
        self.assertTrue(self.widget.queue_table.cellWidget(0, 5).isEnabled())

    def test_paused_queue_shows_resume_and_unlocks_reorder(self) -> None:
        self.widget.set_navigation_state(can_back=False, can_forward=False)
        self.widget.set_series_state(
            series_file="demo.seriesctpr",
            items=self._items(),
            queue_running=False,
            active_item_id="",
            queue_runtime={
                "queue_state": "paused",
                "active_item_id": None,
                "pending_item_ids": ["item-1"],
                "failed_item_id": "item-2",
                "retry_remaining_by_item": {"item-2": 0},
                "last_run_summary": {},
            },
            child_unsynced_dirty=True,
            recovery_loaded=True,
        )
        QtWidgets.QApplication.processEvents()

        self.assertEqual(
            self.widget.queue_table.dragDropMode(),
            QtWidgets.QAbstractItemView.DragDropMode.InternalMove,
        )
        self.assertFalse(self.widget.quick_settings.auto_translate_button.isEnabled())
        self.assertFalse(self.widget.status_panel.resume_button.isHidden())
        self.assertTrue(self.widget.status_panel.resume_button.isEnabled())
        self.assertTrue(self.widget.status_panel.open_failed_button.isEnabled())
        self.assertFalse(self.widget.recovery_badge.isHidden())
        self.assertFalse(self.widget.unsynced_badge.isHidden())

    def test_sort_combo_emits_reorder_and_resets_to_manual(self) -> None:
        self.widget.set_series_state(
            series_file="demo.seriesctpr",
            items=self._items(),
            queue_running=False,
            active_item_id="",
            queue_runtime={"queue_state": "idle", "last_run_summary": {}},
        )
        captured: list[list[str]] = []
        self.widget.reorder_requested.connect(captured.append)

        self.widget.sort_combo.setCurrentIndex(self.widget.sort_combo.findData("date_desc"))
        QtWidgets.QApplication.processEvents()

        self.assertEqual(captured, [["item-2", "item-1"]])
        self.assertEqual(self.widget.sort_combo.currentData(), "manual")

    def test_close_with_visible_hover_preview_cleans_up_safely(self) -> None:
        self.widget.set_series_state(
            series_file="demo.seriesctpr",
            items=self._items(),
            queue_running=False,
            active_item_id="",
            queue_runtime={"queue_state": "idle", "last_run_summary": {}},
        )
        payload = self._items()[0]
        popup = self.widget._hover_preview_popup

        self.widget._queue_hover_requested(payload, self.widget.mapToGlobal(self.widget.rect().center()))
        self.widget._show_pending_hover_preview()
        QtWidgets.QApplication.processEvents()

        self.assertTrue(popup.isVisible())

        self.widget.close()
        QtWidgets.QApplication.processEvents()

        self.assertFalse(isValid(popup) and popup.isVisible())


if __name__ == "__main__":
    unittest.main()
