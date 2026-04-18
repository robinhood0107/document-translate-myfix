from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from app.ui.series_workspace import SeriesWorkspace


class SeriesWorkspaceRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.widget = SeriesWorkspace()
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
        self.addCleanup(self.widget.deleteLater)

    def _items(self) -> list[dict[str, object]]:
        return [
            {
                "series_item_id": "item-1",
                "queue_index": 1,
                "display_name": "chapter-01.ctpr",
                "source_kind": "ctpr_import",
                "source_origin_relpath": "chapter-01.ctpr",
                "source_origin_path": "/tmp/chapter-01.ctpr",
                "status": "pending",
            },
            {
                "series_item_id": "item-2",
                "queue_index": 2,
                "display_name": "chapter-02.ctpr",
                "source_kind": "source_file",
                "source_origin_relpath": "nested/chapter-02.png",
                "source_origin_path": "/tmp/nested/chapter-02.png",
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
        self.assertEqual(self.widget.queue_table.item(0, 4).text(), "Pending")
        self.assertEqual(self.widget.queue_table.item(1, 4).text(), "Running")
        self.assertTrue(self.widget.queue_table.cellWidget(0, 5).isEnabled())


if __name__ == "__main__":
    unittest.main()
