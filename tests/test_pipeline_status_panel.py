from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QImage

from app.ui.pipeline_status_panel import PipelineStatusPanel


class PipelineStatusPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.parent = QtWidgets.QWidget()
        self.parent.setGeometry(0, 0, 1200, 900)
        self.parent.show()
        self.panel = PipelineStatusPanel(self.parent)
        self.panel.set_allowed_area(QtCore.QRect(0, 0, 1200, 900))
        self.addCleanup(self.panel.deleteLater)
        self.addCleanup(self.parent.deleteLater)

    def test_running_event_shows_cancel_and_report(self) -> None:
        self.panel.update_event(
            {
                "phase": "pipeline",
                "status": "running",
                "service": "batch",
                "message": "running",
                "page_total": 3,
                "page_index": 1,
                "image_name": "page01.png",
                "eta_text": "00:01:00",
            }
        )
        QtWidgets.QApplication.processEvents()
        self.assertFalse(self.panel.cancel_button.isHidden())
        self.assertFalse(self.panel.report_button.isHidden())
        self.assertTrue(self.panel.retry_button.isHidden())

    def test_done_event_updates_preview_and_output_button(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "preview.png")
            QImage(32, 32, QImage.Format.Format_RGB32).save(image_path)
            self.panel.set_output_root(temp_dir)
            self.panel.update_event(
                {
                    "phase": "done",
                    "status": "completed",
                    "service": "batch",
                    "message": "done",
                    "preview_path": image_path,
                }
            )
            QtWidgets.QApplication.processEvents()
            self.assertFalse(self.panel.open_output_button.isHidden())
            self.assertFalse(self.panel.close_button.isHidden())
            self.assertEqual(self.panel._preview_path, image_path)
            pixmap = self.panel.preview_label.pixmap()
            self.assertIsNotNone(pixmap)
            self.assertFalse(pixmap.isNull())
