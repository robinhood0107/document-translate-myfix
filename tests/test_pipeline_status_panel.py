from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtTest, QtWidgets
from PySide6.QtGui import QImage

from app.ui.pipeline_status_panel import PipelineInteractionOverlay, PipelineStatusPanel


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

    def test_panel_defaults_to_embedded_mode(self) -> None:
        self.assertEqual(self.panel.display_mode(), PipelineStatusPanel.EMBEDDED_MODE)
        self.assertIs(self.panel.parentWidget(), self.parent)

    def test_toggle_display_mode_switches_to_window_and_back(self) -> None:
        self.panel.show()
        QtWidgets.QApplication.processEvents()

        self.panel.set_display_mode(PipelineStatusPanel.WINDOW_MODE)
        QtWidgets.QApplication.processEvents()
        self.assertEqual(self.panel.display_mode(), PipelineStatusPanel.WINDOW_MODE)
        self.assertTrue(bool(self.panel.windowFlags() & QtCore.Qt.WindowType.Window))

        self.panel.set_display_mode(PipelineStatusPanel.EMBEDDED_MODE)
        QtWidgets.QApplication.processEvents()
        self.assertEqual(self.panel.display_mode(), PipelineStatusPanel.EMBEDDED_MODE)
        self.assertIs(self.panel.parentWidget(), self.parent)

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

    def test_cancelling_event_keeps_status_controls_available(self) -> None:
        self.panel.update_event(
            {
                "phase": "pipeline",
                "status": "cancelling",
                "panel_state": "cancelling",
                "service": "batch",
                "message": "cancelling",
            }
        )
        QtWidgets.QApplication.processEvents()

        self.assertEqual(self.panel._current_state, "cancelling")
        self.assertFalse(self.panel.cancel_button.isHidden())
        self.assertFalse(self.panel.cancel_button.isEnabled())
        self.assertFalse(self.panel.report_button.isHidden())
        self.assertTrue(self.panel.report_button.isEnabled())
        self.assertTrue(self.panel.mode_button.isEnabled())
        self.assertTrue(self.panel.logs_button.isEnabled())

    def test_done_event_updates_preview_and_auto_hides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "preview.png")
            QImage(120, 220, QImage.Format.Format_RGB32).save(image_path)
            self.panel.set_output_root(temp_dir)
            self.panel.update_event(
                {
                    "phase": "done",
                    "status": "completed",
                    "service": "batch",
                    "message": "done",
                    "preview_path": image_path,
                    "auto_hide_ms": 1,
                }
            )
            QtWidgets.QApplication.processEvents()
            self.assertFalse(self.panel.open_output_button.isHidden())
            self.assertFalse(self.panel.close_button.isHidden())
            self.assertEqual(self.panel._preview_path, image_path)
            pixmap = self.panel.preview_label.pixmap()
            self.assertIsNotNone(pixmap)
            self.assertFalse(pixmap.isNull())

            QtTest.QTest.qWait(50)
            QtWidgets.QApplication.processEvents()
            self.assertFalse(self.panel.isVisible())

    def test_running_event_without_preview_keeps_existing_preview(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "preview.png")
            QImage(120, 220, QImage.Format.Format_RGB32).save(image_path)
            self.panel.update_event(
                {
                    "phase": "pipeline",
                    "status": "running",
                    "service": "batch",
                    "message": "preview",
                    "preview_path": image_path,
                }
            )
            self.panel.update_event(
                {
                    "phase": "pipeline",
                    "status": "running",
                    "service": "batch",
                    "message": "still running",
                }
            )
            QtWidgets.QApplication.processEvents()

            self.assertEqual(self.panel._preview_path, image_path)
            pixmap = self.panel.preview_label.pixmap()
            self.assertIsNotNone(pixmap)
            self.assertFalse(pixmap.isNull())

    def test_completed_sub_event_does_not_mark_pipeline_done(self) -> None:
        self.panel.update_event(
            {
                "phase": "pipeline",
                "status": "completed",
                "panel_state": "running",
                "service": "batch",
                "message": "page done",
            }
        )
        QtWidgets.QApplication.processEvents()

        self.assertEqual(self.panel._current_state, "running")
        self.assertTrue(self.panel._pipeline_active)

    def test_hiding_logs_reduces_left_column_width(self) -> None:
        self.panel.show()
        QtWidgets.QApplication.processEvents()
        width_with_logs = self.panel.left_panel.width()
        self.panel.set_logs_visible(False)
        QtWidgets.QApplication.processEvents()
        self.assertLess(self.panel.left_panel.width(), width_with_logs)

    def test_progress_events_do_not_reset_user_resized_geometry(self) -> None:
        self.panel.show()
        self.panel.setGeometry(80, 90, 760, 500)
        QtWidgets.QApplication.processEvents()
        expected = QtCore.QRect(self.panel.geometry())

        self.panel.update_event(
            {
                "phase": "pipeline",
                "status": "running",
                "service": "batch",
                "message": "still running",
                "page_total": 10,
                "page_index": 2,
            }
        )
        QtWidgets.QApplication.processEvents()

        self.assertEqual(self.panel.geometry(), expected)


class PipelineInteractionOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_overlay_uses_light_pass_through_dim_without_center_card(self) -> None:
        parent = QtWidgets.QWidget()
        parent.setGeometry(0, 0, 640, 360)
        overlay = PipelineInteractionOverlay(parent)
        overlay.setGeometry(parent.rect())
        self.addCleanup(overlay.deleteLater)
        self.addCleanup(parent.deleteLater)
        parent.show()
        overlay.show()
        QtWidgets.QApplication.processEvents()

        self.assertTrue(
            overlay.testAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        )
        self.assertEqual(overlay.DIM_COLOR.alpha(), 31)

        image = QImage(640, 360, QImage.Format.Format_ARGB32)
        image.fill(QtCore.Qt.GlobalColor.transparent)
        overlay.render(image)
        self.assertEqual(
            image.pixelColor(12, 12).getRgb(),
            image.pixelColor(320, 180).getRgb(),
        )
