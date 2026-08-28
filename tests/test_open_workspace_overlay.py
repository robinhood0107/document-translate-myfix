from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from app.ui.open_workspace_overlay import OpenWorkspaceOverlay
from app.controllers.open_workspace import OpenWorkspaceCoordinator


class OpenWorkspaceOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_overlay_is_embedded_and_never_a_top_level_window(self) -> None:
        host = QtWidgets.QWidget()
        overlay = OpenWorkspaceOverlay(host)
        self.addCleanup(host.deleteLater)

        overlay.setGeometry(host.rect())
        overlay.start("Checking selected files...")
        QtWidgets.QApplication.processEvents()

        self.assertIs(overlay.parentWidget(), host)
        self.assertFalse(overlay.isWindow())
        self.assertNotIn(overlay, QtWidgets.QApplication.topLevelWidgets())
        self.assertFalse(overlay.close_button.isVisible())

    def test_overlay_switches_between_indeterminate_and_page_progress(self) -> None:
        host = QtWidgets.QWidget()
        overlay = OpenWorkspaceOverlay(host)
        self.addCleanup(host.deleteLater)

        overlay.start("Indexing archive...")
        self.assertEqual(overlay.progress_bar.maximum(), 0)

        overlay.update_progress(
            {
                "message": "Indexing archive pages...",
                "current": 7,
                "total": 12,
            }
        )
        self.assertEqual(overlay.progress_bar.maximum(), 12)
        self.assertEqual(overlay.progress_bar.value(), 7)

        overlay.show_failure("Could not open workspace", "See the log")
        self.assertFalse(overlay.isWindow())
        self.assertFalse(overlay.close_button.isHidden())

    def test_project_preparation_mutates_only_the_staging_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            main = SimpleNamespace(
                temp_dir=temp_dir,
                image_files=["old-page.png"],
                settings_page=object(),
                tr=lambda text: text,
            )
            events: list[dict] = []

            def fake_load(staging, _path):
                staging.image_files = ["new-page.png"]
                staging.curr_img_idx = 0
                return "context"

            with mock.patch(
                "app.projects.project_state.load_state_from_proj_file",
                side_effect=fake_load,
            ):
                snapshot = OpenWorkspaceCoordinator.prepare_project(
                    events.append,
                    main,
                    os.path.join(temp_dir, "example.ctpr"),
                )
            self.addCleanup(snapshot.cleanup)

            self.assertEqual(main.image_files, ["old-page.png"])
            self.assertEqual(snapshot.state.image_files, ["new-page.png"])
            self.assertEqual(snapshot.saved_context, "context")
            self.assertEqual([event["stage"] for event in events], ["validate", "index"])


if __name__ == "__main__":
    unittest.main()
