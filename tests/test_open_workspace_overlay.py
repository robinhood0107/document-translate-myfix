from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from app.ui.open_workspace_overlay import OpenWorkspaceOverlay
from app.controllers.open_workspace import (
    OpenWorkspaceCoordinator,
    PreparedWorkspace,
    _WorkspaceCommitTransaction,
)


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

    @staticmethod
    def _transaction_main(old_handler, old_root: str = ""):
        image = object()
        undo_stack = object()
        main = SimpleNamespace(
            curr_img_idx=0,
            image_files=["old-page.png"],
            image_states={"old-page.png": {"state": "old"}},
            image_data={"old-page.png": image},
            image_history={"old-page.png": ["old-page.png"]},
            in_memory_history={"old-page.png": [image]},
            current_history_index={"old-page.png": 0},
            displayed_images={"old-page.png"},
            image_patches={"old-page.png": []},
            in_memory_patches={},
            image_cards=[object()],
            loaded_images=["old-page.png"],
            undo_stacks={"old-page.png": undo_stack},
            file_handler=old_handler,
            _workspace_owned_temp_roots=[old_root] if old_root else [],
            image_ctrl=SimpleNamespace(
                update_image_cards=mock.Mock(),
                display_image_from_loaded=mock.Mock(),
            ),
            page_list=SimpleNamespace(setCurrentRow=mock.Mock()),
            batch_report_ctrl=SimpleNamespace(refresh_action_buttons=mock.Mock()),
            setWindowTitle=mock.Mock(),
            windowTitle=lambda: "Existing workspace",
            _detach_undo_stack=mock.Mock(),
        )
        return main

    def test_failed_commit_restores_old_workspace_objects_and_file_handler(self) -> None:
        old_handler = mock.Mock()
        new_handler = mock.Mock()
        main = self._transaction_main(old_handler)
        old_image_data = main.image_data
        old_image_files = main.image_files
        old_undo_stacks = main.undo_stacks
        prepared = PreparedWorkspace(new_handler, ["new-page.png"], None)
        transaction = _WorkspaceCommitTransaction(main, prepared)

        transaction.isolate_mutable_state()
        main.image_data["new-page.png"] = object()
        main.image_files = ["new-page.png"]
        main.file_handler = new_handler
        transaction.rollback()

        self.assertIs(main.file_handler, old_handler)
        self.assertIs(main.image_data, old_image_data)
        self.assertIs(main.image_files, old_image_files)
        self.assertIs(main.undo_stacks, old_undo_stacks)
        self.assertEqual(list(main.undo_stacks), ["old-page.png"])
        self.assertEqual(main.image_files, ["old-page.png"])
        old_handler.cleanup.assert_not_called()
        new_handler.cleanup.assert_called_once_with()

    def test_successful_commit_transfers_new_handler_before_old_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            old_root = os.path.join(temp_dir, "old-workspace")
            os.makedirs(old_root)
            old_handler = mock.Mock()
            new_handler = mock.Mock()
            main = self._transaction_main(old_handler, old_root)
            prepared = PreparedWorkspace(new_handler, ["new-page.png"], None)
            transaction = _WorkspaceCommitTransaction(main, prepared)
            old_undo_stacks = main.undo_stacks
            transaction.isolate_mutable_state()
            main.file_handler = new_handler
            main.image_files = ["new-page.png"]

            transaction.commit()
            prepared.cleanup()

            old_handler.cleanup.assert_called_once_with()
            new_handler.cleanup.assert_not_called()
            main._detach_undo_stack.assert_called_once()
            self.assertEqual(old_undo_stacks, {})
            self.assertEqual(main._workspace_owned_temp_roots, [])
            self.assertFalse(os.path.exists(old_root))


if __name__ == "__main__":
    unittest.main()
