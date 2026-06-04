from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from app.controllers.image import ImageStateController
from app.controllers.manual_workflow import ManualWorkflowController


class _ProjectControllerStub:
    def __init__(self) -> None:
        self.clear_recovery_checkpoint_called = False

    def clear_recovery_checkpoint(self) -> None:
        self.clear_recovery_checkpoint_called = True


class _MainStub:
    def __init__(self) -> None:
        self.project_ctrl = _ProjectControllerStub()
        self.image_files: list[str] = []
        self.threaded_calls: list[tuple] = []
        self.errors: list[tuple] = []

    def tr(self, text: str) -> str:
        return text

    def run_threaded(self, *args, **kwargs):
        self.threaded_calls.append((args, kwargs))

    def default_error_handler(self, error_tuple: tuple) -> None:
        self.errors.append(error_tuple)


class BusyEntrypointTests(unittest.TestCase):
    def test_manual_busy_helper_uses_processing_modal(self) -> None:
        main = _MainStub()
        controller = ManualWorkflowController(main)
        busy_dialog = object()

        with mock.patch("app.controllers.manual_workflow.Messages.show_busy", return_value=busy_dialog) as show_busy:
            self.assertIs(controller._show_manual_busy("Preparing OCR..."), busy_dialog)

        show_busy.assert_called_once_with(
            main,
            "Preparing OCR...",
            title="Processing",
            minimum_visible_ms=300,
        )

    def test_image_load_shows_busy_and_closes_on_finish(self) -> None:
        main = _MainStub()
        controller = ImageStateController.__new__(ImageStateController)
        controller.main = main
        controller.clear_state = mock.Mock()
        controller.load_initial_image = mock.Mock(name="load_initial_image")
        controller.on_initial_image_loaded = mock.Mock(name="on_initial_image_loaded")
        busy_dialog = object()

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "page.input")
            with mock.patch("app.controllers.image.Messages.show_busy", return_value=busy_dialog) as show_busy, \
                mock.patch("app.controllers.image.Messages.close_busy") as close_busy:
                controller.thread_load_images([input_path])

                show_busy.assert_called_once()
                self.assertEqual(show_busy.call_args.args[1], "Loading images...")
                self.assertEqual(show_busy.call_args.kwargs["title"], "Import")
                self.assertEqual(show_busy.call_args.kwargs["minimum_visible_ms"], 300)

                threaded_args, _threaded_kwargs = main.threaded_calls[0]
                self.assertIs(threaded_args[0], controller.load_initial_image)
                self.assertIs(threaded_args[1], controller.on_initial_image_loaded)
                threaded_args[3]()

        close_busy.assert_called_once_with(busy_dialog)

    def test_image_insert_shows_busy_and_forwards_errors(self) -> None:
        main = _MainStub()
        main.image_files = ["existing"]
        main.file_handler = mock.Mock()
        main.file_handler.prepare_files.return_value = []
        controller = ImageStateController.__new__(ImageStateController)
        controller.main = main
        busy_dialog = object()

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "new-page.input")
            with mock.patch("app.controllers.image.Messages.show_busy", return_value=busy_dialog) as show_busy, \
                mock.patch("app.controllers.image.Messages.close_busy") as close_busy:
                controller.thread_insert([input_path])

                show_busy.assert_called_once()
                self.assertEqual(show_busy.call_args.args[1], "Importing pages...")
                threaded_args, _threaded_kwargs = main.threaded_calls[0]
                error_tuple = (RuntimeError, RuntimeError("failed"), "")
                threaded_args[2](error_tuple)

        close_busy.assert_called_once_with(busy_dialog, force=True)
        self.assertEqual(main.errors, [error_tuple])


if __name__ == "__main__":
    unittest.main()
