from __future__ import annotations

import os
import unittest
from unittest import mock

from app.controllers.image import ImageStateController


class _DummyCombo:
    def __init__(self, value: str) -> None:
        self._value = value

    def currentText(self) -> str:
        return self._value


class _DummyMain:
    def __init__(self) -> None:
        self.image_states: dict[str, dict] = {}
        self.s_combo = _DummyCombo("Japanese")
        self.t_combo = _DummyCombo("English")
        self.image_files: list[str] = []


class ImageStateControllerTests(unittest.TestCase):
    def test_apply_languages_to_paths_updates_only_selected_pages(self) -> None:
        main = _DummyMain()
        controller = ImageStateController.__new__(ImageStateController)
        controller.main = main

        controller.ensure_page_state("page-a.png")
        controller.ensure_page_state("page-b.png")
        controller.ensure_page_state("page-c.png")

        controller.apply_languages_to_paths(
            ["page-a.png", "page-c.png"],
            "Chinese",
            "Korean",
        )

        self.assertEqual(main.image_states["page-a.png"]["source_lang"], "Chinese")
        self.assertEqual(main.image_states["page-a.png"]["target_lang"], "Korean")
        self.assertEqual(main.image_states["page-c.png"]["source_lang"], "Chinese")
        self.assertEqual(main.image_states["page-c.png"]["target_lang"], "Korean")
        self.assertEqual(main.image_states["page-b.png"]["source_lang"], "Japanese")
        self.assertEqual(main.image_states["page-b.png"]["target_lang"], "English")

    def test_sort_images_by_name_uses_requested_direction(self) -> None:
        main = _DummyMain()
        main.image_files = ["c.png", "a.png", "b.png"]
        controller = ImageStateController.__new__(ImageStateController)
        controller.main = main

        captured: list[list[str]] = []
        controller._apply_image_order = lambda order, clear_active_sort=False: captured.append(order)
        controller._resolve_page_sort_source = lambda path: (path, "file")
        controller._safe_mtime = lambda path: 0.0

        controller.sort_images("name", "asc")
        controller.sort_images("name", "desc")

        self.assertEqual(captured[0], ["a.png", "b.png", "c.png"])
        self.assertEqual(captured[1], ["c.png", "b.png", "a.png"])

    def test_sort_images_by_date_preserves_name_tie_break(self) -> None:
        main = _DummyMain()
        main.image_files = ["b.png", "a.png", "c.png"]
        controller = ImageStateController.__new__(ImageStateController)
        controller.main = main

        captured: list[list[str]] = []
        controller._apply_image_order = lambda order, clear_active_sort=False: captured.append(order)
        controller._resolve_page_sort_source = lambda path: (path, "file")
        mtimes = {"a.png": 10.0, "b.png": 10.0, "c.png": 5.0}
        controller._safe_mtime = lambda path: mtimes[path]

        controller.sort_images("date", "desc")
        controller.sort_images("date", "oldest")

        self.assertEqual(captured[0], ["a.png", "b.png", "c.png"])
        self.assertEqual(captured[1], ["c.png", "a.png", "b.png"])

    def test_thread_load_images_does_not_preserve_old_project_file(self) -> None:
        main = mock.Mock()
        main.project_file = "/tmp/old-project.ctpr"
        main.default_error_handler = mock.Mock()
        main.run_threaded = mock.Mock()
        main.setWindowTitle = mock.Mock()
        main.project_ctrl = mock.Mock()

        controller = ImageStateController.__new__(ImageStateController)
        controller.main = main
        controller.clear_state = mock.Mock(
            side_effect=lambda: setattr(main, "project_file", None)
        )

        controller.thread_load_images(["folder/page.png"])

        main.project_ctrl.clear_recovery_checkpoint.assert_called_once()
        controller.clear_state.assert_called_once()
        self.assertIsNone(main.project_file)
        loaded_paths = main.run_threaded.call_args.args[-1]
        self.assertEqual(loaded_paths, [os.path.abspath("folder/page.png")])


if __name__ == "__main__":
    unittest.main()
