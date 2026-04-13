from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from controller import ComicTranslate


class _Toggle:
    def __init__(self, checked: bool = False) -> None:
        self._checked = checked
        self.set_calls: list[bool] = []

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, value: bool) -> None:
        self._checked = value
        self.set_calls.append(value)


class _WidgetState:
    def __init__(self) -> None:
        self.enabled_values: list[bool] = []
        self.visible_values: list[bool] = []

    def setEnabled(self, value: bool) -> None:
        self.enabled_values.append(value)

    def setVisible(self, value: bool) -> None:
        self.visible_values.append(value)


class _ImageCtrl:
    def __init__(self, page_states: dict[str, dict[str, str]]) -> None:
        self._page_states = page_states
        self.cleared_paths: list[str] | None = None

    def ensure_page_state(self, path: str) -> dict[str, str]:
        return self._page_states[path]

    def clear_page_skip_errors_for_paths(self, paths: list[str]) -> None:
        self.cleared_paths = list(paths)


class _BatchReportCtrl:
    def __init__(self) -> None:
        self.refresh_calls = 0

    def refresh_action_buttons(self) -> None:
        self.refresh_calls += 1


class _DummyController(SimpleNamespace):
    pass


class ControllerBatchPreflightTests(unittest.TestCase):
    def _build_controller(self) -> _DummyController:
        page_states = {
            "page-a.png": {"target_lang": "English", "source_lang": "Japanese"},
            "page-b.png": {"target_lang": "English", "source_lang": "Japanese"},
        }
        controller = _DummyController()
        controller.image_files = list(page_states.keys())
        controller.image_ctrl = _ImageCtrl(page_states)
        controller._start_batch_report = mock.Mock()
        controller._show_automatic_progress_dialog = mock.Mock()
        controller.batch_mode_selected = mock.Mock()
        controller.run_threaded = mock.Mock()
        controller.default_error_handler = mock.Mock()
        controller.on_batch_process_finished = mock.Mock()
        controller.pipeline = SimpleNamespace(batch_process=mock.Mock(), webtoon_batch_process=mock.Mock())
        controller.manual_radio = _Toggle(False)
        controller.automatic_radio = _Toggle(False)
        controller.translate_button = _WidgetState()
        controller.cancel_button = _WidgetState()
        controller.save_as_project_button = _WidgetState()
        controller.webtoon_toggle = _WidgetState()
        controller.progress_bar = _WidgetState()
        controller.batch_report_ctrl = _BatchReportCtrl()
        controller.webtoon_mode = False
        return controller

    def test_start_batch_process_reuses_one_preflight_cache_for_multiple_pages(self) -> None:
        controller = self._build_controller()
        captured_caches: list[dict[str, str]] = []

        def _validate(_main, _target_lang, *, source_lang=None, preflight_cache=None):
            self.assertEqual(source_lang, "Japanese")
            self.assertIsNotNone(preflight_cache)
            captured_caches.append(preflight_cache)
            return True

        with mock.patch("controller.validate_settings", side_effect=_validate):
            result = ComicTranslate._start_batch_process_for_paths(
                controller,
                ["page-a.png", "page-b.png"],
                run_type="batch",
            )

        self.assertTrue(result)
        self.assertEqual(len(captured_caches), 2)
        self.assertIs(captured_caches[0], captured_caches[1])

    def test_one_page_auto_process_uses_batch_entrypoint(self) -> None:
        controller = self._build_controller()
        controller._batch_active = False
        controller.curr_img_idx = 0
        controller._confirm_and_apply_auto_languages = mock.Mock(return_value=True)
        controller._start_batch_process_for_paths = mock.Mock(return_value=True)

        ComicTranslate.start_one_page_auto_process(controller)

        controller._confirm_and_apply_auto_languages.assert_called_once_with(["page-a.png"], "one_page_auto")
        controller._start_batch_process_for_paths.assert_called_once_with(["page-a.png"], run_type="one_page_auto")


if __name__ == "__main__":
    unittest.main()
