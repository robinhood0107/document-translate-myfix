from __future__ import annotations

import unittest
from unittest import mock

from modules.translation.local_runtime import LocalGemmaRuntimeManager
from modules.utils.exceptions import (
    LocalServiceConnectionError,
    LocalServiceResponseError,
    OperationCancelledError,
)


class _DummyGemmaSettingsPage:
    def __init__(
        self,
        *,
        api_url: str = "http://127.0.0.1:38080/v1",
        model: str = "gemma-test.gguf",
    ) -> None:
        self._api_url = api_url
        self._model = model

    def get_credentials(self, provider: str) -> dict:
        assert provider == "Custom Local Server(Gemma)"
        return {"api_url": self._api_url, "model": self._model}


class LocalGemmaRuntimeManagerTests(unittest.TestCase):
    def test_ensure_server_reuses_readiness_cache_for_same_endpoint_model(self) -> None:
        manager = LocalGemmaRuntimeManager()
        settings_page = _DummyGemmaSettingsPage()
        events: list[dict] = []

        with mock.patch.object(manager, "_wait_for_any_probe", return_value=True) as wait_for_probe, \
             mock.patch.object(manager, "_validate_model_with_progress") as validate_model:
            manager.ensure_server(settings_page, progress_callback=events.append)
            manager.ensure_server(settings_page, progress_callback=events.append)

        self.assertEqual(wait_for_probe.call_count, 1)
        self.assertEqual(validate_model.call_count, 1)
        self.assertTrue(any(event.get("readiness_cache_hit") for event in events))

    def test_model_mismatch_does_not_seed_readiness_cache(self) -> None:
        manager = LocalGemmaRuntimeManager()
        settings_page = _DummyGemmaSettingsPage()

        with mock.patch.object(manager, "_wait_for_any_probe", return_value=True) as wait_for_probe, \
             mock.patch.object(
                 manager,
                 "_validate_model_with_progress",
                 side_effect=[
                     LocalServiceResponseError(
                         "model mismatch",
                         service_name="Gemma",
                         settings_page_name="Gemma Local Server Settings",
                     ),
                     None,
                 ],
             ):
            with self.assertRaises(LocalServiceResponseError):
                manager.ensure_server(settings_page)
            manager.ensure_server(settings_page)

        self.assertEqual(wait_for_probe.call_count, 2)

    def test_connection_failure_does_not_seed_readiness_cache(self) -> None:
        manager = LocalGemmaRuntimeManager()
        settings_page = _DummyGemmaSettingsPage()

        with mock.patch.object(manager, "_wait_for_any_probe", side_effect=[False, True]) as wait_for_probe, \
             mock.patch.object(manager, "_validate_model_with_progress"):
            with self.assertRaises(LocalServiceConnectionError):
                manager.ensure_server(settings_page)
            manager.ensure_server(settings_page)

        self.assertEqual(wait_for_probe.call_count, 2)

    def test_cancelled_ensure_does_not_seed_readiness_cache(self) -> None:
        manager = LocalGemmaRuntimeManager()
        settings_page = _DummyGemmaSettingsPage()
        cancel = mock.Mock(side_effect=[True, False])

        with mock.patch.object(manager, "_wait_for_any_probe", return_value=True) as wait_for_probe, \
             mock.patch.object(manager, "_validate_model_with_progress"):
            with self.assertRaises(OperationCancelledError):
                manager.ensure_server(settings_page, cancel_checker=cancel)
            manager.ensure_server(settings_page, cancel_checker=cancel)

        self.assertEqual(wait_for_probe.call_count, 1)


if __name__ == "__main__":
    unittest.main()
