from __future__ import annotations

import json
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
             mock.patch.object(manager, "_validate_model_with_progress") as validate_model, \
             mock.patch.object(manager, "_prewarm_chat_completion_with_progress") as prewarm:
            manager.ensure_server(settings_page, progress_callback=events.append)
            manager.ensure_server(settings_page, progress_callback=events.append)

        self.assertEqual(wait_for_probe.call_count, 1)
        self.assertEqual(validate_model.call_count, 1)
        self.assertEqual(prewarm.call_count, 1)
        self.assertTrue(any(event.get("readiness_cache_hit") for event in events))

    def test_managed_model_mismatch_recreates_container_and_seeds_readiness_cache(self) -> None:
        manager = LocalGemmaRuntimeManager()
        settings_page = _DummyGemmaSettingsPage(api_url="http://127.0.0.1:18080/v1")
        events: list[dict] = []

        with mock.patch("modules.translation.local_runtime.Path.is_file", return_value=True), \
             mock.patch.object(manager, "_wait_for_any_probe", side_effect=[True, True]) as wait_for_probe, \
             mock.patch.object(manager, "_run_compose") as run_compose, \
             mock.patch.object(manager, "_prewarm_chat_completion_with_progress") as prewarm, \
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
            manager.ensure_server(settings_page, progress_callback=events.append)
            manager.ensure_server(settings_page, progress_callback=events.append)

        self.assertEqual(wait_for_probe.call_count, 2)
        self.assertEqual(prewarm.call_count, 1)
        run_compose.assert_called_once_with(
            "up",
            "-d",
            "--force-recreate",
            step_name="recreate",
            model_name="gemma-test.gguf",
        )
        self.assertTrue(any(event.get("step_key") == "compose_recreate" for event in events))
        self.assertTrue(any(event.get("readiness_cache_hit") for event in events))

    def test_connection_failure_does_not_seed_readiness_cache(self) -> None:
        manager = LocalGemmaRuntimeManager()
        settings_page = _DummyGemmaSettingsPage()

        with mock.patch.object(manager, "_wait_for_any_probe", side_effect=[False, True]) as wait_for_probe, \
             mock.patch.object(manager, "_validate_model_with_progress"), \
             mock.patch.object(manager, "_prewarm_chat_completion_with_progress"):
            with self.assertRaises(LocalServiceConnectionError):
                manager.ensure_server(settings_page)
            manager.ensure_server(settings_page)

        self.assertEqual(wait_for_probe.call_count, 2)

    def test_cancelled_ensure_does_not_seed_readiness_cache(self) -> None:
        manager = LocalGemmaRuntimeManager()
        settings_page = _DummyGemmaSettingsPage()
        cancel = mock.Mock(side_effect=[True, False])

        with mock.patch.object(manager, "_wait_for_any_probe", return_value=True) as wait_for_probe, \
             mock.patch.object(manager, "_validate_model_with_progress"), \
             mock.patch.object(manager, "_prewarm_chat_completion_with_progress"):
            with self.assertRaises(OperationCancelledError):
                manager.ensure_server(settings_page, cancel_checker=cancel)
            manager.ensure_server(settings_page, cancel_checker=cancel)

        self.assertEqual(wait_for_probe.call_count, 1)

    def test_validate_loaded_model_accepts_model_id_basename_match(self) -> None:
        manager = LocalGemmaRuntimeManager()

        class _Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(
                    {"data": [{"id": "/models/gemma-4-26B-IQ4_NL.gguf"}]}
                ).encode("utf-8")

        with mock.patch("modules.translation.local_runtime.urlopen", return_value=_Response()):
            manager._validate_loaded_model(
                "http://127.0.0.1:18080/v1",
                "gemma-4-26B-IQ4_NL.gguf",
            )

    def test_managed_runtime_prewarms_chat_completion_after_model_validation(self) -> None:
        manager = LocalGemmaRuntimeManager()
        settings_page = _DummyGemmaSettingsPage(api_url="http://127.0.0.1:18080/v1")
        events: list[dict] = []

        with mock.patch("modules.translation.local_runtime.Path.is_file", return_value=True), \
             mock.patch.object(manager, "_wait_for_any_probe", return_value=True), \
             mock.patch.object(manager, "_run_compose"), \
             mock.patch.object(manager, "_validate_model_with_progress"), \
             mock.patch.object(manager, "_prewarm_chat_completion") as prewarm:
            manager.ensure_server(settings_page, progress_callback=events.append)

        prewarm.assert_called_once()
        self.assertTrue(any(event.get("step_key") == "chat_prewarm" for event in events))


if __name__ == "__main__":
    unittest.main()
