from __future__ import annotations

import base64
import json
import subprocess
import unittest
from types import SimpleNamespace
from unittest import mock

from modules.translation.gemma_runtime_contract import (
    DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
    DEFAULT_GEMMA_MODEL_VOLUME,
)
from modules.translation.local_runtime import LocalGemmaRuntimeManager
from modules.utils.exceptions import (
    LocalServiceConnectionError,
    LocalServiceResponseError,
    LocalServiceSetupError,
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


def _runtime_contract(fingerprint: str = "fingerprint-a") -> SimpleNamespace:
    compose_environment = {
        "LLAMA_CPP_IMAGE": DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
        "LLAMA_CPP_PULL_POLICY": "missing",
        "LLAMA_MODEL_FILE": "gemma-test.gguf",
        "GEMMA_MODEL_VOLUME": DEFAULT_GEMMA_MODEL_VOLUME,
        "GEMMA_RUNTIME_FINGERPRINT": fingerprint,
        "GEMMA_READY_MANIFEST_SHA256": "a" * 64,
        "GEMMA_MODEL_SHA256": "b" * 64,
        "GEMMA_PREPARATION_VERSION": "1",
    }
    return SimpleNamespace(
        fingerprint=fingerprint,
        image_ref=DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
        image_id="sha256:" + ("1" * 64),
        volume_name=DEFAULT_GEMMA_MODEL_VOLUME,
        model_name="gemma-test.gguf",
        model_sha256="b" * 64,
        ready_manifest_sha256="a" * 64,
        preparation_version=1,
        command=("-m", "/models/gemma-test.gguf"),
        compose_environment=lambda: dict(compose_environment),
    )


def _container_state(
    *,
    exists: bool = True,
    running: bool = True,
    matches: bool = True,
) -> dict:
    return {
        "exists": exists,
        "running": running,
        "matches": matches,
        "mismatch_reasons": [] if matches else ["command"],
    }


class LocalGemmaRuntimeManagerTests(unittest.TestCase):
    def test_shutdown_stops_and_preserves_managed_container(self) -> None:
        manager = LocalGemmaRuntimeManager()
        manager._managed_active = True
        manager._active_contract = _runtime_contract()
        manager._readiness_cache.add(
            (
                "http://127.0.0.1:18080/v1",
                "gemma-test.gguf",
                "managed",
                "fingerprint-a",
            )
        )

        with mock.patch.object(manager, "_stop_managed_container") as stop:
            manager.shutdown()

        stop.assert_called_once_with()
        self.assertFalse(manager._managed_active)
        self.assertIsNone(manager._active_contract)
        self.assertFalse(manager._readiness_cache)

    def test_shutdown_clears_managed_state_when_stop_fails(self) -> None:
        manager = LocalGemmaRuntimeManager()
        manager._managed_active = True
        manager._active_contract = _runtime_contract()
        failure = LocalServiceSetupError(
            "stop failed",
            service_name="Gemma",
            settings_page_name="Gemma Local Server Settings",
        )

        with mock.patch.object(
            manager,
            "_stop_managed_container",
            side_effect=failure,
        ), self.assertLogs(
            "modules.translation.local_runtime",
            level="WARNING",
        ):
            manager.shutdown()

        self.assertFalse(manager._managed_active)
        self.assertIsNone(manager._active_contract)
        self.assertFalse(manager._readiness_cache)

    def test_cancelled_startup_can_stop_container_started_by_compose(self) -> None:
        manager = LocalGemmaRuntimeManager()
        settings_page = _DummyGemmaSettingsPage(
            api_url="http://127.0.0.1:18080/v1"
        )
        contract = _runtime_contract()
        cancelled = OperationCancelledError("cancelled during health wait")

        with mock.patch.object(manager, "validate_server"), \
             mock.patch.object(manager, "_load_runtime_contract", return_value=contract), \
             mock.patch.object(
                 manager,
                 "_inspect_managed_container_state",
                 return_value=_container_state(exists=False, running=False, matches=False),
             ), \
             mock.patch.object(manager, "_run_compose") as run_compose, \
             mock.patch.object(manager, "_assert_managed_container_contract"), \
             mock.patch.object(
                 manager,
                 "_wait_for_any_probe",
                 side_effect=cancelled,
             ), \
             mock.patch.object(manager, "_stop_managed_container") as stop:
            with self.assertRaises(OperationCancelledError):
                manager.ensure_server(settings_page)
            manager.shutdown()

        run_compose.assert_called_once_with(
            "up",
            "-d",
            step_name="up",
            runtime_contract=contract,
        )
        stop.assert_called_once_with()

    def test_validate_loaded_model_accepts_model_id_basename_match(self) -> None:
        manager = LocalGemmaRuntimeManager()
        payload = {"data": [{"id": "/models/gemma-test.gguf"}]}
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(payload).encode(
            "utf-8"
        )

        with mock.patch(
            "modules.translation.local_runtime.urlopen",
            return_value=response,
        ):
            manager._validate_loaded_model(
                "http://127.0.0.1:18080/v1",
                "gemma-test.gguf",
            )

    def test_validate_loaded_model_rejects_different_model(self) -> None:
        manager = LocalGemmaRuntimeManager()
        payload = {"data": [{"id": "/models/other.gguf"}]}
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(payload).encode(
            "utf-8"
        )

        with mock.patch(
            "modules.translation.local_runtime.urlopen",
            return_value=response,
        ), self.assertRaises(LocalServiceResponseError):
            manager._validate_loaded_model(
                "http://127.0.0.1:18080/v1",
                "gemma-test.gguf",
            )

    def test_unmanaged_endpoint_reuses_readiness_cache(self) -> None:
        manager = LocalGemmaRuntimeManager()
        settings_page = _DummyGemmaSettingsPage()
        events: list[dict] = []

        with mock.patch.object(
            manager,
            "_wait_for_any_probe",
            return_value=True,
        ) as wait_for_probe, mock.patch.object(
            manager,
            "_validate_model_with_progress",
        ) as validate_model, mock.patch.object(
            manager,
            "_prewarm_chat_completion_with_progress",
        ) as prewarm:
            manager.ensure_server(settings_page, progress_callback=events.append)
            manager.ensure_server(settings_page, progress_callback=events.append)

        self.assertEqual(wait_for_probe.call_count, 1)
        self.assertEqual(validate_model.call_count, 1)
        self.assertEqual(prewarm.call_count, 1)
        self.assertTrue(any(event.get("readiness_cache_hit") for event in events))

    def test_managed_running_container_reuses_exact_contract_and_cache(self) -> None:
        manager = LocalGemmaRuntimeManager()
        settings_page = _DummyGemmaSettingsPage(
            api_url="http://127.0.0.1:18080/v1"
        )
        contract = _runtime_contract()
        events: list[dict] = []

        with mock.patch.object(manager, "validate_server"), \
             mock.patch.object(manager, "_load_runtime_contract", return_value=contract), \
             mock.patch.object(
                 manager,
                 "_inspect_managed_container_state",
                 return_value=_container_state(),
             ) as inspect_state, \
             mock.patch.object(manager, "_probe_url", return_value=True), \
             mock.patch.object(manager, "_wait_for_any_probe", return_value=True), \
             mock.patch.object(manager, "_validate_model_with_progress"), \
             mock.patch.object(manager, "_prewarm_chat_completion_with_progress"), \
             mock.patch.object(manager, "_log_runtime_metadata"), \
             mock.patch.object(manager, "_run_compose") as run_compose, \
             mock.patch.object(manager, "_start_managed_container") as start:
            manager.ensure_server(settings_page, progress_callback=events.append)
            manager.ensure_server(settings_page, progress_callback=events.append)

        self.assertEqual(inspect_state.call_count, 2)
        run_compose.assert_not_called()
        start.assert_not_called()
        self.assertTrue(any(event.get("readiness_cache_hit") for event in events))

    def test_exact_stopped_container_uses_direct_docker_start(self) -> None:
        manager = LocalGemmaRuntimeManager()
        settings_page = _DummyGemmaSettingsPage(
            api_url="http://127.0.0.1:18080/v1"
        )
        contract = _runtime_contract()

        with mock.patch.object(manager, "validate_server"), \
             mock.patch.object(manager, "_load_runtime_contract", return_value=contract), \
             mock.patch.object(
                 manager,
                 "_inspect_managed_container_state",
                 return_value=_container_state(running=False),
             ), \
             mock.patch.object(manager, "_start_managed_container") as start, \
             mock.patch.object(manager, "_run_compose") as run_compose, \
             mock.patch.object(manager, "_wait_for_any_probe", return_value=True), \
             mock.patch.object(manager, "_validate_model_with_progress"), \
             mock.patch.object(manager, "_prewarm_chat_completion_with_progress"), \
             mock.patch.object(manager, "_log_runtime_metadata"):
            manager.ensure_server(settings_page)

        start.assert_called_once_with()
        run_compose.assert_not_called()

    def test_fingerprint_mismatch_force_recreates_container(self) -> None:
        manager = LocalGemmaRuntimeManager()
        settings_page = _DummyGemmaSettingsPage(
            api_url="http://127.0.0.1:18080/v1"
        )
        contract = _runtime_contract()
        events: list[dict] = []

        with mock.patch.object(manager, "validate_server"), \
             mock.patch.object(manager, "_load_runtime_contract", return_value=contract), \
             mock.patch.object(
                 manager,
                 "_inspect_managed_container_state",
                 return_value=_container_state(matches=False),
             ), \
             mock.patch.object(manager, "_run_compose") as run_compose, \
             mock.patch.object(manager, "_assert_managed_container_contract"), \
             mock.patch.object(manager, "_wait_for_any_probe", return_value=True), \
             mock.patch.object(manager, "_validate_model_with_progress"), \
             mock.patch.object(manager, "_prewarm_chat_completion_with_progress"), \
             mock.patch.object(manager, "_log_runtime_metadata"):
            manager.ensure_server(settings_page, progress_callback=events.append)

        run_compose.assert_called_once_with(
            "up",
            "-d",
            "--force-recreate",
            step_name="recreate",
            runtime_contract=contract,
        )
        self.assertTrue(
            any(event.get("step_key") == "compose_recreate" for event in events)
        )

    def test_cache_rechecks_container_and_restarts_if_stopped_externally(self) -> None:
        manager = LocalGemmaRuntimeManager()
        settings_page = _DummyGemmaSettingsPage(
            api_url="http://127.0.0.1:18080/v1"
        )
        contract = _runtime_contract()

        with mock.patch.object(manager, "validate_server"), \
             mock.patch.object(manager, "_load_runtime_contract", return_value=contract), \
             mock.patch.object(
                 manager,
                 "_inspect_managed_container_state",
                 side_effect=[
                     _container_state(),
                     _container_state(running=False),
                     _container_state(running=False),
                 ],
             ), \
             mock.patch.object(manager, "_probe_url", return_value=False), \
             mock.patch.object(manager, "_start_managed_container") as start, \
             mock.patch.object(manager, "_wait_for_any_probe", return_value=True) as wait, \
             mock.patch.object(manager, "_validate_model_with_progress"), \
             mock.patch.object(manager, "_prewarm_chat_completion_with_progress"), \
             mock.patch.object(manager, "_log_runtime_metadata"):
            manager.ensure_server(settings_page)
            manager.ensure_server(settings_page)

        start.assert_called_once_with()
        self.assertEqual(wait.call_count, 2)

    def test_connection_failure_does_not_seed_readiness_cache(self) -> None:
        manager = LocalGemmaRuntimeManager()
        settings_page = _DummyGemmaSettingsPage()

        with mock.patch.object(
            manager,
            "_wait_for_any_probe",
            side_effect=[False, True],
        ) as wait_for_probe, mock.patch.object(
            manager,
            "_validate_model_with_progress",
        ), mock.patch.object(
            manager,
            "_prewarm_chat_completion_with_progress",
        ):
            with self.assertRaises(LocalServiceConnectionError):
                manager.ensure_server(settings_page)
            manager.ensure_server(settings_page)

        self.assertEqual(wait_for_probe.call_count, 2)

    def test_managed_health_timeout_can_retry_without_cached_readiness(self) -> None:
        manager = LocalGemmaRuntimeManager()
        settings_page = _DummyGemmaSettingsPage(
            api_url="http://127.0.0.1:18080/v1"
        )
        contract = _runtime_contract()

        with mock.patch.object(manager, "validate_server"), \
             mock.patch.object(
                 manager,
                 "_load_runtime_contract",
                 return_value=contract,
             ), \
             mock.patch.object(
                 manager,
                 "_inspect_managed_container_state",
                 return_value=_container_state(),
             ), \
             mock.patch.object(
                 manager,
                 "_wait_for_any_probe",
                 side_effect=[False, True],
             ) as wait_for_probe, \
             mock.patch.object(manager, "_validate_model_with_progress"), \
             mock.patch.object(
                 manager,
                 "_prewarm_chat_completion_with_progress",
             ), \
             mock.patch.object(manager, "_log_runtime_metadata"):
            with self.assertRaisesRegex(
                LocalServiceSetupError,
                "Timed out while waiting for Gemma",
            ):
                manager.ensure_server(settings_page)
            self.assertFalse(manager._readiness_cache)

            manager.ensure_server(settings_page)

        self.assertEqual(wait_for_probe.call_count, 2)
        self.assertEqual(len(manager._readiness_cache), 1)

    def test_initial_cancellation_avoids_docker_runtime_probe(self) -> None:
        manager = LocalGemmaRuntimeManager()
        settings_page = _DummyGemmaSettingsPage(
            api_url="http://127.0.0.1:18080/v1"
        )

        with mock.patch.object(manager, "_load_runtime_contract") as load_contract:
            with self.assertRaises(OperationCancelledError):
                manager.ensure_server(
                    settings_page,
                    cancel_checker=lambda: True,
                )

        load_contract.assert_not_called()

    def test_managed_runtime_prewarms_after_model_validation(self) -> None:
        manager = LocalGemmaRuntimeManager()
        settings_page = _DummyGemmaSettingsPage(
            api_url="http://127.0.0.1:18080/v1"
        )
        contract = _runtime_contract()
        events: list[dict] = []

        with mock.patch.object(manager, "validate_server"), \
             mock.patch.object(manager, "_load_runtime_contract", return_value=contract), \
             mock.patch.object(
                 manager,
                 "_inspect_managed_container_state",
                 return_value=_container_state(),
             ), \
             mock.patch.object(manager, "_wait_for_any_probe", return_value=True), \
             mock.patch.object(manager, "_validate_model_with_progress"), \
             mock.patch.object(manager, "_prewarm_chat_completion") as prewarm, \
             mock.patch.object(manager, "_log_runtime_metadata"):
            manager.ensure_server(
                settings_page,
                progress_callback=events.append,
            )

        prewarm.assert_called_once()
        self.assertTrue(
            any(event.get("step_key") == "chat_prewarm" for event in events)
        )

    def test_build_env_uses_exact_runtime_contract(self) -> None:
        manager = LocalGemmaRuntimeManager()
        contract = _runtime_contract()
        env = manager._build_env(runtime_contract=contract)

        self.assertEqual(env["LLAMA_CPP_IMAGE"], DEFAULT_GEMMA_LLAMA_CPP_IMAGE)
        self.assertEqual(env["LLAMA_CPP_PULL_POLICY"], "missing")
        self.assertEqual(env["GEMMA_MODEL_VOLUME"], DEFAULT_GEMMA_MODEL_VOLUME)
        self.assertEqual(env["GEMMA_RUNTIME_FINGERPRINT"], "fingerprint-a")

    def test_load_contract_rejects_model_paths_before_docker_probe(self) -> None:
        manager = LocalGemmaRuntimeManager()

        with mock.patch.object(manager, "_ensure_runtime_image_id") as ensure_image:
            with self.assertRaisesRegex(
                LocalServiceSetupError,
                "Invalid Gemma model filename",
            ):
                manager._load_runtime_contract("../gemma-test.gguf")

        ensure_image.assert_not_called()

    def test_runtime_image_is_pulled_only_when_missing(self) -> None:
        manager = LocalGemmaRuntimeManager()
        present = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="sha256:present\n",
            stderr="",
        )
        missing = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="missing",
        )

        with mock.patch(
            "modules.translation.local_runtime.run_docker_command",
            return_value=present,
        ) as run:
            self.assertEqual(
                manager._ensure_runtime_image_id(DEFAULT_GEMMA_LLAMA_CPP_IMAGE),
                "sha256:present",
            )
        self.assertEqual(run.call_count, 1)

        with mock.patch(
            "modules.translation.local_runtime.run_docker_command",
            side_effect=[missing, present, present],
        ) as run:
            self.assertEqual(
                manager._ensure_runtime_image_id(DEFAULT_GEMMA_LLAMA_CPP_IMAGE),
                "sha256:present",
            )
        self.assertEqual(run.call_count, 3)
        self.assertEqual(run.call_args_list[1].args[0][1], "pull")

    def test_runtime_image_pull_failure_is_reported_as_setup_error(self) -> None:
        manager = LocalGemmaRuntimeManager()
        missing = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="missing",
        )

        with mock.patch(
            "modules.translation.local_runtime.run_docker_command",
            side_effect=[missing, RuntimeError("registry unavailable")],
        ), self.assertRaisesRegex(
            LocalServiceSetupError,
            "Unable to load the pinned Gemma runtime image",
        ):
            manager._ensure_runtime_image_id(DEFAULT_GEMMA_LLAMA_CPP_IMAGE)

    def test_runtime_image_empty_id_after_pull_is_rejected(self) -> None:
        manager = LocalGemmaRuntimeManager()
        missing = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="missing",
        )
        pull = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="pulled",
            stderr="",
        )
        empty_inspect = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="",
        )

        with mock.patch(
            "modules.translation.local_runtime.run_docker_command",
            side_effect=[missing, pull, empty_inspect],
        ), self.assertRaisesRegex(
            LocalServiceSetupError,
            "Docker returned no image ID",
        ):
            manager._ensure_runtime_image_id(DEFAULT_GEMMA_LLAMA_CPP_IMAGE)

    def test_volume_probe_reads_only_manifest_hash_and_model_size(self) -> None:
        manager = LocalGemmaRuntimeManager()
        manifest = b'{"ready":true}'
        stdout = (
            "manifest_sha256=" + ("a" * 64) + "\n"
            "manifest_base64="
            + base64.b64encode(manifest).decode("ascii")
            + "\nmodel_bytes=14585439872\n"
        )
        volume_inspection = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                '{"comic-translate.runtime":"Gemma",'
                '"comic-translate.preparation-version":"1"}'
            ),
            stderr="",
        )
        probe = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=stdout,
            stderr="",
        )

        with mock.patch(
            "modules.translation.local_runtime.run_docker_command",
            side_effect=[volume_inspection, probe],
        ) as run:
            result = manager._probe_model_volume(
                volume_name=DEFAULT_GEMMA_MODEL_VOLUME,
                model_name="gemma-test.gguf",
                image_ref=DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
            )

        self.assertEqual(result, (manifest, "a" * 64, 14_585_439_872))
        command_text = " ".join(run.call_args_list[1].args[0])
        self.assertIn("sha256sum \"$manifest_path\"", command_text)
        self.assertNotIn("sha256sum \"$model_path\"", command_text)

    def test_missing_volume_is_rejected_without_docker_run(self) -> None:
        manager = LocalGemmaRuntimeManager()
        missing = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="No such volume",
        )

        with mock.patch(
            "modules.translation.local_runtime.run_docker_command",
            return_value=missing,
        ) as run, self.assertRaisesRegex(
            LocalServiceSetupError,
            "does not exist",
        ):
            manager._probe_model_volume(
                volume_name=DEFAULT_GEMMA_MODEL_VOLUME,
                model_name="gemma-test.gguf",
                image_ref=DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
            )

        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][1:3], ["volume", "inspect"])

    def test_volume_probe_rejects_malformed_labels(self) -> None:
        manager = LocalGemmaRuntimeManager()
        malformed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="{not-json",
            stderr="",
        )

        with mock.patch(
            "modules.translation.local_runtime.run_docker_command",
            return_value=malformed,
        ) as run, self.assertRaisesRegex(
            LocalServiceSetupError,
            "Unable to parse Docker labels",
        ):
            manager._probe_model_volume(
                volume_name=DEFAULT_GEMMA_MODEL_VOLUME,
                model_name="gemma-test.gguf",
                image_ref=DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
            )

        self.assertEqual(run.call_count, 1)

    def test_volume_probe_rejects_wrong_labels(self) -> None:
        manager = LocalGemmaRuntimeManager()
        wrong_labels = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"comic-translate.runtime":"Other"}',
            stderr="",
        )

        with mock.patch(
            "modules.translation.local_runtime.run_docker_command",
            return_value=wrong_labels,
        ) as run, self.assertRaisesRegex(
            LocalServiceSetupError,
            "volume labels do not match",
        ):
            manager._probe_model_volume(
                volume_name=DEFAULT_GEMMA_MODEL_VOLUME,
                model_name="gemma-test.gguf",
                image_ref=DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
            )

        self.assertEqual(run.call_count, 1)

    def test_volume_probe_rejects_malformed_probe_output(self) -> None:
        manager = LocalGemmaRuntimeManager()
        volume_inspection = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                '{"comic-translate.runtime":"Gemma",'
                '"comic-translate.preparation-version":"1"}'
            ),
            stderr="",
        )
        malformed_probe = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="manifest_sha256=not-enough-fields\n",
            stderr="",
        )

        with mock.patch(
            "modules.translation.local_runtime.run_docker_command",
            side_effect=[volume_inspection, malformed_probe],
        ), self.assertRaisesRegex(
            LocalServiceSetupError,
            "Unable to parse the prepared Gemma volume probe output",
        ):
            manager._probe_model_volume(
                volume_name=DEFAULT_GEMMA_MODEL_VOLUME,
                model_name="gemma-test.gguf",
                image_ref=DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
            )

    def test_compose_failure_is_reported_with_requested_image(self) -> None:
        manager = LocalGemmaRuntimeManager()
        contract = _runtime_contract()

        with mock.patch.object(
            manager,
            "_resolve_compose_command",
            return_value=["docker", "compose"],
        ), mock.patch(
            "modules.translation.local_runtime.run_docker_command",
            side_effect=RuntimeError("compose failed"),
        ), self.assertRaisesRegex(
            LocalServiceSetupError,
            "Requested image:",
        ):
            manager._run_compose(
                "up",
                "-d",
                step_name="up",
                runtime_contract=contract,
            )

    def test_post_create_contract_mismatch_fails_startup(self) -> None:
        manager = LocalGemmaRuntimeManager()
        settings_page = _DummyGemmaSettingsPage(
            api_url="http://127.0.0.1:18080/v1"
        )
        contract = _runtime_contract()
        mismatch = LocalServiceSetupError(
            "post-create mismatch",
            service_name="Gemma",
            settings_page_name="Gemma Local Server Settings",
        )

        with mock.patch.object(manager, "validate_server"), \
             mock.patch.object(
                 manager,
                 "_load_runtime_contract",
                 return_value=contract,
             ), \
             mock.patch.object(
                 manager,
                 "_inspect_managed_container_state",
                 return_value=_container_state(
                     exists=False,
                     running=False,
                     matches=False,
                 ),
             ), \
             mock.patch.object(manager, "_run_compose") as run_compose, \
             mock.patch.object(
                 manager,
                 "_assert_managed_container_contract",
                 side_effect=mismatch,
             ):
            with self.assertRaisesRegex(
                LocalServiceSetupError,
                "post-create mismatch",
            ):
                manager.ensure_server(settings_page)

        run_compose.assert_called_once()
        self.assertFalse(manager._readiness_cache)


if __name__ == "__main__":
    unittest.main()
