from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from modules.ocr.local_runtime import LocalOCRRuntimeManager
from modules.translation.local_runtime import LocalGemmaRuntimeManager
from modules.translation.gemma_runtime_contract import (
    DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
    DEFAULT_GEMMA_MODEL_VOLUME,
)
from modules.utils.local_llama_router import (
    DEFAULT_ROUTER_GEMMA_OPTIONS,
    DEFAULT_ROUTER_OCR_OPTIONS,
    LocalLlamaRouterCoordinator,
    ROUTER_GEMMA_ALIAS,
    ROUTER_GEMMA_MODEL_SHA256,
    RouterRuntimeError,
    _canonical_sha256,
    _model_states,
    _pair_catalog,
)
from pipeline.stage_batched_processor import StageBatchedProcessor


class _SettingsPage:
    def __init__(self, *, engine: str = "PaddleOCR VL") -> None:
        self.engine = engine

    def get_paddleocr_vl_settings(self) -> dict[str, str]:
        return {"server_url": "http://127.0.0.1:18000/v1/chat/completions"}

    def get_paddleocr_vl_spotting_settings(self) -> dict[str, str]:
        return {"server_url": "http://127.0.0.1:18002/v1/chat/completions"}

    def get_tool_selection(self, key: str) -> str:
        if key == "translator":
            return "Custom Local Server(Gemma)"
        return self.engine

    def get_credentials(self, provider: str) -> dict[str, str]:
        if provider != "Custom Local Server(Gemma)":
            return {}
        return {
            "api_url": "http://127.0.0.1:18080/v1",
            "model": ROUTER_GEMMA_ALIAS,
        }


class _FakeCoordinator:
    generation = 3

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.gemma_provider = None

    def register_gemma_identity_provider(self, provider) -> None:
        self.gemma_provider = provider

    def is_router_ocr_candidate(self, engine_key, settings_page) -> bool:
        return True

    def is_router_gemma_candidate(self, settings_page) -> bool:
        return True

    def ensure_ocr_model(self, engine_key, settings_page, identity, **kwargs):
        self.calls.append(("ensure_ocr", str(identity["model_name"])))
        return {"router": True, "model": str(identity["model_name"])}

    def ensure_gemma_model(self, settings_page, identity, **kwargs):
        self.calls.append(("ensure_gemma", str(identity["model_name"])))
        return {"router": True, "model": str(identity["model_name"])}

    def snapshot(self):
        return SimpleNamespace(
            prepared=True,
            container_running=True,
            active_model=ROUTER_GEMMA_ALIAS,
            loaded_count=1,
        )

    def unload_model(self, alias, **kwargs):
        self.calls.append(("unload", str(alias)))
        return {
            "runtime_state": "sleeping",
            "gpu_release_expected": True,
        }

    def stop_pair(self):
        self.calls.append(("stop", "pair"))
        return {"runtime_state": "stopped"}


def _ocr_identity(*, spotting: bool = False) -> dict[str, object]:
    pair = _pair_catalog()["paddle-spotting" if spotting else "paddle-crop"]
    return {
        "model_name": pair.ocr_alias,
        "model_sha256": pair.ocr_model_sha256,
        "mmproj_sha256": pair.ocr_mmproj_sha256,
        "manifest_sha256": "a" * 64,
        "volume": pair.ocr_volume,
        "image_ref": DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
        "image_id": "sha256:" + "1" * 64,
        "runtime_options": {
            **DEFAULT_ROUTER_OCR_OPTIONS,
            "PADDLEOCR_LLAMA_CTX_SIZE": "4096",
        },
    }


def _gemma_identity() -> dict[str, object]:
    return {
        "model_name": ROUTER_GEMMA_ALIAS,
        "model_sha256": ROUTER_GEMMA_MODEL_SHA256,
        "manifest_sha256": "b" * 64,
        "volume": DEFAULT_GEMMA_MODEL_VOLUME,
        "image_ref": DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
        "image_id": "sha256:" + "1" * 64,
        "runtime_options": {
            "LLAMA_CTX_SIZE": "4096",
            "LLAMA_N_PARALLEL": "1",
            "LLAMA_THREADS": "10",
            "LLAMA_BATCH_SIZE": "2048",
            "LLAMA_UBATCH_SIZE": "512",
            "LLAMA_N_GPU_LAYERS": "23",
            "LLAMA_CACHE_TYPE_K": "f16",
            "LLAMA_CACHE_TYPE_V": "f16",
            "LLAMA_CACHE_RAM_MIB": "0",
            "LLAMA_SPEC_TYPE": "none",
            "LLAMA_SPEC_DRAFT_N_MAX": "8",
        },
    }


class LocalLlamaRouterCoordinatorTests(unittest.TestCase):
    def test_pair_catalog_has_two_fixed_ports_and_no_autoload_assets(self) -> None:
        catalog = _pair_catalog()
        self.assertEqual(catalog["paddle-crop"].endpoint, "http://127.0.0.1:18000")
        self.assertEqual(catalog["paddle-spotting"].endpoint, "http://127.0.0.1:18002")
        for pair in catalog.values():
            self.assertTrue(pair.compose_file.is_file())
            preset = pair.preset_file.read_text(encoding="utf-8")
            compose = pair.compose_file.read_text(encoding="utf-8")
            self.assertIn("load-on-startup = false", preset)
            self.assertIn("--models-max", compose)
            self.assertIn("--no-models-autoload", compose)
            self.assertIn("read_only: true", compose)
            self.assertIn(":8080", compose)

    def test_fingerprint_changes_for_image_manifest_compose_and_command(self) -> None:
        coordinator = LocalLlamaRouterCoordinator()
        pair = _pair_catalog()["paddle-crop"]
        base = coordinator._build_contract(pair, _ocr_identity(), _gemma_identity())

        changed_image = _ocr_identity()
        changed_image_gemma = _gemma_identity()
        changed_image["image_id"] = "sha256:" + "2" * 64
        changed_image_gemma["image_id"] = "sha256:" + "2" * 64
        image_contract = coordinator._build_contract(
            pair,
            changed_image,
            changed_image_gemma,
        )
        self.assertNotEqual(base["fingerprint"], image_contract["fingerprint"])

        changed_manifest = _ocr_identity()
        changed_manifest["manifest_sha256"] = "c" * 64
        manifest_contract = coordinator._build_contract(
            pair,
            changed_manifest,
            _gemma_identity(),
        )
        self.assertNotEqual(base["fingerprint"], manifest_contract["fingerprint"])

        with tempfile.TemporaryDirectory() as temp_dir:
            compose_file = Path(temp_dir) / "compose.yaml"
            preset_file = Path(temp_dir) / "models.ini"
            compose_file.write_text(pair.compose_file.read_text(encoding="utf-8") + "\n# test", encoding="utf-8")
            preset_file.write_text(pair.preset_file.read_text(encoding="utf-8"), encoding="utf-8")
            changed_pair = pair.__class__(
                **{
                    **pair.__dict__,
                    "compose_file": compose_file,
                    "preset_file": preset_file,
                }
            )
            changed_contract = coordinator._build_contract(
                changed_pair,
                _ocr_identity(),
                _gemma_identity(),
            )
        self.assertNotEqual(base["fingerprint"], changed_contract["fingerprint"])
        self.assertEqual(base["command_sha256"], _canonical_sha256(base["command"]))
        with mock.patch(
            "modules.utils.local_llama_router._expected_command",
            return_value=("--host", "changed"),
        ):
            command_contract = coordinator._build_contract(
                pair,
                _ocr_identity(),
                _gemma_identity(),
            )
        self.assertNotEqual(base["fingerprint"], command_contract["fingerprint"])

    def test_model_state_parser_and_explicit_load_unload_are_exclusive(self) -> None:
        self.assertEqual(
            _model_states(
                {"models": {"ocr": {"status": "unloaded"}, "gemma": {"state": "loaded"}}}
            ),
            {"ocr": "unloaded", "gemma": "loaded"},
        )
        coordinator = LocalLlamaRouterCoordinator()
        coordinator._prepared = True
        coordinator._pair_key = "paddle-crop"
        coordinator._states = {
            "PaddleOCR-VL-1.6-0.9B": "unloaded",
            ROUTER_GEMMA_ALIAS: "unloaded",
        }
        states = dict(coordinator._states)

        def fake_http(url, *, payload=None, timeout_sec=30.0):
            del timeout_sec
            if "/v1/models?" in url:
                return 200, {"data": [{"id": "PaddleOCR-VL-1.6-0.9B"}]}
            if any(f"/{endpoint}?" in url for endpoint in ("props", "slots")):
                return 200, {"ok": True}
            if "/metrics?" in url:
                return 200, "# HELP llama_server_info 1\n"
            if url.endswith("/models"):
                return 200, {"data": [{"id": key, "state": value} for key, value in states.items()]}
            if url.endswith("/models/load"):
                states[str(payload["model"])] = "loaded"
                return 200, {"ok": True}
            if url.endswith("/models/unload"):
                states[str(payload["model"])] = "unloaded"
                return 200, {"ok": True}
            raise AssertionError(url)

        with mock.patch("modules.utils.local_llama_router._json_http", side_effect=fake_http):
            coordinator._load_model_locked(
                "PaddleOCR-VL-1.6-0.9B",
                cancel_checker=None,
            )
            with self.assertRaises(RouterRuntimeError):
                coordinator._load_model_locked(ROUTER_GEMMA_ALIAS, cancel_checker=None)
            report = coordinator.unload_model("PaddleOCR-VL-1.6-0.9B")

        self.assertEqual(report["runtime_state"], "sleeping")
        self.assertEqual(coordinator.snapshot().loaded_count, 0)
        self.assertIsNone(coordinator.snapshot().active_model)

    def test_no_autoload_probe_releases_command_lock_for_inference_http(self) -> None:
        coordinator = LocalLlamaRouterCoordinator()
        coordinator._prepared = True
        coordinator._pair_key = "paddle-crop"
        coordinator._generation = 1
        coordinator._states = {
            "PaddleOCR-VL-1.6-0.9B": "unloaded",
            ROUTER_GEMMA_ALIAS: "unloaded",
        }
        lock_owned: list[tuple[str, bool]] = []

        def fake_http(url, *, payload=None, timeout_sec=30.0):
            del payload, timeout_sec
            owned = bool(coordinator._command_lock._is_owned())
            if "chat/completions" in url:
                lock_owned.append(("inference", owned))
                return 400, {"error": "model is unloaded"}
            lock_owned.append(("models", owned))
            return 200, {
                "data": [
                    {"id": "PaddleOCR-VL-1.6-0.9B", "state": "unloaded"},
                    {"id": ROUTER_GEMMA_ALIAS, "state": "unloaded"},
                ]
            }

        with mock.patch("modules.utils.local_llama_router._json_http", side_effect=fake_http):
            coordinator._probe_no_autoload_if_needed(
                _pair_catalog()["paddle-crop"],
                cancel_checker=None,
            )

        self.assertEqual(lock_owned[0], ("inference", False))
        self.assertEqual(lock_owned[1], ("models", True))

    def test_release_failure_blocks_generation_and_next_model(self) -> None:
        coordinator = LocalLlamaRouterCoordinator()
        coordinator._prepared = True
        coordinator._pair_key = "paddle-crop"
        coordinator._active_model = "PaddleOCR-VL-1.6-0.9B"
        coordinator._states = {
            "PaddleOCR-VL-1.6-0.9B": "loaded",
            ROUTER_GEMMA_ALIAS: "unloaded",
        }

        def failed_unload(url, *, payload=None, timeout_sec=30.0):
            del payload, timeout_sec
            if url.endswith("/models"):
                return 200, {"data": [{"id": key, "state": value} for key, value in coordinator._states.items()]}
            return 500, {"error": "unload failed"}

        with mock.patch("modules.utils.local_llama_router._json_http", side_effect=failed_unload):
            with self.assertRaises(RouterRuntimeError):
                coordinator.unload_model("PaddleOCR-VL-1.6-0.9B")

        self.assertTrue(coordinator.snapshot().release_failed)
        self.assertEqual(coordinator.snapshot().active_model, "PaddleOCR-VL-1.6-0.9B")
        with self.assertRaises(RouterRuntimeError):
            coordinator.begin_generation()


class RouterManagerAndStageTests(unittest.TestCase):
    def test_managers_keep_separate_runtime_behavior_without_coordinator(self) -> None:
        self.assertIsNone(LocalOCRRuntimeManager()._coordinator)
        self.assertIsNone(LocalGemmaRuntimeManager()._coordinator)

    def test_ocr_manager_uses_router_and_preserves_state_on_release_failure(self) -> None:
        coordinator = _FakeCoordinator()
        manager = LocalOCRRuntimeManager(coordinator)
        identity = _ocr_identity()
        settings = _SettingsPage()
        with mock.patch.object(manager, "validate_engine"), mock.patch.object(
            manager,
            "_router_runtime_identity",
            return_value=identity,
        ):
            manager.ensure_engine("PaddleOCR VL", settings)
        self.assertEqual(manager._router_active_alias, "PaddleOCR-VL-1.6-0.9B")
        self.assertIn(("ensure_ocr", "PaddleOCR-VL-1.6-0.9B"), coordinator.calls)

        coordinator.unload_model = mock.Mock(
            side_effect=RouterRuntimeError("unload failed")
        )
        with self.assertRaises(Exception):
            manager.release_for_handoff()
        self.assertEqual(manager._router_active_alias, "PaddleOCR-VL-1.6-0.9B")

    def test_gemma_manager_uses_router_load_and_unload(self) -> None:
        coordinator = _FakeCoordinator()
        manager = LocalGemmaRuntimeManager(coordinator)
        settings = _SettingsPage()
        contract = SimpleNamespace(
            fingerprint="gemma-fingerprint",
            model_name=ROUTER_GEMMA_ALIAS,
            model_sha256=ROUTER_GEMMA_MODEL_SHA256,
            ready_manifest_sha256="b" * 64,
            volume_name=DEFAULT_GEMMA_MODEL_VOLUME,
            image_ref=DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
            image_id="sha256:" + "1" * 64,
            runtime_options=dict(DEFAULT_ROUTER_GEMMA_OPTIONS),
        )
        identity = {
            "model_name": ROUTER_GEMMA_ALIAS,
            "model_sha256": ROUTER_GEMMA_MODEL_SHA256,
            "manifest_sha256": "b" * 64,
            "volume": DEFAULT_GEMMA_MODEL_VOLUME,
            "image_ref": DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
            "image_id": "sha256:" + "1" * 64,
            "runtime_options": dict(DEFAULT_ROUTER_GEMMA_OPTIONS),
        }
        with mock.patch.object(manager, "validate_server"), mock.patch.object(
            manager,
            "_load_runtime_contract",
            return_value=contract,
        ), mock.patch.object(
            manager,
            "router_runtime_identity",
            return_value=identity,
        ), mock.patch.object(
            manager,
            "_validate_model_with_progress",
        ), mock.patch.object(
            manager,
            "_prewarm_chat_completion_with_progress",
        ):
            manager.ensure_server(settings)
            manager.shutdown()

        self.assertIn(("ensure_gemma", ROUTER_GEMMA_ALIAS), coordinator.calls)
        self.assertIn(("unload", ROUTER_GEMMA_ALIAS), coordinator.calls)

    def test_stage_resets_router_generation_and_finishes_pair_after_cleanup(self) -> None:
        coordinator = mock.Mock()
        coordinator.has_active_pair.return_value = True
        processor = object.__new__(StageBatchedProcessor)
        processor.main_page = SimpleNamespace(
            local_llama_router_coordinator=coordinator,
            local_ocr_runtime_manager=LocalOCRRuntimeManager(),
            local_translation_runtime_manager=LocalGemmaRuntimeManager(),
        )
        processor._prewarm_cancel_event = threading.Event()
        processor._runtime_progress_lock = threading.RLock()
        processor._runtime_progress_started = {}
        processor._runtime_resource_arbiter = mock.Mock(
            return_value=SimpleNamespace(reset=mock.Mock())
        )
        processor._reset_prewarm_lifecycle()
        coordinator.begin_generation.assert_called_once_with()

        with mock.patch.object(processor, "_shutdown_runtime_with_retry"):
            processor._shutdown_managed_runtimes(preserve_sleeping_paddle=True)
        coordinator.finish_pair.assert_called_once_with(keep_container=True)


if __name__ == "__main__":
    unittest.main()
