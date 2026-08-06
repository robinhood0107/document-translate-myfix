from __future__ import annotations

import copy
import errno
import threading
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest import mock

import requests

from modules.ocr.local_runtime import LocalOCRRuntimeManager
from modules.translation.local_runtime import LocalGemmaRuntimeManager
from modules.utils.exceptions import OperationCancelledError
from modules.utils.local_llama_router.adapter import (
    DockerRouterCommandAdapter,
    RouterAdapterError,
    RouterAdapterOwnershipError,
    RouterContainerInspection,
    RouterModelSnapshot,
)
from modules.utils.local_llama_router.contracts import (
    DEFAULT_CROP_ROUTER_ENDPOINT,
    DEFAULT_GEMMA_ROUTER_ENDPOINT,
    DEFAULT_GEMMA_ROUTER_MODEL,
    DEFAULT_SPOTTING_ROUTER_ENDPOINT,
    ROUTER_OWNER_LABEL,
    ROUTER_OWNER_VALUE,
    ROUTER_PAIR_LABEL,
    RouterModelMaterial,
    RouterRuntimeSpec,
    build_router_contract,
    classify_router_pair,
    exact_endpoint_matches,
)
from modules.utils.local_llama_router.coordinator import (
    LocalLlamaRouterCoordinator,
    RouterReleaseError,
    RouterSetupError,
    RouterState,
)
from pipeline.runtime_resource_arbiter import RuntimeLeaseConflictError, RuntimeResourceArbiter


def _gpu_snapshot(
    used_mb: float,
    process_ids: set[int],
    *,
    router_worker_ids: set[int] | None = None,
    router_worker_alias: str = "PaddleOCR-VL-1.6-0.9B",
) -> dict[str, Any]:
    gpu_uuid = "GPU-router-test"
    payload = {
        "driver": {
            "available": True,
            "gpus": [{"uuid": gpu_uuid, "memory_used_mb": used_mb}],
            "primary": {"uuid": gpu_uuid, "memory_used_mb": used_mb},
        },
        "driver_processes": {
            "query_available": True,
            "rows": [
                {
                    "pid": process_id,
                    "gpu_uuid": gpu_uuid,
                    "process_name": "llama-server",
                    "memory_used_mb": used_mb,
                }
                for process_id in sorted(process_ids)
            ],
        },
    }
    if router_worker_ids is not None:
        payload["router_worker_processes"] = {
            "query_available": True,
            "rows": [
                {
                    "pid": process_id,
                    "gpu_uuid": gpu_uuid,
                    "model_alias": router_worker_alias,
                    "gpu_device_attached": True,
                }
                for process_id in sorted(router_worker_ids)
            ],
        }
    return payload


class _SequenceSampler:
    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self._samples = list(samples)
        self._last = self._samples[-1] if self._samples else _gpu_snapshot(0, set())

    def __call__(self) -> dict[str, Any]:
        if self._samples:
            self._last = self._samples.pop(0)
        return self._last


def _spec():
    pair = classify_router_pair(
        "PaddleOCR VL",
        DEFAULT_CROP_ROUTER_ENDPOINT,
        DEFAULT_GEMMA_ROUTER_ENDPOINT,
        DEFAULT_GEMMA_ROUTER_MODEL,
    )
    assert pair is not None
    ocr = RouterModelMaterial(
        alias=pair.ocr_alias,
        model_file="PaddleOCR-VL-1.6-GGUF.gguf",
        model_sha256="ocr-sha",
        mmproj_file="PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
        mmproj_sha256="mmproj-sha",
        volume_name="ocr-volume",
        ready_manifest_sha256="ocr-manifest",
        source_fingerprint="ocr-source",
        runtime_options={"ctx-size": "4096"},
        preparation_version=1,
    )
    gemma = RouterModelMaterial(
        alias=DEFAULT_GEMMA_ROUTER_MODEL,
        model_file=DEFAULT_GEMMA_ROUTER_MODEL,
        model_sha256="gemma-sha",
        volume_name="gemma-volume",
        ready_manifest_sha256="gemma-manifest",
        source_fingerprint="gemma-source",
        runtime_options={"ctx-size": "4096"},
        preparation_version=2,
    )
    return RouterRuntimeSpec(pair=pair, ocr_model=ocr, gemma_model=gemma)


def _contract(spec: RouterRuntimeSpec):
    return build_router_contract(
        spec=spec,
        image_id="sha256:image-id",
        repo_digest="ghcr.io/ggml-org/llama.cpp@sha256:image-digest",
        entrypoint=("/app/llama-server",),
        binary_version="version: b10133",
        resolved_compose_config={
            "services": {"llama-router": {"ports": [18000, 18080]}},
        },
        effective_environment={"NVIDIA_VISIBLE_DEVICES": "all"},
        port_mapping={"ocr": 18000, "gemma": 18080},
        volume_mapping={"ocr": "ocr-volume", "gemma": "gemma-volume"},
        device_mapping={"gpus": "all"},
        server_args=(
            "--models-preset",
            "/router/models.ini",
            "--models-max",
            "1",
            "--no-models-autoload",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
            "--metrics",
        ),
        preset_sha256="preset-sha",
    )


class _FakeAdapter:
    def __init__(self, contract) -> None:
        self.contract = contract
        self.loaded_model = ""
        self.container_running = False
        self.load_calls: list[str] = []
        self.unload_calls: list[str] = []
        self.stop_calls = 0
        self.implicit_autoload_checks = 0

    def build_contract(self, spec, **_kwargs):
        assert spec.pair == self.contract.pair
        return self.contract

    def prepare(self, contract, **_kwargs):
        self.container_running = True
        return RouterContainerInspection(
            name=contract.pair.container_name,
            exists=True,
            running=True,
            image=contract.image_ref,
            image_id=contract.image_id,
            labels=dict(contract.ownership_labels),
            command=contract.server_args,
            entrypoint=contract.entrypoint,
            ports={},
            mounts=(),
            device_requests=(),
            pid=100,
        )

    def model_snapshot(self, _pair, **_kwargs):
        loaded = (self.loaded_model,) if self.loaded_model else ()
        return RouterModelSnapshot(
            loaded_models=loaded,
            loaded_count=len(loaded),
            slots_idle=True,
            slots=({"is_processing": False},),
            raw_models={},
            raw_slots={},
        )

    def load_model(self, _pair, model_alias, **_kwargs):
        if self.loaded_model:
            raise RuntimeError("second model load")
        self.loaded_model = model_alias
        self.load_calls.append(model_alias)

    def unload_model(self, _pair, model_alias, **_kwargs):
        if self.loaded_model != model_alias:
            raise RuntimeError("wrong unload")
        self.loaded_model = ""
        self.unload_calls.append(model_alias)

    def assert_implicit_autoload_rejected(self, _pair, _model_alias):
        self.implicit_autoload_checks += 1

    def stop_pair(self, _contract, **_kwargs):
        self.container_running = False
        self.stop_calls += 1

    def stop_owned_port_occupants(self, _contract, **_kwargs):
        return None

    def owned_gpu_process_ids(self, _contract):
        return frozenset({100}) if self.container_running else frozenset()


class _TransientSlotsFakeAdapter(_FakeAdapter):
    """Model API has completed, but its just-started proxy is not ready yet."""

    def __init__(self, contract) -> None:
        super().__init__(contract)
        self._raised_transient_slots_error = False

    def model_snapshot(self, pair, **kwargs):
        if self.loaded_model and not self._raised_transient_slots_error:
            self._raised_transient_slots_error = True
            raise RouterAdapterError(
                "Router request failed: GET "
                f"{pair.router_base_url}/slots?model={self.loaded_model} "
                "HTTP 500: proxy error: Could not establish connection"
            )
        return super().model_snapshot(pair, **kwargs)


class _RouterSettingsPage:
    def __init__(
        self,
        *,
        crop_endpoint: str = DEFAULT_CROP_ROUTER_ENDPOINT,
        spotting_endpoint: str = DEFAULT_SPOTTING_ROUTER_ENDPOINT,
        gemma_endpoint: str = DEFAULT_GEMMA_ROUTER_ENDPOINT,
        gemma_model: str = DEFAULT_GEMMA_ROUTER_MODEL,
    ) -> None:
        self._crop_endpoint = crop_endpoint
        self._spotting_endpoint = spotting_endpoint
        self._gemma_endpoint = gemma_endpoint
        self._gemma_model = gemma_model

    def get_tool_selection(self, tool_type: str) -> str:
        return "Custom Local Server(Gemma)" if tool_type == "translator" else ""

    def get_credentials(self, provider: str) -> dict[str, str]:
        assert provider == "Custom Local Server(Gemma)"
        return {
            "api_url": self._gemma_endpoint,
            "model": self._gemma_model,
        }

    def get_paddleocr_vl_settings(self) -> dict[str, str]:
        return {"server_url": self._crop_endpoint}

    def get_paddleocr_vl_spotting_settings(self) -> dict[str, str]:
        return {"server_url": self._spotting_endpoint}


class LocalLlamaRouterContractTests(unittest.TestCase):
    def test_only_exact_default_endpoints_are_router_candidates(self) -> None:
        self.assertTrue(
            exact_endpoint_matches(
                DEFAULT_CROP_ROUTER_ENDPOINT,
                DEFAULT_CROP_ROUTER_ENDPOINT,
            )
        )
        rejected = (
            f"{DEFAULT_CROP_ROUTER_ENDPOINT}?probe=1",
            f"{DEFAULT_CROP_ROUTER_ENDPOINT}#fragment",
            "http://user@127.0.0.1:18000/v1/chat/completions",
            "http://localhost:18000/v1/chat/completions",
            "http://127.0.0.1:18000/v1/chat/completions/",
            "http://127.0.0.1:18001/v1/chat/completions",
        )
        for endpoint in rejected:
            with self.subTest(endpoint=endpoint):
                self.assertFalse(
                    exact_endpoint_matches(endpoint, DEFAULT_CROP_ROUTER_ENDPOINT)
                )
                self.assertIsNone(
                    classify_router_pair(
                        "PaddleOCR VL",
                        endpoint,
                        DEFAULT_GEMMA_ROUTER_ENDPOINT,
                        DEFAULT_GEMMA_ROUTER_MODEL,
                    )
                )

    def test_crop_and_spotting_have_only_their_exact_pair(self) -> None:
        crop = classify_router_pair(
            "PaddleOCR VL",
            DEFAULT_CROP_ROUTER_ENDPOINT,
            DEFAULT_GEMMA_ROUTER_ENDPOINT,
            DEFAULT_GEMMA_ROUTER_MODEL,
        )
        spotting = classify_router_pair(
            "PaddleOCR VL Spotting",
            DEFAULT_SPOTTING_ROUTER_ENDPOINT,
            DEFAULT_GEMMA_ROUTER_ENDPOINT,
            DEFAULT_GEMMA_ROUTER_MODEL,
        )
        self.assertIsNotNone(crop)
        self.assertIsNotNone(spotting)
        self.assertNotEqual(crop, spotting)
        self.assertIsNone(
            classify_router_pair(
                "PaddleOCR VL",
                DEFAULT_CROP_ROUTER_ENDPOINT,
                DEFAULT_GEMMA_ROUTER_ENDPOINT,
                "another-model.gguf",
            )
        )

    def test_manager_path_excludes_every_custom_endpoint_shape(self) -> None:
        spec = _spec()
        coordinator = LocalLlamaRouterCoordinator(
            adapter=_FakeAdapter(_contract(spec)),
            gpu_sampler=_SequenceSampler([_gpu_snapshot(0.0, set())]),
        )
        ocr_manager = LocalOCRRuntimeManager(router_coordinator=coordinator)
        gemma_manager = LocalGemmaRuntimeManager(router_coordinator=coordinator)
        ocr_manager.set_router_gemma_manager(gemma_manager)

        defaults = _RouterSettingsPage()
        self.assertIsNotNone(
            ocr_manager.router_pair_for_engine("PaddleOCR VL", defaults)
        )
        self.assertIsNotNone(
            ocr_manager.router_pair_for_engine("PaddleOCR VL Spotting", defaults)
        )
        # A directly created runtime manager has no shared coordinator and
        # must retain the legacy separate-server path, even at default URLs.
        self.assertIsNone(
            LocalOCRRuntimeManager().router_pair_for_engine(
                "PaddleOCR VL", defaults
            )
        )

        custom_crop_endpoints = (
            f"{DEFAULT_CROP_ROUTER_ENDPOINT}?probe=1",
            f"{DEFAULT_CROP_ROUTER_ENDPOINT}#fragment",
            "http://user@127.0.0.1:18000/v1/chat/completions",
            "http://localhost:18000/v1/chat/completions",
            "http://127.0.0.1:18000/v1/chat/completions/",
            "http://127.0.0.1:18001/v1/chat/completions",
        )
        for endpoint in custom_crop_endpoints:
            with self.subTest(kind="crop", endpoint=endpoint):
                self.assertIsNone(
                    ocr_manager.router_pair_for_engine(
                        "PaddleOCR VL",
                        _RouterSettingsPage(crop_endpoint=endpoint),
                    )
                )

        custom_gemma_endpoints = (
            f"{DEFAULT_GEMMA_ROUTER_ENDPOINT}?probe=1",
            f"{DEFAULT_GEMMA_ROUTER_ENDPOINT}#fragment",
            "http://user@127.0.0.1:18080/v1",
            "http://localhost:18080/v1",
            "http://127.0.0.1:18080/v1/",
            "http://127.0.0.1:18081/v1",
        )
        for endpoint in custom_gemma_endpoints:
            with self.subTest(kind="gemma", endpoint=endpoint):
                self.assertIsNone(
                    ocr_manager.router_pair_for_engine(
                        "PaddleOCR VL",
                        _RouterSettingsPage(gemma_endpoint=endpoint),
                    )
                )

    def test_router_compose_and_preset_contracts_are_explicit(self) -> None:
        pairs = (
            (
                "PaddleOCR VL",
                DEFAULT_CROP_ROUTER_ENDPOINT,
                "127.0.0.1:18000:8080",
            ),
            (
                "PaddleOCR VL Spotting",
                DEFAULT_SPOTTING_ROUTER_ENDPOINT,
                "127.0.0.1:18002:8080",
            ),
        )
        for engine_key, endpoint, expected_ocr_port in pairs:
            pair = classify_router_pair(
                engine_key,
                endpoint,
                DEFAULT_GEMMA_ROUTER_ENDPOINT,
                DEFAULT_GEMMA_ROUTER_MODEL,
            )
            assert pair is not None
            with self.subTest(pair=pair.kind.value):
                compose = pair.compose_file.read_text(encoding="utf-8")
                preset = pair.preset_file.read_text(encoding="utf-8")
                self.assertIn(expected_ocr_port, compose)
                self.assertIn("127.0.0.1:18080:8080", compose)
                self.assertIn("paddleocr-router-models:/models/ocr:ro", compose)
                self.assertIn("gemma-router-models:/models/gemma:ro", compose)
                self.assertIn("./router-models.ini:/router/models.ini:ro", compose)
                self.assertIn("--models-max", compose)
                self.assertIn('"1"', compose)
                self.assertIn("--no-models-autoload", compose)
                self.assertIn("load-on-startup = false", preset)
                self.assertIn(pair.ocr_alias, preset)
                self.assertIn(DEFAULT_GEMMA_ROUTER_MODEL, preset)

    def test_fingerprint_carries_all_runtime_ownership_inputs(self) -> None:
        contract = _contract(_spec())
        payload = contract.payload()
        self.assertEqual(payload["image"]["id"], "sha256:image-id")
        self.assertEqual(payload["image"]["repo_digest"], "ghcr.io/ggml-org/llama.cpp@sha256:image-digest")
        self.assertEqual(payload["ocr_model"]["ready_manifest_sha256"], "ocr-manifest")
        self.assertEqual(payload["gemma_model"]["ready_manifest_sha256"], "gemma-manifest")
        self.assertEqual(payload["ocr_model"]["mmproj_sha256"], "mmproj-sha")
        self.assertEqual(payload["preset_sha256"], "preset-sha")
        self.assertEqual(payload["command_sha256"], contract.command_sha256)
        self.assertIn("resolved_compose_config", payload)
        self.assertIn("device_mapping", payload)
        self.assertIn("ownership_labels", payload)

    def test_resolved_compose_contract_rejects_a_writable_model_mount(self) -> None:
        spec = _spec()
        config = {
            "services": {
                "llama-router": {
                    "labels": {
                        ROUTER_OWNER_LABEL: ROUTER_OWNER_VALUE,
                        ROUTER_PAIR_LABEL: spec.pair.kind.value,
                    },
                    "ports": [
                        {
                            "host_ip": "127.0.0.1",
                            "published": "18000",
                            "target": 8080,
                            "protocol": "tcp",
                        },
                        {
                            "host_ip": "127.0.0.1",
                            "published": "18080",
                            "target": 8080,
                            "protocol": "tcp",
                        },
                    ],
                    "volumes": [
                        {
                            "source": "paddleocr-router-models",
                            "target": "/models/ocr",
                            "read_only": True,
                        },
                        {
                            "source": "gemma-router-models",
                            "target": "/models/gemma",
                            "read_only": True,
                        },
                        {
                            "source": "/repo/router-models.ini",
                            "target": "/router/models.ini",
                            "read_only": True,
                        },
                    ],
                    "environment": {
                        "NVIDIA_VISIBLE_DEVICES": "all",
                        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
                    },
                    "gpus": [{"count": -1}],
                }
            },
            "volumes": {
                "paddleocr-router-models": {
                    "external": True,
                    "name": spec.ocr_model.volume_name,
                },
                "gemma-router-models": {
                    "external": True,
                    "name": spec.gemma_model.volume_name,
                },
            },
        }
        adapter = DockerRouterCommandAdapter()

        adapter._assert_resolved_compose_contract(spec, config)
        writable = copy.deepcopy(config)
        writable["services"]["llama-router"]["volumes"][0]["read_only"] = False

        with self.assertRaisesRegex(RouterAdapterError, "read-only"):
            adapter._assert_resolved_compose_contract(spec, writable)

    def test_running_container_contract_rejects_a_writable_model_mount(self) -> None:
        contract = _contract(_spec())
        inspection = RouterContainerInspection(
            name=contract.pair.container_name,
            exists=True,
            running=True,
            image=contract.image_ref,
            image_id=contract.image_id,
            labels=dict(contract.ownership_labels),
            command=contract.server_args,
            entrypoint=contract.entrypoint,
            ports={
                "8080/tcp": [
                    {"HostIp": "127.0.0.1", "HostPort": "18000"},
                    {"HostIp": "127.0.0.1", "HostPort": "18080"},
                ]
            },
            mounts=(
                {
                    "Type": "volume",
                    "Name": contract.ocr_model.volume_name,
                    "Destination": "/models/ocr",
                    "RW": False,
                },
                {
                    "Type": "volume",
                    "Name": contract.gemma_model.volume_name,
                    "Destination": "/models/gemma",
                    "RW": False,
                },
                {
                    "Type": "bind",
                    "Source": "/repo/router-models.ini",
                    "Destination": "/router/models.ini",
                    "RW": False,
                },
            ),
            device_requests=({"Capabilities": [["gpu"]]},),
            pid=100,
        )
        adapter = DockerRouterCommandAdapter()

        adapter._assert_container_contract(inspection, contract)
        writable = replace(
            inspection,
            mounts=(
                {
                    "Type": "volume",
                    "Name": contract.ocr_model.volume_name,
                    "Destination": "/models/ocr",
                    "RW": True,
                },
                *inspection.mounts[1:],
            ),
        )

        with self.assertRaisesRegex(RouterAdapterError, "read-only"):
            adapter._assert_container_contract(writable, contract)

    def test_owned_gpu_process_ids_include_host_and_container_namespaces(self) -> None:
        spec = _spec()
        contract = _contract(spec)
        adapter = DockerRouterCommandAdapter()
        inspection = RouterContainerInspection(
            name=contract.pair.container_name,
            exists=True,
            running=True,
            image=contract.image_ref,
            image_id=contract.image_id,
            labels=dict(contract.ownership_labels),
            command=contract.server_args,
            entrypoint=contract.entrypoint,
            ports={},
            mounts=(),
            device_requests=(),
            pid=26502,
        )
        with mock.patch.object(
            adapter,
            "_inspect_container",
            return_value=inspection,
        ), mock.patch(
            "modules.utils.local_llama_router.adapter.run_docker_command",
            side_effect=[
                SimpleNamespace(returncode=0, stdout="PID\n26363\n26502\n"),
                SimpleNamespace(returncode=0, stdout="PID\n1\n28\n"),
            ],
        ):
            process_ids = adapter.owned_gpu_process_ids(contract)

        self.assertEqual(process_ids, frozenset({1, 28, 26363, 26502}))

    def test_router_get_retries_a_stale_keep_alive_connection_only_once(self) -> None:
        session = mock.MagicMock()
        response = mock.MagicMock()
        response.status_code = 200
        response.json.return_value = {"data": []}
        session.request.side_effect = [
            requests.ConnectionError("stale keep-alive socket"),
            response,
        ]
        adapter = DockerRouterCommandAdapter(request_session=session)

        payload = adapter._request_payload(
            "GET",
            "http://127.0.0.1:18000/models",
            timeout_sec=5.0,
        )

        self.assertEqual(payload, {"data": []})
        self.assertEqual(session.request.call_count, 2)
        session.close.assert_called_once_with()

    def test_router_post_does_not_retry_a_connection_error(self) -> None:
        session = mock.MagicMock()
        session.request.side_effect = requests.ConnectionError("connection reset")
        adapter = DockerRouterCommandAdapter(request_session=session)

        with self.assertRaisesRegex(RouterAdapterError, "Router request failed: POST"):
            adapter._request_payload(
                "POST",
                "http://127.0.0.1:18000/models/load",
                json_payload={"model": "model"},
                timeout_sec=5.0,
            )

        session.request.assert_called_once()
        session.close.assert_not_called()

    def test_direct_listener_is_an_explicit_ownership_error(self) -> None:
        adapter = DockerRouterCommandAdapter()
        probe = mock.MagicMock()
        probe.bind.side_effect = OSError(errno.EADDRINUSE, "Address already in use")

        with mock.patch(
            "modules.utils.local_llama_router.adapter.socket.socket",
            return_value=probe,
        ):
            with self.assertRaisesRegex(
                RouterAdapterOwnershipError,
                "direct or unowned",
            ):
                adapter._assert_host_port_available(
                    18080,
                    cancel_checker=None,
                    timeout_sec=0.0,
                )

        probe.close.assert_called_once_with()

    def test_owned_container_gpu_snapshot_keeps_namespace_pid_with_na_memory(self) -> None:
        spec = _spec()
        contract = _contract(spec)
        adapter = DockerRouterCommandAdapter()
        inspection = RouterContainerInspection(
            name=contract.pair.container_name,
            exists=True,
            running=True,
            image=contract.image_ref,
            image_id=contract.image_id,
            labels=dict(contract.ownership_labels),
            command=contract.server_args,
            entrypoint=contract.entrypoint,
            ports={},
            mounts=(),
            device_requests=(),
            pid=26502,
        )
        with mock.patch.object(
            adapter,
            "_inspect_container",
            return_value=inspection,
        ), mock.patch(
            "modules.utils.local_llama_router.adapter.run_docker_command",
            side_effect=[
                SimpleNamespace(
                    returncode=0,
                    stdout="0, GPU-router, RTX, 12282, 2584, 9698, 0, 0\n",
                ),
                SimpleNamespace(
                    returncode=0,
                    stdout="28, GPU-router, [Not Found], [N/A]\n",
                ),
            ],
        ):
            report = adapter.gpu_snapshot(contract)

        self.assertTrue(report["driver"]["available"])
        self.assertEqual(report["driver"]["primary"]["memory_used_mb"], 2584)
        self.assertEqual(report["driver_processes"]["rows"][0]["pid"], 28)
        self.assertIsNone(report["driver_processes"]["rows"][0]["memory_used_mb"])

    def test_router_gpu_snapshot_falls_back_to_exact_dxg_worker_identity(self) -> None:
        spec = _spec()
        contract = _contract(spec)
        adapter = DockerRouterCommandAdapter()
        inspection = RouterContainerInspection(
            name=contract.pair.container_name,
            exists=True,
            running=True,
            image=contract.image_ref,
            image_id=contract.image_id,
            labels=dict(contract.ownership_labels),
            command=contract.server_args,
            entrypoint=contract.entrypoint,
            ports={},
            mounts=(),
            device_requests=(),
            pid=26502,
        )
        with mock.patch.object(
            adapter,
            "_inspect_container",
            return_value=inspection,
        ), mock.patch(
            "modules.utils.local_llama_router.adapter.run_docker_command",
            side_effect=[
                SimpleNamespace(
                    returncode=0,
                    stdout="0, GPU-router, RTX, 12282, 11742, 540, 0, 0\n",
                ),
                SimpleNamespace(returncode=0, stdout=""),
                SimpleNamespace(
                    returncode=0,
                    stdout=(
                        "67 /app/llama-server --alias gemma-4-26B-IQ4_NL.gguf "
                        "--model /models/gemma/gemma-4-26B-IQ4_NL.gguf\n"
                    ),
                ),
                SimpleNamespace(returncode=0, stdout="/dev/dxg\n"),
            ],
        ):
            report = adapter.gpu_snapshot(contract)

        workers = report["router_worker_processes"]
        self.assertTrue(workers["query_available"])
        self.assertEqual(workers["rows"][0]["pid"], 67)
        self.assertEqual(
            workers["rows"][0]["model_alias"],
            DEFAULT_GEMMA_ROUTER_MODEL,
        )
        self.assertTrue(workers["rows"][0]["gpu_device_attached"])


class LocalLlamaRouterCoordinatorTests(unittest.TestCase):
    def _coordinator(self, *, stopped: bool = False, bad_release: bool = False):
        spec = _spec()
        adapter = _FakeAdapter(_contract(spec))
        after_used = 108.0 if stopped else 250.0
        after_pids = set() if stopped else {100}
        if bad_release:
            after_used = 900.0
        sampler = _SequenceSampler(
            [
                _gpu_snapshot(100.0, set()),  # before container prepare
                _gpu_snapshot(200.0, {100}),  # after container, before model load
                _gpu_snapshot(1200.0, {100}),  # loaded worker attribution
                _gpu_snapshot(1200.0, {100}),  # immediately before unload
                _gpu_snapshot(after_used, after_pids),  # release gate
            ]
        )
        return spec, adapter, LocalLlamaRouterCoordinator(
            adapter=adapter,
            gpu_sampler=sampler,
            router_release_timeout_sec=0.0,
        )

    def test_normal_release_keeps_container_at_zero_loaded_models(self) -> None:
        spec, adapter, coordinator = self._coordinator()
        arbiter = RuntimeResourceArbiter()

        coordinator.load(
            spec,
            spec.ocr_model.alias,
            arbiter=arbiter,
            service="ocr",
        )
        released = coordinator.unload(
            model_alias=spec.ocr_model.alias,
            arbiter=arbiter,
            service="ocr",
        )

        self.assertFalse(released.container_stopped)
        self.assertEqual(released.loaded_count, 0)
        self.assertTrue(released.slots_idle)
        self.assertTrue(released.vram["observed"])
        self.assertTrue(released.verified)
        self.assertEqual(coordinator.snapshot().state, RouterState.CONTAINER_READY)
        self.assertEqual(coordinator.snapshot().loaded_count, 0)
        self.assertTrue(adapter.container_running)
        self.assertEqual(adapter.load_calls, [spec.ocr_model.alias])
        self.assertEqual(adapter.unload_calls, [spec.ocr_model.alias])
        self.assertEqual(adapter.implicit_autoload_checks, 0)

    def test_idempotent_loaded_ensure_does_not_create_a_second_arbiter_lease(self) -> None:
        spec, adapter, coordinator = self._coordinator()
        stage_arbiter = RuntimeResourceArbiter()
        incidental_arbiter = RuntimeResourceArbiter()

        coordinator.load(
            spec,
            DEFAULT_GEMMA_ROUTER_MODEL,
            arbiter=stage_arbiter,
            service="gemma",
        )
        returned = coordinator.load(
            spec,
            DEFAULT_GEMMA_ROUTER_MODEL,
            arbiter=incidental_arbiter,
            service="gemma",
        )

        self.assertEqual(returned.fingerprint, adapter.contract.fingerprint)
        self.assertEqual(stage_arbiter.snapshot().active_model, "gemma")
        self.assertIsNone(incidental_arbiter.snapshot().active_model)
        self.assertEqual(adapter.load_calls, [DEFAULT_GEMMA_ROUTER_MODEL])
        self.assertEqual(coordinator.snapshot().model_generation, 1)

    def test_router_cache_identities_advance_after_each_model_unload(self) -> None:
        spec = _spec()
        adapter = _FakeAdapter(_contract(spec))
        coordinator = LocalLlamaRouterCoordinator(
            adapter=adapter,
            gpu_sampler=_SequenceSampler(
                [
                    _gpu_snapshot(100.0, set()),
                    _gpu_snapshot(200.0, {100}),
                    _gpu_snapshot(1200.0, {100}),
                    _gpu_snapshot(1200.0, {100}),
                    _gpu_snapshot(250.0, {100}),
                    _gpu_snapshot(250.0, {100}),
                    _gpu_snapshot(1300.0, {100}),
                    _gpu_snapshot(1300.0, {100}),
                    _gpu_snapshot(250.0, {100}),
                ]
            ),
            router_release_timeout_sec=0.0,
        )
        ocr_manager = LocalOCRRuntimeManager(router_coordinator=coordinator)
        gemma_manager = LocalGemmaRuntimeManager(router_coordinator=coordinator)
        ocr_manager.set_router_gemma_manager(gemma_manager)
        settings = _RouterSettingsPage()

        coordinator.load(spec, spec.ocr_model.alias, service="ocr")
        ocr_before = ocr_manager.get_ocr_cache_identity(
            "PaddleOCR VL",
            settings,
        )
        assert ocr_before is not None
        coordinator.unload(model_alias=spec.ocr_model.alias, service="ocr")
        ocr_after = ocr_manager.get_ocr_cache_identity(
            "PaddleOCR VL",
            settings,
        )
        assert ocr_after is not None
        self.assertEqual(
            ocr_after["router_model_generation"],
            ocr_before["router_model_generation"] + 1,
        )

        coordinator.load(spec, DEFAULT_GEMMA_ROUTER_MODEL, service="gemma")
        gemma_manager.set_router_spec(spec)
        gemma_manager._router_pair = spec.pair
        gemma_before = gemma_manager.get_translation_cache_identity(settings)
        assert gemma_before is not None
        coordinator.unload(
            model_alias=DEFAULT_GEMMA_ROUTER_MODEL,
            service="gemma",
        )
        gemma_after = gemma_manager.get_translation_cache_identity(settings)
        assert gemma_after is not None
        self.assertEqual(
            gemma_after["router_model_generation"],
            gemma_before["router_model_generation"] + 1,
        )

    def test_load_waits_for_owned_gpu_worker_attribution(self) -> None:
        spec = _spec()
        adapter = _FakeAdapter(_contract(spec))
        sampler = _SequenceSampler(
            [
                _gpu_snapshot(100.0, set()),
                _gpu_snapshot(200.0, set()),
                _gpu_snapshot(1200.0, set()),
                _gpu_snapshot(1200.0, {100}),
            ]
        )
        coordinator = LocalLlamaRouterCoordinator(
            adapter=adapter,
            gpu_sampler=sampler,
            router_release_timeout_sec=0.0,
            router_attribution_timeout_sec=1.0,
        )

        coordinator.load(spec, spec.ocr_model.alias, service="ocr")

        self.assertEqual(coordinator.snapshot().state, RouterState.OCR_LOADED)
        self.assertEqual(adapter.load_calls, [spec.ocr_model.alias])

    def test_load_retries_only_the_short_model_slots_proxy_race(self) -> None:
        spec = _spec()
        adapter = _TransientSlotsFakeAdapter(_contract(spec))
        sampler = _SequenceSampler(
            [
                _gpu_snapshot(100.0, set()),
                _gpu_snapshot(200.0, {100}),
                _gpu_snapshot(1200.0, {100}),
            ]
        )
        coordinator = LocalLlamaRouterCoordinator(
            adapter=adapter,
            gpu_sampler=sampler,
            router_release_timeout_sec=0.0,
        )

        coordinator.load(spec, spec.ocr_model.alias, service="ocr")

        self.assertTrue(adapter._raised_transient_slots_error)
        self.assertEqual(coordinator.snapshot().state, RouterState.OCR_LOADED)
        self.assertEqual(adapter.load_calls, [spec.ocr_model.alias])

    def test_wsl_dxg_worker_identity_allows_only_the_requested_model(self) -> None:
        spec = _spec()
        adapter = _FakeAdapter(_contract(spec))
        sampler = _SequenceSampler(
            [
                _gpu_snapshot(100.0, set(), router_worker_ids=set()),
                _gpu_snapshot(200.0, set(), router_worker_ids=set()),
                _gpu_snapshot(
                    1200.0,
                    set(),
                    router_worker_ids={100},
                    router_worker_alias=spec.ocr_model.alias,
                ),
                _gpu_snapshot(
                    1200.0,
                    set(),
                    router_worker_ids={100},
                    router_worker_alias=spec.ocr_model.alias,
                ),
                _gpu_snapshot(250.0, set(), router_worker_ids=set()),
            ]
        )
        coordinator = LocalLlamaRouterCoordinator(
            adapter=adapter,
            gpu_sampler=sampler,
            router_release_timeout_sec=0.0,
        )

        coordinator.load(spec, spec.ocr_model.alias, service="ocr")
        released = coordinator.unload(
            model_alias=spec.ocr_model.alias,
            service="ocr",
        )

        self.assertTrue(released.verified)
        self.assertEqual(
            released.vram["before_process_source"],
            "router-worker-dxg",
        )

    def test_wsl_dxg_worker_identity_rejects_a_wrong_model_alias(self) -> None:
        spec = _spec()
        adapter = _FakeAdapter(_contract(spec))
        sampler = _SequenceSampler(
            [
                _gpu_snapshot(100.0, set(), router_worker_ids=set()),
                _gpu_snapshot(200.0, set(), router_worker_ids=set()),
                _gpu_snapshot(
                    1200.0,
                    set(),
                    router_worker_ids={100},
                    router_worker_alias=DEFAULT_GEMMA_ROUTER_MODEL,
                ),
                _gpu_snapshot(
                    1200.0,
                    set(),
                    router_worker_ids={100},
                    router_worker_alias=DEFAULT_GEMMA_ROUTER_MODEL,
                ),
                _gpu_snapshot(108.0, set(), router_worker_ids=set()),
            ]
        )
        coordinator = LocalLlamaRouterCoordinator(
            adapter=adapter,
            gpu_sampler=sampler,
            router_release_timeout_sec=0.0,
            router_attribution_timeout_sec=0.0,
        )

        # 기본값은 강건성 우선이다. WSL의 NVML 호환 계층이 worker 를 보고하지
        # 않아도 모델은 적재됐으므로 기동을 막지 않는다.
        coordinator.load(spec, spec.ocr_model.alias, service="ocr")
        self.assertEqual(coordinator.snapshot().loaded_model, spec.ocr_model.alias)

        # 진단용 강제를 켜면 예전처럼 실패한다.
        adapter2 = _FakeAdapter(_contract(spec))
        sampler2 = _SequenceSampler(
            [
                _gpu_snapshot(108.0, set()),
                _gpu_snapshot(108.0, set()),
                _gpu_snapshot(
                    1200.0,
                    set(),
                    router_worker_ids={100},
                    router_worker_alias=DEFAULT_GEMMA_ROUTER_MODEL,
                ),
                _gpu_snapshot(108.0, set(), router_worker_ids=set()),
            ]
        )
        coordinator2 = LocalLlamaRouterCoordinator(
            adapter=adapter2,
            gpu_sampler=sampler2,
            router_release_timeout_sec=0.0,
            router_attribution_timeout_sec=0.0,
        )
        with mock.patch(
            "modules.utils.local_llama_router.coordinator."
            "gpu_release_enforcement_enabled",
            return_value=True,
        ):
            with self.assertRaisesRegex(RouterSetupError, "GPU worker"):
                coordinator2.load(spec, spec.ocr_model.alias, service="ocr")

    def test_kept_container_release_allows_only_owned_worker_pid_to_disappear(self) -> None:
        spec = _spec()
        adapter = _FakeAdapter(_contract(spec))
        sampler = _SequenceSampler(
            [
                _gpu_snapshot(100.0, {10}),
                _gpu_snapshot(200.0, {10}),
                _gpu_snapshot(1200.0, {10, 100}),
                _gpu_snapshot(1200.0, {10, 100}),
                _gpu_snapshot(250.0, {10}),
            ]
        )
        coordinator = LocalLlamaRouterCoordinator(
            adapter=adapter,
            gpu_sampler=sampler,
            router_release_timeout_sec=0.0,
        )
        arbiter = RuntimeResourceArbiter()
        coordinator.load(
            spec,
            spec.ocr_model.alias,
            arbiter=arbiter,
            service="ocr",
        )

        released = coordinator.unload(
            model_alias=spec.ocr_model.alias,
            arbiter=arbiter,
            service="ocr",
        )

        self.assertTrue(released.verified)
        self.assertEqual(coordinator.snapshot().state, RouterState.CONTAINER_READY)

    def test_terminal_release_stops_owned_container_and_returns_to_baseline(self) -> None:
        spec, adapter, coordinator = self._coordinator(stopped=True)
        arbiter = RuntimeResourceArbiter()
        coordinator.load(
            spec,
            DEFAULT_GEMMA_ROUTER_MODEL,
            arbiter=arbiter,
            service="gemma",
        )

        released = coordinator.unload(
            model_alias=DEFAULT_GEMMA_ROUTER_MODEL,
            arbiter=arbiter,
            service="gemma",
            stop_container=True,
        )

        self.assertTrue(released.container_stopped)
        self.assertTrue(released.vram["observed"])
        self.assertTrue(released.verified)
        self.assertEqual(coordinator.snapshot().state, RouterState.IDLE)
        self.assertFalse(adapter.container_running)
        self.assertEqual(adapter.stop_calls, 1)

    def test_release_failure_keeps_arbiter_ownership_and_blocks_next_load(self) -> None:
        # 이 두 테스트는 RELEASE_FAILED 상태 기계 자체가 대상이다. 그 상태는 진단용
        # 강제를 켰을 때만 도달하므로 여기서 명시적으로 켠다.
        patcher = mock.patch(
            "modules.utils.local_llama_router.coordinator."
            "gpu_release_enforcement_enabled",
            return_value=True,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        spec, _adapter, coordinator = self._coordinator(bad_release=True)
        arbiter = RuntimeResourceArbiter()
        coordinator.load(
            spec,
            spec.ocr_model.alias,
            arbiter=arbiter,
            service="ocr",
        )

        with self.assertRaises(RouterReleaseError):
            coordinator.unload(
                model_alias=spec.ocr_model.alias,
                arbiter=arbiter,
                service="ocr",
            )

        self.assertEqual(coordinator.snapshot().state, RouterState.RELEASE_FAILED)
        self.assertIsNotNone(coordinator.snapshot().release_evidence)
        self.assertFalse(coordinator.snapshot().release_evidence.verified)
        self.assertEqual(arbiter.snapshot().active_model, "ocr")
        with self.assertRaises(RuntimeLeaseConflictError):
            coordinator.load(
                spec,
                DEFAULT_GEMMA_ROUTER_MODEL,
                arbiter=arbiter,
                service="gemma",
            )

    def test_failed_model_release_can_only_recover_through_terminal_stop(self) -> None:
        # 이 두 테스트는 RELEASE_FAILED 상태 기계 자체가 대상이다. 그 상태는 진단용
        # 강제를 켰을 때만 도달하므로 여기서 명시적으로 켠다.
        patcher = mock.patch(
            "modules.utils.local_llama_router.coordinator."
            "gpu_release_enforcement_enabled",
            return_value=True,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        spec = _spec()
        adapter = _FakeAdapter(_contract(spec))
        sampler = _SequenceSampler(
            [
                _gpu_snapshot(100.0, set()),
                _gpu_snapshot(200.0, {100}),
                _gpu_snapshot(1200.0, {100}),
                _gpu_snapshot(1200.0, {100}),
                _gpu_snapshot(900.0, {100}),
                _gpu_snapshot(900.0, {100}),
                _gpu_snapshot(108.0, set()),
            ]
        )
        coordinator = LocalLlamaRouterCoordinator(
            adapter=adapter,
            gpu_sampler=sampler,
            router_release_timeout_sec=0.0,
        )
        arbiter = RuntimeResourceArbiter()
        coordinator.load(
            spec,
            spec.ocr_model.alias,
            arbiter=arbiter,
            service="ocr",
        )

        with self.assertRaises(RouterReleaseError):
            coordinator.unload(
                model_alias=spec.ocr_model.alias,
                arbiter=arbiter,
                service="ocr",
            )

        failed = coordinator.snapshot()
        self.assertEqual(failed.state, RouterState.RELEASE_FAILED)
        self.assertIsNotNone(failed.release_evidence)
        self.assertFalse(failed.release_evidence.verified)
        self.assertEqual(arbiter.snapshot().active_model, "ocr")

        recovered = coordinator.stop(arbiter=arbiter, service="ocr")

        self.assertIsNotNone(recovered)
        self.assertTrue(recovered.verified)
        self.assertEqual(coordinator.snapshot().state, RouterState.IDLE)
        self.assertIsNone(arbiter.snapshot().active_model)
        self.assertFalse(adapter.container_running)

    def test_unload_drains_http_lease_without_holding_command_lock(self) -> None:
        spec, adapter, coordinator = self._coordinator()
        arbiter = RuntimeResourceArbiter()
        coordinator.load(
            spec,
            spec.ocr_model.alias,
            arbiter=arbiter,
            service="ocr",
        )
        lease_started = threading.Event()
        release_lease = threading.Event()
        unload_done = threading.Event()
        failures: list[BaseException] = []

        def request() -> None:
            try:
                with coordinator.inference_lease(
                    pair=spec.pair,
                    model_alias=spec.ocr_model.alias,
                ):
                    lease_started.set()
                    release_lease.wait(timeout=2.0)
            except BaseException as exc:  # pragma: no cover - assertion below
                failures.append(exc)

        def unload() -> None:
            try:
                coordinator.unload(
                    model_alias=spec.ocr_model.alias,
                    arbiter=arbiter,
                    service="ocr",
                )
            except BaseException as exc:  # pragma: no cover - assertion below
                failures.append(exc)
            finally:
                unload_done.set()

        request_thread = threading.Thread(target=request)
        request_thread.start()
        self.assertTrue(lease_started.wait(timeout=1.0))
        unload_thread = threading.Thread(target=unload)
        unload_thread.start()
        time.sleep(0.15)
        self.assertEqual(adapter.unload_calls, [])
        self.assertFalse(unload_done.is_set())
        release_lease.set()
        request_thread.join(timeout=2.0)
        unload_thread.join(timeout=2.0)

        self.assertFalse(request_thread.is_alive())
        self.assertFalse(unload_thread.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(adapter.unload_calls, [spec.ocr_model.alias])


class LocalLlamaRouterManagerBoundaryTests(unittest.TestCase):
    def test_router_owner_service_uses_the_loaded_gemma_lease(self) -> None:
        spec = _spec()
        coordinator = mock.MagicMock()
        coordinator.snapshot.return_value = SimpleNamespace(
            loaded_model=DEFAULT_GEMMA_ROUTER_MODEL,
        )
        manager = LocalOCRRuntimeManager(router_coordinator=coordinator)
        manager._active_engine = "PaddleOCR VL"
        manager._router_spec = spec

        self.assertEqual(manager._router_owner_service("paddleocr_vl"), "gemma")

    def test_custom_ocr_selection_terminally_finishes_a_stale_router_pair(self) -> None:
        spec = _spec()
        manager = LocalOCRRuntimeManager(router_coordinator=mock.MagicMock())
        manager._router_pair = spec.pair
        manager._active_engine = "PaddleOCR VL"

        with mock.patch.object(
            manager,
            "router_pair_for_engine",
            return_value=None,
        ), mock.patch.object(manager, "_router_finish") as finish, mock.patch.object(
            manager,
            "_deactivate_active_engine",
        ):
            manager.ensure_engine(
                "PaddleOCR VL",
                _RouterSettingsPage(crop_endpoint="http://localhost:18000/v1/chat/completions"),
                runtime_service="paddleocr_vl",
            )

        finish.assert_called_once_with(
            stop_container=True,
            resource_arbiter=None,
            runtime_service="paddleocr_vl",
            cancel_checker=None,
        )

    def test_crop_to_spotting_transition_terminally_finishes_first_pair(self) -> None:
        crop_spec = _spec()
        spotting_pair = classify_router_pair(
            "PaddleOCR VL Spotting",
            DEFAULT_SPOTTING_ROUTER_ENDPOINT,
            DEFAULT_GEMMA_ROUTER_ENDPOINT,
            DEFAULT_GEMMA_ROUTER_MODEL,
        )
        assert spotting_pair is not None
        coordinator = mock.MagicMock()
        manager = LocalOCRRuntimeManager(router_coordinator=coordinator)
        manager._router_pair = crop_spec.pair
        manager._active_engine = "PaddleOCR VL"
        spotting_spec = replace(crop_spec, pair=spotting_pair)

        with mock.patch.object(manager, "_router_finish") as finish, mock.patch.object(
            manager,
            "_router_runtime_spec",
            return_value=spotting_spec,
        ):
            manager._ensure_router_engine(
                "PaddleOCR VL Spotting",
                _RouterSettingsPage(),
                spotting_pair,
                resource_arbiter=None,
                runtime_service="paddleocr_vl_spotting",
                cancel_checker=None,
            )

        finish.assert_called_once_with(
            stop_container=True,
            resource_arbiter=None,
            runtime_service="paddleocr_vl",
            cancel_checker=None,
        )
        coordinator.load.assert_called_once()

    def test_custom_gemma_selection_terminally_finishes_a_stale_router_pair(self) -> None:
        spec = _spec()
        coordinator = mock.MagicMock()
        coordinator.snapshot.return_value = SimpleNamespace(pair=spec.pair.kind.value)
        manager = LocalGemmaRuntimeManager(router_coordinator=coordinator)
        manager._router_pair = spec.pair

        with mock.patch.object(
            manager,
            "_router_pair_for_server",
            return_value=None,
        ), mock.patch.object(
            manager,
            "_ensure_server_uncached",
        ):
            manager.ensure_server(
                _RouterSettingsPage(gemma_endpoint="http://localhost:18080/v1"),
                runtime_service="gemma",
            )

        coordinator.finish.assert_called_once_with(
            arbiter=None,
            service="gemma",
            stop_container=True,
            cancel_checker=None,
        )
        self.assertIsNone(manager._router_pair)

    def test_custom_gemma_cleanup_uses_the_loaded_ocr_owner_service(self) -> None:
        spec = _spec()
        coordinator = mock.MagicMock()
        coordinator.snapshot.return_value = SimpleNamespace(
            pair=spec.pair.kind.value,
            loaded_model=spec.ocr_model.alias,
        )
        manager = LocalGemmaRuntimeManager(router_coordinator=coordinator)
        manager._router_pair = spec.pair

        manager._finish_router_for_selection_change(
            resource_arbiter=None,
            runtime_service="gemma",
            cancel_checker=None,
        )

        coordinator.finish.assert_called_once_with(
            arbiter=None,
            service="paddleocr_vl",
            stop_container=True,
            cancel_checker=None,
        )

    def test_ocr_router_cancellation_reaches_the_stage_unchanged(self) -> None:
        spec = _spec()
        coordinator = mock.MagicMock()
        coordinator.load.side_effect = OperationCancelledError("cancelled")
        manager = LocalOCRRuntimeManager(router_coordinator=coordinator)

        with mock.patch.object(
            manager,
            "router_pair_for_engine",
            return_value=spec.pair,
        ), mock.patch.object(
            manager,
            "_router_runtime_spec",
            return_value=spec,
        ):
            with self.assertRaisesRegex(OperationCancelledError, "cancelled"):
                manager.ensure_engine(
                    "PaddleOCR VL",
                    _RouterSettingsPage(),
                    runtime_service="paddleocr_vl",
                )

        self.assertIsNone(manager._router_pair)
        coordinator.load.assert_called_once()

    def test_gemma_router_cancellation_reaches_the_stage_unchanged(self) -> None:
        spec = _spec()
        coordinator = mock.MagicMock()
        coordinator.current_pair_for_gemma.return_value = spec.pair
        coordinator.load.side_effect = OperationCancelledError("cancelled")
        manager = LocalGemmaRuntimeManager(router_coordinator=coordinator)
        manager.set_router_spec(spec)

        with self.assertRaisesRegex(OperationCancelledError, "cancelled"):
            manager.ensure_server(
                _RouterSettingsPage(),
                runtime_service="gemma",
            )

        self.assertIsNone(manager._router_pair)
        coordinator.load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
