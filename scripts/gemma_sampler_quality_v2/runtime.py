"""Router-v2-backed Gemma-only replay runtime for the private sampler lab."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

import requests

from modules.ocr.local_runtime import LocalOCRRuntimeManager
from modules.translation.local_runtime import LocalGemmaRuntimeManager
from modules.utils.local_llama_router import (
    DEFAULT_GEMMA_ROUTER_MODEL,
    LocalLlamaRouterCoordinator,
    RouterRuntimeContract,
    RouterRuntimeSpec,
)
from modules.utils.local_llama_router.adapter import DockerRouterCommandAdapter, RouterCommandAdapter
from pipeline.runtime_resource_arbiter import RuntimeResourceArbiter

from .protocol import (
    GEMMA_MODEL_ALIAS,
    PINNED_LLAMA_CPP_IMAGE,
    ProtocolError,
    assert_pinned_sampler_contract,
)


LAB_SERVICE_NAME = "gemma_sampler_quality_v2"


class RuntimeErrorV2(RuntimeError):
    """A Router v2 replay setup, ownership, or drain gate failed."""


class TransientReplayError(RuntimeErrorV2):
    """A retryable transport failure occurred after the safe drain gate."""


@dataclass(frozen=True)
class RouterLabSettings:
    """Minimal settings-page surface used only to build the product Router spec."""

    ocr_endpoint: str = "http://127.0.0.1:18000/v1/chat/completions"
    gemma_endpoint: str = "http://127.0.0.1:18080/v1"
    gemma_model: str = GEMMA_MODEL_ALIAS

    def get_tool_selection(self, key: str) -> str:
        return "Custom Local Server(Gemma)" if key == "translator" else ""

    def get_credentials(self, key: str) -> dict[str, str]:
        if key == "Custom Local Server(Gemma)":
            return {"api_url": self.gemma_endpoint, "model": self.gemma_model}
        return {}

    def get_paddleocr_vl_settings(self) -> dict[str, str]:
        return {"server_url": self.ocr_endpoint}

    def get_paddleocr_vl_spotting_settings(self) -> dict[str, str]:
        return {"server_url": "http://127.0.0.1:18002/v1/chat/completions"}

    def get_hunyuan_ocr_settings(self) -> dict[str, str]:
        return {"server_url": "http://127.0.0.1:28080/v1"}

    def get_mangalmm_ocr_settings(self) -> dict[str, str]:
        return {"server_url": "http://127.0.0.1:28081/v1"}


@dataclass(frozen=True)
class ReplayResponse:
    envelope: Mapping[str, Any]
    latency_ms: float
    completion_tokens: int | None


class RouterGemmaReplayRuntime:
    """Load Gemma once in the Crop Router and enforce terminal Router cleanup."""

    def __init__(
        self,
        *,
        settings: RouterLabSettings | None = None,
        adapter: RouterCommandAdapter | None = None,
        coordinator: LocalLlamaRouterCoordinator | None = None,
        arbiter: RuntimeResourceArbiter | None = None,
        request_session: requests.Session | Any | None = None,
    ) -> None:
        self.settings = settings or RouterLabSettings()
        self.adapter = adapter or DockerRouterCommandAdapter()
        self.coordinator = coordinator or LocalLlamaRouterCoordinator(adapter=self.adapter)
        self.arbiter = arbiter or RuntimeResourceArbiter()
        self.session = request_session or requests.Session()
        self.ocr_manager = LocalOCRRuntimeManager(router_coordinator=self.coordinator)
        self.gemma_manager = LocalGemmaRuntimeManager(router_coordinator=self.coordinator)
        self.ocr_manager.set_router_gemma_manager(self.gemma_manager)
        self.spec: RouterRuntimeSpec | None = None
        self.contract: RouterRuntimeContract | None = None
        self._started = False

    def start(self) -> RouterRuntimeContract:
        """Prepare only the exact Crop default pair, then explicitly load Gemma."""

        if self._started and self.contract is not None:
            return self.contract
        pair = self.coordinator.classify_pair(
            "PaddleOCR VL",
            self.settings.ocr_endpoint,
            self.settings.gemma_endpoint,
            self.settings.gemma_model,
        )
        if pair is None or pair.kind.value != "crop":
            raise RuntimeErrorV2("Sampler v2 requires the exact Crop + default Gemma Router pair.")
        if self.settings.gemma_model != DEFAULT_GEMMA_ROUTER_MODEL:
            raise RuntimeErrorV2("Sampler v2 requires the product Gemma Router alias.")
        try:
            # This reuses product model/volume manifest preparation.  It does
            # not load OCR and never starts the legacy Gemma compose runtime.
            spec = self.ocr_manager._router_runtime_spec("PaddleOCR VL", self.settings, pair)
            self.gemma_manager.set_router_spec(spec)
            contract = self.coordinator.prepare(
                spec,
                arbiter=self.arbiter,
                service=LAB_SERVICE_NAME,
            )
            self.coordinator.load(
                spec,
                DEFAULT_GEMMA_ROUTER_MODEL,
                arbiter=self.arbiter,
                service=LAB_SERVICE_NAME,
            )
        except Exception as exc:
            cleanup_error = self._terminal_cleanup_after_start_failure()
            if cleanup_error is not None:
                raise RuntimeErrorV2(
                    "Router v2 Gemma replay setup failed and terminal cleanup/GPU return "
                    f"verification also failed: setup={exc}; cleanup={cleanup_error}"
                ) from cleanup_error
            if isinstance(exc, RuntimeErrorV2):
                raise
            raise RuntimeErrorV2(
                "Router v2 Gemma replay setup failed after terminal cleanup/GPU return "
                f"verification passed: setup={exc}"
            ) from exc
        snapshot = self.coordinator.snapshot()
        model_snapshot = self.adapter.model_snapshot(pair)
        if (
            snapshot.loaded_count != 1
            or snapshot.loaded_model != DEFAULT_GEMMA_ROUTER_MODEL
            or model_snapshot.loaded_count != 1
            or tuple(model_snapshot.loaded_models) != (DEFAULT_GEMMA_ROUTER_MODEL,)
            or not model_snapshot.slots_idle
        ):
            cleanup_error = self._terminal_cleanup_after_start_failure()
            if cleanup_error is not None:
                raise RuntimeErrorV2(
                    "Router replay started in an invalid model state and terminal cleanup/GPU "
                    f"return verification also failed: {cleanup_error}"
                ) from cleanup_error
            raise RuntimeErrorV2("Router replay did not start with exactly one idle Gemma model.")
        if contract.image_ref != PINNED_LLAMA_CPP_IMAGE:
            cleanup_error = self._terminal_cleanup_after_start_failure()
            if cleanup_error is not None:
                raise RuntimeErrorV2(
                    "Router replay image pin failed and terminal cleanup/GPU return verification "
                    f"also failed: {cleanup_error}"
                ) from cleanup_error
            raise RuntimeErrorV2("Router replay image differs from the sampler v2 pinned build.")
        self.spec = spec
        self.contract = contract
        self._started = True
        return contract

    def request(self, payload: Mapping[str, Any], *, timeout_sec: float) -> ReplayResponse:
        """Issue one HTTP request under an inference lease, never a command lock."""

        if not self._started or self.spec is None or self.contract is None:
            raise RuntimeErrorV2("Router replay request was attempted before start.")
        assert_pinned_sampler_contract(
            image_ref=self.contract.image_ref,
            binary_version=self.contract.binary_version,
            payload=payload,
        )
        started = time.perf_counter()
        try:
            with self.coordinator.inference_lease(
                pair=self.spec.pair,
                model_alias=DEFAULT_GEMMA_ROUTER_MODEL,
            ):
                response = self.session.post(
                    f"{self.settings.gemma_endpoint}/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json=dict(payload),
                    timeout=float(timeout_sec),
                )
        except requests.exceptions.Timeout as exc:
            self.wait_until_safe_to_retry()
            raise TransientReplayError("Gemma replay request timed out after drain verification.") from exc
        except requests.exceptions.ConnectionError as exc:
            self.wait_until_safe_to_retry()
            raise TransientReplayError("Gemma replay connection failed after drain verification.") from exc
        except requests.exceptions.RequestException as exc:
            response = getattr(exc, "response", None)
            if response is not None and int(getattr(response, "status_code", 0) or 0) >= 500:
                self.wait_until_safe_to_retry()
                raise TransientReplayError("Gemma replay server error after drain verification.") from exc
            raise RuntimeErrorV2("Gemma replay request failed permanently.") from exc
        elapsed = (time.perf_counter() - started) * 1000.0
        status_code = int(getattr(response, "status_code", 0) or 0)
        if status_code in {408, 429} or status_code >= 500:
            self.wait_until_safe_to_retry()
            raise TransientReplayError("Gemma replay returned a retryable HTTP status.")
        try:
            response.raise_for_status()
            envelope = response.json()
        except Exception as exc:
            raise RuntimeErrorV2("Gemma replay returned an unusable HTTP response.") from exc
        if not isinstance(envelope, Mapping):
            raise RuntimeErrorV2("Gemma replay returned a non-object response envelope.")
        usage = envelope.get("usage")
        completion_tokens = None
        if isinstance(usage, Mapping):
            value = usage.get("completion_tokens")
            if isinstance(value, int) and not isinstance(value, bool):
                completion_tokens = value
        return ReplayResponse(
            envelope=dict(envelope),
            latency_ms=elapsed,
            completion_tokens=completion_tokens,
        )

    def wait_until_safe_to_retry(self, *, timeout_sec: float = 30.0) -> None:
        """Require both coordinator drain and Router slots idle before a retry."""

        if self.spec is None:
            raise RuntimeErrorV2("Router replay has no active spec for a drain check.")
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        while True:
            coordinator_snapshot = self.coordinator.snapshot()
            model_snapshot = self.adapter.model_snapshot(self.spec.pair)
            if (
                coordinator_snapshot.active_requests == 0
                and model_snapshot.loaded_count == 1
                and tuple(model_snapshot.loaded_models) == (DEFAULT_GEMMA_ROUTER_MODEL,)
                and model_snapshot.slots_idle
            ):
                return
            if time.monotonic() >= deadline:
                raise RuntimeErrorV2("Router did not drain its active request and slots before retry.")
            time.sleep(0.1)

    def close(self) -> None:
        """Terminally unload and stop the exact owned Router, proving GPU return."""

        try:
            self.coordinator.finish(
                arbiter=self.arbiter,
                service=LAB_SERVICE_NAME,
                stop_container=True,
            )
        except Exception as exc:
            raise RuntimeErrorV2("Router replay terminal cleanup or GPU return verification failed.") from exc
        finally:
            self._started = False
            self.spec = None
            self.contract = None

    def _terminal_cleanup_after_start_failure(self) -> Exception | None:
        """Attempt terminal cleanup and return proof failure to the caller.

        A setup error does not excuse a failed unload or unverified GPU return.
        The caller includes both failures in the terminal error instead of
        masking the release failure behind the original setup exception.
        """

        cleanup_error: Exception | None = None
        try:
            self.coordinator.finish(
                arbiter=self.arbiter,
                service=LAB_SERVICE_NAME,
                stop_container=True,
            )
        except Exception as exc:
            cleanup_error = exc
        finally:
            self._started = False
            self.spec = None
            self.contract = None
        return cleanup_error
