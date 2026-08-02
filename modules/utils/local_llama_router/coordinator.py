"""Typed state machine for the shared Paddle OCR + Gemma llama.cpp Router."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterator

from modules.utils.exceptions import OperationCancelledError
from modules.utils.gpu_handoff import (
    DEFAULT_ROUTER_VRAM_RELEASE_TIMEOUT_SEC,
    router_gpu_process_set,
    wait_for_router_container_stop_release,
    wait_for_router_vram_release,
)
from modules.utils.gpu_metrics import query_router_cuda_handoff_metrics
from pipeline.runtime_resource_arbiter import (
    RuntimeLeaseConflictError,
    RuntimeModelState,
    RuntimeResourceArbiter,
)

from .adapter import (
    DockerRouterCommandAdapter,
    RouterAdapterError,
    RouterAdapterOwnershipError,
    RouterCommandAdapter,
    RouterModelSnapshot,
)
from .contracts import (
    DEFAULT_GEMMA_ROUTER_MODEL,
    RouterPair,
    RouterRuntimeContract,
    RouterRuntimeSpec,
    classify_router_pair,
    exact_endpoint_matches,
)


class RouterState(str, Enum):
    IDLE = "idle"
    CONTAINER_READY = "container_ready"
    OCR_LOADED = "ocr_loaded"
    GEMMA_LOADED = "gemma_loaded"
    DRAINING = "draining"
    RELEASE_FAILED = "release_failed"


class RouterStateError(RuntimeError):
    pass


class RouterSetupError(RouterStateError):
    pass


class RouterReleaseError(RouterStateError):
    pass


class RouterOwnershipError(RouterSetupError):
    pass


@dataclass(frozen=True)
class RouterReleaseEvidence:
    model_alias: str
    container_stopped: bool
    loaded_count: int
    slots_idle: bool
    vram: dict[str, Any]
    completed_at: float

    @property
    def verified(self) -> bool:
        """Whether this evidence proves the complete requested release."""

        return bool(
            self.loaded_count == 0
            and self.slots_idle
            and self.vram.get("observed", False)
        )


@dataclass(frozen=True)
class RouterSnapshot:
    state: RouterState
    pair: str | None
    loaded_model: str | None
    loaded_count: int
    active_requests: int
    accepting_requests: bool
    model_generation: int
    container_generation: int
    fingerprint: str
    release_evidence: RouterReleaseEvidence | None
    failure: str


class LocalLlamaRouterCoordinator:
    """Coordinate one explicitly unloaded model in a shared Router container.

    Every command entry point first acquires a ``RuntimeResourceArbiter`` and
    then this coordinator's command lock.  HTTP inference leases use a separate
    condition and deliberately never hold the command lock, allowing the drain
    path to wait for active requests without deadlocking the request itself.
    """

    _SLOT_DRAIN_TIMEOUT_SEC = 30.0
    _SLOT_DRAIN_POLL_SEC = 0.1
    _MODEL_LOAD_TIMEOUT_SEC = 180.0
    _MODEL_UNLOAD_TIMEOUT_SEC = 60.0
    _MODEL_TRANSITION_POLL_SEC = 0.25
    _GPU_ATTRIBUTION_TIMEOUT_SEC = 10.0
    _GPU_ATTRIBUTION_POLL_SEC = 0.25

    def __init__(
        self,
        *,
        adapter: RouterCommandAdapter | None = None,
        gpu_sampler: Callable[[], dict[str, Any]] = query_router_cuda_handoff_metrics,
        router_release_timeout_sec: float = DEFAULT_ROUTER_VRAM_RELEASE_TIMEOUT_SEC,
        router_attribution_timeout_sec: float = _GPU_ATTRIBUTION_TIMEOUT_SEC,
    ) -> None:
        self._adapter = adapter or DockerRouterCommandAdapter()
        self._gpu_sampler = gpu_sampler
        self._router_release_timeout_sec = max(
            0.0,
            float(router_release_timeout_sec),
        )
        self._router_attribution_timeout_sec = max(
            0.0,
            float(router_attribution_timeout_sec),
        )
        self._fallback_arbiter = RuntimeResourceArbiter()
        self._command_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._request_condition = threading.Condition(self._state_lock)
        self._state = RouterState.IDLE
        self._pair: RouterPair | None = None
        self._contract: RouterRuntimeContract | None = None
        self._loaded_model: str | None = None
        self._active_requests = 0
        self._accepting_requests = False
        self._model_generation = 0
        self._container_generation = 0
        self._model_load_baseline: dict[str, Any] | None = None
        self._container_baseline: dict[str, Any] | None = None
        self._last_release_evidence: RouterReleaseEvidence | None = None
        self._failure = ""

    # Candidate classification is intentionally pure so both managers can
    # make a fail-closed Router decision without creating containers.
    @staticmethod
    def classify_pair(
        engine_key: Any,
        ocr_endpoint: Any,
        gemma_endpoint: Any,
        gemma_model: Any,
    ) -> RouterPair | None:
        return classify_router_pair(
            engine_key,
            ocr_endpoint,
            gemma_endpoint,
            gemma_model,
        )

    def current_pair_for_gemma(
        self,
        gemma_endpoint: Any,
        gemma_model: Any,
    ) -> RouterPair | None:
        """Return a current pair only for the exact default Gemma request."""

        with self._state_lock:
            pair = self._pair
            usable = self._state in {
                RouterState.CONTAINER_READY,
                RouterState.GEMMA_LOADED,
                RouterState.OCR_LOADED,
                RouterState.DRAINING,
            }
        if (
            pair is None
            or not usable
            or not exact_endpoint_matches(gemma_endpoint, pair.gemma_endpoint)
            or str(gemma_model or "").strip() != DEFAULT_GEMMA_ROUTER_MODEL
        ):
            return None
        return pair

    def prepare(
        self,
        spec: RouterRuntimeSpec,
        *,
        arbiter: RuntimeResourceArbiter | None = None,
        service: str,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> RouterRuntimeContract:
        """Prepare an owned zero-model Router container for a pair.

        Preparation occupies the Arbiter only while Docker may create CUDA
        context.  It releases that temporary lease before returning because no
        model remains loaded at this point.
        """

        with self._state_lock:
            if self._loaded_model is not None:
                raise RouterStateError(
                    "Router preparation cannot run while a model lease is active."
                )
        active_arbiter = arbiter or self._fallback_arbiter
        token = active_arbiter.token(service)
        try:
            with active_arbiter.model_start(
                token,
                cancel_checker=cancel_checker,
                stale_cleanup=lambda: self._abort_failed_start_locked(),
            ):
                with self._command_lock:
                    contract = self._prepare_locked(spec, cancel_checker=cancel_checker)
            with active_arbiter.model_release(service, target_state=RuntimeModelState.STOPPED):
                pass
            return contract
        except RouterAdapterOwnershipError as exc:
            self._mark_release_failed_if_ownership_remains(exc)
            raise RouterOwnershipError(str(exc)) from exc
        except OperationCancelledError:
            raise
        except RuntimeLeaseConflictError:
            raise
        except RouterStateError:
            raise
        except Exception as exc:
            self._mark_release_failed_if_ownership_remains(exc)
            raise RouterSetupError(f"Router preparation failed: {exc}") from exc

    def load(
        self,
        spec: RouterRuntimeSpec,
        model_alias: str,
        *,
        arbiter: RuntimeResourceArbiter | None = None,
        service: str,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> RouterRuntimeContract:
        """Explicitly load exactly one requested alias and retain its Arbiter lease."""

        self._validate_model_alias(spec.pair, model_alias)
        active_arbiter = arbiter or self._fallback_arbiter
        token = active_arbiter.token(service)
        try:
            matching_loaded_contract = self._matching_loaded_contract(
                spec,
                model_alias,
                cancel_checker=cancel_checker,
            )
            if matching_loaded_contract is not None:
                # A batch prewarm may already own this exact model through its
                # stage Arbiter when a page-level translator asks to ensure it
                # again. Do not create a second fallback-Arbiter lease for an
                # already-loaded Router model.
                return matching_loaded_contract
            with active_arbiter.model_start(
                token,
                cancel_checker=cancel_checker,
                stale_cleanup=lambda: self._abort_failed_start_locked(),
            ):
                with self._command_lock:
                    contract = self._prepare_locked(spec, cancel_checker=cancel_checker)
                    self._load_locked(
                        contract,
                        model_alias,
                        cancel_checker=cancel_checker,
                    )
            return contract
        except RouterAdapterOwnershipError as exc:
            self._mark_release_failed_if_ownership_remains(exc)
            raise RouterOwnershipError(str(exc)) from exc
        except OperationCancelledError:
            raise
        except RuntimeLeaseConflictError:
            raise
        except RouterStateError:
            raise
        except Exception as exc:
            self._mark_release_failed_if_ownership_remains(exc)
            raise RouterSetupError(f"Router model load failed: {exc}") from exc

    def _matching_loaded_contract(
        self,
        spec: RouterRuntimeSpec,
        model_alias: str,
        *,
        cancel_checker: Callable[[], bool] | None,
    ) -> RouterRuntimeContract | None:
        """Return the current contract only for a verified idempotent load.

        This fast path intentionally re-builds the pure Router fingerprint
        before returning. Merely matching an alias or endpoint is insufficient
        because a model/volume/image/preset change must still be treated as a
        pair transition and cannot inherit another run's Arbiter ownership.
        """

        with self._state_lock:
            current_contract = self._contract
            current_pair = self._pair
            current_model = self._loaded_model
            current_state = self._state
        if (
            current_contract is None
            or current_pair != spec.pair
            or current_model != model_alias
            or current_state
            not in {RouterState.OCR_LOADED, RouterState.GEMMA_LOADED}
        ):
            return None
        requested_contract = self._adapter.build_contract(
            spec,
            cancel_checker=cancel_checker,
        )
        with self._state_lock:
            if (
                self._contract is current_contract
                and self._pair == spec.pair
                and self._loaded_model == model_alias
                and self._state
                in {RouterState.OCR_LOADED, RouterState.GEMMA_LOADED}
                and current_contract.fingerprint == requested_contract.fingerprint
            ):
                return current_contract
        return None

    def unload(
        self,
        *,
        model_alias: str,
        arbiter: RuntimeResourceArbiter | None = None,
        service: str,
        stop_container: bool = False,
        cancel_checker: Callable[[], bool] | None = None,
        allow_foreign_owner_teardown: bool = False,
    ) -> RouterReleaseEvidence:
        """Drain requests, unload a model, prove VRAM release, then publish success."""

        active_arbiter = arbiter or self._fallback_arbiter
        try:
            with active_arbiter.model_release(
                service,
                target_state=RuntimeModelState.STOPPED,
                allow_foreign_owner_teardown=allow_foreign_owner_teardown,
            ):
                with self._command_lock:
                    return self._unload_locked(
                        model_alias,
                        stop_container=stop_container,
                        cancel_checker=cancel_checker,
                    )
        except RouterAdapterOwnershipError as exc:
            self._mark_release_failed(exc)
            raise RouterOwnershipError(str(exc)) from exc
        except RouterReleaseError as exc:
            self._mark_release_failed(exc)
            raise
        except Exception as exc:
            self._mark_release_failed(exc)
            raise RouterReleaseError(f"Router model release failed: {exc}") from exc

    def finish(
        self,
        *,
        arbiter: RuntimeResourceArbiter | None = None,
        service: str,
        stop_container: bool,
        cancel_checker: Callable[[], bool] | None = None,
        allow_foreign_owner_teardown: bool = False,
    ) -> RouterReleaseEvidence | None:
        """Finish normal work or terminally stop an owned pair after failure/cancel."""

        with self._state_lock:
            loaded_model = self._loaded_model
            state = self._state
            contract = self._contract
        if state is RouterState.RELEASE_FAILED:
            if not stop_container:
                raise RouterReleaseError(
                    "Router is in RELEASE_FAILED; it retains ownership until terminal cleanup succeeds."
                )
            if contract is None:
                raise RouterReleaseError(
                    "Router is in RELEASE_FAILED without an owned container contract."
                )
            active_arbiter = arbiter or self._fallback_arbiter
            try:
                with active_arbiter.model_release(
                    service,
                    target_state=RuntimeModelState.STOPPED,
                    allow_foreign_owner_teardown=allow_foreign_owner_teardown,
                ):
                    with self._command_lock:
                        self._force_stop_container_locked(
                            contract,
                            cancel_checker=cancel_checker,
                        )
                return self._last_release_evidence
            except Exception as exc:
                self._mark_release_failed(exc)
                raise RouterReleaseError(
                    f"Router failed-state terminal cleanup failed: {exc}"
                ) from exc
        if loaded_model:
            return self.unload(
                model_alias=loaded_model,
                arbiter=arbiter,
                service=service,
                stop_container=stop_container,
                cancel_checker=cancel_checker,
                allow_foreign_owner_teardown=allow_foreign_owner_teardown,
            )
        if state in {RouterState.OCR_LOADED, RouterState.GEMMA_LOADED, RouterState.DRAINING}:
            raise RouterReleaseError(
                "Router has an inconsistent loaded-model state without an alias."
            )
        if contract is None or state is RouterState.IDLE:
            return None
        if not stop_container:
            return None
        active_arbiter = arbiter or self._fallback_arbiter
        try:
            with active_arbiter.model_release(
                service,
                target_state=RuntimeModelState.STOPPED,
                allow_foreign_owner_teardown=allow_foreign_owner_teardown,
            ):
                with self._command_lock:
                    self._stop_container_without_model_locked(
                        contract,
                        cancel_checker=cancel_checker,
                    )
            return self._last_release_evidence
        except Exception as exc:
            self._mark_release_failed(exc)
            raise RouterReleaseError(f"Router terminal cleanup failed: {exc}") from exc

    def stop(
        self,
        *,
        arbiter: RuntimeResourceArbiter | None = None,
        service: str,
        cancel_checker: Callable[[], bool] | None = None,
        allow_foreign_owner_teardown: bool = False,
    ) -> RouterReleaseEvidence | None:
        return self.finish(
            arbiter=arbiter,
            service=service,
            stop_container=True,
            cancel_checker=cancel_checker,
            allow_foreign_owner_teardown=allow_foreign_owner_teardown,
        )

    @contextmanager
    def inference_lease(
        self,
        *,
        pair: RouterPair,
        model_alias: str,
    ) -> Iterator[None]:
        """Protect an HTTP request without holding the Router command lock."""

        with self._request_condition:
            if (
                self._pair != pair
                or self._loaded_model != model_alias
                or self._state
                not in {RouterState.OCR_LOADED, RouterState.GEMMA_LOADED}
                or not self._accepting_requests
            ):
                raise RouterStateError(
                    "Router inference was requested without a matching loaded model lease."
                )
            self._active_requests += 1
        try:
            yield
        finally:
            with self._request_condition:
                self._active_requests = max(0, self._active_requests - 1)
                self._request_condition.notify_all()

    def snapshot(self) -> RouterSnapshot:
        with self._state_lock:
            return RouterSnapshot(
                state=self._state,
                pair=self._pair.kind.value if self._pair is not None else None,
                loaded_model=self._loaded_model,
                loaded_count=1 if self._loaded_model else 0,
                active_requests=self._active_requests,
                accepting_requests=self._accepting_requests,
                model_generation=self._model_generation,
                container_generation=self._container_generation,
                fingerprint=(self._contract.fingerprint if self._contract else ""),
                release_evidence=self._last_release_evidence,
                failure=self._failure,
            )

    def _prepare_locked(
        self,
        spec: RouterRuntimeSpec,
        *,
        cancel_checker: Callable[[], bool] | None,
    ) -> RouterRuntimeContract:
        self._raise_if_failed()
        contract = self._adapter.build_contract(spec, cancel_checker=cancel_checker)
        with self._state_lock:
            current_pair = self._pair
            current_contract = self._contract
            state = self._state
            loaded = self._loaded_model
        if (
            current_pair == spec.pair
            and current_contract is not None
            and current_contract.fingerprint == contract.fingerprint
            and state in {RouterState.CONTAINER_READY, RouterState.OCR_LOADED, RouterState.GEMMA_LOADED}
        ):
            return current_contract
        if loaded:
            raise RouterStateError(
                "Router pair/configuration changed while a model is still loaded."
            )
        if current_contract is not None:
            # Pair changes (including Crop <-> Spotting) stop only a container
            # whose label was already proven product-owned.
            self._stop_container_without_model_locked(
                current_contract,
                cancel_checker=cancel_checker,
            )
        container_baseline = self._gpu_sampler()
        inspection = self._adapter.prepare(contract, cancel_checker=cancel_checker)
        if not inspection.owned_by(contract):
            raise RouterAdapterOwnershipError("Router container ownership changed during prepare.")
        # Publish enough state for the Arbiter's failed-start cleanup before
        # running probes that can still fail after Docker has created a
        # container. Without this, an implicit-autoload contract failure could
        # orphan a zero-model Router process.
        with self._state_lock:
            self._pair = spec.pair
            self._contract = contract
            self._state = RouterState.CONTAINER_READY
            self._loaded_model = None
            self._accepting_requests = False
            self._container_baseline = container_baseline
            self._model_load_baseline = None
            self._container_generation += 1
            self._failure = ""
        snapshot = self._adapter.model_snapshot(spec.pair)
        if snapshot.loaded_count != 0 or not snapshot.slots_idle:
            raise RouterSetupError(
                "Router preparation did not leave zero models and idle slots."
            )
        # Do not send an unloaded inference request here. Router's explicit
        # no-autoload configuration is fingerprinted above; the live rejection
        # probe belongs to isolated validation because it mutates router-side
        # request state before the product's first explicit model load.
        return contract

    def _load_locked(
        self,
        contract: RouterRuntimeContract,
        model_alias: str,
        *,
        cancel_checker: Callable[[], bool] | None,
    ) -> None:
        with self._state_lock:
            if self._state is RouterState.RELEASE_FAILED:
                self._raise_if_failed()
            if self._contract is None or self._contract.fingerprint != contract.fingerprint:
                raise RouterStateError("Router contract changed before model load.")
            if self._loaded_model and self._loaded_model != model_alias:
                raise RouterStateError(
                    "Router cannot load a second model before the first one is released."
                )
            if self._loaded_model == model_alias:
                return
            if self._state is not RouterState.CONTAINER_READY:
                raise RouterStateError(f"Router cannot load from state {self._state.value}.")
        model_baseline = self._container_gpu_sample(contract)
        self._adapter.load_model(
            contract.pair,
            model_alias,
            cancel_checker=cancel_checker,
        )
        models = self._wait_for_model_loaded(
            contract.pair,
            model_alias,
        )
        if (
            models.loaded_count != 1
            or tuple(models.loaded_models) != (model_alias,)
            or not models.slots_idle
        ):
            raise RouterSetupError(
                "Router model load did not produce exactly one idle requested model."
            )
        self._wait_for_loaded_gpu_attribution(
            contract,
            model_baseline,
            model_alias=model_alias,
        )
        with self._state_lock:
            self._loaded_model = model_alias
            self._model_load_baseline = model_baseline
            self._state = (
                RouterState.OCR_LOADED
                if model_alias == contract.ocr_model.alias
                else RouterState.GEMMA_LOADED
            )
            self._accepting_requests = True
            self._model_generation += 1

    @staticmethod
    def _router_gpu_memory(
        sample: dict[str, Any],
        *,
        gpu_uuid: str = "",
    ) -> tuple[str, float | None]:
        driver = sample.get("driver")
        if not isinstance(driver, dict) or not bool(driver.get("available")):
            return str(gpu_uuid or ""), None
        target_uuid = str(gpu_uuid or "").strip()
        rows = driver.get("gpus")
        if isinstance(rows, list) and target_uuid:
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if str(row.get("uuid") or "").strip() != target_uuid:
                    continue
                value = row.get("memory_used_mb")
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return target_uuid, None
                return target_uuid, float(value)
        primary = driver.get("primary")
        if not isinstance(primary, dict):
            return target_uuid, None
        primary_uuid = str(primary.get("uuid") or "").strip()
        if target_uuid and primary_uuid != target_uuid:
            return target_uuid, None
        value = primary.get("memory_used_mb")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return primary_uuid, None
        return primary_uuid, float(value)

    @staticmethod
    def _router_gpu_process_ids(
        sample: dict[str, Any],
        *,
        gpu_uuid: str,
    ) -> tuple[frozenset[int] | None, str]:
        return router_gpu_process_set(sample, gpu_uuid=gpu_uuid)

    @staticmethod
    def _router_worker_aliases(
        sample: dict[str, Any],
        *,
        gpu_uuid: str,
    ) -> tuple[str, ...] | None:
        """Return exact aliases when WSL uses adapter worker-FD evidence."""

        payload = sample.get("router_worker_processes")
        if not isinstance(payload, dict) or not bool(payload.get("query_available")):
            return None
        rows = payload.get("rows")
        if not isinstance(rows, list):
            return None
        aliases: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                return None
            if str(row.get("gpu_uuid") or "").strip() != str(gpu_uuid or "").strip():
                return None
            if not bool(row.get("gpu_device_attached")):
                return None
            alias = str(row.get("model_alias") or "").strip()
            if not alias:
                return None
            aliases.append(alias)
        return tuple(sorted(aliases))

    def _container_gpu_sample(
        self,
        contract: RouterRuntimeContract,
    ) -> dict[str, Any]:
        """Use the adapter's exact-container GPU view when it is available."""

        sampler = getattr(self._adapter, "gpu_snapshot", None)
        if callable(sampler):
            return sampler(contract)
        # Test adapters and legacy in-process fakes deliberately have no
        # Docker namespace. Their injected sampler remains the only source.
        return self._gpu_sampler()

    def _wait_for_loaded_gpu_attribution(
        self,
        contract: RouterRuntimeContract,
        model_baseline: dict[str, Any],
        *,
        model_alias: str,
    ) -> None:
        """Wait until a loaded Router worker is observable on its exact GPU.

        The Router model API can report ``loaded`` before Docker/WSL publishes
        the worker in the driver process table. Publishing the inference lease
        before that point would make a later unload impossible to attribute,
        even if memory returned correctly. Treat missing attribution as a
        startup failure before any HTTP inference can begin.
        """

        baseline_uuid, baseline_used = self._router_gpu_memory(model_baseline)
        deadline = time.monotonic() + self._router_attribution_timeout_sec
        last_reason = ""
        while True:
            sample = self._container_gpu_sample(contract)
            current_uuid, current_used = self._router_gpu_memory(
                sample,
                gpu_uuid=baseline_uuid,
            )
            process_ids, process_source = self._router_gpu_process_ids(
                sample,
                gpu_uuid=current_uuid,
            )
            worker_aliases = self._router_worker_aliases(
                sample,
                gpu_uuid=current_uuid,
            )
            owned_ids = self._adapter.owned_gpu_process_ids(contract)
            memory_delta = (
                current_used - baseline_used
                if isinstance(current_used, (int, float))
                and isinstance(baseline_used, (int, float))
                else None
            )
            if (
                baseline_uuid
                and current_uuid == baseline_uuid
                and isinstance(memory_delta, (int, float))
                and memory_delta > 0.0
                and process_ids is not None
                and owned_ids
                and bool(process_ids.intersection(owned_ids))
                and (
                    process_source != "router-worker-dxg"
                    or worker_aliases == (model_alias,)
                )
            ):
                return
            last_reason = (
                f"gpu={current_uuid or '<missing>'} baseline={baseline_uuid or '<missing>'} "
                f"delta_mb={memory_delta!r} process_ids={sorted(process_ids or ())} "
                f"process_source={process_source} worker_aliases={list(worker_aliases or ())} "
                f"owned_ids={sorted(owned_ids)}"
            )
            if time.monotonic() >= deadline:
                break
            time.sleep(self._GPU_ATTRIBUTION_POLL_SEC)
        raise RouterSetupError(
            "Router model loaded without attributable GPU worker evidence: "
            + last_reason
        )

    def _unload_locked(
        self,
        model_alias: str,
        *,
        stop_container: bool,
        cancel_checker: Callable[[], bool] | None,
    ) -> RouterReleaseEvidence:
        with self._state_lock:
            contract = self._contract
            pair = self._pair
            if contract is None or pair is None:
                raise RouterReleaseError("Router has no active owned pair to release.")
            if self._loaded_model != model_alias:
                raise RouterReleaseError(
                    "Router release model does not match the currently loaded alias."
                )
            if self._state not in {RouterState.OCR_LOADED, RouterState.GEMMA_LOADED}:
                raise RouterReleaseError(
                    f"Router cannot release from state {self._state.value}."
                )
            self._state = RouterState.DRAINING
            self._accepting_requests = False
            deadline = time.monotonic() + self._SLOT_DRAIN_TIMEOUT_SEC
            while self._active_requests > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RouterReleaseError(
                        "Router active inference requests did not drain before unload."
                    )
                self._request_condition.wait(timeout=min(0.25, remaining))
            model_baseline = self._model_load_baseline
            container_baseline = self._container_baseline
        if model_baseline is None or container_baseline is None:
            raise RouterReleaseError("Router GPU baselines are missing for model release.")
        pre_unload = self._adapter.model_snapshot(pair)
        if pre_unload.loaded_count != 1 or tuple(pre_unload.loaded_models) != (model_alias,):
            raise RouterReleaseError("Router model API did not confirm the loaded release target.")
        self._wait_for_router_slots_idle(pair)
        before = self._container_gpu_sample(contract)
        owned_process_ids = self._adapter.owned_gpu_process_ids(contract)
        self._adapter.unload_model(
            pair,
            model_alias,
            cancel_checker=cancel_checker,
        )
        after_unload = self._wait_for_model_unloaded(pair, model_alias)
        if after_unload.loaded_count != 0 or not after_unload.slots_idle:
            raise RouterReleaseError(
                "Router unload did not leave zero loaded models and idle slots."
            )
        if stop_container:
            self._adapter.stop_pair(contract, cancel_checker=cancel_checker)
        vram = wait_for_router_vram_release(
            before,
            model_load_baseline=model_baseline,
            container_baseline=container_baseline,
            container_kept=not stop_container,
            owned_router_process_ids=owned_process_ids,
            timeout_sec=self._router_release_timeout_sec,
            sampler=(
                self._gpu_sampler
                if stop_container
                else lambda: self._container_gpu_sample(contract)
            ),
        )
        evidence = RouterReleaseEvidence(
            model_alias=model_alias,
            container_stopped=stop_container,
            loaded_count=after_unload.loaded_count,
            slots_idle=after_unload.slots_idle,
            vram=vram,
            completed_at=time.time(),
        )
        # Preserve failed proof data in the snapshot too. The caller still
        # receives an exception and the Arbiter lease remains held, but a
        # terminal unload failure cannot be obscured by later cleanup logs.
        with self._state_lock:
            self._last_release_evidence = evidence
        if not evidence.verified:
            raise RouterReleaseError(
                "Router model unload completed but GPU return was not proven: "
                f"{vram.get('status', 'unknown')}"
            )
        with self._state_lock:
            self._loaded_model = None
            self._model_load_baseline = None
            self._accepting_requests = False
            self._model_generation += 1
            self._failure = ""
            if stop_container:
                self._clear_container_locked()
            else:
                self._state = RouterState.CONTAINER_READY
        return evidence

    def _stop_container_without_model_locked(
        self,
        contract: RouterRuntimeContract,
        *,
        cancel_checker: Callable[[], bool] | None,
    ) -> None:
        with self._state_lock:
            if self._contract is None or self._contract.fingerprint != contract.fingerprint:
                raise RouterReleaseError(
                    "Router container changed before terminal zero-model cleanup."
                )
            pair = self._pair
            container_baseline = self._container_baseline
        if pair is None or container_baseline is None:
            raise RouterReleaseError(
                "Router baseline evidence is missing for terminal zero-model cleanup."
            )
        models = self._adapter.model_snapshot(pair)
        if models.loaded_count != 0 or not models.slots_idle:
            raise RouterReleaseError(
                "Router terminal cleanup requires zero loaded models and idle slots."
            )
        self._force_stop_container_locked(
            contract,
            cancel_checker=cancel_checker,
        )

    def _force_stop_container_locked(
        self,
        contract: RouterRuntimeContract,
        *,
        cancel_checker: Callable[[], bool] | None,
    ) -> None:
        """Stop an owned Router after failed startup without trusting its API.

        A load request is asynchronous in Router mode. If it fails or is
        cancelled while the model is still ``loading``, a follow-up model API
        call cannot prove a clean zero-model state. Stopping the exact owned
        container and proving the pre-container GPU baseline is the only safe
        terminal cleanup in that state.
        """
        with self._state_lock:
            if self._contract is None or self._contract.fingerprint != contract.fingerprint:
                raise RouterReleaseError(
                    "Router container changed before forced terminal cleanup."
                )
            container_baseline = self._container_baseline
        if container_baseline is None:
            raise RouterReleaseError(
                "Router baseline evidence is missing for forced terminal cleanup."
            )
        before = self._container_gpu_sample(contract)
        owned_process_ids = self._adapter.owned_gpu_process_ids(contract)
        self._adapter.stop_pair(contract, cancel_checker=cancel_checker)
        vram = wait_for_router_container_stop_release(
            before,
            container_baseline=container_baseline,
            owned_router_process_ids=owned_process_ids,
            timeout_sec=self._router_release_timeout_sec,
            sampler=self._gpu_sampler,
        )
        if not bool(vram.get("observed", False)):
            raise RouterReleaseError(
                "Router zero-model container stop completed but GPU return was not proven: "
                f"{vram.get('status', 'unknown')}"
            )
        evidence = RouterReleaseEvidence(
            model_alias="",
            container_stopped=True,
            loaded_count=0,
            slots_idle=True,
            vram=vram,
            completed_at=time.time(),
        )
        with self._state_lock:
            self._last_release_evidence = evidence
            self._clear_container_locked()

    def _abort_failed_start_locked(self) -> None:
        """Best-effort stale cleanup called while the Arbiter still owns startup."""

        with self._command_lock:
            with self._state_lock:
                contract = self._contract
                if contract is not None:
                    self._state = RouterState.DRAINING
                    self._accepting_requests = False
                    deadline = time.monotonic() + self._SLOT_DRAIN_TIMEOUT_SEC
                    while self._active_requests > 0:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise RouterReleaseError(
                                "Router active inference requests did not drain during failed-start cleanup."
                            )
                        self._request_condition.wait(timeout=min(0.25, remaining))
            if contract is None:
                return
            try:
                self._force_stop_container_locked(
                    contract,
                    cancel_checker=None,
                )
            except BaseException as exc:
                self._mark_release_failed(exc)
                raise

    def _wait_for_router_slots_idle(self, pair: RouterPair) -> None:
        deadline = time.monotonic() + self._SLOT_DRAIN_TIMEOUT_SEC
        last_snapshot: RouterModelSnapshot | None = None
        while True:
            last_snapshot = self._adapter.model_snapshot(pair)
            if last_snapshot.slots_idle:
                return
            if time.monotonic() >= deadline:
                raise RouterReleaseError("Router slots did not become idle before unload.")
            time.sleep(self._SLOT_DRAIN_POLL_SEC)

    def _wait_for_model_loaded(
        self,
        pair: RouterPair,
        model_alias: str,
    ) -> RouterModelSnapshot:
        deadline = time.monotonic() + self._MODEL_LOAD_TIMEOUT_SEC
        last_snapshot: RouterModelSnapshot | None = None
        last_slot_connection_error = ""
        while True:
            try:
                last_snapshot = self._adapter.model_snapshot(pair)
            except RouterAdapterError as exc:
                # Router can publish a model as loaded just before its child
                # server accepts the per-model /slots proxy. That is a short
                # transition, not a successful ready state and not a reason
                # to fall through to an unmanaged endpoint. Retry only this
                # explicit proxy-connect race; all other adapter failures
                # remain fail-closed.
                if not self._is_transient_slot_connection_error(exc):
                    raise RouterSetupError(
                        f"Router model readiness check failed: {exc}"
                    ) from exc
                last_slot_connection_error = str(exc)
                if time.monotonic() >= deadline:
                    raise RouterSetupError(
                        "Router model child did not accept its /slots proxy before "
                        "the transition timeout: " + last_slot_connection_error
                    ) from exc
                time.sleep(self._MODEL_TRANSITION_POLL_SEC)
                continue
            if (
                last_snapshot.loaded_count == 1
                and tuple(last_snapshot.loaded_models) == (model_alias,)
                and last_snapshot.slots_idle
            ):
                return last_snapshot
            if not last_snapshot.transitional_models:
                raise RouterSetupError(
                    "Router model load completed without exactly one idle requested model."
                )
            if time.monotonic() >= deadline:
                raise RouterSetupError(
                    "Router model load did not finish before its transition timeout."
                )
            time.sleep(self._MODEL_TRANSITION_POLL_SEC)

    @staticmethod
    def _is_transient_slot_connection_error(exc: RouterAdapterError) -> bool:
        message = str(exc).lower()
        return bool(
            "get http://127.0.0.1:" in message
            and "/slots?model=" in message
            and "proxy error: could not establish connection" in message
        )

    def _wait_for_model_unloaded(
        self,
        pair: RouterPair,
        model_alias: str,
    ) -> RouterModelSnapshot:
        deadline = time.monotonic() + self._MODEL_UNLOAD_TIMEOUT_SEC
        last_snapshot: RouterModelSnapshot | None = None
        while True:
            last_snapshot = self._adapter.model_snapshot(
                pair,
                include_slots=False,
            )
            if last_snapshot.loaded_count == 0 and last_snapshot.slots_idle:
                return last_snapshot
            if (
                last_snapshot.loaded_count > 0
                and model_alias not in last_snapshot.loaded_models
                and not last_snapshot.transitional_models
            ):
                raise RouterReleaseError(
                    "Router unload changed the loaded model identity unexpectedly."
                )
            if time.monotonic() >= deadline:
                raise RouterReleaseError(
                    "Router model unload did not finish before its transition timeout."
                )
            time.sleep(self._MODEL_TRANSITION_POLL_SEC)

    @staticmethod
    def _validate_model_alias(pair: RouterPair, model_alias: str) -> None:
        allowed = {pair.ocr_alias, DEFAULT_GEMMA_ROUTER_MODEL}
        if str(model_alias or "").strip() not in allowed:
            raise RouterSetupError(
                f"Router model alias is not part of the selected pair: {model_alias!r}"
            )

    def _clear_container_locked(self) -> None:
        self._state = RouterState.IDLE
        self._pair = None
        self._contract = None
        self._loaded_model = None
        self._active_requests = 0
        self._accepting_requests = False
        self._model_load_baseline = None
        self._container_baseline = None
        self._container_generation += 1

    def _mark_release_failed(self, exc: BaseException) -> None:
        with self._state_lock:
            self._state = RouterState.RELEASE_FAILED
            self._accepting_requests = False
            self._failure = f"{type(exc).__name__}: {exc}"
            self._request_condition.notify_all()

    def _mark_release_failed_if_ownership_remains(
        self,
        exc: BaseException,
    ) -> None:
        """Retain the Arbiter only when a failed command still owns Router state."""

        with self._state_lock:
            owns_router = bool(
                self._contract is not None or self._state is not RouterState.IDLE
            )
        if owns_router:
            self._mark_release_failed(exc)

    def _raise_if_failed(self) -> None:
        with self._state_lock:
            if self._state is RouterState.RELEASE_FAILED:
                raise RouterReleaseError(
                    "Router is in RELEASE_FAILED and still owns the GPU lease: "
                    f"{self._failure}"
                )
