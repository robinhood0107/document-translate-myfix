from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator

from modules.utils.exceptions import OperationCancelledError


class RuntimeModelState(str, Enum):
    STOPPED = "stopped"
    MODEL_LOADING = "model_loading"
    MODEL_READY = "model_ready"
    SLEEPING = "sleeping"
    RELEASING = "releasing"
    RELEASE_FAILED = "release_failed"


class RuntimeLeaseConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeCommandToken:
    generation: int
    service: str


@dataclass(frozen=True)
class RuntimeArbiterSnapshot:
    generation: int
    cancelled: bool
    active_model: str | None
    states: dict[str, str]


@dataclass
class RuntimeReleaseContext:
    """Mutable release outcome selected while the command lock is held."""

    target_state: RuntimeModelState


TransitionCallback = Callable[[str, RuntimeModelState, RuntimeModelState | None, str], None]


class RuntimeResourceArbiter:
    """Serialize managed GPU-model transitions for one pipeline run.

    The pipeline may prepare a runtime on a background thread, but every model
    start and release passes through the same command lock.  A generation token
    prevents a cancelled run from publishing a late MODEL_READY transition.
    """

    def __init__(
        self,
        *,
        transition_callback: TransitionCallback | None = None,
    ) -> None:
        self._state_lock = threading.RLock()
        self._command_lock = threading.RLock()
        self._transition_callback = transition_callback
        self._generation = 0
        self._cancelled = False
        self._active_model: str | None = None
        self._states: dict[str, RuntimeModelState] = {}
        self._external_owner: tuple[str, int] | None = None

    def reset(self) -> int:
        """Start a fresh command generation without guessing runtime state."""

        with self._state_lock:
            self._generation += 1
            self._cancelled = False
            return self._generation

    def cancel_generation(self) -> int:
        with self._state_lock:
            self._generation += 1
            self._cancelled = True
            return self._generation

    def token(self, service: str) -> RuntimeCommandToken:
        service = self._normalize_service(service)
        with self._state_lock:
            return RuntimeCommandToken(self._generation, service)

    def snapshot(self) -> RuntimeArbiterSnapshot:
        with self._state_lock:
            return RuntimeArbiterSnapshot(
                generation=self._generation,
                cancelled=self._cancelled,
                active_model=self._active_model,
                states={
                    service: state.value
                    for service, state in sorted(self._states.items())
                },
            )

    def is_cancelled(
        self,
        token: RuntimeCommandToken,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> bool:
        with self._state_lock:
            cancelled = self._cancelled or token.generation != self._generation
        if cancelled:
            return True
        if not callable(cancel_checker):
            return False
        try:
            return bool(cancel_checker())
        except Exception:
            return True

    @contextmanager
    def model_start(
        self,
        token: RuntimeCommandToken,
        *,
        cancel_checker: Callable[[], bool] | None = None,
        stale_cleanup: Callable[[], None] | None = None,
    ) -> Iterator[None]:
        """Run one model start while enforcing the exclusive GPU lease."""

        service = self._normalize_service(token.service)
        with self._command_lock:
            self._raise_if_cancelled(token, cancel_checker, "before startup")
            with self._state_lock:
                if (
                    self._active_model == service
                    and self._states.get(service)
                    is RuntimeModelState.RELEASE_FAILED
                ):
                    raise RuntimeLeaseConflictError(
                        f"Cannot start {service}; its previous release failed."
                    )
                if self._active_model not in {None, service}:
                    raise RuntimeLeaseConflictError(
                        f"Cannot start {service}; GPU model lease is held by "
                        f"{self._active_model}."
                    )
                self._active_model = service
                self._transition(
                    service,
                    RuntimeModelState.MODEL_LOADING,
                    outcome="starting",
                )
            try:
                yield
            except BaseException as startup_error:
                cleanup_error = self._cleanup_failed_start(
                    service,
                    stale_cleanup,
                )
                if cleanup_error is not None:
                    raise cleanup_error from startup_error
                raise
            if self.is_cancelled(token, cancel_checker):
                cleanup_error = self._cleanup_failed_start(
                    service,
                    stale_cleanup,
                )
                if cleanup_error is not None:
                    raise cleanup_error
                raise OperationCancelledError(
                    f"{service} startup was cancelled after model load."
                )
            with self._state_lock:
                self._transition(
                    service,
                    RuntimeModelState.MODEL_READY,
                    outcome="completed",
                )

    @contextmanager
    def model_release(
        self,
        service: str,
        *,
        target_state: RuntimeModelState = RuntimeModelState.STOPPED,
        allow_foreign_owner_teardown: bool = False,
    ) -> Iterator[RuntimeReleaseContext]:
        """Serialize model release against every queued/running model start.

        ``allow_foreign_owner_teardown`` is for terminal cleanup only.  It lets
        a caller stop a second, pre-existing runtime after another runtime has
        already failed to release, without relinquishing the failed owner's
        lease.  It must never be used for a normal stage handoff.
        """

        service = self._normalize_service(service)
        if target_state not in {
            RuntimeModelState.STOPPED,
            RuntimeModelState.SLEEPING,
        }:
            raise ValueError(f"Invalid release target state: {target_state}")
        with self._command_lock:
            with self._state_lock:
                foreign_owner = self._active_model not in {None, service}
                if foreign_owner and not allow_foreign_owner_teardown:
                    raise RuntimeLeaseConflictError(
                        f"Cannot release {service}; GPU model lease is held by "
                        f"{self._active_model}."
                    )
                # A runtime can predate the arbiter (for example after a
                # process restore).  Treat its release as owning the lease
                # until the release is proven successful, so a failed release
                # cannot allow another GPU model to start concurrently.
                if self._active_model is None:
                    self._active_model = service
                self._transition(
                    service,
                    RuntimeModelState.RELEASING,
                    outcome="starting",
                )
            release_context = RuntimeReleaseContext(target_state=target_state)
            try:
                yield release_context
            except BaseException:
                with self._state_lock:
                    self._transition(
                        service,
                        RuntimeModelState.RELEASE_FAILED,
                        outcome="failed",
                    )
                raise
            if release_context.target_state not in {
                RuntimeModelState.STOPPED,
                RuntimeModelState.SLEEPING,
            }:
                with self._state_lock:
                    self._transition(
                        service,
                        RuntimeModelState.RELEASE_FAILED,
                        outcome="failed",
                    )
                raise ValueError(
                    "Invalid release target state: "
                    f"{release_context.target_state}"
                )
            with self._state_lock:
                if self._active_model == service:
                    self._active_model = None
                self._transition(
                    service,
                    release_context.target_state,
                    outcome="completed",
                )

    def acquire_external_model(self, service: str) -> None:
        """Reserve the GPU for an in-process model such as the inpainter."""

        service = self._normalize_service(service)
        self._command_lock.acquire()
        acquired = False
        try:
            owner = threading.get_ident()
            with self._state_lock:
                if self._external_owner is not None:
                    raise RuntimeLeaseConflictError(
                        f"External GPU model lease is already held by "
                        f"{self._external_owner[0]}."
                    )
                if self._active_model not in {None, service}:
                    raise RuntimeLeaseConflictError(
                        f"Cannot acquire {service}; GPU model lease is held by "
                        f"{self._active_model}."
                    )
                self._external_owner = (service, owner)
                self._active_model = service
                self._transition(
                    service,
                    RuntimeModelState.MODEL_LOADING,
                    outcome="acquired",
                )
                acquired = True
        finally:
            if not acquired:
                self._command_lock.release()

    def mark_external_model_ready(self, service: str) -> None:
        service = self._normalize_service(service)
        owner = threading.get_ident()
        with self._state_lock:
            if self._external_owner != (service, owner):
                raise RuntimeLeaseConflictError(
                    f"External GPU model lease for {service} is not held by "
                    "the current thread."
                )
            self._transition(
                service,
                RuntimeModelState.MODEL_READY,
                outcome="completed",
            )

    def release_external_model(
        self,
        service: str,
        *,
        release_succeeded: bool,
    ) -> None:
        service = self._normalize_service(service)
        owner = threading.get_ident()
        with self._state_lock:
            if self._external_owner != (service, owner):
                raise RuntimeLeaseConflictError(
                    f"External GPU model lease for {service} is not held by "
                    "the current thread."
                )
            self._external_owner = None
            if release_succeeded:
                if self._active_model == service:
                    self._active_model = None
                self._transition(
                    service,
                    RuntimeModelState.STOPPED,
                    outcome="completed",
                )
            else:
                self._transition(
                    service,
                    RuntimeModelState.RELEASE_FAILED,
                    outcome="failed",
                )
        self._command_lock.release()

    def external_model_held_by_current_thread(self, service: str) -> bool:
        service = self._normalize_service(service)
        with self._state_lock:
            return self._external_owner == (service, threading.get_ident())

    def _cleanup_failed_start(
        self,
        service: str,
        cleanup: Callable[[], None] | None,
    ) -> BaseException | None:
        cleanup_error: BaseException | None = None
        if callable(cleanup):
            try:
                cleanup()
            except BaseException as exc:
                cleanup_error = exc
        with self._state_lock:
            if cleanup_error is None:
                if self._active_model == service:
                    self._active_model = None
                self._transition(
                    service,
                    RuntimeModelState.STOPPED,
                    outcome="cancelled",
                )
            else:
                self._transition(
                    service,
                    RuntimeModelState.RELEASE_FAILED,
                    outcome="failed",
                )
        return cleanup_error

    def _raise_if_cancelled(
        self,
        token: RuntimeCommandToken,
        cancel_checker: Callable[[], bool] | None,
        when: str,
    ) -> None:
        if self.is_cancelled(token, cancel_checker):
            raise OperationCancelledError(
                f"{token.service} startup was cancelled {when}."
            )

    def _transition(
        self,
        service: str,
        state: RuntimeModelState,
        *,
        outcome: str,
    ) -> None:
        previous = self._states.get(service)
        self._states[service] = state
        callback = self._transition_callback
        if callable(callback):
            try:
                callback(service, state, previous, outcome)
            except Exception:
                # Telemetry and observers must never break runtime ownership.
                pass

    @staticmethod
    def _normalize_service(service: str) -> str:
        normalized = str(service or "").strip().lower()
        if not normalized:
            raise ValueError("Runtime service name must not be empty.")
        return normalized
