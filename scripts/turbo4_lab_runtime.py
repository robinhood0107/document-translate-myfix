#!/usr/bin/env python3
"""Lab-only TurboQuant Gemma runtime adapter.

This module is intentionally kept out of the product runtime contract.  It
allows a benchmark subprocess to replace its already-created
``LocalGemmaRuntimeManager`` with a subclass that starts only a uniquely named
loopback-only lab container when StageBatchedProcessor asks for Gemma after the
inpainter release gate.  It never touches the product Gemma container.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable, Mapping
from urllib.error import URLError
from urllib.request import Request, urlopen

from modules.translation.local_runtime import LocalGemmaRuntimeManager
from modules.utils.exceptions import OperationCancelledError
from modules.utils.gpu_metrics import query_gpu_metrics


LAB_CONTAINER_PREFIX = "ct-gemma-turbo4-"
OFFLOAD_LAB_CONTAINER_PREFIX = "ct-gemma-offload-"
LAB_PROTOCOL_LABEL = "com.comictranslate.benchmark-protocol"
LAB_ROLE_LABEL = "com.comictranslate.benchmark-role"
LAB_COMMIT_LABEL = "com.comictranslate.turbo4-fork-commit"
_SAFE_DOCKER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SAFE_MODEL_NAME = re.compile(r"^[^/\\\x00-\x1f]+\.gguf$", re.IGNORECASE)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class Turbo4LabRuntimeError(RuntimeError):
    """The isolated Turbo4 lab runtime contract was not satisfied."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _run(
    command: list[str],
    *,
    check: bool = True,
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise Turbo4LabRuntimeError(
            f"Command failed ({completed.returncode}): {Path(command[0]).name}\n"
            f"{detail[-4096:]}"
        )
    return completed


def _http_json(url: str, *, timeout: float = 3.0) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {"value": payload}


def _gpu_used_mib() -> int | None:
    metrics = query_gpu_metrics()
    primary = metrics.get("primary") if isinstance(metrics, Mapping) else None
    if not isinstance(primary, Mapping):
        return None
    try:
        return int(primary["memory_used_mb"])
    except (KeyError, TypeError, ValueError):
        return None


def _expected_model_identifiers(model_name: str) -> set[str]:
    """Accept only the two llama.cpp spellings for the mounted model file."""

    return {model_name, f"/models/{model_name}"}


@dataclass(frozen=True)
class Turbo4LabRuntimeConfig:
    protocol_version: str
    candidate_key: str
    image_ref: str
    model_volume: str
    model_name: str
    model_sha256: str
    port: int
    container_name: str
    cache_type_v: str
    fork_commit: str
    image_id: str = ""
    release_timeout_sec: float = 45.0
    release_residual_mib: int = 64

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
    ) -> "Turbo4LabRuntimeConfig":
        config = cls(
            protocol_version=str(raw.get("protocol_version", "") or "").strip(),
            candidate_key=str(raw.get("candidate_key", "") or "").strip(),
            image_ref=str(raw.get("image_ref", "") or "").strip(),
            model_volume=str(raw.get("model_volume", "") or "").strip(),
            model_name=str(raw.get("model_name", "") or "").strip(),
            model_sha256=str(raw.get("model_sha256", "") or "").strip().lower(),
            port=int(raw.get("port", 0) or 0),
            container_name=str(raw.get("container_name", "") or "").strip(),
            cache_type_v=str(raw.get("cache_type_v", "") or "").strip().lower(),
            fork_commit=str(raw.get("fork_commit", "") or "").strip().lower(),
            image_id=str(raw.get("image_id", "") or "").strip(),
            release_timeout_sec=float(raw.get("release_timeout_sec", 45.0) or 45.0),
            release_residual_mib=int(raw.get("release_residual_mib", 64) or 64),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.protocol_version:
            raise Turbo4LabRuntimeError("Turbo4 lab protocol_version is required.")
        if not self.candidate_key or not _SAFE_DOCKER_NAME.fullmatch(self.candidate_key):
            raise Turbo4LabRuntimeError("Turbo4 candidate_key is unsafe.")
        if not self.image_ref:
            raise Turbo4LabRuntimeError("Turbo4 image_ref is required.")
        if not _SAFE_DOCKER_NAME.fullmatch(self.model_volume):
            raise Turbo4LabRuntimeError("Turbo4 model_volume is unsafe.")
        if not _SAFE_MODEL_NAME.fullmatch(self.model_name):
            raise Turbo4LabRuntimeError("Turbo4 model_name is unsafe.")
        if not _SHA256.fullmatch(self.model_sha256):
            raise Turbo4LabRuntimeError("Turbo4 model_sha256 must be 64 lowercase hex.")
        if not 1024 <= int(self.port) <= 65535:
            raise Turbo4LabRuntimeError("Turbo4 lab port must be between 1024 and 65535.")
        if not self.container_name.startswith(LAB_CONTAINER_PREFIX) or not _SAFE_DOCKER_NAME.fullmatch(
            self.container_name
        ):
            raise Turbo4LabRuntimeError("Turbo4 container name must use the lab-only prefix.")
        if self.cache_type_v not in {"f16", "turbo4"}:
            raise Turbo4LabRuntimeError("Turbo4 lab supports only f16 or turbo4 V cache.")
        if self.cache_type_v == "turbo4" and not _COMMIT.fullmatch(self.fork_commit):
            raise Turbo4LabRuntimeError("Turbo4 candidate requires a 40-character fork commit.")
        if self.cache_type_v == "f16" and self.fork_commit and not _COMMIT.fullmatch(
            self.fork_commit
        ):
            raise Turbo4LabRuntimeError("Fork control commit must be 40 lowercase hex.")
        if self.release_timeout_sec <= 0 or self.release_residual_mib < 0:
            raise Turbo4LabRuntimeError("Turbo4 release gate settings are invalid.")

    @property
    def endpoint_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def command(self) -> list[str]:
        # This fixed command deliberately has no draft, MTP, or n-gram option.
        return [
            "-m",
            f"/models/{self.model_name}",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
            "-c",
            "4096",
            "-np",
            "1",
            "-t",
            "10",
            "-b",
            "2048",
            "-ub",
            "512",
            "--n-gpu-layers",
            "23",
            "--fit",
            "off",
            "-fa",
            "on",
            "-ctk",
            "f16",
            "-ctv",
            self.cache_type_v,
            "--kv-offload",
            "--swa-full",
            "--jinja",
            "--reasoning",
            "off",
            "--reasoning-budget",
            "0",
            "--reasoning-format",
            "none",
            "--metrics",
            "--perf",
            "--cache-ram",
            "0",
            "--spec-type",
            "none",
        ]

    @property
    def command_sha256(self) -> str:
        return _canonical_sha256(self.command)


@dataclass(frozen=True)
class ShippingOffloadLabRuntimeConfig:
    """Fixed shipping-image runtime for the CPU-MoE/KV residency lab.

    This intentionally accepts no arbitrary server arguments.  The sibling
    lab may vary only the two residency controls under test while keeping the
    pinned b10133 command otherwise identical to the shipping F16 control.
    """

    protocol_version: str
    candidate_key: str
    image_ref: str
    model_volume: str
    model_name: str
    model_sha256: str
    port: int
    container_name: str
    kv_offload: bool
    n_cpu_moe: int
    image_id: str = ""
    n_gpu_layers: int = 23
    release_timeout_sec: float = 45.0
    # Windows WDDM background usage can drift by roughly 100 MiB after a
    # completely removed container.  This still cannot mask a model-sized
    # allocation, while avoiding a false GPU-release failure in this lab.
    release_residual_mib: int = 128
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    fork_commit: str = ""

    def validate(self) -> None:
        if not self.protocol_version:
            raise Turbo4LabRuntimeError("Offload lab protocol_version is required.")
        if not self.candidate_key or not _SAFE_DOCKER_NAME.fullmatch(self.candidate_key):
            raise Turbo4LabRuntimeError("Offload lab candidate_key is unsafe.")
        if not self.image_ref:
            raise Turbo4LabRuntimeError("Offload lab image_ref is required.")
        if not _SAFE_DOCKER_NAME.fullmatch(self.model_volume):
            raise Turbo4LabRuntimeError("Offload lab model_volume is unsafe.")
        if not _SAFE_MODEL_NAME.fullmatch(self.model_name):
            raise Turbo4LabRuntimeError("Offload lab model_name is unsafe.")
        if not _SHA256.fullmatch(self.model_sha256):
            raise Turbo4LabRuntimeError("Offload lab model_sha256 must be 64 lowercase hex.")
        if not 1024 <= int(self.port) <= 65535:
            raise Turbo4LabRuntimeError("Offload lab port must be between 1024 and 65535.")
        if not self.container_name.startswith(OFFLOAD_LAB_CONTAINER_PREFIX) or not _SAFE_DOCKER_NAME.fullmatch(
            self.container_name
        ):
            raise Turbo4LabRuntimeError("Offload lab container must use the offload-only prefix.")
        if self.cache_type_k != "f16" or self.cache_type_v != "f16":
            raise Turbo4LabRuntimeError("Offload lab must keep F16/F16 KV cache types.")
        if self.fork_commit:
            raise Turbo4LabRuntimeError("Offload lab must use the pinned shipping image, not a fork.")
        if not isinstance(self.kv_offload, bool):
            raise Turbo4LabRuntimeError("Offload lab kv_offload must be boolean.")
        if not 0 <= int(self.n_cpu_moe) <= 64:
            raise Turbo4LabRuntimeError("Offload lab n_cpu_moe must be between 0 and 64.")
        if int(self.n_gpu_layers) != 23:
            raise Turbo4LabRuntimeError("Offload lab must keep n_gpu_layers at 23.")
        if self.release_timeout_sec <= 0 or self.release_residual_mib < 0:
            raise Turbo4LabRuntimeError("Offload lab release gate settings are invalid.")

    @property
    def endpoint_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def command(self) -> list[str]:
        command = [
            "-m",
            f"/models/{self.model_name}",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
            "-c",
            "4096",
            "-np",
            "1",
            "-t",
            "10",
            "-b",
            "2048",
            "-ub",
            "512",
            "--n-gpu-layers",
            str(self.n_gpu_layers),
            "--fit",
            "off",
            "-fa",
            "on",
            "-ctk",
            self.cache_type_k,
            "-ctv",
            self.cache_type_v,
            "--kv-offload" if self.kv_offload else "--no-kv-offload",
            "--swa-full",
            "--jinja",
            "--reasoning",
            "off",
            "--reasoning-budget",
            "0",
            "--reasoning-format",
            "none",
            "--metrics",
            "--perf",
            "--cache-ram",
            "0",
            "--spec-type",
            "none",
        ]
        if self.n_cpu_moe:
            command.extend(("--n-cpu-moe", str(self.n_cpu_moe)))
        return command

    @property
    def command_sha256(self) -> str:
        return _canonical_sha256(self.command)


def _assert_lab_name(name: str) -> None:
    if not str(name).startswith((LAB_CONTAINER_PREFIX, OFFLOAD_LAB_CONTAINER_PREFIX)):
        raise Turbo4LabRuntimeError(
            f"Refusing to operate on a non-lab container: {name!r}"
        )


def _inspect_container(name: str) -> dict[str, Any]:
    _assert_lab_name(name)
    completed = _run(
        ["docker", "inspect", name, "--format", "{{json .}}"],
        check=False,
        timeout=20,
    )
    if completed.returncode != 0:
        return {}
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _container_swap_peak_bytes(name: str) -> int | None:
    _assert_lab_name(name)
    completed = _run(
        [
            "docker",
            "exec",
            name,
            "/bin/sh",
            "-ec",
            (
                "if test -r /sys/fs/cgroup/memory.swap.peak; then "
                "cat /sys/fs/cgroup/memory.swap.peak; "
                "elif test -r /sys/fs/cgroup/memory.swap.current; then "
                "cat /sys/fs/cgroup/memory.swap.current; "
                "else exit 44; fi"
            ),
        ],
        check=False,
        timeout=15,
    )
    text = (completed.stdout or "").strip().splitlines()
    if completed.returncode != 0 or not text:
        return None
    try:
        return max(0, int(text[-1]))
    except ValueError:
        return None


class Turbo4LabRuntimeManager(LocalGemmaRuntimeManager):
    """A ``LocalGemmaRuntimeManager`` subclass scoped to one lab container."""

    def __init__(
        self,
        *,
        inner: LocalGemmaRuntimeManager | None,
        config: Turbo4LabRuntimeConfig,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.config = config
        self._lab_active = False
        self._pre_start_gpu_mib: int | None = None
        self._events: list[dict[str, Any]] = []
        self._last_health: dict[str, Any] = {}
        self._last_models: dict[str, Any] = {}
        self._last_release: dict[str, Any] = {}

    def should_manage_server(self, _settings_page: Any) -> bool:
        return True

    def validate_server(self, _settings_page: Any) -> None:
        self.config.validate()

    def get_translation_cache_identity(self, _settings_page: Any) -> dict[str, Any]:
        return {
            "managed": True,
            "lab_only": True,
            "runtime_fingerprint": _canonical_sha256(
                {
                    "protocol": self.config.protocol_version,
                    "candidate": self.config.candidate_key,
                    "image_ref": self.config.image_ref,
                    "image_id": self.config.image_id,
                    "fork_commit": self.config.fork_commit,
                    "model_sha256": self.config.model_sha256,
                    "command_sha256": self.config.command_sha256,
                }
            ),
            "model_name": self.config.model_name,
            "model_sha256": self.config.model_sha256,
            "runtime_image_ref": self.config.image_ref,
            "runtime_image_id": self.config.image_id,
            "runtime_command_sha256": self.config.command_sha256,
        }

    def ensure_server(
        self,
        _settings_page: Any,
        *,
        timeout_sec: int = 420,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> None:
        with self._lock:
            self.config.validate()
            if self._is_cancelled(cancel_checker):
                raise OperationCancelledError("Cancelled before Turbo4 lab startup.")
            if self._lab_active and self._probe_ready():
                self._emit(progress_callback, "completed", "readiness_cache")
                return
            existing = _inspect_container(self.config.container_name)
            if existing:
                self._remove_exact_container()
            self._pre_start_gpu_mib = _gpu_used_mib()
            if self._pre_start_gpu_mib is None:
                raise Turbo4LabRuntimeError("GPU telemetry is unavailable before Turbo4 startup.")
            self._emit(progress_callback, "starting", "container_start")
            started = time.perf_counter()
            try:
                _run(self._docker_run_command(), timeout=90)
                self._wait_for_ready(
                    timeout_sec=float(timeout_sec),
                    cancel_checker=cancel_checker,
                )
                self._validate_model_identity()
                # Match the product managed runtime: health alone is not a
                # comparable ready state because its first chat completion
                # performs CUDA/model initialization.  This prewarm is kept
                # outside the recorded translation request ledger.
                self._emit(progress_callback, "starting", "chat_prewarm")
                self._prewarm_chat_completion(
                    self.config.endpoint_url,
                    self.config.model_name,
                    cancel_checker=cancel_checker,
                )
                self._emit(progress_callback, "completed", "chat_prewarm")
            except BaseException as startup_error:
                release_started = time.perf_counter()
                cleanup_error: BaseException | None = None
                try:
                    self._remove_exact_container()
                except BaseException as exc:
                    # Do not let a failed Docker stop/rm skip the driver-wide
                    # return check.  The original startup failure is retained
                    # below when cleanup and release both succeed.
                    cleanup_error = exc
                finally:
                    self._lab_active = False
                release = self._wait_for_gpu_release(started_at=release_started)
                self._last_release = release
                self._events.append({"event": "released_after_startup_failure", **release})
                if cleanup_error is not None:
                    raise Turbo4LabRuntimeError(
                        "Turbo4 lab startup failed and its container could not be removed."
                    ) from cleanup_error
                if not release.get("observed", False):
                    raise Turbo4LabRuntimeError(
                        "Turbo4 lab startup failed and GPU release was not confirmed."
                    ) from startup_error
                raise
            self._lab_active = True
            self._events.append(
                {
                    "event": "model_ready",
                    "elapsed_sec": round(time.perf_counter() - started, 6),
                }
            )
            self._emit(progress_callback, "completed", "model_ready")

    def shutdown(self) -> dict[str, Any]:
        with self._lock:
            if not self._lab_active and not _inspect_container(self.config.container_name):
                return dict(self._last_release or {"status": "not-started"})
            started = time.perf_counter()
            peak = _container_swap_peak_bytes(self.config.container_name)
            self._remove_exact_container()
            release = self._wait_for_gpu_release(started_at=started)
            release["cgroup_swap_peak_bytes"] = peak
            self._last_release = release
            self._lab_active = False
            self._events.append({"event": "released", **release})
            if not release.get("observed", False):
                raise Turbo4LabRuntimeError(
                    "Turbo4 lab container stopped but GPU release was not confirmed."
                )
            return dict(release)

    def release_for_handoff(self) -> dict[str, Any]:
        return self.shutdown()

    def recover_after_parent_abort(self, *, baseline_gpu_mib: int | None) -> dict[str, Any]:
        """Remove this exact lab container after an externally killed runner.

        A full-auto benchmark child normally calls ``shutdown()`` itself.  If
        the parent has to terminate that child, the parent owns recovery and
        supplies the GPU baseline captured before the child started.  This
        method remains scoped to the exact lab-only name in ``config``.
        """

        self.config.validate()
        if baseline_gpu_mib is None or int(baseline_gpu_mib) < 0:
            raise Turbo4LabRuntimeError(
                "Parent-abort recovery requires a pre-start GPU baseline."
            )
        self._pre_start_gpu_mib = int(baseline_gpu_mib)
        if _inspect_container(self.config.container_name):
            return self.shutdown()
        started = time.perf_counter()
        release = self._wait_for_gpu_release(started_at=started)
        self._last_release = release
        self._events.append({"event": "released_after_parent_abort", **release})
        if not release.get("observed", False):
            raise Turbo4LabRuntimeError(
                "Parent-abort recovery found no lab container but GPU release was not confirmed."
            )
        return dict(release)

    def evidence(self) -> dict[str, Any]:
        inspection = _inspect_container(self.config.container_name)
        return {
            "config": {
                "protocol_version": self.config.protocol_version,
                "candidate_key": self.config.candidate_key,
                "image_ref": self.config.image_ref,
                "image_id": self.config.image_id,
                "model_volume": self.config.model_volume,
                "model_name": self.config.model_name,
                "model_sha256": self.config.model_sha256,
                "fork_commit": self.config.fork_commit,
                "cache_type_v": self.config.cache_type_v,
                "command": self.config.command,
                "command_sha256": self.config.command_sha256,
            },
            "container": inspection,
            "health": self._last_health,
            "models": self._last_models,
            "events": list(self._events),
            "release": dict(self._last_release),
        }

    def _docker_run_command(self) -> list[str]:
        return [
            "docker",
            "run",
            "--detach",
            "--name",
            self.config.container_name,
            "--pull",
            "never",
            "--gpus",
            "all",
            "--publish",
            f"127.0.0.1:{self.config.port}:8080",
            "--mount",
            (
                "type=volume,source="
                f"{self.config.model_volume},target=/models,readonly"
            ),
            "--label",
            f"{LAB_PROTOCOL_LABEL}={self.config.protocol_version}",
            "--label",
            f"{LAB_ROLE_LABEL}={self.config.candidate_key}",
            "--label",
            f"{LAB_COMMIT_LABEL}={self.config.fork_commit or 'shipping'}",
            self.config.image_ref,
            *self.config.command,
        ]

    def _probe_ready(self) -> bool:
        try:
            self._last_health = _http_json(
                f"http://127.0.0.1:{self.config.port}/health"
            )
            return True
        except (OSError, URLError, ValueError, json.JSONDecodeError):
            return False

    def _wait_for_ready(
        self,
        *,
        timeout_sec: float,
        cancel_checker: Callable[[], bool] | None,
    ) -> None:
        started = time.monotonic()
        last_error = ""
        while time.monotonic() - started < max(1.0, timeout_sec):
            if self._is_cancelled(cancel_checker):
                raise OperationCancelledError("Cancelled while starting Turbo4 lab runtime.")
            if self._probe_ready():
                return
            try:
                _http_json(f"http://127.0.0.1:{self.config.port}/v1/models")
            except (OSError, URLError, ValueError, json.JSONDecodeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            elapsed = time.monotonic() - started
            time.sleep(0.1 if elapsed < 2.0 else (0.25 if elapsed < 10.0 else 1.0))
        raise Turbo4LabRuntimeError(
            "Timed out waiting for the Turbo4 lab runtime: "
            f"{last_error or 'health unavailable'}"
        )

    def _validate_model_identity(self) -> None:
        self._last_models = _http_json(
            f"http://127.0.0.1:{self.config.port}/v1/models"
        )
        raw_models = self._last_models.get("data")
        rows = raw_models if isinstance(raw_models, list) else []
        identifiers = {
            str(row.get("id", "") or "")
            for row in rows
            if isinstance(row, Mapping)
        }
        expected_identifiers = _expected_model_identifiers(self.config.model_name)
        if not identifiers or not identifiers <= expected_identifiers:
            raise Turbo4LabRuntimeError(
                "Turbo4 model identity mismatch: " + ", ".join(sorted(identifiers))
            )
        inspection = _inspect_container(self.config.container_name)
        config = inspection.get("Config") if isinstance(inspection, Mapping) else None
        mounts = inspection.get("Mounts") if isinstance(inspection, Mapping) else None
        command = list(config.get("Cmd") or []) if isinstance(config, Mapping) else []
        labels = config.get("Labels") if isinstance(config, Mapping) else {}
        image_id = str(inspection.get("Image", "") or "")
        expected_mount = next(
            (
                row
                for row in (mounts if isinstance(mounts, list) else [])
                if isinstance(row, Mapping)
                and row.get("Destination") == "/models"
            ),
            None,
        )
        failures: list[str] = []
        if list(command) != self.config.command:
            failures.append("command")
        if not isinstance(expected_mount, Mapping):
            failures.append("models-mount")
        else:
            if expected_mount.get("Type") != "volume":
                failures.append("models-mount-type")
            if expected_mount.get("Name") != self.config.model_volume:
                failures.append("models-volume")
            if bool(expected_mount.get("RW", True)):
                failures.append("models-mount-readonly")
        if not isinstance(labels, Mapping) or labels.get(LAB_PROTOCOL_LABEL) != self.config.protocol_version:
            failures.append("protocol-label")
        elif labels.get(LAB_ROLE_LABEL) != self.config.candidate_key:
            failures.append("role-label")
        elif labels.get(LAB_COMMIT_LABEL) != (self.config.fork_commit or "shipping"):
            failures.append("fork-commit-label")
        if self.config.image_id and image_id != self.config.image_id:
            failures.append("image-id")
        if failures:
            raise Turbo4LabRuntimeError(
                "Turbo4 lab container contract mismatch: " + ", ".join(failures)
            )

    def _remove_exact_container(self) -> None:
        _assert_lab_name(self.config.container_name)
        _run(
            ["docker", "stop", "--timeout", "10", self.config.container_name],
            check=False,
            timeout=30,
        )
        _run(
            ["docker", "rm", self.config.container_name],
            check=False,
            timeout=30,
        )

    def _wait_for_gpu_release(self, *, started_at: float) -> dict[str, Any]:
        baseline = self._pre_start_gpu_mib
        if baseline is None:
            return {
                "status": "unavailable",
                "observed": False,
                "elapsed_sec": round(time.perf_counter() - started_at, 6),
                "baseline_gpu_mib": None,
                "after_gpu_mib": _gpu_used_mib(),
            }
        deadline = time.monotonic() + self.config.release_timeout_sec
        after: int | None = None
        while time.monotonic() < deadline:
            after = _gpu_used_mib()
            if after is not None and after <= baseline + self.config.release_residual_mib:
                return {
                    "status": "released",
                    "observed": True,
                    "elapsed_sec": round(time.perf_counter() - started_at, 6),
                    "baseline_gpu_mib": baseline,
                    "after_gpu_mib": after,
                    "allowed_gpu_mib": baseline + self.config.release_residual_mib,
                }
            time.sleep(0.1)
        return {
            "status": "timeout",
            "observed": False,
            "elapsed_sec": round(time.perf_counter() - started_at, 6),
            "baseline_gpu_mib": baseline,
            "after_gpu_mib": after,
            "allowed_gpu_mib": baseline + self.config.release_residual_mib,
        }

    def _emit(
        self,
        callback: Callable[[dict[str, Any]], None] | None,
        status: str,
        step_key: str,
    ) -> None:
        event = {
            "phase": "gemma_startup",
            "service": "gemma",
            "status": status,
            "step_key": step_key,
            "message": "Turbo4 lab Gemma runtime",
            "detail": self.config.candidate_key,
        }
        self._events.append({"event": "progress", **event})
        if callback is not None:
            callback(event)


class InstalledTurbo4LabRuntime:
    """Restores the original manager after best-effort lab cleanup."""

    def __init__(self, window: Any, original: LocalGemmaRuntimeManager, adapter: Turbo4LabRuntimeManager) -> None:
        self.window = window
        self.original = original
        self.adapter = adapter
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.adapter.shutdown()
        finally:
            self.window.local_translation_runtime_manager = self.original
            self.closed = True


def install_turbo4_lab_runtime_adapter(
    window: Any,
    config: Mapping[str, Any],
) -> InstalledTurbo4LabRuntime:
    """Install the lab adapter on one benchmark window only.

    No container is started here.  The first start can only happen later when
    StageBatchedProcessor invokes ``ensure_server`` during translation.
    """

    original = getattr(window, "local_translation_runtime_manager", None)
    if not isinstance(original, LocalGemmaRuntimeManager):
        raise Turbo4LabRuntimeError(
            "Turbo4 lab requires the product LocalGemmaRuntimeManager instance."
        )
    adapter = Turbo4LabRuntimeManager(
        inner=original,
        config=Turbo4LabRuntimeConfig.from_mapping(config),
    )
    window.local_translation_runtime_manager = adapter
    return InstalledTurbo4LabRuntime(window, original, adapter)
