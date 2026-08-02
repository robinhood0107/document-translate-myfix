#!/usr/bin/env python3
"""Lab-only TurboQuant KV-V ``turbo4`` verification for Gemma.

The runner intentionally has no product promotion path.  It fixes one
TurboQuant fork commit, validates a same-fork F16 control before comparing the
shipping b10133 image, writes raw inputs/responses/resources only into the
managed private archive, and never launches active Paddle+Gemma residency.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_serving_scheduler_matrix import (  # noqa: E402
    DEFAULT_MAX_ROUNDS,
    GPU_BACKGROUND_LIMIT_MIB,
    PADDLE_MEASURED_PEAK_MIB,
    _gpu_snapshot,
    _windows_available_bytes,
    _wsl_swap_used_bytes,
    canonical_full_snapshot_sha256,
    residency_preflight,
    resource_preflight,
    should_continue_adaptive,
    summarize_pairs,
)
from turbo4_lab_runtime import (  # noqa: E402
    LAB_CONTAINER_PREFIX,
    OFFLOAD_LAB_CONTAINER_PREFIX,
    Turbo4LabRuntimeConfig,
    Turbo4LabRuntimeError,
    Turbo4LabRuntimeManager,
)
from modules.utils.exceptions import OperationCancelledError  # noqa: E402
from validation_artifact_harness import (  # noqa: E402
    ManagedArtifactRun,
    select_managed_output_directory,
)


PROTOCOL_VERSION = "gemma-turbo4-kv-v1"
FAMILY_NAME = "gemma-turbo4-kv"
ARTIFACT_CATEGORY = "10-gemma-translation"
PROTOCOL_PATH = ROOT / "benchmarks" / "gemma_turbo4_kv" / "protocol-v1.json"
DOCKERFILE_PATH = ROOT / "benchmarks" / "gemma_turbo4_kv" / "Dockerfile.turboquant"
FORK_REPOSITORY = "https://github.com/TheTom/llama-cpp-turboquant.git"
FORK_REF = "feature/turboquant-kv-cache"
FORK_COMMIT = "8a891f4b566efdbd3cea92fafee3227a0a267683"
SHIPPING_IMAGE = (
    "ghcr.io/ggml-org/llama.cpp@sha256:"
    "22e0e3bfe967af4fd1df6a918022abbfd4e72e4d40a4769e616a4176790acbcb"
)
SHIPPING_COMMIT = "ff067f76dd8e9e05f0528056f1274adf01a54d70"
SHIPPING_BUILD = "b10133"
MODEL_VOLUME = "comic-translate-gemma-models-v2"
MODEL_NAME = "gemma-4-26B-IQ4_NL.gguf"
MODEL_SHA256 = "768a89b94209243b333b2e074b928fe51ea208ebdad6424a510bd73e5cb4d0b8"
READY_MANIFEST = ".comic-translate-gemma-ready-v2.json"
DEFAULT_PORT = 18081
DEFAULT_SEED = 20260801
DEFAULT_INITIAL_ROUNDS = 2
R3_RESIDENCY_THRESHOLD = 0.90
REPLAY_CHUNK_SIZE = 6
REPLAY_MAX_COMPLETION_TOKENS = 512
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_FATAL_RESOURCE_FAILURES = frozenset(
    {"gpu_release_unconfirmed", "oom_detected", "container_orphan"}
)


class Turbo4BenchmarkError(RuntimeError):
    """Raised when a Turbo4 lab step cannot satisfy its fixed contract."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run(
    command: Sequence[str],
    *,
    check: bool = True,
    timeout: float = 120.0,
    cwd: Path = ROOT,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip()
        raise Turbo4BenchmarkError(
            f"Command failed ({completed.returncode}): {Path(command[0]).name}\n"
            f"{message[-4096:]}"
        )
    return completed


def _http_json(
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    timeout: float = 180.0,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urlopen(request, timeout=timeout) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    return decoded if isinstance(decoded, dict) else {"value": decoded}


def _find_powershell() -> str | None:
    return shutil.which("powershell.exe") or shutil.which("powershell")


def query_shared_gpu_used_mib() -> float | None:
    """Read the Windows shared-GPU counter without changing host state."""

    executable = _find_powershell()
    if not executable:
        return None
    command = (
        "$samples=(Get-Counter "
        "'\\GPU Process Memory(*)\\Shared Usage' "
        "-ErrorAction Stop).CounterSamples;"
        "$sum=($samples|Measure-Object -Property CookedValue -Sum).Sum;"
        "[Console]::WriteLine([Math]::Round($sum/1MB,3))"
    )
    completed = _run(
        [
            executable,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            command,
        ],
        check=False,
        timeout=15,
    )
    if completed.returncode != 0:
        return None
    try:
        return float((completed.stdout or "").strip())
    except ValueError:
        return None


def _running_container_names() -> list[str]:
    completed = _run(
        ["docker", "ps", "--format", "{{.Names}}"],
        check=False,
        timeout=20,
    )
    if completed.returncode != 0:
        return ["docker-ps-unavailable"]
    return [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]


def _lab_container_name(candidate_key: str, *, stage: str, round_index: int) -> str:
    if not _SAFE_KEY_RE.fullmatch(candidate_key):
        raise Turbo4BenchmarkError(f"Unsafe Turbo4 candidate key: {candidate_key!r}")
    digest = hashlib.sha256(
        f"{os.getpid()}:{time.time_ns()}:{candidate_key}:{stage}:{round_index}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{LAB_CONTAINER_PREFIX}{candidate_key}-{digest}"


@dataclass(frozen=True)
class TurboCandidate:
    key: str
    role: str
    image_ref: str
    cache_type_v: str
    fork_commit: str = ""

    def validate(self) -> None:
        if not _SAFE_KEY_RE.fullmatch(self.key):
            raise Turbo4BenchmarkError(f"Unsafe candidate key: {self.key!r}")
        if self.role not in {"shipping", "fork"}:
            raise Turbo4BenchmarkError("Turbo4 candidate role must be shipping or fork.")
        if not self.image_ref:
            raise Turbo4BenchmarkError("Turbo4 candidate image is required.")
        if self.cache_type_v not in {"f16", "turbo4"}:
            raise Turbo4BenchmarkError("Only F16 and Turbo4 V cache candidates are allowed.")
        if self.role == "fork" and not _COMMIT_RE.fullmatch(self.fork_commit):
            raise Turbo4BenchmarkError("Fork candidate must pin a 40-character commit.")
        if self.role == "shipping" and self.cache_type_v != "f16":
            raise Turbo4BenchmarkError("Shipping b10133 is F16/F16 control only.")
        if self.cache_type_v == "turbo4" and self.role != "fork":
            raise Turbo4BenchmarkError("Turbo4 is allowed only in the pinned fork image.")


def candidate_catalog(*, turbo_image: str, fork_commit: str) -> dict[str, TurboCandidate]:
    catalog = {
        "shipping-f16": TurboCandidate(
            key="shipping-f16",
            role="shipping",
            image_ref=SHIPPING_IMAGE,
            cache_type_v="f16",
        ),
        "fork-f16": TurboCandidate(
            key="fork-f16",
            role="fork",
            image_ref=turbo_image,
            cache_type_v="f16",
            fork_commit=fork_commit,
        ),
        "turbo4": TurboCandidate(
            key="turbo4",
            role="fork",
            image_ref=turbo_image,
            cache_type_v="turbo4",
            fork_commit=fork_commit,
        ),
    }
    for candidate in catalog.values():
        candidate.validate()
    return catalog


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Turbo4BenchmarkError("Turbo4 protocol must be a JSON object.")
    fork = payload.get("fork")
    shipping = payload.get("shipping")
    model = payload.get("model")
    fixed = payload.get("fixed_server")
    safety = payload.get("safety")
    if not all(isinstance(item, Mapping) for item in (fork, shipping, model, fixed, safety)):
        raise Turbo4BenchmarkError("Turbo4 protocol is missing a required section.")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise Turbo4BenchmarkError("Unexpected Turbo4 protocol version.")
    if (
        fork.get("repository") != FORK_REPOSITORY
        or fork.get("ref") != FORK_REF
        or fork.get("commit") != FORK_COMMIT
    ):
        raise Turbo4BenchmarkError("Turbo4 fork protocol is not pinned to the approved ref.")
    if shipping.get("image") != SHIPPING_IMAGE or shipping.get("commit") != SHIPPING_COMMIT:
        raise Turbo4BenchmarkError("Shipping b10133 protocol mismatch.")
    if model.get("volume") != MODEL_VOLUME or model.get("name") != MODEL_NAME:
        raise Turbo4BenchmarkError("Turbo4 model protocol mismatch.")
    if model.get("sha256") != MODEL_SHA256:
        raise Turbo4BenchmarkError("Turbo4 model SHA protocol mismatch.")
    expected_fixed = {
        "context_size": 4096,
        "n_gpu_layers": 23,
        "n_parallel": 1,
        "threads": 10,
        "batch_size": 2048,
        "ubatch_size": 512,
        "cache_type_k": "f16",
        "cache_type_v_baseline": "f16",
        "cache_type_v_candidate": "turbo4",
        "seed": DEFAULT_SEED,
    }
    if any(fixed.get(key) != value for key, value in expected_fixed.items()):
        raise Turbo4BenchmarkError("Turbo4 fixed server contract mismatch.")
    if float(safety.get("r3_residency_threshold", 0.0)) != R3_RESIDENCY_THRESHOLD:
        raise Turbo4BenchmarkError("Turbo4 R3 threshold must be 90%.")
    if safety.get("active_r3_execution") is not False:
        raise Turbo4BenchmarkError("Turbo4 lab must never enable active R3 execution.")
    forbidden = {str(value).lower() for value in payload.get("forbidden", [])}
    if not {"qat", "mtp", "draft", "ngram", "speculative", "new_gguf"} <= forbidden:
        raise Turbo4BenchmarkError("Turbo4 protocol must explicitly reject old candidates.")
    return payload


def verify_fork_ref(*, fork_commit: str = FORK_COMMIT) -> dict[str, str]:
    if not _COMMIT_RE.fullmatch(fork_commit):
        raise Turbo4BenchmarkError("Fork commit must be exactly 40 lowercase hex.")
    completed = _run(
        ["git", "ls-remote", FORK_REPOSITORY, f"refs/heads/{FORK_REF}"],
        timeout=45,
    )
    fields = (completed.stdout or "").strip().split()
    if len(fields) < 1 or not _COMMIT_RE.fullmatch(fields[0]):
        raise Turbo4BenchmarkError("Unable to resolve the TurboQuant fork ref.")
    resolved = fields[0]
    if resolved != fork_commit:
        raise Turbo4BenchmarkError(
            "TurboQuant fork ref moved; protocol commit does not match the live ref."
        )
    return {"repository": FORK_REPOSITORY, "ref": FORK_REF, "commit": resolved}


def _inspect_image(image_ref: str) -> dict[str, Any]:
    completed = _run(
        ["docker", "image", "inspect", image_ref, "--format", "{{json .}}"],
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        raise Turbo4BenchmarkError(f"Docker image is unavailable: {image_ref}")
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise Turbo4BenchmarkError("Docker image inspect returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise Turbo4BenchmarkError("Docker image inspect returned a non-object.")
    config = payload.get("Config") if isinstance(payload.get("Config"), Mapping) else {}
    labels = config.get("Labels") if isinstance(config, Mapping) else {}
    return {
        "reference": image_ref,
        "id": str(payload.get("Id", "") or ""),
        "repo_digests": list(payload.get("RepoDigests") or []),
        "created": str(payload.get("Created", "") or ""),
        "labels": dict(labels) if isinstance(labels, Mapping) else {},
    }


def _run_docker_build(
    *,
    image_tag: str,
    fork_commit: str,
    output_dir: Path,
) -> dict[str, Any]:
    command = [
        "docker",
        "build",
        "--file",
        str(DOCKERFILE_PATH),
        "--tag",
        image_tag,
        "--build-arg",
        f"TURBOQUANT_REPOSITORY={FORK_REPOSITORY}",
        "--build-arg",
        f"TURBOQUANT_COMMIT={fork_commit}",
        # The Dockerfile clones the pinned fork itself and never COPYs product
        # files.  Keeping this context to the lab folder prevents a large
        # checkout, venv, or private archive from being sent to BuildKit.
        str(DOCKERFILE_PATH.parent),
    ]
    started = time.perf_counter()
    completed = _run(command, check=False, timeout=3600)
    (output_dir / "turbo4-image-build.stdout.log").write_text(
        completed.stdout or "", encoding="utf-8"
    )
    (output_dir / "turbo4-image-build.stderr.log").write_text(
        completed.stderr or "", encoding="utf-8"
    )
    if completed.returncode != 0:
        raise Turbo4BenchmarkError("Turbo4 image build failed; see private build logs.")
    return {
        "command": command,
        "elapsed_sec": round(time.perf_counter() - started, 6),
        "image": _inspect_image(image_tag),
    }


def verify_turbo_image(
    *,
    image_ref: str,
    fork_commit: str,
    output_dir: Path,
) -> dict[str, Any]:
    image = _inspect_image(image_ref)
    labels = image.get("labels") if isinstance(image.get("labels"), Mapping) else {}
    if labels.get("com.comictranslate.turbo4-fork-commit") != fork_commit:
        raise Turbo4BenchmarkError("Turbo4 image label does not pin the expected commit.")
    marker = _run(
        [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--entrypoint",
            "/bin/sh",
            image_ref,
            "-ec",
            "cat /app/TURBOQUANT_COMMIT",
        ],
        timeout=45,
    )
    if (marker.stdout or "").strip() != fork_commit:
        raise Turbo4BenchmarkError("Turbo4 image commit marker mismatch.")
    help_result = _run(
        [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--gpus",
            "all",
            "--entrypoint",
            "/app/llama-server",
            image_ref,
            "--help",
        ],
        check=False,
        timeout=60,
    )
    help_text = (help_result.stdout or "") + (help_result.stderr or "")
    (output_dir / "turbo4-llama-server-help.txt").write_text(help_text, encoding="utf-8")
    if help_result.returncode != 0 or "turbo4" not in help_text.lower():
        raise Turbo4BenchmarkError("Turbo4 image does not expose turbo4 in llama-server --help.")
    return {
        "image": image,
        "fork_commit": fork_commit,
        "help_sha256": hashlib.sha256(help_text.encode("utf-8")).hexdigest(),
        "turbo4_help_present": True,
    }


def verify_model_volume(*, output_dir: Path) -> dict[str, Any]:
    volume = _run(["docker", "volume", "inspect", MODEL_VOLUME, "--format", "{{json .}}"], timeout=30)
    try:
        volume_payload = json.loads(volume.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise Turbo4BenchmarkError("Gemma model volume inspect returned invalid JSON.") from exc
    command = [
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--mount",
        f"type=volume,source={MODEL_VOLUME},target=/models,readonly",
        "--entrypoint",
        "/bin/sh",
        SHIPPING_IMAGE,
        "-ec",
        (
            "printf '%s\\n' '__MANIFEST__'; cat /models/" + READY_MANIFEST
            + "; printf '%s\\n' '__BYTES__'; stat -c '%s' /models/" + MODEL_NAME
            + "; printf '%s\\n' '__SHA256__'; sha256sum /models/" + MODEL_NAME
        ),
    ]
    inspected = _run(command, timeout=60)
    text = inspected.stdout or ""
    if (
        "__MANIFEST__\n" not in text
        or "\n__BYTES__\n" not in text
        or "\n__SHA256__\n" not in text
    ):
        raise Turbo4BenchmarkError("Gemma volume probe returned an invalid format.")
    manifest_text, remainder = text.split("\n__BYTES__\n", 1)
    manifest_text = manifest_text.split("__MANIFEST__\n", 1)[1]
    size_text, sha256_text = remainder.split("\n__SHA256__\n", 1)
    try:
        # Windows-created ready manifests can carry a UTF-8 BOM.  It is not
        # part of the JSON document and must not turn an otherwise exact
        # model identity check into an infrastructure false reject.
        manifest = json.loads(manifest_text.removeprefix("\ufeff"))
        observed_bytes = int(size_text.strip().splitlines()[-1])
        observed_sha256 = sha256_text.strip().split()[0].lower()
    except (json.JSONDecodeError, ValueError, IndexError) as exc:
        raise Turbo4BenchmarkError("Gemma model volume manifest is invalid.") from exc
    files = manifest.get("files") if isinstance(manifest, Mapping) else None
    selected = next(
        (
            item
            for item in (files if isinstance(files, list) else [])
            if isinstance(item, Mapping) and item.get("name") == MODEL_NAME
        ),
        None,
    )
    if not isinstance(selected, Mapping):
        raise Turbo4BenchmarkError("Gemma model is missing from the ready manifest.")
    if (
        selected.get("sha256") != MODEL_SHA256
        or int(selected.get("bytes", -1)) != observed_bytes
        or not _SHA256_RE.fullmatch(observed_sha256)
        or observed_sha256 != MODEL_SHA256
    ):
        raise Turbo4BenchmarkError("Gemma model ready-manifest identity mismatch.")
    evidence = {
        "volume": volume_payload,
        "model_name": MODEL_NAME,
        "model_sha256": MODEL_SHA256,
        "observed_model_sha256": observed_sha256,
        "model_bytes": observed_bytes,
        "ready_manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
    }
    _json_write(output_dir / "model-volume-evidence.json", evidence)
    return evidence


_WINDOWS_RAM_OBSERVATIONS = {
    "windows_available_ram_unavailable",
    "windows_available_ram_below_6gib",
}


def _preflight() -> dict[str, Any]:
    base = resource_preflight()
    # This isolated lab records host-memory and swap values, but does not treat
    # them as candidate rejection conditions. It still fails closed for GPU
    # background conflicts, OOM, and orphaned containers below.
    failures = [
        failure
        for failure in (base.get("failures") or [])
        if failure not in _WINDOWS_RAM_OBSERVATIONS
    ]
    observations = [
        failure
        for failure in (base.get("failures") or [])
        if failure in _WINDOWS_RAM_OBSERVATIONS
    ]
    shared = query_shared_gpu_used_mib()
    active = _running_container_names()
    conflicts = [
        name
        for name in active
        if name in {"gemma-local-server", "paddleocr-llamacpp"}
        or name.startswith(LAB_CONTAINER_PREFIX)
        or name.startswith(OFFLOAD_LAB_CONTAINER_PREFIX)
    ]
    if shared is None:
        observations.append("windows_shared_gpu_metrics_unavailable")
    if conflicts:
        failures.append("managed_or_lab_gpu_container_active")
    return {
        **base,
        "shared_gpu_used_mib": shared,
        "active_conflicts": conflicts,
        "observations": observations,
        "failures": failures,
        "passed": not failures,
    }


def require_preflight(*, output_dir: Path) -> dict[str, Any]:
    """Record preflight state and stop only for active lab safety failures."""

    started = time.monotonic()
    result = _preflight()
    result["settle"] = {
        "attempt_count": 1,
        "waited_sec": round(time.monotonic() - started, 6),
        "host_memory_and_swap": "telemetry_only",
        "attempts": [
            {
                "at_offset_sec": 0.0,
                "scope": "full",
                "failures": list(result.get("failures") or []),
                "observations": list(result.get("observations") or []),
                "windows_available_bytes": result.get("windows_available_bytes"),
            }
        ],
    }
    _json_write(output_dir / "resource-preflight.json", result)
    if not result["passed"]:
        raise Turbo4BenchmarkError("Resource preflight failed: " + ", ".join(result["failures"]))
    return result


def _resource_emergency_reason(samples: Sequence[Mapping[str, Any]]) -> str | None:
    """Host-memory observations never cancel an in-flight lab candidate."""

    del samples
    return None


class ResourceSampler:
    """Collect private one-second GPU/RAM/swap/shared-GPU evidence."""

    def __init__(self, *, interval_sec: float = 1.0) -> None:
        self.interval_sec = max(0.5, float(interval_sec))
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._emergency_reason: str | None = None

    def sample(self) -> dict[str, Any]:
        row = {
            "at_unix": time.time(),
            "gpu": _gpu_snapshot(),
            "windows_available_bytes": _windows_available_bytes(),
            "wsl_swap_used_bytes": _wsl_swap_used_bytes(),
            "shared_gpu_used_mib": query_shared_gpu_used_mib(),
        }
        self.samples.append(row)
        if self._emergency_reason is None:
            self._emergency_reason = _resource_emergency_reason(self.samples)
        return row

    @property
    def emergency_reason(self) -> str | None:
        return self._emergency_reason

    def start(self) -> None:
        self.sample()

        def collect() -> None:
            while not self._stop.wait(self.interval_sec):
                self.sample()

        self._thread = threading.Thread(target=collect, name="ct-turbo4-resource-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self.sample()

    def summary(self) -> dict[str, Any]:
        gpu_used = [
            int((row.get("gpu") or {}).get("used_mib"))
            for row in self.samples
            if isinstance(row.get("gpu"), Mapping)
            and (row.get("gpu") or {}).get("available")
            and (row.get("gpu") or {}).get("used_mib") is not None
        ]
        windows = [
            int(row["windows_available_bytes"])
            for row in self.samples
            if row.get("windows_available_bytes") is not None
        ]
        swaps = [
            int(row["wsl_swap_used_bytes"])
            for row in self.samples
            if row.get("wsl_swap_used_bytes") is not None
        ]
        shared = [
            float(row["shared_gpu_used_mib"])
            for row in self.samples
            if row.get("shared_gpu_used_mib") is not None
        ]
        return {
            "sample_count": len(self.samples),
            "gpu_peak_mib": max(gpu_used) if gpu_used else None,
            "windows_available_min_bytes": min(windows) if windows else None,
            "wsl_swap_before_bytes": swaps[0] if swaps else None,
            "wsl_swap_after_bytes": swaps[-1] if swaps else None,
            "wsl_swap_growth_bytes": max(0, max(swaps) - swaps[0]) if swaps else None,
            "shared_gpu_before_mib": shared[0] if shared else None,
            "shared_gpu_peak_mib": max(shared) if shared else None,
            "shared_gpu_growth_mib": max(0.0, max(shared) - shared[0]) if shared else None,
        }


def resource_gate_report(
    *,
    sampler: ResourceSampler,
    cgroup_swap_peak_bytes: int | None,
) -> dict[str, Any]:
    summary = sampler.summary()
    failures: list[str] = []
    observations: list[str] = []
    if summary["windows_available_min_bytes"] is None:
        observations.append("windows_available_ram_unavailable")
    elif int(summary["windows_available_min_bytes"]) < 6 * 1024**3:
        observations.append("windows_available_ram_below_6gib")
    if summary["wsl_swap_growth_bytes"] is None:
        observations.append("wsl_swap_metrics_unavailable")
    elif int(summary["wsl_swap_growth_bytes"]) != 0:
        observations.append("wsl_swap_growth_observed")
    if summary["shared_gpu_growth_mib"] is None:
        observations.append("windows_shared_gpu_metrics_unavailable")
    elif float(summary["shared_gpu_growth_mib"]) > 0.0:
        observations.append("shared_gpu_growth_observed")
    if cgroup_swap_peak_bytes is None:
        observations.append("container_swap_metrics_unavailable")
    elif cgroup_swap_peak_bytes != 0:
        observations.append("container_swap_observed")
    return {
        **summary,
        "cgroup_swap_peak_bytes": cgroup_swap_peak_bytes,
        "observations": observations,
        "failures": failures,
        "passed": not failures,
    }


def _request_ledger(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for index, payload in enumerate(payloads):
        messages = payload.get("messages")
        schema = payload.get("response_format")
        seed = payload.get("seed")
        rows.append(
            {
                "logical_request_index": index,
                "model": str(payload.get("model", "") or ""),
                "prompt_sha256": _canonical_sha256(messages),
                "schema_sha256": _canonical_sha256(schema),
                "seed": seed,
                "payload_sha256": _canonical_sha256(payload),
            }
        )
    return {"rows": rows, "sha256": _canonical_sha256(rows)}


def _validate_translation_replay_requests(
    requests: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not requests:
        raise Turbo4BenchmarkError("Translation replay requires a non-empty requests list.")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(requests):
        if not isinstance(item, Mapping):
            raise Turbo4BenchmarkError(f"Translation replay request {index} is not an object.")
        request = dict(item)
        if request.get("model") != MODEL_NAME:
            raise Turbo4BenchmarkError("Translation replay model identity differs from the fixed IQ4_NL model.")
        if request.get("seed") != DEFAULT_SEED:
            raise Turbo4BenchmarkError("Translation replay seed must equal the fixed protocol seed.")
        if not isinstance(request.get("messages"), list) or "response_format" not in request:
            raise Turbo4BenchmarkError("Translation replay must include messages and response_format.")
        lowered = json.dumps(request, ensure_ascii=False).lower()
        if any(word in lowered for word in ("draft", "mtp", "speculative", "ngram", "qat")):
            raise Turbo4BenchmarkError("Translation replay must not request draft/MTP/QAT/speculative behavior.")
        normalized.append(request)
    return normalized


def load_translation_replay(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    requests = payload.get("requests") if isinstance(payload, Mapping) else None
    if not isinstance(requests, list):
        raise Turbo4BenchmarkError("Translation replay requires a non-empty requests list.")
    return _validate_translation_replay_requests(requests)


def build_translation_replay_from_page_snapshots(path: Path) -> list[dict[str, Any]]:
    """Build the exact successful contextual-single request order.

    The product's current Gemma path sends one request per target block while
    retaining the six-block contextual window.  This deliberately does not
    synthesize strict/recovery requests: an input that needs a retry belongs in
    the full E2E gate, not in a relaxed fixed-seed structural comparison.
    """

    from benchmark_gemma_translation_only_matrix import (
        build_engine,
        build_single_block_translation_request,
        iter_page_snapshot_chunks,
    )

    if not path.is_file():
        raise Turbo4BenchmarkError("Page snapshot replay input is unavailable.")
    engine = build_engine(
        model=MODEL_NAME,
        source_lang="Japanese",
        target_lang="Korean",
        max_tokens=REPLAY_MAX_COMPLETION_TOKENS,
    )
    requests: list[dict[str, Any]] = []
    for chunk in iter_page_snapshot_chunks(path, chunk_size=REPLAY_CHUNK_SIZE):
        texts = chunk.get("texts")
        if not isinstance(texts, list) or not texts:
            continue
        normalized_texts = [str(text) for text in texts]
        for target_index in range(len(normalized_texts)):
            payload, expected_keys = build_single_block_translation_request(
                engine,
                normalized_texts,
                target_index,
            )
            if expected_keys != ["translation"]:
                raise Turbo4BenchmarkError(
                    "Contextual-single replay did not preserve the product response schema."
                )
            request = dict(payload)
            request["seed"] = DEFAULT_SEED
            requests.append(request)
    return _validate_translation_replay_requests(requests)


def _canonical_response(response: Mapping[str, Any]) -> dict[str, Any]:
    raw_choices = response.get("choices")
    choices = raw_choices if isinstance(raw_choices, list) else []
    normalized = []
    for index, choice in enumerate(choices):
        if not isinstance(choice, Mapping):
            continue
        message = choice.get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        normalized.append(
            {
                "index": int(choice.get("index", index) or index),
                "content": str(content or ""),
                "finish_reason": str(choice.get("finish_reason", "") or ""),
            }
        )
    return {"choices": normalized}


def _replay_response_ledger(responses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [_canonical_response(item) for item in responses]
    return {"rows": rows, "sha256": _canonical_sha256(rows)}


def _runtime_config(
    candidate: TurboCandidate,
    *,
    port: int,
    stage: str,
    round_index: int,
    image_id: str,
) -> Turbo4LabRuntimeConfig:
    return Turbo4LabRuntimeConfig(
        protocol_version=PROTOCOL_VERSION,
        candidate_key=candidate.key,
        image_ref=candidate.image_ref,
        model_volume=MODEL_VOLUME,
        model_name=MODEL_NAME,
        model_sha256=MODEL_SHA256,
        port=port,
        container_name=_lab_container_name(candidate.key, stage=stage, round_index=round_index),
        cache_type_v=candidate.cache_type_v,
        fork_commit=candidate.fork_commit,
        image_id=image_id,
    )


def _read_container_logs(name: str, path: Path) -> str:
    completed = _run(["docker", "logs", "--tail", "1000", name], check=False, timeout=30)
    text = (completed.stdout or "") + (completed.stderr or "")
    path.write_text(text, encoding="utf-8")
    return text


def _assert_no_active_lab_or_product_container() -> list[str]:
    active = _running_container_names()
    return [
        name
        for name in active
        if name in {"gemma-local-server", "paddleocr-llamacpp"}
        or name.startswith(LAB_CONTAINER_PREFIX)
        or name.startswith(OFFLOAD_LAB_CONTAINER_PREFIX)
    ]


def execute_replay_once(
    candidate: TurboCandidate,
    *,
    payloads: Sequence[Mapping[str, Any]],
    run_dir: Path,
    round_index: int,
    port: int,
    image_id: str,
) -> dict[str, Any]:
    candidate.validate()
    preflight = require_preflight(output_dir=run_dir)
    config = _runtime_config(
        candidate,
        port=port,
        stage="replay",
        round_index=round_index,
        image_id=image_id,
    )
    manager = Turbo4LabRuntimeManager(inner=None, config=config)
    sampler = ResourceSampler()
    responses: list[dict[str, Any]] = []
    logs = ""
    startup_error = ""
    sampler.start()
    started = time.perf_counter()
    ready_at: float | None = None
    try:
        manager.ensure_server(
            None,
            cancel_checker=lambda: sampler.emergency_reason is not None,
        )
        if sampler.emergency_reason is not None:
            raise Turbo4BenchmarkError(sampler.emergency_reason)
        ready_at = time.perf_counter()
        for payload in payloads:
            if sampler.emergency_reason is not None:
                raise Turbo4BenchmarkError(sampler.emergency_reason)
            responses.append(
                _http_json(
                    f"{config.endpoint_url}/chat/completions",
                    payload=payload,
                )
            )
        if sampler.emergency_reason is not None:
            raise Turbo4BenchmarkError(sampler.emergency_reason)
        request_wall = time.perf_counter() - ready_at
        logs = _read_container_logs(config.container_name, run_dir / "container.log")
    except (
        HTTPError,
        URLError,
        OSError,
        ValueError,
        subprocess.TimeoutExpired,
        OperationCancelledError,
        Turbo4BenchmarkError,
        Turbo4LabRuntimeError,
    ) as exc:
        startup_error = sampler.emergency_reason or f"{type(exc).__name__}: {exc}"
        request_wall = 0.0
    finally:
        cgroup_peak = None
        try:
            release = manager.shutdown()
            cgroup_peak = release.get("cgroup_swap_peak_bytes")
        except Exception as exc:
            startup_error = startup_error or f"{type(exc).__name__}: {exc}"
        sampler.stop()
    resources = resource_gate_report(
        sampler=sampler,
        cgroup_swap_peak_bytes=(int(cgroup_peak) if cgroup_peak is not None else None),
    )
    release_evidence = manager.evidence().get("release")
    if not isinstance(release_evidence, Mapping) or not bool(release_evidence.get("observed", False)):
        resources["failures"].append("gpu_release_unconfirmed")
        resources["passed"] = False
    emergency_abort = sampler.emergency_reason
    if emergency_abort is not None:
        resources["emergency_abort"] = emergency_abort
        if emergency_abort not in resources["failures"]:
            resources["failures"].append(emergency_abort)
        resources["passed"] = False
    active_after = _assert_no_active_lab_or_product_container()
    if active_after:
        resources["failures"].append("container_orphan")
        resources["passed"] = False
    lower_logs = logs.lower()
    oom = any(token in lower_logs for token in ("out of memory", "cuda error", "cuda oom", "oom-kill"))
    if oom:
        resources["failures"].append("oom_detected")
        resources["passed"] = False
    request_ledger = _request_ledger(payloads)
    response_ledger = _replay_response_ledger(responses)
    result = {
        "candidate": asdict(candidate),
        "status": "passed" if not startup_error and bool(resources["passed"]) else "rejected",
        "error": startup_error,
        "preflight": preflight,
        "runtime": manager.evidence(),
        "start_to_health_sec": round(
            max(0.0, (ready_at or time.perf_counter()) - started),
            6,
        ),
        "request_wall_sec": round(request_wall, 6),
        "request_ledger": request_ledger,
        "response_ledger": response_ledger,
        "resource_gates": resources,
        "orphan_containers": active_after,
        "oom_detected": oom,
    }
    _json_write(run_dir / "raw-replay-requests.json", {"requests": list(payloads)})
    _json_write(run_dir / "raw-replay-responses.json", {"responses": responses})
    _json_write(run_dir / "resource-samples.json", {"samples": sampler.samples})
    _json_write(run_dir / "run-result.json", result)
    return result


def _full_auto_preset(
    candidate: TurboCandidate,
    *,
    port: int,
    round_index: int,
    image_id: str,
) -> dict[str, Any]:
    from benchmark_serving_scheduler_matrix import BASELINE, _build_candidate_preset

    preset = _build_candidate_preset(BASELINE, endpoint_port=18000)
    gemma = dict(preset.get("gemma") or {})
    gemma.update(
        {
            "image": candidate.image_ref,
            "pull_policy": "never",
            # Every comparison arm uses the same lab-only lifecycle.  The
            # shipping control must never start or reuse gemma-local-server.
            "endpoint_url": f"http://127.0.0.1:{port}/v1",
            "model": MODEL_NAME,
            "model_path": f"/models/{MODEL_NAME}",
            "context_size": 4096,
            "n_parallel": 1,
            "threads": 10,
            "n_gpu_layers": 23,
            "batch_size": 2048,
            "ubatch_size": 512,
            "cache_type_k": "f16",
            "cache_type_v": candidate.cache_type_v,
            "spec_type": "none",
            # The product contract permits this inert value only when
            # spec_type=none; the lab adapter itself never receives a draft
            # option in its fixed command.
            "spec_draft_n_max": 8,
        }
    )
    preset["gemma"] = gemma
    preset.setdefault("benchmark_http", {})["gemma_seed"] = DEFAULT_SEED
    preset.setdefault("benchmark_cache_policy", {}).update(
        {
            "paddleocr_persistent": False,
            "translation_persistent": False,
            "exact_tm": False,
            "project_checkpoint": False,
        }
    )
    config = _runtime_config(
        candidate,
        port=port,
        stage="full-auto",
        round_index=round_index,
        image_id=image_id,
    )
    preset["benchmark_turbo4_kv"] = {
        "protocol_version": config.protocol_version,
        "candidate_key": config.candidate_key,
        "image_ref": config.image_ref,
        "model_volume": config.model_volume,
        "model_name": config.model_name,
        "model_sha256": config.model_sha256,
        "port": config.port,
        "container_name": config.container_name,
        "cache_type_v": config.cache_type_v,
        "fork_commit": config.fork_commit,
        "image_id": config.image_id,
    }
    return preset


def _run_product_pipeline_monitored(
    *,
    command: Sequence[str],
    run_dir: Path,
    sampler: ResourceSampler,
    timeout_sec: float = 1800.0,
) -> dict[str, Any]:
    del sampler
    stdout_path = run_dir / "runner.stdout.log"
    stderr_path = run_dir / "runner.stderr.log"
    started = time.monotonic()
    timed_out = False
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            list(command),
            cwd=str(ROOT),
            env={
                **os.environ,
                "QT_QPA_PLATFORM": "offscreen",
                "CT_DISABLE_UPDATE_CHECK": "1",
                "CT_ENABLE_MEMLOG": "1",
                "CT_ENABLE_GPU_BENCH": "1",
                "CT_MEMLOG_INTERVAL_SEC": "1",
                "COMIC_SKIP_STARTUP_MODELS": "1",
            },
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        while process.poll() is None:
            if time.monotonic() - started >= timeout_sec:
                timed_out = True
                process.terminate()
                break
            time.sleep(0.25)
        try:
            returncode = process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            returncode = process.wait(timeout=20)
    return {
        "returncode": returncode,
        "wall_sec": round(time.monotonic() - started, 6),
        "emergency_terminated": False,
        "timed_out": timed_out,
        "command": list(command),
    }


def _recover_full_auto_lab_container(
    config: Turbo4LabRuntimeConfig,
    *,
    baseline_gpu_mib: int | None,
) -> dict[str, Any]:
    """Recover only the exact adapter container left by a killed child."""

    manager = Turbo4LabRuntimeManager(inner=None, config=config)
    inspection = manager.evidence().get("container")
    if not isinstance(inspection, Mapping) or not inspection:
        return {"attempted": False, "container_present": False}
    try:
        release = manager.recover_after_parent_abort(
            baseline_gpu_mib=baseline_gpu_mib,
        )
    except Exception as exc:
        return {
            "attempted": True,
            "container_present": True,
            "error": f"{type(exc).__name__}: {exc}",
            "evidence": manager.evidence(),
        }
    return {
        "attempted": True,
        "container_present": True,
        "release": release,
        "evidence": manager.evidence(),
    }


def _pipeline_request_ledger(
    *,
    preset: Mapping[str, Any],
    metrics_path: Path,
    record_path: Path,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    if metrics_path.is_file():
        for line in metrics_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, Mapping) or row.get("tag") != "translate_start":
                continue
            events.append(
                {
                    "image_index": row.get("image_index"),
                    "block_count": row.get("block_count"),
                    "translator_key": row.get("translator_key"),
                }
            )
    gemma = preset.get("gemma") if isinstance(preset.get("gemma"), Mapping) else {}
    fixed_contract = {
        "model": gemma.get("model"),
        "context_size": gemma.get("context_size"),
        "n_gpu_layers": gemma.get("n_gpu_layers"),
        "n_parallel": gemma.get("n_parallel"),
        "threads": gemma.get("threads"),
        "batch_size": gemma.get("batch_size"),
        "ubatch_size": gemma.get("ubatch_size"),
        "cache_type_k": gemma.get("cache_type_k"),
        "seed": (preset.get("benchmark_http") or {}).get("gemma_seed"),
    }
    runtime_variant = {
        "cache_type_v": gemma.get("cache_type_v"),
        "spec_type": gemma.get("spec_type"),
    }
    attempts: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    parse_failures: list[str] = []
    record_summary: Mapping[str, Any] | None = None
    if not record_path.is_file():
        parse_failures.append("gemma_http_record_missing")
    else:
        for line_number, line in enumerate(
            record_path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                parse_failures.append(f"gemma_http_record_invalid_line_{line_number}")
                continue
            if not isinstance(row, Mapping):
                parse_failures.append(f"gemma_http_record_nonobject_line_{line_number}")
                continue
            if row.get("record_type") == "summary":
                if record_summary is not None:
                    parse_failures.append(f"gemma_http_summary_duplicate_line_{line_number}")
                else:
                    record_summary = row
                continue
            if record_summary is not None:
                parse_failures.append(f"gemma_http_record_after_summary_line_{line_number}")
                continue
            request = row.get("request")
            response = row.get("response")
            if not isinstance(request, Mapping):
                parse_failures.append(f"gemma_http_request_missing_line_{line_number}")
                continue
            if str(row.get("error", "") or ""):
                parse_failures.append(f"gemma_http_error_line_{line_number}")
            if request.get("model") != MODEL_NAME:
                parse_failures.append(f"gemma_http_model_mismatch_line_{line_number}")
            if request.get("seed") != DEFAULT_SEED:
                parse_failures.append(f"gemma_http_seed_mismatch_line_{line_number}")
            if not isinstance(request.get("messages"), list) or "response_format" not in request:
                parse_failures.append(f"gemma_http_schema_missing_line_{line_number}")
            status_code = row.get("status_code")
            if not isinstance(status_code, int) or not 200 <= status_code < 300:
                parse_failures.append(f"gemma_http_status_invalid_line_{line_number}")
            try:
                attempt_index = int(row.get("attempt_index", len(attempts)) or 0)
            except (TypeError, ValueError):
                parse_failures.append(f"gemma_http_attempt_index_invalid_line_{line_number}")
                attempt_index = len(attempts)
            attempts.append(
                {
                    "attempt_index": attempt_index,
                    "model": str(request.get("model", "") or ""),
                    "prompt_sha256": _canonical_sha256(request.get("messages")),
                    "schema_sha256": _canonical_sha256(request.get("response_format")),
                    "seed": request.get("seed"),
                    "payload_sha256": _canonical_sha256(request),
                }
            )
            responses.append(
                {
                    "attempt_index": attempt_index,
                    "canonical_response": _canonical_response(
                        response if isinstance(response, Mapping) else {}
                    ),
                }
            )
    if not attempts:
        parse_failures.append("gemma_http_record_empty")
    attempts.sort(key=lambda item: int(item["attempt_index"]))
    responses.sort(key=lambda item: int(item["attempt_index"]))
    if record_summary is None:
        parse_failures.append("gemma_http_summary_missing")
    else:
        try:
            expected_attempt_count = int(record_summary.get("attempt_count"))
        except (TypeError, ValueError):
            parse_failures.append("gemma_http_summary_attempt_count_invalid")
            expected_attempt_count = -1
        if expected_attempt_count != len(attempts):
            parse_failures.append("gemma_http_summary_attempt_count_mismatch")
        if str(record_summary.get("write_error", "") or ""):
            parse_failures.append("gemma_http_writer_error")
        if [int(item["attempt_index"]) for item in attempts] != list(
            range(max(0, expected_attempt_count))
        ):
            parse_failures.append("gemma_http_attempt_index_sequence_invalid")
    payload = {
        "fixed_contract": fixed_contract,
        "translation_start_order": events,
        "actual_http_attempts": attempts,
    }
    return {
        **payload,
        "runtime_variant": runtime_variant,
        "response_ledger": {
            "rows": responses,
            "sha256": _canonical_sha256(responses),
        },
        "record_path_present": record_path.is_file(),
        "record_summary": dict(record_summary or {}),
        "record_complete": not parse_failures,
        "record_failures": parse_failures,
        "sha256": _canonical_sha256(payload),
    }


def _full_auto_quality_failures(
    *,
    snapshot: Mapping[str, Any],
    summary: Mapping[str, Any],
    expected_page_count: int,
) -> list[str]:
    failures: list[str] = []
    try:
        if int(summary.get("page_failed_count", -1)) != 0:
            failures.append("summary_page_failed")
    except (TypeError, ValueError):
        failures.append("summary_page_failed_invalid")
    try:
        if int(summary.get("page_done_count", -1)) != expected_page_count:
            failures.append("summary_page_done_count_mismatch")
    except (TypeError, ValueError):
        failures.append("summary_page_done_count_invalid")
    pages = snapshot.get("pages")
    if not isinstance(pages, list) or len(pages) != expected_page_count:
        return failures + ["snapshot_page_count_mismatch"]
    for page_index, page in enumerate(pages):
        if not isinstance(page, Mapping):
            failures.append(f"page_{page_index}_invalid")
            continue
        if bool(page.get("page_failed", False)):
            failures.append(f"page_{page_index}_failed")
        stage_status = page.get("stage_status")
        if isinstance(stage_status, Mapping):
            for stage_name, stage in stage_status.items():
                if isinstance(stage, Mapping) and str(stage.get("status", "")).lower() == "failed":
                    failures.append(f"page_{page_index}_{stage_name}_failed")
        if not bool(page.get("translated_image_exists", False)):
            failures.append(f"page_{page_index}_render_missing")
        decoded_hash = str(page.get("translated_image_decoded_pixel_sha256", "") or "")
        if not _SHA256_RE.fullmatch(decoded_hash):
            failures.append(f"page_{page_index}_decoded_pixel_hash_missing")
    return failures


def execute_full_auto_once(
    candidate: TurboCandidate,
    *,
    sample_dir: Path,
    sample_count: int,
    run_dir: Path,
    round_index: int,
    port: int,
    image_id: str,
    python_executable: str,
) -> dict[str, Any]:
    candidate.validate()
    if not sample_dir.is_dir():
        raise Turbo4BenchmarkError("Full-auto sample directory is unavailable.")
    preflight = require_preflight(output_dir=run_dir)
    preset = _full_auto_preset(
        candidate,
        port=port,
        round_index=round_index,
        image_id=image_id,
    )
    raw_runtime_config = preset.get("benchmark_turbo4_kv")
    if not isinstance(raw_runtime_config, Mapping):
        raise Turbo4BenchmarkError("Full-auto Turbo4 lab runtime configuration is missing.")
    runtime_config = Turbo4LabRuntimeConfig.from_mapping(raw_runtime_config)
    preflight_gpu = preflight.get("gpu") if isinstance(preflight.get("gpu"), Mapping) else {}
    baseline_gpu_mib = (
        int(preflight_gpu["used_mib"])
        if preflight_gpu.get("used_mib") is not None
        else None
    )
    record_path = run_dir / "gemma-http-records.jsonl"
    benchmark_http = preset.setdefault("benchmark_http", {})
    if not isinstance(benchmark_http, dict):
        raise Turbo4BenchmarkError("Turbo4 benchmark_http configuration must be an object.")
    benchmark_http["gemma_request_record_path"] = str(record_path)
    preset_path = run_dir / "preset.json"
    _json_write(preset_path, preset)
    command = [
        python_executable,
        str(ROOT / "scripts" / "benchmark_pipeline.py"),
        "--preset",
        str(preset_path),
        "--mode",
        "batch",
        "--repeat",
        "1",
        "--runtime-mode",
        "attach-running",
        "--runtime-services",
        "full",
        "--product-managed-runtime",
        "--sample-dir",
        str(sample_dir),
        "--sample-count",
        str(sample_count),
        "--source-lang",
        "Japanese",
        "--target-lang",
        "Korean",
        "--export-page-snapshots",
        "--stage-ceiling",
        "render",
        "--output-dir",
        str(run_dir),
    ]
    sampler = ResourceSampler()
    sampler.start()
    process_error = ""
    try:
        process = _run_product_pipeline_monitored(
            command=command,
            run_dir=run_dir,
            sampler=sampler,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        process_error = f"{type(exc).__name__}: {exc}"
        process = {
            "returncode": -1,
            "wall_sec": 0.0,
            "emergency_terminated": False,
            "timed_out": False,
            "command": command,
        }
    finally:
        sampler.stop()
    parent_abort_cleanup = _recover_full_auto_lab_container(
        runtime_config,
        baseline_gpu_mib=baseline_gpu_mib,
    )
    snapshot_path = run_dir / "page_snapshots.json"
    summary_path = run_dir / "summary.json"
    snapshot: Mapping[str, Any] = {}
    summary: Mapping[str, Any] = {}
    error = process_error
    if parent_abort_cleanup.get("error"):
        error = error or "parent_lab_cleanup_failed"
    if process["returncode"] != 0:
        error = "product_pipeline_failed"
    elif not snapshot_path.is_file() or not summary_path.is_file():
        error = "product_pipeline_output_missing"
    else:
        try:
            parsed_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            parsed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            snapshot = parsed_snapshot if isinstance(parsed_snapshot, Mapping) else {}
            summary = parsed_summary if isinstance(parsed_summary, Mapping) else {}
        except json.JSONDecodeError:
            error = "product_pipeline_output_invalid"
    quality_failures = _full_auto_quality_failures(
        snapshot=snapshot,
        summary=summary,
        expected_page_count=sample_count,
    )
    if quality_failures:
        error = error or "full_auto_quality_gate_failed"
    lab_runtime: Mapping[str, Any] = {}
    lab_runtime_path = run_dir / "turbo4_lab_runtime.json"
    if not lab_runtime_path.is_file():
        error = error or "lab_runtime_evidence_missing"
    else:
        try:
            parsed = json.loads(lab_runtime_path.read_text(encoding="utf-8"))
            lab_runtime = parsed if isinstance(parsed, Mapping) else {}
        except json.JSONDecodeError:
            error = error or "lab_runtime_evidence_invalid"
    release = lab_runtime.get("release") if isinstance(lab_runtime, Mapping) else {}
    if not isinstance(release, Mapping) or not bool(release.get("observed", False)):
        error = error or "lab_runtime_release_unconfirmed"
    cgroup_peak = release.get("cgroup_swap_peak_bytes") if isinstance(release, Mapping) else None
    resources = resource_gate_report(
        sampler=sampler,
        cgroup_swap_peak_bytes=(int(cgroup_peak) if cgroup_peak is not None else None),
    )
    if not isinstance(release, Mapping) or not bool(release.get("observed", False)):
        resources["failures"].append("gpu_release_unconfirmed")
        resources["passed"] = False
    active_after = _assert_no_active_lab_or_product_container()
    if active_after:
        resources["failures"].append("container_orphan")
        resources["passed"] = False
    logs = ""
    for log_path in (run_dir / "runner.stdout.log", run_dir / "runner.stderr.log"):
        if log_path.is_file():
            logs += log_path.read_text(encoding="utf-8", errors="replace")
    oom = any(token in logs.lower() for token in ("out of memory", "cuda error", "cuda oom", "oom-kill"))
    if oom:
        resources["failures"].append("oom_detected")
        resources["passed"] = False
    request_ledger = _pipeline_request_ledger(
        preset=preset,
        metrics_path=run_dir / "metrics.jsonl",
        record_path=record_path,
    )
    if not request_ledger["record_complete"]:
        error = error or "gemma_request_ledger_incomplete"
    result = {
        "candidate": asdict(candidate),
        "status": "passed" if not error and bool(resources["passed"]) else "rejected",
        "error": error,
        "preflight": preflight,
        "process": process,
        "pipeline_wall_sec": process["wall_sec"],
        "snapshot_sha256": canonical_full_snapshot_sha256(snapshot) if snapshot else "",
        "quality_failures": quality_failures,
        "request_ledger": request_ledger,
        "resource_gates": resources,
        "lab_runtime": lab_runtime,
        "parent_abort_cleanup": parent_abort_cleanup,
        "orphan_containers": active_after,
        "oom_detected": oom,
        "summary_status": summary.get("status") if isinstance(summary, Mapping) else None,
    }
    _json_write(run_dir / "resource-samples.json", {"samples": sampler.samples})
    _json_write(run_dir / "run-result.json", result)
    return result


def _quality_key(result: Mapping[str, Any], *, mode: str) -> str:
    if mode == "replay":
        response = result.get("response_ledger")
        if not isinstance(response, Mapping):
            return ""
        return str(response.get("sha256", "") or "")
    return str(result.get("snapshot_sha256", "") or "")


def _ledger_key(result: Mapping[str, Any]) -> str:
    ledger = result.get("request_ledger")
    return str(ledger.get("sha256", "") or "") if isinstance(ledger, Mapping) else ""


def execute_pair_matrix(
    *,
    baseline: TurboCandidate,
    candidate: TurboCandidate,
    mode: str,
    output_dir: Path,
    execute: Callable[[TurboCandidate, Path, int], dict[str, Any]],
    initial_rounds: int = DEFAULT_INITIAL_ROUNDS,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> dict[str, Any]:
    if baseline.key == candidate.key:
        raise Turbo4BenchmarkError("Baseline and candidate must differ.")
    if mode not in {"replay", "full-auto"}:
        raise Turbo4BenchmarkError("Unsupported Turbo4 pair mode.")
    max_rounds = max(2, min(DEFAULT_MAX_ROUNDS, int(max_rounds)))
    initial_rounds = max(2, min(max_rounds, int(initial_rounds)))
    pairs: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    fatal_resource_failures: list[str] = []
    aborted_early = False
    reference_quality = ""
    reference_ledger = ""
    for round_index in range(1, max_rounds + 1):
        order = (baseline, candidate) if round_index % 2 else (candidate, baseline)
        results: dict[str, dict[str, Any]] = {}
        for profile in order:
            run_dir = output_dir / f"round-{round_index}" / profile.key
            run_dir.mkdir(parents=True, exist_ok=False)
            results[profile.key] = execute(profile, run_dir, round_index)
            resource_failures = set(
                (results[profile.key].get("resource_gates") or {}).get("failures")
                or []
            )
            fatal = sorted(resource_failures & _FATAL_RESOURCE_FAILURES)
            if fatal or results[profile.key].get("status") != "passed":
                aborted_early = True
                fatal_resource_failures.extend(fatal)
                for missing in (baseline, candidate):
                    results.setdefault(
                        missing.key,
                        {
                            "candidate": asdict(missing),
                            "status": "not-run",
                            "request_wall_sec": 0.0,
                            "pipeline_wall_sec": 0.0,
                            "resource_gates": {
                                "passed": False,
                                "failures": ["not_run_after_fatal_or_reject"],
                            },
                        },
                    )
                break
        for profile in (baseline, candidate):
            result = results[profile.key]
            quality = _quality_key(result, mode="replay" if mode == "replay" else "full-auto")
            ledger = _ledger_key(result)
            if not reference_quality and profile.key == baseline.key:
                reference_quality = quality
                reference_ledger = ledger
            if not quality or quality != reference_quality or ledger != reference_ledger:
                mismatches.append(
                    {
                        "round": round_index,
                        "profile": profile.key,
                        "expected_quality_sha256": reference_quality,
                        "actual_quality_sha256": quality,
                        "expected_request_ledger_sha256": reference_ledger,
                        "actual_request_ledger_sha256": ledger,
                    }
                )
        baseline_result = results[baseline.key]
        candidate_result = results[candidate.key]
        timing_key = "request_wall_sec" if mode == "replay" else "pipeline_wall_sec"
        baseline_seconds = float(baseline_result.get(timing_key, 0.0) or 0.0)
        candidate_seconds = float(candidate_result.get(timing_key, 0.0) or 0.0)
        pair = {
            "round": round_index,
            "order": [item.key for item in order],
            "baseline": baseline_result,
            "candidate": candidate_result,
            "baseline_sec": baseline_seconds,
            "candidate_sec": candidate_seconds,
        }
        pairs.append(pair)
        valid_times = all(item["baseline_sec"] > 0 and item["candidate_sec"] > 0 for item in pairs)
        stats = (
            summarize_pairs(
                [item["baseline_sec"] for item in pairs],
                [item["candidate_sec"] for item in pairs],
            )
            if valid_times
            else {}
        )
        _json_write(
            output_dir / "pair-summary.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "mode": mode,
                "baseline": asdict(baseline),
                "candidate": asdict(candidate),
                "rounds": pairs,
                "statistics": stats,
                "quality_mismatches": mismatches,
                "fatal_resource_failures": sorted(set(fatal_resource_failures)),
                "aborted_early": aborted_early,
            },
        )
        if aborted_early:
            break
        if round_index < initial_rounds:
            continue
        if mismatches:
            break
        if not stats or not should_continue_adaptive(stats, rounds=round_index):
            break
    valid_times = all(item["baseline_sec"] > 0 and item["candidate_sec"] > 0 for item in pairs)
    stats = (
        summarize_pairs(
            [item["baseline_sec"] for item in pairs],
            [item["candidate_sec"] for item in pairs],
        )
        if pairs and valid_times
        else {}
    )
    resources_ok = all(
        bool((item[role].get("resource_gates") or {}).get("passed"))
        for item in pairs
        for role in ("baseline", "candidate")
    )
    statuses_ok = all(
        item[role].get("status") == "passed"
        for item in pairs
        for role in ("baseline", "candidate")
    )
    lower = float(stats.get("one_sided_95_bootstrap_lower_percent", 0.0) or 0.0)
    promotion = bool(not mismatches and resources_ok and statuses_ok and lower > 0.0)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "mode": mode,
        "baseline": asdict(baseline),
        "candidate": asdict(candidate),
        "round_count": len(pairs),
        "statistics": stats,
        "quality_exact": not mismatches,
        "quality_mismatches": mismatches,
        "resource_gate_pass": resources_ok,
        "status_gate_pass": statuses_ok,
        "promotion_eligible": promotion,
        "decision": "PASS" if promotion else "REJECT",
        "fatal_resource_failures": sorted(set(fatal_resource_failures)),
        "aborted_early": aborted_early,
        "rounds": pairs,
    }


def execute_structural_gate(
    *,
    candidates: Mapping[str, TurboCandidate],
    payloads: Sequence[Mapping[str, Any]],
    output_dir: Path,
    port: int,
    image_ids: Mapping[str, str],
) -> dict[str, Any]:
    runs: dict[str, dict[str, Any]] = {}
    fatal_resource_failures = {
        "gpu_release_unconfirmed",
        "oom_detected",
        "container_orphan",
    }
    for index, key in enumerate(("shipping-f16", "fork-f16", "turbo4"), start=1):
        run_dir = output_dir / key
        run_dir.mkdir(parents=True, exist_ok=False)
        candidate = candidates[key]
        runs[key] = execute_replay_once(
            candidate,
            payloads=payloads,
            run_dir=run_dir,
            round_index=index,
            port=port,
            image_id=image_ids[key],
        )
        failures = set((runs[key].get("resource_gates") or {}).get("failures") or [])
        fatal = sorted(failures & fatal_resource_failures)
        if fatal:
            result = {
                "protocol_version": PROTOCOL_VERSION,
                "stage": "fixed-seed-structural",
                "image_ids": dict(image_ids),
                "runs": runs,
                "comparisons": {},
                "resource_gate_pass": False,
                "status_gate_pass": False,
                "fatal_resource_failures": fatal,
                "aborted_early": True,
                "decision": "REJECT",
            }
            _json_write(output_dir / "structural-summary.json", result)
            return result
    reference = _quality_key(runs["shipping-f16"], mode="replay")
    request_reference = _ledger_key(runs["shipping-f16"])
    mismatches = {
        key: {
            "response_exact": _quality_key(result, mode="replay") == reference,
            "request_exact": _ledger_key(result) == request_reference,
        }
        for key, result in runs.items()
    }
    resources_ok = all(bool((result.get("resource_gates") or {}).get("passed")) for result in runs.values())
    statuses_ok = all(result.get("status") == "passed" for result in runs.values())
    passed = bool(resources_ok and statuses_ok and all(all(row.values()) for row in mismatches.values()))
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "fixed-seed-structural",
        "image_ids": dict(image_ids),
        "runs": runs,
        "response_reference_sha256": reference,
        "request_reference_sha256": request_reference,
        "comparisons": mismatches,
        "resource_gate_pass": resources_ok,
        "status_gate_pass": statuses_ok,
        "decision": "PASS" if passed else "REJECT",
    }
    _json_write(output_dir / "structural-summary.json", result)
    return result


def execute_fork_replay_abba(
    *,
    candidates: Mapping[str, TurboCandidate],
    payloads: Sequence[Mapping[str, Any]],
    output_dir: Path,
    port: int,
    image_ids: Mapping[str, str],
    initial_rounds: int,
    max_rounds: int,
) -> dict[str, Any]:
    """Measure the only intended fork delta before E2E promotion gates."""

    stage_dir = output_dir / "fork-f16-vs-turbo4-replay"
    stage_dir.mkdir(parents=True, exist_ok=False)

    def execute(
        candidate: TurboCandidate,
        run_dir: Path,
        round_index: int,
    ) -> dict[str, Any]:
        return execute_replay_once(
            candidate,
            payloads=payloads,
            run_dir=run_dir,
            round_index=round_index,
            port=port,
            image_id=image_ids[candidate.key],
        )

    result = execute_pair_matrix(
        baseline=candidates["fork-f16"],
        candidate=candidates["turbo4"],
        mode="replay",
        output_dir=stage_dir,
        execute=execute,
        initial_rounds=initial_rounds,
        max_rounds=max_rounds,
    )
    result["stage"] = "fork-f16-vs-turbo4-replay"
    result["image_ids"] = dict(image_ids)
    result["resource_summary"] = _aggregate_resources(
        [
            result_item
            for row in result["rounds"]
            for result_item in (row["baseline"], row["candidate"])
        ]
    )
    result["r3_estimate"] = r3_estimate_from_results(
        [
            result_item
            for row in result["rounds"]
            for result_item in (row["baseline"], row["candidate"])
        ]
    )
    _json_write(stage_dir / "stage-summary.json", result)
    return result


def r3_estimate_from_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidate_peaks: list[int] = []
    gpu_baselines: list[int] = []
    physical: int | None = None
    for result in results:
        candidate = result.get("candidate")
        if not isinstance(candidate, Mapping) or candidate.get("key") != "turbo4":
            continue
        resources = result.get("resource_gates")
        preflight = result.get("preflight")
        if isinstance(resources, Mapping) and resources.get("gpu_peak_mib") is not None:
            candidate_peaks.append(int(resources["gpu_peak_mib"]))
        if isinstance(preflight, Mapping):
            gpu = preflight.get("gpu")
            if isinstance(gpu, Mapping):
                if gpu.get("used_mib") is not None:
                    gpu_baselines.append(int(gpu["used_mib"]))
                if gpu.get("total_mib") is not None:
                    physical = int(gpu["total_mib"])
    if physical is None or not candidate_peaks or not gpu_baselines:
        return {"available": False, "threshold": R3_RESIDENCY_THRESHOLD}
    gemma_peak = max(0, max(candidate_peaks) - min(gpu_baselines))
    estimate = residency_preflight(
        physical_mib=physical,
        paddle_peak_mib=PADDLE_MEASURED_PEAK_MIB,
        gemma_peak_mib=gemma_peak,
        threshold=R3_RESIDENCY_THRESHOLD,
    )
    return {
        "available": True,
        "paddle_peak_mib_source": "latest-direct-paddle-lab",
        "turbo4_gemma_peak_mib": gemma_peak,
        "active_r3_executed": False,
        **estimate,
    }


def _stage_gate(path: Path, *, expected_stage: str) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("decision") != "PASS":
        raise Turbo4BenchmarkError(f"Required prior {expected_stage} gate did not pass.")
    return payload


def _load_compatible_prior_gate(
    path: Path,
    *,
    expected_stage: str,
    image_ids: Mapping[str, str],
) -> Mapping[str, Any]:
    payload = _stage_gate(path, expected_stage=expected_stage)
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise Turbo4BenchmarkError(f"Prior {expected_stage} gate protocol does not match this lab.")
    if payload.get("stage") != expected_stage:
        raise Turbo4BenchmarkError(f"Prior gate is not the required {expected_stage} stage.")
    if dict(payload.get("image_ids") or {}) != dict(image_ids):
        raise Turbo4BenchmarkError(
            f"Prior {expected_stage} gate image identities differ from this run."
        )
    return payload


def _print_redacted(result: Mapping[str, Any]) -> None:
    """Print only the approved checkpoint fields, never raw prompts/responses."""

    statistics = result.get("statistics") if isinstance(result.get("statistics"), Mapping) else {}
    r3 = result.get("r3_estimate") if isinstance(result.get("r3_estimate"), Mapping) else {}
    resources = result.get("resource_summary") if isinstance(result.get("resource_summary"), Mapping) else {}
    payload = {
        "decision": result.get("decision", "REJECT"),
        "median_baseline_sec": statistics.get("baseline_median_sec"),
        "median_candidate_sec": statistics.get("candidate_median_sec"),
        "one_sided_95_bootstrap_lower_percent": statistics.get("one_sided_95_bootstrap_lower_percent"),
        "peak_vram_mib": resources.get("gpu_peak_mib"),
        "windows_available_min_bytes": resources.get("windows_available_min_bytes"),
        "wsl_swap_growth_bytes": resources.get("wsl_swap_growth_bytes"),
        "shared_gpu_growth_mib": resources.get("shared_gpu_growth_mib"),
        "r3_estimate": r3,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _aggregate_resources(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [item.get("resource_gates") for item in items if isinstance(item.get("resource_gates"), Mapping)]
    if not rows:
        return {}
    values: dict[str, list[float]] = {}
    for row in rows:
        for key in ("gpu_peak_mib", "windows_available_min_bytes", "wsl_swap_growth_bytes", "shared_gpu_growth_mib"):
            value = row.get(key)
            if isinstance(value, (int, float)):
                values.setdefault(key, []).append(float(value))
    return {
        "gpu_peak_mib": max(values.get("gpu_peak_mib", []) or [0]),
        "windows_available_min_bytes": min(values.get("windows_available_min_bytes", []) or [0]),
        "wsl_swap_growth_bytes": max(values.get("wsl_swap_growth_bytes", []) or [0]),
        "shared_gpu_growth_mib": max(values.get("shared_gpu_growth_mib", []) or [0]),
    }


def _load_prepared_runtime_manifest(
    path: Path,
    *,
    image_tag: str,
    fork_commit: str,
) -> dict[str, Any]:
    """Reuse a just-validated immutable lab preparation without reprobe I/O.

    A model-volume probe is intentionally isolated, but Docker Desktop can
    retain several GiB of host pages immediately afterward.  A resume run may
    therefore use the private manifest written by a preceding successful
    ``--mode build`` only when the current image IDs and all pinned identities
    still match.  It does not trust mutable tags or a bare image reference.
    """

    try:
        raw = path.read_bytes()
        prepared = json.loads(raw.decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Turbo4BenchmarkError("Prepared Turbo4 runtime manifest is unreadable.") from exc
    if not isinstance(prepared, Mapping):
        raise Turbo4BenchmarkError("Prepared Turbo4 runtime manifest is invalid.")

    fork = verify_fork_ref(fork_commit=fork_commit)
    expected_fork = prepared.get("fork") if isinstance(prepared.get("fork"), Mapping) else {}
    expected_shipping = (
        prepared.get("shipping_image")
        if isinstance(prepared.get("shipping_image"), Mapping)
        else {}
    )
    expected_turbo = (
        prepared.get("turbo_image")
        if isinstance(prepared.get("turbo_image"), Mapping)
        else {}
    )
    expected_turbo_image = (
        expected_turbo.get("image")
        if isinstance(expected_turbo.get("image"), Mapping)
        else {}
    )
    expected_volume = (
        prepared.get("model_volume")
        if isinstance(prepared.get("model_volume"), Mapping)
        else {}
    )

    shipping = _inspect_image(SHIPPING_IMAGE)
    turbo_image = _inspect_image(image_tag)
    failures: list[str] = []
    if expected_fork.get("commit") != fork_commit or fork.get("commit") != fork_commit:
        failures.append("fork-commit")
    if expected_shipping.get("id") != shipping.get("id"):
        failures.append("shipping-image-id")
    if expected_turbo_image.get("reference") != image_tag:
        failures.append("turbo-image-reference")
    if expected_turbo_image.get("id") != turbo_image.get("id"):
        failures.append("turbo-image-id")
    if expected_turbo.get("fork_commit") != fork_commit or not expected_turbo.get("turbo4_help_present"):
        failures.append("turbo4-help-or-commit")
    if (
        expected_volume.get("model_name") != MODEL_NAME
        or expected_volume.get("model_sha256") != MODEL_SHA256
    ):
        failures.append("model-volume-identity")
    if failures:
        raise Turbo4BenchmarkError(
            "Prepared Turbo4 runtime manifest no longer matches: " + ", ".join(failures)
        )
    return {
        "fork": fork,
        "shipping_image": shipping,
        "turbo_image": {
            "image": turbo_image,
            "fork_commit": fork_commit,
            "help_sha256": expected_turbo.get("help_sha256"),
            "turbo4_help_present": True,
            "reused_prepared_validation": True,
        },
        "build": {
            "image": turbo_image,
            "reused": True,
            "prepared_runtime_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        },
        "model_volume": dict(expected_volume),
    }


def _prepare_images(
    *,
    output_dir: Path,
    image_tag: str,
    fork_commit: str,
    build: bool,
    prepared_runtime_manifest: Path | None = None,
) -> dict[str, Any]:
    require_preflight(output_dir=output_dir)
    if prepared_runtime_manifest is not None:
        if build:
            raise Turbo4BenchmarkError(
                "A prepared runtime manifest may only be used with --reuse-image."
            )
        result = _load_prepared_runtime_manifest(
            prepared_runtime_manifest,
            image_tag=image_tag,
            fork_commit=fork_commit,
        )
        _json_write(output_dir / "runtime-manifest.json", result)
        return result
    fork = verify_fork_ref(fork_commit=fork_commit)
    shipping = _inspect_image(SHIPPING_IMAGE)
    if build:
        build_result = _run_docker_build(
            image_tag=image_tag,
            fork_commit=fork_commit,
            output_dir=output_dir,
        )
    else:
        build_result = {"image": _inspect_image(image_tag), "reused": True}
    turbo = verify_turbo_image(
        image_ref=image_tag,
        fork_commit=fork_commit,
        output_dir=output_dir,
    )
    volume = verify_model_volume(output_dir=output_dir)
    result = {
        "fork": fork,
        "shipping_image": shipping,
        "turbo_image": turbo,
        "build": build_result,
        "model_volume": volume,
    }
    _json_write(output_dir / "runtime-manifest.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the private Gemma Turbo4 KV-V lab.")
    parser.add_argument(
        "--mode",
        choices=("plan", "preflight", "build", "structural", "s1", "s6", "series", "all"),
        default="plan",
    )
    replay_source = parser.add_mutually_exclusive_group()
    replay_source.add_argument("--translation-replay", type=Path)
    replay_source.add_argument("--page-snapshots", type=Path)
    parser.add_argument("--s1-sample-dir", type=Path)
    parser.add_argument("--s6-sample-dir", type=Path)
    parser.add_argument("--series-sample-dir", type=Path)
    parser.add_argument("--structural-gate", type=Path)
    parser.add_argument("--s1-gate", type=Path)
    parser.add_argument("--s6-gate", type=Path)
    parser.add_argument("--sample-count-s1", type=int, default=1)
    parser.add_argument("--sample-count-s6", type=int, default=6)
    parser.add_argument("--sample-count-series", type=int, default=6)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--image-tag", default=f"comic-translate/turbo4-lab:{FORK_COMMIT[:12]}")
    parser.add_argument("--fork-commit", default=FORK_COMMIT)
    parser.add_argument("--reuse-image", action="store_true")
    parser.add_argument(
        "--prepared-runtime-manifest",
        type=Path,
        help="Private runtime-manifest.json from a just-passing --mode build; avoids a duplicate volume probe.",
    )
    parser.add_argument("--initial-rounds", type=int, default=DEFAULT_INITIAL_ROUNDS)
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", type=Path)
    return parser


def _run_full_stage(
    *,
    label: str,
    sample_dir: Path,
    sample_count: int,
    candidates: Mapping[str, TurboCandidate],
    output_dir: Path,
    port: int,
    image_ids: Mapping[str, str],
    python_executable: str,
    initial_rounds: int,
    max_rounds: int,
) -> dict[str, Any]:
    stage_dir = output_dir / label
    stage_dir.mkdir(parents=True, exist_ok=False)

    def execute(candidate: TurboCandidate, run_dir: Path, round_index: int) -> dict[str, Any]:
        image_id = image_ids[candidate.key]
        return execute_full_auto_once(
            candidate,
            sample_dir=sample_dir,
            sample_count=sample_count,
            run_dir=run_dir,
            round_index=round_index,
            port=port,
            image_id=image_id,
            python_executable=python_executable,
        )

    result = execute_pair_matrix(
        baseline=candidates["shipping-f16"],
        candidate=candidates["turbo4"],
        mode="full-auto",
        output_dir=stage_dir,
        execute=execute,
        initial_rounds=initial_rounds,
        max_rounds=max_rounds,
    )
    result["stage"] = label
    result["image_ids"] = dict(image_ids)
    result["r3_estimate"] = r3_estimate_from_results(
        [
            result_item
            for row in result["rounds"]
            for result_item in (row["baseline"], row["candidate"])
        ]
    )
    result["resource_summary"] = _aggregate_resources(
        [
            result_item
            for row in result["rounds"]
            for result_item in (row["baseline"], row["candidate"])
        ]
    )
    _json_write(stage_dir / "stage-summary.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = load_protocol()
    if args.mode == "plan":
        print(
            json.dumps(
                {
                    "protocol_version": protocol["protocol_version"],
                    "fork_commit": FORK_COMMIT,
                    "fixed": protocol["fixed_server"],
                    "r3": {"threshold": R3_RESIDENCY_THRESHOLD, "active_execution": False},
                    "order": ["structural", "s1", "s6", "series"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    output_dir, managed_run = select_managed_output_directory(
        family=FAMILY_NAME,
        category=ARTIFACT_CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode == "preflight":
            result = {"decision": "PASS", "preflight": require_preflight(output_dir=output_dir)}
            _json_write(output_dir / "preflight-summary.json", result)
            _print_redacted(result)
            if managed_run is not None:
                managed_run.complete(metadata={"protocol_version": PROTOCOL_VERSION, "mode": args.mode})
            return 0

        images = _prepare_images(
            output_dir=output_dir,
            image_tag=args.image_tag,
            fork_commit=args.fork_commit,
            build=not args.reuse_image,
            prepared_runtime_manifest=args.prepared_runtime_manifest,
        )
        image_ids = {
            "shipping-f16": str((images["shipping_image"] or {}).get("id", "")),
            "fork-f16": str(((images["turbo_image"] or {}).get("image") or {}).get("id", "")),
            "turbo4": str(((images["turbo_image"] or {}).get("image") or {}).get("id", "")),
        }
        if not all(image_ids.values()):
            raise Turbo4BenchmarkError("Image IDs are required in the Turbo4 runtime manifest.")
        candidates = candidate_catalog(turbo_image=args.image_tag, fork_commit=args.fork_commit)
        if args.mode == "build":
            result = {"decision": "PASS", "runtime_manifest": images}
            _json_write(output_dir / "build-summary.json", result)
            _print_redacted(result)
            if managed_run is not None:
                managed_run.complete(metadata={"protocol_version": PROTOCOL_VERSION, "mode": args.mode})
            return 0

        structural_gate: Mapping[str, Any]
        if args.mode in {"structural", "all"}:
            if args.translation_replay is not None:
                payloads = load_translation_replay(args.translation_replay)
            elif args.page_snapshots is not None:
                payloads = build_translation_replay_from_page_snapshots(args.page_snapshots)
            else:
                raise Turbo4BenchmarkError(
                    "--translation-replay or --page-snapshots is required for the structural gate."
                )
            structural_dir = output_dir / "structural"
            structural_dir.mkdir(parents=True, exist_ok=False)
            structural = execute_structural_gate(
                candidates=candidates,
                payloads=payloads,
                output_dir=structural_dir,
                port=args.port,
                image_ids=image_ids,
            )
            if structural["decision"] != "PASS":
                result = {
                    "protocol_version": PROTOCOL_VERSION,
                    "image_ids": dict(image_ids),
                    "decision": "REJECT",
                    "stage": "fixed-seed-structural",
                    "structural": structural,
                    "resource_summary": _aggregate_resources(list(structural["runs"].values())),
                    "r3_estimate": r3_estimate_from_results(list(structural["runs"].values())),
                }
                _json_write(output_dir / "final-summary.json", result)
                _print_redacted(result)
                if managed_run is not None:
                    managed_run.complete(metadata={"protocol_version": PROTOCOL_VERSION, "mode": args.mode, "decision": "REJECT"})
                return 2
            fork_replay = execute_fork_replay_abba(
                candidates=candidates,
                payloads=payloads,
                output_dir=output_dir,
                port=args.port,
                image_ids=image_ids,
                initial_rounds=args.initial_rounds,
                max_rounds=args.max_rounds,
            )
            if fork_replay["decision"] != "PASS":
                result = {
                    "protocol_version": PROTOCOL_VERSION,
                    "image_ids": dict(image_ids),
                    "decision": "REJECT",
                    "stage": "fork-f16-vs-turbo4-replay",
                    "structural": structural,
                    "fork_replay": fork_replay,
                    "resource_summary": fork_replay["resource_summary"],
                    "r3_estimate": fork_replay["r3_estimate"],
                }
                _json_write(output_dir / "final-summary.json", result)
                _print_redacted(result)
                if managed_run is not None:
                    managed_run.complete(metadata={"protocol_version": PROTOCOL_VERSION, "mode": args.mode, "decision": "REJECT"})
                return 2
            structural_gate = {
                "protocol_version": PROTOCOL_VERSION,
                "image_ids": dict(image_ids),
                "decision": "PASS",
                "stage": "fixed-seed-structural",
                "structural": structural,
                "fork_replay": fork_replay,
            }
        else:
            if args.structural_gate is None:
                raise Turbo4BenchmarkError(
                    "--structural-gate is required when resuming after the structural gate."
                )
            structural_gate = _load_compatible_prior_gate(
                args.structural_gate,
                expected_stage="fixed-seed-structural",
                image_ids=image_ids,
            )
            fork_replay = structural_gate.get("fork_replay")
            if not isinstance(fork_replay, Mapping) or fork_replay.get("decision") != "PASS":
                raise Turbo4BenchmarkError(
                    "Prior structural gate lacks a passing fork F16 versus Turbo4 ABBA result."
                )

        if args.mode == "structural":
            result = {
                **structural_gate,
                "resource_summary": fork_replay["resource_summary"],
                "r3_estimate": fork_replay["r3_estimate"],
            }
            _json_write(output_dir / "final-summary.json", result)
            _print_redacted(result)
            if managed_run is not None:
                managed_run.complete(metadata={"protocol_version": PROTOCOL_VERSION, "mode": args.mode, "decision": "PASS"})
            return 0

        if args.mode in {"s1", "all"}:
            if args.s1_sample_dir is None:
                raise Turbo4BenchmarkError("--s1-sample-dir is required for S1 ABBA.")
            s1 = _run_full_stage(
                label="s1",
                sample_dir=args.s1_sample_dir.resolve(),
                sample_count=max(1, int(args.sample_count_s1)),
                candidates=candidates,
                output_dir=output_dir,
                port=args.port,
                image_ids=image_ids,
                python_executable=str(args.python),
                initial_rounds=args.initial_rounds,
                max_rounds=args.max_rounds,
            )
        else:
            if args.s1_gate is None:
                raise Turbo4BenchmarkError(
                    "--s1-gate is required when resuming at S6 or series."
                )
            s1 = _load_compatible_prior_gate(
                args.s1_gate,
                expected_stage="s1",
                image_ids=image_ids,
            )
        if s1["decision"] != "PASS" or args.mode == "s1":
            result = {"decision": s1["decision"], "stage": "s1", **s1}
            _json_write(output_dir / "final-summary.json", result)
            _print_redacted(result)
            if managed_run is not None:
                managed_run.complete(metadata={"protocol_version": PROTOCOL_VERSION, "mode": args.mode, "decision": result["decision"]})
            return 0 if result["decision"] == "PASS" else 2

        if args.mode in {"s6", "all"}:
            if args.s6_sample_dir is None:
                raise Turbo4BenchmarkError("--s6-sample-dir is required after S1 PASS.")
            s6 = _run_full_stage(
                label="s6",
                sample_dir=args.s6_sample_dir.resolve(),
                sample_count=max(1, int(args.sample_count_s6)),
                candidates=candidates,
                output_dir=output_dir,
                port=args.port,
                image_ids=image_ids,
                python_executable=str(args.python),
                initial_rounds=args.initial_rounds,
                max_rounds=args.max_rounds,
            )
        else:
            if args.s6_gate is None:
                raise Turbo4BenchmarkError(
                    "--s6-gate is required when resuming at the series gate."
                )
            s6 = _load_compatible_prior_gate(
                args.s6_gate,
                expected_stage="s6",
                image_ids=image_ids,
            )
        if s6["decision"] != "PASS" or args.mode == "s6":
            result = {"decision": s6["decision"], "stage": "s6", **s6}
            _json_write(output_dir / "final-summary.json", result)
            _print_redacted(result)
            if managed_run is not None:
                managed_run.complete(metadata={"protocol_version": PROTOCOL_VERSION, "mode": args.mode, "decision": result["decision"]})
            return 0 if result["decision"] == "PASS" else 2

        raise Turbo4BenchmarkError(
            "True series 3+3 requires the dedicated series-queue lab adapter; "
            "a flat six-page batch is not a valid substitute."
        )
    except BaseException as exc:
        if managed_run is not None:
            managed_run.fail(exc, metadata={"protocol_version": PROTOCOL_VERSION, "mode": args.mode})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
