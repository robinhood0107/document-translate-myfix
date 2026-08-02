#!/usr/bin/env python3
"""Lab-only CPU-MoE plus host-KV residency screen for shipping Gemma.

The runner never changes product serving settings.  It uses the pinned b10133
image and the existing fixed-seed translation replay, first finding a physical
VRAM-fitting CPU-MoE level.  A full response-ledger match is required for
quality or speed promotion; a separately labeled sentinel can answer only the
physical-residency question after its own exact-level fit preflight.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for entry in (ROOT, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import benchmark_gemma_turbo4_kv as replay_tools  # noqa: E402
import benchmark_serving_scheduler_matrix as serving_tools  # noqa: E402
from turbo4_lab_runtime import (  # noqa: E402
    OFFLOAD_LAB_CONTAINER_PREFIX,
    ShippingOffloadLabRuntimeConfig,
    Turbo4LabRuntimeError,
    Turbo4LabRuntimeManager,
)
from validation_artifact_harness import (  # noqa: E402
    ManagedArtifactRun,
    select_managed_output_directory,
)


PROTOCOL_VERSION = "gemma-cpu-moe-offload-v1"
FAMILY_NAME = "gemma-cpu-moe-offload"
ARTIFACT_CATEGORY = "10-gemma-translation"
PROTOCOL_PATH = ROOT / "benchmarks" / "gemma_cpu_moe_offload" / "protocol-v1.json"
SHIPPING_IMAGE = replay_tools.SHIPPING_IMAGE
SHIPPING_COMMIT = replay_tools.SHIPPING_COMMIT
SHIPPING_BUILD = replay_tools.SHIPPING_BUILD
MODEL_VOLUME = replay_tools.MODEL_VOLUME
MODEL_NAME = replay_tools.MODEL_NAME
MODEL_SHA256 = replay_tools.MODEL_SHA256
DEFAULT_PORT = 18082
DEFAULT_PADDLE_PORT = 18002
DEFAULT_CPU_MOE_LEVELS = (1, 2, 3, 4, 5, 6, 8, 12)
_SAFE_KEY = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class OffloadLabError(RuntimeError):
    """The shipping CPU-MoE lab contract was not satisfied."""


@dataclass(frozen=True)
class OffloadCandidate:
    key: str
    kv_offload: bool
    n_cpu_moe: int

    def validate(self) -> None:
        if not _SAFE_KEY.fullmatch(self.key):
            raise OffloadLabError(f"Unsafe offload candidate key: {self.key!r}")
        if not isinstance(self.kv_offload, bool):
            raise OffloadLabError("kv_offload must be boolean.")
        if not 0 <= int(self.n_cpu_moe) <= 64:
            raise OffloadLabError("n_cpu_moe must be between 0 and 64.")
        if self.kv_offload and self.n_cpu_moe:
            raise OffloadLabError("The shipping baseline may not carry CPU-MoE offload.")


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    replay_tools._json_write(path, value)


def _canonical_sha256(value: Any) -> str:
    return replay_tools._canonical_sha256(value)


def _running_conflicts() -> list[str]:
    active = replay_tools._running_container_names()
    return [
        name
        for name in active
        if name in {"gemma-local-server", "paddleocr-llamacpp"}
        or name.startswith(OFFLOAD_LAB_CONTAINER_PREFIX)
        or name.startswith(replay_tools.LAB_CONTAINER_PREFIX)
        or name.startswith(serving_tools.LAB_CONTAINER_PREFIX)
    ]


def require_preflight(*, output_dir: Path) -> dict[str, Any]:
    """Fail closed for active GPU conflicts, not host-memory observations."""

    base = serving_tools.resource_preflight()
    telemetry_observations = {
        "windows_available_ram_unavailable",
        "windows_available_ram_below_6gib",
        "wsl_swap_metrics_unavailable",
    }
    failures = [
        value for value in (base.get("failures") or []) if value not in telemetry_observations
    ]
    observations = [
        value for value in (base.get("failures") or []) if value in telemetry_observations
    ]
    conflicts = _running_conflicts()
    if conflicts:
        failures.append("managed_or_lab_gpu_container_active")
    shared = replay_tools.query_shared_gpu_used_mib()
    if shared is None:
        observations.append("windows_shared_gpu_metrics_unavailable")
    result = {
        **base,
        "shared_gpu_used_mib": shared,
        "active_conflicts": conflicts,
        "observations": observations,
        "failures": failures,
        "passed": not failures,
    }
    _json_write(output_dir / "resource-preflight.json", result)
    if not result["passed"]:
        raise OffloadLabError("Resource preflight failed: " + ", ".join(failures))
    return result


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OffloadLabError("Offload lab protocol is unreadable.") from exc
    if not isinstance(payload, dict) or payload.get("protocol_version") != PROTOCOL_VERSION:
        raise OffloadLabError("Offload lab protocol version mismatch.")
    shipping = payload.get("shipping") if isinstance(payload.get("shipping"), Mapping) else {}
    model = payload.get("model") if isinstance(payload.get("model"), Mapping) else {}
    fixed = payload.get("fixed_server") if isinstance(payload.get("fixed_server"), Mapping) else {}
    if (
        shipping.get("image") != SHIPPING_IMAGE
        or shipping.get("commit") != SHIPPING_COMMIT
        or shipping.get("build") != SHIPPING_BUILD
    ):
        raise OffloadLabError("Offload lab must pin shipping b10133.")
    if (
        model.get("volume") != MODEL_VOLUME
        or model.get("name") != MODEL_NAME
        or model.get("sha256") != MODEL_SHA256
    ):
        raise OffloadLabError("Offload lab model identity mismatch.")
    expected_fixed = {
        "context_size": 4096,
        "n_gpu_layers": 23,
        "n_parallel": 1,
        "threads": 10,
        "batch_size": 2048,
        "ubatch_size": 512,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "seed": replay_tools.DEFAULT_SEED,
    }
    if any(fixed.get(key) != value for key, value in expected_fixed.items()):
        raise OffloadLabError("Offload lab fixed server contract mismatch.")
    if set(payload.get("candidate_delta") or []) != {"--no-kv-offload", "--n-cpu-moe N"}:
        raise OffloadLabError("Offload lab candidate delta is not fixed.")
    return payload


def verify_runtime(*, output_dir: Path) -> dict[str, Any]:
    image = replay_tools._inspect_image(SHIPPING_IMAGE)
    help_result = replay_tools._run(
        [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--entrypoint",
            "/app/llama-server",
            SHIPPING_IMAGE,
            "--help",
        ],
        timeout=60,
    )
    help_text = help_result.stdout or ""
    missing = [
        flag
        for flag in ("--no-kv-offload", "--n-cpu-moe", "--n-gpu-layers")
        if flag not in help_text
    ]
    if missing:
        raise OffloadLabError("Shipping b10133 misses offload options: " + ", ".join(missing))
    volume = replay_tools.verify_model_volume(output_dir=output_dir)
    result = {
        "shipping_image": image,
        "shipping_commit": SHIPPING_COMMIT,
        "shipping_build": SHIPPING_BUILD,
        "help_sha256": hashlib.sha256(help_text.encode("utf-8")).hexdigest(),
        "required_help_options": ["--no-kv-offload", "--n-cpu-moe", "--n-gpu-layers"],
        "model_volume": volume,
    }
    _json_write(output_dir / "runtime-manifest.json", result)
    return result


def _lab_container_name(candidate: OffloadCandidate, *, phase: str, round_index: int) -> str:
    token = hashlib.sha256(
        f"{os.getpid()}:{time.time_ns()}:{candidate.key}:{phase}:{round_index}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{OFFLOAD_LAB_CONTAINER_PREFIX}{candidate.key}-{token}"


def _runtime_config(
    candidate: OffloadCandidate,
    *,
    port: int,
    phase: str,
    round_index: int,
    image_id: str,
) -> ShippingOffloadLabRuntimeConfig:
    candidate.validate()
    config = ShippingOffloadLabRuntimeConfig(
        protocol_version=PROTOCOL_VERSION,
        candidate_key=candidate.key,
        image_ref=SHIPPING_IMAGE,
        model_volume=MODEL_VOLUME,
        model_name=MODEL_NAME,
        model_sha256=MODEL_SHA256,
        port=port,
        container_name=_lab_container_name(candidate, phase=phase, round_index=round_index),
        kv_offload=candidate.kv_offload,
        n_cpu_moe=candidate.n_cpu_moe,
        image_id=image_id,
    )
    config.validate()
    return config


def _read_logs(name: str, path: Path) -> str:
    completed = replay_tools._run(["docker", "logs", "--tail", "1000", name], check=False, timeout=30)
    text = (completed.stdout or "") + (completed.stderr or "")
    path.write_text(text, encoding="utf-8")
    return text


def _residency_estimate(result: Mapping[str, Any]) -> dict[str, Any]:
    preflight = result.get("preflight") if isinstance(result.get("preflight"), Mapping) else {}
    resources = result.get("resource_gates") if isinstance(result.get("resource_gates"), Mapping) else {}
    gpu = preflight.get("gpu") if isinstance(preflight.get("gpu"), Mapping) else {}
    baseline = gpu.get("used_mib")
    peak = resources.get("gpu_peak_mib")
    physical = gpu.get("total_mib")
    if baseline is None or peak is None or physical is None:
        return {"available": False}
    gemma_increment = max(0, int(peak) - int(baseline))
    physical_fit = serving_tools.residency_preflight(
        physical_mib=int(physical),
        paddle_peak_mib=serving_tools.PADDLE_MEASURED_PEAK_MIB,
        gemma_peak_mib=gemma_increment,
        threshold=1.0,
    )
    headroom_95 = serving_tools.residency_preflight(
        physical_mib=int(physical),
        paddle_peak_mib=serving_tools.PADDLE_MEASURED_PEAK_MIB,
        gemma_peak_mib=gemma_increment,
        threshold=0.95,
    )
    return {
        "available": True,
        "gpu_baseline_mib": int(baseline),
        "gemma_increment_mib": gemma_increment,
        "paddle_increment_mib": serving_tools.PADDLE_MEASURED_PEAK_MIB,
        "physical_fit": physical_fit,
        "headroom_95": headroom_95,
    }


def execute_replay_once(
    candidate: OffloadCandidate,
    *,
    payloads: Sequence[Mapping[str, Any]],
    run_dir: Path,
    round_index: int,
    port: int,
    image_id: str,
    phase: str,
) -> dict[str, Any]:
    candidate.validate()
    preflight = require_preflight(output_dir=run_dir)
    config = _runtime_config(
        candidate,
        port=port,
        phase=phase,
        round_index=round_index,
        image_id=image_id,
    )
    manager = Turbo4LabRuntimeManager(inner=None, config=config)  # type: ignore[arg-type]
    sampler = replay_tools.ResourceSampler()
    responses: list[dict[str, Any]] = []
    logs = ""
    error = ""
    request_wall = 0.0
    ready_at: float | None = None
    started = time.perf_counter()
    sampler.start()
    try:
        manager.ensure_server(None, cancel_checker=lambda: sampler.emergency_reason is not None)
        if sampler.emergency_reason is not None:
            raise OffloadLabError(sampler.emergency_reason)
        ready_at = time.perf_counter()
        for payload in payloads:
            responses.append(replay_tools._http_json(f"{config.endpoint_url}/chat/completions", payload=payload))
        request_wall = time.perf_counter() - ready_at
        logs = _read_logs(config.container_name, run_dir / "container.log")
    except (
        HTTPError,
        URLError,
        OSError,
        ValueError,
        subprocess.TimeoutExpired,
        OffloadLabError,
        Turbo4LabRuntimeError,
    ) as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        cgroup_peak = None
        try:
            if not logs:
                logs = _read_logs(config.container_name, run_dir / "container.log")
        except Exception as exc:
            error = error or f"{type(exc).__name__}: {exc}"
        try:
            release = manager.shutdown()
            cgroup_peak = release.get("cgroup_swap_peak_bytes")
        except Exception as exc:  # cleanup evidence must remain fail-closed
            error = error or f"{type(exc).__name__}: {exc}"
        sampler.stop()
    resources = replay_tools.resource_gate_report(
        sampler=sampler,
        cgroup_swap_peak_bytes=(int(cgroup_peak) if cgroup_peak is not None else None),
    )
    release = manager.evidence().get("release")
    if not isinstance(release, Mapping) or not release.get("observed", False):
        resources["failures"].append("gpu_release_unconfirmed")
        resources["passed"] = False
    conflicts = _running_conflicts()
    if conflicts:
        resources["failures"].append("container_orphan")
        resources["passed"] = False
    oom = any(token in logs.lower() for token in ("out of memory", "cuda error", "cuda oom", "oom-kill"))
    if oom:
        resources["failures"].append("oom_detected")
        resources["passed"] = False
    result = {
        "candidate": asdict(candidate),
        "phase": phase,
        "status": "passed" if not error and bool(resources["passed"]) else "rejected",
        "error": error,
        "preflight": preflight,
        "runtime": manager.evidence(),
        "start_to_health_sec": round(max(0.0, (ready_at or time.perf_counter()) - started), 6),
        "request_wall_sec": round(request_wall, 6),
        "request_ledger": replay_tools._request_ledger(payloads),
        "response_ledger": replay_tools._replay_response_ledger(responses),
        "resource_gates": resources,
        "orphan_containers": conflicts,
        "oom_detected": oom,
    }
    result["residency"] = _residency_estimate(result)
    _json_write(run_dir / "raw-replay-requests.json", {"requests": list(payloads)})
    _json_write(run_dir / "raw-replay-responses.json", {"responses": responses})
    _json_write(run_dir / "resource-samples.json", {"samples": sampler.samples})
    _json_write(run_dir / "run-result.json", result)
    return result


def _screen_candidates(levels: Sequence[int]) -> list[OffloadCandidate]:
    # The screen must find the *smallest* offload that fits.  CLI append order
    # is user-controlled, so normalize it rather than letting a larger level
    # win solely because it appeared first.
    unique = [0, *sorted({int(level) for level in levels if int(level) > 0})]
    candidates = [
        OffloadCandidate(key=f"no-kv-moe{level}", kv_offload=False, n_cpu_moe=level)
        for level in unique
    ]
    for candidate in candidates:
        candidate.validate()
    return candidates


def _is_physical_fit(result: Mapping[str, Any]) -> bool:
    residency = result.get("residency") if isinstance(result.get("residency"), Mapping) else {}
    physical = residency.get("physical_fit") if isinstance(residency.get("physical_fit"), Mapping) else {}
    return bool(result.get("status") == "passed" and physical.get("may_run_dual_model", False))


def execute_screen(
    *,
    payloads: Sequence[Mapping[str, Any]],
    levels: Sequence[int],
    output_dir: Path,
    port: int,
    image_id: str,
) -> dict[str, Any]:
    if not payloads:
        raise OffloadLabError("Screen requires one fixed replay payload.")
    runs: list[dict[str, Any]] = []
    selected: int | None = None
    for index, candidate in enumerate(_screen_candidates(levels)):
        run_dir = output_dir / f"screen-{candidate.key}"
        run_dir.mkdir(parents=True, exist_ok=False)
        result = execute_replay_once(
            candidate,
            payloads=payloads[:1],
            run_dir=run_dir,
            round_index=index,
            port=port,
            image_id=image_id,
            phase="screen",
        )
        result["physical_fit"] = _is_physical_fit(result)
        _json_write(run_dir / "run-result.json", result)
        runs.append(result)
        if result["physical_fit"]:
            selected = candidate.n_cpu_moe
            break
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "decision": "PASS" if selected is not None else "REJECT",
        "selected_n_cpu_moe": selected,
        "runs": runs,
    }
    _json_write(output_dir / "screen-summary.json", result)
    return result


def execute_selected_physical_fit_preflight(
    *,
    payloads: Sequence[Mapping[str, Any]],
    n_cpu_moe: int,
    output_dir: Path,
    port: int,
    image_id: str,
) -> dict[str, Any]:
    """Measure exactly one sentinel level before allowing a dual load.

    This is deliberately independent from the structural response gate: the
    sentinel may answer the user's physical-residency question after a quality
    reject, but it must never start Paddle for an unmeasured CPU-MoE level.
    """

    if not payloads:
        raise OffloadLabError("Physical-fit preflight requires one fixed replay payload.")
    candidate = OffloadCandidate(
        key=f"no-kv-moe{int(n_cpu_moe)}",
        kv_offload=False,
        n_cpu_moe=int(n_cpu_moe),
    )
    candidate.validate()
    run_dir = output_dir / f"physical-fit-{candidate.key}"
    run_dir.mkdir(parents=True, exist_ok=False)
    run = execute_replay_once(
        candidate,
        payloads=payloads[:1],
        run_dir=run_dir,
        round_index=0,
        port=port,
        image_id=image_id,
        phase="co-resident-preflight",
    )
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": asdict(candidate),
        "decision": "PASS" if _is_physical_fit(run) else "REJECT",
        "run": run,
    }
    _json_write(output_dir / "physical-fit-summary.json", result)
    return result


def execute_structural(
    *,
    payloads: Sequence[Mapping[str, Any]],
    n_cpu_moe: int,
    output_dir: Path,
    port: int,
    image_id: str,
) -> dict[str, Any]:
    baseline_dir = output_dir / "shipping-f16"
    candidate_dir = output_dir / f"no-kv-moe{n_cpu_moe}"
    baseline_dir.mkdir(parents=True, exist_ok=False)
    candidate_dir.mkdir(parents=True, exist_ok=False)
    baseline = execute_replay_once(
        OffloadCandidate(key="shipping-f16", kv_offload=True, n_cpu_moe=0),
        payloads=payloads,
        run_dir=baseline_dir,
        round_index=0,
        port=port,
        image_id=image_id,
        phase="structural",
    )
    candidate = execute_replay_once(
        OffloadCandidate(key=f"no-kv-moe{n_cpu_moe}", kv_offload=False, n_cpu_moe=n_cpu_moe),
        payloads=payloads,
        run_dir=candidate_dir,
        round_index=1,
        port=port,
        image_id=image_id,
        phase="structural",
    )
    same_requests = baseline["request_ledger"]["sha256"] == candidate["request_ledger"]["sha256"]
    same_responses = baseline["response_ledger"]["sha256"] == candidate["response_ledger"]["sha256"]
    residency = candidate.get("residency") if isinstance(candidate.get("residency"), Mapping) else {}
    physical = residency.get("physical_fit") if isinstance(residency.get("physical_fit"), Mapping) else {}
    decision = "PASS" if (
        baseline["status"] == "passed"
        and candidate["status"] == "passed"
        and same_requests
        and same_responses
        and physical.get("may_run_dual_model", False)
    ) else "REJECT"
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "decision": decision,
        "request_ledger_exact": same_requests,
        "response_ledger_exact": same_responses,
        "baseline": baseline,
        "candidate": candidate,
    }
    _json_write(output_dir / "structural-summary.json", result)
    return result


def _remove_paddle_lab(name: str) -> None:
    if not name.startswith(serving_tools.LAB_CONTAINER_PREFIX):
        raise OffloadLabError("Refusing to remove a non-lab Paddle container.")
    replay_tools._run(["docker", "stop", "--timeout", "10", name], check=False, timeout=30)
    replay_tools._run(["docker", "rm", name], check=False, timeout=30)


def execute_co_resident_probe(
    *,
    payload: Mapping[str, Any],
    n_cpu_moe: int,
    output_dir: Path,
    port: int,
    paddle_port: int,
    image_id: str,
) -> dict[str, Any]:
    """Load both models only after an exact-level physical-fit preflight."""

    preflight = require_preflight(output_dir=output_dir)
    candidate = OffloadCandidate(key=f"no-kv-moe{n_cpu_moe}", kv_offload=False, n_cpu_moe=n_cpu_moe)
    config = _runtime_config(candidate, port=port, phase="co-resident", round_index=0, image_id=image_id)
    manager = Turbo4LabRuntimeManager(inner=None, config=config)  # type: ignore[arg-type]
    paddle_name = f"{serving_tools.LAB_CONTAINER_PREFIX}offload-{hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:12]}"
    sampler = replay_tools.ResourceSampler()
    error = ""
    gemma_response: dict[str, Any] = {}
    paddle_started: dict[str, Any] = {}
    gemma_logs = ""
    paddle_logs = ""
    cgroup_peak: int | None = None
    sampler.start()
    try:
        manager.ensure_server(None, cancel_checker=lambda: sampler.emergency_reason is not None)
        paddle_started = serving_tools._start_paddle_container(
            serving_tools.BASELINE,
            name=paddle_name,
            port=paddle_port,
        )
        replay_tools._http_json(f"http://127.0.0.1:{paddle_port}/health")
        gemma_response = replay_tools._http_json(f"{config.endpoint_url}/chat/completions", payload=payload)
        gemma_logs = _read_logs(config.container_name, output_dir / "gemma-container.log")
        paddle_logs = _read_logs(paddle_name, output_dir / "paddle-container.log")
    except (
        HTTPError,
        URLError,
        OSError,
        ValueError,
        subprocess.TimeoutExpired,
        OffloadLabError,
        Turbo4LabRuntimeError,
        serving_tools.BenchmarkContractError,
    ) as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if not gemma_logs:
                gemma_logs = _read_logs(config.container_name, output_dir / "gemma-container.log")
            if not paddle_logs:
                paddle_logs = _read_logs(paddle_name, output_dir / "paddle-container.log")
        except Exception as exc:
            error = error or f"{type(exc).__name__}: {exc}"
        try:
            _remove_paddle_lab(paddle_name)
        except Exception as exc:
            error = error or f"{type(exc).__name__}: {exc}"
        finally:
            try:
                release = manager.shutdown()
                raw_cgroup_peak = release.get("cgroup_swap_peak_bytes")
                cgroup_peak = int(raw_cgroup_peak) if raw_cgroup_peak is not None else None
            except Exception as exc:
                error = error or f"{type(exc).__name__}: {exc}"
            finally:
                sampler.stop()
    resources = replay_tools.resource_gate_report(
        sampler=sampler,
        cgroup_swap_peak_bytes=cgroup_peak,
    )
    release = manager.evidence().get("release")
    if not isinstance(release, Mapping) or not release.get("observed", False):
        resources["failures"].append("gpu_release_unconfirmed")
        resources["passed"] = False
    conflicts = _running_conflicts()
    if conflicts:
        resources["failures"].append("container_orphan")
        resources["passed"] = False
    gpu = preflight.get("gpu") if isinstance(preflight.get("gpu"), Mapping) else {}
    actual_peak = resources.get("gpu_peak_mib")
    physical = gpu.get("total_mib")
    actual_fit = actual_peak is not None and physical is not None and int(actual_peak) <= int(physical)
    logs = (gemma_logs + "\n" + paddle_logs).lower()
    oom = any(token in logs for token in ("out of memory", "cuda error", "cuda oom", "oom-kill"))
    if oom:
        resources["failures"].append("oom_detected")
        resources["passed"] = False
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": asdict(candidate),
        "status": "PASS" if not error and resources["passed"] and actual_fit else "REJECT",
        "error": error,
        "preflight": preflight,
        "paddle_start": paddle_started,
        "gemma_response_ledger": replay_tools._replay_response_ledger([gemma_response]),
        "resource_gates": resources,
        "actual_physical_fit": actual_fit,
        "orphan_containers": conflicts,
        "oom_detected": oom,
    }
    _json_write(output_dir / "co-resident-summary.json", result)
    return result


def _print_redacted(value: Mapping[str, Any]) -> None:
    screen = value.get("screen") if isinstance(value.get("screen"), Mapping) else {}
    structural = value.get("structural") if isinstance(value.get("structural"), Mapping) else {}
    co_resident = value.get("co_resident") if isinstance(value.get("co_resident"), Mapping) else {}
    print(json.dumps({
        "decision": value.get("decision"),
        "selected_n_cpu_moe": value.get("selected_n_cpu_moe", screen.get("selected_n_cpu_moe")),
        "structural": structural.get("decision"),
        "co_resident": co_resident.get("status"),
    }, ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the private Gemma CPU-MoE/KV offload lab.")
    parser.add_argument(
        "--mode",
        choices=("plan", "screen", "structural", "co-resident", "co-resident-sentinel", "all"),
        default="plan",
    )
    parser.add_argument("--translation-replay", type=Path)
    parser.add_argument("--n-cpu-moe", type=int, action="append", dest="cpu_moe_levels")
    parser.add_argument("--selected-n-cpu-moe", type=int)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--paddle-port", type=int, default=DEFAULT_PADDLE_PORT)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = load_protocol()
    if args.mode == "plan":
        print(json.dumps({"protocol_version": protocol["protocol_version"], "order": ["screen", "structural", "co-resident"]}, ensure_ascii=False))
        return 0
    if args.translation_replay is None:
        raise OffloadLabError("--translation-replay is required for every execution mode.")
    payloads = replay_tools.load_translation_replay(args.translation_replay)
    output_dir, managed_run = select_managed_output_directory(
        family=FAMILY_NAME,
        category=ARTIFACT_CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"protocol_version": PROTOCOL_VERSION}
    try:
        runtime = verify_runtime(output_dir=output_dir)
        image_id = str((runtime["shipping_image"] or {}).get("id", ""))
        if not image_id:
            raise OffloadLabError("Shipping image ID is required.")
        levels = tuple(args.cpu_moe_levels or DEFAULT_CPU_MOE_LEVELS)
        screen: Mapping[str, Any] | None = None
        selected = args.selected_n_cpu_moe
        if args.mode in {"screen", "all"}:
            screen = execute_screen(
                payloads=payloads,
                levels=levels,
                output_dir=output_dir / "screen",
                port=args.port,
                image_id=image_id,
            )
            result["screen"] = screen
            selected = screen.get("selected_n_cpu_moe")
            if selected is None:
                result["decision"] = "REJECT"
                _json_write(output_dir / "final-summary.json", result)
                _print_redacted(result)
                return 2
        if selected is not None:
            result["selected_n_cpu_moe"] = int(selected)
        if args.mode in {"structural", "co-resident", "all"}:
            if selected is None:
                raise OffloadLabError("A selected physical-fit n_cpu_moe level is required.")
            structural = execute_structural(
                payloads=payloads,
                n_cpu_moe=int(selected),
                output_dir=output_dir / "structural",
                port=args.port,
                image_id=image_id,
            )
            result["structural"] = structural
            if structural["decision"] != "PASS":
                result["decision"] = "REJECT"
                _json_write(output_dir / "final-summary.json", result)
                _print_redacted(result)
                return 2
        if args.mode in {"co-resident", "co-resident-sentinel", "all"}:
            if selected is None:
                raise OffloadLabError("A selected physical-fit n_cpu_moe level is required.")
            if args.mode == "co-resident-sentinel":
                physical_fit_preflight = execute_selected_physical_fit_preflight(
                    payloads=payloads,
                    n_cpu_moe=int(selected),
                    output_dir=output_dir / "physical-fit-preflight",
                    port=args.port,
                    image_id=image_id,
                )
                result["physical_fit_preflight"] = physical_fit_preflight
                if physical_fit_preflight["decision"] != "PASS":
                    result["decision"] = "REJECT"
                    _json_write(output_dir / "final-summary.json", result)
                    _print_redacted(result)
                    return 2
            co_resident = execute_co_resident_probe(
                payload=payloads[0],
                n_cpu_moe=int(selected),
                output_dir=output_dir / "co-resident",
                port=args.port,
                paddle_port=args.paddle_port,
                image_id=image_id,
            )
            result["co_resident"] = co_resident
            result["decision"] = co_resident["status"]
        else:
            result["decision"] = "PASS"
        _json_write(output_dir / "final-summary.json", result)
        _print_redacted(result)
        return 0 if result["decision"] == "PASS" else 2
    except BaseException as exc:
        if managed_run is not None:
            managed_run.fail(
                exc,
                metadata={"protocol_version": PROTOCOL_VERSION, "mode": args.mode},
            )
        raise
    finally:
        if managed_run is not None:
            managed_run.complete(metadata={"protocol_version": PROTOCOL_VERSION, "mode": args.mode, "decision": result.get("decision", "ERROR")})


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OffloadLabError as exc:
        print(f"[gemma-cpu-moe-offload] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
