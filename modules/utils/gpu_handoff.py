from __future__ import annotations

import gc
import sys
import time
from typing import Any, Callable

from modules.utils.gpu_metrics import query_cuda_handoff_metrics


DEFAULT_VRAM_RELEASE_TIMEOUT_SEC = 5.0
DEFAULT_VRAM_RELEASE_POLL_SEC = 0.1
DEFAULT_VRAM_RELEASE_MIN_DROP_MB = 16.0
DEFAULT_VRAM_RELEASE_EXPECTED_RATIO = 0.9
DEFAULT_VRAM_BASELINE_TOLERANCE_MB = 16.0
# A sleeping llama.cpp server keeps its process, CUDA context, and small
# reusable buffers after unloading the model. This bounds that process-only
# residue; it does not allow a model to remain resident.
DEFAULT_MANAGED_SLEEPING_RESIDUAL_MB = 512.0
DEFAULT_MANAGED_SLEEPING_RELEASE_RATIO = 0.85


def estimate_torch_cuda_storage_mb(resource: Any) -> dict[str, Any]:
    """Estimate unique CUDA parameter/buffer storage retained by a torch model."""

    if resource is None:
        return {
            "available": False,
            "storage_count": 0,
            "total_mb": 0.0,
        }
    tensors: list[Any] = []
    recognized = False
    pending = [resource]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        object_id = id(current)
        if object_id in visited:
            continue
        visited.add(object_id)
        has_tensor_accessor = False
        for accessor_name in ("parameters", "buffers"):
            accessor = getattr(current, accessor_name, None)
            if not callable(accessor):
                continue
            has_tensor_accessor = True
            recognized = True
            try:
                tensors.extend(list(accessor()))
            except Exception:
                continue
        if has_tensor_accessor:
            continue
        try:
            child_values = vars(current).values()
        except TypeError:
            continue
        for child in child_values:
            if callable(getattr(child, "parameters", None)) or callable(
                getattr(child, "buffers", None)
            ):
                pending.append(child)

    storages: dict[tuple[str, int], int] = {}
    for tensor in tensors:
        device = str(getattr(tensor, "device", "") or "").lower()
        if not device.startswith("cuda"):
            continue
        try:
            storage = tensor.untyped_storage()
            storage_ptr = int(storage.data_ptr())
            storage_bytes = int(storage.nbytes())
        except Exception:
            try:
                storage_ptr = int(tensor.data_ptr())
                storage_bytes = int(tensor.numel()) * int(tensor.element_size())
            except Exception:
                continue
        key = (device, storage_ptr)
        storages[key] = max(storages.get(key, 0), storage_bytes)

    return {
        "available": recognized,
        "storage_count": len(storages),
        "total_mb": float(sum(storages.values())) / 1024.0 / 1024.0,
    }


def cleanup_python_cuda_memory(
    *,
    release_cuda_allocator: bool = True,
) -> dict[str, Any]:
    """Run targeted Python/CUDA allocator cleanup after model references are gone."""

    report: dict[str, Any] = {
        "gc_collected": 0,
        "cuda_cleanup_requested": bool(release_cuda_allocator),
        "cuda_available": False,
        "cuda_synchronized": False,
        "cuda_cache_emptied": False,
        "cuda_ipc_collected": False,
        "errors": [],
    }
    try:
        report["gc_collected"] = int(gc.collect())
    except Exception as exc:
        report["errors"].append(f"gc.collect: {type(exc).__name__}: {exc}")

    if not release_cuda_allocator:
        return report

    torch = sys.modules.get("torch")
    if torch is None:
        return report
    try:
        cuda = torch.cuda
        report["cuda_available"] = bool(cuda.is_available())
    except Exception as exc:
        report["errors"].append(f"torch.cuda.is_available: {type(exc).__name__}: {exc}")
        return report
    if not report["cuda_available"]:
        return report

    try:
        cuda.synchronize()
        report["cuda_synchronized"] = True
    except Exception as exc:
        report["errors"].append(f"torch.cuda.synchronize: {type(exc).__name__}: {exc}")
    try:
        cuda.empty_cache()
        report["cuda_cache_emptied"] = True
    except Exception as exc:
        report["errors"].append(f"torch.cuda.empty_cache: {type(exc).__name__}: {exc}")
    try:
        cuda.ipc_collect()
        report["cuda_ipc_collected"] = True
    except Exception as exc:
        report["errors"].append(f"torch.cuda.ipc_collect: {type(exc).__name__}: {exc}")
    try:
        report["gc_collected"] += int(gc.collect())
    except Exception as exc:
        report["errors"].append(f"second gc.collect: {type(exc).__name__}: {exc}")
    return report


def _metric_value(payload: dict[str, Any], *path: str) -> float | None:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return None
    return float(current)


def _driver_process_uuid(payload: dict[str, Any]) -> str:
    selected = payload.get("driver_process", {}).get("selected")
    if not isinstance(selected, dict):
        return ""
    return str(selected.get("gpu_uuid") or "").strip()


def _driver_process_used_mb(
    payload: dict[str, Any],
    *,
    gpu_uuid: str,
) -> float | None:
    driver_process = payload.get("driver_process")
    if not isinstance(driver_process, dict) or not driver_process.get("query_available"):
        return None
    target_uuid = str(gpu_uuid or "").strip()
    if not target_uuid:
        return None
    rows = driver_process.get("rows")
    if not isinstance(rows, list):
        return None
    total = 0.0
    matched = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("gpu_uuid") or "").strip() != target_uuid:
            continue
        value = row.get("memory_used_mb")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        total += float(value)
        matched = True
    return total if matched else 0.0


def _global_gpu_memory_used_mb(
    payload: dict[str, Any],
    *,
    gpu_uuid: str = "",
) -> tuple[str, float | None]:
    """Return one GPU's driver-total usage without assuming a Python PID.

    Docker CUDA processes are not necessarily attributed to the Python process
    that owns the UI.  Managed llama.cpp runtime release therefore needs the
    driver-wide reading for the same GPU, while native inpainter release can
    continue to use the stronger per-process allocator evidence above.
    """

    driver = payload.get("driver")
    if not isinstance(driver, dict) or not bool(driver.get("available")):
        return "", None
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
        return "", None
    primary_uuid = str(primary.get("uuid") or "").strip()
    if target_uuid and primary_uuid != target_uuid:
        return target_uuid, None
    value = primary.get("memory_used_mb")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return primary_uuid, None
    return primary_uuid, float(value)


def _release_deltas(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    driver_gpu_uuid: str = "",
) -> dict[str, float | None]:
    before_allocated = _metric_value(before, "process", "allocated_mb")
    after_allocated = _metric_value(after, "process", "allocated_mb")
    before_reserved = _metric_value(before, "process", "reserved_mb")
    after_reserved = _metric_value(after, "process", "reserved_mb")
    before_driver = _driver_process_used_mb(before, gpu_uuid=driver_gpu_uuid)
    after_driver = _driver_process_used_mb(after, gpu_uuid=driver_gpu_uuid)
    return {
        "process_allocated_drop_mb": (
            before_allocated - after_allocated
            if before_allocated is not None and after_allocated is not None
            else None
        ),
        "process_reserved_drop_mb": (
            before_reserved - after_reserved
            if before_reserved is not None and after_reserved is not None
            else None
        ),
        "process_driver_used_drop_mb": (
            before_driver - after_driver
            if before_driver is not None and after_driver is not None
            else None
        ),
    }


def wait_for_vram_release(
    before: dict[str, Any],
    *,
    gpu_release_expected: bool,
    expected_process_drop_mb: float = 0.0,
    untracked_gpu_resource_count: int = 0,
    driver_baseline: dict[str, Any] | None = None,
    timeout_sec: float = DEFAULT_VRAM_RELEASE_TIMEOUT_SEC,
    poll_interval_sec: float = DEFAULT_VRAM_RELEASE_POLL_SEC,
    min_drop_mb: float = DEFAULT_VRAM_RELEASE_MIN_DROP_MB,
    expected_drop_ratio: float = DEFAULT_VRAM_RELEASE_EXPECTED_RATIO,
    baseline_tolerance_mb: float = DEFAULT_VRAM_BASELINE_TOLERANCE_MB,
    sampler: Callable[[], dict[str, Any]] = query_cuda_handoff_metrics,
) -> dict[str, Any]:
    """Wait until target-sized allocator or PID/device baseline evidence proves release."""

    started = time.monotonic()
    minimum_drop = max(0.0, float(min_drop_mb))
    expected_process_drop = max(0.0, float(expected_process_drop_mb))
    process_threshold = (
        max(
            minimum_drop,
            expected_process_drop * max(0.0, min(1.0, float(expected_drop_ratio))),
        )
        if expected_process_drop > 0.0
        else 0.0
    )
    untracked_count = max(0, int(untracked_gpu_resource_count))
    process_available = bool((before.get("process") or {}).get("available"))
    process_evidence_required = expected_process_drop > 0.0
    driver_evidence_required = untracked_count > 0
    driver_gpu_uuid = _driver_process_uuid(before)
    driver_before = _driver_process_used_mb(before, gpu_uuid=driver_gpu_uuid)
    driver_baseline_used = _driver_process_used_mb(
        driver_baseline or {},
        gpu_uuid=driver_gpu_uuid,
    )
    driver_evidence_available = bool(
        driver_gpu_uuid
        and driver_before is not None
        and driver_baseline_used is not None
    )
    measurement_available = bool(
        (process_evidence_required and process_available)
        or (driver_evidence_required and driver_evidence_available)
    )
    required = bool(gpu_release_expected)

    if not gpu_release_expected:
        after = sampler()
        return {
            "required": False,
            "measurement_available": measurement_available,
            "observed": True,
            "status": "not-required",
            "evidence_source": "not-required",
            "process_threshold_mb": process_threshold,
            "driver_gpu_uuid": driver_gpu_uuid,
            "driver_baseline_mb": driver_baseline_used,
            "elapsed_sec": time.monotonic() - started,
            "before": before,
            "after": after,
            "deltas": _release_deltas(
                before,
                after,
                driver_gpu_uuid=driver_gpu_uuid,
            ),
        }
    if (
        (process_evidence_required and not process_available)
        or (driver_evidence_required and not driver_evidence_available)
        or (not process_evidence_required and not driver_evidence_required)
    ):
        after = sampler()
        return {
            "required": True,
            "measurement_available": False,
            "observed": False,
            "status": "unavailable",
            "evidence_source": "unavailable",
            "process_threshold_mb": process_threshold,
            "driver_gpu_uuid": driver_gpu_uuid,
            "driver_baseline_mb": driver_baseline_used,
            "elapsed_sec": time.monotonic() - started,
            "before": before,
            "after": after,
            "deltas": _release_deltas(
                before,
                after,
                driver_gpu_uuid=driver_gpu_uuid,
            ),
        }

    deadline = started + max(0.0, float(timeout_sec))
    last_after: dict[str, Any] = {}
    last_deltas: dict[str, float | None] = {}
    observed = False
    evidence_parts: list[str] = []
    if process_evidence_required:
        evidence_parts.append("process-allocated")
    if driver_evidence_required:
        evidence_parts.append("pid-device-driver-baseline")
    evidence_source = "+".join(evidence_parts)
    while True:
        last_after = sampler()
        last_deltas = _release_deltas(
            before,
            last_after,
            driver_gpu_uuid=driver_gpu_uuid,
        )
        process_observed = True
        if process_evidence_required:
            observed_drop = last_deltas.get("process_allocated_drop_mb")
            process_observed = bool(
                isinstance(observed_drop, (int, float))
                and observed_drop >= process_threshold
            )
        driver_observed = True
        if driver_evidence_required:
            after_driver = _driver_process_used_mb(
                last_after,
                gpu_uuid=driver_gpu_uuid,
            )
            driver_drop = last_deltas.get("process_driver_used_drop_mb")
            driver_observed = bool(
                isinstance(after_driver, (int, float))
                and isinstance(driver_drop, (int, float))
                and driver_drop >= minimum_drop
                and after_driver
                <= float(driver_baseline_used) + max(0.0, float(baseline_tolerance_mb))
            )
        if process_observed and driver_observed:
            observed = True
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(max(0.01, float(poll_interval_sec)))

    return {
        "required": required,
        "measurement_available": measurement_available,
        "observed": observed,
        "status": "observed" if observed else "timeout",
        "evidence_source": evidence_source,
        "process_threshold_mb": process_threshold,
        "driver_gpu_uuid": driver_gpu_uuid,
        "driver_baseline_mb": driver_baseline_used,
        "elapsed_sec": time.monotonic() - started,
        "before": before,
        "after": last_after,
        "deltas": last_deltas,
    }


def wait_for_global_vram_release(
    before: dict[str, Any],
    *,
    gpu_release_expected: bool,
    driver_baseline: dict[str, Any] | None = None,
    timeout_sec: float = DEFAULT_VRAM_RELEASE_TIMEOUT_SEC,
    poll_interval_sec: float = DEFAULT_VRAM_RELEASE_POLL_SEC,
    min_drop_mb: float = DEFAULT_VRAM_RELEASE_MIN_DROP_MB,
    expected_drop_ratio: float = DEFAULT_VRAM_RELEASE_EXPECTED_RATIO,
    baseline_tolerance_mb: float = DEFAULT_VRAM_BASELINE_TOLERANCE_MB,
    residual_allowance_mb: float = 0.0,
    sampler: Callable[[], dict[str, Any]] = query_cuda_handoff_metrics,
) -> dict[str, Any]:
    """Verify managed-runtime VRAM release with driver-total GPU memory.

    A managed Docker server has a different PID from the UI process, so a
    process allocator reading cannot prove its release.  This gate requires a
    model-sized drop from the pre-stop sample and, when a pre-load baseline
    exists, a return close to that baseline. A sleeping llama.cpp process may
    receive a small explicit CUDA-context allowance, but it must still release
    the configured fraction of its model-load delta. Missing or ambiguous GPU
    measurements fail closed when a release was expected.
    """

    started = time.monotonic()
    minimum_drop = max(0.0, float(min_drop_mb))
    tolerance = max(0.0, float(baseline_tolerance_mb))
    residual_allowance = max(0.0, float(residual_allowance_mb))
    drop_ratio = max(0.0, min(1.0, float(expected_drop_ratio)))
    gpu_uuid, before_used = _global_gpu_memory_used_mb(before)
    baseline_uuid, baseline_used = _global_gpu_memory_used_mb(
        driver_baseline or {},
        gpu_uuid=gpu_uuid,
    )
    if not gpu_uuid:
        gpu_uuid = baseline_uuid
        if gpu_uuid:
            _baseline_uuid, before_used = _global_gpu_memory_used_mb(
                before,
                gpu_uuid=gpu_uuid,
            )

    if not gpu_release_expected:
        after = sampler()
        _after_uuid, after_used = _global_gpu_memory_used_mb(
            after,
            gpu_uuid=gpu_uuid,
        )
        return {
            "required": False,
            "measurement_available": before_used is not None,
            "observed": True,
            "status": "not-required",
            "evidence_source": "not-required",
            "gpu_uuid": gpu_uuid,
            "before_used_mb": before_used,
            "baseline_used_mb": baseline_used,
            "after_used_mb": after_used,
            "drop_mb": (
                before_used - after_used
                if before_used is not None and after_used is not None
                else None
            ),
            "expected_drop_mb": 0.0,
            "expected_drop_ratio": drop_ratio,
            "residual_allowance_mb": residual_allowance,
            "elapsed_sec": time.monotonic() - started,
        }

    if before_used is None or not gpu_uuid:
        after = sampler()
        _after_uuid, after_used = _global_gpu_memory_used_mb(
            after,
            gpu_uuid=gpu_uuid,
        )
        return {
            "required": True,
            "measurement_available": False,
            "observed": False,
            "status": "unavailable",
            "evidence_source": "driver-global-memory",
            "gpu_uuid": gpu_uuid,
            "before_used_mb": before_used,
            "baseline_used_mb": baseline_used,
            "after_used_mb": after_used,
            "drop_mb": (
                before_used - after_used
                if before_used is not None and after_used is not None
                else None
            ),
            "expected_drop_mb": None,
            "expected_drop_ratio": drop_ratio,
            "residual_allowance_mb": residual_allowance,
            "elapsed_sec": time.monotonic() - started,
        }

    model_load_delta = (
        before_used - baseline_used
        if baseline_used is not None
        else None
    )
    expected_drop = max(
        minimum_drop,
        float(model_load_delta) * drop_ratio
        if isinstance(model_load_delta, (int, float))
        and model_load_delta > 0.0
        else 0.0,
    )
    deadline = started + max(0.0, float(timeout_sec))
    after: dict[str, Any] = {}
    after_used: float | None = None
    observed = False
    while True:
        after = sampler()
        _after_uuid, after_used = _global_gpu_memory_used_mb(
            after,
            gpu_uuid=gpu_uuid,
        )
        drop_mb = (
            before_used - after_used
            if after_used is not None
            else None
        )
        drop_observed = bool(
            isinstance(drop_mb, (int, float)) and drop_mb >= expected_drop
        )
        baseline_observed = bool(
            baseline_used is None
            or (
                isinstance(after_used, (int, float))
                and after_used
                <= baseline_used + tolerance + residual_allowance
            )
        )
        if drop_observed and baseline_observed:
            observed = True
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(max(0.01, float(poll_interval_sec)))

    drop_mb = (
        before_used - after_used
        if after_used is not None
        else None
    )
    return {
        "required": True,
        "measurement_available": after_used is not None,
        "observed": observed,
        "status": "observed" if observed else "timeout",
        "evidence_source": "driver-global-memory",
        "gpu_uuid": gpu_uuid,
        "before_used_mb": before_used,
        "baseline_used_mb": baseline_used,
        "after_used_mb": after_used,
        "drop_mb": drop_mb,
        "minimum_drop_mb": minimum_drop,
        "expected_drop_mb": expected_drop,
        "expected_drop_ratio": drop_ratio,
        "baseline_tolerance_mb": tolerance,
        "residual_allowance_mb": residual_allowance,
        "elapsed_sec": time.monotonic() - started,
    }
