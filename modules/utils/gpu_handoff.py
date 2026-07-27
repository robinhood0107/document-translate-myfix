from __future__ import annotations

import gc
import sys
import time
from typing import Any, Callable

from modules.utils.gpu_metrics import query_cuda_handoff_metrics


DEFAULT_VRAM_RELEASE_TIMEOUT_SEC = 5.0
DEFAULT_VRAM_RELEASE_POLL_SEC = 0.1
DEFAULT_VRAM_RELEASE_MIN_DROP_MB = 16.0


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


def _driver_used_mb(payload: dict[str, Any]) -> float | None:
    primary = payload.get("driver", {}).get("primary")
    if not isinstance(primary, dict):
        return None
    value = primary.get("memory_used_mb")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _release_deltas(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, float | None]:
    before_allocated = _metric_value(before, "process", "allocated_mb")
    after_allocated = _metric_value(after, "process", "allocated_mb")
    before_reserved = _metric_value(before, "process", "reserved_mb")
    after_reserved = _metric_value(after, "process", "reserved_mb")
    before_driver = _driver_used_mb(before)
    after_driver = _driver_used_mb(after)
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
        "driver_used_drop_mb": (
            before_driver - after_driver
            if before_driver is not None and after_driver is not None
            else None
        ),
    }


def wait_for_vram_release(
    before: dict[str, Any],
    *,
    gpu_release_expected: bool,
    timeout_sec: float = DEFAULT_VRAM_RELEASE_TIMEOUT_SEC,
    poll_interval_sec: float = DEFAULT_VRAM_RELEASE_POLL_SEC,
    min_drop_mb: float = DEFAULT_VRAM_RELEASE_MIN_DROP_MB,
    sampler: Callable[[], dict[str, Any]] = query_cuda_handoff_metrics,
) -> dict[str, Any]:
    """Wait until process or driver metrics prove that VRAM was returned."""

    started = time.monotonic()
    threshold = max(0.0, float(min_drop_mb))
    process_available = bool((before.get("process") or {}).get("available"))
    driver_available = bool((before.get("driver") or {}).get("available"))
    measurement_available = process_available or driver_available
    required = bool(gpu_release_expected and measurement_available)

    if not gpu_release_expected:
        after = sampler()
        return {
            "required": False,
            "measurement_available": measurement_available,
            "observed": True,
            "status": "not-required",
            "threshold_mb": threshold,
            "elapsed_sec": time.monotonic() - started,
            "before": before,
            "after": after,
            "deltas": _release_deltas(before, after),
        }
    if not measurement_available:
        after = sampler()
        return {
            "required": False,
            "measurement_available": False,
            "observed": None,
            "status": "unavailable",
            "threshold_mb": threshold,
            "elapsed_sec": time.monotonic() - started,
            "before": before,
            "after": after,
            "deltas": _release_deltas(before, after),
        }

    deadline = started + max(0.0, float(timeout_sec))
    last_after: dict[str, Any] = {}
    last_deltas: dict[str, float | None] = {}
    observed = False
    while True:
        last_after = sampler()
        last_deltas = _release_deltas(before, last_after)
        positive_drops = [
            value
            for value in last_deltas.values()
            if isinstance(value, (int, float))
        ]
        if any(value >= threshold for value in positive_drops):
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
        "threshold_mb": threshold,
        "elapsed_sec": time.monotonic() - started,
        "before": before,
        "after": last_after,
        "deltas": last_deltas,
    }
