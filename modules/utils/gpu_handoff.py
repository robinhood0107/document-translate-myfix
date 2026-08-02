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
DEFAULT_ROUTER_VRAM_RELEASE_TIMEOUT_SEC = 30.0
DEFAULT_ROUTER_CONTAINER_RESIDUAL_MB = 528.0
DEFAULT_ROUTER_STOPPED_BASELINE_TOLERANCE_MB = 16.0
DEFAULT_ROUTER_CONTAINER_RELEASE_RATIO = 0.85
DEFAULT_ROUTER_STOPPED_RELEASE_RATIO = 0.90


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


def _gpu_process_set(
    payload: dict[str, Any],
    *,
    gpu_uuid: str,
) -> frozenset[int] | None:
    """Return one GPU's driver process IDs, or ``None`` when unmeasurable."""

    process_payload = payload.get("driver_processes")
    if not isinstance(process_payload, dict) or not bool(
        process_payload.get("query_available")
    ):
        return None
    rows = process_payload.get("rows")
    if not isinstance(rows, list):
        return None
    selected: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            return None
        if str(row.get("gpu_uuid") or "").strip() != str(gpu_uuid or "").strip():
            continue
        pid = row.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            return None
        selected.add(pid)
    return frozenset(selected)


def router_gpu_process_set(
    payload: dict[str, Any],
    *,
    gpu_uuid: str,
) -> tuple[frozenset[int] | None, str]:
    """Return the attributable Router worker set for one GPU.

    Normal Linux Docker installations expose the model child through NVML, so
    the driver process table remains the strongest evidence.  NVIDIA documents
    that WSL NVML does not support active-compute-process queries, however: it
    can report a large GPU memory delta while returning an empty process table.
    The Router adapter therefore supplies an independently verified fallback
    only for that case.  Each fallback row is an exact owned Router child whose
    command matches a configured model and whose ``/dev/dxg`` handle is open.

    This helper deliberately does *not* treat a missing/malformed fallback as
    an empty process set.  Callers must fail closed when neither source can
    prove ownership.
    """

    driver_processes = _gpu_process_set(payload, gpu_uuid=gpu_uuid)
    if driver_processes is None:
        return None, "driver"
    if driver_processes:
        return driver_processes, "driver"

    workers = payload.get("router_worker_processes")
    if isinstance(workers, dict):
        if not bool(workers.get("query_available")):
            # The owned-container adapter explicitly attempted the WSL
            # fallback and could not prove its worker state. Do not turn that
            # failure into an empty process set merely because NVML has the
            # documented active-compute-process blind spot.
            return None, "router-worker-dxg"
        rows = workers.get("rows")
        if not isinstance(rows, list):
            return None, "router-worker-dxg"
        worker_ids: set[int] = set()
        for row in rows:
            if not isinstance(row, dict):
                return None, "router-worker-dxg"
            if str(row.get("gpu_uuid") or "").strip() != str(gpu_uuid or "").strip():
                return None, "router-worker-dxg"
            pid = row.get("pid")
            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
                return None, "router-worker-dxg"
            if not bool(row.get("gpu_device_attached")):
                return None, "router-worker-dxg"
            worker_ids.add(pid)
        # NVML already reported no active PID above. Keep the adapter's
        # container-namespace identity only for this documented WSL gap.
        return frozenset(worker_ids), "router-worker-dxg"

    # A generic post-stop sampler has no owned-container fallback at all. In
    # that case an empty *driver* process set is valid evidence only because
    # the caller already proved the container was stopped.
    return driver_processes, "driver"


def wait_for_router_vram_release(
    before: dict[str, Any],
    *,
    model_load_baseline: dict[str, Any],
    container_baseline: dict[str, Any],
    container_kept: bool,
    owned_router_process_ids: frozenset[int],
    timeout_sec: float = DEFAULT_ROUTER_VRAM_RELEASE_TIMEOUT_SEC,
    poll_interval_sec: float = DEFAULT_VRAM_RELEASE_POLL_SEC,
    sampler: Callable[[], dict[str, Any]] = query_cuda_handoff_metrics,
) -> dict[str, Any]:
    """Prove Router model release on the same GPU without ambiguous deltas.

    The normal (container-kept) path permits only the Router process/context
    residue.  The terminal path additionally requires the process set to
    return to the pre-container baseline and attributes every disappeared GPU
    process to the owned Router container.  Any missing measurement, UUID
    mismatch, or concurrent process-set change fails closed.
    """

    started = time.monotonic()
    gpu_uuid, before_used = _global_gpu_memory_used_mb(before)
    baseline_uuid, model_baseline_used = _global_gpu_memory_used_mb(
        model_load_baseline,
        gpu_uuid=gpu_uuid,
    )
    container_uuid, container_baseline_used = _global_gpu_memory_used_mb(
        container_baseline,
        gpu_uuid=gpu_uuid,
    )
    if not gpu_uuid:
        gpu_uuid = baseline_uuid or container_uuid
        if gpu_uuid:
            _ignored, before_used = _global_gpu_memory_used_mb(
                before,
                gpu_uuid=gpu_uuid,
            )
            _ignored, model_baseline_used = _global_gpu_memory_used_mb(
                model_load_baseline,
                gpu_uuid=gpu_uuid,
            )
            _ignored, container_baseline_used = _global_gpu_memory_used_mb(
                container_baseline,
                gpu_uuid=gpu_uuid,
            )

    before_processes, before_process_source = router_gpu_process_set(
        before,
        gpu_uuid=gpu_uuid,
    )
    model_baseline_processes, model_baseline_process_source = router_gpu_process_set(
        model_load_baseline,
        gpu_uuid=gpu_uuid,
    )
    container_baseline_processes, container_baseline_process_source = router_gpu_process_set(
        container_baseline,
        gpu_uuid=gpu_uuid,
    )
    required_baseline = (
        model_baseline_used if container_kept else container_baseline_used
    )
    required_processes = (
        model_baseline_processes if container_kept else container_baseline_processes
    )
    required_process_source = (
        model_baseline_process_source
        if container_kept
        else container_baseline_process_source
    )
    required_ratio = (
        DEFAULT_ROUTER_CONTAINER_RELEASE_RATIO
        if container_kept
        else DEFAULT_ROUTER_STOPPED_RELEASE_RATIO
    )
    allowed_residual = (
        DEFAULT_ROUTER_CONTAINER_RESIDUAL_MB
        if container_kept
        else DEFAULT_ROUTER_STOPPED_BASELINE_TOLERANCE_MB
    )
    model_load_delta = (
        before_used - model_baseline_used
        if isinstance(before_used, (int, float))
        and isinstance(model_baseline_used, (int, float))
        else None
    )
    measurement_available = bool(
        gpu_uuid
        and isinstance(before_used, (int, float))
        and isinstance(model_baseline_used, (int, float))
        and isinstance(required_baseline, (int, float))
        and before_processes is not None
        and required_processes is not None
        and isinstance(model_load_delta, (int, float))
        and model_load_delta > 0.0
    )
    if not container_kept:
        measurement_available = bool(
            measurement_available
            and owned_router_process_ids
        )
    if not measurement_available:
        after = sampler()
        return {
            "required": True,
            "measurement_available": False,
            "observed": False,
            "status": "unavailable",
            "gpu_uuid": gpu_uuid,
            "before_used_mb": before_used,
            "model_baseline_used_mb": model_baseline_used,
            "container_baseline_used_mb": container_baseline_used,
            "container_kept": container_kept,
            "owned_router_process_ids": sorted(owned_router_process_ids),
            "before_processes": sorted(before_processes or ()),
            "required_processes": sorted(required_processes or ()),
            "before_process_source": before_process_source,
            "required_process_source": required_process_source,
            "after": after,
            "elapsed_sec": time.monotonic() - started,
        }

    expected_drop = float(model_load_delta) * required_ratio
    deadline = started + max(0.0, float(timeout_sec))
    after: dict[str, Any] = {}
    after_used: float | None = None
    after_processes: frozenset[int] | None = None
    after_process_source = ""
    process_set_reason = ""
    observed = False
    while True:
        after = sampler()
        after_uuid, after_used = _global_gpu_memory_used_mb(after, gpu_uuid=gpu_uuid)
        after_processes, after_process_source = router_gpu_process_set(
            after,
            gpu_uuid=gpu_uuid,
        )
        same_gpu = after_uuid == gpu_uuid and after_used is not None
        drop_mb = before_used - after_used if after_used is not None else None
        memory_returned = bool(
            isinstance(drop_mb, (int, float))
            and drop_mb >= expected_drop
            and after_used <= float(required_baseline) + allowed_residual
        )
        process_set_ok = False
        if after_processes is None:
            process_set_reason = "process-set-unavailable"
        elif container_kept:
            removed = before_processes - after_processes
            added = after_processes - before_processes
            process_set_ok = (
                after_processes == required_processes
                and not added
                and removed.issubset(owned_router_process_ids)
            )
            if not process_set_ok:
                process_set_reason = "container-kept-process-set-not-attributable"
        else:
            removed = before_processes - after_processes
            added = after_processes - before_processes
            process_set_ok = bool(
                after_processes == required_processes
                and not added
                and removed
                and removed.issubset(owned_router_process_ids)
            )
            if not process_set_ok:
                process_set_reason = "container-stop-process-set-not-attributable"
        if same_gpu and memory_returned and process_set_ok:
            observed = True
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(max(0.01, float(poll_interval_sec)))

    drop_mb = before_used - after_used if after_used is not None else None
    return {
        "required": True,
        "measurement_available": True,
        "observed": observed,
        "status": "observed" if observed else "timeout",
        "gpu_uuid": gpu_uuid,
        "container_kept": container_kept,
        "before_used_mb": before_used,
        "model_baseline_used_mb": model_baseline_used,
        "container_baseline_used_mb": container_baseline_used,
        "after_used_mb": after_used,
        "model_load_delta_mb": model_load_delta,
        "drop_mb": drop_mb,
        "required_drop_mb": expected_drop,
        "required_drop_ratio": required_ratio,
        "allowed_baseline_residual_mb": allowed_residual,
        "before_processes": sorted(before_processes),
        "required_processes": sorted(required_processes),
        "after_processes": sorted(after_processes or ()),
        "before_process_source": before_process_source,
        "required_process_source": required_process_source,
        "after_process_source": after_process_source,
        "owned_router_process_ids": sorted(owned_router_process_ids),
        "process_set_reason": process_set_reason,
        "elapsed_sec": time.monotonic() - started,
    }


def wait_for_router_container_stop_release(
    before: dict[str, Any],
    *,
    container_baseline: dict[str, Any],
    owned_router_process_ids: frozenset[int],
    timeout_sec: float = DEFAULT_ROUTER_VRAM_RELEASE_TIMEOUT_SEC,
    poll_interval_sec: float = DEFAULT_VRAM_RELEASE_POLL_SEC,
    sampler: Callable[[], dict[str, Any]] = query_cuda_handoff_metrics,
) -> dict[str, Any]:
    """Prove terminal Router cleanup when no model is currently loaded.

    A failed setup, pair switch, or app shutdown can need to stop a Router
    that is already at loaded-model count zero. There is no model-load delta
    to compare in that state, so require the stronger container baseline and
    exact process-set return instead. A process set that changes for an
    unowned PID remains an ambiguous GPU handoff and fails closed.
    """

    started = time.monotonic()
    gpu_uuid, before_used = _global_gpu_memory_used_mb(before)
    baseline_uuid, baseline_used = _global_gpu_memory_used_mb(
        container_baseline,
        gpu_uuid=gpu_uuid,
    )
    if not gpu_uuid:
        gpu_uuid = baseline_uuid
        if gpu_uuid:
            _ignored, before_used = _global_gpu_memory_used_mb(
                before,
                gpu_uuid=gpu_uuid,
            )
            _ignored, baseline_used = _global_gpu_memory_used_mb(
                container_baseline,
                gpu_uuid=gpu_uuid,
            )

    before_processes, before_process_source = router_gpu_process_set(
        before,
        gpu_uuid=gpu_uuid,
    )
    baseline_processes, baseline_process_source = router_gpu_process_set(
        container_baseline,
        gpu_uuid=gpu_uuid,
    )
    container_delta = (
        before_used - baseline_used
        if isinstance(before_used, (int, float))
        and isinstance(baseline_used, (int, float))
        else None
    )
    measurement_available = bool(
        gpu_uuid
        and isinstance(before_used, (int, float))
        and isinstance(baseline_used, (int, float))
        and before_processes is not None
        and baseline_processes is not None
        and owned_router_process_ids
    )
    if not measurement_available:
        after = sampler()
        return {
            "required": True,
            "measurement_available": False,
            "observed": False,
            "status": "unavailable",
            "gpu_uuid": gpu_uuid,
            "before_used_mb": before_used,
            "container_baseline_used_mb": baseline_used,
            "container_delta_mb": container_delta,
            "owned_router_process_ids": sorted(owned_router_process_ids),
            "before_processes": sorted(before_processes or ()),
            "required_processes": sorted(baseline_processes or ()),
            "before_process_source": before_process_source,
            "required_process_source": baseline_process_source,
            "after": after,
            "elapsed_sec": time.monotonic() - started,
        }

    required_drop = max(0.0, float(container_delta or 0.0)) * (
        DEFAULT_ROUTER_STOPPED_RELEASE_RATIO
    )
    deadline = started + max(0.0, float(timeout_sec))
    after: dict[str, Any] = {}
    after_used: float | None = None
    after_processes: frozenset[int] | None = None
    after_process_source = ""
    process_set_reason = ""
    observed = False
    while True:
        after = sampler()
        after_uuid, after_used = _global_gpu_memory_used_mb(
            after,
            gpu_uuid=gpu_uuid,
        )
        after_processes, after_process_source = router_gpu_process_set(
            after,
            gpu_uuid=gpu_uuid,
        )
        drop_mb = before_used - after_used if after_used is not None else None
        # A zero-model Router can sample a few MiB below its pre-container
        # baseline before it is stopped. In that case there is no container
        # allocation to return, so demanding a non-negative "drop" turns
        # harmless sampling jitter into a false release failure. The stricter
        # baseline +16 MiB bound still applies. A positive container delta
        # continues to require the promised 90% return.
        memory_returned = bool(
            after_used is not None
            and after_used
            <= float(baseline_used) + DEFAULT_ROUTER_STOPPED_BASELINE_TOLERANCE_MB
            and (
                not isinstance(container_delta, (int, float))
                or container_delta <= 0.0
                or (
                    isinstance(drop_mb, (int, float))
                    and drop_mb >= required_drop
                )
            )
        )
        process_set_ok = False
        if after_processes is None:
            process_set_reason = "process-set-unavailable"
        else:
            removed = before_processes - after_processes
            added = after_processes - before_processes
            process_set_ok = bool(
                after_processes == baseline_processes
                and not added
                and removed.issubset(owned_router_process_ids)
            )
            if not process_set_ok:
                process_set_reason = "container-stop-process-set-not-attributable"
        if after_uuid == gpu_uuid and memory_returned and process_set_ok:
            observed = True
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(max(0.01, float(poll_interval_sec)))

    drop_mb = before_used - after_used if after_used is not None else None
    return {
        "required": True,
        "measurement_available": True,
        "observed": observed,
        "status": "observed" if observed else "timeout",
        "gpu_uuid": gpu_uuid,
        "before_used_mb": before_used,
        "container_baseline_used_mb": baseline_used,
        "after_used_mb": after_used,
        "container_delta_mb": container_delta,
        "drop_mb": drop_mb,
        "required_drop_mb": required_drop,
        "required_drop_ratio": DEFAULT_ROUTER_STOPPED_RELEASE_RATIO,
        "allowed_baseline_residual_mb": DEFAULT_ROUTER_STOPPED_BASELINE_TOLERANCE_MB,
        "before_processes": sorted(before_processes),
        "required_processes": sorted(baseline_processes),
        "after_processes": sorted(after_processes or ()),
        "before_process_source": before_process_source,
        "required_process_source": baseline_process_source,
        "after_process_source": after_process_source,
        "owned_router_process_ids": sorted(owned_router_process_ids),
        "process_set_reason": process_set_reason,
        "elapsed_sec": time.monotonic() - started,
    }


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
