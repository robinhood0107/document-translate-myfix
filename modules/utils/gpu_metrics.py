from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CONTAINER_NAMES = (
    "gemma-local-server",
    "paddleocr-server",
    "paddleocr-vllm",
)
_GPU_METRICS_CACHE_LOCK = threading.Lock()
_GPU_METRICS_CACHE_VALUE: dict[str, Any] | None = None
_GPU_METRICS_CACHE_EXPIRES_AT = 0.0


def _run_capture_status(
    cmd: list[str],
    *,
    timeout_sec: float = 5.0,
) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=max(0.1, float(timeout_sec)),
        )
    except Exception:
        return False, ""
    return True, (completed.stdout or "").strip()


def _run_capture(cmd: list[str], *, timeout_sec: float = 5.0) -> str:
    _succeeded, output = _run_capture_status(cmd, timeout_sec=timeout_sec)
    return output


def _query_gpu_rows() -> list[dict[str, Any]]:
    output = _run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []

    rows: list[dict[str, Any]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "uuid": parts[1],
                    "name": parts[2],
                    "memory_total_mb": int(parts[3]),
                    "memory_used_mb": int(parts[4]),
                    "memory_free_mb": int(parts[5]),
                    "gpu_util_percent": int(parts[6]),
                    "memory_util_percent": int(parts[7]) if len(parts) > 7 else None,
                }
            )
        except ValueError:
            continue
    return rows


def query_gpu_metrics() -> dict[str, Any]:
    rows = _query_gpu_rows()
    primary = rows[0] if rows else None
    return {
        "available": bool(rows),
        "gpu_count": len(rows),
        "gpus": rows,
        "primary": primary,
        "sampled_at": time.time(),
    }


def query_process_driver_gpu_metrics(
    *,
    pid: int | None = None,
    preferred_gpu_uuid: str | None = None,
) -> dict[str, Any]:
    """Return driver memory attributed to this PID and an exact GPU UUID."""

    requested_pid = int(pid if pid is not None else os.getpid())
    query_succeeded, output = _run_capture_status(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: list[dict[str, Any]] = []
    if query_succeeded:
        for raw_line in output.splitlines():
            parts = [part.strip() for part in raw_line.split(",")]
            if len(parts) < 3:
                continue
            try:
                row_pid = int(parts[0])
                used_mb = float(parts[2])
            except ValueError:
                continue
            if row_pid != requested_pid:
                continue
            rows.append(
                {
                    "pid": row_pid,
                    "gpu_uuid": parts[1],
                    "memory_used_mb": used_mb,
                }
            )

    preferred = str(preferred_gpu_uuid or "").strip()
    candidate_uuids = {str(row["gpu_uuid"]) for row in rows}
    selected_uuid = ""
    if preferred and preferred in candidate_uuids:
        selected_uuid = preferred
    elif not preferred and len(candidate_uuids) == 1:
        selected_uuid = next(iter(candidate_uuids))
    selected_rows = [
        row for row in rows if selected_uuid and row["gpu_uuid"] == selected_uuid
    ]
    selected = (
        {
            "pid": requested_pid,
            "gpu_uuid": selected_uuid,
            "memory_used_mb": sum(float(row["memory_used_mb"]) for row in selected_rows),
        }
        if selected_rows
        else None
    )
    reason = ""
    if not query_succeeded:
        reason = "nvidia-smi-query-failed"
    elif preferred and preferred not in candidate_uuids:
        reason = "preferred-gpu-process-not-found"
    elif len(candidate_uuids) > 1 and not preferred:
        reason = "ambiguous-process-gpu"
    elif not rows:
        reason = "process-gpu-memory-not-reported"
    return {
        "query_available": query_succeeded,
        "available": selected is not None,
        "pid": requested_pid,
        "preferred_gpu_uuid": preferred,
        "rows": rows,
        "selected": selected,
        "reason": reason,
    }


def query_process_cuda_metrics() -> dict[str, Any]:
    """Return CUDA allocator state for this Python process without importing torch.

    Inference code imports torch before this is useful. Avoiding a new import here
    keeps diagnostics from loading the CUDA runtime in CPU-only workflows.
    """

    torch = sys.modules.get("torch")
    if torch is None:
        return {
            "available": False,
            "reason": "torch-not-loaded",
        }
    try:
        cuda = torch.cuda
        if not bool(cuda.is_available()):
            return {
                "available": False,
                "reason": "cuda-unavailable",
            }
        device_index = int(cuda.current_device())
        try:
            properties = cuda.get_device_properties(device_index)
            device_uuid = str(getattr(properties, "uuid", "") or "").strip()
        except Exception:
            device_uuid = ""
        return {
            "available": True,
            "device_index": device_index,
            "device_uuid": device_uuid,
            "allocated_mb": float(cuda.memory_allocated(device_index)) / 1024.0 / 1024.0,
            "reserved_mb": float(cuda.memory_reserved(device_index)) / 1024.0 / 1024.0,
            "max_allocated_mb": float(cuda.max_memory_allocated(device_index)) / 1024.0 / 1024.0,
            "max_reserved_mb": float(cuda.max_memory_reserved(device_index)) / 1024.0 / 1024.0,
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def query_cuda_handoff_metrics() -> dict[str, Any]:
    """Collect process allocator and driver-visible GPU memory for a handoff."""

    process = query_process_cuda_metrics()
    return {
        "sampled_at": time.time(),
        "process": process,
        "driver": query_gpu_metrics(),
        "driver_process": query_process_driver_gpu_metrics(
            preferred_gpu_uuid=str(process.get("device_uuid") or ""),
        ),
    }


def query_gpu_metrics_cached(ttl_sec: float = 1.0) -> dict[str, Any]:
    global _GPU_METRICS_CACHE_VALUE, _GPU_METRICS_CACHE_EXPIRES_AT

    try:
        ttl = max(0.0, float(ttl_sec))
    except (TypeError, ValueError):
        ttl = 1.0

    now = time.monotonic()
    with _GPU_METRICS_CACHE_LOCK:
        if _GPU_METRICS_CACHE_VALUE is not None and now < _GPU_METRICS_CACHE_EXPIRES_AT:
            return copy.deepcopy(_GPU_METRICS_CACHE_VALUE)

    fresh = query_gpu_metrics()
    with _GPU_METRICS_CACHE_LOCK:
        _GPU_METRICS_CACHE_VALUE = copy.deepcopy(fresh)
        _GPU_METRICS_CACHE_EXPIRES_AT = now + ttl
    return copy.deepcopy(fresh)


def _parse_linux_swap_meminfo(
    text: str,
    *,
    source: str,
) -> dict[str, Any]:
    values_kb: dict[str, int] = {}
    for raw_line in str(text or "").splitlines():
        if ":" not in raw_line:
            continue
        key, raw_value = raw_line.split(":", 1)
        normalized_key = key.strip()
        if normalized_key not in {"SwapTotal", "SwapFree", "SwapCached"}:
            continue
        token = raw_value.strip().split()[0] if raw_value.strip() else ""
        try:
            values_kb[normalized_key] = max(0, int(token))
        except (TypeError, ValueError):
            continue
    if "SwapTotal" not in values_kb or "SwapFree" not in values_kb:
        return {
            "available": False,
            "source": source,
            "reason": "swap-meminfo-unavailable",
        }
    total_mb = float(values_kb["SwapTotal"]) / 1024.0
    free_mb = float(values_kb["SwapFree"]) / 1024.0
    cached_mb = float(values_kb.get("SwapCached", 0)) / 1024.0
    return {
        "available": True,
        "source": source,
        "swap_total_mb": round(total_mb, 3),
        "swap_free_mb": round(free_mb, 3),
        "swap_cached_mb": round(cached_mb, 3),
        "swap_used_mb": round(max(0.0, total_mb - free_mb), 3),
        "sampled_at": time.time(),
    }


def _current_process_is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    if not sys.platform.startswith("linux"):
        return False
    try:
        osrelease = Path("/proc/sys/kernel/osrelease").read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return False
    return "microsoft" in osrelease.lower()


def query_wsl_swap_metrics(*, timeout_sec: float = 2.0) -> dict[str, Any]:
    """Return WSL swap usage without changing WSL or Docker state."""

    if _current_process_is_wsl():
        try:
            meminfo = Path("/proc/meminfo").read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            return {
                "available": False,
                "source": "current-wsl",
                "reason": "meminfo-read-failed",
            }
        return _parse_linux_swap_meminfo(meminfo, source="current-wsl")

    if os.name != "nt":
        return {
            "available": False,
            "source": "none",
            "reason": "not-wsl-or-windows",
        }

    succeeded, meminfo = _run_capture_status(
        [
            "wsl.exe",
            "-d",
            "docker-desktop",
            "--",
            "cat",
            "/proc/meminfo",
        ],
        timeout_sec=max(0.1, float(timeout_sec)),
    )
    if not succeeded or not meminfo:
        return {
            "available": False,
            "source": "docker-desktop-wsl",
            "reason": "docker-desktop-wsl-query-failed",
        }
    return _parse_linux_swap_meminfo(
        meminfo,
        source="docker-desktop-wsl",
    )


def _docker_ps_rows(container_names: Iterable[str] | None = None) -> list[dict[str, Any]]:
    requested = {name for name in (container_names or []) if name}
    output = _run_capture(["docker", "ps", "--format", "{{json .}}"])
    if not output:
        return []

    rows: list[dict[str, Any]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = str(row.get("Names", "") or "")
        if requested and name not in requested:
            continue
        rows.append(row)
    return rows


def _docker_inspect_rows(container_names: Iterable[str]) -> dict[str, dict[str, Any]]:
    names = [name for name in container_names if name]
    if not names:
        return {}

    output = _run_capture(["docker", "inspect", *names])
    if not output:
        return {}

    try:
        items = json.loads(output)
    except json.JSONDecodeError:
        return {}

    snapshot: dict[str, dict[str, Any]] = {}
    for item in items:
        name = str(item.get("Name", "") or "").lstrip("/")
        if not name:
            continue
        state = item.get("State", {}) or {}
        config = item.get("Config", {}) or {}
        host_config = item.get("HostConfig", {}) or {}
        snapshot[name] = {
            "name": name,
            "image": config.get("Image", ""),
            "cmd": config.get("Cmd", []) or [],
            "entrypoint": config.get("Entrypoint", []) or [],
            "status": state.get("Status", ""),
            "running": bool(state.get("Running", False)),
            "health": ((state.get("Health") or {}).get("Status")),
            "restart_count": item.get("RestartCount", 0),
            "device_requests": host_config.get("DeviceRequests", []) or [],
            "ports": ((item.get("NetworkSettings") or {}).get("Ports")) or {},
        }
    return snapshot


def collect_runtime_snapshot(
    container_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    names = list(container_names or DEFAULT_CONTAINER_NAMES)
    ps_rows = _docker_ps_rows(names)
    inspect_rows = _docker_inspect_rows(names)
    return {
        "sampled_at": time.time(),
        "container_names": names,
        "docker_ps": ps_rows,
        "containers": inspect_rows,
        "gpu": query_gpu_metrics(),
    }


def write_snapshot_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
