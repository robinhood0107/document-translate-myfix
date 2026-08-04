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
    "paddleocr-llamacpp",
)
_GPU_METRICS_CACHE_LOCK = threading.Lock()
_GPU_METRICS_CACHE_VALUE: dict[str, Any] | None = None
_GPU_METRICS_CACHE_EXPIRES_AT = 0.0
_ROUTER_NVIDIA_SMI_LOCK = threading.Lock()
_ROUTER_NVIDIA_SMI_PREFIX: tuple[str, ...] | None = None


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


def _windows_wsl_distribution_names() -> tuple[str, ...]:
    """Return usable user distributions when this module runs in Windows Python."""

    try:
        completed = subprocess.run(
            ["wsl.exe", "--list", "--quiet"],
            check=False,
            capture_output=True,
            timeout=5.0,
        )
    except Exception:
        return ()
    if completed.returncode != 0:
        return ()
    raw = completed.stdout or b""
    try:
        text = raw.decode("utf-16le") if b"\x00" in raw else raw.decode("utf-8")
    except UnicodeDecodeError:
        return ()
    excluded = {"docker-desktop", "docker-desktop-data"}
    names = [
        value.strip()
        for value in text.replace("\x00", "").splitlines()
        if value.strip() and value.strip().lower() not in excluded
    ]
    return tuple(names)


def _router_nvidia_smi_prefix() -> tuple[str, ...]:
    """Choose the Linux driver view for Docker Router evidence when available.

    ``.venv-win`` launches Windows Python even when the repository itself is
    opened through WSL. Windows ``nvidia-smi.exe`` sees desktop processes and
    represents Docker's CUDA client as an un-attributable System PID. The WSL
    driver view instead reports the Router worker's container namespace PID,
    which the adapter can prove belongs to the owned container.
    """

    global _ROUTER_NVIDIA_SMI_PREFIX
    with _ROUTER_NVIDIA_SMI_LOCK:
        if _ROUTER_NVIDIA_SMI_PREFIX is not None:
            return _ROUTER_NVIDIA_SMI_PREFIX
        if os.name == "nt":
            distributions = _windows_wsl_distribution_names()
            if distributions:
                _ROUTER_NVIDIA_SMI_PREFIX = (
                    "wsl.exe",
                    "-d",
                    distributions[0],
                    "--exec",
                    "/usr/lib/wsl/lib/nvidia-smi",
                )
                return _ROUTER_NVIDIA_SMI_PREFIX
        # Do not switch to the Windows driver view after a WSL distribution
        # has been selected. A transient WSL command failure must make the
        # Router handoff unproven rather than mixing PID namespaces mid-gate.
        # Native nvidia-smi is only the fallback for systems without WSL and
        # for direct Linux execution, where it is already the correct view.
        _ROUTER_NVIDIA_SMI_PREFIX = ("nvidia-smi",)
        return _ROUTER_NVIDIA_SMI_PREFIX


def _run_router_nvidia_smi(args: list[str]) -> tuple[bool, str]:
    prefix = _router_nvidia_smi_prefix()
    return _run_capture_status([*prefix, *args])


def _parse_gpu_rows(output: str) -> list[dict[str, Any]]:
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


def _query_gpu_rows() -> list[dict[str, Any]]:
    return _parse_gpu_rows(
        _run_capture(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ]
        )
    )


def _query_router_gpu_rows() -> list[dict[str, Any]]:
    _succeeded, output = _run_router_nvidia_smi(
        [
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ]
    )
    return _parse_gpu_rows(output)


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


def query_router_gpu_metrics() -> dict[str, Any]:
    """Return GPU totals from the same Linux view as a Docker Router."""

    rows = _query_router_gpu_rows()
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


def _parse_gpu_compute_processes(
    query_succeeded: bool,
    output: str,
) -> dict[str, Any]:
    """Parse a driver process listing, retaining PID-only WSL entries."""

    rows: list[dict[str, Any]] = []
    if query_succeeded:
        for raw_line in output.splitlines():
            parts = [part.strip() for part in raw_line.split(",")]
            if len(parts) < 4:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            try:
                used_mb: float | None = float(parts[-1])
            except ValueError:
                # Docker Desktop / WSL can expose a valid container-namespace
                # PID while reporting ``[Not Found]`` and ``[N/A]`` for its
                # process metadata. The memory value is unusable for an
                # attribution total, but dropping this row would also lose the
                # driver-visible process identity required by Router handoff.
                used_mb = None
            rows.append(
                {
                    "pid": pid,
                    "gpu_uuid": parts[1],
                    "process_name": ",".join(parts[2:-1]).strip(),
                    "memory_used_mb": used_mb,
                    "memory_reported": used_mb is not None,
                }
            )
    return {
        "query_available": query_succeeded,
        "rows": rows,
        "gpu_uuids": sorted(
            {
                str(row.get("gpu_uuid") or "").strip()
                for row in rows
                if str(row.get("gpu_uuid") or "").strip()
            }
        ),
    }


def query_gpu_compute_processes() -> dict[str, Any]:
    """Return the driver-visible compute-process set for every GPU.

    Router release checks need more than total memory: an unrelated process
    entering or leaving the target GPU makes a global-memory delta ambiguous.
    The result deliberately carries an explicit query status so callers can
    fail closed rather than treating an unavailable process list as empty.
    """

    query_succeeded, output = _run_capture_status(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    return _parse_gpu_compute_processes(query_succeeded, output)


def query_router_gpu_compute_processes() -> dict[str, Any]:
    """Return process identities from the same Linux view as a Docker Router."""

    query_succeeded, output = _run_router_nvidia_smi(
        [
            "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    return _parse_gpu_compute_processes(query_succeeded, output)


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
        "driver_processes": query_gpu_compute_processes(),
    }


def query_router_cuda_handoff_metrics() -> dict[str, Any]:
    """Collect Docker Router evidence without mixing Windows desktop PIDs.

    The generic sampler is retained for native/in-process CUDA paths. Router
    state transitions use this dedicated sampler so its GPU UUID, memory
    totals, and process set all originate from the Docker-visible Linux driver
    view.
    """

    process = query_process_cuda_metrics()
    return {
        "sampled_at": time.time(),
        "process": process,
        "driver": query_router_gpu_metrics(),
        "driver_process": {
            "query_available": False,
            "available": False,
            "pid": os.getpid(),
            "preferred_gpu_uuid": str(process.get("device_uuid") or ""),
            "rows": [],
            "selected": None,
            "reason": "router-uses-container-driver-view",
        },
        "driver_processes": query_router_gpu_compute_processes(),
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
