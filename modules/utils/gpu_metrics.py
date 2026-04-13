from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CONTAINER_NAMES = (
    "gemma-local-server",
    "paddleocr-server",
    "paddleocr-vllm",
    "mangalmm-local-server",
)


def _run_capture(cmd: list[str]) -> str:
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return (completed.stdout or "").strip()


def _query_gpu_rows() -> list[dict[str, Any]]:
    output = _run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory",
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
        if len(parts) < 6:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_used_mb": int(parts[3]),
                    "memory_free_mb": int(parts[4]),
                    "gpu_util_percent": int(parts[5]),
                    "memory_util_percent": int(parts[6]) if len(parts) > 6 else None,
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
    names = list(DEFAULT_CONTAINER_NAMES if container_names is None else container_names)
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
