#!/usr/bin/env python3
"""Verify that active managed inference contracts do not start vLLM."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.ocr.managed_backend_policy import (  # noqa: E402
    find_vllm_process_commands,
)


ACTIVE_COMPOSE_FILES = (
    ROOT_DIR / "docker-compose.yaml",
    ROOT_DIR / "docker-compose.gemma-host-rollback.yaml",
    ROOT_DIR / "hunyuanocr_docker_files" / "docker-compose.yaml",
    ROOT_DIR / "mangalmm_docker_files" / "docker-compose.yaml",
    ROOT_DIR / "paddleocr_vl_docker_files" / "docker-compose.yaml",
    ROOT_DIR
    / "paddleocr_vl_spotting_docker_files"
    / "docker-compose.yaml",
)
ACTIVE_CONTAINER_NAMES = (
    "gemma-local-server",
    "hunyuanocr-local-server",
    "mangalmm-local-server",
    "paddleocr-llamacpp",
    "paddleocr-server",
    "paddleocr-spotting-llamacpp",
)
PADDLE_PIPELINE_CONFIG = (
    ROOT_DIR / "paddleocr_vl_docker_files" / "pipeline_conf.yaml"
)


class ManagedRuntimeVerificationError(RuntimeError):
    pass


def _launch_text(service: dict[str, Any]) -> str:
    values: list[str] = []
    for key in ("entrypoint", "command"):
        value = service.get(key, "")
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        elif value:
            values.append(str(value))
    return " ".join(values)


def verify_static_contracts() -> dict[str, Any]:
    checked_services: list[str] = []
    for compose_file in ACTIVE_COMPOSE_FILES:
        payload = yaml.safe_load(compose_file.read_text(encoding="utf-8"))
        services = payload.get("services", {}) if isinstance(payload, dict) else {}
        if not isinstance(services, dict) or not services:
            raise ManagedRuntimeVerificationError(
                f"Managed Compose has no services: {compose_file}"
            )
        for service_name, raw_service in services.items():
            service = raw_service if isinstance(raw_service, dict) else {}
            if "vllm" in str(service_name).lower():
                raise ManagedRuntimeVerificationError(
                    f"Active managed service still names vLLM: {service_name}"
                )
            violations = find_vllm_process_commands(_launch_text(service))
            if violations:
                raise ManagedRuntimeVerificationError(
                    f"Active managed command starts vLLM: {service_name}: "
                    + " | ".join(violations)
                )
            checked_services.append(str(service_name))

    pipeline = yaml.safe_load(
        PADDLE_PIPELINE_CONFIG.read_text(encoding="utf-8")
    )
    genai = pipeline["SubModules"]["VLRecognition"]["genai_config"]
    if genai.get("backend") != "llama-cpp-server":
        raise ManagedRuntimeVerificationError(
            "Paddle crop relay is not pinned to llama-cpp-server."
        )
    if genai.get("server_url") != "http://paddleocr-llamacpp:8080/v1":
        raise ManagedRuntimeVerificationError(
            "Paddle crop relay does not target the managed llama.cpp service."
        )
    return {
        "mode": "static",
        "compose_count": len(ACTIVE_COMPOSE_FILES),
        "services": checked_services,
        "paddle_backend": genai["backend"],
    }


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def verify_live_processes() -> dict[str, Any]:
    checked: list[str] = []
    for container_name in ACTIVE_CONTAINER_NAMES:
        state = _docker(
            "inspect",
            "--format",
            "{{.State.Running}}",
            container_name,
        )
        detail = ((state.stdout or "") + (state.stderr or "")).lower()
        if state.returncode != 0 and (
            "no such object" in detail or "no such container" in detail
        ):
            continue
        if state.returncode != 0:
            raise ManagedRuntimeVerificationError(
                f"Unable to inspect managed container {container_name}: "
                f"{detail.strip()}"
            )
        if (state.stdout or "").strip().lower() != "true":
            continue
        process_table = _docker("top", container_name, "-eo", "args")
        if process_table.returncode != 0:
            raise ManagedRuntimeVerificationError(
                f"Unable to inspect process tree for {container_name}: "
                f"{(process_table.stderr or '').strip()}"
            )
        violations = find_vllm_process_commands(process_table.stdout or "")
        if violations:
            raise ManagedRuntimeVerificationError(
                f"Managed container {container_name} is running vLLM: "
                + " | ".join(violations)
            )
        checked.append(container_name)
    return {"mode": "live", "running_containers": checked}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also inspect process trees of currently running managed containers.",
    )
    args = parser.parse_args()
    try:
        payload = {"static": verify_static_contracts()}
        if args.live:
            payload["live"] = verify_live_processes()
    except (ManagedRuntimeVerificationError, OSError, KeyError, TypeError) as exc:
        print(f"Managed llama.cpp verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
