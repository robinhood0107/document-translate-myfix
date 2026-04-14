from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_LLAMA_CPP_IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda"
DEFAULT_LLAMA_CPP_PULL_POLICY = "always"
WINDOWS_DOCKER_BIN = Path(r"C:\Program Files\Docker\Docker\resources\bin")


def normalize_llama_cpp_image(image_ref: Any = None) -> str:
    text = str(image_ref or "").strip()
    if not text:
        return DEFAULT_LLAMA_CPP_IMAGE
    if "ggml-org/llama.cpp" in text or "local/llama.cpp" in text:
        return DEFAULT_LLAMA_CPP_IMAGE
    return text


def normalize_llama_cpp_pull_policy(_: Any = None) -> str:
    return DEFAULT_LLAMA_CPP_PULL_POLICY


def resolve_docker_executable() -> str:
    docker = shutil.which("docker")
    if docker:
        return docker
    docker_exe = shutil.which("docker.exe")
    if docker_exe:
        return docker_exe
    windows_docker = WINDOWS_DOCKER_BIN / "docker.exe"
    if windows_docker.is_file():
        return str(windows_docker)
    return "docker"


def _augment_env_with_docker_bin(env: dict[str, str] | None = None) -> dict[str, str] | None:
    if not WINDOWS_DOCKER_BIN.is_dir():
        return env
    merged = dict(os.environ if env is None else env)
    path_value = merged.get("PATH", "")
    docker_bin = str(WINDOWS_DOCKER_BIN)
    if docker_bin.lower() not in path_value.lower():
        merged["PATH"] = f"{docker_bin}{os.pathsep}{path_value}" if path_value else docker_bin
    return merged


def _format_command_failure(
    resolved_cmd: list[str],
    completed: subprocess.CompletedProcess[str],
    *,
    cwd: str | Path | None = None,
) -> str:
    return (
        f"Command failed (exit={completed.returncode}): {' '.join(resolved_cmd)}\n"
        f"cwd={cwd}\n"
        f"stdout:\n{(completed.stdout or '').strip()}\n"
        f"stderr:\n{(completed.stderr or '').strip()}"
    )


def run_docker_command(
    cmd: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    resolved_cmd = list(cmd)
    if resolved_cmd and resolved_cmd[0] == "docker":
        resolved_cmd[0] = resolve_docker_executable()
    env = _augment_env_with_docker_bin(env)
    completed = subprocess.run(
        resolved_cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(_format_command_failure(resolved_cmd, completed, cwd=cwd))
    return completed


def resolve_docker_compose_command() -> tuple[str, ...]:
    candidates: list[tuple[str, ...]] = []
    resolved_docker = resolve_docker_executable()
    if resolved_docker and (shutil.which("docker") or shutil.which("docker.exe") or Path(resolved_docker).is_file()):
        candidates.append(("docker", "compose"))
    if shutil.which("docker-compose"):
        candidates.append(("docker-compose",))
    windows_docker_compose = Path(r"C:\Program Files\Docker\Docker\resources\bin\docker-compose.exe")
    if windows_docker_compose.is_file():
        candidates.append((str(windows_docker_compose),))

    for candidate in candidates:
        probe = run_docker_command([*candidate, "version"], check=False)
        if probe.returncode == 0:
            if candidate[0] == "docker":
                return (resolve_docker_executable(), *candidate[1:])
            return candidate
    raise RuntimeError("Docker Compose is not available.")


def docker_compose_pull_latest(
    compose_file: str | Path,
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return run_docker_command(
        [*resolve_docker_compose_command(), "-f", str(compose_file), "pull", "--policy", "always"],
        cwd=cwd,
        env=env,
    )


def _is_retryable_compose_up_failure(completed: subprocess.CompletedProcess[str]) -> bool:
    details = ((completed.stdout or "") + "\n" + (completed.stderr or "")).lower()
    retry_markers = (
        'removal of container',
        'already in progress',
        'container name',
        'is already in use',
        'network has active endpoints',
    )
    return any(marker in details for marker in retry_markers)


def docker_compose_up_force_recreate(
    compose_file: str | Path,
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    resolved_cmd = [*resolve_docker_compose_command(), "-f", str(compose_file), "up", "-d", "--force-recreate"]
    last_completed: subprocess.CompletedProcess[str] | None = None
    for attempt in range(5):
        completed = run_docker_command(resolved_cmd, cwd=cwd, env=env, check=False)
        if completed.returncode == 0:
            return completed
        last_completed = completed
        if attempt < 4 and _is_retryable_compose_up_failure(completed):
            time.sleep(2 + attempt)
            continue
        raise RuntimeError(_format_command_failure(resolved_cmd, completed, cwd=cwd))
    assert last_completed is not None
    raise RuntimeError(_format_command_failure(resolved_cmd, last_completed, cwd=cwd))


def docker_compose_pull_and_up(
    compose_file: str | Path,
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    docker_compose_pull_latest(compose_file, cwd=cwd, env=env)
    docker_compose_up_force_recreate(compose_file, cwd=cwd, env=env)


def inspect_llama_cpp_image_digest(image_ref: str) -> str:
    normalized = normalize_llama_cpp_image(image_ref)
    completed = run_docker_command(
        ["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", normalized],
        check=False,
    )
    if completed.returncode != 0:
        return ""
    raw = (completed.stdout or "").strip()
    if not raw:
        return ""
    try:
        digests = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    if not isinstance(digests, list):
        return ""
    for digest in digests:
        if isinstance(digest, str) and digest.startswith("ghcr.io/ggml-org/llama.cpp@"):
            return digest
    for digest in digests:
        if isinstance(digest, str):
            return digest
    return ""


def _extract_llama_cpp_version(output: str) -> str:
    lines = [line.strip() for line in (output or "").splitlines() if line.strip()]
    for line in lines:
        if line.lower().startswith("version:"):
            return line
    for line in lines:
        if "version" in line.lower() and "llama" in line.lower():
            return line
    return lines[0] if lines else ""


def inspect_llama_cpp_version_from_container(container_name: str) -> str:
    completed = run_docker_command(
        ["docker", "exec", container_name, "/app/llama-server", "--version"],
        check=False,
    )
    output = ((completed.stdout or "") + (completed.stderr or "")).strip()
    return _extract_llama_cpp_version(output)


def inspect_llama_cpp_version_from_image(image_ref: str) -> str:
    normalized = normalize_llama_cpp_image(image_ref)
    completed = run_docker_command(
        ["docker", "run", "--rm", "--entrypoint", "/app/llama-server", normalized, "--version"],
        check=False,
    )
    output = ((completed.stdout or "") + (completed.stderr or "")).strip()
    return _extract_llama_cpp_version(output)


def inspect_llama_cpp_runtime(
    *,
    image_ref: str | None = None,
    container_name: str | None = None,
) -> dict[str, str]:
    image = normalize_llama_cpp_image(image_ref)
    if container_name:
        completed = run_docker_command(
            ["docker", "inspect", "--format", "{{.Config.Image}}", container_name],
            check=False,
        )
        runtime_image = (completed.stdout or "").strip()
        if runtime_image:
            image = runtime_image
    return {
        "llama_cpp_image": image,
        "llama_cpp_digest": inspect_llama_cpp_image_digest(image),
        "llama_cpp_version": (
            inspect_llama_cpp_version_from_container(container_name)
            if container_name
            else inspect_llama_cpp_version_from_image(image)
        ),
    }
