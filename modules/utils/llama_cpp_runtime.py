from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from modules.utils.exceptions import OperationCancelledError


# Windows setup은 CUDA12/CUDA13 Python 경로 모두 같은 검증된 CUDA 이미지만
# 사용한다. 다른 llama.cpp 태그는 설치 상태나 관리형 런타임 계약으로 인정하지 않는다.
SUPPORTED_LLAMA_CPP_IMAGES: tuple[str, ...] = (
    "ghcr.io/ggml-org/llama.cpp:server-cuda",
)
_requested_default_image = os.environ.get("LLAMA_CPP_IMAGE", "").strip()
DEFAULT_LLAMA_CPP_IMAGE = (
    _requested_default_image
    if _requested_default_image in SUPPORTED_LLAMA_CPP_IMAGES
    else "ghcr.io/ggml-org/llama.cpp:server-cuda"
)
DEFAULT_LLAMA_CPP_PULL_POLICY = "always"
DEFAULT_MANAGED_RUNTIME_STOP_TIMEOUT_SEC = 10
DEFAULT_DOCKER_COMMAND_TIMEOUT_SEC = 600.0
DOCKER_COMMAND_POLL_INTERVAL_SEC = 0.1


def is_supported_llama_cpp_image(image_ref: Any = None) -> bool:
    """Report whether a reference is one of the supported CUDA server tags."""

    return str(image_ref or "").strip() in SUPPORTED_LLAMA_CPP_IMAGES


def normalize_llama_cpp_image(image_ref: Any = None) -> str:
    text = str(image_ref or "").strip()
    if not text:
        return DEFAULT_LLAMA_CPP_IMAGE
    if "@sha256:" in text:
        return text
    if is_supported_llama_cpp_image(text):
        return text
    if "ggml-org/llama.cpp" in text or "local/llama.cpp" in text:
        return DEFAULT_LLAMA_CPP_IMAGE
    return text


def normalize_llama_cpp_pull_policy(_: Any = None) -> str:
    return DEFAULT_LLAMA_CPP_PULL_POLICY


def _find_executable_on_path(*names: str) -> str | None:
    """Find an executable even when Windows ``PATHEXT`` is malformed.

    ``shutil.which`` follows ``PATHEXT`` on Windows.  A launcher inherited from
    a constrained shell can expose only an unrelated extension, making an
    existing ``docker.exe`` invisible.  The explicit PATH scan keeps Docker
    discovery fail-closed while avoiding reliance on that ambient setting.
    """

    for name in names:
        resolved = shutil.which(name)
        if resolved:
            return resolved
    directories = tuple(
        raw_directory.strip().strip('"')
        for raw_directory in os.environ.get("PATH", "").split(os.pathsep)
    )
    for name in names:
        for directory in directories:
            if not directory:
                continue
            candidate = Path(directory) / name
            if candidate.is_file():
                return str(candidate)
    return None


def resolve_docker_executable() -> str:
    return _find_executable_on_path("docker.exe", "docker") or "docker"


def remove_named_container(name: str) -> None:
    """이름이 같은 잔여 컨테이너를 조용히 제거한다.

    프로브 컨테이너는 ``--rm``으로 뜨지만, 앱이 강제 종료되면 이름이 남아 다음
    실행이 "name already in use"로 실패한다. 고정 이름을 쓰는 이점을 잃지 않도록
    실행 전에 항상 정리한다. 없으면 아무 일도 일어나지 않는다.
    """

    normalized = str(name or "").strip()
    if not normalized:
        return
    try:
        run_docker_command(
            ["docker", "rm", "-f", normalized],
            check=False,
            timeout_sec=30.0,
        )
    except Exception:
        # 정리 실패는 진단 정보일 뿐이다. 실제 문제라면 이어지는 run 이 드러낸다.
        logger.debug("Could not remove a leftover probe container: %s", normalized)


def run_docker_command(
    cmd: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    timeout_sec: float = DEFAULT_DOCKER_COMMAND_TIMEOUT_SEC,
    cancel_checker: Callable[[], bool] | None = None,
) -> subprocess.CompletedProcess[str]:
    resolved_cmd = list(cmd)
    if resolved_cmd and resolved_cmd[0] == "docker":
        resolved_cmd[0] = resolve_docker_executable()
    if cancel_checker is not None and cancel_checker():
        raise OperationCancelledError("Cancelled before Docker command startup.")
    process = subprocess.Popen(
        resolved_cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **_process_group_popen_kwargs(),
    )
    deadline = time.monotonic() + max(0.1, float(timeout_sec))
    stdout = ""
    stderr = ""
    while True:
        if cancel_checker is not None and cancel_checker():
            _terminate_process(process)
            raise OperationCancelledError(
                f"Cancelled Docker command: {' '.join(resolved_cmd)}"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            _terminate_process(process)
            raise RuntimeError(
                f"Docker command timed out after {float(timeout_sec):.1f}s: "
                f"{' '.join(resolved_cmd)}"
            )
        try:
            stdout, stderr = process.communicate(
                timeout=min(DOCKER_COMMAND_POLL_INTERVAL_SEC, remaining)
            )
            break
        except subprocess.TimeoutExpired:
            continue

    completed = subprocess.CompletedProcess(
        resolved_cmd,
        int(process.returncode or 0),
        stdout or "",
        stderr or "",
    )
    if check and completed.returncode != 0:
        detail = (
            f"Command failed (exit={completed.returncode}): {' '.join(resolved_cmd)}\n"
            f"cwd={cwd}\n"
            f"stdout:\n{(completed.stdout or '').strip()}\n"
            f"stderr:\n{(completed.stderr or '').strip()}"
        )
        raise RuntimeError(detail)
    return completed


def _process_group_popen_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {
            "creationflags": getattr(
                subprocess,
                "CREATE_NEW_PROCESS_GROUP",
                0x00000200,
            )
        }
    return {"start_new_session": True}


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        _terminate_windows_process_tree(process)
    else:
        _terminate_posix_process_group(process)


def _terminate_windows_process_tree(process: subprocess.Popen[str]) -> None:
    pid = getattr(process, "pid", None)
    if pid is not None:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5.0,
            )
        except Exception:
            pass
    _wait_then_force_kill(process)


def _terminate_posix_process_group(process: subprocess.Popen[str]) -> None:
    pid = getattr(process, "pid", None)
    if pid is not None:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:
            pass
    try:
        process.communicate(timeout=2.0)
        return
    except Exception:
        pass
    if pid is not None:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            pass
    _wait_then_force_kill(process)


def _wait_then_force_kill(process: subprocess.Popen[str]) -> None:
    try:
        process.communicate(timeout=2.0)
        return
    except Exception:
        pass
    if process.poll() is not None:
        return
    try:
        process.kill()
        process.communicate(timeout=2.0)
    except Exception:
        pass


def resolve_docker_compose_command(
    *,
    cancel_checker: Callable[[], bool] | None = None,
) -> tuple[str, ...]:
    candidates: list[tuple[str, ...]] = []
    docker = _find_executable_on_path("docker.exe", "docker")
    if docker:
        candidates.append((docker, "compose"))
    docker_compose = _find_executable_on_path("docker-compose.exe", "docker-compose")
    if docker_compose:
        candidates.append((docker_compose,))

    for candidate in candidates:
        probe = run_docker_command(
            [*candidate, "version"],
            check=False,
            cancel_checker=cancel_checker,
        )
        if probe.returncode == 0:
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


def docker_compose_up_force_recreate(
    compose_file: str | Path,
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return run_docker_command(
        [*resolve_docker_compose_command(), "-f", str(compose_file), "up", "-d", "--force-recreate"],
        cwd=cwd,
        env=env,
    )


def docker_compose_pull_and_up(
    compose_file: str | Path,
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    docker_compose_pull_latest(compose_file, cwd=cwd, env=env)
    docker_compose_up_force_recreate(compose_file, cwd=cwd, env=env)


def inspect_llama_cpp_image_digest(
    image_ref: str,
    *,
    cancel_checker: Callable[[], bool] | None = None,
) -> str:
    normalized = normalize_llama_cpp_image(image_ref)
    completed = run_docker_command(
        ["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", normalized],
        check=False,
        cancel_checker=cancel_checker,
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


def inspect_llama_cpp_version_from_container(
    container_name: str,
    *,
    cancel_checker: Callable[[], bool] | None = None,
) -> str:
    completed = run_docker_command(
        ["docker", "exec", container_name, "/app/llama-server", "--version"],
        check=False,
        cancel_checker=cancel_checker,
    )
    output = ((completed.stdout or "") + (completed.stderr or "")).strip()
    return _extract_llama_cpp_version(output)


def inspect_llama_cpp_version_from_image(
    image_ref: str,
    *,
    cancel_checker: Callable[[], bool] | None = None,
) -> str:
    normalized = normalize_llama_cpp_image(image_ref)
    # 이름을 주지 않으면 Docker 가 임의 이름을 붙여, 사용자에게는 정체를 알 수 없는
    # 컨테이너가 떴다 사라지는 것으로 보인다. 제품이 만드는 컨테이너는 모두 이름이
    # 정해져 있어야 한다.
    probe_name = "comic-translate-llamacpp-version-probe"
    remove_named_container(probe_name)
    completed = run_docker_command(
        [
            "docker",
            "run",
            "--name",
            probe_name,
            "--rm",
            "--entrypoint",
            "/app/llama-server",
            normalized,
            "--version",
        ],
        check=False,
        cancel_checker=cancel_checker,
    )
    output = ((completed.stdout or "") + (completed.stderr or "")).strip()
    return _extract_llama_cpp_version(output)


def inspect_llama_cpp_runtime(
    *,
    image_ref: str | None = None,
    container_name: str | None = None,
    cancel_checker: Callable[[], bool] | None = None,
) -> dict[str, str]:
    image = normalize_llama_cpp_image(image_ref)
    if container_name:
        completed = run_docker_command(
            ["docker", "inspect", "--format", "{{.Config.Image}}", container_name],
            check=False,
            cancel_checker=cancel_checker,
        )
        runtime_image = (completed.stdout or "").strip()
        if runtime_image:
            image = runtime_image
    return {
        "llama_cpp_image": image,
        "llama_cpp_digest": inspect_llama_cpp_image_digest(
            image,
            cancel_checker=cancel_checker,
        ),
        "llama_cpp_version": (
            inspect_llama_cpp_version_from_container(
                container_name,
                cancel_checker=cancel_checker,
            )
            if container_name
            else inspect_llama_cpp_version_from_image(
                image,
                cancel_checker=cancel_checker,
            )
        ),
    }
