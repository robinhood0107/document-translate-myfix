from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modules.utils.llama_cpp_runtime import (
    DEFAULT_LLAMA_CPP_IMAGE,
    SUPPORTED_LLAMA_CPP_IMAGES,
    normalize_llama_cpp_image,
    resolve_docker_compose_command,
    resolve_docker_executable,
    run_docker_command,
)
from modules.utils.exceptions import OperationCancelledError


class _BlockingProcess:
    def __init__(self) -> None:
        self.pid = 4242
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def communicate(self, timeout=None):
        if self.terminated:
            self.returncode = -15
            return "", ""
        raise subprocess.TimeoutExpired(["docker", "version"], timeout)

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.terminated = True


class LlamaCppRuntimePolicyTests(unittest.TestCase):
    def test_digest_pinned_image_is_preserved(self) -> None:
        pinned = (
            "ghcr.io/ggml-org/llama.cpp@sha256:"
            "22e0e3bfe967af4fd1df6a918022abbfd4e72e4d40a4769e616a4176790acbcb"
        )
        self.assertEqual(normalize_llama_cpp_image(pinned), pinned)

    def test_repository_default_is_the_cuda13_server_tag(self) -> None:
        self.assertEqual(
            DEFAULT_LLAMA_CPP_IMAGE,
            "ghcr.io/ggml-org/llama.cpp:server-cuda13",
        )
        self.assertEqual(SUPPORTED_LLAMA_CPP_IMAGES[0], DEFAULT_LLAMA_CPP_IMAGE)

    def test_supported_cuda_tags_are_preserved(self) -> None:
        for supported in SUPPORTED_LLAMA_CPP_IMAGES:
            with self.subTest(image=supported):
                self.assertEqual(normalize_llama_cpp_image(supported), supported)

    def test_unsupported_llama_cpp_tags_normalize_to_repository_default(self) -> None:
        self.assertEqual(
            normalize_llama_cpp_image("ghcr.io/ggml-org/llama.cpp:server"),
            DEFAULT_LLAMA_CPP_IMAGE,
        )
        self.assertEqual(
            normalize_llama_cpp_image("local/llama.cpp:latest"),
            DEFAULT_LLAMA_CPP_IMAGE,
        )

    def test_windows_docker_exe_is_found_when_pathext_is_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root)
            shim_directory = root / "shim"
            docker_directory = root / "docker"
            shim_directory.mkdir()
            docker_directory.mkdir()
            (shim_directory / "docker").touch()
            docker_exe = docker_directory / "docker.exe"
            docker_exe.touch()
            with mock.patch.dict(
                "modules.utils.llama_cpp_runtime.os.environ",
                {
                    "PATH": os.pathsep.join((str(shim_directory), str(docker_directory))),
                    "PATHEXT": ".CPL",
                },
                clear=False,
            ), mock.patch(
                "modules.utils.llama_cpp_runtime.shutil.which",
                return_value=None,
            ):
                self.assertEqual(resolve_docker_executable(), str(docker_exe))

    def test_compose_probe_uses_explicitly_discovered_docker_executable(self) -> None:
        docker_exe = r"C:\Docker\docker.exe"
        completed = subprocess.CompletedProcess([docker_exe, "compose", "version"], 0, "", "")
        with mock.patch(
            "modules.utils.llama_cpp_runtime._find_executable_on_path",
            side_effect=[docker_exe, None],
        ), mock.patch(
            "modules.utils.llama_cpp_runtime.run_docker_command",
            return_value=completed,
        ) as run:
            self.assertEqual(resolve_docker_compose_command(), (docker_exe, "compose"))

        run.assert_called_once_with(
            [docker_exe, "compose", "version"],
            check=False,
            cancel_checker=None,
        )

    def test_docker_command_cancellation_terminates_blocked_client(self) -> None:
        process = _BlockingProcess()
        cancel_checks = iter([False, True])

        with mock.patch(
            "modules.utils.llama_cpp_runtime.subprocess.Popen",
            return_value=process,
        ) as popen, mock.patch(
            "modules.utils.llama_cpp_runtime.os.name",
            "nt",
        ), mock.patch(
            "modules.utils.llama_cpp_runtime.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, "", ""),
        ) as run, self.assertRaises(OperationCancelledError):
            run_docker_command(
                ["docker", "version"],
                cancel_checker=lambda: next(cancel_checks),
            )

        self.assertTrue(process.terminated)
        self.assertTrue(
            popen.call_args.kwargs["creationflags"]
            & getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        )
        run.assert_called_once_with(
            ["taskkill", "/PID", "4242", "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
        )

    def test_docker_command_initial_cancellation_does_not_spawn_client(self) -> None:
        with mock.patch(
            "modules.utils.llama_cpp_runtime.subprocess.Popen",
        ) as popen, self.assertRaises(OperationCancelledError):
            run_docker_command(
                ["docker", "version"],
                cancel_checker=lambda: True,
            )

        popen.assert_not_called()

    def test_docker_command_timeout_terminates_blocked_client(self) -> None:
        process = _BlockingProcess()

        with mock.patch(
            "modules.utils.llama_cpp_runtime.subprocess.Popen",
            return_value=process,
        ), mock.patch(
            "modules.utils.llama_cpp_runtime.os.name",
            "nt",
        ), mock.patch(
            "modules.utils.llama_cpp_runtime.time.monotonic",
            side_effect=[0.0, 1.0],
        ), mock.patch(
            "modules.utils.llama_cpp_runtime.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, "", ""),
        ) as run, self.assertRaisesRegex(
            RuntimeError,
            "Docker command timed out",
        ):
            run_docker_command(
                ["docker", "version"],
                timeout_sec=0.1,
            )

        self.assertTrue(process.terminated)
        run.assert_called_once_with(
            ["taskkill", "/PID", "4242", "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
        )

    def test_docker_command_check_false_returns_nonzero_result(self) -> None:
        process = mock.MagicMock()
        process.returncode = 7
        process.communicate.return_value = ("docker output", "docker error")

        with mock.patch(
            "modules.utils.llama_cpp_runtime.subprocess.Popen",
            return_value=process,
        ):
            completed = run_docker_command(
                ["docker", "version"],
                check=False,
            )

        self.assertEqual(completed.returncode, 7)
        self.assertEqual(completed.stdout, "docker output")
        self.assertEqual(completed.stderr, "docker error")

    def test_docker_command_check_true_reports_nonzero_result(self) -> None:
        process = mock.MagicMock()
        process.returncode = 7
        process.communicate.return_value = ("docker output", "docker error")

        with mock.patch(
            "modules.utils.llama_cpp_runtime.subprocess.Popen",
            return_value=process,
        ), self.assertRaisesRegex(
            RuntimeError,
            "exit=7",
        ) as raised:
            run_docker_command(["docker", "version"])

        detail = str(raised.exception)
        self.assertIn("docker output", detail)
        self.assertIn("docker error", detail)


if __name__ == "__main__":
    unittest.main()
