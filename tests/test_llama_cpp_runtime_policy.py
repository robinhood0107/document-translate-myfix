from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from modules.utils.llama_cpp_runtime import (
    DEFAULT_LLAMA_CPP_IMAGE,
    normalize_llama_cpp_image,
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

    def test_mutable_llama_cpp_tags_still_normalize_to_repository_default(self) -> None:
        self.assertEqual(
            normalize_llama_cpp_image("ghcr.io/ggml-org/llama.cpp:server-cuda"),
            DEFAULT_LLAMA_CPP_IMAGE,
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


if __name__ == "__main__":
    unittest.main()
