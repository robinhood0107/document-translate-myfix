from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import unittest
import uuid
from pathlib import Path

from modules.translation.gemma_runtime_contract import (
    DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
    DEFAULT_GEMMA_READY_MANIFEST,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PREPARE_SCRIPT = _REPO_ROOT / "scripts" / "prepare_gemma_runtime.ps1"
_RUN_DOCKER_INTEGRATION = (
    os.environ.get("CT_RUN_GEMMA_VOLUME_FAULT_TESTS", "").strip() == "1"
)
_RESET_READY_SHELL = (
    'set -eu; rm -f "/models/$READY_MANIFEST" '
    '"/models/.${READY_MANIFEST}.partial"'
)
_COPY_MARKER = 'partial="/models/.${TARGET_FILE}.partial"'


def _prepare_script_text() -> str:
    return _PREPARE_SCRIPT.read_text(encoding="utf-8")


def _extract_model_copy_shell() -> str:
    script = _prepare_script_text()
    marker_index = script.index(_COPY_MARKER)
    start = script.rfind("@'\n", 0, marker_index)
    end = script.find("\n'@", marker_index)
    if start < 0 or end < 0:
        raise AssertionError("Unable to extract the Gemma model copy shell payload.")
    return script[start + 3 : end]


class GemmaPrepareScriptStructureTests(unittest.TestCase):
    def test_ready_state_and_atomic_copy_are_ordered_fail_closed(self) -> None:
        script = _prepare_script_text()
        copy_shell = _extract_model_copy_shell()

        reset_index = script.index(_RESET_READY_SHELL)
        copy_index = script.index(_COPY_MARKER)
        smoke_index = script.index("$SmokeResult = $null")
        manifest_index = script.index("$Manifest = [ordered]@{")
        publish_index = script.index(
            'partial="/models/.${READY_MANIFEST}.partial"'
        )

        self.assertLess(reset_index, copy_index)
        self.assertLess(copy_index, smoke_index)
        self.assertLess(smoke_index, manifest_index)
        self.assertLess(manifest_index, publish_index)
        self.assertLess(
            copy_shell.index('actual_sha256="$(sha256sum "$partial"'),
            copy_shell.index('mv -f "$partial" "$target"'),
        )


@unittest.skipUnless(
    _RUN_DOCKER_INTEGRATION,
    "set CT_RUN_GEMMA_VOLUME_FAULT_TESTS=1 for isolated Docker fault tests",
)
class GemmaPrepareDockerFaultTests(unittest.TestCase):
    docker: str

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.docker = shutil.which("docker.exe") or shutil.which("docker") or ""
        if not cls.docker:
            raise AssertionError("Docker is required for Gemma volume fault tests.")
        image = cls._docker(
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
            check=False,
        )
        if image.returncode != 0 or not image.stdout.strip():
            raise AssertionError(
                "The pinned llama.cpp image must already exist; this test never pulls it."
            )

    def setUp(self) -> None:
        suffix = uuid.uuid4().hex
        self.import_volume = f"ct-gemma-prepare-import-{suffix}"
        self.model_volume = f"ct-gemma-prepare-models-{suffix}"
        self._created_volumes: list[str] = []
        for volume in (self.import_volume, self.model_volume):
            self._docker(
                "volume",
                "create",
                "--label",
                "comic-translate.test=gemma-prepare-fault",
                volume,
            )
            self._created_volumes.append(volume)

    def tearDown(self) -> None:
        for volume in reversed(self._created_volumes):
            self._docker("volume", "rm", "-f", volume, check=False)

    @classmethod
    def _docker(
        cls,
        *arguments: str,
        input_bytes: bytes | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        result = subprocess.run(
            [cls.docker, *arguments],
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if check and result.returncode != 0:
            output = (result.stdout + result.stderr).decode(
                "utf-8", errors="replace"
            )
            raise AssertionError(
                f"Docker command failed ({result.returncode}): "
                f"{' '.join(arguments)}\n{output}"
            )
        return result

    def _write_volume_file(
        self,
        volume: str,
        mount_point: str,
        file_name: str,
        content: bytes,
    ) -> None:
        self._docker(
            "run",
            "--rm",
            "--pull",
            "never",
            "-i",
            "-e",
            f"TEST_FILE={file_name}",
            "--mount",
            f"type=volume,source={volume},target={mount_point}",
            "--entrypoint",
            "/bin/sh",
            DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
            "-ec",
            f'set -eu; cat > "{mount_point}/$TEST_FILE"',
            input_bytes=content,
        )

    def _read_volume_file(
        self,
        volume: str,
        mount_point: str,
        file_name: str,
    ) -> bytes:
        result = self._docker(
            "run",
            "--rm",
            "--pull",
            "never",
            "-e",
            f"TEST_FILE={file_name}",
            "--mount",
            f"type=volume,source={volume},target={mount_point},readonly",
            "--entrypoint",
            "/bin/sh",
            DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
            "-ec",
            f'set -eu; cat "{mount_point}/$TEST_FILE"',
        )
        return result.stdout

    def _volume_file_exists(
        self,
        volume: str,
        mount_point: str,
        file_name: str,
    ) -> bool:
        result = self._docker(
            "run",
            "--rm",
            "--pull",
            "never",
            "-e",
            f"TEST_FILE={file_name}",
            "--mount",
            f"type=volume,source={volume},target={mount_point},readonly",
            "--entrypoint",
            "/bin/sh",
            DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
            "-ec",
            f'test -e "{mount_point}/$TEST_FILE"',
            check=False,
        )
        return result.returncode == 0

    def _clear_ready_state(self) -> None:
        self._docker(
            "run",
            "--rm",
            "--pull",
            "never",
            "-e",
            f"READY_MANIFEST={DEFAULT_GEMMA_READY_MANIFEST}",
            "--mount",
            f"type=volume,source={self.model_volume},target=/models",
            "--entrypoint",
            "/bin/sh",
            DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
            "-ec",
            _RESET_READY_SHELL,
        )

    def _run_model_copy(
        self,
        *,
        source_file: str,
        target_file: str,
        expected_bytes: int,
        expected_sha256: str,
        check: bool,
    ) -> subprocess.CompletedProcess[bytes]:
        return self._docker(
            "run",
            "--rm",
            "--pull",
            "never",
            "-e",
            f"SOURCE_FILE={source_file}",
            "-e",
            f"TARGET_FILE={target_file}",
            "-e",
            f"EXPECTED_BYTES={expected_bytes}",
            "-e",
            f"EXPECTED_SHA256={expected_sha256}",
            "--mount",
            f"type=volume,source={self.import_volume},target=/import,readonly",
            "--mount",
            f"type=volume,source={self.model_volume},target=/models",
            "--entrypoint",
            "/bin/sh",
            DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
            "-ec",
            _extract_model_copy_shell(),
            check=check,
        )

    def test_interrupted_partial_is_replaced_without_publishing_ready(self) -> None:
        source_name = "fault-model.gguf"
        partial_name = f".{source_name}.partial"
        expected = (b"expected-gemma-model\n" * 257) + b"complete"
        self._write_volume_file(
            self.import_volume, "/import", source_name, expected
        )
        self._write_volume_file(
            self.model_volume, "/models", partial_name, b"interrupted"
        )
        self._write_volume_file(
            self.model_volume,
            "/models",
            DEFAULT_GEMMA_READY_MANIFEST,
            b'{"ready":true}',
        )

        self._clear_ready_state()
        self._run_model_copy(
            source_file=source_name,
            target_file=source_name,
            expected_bytes=len(expected),
            expected_sha256=hashlib.sha256(expected).hexdigest(),
            check=True,
        )

        self.assertEqual(
            self._read_volume_file(self.model_volume, "/models", source_name),
            expected,
        )
        self.assertFalse(
            self._volume_file_exists(
                self.model_volume, "/models", partial_name
            )
        )
        self.assertFalse(
            self._volume_file_exists(
                self.model_volume,
                "/models",
                DEFAULT_GEMMA_READY_MANIFEST,
            )
        )

    def test_same_size_wrong_hash_cannot_replace_target_or_ready_state(self) -> None:
        source_name = "fault-model.gguf"
        partial_name = f".{source_name}.partial"
        expected = b"A" * 8192
        corrupted = b"B" * len(expected)
        previous_target = b"previous-verified-model"
        self._write_volume_file(
            self.import_volume, "/import", source_name, corrupted
        )
        self._write_volume_file(
            self.model_volume, "/models", source_name, previous_target
        )
        self._write_volume_file(
            self.model_volume,
            "/models",
            DEFAULT_GEMMA_READY_MANIFEST,
            b'{"ready":true}',
        )

        self._clear_ready_state()
        result = self._run_model_copy(
            source_file=source_name,
            target_file=source_name,
            expected_bytes=len(expected),
            expected_sha256=hashlib.sha256(expected).hexdigest(),
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(
            self._read_volume_file(self.model_volume, "/models", source_name),
            previous_target,
        )
        self.assertEqual(
            self._read_volume_file(
                self.model_volume, "/models", partial_name
            ),
            corrupted,
        )
        self.assertFalse(
            self._volume_file_exists(
                self.model_volume,
                "/models",
                DEFAULT_GEMMA_READY_MANIFEST,
            )
        )


if __name__ == "__main__":
    unittest.main()
