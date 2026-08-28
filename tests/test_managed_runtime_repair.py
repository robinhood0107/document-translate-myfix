from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from typing import Any

from modules.translation.gemma_runtime_contract import (
    DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
    DEFAULT_GEMMA_MODEL_VOLUME,
    GEMMA_MODEL_SPECS,
    GemmaRuntimeContractError,
    build_gemma_runtime_contract,
)
from modules.utils import managed_runtime_repair
from modules.utils.managed_runtime_repair import (
    ManagedRuntimeRepairError,
    ManagedRuntimeRepairPlan,
    is_image_identity_only_drift,
    manifest_recorded_image_identity,
    run_managed_runtime_preparation,
)


ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = ROOT / "docker-compose.yaml"
PREPARE_SCRIPTS = (
    "prepare_gemma_runtime.ps1",
    "prepare_hunyuanocr_llamacpp_runtime.ps1",
    "prepare_mangalmm_llamacpp_runtime.ps1",
    "prepare_paddleocr_llamacpp_runtime.ps1",
    "prepare_paddleocr_spotting_llamacpp_runtime.ps1",
)

SEALED_IMAGE_ID = "sha256:" + "a" * 64
CURRENT_IMAGE_ID = "sha256:" + "b" * 64
MODEL_NAME = "gemma-4-26B-IQ4_NL.gguf"
MODEL_SPEC = GEMMA_MODEL_SPECS[MODEL_NAME]


def _gemma_manifest(**overrides: Any) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "runtime": "Gemma",
        "preparation_version": 2,
        "volume_name": DEFAULT_GEMMA_MODEL_VOLUME,
        "ready": True,
        "source_image_ref": DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
        "source_image_digest": SEALED_IMAGE_ID,
        "source_image_id": SEALED_IMAGE_ID,
        "default_model": MODEL_NAME,
        "runtime_configuration": {
            "context_size": 4096,
            "parallel": 1,
            "threads": 10,
            "gpu_layers": 23,
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "cache_ram_mib": 0,
            "speculative_type": "none",
            "speculative_draft_max": 8,
        },
        "files": [
            {
                "name": MODEL_NAME,
                "bytes": MODEL_SPEC["bytes"],
                "sha256": MODEL_SPEC["sha256"],
                "role": MODEL_SPEC["role"],
            }
        ],
        "smoke_test": {
            "passed": True,
            "model": MODEL_NAME,
            "health_status": "ok",
            "models_match": True,
            "chat_response_nonempty": True,
        },
    }
    manifest.update(overrides)
    return manifest


def _encode(manifest: dict[str, Any]) -> bytes:
    return json.dumps(manifest, ensure_ascii=False, indent=4).encode("utf-8")


def _gemma_revalidate(manifest_bytes: bytes, *, observed_bytes: int | None = None):
    import hashlib

    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    def revalidate(image_ref: str, image_id: str):
        return build_gemma_runtime_contract(
            manifest_bytes=manifest_bytes,
            manifest_sha256=manifest_sha256,
            observed_model_bytes=(
                MODEL_SPEC["bytes"] if observed_bytes is None else observed_bytes
            ),
            volume_name=DEFAULT_GEMMA_MODEL_VOLUME,
            model_name=MODEL_NAME,
            image_ref=image_ref,
            image_id=image_id,
            compose_file=COMPOSE_FILE,
            environment={},
        )

    return revalidate


class ImageIdentityDriftTests(unittest.TestCase):
    def test_drift_is_detected_when_only_the_image_identity_moved(self) -> None:
        manifest_bytes = _encode(_gemma_manifest())
        revalidate = _gemma_revalidate(manifest_bytes)

        with self.assertRaises(GemmaRuntimeContractError):
            revalidate(DEFAULT_GEMMA_LLAMA_CPP_IMAGE, CURRENT_IMAGE_ID)

        self.assertTrue(
            is_image_identity_only_drift(
                manifest_bytes,
                current_image_id=CURRENT_IMAGE_ID,
                revalidate=revalidate,
            )
        )

    def test_a_matching_image_identity_is_not_drift(self) -> None:
        manifest_bytes = _encode(_gemma_manifest())
        self.assertFalse(
            is_image_identity_only_drift(
                manifest_bytes,
                current_image_id=SEALED_IMAGE_ID,
                revalidate=_gemma_revalidate(manifest_bytes),
            )
        )

    def test_a_model_size_mismatch_is_never_treated_as_drift(self) -> None:
        # 볼륨을 신뢰할 수 없는 상태를 image drift 로 오인해 다시 봉인하면,
        # 잘못된 모델에 유효 도장을 찍게 된다.
        manifest_bytes = _encode(_gemma_manifest())
        self.assertFalse(
            is_image_identity_only_drift(
                manifest_bytes,
                current_image_id=CURRENT_IMAGE_ID,
                revalidate=_gemma_revalidate(manifest_bytes, observed_bytes=123),
            )
        )

    def test_a_broken_registry_is_never_treated_as_drift(self) -> None:
        manifest_bytes = _encode(
            _gemma_manifest(
                files=[
                    {
                        "name": MODEL_NAME,
                        "bytes": MODEL_SPEC["bytes"],
                        "sha256": "0" * 64,
                        "role": MODEL_SPEC["role"],
                    }
                ]
            )
        )
        self.assertFalse(
            is_image_identity_only_drift(
                manifest_bytes,
                current_image_id=CURRENT_IMAGE_ID,
                revalidate=_gemma_revalidate(manifest_bytes),
            )
        )

    def test_unreadable_manifests_report_no_identity(self) -> None:
        self.assertIsNone(manifest_recorded_image_identity(b"not json"))
        self.assertIsNone(manifest_recorded_image_identity(b"[]"))
        self.assertFalse(
            is_image_identity_only_drift(
                b"not json",
                current_image_id=CURRENT_IMAGE_ID,
                revalidate=lambda _ref, _id: None,
            )
        )


class PreparationRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def _fake_runner(self, stdout: str = "{}", returncode: int = 0):
        def runner(cmd, **kwargs):
            self.calls.append({"cmd": list(cmd), **kwargs})
            return subprocess.CompletedProcess(list(cmd), returncode, stdout, "")

        return runner

    def _plan(self) -> ManagedRuntimeRepairPlan:
        return ManagedRuntimeRepairPlan(
            runtime_label="Gemma",
            prepare_script=ROOT / "scripts" / "prepare_gemma_runtime.ps1",
            volume_name=DEFAULT_GEMMA_MODEL_VOLUME,
        )

    def test_the_runner_invokes_the_prepare_script_with_the_requested_mode(self) -> None:
        original = managed_runtime_repair.run_docker_command
        managed_runtime_repair.run_docker_command = self._fake_runner(
            "Auto mode: resealing.\n" + json.dumps({"mode": "Reseal", "resealed": True})
        )
        try:
            result = run_managed_runtime_preparation(
                self._plan(),
                mode="Auto",
                allow_download=True,
            )
        finally:
            managed_runtime_repair.run_docker_command = original

        self.assertEqual(result, {"mode": "Reseal", "resealed": True})
        self.assertEqual(len(self.calls), 1)
        cmd = self.calls[0]["cmd"]
        self.assertIn("-Mode", cmd)
        self.assertEqual(cmd[cmd.index("-Mode") + 1], "Auto")
        self.assertIn("-VolumeName", cmd)
        self.assertEqual(
            cmd[cmd.index("-VolumeName") + 1],
            DEFAULT_GEMMA_MODEL_VOLUME,
        )
        self.assertIn("-AllowDownload", cmd)
        self.assertIn("-DownloadDirectory", cmd)
        cache_path = cmd[cmd.index("-DownloadDirectory") + 1]
        self.assertTrue(
            cache_path.endswith("models\\managed-runtime-sources")
            or cache_path.endswith("models/managed-runtime-sources")
        )
        self.assertIn("-NonInteractive", cmd)
        self.assertTrue(cmd[cmd.index("-File") + 1].endswith("prepare_gemma_runtime.ps1"))

    def test_downloads_stay_off_unless_requested(self) -> None:
        original = managed_runtime_repair.run_docker_command
        managed_runtime_repair.run_docker_command = self._fake_runner()
        try:
            run_managed_runtime_preparation(self._plan(), mode="Reseal")
        finally:
            managed_runtime_repair.run_docker_command = original
        self.assertNotIn("-AllowDownload", self.calls[0]["cmd"])

    def test_a_failing_script_raises_a_repair_error(self) -> None:
        original = managed_runtime_repair.run_docker_command
        managed_runtime_repair.run_docker_command = self._fake_runner(
            "boom", returncode=1
        )
        try:
            with self.assertRaises(ManagedRuntimeRepairError):
                run_managed_runtime_preparation(self._plan(), mode="Reseal")
        finally:
            managed_runtime_repair.run_docker_command = original

    def test_a_missing_script_raises_before_any_process_starts(self) -> None:
        plan = ManagedRuntimeRepairPlan(
            runtime_label="Gemma",
            prepare_script=ROOT / "scripts" / "does-not-exist.ps1",
            volume_name=DEFAULT_GEMMA_MODEL_VOLUME,
        )
        original = managed_runtime_repair.run_docker_command
        managed_runtime_repair.run_docker_command = self._fake_runner()
        try:
            with self.assertRaises(ManagedRuntimeRepairError):
                run_managed_runtime_preparation(plan)
        finally:
            managed_runtime_repair.run_docker_command = original
        self.assertEqual(self.calls, [])


class GemmaReadOnlyRuntimeTests(unittest.TestCase):
    """The application rejects drift; setup remains the only repair owner."""

    def _manager(self, manifest_bytes: bytes, *, image_id: str):
        import hashlib

        from modules.translation.local_runtime import (
            LocalGemmaRuntimeManager,
            _GemmaVolumeNotProvisioned,
        )

        manager = LocalGemmaRuntimeManager()
        manager._ensure_runtime_image_id = lambda _ref: image_id  # type: ignore[method-assign]
        manager._probe_model_volume = lambda **_kwargs: (  # type: ignore[method-assign]
            manifest_bytes,
            hashlib.sha256(manifest_bytes).hexdigest(),
            MODEL_SPEC["bytes"],
        )
        return manager, _GemmaVolumeNotProvisioned

    def test_image_drift_is_rejected_without_a_repair_hook(self) -> None:
        from modules.utils.exceptions import LocalServiceSetupError

        drifted = _encode(_gemma_manifest())
        manager, _marker = self._manager(drifted, image_id=CURRENT_IMAGE_ID)

        with self.assertRaises(LocalServiceSetupError):
            manager._load_runtime_contract(MODEL_NAME)

        self.assertFalse(hasattr(manager, "_repair_runtime_volume"))

    def test_missing_volume_is_rejected_without_provisioning(self) -> None:
        from modules.utils.exceptions import LocalServiceSetupError

        manifest = _encode(
            _gemma_manifest(
                source_image_digest=CURRENT_IMAGE_ID,
                source_image_id=CURRENT_IMAGE_ID,
            )
        )
        manager, marker = self._manager(manifest, image_id=CURRENT_IMAGE_ID)

        def missing(**_kwargs):
            raise marker("Prepared Gemma model volume does not exist: x")

        manager._probe_model_volume = missing  # type: ignore[method-assign]

        with self.assertRaisesRegex(LocalServiceSetupError, "matching setup BAT"):
            manager._load_runtime_contract(MODEL_NAME)


class PrepareScriptContractTests(unittest.TestCase):
    def test_docker_volume_helpers_have_one_owner(self) -> None:
        module = (
            ROOT / "scripts" / "lib" / "ManagedRuntimeDocker.psm1"
        ).read_text(encoding="utf-8")
        helpers = (
            "Invoke-DockerResult",
            "Invoke-Docker",
            "Get-PinnedImageId",
            "Assert-ManagedContainerStopped",
            "Assert-VolumeLabels",
            "Get-VolumeFileHash",
            "Read-ReadyManifest",
            "Get-VolumeFileSize",
            "Test-VolumeHoldsEveryModel",
        )
        for helper in helpers:
            self.assertIn(f"function {helper}", module)
        for name in PREPARE_SCRIPTS:
            with self.subTest(script=name):
                script = (ROOT / "scripts" / name).read_text(encoding="utf-8")
                self.assertIn("Initialize-ManagedRuntimeDocker", script)
                for helper in helpers:
                    self.assertNotIn(f"function {helper}", script)

    def test_every_prepare_script_offers_reseal_and_auto(self) -> None:
        for name in PREPARE_SCRIPTS:
            with self.subTest(script=name):
                script = (ROOT / "scripts" / name).read_text(encoding="utf-8")
                self.assertIn(
                    "[ValidateSet('Prepare', 'Verify', 'Reseal', 'Auto')]",
                    script,
                )
                self.assertIn("$IsReseal = $Mode -eq 'Reseal'", script)
                self.assertIn("if ($Mode -eq 'Auto') {", script)
                self.assertIn("mode = 'Reuse'", script)
                self.assertIn("without a full hash or GPU smoke", script)
                self.assertLess(
                    script.index("mode = 'Reuse'"),
                    script.index("$IsReseal = $Mode -eq 'Reseal'"),
                )

    def test_every_prepare_script_normalizes_crlf_for_container_shells(self) -> None:
        # 컨테이너의 /bin/sh 는 dash 다. here-string 의 CR 이 그대로 넘어가면
        # 첫 줄 `set -eu` 부터 "Illegal option -" 로 죽는다.
        module = (
            ROOT / "scripts" / "lib" / "ManagedRuntimeDocker.psm1"
        ).read_text(encoding="utf-8")
        self.assertIn("$NormalizedArguments", module)
        self.assertIn('-replace "`r`n", "`n"', module)

    def test_smoke_failures_preserve_logs_until_explicit_cleanup(self) -> None:
        module = (
            ROOT / "scripts" / "lib" / "ManagedRuntimeDocker.psm1"
        ).read_text(encoding="utf-8")
        self.assertIn("function Test-ManagedRuntimeContainerRunning", module)
        self.assertIn("function Remove-ManagedRuntimeContainer", module)
        for name in PREPARE_SCRIPTS:
            with self.subTest(script=name):
                script = (ROOT / "scripts" / name).read_text(encoding="utf-8")
                self.assertNotIn("'run', '-d', '--rm'", script)
                self.assertIn("Test-ManagedRuntimeContainerRunning", script)
                self.assertIn("Remove-ManagedRuntimeContainer", script)

    def test_every_prepare_script_accepts_only_the_managed_cuda_image(self) -> None:
        module = (
            ROOT / "scripts" / "lib" / "ManagedRuntimeDocker.psm1"
        ).read_text(encoding="utf-8")
        self.assertIn("ghcr.io/ggml-org/llama.cpp:server-cuda'", module)
        self.assertIn("Supported = @($ImageRef)", module)
        for name in PREPARE_SCRIPTS:
            with self.subTest(script=name):
                script = (ROOT / "scripts" / name).read_text(encoding="utf-8")
                self.assertIn("[string]$ImageRef", script)
                self.assertIn("[int64]$MinimumFreeBytes = 0", script)
                self.assertIn("Get-ManagedLlamaCppImagePolicy", script)

    def test_prepare_does_not_mount_a_missing_volume_before_creation(self) -> None:
        for name in PREPARE_SCRIPTS[:4]:
            with self.subTest(script=name):
                script = (ROOT / "scripts" / name).read_text(encoding="utf-8")
                self.assertIn("$VolumeExistsBeforePrepare -and", script)

    def test_shared_model_download_retries_without_discarding_partial_data(self) -> None:
        source = (
            ROOT / "scripts" / "lib" / "ManagedRuntimeModelSource.psm1"
        ).read_text(encoding="utf-8")
        self.assertIn("function Invoke-ManagedRuntimeDownloadAttempt", source)
        self.assertIn("[int]$MaximumAttempts = 5", source)
        self.assertIn("Resuming in", source)
        self.assertIn("$Handler.AllowAutoRedirect = $false", source)
        self.assertIn("for ($Redirect = 0; $Redirect -le 10; $Redirect++)", source)
        self.assertIn("$Request.Headers.Range", source)
        self.assertIn("[System.Security.Cryptography.SHA256]::Create()", source)
        self.assertNotIn("Get-FileHash", source)
        self.assertIn("$RequiredBytes = $Bytes + 536870912L", source)

    def test_every_contracted_model_has_a_registered_download_source(self) -> None:
        # 원본이 없으면 자가복구가 볼륨을 채울 수 없다. 등록된 출처를 계약으로 고정해
        # 새 모델을 추가할 때 URL 을 빠뜨리지 않게 한다.
        expected = {
            "prepare_gemma_runtime.ps1": 1,
            "prepare_hunyuanocr_llamacpp_runtime.ps1": 2,
            "prepare_mangalmm_llamacpp_runtime.ps1": 2,
            "prepare_paddleocr_llamacpp_runtime.ps1": 2,
            # Spotting 은 대상 GGUF 하나와, 파생 전 원본 projector 하나를 등록한다.
            "prepare_paddleocr_spotting_llamacpp_runtime.ps1": 2,
        }
        for name, count in expected.items():
            with self.subTest(script=name):
                script = (ROOT / "scripts" / name).read_text(encoding="utf-8")
                self.assertNotIn("DownloadUrl = ''", script)
                registered = script.count("https://huggingface.co/")
                self.assertEqual(registered, count)

    def test_reseal_never_creates_a_volume(self) -> None:
        for name in PREPARE_SCRIPTS:
            with self.subTest(script=name):
                script = (ROOT / "scripts" / name).read_text(encoding="utf-8")
                create_index = script.index("'volume', 'create',")
                guard_index = script.rindex("if ($IsReseal) {", 0, create_index)
                self.assertIn("volume to reseal does not exist", script[guard_index:create_index])
