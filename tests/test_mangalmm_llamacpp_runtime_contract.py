from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from modules.ocr.mangalmm_llamacpp_runtime_contract import (
    DEFAULT_MANGALMM_LLAMA_CPP_IMAGE,
    DEFAULT_MANGALMM_MODEL_VOLUME,
    MANGALMM_MMPROJ_NAME,
    MANGALMM_MODEL_ALIAS,
    MANGALMM_MODEL_NAME,
    MANGALMM_MODEL_SPECS,
    MangaLMMRuntimeContractError,
    build_mangalmm_runtime_contract,
    resolve_mangalmm_runtime_options,
    validate_mangalmm_volume_name,
)


def _manifest_payload() -> dict:
    return {
        "schema_version": 1,
        "runtime": "MangaLMM-llama.cpp",
        "preparation_version": 2,
        "volume_name": DEFAULT_MANGALMM_MODEL_VOLUME,
        "ready": True,
        "source_image_ref": DEFAULT_MANGALMM_LLAMA_CPP_IMAGE,
        "source_image_id": DEFAULT_MANGALMM_LLAMA_CPP_IMAGE.rsplit("@", 1)[-1],
        "smoke_test": {
            "passed": True,
            "device": "CUDA",
            "model_alias": MANGALMM_MODEL_ALIAS,
        },
        "files": [
            {
                "name": name,
                "bytes": spec["bytes"],
                "sha256": spec["sha256"],
                "role": spec["role"],
            }
            for name, spec in MANGALMM_MODEL_SPECS.items()
        ],
    }


class MangaLMMRuntimeContractTests(unittest.TestCase):
    def _build_contract(
        self,
        root: Path,
        *,
        manifest: dict | None = None,
        environment: dict | None = None,
    ):
        compose_file = root / "docker-compose.yaml"
        compose_file.write_text("services: {}\n", encoding="utf-8")
        manifest_bytes = json.dumps(
            manifest or _manifest_payload(),
            sort_keys=True,
        ).encode("utf-8")
        return build_mangalmm_runtime_contract(
            manifest_bytes=manifest_bytes,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            observed_file_bytes={
                name: int(spec["bytes"])
                for name, spec in MANGALMM_MODEL_SPECS.items()
            },
            volume_name=DEFAULT_MANGALMM_MODEL_VOLUME,
            llama_image_ref=DEFAULT_MANGALMM_LLAMA_CPP_IMAGE,
            llama_image_id=DEFAULT_MANGALMM_LLAMA_CPP_IMAGE.rsplit("@", 1)[-1],
            compose_file=compose_file,
            environment=environment,
        )

    def test_contract_includes_named_volume_and_exact_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            contract = self._build_contract(Path(temp_dir))

        self.assertEqual(contract.volume_name, DEFAULT_MANGALMM_MODEL_VOLUME)
        self.assertIn(f"/models/{MANGALMM_MODEL_NAME}", contract.command)
        self.assertIn(f"/models/{MANGALMM_MMPROJ_NAME}", contract.command)
        self.assertIn("--cache-ram", contract.command)
        self.assertEqual(
            contract.compose_environment()["MANGALMM_RUNTIME_FINGERPRINT"],
            contract.fingerprint,
        )

    def test_runtime_option_changes_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline = self._build_contract(root)
            changed = self._build_contract(
                root,
                environment={"MANGALMM_LLAMA_CTX_SIZE": "12288"},
            )

        self.assertNotEqual(baseline.command_sha256, changed.command_sha256)
        self.assertNotEqual(baseline.fingerprint, changed.fingerprint)

    def test_manifest_rejects_unverified_smoke(self) -> None:
        manifest = _manifest_payload()
        manifest["smoke_test"]["passed"] = False
        with tempfile.TemporaryDirectory() as temp_dir, self.assertRaisesRegex(
            MangaLMMRuntimeContractError,
            "passed smoke test",
        ):
            self._build_contract(Path(temp_dir), manifest=manifest)

    def test_manifest_rejects_cpu_or_wrong_alias_smoke(self) -> None:
        for field, value in (
            ("device", "CPU"),
            ("model_alias", "unexpected"),
        ):
            manifest = _manifest_payload()
            manifest["smoke_test"][field] = value
            with self.subTest(field=field), tempfile.TemporaryDirectory(
            ) as temp_dir, self.assertRaisesRegex(
                MangaLMMRuntimeContractError,
                "CUDA device and exact model alias",
            ):
                self._build_contract(Path(temp_dir), manifest=manifest)

    def test_manifest_rejects_wrong_model_hash(self) -> None:
        manifest = _manifest_payload()
        manifest["files"][0]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temp_dir, self.assertRaisesRegex(
            MangaLMMRuntimeContractError,
            "model contract mismatch",
        ):
            self._build_contract(Path(temp_dir), manifest=manifest)

    def test_runtime_options_reject_ubatch_larger_than_batch(self) -> None:
        with self.assertRaisesRegex(
            MangaLMMRuntimeContractError,
            "may not exceed",
        ):
            resolve_mangalmm_runtime_options(
                {
                    "MANGALMM_LLAMA_BATCH_SIZE": "128",
                    "MANGALMM_LLAMA_UBATCH_SIZE": "512",
                }
            )

    def test_volume_name_rejects_shell_metacharacters(self) -> None:
        for value in ("", "../models", "volume;rm", "volume name"):
            with self.subTest(value=value), self.assertRaises(
                MangaLMMRuntimeContractError
            ):
                validate_mangalmm_volume_name(value)


if __name__ == "__main__":
    unittest.main()
