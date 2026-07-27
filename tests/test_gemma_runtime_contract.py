from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from modules.translation.gemma_runtime_contract import (
    DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
    DEFAULT_GEMMA_MODEL_VOLUME,
    DEFAULT_GEMMA_PREPARATION_RUNTIME_CONFIGURATION,
    DEFAULT_GEMMA_READY_MANIFEST,
    GEMMA_MODEL_SPECS,
    GEMMA_RUNTIME_FINGERPRINT_LABEL,
    GEMMA_RUNTIME_KIND_LABEL,
    GEMMA_RUNTIME_MANIFEST_SHA_LABEL,
    GEMMA_RUNTIME_MODEL_SHA_LABEL,
    GEMMA_RUNTIME_PREPARATION_LABEL,
    GEMMA_RUNTIME_VOLUME_LABEL,
    GemmaRuntimeContract,
    GemmaRuntimeContractError,
    build_gemma_runtime_contract,
    container_contract_mismatch_reasons,
    resolve_gemma_runtime_options,
    validate_gemma_model_name,
    validate_gemma_volume_name,
)


_IMAGE_ID = "sha256:22e0e3bfe967af4fd1df6a918022abbfd4e72e4d40a4769e616a4176790acbcb"
_LEGACY_MODEL = "gemma-4-26B-IQ4_NL.gguf"


def _manifest_payload() -> dict:
    return {
        "schema_version": 1,
        "runtime": "Gemma",
        "preparation_version": 1,
        "volume_name": DEFAULT_GEMMA_MODEL_VOLUME,
        "ready": True,
        "source_image_ref": DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
        "source_image_digest": _IMAGE_ID,
        "source_image_id": _IMAGE_ID,
        "default_model": _LEGACY_MODEL,
        "runtime_configuration": dict(
            DEFAULT_GEMMA_PREPARATION_RUNTIME_CONFIGURATION
        ),
        "files": [
            {
                "name": name,
                "bytes": spec["bytes"],
                "sha256": spec["sha256"],
                "role": spec["role"],
            }
            for name, spec in GEMMA_MODEL_SPECS.items()
        ],
        "smoke_test": {
            "passed": True,
            "model": (
                "Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ4_XS.gguf"
            ),
        },
    }


def _manifest_bytes(payload: dict | None = None) -> bytes:
    return json.dumps(
        payload or _manifest_payload(),
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")


class GemmaRuntimeContractTests(unittest.TestCase):
    def _build_contract(
        self,
        *,
        payload: dict | None = None,
        model_name: str = _LEGACY_MODEL,
        environment: dict[str, str] | None = None,
        compose_text: str = "services: {}\n",
        observed_model_bytes: int | None = None,
    ) -> GemmaRuntimeContract:
        manifest_bytes = _manifest_bytes(payload)
        model_spec = GEMMA_MODEL_SPECS[model_name]
        with tempfile.TemporaryDirectory() as temp_dir:
            compose_file = Path(temp_dir) / "docker-compose.yaml"
            compose_file.write_text(compose_text, encoding="utf-8")
            return build_gemma_runtime_contract(
                manifest_bytes=manifest_bytes,
                manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                observed_model_bytes=(
                    int(model_spec["bytes"])
                    if observed_model_bytes is None
                    else observed_model_bytes
                ),
                volume_name=DEFAULT_GEMMA_MODEL_VOLUME,
                model_name=model_name,
                image_ref=DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
                image_id=_IMAGE_ID,
                compose_file=compose_file,
                environment=environment or {},
            )

    def test_build_contract_contains_pinned_runtime_identity(self) -> None:
        contract = self._build_contract()

        self.assertEqual(contract.model_name, _LEGACY_MODEL)
        self.assertEqual(contract.image_ref, DEFAULT_GEMMA_LLAMA_CPP_IMAGE)
        self.assertEqual(contract.image_id, _IMAGE_ID)
        self.assertEqual(contract.volume_name, DEFAULT_GEMMA_MODEL_VOLUME)
        self.assertEqual(contract.ready_manifest_name, DEFAULT_GEMMA_READY_MANIFEST)
        self.assertIn("-ctk", contract.command)
        self.assertIn("--spec-type", contract.command)

        compose_environment = contract.compose_environment()
        self.assertEqual(compose_environment["LLAMA_CPP_PULL_POLICY"], "missing")
        self.assertEqual(
            compose_environment["GEMMA_RUNTIME_FINGERPRINT"],
            contract.fingerprint,
        )
        self.assertEqual(
            compose_environment["GEMMA_READY_MANIFEST_SHA256"],
            contract.ready_manifest_sha256,
        )

    def test_contract_rejects_manifest_sha_mismatch(self) -> None:
        manifest_bytes = _manifest_bytes()
        with tempfile.TemporaryDirectory() as temp_dir:
            compose_file = Path(temp_dir) / "docker-compose.yaml"
            compose_file.write_text("services: {}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                GemmaRuntimeContractError,
                "does not match the mounted manifest bytes",
            ):
                build_gemma_runtime_contract(
                    manifest_bytes=manifest_bytes,
                    manifest_sha256="0" * 64,
                    observed_model_bytes=GEMMA_MODEL_SPECS[_LEGACY_MODEL]["bytes"],
                    volume_name=DEFAULT_GEMMA_MODEL_VOLUME,
                    model_name=_LEGACY_MODEL,
                    image_ref=DEFAULT_GEMMA_LLAMA_CPP_IMAGE,
                    image_id=_IMAGE_ID,
                    compose_file=compose_file,
                )

    def test_contract_rejects_image_digest_or_failed_smoke(self) -> None:
        for field, value, expected_message in (
            ("source_image_digest", "sha256:" + ("0" * 64), "source_image_digest"),
            ("smoke_test", {"passed": False}, "successful smoke test"),
        ):
            with self.subTest(field=field):
                payload = _manifest_payload()
                payload[field] = value
                with self.assertRaisesRegex(
                    GemmaRuntimeContractError,
                    expected_message,
                ):
                    self._build_contract(payload=payload)

    def test_contract_rejects_mounted_model_size_mismatch(self) -> None:
        with self.assertRaisesRegex(
            GemmaRuntimeContractError,
            "Mounted Gemma model size mismatch",
        ):
            self._build_contract(observed_model_bytes=123)

    def test_contract_rejects_incomplete_registry_or_runtime_configuration(self) -> None:
        payload = _manifest_payload()
        payload["files"].pop()
        with self.assertRaisesRegex(
            GemmaRuntimeContractError,
            "model registry",
        ):
            self._build_contract(payload=payload)

        payload = _manifest_payload()
        payload["runtime_configuration"]["threads"] = 11
        with self.assertRaisesRegex(
            GemmaRuntimeContractError,
            "runtime configuration",
        ):
            self._build_contract(payload=payload)

    def test_runtime_options_accept_only_validated_candidate_values(self) -> None:
        options = resolve_gemma_runtime_options(
            {
                "LLAMA_CACHE_TYPE_K": "Q8_0",
                "LLAMA_CACHE_TYPE_V": "q8_0",
                "LLAMA_CACHE_RAM_MIB": "256",
                "LLAMA_SPEC_TYPE": "NGRAM-MOD",
                "LLAMA_SPEC_DRAFT_N_MAX": "4",
                "LLAMA_N_GPU_LAYERS": 0,
            }
        )
        self.assertEqual(options["LLAMA_CACHE_TYPE_K"], "q8_0")
        self.assertEqual(options["LLAMA_CACHE_RAM_MIB"], "256")
        self.assertEqual(options["LLAMA_SPEC_TYPE"], "ngram-mod")
        self.assertEqual(options["LLAMA_SPEC_DRAFT_N_MAX"], "4")
        self.assertEqual(options["LLAMA_N_GPU_LAYERS"], "0")

        invalid_values = {
            "LLAMA_CTX_SIZE": "512",
            "LLAMA_CACHE_TYPE_K": "q4_0",
            "LLAMA_CACHE_RAM_MIB": "512",
            "LLAMA_SPEC_TYPE": "draft-model",
            "LLAMA_SPEC_DRAFT_N_MAX": "16",
        }
        for key, value in invalid_values.items():
            with self.subTest(key=key):
                with self.assertRaises(GemmaRuntimeContractError):
                    resolve_gemma_runtime_options({key: value})

    def test_fingerprint_changes_with_compose_or_runtime_configuration(self) -> None:
        baseline = self._build_contract()
        changed_compose = self._build_contract(compose_text="services:\n  gemma: {}\n")
        changed_runtime = self._build_contract(
            environment={"LLAMA_CACHE_TYPE_K": "q8_0"}
        )

        self.assertNotEqual(baseline.fingerprint, changed_compose.fingerprint)
        self.assertNotEqual(baseline.fingerprint, changed_runtime.fingerprint)
        self.assertNotEqual(baseline.command_sha256, changed_runtime.command_sha256)

    def test_exact_container_contract_has_no_mismatch(self) -> None:
        contract = self._build_contract()
        inspection = self._matching_inspection(contract)
        self.assertEqual(
            container_contract_mismatch_reasons(inspection, contract),
            [],
        )

    def test_container_contract_detects_identity_command_and_mount_changes(self) -> None:
        contract = self._build_contract()
        mutations = {
            "fingerprint-label": lambda item: item["Config"]["Labels"].update(
                {GEMMA_RUNTIME_FINGERPRINT_LABEL: "different"}
            ),
            "image-id": lambda item: item.update({"Image": "sha256:different"}),
            "command": lambda item: item["Config"].update({"Cmd": ["--wrong"]}),
            "volume": lambda item: item["Mounts"][0].update({"Name": "other-volume"}),
            "read-write": lambda item: item["Mounts"][0].update({"RW": True}),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                inspection = self._matching_inspection(contract)
                mutate(inspection)
                self.assertTrue(
                    container_contract_mismatch_reasons(inspection, contract)
                )

    def test_model_and_volume_names_reject_paths_or_shell_metacharacters(self) -> None:
        self.assertEqual(
            validate_gemma_model_name(_LEGACY_MODEL),
            _LEGACY_MODEL,
        )
        self.assertEqual(
            validate_gemma_volume_name(DEFAULT_GEMMA_MODEL_VOLUME),
            DEFAULT_GEMMA_MODEL_VOLUME,
        )
        for invalid in ("../model.gguf", "folder/model.gguf", "model.bin"):
            with self.subTest(model=invalid):
                with self.assertRaises(GemmaRuntimeContractError):
                    validate_gemma_model_name(invalid)
        for invalid in ("bad volume", "$(touch-x)", ""):
            with self.subTest(volume=invalid):
                with self.assertRaises(GemmaRuntimeContractError):
                    validate_gemma_volume_name(invalid)

    @staticmethod
    def _matching_inspection(contract: GemmaRuntimeContract) -> dict:
        return {
            "Config": {
                "Image": contract.image_ref,
                "Cmd": list(contract.command),
                "Labels": {
                    GEMMA_RUNTIME_KIND_LABEL: "gemma",
                    GEMMA_RUNTIME_FINGERPRINT_LABEL: contract.fingerprint,
                    GEMMA_RUNTIME_VOLUME_LABEL: contract.volume_name,
                    GEMMA_RUNTIME_MANIFEST_SHA_LABEL: (
                        contract.ready_manifest_sha256
                    ),
                    GEMMA_RUNTIME_MODEL_SHA_LABEL: contract.model_sha256,
                    GEMMA_RUNTIME_PREPARATION_LABEL: str(
                        contract.preparation_version
                    ),
                },
            },
            "Image": contract.image_id,
            "Mounts": [
                {
                    "Destination": "/models",
                    "Type": "volume",
                    "Name": contract.volume_name,
                    "RW": False,
                }
            ],
            "State": {"Running": False},
        }


if __name__ == "__main__":
    unittest.main()
