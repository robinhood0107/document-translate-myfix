from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from modules.ocr.paddleocr_vl_spotting.runtime_contract import (
    DEFAULT_PADDLE_SPOTTING_LLAMA_CPP_IMAGE,
    DEFAULT_PADDLE_SPOTTING_MODEL_VOLUME,
    PADDLE_SPOTTING_IMAGE_MAX_PIXELS,
    PADDLE_SPOTTING_MMPROJ_NAME,
    PADDLE_SPOTTING_MODEL_ALIAS,
    PADDLE_SPOTTING_MODEL_NAME,
    PADDLE_SPOTTING_MODEL_SPECS,
    PADDLE_SPOTTING_RUNTIME_PREPARATION_VERSION,
    PaddleSpottingRuntimeContractError,
    build_paddle_spotting_runtime_contract,
    resolve_paddle_spotting_runtime_options,
    validate_paddle_spotting_volume_name,
)


_IMAGE_ID = "sha256:" + ("1" * 64)


def _manifest_payload() -> dict:
    return {
        "schema_version": 1,
        "runtime": "PaddleOCR-VL-Spotting-llama.cpp",
        "preparation_version": (
            PADDLE_SPOTTING_RUNTIME_PREPARATION_VERSION
        ),
        "volume_name": DEFAULT_PADDLE_SPOTTING_MODEL_VOLUME,
        "ready": True,
        "source_image_ref": DEFAULT_PADDLE_SPOTTING_LLAMA_CPP_IMAGE,
        "source_image_id": _IMAGE_ID,
        "spotting_contract": {
            "prompt": "Spotting:",
            "special_tokens": True,
            "clip.vision.image_max_pixels": PADDLE_SPOTTING_IMAGE_MAX_PIXELS,
        },
        "smoke_test": {
            "passed": True,
            "device": "CUDA",
            "model_alias": PADDLE_SPOTTING_MODEL_ALIAS,
        },
        "files": [
            {
                "name": name,
                "bytes": spec["bytes"],
                "sha256": spec["sha256"],
                "role": spec["role"],
                **(
                    {"derived_from_sha256": spec["derived_from_sha256"]}
                    if spec.get("derived_from_sha256")
                    else {}
                ),
            }
            for name, spec in PADDLE_SPOTTING_MODEL_SPECS.items()
        ],
    }


class PaddleSpottingRuntimeContractTests(unittest.TestCase):
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
        image_id = _IMAGE_ID
        return build_paddle_spotting_runtime_contract(
            manifest_bytes=manifest_bytes,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            observed_file_bytes={
                name: int(spec["bytes"])
                for name, spec in PADDLE_SPOTTING_MODEL_SPECS.items()
            },
            volume_name=DEFAULT_PADDLE_SPOTTING_MODEL_VOLUME,
            llama_image_ref=DEFAULT_PADDLE_SPOTTING_LLAMA_CPP_IMAGE,
            llama_image_id=image_id,
            compose_file=compose_file,
            environment=environment,
        )

    def test_contract_uses_only_official_spotting_projector_and_special_tokens(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            contract = self._build_contract(Path(temp_dir))

        self.assertEqual(contract.volume_name, DEFAULT_PADDLE_SPOTTING_MODEL_VOLUME)
        self.assertIn(f"/models/{PADDLE_SPOTTING_MODEL_NAME}", contract.command)
        self.assertIn(f"/models/{PADDLE_SPOTTING_MMPROJ_NAME}", contract.command)
        self.assertIn("--special", contract.command)
        self.assertNotIn("PaddleOCR-VL-1.6-mmproj.gguf", contract.command)

    def test_rejects_crop_projector_metadata_or_hash(self) -> None:
        for field, value in (
            ("clip.vision.image_max_pixels", 1_003_520),
            ("prompt", "OCR:"),
            ("special_tokens", False),
        ):
            manifest = _manifest_payload()
            manifest["spotting_contract"][field] = value
            with self.subTest(field=field), tempfile.TemporaryDirectory(
            ) as temp_dir, self.assertRaisesRegex(
                PaddleSpottingRuntimeContractError,
                "official Spotting projector contract",
            ):
                self._build_contract(Path(temp_dir), manifest=manifest)

        manifest = _manifest_payload()
        for entry in manifest["files"]:
            if entry["name"] == PADDLE_SPOTTING_MMPROJ_NAME:
                entry["sha256"] = (
                    "204d757d7610d9b3faab10d506d69e5b244e32bf765e2bab2d0167e65e0a058a"
                )
        with tempfile.TemporaryDirectory() as temp_dir, self.assertRaisesRegex(
            PaddleSpottingRuntimeContractError,
            "model contract mismatch",
        ):
            self._build_contract(Path(temp_dir), manifest=manifest)

    def test_runtime_option_changes_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline = self._build_contract(root)
            changed = self._build_contract(
                root,
                environment={"PADDLEOCR_SPOTTING_LLAMA_CTX_SIZE": "8192"},
            )

        self.assertNotEqual(baseline.command_sha256, changed.command_sha256)
        self.assertNotEqual(baseline.fingerprint, changed.fingerprint)

    def test_rejects_unverified_device_or_alias_smoke(self) -> None:
        for field, value in (
            ("device", "CPU"),
            ("model_alias", "PaddleOCR-VL-1.6-0.9B"),
        ):
            manifest = _manifest_payload()
            manifest["smoke_test"][field] = value
            with self.subTest(field=field), tempfile.TemporaryDirectory(
            ) as temp_dir, self.assertRaisesRegex(
                PaddleSpottingRuntimeContractError,
                "CUDA device and exact model alias",
            ):
                self._build_contract(Path(temp_dir), manifest=manifest)

    def test_runtime_options_reject_invalid_batch_relationship(self) -> None:
        with self.assertRaisesRegex(
            PaddleSpottingRuntimeContractError,
            "may not exceed",
        ):
            resolve_paddle_spotting_runtime_options(
                {
                    "PADDLEOCR_SPOTTING_LLAMA_BATCH_SIZE": "128",
                    "PADDLEOCR_SPOTTING_LLAMA_UBATCH_SIZE": "512",
                }
            )

    def test_volume_name_rejects_shell_metacharacters(self) -> None:
        for value in ("", "../models", "volume;rm", "volume name"):
            with self.subTest(value=value), self.assertRaises(
                PaddleSpottingRuntimeContractError
            ):
                validate_paddle_spotting_volume_name(value)


if __name__ == "__main__":
    unittest.main()
