from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from modules.ocr.paddle_llamacpp_runtime_contract import (
    DEFAULT_PADDLE_LLAMA_CPP_IMAGE,
)


PADDLE_SPOTTING_RUNTIME_MANIFEST_SCHEMA_VERSION = 1
PADDLE_SPOTTING_RUNTIME_PREPARATION_VERSION = 2
PADDLE_SPOTTING_IMAGE_MAX_PIXELS = 1_605_632
DEFAULT_PADDLE_SPOTTING_MODEL_VOLUME = (
    "comic-translate-paddleocr-vl-spotting-llamacpp-models-v2"
)
DEFAULT_PADDLE_SPOTTING_READY_MANIFEST = (
    ".comic-translate-paddleocr-vl-spotting-llamacpp-ready-v2.json"
)
DEFAULT_PADDLE_SPOTTING_LLAMA_CPP_IMAGE = (
    DEFAULT_PADDLE_LLAMA_CPP_IMAGE
)
PADDLE_SPOTTING_MODEL_NAME = (
    "PaddleOCR-VL-1.6-Spotting-GGUF.gguf"
)
PADDLE_SPOTTING_MMPROJ_NAME = (
    "PaddleOCR-VL-1.6-Spotting-mmproj.gguf"
)
PADDLE_SPOTTING_MODEL_ALIAS = "PaddleOCR-VL-1.6-Spotting"
PADDLE_SPOTTING_MODEL_SPECS: dict[str, dict[str, Any]] = {
    PADDLE_SPOTTING_MODEL_NAME: {
        "bytes": 935_769_056,
        "sha256": (
            "f3ae46ec885050acf4b3d31944431e1fd90d50664fb09126af4a3c050ba14ee8"
        ),
        "role": "vlm",
    },
    PADDLE_SPOTTING_MMPROJ_NAME: {
        "bytes": 881_770_560,
        "sha256": (
            "8e011479092c5e82c8c1c2d85d52b9ac48df12183c5c7bc3190190732259db09"
        ),
        "role": "vision-projector",
        "derived_from_sha256": (
            "204d757d7610d9b3faab10d506d69e5b244e32bf765e2bab2d0167e65e0a058a"
        ),
    },
}

DEFAULT_PADDLE_SPOTTING_RUNTIME_OPTIONS: dict[str, str] = {
    "PADDLEOCR_SPOTTING_LLAMA_CTX_SIZE": "4096",
    "PADDLEOCR_SPOTTING_LLAMA_PARALLEL": "1",
    "PADDLEOCR_SPOTTING_LLAMA_THREADS": "10",
    "PADDLEOCR_SPOTTING_LLAMA_BATCH_SIZE": "2048",
    "PADDLEOCR_SPOTTING_LLAMA_UBATCH_SIZE": "512",
    "PADDLEOCR_SPOTTING_LLAMA_GPU_LAYERS": "all",
    "PADDLEOCR_SPOTTING_LLAMA_SLEEP_IDLE_SECONDS": "5",
}

PADDLE_SPOTTING_RUNTIME_FINGERPRINT_LABEL = (
    "com.comictranslate.paddleocr-spotting-runtime-fingerprint"
)
PADDLE_SPOTTING_RUNTIME_VOLUME_LABEL = (
    "com.comictranslate.paddleocr-spotting-model-volume"
)
PADDLE_SPOTTING_RUNTIME_MANIFEST_SHA_LABEL = (
    "com.comictranslate.paddleocr-spotting-ready-manifest-sha256"
)
PADDLE_SPOTTING_RUNTIME_MODEL_SHA_LABEL = (
    "com.comictranslate.paddleocr-spotting-model-sha256"
)
PADDLE_SPOTTING_RUNTIME_MMPROJ_SHA_LABEL = (
    "com.comictranslate.paddleocr-spotting-mmproj-sha256"
)

_SAFE_DOCKER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PaddleSpottingRuntimeContractError(ValueError):
    pass


@dataclass(frozen=True)
class PaddleSpottingRuntimeContract:
    volume_name: str
    ready_manifest_name: str
    ready_manifest_sha256: str
    preparation_version: int
    llama_image_ref: str
    llama_image_id: str
    compose_file_sha256: str
    command_sha256: str
    fingerprint: str
    command: tuple[str, ...]
    runtime_options: Mapping[str, str]

    def compose_environment(self) -> dict[str, str]:
        values = dict(self.runtime_options)
        values.update(
            {
                "PADDLEOCR_SPOTTING_LLAMA_CPP_IMAGE": (
                    self.llama_image_ref
                ),
                "PADDLEOCR_SPOTTING_MODEL_VOLUME": self.volume_name,
                "PADDLEOCR_SPOTTING_MODEL_FILE": (
                    PADDLE_SPOTTING_MODEL_NAME
                ),
                "PADDLEOCR_SPOTTING_MMPROJ_FILE": (
                    PADDLE_SPOTTING_MMPROJ_NAME
                ),
                "PADDLEOCR_SPOTTING_MODEL_ALIAS": (
                    PADDLE_SPOTTING_MODEL_ALIAS
                ),
                "PADDLEOCR_SPOTTING_RUNTIME_FINGERPRINT": (
                    self.fingerprint
                ),
                "PADDLEOCR_SPOTTING_READY_MANIFEST_SHA256": (
                    self.ready_manifest_sha256
                ),
                "PADDLEOCR_SPOTTING_MODEL_SHA256": str(
                    PADDLE_SPOTTING_MODEL_SPECS[
                        PADDLE_SPOTTING_MODEL_NAME
                    ]["sha256"]
                ),
                "PADDLEOCR_SPOTTING_MMPROJ_SHA256": str(
                    PADDLE_SPOTTING_MODEL_SPECS[
                        PADDLE_SPOTTING_MMPROJ_NAME
                    ]["sha256"]
                ),
                "PADDLEOCR_SPOTTING_PREPARATION_VERSION": str(
                    self.preparation_version
                ),
            }
        )
        return values


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_paddle_spotting_volume_name(value: str) -> str:
    normalized = str(value or "").strip()
    if not _SAFE_DOCKER_NAME.fullmatch(normalized):
        raise PaddleSpottingRuntimeContractError(
            "Invalid PaddleOCR-VL Spotting model volume name: "
            f"{value!r}"
        )
    return normalized


def resolve_paddle_spotting_runtime_options(
    environment: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    source = environment or {}
    values: dict[str, str] = {}
    for key, default in DEFAULT_PADDLE_SPOTTING_RUNTIME_OPTIONS.items():
        raw_value = source.get(key, default)
        normalized = str(raw_value).strip() if raw_value is not None else ""
        values[key] = normalized or default

    ranges = {
        "PADDLEOCR_SPOTTING_LLAMA_CTX_SIZE": (4096, 16_384),
        "PADDLEOCR_SPOTTING_LLAMA_PARALLEL": (1, 4),
        "PADDLEOCR_SPOTTING_LLAMA_THREADS": (1, 64),
        "PADDLEOCR_SPOTTING_LLAMA_BATCH_SIZE": (128, 4096),
        "PADDLEOCR_SPOTTING_LLAMA_UBATCH_SIZE": (64, 2048),
        "PADDLEOCR_SPOTTING_LLAMA_SLEEP_IDLE_SECONDS": (1, 300),
    }
    for key, (minimum, maximum) in ranges.items():
        try:
            parsed = int(values[key])
        except ValueError as exc:
            raise PaddleSpottingRuntimeContractError(
                f"{key} must be an integer."
            ) from exc
        if parsed < minimum or parsed > maximum:
            raise PaddleSpottingRuntimeContractError(
                f"{key} must be between {minimum} and {maximum}: {parsed}"
            )
        values[key] = str(parsed)
    if int(values["PADDLEOCR_SPOTTING_LLAMA_UBATCH_SIZE"]) > int(
        values["PADDLEOCR_SPOTTING_LLAMA_BATCH_SIZE"]
    ):
        raise PaddleSpottingRuntimeContractError(
            "PADDLEOCR_SPOTTING_LLAMA_UBATCH_SIZE may not exceed "
            "PADDLEOCR_SPOTTING_LLAMA_BATCH_SIZE."
        )

    gpu_layers = values[
        "PADDLEOCR_SPOTTING_LLAMA_GPU_LAYERS"
    ].lower()
    if gpu_layers != "all":
        try:
            parsed_layers = int(gpu_layers)
        except ValueError as exc:
            raise PaddleSpottingRuntimeContractError(
                "PADDLEOCR_SPOTTING_LLAMA_GPU_LAYERS must be 'all' "
                "or an integer."
            ) from exc
        if parsed_layers < 0 or parsed_layers > 999:
            raise PaddleSpottingRuntimeContractError(
                "PADDLEOCR_SPOTTING_LLAMA_GPU_LAYERS must be between "
                "0 and 999."
            )
        gpu_layers = str(parsed_layers)
    values["PADDLEOCR_SPOTTING_LLAMA_GPU_LAYERS"] = gpu_layers
    return values


def build_paddle_spotting_server_command(
    runtime_options: Mapping[str, str],
) -> tuple[str, ...]:
    return (
        "-m",
        f"/models/{PADDLE_SPOTTING_MODEL_NAME}",
        "--mmproj",
        f"/models/{PADDLE_SPOTTING_MMPROJ_NAME}",
        "--alias",
        PADDLE_SPOTTING_MODEL_ALIAS,
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "-c",
        runtime_options["PADDLEOCR_SPOTTING_LLAMA_CTX_SIZE"],
        "-np",
        runtime_options["PADDLEOCR_SPOTTING_LLAMA_PARALLEL"],
        "-t",
        runtime_options["PADDLEOCR_SPOTTING_LLAMA_THREADS"],
        "-b",
        runtime_options["PADDLEOCR_SPOTTING_LLAMA_BATCH_SIZE"],
        "-ub",
        runtime_options["PADDLEOCR_SPOTTING_LLAMA_UBATCH_SIZE"],
        "--n-gpu-layers",
        runtime_options["PADDLEOCR_SPOTTING_LLAMA_GPU_LAYERS"],
        "--fit",
        "off",
        "--flash-attn",
        "on",
        "--temp",
        "0",
        "--special",
        "--metrics",
        "--sleep-idle-seconds",
        runtime_options[
            "PADDLEOCR_SPOTTING_LLAMA_SLEEP_IDLE_SECONDS"
        ],
    )


def _manifest_file_entries(manifest: dict[str, Any]) -> dict[str, Any]:
    files = manifest.get("files")
    if isinstance(files, list):
        return {
            str(entry.get("name", "")): entry
            for entry in files
            if isinstance(entry, dict)
        }
    if isinstance(files, dict):
        return {
            str(name): entry
            for name, entry in files.items()
            if isinstance(entry, dict)
        }
    return {}


def build_paddle_spotting_runtime_contract(
    *,
    manifest_bytes: bytes,
    manifest_sha256: str,
    observed_file_bytes: Mapping[str, int],
    volume_name: str,
    llama_image_ref: str,
    llama_image_id: str,
    compose_file: Path,
    environment: Mapping[str, Any] | None = None,
) -> PaddleSpottingRuntimeContract:
    safe_volume = validate_paddle_spotting_volume_name(volume_name)
    normalized_manifest_sha = str(manifest_sha256 or "").strip().lower()
    if not _SHA256.fullmatch(normalized_manifest_sha):
        raise PaddleSpottingRuntimeContractError(
            "Invalid PaddleOCR-VL Spotting ready manifest SHA-256."
        )
    if hashlib.sha256(manifest_bytes).hexdigest() != normalized_manifest_sha:
        raise PaddleSpottingRuntimeContractError(
            "PaddleOCR-VL Spotting ready manifest SHA-256 does not "
            "match the mounted bytes."
        )
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PaddleSpottingRuntimeContractError(
            "Unable to parse the PaddleOCR-VL Spotting ready manifest."
        ) from exc
    if not isinstance(manifest, dict):
        raise PaddleSpottingRuntimeContractError(
            "PaddleOCR-VL Spotting ready manifest must be an object."
        )

    required_header = {
        "schema_version": PADDLE_SPOTTING_RUNTIME_MANIFEST_SCHEMA_VERSION,
        "runtime": "PaddleOCR-VL-Spotting-llama.cpp",
        "preparation_version": PADDLE_SPOTTING_RUNTIME_PREPARATION_VERSION,
        "volume_name": safe_volume,
        "ready": True,
        "source_image_ref": llama_image_ref,
        "source_image_id": llama_image_id,
    }
    for key, expected in required_header.items():
        if manifest.get(key) != expected:
            raise PaddleSpottingRuntimeContractError(
                "PaddleOCR-VL Spotting manifest field "
                f"{key!r} does not match the runtime contract."
            )
    spotting_contract = manifest.get("spotting_contract")
    if not isinstance(spotting_contract, dict) or {
        "prompt": spotting_contract.get("prompt"),
        "special_tokens": spotting_contract.get("special_tokens"),
        "clip.vision.image_max_pixels": spotting_contract.get(
            "clip.vision.image_max_pixels"
        ),
    } != {
        "prompt": "Spotting:",
        "special_tokens": True,
        "clip.vision.image_max_pixels": (
            PADDLE_SPOTTING_IMAGE_MAX_PIXELS
        ),
    }:
        raise PaddleSpottingRuntimeContractError(
            "PaddleOCR-VL Spotting manifest does not contain the "
            "official Spotting projector contract."
        )
    smoke_test = manifest.get("smoke_test")
    if not isinstance(smoke_test, dict) or smoke_test.get("passed") is not True:
        raise PaddleSpottingRuntimeContractError(
            "PaddleOCR-VL Spotting manifest does not contain a passed "
            "smoke test."
        )
    if (
        str(smoke_test.get("device", "")).strip().upper() != "CUDA"
        or str(smoke_test.get("model_alias", "")).strip()
        != PADDLE_SPOTTING_MODEL_ALIAS
    ):
        raise PaddleSpottingRuntimeContractError(
            "PaddleOCR-VL Spotting smoke test must record the CUDA "
            "device and exact model alias."
        )

    entries = _manifest_file_entries(manifest)
    if set(entries) != set(PADDLE_SPOTTING_MODEL_SPECS):
        raise PaddleSpottingRuntimeContractError(
            "PaddleOCR-VL Spotting manifest file registry does not "
            "match the product registry."
        )
    for name, spec in PADDLE_SPOTTING_MODEL_SPECS.items():
        entry = entries[name]
        if (
            int(entry.get("bytes", -1)) != int(spec["bytes"])
            or str(entry.get("sha256", "")).lower()
            != str(spec["sha256"])
            or str(entry.get("role", "")) != str(spec["role"])
            or int(observed_file_bytes.get(name, -1))
            != int(spec["bytes"])
        ):
            raise PaddleSpottingRuntimeContractError(
                "PaddleOCR-VL Spotting model contract mismatch: "
                f"{name}"
            )
        derived_from = spec.get("derived_from_sha256")
        if derived_from and str(
            entry.get("derived_from_sha256", "")
        ).lower() != str(derived_from):
            raise PaddleSpottingRuntimeContractError(
                "PaddleOCR-VL Spotting projector provenance mismatch."
            )

    runtime_options = resolve_paddle_spotting_runtime_options(environment)
    command = build_paddle_spotting_server_command(runtime_options)
    compose_sha256 = hashlib.sha256(compose_file.read_bytes()).hexdigest()
    command_sha256 = _canonical_sha256(command)
    fingerprint = _canonical_sha256(
        {
            "contract_schema_version": 1,
            "volume_name": safe_volume,
            "ready_manifest_name": (
                DEFAULT_PADDLE_SPOTTING_READY_MANIFEST
            ),
            "ready_manifest_sha256": normalized_manifest_sha,
            "llama_image_ref": llama_image_ref,
            "llama_image_id": llama_image_id,
            "compose_file_sha256": compose_sha256,
            "command_sha256": command_sha256,
            "files": PADDLE_SPOTTING_MODEL_SPECS,
            "spotting_contract": {
                "prompt": "Spotting:",
                "special_tokens": True,
                "clip.vision.image_max_pixels": (
                    PADDLE_SPOTTING_IMAGE_MAX_PIXELS
                ),
            },
        }
    )
    return PaddleSpottingRuntimeContract(
        volume_name=safe_volume,
        ready_manifest_name=DEFAULT_PADDLE_SPOTTING_READY_MANIFEST,
        ready_manifest_sha256=normalized_manifest_sha,
        preparation_version=PADDLE_SPOTTING_RUNTIME_PREPARATION_VERSION,
        llama_image_ref=llama_image_ref,
        llama_image_id=llama_image_id,
        compose_file_sha256=compose_sha256,
        command_sha256=command_sha256,
        fingerprint=fingerprint,
        command=command,
        runtime_options=runtime_options,
    )
