from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PADDLE_LLAMA_RUNTIME_MANIFEST_SCHEMA_VERSION = 1
PADDLE_LLAMA_RUNTIME_PREPARATION_VERSION = 1
DEFAULT_PADDLE_LLAMA_MODEL_VOLUME = (
    "comic-translate-paddleocr-vl-llamacpp-models-v1"
)
DEFAULT_PADDLE_LLAMA_READY_MANIFEST = (
    ".comic-translate-paddleocr-vl-llamacpp-ready-v1.json"
)
DEFAULT_PADDLE_LLAMA_CPP_IMAGE = (
    "ghcr.io/ggml-org/llama.cpp@sha256:"
    "22e0e3bfe967af4fd1df6a918022abbfd4e72e4d40a4769e616a4176790acbcb"
)
DEFAULT_PADDLE_LAYOUT_IMAGE = (
    "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/"
    "paddlepaddle/paddleocr-genai-vllm-server@sha256:"
    "d0d32c04a2119613d25a0a4c292e165ccc107954b74580613cf59e378037f8f5"
)

PADDLE_LLAMA_MODEL_NAME = "PaddleOCR-VL-1.6-GGUF.gguf"
PADDLE_LLAMA_MMPROJ_NAME = "PaddleOCR-VL-1.6-GGUF-mmproj.gguf"
PADDLE_LLAMA_MODEL_ALIAS = "PaddleOCR-VL-1.6-0.9B"
PADDLE_LLAMA_MODEL_SPECS: dict[str, dict[str, Any]] = {
    PADDLE_LLAMA_MODEL_NAME: {
        "bytes": 935_769_056,
        "sha256": "f3ae46ec885050acf4b3d31944431e1fd90d50664fb09126af4a3c050ba14ee8",
        "role": "vlm",
    },
    PADDLE_LLAMA_MMPROJ_NAME: {
        "bytes": 881_770_560,
        "sha256": "204d757d7610d9b3faab10d506d69e5b244e32bf765e2bab2d0167e65e0a058a",
        "role": "vision-projector",
    },
}

DEFAULT_PADDLE_LLAMA_RUNTIME_OPTIONS: dict[str, str] = {
    "PADDLEOCR_LLAMA_CTX_SIZE": "4096",
    "PADDLEOCR_LLAMA_PARALLEL": "1",
    "PADDLEOCR_LLAMA_THREADS": "10",
    "PADDLEOCR_LLAMA_BATCH_SIZE": "2048",
    "PADDLEOCR_LLAMA_UBATCH_SIZE": "512",
    "PADDLEOCR_LLAMA_GPU_LAYERS": "all",
    "PADDLEOCR_LLAMA_SLEEP_IDLE_SECONDS": "5",
}

PADDLE_RUNTIME_FINGERPRINT_LABEL = (
    "com.comictranslate.paddleocr-runtime-fingerprint"
)
PADDLE_RUNTIME_VOLUME_LABEL = "com.comictranslate.paddleocr-model-volume"
PADDLE_RUNTIME_MANIFEST_SHA_LABEL = (
    "com.comictranslate.paddleocr-ready-manifest-sha256"
)
PADDLE_RUNTIME_MODEL_SHA_LABEL = (
    "com.comictranslate.paddleocr-model-sha256"
)
PADDLE_RUNTIME_MMPROJ_SHA_LABEL = (
    "com.comictranslate.paddleocr-mmproj-sha256"
)

_SAFE_DOCKER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PaddleLlamaRuntimeContractError(ValueError):
    pass


@dataclass(frozen=True)
class PaddleLlamaRuntimeContract:
    volume_name: str
    ready_manifest_name: str
    ready_manifest_sha256: str
    preparation_version: int
    llama_image_ref: str
    llama_image_id: str
    layout_image_ref: str
    layout_image_id: str
    compose_file_sha256: str
    pipeline_config_sha256: str
    command_sha256: str
    fingerprint: str
    command: tuple[str, ...]
    runtime_options: Mapping[str, str]

    def compose_environment(self) -> dict[str, str]:
        values = dict(self.runtime_options)
        values.update(
            {
                "PADDLEOCR_LLAMA_CPP_IMAGE": self.llama_image_ref,
                "PADDLEOCR_LAYOUT_IMAGE": self.layout_image_ref,
                "PADDLEOCR_LLAMA_MODEL_VOLUME": self.volume_name,
                "PADDLEOCR_LLAMA_MODEL_FILE": PADDLE_LLAMA_MODEL_NAME,
                "PADDLEOCR_LLAMA_MMPROJ_FILE": PADDLE_LLAMA_MMPROJ_NAME,
                "PADDLEOCR_LLAMA_MODEL_ALIAS": PADDLE_LLAMA_MODEL_ALIAS,
                "PADDLEOCR_RUNTIME_FINGERPRINT": self.fingerprint,
                "PADDLEOCR_READY_MANIFEST_SHA256": (
                    self.ready_manifest_sha256
                ),
                "PADDLEOCR_MODEL_SHA256": str(
                    PADDLE_LLAMA_MODEL_SPECS[PADDLE_LLAMA_MODEL_NAME]["sha256"]
                ),
                "PADDLEOCR_MMPROJ_SHA256": str(
                    PADDLE_LLAMA_MODEL_SPECS[PADDLE_LLAMA_MMPROJ_NAME]["sha256"]
                ),
                "PADDLEOCR_PREPARATION_VERSION": str(
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


def validate_paddle_llama_volume_name(value: str) -> str:
    normalized = str(value or "").strip()
    if not _SAFE_DOCKER_NAME.fullmatch(normalized):
        raise PaddleLlamaRuntimeContractError(
            f"Invalid PaddleOCR llama.cpp model volume name: {value!r}"
        )
    return normalized


def resolve_paddle_llama_runtime_options(
    environment: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    source = environment or {}
    values: dict[str, str] = {}
    for key, default in DEFAULT_PADDLE_LLAMA_RUNTIME_OPTIONS.items():
        raw_value = source.get(key, default)
        normalized = str(raw_value).strip() if raw_value is not None else ""
        values[key] = normalized or default

    numeric_ranges = {
        "PADDLEOCR_LLAMA_CTX_SIZE": (1024, 16_384),
        "PADDLEOCR_LLAMA_PARALLEL": (1, 8),
        "PADDLEOCR_LLAMA_THREADS": (1, 64),
        "PADDLEOCR_LLAMA_BATCH_SIZE": (128, 4096),
        "PADDLEOCR_LLAMA_UBATCH_SIZE": (64, 2048),
        "PADDLEOCR_LLAMA_SLEEP_IDLE_SECONDS": (1, 300),
    }
    for key, (minimum, maximum) in numeric_ranges.items():
        try:
            parsed = int(values[key])
        except ValueError as exc:
            raise PaddleLlamaRuntimeContractError(
                f"{key} must be an integer."
            ) from exc
        if parsed < minimum or parsed > maximum:
            raise PaddleLlamaRuntimeContractError(
                f"{key} must be between {minimum} and {maximum}: {parsed}"
            )
        values[key] = str(parsed)

    if int(values["PADDLEOCR_LLAMA_UBATCH_SIZE"]) > int(
        values["PADDLEOCR_LLAMA_BATCH_SIZE"]
    ):
        raise PaddleLlamaRuntimeContractError(
            "PADDLEOCR_LLAMA_UBATCH_SIZE may not exceed "
            "PADDLEOCR_LLAMA_BATCH_SIZE."
        )

    gpu_layers = values["PADDLEOCR_LLAMA_GPU_LAYERS"].lower()
    if gpu_layers != "all":
        try:
            parsed_gpu_layers = int(gpu_layers)
        except ValueError as exc:
            raise PaddleLlamaRuntimeContractError(
                "PADDLEOCR_LLAMA_GPU_LAYERS must be 'all' or an integer."
            ) from exc
        if parsed_gpu_layers < 0 or parsed_gpu_layers > 999:
            raise PaddleLlamaRuntimeContractError(
                "PADDLEOCR_LLAMA_GPU_LAYERS must be between 0 and 999."
            )
        gpu_layers = str(parsed_gpu_layers)
    values["PADDLEOCR_LLAMA_GPU_LAYERS"] = gpu_layers
    return values


def build_paddle_llama_server_command(
    runtime_options: Mapping[str, str],
) -> tuple[str, ...]:
    return (
        "-m",
        f"/models/{PADDLE_LLAMA_MODEL_NAME}",
        "--mmproj",
        f"/models/{PADDLE_LLAMA_MMPROJ_NAME}",
        "--alias",
        PADDLE_LLAMA_MODEL_ALIAS,
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "-c",
        runtime_options["PADDLEOCR_LLAMA_CTX_SIZE"],
        "-np",
        runtime_options["PADDLEOCR_LLAMA_PARALLEL"],
        "-t",
        runtime_options["PADDLEOCR_LLAMA_THREADS"],
        "-b",
        runtime_options["PADDLEOCR_LLAMA_BATCH_SIZE"],
        "-ub",
        runtime_options["PADDLEOCR_LLAMA_UBATCH_SIZE"],
        "--n-gpu-layers",
        runtime_options["PADDLEOCR_LLAMA_GPU_LAYERS"],
        "--fit",
        "off",
        "--flash-attn",
        "on",
        "--temp",
        "0",
        "--metrics",
        "--sleep-idle-seconds",
        runtime_options["PADDLEOCR_LLAMA_SLEEP_IDLE_SECONDS"],
    )


def build_paddle_llama_runtime_contract(
    *,
    manifest_bytes: bytes,
    manifest_sha256: str,
    observed_file_bytes: Mapping[str, int],
    volume_name: str,
    llama_image_ref: str,
    llama_image_id: str,
    layout_image_ref: str,
    layout_image_id: str,
    compose_file: Path,
    pipeline_config_file: Path,
    environment: Mapping[str, Any] | None = None,
) -> PaddleLlamaRuntimeContract:
    safe_volume = validate_paddle_llama_volume_name(volume_name)
    normalized_manifest_sha = str(manifest_sha256 or "").strip().lower()
    if not _SHA256.fullmatch(normalized_manifest_sha):
        raise PaddleLlamaRuntimeContractError(
            "Invalid PaddleOCR llama.cpp ready manifest SHA-256."
        )
    actual_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_manifest_sha != normalized_manifest_sha:
        raise PaddleLlamaRuntimeContractError(
            "PaddleOCR llama.cpp ready manifest SHA-256 does not match "
            "the mounted manifest bytes."
        )

    try:
        manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PaddleLlamaRuntimeContractError(
            "Unable to parse the PaddleOCR llama.cpp ready manifest."
        ) from exc
    if not isinstance(manifest, dict):
        raise PaddleLlamaRuntimeContractError(
            "PaddleOCR llama.cpp ready manifest must be an object."
        )

    required_header = {
        "schema_version": PADDLE_LLAMA_RUNTIME_MANIFEST_SCHEMA_VERSION,
        "runtime": "PaddleOCR-VL-llama.cpp",
        "preparation_version": PADDLE_LLAMA_RUNTIME_PREPARATION_VERSION,
        "volume_name": safe_volume,
        "ready": True,
        "source_image_ref": llama_image_ref,
        "source_image_id": llama_image_id,
    }
    for key, expected in required_header.items():
        if manifest.get(key) != expected:
            raise PaddleLlamaRuntimeContractError(
                f"PaddleOCR llama.cpp manifest field {key!r} does not "
                f"match the runtime contract."
            )
    smoke_test = manifest.get("smoke_test")
    if not isinstance(smoke_test, dict) or smoke_test.get("passed") is not True:
        raise PaddleLlamaRuntimeContractError(
            "PaddleOCR llama.cpp manifest does not contain a passed smoke test."
        )

    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != len(
        PADDLE_LLAMA_MODEL_SPECS
    ):
        raise PaddleLlamaRuntimeContractError(
            "PaddleOCR llama.cpp manifest file registry is incomplete."
        )
    entries = {
        str(entry.get("name", "")): entry
        for entry in files
        if isinstance(entry, dict)
    }
    if set(entries) != set(PADDLE_LLAMA_MODEL_SPECS):
        raise PaddleLlamaRuntimeContractError(
            "PaddleOCR llama.cpp manifest file registry does not match "
            "the product registry."
        )
    for name, spec in PADDLE_LLAMA_MODEL_SPECS.items():
        entry = entries[name]
        if (
            int(entry.get("bytes", -1)) != int(spec["bytes"])
            or str(entry.get("sha256", "")).lower() != str(spec["sha256"])
            or str(entry.get("role", "")) != str(spec["role"])
            or int(observed_file_bytes.get(name, -1)) != int(spec["bytes"])
        ):
            raise PaddleLlamaRuntimeContractError(
                f"PaddleOCR llama.cpp model contract mismatch: {name}"
            )

    runtime_options = resolve_paddle_llama_runtime_options(environment)
    command = build_paddle_llama_server_command(runtime_options)
    compose_sha256 = hashlib.sha256(compose_file.read_bytes()).hexdigest()
    pipeline_sha256 = hashlib.sha256(
        pipeline_config_file.read_bytes()
    ).hexdigest()
    command_sha256 = _canonical_sha256(command)
    fingerprint = _canonical_sha256(
        {
            "contract_schema_version": 1,
            "volume_name": safe_volume,
            "ready_manifest_name": DEFAULT_PADDLE_LLAMA_READY_MANIFEST,
            "ready_manifest_sha256": normalized_manifest_sha,
            "llama_image_ref": llama_image_ref,
            "llama_image_id": llama_image_id,
            "layout_image_ref": layout_image_ref,
            "layout_image_id": layout_image_id,
            "compose_file_sha256": compose_sha256,
            "pipeline_config_sha256": pipeline_sha256,
            "command_sha256": command_sha256,
            "files": PADDLE_LLAMA_MODEL_SPECS,
        }
    )
    return PaddleLlamaRuntimeContract(
        volume_name=safe_volume,
        ready_manifest_name=DEFAULT_PADDLE_LLAMA_READY_MANIFEST,
        ready_manifest_sha256=normalized_manifest_sha,
        preparation_version=PADDLE_LLAMA_RUNTIME_PREPARATION_VERSION,
        llama_image_ref=llama_image_ref,
        llama_image_id=llama_image_id,
        layout_image_ref=layout_image_ref,
        layout_image_id=layout_image_id,
        compose_file_sha256=compose_sha256,
        pipeline_config_sha256=pipeline_sha256,
        command_sha256=command_sha256,
        fingerprint=fingerprint,
        command=command,
        runtime_options=runtime_options,
    )
