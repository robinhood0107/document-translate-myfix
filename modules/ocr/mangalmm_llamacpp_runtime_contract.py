from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


MANGALMM_RUNTIME_MANIFEST_SCHEMA_VERSION = 1
MANGALMM_RUNTIME_PREPARATION_VERSION = 2
DEFAULT_MANGALMM_MODEL_VOLUME = "comic-translate-mangalmm-models-v2"
DEFAULT_MANGALMM_READY_MANIFEST = ".comic-translate-mangalmm-ready-v2.json"
DEFAULT_MANGALMM_LLAMA_CPP_IMAGE = (
    "ghcr.io/ggml-org/llama.cpp@sha256:"
    "22e0e3bfe967af4fd1df6a918022abbfd4e72e4d40a4769e616a4176790acbcb"
)

MANGALMM_MODEL_NAME = "MangaLMM.Q8_0.gguf"
MANGALMM_MMPROJ_NAME = "MangaLMM.mmproj-Q8_0.gguf"
MANGALMM_MODEL_ALIAS = "MangaLMM"
MANGALMM_MODEL_SPECS: dict[str, dict[str, Any]] = {
    MANGALMM_MODEL_NAME: {
        "bytes": 8_098_524_160,
        "sha256": "55e42d513ee22ab1a301b5fa8f04a2812b69d6b351e7d34efdff2b8d8e8fa01a",
        "role": "vlm",
    },
    MANGALMM_MMPROJ_NAME: {
        "bytes": 853_119_744,
        "sha256": "24f43da26996b54bf5764177a954e49b24ec38a53de34d8231764747b0dcd8d7",
        "role": "vision-projector",
    },
}

DEFAULT_MANGALMM_RUNTIME_OPTIONS: dict[str, str] = {
    "MANGALMM_LLAMA_CTX_SIZE": "8192",
    "MANGALMM_LLAMA_PARALLEL": "1",
    "MANGALMM_LLAMA_THREADS": "12",
    "MANGALMM_LLAMA_BATCH_SIZE": "2048",
    "MANGALMM_LLAMA_UBATCH_SIZE": "512",
    "MANGALMM_LLAMA_GPU_LAYERS": "all",
}

MANGALMM_RUNTIME_FINGERPRINT_LABEL = (
    "com.comictranslate.mangalmm-runtime-fingerprint"
)
MANGALMM_RUNTIME_VOLUME_LABEL = "com.comictranslate.mangalmm-model-volume"
MANGALMM_RUNTIME_MANIFEST_SHA_LABEL = (
    "com.comictranslate.mangalmm-ready-manifest-sha256"
)
MANGALMM_RUNTIME_MODEL_SHA_LABEL = (
    "com.comictranslate.mangalmm-model-sha256"
)
MANGALMM_RUNTIME_MMPROJ_SHA_LABEL = (
    "com.comictranslate.mangalmm-mmproj-sha256"
)

_SAFE_DOCKER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class MangaLMMRuntimeContractError(ValueError):
    pass


@dataclass(frozen=True)
class MangaLMMRuntimeContract:
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
                "MANGALMM_LLAMA_CPP_IMAGE": self.llama_image_ref,
                "MANGALMM_MODEL_VOLUME": self.volume_name,
                "MANGALMM_MODEL_FILE": MANGALMM_MODEL_NAME,
                "MANGALMM_MMPROJ_FILE": MANGALMM_MMPROJ_NAME,
                "MANGALMM_MODEL_ALIAS": MANGALMM_MODEL_ALIAS,
                "MANGALMM_RUNTIME_FINGERPRINT": self.fingerprint,
                "MANGALMM_READY_MANIFEST_SHA256": (
                    self.ready_manifest_sha256
                ),
                "MANGALMM_MODEL_SHA256": str(
                    MANGALMM_MODEL_SPECS[MANGALMM_MODEL_NAME]["sha256"]
                ),
                "MANGALMM_MMPROJ_SHA256": str(
                    MANGALMM_MODEL_SPECS[MANGALMM_MMPROJ_NAME]["sha256"]
                ),
                "MANGALMM_PREPARATION_VERSION": str(
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


def validate_mangalmm_volume_name(value: str) -> str:
    normalized = str(value or "").strip()
    if not _SAFE_DOCKER_NAME.fullmatch(normalized):
        raise MangaLMMRuntimeContractError(
            f"Invalid MangaLMM model volume name: {value!r}"
        )
    return normalized


def resolve_mangalmm_runtime_options(
    environment: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    source = environment or {}
    values: dict[str, str] = {}
    for key, default in DEFAULT_MANGALMM_RUNTIME_OPTIONS.items():
        raw_value = source.get(key, default)
        normalized = str(raw_value).strip() if raw_value is not None else ""
        values[key] = normalized or default

    numeric_ranges = {
        "MANGALMM_LLAMA_CTX_SIZE": (4096, 16_384),
        "MANGALMM_LLAMA_PARALLEL": (1, 2),
        "MANGALMM_LLAMA_THREADS": (1, 64),
        "MANGALMM_LLAMA_BATCH_SIZE": (128, 4096),
        "MANGALMM_LLAMA_UBATCH_SIZE": (64, 2048),
    }
    for key, (minimum, maximum) in numeric_ranges.items():
        try:
            parsed = int(values[key])
        except ValueError as exc:
            raise MangaLMMRuntimeContractError(
                f"{key} must be an integer."
            ) from exc
        if parsed < minimum or parsed > maximum:
            raise MangaLMMRuntimeContractError(
                f"{key} must be between {minimum} and {maximum}: {parsed}"
            )
        values[key] = str(parsed)

    if int(values["MANGALMM_LLAMA_UBATCH_SIZE"]) > int(
        values["MANGALMM_LLAMA_BATCH_SIZE"]
    ):
        raise MangaLMMRuntimeContractError(
            "MANGALMM_LLAMA_UBATCH_SIZE may not exceed "
            "MANGALMM_LLAMA_BATCH_SIZE."
        )

    gpu_layers = values["MANGALMM_LLAMA_GPU_LAYERS"].lower()
    if gpu_layers != "all":
        try:
            parsed_gpu_layers = int(gpu_layers)
        except ValueError as exc:
            raise MangaLMMRuntimeContractError(
                "MANGALMM_LLAMA_GPU_LAYERS must be 'all' or an integer."
            ) from exc
        if parsed_gpu_layers < 0 or parsed_gpu_layers > 999:
            raise MangaLMMRuntimeContractError(
                "MANGALMM_LLAMA_GPU_LAYERS must be between 0 and 999."
            )
        gpu_layers = str(parsed_gpu_layers)
    values["MANGALMM_LLAMA_GPU_LAYERS"] = gpu_layers
    return values


def build_mangalmm_server_command(
    runtime_options: Mapping[str, str],
) -> tuple[str, ...]:
    return (
        "-m",
        f"/models/{MANGALMM_MODEL_NAME}",
        "--mmproj",
        f"/models/{MANGALMM_MMPROJ_NAME}",
        "--alias",
        MANGALMM_MODEL_ALIAS,
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "-c",
        runtime_options["MANGALMM_LLAMA_CTX_SIZE"],
        "-np",
        runtime_options["MANGALMM_LLAMA_PARALLEL"],
        "-t",
        runtime_options["MANGALMM_LLAMA_THREADS"],
        "-b",
        runtime_options["MANGALMM_LLAMA_BATCH_SIZE"],
        "-ub",
        runtime_options["MANGALMM_LLAMA_UBATCH_SIZE"],
        "--n-gpu-layers",
        runtime_options["MANGALMM_LLAMA_GPU_LAYERS"],
        "--fit",
        "off",
        "--flash-attn",
        "on",
        "--temp",
        "0",
        "--metrics",
        "--cache-ram",
        "0",
    )


def build_mangalmm_runtime_contract(
    *,
    manifest_bytes: bytes,
    manifest_sha256: str,
    observed_file_bytes: Mapping[str, int],
    volume_name: str,
    llama_image_ref: str,
    llama_image_id: str,
    compose_file: Path,
    environment: Mapping[str, Any] | None = None,
) -> MangaLMMRuntimeContract:
    safe_volume = validate_mangalmm_volume_name(volume_name)
    normalized_manifest_sha = str(manifest_sha256 or "").strip().lower()
    if not _SHA256.fullmatch(normalized_manifest_sha):
        raise MangaLMMRuntimeContractError(
            "Invalid MangaLMM ready manifest SHA-256."
        )
    if hashlib.sha256(manifest_bytes).hexdigest() != normalized_manifest_sha:
        raise MangaLMMRuntimeContractError(
            "MangaLMM ready manifest SHA-256 does not match mounted bytes."
        )
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MangaLMMRuntimeContractError(
            "Unable to parse the MangaLMM ready manifest."
        ) from exc
    if not isinstance(manifest, dict):
        raise MangaLMMRuntimeContractError(
            "MangaLMM ready manifest must be an object."
        )

    required_header = {
        "schema_version": MANGALMM_RUNTIME_MANIFEST_SCHEMA_VERSION,
        "runtime": "MangaLMM-llama.cpp",
        "preparation_version": MANGALMM_RUNTIME_PREPARATION_VERSION,
        "volume_name": safe_volume,
        "ready": True,
        "source_image_ref": llama_image_ref,
        "source_image_id": llama_image_id,
    }
    for key, expected in required_header.items():
        if manifest.get(key) != expected:
            raise MangaLMMRuntimeContractError(
                f"MangaLMM manifest field {key!r} does not match the runtime contract."
            )
    smoke_test = manifest.get("smoke_test")
    if not isinstance(smoke_test, dict) or smoke_test.get("passed") is not True:
        raise MangaLMMRuntimeContractError(
            "MangaLMM manifest does not contain a passed smoke test."
        )
    if (
        smoke_test.get("device") != "CUDA"
        or smoke_test.get("model_alias") != MANGALMM_MODEL_ALIAS
    ):
        raise MangaLMMRuntimeContractError(
            "MangaLMM smoke test must verify the CUDA device and exact model alias."
        )

    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != len(
        MANGALMM_MODEL_SPECS
    ):
        raise MangaLMMRuntimeContractError(
            "MangaLMM manifest file registry is incomplete."
        )
    entries = {
        str(entry.get("name", "")): entry
        for entry in files
        if isinstance(entry, dict)
    }
    if set(entries) != set(MANGALMM_MODEL_SPECS):
        raise MangaLMMRuntimeContractError(
            "MangaLMM manifest file registry does not match the product registry."
        )
    for name, spec in MANGALMM_MODEL_SPECS.items():
        entry = entries[name]
        if (
            int(entry.get("bytes", -1)) != int(spec["bytes"])
            or str(entry.get("sha256", "")).lower() != str(spec["sha256"])
            or str(entry.get("role", "")) != str(spec["role"])
            or int(observed_file_bytes.get(name, -1)) != int(spec["bytes"])
        ):
            raise MangaLMMRuntimeContractError(
                f"MangaLMM model contract mismatch: {name}"
            )

    runtime_options = resolve_mangalmm_runtime_options(environment)
    command = build_mangalmm_server_command(runtime_options)
    compose_sha256 = hashlib.sha256(compose_file.read_bytes()).hexdigest()
    command_sha256 = _canonical_sha256(command)
    fingerprint = _canonical_sha256(
        {
            "contract_schema_version": 1,
            "volume_name": safe_volume,
            "ready_manifest_name": DEFAULT_MANGALMM_READY_MANIFEST,
            "ready_manifest_sha256": normalized_manifest_sha,
            "llama_image_ref": llama_image_ref,
            "llama_image_id": llama_image_id,
            "compose_file_sha256": compose_sha256,
            "command_sha256": command_sha256,
            "model_registry": MANGALMM_MODEL_SPECS,
            "runtime_options": runtime_options,
        }
    )
    return MangaLMMRuntimeContract(
        volume_name=safe_volume,
        ready_manifest_name=DEFAULT_MANGALMM_READY_MANIFEST,
        ready_manifest_sha256=normalized_manifest_sha,
        preparation_version=MANGALMM_RUNTIME_PREPARATION_VERSION,
        llama_image_ref=llama_image_ref,
        llama_image_id=llama_image_id,
        compose_file_sha256=compose_sha256,
        command_sha256=command_sha256,
        fingerprint=fingerprint,
        command=command,
        runtime_options=runtime_options,
    )
