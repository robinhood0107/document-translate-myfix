from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


GEMMA_RUNTIME_MANIFEST_SCHEMA_VERSION = 1
GEMMA_RUNTIME_PREPARATION_VERSION = 1
DEFAULT_GEMMA_MODEL_VOLUME = "comic-translate-gemma-models-v1"
DEFAULT_GEMMA_READY_MANIFEST = ".comic-translate-gemma-ready-v1.json"
DEFAULT_GEMMA_LLAMA_CPP_IMAGE = (
    "ghcr.io/ggml-org/llama.cpp@sha256:"
    "22e0e3bfe967af4fd1df6a918022abbfd4e72e4d40a4769e616a4176790acbcb"
)
DEFAULT_GEMMA_LLAMA_CPP_PULL_POLICY = "missing"

GEMMA_RUNTIME_FINGERPRINT_LABEL = "comic-translate.runtime-fingerprint"
GEMMA_RUNTIME_KIND_LABEL = "comic-translate.runtime"
GEMMA_RUNTIME_VOLUME_LABEL = "comic-translate.model-volume"
GEMMA_RUNTIME_MANIFEST_SHA_LABEL = "comic-translate.ready-manifest-sha256"
GEMMA_RUNTIME_MODEL_SHA_LABEL = "comic-translate.model-sha256"
GEMMA_RUNTIME_PREPARATION_LABEL = "comic-translate.preparation-version"

DEFAULT_GEMMA_RUNTIME_OPTIONS: dict[str, str] = {
    "LLAMA_CTX_SIZE": "4096",
    "LLAMA_N_PARALLEL": "1",
    "LLAMA_THREADS": "10",
    "LLAMA_N_GPU_LAYERS": "23",
    "LLAMA_CACHE_TYPE_K": "f16",
    "LLAMA_CACHE_TYPE_V": "f16",
    "LLAMA_CACHE_RAM_MIB": "0",
    "LLAMA_SPEC_TYPE": "none",
    "LLAMA_SPEC_DRAFT_N_MAX": "8",
}

GEMMA_MODEL_SPECS: dict[str, dict[str, Any]] = {
    "gemma-4-26B-IQ4_NL.gguf": {
        "bytes": 14_585_439_872,
        "sha256": "768a89b94209243b333b2e074b928fe51ea208ebdad6424a510bd73e5cb4d0b8",
        "role": "legacy-rollback",
    },
    "Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ4_XS.gguf": {
        "bytes": 13_917_726_048,
        "sha256": "61b277f4dde555fc6c04c9024a9580ef8c83f2f19504f3989a15f95684257426",
        "role": "product-candidate",
    },
}
DEFAULT_GEMMA_PREPARATION_RUNTIME_CONFIGURATION: dict[str, Any] = {
    "context_size": 4096,
    "parallel": 1,
    "threads": 10,
    "gpu_layers": 23,
    "cache_type_k": "f16",
    "cache_type_v": "f16",
    "cache_ram_mib": 0,
    "speculative_type": "none",
    "speculative_draft_max": 8,
}
DEFAULT_GEMMA_PREPARED_MODEL = "gemma-4-26B-IQ4_NL.gguf"
DEFAULT_GEMMA_SMOKE_MODEL = (
    "Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ4_XS.gguf"
)

_SAFE_DOCKER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SAFE_MODEL_NAME = re.compile(r"^[^/\\\x00-\x1f]+\.gguf$", re.IGNORECASE)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class GemmaRuntimeContractError(ValueError):
    pass


@dataclass(frozen=True)
class GemmaRuntimeContract:
    model_name: str
    model_bytes: int
    model_sha256: str
    volume_name: str
    ready_manifest_name: str
    ready_manifest_sha256: str
    preparation_version: int
    image_ref: str
    image_id: str
    compose_file_sha256: str
    command_sha256: str
    fingerprint: str
    command: tuple[str, ...]
    runtime_options: Mapping[str, str]

    def compose_environment(self) -> dict[str, str]:
        values = dict(self.runtime_options)
        values.update(
            {
                "LLAMA_CPP_IMAGE": self.image_ref,
                "LLAMA_CPP_PULL_POLICY": DEFAULT_GEMMA_LLAMA_CPP_PULL_POLICY,
                "LLAMA_MODEL_FILE": self.model_name,
                "GEMMA_MODEL_VOLUME": self.volume_name,
                "GEMMA_RUNTIME_FINGERPRINT": self.fingerprint,
                "GEMMA_READY_MANIFEST_SHA256": self.ready_manifest_sha256,
                "GEMMA_MODEL_SHA256": self.model_sha256,
                "GEMMA_PREPARATION_VERSION": str(self.preparation_version),
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


def _require_safe_volume_name(value: str) -> str:
    normalized = str(value or "").strip()
    if not _SAFE_DOCKER_NAME.fullmatch(normalized):
        raise GemmaRuntimeContractError(f"Invalid Gemma model volume name: {value!r}")
    return normalized


def _require_safe_model_name(value: str) -> str:
    normalized = Path(str(value or "").strip()).name
    if normalized != str(value or "").strip() or not _SAFE_MODEL_NAME.fullmatch(normalized):
        raise GemmaRuntimeContractError(f"Invalid Gemma model filename: {value!r}")
    return normalized


def validate_gemma_volume_name(value: str) -> str:
    return _require_safe_volume_name(value)


def validate_gemma_model_name(value: str) -> str:
    return _require_safe_model_name(value)


def resolve_gemma_runtime_options(
    environment: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    source = environment or {}
    values: dict[str, str] = {}
    for key, default in DEFAULT_GEMMA_RUNTIME_OPTIONS.items():
        raw_value = source.get(key, default)
        normalized = str(raw_value).strip() if raw_value is not None else ""
        values[key] = normalized or default

    numeric_ranges = {
        "LLAMA_CTX_SIZE": (1024, 32768),
        "LLAMA_N_PARALLEL": (1, 4),
        "LLAMA_THREADS": (1, 64),
        "LLAMA_N_GPU_LAYERS": (0, 99),
    }
    for key, (minimum, maximum) in numeric_ranges.items():
        try:
            parsed = int(values[key])
        except ValueError as exc:
            raise GemmaRuntimeContractError(f"{key} must be an integer.") from exc
        if parsed < minimum or parsed > maximum:
            raise GemmaRuntimeContractError(
                f"{key} must be between {minimum} and {maximum}: {parsed}"
            )
        values[key] = str(parsed)

    for key in ("LLAMA_CACHE_TYPE_K", "LLAMA_CACHE_TYPE_V"):
        normalized = values[key].lower()
        if normalized not in {"f16", "q8_0"}:
            raise GemmaRuntimeContractError(
                f"{key} must be one of f16 or q8_0: {values[key]!r}"
            )
        values[key] = normalized

    try:
        cache_ram = int(values["LLAMA_CACHE_RAM_MIB"])
    except ValueError as exc:
        raise GemmaRuntimeContractError("LLAMA_CACHE_RAM_MIB must be an integer.") from exc
    if cache_ram not in {0, 256}:
        raise GemmaRuntimeContractError(
            f"LLAMA_CACHE_RAM_MIB must be 0 or 256: {cache_ram}"
        )
    values["LLAMA_CACHE_RAM_MIB"] = str(cache_ram)

    spec_type = values["LLAMA_SPEC_TYPE"].lower()
    if spec_type not in {"none", "ngram-mod"}:
        raise GemmaRuntimeContractError(
            f"LLAMA_SPEC_TYPE must be none or ngram-mod: {values['LLAMA_SPEC_TYPE']!r}"
        )
    values["LLAMA_SPEC_TYPE"] = spec_type

    try:
        draft_max = int(values["LLAMA_SPEC_DRAFT_N_MAX"])
    except ValueError as exc:
        raise GemmaRuntimeContractError(
            "LLAMA_SPEC_DRAFT_N_MAX must be an integer."
        ) from exc
    if draft_max not in {2, 4, 8}:
        raise GemmaRuntimeContractError(
            f"LLAMA_SPEC_DRAFT_N_MAX must be 2, 4, or 8: {draft_max}"
        )
    values["LLAMA_SPEC_DRAFT_N_MAX"] = str(draft_max)
    return values


def build_gemma_server_command(
    model_name: str,
    runtime_options: Mapping[str, str],
) -> tuple[str, ...]:
    model = _require_safe_model_name(model_name)
    return (
        "-m",
        f"/models/{model}",
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "-c",
        runtime_options["LLAMA_CTX_SIZE"],
        "-np",
        runtime_options["LLAMA_N_PARALLEL"],
        "-t",
        runtime_options["LLAMA_THREADS"],
        "--n-gpu-layers",
        runtime_options["LLAMA_N_GPU_LAYERS"],
        "--fit",
        "off",
        "-fa",
        "on",
        "-ctk",
        runtime_options["LLAMA_CACHE_TYPE_K"],
        "-ctv",
        runtime_options["LLAMA_CACHE_TYPE_V"],
        "--kv-offload",
        "--swa-full",
        "--jinja",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
        "--reasoning-format",
        "none",
        "--metrics",
        "--perf",
        "--cache-ram",
        runtime_options["LLAMA_CACHE_RAM_MIB"],
        "--spec-type",
        runtime_options["LLAMA_SPEC_TYPE"],
        "--spec-draft-n-max",
        runtime_options["LLAMA_SPEC_DRAFT_N_MAX"],
    )


def _validate_manifest(
    manifest_bytes: bytes,
    *,
    volume_name: str,
    model_name: str,
    image_ref: str,
    image_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        payload = json.loads(manifest_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GemmaRuntimeContractError("Gemma ready manifest is not valid UTF-8 JSON.") from exc
    if not isinstance(payload, dict):
        raise GemmaRuntimeContractError("Gemma ready manifest must be a JSON object.")

    expected_header = {
        "schema_version": GEMMA_RUNTIME_MANIFEST_SCHEMA_VERSION,
        "runtime": "Gemma",
        "preparation_version": GEMMA_RUNTIME_PREPARATION_VERSION,
        "volume_name": volume_name,
        "source_image_ref": image_ref,
        "source_image_digest": image_id,
        "source_image_id": image_id,
    }
    for key, expected in expected_header.items():
        actual = payload.get(key)
        if actual != expected:
            raise GemmaRuntimeContractError(
                f"Gemma ready manifest mismatch for {key}: expected={expected!r}, actual={actual!r}"
            )
    if payload.get("ready") is not True:
        raise GemmaRuntimeContractError("Gemma ready manifest is not marked ready.")
    if payload.get("default_model") != DEFAULT_GEMMA_PREPARED_MODEL:
        raise GemmaRuntimeContractError(
            "Gemma ready manifest default model does not match the product contract."
        )
    if (
        payload.get("runtime_configuration")
        != DEFAULT_GEMMA_PREPARATION_RUNTIME_CONFIGURATION
    ):
        raise GemmaRuntimeContractError(
            "Gemma ready manifest runtime configuration does not match "
            "the preparation contract."
        )
    smoke_test = payload.get("smoke_test")
    if (
        not isinstance(smoke_test, dict)
        or smoke_test.get("passed") is not True
        or smoke_test.get("model") != DEFAULT_GEMMA_SMOKE_MODEL
    ):
        raise GemmaRuntimeContractError(
            "Gemma ready manifest does not contain a successful smoke test."
        )

    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise GemmaRuntimeContractError("Gemma ready manifest contains no model files.")
    entries: dict[str, dict[str, Any]] = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise GemmaRuntimeContractError("Gemma ready manifest file entries must be objects.")
        name = _require_safe_model_name(str(entry.get("name", "")))
        if name in entries:
            raise GemmaRuntimeContractError(f"Duplicate Gemma ready manifest entry: {name}")
        entries[name] = entry

    if set(entries) != set(GEMMA_MODEL_SPECS):
        raise GemmaRuntimeContractError(
            "Gemma ready manifest model registry does not match the product registry."
        )
    for registered_name, expected_spec in GEMMA_MODEL_SPECS.items():
        entry = entries[registered_name]
        try:
            actual_bytes = int(entry.get("bytes"))
        except (TypeError, ValueError) as exc:
            raise GemmaRuntimeContractError(
                f"Invalid byte count in Gemma ready manifest for {registered_name}."
            ) from exc
        actual_sha = str(entry.get("sha256", "")).strip().lower()
        actual_role = str(entry.get("role", "")).strip()
        if actual_bytes != int(expected_spec["bytes"]):
            raise GemmaRuntimeContractError(
                f"Gemma model size mismatch in ready manifest for {registered_name}: "
                f"expected={expected_spec['bytes']}, actual={actual_bytes}"
            )
        if (
            not _SHA256.fullmatch(actual_sha)
            or actual_sha != expected_spec["sha256"]
        ):
            raise GemmaRuntimeContractError(
                f"Gemma model SHA-256 mismatch in ready manifest for {registered_name}."
            )
        if actual_role != expected_spec["role"]:
            raise GemmaRuntimeContractError(
                f"Gemma model role mismatch in ready manifest for {registered_name}: "
                f"expected={expected_spec['role']!r}, actual={actual_role!r}"
            )

    model = _require_safe_model_name(model_name)
    expected_spec = GEMMA_MODEL_SPECS.get(model)
    if expected_spec is None:
        raise GemmaRuntimeContractError(
            f"Configured Gemma model is not in the prepared product registry: {model}"
        )
    selected = entries.get(model)
    if selected is None:
        raise GemmaRuntimeContractError(
            f"Configured Gemma model is missing from the ready manifest: {model}"
        )
    return payload, selected


def build_gemma_runtime_contract(
    *,
    manifest_bytes: bytes,
    manifest_sha256: str,
    observed_model_bytes: int,
    volume_name: str,
    model_name: str,
    image_ref: str,
    image_id: str,
    compose_file: str | Path,
    environment: Mapping[str, Any] | None = None,
) -> GemmaRuntimeContract:
    volume = _require_safe_volume_name(volume_name)
    model = _require_safe_model_name(model_name)
    manifest_sha = str(manifest_sha256 or "").strip().lower()
    if not _SHA256.fullmatch(manifest_sha):
        raise GemmaRuntimeContractError("Invalid Gemma ready manifest SHA-256.")
    actual_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha != actual_manifest_sha:
        raise GemmaRuntimeContractError(
            "Gemma ready manifest SHA-256 does not match the mounted manifest bytes."
        )

    _, selected = _validate_manifest(
        manifest_bytes,
        volume_name=volume,
        model_name=model,
        image_ref=image_ref,
        image_id=image_id,
    )
    model_bytes = int(selected["bytes"])
    if int(observed_model_bytes) != model_bytes:
        raise GemmaRuntimeContractError(
            f"Mounted Gemma model size mismatch for {model}: "
            f"manifest={model_bytes}, actual={observed_model_bytes}"
        )

    compose_path = Path(compose_file)
    compose_bytes = compose_path.read_bytes()
    compose_sha = hashlib.sha256(compose_bytes).hexdigest()
    runtime_options = resolve_gemma_runtime_options(environment)
    command = build_gemma_server_command(model, runtime_options)
    command_sha = _canonical_sha256(command)
    preparation_version = int(GEMMA_RUNTIME_PREPARATION_VERSION)
    fingerprint_payload = {
        "image_ref": image_ref,
        "image_id": image_id,
        "compose_file_sha256": compose_sha,
        "command_sha256": command_sha,
        "volume_name": volume,
        "ready_manifest_sha256": manifest_sha,
        "model_name": model,
        "model_sha256": str(selected["sha256"]).lower(),
        "preparation_version": preparation_version,
    }
    return GemmaRuntimeContract(
        model_name=model,
        model_bytes=model_bytes,
        model_sha256=str(selected["sha256"]).lower(),
        volume_name=volume,
        ready_manifest_name=DEFAULT_GEMMA_READY_MANIFEST,
        ready_manifest_sha256=manifest_sha,
        preparation_version=preparation_version,
        image_ref=image_ref,
        image_id=image_id,
        compose_file_sha256=compose_sha,
        command_sha256=command_sha,
        fingerprint=_canonical_sha256(fingerprint_payload),
        command=command,
        runtime_options=runtime_options,
    )


def container_contract_mismatch_reasons(
    inspection: Mapping[str, Any],
    contract: GemmaRuntimeContract,
) -> list[str]:
    reasons: list[str] = []
    config = inspection.get("Config")
    if not isinstance(config, Mapping):
        return ["missing container config"]
    labels = config.get("Labels")
    if not isinstance(labels, Mapping):
        labels = {}

    expected_labels = {
        GEMMA_RUNTIME_KIND_LABEL: "gemma",
        GEMMA_RUNTIME_FINGERPRINT_LABEL: contract.fingerprint,
        GEMMA_RUNTIME_VOLUME_LABEL: contract.volume_name,
        GEMMA_RUNTIME_MANIFEST_SHA_LABEL: contract.ready_manifest_sha256,
        GEMMA_RUNTIME_MODEL_SHA_LABEL: contract.model_sha256,
        GEMMA_RUNTIME_PREPARATION_LABEL: str(contract.preparation_version),
    }
    for key, expected in expected_labels.items():
        if str(labels.get(key, "")) != expected:
            reasons.append(f"label:{key}")

    if str(config.get("Image", "")) != contract.image_ref:
        reasons.append("image-ref")
    if str(inspection.get("Image", "")) != contract.image_id:
        reasons.append("image-id")

    actual_command = config.get("Cmd")
    if list(actual_command or []) != list(contract.command):
        reasons.append("command")

    mounts = inspection.get("Mounts")
    if not isinstance(mounts, list):
        mounts = []
    matching_mount = next(
        (
            mount
            for mount in mounts
            if isinstance(mount, Mapping)
            and str(mount.get("Destination", "")) == "/models"
        ),
        None,
    )
    if matching_mount is None:
        reasons.append("models-mount")
    else:
        if str(matching_mount.get("Type", "")) != "volume":
            reasons.append("models-mount-type")
        if str(matching_mount.get("Name", "")) != contract.volume_name:
            reasons.append("models-volume")
        if bool(matching_mount.get("RW", True)):
            reasons.append("models-mount-readonly")
    return reasons
