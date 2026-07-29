#!/usr/bin/env python3
"""Run the external-data llama.cpp Gemma target/speculation tournament.

Benchmark policy and raw translations deliberately live outside the product
runtime and outside Git.  This module only provides the generic protocol,
runtime contract validation, execution hooks, and report-safe summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import struct
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import requests


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.translation.llm.custom_local_gemma import (  # noqa: E402
    DEFAULT_GEMMA_PROMPT_PROFILE,
    DEFAULT_GEMMA_RESPONSE_FORMAT_MODE,
    DEFAULT_GEMMA_RESPONSE_SCHEMA_MODE,
    DEFAULT_GEMMA_THINK_BRIEFLY_PROMPT,
    DEFAULT_GEMMA_TRANSLATION_MIN_P,
    DEFAULT_GEMMA_TRANSLATION_TEMPERATURE,
    DEFAULT_GEMMA_TRANSLATION_TOP_K,
    DEFAULT_GEMMA_TRANSLATION_TOP_P,
    GEMMA_REQUEST_MODE_CONTEXTUAL_SINGLE,
    CustomLocalGemmaTranslation,
)
from modules.utils.textblock import TextBlock  # noqa: E402


PROTOCOL_VERSION = 1
INVENTORY_LOCK_VERSION = 2
RESULT_VERSION = 1
DEFAULT_IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda"
DEFAULT_IMAGE_ID = (
    "sha256:22e0e3bfe967af4fd1df6a918022abbfd4e72e4d40a4769e616a4176790acbcb"
)
DEFAULT_HELPER_IMAGE = "alpine:3.22"
DEFAULT_PORT = 18080
DEFAULT_CONTEXT_SIZE = 4096
DEFAULT_THREADS = 10
DEFAULT_CHUNK_SIZE = 6
DEFAULT_MAX_COMPLETION_TOKENS = 512
DEFAULT_TARGET_NGL = 23
DEFAULT_BOOTSTRAP_SAMPLES = 20_000
DEFAULT_BOOTSTRAP_SEED = 20260729
DEFAULT_START_TIMEOUT_SEC = 600
DEFAULT_REQUEST_TIMEOUT_SEC = 240
DEFAULT_STOP_TIMEOUT_SEC = 60
DEFAULT_MAX_IDLE_GPU_USED_MB = 2048
DEFAULT_MAX_SWAP_GROWTH_MB = 128
DEFAULT_MAX_SHARED_GPU_GROWTH_MB = 512
SAFE_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,191}\.gguf$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STRUCTURAL_STATS = (
    "gemma_truncated_count",
    "gemma_empty_content_count",
    "gemma_missing_key_count",
    "gemma_nested_value_count",
    "gemma_duplicate_key_count",
    "gemma_trailing_content_count",
    "gemma_top_level_type_error_count",
    "gemma_invalid_value_count",
    "gemma_unexpected_key_count",
    "gemma_repetition_guard_count",
    "gemma_parser_error_count",
    "gemma_schema_validation_fail_count",
    "gemma_contextual_merge_fallback_count",
    "gemma_chunk_retry_events",
    "gemma_request_retry_count",
    "gemma_http_retry_count",
    "gemma_strict_single_retry_count",
)
TOKENIZER_METADATA_KEYS = (
    "general.architecture",
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.add_eos_token",
    "tokenizer.chat_template",
)
GGUF_CONTRACT_METADATA_KEYS = (
    *TOKENIZER_METADATA_KEYS,
    "gemma4.block_count",
)
VOCABULARY_METADATA_KEYS = tuple(
    key
    for key in TOKENIZER_METADATA_KEYS
    if key not in {"general.architecture", "tokenizer.chat_template"}
)
MTP_CORE_VOCABULARY_KEYS = (
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.merges",
)
PROMETHEUS_INTEREST_RE = re.compile(
    r"(?:draft|accept|prompt|predict|token|request|kv|cache)",
    re.IGNORECASE,
)
DRAFT_ACCEPTANCE_RE = re.compile(
    r"draft acceptance\s*=\s*[0-9]+(?:\.[0-9]+)?\s*"
    r"\(\s*(?P<accepted>[0-9]+)\s+accepted\s*/\s*"
    r"(?P<generated>[0-9]+)\s+generated\),\s*"
    r"mean len\s*=\s*(?P<mean_length>[0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


class ProtocolError(ValueError):
    """Raised when a benchmark manifest or result violates the protocol."""


@dataclass(frozen=True)
class VolumeSpec:
    id: str
    name: str
    managed: bool


@dataclass(frozen=True)
class ArtifactSpec:
    id: str
    filename: str
    volume_filename: str
    local_path: Path
    volume_id: str
    expected_size: int | None
    expected_sha256: str | None


@dataclass(frozen=True)
class TargetSpec(ArtifactSpec):
    baseline: bool
    mtp_draft_id: str | None
    initial_ngl: int


@dataclass(frozen=True)
class DraftSpec(ArtifactSpec):
    allowed_target_ids: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkManifest:
    path: Path
    image: str
    expected_image_id: str
    helper_image: str
    volumes: Mapping[str, VolumeSpec]
    targets: Mapping[str, TargetSpec]
    drafts: Mapping[str, DraftSpec]
    max_idle_gpu_used_mb: int
    max_swap_growth_mb: int
    max_shared_gpu_growth_mb: int

    @property
    def baseline(self) -> TargetSpec:
        baselines = [target for target in self.targets.values() if target.baseline]
        if len(baselines) != 1:
            raise ProtocolError("Exactly one target must be marked baseline")
        return baselines[0]


@dataclass(frozen=True)
class Profile:
    id: str
    target_id: str
    speculation: str
    draft_n: int
    target_ngl: int
    draft_ngl: str
    draft_id: str | None = None


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    lock_path = path.with_name(path.name + ".write-lock")
    lock_fd: int | None = None
    lock_acquired = False
    try:
        lock_fd = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        lock_acquired = True
        os.write(lock_fd, f"{os.getpid()}\n".encode("ascii"))
        os.close(lock_fd)
        lock_fd = None
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite immutable JSON: {path}")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        try:
            os.link(temporary, path)
            temporary.unlink()
        except OSError:
            if path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite immutable JSON: {path}"
                )
            os.replace(temporary, path)
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if temporary.exists():
            temporary.unlink()
        if lock_acquired and lock_path.exists():
            lock_path.unlink()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_external_path(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return resolved
    raise ProtocolError(f"{label} must be outside the Git repository")


def _require_identifier(value: Any, *, label: str) -> str:
    normalized = str(value or "").strip()
    if not SAFE_IDENTIFIER_RE.fullmatch(normalized):
        raise ProtocolError(f"{label} is not a safe identifier")
    return normalized


def _require_filename(value: Any, *, label: str) -> str:
    normalized = str(value or "").strip()
    if not SAFE_FILENAME_RE.fullmatch(normalized):
        raise ProtocolError(f"{label} must be a plain .gguf filename")
    return normalized


def _optional_sha256(value: Any, *, label: str) -> str | None:
    normalized = str(value or "").strip().casefold()
    if not normalized:
        return None
    if not SHA256_RE.fullmatch(normalized):
        raise ProtocolError(f"{label} must be a 64-character SHA-256")
    return normalized


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected a JSON object: {path.name}")
    return payload


def load_manifest(path: Path) -> BenchmarkManifest:
    manifest_path = _require_external_path(path, label="model manifest")
    payload = _load_object(manifest_path)
    if int(payload.get("protocol_version", 0) or 0) != PROTOCOL_VERSION:
        raise ProtocolError("Unsupported model manifest protocol_version")

    image_payload = payload.get("image")
    if not isinstance(image_payload, Mapping):
        raise ProtocolError("image must be an object")
    image = str(image_payload.get("reference") or "").strip()
    expected_image_id = str(image_payload.get("expected_id") or "").strip()
    if not image:
        raise ProtocolError("image.reference is required")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_image_id.casefold()):
        raise ProtocolError("image.expected_id must be a Docker sha256 image ID")

    volumes_payload = payload.get("volumes")
    if not isinstance(volumes_payload, list) or not volumes_payload:
        raise ProtocolError("volumes must be a non-empty list")
    volumes: dict[str, VolumeSpec] = {}
    for index, item in enumerate(volumes_payload):
        if not isinstance(item, Mapping):
            raise ProtocolError(f"volumes[{index}] must be an object")
        volume_id = _require_identifier(item.get("id"), label=f"volumes[{index}].id")
        name = _require_identifier(item.get("name"), label=f"volumes[{index}].name")
        if volume_id in volumes:
            raise ProtocolError(f"Duplicate volume id: {volume_id}")
        volumes[volume_id] = VolumeSpec(
            id=volume_id,
            name=name,
            managed=bool(item.get("managed", False)),
        )

    def artifact_fields(
        item: Mapping[str, Any],
        *,
        label: str,
    ) -> tuple[str, str, str, Path, str, int | None, str | None]:
        artifact_id = _require_identifier(item.get("id"), label=f"{label}.id")
        filename = _require_filename(item.get("filename"), label=f"{label}.filename")
        volume_filename = _require_filename(
            item.get("volume_filename") or filename,
            label=f"{label}.volume_filename",
        )
        raw_local_path = Path(str(item.get("local_path") or "")).expanduser()
        if not raw_local_path.is_absolute():
            raise ProtocolError(f"{label}.local_path must be absolute")
        local_path = raw_local_path.resolve()
        volume_id = _require_identifier(
            item.get("volume_id"),
            label=f"{label}.volume_id",
        )
        if volume_id not in volumes:
            raise ProtocolError(f"{label} references an unknown volume")
        raw_size = item.get("expected_size")
        expected_size = int(raw_size) if raw_size not in (None, "") else None
        if expected_size is not None and expected_size <= 0:
            raise ProtocolError(f"{label}.expected_size must be positive")
        expected_sha = _optional_sha256(
            item.get("expected_sha256"),
            label=f"{label}.expected_sha256",
        )
        return (
            artifact_id,
            filename,
            volume_filename,
            local_path,
            volume_id,
            expected_size,
            expected_sha,
        )

    targets_payload = payload.get("targets")
    if not isinstance(targets_payload, list) or not targets_payload:
        raise ProtocolError("targets must be a non-empty list")
    targets: dict[str, TargetSpec] = {}
    for index, item in enumerate(targets_payload):
        if not isinstance(item, Mapping):
            raise ProtocolError(f"targets[{index}] must be an object")
        fields = artifact_fields(item, label=f"targets[{index}]")
        target = TargetSpec(
            *fields,
            baseline=bool(item.get("baseline", False)),
            mtp_draft_id=(
                _require_identifier(
                    item.get("mtp_draft_id"),
                    label=f"targets[{index}].mtp_draft_id",
                )
                if item.get("mtp_draft_id")
                else None
            ),
            initial_ngl=int(item.get("initial_ngl", DEFAULT_TARGET_NGL)),
        )
        if target.id in targets:
            raise ProtocolError(f"Duplicate target id: {target.id}")
        if target.initial_ngl < 0:
            raise ProtocolError("initial_ngl must be non-negative")
        targets[target.id] = target

    drafts_payload = payload.get("drafts")
    if not isinstance(drafts_payload, list):
        raise ProtocolError("drafts must be a list")
    drafts: dict[str, DraftSpec] = {}
    for index, item in enumerate(drafts_payload):
        if not isinstance(item, Mapping):
            raise ProtocolError(f"drafts[{index}] must be an object")
        fields = artifact_fields(item, label=f"drafts[{index}]")
        allowed_payload = item.get("allowed_target_ids")
        if not isinstance(allowed_payload, list) or not allowed_payload:
            raise ProtocolError("Every draft requires allowed_target_ids")
        allowed = tuple(
            _require_identifier(value, label="allowed_target_ids item")
            for value in allowed_payload
        )
        if len(set(allowed)) != len(allowed):
            raise ProtocolError("allowed_target_ids contains duplicates")
        if any(target_id not in targets for target_id in allowed):
            raise ProtocolError("A draft references an unknown target")
        draft = DraftSpec(*fields, allowed_target_ids=allowed)
        if draft.id in drafts or draft.id in targets:
            raise ProtocolError(f"Duplicate artifact id: {draft.id}")
        drafts[draft.id] = draft

    volume_destinations: dict[tuple[str, str], str] = {}
    for artifact in [*targets.values(), *drafts.values()]:
        destination = (artifact.volume_id, artifact.volume_filename)
        previous = volume_destinations.get(destination)
        if previous is not None:
            raise ProtocolError(
                "Duplicate volume destination: "
                f"{previous} and {artifact.id} both map to "
                f"{artifact.volume_id}/{artifact.volume_filename}"
            )
        volume_destinations[destination] = artifact.id

    baselines = [target.id for target in targets.values() if target.baseline]
    if len(baselines) != 1:
        raise ProtocolError("Exactly one target must be marked baseline")
    baseline_target = targets[baselines[0]]
    if volumes[baseline_target.volume_id].managed:
        raise ProtocolError("The product baseline volume must never be managed")
    for target in targets.values():
        if target.mtp_draft_id is None:
            continue
        draft = drafts.get(target.mtp_draft_id)
        if draft is None:
            raise ProtocolError(f"{target.id} references an unknown MTP draft")
        if target.id not in draft.allowed_target_ids:
            raise ProtocolError(
                f"Invalid MTP pairing: {target.id} -> {target.mtp_draft_id}"
            )

    cache_contract = payload.get("cache_contract")
    expected_cache_contract = {
        "persistent_translation_cache": False,
        "exact_translation_memory": False,
        "project_checkpoint": False,
        "llama_prompt_cache_ram_mib": 0,
    }
    if cache_contract != expected_cache_contract:
        raise ProtocolError(
            "Cold benchmark cache_contract must disable every product cache "
            "and set llama_prompt_cache_ram_mib to 0"
        )

    preflight = payload.get("preflight") or {}
    if not isinstance(preflight, Mapping):
        raise ProtocolError("preflight must be an object")
    return BenchmarkManifest(
        path=manifest_path,
        image=image,
        expected_image_id=expected_image_id.casefold(),
        helper_image=str(payload.get("helper_image") or DEFAULT_HELPER_IMAGE),
        volumes=volumes,
        targets=targets,
        drafts=drafts,
        max_idle_gpu_used_mb=int(
            preflight.get("max_idle_gpu_used_mb", DEFAULT_MAX_IDLE_GPU_USED_MB)
        ),
        max_swap_growth_mb=int(
            preflight.get("max_swap_growth_mb", DEFAULT_MAX_SWAP_GROWTH_MB)
        ),
        max_shared_gpu_growth_mb=int(
            preflight.get(
                "max_shared_gpu_growth_mb",
                DEFAULT_MAX_SHARED_GPU_GROWTH_MB,
            )
        ),
    )


def enumerate_profiles(manifest: BenchmarkManifest) -> list[Profile]:
    profiles: list[Profile] = []
    for target in manifest.targets.values():
        profiles.append(
            Profile(
                id=f"{target.id}__none",
                target_id=target.id,
                speculation="none",
                draft_n=0,
                target_ngl=target.initial_ngl,
                draft_ngl="",
            )
        )
        for draft_n in (2, 4, 8):
            profiles.append(
                Profile(
                    id=f"{target.id}__ngram-{draft_n}",
                    target_id=target.id,
                    speculation="ngram",
                    draft_n=draft_n,
                    target_ngl=target.initial_ngl,
                    draft_ngl="",
                )
            )
        if target.mtp_draft_id:
            for draft_n in (2, 4, 8):
                profiles.append(
                    Profile(
                        id=f"{target.id}__mtp-{draft_n}",
                        target_id=target.id,
                        speculation="mtp",
                        draft_n=draft_n,
                        target_ngl=target.initial_ngl,
                        draft_ngl="all",
                        draft_id=target.mtp_draft_id,
                    )
                )
    return profiles


def find_profile(
    manifest: BenchmarkManifest,
    profile_id: str,
    *,
    target_ngl: int | None = None,
    draft_ngl: str | None = None,
) -> Profile:
    matches = [
        profile for profile in enumerate_profiles(manifest) if profile.id == profile_id
    ]
    if len(matches) != 1:
        raise ProtocolError(f"Unknown profile: {profile_id}")
    profile = matches[0]
    if target_ngl is not None:
        if target_ngl < 0:
            raise ProtocolError("target_ngl must be non-negative")
        profile = replace(profile, target_ngl=int(target_ngl))
    if draft_ngl is not None:
        normalized = str(draft_ngl).strip().casefold()
        if normalized != "all" and not normalized.isdigit():
            raise ProtocolError("draft_ngl must be 'all' or a non-negative integer")
        profile = replace(profile, draft_ngl=normalized)
    return profile


def _read_exact(stream: Any, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ProtocolError("Unexpected end of GGUF metadata")
    return data


def _read_u32(stream: Any) -> int:
    return struct.unpack("<I", _read_exact(stream, 4))[0]


def _read_u64(stream: Any) -> int:
    return struct.unpack("<Q", _read_exact(stream, 8))[0]


def _read_gguf_string(stream: Any) -> bytes:
    length = _read_u64(stream)
    if length > 512 * 1024 * 1024:
        raise ProtocolError("Unreasonable GGUF metadata string length")
    return _read_exact(stream, int(length))


_GGUF_SCALAR_FORMATS: dict[int, str] = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


def _consume_gguf_value(
    stream: Any,
    value_type: int,
    *,
    digest: hashlib._Hash | None,
) -> Any:
    if value_type in _GGUF_SCALAR_FORMATS:
        fmt = _GGUF_SCALAR_FORMATS[value_type]
        size = struct.calcsize(fmt)
        raw = _read_exact(stream, size)
        if digest is not None:
            digest.update(struct.pack("<I", value_type))
            digest.update(raw)
        return struct.unpack(fmt, raw)[0]
    if value_type == 8:
        raw = _read_gguf_string(stream)
        if digest is not None:
            digest.update(struct.pack("<I", value_type))
            digest.update(struct.pack("<Q", len(raw)))
            digest.update(raw)
        return raw.decode("utf-8", errors="replace")
    if value_type == 9:
        element_type = _read_u32(stream)
        count = _read_u64(stream)
        if count > 100_000_000:
            raise ProtocolError("Unreasonable GGUF metadata array length")
        if digest is not None:
            digest.update(struct.pack("<I", value_type))
            digest.update(struct.pack("<I", element_type))
            digest.update(struct.pack("<Q", count))
        for _ in range(int(count)):
            _consume_gguf_value(stream, element_type, digest=digest)
        return {"element_type": element_type, "count": int(count)}
    raise ProtocolError(f"Unsupported GGUF metadata value type: {value_type}")


def gguf_metadata_contract(path: Path) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    selected_digests: dict[str, str] = {}
    with path.open("rb") as stream:
        if _read_exact(stream, 4) != b"GGUF":
            raise ProtocolError(f"Not a GGUF file: {path.name}")
        version = _read_u32(stream)
        if version not in {2, 3}:
            raise ProtocolError(f"Unsupported GGUF version: {version}")
        tensor_count = _read_u64(stream)
        metadata_count = _read_u64(stream)
        if metadata_count > 1_000_000:
            raise ProtocolError("Unreasonable GGUF metadata count")
        for _ in range(int(metadata_count)):
            key = _read_gguf_string(stream).decode("utf-8", errors="replace")
            value_type = _read_u32(stream)
            capture = key in GGUF_CONTRACT_METADATA_KEYS
            digest = hashlib.sha256() if capture else None
            value = _consume_gguf_value(stream, value_type, digest=digest)
            if capture:
                selected_digests[key] = digest.hexdigest() if digest else ""
                if not isinstance(value, dict):
                    selected[key] = value
    tokenizer_fingerprint = _canonical_sha256(
        {
            key: selected_digests.get(key, "")
            for key in TOKENIZER_METADATA_KEYS
            if key.startswith("tokenizer.")
        }
    )
    return {
        "gguf_version": version,
        "tensor_count": int(tensor_count),
        "metadata_count": int(metadata_count),
        "selected": selected,
        "selected_value_sha256": selected_digests,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "architecture": str(selected.get("general.architecture") or ""),
    }


def mtp_metadata_compatibility(
    target_contract: Mapping[str, Any],
    draft_contract: Mapping[str, Any],
) -> dict[str, Any]:
    target_digests = target_contract.get("selected_value_sha256") or {}
    draft_digests = draft_contract.get("selected_value_sha256") or {}
    if not isinstance(target_digests, Mapping) or not isinstance(
        draft_digests,
        Mapping,
    ):
        raise ProtocolError("GGUF tokenizer metadata digest contract is invalid")
    shared = sorted(
        key
        for key in VOCABULARY_METADATA_KEYS
        if key in target_digests and key in draft_digests
    )
    core_shared = [
        key
        for key in MTP_CORE_VOCABULARY_KEYS
        if key in target_digests and key in draft_digests
    ]
    core_mismatched = [
        key
        for key in core_shared
        if str(target_digests[key]) != str(draft_digests[key])
    ]
    ancillary_mismatched = [
        key
        for key in shared
        if key not in MTP_CORE_VOCABULARY_KEYS
        and str(target_digests[key]) != str(draft_digests[key])
    ]
    missing_core_in_draft = sorted(
        key
        for key in MTP_CORE_VOCABULARY_KEYS
        if key in target_digests and key not in draft_digests
    )
    model_key = "tokenizer.ggml.model"
    model_metadata_present = (
        model_key in target_digests and model_key in draft_digests
    )
    compatible = model_metadata_present and not core_mismatched
    return {
        "metadata_compatible": compatible,
        "coverage": "full" if not missing_core_in_draft else "partial",
        "shared_keys": shared,
        "core_mismatched_keys": core_mismatched,
        "ancillary_mismatched_keys": ancillary_mismatched,
        "missing_core_in_draft": missing_core_in_draft,
        "load_generation_smoke_required": True,
    }


def build_inventory(
    manifest: BenchmarkManifest,
    *,
    output_path: Path,
) -> dict[str, Any]:
    output = _require_external_path(output_path, label="inventory lock")
    artifacts: list[ArtifactSpec] = [
        *manifest.targets.values(),
        *manifest.drafts.values(),
    ]
    locked: dict[str, Any] = {}
    for artifact in artifacts:
        if not artifact.local_path.is_file():
            raise FileNotFoundError(f"Missing GGUF artifact: {artifact.id}")
        size = artifact.local_path.stat().st_size
        if artifact.local_path.name != artifact.filename:
            raise ProtocolError(
                f"Local filename mismatch for {artifact.id}: "
                f"{artifact.local_path.name} != {artifact.filename}"
            )
        if artifact.expected_size is not None and size != artifact.expected_size:
            raise ProtocolError(f"Size mismatch for {artifact.id}")
        digest = sha256_file(artifact.local_path)
        if (
            artifact.expected_sha256 is not None
            and digest != artifact.expected_sha256
        ):
            raise ProtocolError(f"SHA-256 mismatch for {artifact.id}")
        locked[artifact.id] = {
            "filename": artifact.filename,
            "volume_filename": artifact.volume_filename,
            "local_path": str(artifact.local_path),
            "volume_id": artifact.volume_id,
            "size": size,
            "sha256": digest,
            "gguf": gguf_metadata_contract(artifact.local_path),
        }
    image_contract = inspect_image(manifest.image)
    if image_contract["id"] != manifest.expected_image_id:
        raise ProtocolError(
            "llama.cpp image ID differs from image.expected_id "
            f"({image_contract['id']} != {manifest.expected_image_id})"
        )
    compatibility: dict[str, Any] = {}
    for target in manifest.targets.values():
        if not target.mtp_draft_id:
            continue
        target_contract = locked[target.id]["gguf"]
        draft_contract = locked[target.mtp_draft_id]["gguf"]
        metadata_compatibility = mtp_metadata_compatibility(
            target_contract,
            draft_contract,
        )
        compatibility[target.id] = {
            "draft_id": target.mtp_draft_id,
            "manifest_pair_allowed": target.id
            in manifest.drafts[target.mtp_draft_id].allowed_target_ids,
            **metadata_compatibility,
        }
    payload = {
        "inventory_lock_version": INVENTORY_LOCK_VERSION,
        "manifest_sha256": sha256_file(manifest.path),
        "image": image_contract,
        "artifacts": locked,
        "mtp_compatibility": compatibility,
        "created_at_unix": time.time(),
    }
    payload["lock_sha256"] = _canonical_sha256(payload)
    _atomic_write_json(output, payload)
    return payload


def load_inventory_lock(
    manifest: BenchmarkManifest,
    path: Path,
) -> dict[str, Any]:
    lock_path = _require_external_path(path, label="inventory lock")
    payload = _load_object(lock_path)
    if int(payload.get("inventory_lock_version", 0)) != INVENTORY_LOCK_VERSION:
        raise ProtocolError("Unsupported inventory lock version")
    claimed = str(payload.get("lock_sha256") or "")
    unsigned = dict(payload)
    unsigned.pop("lock_sha256", None)
    if not SHA256_RE.fullmatch(claimed) or _canonical_sha256(unsigned) != claimed:
        raise ProtocolError("Inventory lock checksum is invalid")
    if str(payload.get("manifest_sha256") or "") != sha256_file(manifest.path):
        raise ProtocolError("Model manifest changed after inventory lock")
    image = payload.get("image") or {}
    if str(image.get("id") or "") != manifest.expected_image_id:
        raise ProtocolError("Inventory image ID differs from manifest")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ProtocolError("Inventory artifacts must be an object")
    expected_ids = set(manifest.targets) | set(manifest.drafts)
    if set(artifacts) != expected_ids:
        raise ProtocolError("Inventory artifact set is incomplete or has extras")
    for artifact_id, artifact in artifacts.items():
        if not isinstance(artifact, Mapping):
            raise ProtocolError(f"Inventory artifact is invalid: {artifact_id}")
        manifest_artifact = (
            manifest.targets.get(artifact_id) or manifest.drafts.get(artifact_id)
        )
        assert manifest_artifact is not None
        if (
            str(artifact.get("filename") or "") != manifest_artifact.filename
            or str(artifact.get("volume_filename") or "")
            != manifest_artifact.volume_filename
            or str(artifact.get("volume_id") or "") != manifest_artifact.volume_id
        ):
            raise ProtocolError(
                f"Inventory artifact location differs from manifest: {artifact_id}"
            )
        digest = str(artifact.get("sha256") or "")
        size = int(artifact.get("size", 0) or 0)
        if not SHA256_RE.fullmatch(digest) or size <= 0:
            raise ProtocolError(f"Inventory artifact is unlocked: {artifact_id}")
    return payload


def run_process(
    arguments: Sequence[str],
    *,
    timeout_sec: float = 60,
    allow_failure: bool = False,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [str(argument) for argument in arguments],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_sec,
        check=False,
    )
    if completed.returncode != 0 and not allow_failure:
        command = " ".join(str(argument) for argument in arguments[:4])
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {command}\n"
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed


def docker_executable() -> str:
    for candidate in ("docker.exe", "docker"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError("docker.exe/docker is unavailable")


def docker_json(arguments: Sequence[str], *, timeout_sec: float = 60) -> Any:
    completed = run_process(
        [docker_executable(), *arguments],
        timeout_sec=timeout_sec,
    )
    return json.loads(completed.stdout)


def inspect_image(reference: str) -> dict[str, Any]:
    payload = docker_json(["image", "inspect", reference], timeout_sec=60)
    if not isinstance(payload, list) or len(payload) != 1:
        raise ProtocolError(f"Unable to inspect image: {reference}")
    item = payload[0]
    return {
        "reference": reference,
        "id": str(item.get("Id") or "").casefold(),
        "repo_digests": sorted(str(value) for value in item.get("RepoDigests") or []),
        "created": str(item.get("Created") or ""),
    }


def build_server_command(
    manifest: BenchmarkManifest,
    inventory: Mapping[str, Any],
    profile: Profile,
) -> list[str]:
    target = manifest.targets[profile.target_id]
    target_artifact = inventory["artifacts"][target.id]
    command = [
        "-m",
        f"/volumes/{target.volume_id}/{target.volume_filename}",
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "-c",
        str(DEFAULT_CONTEXT_SIZE),
        "-np",
        "1",
        "-t",
        str(DEFAULT_THREADS),
        "--n-gpu-layers",
        str(profile.target_ngl),
        "--fit",
        "off",
        "-fa",
        "on",
        "-ctk",
        "f16",
        "-ctv",
        "f16",
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
        "0",
    ]
    if profile.speculation == "ngram":
        command.extend(
            [
                "--spec-type",
                "ngram-mod",
                "--spec-ngram-mod-n-min",
                str(profile.draft_n),
                "--spec-ngram-mod-n-max",
                str(profile.draft_n),
            ]
        )
    elif profile.speculation == "mtp":
        if not profile.draft_id:
            raise ProtocolError("MTP profile is missing draft_id")
        draft = manifest.drafts[profile.draft_id]
        if target.id not in draft.allowed_target_ids:
            raise ProtocolError("MTP profile violates allowed pairing")
        compatibility = (inventory.get("mtp_compatibility") or {}).get(target.id)
        if not isinstance(compatibility, Mapping):
            raise ProtocolError("MTP compatibility contract is missing")
        if not bool(compatibility.get("manifest_pair_allowed")):
            raise ProtocolError("MTP manifest pairing was not inventory-validated")
        if not bool(compatibility.get("metadata_compatible")):
            raise ProtocolError(
                "MTP tokenizer/vocabulary metadata conflicts with the target"
            )
        draft_artifact = inventory["artifacts"][draft.id]
        if not draft_artifact.get("sha256") or not target_artifact.get("sha256"):
            raise ProtocolError("Target and draft must be inventory-locked")
        command.extend(
            [
                "-md",
                f"/volumes/{draft.volume_id}/{draft.volume_filename}",
                "--spec-type",
                "draft-mtp",
                "--spec-draft-n-max",
                str(profile.draft_n),
                "--spec-draft-ngl",
                profile.draft_ngl or "all",
                "-ctkd",
                "f16",
                "-ctvd",
                "f16",
            ]
        )
    elif profile.speculation != "none":
        raise ProtocolError(f"Unsupported speculation mode: {profile.speculation}")
    if "-cd" in command or "--draft" in command:
        raise ProtocolError("Removed llama.cpp speculative flags are forbidden")
    return command


def runtime_fingerprint(
    manifest: BenchmarkManifest,
    inventory: Mapping[str, Any],
    profile: Profile,
) -> str:
    target = manifest.targets[profile.target_id]
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "image_id": manifest.expected_image_id,
        "target": {
            "id": target.id,
            "size": inventory["artifacts"][target.id]["size"],
            "sha256": inventory["artifacts"][target.id]["sha256"],
        },
        "command": build_server_command(manifest, inventory, profile),
        "volumes": {
            volume_id: manifest.volumes[volume_id].name
            for volume_id in sorted(
                {
                    target.volume_id,
                    *(
                        [manifest.drafts[profile.draft_id].volume_id]
                        if profile.draft_id
                        else []
                    ),
                }
            )
        },
    }
    if profile.draft_id:
        payload["draft"] = {
            "id": profile.draft_id,
            "size": inventory["artifacts"][profile.draft_id]["size"],
            "sha256": inventory["artifacts"][profile.draft_id]["sha256"],
        }
    return _canonical_sha256(payload)


def _mount_arguments(
    manifest: BenchmarkManifest,
    profile: Profile,
) -> list[str]:
    target = manifest.targets[profile.target_id]
    volume_ids = {target.volume_id}
    if profile.draft_id:
        volume_ids.add(manifest.drafts[profile.draft_id].volume_id)
    arguments: list[str] = []
    for volume_id in sorted(volume_ids):
        volume = manifest.volumes[volume_id]
        arguments.extend(
            [
                "--mount",
                (
                    f"type=volume,source={volume.name},"
                    f"target=/volumes/{volume.id},readonly"
                ),
            ]
        )
    return arguments


def expected_container_name(profile: Profile, fingerprint: str) -> str:
    base = re.sub(r"[^a-z0-9_.-]+", "-", profile.id.casefold())
    return f"ct-gemma-tournament-{base[:42]}-{fingerprint[:12]}"


def _container_contract(
    manifest: BenchmarkManifest,
    inventory: Mapping[str, Any],
    profile: Profile,
) -> dict[str, Any]:
    fingerprint = runtime_fingerprint(manifest, inventory, profile)
    name = expected_container_name(profile, fingerprint)
    return {
        "name": name,
        "fingerprint": fingerprint,
        "command": build_server_command(manifest, inventory, profile),
        "mount_args": _mount_arguments(manifest, profile),
    }


def _inspect_optional_container(name: str) -> dict[str, Any] | None:
    completed = run_process(
        [docker_executable(), "inspect", name],
        timeout_sec=30,
        allow_failure=True,
    )
    if completed.returncode != 0:
        return None
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list) or len(payload) != 1:
        raise ProtocolError(f"Unexpected Docker inspect response: {name}")
    return payload[0]


def container_contract_errors(
    *,
    inspected: Mapping[str, Any],
    manifest: BenchmarkManifest,
    profile: Profile,
    contract: Mapping[str, Any],
) -> list[str]:
    config = inspected.get("Config") or {}
    host_config = inspected.get("HostConfig") or {}
    labels = config.get("Labels") or {}
    errors: list[str] = []
    if str(inspected.get("Image") or "").casefold() != manifest.expected_image_id:
        errors.append("image ID")
    if (
        str(labels.get("comic-translate.config-fingerprint") or "")
        != contract["fingerprint"]
    ):
        errors.append("fingerprint label")
    if (
        str(labels.get("comic-translate.runtime") or "")
        != "gemma-profile-tournament"
    ):
        errors.append("runtime label")
    if [str(value) for value in config.get("Cmd") or []] != contract["command"]:
        errors.append("command")
    if [str(value) for value in config.get("Entrypoint") or []] != [
        "/app/llama-server"
    ]:
        errors.append("entrypoint")
    if bool(host_config.get("Privileged")):
        errors.append("privileged")
    if bool(host_config.get("AutoRemove")):
        errors.append("auto-remove")
    if str((host_config.get("RestartPolicy") or {}).get("Name") or "no") != "no":
        errors.append("restart policy")
    if not bool(host_config.get("DeviceRequests") or []):
        errors.append("GPU device request")
    if str(host_config.get("NetworkMode") or "") not in {"default", "bridge"}:
        errors.append("network mode")
    port_bindings = host_config.get("PortBindings") or {}
    bindings = port_bindings.get("8080/tcp") or []
    normalized_bindings = [
        {
            "host_ip": str(binding.get("HostIp") or ""),
            "host_port": str(binding.get("HostPort") or ""),
        }
        for binding in bindings
        if isinstance(binding, Mapping)
    ]
    if set(port_bindings) != {"8080/tcp"} or normalized_bindings != [
        {"host_ip": "127.0.0.1", "host_port": str(DEFAULT_PORT)}
    ]:
        errors.append("loopback port binding")

    target = manifest.targets[profile.target_id]
    expected_volume_ids = {target.volume_id}
    if profile.draft_id:
        expected_volume_ids.add(manifest.drafts[profile.draft_id].volume_id)
    expected_mounts = {
        f"/volumes/{volume_id}": manifest.volumes[volume_id].name
        for volume_id in expected_volume_ids
    }
    actual_mounts: dict[str, str] = {}
    for mount in inspected.get("Mounts") or []:
        destination = str(mount.get("Destination") or "")
        if not destination.startswith("/volumes/"):
            continue
        actual_mounts[destination] = str(mount.get("Name") or "")
        if str(mount.get("Type") or "") != "volume" or bool(mount.get("RW")):
            errors.append(f"read-only volume mount {destination}")
    if actual_mounts != expected_mounts:
        errors.append("volume mapping")
    return errors


def ensure_stopped_container(
    manifest: BenchmarkManifest,
    inventory: Mapping[str, Any],
    profile: Profile,
) -> dict[str, Any]:
    contract = _container_contract(manifest, inventory, profile)
    existing = _inspect_optional_container(contract["name"])
    if existing is not None:
        errors = container_contract_errors(
            inspected=existing,
            manifest=manifest,
            profile=profile,
            contract=contract,
        )
        state = str((existing.get("State") or {}).get("Status") or "")
        if state not in {"created", "exited"}:
            errors.append(f"state={state}")
        if errors:
            raise ProtocolError(
                f"Stopped container contract mismatch for {contract['name']}: "
                + ", ".join(errors)
            )
        contract["container_id"] = str(existing.get("Id") or "")
        contract["reused"] = True
        return contract

    arguments = [
        docker_executable(),
        "create",
        "--name",
        contract["name"],
        "--label",
        "comic-translate.runtime=gemma-profile-tournament",
        "--label",
        f"comic-translate.config-fingerprint={contract['fingerprint']}",
        "--label",
        f"comic-translate.profile={profile.id}",
        "--gpus",
        "all",
        "-e",
        "NVIDIA_VISIBLE_DEVICES=all",
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        "-p",
        f"127.0.0.1:{DEFAULT_PORT}:8080",
        *contract["mount_args"],
        "--entrypoint",
        "/app/llama-server",
        manifest.image,
        *contract["command"],
    ]
    completed = run_process(arguments, timeout_sec=120)
    created = _inspect_optional_container(contract["name"])
    if created is None:
        raise ProtocolError("Created container cannot be inspected")
    errors = container_contract_errors(
        inspected=created,
        manifest=manifest,
        profile=profile,
        contract=contract,
    )
    if errors:
        raise ProtocolError(
            f"Created container contract mismatch for {contract['name']}: "
            + ", ".join(errors)
        )
    contract["container_id"] = str(created.get("Id") or completed.stdout.strip())
    contract["reused"] = False
    return contract


def _active_port_containers() -> list[dict[str, str]]:
    completed = run_process(
        [
            docker_executable(),
            "ps",
            "--filter",
            f"publish={DEFAULT_PORT}",
            "--format",
            "{{.ID}}\t{{.Names}}",
        ],
        timeout_sec=30,
    )
    output = []
    for raw in completed.stdout.splitlines():
        if not raw.strip():
            continue
        container_id, _, name = raw.partition("\t")
        output.append({"id": container_id.strip(), "name": name.strip()})
    return output


def stop_owned_port_containers(*, except_name: str = "") -> None:
    for item in _active_port_containers():
        if item["name"] == except_name:
            continue
        inspected = _inspect_optional_container(item["name"]) or {}
        labels = (inspected.get("Config") or {}).get("Labels") or {}
        if str(labels.get("comic-translate.runtime") or "") not in {
            "gemma-profile-tournament",
            "gemma-probe",
        }:
            raise ProtocolError(
                f"Port {DEFAULT_PORT} is occupied by an unrelated container: "
                f"{item['name']}"
            )
        run_process(
            [
                docker_executable(),
                "stop",
                "--time",
                str(DEFAULT_STOP_TIMEOUT_SEC),
                item["name"],
            ],
            timeout_sec=DEFAULT_STOP_TIMEOUT_SEC + 30,
        )


def _running_gpu_containers() -> list[str]:
    ids = run_process(
        [docker_executable(), "ps", "-q"],
        timeout_sec=30,
    ).stdout.split()
    unrelated: list[str] = []
    for container_id in ids:
        payload = docker_json(["inspect", container_id], timeout_sec=30)
        item = payload[0]
        device_requests = (item.get("HostConfig") or {}).get("DeviceRequests") or []
        if not device_requests:
            continue
        labels = (item.get("Config") or {}).get("Labels") or {}
        runtime_label = str(labels.get("comic-translate.runtime") or "")
        if runtime_label in {"gemma-profile-tournament", "gemma-probe"}:
            continue
        unrelated.append(str((item.get("Name") or "").lstrip("/")))
    return sorted(unrelated)


def query_gpu_snapshot() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi.exe") or shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "reason": "nvidia-smi unavailable"}
    completed = run_process(
        [
            executable,
            "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        timeout_sec=15,
        allow_failure=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return {
            "available": False,
            "reason": completed.stderr.strip() or "nvidia-smi failed",
        }
    first = completed.stdout.splitlines()[0]
    values = [value.strip() for value in first.split(",")]
    if len(values) < 5:
        return {"available": False, "reason": "unexpected nvidia-smi output"}
    try:
        memory_total = float(values[0])
        memory_used = float(values[1])
        memory_free = float(values[2])
    except ValueError:
        return {"available": False, "reason": "invalid nvidia-smi memory output"}

    def optional_float(value: str) -> float | None:
        try:
            return float(value)
        except ValueError:
            return None

    return {
        "available": True,
        "memory_total_mb": memory_total,
        "memory_used_mb": memory_used,
        "memory_free_mb": memory_free,
        "utilization_percent": optional_float(values[3]),
        "power_watts": optional_float(values[4]),
    }


def query_wsl_swap_used_mb() -> float | None:
    if os.name == "nt":
        executable = shutil.which("wsl.exe") or shutil.which("wsl")
        if not executable:
            return None
        try:
            completed = run_process(
                [
                    executable,
                    "-e",
                    "sh",
                    "-lc",
                    "awk '/SwapTotal:/{t=$2}/SwapFree:/{f=$2}END{print (t-f)/1024}' /proc/meminfo",
                ],
                timeout_sec=15,
                allow_failure=True,
            )
        except subprocess.TimeoutExpired:
            return None
        raw = completed.stdout.strip()
    else:
        meminfo = Path("/proc/meminfo")
        if not meminfo.is_file():
            return None
        values: dict[str, float] = {}
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            if key in {"SwapTotal", "SwapFree"}:
                values[key] = float(rest.strip().split()[0])
        if set(values) != {"SwapTotal", "SwapFree"}:
            return None
        raw = str((values["SwapTotal"] - values["SwapFree"]) / 1024.0)
    try:
        return float(raw)
    except ValueError:
        return None


def query_shared_gpu_used_mb() -> float | None:
    executable = shutil.which("powershell.exe") or shutil.which("powershell")
    if not executable:
        return None
    script = (
        "$samples=(Get-Counter "
        "'\\GPU Process Memory(*)\\Shared Usage' "
        "-ErrorAction Stop).CounterSamples;"
        "$sum=($samples|Measure-Object -Property CookedValue -Sum).Sum;"
        "[Console]::WriteLine([Math]::Round($sum/1MB,3))"
    )
    completed = run_process(
        [
            executable,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            script,
        ],
        timeout_sec=15,
        allow_failure=True,
    )
    if completed.returncode != 0:
        return None
    try:
        return float(completed.stdout.strip())
    except ValueError:
        return None


def preflight_environment(manifest: BenchmarkManifest) -> dict[str, Any]:
    image = inspect_image(manifest.image)
    if image["id"] != manifest.expected_image_id:
        raise ProtocolError("Pinned llama.cpp image ID is unavailable")
    unrelated_gpu = _running_gpu_containers()
    gpu = query_gpu_snapshot()
    failures: list[str] = []
    if unrelated_gpu:
        failures.append(
            "unrelated running GPU containers: " + ", ".join(unrelated_gpu)
        )
    if not gpu.get("available"):
        failures.append("GPU telemetry unavailable")
    elif float(gpu.get("memory_used_mb", math.inf)) > manifest.max_idle_gpu_used_mb:
        failures.append(
            "idle GPU memory exceeds contract "
            f"({gpu['memory_used_mb']:.0f} MiB > "
            f"{manifest.max_idle_gpu_used_mb} MiB)"
        )
    occupied = _active_port_containers()
    for item in occupied:
        inspected = _inspect_optional_container(item["name"]) or {}
        labels = (inspected.get("Config") or {}).get("Labels") or {}
        if str(labels.get("comic-translate.runtime") or "") not in {
            "gemma-profile-tournament",
            "gemma-probe",
        }:
            failures.append(
                f"port {DEFAULT_PORT} occupied by {item['name']}"
            )
    result = {
        "passed": not failures,
        "failures": failures,
        "image": image,
        "gpu": gpu,
        "wsl_swap_used_mb": query_wsl_swap_used_mb(),
        "shared_gpu_used_mb": query_shared_gpu_used_mb(),
        "unrelated_gpu_containers": unrelated_gpu,
        "port_containers": occupied,
    }
    if failures:
        raise ProtocolError("; ".join(failures))
    return result


class ResourceSampler:
    def __init__(self, *, interval_sec: float = 1.0):
        self.interval_sec = max(0.5, float(interval_sec))
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "ResourceSampler":
        def collect() -> None:
            next_swap_sample_at = 0.0
            next_shared_gpu_sample_at = 0.0
            last_swap: float | None = None
            last_shared_gpu: float | None = None
            while not self._stop.is_set():
                now = time.monotonic()
                if now >= next_swap_sample_at:
                    last_swap = query_wsl_swap_used_mb()
                    next_swap_sample_at = now + 10.0
                if now >= next_shared_gpu_sample_at:
                    last_shared_gpu = query_shared_gpu_used_mb()
                    next_shared_gpu_sample_at = now + 30.0
                self.samples.append(
                    {
                        "at_unix": time.time(),
                        "gpu": query_gpu_snapshot(),
                        "wsl_swap_used_mb": last_swap,
                        "shared_gpu_used_mb": last_shared_gpu,
                    }
                )
                self._stop.wait(self.interval_sec)

        self._thread = threading.Thread(target=collect, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def summary(self) -> dict[str, Any]:
        gpu_used = [
            float(item["gpu"]["memory_used_mb"])
            for item in self.samples
            if (item.get("gpu") or {}).get("available")
        ]
        gpu_util = [
            float(item["gpu"]["utilization_percent"])
            for item in self.samples
            if (
                (item.get("gpu") or {}).get("available")
                and (item.get("gpu") or {}).get("utilization_percent")
                is not None
            )
        ]
        swaps = [
            float(item["wsl_swap_used_mb"])
            for item in self.samples
            if item.get("wsl_swap_used_mb") is not None
        ]
        shared_gpu = [
            float(item["shared_gpu_used_mb"])
            for item in self.samples
            if item.get("shared_gpu_used_mb") is not None
        ]
        return {
            "sample_count": len(self.samples),
            "gpu_memory_used_peak_mb": max(gpu_used) if gpu_used else None,
            "gpu_utilization_peak_percent": max(gpu_util) if gpu_util else None,
            "wsl_swap_used_min_mb": min(swaps) if swaps else None,
            "wsl_swap_used_peak_mb": max(swaps) if swaps else None,
            "wsl_swap_growth_mb": (
                max(swaps) - min(swaps) if len(swaps) >= 2 else 0.0
            ),
            "shared_gpu_used_min_mb": min(shared_gpu) if shared_gpu else None,
            "shared_gpu_used_peak_mb": max(shared_gpu) if shared_gpu else None,
            "shared_gpu_growth_mb": (
                max(shared_gpu) - min(shared_gpu)
                if len(shared_gpu) >= 2
                else 0.0
            ),
        }


def load_corpus(path: Path) -> dict[str, Any]:
    corpus_path = _require_external_path(path, label="benchmark corpus")
    payload = _load_object(corpus_path)
    if int(payload.get("protocol_version", 0)) != PROTOCOL_VERSION:
        raise ProtocolError("Unsupported corpus protocol_version")
    sensitive = payload.get("sensitive_groups")
    contiguous = payload.get("contiguous_groups")
    if not isinstance(sensitive, list) or not isinstance(contiguous, list):
        raise ProtocolError("Corpus requires sensitive_groups and contiguous_groups")

    def validate_groups(groups: list[Any], *, label: str) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for group_index, raw_group in enumerate(groups):
            if not isinstance(raw_group, Mapping):
                raise ProtocolError(f"{label}[{group_index}] must be an object")
            group_id = _require_identifier(
                raw_group.get("id"),
                label=f"{label}[{group_index}].id",
            )
            source_language = str(raw_group.get("source_language") or "").strip()
            target_language = str(raw_group.get("target_language") or "").strip()
            if source_language not in {"Japanese", "Chinese", "English"}:
                raise ProtocolError(f"Unsupported source language: {source_language}")
            if target_language != "Korean":
                raise ProtocolError("Tournament target language must be Korean")
            raw_items = raw_group.get("items")
            if not isinstance(raw_items, list) or not raw_items:
                raise ProtocolError(f"{group_id} has no items")
            items: list[dict[str, Any]] = []
            for item_index, raw_item in enumerate(raw_items):
                if not isinstance(raw_item, Mapping):
                    raise ProtocolError(f"{group_id} item must be an object")
                item_id = _require_identifier(
                    raw_item.get("id"),
                    label=f"{group_id}.items[{item_index}].id",
                )
                if item_id in seen_ids:
                    raise ProtocolError(f"Duplicate corpus item id: {item_id}")
                seen_ids.add(item_id)
                text = str(raw_item.get("text") or "")
                if not text.strip():
                    raise ProtocolError(f"Empty corpus text: {item_id}")
                items.append(
                    {
                        "id": item_id,
                        "text": text,
                        "reference": str(raw_item.get("reference") or ""),
                        "review_focus": str(raw_item.get("review_focus") or ""),
                    }
                )
            normalized.append(
                {
                    "id": group_id,
                    "source_language": source_language,
                    "target_language": target_language,
                    "items": items,
                }
            )
        return normalized

    normalized_sensitive = validate_groups(sensitive, label="sensitive_groups")
    normalized_contiguous = validate_groups(contiguous, label="contiguous_groups")
    sensitive_count = sum(len(group["items"]) for group in normalized_sensitive)
    contiguous_count = sum(len(group["items"]) for group in normalized_contiguous)
    language_counts = {
        language: sum(
            len(group["items"])
            for group in normalized_contiguous
            if group["source_language"] == language
        )
        for language in ("Japanese", "Chinese", "English")
    }
    if sensitive_count != 15:
        raise ProtocolError(f"sensitive corpus must contain 15 items, got {sensitive_count}")
    if contiguous_count != 54 or any(value != 18 for value in language_counts.values()):
        raise ProtocolError(
            "contiguous corpus must contain 18 Japanese, 18 Chinese, and "
            "18 English items"
        )
    normalized = {
        "protocol_version": PROTOCOL_VERSION,
        "sensitive_groups": normalized_sensitive,
        "contiguous_groups": normalized_contiguous,
        "counts": {
            "sensitive": sensitive_count,
            "contiguous": contiguous_count,
            "by_language": language_counts,
        },
    }
    normalized["corpus_sha256"] = _canonical_sha256(normalized)
    return normalized


def _stage_groups(
    corpus: Mapping[str, Any],
    stage: str,
) -> list[dict[str, Any]]:
    if stage == "smoke":
        groups = []
        for language in ("Japanese", "Chinese", "English"):
            source_groups = [
                group
                for group in corpus["sensitive_groups"]
                if group["source_language"] == language
            ]
            if not source_groups:
                raise ProtocolError(f"No sensitive group for {language}")
            groups.append(
                {
                    **source_groups[0],
                    "id": f"smoke-{language.casefold()}",
                    "items": source_groups[0]["items"][:1],
                }
            )
        explicit = next(
            (
                group
                for group in corpus["sensitive_groups"]
                if any(
                    "명시" in item["review_focus"]
                    or "성적" in item["review_focus"]
                    or "폭력" in item["review_focus"]
                    for item in group["items"]
                )
            ),
            corpus["sensitive_groups"][0],
        )
        groups.append(
            {
                **explicit,
                "id": "smoke-explicit",
                "items": explicit["items"][-1:],
            }
        )
        return groups
    if stage == "sensitive15":
        return [dict(group) for group in corpus["sensitive_groups"]]
    if stage == "screen18":
        selected: list[dict[str, Any]] = []
        for language in ("Japanese", "Chinese", "English"):
            remaining = 6
            for group in corpus["contiguous_groups"]:
                if group["source_language"] != language or remaining <= 0:
                    continue
                take = min(remaining, len(group["items"]))
                selected.append(
                    {
                        **group,
                        "id": f"screen18-{group['id']}",
                        "items": group["items"][:take],
                    }
                )
                remaining -= take
            if remaining:
                raise ProtocolError(f"Insufficient screen18 items for {language}")
        return selected
    if stage == "final54":
        return [dict(group) for group in corpus["contiguous_groups"]]
    if stage.startswith("breakeven"):
        count = int(stage.removeprefix("breakeven"))
        if count not in {6, 15, 30, 54}:
            raise ProtocolError(f"Unsupported break-even stage: {stage}")
        selected: list[dict[str, Any]] = []
        per_language = count // 3
        if per_language * 3 != count:
            raise ProtocolError("Break-even size must divide evenly by language")
        for language in ("Japanese", "Chinese", "English"):
            remaining = per_language
            for group in corpus["contiguous_groups"]:
                if group["source_language"] != language or remaining <= 0:
                    continue
                take = min(remaining, len(group["items"]))
                selected.append(
                    {
                        **group,
                        "id": f"{stage}-{group['id']}",
                        "items": group["items"][:take],
                    }
                )
                remaining -= take
            if remaining:
                raise ProtocolError(
                    f"Insufficient {language} items for {stage}"
                )
        return selected
    raise ProtocolError(f"Unknown benchmark stage: {stage}")


def _build_engine(
    *,
    model_name: str,
    source_language: str,
    request_timeout_sec: int,
) -> CustomLocalGemmaTranslation:
    engine = CustomLocalGemmaTranslation()
    engine.api_base_url = f"http://127.0.0.1:{DEFAULT_PORT}/v1"
    engine.model = model_name
    engine.source_lang = source_language
    engine.target_lang = "Korean"
    engine.request_mode = GEMMA_REQUEST_MODE_CONTEXTUAL_SINGLE
    engine.chunk_size = DEFAULT_CHUNK_SIZE
    engine.max_tokens = DEFAULT_MAX_COMPLETION_TOKENS
    engine.timeout = int(request_timeout_sec)
    engine.raw_response_logging = False
    engine.prompt_profile = DEFAULT_GEMMA_PROMPT_PROFILE
    engine.response_format_mode = DEFAULT_GEMMA_RESPONSE_FORMAT_MODE
    engine.response_schema_mode = DEFAULT_GEMMA_RESPONSE_SCHEMA_MODE
    engine.think_briefly_prompt = DEFAULT_GEMMA_THINK_BRIEFLY_PROMPT
    engine.temperature = DEFAULT_GEMMA_TRANSLATION_TEMPERATURE
    engine.top_k = DEFAULT_GEMMA_TRANSLATION_TOP_K
    engine.top_p = DEFAULT_GEMMA_TRANSLATION_TOP_P
    engine.min_p = DEFAULT_GEMMA_TRANSLATION_MIN_P
    engine.contextual_merge_input = True
    engine.configure_translation_memory(
        None,
        {
            "persistent_cache_enabled": False,
            "exact_tm_enabled": False,
        },
    )
    return engine


def _make_blocks(items: Sequence[Mapping[str, Any]], source_language: str) -> list[TextBlock]:
    return [
        TextBlock(
            text_bbox=np.asarray([0, 0, 100, 20], dtype=np.int32),
            text=str(item["text"]),
            source_lang=source_language,
            target_lang="Korean",
        )
        for item in items
    ]


def _sum_stats(
    destination: dict[str, int | float],
    source: Mapping[str, Any],
) -> None:
    for key, value in source.items():
        if key == "gemma_telemetry_schema_version":
            destination[key] = max(
                int(destination.get(key, 0) or 0),
                int(value or 0),
            )
        elif key == "gemma_configured_group_size":
            destination[key] = int(value or destination.get(key, 0) or 0)
        elif isinstance(value, float):
            destination[key] = float(destination.get(key, 0.0) or 0.0) + value
        else:
            destination[key] = int(destination.get(key, 0) or 0) + int(value or 0)


def _health_wait(
    *,
    container_name: str,
    model_filename: str,
    timeout_sec: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    deadline = time.monotonic() + timeout_sec
    session = requests.Session()
    last_error = ""
    while time.monotonic() < deadline:
        try:
            response = session.get(
                f"http://127.0.0.1:{DEFAULT_PORT}/health",
                timeout=2,
            )
            if response.status_code < 500:
                model_response = session.get(
                    f"http://127.0.0.1:{DEFAULT_PORT}/v1/models",
                    timeout=10,
                )
                model_response.raise_for_status()
                model_payload = model_response.json()
                names = {
                    Path(str(item.get("id") or "")).name
                    for item in model_payload.get("data") or []
                    if isinstance(item, Mapping)
                }
                if model_filename not in names:
                    raise ProtocolError(
                        f"Loaded model mismatch: {sorted(names)}"
                    )
                return {
                    "ready_elapsed_sec": time.perf_counter() - started,
                    "loaded_models": sorted(names),
                }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            inspected = _inspect_optional_container(container_name) or {}
            state = str((inspected.get("State") or {}).get("Status") or "")
            if state in {"exited", "dead"}:
                logs = run_process(
                    [docker_executable(), "logs", container_name],
                    timeout_sec=30,
                    allow_failure=True,
                )
                raise RuntimeError(
                    f"llama.cpp exited before health: {last_error}\n"
                    f"{logs.stderr[-8000:] or logs.stdout[-8000:]}"
                )
        time.sleep(1)
    raise TimeoutError(f"llama.cpp health timeout: {last_error}")


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name_and_labels, separator, raw_value = line.rpartition(" ")
        if not separator:
            continue
        name = name_and_labels.split("{", 1)[0]
        if not PROMETHEUS_INTEREST_RE.search(name):
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        key = name_and_labels
        values[key] = value
    return values


def fetch_metrics() -> dict[str, float]:
    response = requests.get(
        f"http://127.0.0.1:{DEFAULT_PORT}/metrics",
        timeout=15,
    )
    response.raise_for_status()
    return parse_prometheus_metrics(response.text)


def metric_delta(
    before: Mapping[str, float],
    after: Mapping[str, float],
) -> dict[str, float]:
    return {
        key: float(after_value) - float(before.get(key, 0.0))
        for key, after_value in after.items()
        if float(after_value) - float(before.get(key, 0.0)) != 0.0
    }


def fetch_container_logs(container_name: str) -> str:
    completed = run_process(
        [docker_executable(), "logs", container_name],
        timeout_sec=30,
        allow_failure=True,
    )
    return completed.stdout + completed.stderr


def classify_profile_failure(exc: BaseException, logs: str) -> str:
    combined = f"{type(exc).__name__}: {exc}\n{logs}".casefold()
    if "failed to load draft model" in combined:
        return "draft_model_load"
    if "out of memory" in combined or "cuda error" in combined:
        return "oom_or_cuda"
    if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)):
        return "timeout"
    return "runtime"


def parse_draft_acceptance_logs(text: str) -> dict[str, Any]:
    accepted = 0
    generated = 0
    weighted_mean_length = 0.0
    line_count = 0
    for match in DRAFT_ACCEPTANCE_RE.finditer(text):
        line_accepted = int(match.group("accepted"))
        line_generated = int(match.group("generated"))
        mean_length = float(match.group("mean_length"))
        accepted += line_accepted
        generated += line_generated
        weighted_mean_length += mean_length * line_generated
        line_count += 1
    return {
        "line_count": line_count,
        "draft_tokens": generated,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / generated if generated > 0 else None,
        "mean_accepted_length": (
            weighted_mean_length / generated if generated > 0 else None
        ),
    }


def summarize_speculation_metrics(
    delta: Mapping[str, float],
    stats: Mapping[str, Any],
    log_telemetry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    def sum_matching(
        *,
        required: Sequence[str],
        excluded: Sequence[str] = (),
    ) -> float:
        return sum(
            float(value)
            for key, value in delta.items()
            if all(token in key.casefold() for token in required)
            and not any(token in key.casefold() for token in excluded)
        )

    drafted = sum_matching(
        required=("draft", "token", "total"),
        excluded=("accept",),
    )
    accepted = sum_matching(required=("accept", "token", "total"))
    if accepted == 0:
        accepted = sum_matching(required=("accept", "draft", "total"))
    telemetry_source = "prometheus"
    if drafted <= 0 and log_telemetry:
        drafted = float(log_telemetry.get("draft_tokens", 0) or 0)
        accepted = float(log_telemetry.get("accepted_tokens", 0) or 0)
        telemetry_source = "llama-log"
    completion_tokens = int(stats.get("gemma_completion_tokens", 0) or 0)
    decode_ms = float(stats.get("gemma_decode_ms", 0.0) or 0.0)
    http_attempts = int(stats.get("gemma_http_attempt_count", 0) or 0)
    return {
        "draft_tokens": drafted,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / drafted if drafted > 0 else None,
        "telemetry_source": telemetry_source if drafted > 0 else "none",
        "log_telemetry": dict(log_telemetry or {}),
        "accepted_tokens_per_http_attempt": (
            accepted / http_attempts if http_attempts > 0 else None
        ),
        "completion_tokens": completion_tokens,
        "tpot_ms": (
            decode_ms / completion_tokens if completion_tokens > 0 else None
        ),
        "prompt_eval_ms": float(
            stats.get("gemma_prompt_eval_ms", 0.0) or 0.0
        ),
        "decode_ms": decode_ms,
        "raw_metric_delta": dict(delta),
    }


_MEMORY_UNITS = {
    "b": 1.0 / (1024 * 1024),
    "kb": 1000.0 / (1024 * 1024),
    "kib": 1.0 / 1024,
    "mb": 1_000_000.0 / (1024 * 1024),
    "mib": 1.0,
    "gb": 1_000_000_000.0 / (1024 * 1024),
    "gib": 1024.0,
}


def parse_memory_mib(value: str) -> float | None:
    match = re.fullmatch(
        r"\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)\s*",
        str(value or ""),
    )
    if not match:
        return None
    multiplier = _MEMORY_UNITS.get(match.group(2).casefold())
    if multiplier is None:
        return None
    return float(match.group(1)) * multiplier


def query_container_stats(container_name: str) -> dict[str, Any]:
    completed = run_process(
        [
            docker_executable(),
            "stats",
            "--no-stream",
            "--format",
            "{{json .}}",
            container_name,
        ],
        timeout_sec=30,
        allow_failure=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return {
            "available": False,
            "reason": completed.stderr.strip() or "docker stats failed",
        }
    try:
        payload = json.loads(completed.stdout.splitlines()[0])
    except json.JSONDecodeError:
        return {"available": False, "reason": "invalid docker stats JSON"}
    memory_used_raw = str(payload.get("MemUsage") or "").split("/", 1)[0].strip()
    return {
        "available": True,
        "memory_used_mib": parse_memory_mib(memory_used_raw),
        "memory_percent": str(payload.get("MemPerc") or ""),
        "cpu_percent": str(payload.get("CPUPerc") or ""),
        "pids": str(payload.get("PIDs") or ""),
    }


def _warm_runtime(model_filename: str, *, timeout_sec: int) -> dict[str, Any]:
    engine = _build_engine(
        model_name=model_filename,
        source_language="Japanese",
        request_timeout_sec=timeout_sec,
    )
    blocks = _make_blocks(
        [{"text": "これはランタイムのウォームアップです。"}],
        "Japanese",
    )
    started = time.perf_counter()
    engine.translate(blocks, np.zeros((1, 1, 3), dtype=np.uint8), "")
    elapsed = time.perf_counter() - started
    if not str(blocks[0].translation or "").strip():
        raise ProtocolError("Warm-up translation is empty")
    return {
        "elapsed_sec": elapsed,
        "stats": dict(engine.last_benchmark_stats),
    }


def _stage_order(
    groups: Sequence[dict[str, Any]],
    *,
    reverse: bool,
) -> list[dict[str, Any]]:
    ordered = [dict(group) for group in groups]
    if reverse:
        ordered.reverse()
        for group in ordered:
            group["items"] = list(reversed(group["items"]))
    return ordered


def _run_translation_groups(
    *,
    model_filename: str,
    groups: Sequence[dict[str, Any]],
    request_timeout_sec: int,
) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    chunk_timings: list[dict[str, Any]] = []
    combined_stats: dict[str, int | float] = {}
    for group in groups:
        source_language = str(group["source_language"])
        engine = _build_engine(
            model_name=model_filename,
            source_language=source_language,
            request_timeout_sec=request_timeout_sec,
        )
        items = list(group["items"])
        for chunk_index, start in enumerate(range(0, len(items), DEFAULT_CHUNK_SIZE)):
            chunk_items = items[start : start + DEFAULT_CHUNK_SIZE]
            blocks = _make_blocks(chunk_items, source_language)
            started = time.perf_counter()
            engine.translate(
                blocks,
                np.zeros((1, 1, 3), dtype=np.uint8),
                "",
            )
            elapsed = time.perf_counter() - started
            stats = dict(engine.last_benchmark_stats)
            _sum_stats(combined_stats, stats)
            chunk_id = f"{group['id']}::chunk-{chunk_index:02d}"
            chunk_timings.append(
                {
                    "chunk_id": chunk_id,
                    "group_id": group["id"],
                    "source_language": source_language,
                    "block_count": len(chunk_items),
                    "elapsed_sec": elapsed,
                    "request_wall_sec": (
                        float(stats.get("gemma_request_wall_ms", 0.0) or 0.0)
                        / 1000.0
                    ),
                    "prompt_eval_sec": (
                        float(stats.get("gemma_prompt_eval_ms", 0.0) or 0.0)
                        / 1000.0
                    ),
                    "decode_sec": (
                        float(stats.get("gemma_decode_ms", 0.0) or 0.0)
                        / 1000.0
                    ),
                }
            )
            for item, block in zip(chunk_items, blocks):
                translation = str(block.translation or "")
                outputs.append(
                    {
                        "item_id": item["id"],
                        "group_id": group["id"],
                        "source_language": source_language,
                        "source": item["text"],
                        "source_sha256": hashlib.sha256(
                            str(item["text"]).encode(
                                "utf-8",
                                errors="surrogatepass",
                            )
                        ).hexdigest(),
                        "reference": item["reference"],
                        "review_focus": item["review_focus"],
                        "translation": translation,
                        "translation_sha256": hashlib.sha256(
                            translation.encode("utf-8", errors="surrogatepass")
                        ).hexdigest(),
                        "empty": not bool(translation.strip()),
                    }
                )
    return {
        "outputs": outputs,
        "chunk_timings": chunk_timings,
        "stats": combined_stats,
        "elapsed_sec": sum(
            float(item["elapsed_sec"]) for item in chunk_timings
        ),
    }


def _streaming_ttft_probe(
    *,
    model_filename: str,
    source_language: str,
    source_text: str,
    timeout_sec: int,
) -> dict[str, Any]:
    engine = _build_engine(
        model_name=model_filename,
        source_language=source_language,
        request_timeout_sec=timeout_sec,
    )
    block = _make_blocks([{"text": source_text}], source_language)[0]
    context = engine._create_request_context([block])
    system_prompt = engine._build_system_prompt(  # type: ignore[attr-defined]
        "",
        prompt_profile=engine.prompt_profile,
    )
    user_prompt = engine._build_contextual_single_request_prompt(context, 0)
    payload = {
        "model": model_filename,
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            },
        ],
        "temperature": engine.temperature,
        "top_k": engine.top_k,
        "top_p": engine.top_p,
        "min_p": engine.min_p,
        "max_completion_tokens": engine.max_tokens,
        "response_format": engine._build_response_format(  # type: ignore[attr-defined]
            user_prompt,
            expected_keys=["translation"],
        ),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_content_at: float | None = None
    finish_reason = ""
    usage: dict[str, Any] = {}
    with requests.post(
        f"http://127.0.0.1:{DEFAULT_PORT}/v1/chat/completions",
        json=payload,
        timeout=timeout_sec,
        stream=True,
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            line = str(raw_line or "").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if isinstance(chunk.get("usage"), Mapping):
                usage = dict(chunk["usage"])
            choices = chunk.get("choices") or []
            for choice in choices:
                delta = choice.get("delta") or {}
                if str(delta.get("content") or "") and first_content_at is None:
                    first_content_at = time.perf_counter()
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])
    ended = time.perf_counter()
    return {
        "ttft_ms": (
            (first_content_at - started) * 1000.0
            if first_content_at is not None
            else None
        ),
        "total_ms": (ended - started) * 1000.0,
        "finish_reason": finish_reason,
        "usage": usage,
    }


def _result_gates(
    *,
    result: Mapping[str, Any],
    expected_item_ids: Sequence[str],
    resource_summary: Mapping[str, Any],
    max_swap_growth_mb: int,
    max_shared_gpu_growth_mb: int,
) -> dict[str, Any]:
    outputs = result.get("outputs") or []
    actual_ids = [str(output.get("item_id") or "") for output in outputs]
    empty_count = sum(bool(output.get("empty")) for output in outputs)
    stats = result.get("stats") or {}
    structural = {
        key: int(stats.get(key, 0) or 0)
        for key in STRUCTURAL_STATS
        if int(stats.get(key, 0) or 0)
    }
    swap_growth = float(resource_summary.get("wsl_swap_growth_mb") or 0.0)
    shared_gpu_growth = float(
        resource_summary.get("shared_gpu_growth_mb") or 0.0
    )
    return {
        "expected_count": len(expected_item_ids),
        "actual_count": len(actual_ids),
        "count_ok": len(actual_ids) == len(expected_item_ids),
        "order_preserved": actual_ids == list(expected_item_ids),
        "empty_count": empty_count,
        "structural_stats": structural,
        "unresolved_fallback_count": int(
            stats.get("gemma_contextual_merge_fallback_count", 0) or 0
        ),
        "finish_reason_length_count": int(
            stats.get("gemma_truncated_count", 0) or 0
        ),
        "swap_growth_mb": swap_growth,
        "swap_growth_ok": swap_growth <= max_swap_growth_mb,
        "shared_gpu_growth_mb": shared_gpu_growth,
        "shared_gpu_growth_ok": (
            shared_gpu_growth <= max_shared_gpu_growth_mb
        ),
        "hard_gate_passed": (
            len(actual_ids) == len(expected_item_ids)
            and actual_ids == list(expected_item_ids)
            and empty_count == 0
            and not structural
            and swap_growth <= max_swap_growth_mb
            and shared_gpu_growth <= max_shared_gpu_growth_mb
        ),
    }


def run_profile_stage(
    *,
    manifest: BenchmarkManifest,
    inventory: Mapping[str, Any],
    corpus: Mapping[str, Any],
    profile: Profile,
    stage: str,
    round_index: int,
    output_dir: Path,
    start_timeout_sec: int,
    request_timeout_sec: int,
) -> dict[str, Any]:
    if round_index not in {1, 2, 3}:
        raise ProtocolError("round_index must be 1, 2, or 3")
    output_root = _require_external_path(output_dir, label="result output")
    output_root.mkdir(parents=True, exist_ok=True)
    stage_groups = _stage_order(
        _stage_groups(corpus, stage),
        reverse=round_index % 2 == 0,
    )
    expected_item_ids = [
        item["id"] for group in stage_groups for item in group["items"]
    ]
    contract = _container_contract(manifest, inventory, profile)
    result_path = output_root / (
        f"{stage}__round-{round_index}__{profile.id}"
        f"__ngl-{profile.target_ngl}.json"
    )
    if result_path.exists():
        raise FileExistsError(f"Result already exists: {result_path.name}")

    result: dict[str, Any] = {
        "result_version": RESULT_VERSION,
        "profile": asdict(profile),
        "stage": stage,
        "round": round_index,
        "status": "running",
        "runtime_fingerprint": contract["fingerprint"],
        "corpus_sha256": corpus["corpus_sha256"],
        "inventory_lock_sha256": inventory["lock_sha256"],
        "request_contract": {
            "request_mode": GEMMA_REQUEST_MODE_CONTEXTUAL_SINGLE,
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "context_size": DEFAULT_CONTEXT_SIZE,
            "threads": DEFAULT_THREADS,
            "target_kv": "f16",
            "draft_kv": "f16",
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "prompt_profile": DEFAULT_GEMMA_PROMPT_PROFILE,
            "response_format_mode": DEFAULT_GEMMA_RESPONSE_FORMAT_MODE,
            "response_schema_mode": DEFAULT_GEMMA_RESPONSE_SCHEMA_MODE,
            "temperature": DEFAULT_GEMMA_TRANSLATION_TEMPERATURE,
            "top_k": DEFAULT_GEMMA_TRANSLATION_TOP_K,
            "top_p": DEFAULT_GEMMA_TRANSLATION_TOP_P,
            "min_p": DEFAULT_GEMMA_TRANSLATION_MIN_P,
            "persistent_translation_cache": False,
            "exact_translation_memory": False,
            "project_checkpoint": False,
            "llama_prompt_cache_ram_mib": 0,
        },
        "started_at_unix": time.time(),
    }
    container: dict[str, Any] | None = None
    started = False
    idle_swap = query_wsl_swap_used_mb()
    try:
        result["volume_contract"] = verify_profile_volume_artifacts(
            manifest,
            inventory,
            profile,
        )
        preflight_environment(manifest)
        stop_owned_port_containers()
        container = ensure_stopped_container(manifest, inventory, profile)
        result["container"] = {
            "name": container["name"],
            "id": container["container_id"],
            "reused": container["reused"],
        }
        result["gpu_before_start"] = query_gpu_snapshot()
        with ResourceSampler() as sampler:
            runtime_started = time.perf_counter()
            run_process(
                [docker_executable(), "start", container["name"]],
                timeout_sec=120,
            )
            started = True
            target = manifest.targets[profile.target_id]
            ready = _health_wait(
                container_name=container["name"],
                model_filename=target.volume_filename,
                timeout_sec=start_timeout_sec,
            )
            result["runtime_ready"] = ready
            result["runtime_start_to_ready_sec"] = (
                time.perf_counter() - runtime_started
            )
            result["container_stats_ready"] = query_container_stats(
                container["name"]
            )
            metrics_before_warmup = fetch_metrics()
            warmup = _warm_runtime(
                target.volume_filename,
                timeout_sec=request_timeout_sec,
            )
            metrics_before_stage = fetch_metrics()
            logs_before_stage = fetch_container_logs(container["name"])
            stage_started = time.perf_counter()
            translation = _run_translation_groups(
                model_filename=target.volume_filename,
                groups=stage_groups,
                request_timeout_sec=request_timeout_sec,
            )
            result["request_only_elapsed_sec"] = (
                time.perf_counter() - stage_started
            )
            metrics_after_stage = fetch_metrics()
            logs_after_stage = fetch_container_logs(container["name"])
            stage_logs = (
                logs_after_stage[len(logs_before_stage) :]
                if logs_after_stage.startswith(logs_before_stage)
                else logs_after_stage
            )
            log_speculation = parse_draft_acceptance_logs(stage_logs)
            result["container_stats_after_stage"] = query_container_stats(
                container["name"]
            )
            result["warmup"] = warmup
            result.update(translation)
            first_group = stage_groups[0]
            result["streaming_probe"] = _streaming_ttft_probe(
                model_filename=target.volume_filename,
                source_language=first_group["source_language"],
                source_text=first_group["items"][0]["text"],
                timeout_sec=request_timeout_sec,
            )
            stage_metric_delta = metric_delta(
                metrics_before_stage,
                metrics_after_stage,
            )
            result["metrics"] = {
                "warmup_delta": metric_delta(
                    metrics_before_warmup,
                    metrics_before_stage,
                ),
                "stage_delta": stage_metric_delta,
            }
            result["speculation_telemetry"] = summarize_speculation_metrics(
                stage_metric_delta,
                translation["stats"],
                log_speculation,
            )
        result["resources"] = sampler.summary()
        result["gpu_before_stop"] = query_gpu_snapshot()
        result["gates"] = _result_gates(
            result=result,
            expected_item_ids=expected_item_ids,
            resource_summary=result["resources"],
            max_swap_growth_mb=manifest.max_swap_growth_mb,
            max_shared_gpu_growth_mb=manifest.max_shared_gpu_growth_mb,
        )
        result["total_elapsed_sec"] = (
            float(result["runtime_start_to_ready_sec"])
            + float(result["warmup"]["elapsed_sec"])
            + float(result["request_only_elapsed_sec"])
        )
        result["status"] = (
            "passed" if result["gates"]["hard_gate_passed"] else "failed"
        )
    except BaseException as exc:
        result["status"] = "failed"
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        if container is not None:
            logs = fetch_container_logs(container["name"])
            result["container_log_tail"] = logs[-12_000:]
            result["failure_kind"] = classify_profile_failure(exc, logs)
        else:
            result["failure_kind"] = classify_profile_failure(exc, "")
    finally:
        if started and container is not None:
            stopped = run_process(
                [
                    docker_executable(),
                    "stop",
                    "--time",
                    str(DEFAULT_STOP_TIMEOUT_SEC),
                    container["name"],
                ],
                timeout_sec=DEFAULT_STOP_TIMEOUT_SEC + 30,
                allow_failure=True,
            )
            result["container_stopped"] = stopped.returncode == 0
            if stopped.returncode != 0:
                result["status"] = "failed"
                result["stop_failure"] = stopped.stderr.strip()
        result["gpu_after_stop"] = query_gpu_snapshot()
        final_swap = query_wsl_swap_used_mb()
        result["idle_swap_before_mb"] = idle_swap
        result["idle_swap_after_mb"] = final_swap
        if idle_swap is not None and final_swap is not None:
            result["idle_swap_delta_mb"] = final_swap - idle_swap
        result["completed_at_unix"] = time.time()
        unsigned = dict(result)
        unsigned.pop("result_sha256", None)
        result["result_sha256"] = _canonical_sha256(unsigned)
        _atomic_write_json(result_path, result)
    if result["status"] != "passed":
        raise RuntimeError(
            f"Profile stage failed: {profile.id} {stage} round {round_index}; "
            f"see {result_path}"
        )
    return result


def paired_speed_bootstrap(
    baseline_seconds: Sequence[float],
    candidate_seconds: Sequence[float],
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, float | int | bool]:
    if len(baseline_seconds) != len(candidate_seconds) or not baseline_seconds:
        raise ProtocolError("Paired timing arrays must have the same non-zero length")
    baseline = np.asarray(baseline_seconds, dtype=float)
    candidate = np.asarray(candidate_seconds, dtype=float)
    if np.any(baseline <= 0) or np.any(candidate <= 0):
        raise ProtocolError("Paired timings must be positive")
    gains = ((baseline - candidate) / baseline) * 100.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(gains), size=(samples, len(gains)))
    boot = np.mean(gains[indices], axis=1)
    lower = float(np.quantile(boot, 0.05))
    upper = float(np.quantile(boot, 0.95))
    estimate = float(np.mean(gains))
    return {
        "pair_count": len(gains),
        "bootstrap_samples": int(samples),
        "mean_gain_percent": estimate,
        "one_sided_95_lower_percent": lower,
        "one_sided_95_upper_percent": upper,
        "proven_faster": lower > 0.0,
        "proven_slower": upper < 0.0,
        "uncertain": lower <= 0.0 <= upper,
    }


def _load_result(path: Path) -> dict[str, Any]:
    payload = _load_object(path)
    claimed = str(payload.get("result_sha256") or "")
    unsigned = dict(payload)
    unsigned.pop("result_sha256", None)
    if not SHA256_RE.fullmatch(claimed) or _canonical_sha256(unsigned) != claimed:
        raise ProtocolError(f"Result checksum mismatch: {path.name}")
    return payload


def _comparison_result_contract(
    results: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    if not results:
        raise ProtocolError(f"{label} results are empty")
    if any(result.get("status") != "passed" for result in results):
        raise ProtocolError("Only passed results can be compared")
    profile_ids = {
        str((result.get("profile") or {}).get("id") or "")
        for result in results
    }
    stages = {str(result.get("stage") or "") for result in results}
    corpus_digests = {
        str(result.get("corpus_sha256") or "") for result in results
    }
    inventory_digests = {
        str(result.get("inventory_lock_sha256") or "") for result in results
    }
    request_contract_digests = {
        _canonical_sha256(result.get("request_contract") or {})
        for result in results
    }
    required_request_contract = {
        "request_mode": GEMMA_REQUEST_MODE_CONTEXTUAL_SINGLE,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "context_size": DEFAULT_CONTEXT_SIZE,
        "threads": DEFAULT_THREADS,
        "target_kv": "f16",
        "draft_kv": "f16",
        "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
        "prompt_profile": DEFAULT_GEMMA_PROMPT_PROFILE,
        "response_format_mode": DEFAULT_GEMMA_RESPONSE_FORMAT_MODE,
        "response_schema_mode": DEFAULT_GEMMA_RESPONSE_SCHEMA_MODE,
        "temperature": DEFAULT_GEMMA_TRANSLATION_TEMPERATURE,
        "top_k": DEFAULT_GEMMA_TRANSLATION_TOP_K,
        "top_p": DEFAULT_GEMMA_TRANSLATION_TOP_P,
        "min_p": DEFAULT_GEMMA_TRANSLATION_MIN_P,
        "persistent_translation_cache": False,
        "exact_translation_memory": False,
        "project_checkpoint": False,
        "llama_prompt_cache_ram_mib": 0,
    }
    for result in results:
        request_contract = result.get("request_contract")
        if not isinstance(request_contract, Mapping):
            raise ProtocolError(f"{label} request contract is missing")
        for key, expected in required_request_contract.items():
            if request_contract.get(key) != expected:
                raise ProtocolError(
                    f"{label} request contract changed: {key}"
                )
    rounds = [int(result.get("round", 0) or 0) for result in results]
    if (
        len(profile_ids) != 1
        or "" in profile_ids
        or len(stages) != 1
        or "" in stages
        or len(corpus_digests) != 1
        or not all(SHA256_RE.fullmatch(value) for value in corpus_digests)
        or len(inventory_digests) != 1
        or not all(SHA256_RE.fullmatch(value) for value in inventory_digests)
        or len(request_contract_digests) != 1
        or len(set(rounds)) != len(rounds)
        or any(round_index not in {1, 2, 3} for round_index in rounds)
    ):
        raise ProtocolError(f"{label} result contract is inconsistent")
    output_contracts: list[list[tuple[str, str]]] = []
    for result in results:
        outputs = result.get("outputs")
        if not isinstance(outputs, list):
            raise ProtocolError(f"{label} outputs are missing")
        contract: list[tuple[str, str]] = []
        for output in outputs:
            if not isinstance(output, Mapping):
                raise ProtocolError(f"{label} output item is invalid")
            item_id = str(output.get("item_id") or "")
            source_sha = str(output.get("source_sha256") or "")
            if not item_id or not SHA256_RE.fullmatch(source_sha):
                raise ProtocolError(f"{label} output source contract is invalid")
            contract.append((item_id, source_sha))
        output_contracts.append(sorted(contract))
    if any(contract != output_contracts[0] for contract in output_contracts[1:]):
        raise ProtocolError(f"{label} output corpus differs between rounds")
    return {
        "profile_id": next(iter(profile_ids)),
        "stage": next(iter(stages)),
        "corpus_sha256": next(iter(corpus_digests)),
        "inventory_lock_sha256": next(iter(inventory_digests)),
        "request_contract_sha256": next(iter(request_contract_digests)),
        "rounds": sorted(rounds),
        "output_contract": output_contracts[0],
    }


def compare_profile_results(
    *,
    baseline_paths: Sequence[Path],
    candidate_paths: Sequence[Path],
) -> dict[str, Any]:
    if len(baseline_paths) != len(candidate_paths) or not baseline_paths:
        raise ProtocolError("Baseline and candidate result counts must match")
    baseline_results = [_load_result(path) for path in baseline_paths]
    candidate_results = [_load_result(path) for path in candidate_paths]
    baseline_contract = _comparison_result_contract(
        baseline_results,
        label="baseline",
    )
    candidate_contract = _comparison_result_contract(
        candidate_results,
        label="candidate",
    )
    for key in (
        "stage",
        "corpus_sha256",
        "inventory_lock_sha256",
        "request_contract_sha256",
        "rounds",
        "output_contract",
    ):
        if baseline_contract[key] != candidate_contract[key]:
            raise ProtocolError(
                f"Baseline and candidate {key} contracts differ"
            )
    if baseline_contract["profile_id"] == candidate_contract["profile_id"]:
        raise ProtocolError("Baseline and candidate profiles must differ")
    baseline_by_chunk: dict[str, list[float]] = {}
    candidate_by_chunk: dict[str, list[float]] = {}
    for result, destination in (
        *((result, baseline_by_chunk) for result in baseline_results),
        *((result, candidate_by_chunk) for result in candidate_results),
    ):
        for timing in result.get("chunk_timings") or []:
            chunk_id = str(timing.get("chunk_id") or "")
            destination.setdefault(chunk_id, []).append(
                float(timing["elapsed_sec"])
            )
    if set(baseline_by_chunk) != set(candidate_by_chunk):
        raise ProtocolError("Paired result chunk IDs differ")
    baseline_values: list[float] = []
    candidate_values: list[float] = []
    for chunk_id in sorted(baseline_by_chunk):
        baseline_values.append(statistics.median(baseline_by_chunk[chunk_id]))
        candidate_values.append(statistics.median(candidate_by_chunk[chunk_id]))
    bootstrap = paired_speed_bootstrap(baseline_values, candidate_values)
    baseline_total = statistics.median(
        float(result["request_only_elapsed_sec"]) for result in baseline_results
    )
    candidate_total = statistics.median(
        float(result["request_only_elapsed_sec"]) for result in candidate_results
    )
    bootstrap.update(
        {
            "baseline_request_median_sec": baseline_total,
            "candidate_request_median_sec": candidate_total,
            "candidate_total_gain_percent": (
                (baseline_total - candidate_total) / baseline_total * 100.0
            ),
            "baseline_runtime_ready_median_sec": statistics.median(
                float(result["runtime_start_to_ready_sec"])
                for result in baseline_results
            ),
            "candidate_runtime_ready_median_sec": statistics.median(
                float(result["runtime_start_to_ready_sec"])
                for result in candidate_results
            ),
            "requires_third_round": bool(bootstrap["uncertain"]),
        }
    )
    return bootstrap


def prepare_managed_volumes(
    manifest: BenchmarkManifest,
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    prepared: dict[str, Any] = {}
    for volume in manifest.volumes.values():
        if not volume.managed:
            continue
        inspect = run_process(
            [docker_executable(), "volume", "inspect", volume.name],
            timeout_sec=30,
            allow_failure=True,
        )
        if inspect.returncode != 0:
            run_process(
                [
                    docker_executable(),
                    "volume",
                    "create",
                    "--label",
                    "comic-translate.purpose=gemma-profile-tournament",
                    volume.name,
                ],
                timeout_sec=30,
            )
        artifacts = [
            artifact
            for artifact in [*manifest.targets.values(), *manifest.drafts.values()]
            if artifact.volume_id == volume.id
        ]
        for artifact in artifacts:
            locked = inventory["artifacts"][artifact.id]
            source_parent = str(artifact.local_path.parent)
            partial = artifact.volume_filename + ".partial"
            copy_script = (
                "set -eu; "
                "mkdir /models/.prepare.lock; "
                "trap 'rmdir /models/.prepare.lock' EXIT; "
                "if [ -e \"/models/$2\" ]; then "
                "actual_size=$(stat -c %s \"/models/$2\"); "
                "[ \"$actual_size\" = \"$4\" ] || { "
                "printf 'existing destination size mismatch: %s != %s\\n' "
                "\"$actual_size\" \"$4\" >&2; exit 21; }; "
                "actual_sha=$(sha256sum \"/models/$2\" | cut -d' ' -f1); "
                "[ \"$actual_sha\" = \"$5\" ] || { "
                "printf 'existing destination SHA-256 mismatch\\n' >&2; "
                "exit 22; }; "
                "exit 0; "
                "fi; "
                "[ ! -e \"/models/$3\" ] || { "
                "printf 'partial destination already exists\\n' >&2; exit 23; }; "
                "cp \"/import/$1\" \"/models/$3\"; "
                "test \"$(stat -c %s \"/models/$3\")\" = \"$4\"; "
                "test \"$(sha256sum \"/models/$3\" | cut -d' ' -f1)\" = \"$5\"; "
                "mv \"/models/$3\" \"/models/$2\""
            )
            try:
                run_process(
                    [
                        docker_executable(),
                        "run",
                        "--rm",
                        "--mount",
                        (
                            f"type=bind,source={source_parent},"
                            "target=/import,readonly"
                        ),
                        "--mount",
                        f"type=volume,source={volume.name},target=/models",
                        "--entrypoint",
                        "sh",
                        manifest.helper_image,
                        "-ceu",
                        copy_script,
                        "_",
                        artifact.filename,
                        artifact.volume_filename,
                        partial,
                        str(locked["size"]),
                        str(locked["sha256"]),
                    ],
                    timeout_sec=7200,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    "Failed to prepare managed volume artifact "
                    f"{artifact.id} at "
                    f"{volume.id}/{artifact.volume_filename}: {exc}"
                ) from exc
        ready_payload = {
            "version": 1,
            "volume_id": volume.id,
            "artifacts": {
                artifact.id: {
                    "filename": artifact.filename,
                    "volume_filename": artifact.volume_filename,
                    "size": inventory["artifacts"][artifact.id]["size"],
                    "sha256": inventory["artifacts"][artifact.id]["sha256"],
                }
                for artifact in artifacts
            },
        }
        encoded = json.dumps(
            ready_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        import base64

        ready_base64 = base64.b64encode(encoded).decode("ascii")
        run_process(
            [
                docker_executable(),
                "run",
                "--rm",
                "--mount",
                f"type=volume,source={volume.name},target=/models",
                "--entrypoint",
                "sh",
                manifest.helper_image,
                "-ceu",
                (
                    "set -eu; "
                    "mkdir /models/.prepare.lock; "
                    "trap 'rmdir /models/.prepare.lock' EXIT; "
                    "test ! -e /models/.ready-v1.json.partial; "
                    "printf '%s' \"$1\" | base64 -d "
                    "> /models/.ready-v1.json.partial; "
                    "mv /models/.ready-v1.json.partial /models/.ready-v1.json"
                ),
                "_",
                ready_base64,
            ],
            timeout_sec=60,
        )
        prepared[volume.id] = ready_payload
    return prepared


def verify_volume_artifacts(
    manifest: BenchmarkManifest,
    inventory: Mapping[str, Any],
    *,
    full_hash: bool,
    artifact_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    artifacts_by_id: dict[str, ArtifactSpec] = {
        artifact.id: artifact
        for artifact in [*manifest.targets.values(), *manifest.drafts.values()]
    }
    selected_ids = (
        sorted(artifacts_by_id)
        if artifact_ids is None
        else sorted(set(str(value) for value in artifact_ids))
    )
    unknown = set(selected_ids) - set(artifacts_by_id)
    if unknown:
        raise ProtocolError(
            "Unknown volume artifact IDs: " + ", ".join(sorted(unknown))
        )
    for artifact_id in selected_ids:
        artifact = artifacts_by_id[artifact_id]
        volume = manifest.volumes[artifact.volume_id]
        locked = inventory["artifacts"][artifact.id]
        shell = (
            "test -r \"/models/$1\"; "
            "printf '%s\\n' \"$(stat -c %s \"/models/$1\")\""
        )
        if full_hash:
            shell += "; sha256sum \"/models/$1\" | cut -d' ' -f1"
        completed = run_process(
            [
                docker_executable(),
                "run",
                "--rm",
                "--mount",
                f"type=volume,source={volume.name},target=/models,readonly",
                "--entrypoint",
                "sh",
                manifest.helper_image,
                "-ceu",
                shell,
                "_",
                artifact.volume_filename,
            ],
            timeout_sec=1800 if full_hash else 60,
        )
        lines = completed.stdout.splitlines()
        size = int(lines[0])
        digest = lines[1].strip() if full_hash else None
        if size != int(locked["size"]):
            raise ProtocolError(f"Volume size mismatch: {artifact.id}")
        if full_hash and digest != locked["sha256"]:
            raise ProtocolError(f"Volume SHA-256 mismatch: {artifact.id}")
        results[artifact.id] = {
            "volume_id": artifact.volume_id,
            "filename": artifact.filename,
            "volume_filename": artifact.volume_filename,
            "size": size,
            "sha256": digest,
        }
    return results


def verify_profile_volume_artifacts(
    manifest: BenchmarkManifest,
    inventory: Mapping[str, Any],
    profile: Profile,
) -> dict[str, Any]:
    artifact_ids = [profile.target_id]
    if profile.draft_id:
        artifact_ids.append(profile.draft_id)
    return verify_volume_artifacts(
        manifest,
        inventory,
        full_hash=False,
        artifact_ids=artifact_ids,
    )


def probe_profile_load(
    *,
    manifest: BenchmarkManifest,
    inventory: Mapping[str, Any],
    profile: Profile,
    start_timeout_sec: int,
    request_timeout_sec: int,
) -> dict[str, Any]:
    """Load, generate one product-contract response, then normally stop."""

    result: dict[str, Any] = {
        "profile": asdict(profile),
        "status": "running",
        "started_at_unix": time.time(),
    }
    container: dict[str, Any] | None = None
    started = False
    try:
        result["volume_contract"] = verify_profile_volume_artifacts(
            manifest,
            inventory,
            profile,
        )
        preflight_environment(manifest)
        stop_owned_port_containers()
        container = ensure_stopped_container(manifest, inventory, profile)
        result["container"] = {
            "name": container["name"],
            "id": container["container_id"],
            "reused": container["reused"],
        }
        with ResourceSampler() as sampler:
            runtime_started = time.perf_counter()
            run_process(
                [docker_executable(), "start", container["name"]],
                timeout_sec=120,
            )
            started = True
            target = manifest.targets[profile.target_id]
            ready = _health_wait(
                container_name=container["name"],
                model_filename=target.volume_filename,
                timeout_sec=start_timeout_sec,
            )
            result["runtime_ready"] = ready
            result["runtime_start_to_ready_sec"] = (
                time.perf_counter() - runtime_started
            )
            result["container_stats_ready"] = query_container_stats(
                container["name"]
            )
            result["generation_probe"] = _warm_runtime(
                target.volume_filename,
                timeout_sec=request_timeout_sec,
            )
            result["container_stats_after_probe"] = query_container_stats(
                container["name"]
            )
        result["resources"] = sampler.summary()
        swap_growth = float(
            result["resources"].get("wsl_swap_growth_mb") or 0.0
        )
        shared_gpu_growth = float(
            result["resources"].get("shared_gpu_growth_mb") or 0.0
        )
        result["resource_gates"] = {
            "swap_growth_mb": swap_growth,
            "swap_growth_ok": swap_growth <= manifest.max_swap_growth_mb,
            "shared_gpu_growth_mb": shared_gpu_growth,
            "shared_gpu_growth_ok": (
                shared_gpu_growth <= manifest.max_shared_gpu_growth_mb
            ),
        }
        result["status"] = (
            "passed"
            if (
                result["resource_gates"]["swap_growth_ok"]
                and result["resource_gates"]["shared_gpu_growth_ok"]
            )
            else "failed"
        )
        if result["status"] == "failed":
            result["failure_reason"] = (
                f"WSL swap growth {swap_growth:.1f} MiB exceeds "
                f"{manifest.max_swap_growth_mb} MiB or shared GPU growth "
                f"{shared_gpu_growth:.1f} MiB exceeds "
                f"{manifest.max_shared_gpu_growth_mb} MiB"
            )
    except BaseException as exc:
        result["status"] = "failed"
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        if container is not None:
            logs = fetch_container_logs(container["name"])
            result["container_log_tail"] = logs[-12_000:]
            result["failure_kind"] = classify_profile_failure(exc, logs)
        else:
            result["failure_kind"] = classify_profile_failure(exc, "")
    finally:
        if started and container is not None:
            stopped = run_process(
                [
                    docker_executable(),
                    "stop",
                    "--time",
                    str(DEFAULT_STOP_TIMEOUT_SEC),
                    container["name"],
                ],
                timeout_sec=DEFAULT_STOP_TIMEOUT_SEC + 30,
                allow_failure=True,
            )
            result["container_stopped"] = stopped.returncode == 0
            if stopped.returncode != 0:
                result["status"] = "failed"
                result["stop_failure"] = stopped.stderr.strip()
        result["completed_at_unix"] = time.time()
    return result


def next_ngl_probe_values(
    *,
    initial_ngl: int,
    max_ngl: int,
    initial_passed: bool,
    initial_swap_only_failure: bool = False,
) -> list[int]:
    if initial_ngl < 0 or max_ngl < initial_ngl:
        raise ProtocolError("Invalid NGL probe bounds")
    if initial_passed or initial_swap_only_failure:
        return list(range(initial_ngl + 1, max_ngl + 1))
    return list(range(initial_ngl - 1, -1, -1))


def tune_profile_ngl(
    *,
    manifest: BenchmarkManifest,
    inventory: Mapping[str, Any],
    profile: Profile,
    max_ngl: int,
    output_path: Path,
    start_timeout_sec: int,
    request_timeout_sec: int,
) -> dict[str, Any]:
    """Find the load/generation-safe boundary without deleting containers."""

    output = _require_external_path(output_path, label="NGL tuning output")
    if output.exists():
        raise FileExistsError(f"NGL tuning output already exists: {output}")
    target_inventory = inventory["artifacts"][profile.target_id]
    target_selected_metadata = (
        (target_inventory.get("gguf") or {}).get("selected") or {}
    )
    block_count = int(
        target_selected_metadata.get("gemma4.block_count", 0) or 0
    )
    meaningful_max_ngl = block_count + 1 if block_count > 0 else max_ngl
    effective_max_ngl = min(max_ngl, meaningful_max_ngl)
    if effective_max_ngl < profile.target_ngl:
        raise ProtocolError("max_ngl must be at least the initial target NGL")

    attempts: list[dict[str, Any]] = []

    def attempt(target_ngl: int, draft_ngl: str) -> dict[str, Any]:
        candidate = replace(
            profile,
            target_ngl=target_ngl,
            draft_ngl=draft_ngl,
        )
        result = probe_profile_load(
            manifest=manifest,
            inventory=inventory,
            profile=candidate,
            start_timeout_sec=start_timeout_sec,
            request_timeout_sec=request_timeout_sec,
        )
        attempts.append(result)
        return result

    draft_modes = [profile.draft_ngl or ""]
    if profile.speculation == "mtp" and profile.draft_ngl != "0":
        draft_modes.append("0")

    selected_draft_ngl = ""
    safe_ngls: list[int] = []
    first_failed_ngl: int | None = None
    for draft_mode in draft_modes:
        initial = attempt(profile.target_ngl, draft_mode)
        initial_passed = initial["status"] == "passed"
        if (
            initial.get("failure_kind") == "draft_model_load"
            and draft_mode != "0"
        ):
            continue
        initial_resource_gates = initial.get("resource_gates") or {}
        initial_swap_only_failure = bool(
            not initial_passed
            and initial_resource_gates
            and not initial_resource_gates.get("swap_growth_ok", True)
            and initial_resource_gates.get("shared_gpu_growth_ok", False)
        )
        search_upward = initial_passed or initial_swap_only_failure
        if initial_passed:
            safe_ngls.append(profile.target_ngl)
        probe_values = next_ngl_probe_values(
            initial_ngl=profile.target_ngl,
            max_ngl=effective_max_ngl,
            initial_passed=initial_passed,
            initial_swap_only_failure=initial_swap_only_failure,
        )
        for ngl in probe_values:
            probed = attempt(ngl, draft_mode)
            if probed["status"] == "passed":
                safe_ngls.append(ngl)
                if not search_upward:
                    break
            elif search_upward and safe_ngls:
                first_failed_ngl = ngl
                break
        if safe_ngls:
            selected_draft_ngl = draft_mode
            break

    safe_ngls = sorted(set(safe_ngls))
    safe_max = max(safe_ngls) if safe_ngls else None
    comparison_ngls = (
        sorted({safe_max, max(0, safe_max - 1)})
        if safe_max is not None
        else []
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "profile_id": profile.id,
        "initial_target_ngl": profile.target_ngl,
        "requested_max_target_ngl": max_ngl,
        "model_block_count": block_count or None,
        "effective_max_target_ngl": effective_max_ngl,
        "selected_draft_ngl": selected_draft_ngl,
        "safe_target_ngls": safe_ngls,
        "safe_max_target_ngl": safe_max,
        "first_failed_target_ngl": first_failed_ngl,
        "screen_comparison_target_ngls": comparison_ngls,
        "status": "passed" if safe_max is not None else "failed",
        "attempts": attempts,
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    _atomic_write_json(output, payload)
    return payload


def _manifest_summary(manifest: BenchmarkManifest) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "image": manifest.image,
        "expected_image_id": manifest.expected_image_id,
        "baseline_target": manifest.baseline.id,
        "target_ids": sorted(manifest.targets),
        "draft_ids": sorted(manifest.drafts),
        "profile_ids": [profile.id for profile in enumerate_profiles(manifest)],
        "volumes": {
            volume.id: {
                "name": volume.name,
                "managed": volume.managed,
            }
            for volume in manifest.volumes.values()
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "External-data llama.cpp Gemma target/MTP/ngram tournament runner"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_manifest(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--model-manifest", required=True)

    validate = subparsers.add_parser("validate-manifest")
    add_manifest(validate)

    inventory = subparsers.add_parser("inventory")
    add_manifest(inventory)
    inventory.add_argument("--output", required=True)

    profiles = subparsers.add_parser("list-profiles")
    add_manifest(profiles)
    profiles.add_argument("--inventory-lock")

    prepare = subparsers.add_parser("prepare-managed-volumes")
    add_manifest(prepare)
    prepare.add_argument("--inventory-lock", required=True)

    verify = subparsers.add_parser("verify-volumes")
    add_manifest(verify)
    verify.add_argument("--inventory-lock", required=True)
    verify.add_argument("--full-hash", action="store_true")

    preflight = subparsers.add_parser("preflight")
    add_manifest(preflight)
    preflight.add_argument("--inventory-lock", required=True)
    preflight.add_argument("--corpus", required=True)

    tune = subparsers.add_parser("tune-ngl")
    add_manifest(tune)
    tune.add_argument("--inventory-lock", required=True)
    tune.add_argument("--profile", required=True)
    tune.add_argument("--max-ngl", type=int, default=40)
    tune.add_argument("--output", required=True)
    tune.add_argument("--target-ngl", type=int)
    tune.add_argument("--draft-ngl")
    tune.add_argument("--start-timeout-sec", type=int, default=DEFAULT_START_TIMEOUT_SEC)
    tune.add_argument(
        "--request-timeout-sec",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_SEC,
    )

    run = subparsers.add_parser("run-profile")
    add_manifest(run)
    run.add_argument("--inventory-lock", required=True)
    run.add_argument("--corpus", required=True)
    run.add_argument("--profile", required=True)
    run.add_argument(
        "--stage",
        choices=(
            "smoke",
            "sensitive15",
            "screen18",
            "final54",
            "breakeven6",
            "breakeven15",
            "breakeven30",
            "breakeven54",
        ),
        required=True,
    )
    run.add_argument("--round", type=int, required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--target-ngl", type=int)
    run.add_argument("--draft-ngl")
    run.add_argument("--start-timeout-sec", type=int, default=DEFAULT_START_TIMEOUT_SEC)
    run.add_argument(
        "--request-timeout-sec",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_SEC,
    )

    compare = subparsers.add_parser("compare")
    compare.add_argument("--baseline", action="append", required=True)
    compare.add_argument("--candidate", action="append", required=True)
    compare.add_argument("--output")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "compare":
        summary = compare_profile_results(
            baseline_paths=[Path(value) for value in args.baseline],
            candidate_paths=[Path(value) for value in args.candidate],
        )
        if args.output:
            output = _require_external_path(Path(args.output), label="comparison output")
            _atomic_write_json(output, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    manifest = load_manifest(Path(args.model_manifest))
    if args.command == "validate-manifest":
        print(json.dumps(_manifest_summary(manifest), ensure_ascii=False, indent=2))
        return 0
    if args.command == "inventory":
        payload = build_inventory(manifest, output_path=Path(args.output))
        print(
            json.dumps(
                {
                    "inventory_lock": str(Path(args.output).resolve()),
                    "lock_sha256": payload["lock_sha256"],
                    "artifact_count": len(payload["artifacts"]),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    inventory = load_inventory_lock(manifest, Path(args.inventory_lock))
    if args.command == "list-profiles":
        print(
            json.dumps(
                [asdict(profile) for profile in enumerate_profiles(manifest)],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "prepare-managed-volumes":
        print(
            json.dumps(
                prepare_managed_volumes(manifest, inventory),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "verify-volumes":
        print(
            json.dumps(
                verify_volume_artifacts(
                    manifest,
                    inventory,
                    full_hash=bool(args.full_hash),
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "tune-ngl":
        profile = find_profile(
            manifest,
            args.profile,
            target_ngl=args.target_ngl,
            draft_ngl=args.draft_ngl,
        )
        payload = tune_profile_ngl(
            manifest=manifest,
            inventory=inventory,
            profile=profile,
            max_ngl=int(args.max_ngl),
            output_path=Path(args.output),
            start_timeout_sec=int(args.start_timeout_sec),
            request_timeout_sec=int(args.request_timeout_sec),
        )
        print(
            json.dumps(
                {
                    key: payload[key]
                    for key in (
                        "status",
                        "profile_id",
                        "selected_draft_ngl",
                        "safe_max_target_ngl",
                        "first_failed_target_ngl",
                        "screen_comparison_target_ngls",
                    )
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    corpus = load_corpus(Path(args.corpus))
    if args.command == "preflight":
        output = {
            "manifest": _manifest_summary(manifest),
            "inventory_lock_sha256": inventory["lock_sha256"],
            "corpus_sha256": corpus["corpus_sha256"],
            "volumes": verify_volume_artifacts(
                manifest,
                inventory,
                full_hash=False,
            ),
            "environment": preflight_environment(manifest),
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-profile":
        profile = find_profile(
            manifest,
            args.profile,
            target_ngl=args.target_ngl,
            draft_ngl=args.draft_ngl,
        )
        result = run_profile_stage(
            manifest=manifest,
            inventory=inventory,
            corpus=corpus,
            profile=profile,
            stage=args.stage,
            round_index=int(args.round),
            output_dir=Path(args.output_dir),
            start_timeout_sec=int(args.start_timeout_sec),
            request_timeout_sec=int(args.request_timeout_sec),
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "profile": profile.id,
                    "stage": args.stage,
                    "round": args.round,
                    "request_only_elapsed_sec": result["request_only_elapsed_sec"],
                    "total_elapsed_sec": result["total_elapsed_sec"],
                    "gates": result["gates"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
