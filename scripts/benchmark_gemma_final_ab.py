#!/usr/bin/env python3
"""Build and validate the protocol-v4 blind Gemma A/B quality review.

This tool is deliberately report-only.  It reads a locked protocol-v3 suite,
revalidates every source contract and result digest, and writes local blind
review artifacts.  It does not import Docker, open a network connection, or
make a model request.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import html
import json
import os
from pathlib import Path
import re
import secrets
import statistics
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]

SOURCE_PROTOCOL_VERSION = 3
REPORT_PROTOCOL_VERSION = 4
EXPECTED_PAGE_COUNT = 22
EXPECTED_BLOCK_COUNT = 292
EXPECTED_MODEL_NAME = (
    "Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ4_XS.gguf"
)
EXPECTED_MODEL_SIZE_BYTES = 13_917_726_048
EXPECTED_MODEL_SHA256 = (
    "61b277f4dde555fc6c04c9024a9580ef8c83f2f19504f3989a15f95684257426"
)
EXPECTED_IMAGE_ID = (
    "sha256:22e0e3bfe967af4fd1df6a918022abbfd4e72e4d40a4769e616a4176790acbcb"
)
EXPECTED_HASH_HELPER_IMAGE_ID = (
    "sha256:14358309a308569c32bdc37e2e0e9694be33a9d99e68afb0f5ff33cc1f695dce"
)
EXPECTED_INPUT_MANIFEST_SHA256 = (
    "63cdfa53fc7c48efa9e6f1f11aae3e86bb5ea0aadcba361491ee34bc94cc9b1e"
)
EXPECTED_OCR_SNAPSHOT_SHA256 = (
    "22fd706b63da75a5a4c7f4175cec6d23f9b9e9a831e82365329fca42e1c84605"
)
EXPECTED_SNAPSHOT_CONTRACT_SHA256 = (
    "57e9bf710c09b72903b3c7ae4ea4c5487618fc691f2ce9aa8a5dc4415eaaf793"
)
EXPECTED_TRANSLATION_CONTRACT_SHA256 = (
    "b7d702ad7d3ca32a3144c60fb8247a5f5e2f7854e3656f00df8d7823e07134b5"
)
EXPECTED_SOURCE_SUITE_FINGERPRINT = (
    "33980d453a26bf2f0034f04c62f7d76ca5a9d5268f12fc484fa0fe9813c49992"
)

CANDIDATE_BASELINE = "current-contextual-single"
CANDIDATE_GROUPED_F16 = "grouped-f16"
CANDIDATE_GROUPED_Q8 = "grouped-q8"
SOURCE_CANDIDATES = (
    CANDIDATE_BASELINE,
    CANDIDATE_GROUPED_F16,
    CANDIDATE_GROUPED_Q8,
)
IMPORTED_CANDIDATES = (
    CANDIDATE_BASELINE,
    CANDIDATE_GROUPED_F16,
)
IMPORTED_RUN_KEYS = (
    (1, CANDIDATE_BASELINE),
    (1, CANDIDATE_GROUPED_F16),
    (2, CANDIDATE_BASELINE),
    (2, CANDIDATE_GROUPED_F16),
)
SOURCE_RUN_ORDER = (
    (1, CANDIDATE_BASELINE),
    (1, CANDIDATE_GROUPED_F16),
    (1, CANDIDATE_GROUPED_Q8),
    (2, CANDIDATE_GROUPED_F16),
    (2, CANDIDATE_GROUPED_Q8),
    (2, CANDIDATE_BASELINE),
    (3, CANDIDATE_GROUPED_Q8),
)
SOURCE_RUN_FILENAMES = {
    (round_index, candidate): (
        f"runs/round-{round_index}_{candidate}.json"
    )
    for round_index, candidate in SOURCE_RUN_ORDER
}
LOCKED_RUN_SHA256 = {
    (1, CANDIDATE_BASELINE): (
        "b5fd4b0f702e9222bd82f6d3be38834eeee171934ddf7434faa9cfc1f77d90ce"
    ),
    (1, CANDIDATE_GROUPED_F16): (
        "39da91857d613ff73fca60bec2e0e65c32326d60e1417fe7ed96af1abaa4fdfc"
    ),
    (1, CANDIDATE_GROUPED_Q8): (
        "e9cfb4902ea3ba8e6c5c2d5dfcdfa5c046323f3820885e0a2e8e9b72626fc94f"
    ),
    (2, CANDIDATE_GROUPED_F16): (
        "8cf729c8e4d2718da7cbcf411e97f4772b229741abbc94fdd41f59f1ccd3ddb1"
    ),
    (2, CANDIDATE_GROUPED_Q8): (
        "d4331b4002adca80089411007fda4f3edfbd4f8e147f7c50b12c40c5765513bf"
    ),
    (2, CANDIDATE_BASELINE): (
        "84787671a3bafa37afb3a9c237d0d17266c8c3e1c52d5750f3e89400ddc52cd4"
    ),
    (3, CANDIDATE_GROUPED_Q8): (
        "6a2497e5a0a57ba9a0d08f5746fb531e713a9c82551a438b99b160971eacc7ae"
    ),
}
EXPECTED_RUNTIME_FINGERPRINTS = {
    CANDIDATE_BASELINE: (
        "628535e4d71e1c9015cd3e5c0275113c621039e176c79a7b3db901688dab38ae"
    ),
    CANDIDATE_GROUPED_F16: (
        "e6d83a374327b5e242dc0fa1b7cf59fdbd5a89f2830484829639f75089816128"
    ),
    CANDIDATE_GROUPED_Q8: (
        "eb765b6b40cb9bbfbf4af7bdb6f8ba767e4cea75c965b10c9d237fa8f31d9a27"
    ),
}
EXPECTED_RUNTIME_PROFILE_LABELS = {
    CANDIDATE_BASELINE: (
        "26b-iq4xs_ngram_f16_ngl23_ctx4096_t10_cache0_draft8"
    ),
    CANDIDATE_GROUPED_F16: (
        "26b-iq4xs_none_f16_ngl23_ctx4096_t10_cache0"
    ),
    CANDIDATE_GROUPED_Q8: (
        "26b-iq4xs_none_q8_ngl23_ctx4096_t10_cache0"
    ),
}

REVIEW_COLUMNS = (
    "row",
    "row_id",
    "page",
    "block",
    "source",
    "A1",
    "A2",
    "B1",
    "B2",
    "A1_regression",
    "A2_regression",
    "B1_regression",
    "B2_regression",
    "regression_types",
    "notes",
)
REVIEW_FLAG_COLUMNS = (
    "A1_regression",
    "A2_regression",
    "B1_regression",
    "B2_regression",
)
REGRESSION_TYPES = (
    "speaker",
    "relationship",
    "negation",
    "action",
    "target",
    "number",
    "proper_noun",
    "explicit_meaning",
)
REGRESSION_TYPE_LABELS = {
    "speaker": "화자",
    "relationship": "관계",
    "negation": "부정",
    "action": "행동",
    "target": "대상",
    "number": "숫자",
    "proper_noun": "고유명사",
    "explicit_meaning": "명시적 의미",
}

PRIVATE_DIRNAME = "private"
STATE_FILENAME = "protocol_v4_state.json"
KEY_FILENAME = "blind_key.json"
PAYLOAD_FILENAME = "blind_payload.json"
AUDIT_FILENAME = "source_import_audit.json"
REVIEW_FILENAME = "blind_review.csv"
REVIEW_HTML_FILENAME = "blind_review.html"
REVIEW_INSTRUCTIONS_FILENAME = "review_instructions.md"
REVIEW_VALIDATION_FILENAME = "review_validation.json"
UNBLIND_SUMMARY_FILENAME = "unblind_summary.json"
UNBLIND_MARKDOWN_FILENAME = "unblind_summary.md"


@dataclass(frozen=True)
class SourceContract:
    protocol_version: int
    page_count: int
    block_count: int
    suite_fingerprint: str
    input_manifest_sha256: str
    snapshot_sha256: str
    snapshot_contract_sha256: str
    model_name: str
    model_size_bytes: int
    model_sha256: str
    image_id: str
    hash_helper_image_id: str
    translation_contract_sha256: str
    runtime_fingerprints: Mapping[str, str]
    run_order: tuple[tuple[int, str], ...]
    run_filenames: Mapping[tuple[int, str], str]
    run_sha256: Mapping[tuple[int, str], str]
    round_difference_counts: Mapping[str, int]
    q8_slowdown_percent: float


LOCKED_SOURCE_CONTRACT = SourceContract(
    protocol_version=SOURCE_PROTOCOL_VERSION,
    page_count=EXPECTED_PAGE_COUNT,
    block_count=EXPECTED_BLOCK_COUNT,
    suite_fingerprint=EXPECTED_SOURCE_SUITE_FINGERPRINT,
    input_manifest_sha256=EXPECTED_INPUT_MANIFEST_SHA256,
    snapshot_sha256=EXPECTED_OCR_SNAPSHOT_SHA256,
    snapshot_contract_sha256=EXPECTED_SNAPSHOT_CONTRACT_SHA256,
    model_name=EXPECTED_MODEL_NAME,
    model_size_bytes=EXPECTED_MODEL_SIZE_BYTES,
    model_sha256=EXPECTED_MODEL_SHA256,
    image_id=EXPECTED_IMAGE_ID,
    hash_helper_image_id=EXPECTED_HASH_HELPER_IMAGE_ID,
    translation_contract_sha256=EXPECTED_TRANSLATION_CONTRACT_SHA256,
    runtime_fingerprints=EXPECTED_RUNTIME_FINGERPRINTS,
    run_order=SOURCE_RUN_ORDER,
    run_filenames=SOURCE_RUN_FILENAMES,
    run_sha256=LOCKED_RUN_SHA256,
    round_difference_counts={
        CANDIDATE_BASELINE: 124,
        CANDIDATE_GROUPED_F16: 133,
    },
    q8_slowdown_percent=4.730,
)


class ReviewIncompleteError(ValueError):
    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        super().__init__(
            f"Blind review is incomplete or invalid ({len(self.errors)} errors)"
        )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json_with_sha256(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read JSON object: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return payload, hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload, _digest = read_json_with_sha256(path)
    return payload


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding=encoding)
    os.replace(temporary, path)


def write_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def _write_private_json(
    path: Path,
    payload: Mapping[str, Any] | list[Any],
) -> None:
    write_json(path, payload)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _validated_sha256(value: Any, *, label: str) -> str:
    normalized = str(value or "").strip().casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"{label} must be a 64-character SHA-256 digest")
    return normalized


def _expect_equal(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise ValueError(
            f"{label} differs (expected={expected!r}, actual={actual!r})"
        )


def _as_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _as_list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _relative_file(
    root: Path,
    relative: str,
    *,
    allowed_root: Path,
    label: str,
) -> Path:
    candidate_relative = Path(str(relative or ""))
    if candidate_relative.is_absolute():
        raise ValueError(f"{label} must be relative")
    candidate = (root / candidate_relative).resolve()
    try:
        candidate.relative_to(allowed_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes its allowed directory") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} is missing: {candidate_relative}")
    return candidate


def _runtime_command(candidate: str, model_name: str) -> list[str]:
    kv_type = "q8_0" if candidate == CANDIDATE_GROUPED_Q8 else "f16"
    command = [
        "-m",
        f"/models/{model_name}",
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "-c",
        "4096",
        "-np",
        "1",
        "-t",
        "10",
        "--n-gpu-layers",
        "23",
        "--fit",
        "off",
        "-fa",
        "on",
        "-ctk",
        kv_type,
        "-ctv",
        kv_type,
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
    if candidate == CANDIDATE_BASELINE:
        command.extend(
            [
                "--spec-type",
                "ngram-mod",
                "--spec-draft-n-max",
                "8",
            ]
        )
    return command


def _validate_source_state(
    state: Mapping[str, Any],
    *,
    contract: SourceContract,
) -> Mapping[str, Any]:
    _expect_equal(state.get("status"), "failed", label="source suite status")
    _expect_equal(
        state.get("quality_status"),
        "pending_user_review",
        label="source quality status",
    )
    _expect_equal(
        state.get("full_pipeline_executed"),
        False,
        label="source full-pipeline flag",
    )
    suite_contract = _as_mapping(
        state.get("suite_contract"),
        label="source suite contract",
    )
    suite_fingerprint = canonical_sha256(suite_contract)
    _expect_equal(
        suite_fingerprint,
        contract.suite_fingerprint,
        label="source suite contract SHA-256",
    )
    _expect_equal(
        state.get("contract_fingerprint"),
        suite_fingerprint,
        label="source suite fingerprint",
    )
    expected_contract_values = {
        "benchmark_protocol_version": contract.protocol_version,
        "input_manifest_sha256": contract.input_manifest_sha256,
        "snapshot_sha256": contract.snapshot_sha256,
        "snapshot_contract_sha256": contract.snapshot_contract_sha256,
        "model_sha256": contract.model_sha256,
        "image_id": contract.image_id,
        "hash_helper_image_id": contract.hash_helper_image_id,
        "translation_behavior_contract_sha256": (
            contract.translation_contract_sha256
        ),
        "group_size": 7,
        "max_completion_tokens": 512,
    }
    for key, expected in expected_contract_values.items():
        _expect_equal(
            suite_contract.get(key),
            expected,
            label=f"source suite contract {key}",
        )
    runtime_fingerprints = _as_mapping(
        suite_contract.get("runtime_config_fingerprints"),
        label="source runtime fingerprints",
    )
    _expect_equal(
        dict(runtime_fingerprints),
        dict(contract.runtime_fingerprints),
        label="source runtime fingerprints",
    )

    behavior = _as_mapping(
        state.get("translation_behavior_contract"),
        label="translation behavior contract",
    )
    behavior_without_digest = dict(behavior)
    recorded_behavior_digest = _validated_sha256(
        behavior_without_digest.pop("contract_sha256", ""),
        label="translation behavior contract digest",
    )
    _expect_equal(
        canonical_sha256(behavior_without_digest),
        recorded_behavior_digest,
        label="translation behavior canonical digest",
    )
    _expect_equal(
        recorded_behavior_digest,
        contract.translation_contract_sha256,
        label="translation behavior locked digest",
    )
    engine = _as_mapping(
        behavior.get("engine"),
        label="translation engine contract",
    )
    for field_name, expected in {
        "source_language": "Japanese",
        "target_language": "Korean",
        "max_completion_tokens": 512,
        "contextual_merge_input": True,
        "persistent_cache_enabled": False,
        "exact_tm_enabled": False,
    }.items():
        _expect_equal(
            engine.get(field_name),
            expected,
            label=f"translation engine {field_name}",
        )
    request_contract = _as_mapping(
        behavior.get("candidate_request_contract"),
        label="candidate request contract",
    )
    _expect_equal(
        request_contract.get(CANDIDATE_BASELINE),
        {"request_mode": "contextual-single", "group_size": 6},
        label="baseline request contract",
    )
    for candidate in (CANDIDATE_GROUPED_F16, CANDIDATE_GROUPED_Q8):
        _expect_equal(
            request_contract.get(candidate),
            {"request_mode": "contextual-grouped", "group_size": 7},
            label=f"{candidate} request contract",
        )
    return suite_contract


def _validate_frozen_assets(
    source_dir: Path,
    *,
    suite_contract: Mapping[str, Any],
    contract: SourceContract,
) -> list[Mapping[str, Any]]:
    assets = read_json(source_dir / "frozen_assets.json")
    _expect_equal(
        assets.get("expected_input_manifest_sha256"),
        contract.input_manifest_sha256,
        label="frozen expected input manifest SHA-256",
    )
    _expect_equal(
        assets.get("expected_ocr_snapshot_sha256"),
        contract.snapshot_sha256,
        label="frozen expected OCR snapshot SHA-256",
    )
    input_manifest = _as_mapping(
        assets.get("input_manifest"),
        label="frozen input manifest",
    )
    files = _as_list(
        input_manifest.get("files"),
        label="frozen input files",
    )
    _expect_equal(
        input_manifest.get("file_count"),
        contract.page_count,
        label="frozen input file count",
    )
    _expect_equal(
        len(files),
        contract.page_count,
        label="frozen input file records",
    )
    _expect_equal(
        canonical_sha256(files),
        contract.input_manifest_sha256,
        label="frozen input manifest canonical digest",
    )
    _expect_equal(
        input_manifest.get("manifest_sha256"),
        contract.input_manifest_sha256,
        label="frozen input manifest recorded digest",
    )

    snapshot = _as_mapping(
        assets.get("snapshot"),
        label="frozen snapshot contract",
    )
    for field_name, expected in {
        "snapshot_sha256": contract.snapshot_sha256,
        "contract_sha256": contract.snapshot_contract_sha256,
        "page_count": contract.page_count,
        "block_count": contract.block_count,
        "source_language": "Japanese",
        "target_language": "Korean",
    }.items():
        _expect_equal(
            snapshot.get(field_name),
            expected,
            label=f"frozen snapshot {field_name}",
        )
    page_contract = _as_list(
        snapshot.get("page_contract"),
        label="frozen page contract",
    )
    _expect_equal(
        len(page_contract),
        contract.page_count,
        label="frozen page contract count",
    )
    _expect_equal(
        canonical_sha256(page_contract),
        contract.snapshot_contract_sha256,
        label="frozen page contract canonical digest",
    )
    _expect_equal(
        suite_contract.get("input_manifest_sha256"),
        input_manifest.get("manifest_sha256"),
        label="suite/input manifest cross-check",
    )
    _expect_equal(
        suite_contract.get("snapshot_sha256"),
        snapshot.get("snapshot_sha256"),
        label="suite/snapshot cross-check",
    )
    _expect_equal(
        suite_contract.get("snapshot_contract_sha256"),
        snapshot.get("contract_sha256"),
        label="suite/snapshot contract cross-check",
    )
    return [
        _as_mapping(page, label=f"frozen page {index}")
        for index, page in enumerate(page_contract, start=1)
    ]


def _validate_model_contract(
    source_dir: Path,
    *,
    contract: SourceContract,
) -> Mapping[str, Any]:
    model = read_json(source_dir / "model_contract.json")
    for key, expected in {
        "model_name": contract.model_name,
        "expected_size_bytes": contract.model_size_bytes,
        "source_size_bytes": contract.model_size_bytes,
        "volume_size_bytes": contract.model_size_bytes,
        "expected_sha256": contract.model_sha256,
        "source_sha256": contract.model_sha256,
        "volume_sha256": contract.model_sha256,
        "model_copy_verified": True,
    }.items():
        _expect_equal(model.get(key), expected, label=f"model contract {key}")
    helper = _as_mapping(
        model.get("hash_helper_image"),
        label="model hash helper contract",
    )
    _expect_equal(
        helper.get("image_id"),
        contract.hash_helper_image_id,
        label="model hash helper image ID",
    )
    return model


def _validate_runtime_contracts(
    source_dir: Path,
    *,
    suite_contract: Mapping[str, Any],
    model_contract: Mapping[str, Any],
    contract: SourceContract,
) -> dict[str, Mapping[str, Any]]:
    payload = json.loads(
        (source_dir / "runtime_contracts.json").read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    runtime_items = _as_list(payload, label="runtime contracts")
    _expect_equal(
        [str(item.get("candidate") or "") for item in runtime_items],
        list(SOURCE_CANDIDATES),
        label="runtime candidate order",
    )
    runtime_by_candidate: dict[str, Mapping[str, Any]] = {}
    model_volume = str(model_contract.get("volume_name") or "")
    for raw_item in runtime_items:
        item = _as_mapping(raw_item, label="runtime contract item")
        candidate = str(item.get("candidate") or "")
        if candidate in runtime_by_candidate:
            raise ValueError(f"Duplicate runtime candidate: {candidate}")
        runtime_by_candidate[candidate] = item
        for key, expected in {
            "image_id": contract.image_id,
            "config_fingerprint": contract.runtime_fingerprints[candidate],
            "profile_label": EXPECTED_RUNTIME_PROFILE_LABELS[candidate],
            "model_volume": model_volume,
            "model_mount_read_only": True,
            "network_ports": {},
            "command": _runtime_command(candidate, contract.model_name),
        }.items():
            _expect_equal(
                item.get(key),
                expected,
                label=f"{candidate} runtime {key}",
            )
        if not str(item.get("container_id") or ""):
            raise ValueError(f"{candidate} runtime container ID is missing")
    _expect_equal(
        {
            candidate: str(item.get("config_fingerprint") or "")
            for candidate, item in runtime_by_candidate.items()
        },
        dict(suite_contract.get("runtime_config_fingerprints") or {}),
        label="runtime/suite fingerprint cross-check",
    )
    return runtime_by_candidate


def _validate_measurement_environment(
    source_dir: Path,
    *,
    suite_contract: Mapping[str, Any],
) -> None:
    environment = read_json(source_dir / "measurement_environment.json")
    _expect_equal(
        environment,
        suite_contract.get("measurement_environment"),
        label="measurement environment contract",
    )
    docker_server = _as_mapping(
        environment.get("docker_server"),
        label="Docker measurement identity",
    )
    nvidia_driver = _as_mapping(
        environment.get("nvidia_driver"),
        label="NVIDIA measurement identity",
    )
    _expect_equal(
        docker_server.get("available"),
        True,
        label="Docker measurement availability",
    )
    _expect_equal(
        nvidia_driver.get("available"),
        True,
        label="NVIDIA measurement availability",
    )


def _output_identity(output: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(output.get("row_id") or ""),
        int(output.get("page_index") or 0),
        str(output.get("page_name") or ""),
        int(output.get("block_index") or 0),
        str(output.get("source") or ""),
        str(output.get("source_sha256") or ""),
    )


def _validate_output(
    output: Mapping[str, Any],
    *,
    label: str,
) -> None:
    source = str(output.get("source") or "")
    translation = str(output.get("translation") or "")
    if not source:
        raise ValueError(f"{label} source is empty")
    if not translation.strip():
        raise ValueError(f"{label} translation is empty")
    source_digest = hashlib.sha256(
        source.encode("utf-8", errors="surrogatepass")
    ).hexdigest()
    translation_digest = hashlib.sha256(
        translation.encode("utf-8", errors="surrogatepass")
    ).hexdigest()
    _expect_equal(
        output.get("source_sha256"),
        source_digest,
        label=f"{label} source SHA-256",
    )
    _expect_equal(
        output.get("translation_sha256"),
        translation_digest,
        label=f"{label} translation SHA-256",
    )
    _expect_equal(output.get("empty"), False, label=f"{label} empty flag")
    _expect_equal(
        output.get("structural_output"),
        False,
        label=f"{label} structural flag",
    )


def _validate_run_result(
    result: Mapping[str, Any],
    *,
    round_index: int,
    candidate: str,
    result_digest: str,
    runtime: Mapping[str, Any],
    contract: SourceContract,
) -> list[Mapping[str, Any]]:
    _expect_equal(result.get("round"), round_index, label="run round")
    _expect_equal(result.get("candidate"), candidate, label="run candidate")
    _expect_equal(
        result.get("contract_fingerprint"),
        contract.suite_fingerprint,
        label="run source contract fingerprint",
    )
    _expect_equal(
        result.get("container_id"),
        runtime.get("container_id"),
        label=f"round {round_index} {candidate} container ID",
    )
    _expect_equal(
        result.get("container_name"),
        runtime.get("container_name"),
        label=f"round {round_index} {candidate} container name",
    )
    _expect_equal(
        result.get("container_stopped"),
        True,
        label=f"round {round_index} {candidate} stop status",
    )
    _expect_equal(
        result.get("page_count"),
        contract.page_count,
        label=f"round {round_index} {candidate} page count",
    )
    _expect_equal(
        result.get("block_count"),
        contract.block_count,
        label=f"round {round_index} {candidate} block count",
    )
    expected_mode = (
        "contextual-single"
        if candidate == CANDIDATE_BASELINE
        else "contextual-grouped"
    )
    expected_group = 6 if candidate == CANDIDATE_BASELINE else 7
    _expect_equal(
        result.get("request_mode"),
        expected_mode,
        label=f"round {round_index} {candidate} request mode",
    )
    _expect_equal(
        result.get("configured_group_size"),
        expected_group,
        label=f"round {round_index} {candidate} group size",
    )
    elapsed = float(result.get("translation_elapsed_sec") or 0.0)
    if elapsed <= 0:
        raise ValueError(f"round {round_index} {candidate} elapsed time is invalid")
    expected_status = (
        "failed"
        if (round_index, candidate) == (3, CANDIDATE_GROUPED_Q8)
        else "passed"
    )
    _expect_equal(
        result.get("status"),
        expected_status,
        label=f"round {round_index} {candidate} status",
    )
    gates = _as_mapping(
        result.get("gates"),
        label=f"round {round_index} {candidate} gates",
    )
    if expected_status == "passed":
        for key in (
            "page_count_ok",
            "block_count_ok",
            "order_preserved",
            "request_contract_passed",
            "hard_gate_passed",
            "clean_run_passed",
        ):
            _expect_equal(
                gates.get(key),
                True,
                label=f"round {round_index} {candidate} gate {key}",
            )
        for key in (
            "empty_translation_count",
            "structural_output_count",
            "structural_telemetry_count",
            "unresolved_failure_count",
            "clean_run_telemetry_count",
        ):
            _expect_equal(
                int(gates.get(key, -1)),
                0,
                label=f"round {round_index} {candidate} gate {key}",
            )
    else:
        _expect_equal(
            gates.get("hard_gate_passed"),
            False,
            label="Q8 failed hard gate",
        )
        _expect_equal(
            gates.get("clean_run_passed"),
            False,
            label="Q8 failed clean gate",
        )

    stats = _as_mapping(
        result.get("stats"),
        label=f"round {round_index} {candidate} telemetry",
    )
    for key in (
        "gemma_tm_result_cache_hit_count",
        "gemma_tm_exact_hit_count",
        "gemma_tm_cache_write_count",
        "gemma_tm_runtime_skipped_count",
    ):
        _expect_equal(
            int(stats.get(key, -1)),
            0,
            label=f"round {round_index} {candidate} telemetry {key}",
        )
    outputs = [
        _as_mapping(output, label=f"{candidate} output {index}")
        for index, output in enumerate(
            _as_list(
                result.get("outputs"),
                label=f"round {round_index} {candidate} outputs",
            ),
            start=1,
        )
    ]
    _expect_equal(
        len(outputs),
        contract.block_count,
        label=f"round {round_index} {candidate} output count",
    )
    for index, output in enumerate(outputs, start=1):
        _validate_output(
            output,
            label=f"round {round_index} {candidate} output {index}",
        )
    if len({_output_identity(output)[0] for output in outputs}) != len(outputs):
        raise ValueError(
            f"round {round_index} {candidate} contains duplicate row IDs"
        )
    _expect_equal(
        result_digest,
        contract.run_sha256[(round_index, candidate)],
        label=f"round {round_index} {candidate} locked result digest",
    )
    return outputs


def _validate_output_order(
    results: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    page_contract: Sequence[Mapping[str, Any]],
    contract: SourceContract,
) -> None:
    reference_outputs = _as_list(
        results[(1, CANDIDATE_BASELINE)].get("outputs"),
        label="baseline round-1 outputs",
    )
    reference_identity = [
        _output_identity(_as_mapping(output, label="reference output"))
        for output in reference_outputs
    ]
    for run_key in IMPORTED_RUN_KEYS:
        outputs = _as_list(
            results[run_key].get("outputs"),
            label=f"{run_key} outputs",
        )
        identities = [
            _output_identity(_as_mapping(output, label=f"{run_key} output"))
            for output in outputs
        ]
        _expect_equal(
            identities,
            reference_identity,
            label=f"{run_key} output order",
        )

    expected_source_digests: list[str] = []
    expected_page_metadata: list[tuple[int, str, int]] = []
    total_blocks = 0
    for expected_page_index, page in enumerate(page_contract, start=1):
        page_index = int(page.get("page_index") or 0)
        page_name = str(page.get("page_name") or "")
        block_count = int(page.get("block_count") or 0)
        _expect_equal(
            page_index,
            expected_page_index,
            label="frozen page order",
        )
        source_digests = [
            str(value)
            for value in _as_list(
                page.get("ordered_source_sha256"),
                label=f"frozen page {page_index} source order",
            )
        ]
        _expect_equal(
            len(source_digests),
            block_count,
            label=f"frozen page {page_index} block order count",
        )
        expected_source_digests.extend(source_digests)
        expected_page_metadata.extend(
            (page_index, page_name, block_index)
            for block_index in range(1, block_count + 1)
        )
        total_blocks += block_count
    _expect_equal(
        total_blocks,
        contract.block_count,
        label="frozen total block count",
    )
    _expect_equal(
        [identity[5] for identity in reference_identity],
        expected_source_digests,
        label="frozen/result source digest order",
    )
    _expect_equal(
        [(identity[1], identity[2], identity[3]) for identity in reference_identity],
        expected_page_metadata,
        label="frozen/result page and block order",
    )


def validate_source_suite(
    source_dir: Path,
    *,
    contract: SourceContract = LOCKED_SOURCE_CONTRACT,
) -> dict[str, Any]:
    source_dir = source_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source suite directory is missing: {source_dir}")
    state = read_json(source_dir / "suite_state.json")
    suite_contract = _validate_source_state(state, contract=contract)
    page_contract = _validate_frozen_assets(
        source_dir,
        suite_contract=suite_contract,
        contract=contract,
    )
    model_contract = _validate_model_contract(source_dir, contract=contract)
    runtime_by_candidate = _validate_runtime_contracts(
        source_dir,
        suite_contract=suite_contract,
        model_contract=model_contract,
        contract=contract,
    )
    _validate_measurement_environment(
        source_dir,
        suite_contract=suite_contract,
    )

    run_records = _as_list(state.get("runs"), label="source run records")
    actual_run_order = [
        (int(record.get("round") or 0), str(record.get("candidate") or ""))
        for record in run_records
        if isinstance(record, Mapping)
    ]
    _expect_equal(
        actual_run_order,
        list(contract.run_order),
        label="source run record order",
    )
    results: dict[tuple[int, str], Mapping[str, Any]] = {}
    verified_run_digests: dict[str, str] = {}
    runs_root = (source_dir / "runs").resolve()
    for raw_record, run_key in zip(run_records, contract.run_order):
        record = _as_mapping(raw_record, label=f"source run record {run_key}")
        expected_relative = contract.run_filenames[run_key]
        _expect_equal(
            record.get("result_file"),
            expected_relative,
            label=f"{run_key} result path",
        )
        result_path = _relative_file(
            source_dir,
            expected_relative,
            allowed_root=runs_root,
            label=f"{run_key} result",
        )
        result, actual_digest = read_json_with_sha256(result_path)
        recorded_digest = _validated_sha256(
            record.get("result_sha256"),
            label=f"{run_key} recorded result digest",
        )
        _expect_equal(
            actual_digest,
            recorded_digest,
            label=f"{run_key} result file digest",
        )
        _expect_equal(
            actual_digest,
            contract.run_sha256[run_key],
            label=f"{run_key} protocol-v4 locked result digest",
        )
        outputs = _validate_run_result(
            result,
            round_index=run_key[0],
            candidate=run_key[1],
            result_digest=actual_digest,
            runtime=runtime_by_candidate[run_key[1]],
            contract=contract,
        )
        _expect_equal(
            record.get("status"),
            result.get("status"),
            label=f"{run_key} state/result status",
        )
        _expect_equal(
            float(record.get("translation_elapsed_sec") or 0.0),
            float(result.get("translation_elapsed_sec") or 0.0),
            label=f"{run_key} state/result elapsed time",
        )
        result_gates = _as_mapping(
            result.get("gates"),
            label=f"{run_key} result gates",
        )
        _expect_equal(
            record.get("hard_gate_passed"),
            result_gates.get("hard_gate_passed"),
            label=f"{run_key} state/result hard gate",
        )
        _expect_equal(
            record.get("clean_run_passed"),
            result_gates.get("clean_run_passed"),
            label=f"{run_key} state/result clean gate",
        )
        _expect_equal(
            len(outputs),
            contract.block_count,
            label=f"{run_key} verified outputs",
        )
        results[run_key] = result
        verified_run_digests[
            f"round-{run_key[0]}:{run_key[1]}"
        ] = actual_digest
    _validate_output_order(
        results,
        page_contract=page_contract,
        contract=contract,
    )
    round_difference_counts: dict[str, int] = {}
    for candidate in IMPORTED_CANDIDATES:
        first_outputs = _as_list(
            results[(1, candidate)].get("outputs"),
            label=f"{candidate} round-1 outputs",
        )
        second_outputs = _as_list(
            results[(2, candidate)].get("outputs"),
            label=f"{candidate} round-2 outputs",
        )
        difference_count = sum(
            str(first.get("translation_sha256") or "")
            != str(second.get("translation_sha256") or "")
            for first, second in zip(first_outputs, second_outputs)
            if isinstance(first, Mapping) and isinstance(second, Mapping)
        )
        _expect_equal(
            difference_count,
            contract.round_difference_counts[candidate],
            label=f"{candidate} cross-round output difference count",
        )
        round_difference_counts[candidate] = difference_count

    imported_keys = set(IMPORTED_RUN_KEYS)
    if any(
        candidate == CANDIDATE_GROUPED_Q8
        for _round_index, candidate in imported_keys
    ):
        raise ValueError("Q8 must never be included in protocol-v4 candidates")
    if imported_keys != {
        (round_index, candidate)
        for round_index in (1, 2)
        for candidate in IMPORTED_CANDIDATES
    }:
        raise ValueError("Protocol-v4 imported run set is invalid")

    q8_failed = results[(3, CANDIDATE_GROUPED_Q8)]
    q8_stats = _as_mapping(q8_failed.get("stats"), label="Q8 failure telemetry")
    for key in (
        "gemma_partial_response_count",
        "gemma_partial_fallback_block_count",
        "gemma_invalid_value_count",
    ):
        if int(q8_stats.get(key, 0) or 0) < 1:
            raise ValueError(f"Q8 elimination evidence is missing: {key}")
    f16_times = [
        float(results[(round_index, CANDIDATE_GROUPED_F16)][
            "translation_elapsed_sec"
        ])
        for round_index in (1, 2)
    ]
    q8_times = [
        float(results[(round_index, CANDIDATE_GROUPED_Q8)][
            "translation_elapsed_sec"
        ])
        for round_index in (1, 2)
    ]
    f16_median = statistics.median(f16_times)
    q8_median = statistics.median(q8_times)
    q8_slowdown_percent = (q8_median - f16_median) / f16_median * 100.0
    if q8_slowdown_percent <= 0:
        raise ValueError("Q8 elimination timing evidence no longer shows a slowdown")
    if abs(q8_slowdown_percent - contract.q8_slowdown_percent) > 0.001:
        raise ValueError(
            "Q8 elimination timing evidence differs "
            f"(expected={contract.q8_slowdown_percent:.3f}%, "
            f"actual={q8_slowdown_percent:.3f}%)"
        )

    return {
        "source_dir": source_dir,
        "state": state,
        "suite_contract": suite_contract,
        "results": results,
        "verified_run_digests": verified_run_digests,
        "round_difference_counts": round_difference_counts,
        "q8_elimination": {
            "candidate": CANDIDATE_GROUPED_Q8,
            "status": "excluded",
            "f16_median_seconds": round(f16_median, 3),
            "q8_median_seconds": round(q8_median, 3),
            "q8_slowdown_percent": round(q8_slowdown_percent, 3),
            "round_3_partial_response_count": int(
                q8_stats["gemma_partial_response_count"]
            ),
            "round_3_partial_fallback_block_count": int(
                q8_stats["gemma_partial_fallback_block_count"]
            ),
            "round_3_invalid_value_count": int(
                q8_stats["gemma_invalid_value_count"]
            ),
        },
    }


def _csv_safe(value: Any) -> str:
    raw = str(value if value is not None else "")
    if raw.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + raw
    return raw


def _select_blind_mapping() -> dict[str, str]:
    candidates = list(IMPORTED_CANDIDATES)
    if secrets.randbelow(2):
        candidates.reverse()
    return dict(zip(("A", "B"), candidates))


def build_blind_payload(
    audit: Mapping[str, Any],
    *,
    mapping: Mapping[str, str] | None = None,
    contract: SourceContract = LOCKED_SOURCE_CONTRACT,
) -> tuple[dict[str, Any], dict[str, str]]:
    label_to_candidate = dict(mapping or _select_blind_mapping())
    if set(label_to_candidate) != {"A", "B"}:
        raise ValueError("Blind mapping must contain exactly A and B")
    if set(label_to_candidate.values()) != set(IMPORTED_CANDIDATES):
        raise ValueError("Blind mapping candidates are invalid")
    if CANDIDATE_GROUPED_Q8 in label_to_candidate.values():
        raise ValueError("Q8 cannot be included in the A/B mapping")
    results = _as_mapping(audit.get("results"), label="verified source results")
    outputs_by_run: dict[tuple[int, str], dict[str, Mapping[str, Any]]] = {}
    for run_key in IMPORTED_RUN_KEYS:
        result = _as_mapping(results.get(run_key), label=f"verified result {run_key}")
        outputs = _as_list(result.get("outputs"), label=f"outputs {run_key}")
        outputs_by_run[run_key] = {
            str(output.get("row_id") or ""): _as_mapping(
                output,
                label=f"output {run_key}",
            )
            for output in outputs
            if isinstance(output, Mapping)
        }
    reference_outputs = _as_list(
        _as_mapping(
            results[(1, CANDIDATE_BASELINE)],
            label="baseline result",
        ).get("outputs"),
        label="baseline outputs",
    )
    rows: list[dict[str, Any]] = []
    for row_number, raw_reference in enumerate(reference_outputs, start=1):
        reference = _as_mapping(raw_reference, label=f"row {row_number}")
        row_id = str(reference.get("row_id") or "")
        row = {
            "row": row_number,
            "row_id": row_id,
            "page": int(reference.get("page_index") or 0),
            "block": int(reference.get("block_index") or 0),
            "source": str(reference.get("source") or ""),
        }
        for label, candidate in label_to_candidate.items():
            for round_index in (1, 2):
                output = outputs_by_run[(round_index, candidate)].get(row_id)
                if output is None:
                    raise ValueError(
                        f"Blind row {row_id} is missing from {label}{round_index}"
                    )
                row[f"{label}{round_index}"] = str(
                    output.get("translation") or ""
                )
        rows.append(row)
    _expect_equal(
        len(rows),
        contract.block_count,
        label="blind review row count",
    )
    payload = {
        "report_protocol_version": REPORT_PROTOCOL_VERSION,
        "source_protocol_version": contract.protocol_version,
        "source_suite_fingerprint": contract.suite_fingerprint,
        "row_count": contract.block_count,
        "rounds": [1, 2],
        "labels": ["A", "B"],
        "candidate_names_hidden": True,
        "timings_hidden": True,
        "rows": rows,
    }
    return payload, label_to_candidate


def _review_csv_row(row: Mapping[str, Any]) -> list[Any]:
    return [
        row["row"],
        _csv_safe(row["row_id"]),
        row["page"],
        row["block"],
        _csv_safe(row["source"]),
        _csv_safe(row["A1"]),
        _csv_safe(row["A2"]),
        _csv_safe(row["B1"]),
        _csv_safe(row["B2"]),
        "",
        "",
        "",
        "",
        "",
        "",
    ]


def write_review_csv(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(REVIEW_COLUMNS)
        for raw_row in _as_list(payload.get("rows"), label="blind rows"):
            row = _as_mapping(raw_row, label="blind row")
            writer.writerow(_review_csv_row(row))
    os.replace(temporary, path)


def _html_review_row(row: Mapping[str, Any]) -> str:
    translations = []
    for cell in ("A1", "A2", "B1", "B2"):
        translations.append(
            (
                f'<td class="translation"><div>{html.escape(str(row[cell]))}</div>'
                f'<select data-field="{cell}_regression" '
                'aria-label="의미 회귀 판정">'
                '<option value="">미검수</option>'
                '<option value="no">회귀 없음</option>'
                '<option value="yes">의미 회귀</option>'
                "</select></td>"
            )
        )
    checkboxes = "".join(
        (
            '<label><input type="checkbox" data-regression-type="'
            + html.escape(regression_type)
            + '"> '
            + html.escape(REGRESSION_TYPE_LABELS[regression_type])
            + "</label>"
        )
        for regression_type in REGRESSION_TYPES
    )
    return (
        f'<tr data-row-id="{html.escape(str(row["row_id"]), quote=True)}">'
        f'<td class="number">{int(row["row"])}</td>'
        f'<td class="number">{int(row["page"])}</td>'
        f'<td class="number">{int(row["block"])}</td>'
        f'<td class="source">{html.escape(str(row["source"]))}</td>'
        + "".join(translations)
        + f'<td class="types">{checkboxes}</td>'
        + '<td><textarea data-field="notes" rows="3" '
        + 'placeholder="회귀가 있으면 이유를 적으세요"></textarea></td>'
        + "</tr>"
    )


def render_review_html(
    payload: Mapping[str, Any],
    *,
    review_fingerprint: str,
) -> str:
    rows = [
        _as_mapping(row, label="blind HTML row")
        for row in _as_list(payload.get("rows"), label="blind HTML rows")
    ]
    browser_payload = json.dumps(
        rows,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    browser_payload = (
        browser_payload.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    return """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gemma A/B blind 품질 검수</title>
<style>
:root {{ color-scheme: light; font-family: "Segoe UI", "Noto Sans KR", sans-serif; }}
body {{ margin: 0; background: #f5f6f8; color: #17191c; }}
header {{ position: sticky; top: 0; z-index: 3; padding: 16px 22px;
  background: #17202b; color: #fff; box-shadow: 0 2px 8px #0003; }}
h1 {{ margin: 0 0 8px; font-size: 21px; }}
header p {{ margin: 4px 0; line-height: 1.45; }}
.toolbar {{ display: flex; gap: 12px; align-items: center; margin-top: 10px; }}
button {{ border: 0; border-radius: 6px; padding: 9px 14px; cursor: pointer;
  background: #2f7d4d; color: #fff; font-weight: 700; }}
button:hover {{ background: #25643e; }}
#progress {{ font-weight: 700; }}
main {{ padding: 14px; overflow: auto; }}
table {{ border-collapse: separate; border-spacing: 0; min-width: 1800px;
  width: 100%; background: #fff; }}
th, td {{ border-right: 1px solid #d7dbe0; border-bottom: 1px solid #d7dbe0;
  padding: 8px; vertical-align: top; }}
th {{ position: sticky; top: 138px; z-index: 2; background: #e8edf3; }}
tr:nth-child(even) td {{ background: #fafbfc; }}
.number {{ width: 48px; text-align: center; }}
.source {{ min-width: 220px; white-space: pre-wrap; }}
.translation {{ min-width: 245px; white-space: pre-wrap; }}
.translation select {{ display: block; width: 100%; margin-top: 9px; padding: 5px; }}
.types {{ min-width: 170px; }}
.types label {{ display: block; margin: 2px 0; }}
textarea {{ width: 210px; box-sizing: border-box; resize: vertical; }}
.incomplete {{ outline: 2px solid #d29322; outline-offset: -2px; }}
</style>
</head>
<body>
<header>
  <h1>Gemma protocol v4 A/B blind 품질 검수</h1>
  <p>설정명·속도·blind key는 숨겨져 있으며 A와 B의 매핑은 두 라운드에서 같습니다.</p>
  <p>자연스러운 표현 차이는 허용합니다. 화자·관계·부정·행동·대상·숫자·고유명사·명시적 의미가 달라질 때만 의미 회귀로 표시하세요.</p>
  <div class="toolbar">
    <button id="export" type="button">완료 CSV 내보내기</button>
    <span id="progress">0 / {row_count}행 검수</span>
  </div>
</header>
<main>
<table>
<thead><tr>
  <th>행</th><th>페이지</th><th>블록</th><th>원문</th>
  <th>A1</th><th>A2</th><th>B1</th><th>B2</th>
  <th>회귀 유형</th><th>메모</th>
</tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
</main>
<script>
"use strict";
const immutableRows = {browser_payload};
const storageKey = "ct-gemma-final-ab-v4-{review_fingerprint}-" +
  window.location.pathname;
const tableRows = Array.from(document.querySelectorAll("tbody tr"));
function rowState(tr) {{
  const flags = {{}};
  for (const select of tr.querySelectorAll("select[data-field]")) {{
    flags[select.dataset.field] = select.value;
  }}
  const regressionTypes = Array.from(
    tr.querySelectorAll("input[data-regression-type]:checked")
  ).map((input) => input.dataset.regressionType);
  return {{
    ...flags,
    regression_types: regressionTypes.join(","),
    notes: tr.querySelector('[data-field="notes"]').value
  }};
}}
function save() {{
  const state = {{}};
  for (const tr of tableRows) state[tr.dataset.rowId] = rowState(tr);
  localStorage.setItem(storageKey, JSON.stringify(state));
  updateProgress();
}}
function restore() {{
  let state = {{}};
  try {{ state = JSON.parse(localStorage.getItem(storageKey) || "{{}}"); }}
  catch (_error) {{ state = {{}}; }}
  for (const tr of tableRows) {{
    const saved = state[tr.dataset.rowId];
    if (!saved) continue;
    for (const select of tr.querySelectorAll("select[data-field]")) {{
      select.value = saved[select.dataset.field] || "";
    }}
    const selectedTypes = new Set((saved.regression_types || "").split(","));
    for (const input of tr.querySelectorAll("input[data-regression-type]")) {{
      input.checked = selectedTypes.has(input.dataset.regressionType);
    }}
    tr.querySelector('[data-field="notes"]').value = saved.notes || "";
  }}
  updateProgress();
}}
function updateProgress() {{
  let complete = 0;
  for (const tr of tableRows) {{
    const flags = Array.from(tr.querySelectorAll("select[data-field]"));
    const done = flags.every((select) => select.value === "yes" || select.value === "no");
    tr.classList.toggle("incomplete", !done);
    if (done) complete += 1;
  }}
  document.getElementById("progress").textContent =
    `${{complete}} / ${{tableRows.length}}행 검수`;
}}
function csvCell(value) {{
  let raw = String(value ?? "");
  if (/^\\s*[=+\\-@]/.test(raw)) raw = "'" + raw;
  return '"' + raw.replaceAll('"', '""') + '"';
}}
function exportCsv() {{
  save();
  const header = {review_columns};
  const lines = [header.map(csvCell).join(",")];
  immutableRows.forEach((row, index) => {{
    const review = rowState(tableRows[index]);
    const values = [
      row.row, row.row_id, row.page, row.block, row.source,
      row.A1, row.A2, row.B1, row.B2,
      review.A1_regression, review.A2_regression,
      review.B1_regression, review.B2_regression,
      review.regression_types, review.notes
    ];
    lines.push(values.map(csvCell).join(","));
  }});
  const blob = new Blob(["\\ufeff" + lines.join("\\r\\n") + "\\r\\n"],
    {{type: "text/csv;charset=utf-8"}});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "blind_review_completed.csv";
  link.click();
  URL.revokeObjectURL(link.href);
}}
for (const tr of tableRows) {{
  tr.addEventListener("change", save);
  tr.addEventListener("input", save);
}}
document.getElementById("export").addEventListener("click", exportCsv);
restore();
</script>
</body>
</html>
""".format(
        table_rows="\n".join(_html_review_row(row) for row in rows),
        browser_payload=browser_payload,
        review_fingerprint=review_fingerprint,
        review_columns=json.dumps(REVIEW_COLUMNS),
        row_count=len(rows),
    )


def _ensure_output_is_untracked(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    try:
        relative = resolved.relative_to(ROOT.resolve())
    except ValueError:
        return
    if not relative.parts or relative.parts[0] == ".git":
        raise ValueError("Report output cannot be written inside .git")
    completed = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative.as_posix()],
        cwd=ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise ValueError(
            "Report output inside the repository must be ignored by Git"
        )


def import_clean_suite(
    source_dir: Path,
    output_dir: Path,
    *,
    contract: SourceContract = LOCKED_SOURCE_CONTRACT,
    mapping: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    try:
        output_dir.relative_to(source_dir)
    except ValueError:
        pass
    else:
        raise ValueError("Report output must not be inside the source suite")
    _ensure_output_is_untracked(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; choose a new path: {output_dir}"
        )
    audit = validate_source_suite(source_dir, contract=contract)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    payload, label_to_candidate = build_blind_payload(
        audit,
        mapping=mapping,
        contract=contract,
    )
    private_dir = output_dir / PRIVATE_DIRNAME
    private_dir.mkdir(parents=True, exist_ok=True)
    payload_path = private_dir / PAYLOAD_FILENAME
    key_path = private_dir / KEY_FILENAME
    audit_path = private_dir / AUDIT_FILENAME
    _write_private_json(payload_path, payload)
    _write_private_json(
        key_path,
        {
            "report_protocol_version": REPORT_PROTOCOL_VERSION,
            "label_to_candidate": label_to_candidate,
            "disclosure_status": "keep_private_until_complete_user_review",
        },
    )
    source_results = _as_mapping(audit.get("results"), label="source results")
    private_audit = {
        "report_protocol_version": REPORT_PROTOCOL_VERSION,
        "source_protocol_version": contract.protocol_version,
        "source_suite_fingerprint": contract.suite_fingerprint,
        "source_contracts_revalidated": {
            "input_manifest": True,
            "ocr_snapshot": True,
            "model": True,
            "runtime": True,
            "translation_behavior": True,
            "result_sha256": True,
            "output_order": True,
        },
        "imported_runs": [
            {
                "round": round_index,
                "candidate": candidate,
                "result_sha256": contract.run_sha256[(round_index, candidate)],
                "translation_elapsed_sec": float(
                    source_results[(round_index, candidate)][
                        "translation_elapsed_sec"
                    ]
                ),
            }
            for round_index, candidate in IMPORTED_RUN_KEYS
        ],
        "cross_round_output_difference_counts": audit[
            "round_difference_counts"
        ],
        "excluded_q8": audit["q8_elimination"],
        "docker_or_model_requests_executed": False,
    }
    _write_private_json(audit_path, private_audit)
    review_fingerprint = canonical_sha256(payload)
    write_review_csv(output_dir / REVIEW_FILENAME, payload)
    _atomic_write_text(
        output_dir / REVIEW_HTML_FILENAME,
        render_review_html(payload, review_fingerprint=review_fingerprint),
    )
    _atomic_write_text(
        output_dir / REVIEW_INSTRUCTIONS_FILENAME,
        "\n".join(
            [
                "# Gemma A/B blind 품질 검수",
                "",
                "- 설정명, 속도, blind key는 검수 완료 전 공개하지 않는다.",
                "- A/B 매핑은 1·2라운드에 동일하게 적용됐다.",
                "- HTML에서 292행의 A1·A2·B1·B2를 모두 판정하고 완료 CSV를 내보낸다.",
                "- 자연스러운 표현 차이는 허용한다.",
                "- 화자·관계·부정·행동·대상·숫자·고유명사·명시적 의미 변화만 회귀로 표시한다.",
                "- 회귀가 있으면 유형과 메모를 반드시 작성한다.",
                "- 검수가 끝나기 전에는 `private` 폴더를 열지 않는다.",
                "",
                "검수 완료 CSV는 먼저 `validate-review`로 검사한다. "
                "292행이 모두 유효할 때만 `unblind`가 허용된다.",
                "",
            ]
        ),
    )
    state = {
        "report_protocol_version": REPORT_PROTOCOL_VERSION,
        "source_protocol_version": contract.protocol_version,
        "status": "awaiting_user_quality_review",
        "expected_page_count": contract.page_count,
        "expected_row_count": contract.block_count,
        "labels": ["A", "B"],
        "rounds": [1, 2],
        "candidate_names_hidden": True,
        "timings_hidden": True,
        "source_suite_fingerprint": contract.suite_fingerprint,
        "report_tool_sha256": sha256_file(Path(__file__).resolve()),
        "blind_payload_sha256": sha256_file(payload_path),
        "blind_key_sha256": sha256_file(key_path),
        "review_fingerprint": review_fingerprint,
        "q8_in_candidate_set": False,
        "q8_elimination_evidence_verified": True,
        "docker_or_model_requests_executed": False,
        "unblind_allowed": False,
    }
    write_json(output_dir / STATE_FILENAME, state)
    return state


def _expected_review_immutable(
    row: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "row": str(row["row"]),
        "row_id": _csv_safe(row["row_id"]),
        "page": str(row["page"]),
        "block": str(row["block"]),
        "source": _csv_safe(row["source"]),
        "A1": _csv_safe(row["A1"]),
        "A2": _csv_safe(row["A2"]),
        "B1": _csv_safe(row["B1"]),
        "B2": _csv_safe(row["B2"]),
    }


def _load_review_context(
    suite_dir: Path,
    *,
    contract: SourceContract = LOCKED_SOURCE_CONTRACT,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    suite_dir = suite_dir.expanduser().resolve()
    state = read_json(suite_dir / STATE_FILENAME)
    _expect_equal(
        state.get("report_protocol_version"),
        REPORT_PROTOCOL_VERSION,
        label="review suite protocol version",
    )
    for field_name, expected in {
        "source_protocol_version": contract.protocol_version,
        "expected_page_count": contract.page_count,
        "expected_row_count": contract.block_count,
        "labels": ["A", "B"],
        "rounds": [1, 2],
        "candidate_names_hidden": True,
        "timings_hidden": True,
        "source_suite_fingerprint": contract.suite_fingerprint,
        "q8_in_candidate_set": False,
        "q8_elimination_evidence_verified": True,
        "docker_or_model_requests_executed": False,
        "unblind_allowed": False,
    }.items():
        _expect_equal(
            state.get(field_name),
            expected,
            label=f"review suite {field_name}",
        )
    _expect_equal(
        state.get("status"),
        "awaiting_user_quality_review",
        label="review suite status",
    )
    private_dir = suite_dir / PRIVATE_DIRNAME
    payload_path = private_dir / PAYLOAD_FILENAME
    key_path = private_dir / KEY_FILENAME
    _expect_equal(
        state.get("report_tool_sha256"),
        sha256_file(Path(__file__).resolve()),
        label="protocol-v4 report tool digest",
    )
    payload, payload_digest = read_json_with_sha256(payload_path)
    key, key_digest = read_json_with_sha256(key_path)
    _expect_equal(
        payload_digest,
        state.get("blind_payload_sha256"),
        label="blind payload file digest",
    )
    _expect_equal(
        key_digest,
        state.get("blind_key_sha256"),
        label="blind key file digest",
    )
    _expect_equal(
        canonical_sha256(payload),
        state.get("review_fingerprint"),
        label="blind payload canonical fingerprint",
    )
    for field_name, expected in {
        "report_protocol_version": REPORT_PROTOCOL_VERSION,
        "source_protocol_version": contract.protocol_version,
        "source_suite_fingerprint": contract.suite_fingerprint,
        "row_count": contract.block_count,
        "rounds": [1, 2],
        "labels": ["A", "B"],
        "candidate_names_hidden": True,
        "timings_hidden": True,
    }.items():
        _expect_equal(
            payload.get(field_name),
            expected,
            label=f"blind payload {field_name}",
        )
    _expect_equal(
        key.get("report_protocol_version"),
        REPORT_PROTOCOL_VERSION,
        label="blind key protocol version",
    )
    _expect_equal(
        key.get("disclosure_status"),
        "keep_private_until_complete_user_review",
        label="blind key disclosure status",
    )
    mapping = _as_mapping(
        key.get("label_to_candidate"),
        label="blind key mapping",
    )
    if set(mapping) != {"A", "B"} or set(mapping.values()) != set(
        IMPORTED_CANDIDATES
    ):
        raise ValueError("Blind key mapping is invalid")
    if CANDIDATE_GROUPED_Q8 in mapping.values():
        raise ValueError("Blind key illegally includes Q8")
    return state, payload, key


def validate_review_file(
    suite_dir: Path,
    review_file: Path,
    *,
    contract: SourceContract = LOCKED_SOURCE_CONTRACT,
) -> dict[str, Any]:
    state, payload, _key = _load_review_context(
        suite_dir,
        contract=contract,
    )
    expected_rows = [
        _as_mapping(row, label="expected blind row")
        for row in _as_list(payload.get("rows"), label="expected blind rows")
    ]
    review_file = review_file.expanduser().resolve()
    if not review_file.is_file():
        raise FileNotFoundError(f"Review CSV is missing: {review_file}")
    csv.field_size_limit(max(csv.field_size_limit(), 16 * 1024 * 1024))
    with review_file.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != REVIEW_COLUMNS:
            raise ReviewIncompleteError(
                ["CSV header differs from the locked review schema"]
            )
        actual_rows = list(reader)

    errors: list[str] = []
    if len(actual_rows) != len(expected_rows):
        errors.append(
            f"expected {len(expected_rows)} rows, found {len(actual_rows)}"
        )
    normalized_rows: list[dict[str, Any]] = []
    flagged_output_count = 0
    flagged_row_count = 0
    label_flag_counts = {"A": 0, "B": 0}
    for index, expected in enumerate(expected_rows):
        if index >= len(actual_rows):
            errors.append(f"row {index + 1} is missing")
            continue
        actual = actual_rows[index]
        immutable = _expected_review_immutable(expected)
        for column, expected_value in immutable.items():
            if str(actual.get(column) or "") != expected_value:
                errors.append(
                    f"row {index + 1} immutable column {column} differs"
                )
        flags: dict[str, str] = {}
        for column in REVIEW_FLAG_COLUMNS:
            value = str(actual.get(column) or "").strip().casefold()
            if value not in {"yes", "no"}:
                errors.append(
                    f"row {index + 1} {column} must be yes or no"
                )
            flags[column] = value
        raw_types = str(actual.get("regression_types") or "").strip()
        regression_types = [
            item.strip()
            for item in raw_types.split(",")
            if item.strip()
        ]
        invalid_types = sorted(set(regression_types) - set(REGRESSION_TYPES))
        if invalid_types:
            errors.append(
                f"row {index + 1} has invalid regression types: "
                + ", ".join(invalid_types)
            )
        if len(set(regression_types)) != len(regression_types):
            errors.append(f"row {index + 1} has duplicate regression types")
        yes_columns = [
            column
            for column, value in flags.items()
            if value == "yes"
        ]
        notes = str(actual.get("notes") or "").strip()
        if yes_columns and not regression_types:
            errors.append(
                f"row {index + 1} needs a regression type"
            )
        if yes_columns and not notes:
            errors.append(f"row {index + 1} needs regression notes")
        if not yes_columns and regression_types:
            errors.append(
                f"row {index + 1} has regression types without a yes flag"
            )
        if yes_columns:
            flagged_row_count += 1
            flagged_output_count += len(yes_columns)
            for column in yes_columns:
                label_flag_counts[column[0]] += 1
        normalized_rows.append(
            {
                "row": int(expected["row"]),
                "row_id": str(expected["row_id"]),
                "flags": flags,
                "regression_types": regression_types,
                "notes": notes,
            }
        )
    if len(actual_rows) > len(expected_rows):
        errors.append(
            f"{len(actual_rows) - len(expected_rows)} unexpected rows were added"
        )
    if errors:
        raise ReviewIncompleteError(errors)
    return {
        "report_protocol_version": REPORT_PROTOCOL_VERSION,
        "status": "complete",
        "review_file_sha256": sha256_file(review_file),
        "review_fingerprint": state["review_fingerprint"],
        "reviewed_row_count": len(normalized_rows),
        "flagged_row_count": flagged_row_count,
        "flagged_output_count": flagged_output_count,
        "blind_label_flag_counts": label_flag_counts,
        "normalized_rows": normalized_rows,
        "unblind_allowed": True,
    }


def _write_review_validation(
    suite_dir: Path,
    payload: Mapping[str, Any],
) -> None:
    write_json(suite_dir / REVIEW_VALIDATION_FILENAME, payload)


def validate_review_command(
    suite_dir: Path,
    review_file: Path,
) -> int:
    suite_dir = suite_dir.expanduser().resolve()
    try:
        validation = validate_review_file(suite_dir, review_file)
    except ReviewIncompleteError as exc:
        _write_review_validation(
            suite_dir,
            {
                "report_protocol_version": REPORT_PROTOCOL_VERSION,
                "status": "incomplete",
                "error_count": len(exc.errors),
                "errors": exc.errors[:100],
                "unblind_allowed": False,
            },
        )
        print(str(exc), file=sys.stderr)
        for error in exc.errors[:20]:
            print(f"- {error}", file=sys.stderr)
        return 2
    _write_review_validation(suite_dir, validation)
    print(
        f"Review complete: {validation['reviewed_row_count']} rows; "
        "unblind is now allowed"
    )
    return 0


def _confirmation_phrase(expected_rows: int) -> str:
    return f"{expected_rows}-ROWS-REVIEWED"


def unblind_review(
    suite_dir: Path,
    review_file: Path,
    *,
    confirmation: str,
    contract: SourceContract = LOCKED_SOURCE_CONTRACT,
) -> dict[str, Any]:
    suite_dir = suite_dir.expanduser().resolve()
    state, _payload, key = _load_review_context(
        suite_dir,
        contract=contract,
    )
    expected_confirmation = _confirmation_phrase(
        int(state["expected_row_count"])
    )
    if confirmation != expected_confirmation:
        raise ValueError(
            "Unblind confirmation differs; expected "
            f"{expected_confirmation!r}"
        )
    validation = validate_review_file(
        suite_dir,
        review_file,
        contract=contract,
    )
    mapping = {
        str(label): str(candidate)
        for label, candidate in _as_mapping(
            key.get("label_to_candidate"),
            label="blind mapping",
        ).items()
    }
    candidate_counts = {
        candidate: int(
            validation["blind_label_flag_counts"][label]
        )
        for label, candidate in mapping.items()
    }
    grouped_regression_count = candidate_counts[CANDIDATE_GROUPED_F16]
    quality_approved = grouped_regression_count == 0
    summary = {
        "report_protocol_version": REPORT_PROTOCOL_VERSION,
        "status": (
            "quality_approved" if quality_approved else "quality_rejected"
        ),
        "reviewed_row_count": validation["reviewed_row_count"],
        "review_file_sha256": validation["review_file_sha256"],
        "label_to_candidate": mapping,
        "candidate_regression_output_counts": candidate_counts,
        "grouped_candidate_semantic_regression_count": (
            grouped_regression_count
        ),
        "grouped_candidate_quality_approved": quality_approved,
        "product_default_implementation_allowed": quality_approved,
        "final_product_promotion_allowed": False,
        "full_pipeline_comparison_required": quality_approved,
        "stop_reason": (
            None
            if quality_approved
            else "grouped candidate has an explicit semantic regression"
        ),
    }
    write_json(suite_dir / UNBLIND_SUMMARY_FILENAME, summary)
    lines = [
        "# Gemma A/B blind 검수 unblind 결과",
        "",
        f"- 상태: `{summary['status']}`",
        f"- 검수 행: {summary['reviewed_row_count']}",
        f"- A: `{mapping['A']}`",
        f"- B: `{mapping['B']}`",
        "- 후보별 회귀 출력 수:",
        f"  - `{CANDIDATE_BASELINE}`: "
        f"{candidate_counts[CANDIDATE_BASELINE]}",
        f"  - `{CANDIDATE_GROUPED_F16}`: "
        f"{candidate_counts[CANDIDATE_GROUPED_F16]}",
        "- grouped 품질 승인: "
        + ("PASS" if quality_approved else "FAIL"),
        "- 제품 기본값 구현 단계 진입 허용: "
        + ("YES" if quality_approved else "NO"),
        "- 최종 제품 승격 허용: NO (전체 파이프라인 비교 필요)",
        "",
    ]
    _atomic_write_text(
        suite_dir / UNBLIND_MARKDOWN_FILENAME,
        "\n".join(lines),
    )
    _write_review_validation(suite_dir, validation)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Protocol-v4 report-only Gemma A/B blind review tool"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser(
        "import-clean",
        help="revalidate the locked v3 suite and create blind A/B artifacts",
    )
    import_parser.add_argument("--source-suite", type=Path, required=True)
    import_parser.add_argument("--output-dir", type=Path, required=True)

    validate_parser = subparsers.add_parser(
        "validate-review",
        help="validate all 292 completed blind review rows",
    )
    validate_parser.add_argument("--suite-dir", type=Path, required=True)
    validate_parser.add_argument("--review-file", type=Path, required=True)

    unblind_parser = subparsers.add_parser(
        "unblind",
        help="unblind only after a complete, valid review",
    )
    unblind_parser.add_argument("--suite-dir", type=Path, required=True)
    unblind_parser.add_argument("--review-file", type=Path, required=True)
    unblind_parser.add_argument(
        "--confirm-user-review",
        required=True,
        help="must be exactly 292-ROWS-REVIEWED",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "import-clean":
            state = import_clean_suite(
                args.source_suite,
                args.output_dir,
            )
            output_dir = args.output_dir.expanduser().resolve()
            print(
                "Protocol-v4 blind review created without Docker/model calls:"
            )
            print(output_dir / REVIEW_HTML_FILENAME)
            print(output_dir / REVIEW_FILENAME)
            print(
                f"Rows: {state['expected_row_count']}; "
                "status: awaiting user quality review"
            )
            return 0
        if args.command == "validate-review":
            return validate_review_command(
                args.suite_dir,
                args.review_file,
            )
        if args.command == "unblind":
            summary = unblind_review(
                args.suite_dir,
                args.review_file,
                confirmation=args.confirm_user_review,
            )
            print(
                f"Unblind complete: {summary['status']}; "
                f"grouped regressions="
                f"{summary['grouped_candidate_semantic_regression_count']}"
            )
            return 0 if summary["grouped_candidate_quality_approved"] else 3
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
