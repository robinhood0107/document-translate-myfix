from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping


HOLDOUT_LOCK_SCHEMA = "inpaint-holdout-execution-lock-v4"
HOLDOUT_PREREQUISITE_SCHEMA = "inpaint-holdout-prerequisites-v4"


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return (
        len(text) == 64
        and text == text.lower()
        and all(character in "0123456789abcdef" for character in text)
    )


def _is_code_commit(value: object) -> bool:
    text = str(value or "")
    return (
        len(text) in {40, 64}
        and text == text.lower()
        and all(character in "0123456789abcdef" for character in text)
    )


def execution_argv_sha256(argv: list[str]) -> str:
    encoded = json.dumps(
        argv,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def execution_binding_sha256(
    *,
    argv: list[str],
    manifest_sha256: object,
    model_sha256: object,
    runtime_provider: object,
) -> str:
    encoded = json.dumps(
        {
            "argv": argv,
            "manifest_sha256": str(manifest_sha256),
            "model_sha256": str(model_sha256),
            "runtime_provider": str(runtime_provider),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_holdout_lock_path(
    prerequisites_path: Path,
    holdout_id: object,
) -> Path:
    """Return the only lock path permitted for one sealed prerequisite file."""

    identity = str(holdout_id or "").strip()
    if not identity:
        raise ValueError("holdout prerequisites require an id")
    sealed_path = prerequisites_path.resolve()
    lock_identity = hashlib.sha256(
        f"{sealed_path.name}\0{identity}".encode("utf-8")
    ).hexdigest()
    return sealed_path.parent / f".inpaint-holdout-{lock_identity}.execution-lock.json"


def validate_holdout_prerequisites(payload: Mapping[str, object]) -> None:
    if payload.get("schema_version") != HOLDOUT_PREREQUISITE_SCHEMA:
        raise ValueError("unsupported inpaint holdout prerequisite schema")
    if not str(payload.get("holdout_id") or "").strip():
        raise ValueError("holdout prerequisites require an id")
    for field in ("manifest_sha256", "model_sha256"):
        if not _is_sha256(payload.get(field)):
            raise ValueError(f"holdout prerequisite {field} must be a SHA-256")
    if not _is_code_commit(payload.get("code_commit")):
        raise ValueError("holdout prerequisite code_commit must be a Git commit id")
    if payload.get("runtime_provider") != "CUDAExecutionProvider":
        raise ValueError("holdout prerequisites require CUDAExecutionProvider")
    execution = payload.get("execution_binding")
    if not isinstance(execution, dict):
        raise ValueError("holdout prerequisites require an execution binding")
    expected_argv = execution.get("argv")
    if (
        not isinstance(expected_argv, list)
        or not expected_argv
        or any(not isinstance(value, str) for value in expected_argv)
        or not expected_argv[0].strip()
    ):
        raise ValueError("holdout execution binding requires a non-empty argv")
    expected_argv_sha256 = execution.get("argv_sha256")
    if not _is_sha256(expected_argv_sha256) or expected_argv_sha256 != (
        execution_argv_sha256(expected_argv)
    ):
        raise ValueError("holdout execution argv SHA-256 does not match argv")
    for field in ("manifest_sha256", "model_sha256", "runtime_provider"):
        if execution.get(field) != payload.get(field):
            raise ValueError(f"holdout execution binding {field} does not match")
    if execution.get("binding_sha256") != execution_binding_sha256(
        argv=expected_argv,
        manifest_sha256=execution["manifest_sha256"],
        model_sha256=execution["model_sha256"],
        runtime_provider=execution["runtime_provider"],
    ):
        raise ValueError("holdout execution binding SHA-256 does not match")
    required_true = (
        "source_only_manifest_sealed",
        "synthetic_gate_passed",
        "e1_mask_gate_passed",
        "e1_cuda_gate_passed",
        "visual_review_passed",
        "onnx_parity_passed",
        "product_stack_frozen",
    )
    missing = [field for field in required_true if payload.get(field) is not True]
    if missing:
        raise ValueError("holdout prerequisite gate is not closed: " + missing[0])
    onnx_xor = payload.get("onnx_final_binary_xor_pixels")
    if isinstance(onnx_xor, bool) or not isinstance(onnx_xor, int) or onnx_xor != 0:
        raise ValueError("holdout prerequisites require final ONNX mask XOR zero")
    development = payload.get("development_source_sha256")
    holdout = payload.get("holdout_source_sha256")
    if (
        not isinstance(development, list)
        or not development
        or not isinstance(holdout, list)
        or not holdout
    ):
        raise ValueError("holdout prerequisites require development and holdout source SHAs")
    development_set = {str(value) for value in development}
    holdout_set = {str(value) for value in holdout}
    if any(not _is_sha256(value) for value in development_set | holdout_set):
        raise ValueError("holdout source identities must be SHA-256 values")
    if len(development_set) != len(development) or len(holdout_set) != len(holdout):
        raise ValueError("holdout source identities must not contain duplicates")
    overlap = sorted(development_set & holdout_set)
    if overlap:
        raise ValueError("holdout source overlaps development data: " + overlap[0])


def claim_holdout_once(
    prerequisites_path: Path,
    *,
    prerequisites: Mapping[str, object],
) -> dict[str, object]:
    """Atomically consume a final holdout before any candidate process starts."""

    prerequisite_bytes = prerequisites_path.read_bytes()
    prerequisite_file_payload = json.loads(prerequisite_bytes.decode("utf-8"))
    if not isinstance(prerequisite_file_payload, dict) or prerequisite_file_payload != dict(
        prerequisites
    ):
        raise ValueError("holdout prerequisite payload differs from its sealed file")
    validate_holdout_prerequisites(prerequisites)
    lock_path = canonical_holdout_lock_path(
        prerequisites_path,
        prerequisites["holdout_id"],
    )
    execution = prerequisites["execution_binding"]
    assert isinstance(execution, dict)
    payload = {
        "schema_version": HOLDOUT_LOCK_SCHEMA,
        "holdout_id": str(prerequisites["holdout_id"]),
        "status": "claimed_consumed",
        "prerequisites_sha256": hashlib.sha256(prerequisite_bytes).hexdigest(),
        "manifest_sha256": str(prerequisites["manifest_sha256"]),
        "code_commit": str(prerequisites["code_commit"]),
        "model_sha256": str(prerequisites["model_sha256"]),
        "runtime_provider": str(prerequisites["runtime_provider"]),
        "execution_argv_sha256": str(execution["argv_sha256"]),
        "execution_binding_sha256": str(execution["binding_sha256"]),
    }
    descriptor = os.open(
        str(lock_path),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # The exclusive file is intentionally retained.  Once a final holdout
        # is claimed it is consumed even if the candidate process later fails.
        raise
    return payload
