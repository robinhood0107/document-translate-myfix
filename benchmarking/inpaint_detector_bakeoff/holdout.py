from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Mapping


HOLDOUT_LOCK_SCHEMA = "inpaint-holdout-execution-lock-v4"
HOLDOUT_PREREQUISITE_SCHEMA = "inpaint-holdout-prerequisites-v4"
REPO_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_LOCK_ROOT = (
    REPO_ROOT
    / "banchmark_result_log"
    / "40-inpaint-mask-render"
    / "holdout-execution-locks-v4"
)
HOLDOUT_RUNNER_RELATIVE_PATH = "scripts/run_inpaint_a5_candidate_v4.py"
A5_PRODUCT_STACK_RUNNER_AVAILABLE = False
A5_UNAVAILABLE_MESSAGE = (
    "A5 unavailable until verified product-stack evidence sealer/runner is implemented"
)


def require_a5_product_stack_runner() -> None:
    if A5_PRODUCT_STACK_RUNNER_AVAILABLE is not True:
        raise RuntimeError(A5_UNAVAILABLE_MESSAGE)


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


def execution_binding_sha256(binding: Mapping[str, object]) -> str:
    fields = (
        "runner_relative_path",
        "runner_sha256",
        "holdout_id",
        "manifest_path",
        "manifest_sha256",
        "matrix_path",
        "matrix_sha256",
        "model_path",
        "model_sha256",
        "runtime_provider",
        "source_inventory_sha256",
        "output_dir",
        "canonical_argv_sha256",
    )
    encoded = json.dumps(
        {field: str(binding.get(field) or "") for field in fields},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_holdout_lock_path(
    prerequisites_path: Path,
    source_inventory_sha256: object,
) -> Path:
    """Return the repository-canonical lock for one sealed source inventory."""

    del prerequisites_path  # Copies and renames must retain the same lock identity.
    identity = str(source_inventory_sha256 or "").strip()
    if not _is_sha256(identity):
        raise ValueError("holdout source inventory identity must be a SHA-256")
    return HOLDOUT_LOCK_ROOT / f"inpaint-holdout-{identity}.execution-lock.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_path(value: object) -> Path | None:
    if isinstance(value, str) and value.strip():
        return Path(value).resolve()
    if isinstance(value, dict):
        nested = value.get("path")
        if isinstance(nested, str) and nested.strip():
            return Path(nested).resolve()
    return None


def holdout_manifest_page_artifact_sha256(
    page: Mapping[str, object],
) -> dict[str, object]:
    """Hash every source/evaluation artifact referenced by one strict v4 page."""

    hashes: dict[str, object] = {}
    for field in (
        "path",
        "target_text_mask",
        "preserve_mask",
        "protected_structure_mask",
        "ambiguous_structure_mask",
        "ownership_mask",
        "bubble_interior_mask",
        "corner_protect_mask",
        "claim_seed_mask",
        "existing_source_edit_mask",
        "baseline",
        "baseline_mask",
        "known_background",
    ):
        path = _artifact_path(page.get(field))
        if path is not None:
            if not path.is_file():
                raise ValueError(f"holdout manifest artifact is missing: {field}")
            hashes[field] = _sha256_file(path)
    instances = page.get("target_instances")
    if not isinstance(instances, list):
        raise ValueError("holdout manifest target_instances must be an array")
    instance_hashes: dict[str, str] = {}
    for value in instances:
        if not isinstance(value, dict):
            raise ValueError("holdout manifest target instance must be an object")
        instance_id = str(value.get("instance_id") or "").strip()
        path = _artifact_path(value.get("mask_path", value.get("mask")))
        if not instance_id or instance_id in instance_hashes or path is None:
            raise ValueError("holdout manifest target instance artifact is invalid")
        if not path.is_file():
            raise ValueError("holdout manifest target instance artifact is missing")
        instance_hashes[instance_id] = _sha256_file(path)
    hashes["target_instances"] = instance_hashes
    regions = page.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError("holdout manifest regions must be a non-empty array")
    region_hashes: dict[str, dict[str, str]] = {}
    for value in regions:
        if not isinstance(value, dict):
            raise ValueError("holdout manifest region must be an object")
        region_id = str(value.get("region_id") or "").strip()
        if not region_id or region_id in region_hashes:
            raise ValueError("holdout manifest region identity is invalid")
        current: dict[str, str] = {}
        for field in (
            "bubble_interior_mask",
            "ownership_mask",
            "protected_structure_mask",
            "ambiguous_structure_mask",
            "corner_protect_mask",
        ):
            path = _artifact_path(value.get(field))
            if path is None or not path.is_file():
                raise ValueError(f"holdout manifest region artifact is invalid: {field}")
            current[field] = _sha256_file(path)
        region_hashes[region_id] = current
    hashes["regions"] = region_hashes
    reference = page.get("paired_reference")
    if reference is not None:
        if not isinstance(reference, dict) or reference.get("proposal_only") is not True:
            raise ValueError("holdout manifest paired reference must be proposal_only")
        path = _artifact_path(reference.get("path"))
        if path is None or not path.is_file():
            raise ValueError("holdout manifest paired reference artifact is invalid")
        actual = _sha256_file(path)
        if actual != reference.get("reference_sha256"):
            raise ValueError("holdout manifest paired reference SHA-256 does not match")
        hashes["paired_reference"] = actual
    return hashes


def validate_holdout_source_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("holdout source manifest root must be an object")
    if payload.get("schema_version") != "inpaint-factorized-source-manifest-v4":
        raise ValueError("holdout requires strict source-only manifest v4")
    if payload.get("split_role") != "final_holdout_source_only":
        raise ValueError("holdout manifest split is not final source-only")
    for field in (
        "annotation_frozen_before_candidate",
        "target_extent_independent",
        "target_inventory_independent",
        "target_review_complete",
    ):
        if payload.get(field) is not True:
            raise ValueError(f"holdout manifest is not frozen source-only: {field}")
    if payload.get("candidate_seen") is not False:
        raise ValueError("holdout manifest was derived after candidate inspection")
    pages = payload.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("holdout manifest pages must be a non-empty array")
    page_ids: list[str] = []
    source_shas: list[str] = []
    for page in pages:
        if not isinstance(page, dict):
            raise ValueError("holdout manifest page must be an object")
        page_id = str(page.get("page_id") or "").strip()
        source_sha = str(page.get("source_sha256") or "").strip().lower()
        if not page_id or page_id in page_ids or not _is_sha256(source_sha):
            raise ValueError("holdout manifest page identity is invalid")
        if source_sha in source_shas:
            raise ValueError("holdout manifest source SHA-256 values must be unique")
        if page.get("candidate_seen") is not False or any(
            page.get(field) is not True
            for field in (
                "annotation_frozen_before_candidate",
                "target_extent_independent",
                "target_inventory_independent",
                "target_review_complete",
            )
        ):
            raise ValueError("holdout manifest page is not frozen source-only")
        actual_artifacts = holdout_manifest_page_artifact_sha256(page)
        if page.get("artifact_sha256") != actual_artifacts:
            raise ValueError("holdout manifest page artifact SHA-256 does not match")
        if actual_artifacts.get("path") != source_sha:
            raise ValueError("holdout manifest page source SHA-256 does not match")
        page_ids.append(page_id)
        source_shas.append(source_sha)
    inventory = {
        "page_ids": page_ids,
        "source_sha256": sorted(source_shas),
    }
    inventory_sha = _canonical_sha256(inventory)
    if payload.get("page_count") != len(page_ids):
        raise ValueError("holdout manifest page count does not match")
    if payload.get("page_inventory_sha256") != inventory_sha:
        raise ValueError("holdout manifest page inventory SHA-256 does not match")
    return {**inventory, "sha256": inventory_sha}


def _bound_file(
    execution: Mapping[str, object],
    *,
    path_field: str,
    sha_field: str,
) -> Path:
    raw_path = execution.get(path_field)
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"holdout execution binding requires {path_field}")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise ValueError(f"holdout execution binding {path_field} is not a file")
    expected_sha = execution.get(sha_field)
    if not _is_sha256(expected_sha) or _sha256_file(path) != expected_sha:
        raise ValueError(f"holdout execution binding {sha_field} does not match file")
    return path


def canonical_holdout_command(execution: Mapping[str, object]) -> list[str]:
    """Build the sole permitted A5 command from structured sealed fields."""

    require_a5_product_stack_runner()
    if execution.get("runner_relative_path") != HOLDOUT_RUNNER_RELATIVE_PATH:
        raise ValueError("holdout execution binding runner is not permitted")
    runner_path = (REPO_ROOT / HOLDOUT_RUNNER_RELATIVE_PATH).resolve()
    if not runner_path.is_file():
        raise ValueError("holdout execution runner is missing")
    if (
        not _is_sha256(execution.get("runner_sha256"))
        or _sha256_file(runner_path) != execution.get("runner_sha256")
    ):
        raise ValueError("holdout execution runner SHA-256 does not match")
    manifest_path = _bound_file(
        execution,
        path_field="manifest_path",
        sha_field="manifest_sha256",
    )
    matrix_path = _bound_file(
        execution,
        path_field="matrix_path",
        sha_field="matrix_sha256",
    )
    model_path = _bound_file(
        execution,
        path_field="model_path",
        sha_field="model_sha256",
    )
    output_dir = execution.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("holdout execution binding requires output_dir")
    provider = execution.get("runtime_provider")
    if provider != "CUDAExecutionProvider":
        raise ValueError("holdout execution requires CUDAExecutionProvider")
    holdout_id = str(execution.get("holdout_id") or "").strip()
    lock_path = canonical_holdout_lock_path(
        Path(),
        execution.get("source_inventory_sha256"),
    )
    return [
        sys.executable,
        str(runner_path),
        "--manifest",
        str(manifest_path),
        "--manifest-sha256",
        str(execution["manifest_sha256"]),
        "--matrix",
        str(matrix_path),
        "--matrix-sha256",
        str(execution["matrix_sha256"]),
        "--model",
        str(model_path),
        "--model-sha256",
        str(execution["model_sha256"]),
        "--provider",
        str(provider),
        "--source-inventory-sha256",
        str(execution["source_inventory_sha256"]),
        "--holdout-id",
        holdout_id,
        "--execution-lock",
        str(lock_path),
        "--output-dir",
        str(Path(output_dir).resolve()),
    ]


def validate_holdout_prerequisites(payload: Mapping[str, object]) -> None:
    # Fail before inspecting manifest/model paths or accepting self-declared
    # gate booleans.  No current sealer binds the exact product-stack evidence.
    require_a5_product_stack_runner()
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
    expected_execution_fields = {
        "runner_relative_path",
        "runner_sha256",
        "holdout_id",
        "manifest_path",
        "manifest_sha256",
        "matrix_path",
        "matrix_sha256",
        "model_path",
        "model_sha256",
        "runtime_provider",
        "source_inventory_sha256",
        "output_dir",
        "canonical_argv_sha256",
        "binding_sha256",
    }
    if set(execution) != expected_execution_fields:
        raise ValueError("holdout execution binding fields are not canonical")
    for field in (
        "holdout_id",
        "manifest_sha256",
        "model_sha256",
        "runtime_provider",
    ):
        if execution.get(field) != payload.get(field):
            raise ValueError(f"holdout execution binding {field} does not match")
    canonical_argv = canonical_holdout_command(execution)
    if execution.get("canonical_argv_sha256") != execution_argv_sha256(
        canonical_argv
    ):
        raise ValueError("holdout canonical execution argv SHA-256 does not match")
    if execution.get("binding_sha256") != execution_binding_sha256(execution):
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
    inventory = validate_holdout_source_manifest(
        Path(str(execution["manifest_path"])).resolve()
    )
    if inventory["source_sha256"] != sorted(holdout_set):
        raise ValueError("holdout prerequisite source inventory differs from manifest")
    if execution.get("source_inventory_sha256") != inventory["sha256"]:
        raise ValueError("holdout execution source inventory SHA-256 does not match")


def claim_holdout_once(
    prerequisites_path: Path,
    *,
    prerequisites: Mapping[str, object],
) -> dict[str, object]:
    """Atomically consume a final holdout before any candidate process starts."""

    require_a5_product_stack_runner()
    prerequisite_bytes = prerequisites_path.read_bytes()
    prerequisite_file_payload = json.loads(prerequisite_bytes.decode("utf-8"))
    if not isinstance(prerequisite_file_payload, dict) or prerequisite_file_payload != dict(
        prerequisites
    ):
        raise ValueError("holdout prerequisite payload differs from its sealed file")
    validate_holdout_prerequisites(prerequisites)
    lock_path = canonical_holdout_lock_path(
        prerequisites_path,
        prerequisites["execution_binding"]["source_inventory_sha256"],
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
        "execution_argv_sha256": str(execution["canonical_argv_sha256"]),
        "execution_binding_sha256": str(execution["binding_sha256"]),
        "source_inventory_sha256": str(execution["source_inventory_sha256"]),
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
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
