#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.contracts import (  # noqa: E402
    binary_mask,
    mask_sha256,
)
from benchmarking.inpaint_detector_bakeoff.incremental_plan import (  # noqa: E402
    IncrementalInpaintPlan,
    IncrementalInpaintReceipt,
    PageIncrementalInpaintPlan,
    composite_incremental_result,
    execute_incremental_inpaint,
    union_incremental_plans,
)
from benchmarking.inpaint_detector_bakeoff.semantic import (  # noqa: E402
    PRESERVE,
    TRANSLATE,
    product_semantic_decision,
)
from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    _resolve_manifest_artifact,
    load_page_masks,
    load_stage1_manifest,
    validate_source_only_manifest_v4,
)
from benchmarking.inpaint_detector_bakeoff.stage2 import (  # noqa: E402
    changed_mask,
    evaluate_relative_product_gate,
    residue_score,
)
from modules.inpainting.source_lama_blockwise import SourceLaMaLarge  # noqa: E402
from modules.inpainting.runtime_contract import (  # noqa: E402
    inspect_learned_inpainter_runtime,
)
from modules.utils.download import ModelDownloader, ModelID  # noqa: E402
from scripts.benchmark_inpaint_glyph_refinement_v34 import (  # noqa: E402
    INVENTORY_SCHEMA_VERSION as MASK_INVENTORY_SCHEMA_VERSION,
    SCHEMA_VERSION as MASK_RESULT_SCHEMA_VERSION,
    validate_output_inventory as validate_mask_output_inventory,
)
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-glyph-refinement-stage2-v34"
CATEGORY = "40-inpaint-mask-render"
SCHEMA_VERSION = "inpaint-glyph-refinement-stage2-results-v34"
OUTPUT_INVENTORY_SCHEMA_VERSION = (
    "inpaint-glyph-refinement-stage2-output-inventory-v34"
)
SUPPORTED_CANDIDATES = {
    "b1_context_additive": "context_additive",
    "b2_narrow_replacement": "narrow_replacement",
    "b3_conditional_segmenter": "conditional_segmenter",
}
REQUIRED_INPUT_MASK_ROLES = (
    "owned_positive_seed",
    "generation_mask",
    "commit_mask",
    "replacement_source_edit",
)
OUTPUT_ARTIFACT_ROLES = (
    "seed",
    "generation_mask",
    "commit_mask",
    "replacement_source_edit",
    "affected_union",
    "final_mask",
    "incremental_changed_mask",
    "candidate_image",
)
OUTPUT_SOURCE_BINDING_ROLES = {
    "seed": "owned_positive_seed",
    "generation_mask": "generation_mask",
    "commit_mask": "commit_mask",
    "replacement_source_edit": "replacement_source_edit",
}
OFFICIAL_CODE_DEPENDENCIES = (
    "scripts/benchmark_inpaint_glyph_refinement_stage2_v34.py",
    "scripts/benchmark_inpaint_glyph_refinement_v34.py",
    "scripts/build_inpaint_seedless_ocr_overlay_v34.py",
    "scripts/build_inpaint_source_routing_overlay_v34.py",
    "scripts/export_inpaint_source_ocr_evidence_v34.py",
    "scripts/build_inpaint_product_policy_overlay_v33.py",
    "benchmarking/inpaint_detector_bakeoff/glyph_refinement.py",
    "benchmarking/inpaint_detector_bakeoff/incremental_plan.py",
    "benchmarking/inpaint_detector_bakeoff/contracts.py",
    "benchmarking/inpaint_detector_bakeoff/evidence_ledger.py",
    "benchmarking/inpaint_detector_bakeoff/semantic.py",
    "benchmarking/inpaint_detector_bakeoff/stage1.py",
    "benchmarking/inpaint_detector_bakeoff/stage2.py",
    "modules/inpainting/source_lama_blockwise.py",
    "modules/inpainting/lama_torch_network.py",
    "modules/inpainting/ffc_torch.py",
    "modules/inpainting/runtime_contract.py",
    "modules/source_parity_vendor/utils/imgproc_utils.py",
    "modules/utils/inpaint_composite.py",
    "modules/utils/download.py",
    "scripts/validation_artifact_harness.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_bytes_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_image(path: str | Path, flags: int) -> np.ndarray:
    result = cv2.imdecode(np.fromfile(Path(path), dtype=np.uint8), flags)
    if result is None or result.size == 0:
        raise FileNotFoundError(path)
    return np.ascontiguousarray(result)


def _read_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    result = _read_image(path, cv2.IMREAD_GRAYSCALE)
    if result.shape != shape:
        raise ValueError(f"v3.4 stage2 mask shape mismatch: {result.shape} != {shape}")
    values = np.unique(result)
    if np.any((values != 0) & (values != 255)):
        raise ValueError("v3.4 stage2 mask must be strict binary")
    return binary_mask(result, shape)


def _write_png(path: Path, value: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"v3.4 stage2 output must be fresh: {path}")
    normalized = np.ascontiguousarray(value)
    encoded, buffer = cv2.imencode(".png", normalized)
    if not encoded:
        raise OSError(f"failed to encode v3.4 stage2 output: {path}")
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_bytes(buffer.tobytes())
    temporary.replace(path)
    decoded = _read_image(
        path,
        cv2.IMREAD_COLOR if normalized.ndim == 3 else cv2.IMREAD_GRAYSCALE,
    )
    if not np.array_equal(decoded, normalized):
        raise RuntimeError(f"v3.4 stage2 output changed during encoding: {path}")
    result: dict[str, object] = {
        "file_sha256": _sha256(path),
        "pixel_bytes_sha256": _array_bytes_sha256(decoded),
        "shape": [int(value) for value in decoded.shape],
        "dtype": str(decoded.dtype),
        "size_bytes": path.stat().st_size,
    }
    if decoded.ndim == 2:
        result["mask_pixel_sha256"] = mask_sha256(decoded)
        result["pixel_count"] = int(np.count_nonzero(decoded))
    return result


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _tracked_worktree_clean() -> bool:
    return not bool(
        subprocess.check_output(
            ["git", "status", "--short", "--untracked-files=no"],
            cwd=ROOT,
            text=True,
        ).strip()
    )


def _committed_dependency_inventory() -> dict[str, object]:
    """Bind an official run to committed bytes, not an untracked worktree."""

    head = _git_head().lower()
    if len(head) != 40 or any(char not in "0123456789abcdef" for char in head):
        raise RuntimeError("v3.4 Stage 2 Git HEAD is invalid")
    if not _tracked_worktree_clean():
        raise RuntimeError("v3.4 Stage 2 requires a clean tracked worktree")
    files: list[dict[str, str]] = []
    for relative in OFFICIAL_CODE_DEPENDENCIES:
        path = (ROOT / relative).resolve()
        try:
            path.relative_to(ROOT.resolve())
        except ValueError as error:
            raise RuntimeError("v3.4 Stage 2 dependency escapes the repo") from error
        if not path.is_file():
            raise RuntimeError(f"v3.4 Stage 2 dependency is missing: {relative}")
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        if tracked.returncode != 0:
            raise RuntimeError(
                f"v3.4 Stage 2 dependency is not committed: {relative}"
            )
        committed = subprocess.run(
            ["git", "show", f"{head}:{relative}"],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        working_bytes = path.read_bytes()
        if committed.returncode != 0 or committed.stdout != working_bytes:
            raise RuntimeError(
                f"v3.4 Stage 2 dependency differs from HEAD: {relative}"
            )
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(working_bytes).hexdigest(),
            }
        )
    if _git_head().lower() != head or not _tracked_worktree_clean():
        raise RuntimeError(
            "v3.4 Stage 2 code changed while dependency inventory was read"
        )
    return {
        "code_commit": head,
        "files": files,
        "inventory_sha256": _canonical_sha256(files),
    }


def _assert_dependency_inventory_unchanged(
    expected: Mapping[str, object],
) -> dict[str, object]:
    """Reopen every committed dependency and reject run-time TOCTOU drift."""

    actual = _committed_dependency_inventory()
    if actual != dict(expected):
        raise RuntimeError("v3.4 Stage 2 code changed during the official run")
    return actual


def _manifest_entries(path: Path) -> dict[str, Mapping[str, object]]:
    payload = _read_json(path)
    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise ValueError("v3.4 stage2 relative manifest lacks pages")
    result = {
        str(row.get("page_id") or ""): row
        for row in pages
        if isinstance(row, Mapping)
    }
    if not result or "" in result or len(result) != len(pages):
        raise ValueError("v3.4 stage2 relative page inventory is invalid")
    return result


@dataclass(frozen=True, slots=True)
class MaskBinding:
    page_id: str
    role: str
    shape: tuple[int, int]
    pixel_sha256: str
    pixel_count: int
    storage: str
    artifact_id: str | None
    artifact_path: Path | None
    artifact_file_sha256: str | None


@dataclass(frozen=True, slots=True)
class MaskOnlyCandidateInput:
    candidate_id: str
    mode: str
    code_commit: str
    inventory_sha256: str
    mask_result_sha256: str
    candidate_metrics: Mapping[str, object]
    page_metrics: Mapping[str, Mapping[str, object]]
    bindings: Mapping[str, Mapping[str, MaskBinding]]


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def load_mask_only_candidate(
    mask_only_result_path: Path,
    *,
    candidate_id: str,
    relative_manifest_sha256: str,
    page_ids: set[str],
    require_clean_head: bool = True,
) -> MaskOnlyCandidateInput:
    """Authenticate the normalized v3.4 mask inventory before opening masks."""

    if candidate_id not in SUPPORTED_CANDIDATES:
        raise ValueError("v3.4 stage2 candidate is unsupported")
    root = mask_only_result_path.resolve().parent
    payload = _read_json(mask_only_result_path)
    if payload.get("schema_version") != MASK_RESULT_SCHEMA_VERSION:
        raise ValueError("v3.4 stage2 mask-only schema differs")
    if payload.get("relative_manifest", {}).get("manifest_sha256") != (
        relative_manifest_sha256
    ):
        raise ValueError("v3.4 stage2 baseline manifest binding differs")
    shortlist = payload.get("shortlist")
    if not isinstance(shortlist, list) or candidate_id not in shortlist:
        raise ValueError("v3.4 stage2 candidate is not in the sealed shortlist")
    candidate = next(
        (
            row
            for row in payload.get("candidates", [])
            if isinstance(row, Mapping) and row.get("candidate_id") == candidate_id
        ),
        None,
    )
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("mask_only_safety_pass") is not True
        or candidate.get("target_coverage_nonregression") is not True
        or candidate.get("mode") != SUPPORTED_CANDIDATES[candidate_id]
    ):
        raise ValueError("v3.4 stage2 candidate failed mask-only admission")

    code_commit = str(payload.get("code_commit") or "").lower()
    if len(code_commit) != 40 or any(char not in "0123456789abcdef" for char in code_commit):
        raise ValueError("v3.4 mask-only code commit is invalid")
    if require_clean_head:
        if payload.get("tracked_worktree_clean") is not True:
            raise ValueError("v3.4 mask-only evidence was not created on a clean HEAD")
        if code_commit != _git_head().lower() or not _tracked_worktree_clean():
            raise ValueError("v3.4 stage2 must run on the exact clean mask-only HEAD")

    output = payload.get("output_inventory")
    if not isinstance(output, Mapping):
        raise ValueError("v3.4 mask-only output inventory binding is missing")
    relative = Path(str(output.get("relative_path") or ""))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("v3.4 mask-only inventory path is unsafe")
    inventory_path = (root / relative).resolve()
    try:
        inventory_path.relative_to(root)
    except ValueError as error:
        raise ValueError("v3.4 mask-only inventory escapes its run") from error
    if not inventory_path.is_file() or _sha256(inventory_path) != output.get(
        "artifact_sha256"
    ):
        raise ValueError("v3.4 mask-only inventory file SHA differs")
    inventory = _read_json(inventory_path)
    if inventory.get("schema_version") != MASK_INVENTORY_SCHEMA_VERSION:
        raise ValueError("v3.4 mask-only inventory schema differs")
    if inventory.get("inventory_sha256") != output.get("inventory_sha256"):
        raise ValueError("v3.4 mask-only inventory canonical SHA differs")
    source_manifest = payload.get("source_manifest")
    if (
        inventory.get("relative_manifest_sha256") != relative_manifest_sha256
        or not isinstance(source_manifest, Mapping)
        or inventory.get("source_manifest_sha256")
        != source_manifest.get("manifest_sha256")
    ):
        raise ValueError("v3.4 mask-only inventory manifest binding differs")
    validate_mask_output_inventory(inventory, output_root=root)

    inventory_pages = {str(value) for value in inventory.get("page_ids", [])}
    if inventory_pages != page_ids:
        raise ValueError("v3.4 mask-only page inventory differs")
    artifact_rows = inventory.get("artifacts")
    binding_rows = inventory.get("mask_bindings")
    if not isinstance(artifact_rows, list) or not isinstance(binding_rows, list):
        raise ValueError("v3.4 mask-only normalized inventory is incomplete")
    artifacts = {
        str(row.get("artifact_id") or ""): row
        for row in artifact_rows
        if isinstance(row, Mapping)
    }
    selected: dict[str, dict[str, MaskBinding]] = {
        page_id: {} for page_id in page_ids
    }
    for row in binding_rows:
        if not isinstance(row, Mapping) or row.get("candidate_id") != candidate_id:
            continue
        page_id = str(row.get("page_id") or "")
        role = str(row.get("role") or "")
        if page_id not in selected or role not in REQUIRED_INPUT_MASK_ROLES:
            continue
        raw_shape = row.get("shape")
        if (
            not isinstance(raw_shape, list)
            or len(raw_shape) != 2
            or any(not isinstance(value, int) or value <= 0 for value in raw_shape)
        ):
            raise ValueError("v3.4 mask-only binding shape is invalid")
        shape = (int(raw_shape[0]), int(raw_shape[1]))
        storage = str(row.get("storage") or "")
        artifact_id: str | None = None
        artifact_path: Path | None = None
        artifact_file_sha: str | None = None
        if storage == "artifact":
            artifact_id = str(row.get("artifact_id") or "")
            artifact = artifacts.get(artifact_id)
            if artifact is None:
                raise ValueError("v3.4 mask-only binding artifact is missing")
            artifact_relative = Path(str(artifact.get("relative_path") or ""))
            artifact_path = (root / artifact_relative).resolve()
            try:
                artifact_path.relative_to(root)
            except ValueError as error:
                raise ValueError("v3.4 mask-only artifact escapes its run") from error
            artifact_file_sha = str(artifact.get("file_sha256") or "")
        elif storage != "inline_zero":
            raise ValueError("v3.4 mask-only binding storage differs")
        if role in selected[page_id]:
            raise ValueError("v3.4 mask-only candidate binding is duplicated")
        selected[page_id][role] = MaskBinding(
            page_id=page_id,
            role=role,
            shape=shape,
            pixel_sha256=str(row.get("pixel_sha256") or ""),
            pixel_count=_positive_int(row.get("pixel_count"), label="pixel_count"),
            storage=storage,
            artifact_id=artifact_id,
            artifact_path=artifact_path,
            artifact_file_sha256=artifact_file_sha,
        )
    if any(set(rows) != set(REQUIRED_INPUT_MASK_ROLES) for rows in selected.values()):
        raise ValueError("v3.4 mask-only candidate binding roles are incomplete")

    stored_pages = payload.get("pages")
    candidate_pages = (
        stored_pages.get(candidate_id) if isinstance(stored_pages, Mapping) else None
    )
    if not isinstance(candidate_pages, list):
        raise ValueError("v3.4 mask-only candidate page metrics are missing")
    page_metrics = {
        str(row.get("page_id") or ""): row
        for row in candidate_pages
        if isinstance(row, Mapping)
    }
    if set(page_metrics) != page_ids:
        raise ValueError("v3.4 mask-only candidate page metrics differ")
    return MaskOnlyCandidateInput(
        candidate_id=candidate_id,
        mode=SUPPORTED_CANDIDATES[candidate_id],
        code_commit=code_commit,
        inventory_sha256=str(inventory["inventory_sha256"]),
        mask_result_sha256=_sha256(mask_only_result_path),
        candidate_metrics=candidate,
        page_metrics=page_metrics,
        bindings=selected,
    )


def reopen_bound_mask(binding: MaskBinding) -> np.ndarray:
    if binding.storage == "inline_zero":
        result = np.zeros(binding.shape, dtype=np.uint8)
    else:
        if (
            binding.artifact_path is None
            or binding.artifact_file_sha256 is None
            or _sha256(binding.artifact_path) != binding.artifact_file_sha256
        ):
            raise ValueError("v3.4 stage2 bound mask file SHA differs")
        result = _read_mask(binding.artifact_path, binding.shape)
    if (
        mask_sha256(result) != binding.pixel_sha256
        or int(np.count_nonzero(result)) != binding.pixel_count
    ):
        raise ValueError("v3.4 stage2 bound mask pixels differ")
    return np.ascontiguousarray(result)


def reconstruct_page_plan(
    *,
    mode: str,
    generation_mask: np.ndarray,
    commit_mask: np.ndarray,
    replacement_source_edit: np.ndarray,
    baseline_mask: np.ndarray,
) -> PageIncrementalInpaintPlan:
    """Reconstruct exactly one normalized mask-only page plan."""

    baseline = binary_mask(baseline_mask)
    shape = baseline.shape
    generation = binary_mask(generation_mask, shape)
    commit = binary_mask(commit_mask, shape)
    replacement = binary_mask(replacement_source_edit, shape)
    if not np.any(commit):
        if np.any(generation) or np.any(replacement):
            raise ValueError("empty v3.4 commit must have empty generation/replacement")
        return union_incremental_plans((), shape=shape)
    if mode == "context_additive":
        if np.any(replacement):
            raise ValueError("context-additive plan cannot replace PR6 pixels")
        if np.any((commit > 0) & (baseline > 0)):
            raise ValueError("context-additive commit overlaps PR6 baseline")
        context = np.where(
            (generation > 0) & (baseline > 0), 255, 0
        ).astype(np.uint8)
        expected_generation = cv2.bitwise_or(commit, context)
        if not np.array_equal(generation, expected_generation):
            raise ValueError("context-additive generation contains unbound context")
        plan = IncrementalInpaintPlan(
            mode=mode,
            generation_mask=generation,
            commit_mask=commit,
            existing_source_edit=context,
        )
    elif mode in {"narrow_replacement", "conditional_segmenter"}:
        if np.any((replacement > 0) & (baseline == 0)):
            raise ValueError("replacement source edit escapes PR6 baseline")
        expected_generation = cv2.bitwise_or(commit, replacement)
        if not np.array_equal(generation, expected_generation):
            raise ValueError("replacement generation differs from commit/context union")
        plan = IncrementalInpaintPlan(
            mode=mode,
            generation_mask=generation,
            commit_mask=commit,
            existing_source_edit=replacement,
        )
    else:
        raise ValueError(f"unsupported v3.4 stage2 mode: {mode}")
    return union_incremental_plans((plan,), shape=shape)


@dataclass(frozen=True, slots=True)
class Stage2PageExecution:
    generated: np.ndarray | None
    receipt: IncrementalInpaintReceipt
    candidate: np.ndarray
    final_mask: np.ndarray
    affected_union: np.ndarray
    incremental_changed: np.ndarray


def execute_stage2_page(
    *,
    source: np.ndarray,
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    plan: PageIncrementalInpaintPlan,
    inpaint: Callable[[np.ndarray, np.ndarray], np.ndarray],
    provider: str,
    expected_final_mask_sha256: str | None = None,
) -> Stage2PageExecution:
    """Generate from the immutable source and commit only the affected union."""

    generated, receipt = execute_incremental_inpaint(
        source,
        plan,
        inpaint,
        provider=provider,
    )
    candidate, final_mask = composite_incremental_result(
        source,
        baseline,
        generated,
        baseline_mask,
        plan,
        receipt,
    )
    baseline_binary = binary_mask(baseline_mask, plan.shape)
    expected_final = cv2.bitwise_or(
        np.where(
            (baseline_binary > 0) & (plan.replacement_source_edit == 0),
            255,
            0,
        ).astype(np.uint8),
        plan.commit_mask,
    )
    if not np.array_equal(final_mask, expected_final):
        raise AssertionError("v3.4 stage2 final mask differs from the exact plan")
    if expected_final_mask_sha256 is not None and mask_sha256(final_mask) != (
        expected_final_mask_sha256
    ):
        raise ValueError("v3.4 stage2 final mask differs from mask-only scoring")
    affected = cv2.bitwise_or(plan.commit_mask, plan.replacement_source_edit)
    incremental_changed = changed_mask(baseline, candidate)
    if np.any((incremental_changed > 0) & (affected == 0)):
        raise AssertionError("v3.4 stage2 changed outside its affected union")
    if plan.lama_request_count == 0 and (
        generated is not None
        or not np.array_equal(candidate, baseline)
        or not np.array_equal(final_mask, baseline_binary)
    ):
        raise AssertionError("empty v3.4 stage2 plan changed PR6 bytes")
    return Stage2PageExecution(
        generated=generated,
        receipt=receipt,
        candidate=np.ascontiguousarray(candidate),
        final_mask=np.ascontiguousarray(final_mask),
        affected_union=np.ascontiguousarray(affected),
        incremental_changed=np.ascontiguousarray(incremental_changed),
    )


def validate_cuda_bf16_diagnostics(
    diagnostics: Sequence[Mapping[str, object]],
    *,
    expected_request_count: int,
) -> dict[str, object]:
    if len(diagnostics) != expected_request_count:
        raise RuntimeError("v3.4 LaMa diagnostic count differs from request count")
    cpu_fallback_count = 0
    for row in diagnostics:
        device = str(row.get("actual_device") or "").lower()
        precision = str(row.get("actual_precision") or "").lower()
        status = str(row.get("status") or "")
        cpu_fallback = bool(row.get("cpu_fallback_used", False))
        cpu_fallback_count += int(cpu_fallback)
        if (
            not device.startswith("cuda")
            or precision != "bf16"
            or cpu_fallback
            or row.get("device_verified_from_model") is not True
            or status not in {"completed", "completed_after_roi_retry"}
        ):
            raise RuntimeError("v3.4 Stage 2 requires verified CUDA bf16 LaMa")
    return {
        "runtime_telemetry_complete": True,
        "cpu_fallback_count": cpu_fallback_count,
        "actual_devices": sorted(
            {str(row.get("actual_device") or "") for row in diagnostics}
        ),
        "actual_precisions": sorted(
            {str(row.get("actual_precision") or "") for row in diagnostics}
        ),
    }


def _target_instance_scores(page, final_mask: np.ndarray) -> list[dict[str, object]]:
    scores: list[dict[str, object]] = []
    for record in page.target_instances:
        if record.priority != "required":
            continue
        target = _read_mask(record.mask_path, final_mask.shape)
        pixels = int(np.count_nonzero(target))
        covered = int(np.count_nonzero((target > 0) & (final_mask > 0)))
        scores.append(
            {
                "instance_id": record.instance_id,
                "coverage": float(covered) / pixels if pixels else 0.0,
            }
        )
    return scores


def _optional_neutral_mask(page, shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=np.uint8)
    for record in page.target_instances:
        if record.priority == "optional":
            result[_read_mask(record.mask_path, shape) > 0] = 255
    return result


def _runtime_action_masks(
    entry: Mapping[str, object],
    masks,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    raw_regions = entry.get("regions")
    if not isinstance(raw_regions, list) or len(raw_regions) != len(masks.regions):
        raise ValueError("v3.4 stage2 region action inventory differs")
    preserve = np.zeros(shape, dtype=np.uint8)
    abstain = np.zeros(shape, dtype=np.uint8)
    for raw_region, region_masks in zip(raw_regions, masks.regions):
        if not isinstance(raw_region, Mapping) or str(
            raw_region.get("region_id") or ""
        ) != region_masks.region_id:
            raise ValueError("v3.4 stage2 region action order differs")
        decision = product_semantic_decision(raw_region)
        if decision.action == PRESERVE:
            preserve[region_masks.ownership > 0] = 255
        elif decision.action != TRANSLATE or not decision.available:
            abstain[region_masks.ownership > 0] = 255
    return preserve, abstain


def _aggregate_residue(rows: Sequence[Mapping[str, object]]) -> float | None:
    total = sum(float(row["residue_score_sum"]) for row in rows)
    count = sum(int(row["residue_source_contrast_pixel_count"]) for row in rows)
    return total / count if count else None


def _aggregate_coverage(rows: Sequence[Mapping[str, object]]) -> float:
    total = sum(int(row["target_pixel_count"]) for row in rows)
    covered = sum(int(row["target_edit_pixel_count"]) for row in rows)
    return float(covered) / total if total else 0.0


def _mask_set_sha(
    rows: Sequence[Mapping[str, object]], field: str
) -> str:
    return _canonical_sha256(
        sorted(
            (
                {"page_id": str(row["page_id"]), field: str(row[field])}
                for row in rows
            ),
            key=lambda row: row["page_id"],
        )
    )


def _artifact_record(
    *,
    output_root: Path,
    page_id: str,
    role: str,
    value: np.ndarray,
    source_binding: MaskBinding | None = None,
) -> dict[str, object]:
    if role not in OUTPUT_ARTIFACT_ROLES and role != "generated_image":
        raise ValueError(f"unknown v3.4 Stage 2 artifact role: {role}")
    path = output_root / role / f"{page_id}.png"
    result = {
        "page_id": page_id,
        "role": role,
        "relative_path": path.relative_to(output_root).as_posix(),
        **_write_png(path, value),
    }
    if source_binding is not None:
        result["source_mask_binding"] = {
            "role": source_binding.role,
            "storage": source_binding.storage,
            "artifact_id": source_binding.artifact_id,
            "pixel_sha256": source_binding.pixel_sha256,
            "pixel_count": source_binding.pixel_count,
        }
    return result


def validate_stage2_output_inventory(
    inventory: Mapping[str, object], *, output_root: Path
) -> None:
    unsigned = {
        key: value for key, value in inventory.items() if key != "inventory_sha256"
    }
    if inventory.get("inventory_sha256") != _canonical_sha256(unsigned):
        raise ValueError("v3.4 Stage 2 output inventory canonical SHA differs")
    if inventory.get("schema_version") != OUTPUT_INVENTORY_SCHEMA_VERSION:
        raise ValueError("v3.4 Stage 2 output inventory schema differs")
    page_ids = {str(value) for value in inventory.get("page_ids", [])}
    records = inventory.get("records")
    if not page_ids or not isinstance(records, list):
        raise ValueError("v3.4 Stage 2 output inventory is incomplete")
    identities: set[tuple[str, str]] = set()
    required = {
        (page_id, role) for page_id in page_ids for role in OUTPUT_ARTIFACT_ROLES
    }
    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError("v3.4 Stage 2 output record is invalid")
        page_id = str(row.get("page_id") or "")
        role = str(row.get("role") or "")
        identity = (page_id, role)
        if page_id not in page_ids or identity in identities:
            raise ValueError("v3.4 Stage 2 output identity differs")
        identities.add(identity)
        if role not in set(OUTPUT_ARTIFACT_ROLES) | {"generated_image"}:
            raise ValueError("v3.4 Stage 2 output role differs")
        relative = Path(str(row.get("relative_path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("v3.4 Stage 2 output path is unsafe")
        path = (output_root / relative).resolve()
        try:
            path.relative_to(output_root.resolve())
        except ValueError as error:
            raise ValueError("v3.4 Stage 2 output escapes its run") from error
        if _sha256(path) != row.get("file_sha256"):
            raise ValueError("v3.4 Stage 2 output file SHA differs")
        raw_shape = row.get("shape")
        if (
            not isinstance(raw_shape, list)
            or len(raw_shape) not in {2, 3}
            or any(not isinstance(value, int) or value <= 0 for value in raw_shape)
            or row.get("dtype") != "uint8"
        ):
            raise ValueError("v3.4 Stage 2 output array metadata differs")
        decoded = _read_image(
            path,
            cv2.IMREAD_COLOR
            if len(raw_shape) == 3
            else cv2.IMREAD_GRAYSCALE,
        )
        if list(decoded.shape) != raw_shape or str(decoded.dtype) != row.get("dtype"):
            raise ValueError("v3.4 Stage 2 output decoded shape/dtype differs")
        if _array_bytes_sha256(decoded) != row.get("pixel_bytes_sha256"):
            raise ValueError("v3.4 Stage 2 output pixel bytes differ")
        if decoded.ndim == 2 and (
            np.any((decoded != 0) & (decoded != 255))
            or mask_sha256(decoded) != row.get("mask_pixel_sha256")
            or int(np.count_nonzero(decoded)) != row.get("pixel_count")
        ):
            raise ValueError("v3.4 Stage 2 output mask pixels differ")
        expected_source_role = OUTPUT_SOURCE_BINDING_ROLES.get(role)
        source_binding = row.get("source_mask_binding")
        if expected_source_role is not None:
            if (
                not isinstance(source_binding, Mapping)
                or source_binding.get("role") != expected_source_role
                or source_binding.get("pixel_sha256")
                != row.get("mask_pixel_sha256")
                or source_binding.get("pixel_count") != row.get("pixel_count")
                or source_binding.get("storage") not in {"artifact", "inline_zero"}
            ):
                raise ValueError("v3.4 Stage 2 output source-mask binding differs")
        elif source_binding is not None:
            raise ValueError("derived v3.4 Stage 2 output has a source binding")
    if not required.issubset(identities):
        raise ValueError("v3.4 Stage 2 output lacks required actual bytes")


def run_candidate(
    *,
    relative_manifest_path: Path,
    mask_only_result_path: Path,
    candidate_id: str,
    output_root: Path,
    device: str = "cuda:0",
    precision: str = "bf16",
    inpaint_size: int = 1536,
) -> dict[str, object]:
    if not str(device).lower().startswith("cuda") or precision != "bf16":
        raise ValueError("official v3.4 Stage 2 requires CUDA bf16")
    code_inventory_start = _committed_dependency_inventory()
    manifest_binding = validate_source_only_manifest_v4(relative_manifest_path)
    pages = load_stage1_manifest(relative_manifest_path)
    page_ids = {page.page_id for page in pages}
    entries = _manifest_entries(relative_manifest_path)
    if set(entries) != page_ids:
        raise ValueError("v3.4 Stage 2 manifest page inventory differs")
    mask_input = load_mask_only_candidate(
        mask_only_result_path,
        candidate_id=candidate_id,
        relative_manifest_sha256=str(manifest_binding["manifest_sha256"]),
        page_ids=page_ids,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise FileExistsError("v3.4 Stage 2 output must be fresh")

    model_path = Path(
        ModelDownloader.primary_path(ModelID.LAMA_LARGE_512PX)
    ).resolve()
    model_sha_before = _sha256(model_path)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    inpainter = SourceLaMaLarge(
        device=device,
        precision=precision,
        inpaint_size=inpaint_size,
    )
    inpainter.ensure_loaded()
    loaded_runtime = inspect_learned_inpainter_runtime(
        inpainter,
        inpainter_key="lama_large_512px",
        requested_device=device,
        requested_precision=precision,
    )
    if (
        not str(loaded_runtime.get("actual_device") or "").lower().startswith(
            "cuda"
        )
        or str(loaded_runtime.get("actual_precision") or "").lower() != "bf16"
        or loaded_runtime.get("device_verified_from_model") is not True
        or loaded_runtime.get("cpu_fallback_used") is not False
    ):
        raise RuntimeError("v3.4 Stage 2 loaded runtime is not CUDA bf16")
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    diagnostics_before = len(inpainter.run_diagnostics)
    inference_count = 0
    baseline_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    artifacts: list[dict[str, object]] = []
    source_inventory: list[dict[str, object]] = []

    def inpaint(source_bgr: np.ndarray, generation_mask: np.ndarray) -> np.ndarray:
        generated_rgb = inpainter.memory_safe_inpaint(
            cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB),
            generation_mask,
        )
        return np.ascontiguousarray(cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2BGR))

    for page in pages:
        entry = entries[page.page_id]
        source_path = Path(page.source_image).resolve()
        source = _read_image(source_path, cv2.IMREAD_COLOR)
        shape = source.shape[:2]
        declared_source_sha = str(entry.get("source_sha256") or "").lower()
        actual_source_sha = _sha256(source_path)
        if declared_source_sha != actual_source_sha:
            raise ValueError("v3.4 Stage 2 source SHA differs from sealed manifest")
        baseline_path = _resolve_manifest_artifact(
            relative_manifest_path, entry.get("baseline")
        )
        baseline_mask_path = _resolve_manifest_artifact(
            relative_manifest_path, entry.get("baseline_mask")
        )
        if baseline_path is None or baseline_mask_path is None:
            raise ValueError(f"v3.4 Stage 2 baseline is missing: {page.page_id}")
        baseline = _read_image(baseline_path, cv2.IMREAD_COLOR)
        baseline_mask = _read_mask(baseline_mask_path, shape)
        if baseline.shape != source.shape:
            raise ValueError("v3.4 Stage 2 baseline image shape differs")
        bindings = mask_input.bindings[page.page_id]
        if any(binding.shape != shape for binding in bindings.values()):
            raise ValueError("v3.4 Stage 2 mask/source shape differs")
        seed = reopen_bound_mask(bindings["owned_positive_seed"])
        generation = reopen_bound_mask(bindings["generation_mask"])
        commit = reopen_bound_mask(bindings["commit_mask"])
        replacement = reopen_bound_mask(bindings["replacement_source_edit"])
        if np.any(commit) and not np.any(seed):
            raise ValueError("v3.4 Stage 2 commit lacks a sealed positive seed")
        plan = reconstruct_page_plan(
            mode=mask_input.mode,
            generation_mask=generation,
            commit_mask=commit,
            replacement_source_edit=replacement,
            baseline_mask=baseline_mask,
        )
        mask_page = mask_input.page_metrics[page.page_id]
        expected_final_sha = str(mask_page.get("final_mask_pixel_sha256") or "")
        execution = execute_stage2_page(
            source=source,
            baseline=baseline,
            baseline_mask=baseline_mask,
            plan=plan,
            inpaint=inpaint,
            provider=f"{device}/source_lama_large/{precision}",
            expected_final_mask_sha256=expected_final_sha,
        )
        inference_count += execution.receipt.request_count

        # Evaluation masks are deliberately opened only after source-derived
        # masks, plan reconstruction, generation, and exact compositing finish.
        masks = load_page_masks(page, shape, strict_binary=True)
        baseline_score, baseline_sum, baseline_count = residue_score(
            source, baseline, masks.target
        )
        candidate_score, candidate_sum, candidate_count = residue_score(
            source, execution.candidate, masks.target
        )
        if candidate_count != baseline_count:
            raise AssertionError("v3.4 Stage 2 residue denominator changed")
        baseline_instance_scores = _target_instance_scores(page, baseline_mask)
        candidate_instance_scores = _target_instance_scores(
            page, execution.final_mask
        )
        baseline_instance_map = {
            str(row["instance_id"]): float(row["coverage"])
            for row in baseline_instance_scores
        }
        for row in candidate_instance_scores:
            before = baseline_instance_map[str(row["instance_id"])]
            after = float(row["coverage"])
            row["baseline_coverage"] = before
            row["coverage_delta"] = after - before
            row["regressed_from_98"] = before >= 0.98 and after < 0.98

        optional = _optional_neutral_mask(page, shape)
        explicit_preserve, explicit_abstain = _runtime_action_masks(
            entry, masks, shape
        )
        baseline_changed = changed_mask(source, baseline)
        candidate_changed = changed_mask(source, execution.candidate)
        newly_changed = np.where(
            (candidate_changed > 0) & (baseline_changed == 0), 255, 0
        ).astype(np.uint8)
        restored = np.where(
            (baseline_changed > 0) & (candidate_changed == 0), 255, 0
        ).astype(np.uint8)
        target_pixels = int(np.count_nonzero(masks.target))
        baseline_covered = int(
            np.count_nonzero((masks.target > 0) & (baseline_mask > 0))
        )
        candidate_covered = int(
            np.count_nonzero((masks.target > 0) & (execution.final_mask > 0))
        )
        base_row = {
            "page_id": page.page_id,
            "target_pixel_count": target_pixels,
            "target_edit_pixel_count": baseline_covered,
            "target_instance_edit_scores": baseline_instance_scores,
            "residue_score": baseline_score,
            "residue_score_sum": baseline_sum,
            "residue_source_contrast_pixel_count": baseline_count,
            "output_mask_pixel_sha256": mask_sha256(baseline_mask),
        }
        candidate_row = {
            "page_id": page.page_id,
            "expected_edit": page.expected_edit,
            "target_pixel_count": target_pixels,
            "target_edit_pixel_count": candidate_covered,
            "target_instance_edit_scores": candidate_instance_scores,
            "residue_score": candidate_score,
            "residue_score_sum": candidate_sum,
            "residue_source_contrast_pixel_count": candidate_count,
            "residue_delta_from_pr6": (
                float(candidate_score) - float(baseline_score)
                if candidate_score is not None and baseline_score is not None
                else None
            ),
            "seed_pixel_count": int(np.count_nonzero(seed)),
            "generation_pixel_count": int(np.count_nonzero(generation)),
            "commit_pixel_count": int(np.count_nonzero(commit)),
            "replacement_source_edit_pixel_count": int(
                np.count_nonzero(replacement)
            ),
            "affected_union_pixel_count": int(
                np.count_nonzero(execution.affected_union)
            ),
            "incremental_changed_pixel_count": int(
                np.count_nonzero(execution.incremental_changed)
            ),
            "incremental_changed_outside_affected_union_pixel_count": int(
                np.count_nonzero(
                    (execution.incremental_changed > 0)
                    & (execution.affected_union == 0)
                )
            ),
            "new_protected_changed_pixel_count": int(
                np.count_nonzero((newly_changed > 0) & (masks.protected > 0))
            ),
            "new_ambiguous_changed_pixel_count": int(
                np.count_nonzero((newly_changed > 0) & (masks.ambiguous > 0))
            ),
            "commit_protected_overlap_pixel_count": int(
                np.count_nonzero((commit > 0) & (masks.protected > 0))
            ),
            "commit_ambiguous_overlap_pixel_count": int(
                np.count_nonzero((commit > 0) & (masks.ambiguous > 0))
            ),
            "commit_corner_overlap_pixel_count": int(
                np.count_nonzero((commit > 0) & (masks.corner > 0))
                if masks.corner is not None
                else 0
            ),
            "commit_preserve_overlap_pixel_count": int(
                np.count_nonzero((commit > 0) & (explicit_preserve > 0))
            ),
            "commit_abstain_overlap_pixel_count": int(
                np.count_nonzero((commit > 0) & (explicit_abstain > 0))
            ),
            "commit_ownership_leak_pixel_count": int(
                np.count_nonzero((commit > 0) & (masks.ownership == 0))
            ),
            "no_edit_non_neutral_commit_pixel_count": (
                int(np.count_nonzero((commit > 0) & (optional == 0)))
                if page.expected_edit == "none"
                else 0
            ),
            "optional_neutral_commit_pixel_count": int(
                np.count_nonzero((commit > 0) & (optional > 0))
            ),
            "outside_final_changed_pixel_count": int(
                np.count_nonzero(
                    (candidate_changed > 0) & (execution.final_mask == 0)
                )
            ),
            "inherited_pr6_protected_changed_pixel_count": int(
                np.count_nonzero((baseline_changed > 0) & (masks.protected > 0))
            ),
            "retained_inherited_protected_changed_pixel_count": int(
                np.count_nonzero(
                    (baseline_changed > 0)
                    & (candidate_changed > 0)
                    & (masks.protected > 0)
                )
            ),
            "restored_inherited_protected_pixel_count": int(
                np.count_nonzero((restored > 0) & (masks.protected > 0))
            ),
            "final_protected_changed_pixel_count": int(
                np.count_nonzero((candidate_changed > 0) & (masks.protected > 0))
            ),
            "additional_lama_inference_call_count": execution.receipt.request_count,
            "receipt": {
                "provider": execution.receipt.provider,
                "request_count": execution.receipt.request_count,
                "source_sha256": execution.receipt.source_sha256,
                "generation_mask_sha256": execution.receipt.generation_mask_sha256,
                "generated_sha256": execution.receipt.generated_sha256,
            },
            "seed_pixel_sha256": mask_sha256(seed),
            "generation_mask_pixel_sha256": mask_sha256(generation),
            "commit_mask_pixel_sha256": mask_sha256(commit),
            "replacement_source_edit_pixel_sha256": mask_sha256(replacement),
            "affected_union_pixel_sha256": mask_sha256(execution.affected_union),
            "output_mask_pixel_sha256": mask_sha256(execution.final_mask),
            "candidate_pixel_bytes_sha256": _array_bytes_sha256(
                execution.candidate
            ),
        }
        baseline_rows.append(base_row)
        candidate_rows.append(candidate_row)
        source_inventory.append(
            {
                "page_id": page.page_id,
                "source_file_sha256": actual_source_sha,
                "source_pixel_bytes_sha256": _array_bytes_sha256(source),
                "baseline_file_sha256": _sha256(baseline_path),
                "baseline_pixel_bytes_sha256": _array_bytes_sha256(baseline),
                "baseline_mask_file_sha256": _sha256(baseline_mask_path),
                "baseline_mask_pixel_sha256": mask_sha256(baseline_mask),
            }
        )
        for role, value, binding in (
            ("seed", seed, bindings["owned_positive_seed"]),
            ("generation_mask", generation, bindings["generation_mask"]),
            ("commit_mask", commit, bindings["commit_mask"]),
            (
                "replacement_source_edit",
                replacement,
                bindings["replacement_source_edit"],
            ),
            ("affected_union", execution.affected_union, None),
            ("final_mask", execution.final_mask, None),
            ("incremental_changed_mask", execution.incremental_changed, None),
            ("candidate_image", execution.candidate, None),
        ):
            artifacts.append(
                _artifact_record(
                    output_root=output_root,
                    page_id=page.page_id,
                    role=role,
                    value=value,
                    source_binding=binding,
                )
            )
        if execution.generated is not None:
            artifacts.append(
                _artifact_record(
                    output_root=output_root,
                    page_id=page.page_id,
                    role="generated_image",
                    value=execution.generated,
                )
            )

    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    diagnostics = inpainter.run_diagnostics[diagnostics_before:]
    runtime_validation = validate_cuda_bf16_diagnostics(
        diagnostics,
        expected_request_count=inference_count,
    )
    model_sha_after = _sha256(model_path)
    if model_sha_after != model_sha_before:
        raise RuntimeError("v3.4 LaMa model changed during Stage 2")

    incremental_zero_fields = {
        "protected_structure_overlap": sum(
            int(row["commit_protected_overlap_pixel_count"])
            for row in candidate_rows
        ),
        "protected_structure_changed": sum(
            int(row["new_protected_changed_pixel_count"])
            for row in candidate_rows
        ),
        "ambiguous_structure_overlap": sum(
            int(row["commit_ambiguous_overlap_pixel_count"])
            for row in candidate_rows
        ),
        "ambiguous_structure_changed": sum(
            int(row["new_ambiguous_changed_pixel_count"])
            for row in candidate_rows
        ),
        "preserve_edit_overlap": sum(
            int(row["commit_preserve_overlap_pixel_count"])
            for row in candidate_rows
        ),
        "ownership_leak_pixel_count": sum(
            int(row["commit_ownership_leak_pixel_count"])
            for row in candidate_rows
        ),
        "corner_edit_overlap_pixel_count": sum(
            int(row["commit_corner_overlap_pixel_count"])
            for row in candidate_rows
        ),
        "outside_final_changed": sum(
            int(row["outside_final_changed_pixel_count"])
            for row in candidate_rows
        ),
        "broad_route_false_positive": 0,
        "no_edit_false_edit": sum(
            int(row["no_edit_non_neutral_commit_pixel_count"])
            for row in candidate_rows
        ),
        "required_skip_count": 0,
        "cpu_fallback_count": int(runtime_validation["cpu_fallback_count"]),
    }
    baseline_metrics = {
        "aggregate_target_coverage": _aggregate_coverage(baseline_rows),
        "aggregate_residue_score": _aggregate_residue(baseline_rows),
        "output_mask_set_sha256": _mask_set_sha(
            baseline_rows, "output_mask_pixel_sha256"
        ),
    }
    required_count = int(
        mask_input.candidate_metrics.get("required_target_instance_count") or 0
    )
    seeded_count = int(
        mask_input.candidate_metrics.get(
            "source_positive_seeded_target_instance_count"
        )
        or 0
    )
    candidate_metrics = {
        **incremental_zero_fields,
        "aggregate_target_coverage": _aggregate_coverage(candidate_rows),
        "aggregate_residue_score": _aggregate_residue(candidate_rows),
        "output_mask_set_sha256": _mask_set_sha(
            candidate_rows, "output_mask_pixel_sha256"
        ),
        "runtime_telemetry_complete": runtime_validation[
            "runtime_telemetry_complete"
        ],
        "maximum_positive_lama_inference_per_page": max(
            (
                int(row["additional_lama_inference_call_count"])
                for row in candidate_rows
            ),
            default=0,
        ),
        "positive_lama_inference_count": inference_count,
        "lama_runtime_provider": device,
        "lama_runtime_precision": precision,
        "target_instance_seed_recall": (
            float(seeded_count) / required_count if required_count else 0.0
        ),
        "missed_target_instance_count": required_count - seeded_count,
    }
    gate = evaluate_relative_product_gate(
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        baseline_pages=baseline_rows,
        candidate_pages=candidate_rows,
        candidate_kind="balanced",
    )
    abstain_overlap = sum(
        int(row["commit_abstain_overlap_pixel_count"])
        for row in candidate_rows
    )
    affected_outside = sum(
        int(row["incremental_changed_outside_affected_union_pixel_count"])
        for row in candidate_rows
    )
    if abstain_overlap or affected_outside:
        failures = list(gate["gate_failures"])
        if abstain_overlap:
            failures.append("safety_nonzero:abstain_edit_overlap")
        if affected_outside:
            failures.append("safety_nonzero:outside_affected_union")
        gate["relative_product_pass"] = False
        gate["gate_failures"] = sorted(set(failures))

    artifacts.sort(key=lambda row: (str(row["page_id"]), str(row["role"])))
    inventory_unsigned: dict[str, object] = {
        "schema_version": OUTPUT_INVENTORY_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "code_commit": code_inventory_start["code_commit"],
        "code_dependency_inventory_sha256": code_inventory_start[
            "inventory_sha256"
        ],
        "source_manifest_sha256": manifest_binding["manifest_sha256"],
        "source_page_inventory_sha256": manifest_binding["page_inventory_sha256"],
        "mask_only_result_sha256": mask_input.mask_result_sha256,
        "mask_only_output_inventory_sha256": mask_input.inventory_sha256,
        "page_ids": sorted(page_ids),
        "source_inputs": sorted(source_inventory, key=lambda row: row["page_id"]),
        "records": artifacts,
    }
    inventory = dict(inventory_unsigned)
    inventory["inventory_sha256"] = _canonical_sha256(inventory_unsigned)
    inventory_path = output_root / "output-artifact-inventory.json"
    _write_json(inventory_path, inventory)
    validate_stage2_output_inventory(inventory, output_root=output_root)
    code_inventory_end = _assert_dependency_inventory_unchanged(
        code_inventory_start
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "candidate_mode": mask_input.mode,
        "code_commit": _git_head(),
        "tracked_worktree_clean": _tracked_worktree_clean(),
        "code_dependency_inventory": code_inventory_end,
        "source_manifest": dict(manifest_binding),
        "mask_only": {
            "result_sha256": mask_input.mask_result_sha256,
            "output_inventory_sha256": mask_input.inventory_sha256,
            "code_commit": mask_input.code_commit,
            "candidate_output_mask_set_sha256": mask_input.candidate_metrics.get(
                "output_mask_set_sha256"
            ),
        },
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "relative_gate": gate,
        "incremental": {
            **incremental_zero_fields,
            "abstain_edit_overlap": abstain_overlap,
            "outside_affected_union": affected_outside,
            "optional_neutral_commit_pixel_count": sum(
                int(row["optional_neutral_commit_pixel_count"])
                for row in candidate_rows
            ),
        },
        "inherited_pr6": {
            "protected_changed_pixel_count": sum(
                int(row["inherited_pr6_protected_changed_pixel_count"])
                for row in candidate_rows
            ),
            "retained_protected_changed_pixel_count": sum(
                int(row["retained_inherited_protected_changed_pixel_count"])
                for row in candidate_rows
            ),
            "restored_protected_pixel_count": sum(
                int(row["restored_inherited_protected_pixel_count"])
                for row in candidate_rows
            ),
            "note": (
                "Inherited PR6 damage is reported separately; replacement "
                "restoration is not counted as newly introduced damage."
            ),
        },
        "runtime": {
            "requested_device": device,
            "requested_precision": precision,
            "inpaint_size": inpaint_size,
            "page_count": len(candidate_rows),
            "elapsed_seconds": elapsed,
            "additional_lama_inference_count": inference_count,
            "maximum_additional_lama_inference_per_page": max(
                (
                    int(row["additional_lama_inference_call_count"])
                    for row in candidate_rows
                ),
                default=0,
            ),
            "cpu_fallback_count": runtime_validation["cpu_fallback_count"],
            "actual_devices": runtime_validation["actual_devices"],
            "actual_precisions": runtime_validation["actual_precisions"],
            "loaded_runtime": loaded_runtime,
            "torch_cuda_version": str(torch.version.cuda or ""),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "peak_vram_allocated_mib": float(torch.cuda.max_memory_allocated(device))
            / (1024.0 * 1024.0),
            "peak_vram_reserved_mib": float(torch.cuda.max_memory_reserved(device))
            / (1024.0 * 1024.0),
            "lama_model_sha256_before": model_sha_before,
            "lama_model_sha256_after": model_sha_after,
            "diagnostics": diagnostics,
        },
        "baseline_pages": baseline_rows,
        "pages": candidate_rows,
        "output_inventory": {
            "relative_path": inventory_path.relative_to(output_root).as_posix(),
            "artifact_sha256": _sha256(inventory_path),
            "inventory_sha256": inventory["inventory_sha256"],
            "artifact_count": len(artifacts),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one sealed v3.4 glyph-refinement candidate with CUDA bf16 LaMa."
        )
    )
    parser.add_argument("--relative-manifest", type=Path, required=True)
    parser.add_argument("--mask-only-result", type=Path, required=True)
    parser.add_argument(
        "--candidate-id", choices=tuple(sorted(SUPPORTED_CANDIDATES)), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inpaint-size", type=int, default=1536)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--require-relative-gate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    try:
        result = run_candidate(
            relative_manifest_path=args.relative_manifest.resolve(),
            mask_only_result_path=args.mask_only_result.resolve(),
            candidate_id=args.candidate_id,
            output_root=output_root,
            device=args.device,
            precision="bf16",
            inpaint_size=args.inpaint_size,
        )
        code_inventory = result.get("code_dependency_inventory")
        if not isinstance(code_inventory, Mapping):
            raise RuntimeError("v3.4 Stage 2 result lacks its code inventory")
        _assert_dependency_inventory_unchanged(code_inventory)
        result_path = output_root / "stage2-results.json"
        _write_json(result_path, result)
        _assert_dependency_inventory_unchanged(code_inventory)
        if managed is not None:
            managed.complete(
                metadata={
                    "candidate_id": args.candidate_id,
                    "relative_product_pass": result["relative_gate"][
                        "relative_product_pass"
                    ],
                    "source_page_inventory_sha256": result["source_manifest"][
                        "page_inventory_sha256"
                    ],
                    "output_inventory_sha256": result["output_inventory"][
                        "inventory_sha256"
                    ],
                }
            )
            mismatches = managed.verify()
            if mismatches:
                raise RuntimeError(
                    "managed artifact verification failed: " + "; ".join(mismatches)
                )
            print(managed.run_root)
        else:
            print(result_path)
        if args.require_relative_gate and not result["relative_gate"][
            "relative_product_pass"
        ]:
            return 1
        return 0
    except BaseException as error:
        if managed is not None:
            managed.fail(error, metadata={"candidate_id": args.candidate_id})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
