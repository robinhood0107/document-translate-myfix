#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

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
from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    _resolve_manifest_artifact,
    load_page_masks,
    load_stage1_manifest,
    validate_source_only_manifest_v4,
)
from benchmarking.inpaint_detector_bakeoff.stage2 import (  # noqa: E402
    changed_mask,
    composite_positive_result,
    evaluate_relative_product_gate,
    residue_score,
)
from benchmarking.inpaint_detector_bakeoff.semantic import (  # noqa: E402
    PRESERVE,
    TRANSLATE,
    product_semantic_decision,
)
from modules.inpainting.source_lama_blockwise import SourceLaMaLarge  # noqa: E402
from modules.utils.download import ModelDownloader, ModelID  # noqa: E402
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-proposal-stage2-v33"
CATEGORY = "40-inpaint-mask-render"
SCHEMA_VERSION = "inpaint-proposal-stage2-results-v33"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _read_image(path: str | Path, flags: int) -> np.ndarray:
    result = cv2.imdecode(np.fromfile(Path(path), dtype=np.uint8), flags)
    if result is None or result.size == 0:
        raise FileNotFoundError(path)
    return result


def _read_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    result = _read_image(path, cv2.IMREAD_GRAYSCALE)
    if result.shape != shape:
        raise ValueError(f"v3.3 stage2 mask shape mismatch: {result.shape} != {shape}")
    unique = np.unique(result)
    if np.any((unique != 0) & (unique != 255)):
        raise ValueError("v3.3 stage2 mask must be strict binary")
    return binary_mask(result, shape)


def _write_png(path: Path, value: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"v3.3 stage2 output must be fresh: {path}")
    encoded, buffer = cv2.imencode(".png", np.ascontiguousarray(value))
    if not encoded:
        raise OSError(f"failed to encode v3.3 stage2 output: {path}")
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_bytes(buffer.tobytes())
    temporary.replace(path)
    decoded = _read_image(
        path,
        cv2.IMREAD_COLOR if value.ndim == 3 else cv2.IMREAD_GRAYSCALE,
    )
    if not np.array_equal(decoded, value):
        raise RuntimeError(f"v3.3 stage2 output changed during encoding: {path}")
    return {
        "file_sha256": _sha256(path),
        "pixel_sha256": hashlib.sha256(
            np.ascontiguousarray(decoded).tobytes()
        ).hexdigest(),
        "shape": list(decoded.shape),
        "dtype": str(decoded.dtype),
    }


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _manifest_entries(path: Path) -> dict[str, Mapping[str, object]]:
    payload = _read_json(path)
    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise ValueError("v3.3 stage2 relative manifest lacks pages")
    result = {
        str(row.get("page_id") or ""): row
        for row in pages
        if isinstance(row, Mapping)
    }
    if not result or "" in result or len(result) != len(pages):
        raise ValueError("v3.3 stage2 relative page inventory is invalid")
    return result


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
        raise ValueError("v3.3 stage2 region action inventory differs")
    preserve = np.zeros(shape, dtype=np.uint8)
    abstain = np.zeros(shape, dtype=np.uint8)
    for raw_region, region_masks in zip(raw_regions, masks.regions):
        if not isinstance(raw_region, Mapping) or str(
            raw_region.get("region_id") or ""
        ) != region_masks.region_id:
            raise ValueError("v3.3 stage2 region action order differs")
        decision = product_semantic_decision(raw_region)
        if decision.action == PRESERVE:
            preserve[region_masks.ownership > 0] = 255
        elif decision.action != TRANSLATE or not decision.available:
            abstain[region_masks.ownership > 0] = 255
    return preserve, abstain


def _aggregate_residue(rows: list[Mapping[str, object]]) -> float | None:
    total = sum(float(row["residue_score_sum"]) for row in rows)
    count = sum(int(row["residue_source_contrast_pixel_count"]) for row in rows)
    return total / count if count else None


def _aggregate_coverage(rows: list[Mapping[str, object]]) -> float:
    total = sum(int(row["target_pixel_count"]) for row in rows)
    covered = sum(int(row["target_edit_pixel_count"]) for row in rows)
    return float(covered) / total if total else 0.0


def _mask_set_sha(rows: list[Mapping[str, object]], field: str) -> str:
    return _canonical_sha256(
        sorted(
            (
                {"page_id": str(row["page_id"]), field: str(row[field])}
                for row in rows
            ),
            key=lambda row: row["page_id"],
        )
    )


def _validated_addition_paths(
    *,
    mask_only_result_path: Path,
    mask_only: Mapping[str, object],
    candidate_id: str,
    page_ids: set[str],
) -> dict[str, Path]:
    binding = mask_only.get("output_inventory")
    if not isinstance(binding, Mapping):
        raise ValueError("v3.3 mask-only result lacks output inventory")
    relative = str(binding.get("relative_path") or "")
    if not relative or Path(relative).is_absolute():
        raise ValueError("v3.3 mask-only inventory path must be relative")
    root = mask_only_result_path.resolve().parent
    inventory_path = (root / relative).resolve()
    try:
        inventory_path.relative_to(root)
    except ValueError as error:
        raise ValueError("v3.3 mask-only inventory escapes its run") from error
    if not inventory_path.is_file() or binding.get("artifact_sha256") != _sha256(
        inventory_path
    ):
        raise ValueError("v3.3 mask-only inventory file SHA differs")
    inventory = _read_json(inventory_path)
    if inventory.get("schema_version") != (
        "inpaint-proposal-refinement-output-inventory-v33"
    ):
        raise ValueError("v3.3 mask-only inventory schema differs")
    expected_inventory_sha = _canonical_sha256(
        {key: value for key, value in inventory.items() if key != "inventory_sha256"}
    )
    if (
        inventory.get("inventory_sha256") != expected_inventory_sha
        or binding.get("inventory_sha256") != expected_inventory_sha
    ):
        raise ValueError("v3.3 mask-only inventory SHA differs")
    records = inventory.get("artifacts")
    if not isinstance(records, list) or binding.get("artifact_count") != len(records):
        raise ValueError("v3.3 mask-only inventory record count differs")
    selected: dict[str, Path] = {}
    for value in records:
        if not isinstance(value, Mapping):
            raise ValueError("v3.3 mask-only inventory record is invalid")
        if value.get("candidate_id") != candidate_id:
            continue
        page_id = str(value.get("page_id") or "")
        if value.get("role") != "safe_addition" or page_id in selected:
            raise ValueError("v3.3 mask-only selected artifact identity differs")
        relative_mask = str(value.get("relative_path") or "")
        if not relative_mask or Path(relative_mask).is_absolute():
            raise ValueError("v3.3 mask-only artifact path must be relative")
        path = (root / relative_mask).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("v3.3 mask-only artifact escapes its run") from error
        if not path.is_file() or value.get("file_sha256") != _sha256(path):
            raise ValueError("v3.3 mask-only safe addition file SHA differs")
        decoded = _read_image(path, cv2.IMREAD_GRAYSCALE)
        unique = np.unique(decoded)
        if np.any((unique != 0) & (unique != 255)):
            raise ValueError("v3.3 mask-only safe addition is not binary")
        normalized = binary_mask(decoded)
        if (
            value.get("pixel_sha256") != mask_sha256(normalized)
            or value.get("pixel_count") != int(np.count_nonzero(normalized))
        ):
            raise ValueError("v3.3 mask-only safe addition pixels differ")
        selected[page_id] = path
    if set(selected) != page_ids:
        raise ValueError("v3.3 mask-only selected page inventory differs")
    stored_pages = mask_only.get("pages")
    candidate_pages = (
        stored_pages.get(candidate_id)
        if isinstance(stored_pages, Mapping)
        else None
    )
    if not isinstance(candidate_pages, list):
        raise ValueError("v3.3 mask-only selected page metrics are missing")
    by_page = {
        str(row.get("page_id") or ""): row
        for row in candidate_pages
        if isinstance(row, Mapping)
    }
    if set(by_page) != page_ids:
        raise ValueError("v3.3 mask-only selected metric pages differ")
    for page_id, path in selected.items():
        decoded = _read_image(path, cv2.IMREAD_GRAYSCALE)
        if by_page[page_id].get("output_safe_addition_pixel_sha256") != (
            mask_sha256(decoded)
        ):
            raise ValueError("v3.3 mask-only selected page SHA differs")
    return selected


def run_candidate(
    *,
    relative_manifest_path: Path,
    mask_only_result_path: Path,
    candidate_id: str,
    output_root: Path,
    device: str,
    precision: str,
    inpaint_size: int,
) -> dict[str, object]:
    manifest_binding = validate_source_only_manifest_v4(relative_manifest_path)
    mask_only = _read_json(mask_only_result_path)
    if mask_only.get("schema_version") != "inpaint-proposal-refinement-results-v33":
        raise ValueError("v3.3 stage2 mask-only schema differs")
    if mask_only.get("relative_manifest", {}).get("manifest_sha256") != (
        manifest_binding["manifest_sha256"]
    ):
        raise ValueError("v3.3 stage2 baseline manifest binding differs")
    shortlist = mask_only.get("shortlist")
    if not isinstance(shortlist, list) or candidate_id not in shortlist:
        raise ValueError("v3.3 stage2 candidate is not in the sealed shortlist")
    mask_candidate = next(
        (
            row
            for row in mask_only.get("candidates", [])
            if isinstance(row, Mapping) and row.get("candidate_id") == candidate_id
        ),
        None,
    )
    if not isinstance(mask_candidate, Mapping) or not bool(
        mask_candidate.get("incremental_safety_pass")
    ):
        raise ValueError("v3.3 stage2 candidate failed mask-only safety")

    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise FileExistsError("v3.3 stage2 output must be fresh")
    pages = load_stage1_manifest(relative_manifest_path)
    entries = _manifest_entries(relative_manifest_path)
    addition_paths = _validated_addition_paths(
        mask_only_result_path=mask_only_result_path,
        mask_only=mask_only,
        candidate_id=candidate_id,
        page_ids={page.page_id for page in pages},
    )

    lama_model_path = Path(
        ModelDownloader.primary_path(ModelID.LAMA_LARGE_512PX)
    ).resolve()
    lama_model_sha256_before = _sha256(lama_model_path)
    inpainter = SourceLaMaLarge(
        device=device,
        precision=precision,
        inpaint_size=inpaint_size,
    )
    inpainter.ensure_loaded()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    diagnostics_before = len(inpainter.run_diagnostics)
    started = time.perf_counter()
    baseline_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    artifacts: list[dict[str, object]] = []
    inference_count = 0
    for page in pages:
        entry = entries[page.page_id]
        source = _read_image(page.source_image, cv2.IMREAD_COLOR)
        shape = source.shape[:2]
        masks = load_page_masks(page, shape, strict_binary=True)
        baseline_path = _resolve_manifest_artifact(
            relative_manifest_path, entry.get("baseline")
        )
        baseline_mask_path = _resolve_manifest_artifact(
            relative_manifest_path, entry.get("baseline_mask")
        )
        if baseline_path is None or baseline_mask_path is None:
            raise ValueError(f"v3.3 stage2 baseline is missing: {page.page_id}")
        baseline = _read_image(baseline_path, cv2.IMREAD_COLOR)
        if baseline.shape != source.shape:
            raise ValueError("v3.3 stage2 baseline image shape differs")
        baseline_mask = _read_mask(baseline_mask_path, shape)
        addition = _read_mask(addition_paths[page.page_id], shape)
        if np.any((addition > 0) & (baseline_mask > 0)):
            raise ValueError("v3.3 stage2 addition overlaps PR6 mask")
        if np.any(addition):
            generated_rgb = inpainter.memory_safe_inpaint(
                cv2.cvtColor(source, cv2.COLOR_BGR2RGB),
                addition,
            )
            generated = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2BGR)
            inference_count += 1
        else:
            generated = source
        candidate, final_mask = composite_positive_result(
            baseline,
            generated,
            addition,
            baseline_mask,
        )
        incremental_changed = changed_mask(baseline, candidate)
        if np.any((incremental_changed > 0) & (addition == 0)):
            raise AssertionError("v3.3 candidate changed outside its addition")
        if not np.any(addition):
            if not np.array_equal(candidate, baseline) or not np.array_equal(
                final_mask, baseline_mask
            ):
                raise AssertionError("empty v3.3 addition changed PR6 bytes")

        baseline_score, baseline_sum, baseline_count = residue_score(
            source, baseline, masks.target
        )
        candidate_score, candidate_sum, candidate_count = residue_score(
            source, candidate, masks.target
        )
        if candidate_count != baseline_count:
            raise AssertionError("v3.3 residue denominator changed")
        optional = _optional_neutral_mask(page, shape)
        explicit_preserve, explicit_abstain = _runtime_action_masks(
            entry, masks, shape
        )
        baseline_instance_scores = _target_instance_scores(page, baseline_mask)
        candidate_instance_scores = _target_instance_scores(page, final_mask)
        target_pixels = int(np.count_nonzero(masks.target))
        baseline_covered = int(
            np.count_nonzero((masks.target > 0) & (baseline_mask > 0))
        )
        candidate_covered = int(
            np.count_nonzero((masks.target > 0) & (final_mask > 0))
        )
        baseline_changed = changed_mask(source, baseline)
        candidate_changed = changed_mask(source, candidate)
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
            "safe_addition_pixel_count": int(np.count_nonzero(addition)),
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
            "incremental_changed_pixel_count": int(
                np.count_nonzero(incremental_changed)
            ),
            "incremental_changed_outside_addition_pixel_count": int(
                np.count_nonzero((incremental_changed > 0) & (addition == 0))
            ),
            "incremental_protected_changed_pixel_count": int(
                np.count_nonzero((incremental_changed > 0) & (masks.protected > 0))
            ),
            "incremental_protected_edit_overlap_pixel_count": int(
                np.count_nonzero((addition > 0) & (masks.protected > 0))
            ),
            "incremental_ambiguous_changed_pixel_count": int(
                np.count_nonzero((incremental_changed > 0) & (masks.ambiguous > 0))
            ),
            "incremental_ambiguous_edit_overlap_pixel_count": int(
                np.count_nonzero((addition > 0) & (masks.ambiguous > 0))
            ),
            "incremental_corner_changed_pixel_count": int(
                np.count_nonzero((incremental_changed > 0) & (masks.corner > 0))
                if masks.corner is not None
                else 0
            ),
            "incremental_corner_edit_overlap_pixel_count": int(
                np.count_nonzero((addition > 0) & (masks.corner > 0))
                if masks.corner is not None
                else 0
            ),
            "explicit_preserve_addition_pixel_count": int(
                np.count_nonzero((addition > 0) & (explicit_preserve > 0))
            ),
            "explicit_abstain_addition_pixel_count": int(
                np.count_nonzero((addition > 0) & (explicit_abstain > 0))
            ),
            "no_edit_non_neutral_addition_pixel_count": (
                int(np.count_nonzero((addition > 0) & (optional == 0)))
                if page.expected_edit == "none"
                else 0
            ),
            "incremental_ownership_leak_pixel_count": int(
                np.count_nonzero((addition > 0) & (masks.ownership == 0))
            ),
            "optional_neutral_addition_pixel_count": int(
                np.count_nonzero((addition > 0) & (optional > 0))
            ),
            "inherited_pr6_protected_changed_pixel_count": int(
                np.count_nonzero((baseline_changed > 0) & (masks.protected > 0))
            ),
            "final_protected_changed_pixel_count": int(
                np.count_nonzero((candidate_changed > 0) & (masks.protected > 0))
            ),
            "positive_lama_inference_call_count": int(np.any(addition)),
            "addition_pixel_sha256": mask_sha256(addition),
            "output_mask_pixel_sha256": mask_sha256(final_mask),
            "candidate_pixel_sha256": hashlib.sha256(
                np.ascontiguousarray(candidate).tobytes()
            ).hexdigest(),
        }
        baseline_rows.append(base_row)
        candidate_rows.append(candidate_row)
        for role, value in (
            ("safe_addition", addition),
            ("candidate_image", candidate),
            ("final_mask", final_mask),
            ("incremental_changed_mask", incremental_changed),
        ):
            path = output_root / role / f"{page.page_id}.png"
            record = _write_png(path, value)
            artifacts.append(
                {
                    "page_id": page.page_id,
                    "role": role,
                    "relative_path": path.relative_to(output_root).as_posix(),
                    **record,
                }
            )

    diagnostics = inpainter.run_diagnostics[diagnostics_before:]
    lama_model_sha256_after = _sha256(lama_model_path)
    if lama_model_sha256_after != lama_model_sha256_before:
        raise RuntimeError("v3.3 LaMa model changed during evaluation")
    cpu_fallback_count = sum(
        int(bool(row.get("cpu_fallback_used", False))) for row in diagnostics
    )
    runtime_complete = len(diagnostics) == inference_count
    incremental_zero_fields = {
        "protected_structure_overlap": sum(
            int(row["incremental_protected_edit_overlap_pixel_count"])
            for row in candidate_rows
        ),
        "protected_structure_changed": sum(
            int(row["incremental_protected_changed_pixel_count"])
            for row in candidate_rows
        ),
        "ambiguous_structure_overlap": sum(
            int(row["incremental_ambiguous_edit_overlap_pixel_count"])
            for row in candidate_rows
        ),
        "ambiguous_structure_changed": sum(
            int(row["incremental_ambiguous_changed_pixel_count"])
            for row in candidate_rows
        ),
        "preserve_edit_overlap": sum(
            int(row["explicit_preserve_addition_pixel_count"])
            for row in candidate_rows
        ),
        "ownership_leak_pixel_count": sum(
            int(row["incremental_ownership_leak_pixel_count"])
            for row in candidate_rows
        ),
        "corner_edit_overlap_pixel_count": sum(
            int(row["incremental_corner_edit_overlap_pixel_count"])
            for row in candidate_rows
        ),
        "outside_final_changed": sum(
            int(row["incremental_changed_outside_addition_pixel_count"])
            for row in candidate_rows
        ),
        "broad_route_false_positive": 0,
        "no_edit_false_edit": sum(
            int(row["no_edit_non_neutral_addition_pixel_count"])
            for row in candidate_rows
        ),
        "required_skip_count": 0,
        "cpu_fallback_count": cpu_fallback_count,
    }
    baseline_metrics = {
        "aggregate_target_coverage": _aggregate_coverage(baseline_rows),
        "aggregate_residue_score": _aggregate_residue(baseline_rows),
        "output_mask_set_sha256": _mask_set_sha(
            baseline_rows, "output_mask_pixel_sha256"
        ),
    }
    candidate_metrics = {
        **incremental_zero_fields,
        "aggregate_target_coverage": _aggregate_coverage(candidate_rows),
        "aggregate_residue_score": _aggregate_residue(candidate_rows),
        "output_mask_set_sha256": _mask_set_sha(
            candidate_rows, "output_mask_pixel_sha256"
        ),
        "runtime_telemetry_complete": runtime_complete,
        "maximum_positive_lama_inference_per_page": max(
            (int(row["positive_lama_inference_call_count"]) for row in candidate_rows),
            default=0,
        ),
        "positive_lama_inference_count": inference_count,
        "lama_runtime_provider": device,
        "lama_runtime_precision": precision,
        "target_instance_seed_recall": mask_candidate.get(
            "target_instance_seed_recall"
        ),
        "missed_target_instance_count": mask_candidate.get(
            "missed_target_instance_count"
        ),
    }
    gate = evaluate_relative_product_gate(
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        baseline_pages=baseline_rows,
        candidate_pages=candidate_rows,
        candidate_kind="balanced",
    )
    abstain_overlap = sum(
        int(row["explicit_abstain_addition_pixel_count"])
        for row in candidate_rows
    )
    if abstain_overlap:
        gate["relative_product_pass"] = False
        gate["gate_failures"] = sorted(
            set([*gate["gate_failures"], "safety_nonzero:abstain_edit_overlap"])
        )
    artifacts.sort(key=lambda row: (str(row["role"]), str(row["page_id"])))
    inventory = {
        "schema_version": "inpaint-proposal-stage2-output-inventory-v33",
        "candidate_id": candidate_id,
        "page_ids": sorted(row["page_id"] for row in candidate_rows),
        "records": artifacts,
    }
    inventory["inventory_sha256"] = _canonical_sha256(inventory)
    inventory_path = output_root / "output-artifact-inventory.json"
    _write_json(inventory_path, inventory)
    result = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "tracked_worktree_clean": not bool(
            subprocess.check_output(
                ["git", "status", "--short", "--untracked-files=no"],
                cwd=ROOT,
                text=True,
            ).strip()
        ),
        "relative_manifest_sha256": manifest_binding["manifest_sha256"],
        "mask_only_result_sha256": _sha256(mask_only_result_path),
        "mask_only_output_mask_set_sha256": mask_candidate.get(
            "output_mask_set_sha256"
        ),
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "relative_gate": gate,
        "inherited_pr6": {
            "protected_changed_pixel_count": sum(
                int(row["inherited_pr6_protected_changed_pixel_count"])
                for row in candidate_rows
            ),
            "note": "reported separately; v3.3 does not rewrite PR6 pixels",
        },
        "incremental": {
            **incremental_zero_fields,
            "abstain_edit_overlap": abstain_overlap,
            "optional_neutral_addition_pixel_count": sum(
                int(row["optional_neutral_addition_pixel_count"])
                for row in candidate_rows
            ),
        },
        "runtime": {
            "device": device,
            "precision": precision,
            "inpaint_size": inpaint_size,
            "page_count": len(candidate_rows),
            "elapsed_seconds": time.perf_counter() - started,
            "positive_lama_inference_count": inference_count,
            "cpu_fallback_count": cpu_fallback_count,
            "peak_vram_allocated_mib": (
                float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
                if device == "cuda"
                else None
            ),
            "peak_vram_reserved_mib": (
                float(torch.cuda.max_memory_reserved()) / (1024.0 * 1024.0)
                if device == "cuda"
                else None
            ),
            "lama_model_path": str(lama_model_path),
            "lama_model_sha256_before": lama_model_sha256_before,
            "lama_model_sha256_after": lama_model_sha256_after,
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
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one incremental PR6 detector-proposal candidate."
    )
    parser.add_argument("--relative-manifest", type=Path, required=True)
    parser.add_argument("--mask-only-result", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
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
            precision=args.precision,
            inpaint_size=args.inpaint_size,
        )
        result_path = output_root / "stage2-results.json"
        _write_json(result_path, result)
        if managed is not None:
            managed.complete(
                metadata={
                    "candidate_id": args.candidate_id,
                    "relative_product_pass": result["relative_gate"][
                        "relative_product_pass"
                    ],
                    "relative_manifest_sha256": result[
                        "relative_manifest_sha256"
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
