#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from itertools import combinations
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.contracts import binary_mask  # noqa: E402
from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    detector_roi_trigger_mask,
    fuse_detector_claims,
    load_page_masks,
    load_stage1_manifest,
    positive_edit_from_claim,
)
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-detector-fusions-v4"
CATEGORY = "40-inpaint-mask-render"
SCHEMA_VERSION = "inpaint-detector-fusion-spec-v4"


@dataclass(slots=True)
class _Totals:
    target_pixels: int = 0
    target_edit_pixels: int = 0
    instance_count: int = 0
    seeded_instances: int = 0
    minimum_instance_coverage: float | None = None
    protected_overlap: int = 0
    ambiguous_overlap: int = 0
    preserve_overlap: int = 0
    ownership_leak: int = 0
    no_edit_false_edit: int = 0
    mask_digest: Any = field(default_factory=hashlib.sha256)

    def update(
        self,
        *,
        page,
        claim: np.ndarray,
        edit: np.ndarray,
        masks,
        instance_indices: tuple[np.ndarray, ...],
    ) -> None:
        flat_claim = claim.reshape(-1)
        flat_edit = edit.reshape(-1)
        self.target_pixels += int(np.count_nonzero(masks.target))
        self.target_edit_pixels += int(
            np.count_nonzero((masks.target > 0) & (edit > 0))
        )
        for indices in instance_indices:
            pixels = int(indices.size)
            seed = int(np.count_nonzero(flat_claim[indices]))
            covered = int(np.count_nonzero(flat_edit[indices]))
            coverage = float(covered) / float(pixels) if pixels else 0.0
            self.instance_count += 1
            self.seeded_instances += int(seed > 0)
            self.minimum_instance_coverage = (
                coverage
                if self.minimum_instance_coverage is None
                else min(self.minimum_instance_coverage, coverage)
            )
        self.protected_overlap += int(
            np.count_nonzero((edit > 0) & (masks.protected > 0))
        )
        self.ambiguous_overlap += int(
            np.count_nonzero((edit > 0) & (masks.ambiguous > 0))
        )
        if masks.preserve is not None:
            self.preserve_overlap += int(
                np.count_nonzero((edit > 0) & (masks.preserve > 0))
            )
        self.ownership_leak += int(
            np.count_nonzero((edit > 0) & (masks.ownership == 0))
        )
        if page.no_edit:
            self.no_edit_false_edit += int(np.count_nonzero(edit))
        self.mask_digest.update(page.page_id.encode("utf-8"))
        self.mask_digest.update(b"\0")
        self.mask_digest.update(edit.tobytes(order="C"))

    def summary(self) -> dict[str, object]:
        seed_recall = (
            float(self.seeded_instances) / float(self.instance_count)
            if self.instance_count
            else None
        )
        coverage = (
            float(self.target_edit_pixels) / float(self.target_pixels)
            if self.target_pixels
            else None
        )
        return {
            "target_instance_count": self.instance_count,
            "seeded_target_instance_count": self.seeded_instances,
            "missed_target_instance_count": self.instance_count - self.seeded_instances,
            "target_instance_seed_recall": seed_recall,
            "aggregate_target_coverage": coverage,
            "minimum_target_instance_coverage": self.minimum_instance_coverage,
            "protected_edit_overlap": self.protected_overlap,
            "ambiguous_edit_overlap": self.ambiguous_overlap,
            "preserve_edit_overlap": self.preserve_overlap,
            "ownership_leak_pixel_count": self.ownership_leak,
            "false_edit_pixel_count": self.no_edit_false_edit,
            "output_mask_set_sha256": self.mask_digest.hexdigest(),
        }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_edit_paths(manifest_path: Path) -> dict[str, str]:
    payload = _read_json(manifest_path)
    paths: dict[str, str] = {}
    for page in payload.get("pages", []):
        if not isinstance(page, dict):
            continue
        value = page.get("existing_source_edit_mask", page.get("baseline_mask"))
        if isinstance(value, dict):
            value = value.get("path")
        if isinstance(value, str) and value.strip():
            paths[str(page.get("page_id") or "")] = value.strip()
    return paths


def _template_path(template: str, page_id: str) -> Path:
    return Path(template.format(page_id=page_id)).resolve()


def _read_mask(template: str, page_id: str, shape: tuple[int, int]) -> np.ndarray:
    path = _template_path(template, page_id)
    value = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if value is None or value.size == 0:
        raise FileNotFoundError(path)
    return binary_mask(value, shape)


def _logical_runs(
    candidate_ids: tuple[str, ...],
    roi_candidate_ids: frozenset[str],
) -> list[dict[str, str]]:
    runs = [
        {"run_id": candidate, "fusion": "single", "primary": candidate, "secondary": ""}
        for candidate in candidate_ids
    ]
    for left, right in combinations(candidate_ids, 2):
        for mode in ("or", "and"):
            runs.append(
                {
                    "run_id": f"{left}__{mode}__{right}",
                    "fusion": mode,
                    "primary": left,
                    "secondary": right,
                }
            )
    for secondary in sorted(roi_candidate_ids):
        for primary in candidate_ids:
            if primary == secondary:
                continue
            for trigger in (
                "seed_missing",
                "raw_refined_disagreement",
                "source_seed_unavailable",
                "union",
            ):
                runs.append(
                    {
                        "run_id": f"{primary}__gated_{trigger}__{secondary}",
                        "fusion": "gated_recovery",
                        "primary": primary,
                        "secondary": secondary,
                        "trigger": trigger,
                    }
                )
    return runs


def _hard_gate_passes(summary: dict[str, object]) -> bool:
    zero = (
        "missed_target_instance_count",
        "protected_edit_overlap",
        "ambiguous_edit_overlap",
        "preserve_edit_overlap",
        "ownership_leak_pixel_count",
        "false_edit_pixel_count",
    )
    return (
        summary.get("target_extent_independent") is True
        and summary.get("target_inventory_independent") is True
        and summary.get("target_review_complete") is True
        and
        all(int(summary[name]) == 0 for name in zero)
        and float(summary.get("aggregate_target_coverage") or 0.0) >= 0.98
        and float(summary.get("minimum_target_instance_coverage") or 0.0) >= 0.98
    )


def run_fusion_matrix(
    manifest_path: Path,
    spec_path: Path,
) -> dict[str, object]:
    spec = _read_json(spec_path)
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported detector fusion spec")
    raw_candidates = spec.get("candidates")
    if not isinstance(raw_candidates, dict) or not raw_candidates:
        raise ValueError("fusion spec requires candidates")
    candidates = {
        str(candidate_id): dict(value)
        for candidate_id, value in raw_candidates.items()
        if isinstance(value, dict)
    }
    if len(candidates) != len(raw_candidates):
        raise ValueError("candidate specs must be objects")
    roi_candidates = frozenset(
        candidate_id
        for candidate_id, value in candidates.items()
        if bool(value.get("roi_detector", False))
    )
    runs = _logical_runs(tuple(candidates), roi_candidates)
    totals = {row["run_id"]: _Totals() for row in runs}
    pages = load_stage1_manifest(manifest_path)
    target_extent_independent = all(
        page.target_extent_independent for page in pages
    )
    target_mask_provenance = sorted(
        {page.target_mask_provenance for page in pages}
    )
    target_inventory_independent = all(
        page.target_inventory_independent for page in pages
    )
    target_review_complete = all(page.target_review_complete for page in pages)
    existing_edit_paths = _existing_edit_paths(manifest_path)
    for page in pages:
        source = cv2.imdecode(
            np.fromfile(page.source_image, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if source is None or source.size == 0:
            raise FileNotFoundError(page.source_image)
        shape = source.shape[:2]
        masks = load_page_masks(
            page,
            shape,
            existing_edit_path=existing_edit_paths.get(page.page_id),
        )
        source_seed = binary_mask(masks.claim_seed, shape)
        loaded: dict[str, dict[str, np.ndarray]] = {}
        for candidate_id, value in candidates.items():
            templates = value.get("templates")
            if not isinstance(templates, dict) or not templates.get("raw"):
                raise ValueError(f"candidate {candidate_id} lacks raw template")
            raw = _read_mask(str(templates["raw"]), page.page_id, shape)
            refined = _read_mask(
                str(templates.get("refined") or templates["raw"]),
                page.page_id,
                shape,
            )
            loaded[candidate_id] = {"raw": raw, "refined": refined}
        instance_indices = tuple(
            np.flatnonzero(
                _read_mask(record.mask_path, page.page_id, shape).reshape(-1)
            )
            for record in page.target_instances
            if record.priority == "required"
        )
        trigger_cache: dict[tuple[str, str], np.ndarray] = {}
        for run in runs:
            primary = loaded[run["primary"]]
            if run["fusion"] == "single":
                claim = fuse_detector_claims(
                    "single",
                    primary["raw"],
                    primary["raw"],
                    ownership=masks.ownership,
                )
            else:
                secondary = loaded[run["secondary"]]
                trigger_mask = None
                if run["fusion"] == "gated_recovery":
                    key = (run["primary"], run["trigger"])
                    trigger_mask = trigger_cache.get(key)
                    if trigger_mask is None:
                        trigger_mask = detector_roi_trigger_mask(
                            run["trigger"],
                            ownership=masks.ownership,
                            primary_raw=primary["raw"],
                            primary_refined=primary["refined"],
                            source_seed=source_seed,
                        )
                        trigger_cache[key] = trigger_mask
                claim = fuse_detector_claims(
                    run["fusion"],
                    primary["raw"],
                    secondary["raw"],
                    ownership=masks.ownership,
                    trigger_mask=trigger_mask,
                )
            edit = positive_edit_from_claim(claim, masks)
            totals[run["run_id"]].update(
                page=page,
                claim=claim,
                edit=edit,
                masks=masks,
                instance_indices=instance_indices,
            )
    output_runs: list[dict[str, object]] = []
    content_owner: dict[str, str] = {}
    closure: list[dict[str, object]] = []
    for run in runs:
        summary = totals[run["run_id"]].summary()
        summary["target_extent_independent"] = target_extent_independent
        summary["target_inventory_independent"] = target_inventory_independent
        summary["target_review_complete"] = target_review_complete
        summary["target_mask_provenance"] = target_mask_provenance
        content_sha = str(summary["output_mask_set_sha256"])
        reused_from = content_owner.get(content_sha, "")
        state = "reused_by_sha" if reused_from else "executed"
        if not reused_from:
            content_owner[content_sha] = run["run_id"]
        closure.append(
            {
                "logical_id": run["run_id"],
                "selection": dict(run),
                "closure_state": state,
                "reason": "",
                "content_sha256": content_sha,
                "reused_from": reused_from,
            }
        )
        output_runs.append(
            {
                **run,
                "status": (
                    "family_complete"
                    if _hard_gate_passes(summary)
                    else (
                        "information_limited"
                        if not target_extent_independent
                        or not target_inventory_independent
                        or not target_review_complete
                        else "dominated"
                    )
                ),
                "closure_reason": (
                    ""
                    if _hard_gate_passes(summary)
                    else (
                        (
                            "target_extent_not_independent"
                            if not target_extent_independent
                            else (
                                "target_inventory_not_independent"
                                if not target_inventory_independent
                                else "target_review_incomplete"
                            )
                        )
                        if not target_extent_independent
                        or not target_inventory_independent
                        or not target_review_complete
                        else "hard_gate_failed"
                    )
                ),
                "metrics": summary,
            }
        )
    return {
        "schema_version": "inpaint-detector-fusion-results-v4",
        "manifest_sha256": _sha256(manifest_path),
        "spec_sha256": _sha256(spec_path),
        "candidate_count": len(candidates),
        "logical_combination_count": len(runs),
        "physical_output_count": len(content_owner),
        "unaccounted_combination_count": 0,
        "closure_ledger": closure,
        "runs": output_runs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score all source-only single/two-detector fusion roles on E1."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        payload = run_fusion_matrix(args.manifest.resolve(), args.spec.resolve())
        _write_json(output_root / "fusion-results.json", payload)
        if managed is not None:
            managed.complete(
                metadata={
                    "manifest_sha256": payload["manifest_sha256"],
                    "spec_sha256": payload["spec_sha256"],
                    "logical_combination_count": payload["logical_combination_count"],
                    "physical_output_count": payload["physical_output_count"],
                }
            )
            mismatches = managed.verify()
            if mismatches:
                raise RuntimeError("managed artifact verification failed: " + "; ".join(mismatches))
            print(managed.run_root)
        else:
            print(output_root)
        return 0
    except BaseException as error:
        if managed is not None:
            managed.fail(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
