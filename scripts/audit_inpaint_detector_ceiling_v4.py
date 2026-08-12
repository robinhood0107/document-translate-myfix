#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.contracts import binary_mask  # noqa: E402
from benchmarking.inpaint_detector_bakeoff.stage1 import load_stage1_manifest  # noqa: E402
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-detector-ceiling-v4"
CATEGORY = "40-inpaint-mask-render"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    value = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if value is None or value.size == 0:
        raise FileNotFoundError(path)
    return binary_mask(value, shape)


def audit_detector_ceiling(manifest_path: Path, spec_path: Path) -> dict[str, object]:
    spec = _read_json(spec_path)
    raw_candidates = spec.get("candidates")
    if not isinstance(raw_candidates, dict) or not raw_candidates:
        raise ValueError("detector ceiling spec requires candidates")
    candidates = {
        str(candidate_id): value
        for candidate_id, value in raw_candidates.items()
        if isinstance(value, dict)
    }
    if len(candidates) != len(raw_candidates):
        raise ValueError("detector ceiling candidates must be objects")
    missing: list[dict[str, object]] = []
    total = seeded = 0
    per_candidate = {
        candidate_id: {"required_instance_count": 0, "seeded_instance_count": 0}
        for candidate_id in candidates
    }
    page_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    for page in load_stage1_manifest(manifest_path):
        source = cv2.imdecode(
            np.fromfile(page.source_image, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        if source is None or source.size == 0:
            raise FileNotFoundError(page.source_image)
        shape = source.shape
        loaded: dict[str, np.ndarray] = {}
        union = np.zeros(shape, np.uint8)
        for candidate_id, value in candidates.items():
            templates = value.get("templates")
            if not isinstance(templates, dict) or not templates.get("raw"):
                raise ValueError(f"candidate {candidate_id} lacks raw template")
            path = Path(str(templates["raw"]).format(page_id=page.page_id)).resolve()
            loaded[candidate_id] = _read_mask(path, shape)
            union[loaded[candidate_id] > 0] = 255
        for record in page.target_instances:
            if record.priority != "required":
                continue
            target = _read_mask(Path(record.mask_path), shape)
            target_pixels = int(np.count_nonzero(target))
            total += 1
            union_pixels = int(np.count_nonzero((target > 0) & (union > 0)))
            seeded += int(union_pixels > 0)
            provider_hits: list[str] = []
            for candidate_id, claim in loaded.items():
                hit = int(np.count_nonzero((target > 0) & (claim > 0)))
                per_candidate[candidate_id]["required_instance_count"] += 1
                per_candidate[candidate_id]["seeded_instance_count"] += int(hit > 0)
                if hit > 0:
                    provider_hits.append(candidate_id)
            if union_pixels > 0:
                continue
            page_counts[page.page_id] = page_counts.get(page.page_id, 0) + 1
            role_counts[record.semantic_role] = role_counts.get(record.semantic_role, 0) + 1
            missing.append(
                {
                    "page_id": page.page_id,
                    "instance_id": record.instance_id,
                    "region_id": record.region_id,
                    "semantic_role": record.semantic_role,
                    "target_pixel_count": target_pixels,
                    "provider_hits": provider_hits,
                }
            )
    candidates_summary = []
    for candidate_id, counts in per_candidate.items():
        count = int(counts["required_instance_count"])
        hits = int(counts["seeded_instance_count"])
        candidates_summary.append(
            {
                "candidate_id": candidate_id,
                **counts,
                "seed_recall": float(hits) / float(count) if count else None,
            }
        )
    return {
        "schema_version": "inpaint-detector-ceiling-results-v4",
        "manifest_sha256": _sha256(manifest_path),
        "spec_sha256": _sha256(spec_path),
        "candidate_count": len(candidates),
        "required_instance_count": total,
        "all_candidate_union_seeded_instance_count": seeded,
        "all_candidate_union_seed_recall": float(seeded) / float(total) if total else None,
        "all_candidate_union_missed_instance_count": total - seeded,
        "missing_by_page": dict(sorted(page_counts.items())),
        "missing_by_semantic_role": dict(sorted(role_counts.items())),
        "missing_instances": missing,
        "candidate_summaries": candidates_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the source-only seed-recall ceiling of all registered detectors."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output.mkdir(parents=True, exist_ok=True)
    try:
        payload = audit_detector_ceiling(args.manifest.resolve(), args.spec.resolve())
        (output / "detector-ceiling-results.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if managed is not None:
            managed.complete(
                metadata={
                    "manifest_sha256": payload["manifest_sha256"],
                    "spec_sha256": payload["spec_sha256"],
                    "candidate_count": payload["candidate_count"],
                    "missed_instance_count": payload[
                        "all_candidate_union_missed_instance_count"
                    ],
                }
            )
            mismatches = managed.verify()
            if mismatches:
                raise RuntimeError("managed artifact verification failed: " + "; ".join(mismatches))
            print(managed.run_root)
        else:
            print(output)
        return 0
    except BaseException as error:
        if managed is not None:
            managed.fail(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
