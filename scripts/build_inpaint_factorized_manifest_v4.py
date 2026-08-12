#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    load_page_masks,
    load_stage1_manifest,
)


SCHEMA_VERSION = "inpaint-detector-bakeoff-manifest-v4"
DECISIONS_SCHEMA_VERSION = "inpaint-factorized-source-decisions-v4"
SEAL_SCHEMA_VERSION = "inpaint-factorized-manifest-seal-v4"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_value(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        nested = value.get("path")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _indexed_pages(payload: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise ValueError(f"{label} must contain a pages array")
    indexed: dict[str, dict[str, Any]] = {}
    for value in pages:
        if not isinstance(value, dict):
            raise ValueError(f"{label} page must be an object")
        page_id = str(value.get("page_id") or "").strip()
        if not page_id or page_id in indexed:
            raise ValueError(f"{label} contains an empty or duplicate page id")
        indexed[page_id] = value
    return indexed


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _artifact_hashes(page: dict[str, Any]) -> dict[str, object]:
    hashes: dict[str, object] = {}
    scalar = (
        "path",
        "target_text_mask",
        "preserve_mask",
        "protected_structure_mask",
        "ambiguous_structure_mask",
        "ownership_mask",
        "corner_protect_mask",
        "claim_seed_mask",
        "existing_source_edit_mask",
        "baseline",
        "baseline_mask",
        "known_background",
    )
    for field in scalar:
        value = _path_value(page.get(field))
        if value is not None:
            artifact = Path(value)
            if not artifact.is_file():
                raise FileNotFoundError(value)
            hashes[field] = _sha256(artifact)
    instance_hashes: dict[str, str] = {}
    for instance in page.get("target_instances", []):
        if not isinstance(instance, dict):
            raise ValueError("target instance must be an object")
        instance_id = str(instance.get("instance_id") or "").strip()
        value = _path_value(instance.get("mask_path", instance.get("mask")))
        if not instance_id or value is None:
            raise ValueError("target instance is missing identity or mask")
        instance_hashes[instance_id] = _sha256(Path(value))
    hashes["target_instances"] = instance_hashes
    region_hashes: dict[str, dict[str, str]] = {}
    for region in page.get("regions", []):
        if not isinstance(region, dict):
            raise ValueError("region must be an object")
        region_id = str(region.get("region_id") or "").strip()
        if not region_id:
            raise ValueError("region id must not be empty")
        current: dict[str, str] = {}
        for field in (
            "bubble_interior_mask",
            "ownership_mask",
            "protected_structure_mask",
            "ambiguous_structure_mask",
            "corner_protect_mask",
        ):
            value = _path_value(region.get(field))
            if value is None:
                raise ValueError(f"region {region_id} lacks {field}")
            current[field] = _sha256(Path(value))
        region_hashes[region_id] = current
    hashes["regions"] = region_hashes
    reference = page.get("paired_reference")
    if reference is not None:
        if not isinstance(reference, dict) or reference.get("proposal_only") is not True:
            raise ValueError("paired reference must be proposal_only")
        value = _path_value(reference.get("path"))
        if value is None:
            raise ValueError("paired reference path is missing")
        actual = _sha256(Path(value))
        if actual != str(reference.get("reference_sha256") or "").lower():
            raise ValueError("paired reference SHA mismatch")
        hashes["paired_reference"] = actual
    return hashes


def build_manifest(
    source_manifest: Path,
    decisions_path: Path,
    baseline_manifest: Path | None = None,
) -> dict[str, Any]:
    source_payload = _read_json(source_manifest)
    decisions_payload = _read_json(decisions_path)
    if decisions_payload.get("schema_version") != DECISIONS_SCHEMA_VERSION:
        raise ValueError("unsupported source-only v4 decisions schema")
    if decisions_payload.get("candidate_seen") is not False:
        raise ValueError("v4 decisions must be frozen before viewing candidates")
    sources = _indexed_pages(source_payload, "source manifest")
    decisions = _indexed_pages(decisions_payload, "source decisions")
    if set(sources) != set(decisions):
        raise ValueError("source and decision page ids differ")
    baselines = (
        _indexed_pages(_read_json(baseline_manifest), "baseline manifest")
        if baseline_manifest is not None
        else None
    )
    if baselines is not None and set(baselines) != set(sources):
        raise ValueError("source and baseline page ids differ")
    pages: list[dict[str, Any]] = []
    for page_id, source in sources.items():
        decision = decisions[page_id]
        page = dict(source)
        for field in (
            "target_text_mask",
            "preserve_mask",
            "target_instances",
            "regions",
            "protected_structure_mask",
            "ambiguous_structure_mask",
            "ownership_mask",
            "claim_seed_mask",
            "bubble_interior_mask",
            "corner_protect_mask",
            "paired_reference",
            "expected_edit",
        ):
            if field in decision:
                page[field] = decision[field]
        page.update(
            {
                "page_id": page_id,
                "annotation_basis": "source_only_v4",
            }
        )
        if baselines is not None:
            baseline = baselines[page_id]
            source_path = _path_value(source.get("path"))
            baseline_source_path = _path_value(baseline.get("path"))
            if source_path is None or baseline_source_path is None:
                raise ValueError(f"baseline source path is missing: {page_id}")
            if _sha256(Path(source_path)) != _sha256(Path(baseline_source_path)):
                raise ValueError(f"baseline source mismatch: {page_id}")
            baseline_image = _path_value(baseline.get("baseline"))
            baseline_mask = _path_value(baseline.get("baseline_mask"))
            if baseline_image is None or baseline_mask is None:
                raise ValueError(f"baseline artifacts are missing: {page_id}")
            page["baseline"] = baseline_image
            page["baseline_mask"] = baseline_mask
            page["existing_source_edit_mask"] = baseline_mask
        paired = page.get("paired_reference")
        if isinstance(paired, dict):
            source_path = _path_value(page.get("path"))
            if source_path is None:
                raise ValueError(f"paired source path is missing: {page_id}")
            if _sha256(Path(source_path)) != str(paired.get("source_sha256") or "").lower():
                raise ValueError(f"paired source SHA mismatch: {page_id}")
        page["artifact_sha256"] = _artifact_hashes(page)
        pages.append(page)
    return {
        "schema_version": SCHEMA_VERSION,
        "corpus_id": decisions_payload.get("corpus_id", source_payload.get("corpus_id")),
        "split_role": "development_source_only",
        "source_manifest_sha256": _sha256(source_manifest),
        "source_decisions_sha256": _sha256(decisions_path),
        "baseline_manifest_sha256": _sha256(baseline_manifest) if baseline_manifest else None,
        "annotation_frozen_before_candidate": True,
        "pages": pages,
    }


def validate_manifest(path: Path) -> None:
    for page in load_stage1_manifest(path):
        image = cv2.imread(page.source_image, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise FileNotFoundError(page.source_image)
        load_page_masks(page, image.shape[:2])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and seal source-only manifest v4.")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path)
    args = parser.parse_args(argv)
    source = args.source_manifest.resolve()
    decisions = args.decisions.resolve()
    baseline = args.baseline_manifest.resolve() if args.baseline_manifest else None
    output = args.output.resolve()
    payload = build_manifest(source, decisions, baseline)
    _write_json(output, payload)
    validate_manifest(output)
    _write_json(
        output.with_suffix(output.suffix + ".seal.json"),
        {
            "schema_version": SEAL_SCHEMA_VERSION,
            "manifest": output.name,
            "manifest_sha256": _sha256(output),
            "source_manifest_sha256": _sha256(source),
            "source_decisions_sha256": _sha256(decisions),
            "baseline_manifest_sha256": _sha256(baseline) if baseline else None,
            "candidate_generated": False,
        },
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
