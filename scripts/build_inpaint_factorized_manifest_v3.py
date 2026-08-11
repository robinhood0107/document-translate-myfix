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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _indexed_pages(payload: dict[str, object], label: str) -> dict[str, dict[str, object]]:
    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise ValueError(f"{label} must contain a pages array")
    indexed: dict[str, dict[str, object]] = {}
    for entry in pages:
        if not isinstance(entry, dict):
            raise ValueError(f"{label} page must be an object")
        page_id = str(entry.get("page_id") or "").strip()
        if not page_id or page_id in indexed:
            raise ValueError(f"{label} contains an empty or duplicate page id")
        indexed[page_id] = entry
    return indexed


def _path_value(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        nested = value.get("path")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return None


def attach_artifact_hashes(payload: dict[str, object]) -> None:
    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise ValueError("manifest must contain pages before artifact sealing")
    scalar_fields = (
        "path",
        "target_text_mask",
        "bubble_interior_mask",
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
    for page in pages:
        assert isinstance(page, dict)
        hashes: dict[str, object] = {}
        for field in scalar_fields:
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
            artifact = Path(value)
            if not artifact.is_file():
                raise FileNotFoundError(value)
            instance_hashes[instance_id] = _sha256(artifact)
        hashes["target_instances"] = instance_hashes
        page["artifact_sha256"] = hashes


def build_manifest(
    source_manifest: Path,
    decisions_path: Path,
    baseline_manifest: Path | None = None,
) -> dict[str, object]:
    source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    decisions_payload = json.loads(decisions_path.read_text(encoding="utf-8"))
    if decisions_payload.get("schema_version") != "inpaint-factorized-source-decisions-v3":
        raise ValueError("unsupported source-only decisions schema")
    source_pages = _indexed_pages(source_payload, "source manifest")
    decisions = _indexed_pages(decisions_payload, "source-only decisions")
    baseline_pages: dict[str, dict[str, object]] | None = None
    if baseline_manifest is not None:
        baseline_payload = json.loads(baseline_manifest.read_text(encoding="utf-8"))
        baseline_pages = _indexed_pages(baseline_payload, "baseline manifest")
    if set(source_pages) != set(decisions):
        missing = sorted(set(source_pages).difference(decisions))
        extra = sorted(set(decisions).difference(source_pages))
        raise ValueError(f"source-only decisions page mismatch: missing={missing}, extra={extra}")
    if baseline_pages is not None and set(source_pages) != set(baseline_pages):
        missing = sorted(set(source_pages).difference(baseline_pages))
        extra = sorted(set(baseline_pages).difference(source_pages))
        raise ValueError(f"baseline page mismatch: missing={missing}, extra={extra}")

    pages: list[dict[str, object]] = []
    for page_id, source in source_pages.items():
        decision = decisions[page_id]
        expected_edit = str(
            decision.get("expected_edit", source.get("expected_edit", "none"))
        ).strip().lower()
        page = dict(source)
        page.update(
            {
                "page_id": page_id,
                "target_text_mask": decision.get("target_text_mask"),
                "target_instances": decision.get("target_instances", []),
                "bubble_route_class": decision.get("bubble_route_class"),
                "bubble_interior_mask": decision.get("bubble_interior_mask"),
                "protected_structure_mask": decision.get("protected_structure_mask"),
                "ambiguous_structure_mask": decision.get("ambiguous_structure_mask"),
                "ownership_mask": decision.get("ownership_mask"),
                "corner_protect_mask": decision.get("corner_protect_mask"),
                "expected_edit": expected_edit,
                "annotation_basis": "source_only_v3",
            }
        )
        page.pop("target_glyph_mask", None)
        if baseline_pages is not None:
            baseline = baseline_pages[page_id]
            source_path = _path_value(source.get("path"))
            baseline_source = _path_value(baseline.get("path"))
            if source_path is None or baseline_source is None:
                raise ValueError(f"baseline source path is missing: {page_id}")
            if _sha256(Path(source_path)) != _sha256(Path(baseline_source)):
                raise ValueError(f"baseline source does not match source-only page: {page_id}")
            baseline_image = _path_value(baseline.get("baseline"))
            baseline_mask = _path_value(baseline.get("baseline_mask"))
            if baseline_image is None or baseline_mask is None:
                raise ValueError(f"baseline artifacts are missing: {page_id}")
            page["baseline"] = baseline_image
            page["baseline_mask"] = baseline_mask
        pages.append(page)
    return {
        "schema_version": "inpaint-detector-bakeoff-manifest-v3",
        "corpus_id": decisions_payload.get("corpus_id", source_payload.get("corpus_id")),
        "split_role": "development_source_only",
        "source_manifest_sha256": _sha256(source_manifest),
        "source_decisions_sha256": _sha256(decisions_path),
        "baseline_manifest_sha256": (
            _sha256(baseline_manifest) if baseline_manifest is not None else None
        ),
        "annotation_frozen_before_candidate": True,
        "pages": pages,
    }


def validate_manifest(path: Path) -> None:
    for page in load_stage1_manifest(path):
        image = cv2.imread(page.source_image, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise FileNotFoundError(page.source_image)
        load_page_masks(page, image.shape[:2])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and seal an A1 v3 manifest from source-only decisions. "
            "Candidate images are not inputs to this command."
        )
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.source_manifest.resolve()
    decisions = args.decisions.resolve()
    output = args.output.resolve()
    baseline = args.baseline_manifest.resolve() if args.baseline_manifest else None
    payload = build_manifest(source, decisions, baseline)
    attach_artifact_hashes(payload)
    _write_json(output, payload)
    validate_manifest(output)
    seal = {
        "schema_version": "inpaint-factorized-manifest-seal-v3",
        "manifest": output.name,
        "manifest_sha256": _sha256(output),
        "source_manifest_sha256": _sha256(source),
        "source_decisions_sha256": _sha256(decisions),
        "baseline_manifest_sha256": _sha256(baseline) if baseline else None,
        "candidate_generated": False,
    }
    _write_json(output.with_suffix(output.suffix + ".seal.json"), seal)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
