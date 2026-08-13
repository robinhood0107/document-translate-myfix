#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    manifest_page_artifact_sha256,
    source_manifest_page_inventory_sha256,
    validate_source_only_manifest_v4,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _absolute_artifact_value(base: Path, value: object) -> object:
    if isinstance(value, str) and value.strip():
        path = Path(value.strip())
        return str((path if path.is_absolute() else base / path).resolve())
    if isinstance(value, dict):
        normalized = dict(value)
        nested = normalized.get("path")
        if isinstance(nested, str) and nested.strip():
            normalized["path"] = _absolute_artifact_value(base, nested)
        return normalized
    return value


def _normalize_page_artifact_paths(page: dict[str, object], base: Path) -> None:
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
        if field in page:
            page[field] = _absolute_artifact_value(base, page[field])
    # A declared null is the canonical no-existing-edit contract.  Omission is
    # unsafe because it is indistinguishable from a producer bug.
    page.setdefault("existing_source_edit_mask", None)
    for instance in page.get("target_instances", []):
        if isinstance(instance, dict):
            field = "mask_path" if "mask_path" in instance else "mask"
            if field in instance:
                instance[field] = _absolute_artifact_value(base, instance[field])
    for region in page.get("regions", []):
        if not isinstance(region, dict):
            continue
        for field in (
            "bubble_interior_mask",
            "ownership_mask",
            "protected_structure_mask",
            "ambiguous_structure_mask",
            "corner_protect_mask",
        ):
            if field in region:
                region[field] = _absolute_artifact_value(base, region[field])
    reference = page.get("paired_reference")
    if isinstance(reference, dict) and "path" in reference:
        reference["path"] = _absolute_artifact_value(base, reference["path"])


def reseal_manifest(source: Path, output_dir: Path) -> Path:
    """Re-hash frozen source-only annotations without changing annotation fields."""

    source_path = source.resolve()
    destination = output_dir.resolve()
    if destination.exists():
        raise FileExistsError(
            f"re-seal output directory must be fresh and absent: {destination}"
        )
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source manifest root must be an object")
    if payload.get("schema_version") != "inpaint-factorized-source-manifest-v4":
        raise ValueError("re-seal requires source-only manifest v4")
    if payload.get("annotation_frozen_before_candidate") is not True:
        raise ValueError("re-seal requires frozen annotations")
    if payload.get("candidate_seen") is not False:
        raise ValueError("re-seal forbids candidate-seen annotations")
    source_seal_path = source_path.with_suffix(source_path.suffix + ".seal.json")
    if not source_seal_path.is_file():
        raise ValueError("re-seal source lacks its original seal")
    source_seal = json.loads(source_seal_path.read_text(encoding="utf-8"))
    if not isinstance(source_seal, dict):
        raise ValueError("re-seal source seal root must be an object")
    if source_seal.get("manifest_sha256") != _sha256(source_path):
        raise ValueError("re-seal source bytes differ from the original seal")
    if source_seal.get("schema_version") not in {
        "inpaint-factorized-manifest-seal-v4-independent",
        "inpaint-factorized-manifest-seal-v4-synthetic",
        "inpaint-factorized-manifest-seal-v4",
    }:
        raise ValueError("re-seal source seal schema is unsupported")
    if source_seal.get("candidate_generated") is not False:
        raise ValueError("re-seal source seal does not predate candidate generation")
    if source_seal.get("candidate_seen") is not False:
        raise ValueError("re-seal source seal records candidate inspection")
    if source_seal.get("annotation_frozen_before_candidate") is not True:
        raise ValueError("re-seal source seal lacks frozen annotation status")
    pages = payload.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("re-seal requires a non-empty page inventory")
    page_ids: list[str] = []
    destination.mkdir(parents=True)
    output = destination / source_path.name
    try:
        for value in pages:
            if not isinstance(value, dict):
                raise ValueError("re-seal manifest page must be an object")
            page_id = str(value.get("page_id") or "").strip()
            if not page_id or page_id in page_ids:
                raise ValueError("re-seal found an empty or duplicate page id")
            page_ids.append(page_id)
            if value.get("candidate_seen") is not False:
                raise ValueError(f"re-seal page {page_id} is not candidate-independent")
            if value.get("annotation_frozen_before_candidate") is not True:
                raise ValueError(f"re-seal page {page_id} was not frozen")
            if str(value.get("annotation_basis") or "") not in {
                "source_only_v4",
                "synthetic_known_ground_truth_v4",
            }:
                raise ValueError(f"re-seal page {page_id} lacks annotation_basis")
            for field in (
                "target_extent_independent",
                "target_inventory_independent",
                "target_review_complete",
            ):
                if value.get(field) is not True:
                    raise ValueError(f"re-seal page {page_id} lacks {field}")
            declared_artifacts = value.get("artifact_sha256")
            if not isinstance(declared_artifacts, dict):
                raise ValueError(
                    f"re-seal page {page_id} lacks its original artifact binding"
                )
            current_artifacts = manifest_page_artifact_sha256(source_path, value)
            if declared_artifacts != current_artifacts:
                raise ValueError(
                    f"re-seal page {page_id} artifact bytes differ from the original binding"
                )
            if value.get("source_sha256") != current_artifacts.get("path"):
                raise ValueError(
                    f"re-seal page {page_id} source bytes differ from the original binding"
                )
            _normalize_page_artifact_paths(value, source_path.parent)
            value["artifact_sha256"] = manifest_page_artifact_sha256(source_path, value)
            value["source_sha256"] = value["artifact_sha256"]["path"]
        sorted_ids = sorted(page_ids)
        payload["page_count"] = len(sorted_ids)
        payload["page_ids"] = sorted_ids
        payload["page_inventory_sha256"] = source_manifest_page_inventory_sha256(
            pages
        )
        _write_json(output, payload)
        seal = {
            "schema_version": "inpaint-factorized-manifest-seal-v4-independent",
            "manifest": output.name,
            "manifest_sha256": _sha256(output),
            "source_manifest_sha256": _sha256(source_path),
            "source_seal_sha256": _sha256(source_seal_path),
            "annotation_frozen_before_candidate": True,
            "candidate_seen": False,
            "candidate_generated": False,
            "metadata_only_reseal": True,
        }
        _write_json(output.with_suffix(output.suffix + ".seal.json"), seal)
        validate_source_only_manifest_v4(output)
    except BaseException:
        # The fresh run remains visibly incomplete for forensic inspection; a
        # caller must choose another fresh directory after correcting input.
        raise
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-hash a frozen source-only v4 manifest into a fresh private run "
            "without changing annotation or artifact paths."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(reseal_manifest(args.source, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
