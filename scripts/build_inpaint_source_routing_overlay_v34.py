#!/usr/bin/env python3
"""Seal explicit source-only ownership/protection artifacts for v3.4 routing.

The builder never guesses sibling filenames. Every source mask is named by an
upstream source-only evidence record and is reopened to verify its file and
pixel SHA before it can enter the overlay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.contracts import (  # noqa: E402
    mask_sha256,
)
from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    validate_source_only_manifest_v4,
)
from benchmarking.inpaint_detector_bakeoff.semantic import (  # noqa: E402
    PRESERVE,
    REVIEW,
    TRANSLATE,
)


SOURCE_EVIDENCE_SCHEMA_VERSION = "inpaint-source-routing-artifact-evidence-v34"
OVERLAY_SCHEMA_VERSION = "inpaint-source-routing-overlay-v34"
OVERLAY_SEAL_SCHEMA_VERSION = "inpaint-source-routing-overlay-seal-v34"
MASK_ROLES = ("ownership", "protected", "ambiguous", "corner")

_EVIDENCE_TOP_KEYS = frozenset(
    {
        "schema_version",
        "source_manifest_sha256",
        "relative_manifest_sha256",
        "source_page_inventory_sha256",
        "candidate_generated",
        "candidate_seen",
        "annotation_frozen_before_candidate",
        "producer",
        "pages",
        "evidence_sha256",
    }
)
_EVIDENCE_PAGE_KEYS = frozenset(
    {
        "page_id",
        "source_sha256",
        "candidate_generated",
        "candidate_seen",
        "annotation_frozen_before_candidate",
        "regions",
    }
)
_EVIDENCE_REGION_KEYS = frozenset(
    {
        "region_id",
        "source_region_record_sha256",
        "candidate_generated",
        "candidate_seen",
        "annotation_frozen_before_candidate",
        "artifacts",
    }
)
_ARTIFACT_KEYS = frozenset({"path", "file_sha256", "pixel_sha256"})


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _file_sha256(path: Path) -> str:
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


def _unsigned_sha256(payload: Mapping[str, object], field: str) -> str:
    return _canonical_sha256({key: value for key, value in payload.items() if key != field})


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    if set(value) != set(expected):
        raise ValueError(f"v3.4 {label} fields differ")


def _resolve_evidence_path(evidence_path: Path, value: object) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("v3.4 source routing artifact path is empty")
    path = Path(raw)
    if not path.is_absolute():
        path = evidence_path.parent / path
    return path.resolve()


def _read_binary_mask(path: Path) -> np.ndarray:
    decoded = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if decoded is None or decoded.ndim != 2 or decoded.size == 0:
        raise ValueError("v3.4 source routing artifact is not a mask")
    values = np.unique(decoded)
    if np.any((values != 0) & (values != 255)):
        raise ValueError("v3.4 source routing artifact is not binary")
    return np.ascontiguousarray(decoded)


def _sealed_semantic_decision(region: Mapping[str, object]) -> dict[str, object]:
    proposal = region.get("proposal")
    evidence = proposal if isinstance(proposal, Mapping) else region
    text_class = str(evidence.get("text_class") or "").strip().lower()
    explicit_action = str(
        evidence.get("processing_action")
        or region.get("processing_action")
        or ""
    ).strip().lower()
    explicit_role = str(
        evidence.get("semantic_role") or region.get("semantic_role") or ""
    ).strip().lower()
    preserve_classes = {"sfx", "onomatopoeia", "decorative", "decoration"}
    translate_classes = {"text_bubble", "text_free", "narration", "caption"}
    class_action = (
        PRESERVE
        if text_class in preserve_classes
        else TRANSLATE
        if text_class in translate_classes
        else REVIEW
    )
    role = explicit_role or {
        "text_bubble": "dialogue_bubble",
        "text_free": "dialogue_free",
        "narration": "narration",
        "caption": "narration",
    }.get(text_class, text_class or "ambiguous")
    if explicit_action:
        if explicit_action not in {TRANSLATE, PRESERVE, REVIEW}:
            return {
                "semantic_role": "ambiguous",
                "semantic_action": REVIEW,
                "semantic_available": False,
                "semantic_reason": "invalid_processing_action",
                "semantic_provenance": "explicit_processing_action",
            }
        if explicit_action == REVIEW:
            return {
                "semantic_role": role,
                "semantic_action": REVIEW,
                "semantic_available": True,
                "semantic_reason": "explicit_review",
                "semantic_provenance": "explicit_processing_action",
            }
        role_conflict = (
            explicit_role in {"sfx", "onomatopoeia", "decorative", "decoration"}
            and explicit_action != PRESERVE
        )
        class_conflict = class_action != REVIEW and class_action != explicit_action
        if role_conflict or class_conflict:
            return {
                "semantic_role": "ambiguous",
                "semantic_action": REVIEW,
                "semantic_available": False,
                "semantic_reason": "semantic_role_action_conflict",
                "semantic_provenance": "explicit_processing_action",
            }
        return {
            "semantic_role": role,
            "semantic_action": explicit_action,
            "semantic_available": True,
            "semantic_reason": "explicit_processing_action",
            "semantic_provenance": "explicit_processing_action",
        }
    if class_action == REVIEW:
        return {
            "semantic_role": "ambiguous",
            "semantic_action": REVIEW,
            "semantic_available": False,
            "semantic_reason": "semantic_evidence_missing_or_unknown",
            "semantic_provenance": "proposal_text_class_fallback",
        }
    return {
        "semantic_role": role,
        "semantic_action": class_action,
        "semantic_available": True,
        "semantic_reason": "proposal_text_class_fallback",
        "semantic_provenance": "proposal_text_class_fallback",
    }


def _manifest_pages(
    source_manifest_path: Path,
    relative_manifest_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, dict[str, object]]]:
    source_binding = validate_source_only_manifest_v4(source_manifest_path)
    relative_binding = validate_source_only_manifest_v4(relative_manifest_path)
    source_payload = _read_json(source_manifest_path)
    relative_payload = _read_json(relative_manifest_path)
    relative_seal_path = relative_manifest_path.with_suffix(
        relative_manifest_path.suffix + ".seal.json"
    )
    relative_seal = _read_json(relative_seal_path)
    if relative_seal.get("source_manifest_sha256") != source_binding["manifest_sha256"]:
        raise ValueError("v3.4 source routing relative manifest binding differs")
    source_rows = source_payload.get("pages")
    relative_rows = relative_payload.get("pages")
    if not isinstance(source_rows, list) or not isinstance(relative_rows, list):
        raise ValueError("v3.4 source routing manifest pages are invalid")
    sources = {
        str(row.get("page_id") or ""): row
        for row in source_rows
        if isinstance(row, Mapping)
    }
    relatives = {
        str(row.get("page_id") or ""): row
        for row in relative_rows
        if isinstance(row, Mapping)
    }
    if not sources or set(sources) != set(relatives):
        raise ValueError("v3.4 source routing manifest page inventory differs")
    pages: dict[str, dict[str, object]] = {}
    for page_id in sorted(sources):
        source = sources[page_id]
        relative = relatives[page_id]
        source_sha = str(source.get("source_sha256") or "").lower()
        if source_sha != str(relative.get("source_sha256") or source_sha).lower():
            raise ValueError("v3.4 source routing source page SHA differs")
        raw_regions = relative.get("regions")
        if not isinstance(raw_regions, list):
            raise ValueError("v3.4 source routing regions are invalid")
        regions: dict[str, Mapping[str, object]] = {}
        for raw_region in raw_regions:
            if not isinstance(raw_region, Mapping):
                raise ValueError("v3.4 source routing region is invalid")
            region_id = str(raw_region.get("region_id") or "").strip()
            if not region_id or region_id in regions:
                raise ValueError("v3.4 source routing region identity differs")
            regions[region_id] = raw_region
        pages[page_id] = {"source_sha256": source_sha, "regions": regions}
    return source_binding, relative_binding, pages


def build_source_routing_overlay(
    source_manifest_path: Path,
    relative_manifest_path: Path,
    source_evidence_path: Path,
) -> dict[str, object]:
    source_manifest_path = source_manifest_path.resolve()
    relative_manifest_path = relative_manifest_path.resolve()
    source_evidence_path = source_evidence_path.resolve()
    source_binding, relative_binding, manifest_pages = _manifest_pages(
        source_manifest_path,
        relative_manifest_path,
    )
    evidence = _read_json(source_evidence_path)
    _exact_keys(evidence, _EVIDENCE_TOP_KEYS, label="source routing evidence")
    if evidence.get("schema_version") != SOURCE_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("v3.4 source routing evidence schema differs")
    if evidence.get("source_manifest_sha256") != source_binding["manifest_sha256"]:
        raise ValueError("v3.4 source routing evidence source manifest differs")
    if evidence.get("relative_manifest_sha256") != relative_binding["manifest_sha256"]:
        raise ValueError("v3.4 source routing evidence relative manifest differs")
    if evidence.get("source_page_inventory_sha256") != source_binding["page_inventory_sha256"]:
        raise ValueError("v3.4 source routing evidence page inventory differs")
    if (
        evidence.get("candidate_generated") is not False
        or evidence.get("candidate_seen") is not False
        or evidence.get("annotation_frozen_before_candidate") is not True
    ):
        raise ValueError("v3.4 source routing evidence is not source-only")
    producer = str(evidence.get("producer") or "").strip()
    if not producer:
        raise ValueError("v3.4 source routing evidence producer is missing")
    if evidence.get("evidence_sha256") != _unsigned_sha256(evidence, "evidence_sha256"):
        raise ValueError("v3.4 source routing evidence canonical SHA differs")
    raw_pages = evidence.get("pages")
    if not isinstance(raw_pages, list):
        raise ValueError("v3.4 source routing evidence pages are invalid")
    evidence_pages = {
        str(row.get("page_id") or ""): row
        for row in raw_pages
        if isinstance(row, Mapping)
    }
    if set(evidence_pages) != set(manifest_pages) or len(evidence_pages) != len(raw_pages):
        raise ValueError("v3.4 source routing evidence page inventory differs")

    output_pages: list[dict[str, object]] = []
    artifact_count = 0
    for page_id in sorted(manifest_pages):
        raw_page = evidence_pages[page_id]
        _exact_keys(raw_page, _EVIDENCE_PAGE_KEYS, label="source routing page")
        source_page = manifest_pages[page_id]
        if str(raw_page.get("source_sha256") or "").lower() != source_page["source_sha256"]:
            raise ValueError("v3.4 source routing evidence source SHA differs")
        if (
            raw_page.get("candidate_generated") is not False
            or raw_page.get("candidate_seen") is not False
            or raw_page.get("annotation_frozen_before_candidate") is not True
        ):
            raise ValueError("v3.4 source routing evidence page is not source-only")
        raw_regions = raw_page.get("regions")
        if not isinstance(raw_regions, list):
            raise ValueError("v3.4 source routing evidence regions are invalid")
        evidence_regions = {
            str(row.get("region_id") or ""): row
            for row in raw_regions
            if isinstance(row, Mapping)
        }
        manifest_regions = source_page["regions"]
        assert isinstance(manifest_regions, Mapping)
        if set(evidence_regions) != set(manifest_regions) or len(evidence_regions) != len(raw_regions):
            raise ValueError("v3.4 source routing evidence region inventory differs")
        output_regions: list[dict[str, object]] = []
        for region_id in sorted(manifest_regions):
            raw_region = evidence_regions[region_id]
            _exact_keys(raw_region, _EVIDENCE_REGION_KEYS, label="source routing region")
            if (
                raw_region.get("candidate_generated") is not False
                or raw_region.get("candidate_seen") is not False
                or raw_region.get("annotation_frozen_before_candidate") is not True
            ):
                raise ValueError("v3.4 source routing evidence region is not source-only")
            manifest_record_sha = _canonical_sha256(dict(manifest_regions[region_id]))
            if raw_region.get("source_region_record_sha256") != manifest_record_sha:
                raise ValueError("v3.4 source routing region record SHA differs")
            raw_artifacts = raw_region.get("artifacts")
            if not isinstance(raw_artifacts, Mapping) or set(raw_artifacts) != set(MASK_ROLES):
                raise ValueError("v3.4 source routing artifact role inventory differs")
            artifacts: dict[str, dict[str, object]] = {}
            common_shape: tuple[int, int] | None = None
            for role in MASK_ROLES:
                descriptor = raw_artifacts[role]
                if not isinstance(descriptor, Mapping):
                    raise ValueError("v3.4 source routing artifact descriptor is invalid")
                _exact_keys(descriptor, _ARTIFACT_KEYS, label="source routing artifact")
                path = _resolve_evidence_path(source_evidence_path, descriptor.get("path"))
                mask = _read_binary_mask(path)
                shape = tuple(mask.shape)
                if common_shape is None:
                    common_shape = shape
                elif shape != common_shape:
                    raise ValueError("v3.4 source routing artifact shape differs")
                file_sha = _file_sha256(path)
                pixel_sha = mask_sha256(mask)
                if (
                    descriptor.get("file_sha256") != file_sha
                    or descriptor.get("pixel_sha256") != pixel_sha
                ):
                    raise ValueError("v3.4 source routing artifact SHA differs")
                if role == "ownership":
                    relative_ownership_raw = manifest_regions[region_id].get(
                        "ownership_mask"
                    )
                    if not str(relative_ownership_raw or "").strip():
                        raise ValueError(
                            "v3.4 relative manifest lacks exact ownership role"
                        )
                    relative_ownership = Path(str(relative_ownership_raw))
                    if not relative_ownership.is_absolute():
                        relative_ownership = (
                            relative_manifest_path.parent / relative_ownership
                        )
                    relative_ownership = relative_ownership.resolve()
                    relative_mask = _read_binary_mask(relative_ownership)
                    if (
                        relative_ownership != path
                        or _file_sha256(relative_ownership) != file_sha
                        or mask_sha256(relative_mask) != pixel_sha
                        or tuple(relative_mask.shape) != shape
                    ):
                        raise ValueError(
                            "v3.4 relative ownership role differs from source evidence"
                        )
                artifacts[role] = {
                    "path": str(path),
                    "file_sha256": file_sha,
                    "pixel_sha256": pixel_sha,
                    "shape": [int(value) for value in shape],
                }
                artifact_count += 1
            output_regions.append(
                {
                    "region_id": region_id,
                    "source_region_record_sha256": manifest_record_sha,
                    **_sealed_semantic_decision(manifest_regions[region_id]),
                    "artifacts": artifacts,
                }
            )
        output_pages.append(
            {
                "page_id": page_id,
                "source_sha256": source_page["source_sha256"],
                "regions": output_regions,
            }
        )

    payload: dict[str, object] = {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "source_manifest_sha256": source_binding["manifest_sha256"],
        "source_manifest_file_sha256": _file_sha256(source_manifest_path),
        "source_manifest_seal_sha256": source_binding["seal_sha256"],
        "relative_manifest_sha256": relative_binding["manifest_sha256"],
        "relative_manifest_file_sha256": _file_sha256(relative_manifest_path),
        "relative_manifest_seal_sha256": relative_binding["seal_sha256"],
        "source_page_inventory_sha256": source_binding["page_inventory_sha256"],
        "source_evidence_schema_version": SOURCE_EVIDENCE_SCHEMA_VERSION,
        "source_evidence_path": str(source_evidence_path),
        "source_evidence_file_sha256": _file_sha256(source_evidence_path),
        "source_evidence_payload_sha256": evidence["evidence_sha256"],
        "source_evidence_producer": producer,
        "candidate_generated": False,
        "candidate_seen": False,
        "annotation_frozen_before_candidate": True,
        "summary": {
            "page_count": len(output_pages),
            "region_count": sum(len(page["regions"]) for page in output_pages),
            "artifact_count": artifact_count,
        },
        "pages": output_pages,
    }
    payload["overlay_sha256"] = _unsigned_sha256(payload, "overlay_sha256")
    return payload


def write_source_routing_overlay(
    source_manifest_path: Path,
    relative_manifest_path: Path,
    source_evidence_path: Path,
    output_path: Path,
) -> tuple[Path, Path, dict[str, object]]:
    output_path = output_path.resolve()
    seal_path = output_path.with_suffix(output_path.suffix + ".seal.json")
    temporary_output = output_path.with_name(f".{output_path.name}.partial")
    temporary_seal = seal_path.with_name(f".{seal_path.name}.partial")
    for path in (output_path, seal_path, temporary_output, temporary_seal):
        if path.exists():
            raise FileExistsError(f"v3.4 source routing overlay must be fresh: {path}")
    payload = build_source_routing_overlay(
        source_manifest_path,
        relative_manifest_path,
        source_evidence_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    seal: dict[str, object] = {
        "schema_version": OVERLAY_SEAL_SCHEMA_VERSION,
        "overlay_file_sha256": _file_sha256(temporary_output),
        "overlay_sha256": payload["overlay_sha256"],
        "source_manifest_sha256": payload["source_manifest_sha256"],
        "source_manifest_file_sha256": payload["source_manifest_file_sha256"],
        "source_manifest_seal_sha256": payload["source_manifest_seal_sha256"],
        "relative_manifest_sha256": payload["relative_manifest_sha256"],
        "relative_manifest_file_sha256": payload["relative_manifest_file_sha256"],
        "relative_manifest_seal_sha256": payload["relative_manifest_seal_sha256"],
        "source_page_inventory_sha256": payload["source_page_inventory_sha256"],
        "source_evidence_file_sha256": payload["source_evidence_file_sha256"],
        "source_evidence_payload_sha256": payload["source_evidence_payload_sha256"],
        "candidate_generated": False,
        "candidate_seen": False,
        "annotation_frozen_before_candidate": True,
    }
    seal["seal_payload_sha256"] = _unsigned_sha256(seal, "seal_payload_sha256")
    temporary_seal.write_text(
        json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_output.replace(output_path)
    temporary_seal.replace(seal_path)
    return output_path, seal_path, payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seal explicit source-only routing masks for v3.4."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--relative-manifest", type=Path, required=True)
    parser.add_argument("--source-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output, seal, payload = write_source_routing_overlay(
        args.source_manifest,
        args.relative_manifest,
        args.source_evidence,
        args.output,
    )
    print(
        json.dumps(
            {"output": str(output), "seal": str(seal), "summary": payload["summary"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
