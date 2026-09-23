#!/usr/bin/env python3
"""Seal source-only OCR/block evidence for the v3.4 seedless B3 probe.

The input to this script is a deliberately small, normalized export from the
product OCR/block path.  It contains identities and provenance only; geometry,
candidate output, and evaluation masks are not part of the schema.  Incomplete
region evidence is reported as information-limited and is never admitted to the
runner overlay.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]


SOURCE_MANIFEST_SCHEMA_VERSION = "inpaint-factorized-source-manifest-v4"
SOURCE_EVIDENCE_SCHEMA_VERSION = "inpaint-source-ocr-block-evidence-v34"
SOURCE_EVIDENCE_SEAL_SCHEMA_VERSION = "inpaint-source-ocr-block-evidence-seal-v34"
SOURCE_EVIDENCE_RECEIPT_SCHEMA_VERSION = "inpaint-source-ocr-runtime-receipt-v34"
OVERLAY_SCHEMA_VERSION = "inpaint-seedless-ocr-evidence-overlay-v34"
OVERLAY_SEAL_SCHEMA_VERSION = "inpaint-seedless-ocr-evidence-seal-v34"

TRANSLATE_ACTION = "translate_inpaint"
SEEDLESS_ROUTE_CLASS = "clean_translucent"
EXACT_OWNER_BINDING_KINDS = frozenset(
    {
        "canonical_block_region_id",
        "exact_product_region_id",
    }
)
TRUSTED_CONFIDENCE_KINDS = frozenset(
    {
        "detector_block_confidence",
        "recognizer_confidence",
        "calibrated_ocr_confidence",
    }
)
TRACKED_RUNTIME_DEPENDENCIES = (
    "scripts/build_inpaint_seedless_ocr_overlay_v34.py",
    "scripts/export_inpaint_source_ocr_evidence_v34.py",
    "modules/detection/processor.py",
    "modules/detection/factory.py",
    "modules/detection/base.py",
    "modules/detection/rtdetr_v2_onnx.py",
    "modules/detection/utils/slicer.py",
    "modules/ocr/local_runtime.py",
    "modules/ocr/paddle_crop/engine.py",
    "modules/ocr/paddle_crop/crop_policy.py",
    "modules/ocr/paddle_crop/response_parser.py",
    "modules/ocr/paddle_crop/runtime.py",
    "modules/ocr/paddle_crop/transport.py",
    "modules/ocr/common/result_contract.py",
    "modules/utils/textblock.py",
)

_TOP_EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "source_manifest_sha256",
        "source_page_inventory_sha256",
        "candidate_seen",
        "annotation_frozen_before_candidate",
        "producer",
        "pages",
    }
)
_PAGE_EVIDENCE_KEYS = frozenset(
    {
        "page_id",
        "source_sha256",
        "candidate_seen",
        "annotation_frozen_before_candidate",
        "regions",
    }
)
_REGION_EVIDENCE_KEYS = frozenset(
    {
        "region_id",
        "owner_region_id",
        "owner_count",
        "owner_binding_kind",
        "canonical_block_id",
        "ocr_text",
        "ocr_script",
        "provider",
        "ocr_confidence",
        "confidence_kind",
        "processing_action",
        "route_class",
        "authoritative_ocr",
        "candidate_seen",
        "annotation_frozen_before_candidate",
        "provenance",
    }
)
_PROVENANCE_KEYS = frozenset(
    {
        "ocr_artifact_sha256",
        "confidence_artifact_sha256",
        "owner_binding_sha256",
        "action_artifact_sha256",
        "route_artifact_sha256",
    }
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


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


def _overlay_sha256(payload: Mapping[str, object]) -> str:
    return _canonical_sha256(
        {key: value for key, value in payload.items() if key != "overlay_sha256"}
    )


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def _require_exact_keys(
    value: Mapping[str, object],
    allowed: frozenset[str],
    *,
    label: str,
) -> None:
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise ValueError(f"{label} contains unsupported fields: {unexpected}")


def _source_manifest_inventory(
    manifest_path: Path,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Read only sealed identities from a v4 source manifest.

    In particular, this function never resolves or opens any path-bearing page,
    target, protection, or annotation field from the manifest.
    """

    payload = _read_json(manifest_path)
    seal_path = manifest_path.with_suffix(manifest_path.suffix + ".seal.json")
    seal = _read_json(seal_path)
    manifest_sha256 = _file_sha256(manifest_path)
    if payload.get("schema_version") != SOURCE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("v3.4 OCR overlay requires a source-only v4 manifest")
    if seal.get("schema_version") not in {
        "inpaint-factorized-manifest-seal-v4-independent",
        "inpaint-factorized-manifest-seal-v4-synthetic",
        "inpaint-factorized-manifest-seal-v4",
    }:
        raise ValueError("v3.4 OCR overlay source manifest seal schema differs")
    if seal.get("manifest_sha256") != manifest_sha256:
        raise ValueError("v3.4 OCR overlay source manifest SHA differs")
    if (
        payload.get("candidate_seen") is not False
        or payload.get("annotation_frozen_before_candidate") is not True
        or seal.get("candidate_generated") is not False
        or seal.get("candidate_seen") is not False
        or seal.get("annotation_frozen_before_candidate") is not True
    ):
        raise ValueError("v3.4 OCR overlay source manifest is not source-only")

    page_inventory_sha256 = payload.get("page_inventory_sha256")
    if not _is_sha256(page_inventory_sha256):
        raise ValueError("v3.4 OCR overlay source page inventory SHA is invalid")
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        raise ValueError("v3.4 OCR overlay source pages are missing")
    if payload.get("page_count") != len(raw_pages):
        raise ValueError("v3.4 OCR overlay source page count differs")

    pages: dict[str, dict[str, object]] = {}
    for raw_page in raw_pages:
        if not isinstance(raw_page, Mapping):
            raise ValueError("v3.4 OCR overlay source page is invalid")
        page_id = str(raw_page.get("page_id") or "").strip()
        source_sha256 = str(raw_page.get("source_sha256") or "").lower()
        if not page_id or page_id in pages or not _is_sha256(source_sha256):
            raise ValueError("v3.4 OCR overlay source page identity is invalid")
        if (
            raw_page.get("candidate_seen") is not False
            or raw_page.get("annotation_frozen_before_candidate") is not True
        ):
            raise ValueError("v3.4 OCR overlay source page is not source-only")
        raw_regions = raw_page.get("regions")
        if not isinstance(raw_regions, list):
            raise ValueError("v3.4 OCR overlay source regions are invalid")
        regions: dict[str, bool] = {}
        for raw_region in raw_regions:
            if not isinstance(raw_region, Mapping):
                raise ValueError("v3.4 OCR overlay source region is invalid")
            region_id = str(raw_region.get("region_id") or "").strip()
            if not region_id or region_id in regions:
                raise ValueError("v3.4 OCR overlay source region identity is invalid")
            regions[region_id] = raw_region.get("source_reviewed") is True
        pages[page_id] = {
            "source_sha256": source_sha256,
            "regions": regions,
        }

    binding: dict[str, object] = {
        "manifest_sha256": manifest_sha256,
        "manifest_file_sha256": manifest_sha256,
        "manifest_seal_sha256": _file_sha256(seal_path),
        "page_inventory_sha256": str(page_inventory_sha256).lower(),
        "page_count": len(pages),
    }
    return binding, pages


def _validate_source_evidence_chain(
    evidence_path: Path,
    *,
    evidence: Mapping[str, object],
    binding: Mapping[str, object],
    source_pages: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    receipt_path = evidence_path.with_suffix(evidence_path.suffix + ".receipt.json")
    seal_path = evidence_path.with_suffix(evidence_path.suffix + ".seal.json")
    receipt = _read_json(receipt_path)
    seal = _read_json(seal_path)
    seal_keys = {
        "schema_version",
        "evidence_file_sha256",
        "receipt_file_sha256",
        "evidence_payload_sha256",
        "receipt_payload_sha256",
        "source_region_inventory_sha256",
        "record_inventory_sha256",
        "source_manifest_sha256",
        "source_page_inventory_sha256",
        "provider_identity_sha256",
        "tracked_dependency_identity_sha256",
        "page_count",
        "complete_page_set",
        "candidate_generated",
        "candidate_seen",
        "annotation_frozen_before_candidate",
    }
    receipt_keys = {
        "schema_version",
        "source_manifest_sha256",
        "source_manifest_seal_sha256",
        "source_page_inventory_sha256",
        "candidate_seen",
        "annotation_frozen_before_candidate",
        "git_head",
        "provider",
        "provider_identity_sha256",
        "evidence_payload_sha256",
        "evidence_file_sha256",
        "source_region_inventory_sha256",
        "record_inventory_sha256",
        "source_region_inventory",
        "record_inventory",
        "detector_identity",
        "detector_identity_sha256",
        "ocr_identity",
        "ocr_identity_sha256",
        "tracked_dependency_identity",
        "tracked_dependency_identity_sha256",
        "reuse_evidence_sha256",
        "summary",
        "pages",
    }
    _require_exact_keys(seal, frozenset(seal_keys), label="source evidence seal")
    _require_exact_keys(receipt, frozenset(receipt_keys), label="source evidence receipt")
    evidence_file_sha = _file_sha256(evidence_path)
    receipt_file_sha = _file_sha256(receipt_path)
    evidence_payload_sha = _canonical_sha256(dict(evidence))
    receipt_payload_sha = _canonical_sha256(dict(receipt))
    if (
        seal.get("schema_version") != SOURCE_EVIDENCE_SEAL_SCHEMA_VERSION
        or receipt.get("schema_version") != SOURCE_EVIDENCE_RECEIPT_SCHEMA_VERSION
        or seal.get("evidence_file_sha256") != evidence_file_sha
        or seal.get("receipt_file_sha256") != receipt_file_sha
        or seal.get("evidence_payload_sha256") != evidence_payload_sha
        or seal.get("receipt_payload_sha256") != receipt_payload_sha
        or receipt.get("evidence_file_sha256") != evidence_file_sha
        or receipt.get("evidence_payload_sha256") != evidence_payload_sha
        or seal.get("source_manifest_sha256") != binding["manifest_sha256"]
        or receipt.get("source_manifest_sha256") != binding["manifest_sha256"]
        or receipt.get("source_manifest_seal_sha256")
        != binding["manifest_seal_sha256"]
        or seal.get("source_page_inventory_sha256")
        != binding["page_inventory_sha256"]
        or receipt.get("source_page_inventory_sha256")
        != binding["page_inventory_sha256"]
        or seal.get("candidate_generated") is not False
        or seal.get("candidate_seen") is not False
        or seal.get("annotation_frozen_before_candidate") is not True
        or receipt.get("candidate_seen") is not False
        or receipt.get("annotation_frozen_before_candidate") is not True
        or seal.get("complete_page_set") is not True
        or seal.get("page_count") != len(source_pages)
    ):
        raise ValueError("v3.4 source OCR evidence seal/receipt binding differs")
    detector = receipt.get("detector_identity")
    ocr = receipt.get("ocr_identity")
    tracked = receipt.get("tracked_dependency_identity")
    if not all(isinstance(value, Mapping) for value in (detector, ocr, tracked)):
        raise ValueError("v3.4 source OCR runtime identity is invalid")
    if (
        receipt.get("detector_identity_sha256")
        != _canonical_sha256(dict(detector))
        or receipt.get("ocr_identity_sha256") != _canonical_sha256(dict(ocr))
        or receipt.get("tracked_dependency_identity_sha256")
        != _canonical_sha256(dict(tracked))
        or seal.get("tracked_dependency_identity_sha256")
        != receipt.get("tracked_dependency_identity_sha256")
    ):
        raise ValueError("v3.4 source OCR runtime identity SHA differs")
    provider = str(receipt.get("provider") or "")
    provider_identity = _canonical_sha256(
        {
            "provider": provider.split(":", 1)[0],
            "detector": dict(detector),
            "ocr": dict(ocr),
            "code": dict(tracked),
        }
    )
    if (
        not provider
        or evidence.get("producer") != provider
        or receipt.get("provider_identity_sha256") != provider_identity
        or seal.get("provider_identity_sha256") != provider_identity
        or not provider.endswith(f":{provider_identity}")
    ):
        raise ValueError("v3.4 source OCR provider identity differs")
    dependencies = tracked.get("dependencies")
    expected_paths = set(TRACKED_RUNTIME_DEPENDENCIES)
    actual_paths = {
        str(row.get("path") or "")
        for row in dependencies
        if isinstance(row, Mapping)
    } if isinstance(dependencies, list) else set()
    if (
        tracked.get("git_head") != receipt.get("git_head")
        or tracked.get("tracked_worktree_clean") is not True
        or not isinstance(dependencies, list)
        or tracked.get("dependency_count") != len(dependencies)
        or tracked.get("dependency_inventory_sha256")
        != _canonical_sha256(dependencies)
        or len(dependencies) != len(expected_paths)
        or actual_paths != expected_paths
    ):
        raise ValueError("v3.4 source OCR tracked dependency proof differs")
    current_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_head != tracked.get("git_head"):
        raise ValueError("v3.4 source OCR evidence HEAD differs")
    for row in dependencies:
        if not isinstance(row, Mapping):
            raise ValueError("v3.4 source OCR dependency row is invalid")
        relative = str(row.get("path") or "")
        path = (ROOT / relative).resolve()
        if (
            not relative
            or ROOT.resolve() not in path.parents
            or row.get("tracked") is not True
            or row.get("unchanged_from_head") is not True
            or not str(row.get("head_blob_id") or "")
            or row.get("working_file_sha256") != _file_sha256(path)
        ):
            raise ValueError("v3.4 source OCR dependency changed after receipt")
        tracked_result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        blob_result = subprocess.run(
            ["git", "rev-parse", f"HEAD:{relative}"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if (
            tracked_result.returncode != 0
            or blob_result.returncode != 0
            or blob_result.stdout.strip() != row.get("head_blob_id")
        ):
            raise ValueError("v3.4 source OCR dependency Git binding differs")
    source_inventory = receipt.get("source_region_inventory")
    record_inventory = receipt.get("record_inventory")
    if not isinstance(source_inventory, list) or not isinstance(record_inventory, list):
        raise ValueError("v3.4 source OCR inventory is invalid")
    if (
        receipt.get("source_region_inventory_sha256")
        != _canonical_sha256(source_inventory)
        or seal.get("source_region_inventory_sha256")
        != receipt.get("source_region_inventory_sha256")
        or receipt.get("record_inventory_sha256")
        != _canonical_sha256(record_inventory)
        or seal.get("record_inventory_sha256")
        != receipt.get("record_inventory_sha256")
    ):
        raise ValueError("v3.4 source OCR inventory SHA differs")
    source_ids = {
        str(page.get("page_id") or ""): {
            str(region.get("region_id") or "")
            for region in page.get("regions", [])
            if isinstance(region, Mapping)
        }
        for page in source_inventory
        if isinstance(page, Mapping)
    }
    expected_ids = {
        page_id: set(str(region_id) for region_id in page["regions"])
        for page_id, page in source_pages.items()
    }
    if source_ids != expected_ids:
        raise ValueError("v3.4 source OCR source region inventory differs")
    actual_records: list[dict[str, object]] = []
    for page in evidence.get("pages", []):
        if not isinstance(page, Mapping):
            raise ValueError("v3.4 source OCR evidence page is invalid")
        page_id = str(page.get("page_id") or "")
        source_sha = str(page.get("source_sha256") or "")
        raw_regions = page.get("regions")
        if not isinstance(raw_regions, list):
            raise ValueError("v3.4 source OCR evidence regions are invalid")
        for raw in raw_regions:
            if not isinstance(raw, Mapping):
                raise ValueError("v3.4 source OCR evidence region is invalid")
            provenance = raw.get("provenance")
            actual_records.append(
                {
                    "page_id": page_id,
                    "source_sha256": source_sha,
                    "region_id": str(raw.get("region_id") or ""),
                    "source_record_sha256": _canonical_sha256(dict(raw)),
                    "provider": str(raw.get("provider") or ""),
                    "owner_count": raw.get("owner_count"),
                    "owner_binding_sha256": (
                        provenance.get("owner_binding_sha256")
                        if isinstance(provenance, Mapping)
                        else None
                    ),
                    "confidence_artifact_sha256": (
                        provenance.get("confidence_artifact_sha256")
                        if isinstance(provenance, Mapping)
                        else None
                    ),
                }
            )
    actual_records.sort(key=lambda row: (str(row["page_id"]), str(row["region_id"])))
    if record_inventory != actual_records:
        raise ValueError("v3.4 source OCR record inventory differs")
    return {
        "source_evidence_receipt_path": str(receipt_path),
        "source_evidence_receipt_file_sha256": receipt_file_sha,
        "source_evidence_receipt_payload_sha256": receipt_payload_sha,
        "source_evidence_seal_path": str(seal_path),
        "source_evidence_seal_file_sha256": _file_sha256(seal_path),
        "source_evidence_provider_identity_sha256": provider_identity,
        "source_evidence_tracked_dependency_identity_sha256": receipt[
            "tracked_dependency_identity_sha256"
        ],
        "source_evidence_region_inventory_sha256": receipt[
            "source_region_inventory_sha256"
        ],
        "source_evidence_record_inventory_sha256": receipt[
            "record_inventory_sha256"
        ],
    }


def _normalized_evidence(
    evidence_path: Path,
    *,
    binding: Mapping[str, object],
    source_pages: Mapping[str, Mapping[str, object]],
) -> tuple[
    str,
    dict[tuple[str, str], Mapping[str, object]],
    dict[str, object],
]:
    payload = _read_json(evidence_path)
    chain = _validate_source_evidence_chain(
        evidence_path,
        evidence=payload,
        binding=binding,
        source_pages=source_pages,
    )
    _require_exact_keys(payload, _TOP_EVIDENCE_KEYS, label="source OCR evidence")
    if payload.get("schema_version") != SOURCE_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("v3.4 source OCR evidence schema differs")
    if payload.get("source_manifest_sha256") != binding["manifest_sha256"]:
        raise ValueError("v3.4 source OCR evidence manifest binding differs")
    if (
        payload.get("source_page_inventory_sha256")
        != binding["page_inventory_sha256"]
    ):
        raise ValueError("v3.4 source OCR evidence page inventory differs")
    if (
        payload.get("candidate_seen") is not False
        or payload.get("annotation_frozen_before_candidate") is not True
    ):
        raise ValueError("v3.4 source OCR evidence is not source-only")
    producer = str(payload.get("producer") or "").strip()
    if not producer:
        raise ValueError("v3.4 source OCR evidence producer is missing")
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list):
        raise ValueError("v3.4 source OCR evidence pages must be an array")

    records: dict[tuple[str, str], Mapping[str, object]] = {}
    seen_pages: set[str] = set()
    for raw_page in raw_pages:
        if not isinstance(raw_page, Mapping):
            raise ValueError("v3.4 source OCR evidence page is invalid")
        _require_exact_keys(raw_page, _PAGE_EVIDENCE_KEYS, label="OCR evidence page")
        page_id = str(raw_page.get("page_id") or "").strip()
        if not page_id or page_id in seen_pages or page_id not in source_pages:
            raise ValueError("v3.4 source OCR evidence page identity differs")
        seen_pages.add(page_id)
        if str(raw_page.get("source_sha256") or "").lower() != source_pages[
            page_id
        ]["source_sha256"]:
            raise ValueError("v3.4 source OCR evidence page source SHA differs")
        if (
            raw_page.get("candidate_seen") is not False
            or raw_page.get("annotation_frozen_before_candidate") is not True
        ):
            raise ValueError("v3.4 source OCR evidence page is not source-only")
        raw_regions = raw_page.get("regions")
        if not isinstance(raw_regions, list):
            raise ValueError("v3.4 source OCR evidence regions must be an array")
        known_regions = source_pages[page_id]["regions"]
        assert isinstance(known_regions, Mapping)
        for raw_region in raw_regions:
            if not isinstance(raw_region, Mapping):
                raise ValueError("v3.4 source OCR evidence region is invalid")
            _require_exact_keys(
                raw_region,
                _REGION_EVIDENCE_KEYS,
                label="OCR evidence region",
            )
            region_id = str(raw_region.get("region_id") or "").strip()
            identity = (page_id, region_id)
            if not region_id or region_id not in known_regions or identity in records:
                raise ValueError("v3.4 source OCR evidence region identity differs")
            provenance = raw_region.get("provenance")
            if provenance is not None:
                if not isinstance(provenance, Mapping):
                    raise ValueError("v3.4 OCR evidence provenance must be an object")
                _require_exact_keys(
                    provenance,
                    _PROVENANCE_KEYS,
                    label="OCR evidence provenance",
                )
            records[identity] = raw_region
    return producer, records, chain


def _region_limitations(
    raw: Mapping[str, object],
    *,
    region_id: str,
    source_reviewed: bool,
) -> list[str]:
    limitations: list[str] = []
    if not source_reviewed:
        limitations.append("source_region_not_reviewed")
    if raw.get("candidate_seen") is not False:
        limitations.append("region_source_only_status_missing")
    if raw.get("annotation_frozen_before_candidate") is not True:
        limitations.append("region_freeze_status_missing")
    if raw.get("authoritative_ocr") is not True:
        limitations.append("authoritative_ocr_missing")

    owner_region_id = str(raw.get("owner_region_id") or "").strip()
    if owner_region_id != region_id:
        limitations.append("exact_owner_region_mismatch")
    owner_count = raw.get("owner_count")
    if isinstance(owner_count, bool) or not isinstance(owner_count, int) or owner_count != 1:
        limitations.append("exact_owner_count_unclear")
    owner_binding_kind = str(raw.get("owner_binding_kind") or "").strip().lower()
    if owner_binding_kind not in EXACT_OWNER_BINDING_KINDS:
        limitations.append("exact_owner_binding_unclear")
    if not str(raw.get("canonical_block_id") or "").strip():
        limitations.append("canonical_block_id_missing")

    if not str(raw.get("ocr_text") or "").strip():
        limitations.append("ocr_text_missing")
    if not str(raw.get("ocr_script") or "").strip():
        limitations.append("ocr_script_missing")
    if not str(raw.get("provider") or "").strip():
        limitations.append("ocr_provider_missing")

    confidence = raw.get("ocr_confidence")
    valid_confidence = (
        isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and math.isfinite(float(confidence))
        and 0.0 < float(confidence) <= 1.0
    )
    if not valid_confidence:
        limitations.append("ocr_confidence_unavailable")
    confidence_kind = str(raw.get("confidence_kind") or "").strip().lower()
    if confidence_kind not in TRUSTED_CONFIDENCE_KINDS:
        limitations.append("ocr_confidence_kind_untrusted")

    if str(raw.get("processing_action") or "").strip().lower() != TRANSLATE_ACTION:
        limitations.append("processing_action_not_translate")
    if str(raw.get("route_class") or "").strip().lower() != SEEDLESS_ROUTE_CLASS:
        limitations.append("route_not_clean_translucent")

    provenance = raw.get("provenance")
    if not isinstance(provenance, Mapping):
        limitations.append("source_provenance_missing")
    else:
        for key in sorted(_PROVENANCE_KEYS):
            if not _is_sha256(provenance.get(key)):
                limitations.append(f"{key}_missing")
    return sorted(set(limitations))


def build_seedless_ocr_overlay(
    manifest_path: Path,
    product_evidence_path: Path,
) -> dict[str, object]:
    """Build a runner-compatible overlay without opening evaluation artifacts."""

    manifest_path = manifest_path.resolve()
    product_evidence_path = product_evidence_path.resolve()
    binding, source_pages = _source_manifest_inventory(manifest_path)
    producer, raw_records, source_chain = _normalized_evidence(
        product_evidence_path,
        binding=binding,
        source_pages=source_pages,
    )

    output_pages: list[dict[str, object]] = []
    reason_counts: Counter[str] = Counter()
    admitted_count = 0
    missing_count = 0
    limited_count = 0
    for page_id in sorted(source_pages):
        source_page = source_pages[page_id]
        source_regions = source_page["regions"]
        assert isinstance(source_regions, Mapping)
        admitted: list[dict[str, object]] = []
        information_limited: list[dict[str, object]] = []
        for region_id in sorted(str(key) for key in source_regions):
            raw = raw_records.get((page_id, region_id))
            if raw is None:
                missing_count += 1
                reason_counts["source_ocr_evidence_missing"] += 1
                information_limited.append(
                    {
                        "region_id": region_id,
                        "status": "missing",
                        "reasons": ["source_ocr_evidence_missing"],
                    }
                )
                continue
            limitations = _region_limitations(
                raw,
                region_id=region_id,
                source_reviewed=bool(source_regions[region_id]),
            )
            if limitations:
                limited_count += 1
                reason_counts.update(limitations)
                information_limited.append(
                    {
                        "region_id": region_id,
                        "status": "information_limited",
                        "reasons": limitations,
                    }
                )
                continue

            provenance = raw["provenance"]
            assert isinstance(provenance, Mapping)
            admitted.append(
                {
                    "region_id": region_id,
                    "owner_region_id": region_id,
                    "owner_count": 1,
                    "provider": str(raw["provider"]).strip(),
                    "ocr_text": str(raw["ocr_text"]).strip(),
                    "ocr_script": str(raw["ocr_script"]).strip(),
                    "ocr_confidence": float(raw["ocr_confidence"]),
                    "processing_action": TRANSLATE_ACTION,
                    "route_class": SEEDLESS_ROUTE_CLASS,
                    "authoritative_ocr": True,
                    "candidate_seen": False,
                    "annotation_frozen_before_candidate": True,
                    "confidence_kind": str(raw["confidence_kind"]).strip().lower(),
                    "owner_binding_kind": str(raw["owner_binding_kind"])
                    .strip()
                    .lower(),
                    "canonical_block_id": str(raw["canonical_block_id"]).strip(),
                    "source_record_sha256": _canonical_sha256(dict(raw)),
                    "source_provenance_sha256": _canonical_sha256(
                        dict(sorted(provenance.items()))
                    ),
                }
            )
            admitted_count += 1
        output_pages.append(
            {
                "page_id": page_id,
                "source_sha256": source_page["source_sha256"],
                "regions": admitted,
                "information_limited_regions": information_limited,
            }
        )

    total_region_count = sum(
        len(page["regions"])
        for page in source_pages.values()
        if isinstance(page["regions"], Mapping)
    )
    payload: dict[str, object] = {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "source_manifest_sha256": binding["manifest_sha256"],
        "source_manifest_file_sha256": binding["manifest_file_sha256"],
        "source_manifest_seal_sha256": binding["manifest_seal_sha256"],
        "source_page_inventory_sha256": binding["page_inventory_sha256"],
        "source_evidence_schema_version": SOURCE_EVIDENCE_SCHEMA_VERSION,
        "source_evidence_path": str(product_evidence_path),
        "source_evidence_file_sha256": _file_sha256(product_evidence_path),
        **source_chain,
        "source_evidence_producer": producer,
        "candidate_generated": False,
        "candidate_seen": False,
        "annotation_frozen_before_candidate": True,
        "summary": {
            "page_count": len(source_pages),
            "source_region_count": total_region_count,
            "input_evidence_record_count": len(raw_records),
            "admitted_record_count": admitted_count,
            "missing_record_count": missing_count,
            "information_limited_record_count": limited_count,
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "pages": output_pages,
    }
    payload["overlay_sha256"] = _overlay_sha256(payload)
    return payload


def write_seedless_ocr_overlay(
    manifest_path: Path,
    product_evidence_path: Path,
    output_path: Path,
) -> tuple[Path, Path, dict[str, object]]:
    output_path = output_path.resolve()
    seal_path = output_path.with_suffix(output_path.suffix + ".seal.json")
    temporary_output = output_path.with_name(f".{output_path.name}.partial")
    temporary_seal = seal_path.with_name(f".{seal_path.name}.partial")
    for path in (output_path, seal_path, temporary_output, temporary_seal):
        if path.exists():
            raise FileExistsError(f"v3.4 OCR overlay output must be fresh: {path}")

    payload = build_seedless_ocr_overlay(manifest_path, product_evidence_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    seal = {
        "schema_version": OVERLAY_SEAL_SCHEMA_VERSION,
        "overlay_file_sha256": _file_sha256(temporary_output),
        "overlay_sha256": payload["overlay_sha256"],
        "source_manifest_sha256": payload["source_manifest_sha256"],
        "source_manifest_file_sha256": payload["source_manifest_file_sha256"],
        "source_manifest_seal_sha256": payload["source_manifest_seal_sha256"],
        "source_page_inventory_sha256": payload["source_page_inventory_sha256"],
        "source_evidence_path": payload["source_evidence_path"],
        "source_evidence_file_sha256": payload["source_evidence_file_sha256"],
        "source_evidence_receipt_path": payload[
            "source_evidence_receipt_path"
        ],
        "source_evidence_receipt_file_sha256": payload[
            "source_evidence_receipt_file_sha256"
        ],
        "source_evidence_receipt_payload_sha256": payload[
            "source_evidence_receipt_payload_sha256"
        ],
        "source_evidence_seal_path": payload["source_evidence_seal_path"],
        "source_evidence_seal_file_sha256": payload[
            "source_evidence_seal_file_sha256"
        ],
        "source_evidence_provider_identity_sha256": payload[
            "source_evidence_provider_identity_sha256"
        ],
        "source_evidence_tracked_dependency_identity_sha256": payload[
            "source_evidence_tracked_dependency_identity_sha256"
        ],
        "source_evidence_region_inventory_sha256": payload[
            "source_evidence_region_inventory_sha256"
        ],
        "source_evidence_record_inventory_sha256": payload[
            "source_evidence_record_inventory_sha256"
        ],
        "candidate_generated": False,
        "candidate_seen": False,
        "annotation_frozen_before_candidate": True,
    }
    seal["seal_payload_sha256"] = _canonical_sha256(seal)
    temporary_seal.write_text(
        json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_output.replace(output_path)
    temporary_seal.replace(seal_path)
    return output_path, seal_path, payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Seal normalized source-only product OCR/block evidence for the "
            "v3.4 seedless B3 runner."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--product-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output, seal, payload = write_seedless_ocr_overlay(
        args.manifest,
        args.product_evidence,
        args.output,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "seal": str(seal),
                "summary": payload["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
