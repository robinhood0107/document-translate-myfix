#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.contracts import (  # noqa: E402
    binary_mask,
    mask_sha256,
)
from benchmarking.inpaint_detector_bakeoff.evidence_ledger import (  # noqa: E402
    _validate_stage1_output_artifacts,
)
from benchmarking.inpaint_detector_bakeoff.glyph_refinement import (  # noqa: E402
    GlyphRefinementResult,
    SeedlessRoiEvidence,
    extract_roi_local_glyph_masks,
    extract_seedless_roi_glyph_masks,
)
from benchmarking.inpaint_detector_bakeoff.incremental_plan import (  # noqa: E402
    IncrementalInpaintPlan,
    PageIncrementalInpaintPlan,
    union_incremental_plans,
)
from benchmarking.inpaint_detector_bakeoff.semantic import (  # noqa: E402
    PRESERVE,
    REVIEW,
    TRANSLATE,
)
from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    PageMasks,
    _resolve_manifest_artifact,
    load_page_masks,
    load_stage1_manifest,
    validate_source_only_manifest_v4,
)
from scripts.build_inpaint_product_policy_overlay_v33 import (  # noqa: E402
    validate_policy_overlay,
)
from scripts import build_inpaint_seedless_ocr_overlay_v34 as seedless_ocr_contract  # noqa: E402
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-glyph-refinement-v34"
CATEGORY = "40-inpaint-mask-render"
SCHEMA_VERSION = "inpaint-glyph-refinement-mask-results-v34"
INVENTORY_SCHEMA_VERSION = "inpaint-glyph-refinement-output-inventory-v34"
SEEDLESS_OVERLAY_SCHEMA_VERSION = "inpaint-seedless-ocr-evidence-overlay-v34"
SEEDLESS_OVERLAY_SEAL_SCHEMA_VERSION = "inpaint-seedless-ocr-evidence-seal-v34"
SOURCE_ROUTING_OVERLAY_SCHEMA_VERSION = "inpaint-source-routing-overlay-v34"
SOURCE_ROUTING_OVERLAY_SEAL_SCHEMA_VERSION = (
    "inpaint-source-routing-overlay-seal-v34"
)
SOURCE_ROUTING_EVIDENCE_SCHEMA_VERSION = (
    "inpaint-source-routing-artifact-evidence-v34"
)
SOURCE_ROUTING_MASK_ROLES = ("ownership", "protected", "ambiguous", "corner")

ARTIFACT_ROLES = (
    "detector_seed",
    "source_claim_seed",
    "seedless_source_seed",
    "owned_positive_seed",
    "glyph_core",
    "glyph_effect",
    "hard_protect",
    "generation_mask",
    "commit_mask",
    "replacement_source_edit",
)
SHARED_SOURCE_ROLES = (
    "detector_seed",
    "source_claim_seed",
    "hard_protect",
)
SHARED_GLYPH_ROLES = (
    "owned_positive_seed",
    "glyph_core",
    "glyph_effect",
)
SHARED_CONDITIONAL_ONLY_ROLES = ("seedless_source_seed",)
PLAN_ARTIFACT_ROLES = (
    "generation_mask",
    "commit_mask",
    "replacement_source_edit",
)
MAX_SHARED_ARTIFACT_WRITES_PER_PAGE = (
    len(SHARED_SOURCE_ROLES)
    + (2 * len(SHARED_GLYPH_ROLES))
    + len(SHARED_CONDITIONAL_ONLY_ROLES)
)


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    candidate_id: str
    mode: str
    commit_source: str


@dataclass(frozen=True, slots=True)
class DetectorBundleExpectation:
    bundle_id: str
    candidate_id: str
    role_model_sha256: str
    model_asset_sha256: str
    preprocessing_contract_sha256: str


CANDIDATES = (
    CandidateSpec("b0_pr6", "baseline", "none"),
    CandidateSpec("b1_context_additive", "context_additive", "refined"),
    CandidateSpec("b2_narrow_replacement", "narrow_replacement", "core"),
    CandidateSpec("b3_conditional_segmenter", "conditional_segmenter", "refined"),
)
MAX_CANDIDATE_ARTIFACT_WRITES_PER_PAGE = (
    (len(CANDIDATES) - 1) * len(PLAN_ARTIFACT_ROLES)
)

DETECTOR_BUNDLE_EXPECTATIONS = {
    "finetune": DetectorBundleExpectation(
        bundle_id="finetune_e6_native3",
        candidate_id="ctd-synthetic-low-contrast-finetune-v4",
        role_model_sha256="c9b0e9465b858ccf2c688309a13a73e13f1e06276212fd016bfdf13748d438d7",
        model_asset_sha256="1f90fa60aeeb1eb82e2ac1167a66bf139a8a61b8780acd351ead55268540cccb",
        preprocessing_contract_sha256="a8288d925769ec9679068b37b83afa30fdca7ca6ffe7ebb5faf29dc11a28f990",
    ),
    "tiled": DetectorBundleExpectation(
        bundle_id="tiled512_native3",
        candidate_id="ballons-ctd-tiled",
        role_model_sha256="b39223bdad3927e966e5f7b326f5b4f66dc2a3294b50af57e717f62689c7e58e",
        model_asset_sha256="1f90fa60aeeb1eb82e2ac1167a66bf139a8a61b8780acd351ead55268540cccb",
        preprocessing_contract_sha256="0cdc6a555cd0bd61ba4b8475b8c1a015fdfe03caf97cd12777ebb3ae4450c57b",
    ),
}

CORE_CODE_PATHS = (
    "scripts/benchmark_inpaint_glyph_refinement_v34.py",
    "scripts/build_inpaint_seedless_ocr_overlay_v34.py",
    "scripts/build_inpaint_source_routing_overlay_v34.py",
    "scripts/export_inpaint_source_ocr_evidence_v34.py",
    "scripts/build_inpaint_product_policy_overlay_v33.py",
    "scripts/validation_artifact_harness.py",
    "benchmarking/inpaint_detector_bakeoff/glyph_refinement.py",
    "benchmarking/inpaint_detector_bakeoff/incremental_plan.py",
    "benchmarking/inpaint_detector_bakeoff/contracts.py",
    "benchmarking/inpaint_detector_bakeoff/evidence_ledger.py",
    "benchmarking/inpaint_detector_bakeoff/semantic.py",
    "benchmarking/inpaint_detector_bakeoff/stage1.py",
    "benchmarking/inpaint_detector_bakeoff/stage2.py",
)


@dataclass(frozen=True, slots=True)
class SourceRoutingEvidence:
    authoritative_translate_ownership: np.ndarray
    authoritative_translate_owner_masks: tuple[tuple[str, np.ndarray], ...]
    all_ownership: np.ndarray
    preserve_or_abstain: np.ndarray
    ownership_conflict: np.ndarray
    source_region_protect: np.ndarray
    source_region_ambiguous: np.ndarray
    source_corner_protect: np.ndarray
    hard_protect: np.ndarray
    semantic_actions: tuple[dict[str, object], ...]
    artifact_inventory: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class SourceRegionEvidence:
    region_id: str
    ownership: np.ndarray
    protected: np.ndarray
    ambiguous: np.ndarray
    corner: np.ndarray
    available: bool
    failure_reason: str
    artifacts: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class SeedlessOcrRegionRecord:
    page_id: str
    source_sha256: str
    region_id: str
    owner_region_id: str
    owner_count: int
    provider: str
    ocr_text: str
    ocr_script: str
    ocr_confidence: float
    confidence_kind: str
    owner_binding_kind: str
    canonical_block_id: str
    source_record_sha256: str
    source_provenance_sha256: str
    processing_action: str
    route_class: str


@dataclass(frozen=True, slots=True)
class SourceRoutingOverlayRegionRecord:
    page_id: str
    source_sha256: str
    region_id: str
    source_region_record_sha256: str
    semantic_role: str
    semantic_action: str
    semantic_available: bool
    semantic_reason: str
    semantic_provenance: str
    artifacts: Mapping[str, Mapping[str, object]]


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


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _canonical_without_field(value: Mapping[str, object], field: str) -> str:
    return _canonical_sha256({key: item for key, item in value.items() if key != field})


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str] | frozenset[str],
    *,
    label: str,
) -> None:
    if set(value) != set(expected):
        raise ValueError(f"v3.4 {label} fields differ")


def capture_official_code_identity(
    *,
    repo_root: Path = ROOT,
    paths: Sequence[str] = CORE_CODE_PATHS,
) -> dict[str, object]:
    """Bind official evidence to clean tracked bytes at one exact HEAD."""

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=repo_root,
        text=True,
    ).strip()
    if status:
        raise RuntimeError("v3.4 official run requires a clean tracked worktree")
    records: list[dict[str, object]] = []
    for relative in paths:
        path = (repo_root / relative).resolve()
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=repo_root,
            check=False,
            capture_output=True,
        )
        blob = subprocess.run(
            ["git", "rev-parse", f"{head}:{relative}"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0 or blob.returncode != 0 or not path.is_file():
            raise RuntimeError(f"v3.4 official code is not tracked: {relative}")
        head_bytes = subprocess.check_output(
            ["git", "show", f"{head}:{relative}"], cwd=repo_root
        )
        working_bytes = path.read_bytes()
        if head_bytes != working_bytes:
            raise RuntimeError(f"v3.4 official code differs from HEAD: {relative}")
        records.append(
            {
                "path": relative,
                "head_blob_id": blob.stdout.strip(),
                "file_sha256": hashlib.sha256(working_bytes).hexdigest(),
            }
        )
    payload: dict[str, object] = {
        "git_head": head,
        "tracked_worktree_clean": True,
        "files": records,
    }
    payload["binding_sha256"] = _canonical_sha256(payload)
    return payload


def verify_official_code_identity(
    expected: Mapping[str, object],
    *,
    repo_root: Path = ROOT,
) -> dict[str, object]:
    raw_files = expected.get("files")
    if not isinstance(raw_files, list):
        raise RuntimeError("v3.4 official code identity is invalid")
    recorded_paths = [
        str(row.get("path") or "")
        for row in raw_files
        if isinstance(row, Mapping)
    ]
    if len(recorded_paths) != len(CORE_CODE_PATHS) or set(recorded_paths) != set(CORE_CODE_PATHS):
        raise RuntimeError("v3.4 official code dependency inventory is incomplete")
    current = capture_official_code_identity(
        repo_root=repo_root,
        paths=tuple(recorded_paths),
    )
    if dict(expected) != current:
        raise RuntimeError("v3.4 official code identity changed during execution")
    return current


def _unsigned_overlay_sha256(payload: Mapping[str, object]) -> str:
    return _canonical_sha256(
        {key: value for key, value in payload.items() if key != "overlay_sha256"}
    )


def _source_semantic_decision(region: Mapping[str, object]) -> dict[str, object]:
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
    translate_roles = translate_classes | {"dialogue_bubble", "dialogue_free"}
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
    role_class_conflict = (
        (explicit_role in preserve_classes and class_action == TRANSLATE)
        or (explicit_role in translate_roles and class_action == PRESERVE)
    )
    if role_class_conflict:
        return {
            "semantic_role": "ambiguous",
            "semantic_action": REVIEW,
            "semantic_available": False,
            "semantic_reason": "semantic_role_action_conflict",
            "semantic_provenance": "source_role_and_class",
        }
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
            explicit_role in preserve_classes and explicit_action != PRESERVE
        ) or (explicit_role in translate_roles and explicit_action != TRANSLATE)
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


def validate_source_routing_overlay(
    overlay_path: Path,
    *,
    source_manifest_path: Path,
    relative_manifest_path: Path,
    source_binding: Mapping[str, object],
    relative_binding: Mapping[str, object],
    source_manifest_payload: Mapping[str, object],
    relative_manifest_payload: Mapping[str, object],
) -> dict[str, object]:
    """Authenticate exact source-only routing paths without sibling discovery."""

    seal_path = overlay_path.with_suffix(overlay_path.suffix + ".seal.json")
    payload = _read_json(overlay_path)
    seal = _read_json(seal_path)
    overlay_keys = {
        "schema_version",
        "source_manifest_sha256",
        "source_manifest_file_sha256",
        "source_manifest_seal_sha256",
        "relative_manifest_sha256",
        "relative_manifest_file_sha256",
        "relative_manifest_seal_sha256",
        "source_page_inventory_sha256",
        "source_evidence_schema_version",
        "source_evidence_path",
        "source_evidence_file_sha256",
        "source_evidence_payload_sha256",
        "source_evidence_producer",
        "candidate_generated",
        "candidate_seen",
        "annotation_frozen_before_candidate",
        "summary",
        "pages",
        "overlay_sha256",
    }
    seal_keys = {
        "schema_version",
        "overlay_file_sha256",
        "overlay_sha256",
        "source_manifest_sha256",
        "source_manifest_file_sha256",
        "source_manifest_seal_sha256",
        "relative_manifest_sha256",
        "relative_manifest_file_sha256",
        "relative_manifest_seal_sha256",
        "source_page_inventory_sha256",
        "source_evidence_file_sha256",
        "source_evidence_payload_sha256",
        "candidate_generated",
        "candidate_seen",
        "annotation_frozen_before_candidate",
        "seal_payload_sha256",
    }
    _require_exact_keys(payload, overlay_keys, label="source routing overlay")
    _require_exact_keys(seal, seal_keys, label="source routing overlay seal")
    if payload.get("schema_version") != SOURCE_ROUTING_OVERLAY_SCHEMA_VERSION:
        raise ValueError("v3.4 source routing overlay schema differs")
    if seal.get("schema_version") != SOURCE_ROUTING_OVERLAY_SEAL_SCHEMA_VERSION:
        raise ValueError("v3.4 source routing overlay seal schema differs")
    if payload.get("overlay_sha256") != _unsigned_overlay_sha256(payload):
        raise ValueError("v3.4 source routing overlay canonical SHA differs")
    if seal.get("seal_payload_sha256") != _canonical_without_field(
        seal, "seal_payload_sha256"
    ):
        raise ValueError("v3.4 source routing overlay seal payload differs")
    expected_bindings = {
        "source_manifest_sha256": source_binding.get("manifest_sha256"),
        "source_manifest_file_sha256": _sha256(source_manifest_path),
        "source_manifest_seal_sha256": source_binding.get("seal_sha256"),
        "relative_manifest_sha256": relative_binding.get("manifest_sha256"),
        "relative_manifest_file_sha256": _sha256(relative_manifest_path),
        "relative_manifest_seal_sha256": relative_binding.get("seal_sha256"),
        "source_page_inventory_sha256": source_binding.get(
            "page_inventory_sha256"
        ),
    }
    for field, expected in expected_bindings.items():
        if payload.get(field) != expected or seal.get(field) != expected:
            raise ValueError(f"v3.4 source routing {field} differs")
    if (
        seal.get("overlay_file_sha256") != _sha256(overlay_path)
        or seal.get("overlay_sha256") != payload["overlay_sha256"]
        or payload.get("candidate_generated") is not False
        or payload.get("candidate_seen") is not False
        or payload.get("annotation_frozen_before_candidate") is not True
        or seal.get("candidate_generated") is not False
        or seal.get("candidate_seen") is not False
        or seal.get("annotation_frozen_before_candidate") is not True
    ):
        raise ValueError("v3.4 source routing overlay is not sealed source-only")

    evidence_path = Path(str(payload.get("source_evidence_path") or ""))
    if not evidence_path.is_absolute():
        raise ValueError("v3.4 source routing evidence path is not absolute")
    evidence = _read_json(evidence_path)
    if (
        payload.get("source_evidence_schema_version")
        != SOURCE_ROUTING_EVIDENCE_SCHEMA_VERSION
        or evidence.get("schema_version")
        != SOURCE_ROUTING_EVIDENCE_SCHEMA_VERSION
        or payload.get("source_evidence_file_sha256") != _sha256(evidence_path)
        or seal.get("source_evidence_file_sha256")
        != payload.get("source_evidence_file_sha256")
        or payload.get("source_evidence_payload_sha256")
        != evidence.get("evidence_sha256")
        or seal.get("source_evidence_payload_sha256")
        != evidence.get("evidence_sha256")
        or evidence.get("evidence_sha256")
        != _canonical_without_field(evidence, "evidence_sha256")
        or evidence.get("candidate_generated") is not False
        or evidence.get("candidate_seen") is not False
        or evidence.get("annotation_frozen_before_candidate") is not True
    ):
        raise ValueError("v3.4 source routing source evidence differs")

    source_pages_raw = source_manifest_payload.get("pages")
    relative_pages_raw = relative_manifest_payload.get("pages")
    overlay_pages_raw = payload.get("pages")
    if not all(
        isinstance(value, list)
        for value in (source_pages_raw, relative_pages_raw, overlay_pages_raw)
    ):
        raise ValueError("v3.4 source routing page records are invalid")
    source_pages = {
        str(row.get("page_id") or ""): row
        for row in source_pages_raw
        if isinstance(row, Mapping)
    }
    relative_pages = {
        str(row.get("page_id") or ""): row
        for row in relative_pages_raw
        if isinstance(row, Mapping)
    }
    overlay_pages = {
        str(row.get("page_id") or ""): row
        for row in overlay_pages_raw
        if isinstance(row, Mapping)
    }
    if (
        not source_pages
        or set(source_pages) != set(relative_pages)
        or set(source_pages) != set(overlay_pages)
        or len(overlay_pages) != len(overlay_pages_raw)
    ):
        raise ValueError("v3.4 source routing overlay page inventory differs")
    records: dict[tuple[str, str], SourceRoutingOverlayRegionRecord] = {}
    artifact_count = 0
    for page_id in sorted(source_pages):
        source_sha = str(source_pages[page_id].get("source_sha256") or "").lower()
        overlay_page = overlay_pages[page_id]
        _require_exact_keys(
            overlay_page,
            {"page_id", "source_sha256", "regions"},
            label="source routing overlay page",
        )
        if str(overlay_page.get("source_sha256") or "").lower() != source_sha:
            raise ValueError("v3.4 source routing overlay source SHA differs")
        relative_regions_raw = relative_pages[page_id].get("regions")
        overlay_regions_raw = overlay_page.get("regions")
        if not isinstance(relative_regions_raw, list) or not isinstance(
            overlay_regions_raw, list
        ):
            raise ValueError("v3.4 source routing overlay regions are invalid")
        relative_regions = {
            str(row.get("region_id") or ""): row
            for row in relative_regions_raw
            if isinstance(row, Mapping)
        }
        overlay_regions = {
            str(row.get("region_id") or ""): row
            for row in overlay_regions_raw
            if isinstance(row, Mapping)
        }
        if (
            set(relative_regions) != set(overlay_regions)
            or len(overlay_regions) != len(overlay_regions_raw)
        ):
            raise ValueError("v3.4 source routing overlay region inventory differs")
        for region_id in sorted(relative_regions):
            row = overlay_regions[region_id]
            _require_exact_keys(
                row,
                {
                    "region_id",
                    "source_region_record_sha256",
                    "semantic_role",
                    "semantic_action",
                    "semantic_available",
                    "semantic_reason",
                    "semantic_provenance",
                    "artifacts",
                },
                label="source routing overlay region",
            )
            record_sha = _canonical_sha256(dict(relative_regions[region_id]))
            if row.get("source_region_record_sha256") != record_sha:
                raise ValueError("v3.4 source routing overlay region SHA differs")
            semantic = _source_semantic_decision(relative_regions[region_id])
            if any(row.get(field) != value for field, value in semantic.items()):
                raise ValueError("v3.4 source routing sealed semantic differs")
            artifacts = row.get("artifacts")
            if not isinstance(artifacts, Mapping) or set(artifacts) != set(
                SOURCE_ROUTING_MASK_ROLES
            ):
                raise ValueError("v3.4 source routing overlay role inventory differs")
            for role in SOURCE_ROUTING_MASK_ROLES:
                descriptor = artifacts[role]
                if not isinstance(descriptor, Mapping):
                    raise ValueError("v3.4 source routing descriptor is invalid")
                _require_exact_keys(
                    descriptor,
                    {"path", "file_sha256", "pixel_sha256", "shape"},
                    label="source routing overlay artifact",
                )
                shape = descriptor.get("shape")
                if (
                    not Path(str(descriptor.get("path") or "")).is_absolute()
                    or not _is_sha256(descriptor.get("file_sha256"))
                    or not _is_sha256(descriptor.get("pixel_sha256"))
                    or not isinstance(shape, list)
                    or len(shape) != 2
                    or any(
                        not isinstance(value, int) or value <= 0 for value in shape
                    )
                ):
                    raise ValueError("v3.4 source routing artifact binding is invalid")
                artifact_count += 1
            records[(page_id, region_id)] = SourceRoutingOverlayRegionRecord(
                page_id=page_id,
                source_sha256=source_sha,
                region_id=region_id,
                source_region_record_sha256=record_sha,
                semantic_role=str(row["semantic_role"]),
                semantic_action=str(row["semantic_action"]),
                semantic_available=bool(row["semantic_available"]),
                semantic_reason=str(row["semantic_reason"]),
                semantic_provenance=str(row["semantic_provenance"]),
                artifacts=MappingProxyType(
                    {role: MappingProxyType(dict(artifacts[role])) for role in SOURCE_ROUTING_MASK_ROLES}
                ),
            )
    summary = payload.get("summary")
    if not isinstance(summary, Mapping) or dict(summary) != {
        "page_count": len(source_pages),
        "region_count": len(records),
        "artifact_count": artifact_count,
    }:
        raise ValueError("v3.4 source routing overlay summary differs")
    return {
        "overlay_sha256": payload["overlay_sha256"],
        "artifact_sha256": _sha256(overlay_path),
        "seal_sha256": _sha256(seal_path),
        "source_evidence_file_sha256": payload["source_evidence_file_sha256"],
        "source_evidence_payload_sha256": payload[
            "source_evidence_payload_sha256"
        ],
        "records": records,
    }


def validate_seedless_ocr_overlay(
    overlay_path: Path | None,
    *,
    source_manifest_path: Path | None = None,
    source_binding: Mapping[str, object],
    source_manifest_payload: Mapping[str, object],
) -> dict[str, object]:
    """Authenticate optional OCR evidence without consulting eval annotations."""

    if overlay_path is None:
        return {
            "available": False,
            "status": "overlay_missing",
            "overlay_sha256": None,
            "artifact_sha256": None,
            "seal_sha256": None,
            "records": {},
        }
    payload = _read_json(overlay_path)
    seal_path = overlay_path.with_suffix(overlay_path.suffix + ".seal.json")
    seal = _read_json(seal_path)
    overlay_keys = {
        "schema_version",
        "source_manifest_sha256",
        "source_manifest_file_sha256",
        "source_manifest_seal_sha256",
        "source_page_inventory_sha256",
        "source_evidence_schema_version",
        "source_evidence_path",
        "source_evidence_file_sha256",
        "source_evidence_receipt_path",
        "source_evidence_receipt_file_sha256",
        "source_evidence_receipt_payload_sha256",
        "source_evidence_seal_path",
        "source_evidence_seal_file_sha256",
        "source_evidence_provider_identity_sha256",
        "source_evidence_tracked_dependency_identity_sha256",
        "source_evidence_region_inventory_sha256",
        "source_evidence_record_inventory_sha256",
        "source_evidence_producer",
        "candidate_generated",
        "candidate_seen",
        "annotation_frozen_before_candidate",
        "summary",
        "pages",
        "overlay_sha256",
    }
    seal_keys = {
        "schema_version",
        "overlay_file_sha256",
        "overlay_sha256",
        "source_manifest_sha256",
        "source_manifest_file_sha256",
        "source_manifest_seal_sha256",
        "source_page_inventory_sha256",
        "source_evidence_path",
        "source_evidence_file_sha256",
        "source_evidence_receipt_path",
        "source_evidence_receipt_file_sha256",
        "source_evidence_receipt_payload_sha256",
        "source_evidence_seal_path",
        "source_evidence_seal_file_sha256",
        "source_evidence_provider_identity_sha256",
        "source_evidence_tracked_dependency_identity_sha256",
        "source_evidence_region_inventory_sha256",
        "source_evidence_record_inventory_sha256",
        "candidate_generated",
        "candidate_seen",
        "annotation_frozen_before_candidate",
        "seal_payload_sha256",
    }
    _require_exact_keys(payload, overlay_keys, label="seedless OCR overlay")
    _require_exact_keys(seal, seal_keys, label="seedless OCR overlay seal")
    if seal.get("schema_version") != SEEDLESS_OVERLAY_SEAL_SCHEMA_VERSION:
        raise ValueError("v3.4 seedless OCR overlay seal schema differs")
    if (
        seal.get("overlay_file_sha256") != _sha256(overlay_path)
        or seal.get("overlay_sha256") != payload.get("overlay_sha256")
        or seal.get("seal_payload_sha256")
        != _canonical_without_field(seal, "seal_payload_sha256")
    ):
        raise ValueError("v3.4 seedless OCR overlay seal binding differs")
    if (
        seal.get("candidate_generated") is not False
        or seal.get("candidate_seen") is not False
        or seal.get("annotation_frozen_before_candidate") is not True
    ):
        raise ValueError("v3.4 seedless OCR overlay seal is not source-only")
    if payload.get("schema_version") != SEEDLESS_OVERLAY_SCHEMA_VERSION:
        raise ValueError("v3.4 seedless OCR overlay schema differs")
    if payload.get("source_manifest_sha256") != source_binding.get("manifest_sha256"):
        raise ValueError("v3.4 seedless OCR overlay source manifest differs")
    if payload.get("source_page_inventory_sha256") != source_binding.get(
        "page_inventory_sha256"
    ):
        raise ValueError("v3.4 seedless OCR overlay page inventory differs")
    if payload.get("candidate_generated") is not False:
        raise ValueError("v3.4 seedless OCR overlay was candidate-derived")
    if payload.get("candidate_seen") is not False:
        raise ValueError("v3.4 seedless OCR overlay was candidate-derived")
    if payload.get("annotation_frozen_before_candidate") is not True:
        raise ValueError("v3.4 seedless OCR overlay was not frozen")
    if payload.get("overlay_sha256") != _unsigned_overlay_sha256(payload):
        raise ValueError("v3.4 seedless OCR overlay canonical SHA differs")
    if source_manifest_path is None:
        raise ValueError("v3.4 seedless OCR overlay needs source manifest path")
    manifest_seal_path = source_manifest_path.with_suffix(
        source_manifest_path.suffix + ".seal.json"
    )
    if (
        payload.get("source_manifest_file_sha256") != _sha256(source_manifest_path)
        or payload.get("source_manifest_seal_sha256")
        != _sha256(manifest_seal_path)
    ):
        raise ValueError("v3.4 seedless OCR source manifest file binding differs")
    for field in (
        "source_manifest_sha256",
        "source_manifest_file_sha256",
        "source_manifest_seal_sha256",
        "source_page_inventory_sha256",
        "source_evidence_path",
        "source_evidence_file_sha256",
        "source_evidence_receipt_path",
        "source_evidence_receipt_file_sha256",
        "source_evidence_receipt_payload_sha256",
        "source_evidence_seal_path",
        "source_evidence_seal_file_sha256",
        "source_evidence_provider_identity_sha256",
        "source_evidence_tracked_dependency_identity_sha256",
        "source_evidence_region_inventory_sha256",
        "source_evidence_record_inventory_sha256",
    ):
        if seal.get(field) != payload.get(field):
            raise ValueError(f"v3.4 seedless OCR seal {field} differs")

    evidence_path = Path(str(payload.get("source_evidence_path") or ""))
    receipt_path = Path(str(payload.get("source_evidence_receipt_path") or ""))
    evidence_seal_path = Path(str(payload.get("source_evidence_seal_path") or ""))
    if not all(path.is_absolute() for path in (evidence_path, receipt_path, evidence_seal_path)):
        raise ValueError("v3.4 seedless OCR source evidence path is invalid")
    evidence = _read_json(evidence_path)
    receipt = _read_json(receipt_path)
    evidence_seal = _read_json(evidence_seal_path)
    if (
        payload.get("source_evidence_schema_version")
        != "inpaint-source-ocr-block-evidence-v34"
        or evidence.get("schema_version")
        != "inpaint-source-ocr-block-evidence-v34"
        or payload.get("source_evidence_file_sha256") != _sha256(evidence_path)
        or payload.get("source_evidence_receipt_file_sha256")
        != _sha256(receipt_path)
        or payload.get("source_evidence_receipt_payload_sha256")
        != _canonical_sha256(receipt)
        or payload.get("source_evidence_seal_file_sha256")
        != _sha256(evidence_seal_path)
        or evidence_seal.get("schema_version")
        != "inpaint-source-ocr-block-evidence-seal-v34"
        or receipt.get("schema_version")
        != "inpaint-source-ocr-runtime-receipt-v34"
        or evidence_seal.get("evidence_file_sha256") != _sha256(evidence_path)
        or evidence_seal.get("receipt_file_sha256") != _sha256(receipt_path)
        or evidence_seal.get("evidence_payload_sha256")
        != _canonical_sha256(evidence)
        or evidence_seal.get("receipt_payload_sha256")
        != _canonical_sha256(receipt)
        or receipt.get("evidence_file_sha256") != _sha256(evidence_path)
        or receipt.get("evidence_payload_sha256") != _canonical_sha256(evidence)
        or evidence_seal.get("candidate_generated") is not False
        or evidence_seal.get("candidate_seen") is not False
        or evidence_seal.get("annotation_frozen_before_candidate") is not True
        or receipt.get("candidate_seen") is not False
        or receipt.get("annotation_frozen_before_candidate") is not True
    ):
        raise ValueError("v3.4 seedless OCR source evidence chain differs")
    for overlay_field, source_field in (
        ("source_evidence_provider_identity_sha256", "provider_identity_sha256"),
        (
            "source_evidence_tracked_dependency_identity_sha256",
            "tracked_dependency_identity_sha256",
        ),
        ("source_evidence_region_inventory_sha256", "source_region_inventory_sha256"),
        ("source_evidence_record_inventory_sha256", "record_inventory_sha256"),
    ):
        if (
            payload.get(overlay_field) != receipt.get(source_field)
            or payload.get(overlay_field) != evidence_seal.get(source_field)
        ):
            raise ValueError(f"v3.4 seedless OCR {overlay_field} differs")
    for field, value in (
        ("detector_identity_sha256", receipt.get("detector_identity")),
        ("ocr_identity_sha256", receipt.get("ocr_identity")),
        (
            "tracked_dependency_identity_sha256",
            receipt.get("tracked_dependency_identity"),
        ),
        ("source_region_inventory_sha256", receipt.get("source_region_inventory")),
        ("record_inventory_sha256", receipt.get("record_inventory")),
    ):
        if not isinstance(value, (Mapping, list)) or receipt.get(field) != _canonical_sha256(value):
            raise ValueError(f"v3.4 seedless OCR receipt {field} differs")
    detector_identity = receipt["detector_identity"]
    ocr_identity = receipt["ocr_identity"]
    tracked_identity = receipt["tracked_dependency_identity"]
    assert isinstance(detector_identity, Mapping)
    assert isinstance(ocr_identity, Mapping)
    assert isinstance(tracked_identity, Mapping)
    provider = str(receipt.get("provider") or "")
    provider_identity_sha = _canonical_sha256(
        {
            "provider": provider.split(":", 1)[0],
            "detector": dict(detector_identity),
            "ocr": dict(ocr_identity),
            "code": dict(tracked_identity),
        }
    )
    dependencies = tracked_identity.get("dependencies")
    expected_dependencies = set(
        seedless_ocr_contract.TRACKED_RUNTIME_DEPENDENCIES
    )
    actual_dependencies = {
        str(row.get("path") or "")
        for row in dependencies
        if isinstance(row, Mapping)
    } if isinstance(dependencies, list) else set()
    current_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if (
        receipt.get("provider_identity_sha256") != provider_identity_sha
        or payload.get("source_evidence_provider_identity_sha256")
        != provider_identity_sha
        or not provider.endswith(f":{provider_identity_sha}")
        or tracked_identity.get("git_head") != current_head
        or receipt.get("git_head") != current_head
        or tracked_identity.get("tracked_worktree_clean") is not True
        or not isinstance(dependencies, list)
        or tracked_identity.get("dependency_count") != len(dependencies)
        or tracked_identity.get("dependency_inventory_sha256")
        != _canonical_sha256(dependencies)
        or len(dependencies) != len(expected_dependencies)
        or actual_dependencies != expected_dependencies
    ):
        raise ValueError("v3.4 seedless OCR provider/dependency binding differs")
    for dependency in dependencies:
        if not isinstance(dependency, Mapping):
            raise ValueError("v3.4 seedless OCR dependency row is invalid")
        relative = str(dependency.get("path") or "")
        path = (ROOT / relative).resolve()
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        head_blob = subprocess.run(
            ["git", "rev-parse", f"HEAD:{relative}"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if (
            not relative
            or ROOT.resolve() not in path.parents
            or tracked.returncode != 0
            or head_blob.returncode != 0
            or dependency.get("tracked") is not True
            or dependency.get("unchanged_from_head") is not True
            or dependency.get("head_blob_id") != head_blob.stdout.strip()
            or dependency.get("working_file_sha256") != _sha256(path)
        ):
            raise ValueError("v3.4 seedless OCR dependency changed after sealing")
    if (
        evidence.get("producer") != payload.get("source_evidence_producer")
        or receipt.get("provider") != payload.get("source_evidence_producer")
    ):
        raise ValueError("v3.4 seedless OCR producer identity differs")
    source_pages = source_manifest_payload.get("pages")
    if not isinstance(source_pages, list):
        raise ValueError("v3.4 source manifest lacks pages for OCR overlay")
    page_index = {
        str(page.get("page_id") or ""): page
        for page in source_pages
        if isinstance(page, Mapping)
    }
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list):
        raise ValueError("v3.4 seedless OCR overlay pages must be an array")
    records: dict[tuple[str, str], SeedlessOcrRegionRecord] = {}
    source_evidence_records: dict[tuple[str, str], Mapping[str, object]] = {}
    for source_page in evidence.get("pages", []):
        if not isinstance(source_page, Mapping):
            raise ValueError("v3.4 seedless OCR source evidence page is invalid")
        source_page_id = str(source_page.get("page_id") or "")
        for source_record in source_page.get("regions", []):
            if not isinstance(source_record, Mapping):
                raise ValueError("v3.4 seedless OCR source evidence region is invalid")
            identity = (source_page_id, str(source_record.get("region_id") or ""))
            if not all(identity) or identity in source_evidence_records:
                raise ValueError("v3.4 seedless OCR source evidence identity differs")
            source_evidence_records[identity] = source_record
    seen_pages: set[str] = set()
    for raw_page in raw_pages:
        if not isinstance(raw_page, Mapping):
            raise ValueError("v3.4 seedless OCR overlay page is invalid")
        page_id = str(raw_page.get("page_id") or "").strip()
        if not page_id or page_id in seen_pages or page_id not in page_index:
            raise ValueError("v3.4 seedless OCR overlay page identity differs")
        seen_pages.add(page_id)
        source_sha = str(raw_page.get("source_sha256") or "").lower()
        if source_sha != str(page_index[page_id].get("source_sha256") or "").lower():
            raise ValueError("v3.4 seedless OCR overlay source SHA differs")
        known_regions = {
            str(region.get("region_id") or "")
            for region in page_index[page_id].get("regions", [])
            if isinstance(region, Mapping)
        }
        raw_regions = raw_page.get("regions")
        if not isinstance(raw_regions, list):
            raise ValueError("v3.4 seedless OCR overlay regions must be an array")
        for raw in raw_regions:
            if not isinstance(raw, Mapping):
                raise ValueError("v3.4 seedless OCR overlay region is invalid")
            _require_exact_keys(
                raw,
                {
                    "region_id",
                    "owner_region_id",
                    "owner_count",
                    "provider",
                    "ocr_text",
                    "ocr_script",
                    "ocr_confidence",
                    "processing_action",
                    "route_class",
                    "authoritative_ocr",
                    "candidate_seen",
                    "annotation_frozen_before_candidate",
                    "confidence_kind",
                    "owner_binding_kind",
                    "canonical_block_id",
                    "source_record_sha256",
                    "source_provenance_sha256",
                },
                label="seedless OCR overlay region",
            )
            region_id = str(raw.get("region_id") or "").strip()
            identity = (page_id, region_id)
            if not region_id or region_id not in known_regions or identity in records:
                raise ValueError("v3.4 seedless OCR overlay region identity differs")
            owner_region_id = str(raw.get("owner_region_id") or "").strip()
            owner_count = raw.get("owner_count")
            provider = str(raw.get("provider") or "").strip()
            text = str(raw.get("ocr_text") or "").strip()
            script = str(raw.get("ocr_script") or "").strip()
            confidence = raw.get("ocr_confidence")
            action = str(raw.get("processing_action") or "").strip().lower()
            route_class = str(raw.get("route_class") or "").strip().lower()
            confidence_kind = str(raw.get("confidence_kind") or "").strip().lower()
            owner_binding_kind = str(raw.get("owner_binding_kind") or "").strip().lower()
            canonical_block_id = str(raw.get("canonical_block_id") or "").strip()
            source_record = source_evidence_records.get(identity)
            provenance = (
                source_record.get("provenance")
                if isinstance(source_record, Mapping)
                else None
            )
            valid_confidence = (
                isinstance(confidence, (int, float))
                and not isinstance(confidence, bool)
                and math.isfinite(float(confidence))
                and 0.0 <= float(confidence) <= 1.0
            )
            if (
                raw.get("candidate_seen") is not False
                or raw.get("annotation_frozen_before_candidate") is not True
                or raw.get("authoritative_ocr") is not True
                or owner_region_id != region_id
                or owner_count != 1
                or not provider
                or not text
                or not script
                or not valid_confidence
                or action != TRANSLATE
                or route_class != "clean_translucent"
                or confidence_kind
                not in {
                    "detector_block_confidence",
                    "recognizer_confidence",
                    "calibrated_ocr_confidence",
                }
                or owner_binding_kind
                not in {"canonical_block_region_id", "exact_product_region_id"}
                or not canonical_block_id
                or not isinstance(source_record, Mapping)
                or not isinstance(provenance, Mapping)
                or raw.get("source_record_sha256")
                != _canonical_sha256(dict(source_record))
                or raw.get("source_provenance_sha256")
                != _canonical_sha256(dict(sorted(provenance.items())))
                or any(
                    not _is_sha256(provenance.get(field))
                    for field in (
                        "ocr_artifact_sha256",
                        "confidence_artifact_sha256",
                        "owner_binding_sha256",
                        "action_artifact_sha256",
                        "route_artifact_sha256",
                    )
                )
                or any(
                    raw.get(field) != source_record.get(field)
                    for field in (
                        "owner_region_id",
                        "owner_count",
                        "provider",
                        "ocr_text",
                        "ocr_script",
                        "ocr_confidence",
                        "processing_action",
                        "route_class",
                        "authoritative_ocr",
                        "confidence_kind",
                        "owner_binding_kind",
                        "canonical_block_id",
                    )
                )
            ):
                raise ValueError("v3.4 seedless OCR region evidence is incomplete")
            records[identity] = SeedlessOcrRegionRecord(
                page_id=page_id,
                source_sha256=source_sha,
                region_id=region_id,
                owner_region_id=owner_region_id,
                owner_count=1,
                provider=provider,
                ocr_text=text,
                ocr_script=script,
                ocr_confidence=float(confidence),
                confidence_kind=confidence_kind,
                owner_binding_kind=owner_binding_kind,
                canonical_block_id=canonical_block_id,
                source_record_sha256=str(raw["source_record_sha256"]),
                source_provenance_sha256=str(raw["source_provenance_sha256"]),
                processing_action=action,
                route_class=route_class,
            )
    return {
        "available": True,
        "status": "validated",
        "overlay_sha256": payload["overlay_sha256"],
        "artifact_sha256": _sha256(overlay_path),
        "seal_sha256": _sha256(seal_path),
        "source_evidence_file_sha256": payload[
            "source_evidence_file_sha256"
        ],
        "source_evidence_receipt_file_sha256": payload[
            "source_evidence_receipt_file_sha256"
        ],
        "source_evidence_seal_file_sha256": payload[
            "source_evidence_seal_file_sha256"
        ],
        "records": records,
    }


def _read_image(path: str | Path, flags: int) -> np.ndarray:
    value = cv2.imdecode(np.fromfile(Path(path), dtype=np.uint8), flags)
    if value is None or value.size == 0:
        raise FileNotFoundError(path)
    return value


def _read_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    value = _read_image(path, cv2.IMREAD_GRAYSCALE)
    if value.shape != shape:
        raise ValueError(f"v3.4 mask shape mismatch: {value.shape} != {shape}")
    values = np.unique(value)
    if np.any((values != 0) & (values != 255)):
        raise ValueError(f"v3.4 input mask is not binary: {path}")
    return binary_mask(value, shape)


def _safe_page_name(page_id: str) -> str:
    normalized = str(page_id).strip()
    if (
        not normalized
        or normalized in {".", ".."}
        or "/" in normalized
        or "\\" in normalized
    ):
        raise ValueError("v3.4 page id is not path-safe")
    return normalized


def _write_mask(path: Path, mask: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"v3.4 output mask must be fresh: {path}")
    normalized = binary_mask(mask)
    encoded, buffer = cv2.imencode(".png", normalized)
    if not encoded:
        raise RuntimeError("failed to encode v3.4 output mask")
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_bytes(buffer.tobytes())
    temporary.replace(path)
    decoded = _read_mask(path, normalized.shape)
    if not np.array_equal(decoded, normalized):
        raise RuntimeError("v3.4 output mask changed during encoding")
    return {
        "file_sha256": _sha256(path),
        "pixel_sha256": mask_sha256(decoded),
        "pixel_count": int(np.count_nonzero(decoded)),
        "size_bytes": path.stat().st_size,
    }


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _stage1_input(
    run_root: Path,
    *,
    expected_variant: str,
    expected_source_manifest_sha256: str,
    page_ids: tuple[str, ...],
) -> dict[str, object]:
    result_path = run_root / "stage1-results.json"
    payload = _read_json(result_path)
    _validate_stage1_output_artifacts(payload, result_path)
    if payload.get("manifest_sha256") != expected_source_manifest_sha256:
        raise ValueError("v3.4 detector run uses a different source manifest")
    if str(payload.get("variant") or "") != expected_variant:
        raise ValueError("v3.4 detector run variant differs")
    summary = payload.get("summary")
    if not isinstance(summary, Mapping) or summary.get("page_count") != len(page_ids):
        raise ValueError("v3.4 detector run page count differs")
    variant_dir = "dilated" if expected_variant == "dilated" else "raw"
    mask_root = run_root / "native_masks" / variant_dir
    files = {path.stem: path.resolve() for path in mask_root.glob("*.png")}
    if set(files) != set(page_ids):
        raise ValueError("v3.4 detector mask inventory differs from source manifest")
    role_candidate = payload.get("role_candidate")
    model = payload.get("model")
    variant_output_identity = payload.get("variant_output_identity")
    if not all(
        isinstance(value, Mapping)
        for value in (role_candidate, model, variant_output_identity)
    ):
        raise ValueError("v3.4 detector inference bundle evidence is missing")
    return {
        "run_root": str(run_root.resolve()),
        "result_path": str(result_path.resolve()),
        "result_sha256": _sha256(result_path),
        "candidate": str(payload.get("candidate") or ""),
        "variant": expected_variant,
        "mask_root": str(mask_root.resolve()),
        "masks": files,
        "role_candidate": dict(role_candidate),
        "model": dict(model),
        "variant_output_identity": dict(variant_output_identity),
    }


def bind_detector_inference_bundle(
    raw: Mapping[str, object],
    dilated: Mapping[str, object],
    expectation: DetectorBundleExpectation,
) -> dict[str, object]:
    """Require raw/native3 to describe one exact, preselected inference bundle."""

    raw_role = raw.get("role_candidate")
    dilated_role = dilated.get("role_candidate")
    raw_identity = raw.get("variant_output_identity")
    dilated_identity = dilated.get("variant_output_identity")
    raw_model = raw.get("model")
    dilated_model = dilated.get("model")
    if not all(
        isinstance(value, Mapping)
        for value in (
            raw_role,
            dilated_role,
            raw_identity,
            dilated_identity,
            raw_model,
            dilated_model,
        )
    ):
        raise ValueError("v3.4 detector bundle evidence is incomplete")
    if dict(raw_role) != dict(dilated_role):
        raise ValueError("v3.4 raw/native3 role candidate bundle differs")
    if dict(raw_identity) != dict(dilated_identity):
        raise ValueError("v3.4 raw/native3 output identity bundle differs")
    if dict(raw_model) != dict(dilated_model):
        raise ValueError("v3.4 raw/native3 model bundle differs")
    expected_role = {
        "candidate_id": expectation.candidate_id,
        "model_sha256": expectation.role_model_sha256,
        "preprocessing_contract_sha256": expectation.preprocessing_contract_sha256,
    }
    for field, expected in expected_role.items():
        if raw_role.get(field) != expected:
            raise ValueError(f"v3.4 detector bundle expected {field} differs")
    if raw.get("candidate") != expectation.candidate_id or dilated.get(
        "candidate"
    ) != expectation.candidate_id:
        raise ValueError("v3.4 detector bundle candidate differs")
    if raw_model.get("sha256") != expectation.model_asset_sha256:
        raise ValueError("v3.4 detector bundle model asset SHA differs")
    # Output-set SHA includes the exact page inventory. It must be verified
    # against each Stage 1 artifact inventory, not pinned to an older E1-only
    # page set when the source-only corpus grows to 130 pages.
    output_hashes: dict[str, str] = {}
    for variant in ("raw", "dilated"):
        identity = raw_identity.get(variant)
        output_sha = (
            identity.get("output_mask_set_sha256")
            if isinstance(identity, Mapping)
            else None
        )
        if not _is_sha256(output_sha):
            raise ValueError(
                f"v3.4 detector bundle {variant} output set is invalid"
            )
        output_hashes[variant] = str(output_sha)
    binding = {
        "bundle_id": expectation.bundle_id,
        "candidate_id": expectation.candidate_id,
        "role_model_sha256": expectation.role_model_sha256,
        "model_asset_sha256": expectation.model_asset_sha256,
        "preprocessing_contract_sha256": expectation.preprocessing_contract_sha256,
        "raw_output_mask_set_sha256": output_hashes["raw"],
        "dilated_output_mask_set_sha256": output_hashes["dilated"],
        "role_candidate_sha256": _canonical_sha256(dict(raw_role)),
        "variant_output_identity_sha256": _canonical_sha256(dict(raw_identity)),
        "raw_stage1_result_sha256": raw.get("result_sha256"),
        "dilated_stage1_result_sha256": dilated.get("result_sha256"),
    }
    binding["inference_bundle_sha256"] = _canonical_sha256(binding)
    return binding


def load_source_region_evidence(
    raw_region: Mapping[str, object],
    overlay_region: SourceRoutingOverlayRegionRecord,
    *,
    shape: tuple[int, int],
) -> SourceRegionEvidence:
    """Reopen only exact source-routing artifacts named by the sealed overlay."""

    region_id = str(raw_region.get("region_id") or "").strip()
    if (
        not region_id
        or overlay_region.region_id != region_id
        or overlay_region.source_region_record_sha256
        != _canonical_sha256(dict(raw_region))
    ):
        raise ValueError("v3.4 source routing overlay region identity differs")
    artifacts: list[dict[str, object]] = []
    masks: dict[str, np.ndarray] = {}
    common_shape: tuple[int, int] | None = None
    for role in SOURCE_ROUTING_MASK_ROLES:
        descriptor = overlay_region.artifacts.get(role)
        if not isinstance(descriptor, Mapping):
            raise ValueError("v3.4 source routing artifact role is missing")
        path = Path(str(descriptor.get("path") or ""))
        if not path.is_absolute():
            raise ValueError("v3.4 source routing artifact path is not absolute")
        value = _read_mask(path, shape)
        actual_shape = tuple(value.shape)
        if common_shape is None:
            common_shape = actual_shape
        elif actual_shape != common_shape:
            raise ValueError("v3.4 source routing artifact shape differs")
        if (
            descriptor.get("shape") != [int(item) for item in actual_shape]
            or descriptor.get("file_sha256") != _sha256(path)
            or descriptor.get("pixel_sha256") != mask_sha256(value)
        ):
            raise ValueError("v3.4 source routing artifact changed after sealing")
        artifacts.append(
            {
                "region_id": region_id,
                "role": role,
                "path": str(path),
                "file_sha256": descriptor["file_sha256"],
                "pixel_sha256": descriptor["pixel_sha256"],
                "shape": descriptor["shape"],
            }
        )
        masks[role] = value
    return SourceRegionEvidence(
        region_id=region_id,
        ownership=np.ascontiguousarray(masks["ownership"]),
        protected=np.ascontiguousarray(masks["protected"]),
        ambiguous=np.ascontiguousarray(masks["ambiguous"]),
        corner=np.ascontiguousarray(masks["corner"]),
        available=True,
        failure_reason="",
        artifacts=tuple(artifacts),
    )


def build_source_routing_evidence(
    raw_regions: Sequence[Mapping[str, object]],
    source_regions: Sequence[SourceRegionEvidence],
    overlay_regions: Sequence[SourceRoutingOverlayRegionRecord],
    *,
    shape: tuple[int, int],
) -> SourceRoutingEvidence:
    """Build routing only from source-reviewed region evidence.

    Page-level target/protected/ambiguous/preserve evaluation masks are absent
    from this interface on purpose.  Source-reviewed region protection and
    corner masks, exact region ownership, and runtime semantic actions are the
    only inputs allowed to influence candidate pixels.
    """

    if (
        len(raw_regions) != len(source_regions)
        or len(raw_regions) != len(overlay_regions)
    ):
        raise ValueError("v3.4 source region inventory differs")
    ownership_count = np.zeros(shape, dtype=np.uint16)
    translate_regions: list[np.ndarray] = []
    semantic_guard = np.zeros(shape, dtype=np.uint8)
    region_protect = np.zeros(shape, dtype=np.uint8)
    region_ambiguous = np.zeros(shape, dtype=np.uint8)
    corner_protect = np.zeros(shape, dtype=np.uint8)
    decisions: list[dict[str, object]] = []
    artifacts: list[dict[str, object]] = []
    ids: set[str] = set()
    translate_owner_rows: list[tuple[str, np.ndarray]] = []
    for raw, masks, sealed in zip(raw_regions, source_regions, overlay_regions):
        region_id = str(raw.get("region_id") or "").strip()
        if (
            not region_id
            or region_id != masks.region_id
            or region_id != sealed.region_id
            or region_id in ids
        ):
            raise ValueError("v3.4 source region id/order differs")
        ids.add(region_id)
        ownership = binary_mask(masks.ownership, shape)
        ownership_count += (ownership > 0).astype(np.uint16)
        expected_semantic = _source_semantic_decision(raw)
        if any(
            getattr(sealed, field) != value
            for field, value in expected_semantic.items()
        ):
            raise ValueError("v3.4 source routing semantic seal differs")
        if (
            masks.available
            and sealed.semantic_available
            and sealed.semantic_action == TRANSLATE
        ):
            translate_regions.append(ownership)
            translate_owner_rows.append((region_id, ownership))
        else:
            semantic_guard[ownership > 0] = 255
        region_protect[masks.protected > 0] = 255
        region_ambiguous[masks.ambiguous > 0] = 255
        corner_protect[masks.corner > 0] = 255
        artifacts.extend(masks.artifacts)
        decisions.append(
            {
                "region_id": region_id,
                "semantic_role": sealed.semantic_role,
                "semantic_action": sealed.semantic_action,
                "semantic_available": sealed.semantic_available and masks.available,
                "semantic_reason": (
                    masks.failure_reason
                    if not masks.available
                    else sealed.semantic_reason
                ),
                "semantic_provenance": sealed.semantic_provenance,
                "source_artifacts_available": masks.available,
            }
        )
    conflict = np.where(ownership_count > 1, 255, 0).astype(np.uint8)
    authoritative = np.zeros(shape, dtype=np.uint8)
    for ownership in translate_regions:
        authoritative[(ownership > 0) & (ownership_count == 1)] = 255
    # Distinct authoritative owners can touch without overlapping. The glyph
    # extractor labels connected ownership pixels, so keep a one-pixel veto on
    # both sides of any eight-neighbour owner boundary before extraction.
    if len(translate_owner_rows) >= np.iinfo(np.uint16).max:
        raise ValueError("v3.4 source owner inventory exceeds label capacity")
    owner_labels = np.zeros(shape, dtype=np.uint16)
    for index, (_region_id, ownership) in enumerate(translate_owner_rows, 1):
        owner_labels[(ownership > 0) & (ownership_count == 1)] = index
    if len(translate_owner_rows) > 1:
        kernel = np.ones((3, 3), dtype=np.uint8)
        neighbor_max = cv2.dilate(owner_labels, kernel)
        positive_or_max = np.where(
            owner_labels > 0, owner_labels, np.iinfo(np.uint16).max
        ).astype(np.uint16)
        neighbor_min = cv2.erode(positive_or_max, kernel)
        owner_contact = (owner_labels > 0) & (neighbor_min != neighbor_max)
    else:
        owner_contact = np.zeros(shape, dtype=bool)
    hard = np.where(
        (region_protect > 0)
        | (region_ambiguous > 0)
        | (corner_protect > 0)
        | (semantic_guard > 0)
        | (conflict > 0)
        | owner_contact,
        255,
        0,
    ).astype(np.uint8)
    authoritative[hard > 0] = 0
    exact_owner_masks: list[tuple[str, np.ndarray]] = []
    for region_id, ownership in translate_owner_rows:
        exact = np.where(
            (ownership > 0) & (authoritative > 0), 255, 0
        ).astype(np.uint8)
        if np.any(exact):
            exact_owner_masks.append((region_id, np.ascontiguousarray(exact)))
    return SourceRoutingEvidence(
        authoritative_translate_ownership=np.ascontiguousarray(authoritative),
        authoritative_translate_owner_masks=tuple(exact_owner_masks),
        all_ownership=np.where(ownership_count > 0, 255, 0).astype(np.uint8),
        preserve_or_abstain=np.ascontiguousarray(semantic_guard),
        ownership_conflict=np.ascontiguousarray(conflict),
        source_region_protect=np.ascontiguousarray(region_protect),
        source_region_ambiguous=np.ascontiguousarray(region_ambiguous),
        source_corner_protect=np.ascontiguousarray(corner_protect),
        hard_protect=np.ascontiguousarray(hard),
        semantic_actions=tuple(decisions),
        artifact_inventory=tuple(artifacts),
    )


def connected_existing_source_edit(
    existing_edit: np.ndarray,
    anchor: np.ndarray,
) -> np.ndarray:
    """Select only PR6 components overlapping or 8-neighbouring an anchor."""

    existing = binary_mask(existing_edit)
    seed = binary_mask(anchor, existing.shape)
    if not np.any(existing) or not np.any(seed):
        return np.zeros(existing.shape, dtype=np.uint8)
    count, labels = cv2.connectedComponents(
        (existing > 0).astype(np.uint8),
        connectivity=8,
    )
    if count <= 1:
        return np.zeros(existing.shape, dtype=np.uint8)
    adjacent = cv2.dilate(
        seed,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    selected_labels = np.unique(labels[adjacent > 0])
    selected_labels = selected_labels[selected_labels > 0]
    if selected_labels.size == 0:
        return np.zeros(existing.shape, dtype=np.uint8)
    return np.ascontiguousarray(
        np.where(np.isin(labels, selected_labels), 255, 0).astype(np.uint8)
    )


def connected_existing_source_edit_by_owner(
    existing_edit: np.ndarray,
    anchor: np.ndarray,
    *,
    authoritative_owner_masks: Sequence[tuple[str, np.ndarray]],
    all_ownership: np.ndarray,
) -> np.ndarray:
    """Select adjacent PR6 components only when wholly owned by one exact owner."""

    existing = binary_mask(existing_edit)
    seed = binary_mask(anchor, existing.shape)
    all_owned = binary_mask(all_ownership, existing.shape)
    if not np.any(existing) or not np.any(seed):
        return np.zeros(existing.shape, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (existing > 0).astype(np.uint8), connectivity=8
    )
    selected = np.zeros(existing.shape, dtype=np.uint8)
    if len(authoritative_owner_masks) >= np.iinfo(np.uint16).max:
        raise ValueError("v3.4 exact owner inventory exceeds label capacity")
    owner_labels = np.zeros(existing.shape, dtype=np.uint16)
    for owner_id, (_region_id, raw_owner) in enumerate(authoritative_owner_masks, 1):
        owner = binary_mask(raw_owner, existing.shape)
        if np.any((owner > 0) & (owner_labels > 0)):
            raise ValueError("v3.4 exact owner masks overlap")
        owner_labels[owner > 0] = owner_id
    owned_seed = np.where((seed > 0) & (owner_labels > 0), 255, 0).astype(np.uint8)
    if not np.any(owned_seed):
        return selected
    adjacent = cv2.dilate(owned_seed, np.ones((3, 3), dtype=np.uint8))
    height, width = existing.shape
    for label in np.unique(labels[adjacent > 0]):
        label_id = int(label)
        if label_id <= 0 or label_id >= count:
            continue
        x, y, box_width, box_height, _area = (int(v) for v in stats[label_id])
        component_slice = (slice(y, y + box_height), slice(x, x + box_width))
        component = labels[component_slice] == label_id
        anchor_owners = np.unique(
            owner_labels[component_slice][
                component & (adjacent[component_slice] > 0)
            ]
        )
        anchor_owners = anchor_owners[anchor_owners > 0]
        if anchor_owners.size != 1:
            continue
        owner_id = int(anchor_owners[0])
        if np.any(component & (owner_labels[component_slice] != owner_id)):
            continue
        x0, y0 = max(0, x - 1), max(0, y - 1)
        x1, y1 = min(width, x + box_width + 1), min(height, y + box_height + 1)
        boundary_slice = (slice(y0, y1), slice(x0, x1))
        padded = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        padded[y - y0 : y - y0 + box_height, x - x0 : x - x0 + box_width][component] = 255
        boundary = cv2.dilate(padded, np.ones((3, 3), dtype=np.uint8))
        other_owner = (all_owned[boundary_slice] > 0) & (
            owner_labels[boundary_slice] != owner_id
        )
        if np.any((boundary > 0) & other_owner):
            continue
        selected[component_slice][component] = 255
    return np.ascontiguousarray(selected)


_GLYPH_MASK_HASH_FIELDS = (
    ("owned_detector_seed", "owned_detector_seed_sha256"),
    ("effect_support", "effect_support_sha256"),
    ("candidate_mask", "candidate_mask_sha256"),
    ("glyph_core", "glyph_core_sha256"),
    ("glyph_effect", "glyph_effect_sha256"),
    ("refined_mask", "refined_mask_sha256"),
    ("rejected_mask", "rejected_mask_sha256"),
    ("hard_protect", "hard_protect_sha256"),
)


def validate_glyph_result_integrity(result: GlyphRefinementResult) -> None:
    shape = tuple(np.asarray(result.refined_mask).shape)
    if len(shape) != 2:
        raise ValueError("v3.4 glyph result is not a page mask")
    for field, hash_field in _GLYPH_MASK_HASH_FIELDS:
        value = np.asarray(getattr(result, field))
        if value.flags.writeable:
            raise ValueError(f"v3.4 glyph result {field} is not readonly")
        actual = mask_sha256(binary_mask(value, shape))
        if result.provenance.get(hash_field) != actual:
            raise ValueError(f"v3.4 glyph result {field} SHA differs")


def _readonly_binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    result = binary_mask(mask, shape).copy()
    result.setflags(write=False)
    return result


def merge_glyph_refinement_results(
    results: Sequence[GlyphRefinementResult],
    *,
    shape: tuple[int, int],
    hard_protect: np.ndarray,
) -> GlyphRefinementResult:
    """Union independently gated seeded and seedless ROI results."""

    protect = binary_mask(hard_protect, shape)
    zero = np.zeros(shape, dtype=np.uint8)
    seed = zero.copy()
    candidate = zero.copy()
    core = zero.copy()
    effect = zero.copy()
    support = zero.copy()
    rejected = zero.copy()
    records = []
    owner_records = []
    child_provenance: list[dict[str, object]] = []
    for result in results:
        validate_glyph_result_integrity(result)
        if result.refined_mask.shape != shape:
            raise ValueError("v3.4 glyph result shape differs")
        if not np.array_equal(binary_mask(result.hard_protect, shape), protect):
            raise ValueError("v3.4 glyph result hard protection differs")
        seed = cv2.bitwise_or(seed, binary_mask(result.owned_detector_seed, shape))
        candidate = cv2.bitwise_or(candidate, binary_mask(result.candidate_mask, shape))
        support = cv2.bitwise_or(support, binary_mask(result.effect_support, shape))
        core = cv2.bitwise_or(core, binary_mask(result.glyph_core, shape))
        effect = cv2.bitwise_or(effect, binary_mask(result.glyph_effect, shape))
        rejected = cv2.bitwise_or(rejected, binary_mask(result.rejected_mask, shape))
        records.extend(result.component_records)
        owner_records.extend(result.owner_records)
        if (
            str(result.provenance.get("seed_mode") or "")
            == "authoritative_ocr_seedless"
            and np.any(result.refined_mask)
        ):
            # A sealed OCR authority plus two intersecting source cues is the
            # positive seed for B3. It is not a detector pixel and remains
            # separately identified in child provenance.
            seed = cv2.bitwise_or(seed, binary_mask(result.glyph_core, shape))
        child_provenance.append(dict(result.provenance))
    core[protect > 0] = 0
    effect[(protect > 0) | (core > 0)] = 0
    refined = cv2.bitwise_or(core, effect)
    if np.any(refined) and not np.any(seed):
        raise ValueError("v3.4 glyph union expanded without a source-positive seed")
    output_masks = {
        "owned_detector_seed": _readonly_binary(seed, shape),
        "effect_support": _readonly_binary(support, shape),
        "candidate_mask": _readonly_binary(candidate, shape),
        "glyph_core": _readonly_binary(core, shape),
        "glyph_effect": _readonly_binary(effect, shape),
        "refined_mask": _readonly_binary(refined, shape),
        "rejected_mask": _readonly_binary(rejected, shape),
        "hard_protect": _readonly_binary(protect, shape),
    }
    hashes = {
        hash_field: mask_sha256(output_masks[field])
        for field, hash_field in _GLYPH_MASK_HASH_FIELDS
    }
    provenance = MappingProxyType(
        {
            "schema_version": "glyph-refinement-union-v34",
            "child_count": len(results),
            "child_provenance": child_provenance,
            "owned_positive_seed_pixel_count": int(np.count_nonzero(seed)),
            "glyph_core_pixel_count": int(np.count_nonzero(core)),
            "glyph_effect_pixel_count": int(np.count_nonzero(effect)),
            "refined_pixel_count": int(np.count_nonzero(refined)),
            "no_seed_no_expansion": not np.any(refined) or bool(np.any(seed)),
            **hashes,
        }
    )
    merged = GlyphRefinementResult(
        **output_masks,
        component_records=tuple(records),
        owner_records=tuple(owner_records),
        provenance=provenance,
    )
    validate_glyph_result_integrity(merged)
    return merged


def extract_page_seedless_results(
    source_image: np.ndarray,
    *,
    page_id: str,
    raw_regions: Sequence[Mapping[str, object]],
    source_regions: Sequence[SourceRegionEvidence],
    routing: SourceRoutingEvidence,
    evidence_records: Mapping[tuple[str, str], SeedlessOcrRegionRecord],
) -> tuple[tuple[GlyphRefinementResult, ...], dict[str, object], np.ndarray]:
    """Run only overlay-authorized B3 regions; missing evidence stays empty."""

    shape = source_image.shape[:2]
    results: list[GlyphRefinementResult] = []
    statuses: list[dict[str, object]] = []
    seedless_positive = np.zeros(shape, dtype=np.uint8)
    actions = {
        str(row["region_id"]): row for row in routing.semantic_actions
    }
    for raw, region in zip(raw_regions, source_regions):
        region_id = str(raw.get("region_id") or "")
        record = evidence_records.get((page_id, region_id))
        if record is None:
            statuses.append(
                {
                    "region_id": region_id,
                    "status": "information_limited",
                    "reason": "ocr_evidence_overlay_missing",
                }
            )
            continue
        action = actions.get(region_id, {})
        if (
            not region.available
            or action.get("semantic_available") is not True
            or action.get("semantic_action") != TRANSLATE
        ):
            statuses.append(
                {
                    "region_id": region_id,
                    "status": "rejected",
                    "reason": "runtime_region_not_authoritative_translate",
                }
            )
            continue
        ownership = np.where(
            (region.ownership > 0)
            & (routing.authoritative_translate_ownership > 0),
            255,
            0,
        ).astype(np.uint8)
        if not np.any(ownership):
            statuses.append(
                {
                    "region_id": region_id,
                    "status": "rejected",
                    "reason": "runtime_authoritative_ownership_empty",
                }
            )
            continue
        evidence = SeedlessRoiEvidence.seal(
            source_image,
            ocr_ownership=ownership,
            hard_protect=routing.hard_protect,
            owner_region_id=record.owner_region_id,
            owner_count=record.owner_count,
            authoritative_ocr=True,
            ocr_text=record.ocr_text,
            ocr_script=record.ocr_script,
            ocr_confidence=record.ocr_confidence,
            ocr_provider=record.provider,
            processing_action=record.processing_action,
            route_class=record.route_class,
        )
        result = extract_seedless_roi_glyph_masks(
            source_image,
            ocr_ownership=ownership,
            hard_protect=routing.hard_protect,
            evidence=evidence,
        )
        results.append(result)
        status = str(result.provenance.get("status") or "rejected")
        reason = str(result.provenance.get("reason") or "")
        if status == "completed" and np.any(result.refined_mask):
            seedless_positive = cv2.bitwise_or(
                seedless_positive,
                binary_mask(result.glyph_core, shape),
            )
            # OCR ownership is a region gate, not a spatial text cue. The
            # current ROI segmenter may therefore surface texture anywhere in
            # that owner. Preserve the diagnostic mask, but make it ineligible
            # for product shortlisting until an independent source-space cue is
            # sealed.
            status = "information_limited"
            reason = "independent_spatial_text_cue_missing"
        statuses.append(
            {
                "region_id": region_id,
                "status": status,
                "reason": reason,
                "spatial_text_cue_verified": False,
                "finalist_eligible": not np.any(result.refined_mask),
                "evidence_seal_sha256": evidence.seal_sha256,
                "refined_pixel_count": int(np.count_nonzero(result.refined_mask)),
            }
        )
    counts = Counter(str(row["status"]) for row in statuses)
    return (
        tuple(results),
        {
            "region_count": len(source_regions),
            "overlay_record_count": sum(
                (page_id, region.region_id) in evidence_records
                for region in source_regions
            ),
            "status_counts": dict(sorted(counts.items())),
            "regions": statuses,
            "information_limited": any(
                row["status"] == "information_limited" for row in statuses
            ),
            "finalist_eligible": not any(
                row.get("finalist_eligible") is False for row in statuses
            ),
            "seedless_positive_pixel_count": int(
                np.count_nonzero(seedless_positive)
            ),
        },
        np.ascontiguousarray(seedless_positive),
    )


def build_candidate_plan(
    spec: CandidateSpec,
    *,
    glyph: GlyphRefinementResult,
    baseline_mask: np.ndarray,
    authoritative_owner_masks: Sequence[tuple[str, np.ndarray]] = (),
    all_ownership: np.ndarray | None = None,
) -> PageIncrementalInpaintPlan:
    validate_glyph_result_integrity(glyph)
    shape = glyph.refined_mask.shape
    baseline = binary_mask(baseline_mask, shape)
    zero = np.zeros(shape, dtype=np.uint8)
    if spec.mode == "baseline":
        return union_incremental_plans((), shape=shape)
    selected = (
        binary_mask(glyph.glyph_core, shape)
        if spec.commit_source == "core"
        else binary_mask(glyph.refined_mask, shape)
    )
    if not np.any(selected):
        # A detector/PR2 seed can legitimately yield no source-safe glyph
        # component. It must not restore any inherited PR6 component merely
        # because that rejected seed touched the baseline.
        return union_incremental_plans((), shape=shape)
    if not authoritative_owner_masks or all_ownership is None:
        raise ValueError("v3.4 nonbaseline plan requires exact owner masks")
    if spec.mode == "context_additive":
        commit = np.where(
            (selected > 0) & (baseline == 0), 255, 0
        ).astype(np.uint8)
        context = connected_existing_source_edit_by_owner(
            baseline,
            commit,
            authoritative_owner_masks=authoritative_owner_masks,
            all_ownership=all_ownership,
        )
        generation = cv2.bitwise_or(commit, context)
        plan = IncrementalInpaintPlan(
            mode="context_additive",
            generation_mask=generation,
            commit_mask=commit,
            existing_source_edit=context,
        )
    elif spec.mode in {"narrow_replacement", "conditional_segmenter"}:
        commit = selected
        replacement = connected_existing_source_edit_by_owner(
            baseline,
            commit,
            authoritative_owner_masks=authoritative_owner_masks,
            all_ownership=all_ownership,
        )
        generation = cv2.bitwise_or(commit, replacement)
        plan = IncrementalInpaintPlan(
            mode=spec.mode,
            generation_mask=generation,
            commit_mask=commit,
            existing_source_edit=replacement,
        )
    else:
        raise ValueError(f"unsupported v3.4 candidate mode: {spec.mode}")
    page = union_incremental_plans((plan,), shape=shape)
    if page.lama_request_count > 1:
        raise AssertionError("v3.4 page plan created more than one fill request")
    if np.any((page.commit_mask > 0) & (glyph.hard_protect > 0)):
        raise AssertionError("v3.4 commit overlaps source-only hard protection")
    if np.any(page.commit_mask) and not np.any(glyph.owned_detector_seed):
        raise AssertionError("v3.4 expanded without an owned source seed")
    return page


def mask_only_final_mask(
    baseline_mask: np.ndarray,
    plan: PageIncrementalInpaintPlan,
) -> np.ndarray:
    baseline = binary_mask(baseline_mask, plan.shape)
    retained = np.where(
        (baseline > 0) & (plan.replacement_source_edit == 0),
        255,
        0,
    ).astype(np.uint8)
    return np.ascontiguousarray(cv2.bitwise_or(retained, plan.commit_mask))


def _instance_scores(
    *,
    page,
    shape: tuple[int, int],
    raw_detector_seed: np.ndarray,
    source_positive_seed: np.ndarray,
    semantic_guard: np.ndarray,
    refined_mask: np.ndarray,
    baseline_mask: np.ndarray,
    final_mask: np.ndarray,
) -> tuple[list[dict[str, object]], list[tuple[str, np.ndarray]]]:
    rows: list[dict[str, object]] = []
    optional: list[tuple[str, np.ndarray]] = []
    for instance in page.target_instances:
        target = _read_mask(instance.mask_path, shape)
        if instance.priority == "optional":
            optional.append((instance.instance_id, target))
            continue
        if instance.priority != "required":
            continue
        pixels = int(np.count_nonzero(target))
        detector_pixels = int(
            np.count_nonzero((target > 0) & (raw_detector_seed > 0))
        )
        source_seed_pixels = int(
            np.count_nonzero((target > 0) & (source_positive_seed > 0))
        )
        semantic_denied = int(
            np.count_nonzero((target > 0) & (semantic_guard > 0))
        )
        refined_pixels = int(
            np.count_nonzero((target > 0) & (refined_mask > 0))
        )
        baseline_pixels = int(
            np.count_nonzero((target > 0) & (baseline_mask > 0))
        )
        final_pixels = int(np.count_nonzero((target > 0) & (final_mask > 0)))
        baseline_coverage = float(baseline_pixels) / pixels if pixels else 0.0
        coverage = float(final_pixels) / pixels if pixels else 0.0
        primary_cause: str | None = None
        if coverage < 0.98:
            if detector_pixels == 0 and source_seed_pixels == 0:
                primary_cause = "미검출"
            elif refined_pixels == 0 and semantic_denied > 0:
                primary_cause = "semantic 거절"
            else:
                primary_cause = "마스크 부족"
        rows.append(
            {
                "instance_id": instance.instance_id,
                "detector_seeded": detector_pixels > 0,
                "detector_seed_pixel_count": detector_pixels,
                "source_positive_seeded": source_seed_pixels > 0,
                "source_positive_seed_pixel_count": source_seed_pixels,
                "semantic_denied_pixel_count": semantic_denied,
                "refined_pixel_count": refined_pixels,
                "baseline_coverage": baseline_coverage,
                "coverage": coverage,
                "primary_mask_failure_cause": primary_cause,
                "image_only_failure_cause_to_check": (
                    "LaMa 재생성" if coverage >= 0.98 else None
                ),
            }
        )
    return rows, optional


def score_candidate_page(
    *,
    page,
    evaluation_masks: PageMasks,
    source_routing: SourceRoutingEvidence,
    raw_detector_seed: np.ndarray,
    source_positive_seed: np.ndarray,
    glyph: GlyphRefinementResult,
    baseline_mask: np.ndarray,
    plan: PageIncrementalInpaintPlan,
) -> dict[str, object]:
    """Score after candidate construction; evaluation masks cannot alter plans."""

    final_mask = mask_only_final_mask(baseline_mask, plan)
    delta = cv2.bitwise_xor(binary_mask(baseline_mask), final_mask)
    required, optional = _instance_scores(
        page=page,
        shape=plan.shape,
        raw_detector_seed=raw_detector_seed,
        source_positive_seed=source_positive_seed,
        semantic_guard=source_routing.preserve_or_abstain,
        refined_mask=glyph.refined_mask,
        baseline_mask=baseline_mask,
        final_mask=final_mask,
    )
    optional_union = np.zeros(plan.shape, dtype=np.uint8)
    for _instance_id, mask in optional:
        optional_union[mask > 0] = 255
    target_pixels = int(np.count_nonzero(evaluation_masks.target))
    baseline_target = int(
        np.count_nonzero(
            (evaluation_masks.target > 0) & (baseline_mask > 0)
        )
    )
    final_target = int(
        np.count_nonzero((evaluation_masks.target > 0) & (final_mask > 0))
    )
    source_failures = {
        "source_hard_protect_commit_overlap": int(
            np.count_nonzero(
                (plan.commit_mask > 0) & (source_routing.hard_protect > 0)
            )
        ),
        "source_ownership_commit_leak": int(
            np.count_nonzero(
                (plan.commit_mask > 0)
                & (source_routing.authoritative_translate_ownership == 0)
            )
        ),
        "no_seed_expansion_pixel_count": (
            int(np.count_nonzero(plan.commit_mask))
            if not np.any(source_positive_seed)
            else 0
        ),
        "replacement_outside_baseline": int(
            np.count_nonzero(
                (plan.replacement_source_edit > 0) & (baseline_mask == 0)
            )
        ),
        "replacement_outside_authoritative_owner": int(
            np.count_nonzero(
                (plan.replacement_source_edit > 0)
                & (source_routing.authoritative_translate_ownership == 0)
            )
        ),
        "replacement_source_hard_protect_overlap": int(
            np.count_nonzero(
                (plan.replacement_source_edit > 0)
                & (source_routing.hard_protect > 0)
            )
        ),
    }
    evaluation_failures = {
        "protected_delta_overlap": int(
            np.count_nonzero((delta > 0) & (evaluation_masks.protected > 0))
        ),
        "ambiguous_delta_overlap": int(
            np.count_nonzero((delta > 0) & (evaluation_masks.ambiguous > 0))
        ),
        "preserve_delta_overlap": int(
            np.count_nonzero((delta > 0) & (evaluation_masks.preserve > 0))
            if evaluation_masks.preserve is not None
            else 0
        ),
        "no_edit_false_delta": (
            int(np.count_nonzero(delta)) if page.no_edit else 0
        ),
        "commit_protected_overlap": int(
            np.count_nonzero(
                (plan.commit_mask > 0) & (evaluation_masks.protected > 0)
            )
        ),
        "commit_ambiguous_overlap": int(
            np.count_nonzero(
                (plan.commit_mask > 0) & (evaluation_masks.ambiguous > 0)
            )
        ),
        "commit_preserve_overlap": int(
            np.count_nonzero(
                (plan.commit_mask > 0) & (evaluation_masks.preserve > 0)
            )
            if evaluation_masks.preserve is not None
            else 0
        ),
        "commit_outside_evaluation_ownership": int(
            np.count_nonzero(
                (plan.commit_mask > 0) & (evaluation_masks.ownership == 0)
            )
        ),
        "no_edit_commit_pixel_count": (
            int(np.count_nonzero(plan.commit_mask)) if page.no_edit else 0
        ),
        "restoration_protected_overlap": int(
            np.count_nonzero(
                (plan.replacement_source_edit > 0)
                & (evaluation_masks.protected > 0)
            )
        ),
        "restoration_ambiguous_overlap": int(
            np.count_nonzero(
                (plan.replacement_source_edit > 0)
                & (evaluation_masks.ambiguous > 0)
            )
        ),
        "restoration_preserve_overlap": int(
            np.count_nonzero(
                (plan.replacement_source_edit > 0)
                & (evaluation_masks.preserve > 0)
            )
            if evaluation_masks.preserve is not None
            else 0
        ),
        "restoration_outside_evaluation_ownership": int(
            np.count_nonzero(
                (plan.replacement_source_edit > 0)
                & (evaluation_masks.ownership == 0)
            )
        ),
        "no_edit_restoration_pixel_count": (
            int(np.count_nonzero(plan.replacement_source_edit))
            if page.no_edit
            else 0
        ),
    }
    return {
        "page_id": page.page_id,
        "expected_edit": page.expected_edit,
        "target_pixel_count": target_pixels,
        "baseline_target_covered_pixel_count": baseline_target,
        "target_covered_pixel_count": final_target,
        "baseline_target_coverage": (
            float(baseline_target) / target_pixels if target_pixels else None
        ),
        "target_coverage": (
            float(final_target) / target_pixels if target_pixels else None
        ),
        "target_instance_scores": required,
        "raw_detector_seed_pixel_count": int(np.count_nonzero(raw_detector_seed)),
        "source_positive_seed_pixel_count": int(
            np.count_nonzero(source_positive_seed)
        ),
        "glyph_core_pixel_count": int(np.count_nonzero(glyph.glyph_core)),
        "glyph_effect_pixel_count": int(np.count_nonzero(glyph.glyph_effect)),
        "generation_pixel_count": int(np.count_nonzero(plan.generation_mask)),
        "commit_pixel_count": int(np.count_nonzero(plan.commit_mask)),
        "replacement_source_edit_pixel_count": int(
            np.count_nonzero(plan.replacement_source_edit)
        ),
        "final_mask_pixel_count": int(np.count_nonzero(final_mask)),
        "optional_neutral_delta_pixel_count": int(
            np.count_nonzero((delta > 0) & (optional_union > 0))
        ),
        "maximum_additional_lama_inference_per_page": plan.lama_request_count,
        "source_safety": source_failures,
        "evaluation_safety": evaluation_failures,
        "semantic_actions": list(source_routing.semantic_actions),
        "glyph_provenance": dict(glyph.provenance),
        "final_mask_pixel_sha256": mask_sha256(final_mask),
    }


def aggregate_candidate(
    spec: CandidateSpec,
    pages: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    instance_scores = [
        score
        for page in pages
        for score in page["target_instance_scores"]  # type: ignore[index]
    ]
    target_pixels = sum(int(page["target_pixel_count"]) for page in pages)
    baseline_target = sum(
        int(page["baseline_target_covered_pixel_count"]) for page in pages
    )
    final_target = sum(int(page["target_covered_pixel_count"]) for page in pages)
    source_keys = tuple(pages[0]["source_safety"]) if pages else ()  # type: ignore[arg-type]
    evaluation_keys = tuple(pages[0]["evaluation_safety"]) if pages else ()  # type: ignore[arg-type]
    source_safety = {
        key: sum(int(page["source_safety"][key]) for page in pages)  # type: ignore[index]
        for key in source_keys
    }
    evaluation_safety = {
        key: sum(int(page["evaluation_safety"][key]) for page in pages)  # type: ignore[index]
        for key in evaluation_keys
    }
    baseline_98 = sum(
        float(row["baseline_coverage"]) >= 0.98 for row in instance_scores
    )
    final_98 = sum(float(row["coverage"]) >= 0.98 for row in instance_scores)
    newly_missed = sum(
        float(row["baseline_coverage"]) > 0.0
        and float(row["coverage"]) <= 0.0
        for row in instance_scores
    )
    cause_counts = Counter(
        str(row["primary_mask_failure_cause"])
        for row in instance_scores
        if row.get("primary_mask_failure_cause")
    )
    seedless_status_counts = Counter(
        status
        for page in pages
        for status, count in (
            page.get("seedless_roi", {}).get("status_counts", {}).items()  # type: ignore[union-attr]
            if isinstance(page.get("seedless_roi"), Mapping)
            else ()
        )
        for _ in range(int(count))
    )
    safety_failures = [
        key
        for key, value in {**source_safety, **evaluation_safety}.items()
        if value
    ]
    seedless_finalist_ineligible_pages = sum(
        page.get("seedless_roi", {}).get("finalist_eligible") is False  # type: ignore[union-attr]
        if isinstance(page.get("seedless_roi"), Mapping)
        else False
        for page in pages
    )
    if spec.candidate_id == "b3_conditional_segmenter" and seedless_finalist_ineligible_pages:
        safety_failures.append("seedless_independent_spatial_text_cue_missing")
    baseline_coverage = (
        float(baseline_target) / target_pixels if target_pixels else None
    )
    coverage = float(final_target) / target_pixels if target_pixels else None
    nonregression = (
        coverage is None
        or baseline_coverage is None
        or coverage >= baseline_coverage
    )
    return {
        "candidate_id": spec.candidate_id,
        "mode": spec.mode,
        "commit_source": spec.commit_source,
        "page_count": len(pages),
        "required_target_instance_count": len(instance_scores),
        "detector_seeded_target_instance_count": sum(
            bool(row["detector_seeded"]) for row in instance_scores
        ),
        "source_positive_seeded_target_instance_count": sum(
            bool(row["source_positive_seeded"]) for row in instance_scores
        ),
        "target_pixel_count": target_pixels,
        "baseline_target_covered_pixel_count": baseline_target,
        "target_covered_pixel_count": final_target,
        "baseline_aggregate_target_coverage": baseline_coverage,
        "aggregate_target_coverage": coverage,
        "baseline_98_instance_count": baseline_98,
        "coverage_98_instance_count": final_98,
        "newly_missed_required_instance_count": newly_missed,
        "mask_failure_cause_counts": dict(sorted(cause_counts.items())),
        "seedless_roi_status_counts": dict(sorted(seedless_status_counts.items())),
        "seedless_roi_information_limited_page_count": sum(
            bool(page.get("seedless_roi", {}).get("information_limited"))  # type: ignore[union-attr]
            if isinstance(page.get("seedless_roi"), Mapping)
            else False
            for page in pages
        ),
        "seedless_roi_finalist_ineligible_page_count": (
            seedless_finalist_ineligible_pages
        ),
        "commit_pixel_count": sum(int(page["commit_pixel_count"]) for page in pages),
        "replacement_source_edit_pixel_count": sum(
            int(page["replacement_source_edit_pixel_count"]) for page in pages
        ),
        "maximum_additional_lama_inference_per_page": max(
            (int(page["maximum_additional_lama_inference_per_page"]) for page in pages),
            default=0,
        ),
        "source_safety": source_safety,
        "evaluation_safety": evaluation_safety,
        "mask_only_safety_pass": not safety_failures and newly_missed == 0,
        "mask_only_safety_failures": sorted(
            safety_failures
            + (["newly_missed_required_instance"] if newly_missed else [])
        ),
        "target_coverage_nonregression": nonregression,
        "relative_product_pass": None,
        "relative_product_reason": "cuda_stage2_not_run",
        "output_mask_set_sha256": _canonical_sha256(
            sorted(
                (
                    {
                        "page_id": str(page["page_id"]),
                        "final_mask": str(page["final_mask_pixel_sha256"]),
                    }
                    for page in pages
                ),
                key=lambda row: row["page_id"],
            )
        ),
    }


def shortlist_candidates(candidates: Sequence[Mapping[str, object]]) -> list[str]:
    eligible = [
        row
        for row in candidates
        if row.get("candidate_id") != "b0_pr6"
        and row.get("mask_only_safety_pass") is True
        and row.get("target_coverage_nonregression") is True
    ]
    eligible.sort(
        key=lambda row: (
            -int(row["coverage_98_instance_count"]),
            -float(row.get("aggregate_target_coverage") or 0.0),
            int(row["commit_pixel_count"]),
            str(row["candidate_id"]),
        )
    )
    return [str(row["candidate_id"]) for row in eligible[:2]]


def _artifact_id(scope_id: str, page_id: str, role: str) -> str:
    if not scope_id or "/" in scope_id or "\\" in scope_id:
        raise ValueError("v3.4 artifact scope is not path-safe")
    if role not in ARTIFACT_ROLES:
        raise ValueError(f"v3.4 artifact role is unknown: {role}")
    return f"{scope_id}:{_safe_page_name(page_id)}:{role}"


def _write_scoped_artifact(
    *,
    output_root: Path,
    scope_id: str,
    scope_kind: str,
    page_id: str,
    role: str,
    mask: np.ndarray,
) -> dict[str, object]:
    if scope_kind not in {"shared", "candidate"}:
        raise ValueError("v3.4 artifact scope kind is invalid")
    safe_page = _safe_page_name(page_id)
    if scope_kind == "shared":
        if not scope_id.startswith("shared_"):
            raise ValueError("v3.4 shared artifact scope is invalid")
        directory = output_root / "shared" / scope_id.removeprefix("shared_")
    else:
        if scope_id not in {spec.candidate_id for spec in CANDIDATES}:
            raise ValueError("v3.4 candidate artifact scope is invalid")
        directory = output_root / "runs" / scope_id
    path = directory / role / f"{safe_page}.png"
    normalized = binary_mask(mask)
    record = _write_mask(path, normalized)
    return {
        "artifact_id": _artifact_id(scope_id, page_id, role),
        "scope_id": scope_id,
        "scope_kind": scope_kind,
        "page_id": page_id,
        "role": role,
        "shape": [int(value) for value in normalized.shape],
        "relative_path": path.relative_to(output_root).as_posix(),
        **record,
    }


def _mask_binding(
    *,
    candidate_id: str,
    page_id: str,
    role: str,
    mask: np.ndarray,
    artifact: Mapping[str, object] | None,
    zero_pixel_sha256: str,
) -> dict[str, object]:
    normalized = binary_mask(mask)
    pixels = int(np.count_nonzero(normalized))
    pixel_sha = (
        str(artifact["pixel_sha256"])
        if artifact is not None
        else zero_pixel_sha256
    )
    binding: dict[str, object] = {
        "candidate_id": candidate_id,
        "page_id": page_id,
        "role": role,
        "shape": [int(value) for value in normalized.shape],
        "pixel_sha256": pixel_sha,
        "pixel_count": pixels,
    }
    if artifact is None:
        if pixels:
            raise ValueError("v3.4 nonempty mask lacks a sealed artifact")
        binding["storage"] = "inline_zero"
    else:
        binding["storage"] = "artifact"
        binding["artifact_id"] = artifact["artifact_id"]
    return binding


def write_normalized_page_artifacts(
    *,
    output_root: Path,
    page_id: str,
    detector_seed: np.ndarray,
    source_claim_seed: np.ndarray,
    seedless_source_seed: np.ndarray,
    hard_protect: np.ndarray,
    seeded_glyph: GlyphRefinementResult,
    conditional_glyph: GlyphRefinementResult,
    candidate_plans: Mapping[str, PageIncrementalInpaintPlan],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    """Write each shared mask once and bind every candidate to exact identities."""

    validate_glyph_result_integrity(seeded_glyph)
    validate_glyph_result_integrity(conditional_glyph)
    shape = tuple(np.asarray(detector_seed).shape)
    if len(shape) != 2:
        raise ValueError("v3.4 normalized artifact page shape is invalid")
    source_masks = {
        "detector_seed": binary_mask(detector_seed, shape),
        "source_claim_seed": binary_mask(source_claim_seed, shape),
        "hard_protect": binary_mask(hard_protect, shape),
    }
    seeded_masks = {
        "owned_positive_seed": binary_mask(seeded_glyph.owned_detector_seed, shape),
        "glyph_core": binary_mask(seeded_glyph.glyph_core, shape),
        "glyph_effect": binary_mask(seeded_glyph.glyph_effect, shape),
    }
    conditional_masks = {
        "seedless_source_seed": binary_mask(seedless_source_seed, shape),
        "owned_positive_seed": binary_mask(
            conditional_glyph.owned_detector_seed, shape
        ),
        "glyph_core": binary_mask(conditional_glyph.glyph_core, shape),
        "glyph_effect": binary_mask(conditional_glyph.glyph_effect, shape),
    }
    if not np.array_equal(binary_mask(conditional_glyph.hard_protect, shape), source_masks["hard_protect"]):
        raise ValueError("v3.4 conditional hard protection differs")
    if not np.array_equal(binary_mask(seeded_glyph.hard_protect, shape), source_masks["hard_protect"]):
        raise ValueError("v3.4 seeded hard protection differs")
    if set(candidate_plans) != {spec.candidate_id for spec in CANDIDATES}:
        raise ValueError("v3.4 candidate plan inventory differs")
    zero_pixel_sha256 = mask_sha256(np.zeros(shape, dtype=np.uint8))

    artifacts: list[dict[str, object]] = []

    def store(
        scope_id: str,
        scope_kind: str,
        role: str,
        mask: np.ndarray,
    ) -> dict[str, object] | None:
        normalized = binary_mask(mask, shape)
        if not np.any(normalized):
            return None
        artifact = _write_scoped_artifact(
            output_root=output_root,
            scope_id=scope_id,
            scope_kind=scope_kind,
            page_id=page_id,
            role=role,
            mask=normalized,
        )
        artifacts.append(artifact)
        return artifact

    source_artifacts = {
        role: store("shared_source", "shared", role, source_masks[role])
        for role in SHARED_SOURCE_ROLES
    }
    seeded_artifacts = {
        role: store("shared_seeded", "shared", role, seeded_masks[role])
        for role in SHARED_GLYPH_ROLES
    }
    conditional_artifacts: dict[str, dict[str, object] | None] = {
        "seedless_source_seed": store(
            "shared_conditional",
            "shared",
            "seedless_source_seed",
            conditional_masks["seedless_source_seed"],
        )
    }
    for role in SHARED_GLYPH_ROLES:
        conditional_artifacts[role] = (
            seeded_artifacts[role]
            if np.array_equal(conditional_masks[role], seeded_masks[role])
            else store(
                "shared_conditional",
                "shared",
                role,
                conditional_masks[role],
            )
        )

    bindings: list[dict[str, object]] = []
    for spec in CANDIDATES:
        conditional = spec.candidate_id == "b3_conditional_segmenter"
        glyph_masks = conditional_masks if conditional else seeded_masks
        glyph_artifacts = conditional_artifacts if conditional else seeded_artifacts
        role_masks: dict[str, np.ndarray] = {
            **source_masks,
            "seedless_source_seed": (
                conditional_masks["seedless_source_seed"]
                if conditional
                else np.zeros(shape, dtype=np.uint8)
            ),
            **{role: glyph_masks[role] for role in SHARED_GLYPH_ROLES},
        }
        role_artifacts: dict[str, Mapping[str, object] | None] = {
            **source_artifacts,
            "seedless_source_seed": (
                conditional_artifacts["seedless_source_seed"]
                if conditional
                else None
            ),
            **{role: glyph_artifacts[role] for role in SHARED_GLYPH_ROLES},
        }
        plan = candidate_plans[spec.candidate_id]
        plan_masks = {
            "generation_mask": plan.generation_mask,
            "commit_mask": plan.commit_mask,
            "replacement_source_edit": plan.replacement_source_edit,
        }
        if spec.mode == "baseline" and any(
            np.any(mask) for mask in plan_masks.values()
        ):
            raise ValueError("v3.4 baseline candidate plan must be empty")
        for role in PLAN_ARTIFACT_ROLES:
            role_masks[role] = binary_mask(plan_masks[role], shape)
            role_artifacts[role] = store(
                spec.candidate_id,
                "candidate",
                role,
                role_masks[role],
            )
        if set(role_masks) != set(ARTIFACT_ROLES):
            raise AssertionError("v3.4 normalized artifact roles are incomplete")
        for role in ARTIFACT_ROLES:
            bindings.append(
                _mask_binding(
                    candidate_id=spec.candidate_id,
                    page_id=page_id,
                    role=role,
                    mask=role_masks[role],
                    artifact=role_artifacts[role],
                    zero_pixel_sha256=zero_pixel_sha256,
                )
            )

    shared_writes = sum(row["scope_kind"] == "shared" for row in artifacts)
    candidate_writes = sum(row["scope_kind"] == "candidate" for row in artifacts)
    if shared_writes > MAX_SHARED_ARTIFACT_WRITES_PER_PAGE:
        raise AssertionError("v3.4 shared artifact write bound exceeded")
    if candidate_writes > MAX_CANDIDATE_ARTIFACT_WRITES_PER_PAGE:
        raise AssertionError("v3.4 candidate artifact write bound exceeded")
    return artifacts, bindings, {
        "shared_artifact_write_count": int(shared_writes),
        "candidate_artifact_write_count": int(candidate_writes),
        "artifact_write_count": int(len(artifacts)),
        "maximum_shared_artifact_writes_per_page": (
            MAX_SHARED_ARTIFACT_WRITES_PER_PAGE
        ),
        "maximum_candidate_artifact_writes_per_page": (
            MAX_CANDIDATE_ARTIFACT_WRITES_PER_PAGE
        ),
    }


def validate_output_inventory(
    inventory: Mapping[str, object],
    *,
    output_root: Path,
) -> None:
    expected_sha = inventory.get("inventory_sha256")
    unsigned = {key: value for key, value in inventory.items() if key != "inventory_sha256"}
    if expected_sha != _canonical_sha256(unsigned):
        raise ValueError("v3.4 output inventory SHA differs")
    artifacts = inventory.get("artifacts")
    bindings = inventory.get("mask_bindings")
    if not isinstance(artifacts, list) or not isinstance(bindings, list):
        raise ValueError("v3.4 output inventory rows are invalid")
    artifacts_by_id: dict[str, Mapping[str, object]] = {}
    for row in artifacts:
        if not isinstance(row, Mapping):
            raise ValueError("v3.4 output inventory row is invalid")
        artifact_id = str(row.get("artifact_id") or "")
        if not artifact_id or artifact_id in artifacts_by_id:
            raise ValueError("v3.4 output inventory identity is invalid")
        artifacts_by_id[artifact_id] = row
        if row.get("scope_kind") not in {"shared", "candidate"}:
            raise ValueError("v3.4 output artifact scope is invalid")
        if row.get("role") not in ARTIFACT_ROLES:
            raise ValueError("v3.4 output artifact role is invalid")
        scope_id = str(row.get("scope_id") or "")
        page_id = str(row.get("page_id") or "")
        role = str(row.get("role") or "")
        if artifact_id != _artifact_id(scope_id, page_id, role):
            raise ValueError("v3.4 output artifact identity fields differ")
        if row.get("scope_kind") == "shared":
            if scope_id not in {
                "shared_source",
                "shared_seeded",
                "shared_conditional",
            }:
                raise ValueError("v3.4 shared output artifact scope differs")
            expected_relative = (
                Path("shared")
                / scope_id.removeprefix("shared_")
                / role
                / f"{_safe_page_name(page_id)}.png"
            )
        else:
            if scope_id not in {spec.candidate_id for spec in CANDIDATES}:
                raise ValueError("v3.4 candidate output artifact scope differs")
            expected_relative = (
                Path("runs")
                / scope_id
                / role
                / f"{_safe_page_name(page_id)}.png"
            )
        relative = Path(str(row.get("relative_path") or ""))
        if relative != expected_relative:
            raise ValueError("v3.4 output artifact relative path differs")
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("v3.4 output inventory path escapes run root")
        artifact = (output_root / relative).resolve()
        if output_root.resolve() not in artifact.parents:
            raise ValueError("v3.4 output inventory path escapes run root")
        if _sha256(artifact) != row.get("file_sha256"):
            raise ValueError("v3.4 output artifact SHA differs")
        if artifact.suffix.lower() == ".png":
            raw_shape = row.get("shape")
            if (
                not isinstance(raw_shape, list)
                or len(raw_shape) != 2
                or any(not isinstance(value, int) or value <= 0 for value in raw_shape)
            ):
                raise ValueError("v3.4 output artifact shape is invalid")
            shape = (int(raw_shape[0]), int(raw_shape[1]))
            decoded = _read_mask(artifact, shape)
            if mask_sha256(decoded) != row.get("pixel_sha256"):
                raise ValueError("v3.4 output mask pixel SHA differs")
            if int(np.count_nonzero(decoded)) != row.get("pixel_count"):
                raise ValueError("v3.4 output mask pixel count differs")

    expected_candidates = {
        str(value) for value in inventory.get("candidate_ids", [])
    }
    expected_pages = {str(value) for value in inventory.get("page_ids", [])}
    expected_roles = {str(value) for value in inventory.get("artifact_roles", [])}
    if (
        expected_candidates != {spec.candidate_id for spec in CANDIDATES}
        or not expected_pages
        or expected_roles != set(ARTIFACT_ROLES)
    ):
        raise ValueError("v3.4 output binding contract differs")
    expected_binding_ids = {
        (candidate_id, page_id, role)
        for candidate_id in expected_candidates
        for page_id in expected_pages
        for role in expected_roles
    }
    binding_ids: set[tuple[str, str, str]] = set()
    referenced_artifacts: set[str] = set()
    zero_hashes: dict[tuple[int, int], str] = {}
    for row in bindings:
        if not isinstance(row, Mapping):
            raise ValueError("v3.4 output mask binding is invalid")
        identity = (
            str(row.get("candidate_id") or ""),
            str(row.get("page_id") or ""),
            str(row.get("role") or ""),
        )
        if identity not in expected_binding_ids or identity in binding_ids:
            raise ValueError("v3.4 output mask binding identity differs")
        binding_ids.add(identity)
        raw_shape = row.get("shape")
        if (
            not isinstance(raw_shape, list)
            or len(raw_shape) != 2
            or any(not isinstance(value, int) or value <= 0 for value in raw_shape)
        ):
            raise ValueError("v3.4 output mask binding shape is invalid")
        shape = (int(raw_shape[0]), int(raw_shape[1]))
        storage = row.get("storage")
        if storage == "inline_zero":
            zero_hash = zero_hashes.get(shape)
            if zero_hash is None:
                zero_hash = mask_sha256(np.zeros(shape, dtype=np.uint8))
                zero_hashes[shape] = zero_hash
            if (
                row.get("pixel_count") != 0
                or row.get("pixel_sha256") != zero_hash
                or row.get("artifact_id") is not None
            ):
                raise ValueError("v3.4 inline zero binding differs")
        elif storage == "artifact":
            artifact_id = str(row.get("artifact_id") or "")
            artifact = artifacts_by_id.get(artifact_id)
            if artifact is None:
                raise ValueError("v3.4 output mask binding artifact is missing")
            referenced_artifacts.add(artifact_id)
            if (
                artifact.get("page_id") != identity[1]
                or artifact.get("role") != identity[2]
                or artifact.get("shape") != raw_shape
                or artifact.get("pixel_sha256") != row.get("pixel_sha256")
                or artifact.get("pixel_count") != row.get("pixel_count")
            ):
                raise ValueError("v3.4 output mask binding artifact differs")
            scope_id = str(artifact.get("scope_id") or "")
            role = identity[2]
            candidate_id = identity[0]
            if role in SHARED_SOURCE_ROLES:
                allowed_scopes = {"shared_source"}
            elif role == "seedless_source_seed":
                allowed_scopes = (
                    {"shared_conditional"}
                    if candidate_id == "b3_conditional_segmenter"
                    else set()
                )
            elif role in SHARED_GLYPH_ROLES:
                allowed_scopes = (
                    {"shared_seeded", "shared_conditional"}
                    if candidate_id == "b3_conditional_segmenter"
                    else {"shared_seeded"}
                )
            else:
                allowed_scopes = (
                    set() if candidate_id == "b0_pr6" else {candidate_id}
                )
            if scope_id not in allowed_scopes:
                raise ValueError("v3.4 output mask binding scope differs")
        else:
            raise ValueError("v3.4 output mask binding storage is invalid")
    if binding_ids != expected_binding_ids:
        raise ValueError("v3.4 output mask binding inventory is incomplete")
    if referenced_artifacts != set(artifacts_by_id):
        raise ValueError("v3.4 output artifact has no exact candidate binding")

    counts = inventory.get("write_counts")
    per_page_counts = inventory.get("per_page_write_counts")
    if not isinstance(counts, Mapping) or not isinstance(per_page_counts, list):
        raise ValueError("v3.4 output write counts are missing")
    actual_shared = sum(row.get("scope_kind") == "shared" for row in artifacts)
    actual_candidate = sum(row.get("scope_kind") == "candidate" for row in artifacts)
    actual_by_page = {
        page_id: {
            "shared_artifact_write_count": sum(
                row.get("page_id") == page_id and row.get("scope_kind") == "shared"
                for row in artifacts
            ),
            "candidate_artifact_write_count": sum(
                row.get("page_id") == page_id and row.get("scope_kind") == "candidate"
                for row in artifacts
            ),
        }
        for page_id in expected_pages
    }
    recorded_by_page = {
        str(row.get("page_id") or ""): row
        for row in per_page_counts
        if isinstance(row, Mapping)
    }
    if set(recorded_by_page) != expected_pages:
        raise ValueError("v3.4 per-page write count inventory differs")
    for page_id, actual in actual_by_page.items():
        recorded = recorded_by_page[page_id]
        if (
            recorded.get("shared_artifact_write_count")
            != actual["shared_artifact_write_count"]
            or recorded.get("candidate_artifact_write_count")
            != actual["candidate_artifact_write_count"]
            or recorded.get("artifact_write_count")
            != sum(actual.values())
            or actual["shared_artifact_write_count"]
            > MAX_SHARED_ARTIFACT_WRITES_PER_PAGE
            or actual["candidate_artifact_write_count"]
            > MAX_CANDIDATE_ARTIFACT_WRITES_PER_PAGE
        ):
            raise ValueError("v3.4 per-page write count contract differs")
    if (
        counts.get("shared_artifact_write_count") != actual_shared
        or counts.get("candidate_artifact_write_count") != actual_candidate
        or counts.get("artifact_write_count") != len(artifacts)
        or counts.get("maximum_shared_artifact_writes_per_page")
        != MAX_SHARED_ARTIFACT_WRITES_PER_PAGE
        or counts.get("maximum_candidate_artifact_writes_per_page")
        != MAX_CANDIDATE_ARTIFACT_WRITES_PER_PAGE
        or actual_shared > len(expected_pages) * MAX_SHARED_ARTIFACT_WRITES_PER_PAGE
        or actual_candidate
        > len(expected_pages) * MAX_CANDIDATE_ARTIFACT_WRITES_PER_PAGE
    ):
        raise ValueError("v3.4 output write count contract differs")


def run_mask_only(
    *,
    source_manifest_path: Path,
    relative_manifest_path: Path,
    source_routing_overlay_path: Path,
    policy_overlay_path: Path,
    finetune_raw_run: Path,
    finetune_native3_run: Path,
    tiled_raw_run: Path,
    tiled_native3_run: Path,
    seedless_ocr_overlay_path: Path | None,
    output_root: Path,
) -> dict[str, object]:
    start_code_identity = capture_official_code_identity()
    source_binding = validate_source_only_manifest_v4(source_manifest_path)
    source_manifest_payload = _read_json(source_manifest_path)
    relative_binding = validate_source_only_manifest_v4(relative_manifest_path)
    raw_relative = _read_json(relative_manifest_path)
    relative_seal_path = relative_manifest_path.with_suffix(
        relative_manifest_path.suffix + ".seal.json"
    )
    relative_seal = _read_json(relative_seal_path)
    if relative_seal.get("source_manifest_sha256") != source_binding["manifest_sha256"]:
        raise ValueError("v3.4 relative baseline is not bound to source manifest")
    source_routing_binding = validate_source_routing_overlay(
        source_routing_overlay_path,
        source_manifest_path=source_manifest_path,
        relative_manifest_path=relative_manifest_path,
        source_binding=source_binding,
        relative_binding=relative_binding,
        source_manifest_payload=source_manifest_payload,
        relative_manifest_payload=raw_relative,
    )
    overlay = validate_policy_overlay(
        policy_overlay_path,
        manifest_path=source_manifest_path,
    )
    seedless_binding = validate_seedless_ocr_overlay(
        seedless_ocr_overlay_path,
        source_manifest_path=source_manifest_path,
        source_binding=source_binding,
        source_manifest_payload=source_manifest_payload,
    )
    pages = load_stage1_manifest(relative_manifest_path)
    page_ids = tuple(page.page_id for page in pages)
    if tuple(sorted(page_ids)) != tuple(source_binding["page_ids"]):
        raise ValueError("v3.4 relative and source page inventories differ")
    entries = {
        str(row.get("page_id") or ""): row
        for row in raw_relative.get("pages", [])
        if isinstance(row, dict)
    }
    if set(entries) != set(page_ids):
        raise ValueError("v3.4 relative raw page inventory differs")

    inputs = {
        "finetune_raw": _stage1_input(
            finetune_raw_run,
            expected_variant="raw",
            expected_source_manifest_sha256=str(source_binding["manifest_sha256"]),
            page_ids=page_ids,
        ),
        "finetune_native3": _stage1_input(
            finetune_native3_run,
            expected_variant="dilated",
            expected_source_manifest_sha256=str(source_binding["manifest_sha256"]),
            page_ids=page_ids,
        ),
        "tiled_raw": _stage1_input(
            tiled_raw_run,
            expected_variant="raw",
            expected_source_manifest_sha256=str(source_binding["manifest_sha256"]),
            page_ids=page_ids,
        ),
        "tiled_native3": _stage1_input(
            tiled_native3_run,
            expected_variant="dilated",
            expected_source_manifest_sha256=str(source_binding["manifest_sha256"]),
            page_ids=page_ids,
        ),
    }
    detector_bundles = {
        "finetune": bind_detector_inference_bundle(
            inputs["finetune_raw"],
            inputs["finetune_native3"],
            DETECTOR_BUNDLE_EXPECTATIONS["finetune"],
        ),
        "tiled": bind_detector_inference_bundle(
            inputs["tiled_raw"],
            inputs["tiled_native3"],
            DETECTOR_BUNDLE_EXPECTATIONS["tiled"],
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise FileExistsError("v3.4 mask-only output must be fresh")

    candidate_pages: dict[str, list[dict[str, object]]] = {
        spec.candidate_id: [] for spec in CANDIDATES
    }
    output_artifacts: list[dict[str, object]] = []
    output_mask_bindings: list[dict[str, object]] = []
    page_write_counts: list[dict[str, object]] = []
    source_region_artifacts: list[dict[str, object]] = []
    decisions_partial = output_root / ".decisions.jsonl.partial"
    baseline_inventory: list[dict[str, str]] = []
    with decisions_partial.open("w", encoding="utf-8", newline="\n") as stream:
        for page in pages:
            entry = entries[page.page_id]
            source = _read_image(page.source_image, cv2.IMREAD_COLOR)
            shape = source.shape[:2]
            baseline_path = _resolve_manifest_artifact(
                relative_manifest_path,
                entry.get("baseline_mask"),
            )
            if baseline_path is None:
                raise ValueError(f"v3.4 baseline mask is missing: {page.page_id}")
            baseline = _read_mask(baseline_path, shape)
            baseline_inventory.append(
                {
                    "page_id": page.page_id,
                    "file_sha256": _sha256(baseline_path),
                    "pixel_sha256": mask_sha256(baseline),
                }
            )
            raw_regions = entry.get("regions")
            if not isinstance(raw_regions, list) or any(
                not isinstance(row, Mapping) for row in raw_regions
            ):
                raise ValueError("v3.4 source region records are invalid")
            overlay_regions: list[SourceRoutingOverlayRegionRecord] = []
            for row in raw_regions:
                region_id = str(row.get("region_id") or "")
                record = source_routing_binding["records"].get(
                    (page.page_id, region_id)
                )
                if not isinstance(record, SourceRoutingOverlayRegionRecord):
                    raise ValueError("v3.4 source routing overlay region is missing")
                overlay_regions.append(record)
            source_regions = tuple(
                load_source_region_evidence(
                    row,
                    overlay_region,
                    shape=shape,
                )
                for row, overlay_region in zip(raw_regions, overlay_regions)
            )
            routing = build_source_routing_evidence(
                raw_regions,
                source_regions,
                overlay_regions,
                shape=shape,
            )
            source_region_artifacts.extend(
                {
                    "page_id": page.page_id,
                    **record,
                }
                for record in routing.artifact_inventory
            )
            detector_masks = {
                key: _read_mask(value["masks"][page.page_id], shape)  # type: ignore[index]
                for key, value in inputs.items()
            }
            raw_or = cv2.bitwise_or(
                detector_masks["finetune_raw"], detector_masks["tiled_raw"]
            )
            native3_or = cv2.bitwise_or(
                detector_masks["finetune_native3"],
                detector_masks["tiled_native3"],
            )
            if np.any((raw_or > 0) & (native3_or == 0)):
                raise ValueError("v3.4 native3 union does not contain raw union")
            source_claim_seed_path = _resolve_manifest_artifact(
                relative_manifest_path,
                entry.get("claim_seed_mask"),
            )
            if source_claim_seed_path is None:
                source_claim_seed = np.zeros(shape, dtype=np.uint8)
            else:
                source_claim_seed = _read_mask(source_claim_seed_path, shape)
            detector_seed = cv2.bitwise_or(raw_or, source_claim_seed)
            seeded_glyph = extract_roi_local_glyph_masks(
                source,
                detector_seed=detector_seed,
                ocr_ownership=routing.authoritative_translate_ownership,
                hard_protect=routing.hard_protect,
                detector_provider="finetune_e6_raw_or_tiled512_raw_or_pr2_source_seed",
                ownership_provider="source_reviewed_region_ownership",
                effect_support=native3_or,
                effect_support_provider="finetune_e6_native3_or_tiled512_native3",
            )
            seedless_results, seedless_status, seedless_positive = (
                extract_page_seedless_results(
                    source,
                    page_id=page.page_id,
                    raw_regions=raw_regions,
                    source_regions=source_regions,
                    routing=routing,
                    evidence_records=seedless_binding["records"],
                )
            )
            conditional_glyph = merge_glyph_refinement_results(
                (seeded_glyph, *seedless_results),
                shape=shape,
                hard_protect=routing.hard_protect,
            )
            glyph_by_candidate = {
                spec.candidate_id: (
                    conditional_glyph
                    if spec.candidate_id == "b3_conditional_segmenter"
                    else seeded_glyph
                )
                for spec in CANDIDATES
            }
            candidate_plans = {
                spec.candidate_id: build_candidate_plan(
                    spec,
                    glyph=glyph_by_candidate[spec.candidate_id],
                    baseline_mask=baseline,
                    authoritative_owner_masks=(
                        routing.authoritative_translate_owner_masks
                    ),
                    all_ownership=routing.all_ownership,
                )
                for spec in CANDIDATES
            }

            owned_raw_detector = np.where(
                (raw_or > 0)
                & (routing.authoritative_translate_ownership > 0),
                255,
                0,
            ).astype(np.uint8)
            owned_source_claim = np.where(
                (source_claim_seed > 0)
                & (routing.authoritative_translate_ownership > 0),
                255,
                0,
            ).astype(np.uint8)
            artifacts, bindings, write_counts = write_normalized_page_artifacts(
                output_root=output_root,
                page_id=page.page_id,
                detector_seed=owned_raw_detector,
                source_claim_seed=owned_source_claim,
                seedless_source_seed=seedless_positive,
                hard_protect=routing.hard_protect,
                seeded_glyph=seeded_glyph,
                conditional_glyph=conditional_glyph,
                candidate_plans=candidate_plans,
            )
            output_artifacts.extend(artifacts)
            output_mask_bindings.extend(bindings)
            page_write_counts.append({"page_id": page.page_id, **write_counts})

            # Candidate pixels are fixed above.  Only now open page-level
            # target/protected/ambiguous/preserve annotations for scoring.
            evaluation_masks = load_page_masks(
                page,
                shape,
                existing_edit_path=str(baseline_path),
                strict_binary=True,
            )
            for spec in CANDIDATES:
                plan = candidate_plans[spec.candidate_id]
                candidate_glyph = glyph_by_candidate[spec.candidate_id]
                score = score_candidate_page(
                    page=page,
                    evaluation_masks=evaluation_masks,
                    source_routing=routing,
                    raw_detector_seed=owned_raw_detector,
                    source_positive_seed=candidate_glyph.owned_detector_seed,
                    glyph=candidate_glyph,
                    baseline_mask=baseline,
                    plan=plan,
                )
                score["seedless_roi"] = (
                    seedless_status
                    if spec.candidate_id == "b3_conditional_segmenter"
                    else {
                        "status_counts": {},
                        "information_limited": False,
                        "seedless_positive_pixel_count": 0,
                        "status": "not_applicable",
                    }
                )
                candidate_pages[spec.candidate_id].append(score)
                for row in score["target_instance_scores"]:
                    stream.write(
                        json.dumps(
                            {
                                "candidate_id": spec.candidate_id,
                                "page_id": page.page_id,
                                **row,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                for record in candidate_glyph.component_records:
                    stream.write(
                        json.dumps(
                            {
                                "candidate_id": spec.candidate_id,
                                "page_id": page.page_id,
                                "record_type": "glyph_component",
                                **asdict(record),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
    decisions_path = output_root / "decisions.jsonl"
    decisions_partial.replace(decisions_path)
    candidates = [
        aggregate_candidate(spec, candidate_pages[spec.candidate_id])
        for spec in CANDIDATES
    ]
    shortlist = shortlist_candidates(candidates)
    output_artifacts.sort(
        key=lambda row: (
            str(row["scope_id"]),
            str(row["page_id"]),
            str(row["role"]),
        )
    )
    output_mask_bindings.sort(
        key=lambda row: (
            str(row["candidate_id"]),
            str(row["page_id"]),
            str(row["role"]),
        )
    )
    write_counts = {
        "shared_artifact_write_count": sum(
            int(row["shared_artifact_write_count"])
            for row in page_write_counts
        ),
        "candidate_artifact_write_count": sum(
            int(row["candidate_artifact_write_count"])
            for row in page_write_counts
        ),
        "artifact_write_count": len(output_artifacts),
        "maximum_shared_artifact_writes_per_page": (
            MAX_SHARED_ARTIFACT_WRITES_PER_PAGE
        ),
        "maximum_candidate_artifact_writes_per_page": (
            MAX_CANDIDATE_ARTIFACT_WRITES_PER_PAGE
        ),
    }
    inventory_unsigned: dict[str, object] = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "source_manifest_sha256": source_binding["manifest_sha256"],
        "relative_manifest_sha256": relative_binding["manifest_sha256"],
        "source_routing_overlay_sha256": source_routing_binding[
            "overlay_sha256"
        ],
        "policy_overlay_sha256": overlay["overlay_sha256"],
        "seedless_ocr_overlay_sha256": seedless_binding["overlay_sha256"],
        "source_region_artifact_inventory_sha256": _canonical_sha256(
            sorted(
                source_region_artifacts,
                key=lambda row: (
                    str(row["page_id"]),
                    str(row["region_id"]),
                    str(row["role"]),
                ),
            )
        ),
        "detector_result_sha256": {
            key: value["result_sha256"] for key, value in sorted(inputs.items())
        },
        "detector_inference_bundle_sha256": {
            key: value["inference_bundle_sha256"]
            for key, value in sorted(detector_bundles.items())
        },
        "evaluator_sha256": _sha256(Path(__file__).resolve()),
        "glyph_refinement_sha256": _sha256(
            ROOT / "benchmarking/inpaint_detector_bakeoff/glyph_refinement.py"
        ),
        "incremental_plan_sha256": _sha256(
            ROOT / "benchmarking/inpaint_detector_bakeoff/incremental_plan.py"
        ),
        "official_code_identity": start_code_identity,
        "candidate_ids": [spec.candidate_id for spec in CANDIDATES],
        "page_ids": sorted(page_ids),
        "artifact_roles": list(ARTIFACT_ROLES),
        "mask_bindings": output_mask_bindings,
        "write_counts": write_counts,
        "per_page_write_counts": page_write_counts,
        "artifacts": output_artifacts,
    }
    inventory = dict(inventory_unsigned)
    inventory["inventory_sha256"] = _canonical_sha256(inventory_unsigned)
    inventory_path = output_root / "output-artifact-inventory.json"
    _atomic_json(inventory_path, inventory)
    validate_output_inventory(inventory, output_root=output_root)
    end_code_identity = verify_official_code_identity(start_code_identity)
    return {
        "schema_version": SCHEMA_VERSION,
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
        "official_code_identity": end_code_identity,
        "evaluator_sha256": _sha256(Path(__file__).resolve()),
        "glyph_refinement_sha256": _sha256(
            ROOT / "benchmarking/inpaint_detector_bakeoff/glyph_refinement.py"
        ),
        "incremental_plan_sha256": _sha256(
            ROOT / "benchmarking/inpaint_detector_bakeoff/incremental_plan.py"
        ),
        "source_manifest": dict(source_binding),
        "relative_manifest": dict(relative_binding),
        "relative_manifest_seal_sha256": _sha256(relative_seal_path),
        "source_routing_overlay": {
            key: source_routing_binding[key]
            for key in (
                "overlay_sha256",
                "artifact_sha256",
                "seal_sha256",
                "source_evidence_file_sha256",
                "source_evidence_payload_sha256",
            )
        },
        "policy_overlay": {
            "artifact_sha256": _sha256(policy_overlay_path),
            "overlay_sha256": overlay["overlay_sha256"],
            "policy_id": overlay["policy_id"],
            "used_for_candidate_generation": False,
        },
        "seedless_ocr_overlay": {
            "available": seedless_binding["available"],
            "status": seedless_binding["status"],
            "overlay_sha256": seedless_binding["overlay_sha256"],
            "artifact_sha256": seedless_binding["artifact_sha256"],
            "seal_sha256": seedless_binding["seal_sha256"],
            "record_count": len(seedless_binding["records"]),
            "used_for_candidate_generation": bool(seedless_binding["records"]),
        },
        "candidate_generation_contract": {
            "evaluation_target_used": False,
            "evaluation_page_protected_used": False,
            "evaluation_ambiguous_used": False,
            "evaluation_preserve_used": False,
            "source_region_protect_used": True,
            "source_region_ambiguous_used": True,
            "source_region_corner_used": True,
            "source_region_ownership_used": True,
            "source_routing_overlay_required": True,
            "source_routing_sibling_discovery_used": False,
            "runtime_semantic_action_used": True,
            "no_seed_no_expansion": True,
        },
        "input_detector_runs": {
            key: {
                field: value[field]
                for field in (
                    "run_root",
                    "result_path",
                    "result_sha256",
                    "candidate",
                    "variant",
                    "mask_root",
                )
            }
            for key, value in inputs.items()
        },
        "detector_inference_bundles": detector_bundles,
        "source_region_artifacts": {
            "inventory_sha256": _canonical_sha256(
                sorted(
                    source_region_artifacts,
                    key=lambda row: (
                        str(row["page_id"]),
                        str(row["region_id"]),
                        str(row["role"]),
                    ),
                )
            ),
            "artifacts": sorted(
                source_region_artifacts,
                key=lambda row: (
                    str(row["page_id"]),
                    str(row["region_id"]),
                    str(row["role"]),
                ),
            ),
        },
        "baseline": {
            "mask_set_sha256": _canonical_sha256(
                sorted(baseline_inventory, key=lambda row: row["page_id"])
            ),
            "pages": baseline_inventory,
        },
        "candidates": candidates,
        "pages": candidate_pages,
        "shortlist": shortlist,
        "shortlist_limit": 2,
        "failure_cause_labels": [
            "미검출",
            "마스크 부족",
            "semantic 거절",
            "LaMa 재생성",
        ],
        "decisions": {
            "relative_path": decisions_path.relative_to(output_root).as_posix(),
            "artifact_sha256": _sha256(decisions_path),
            "size_bytes": decisions_path.stat().st_size,
        },
        "output_inventory": {
            "relative_path": inventory_path.relative_to(output_root).as_posix(),
            "artifact_sha256": _sha256(inventory_path),
            "inventory_sha256": inventory["inventory_sha256"],
            "artifact_count": len(output_artifacts),
            "mask_binding_count": len(output_mask_bindings),
            "write_counts": write_counts,
            "per_page_write_counts": page_write_counts,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate source-only glyph mask refinements over a sealed baseline."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--relative-manifest", type=Path, required=True)
    parser.add_argument("--source-routing-overlay", type=Path, required=True)
    parser.add_argument("--policy-overlay", type=Path, required=True)
    parser.add_argument("--finetune-raw-run", type=Path, required=True)
    parser.add_argument("--finetune-native3-run", type=Path, required=True)
    parser.add_argument("--tiled-raw-run", type=Path, required=True)
    parser.add_argument("--tiled-native3-run", type=Path, required=True)
    parser.add_argument("--seedless-ocr-overlay", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    try:
        result = run_mask_only(
            source_manifest_path=args.source_manifest.resolve(),
            relative_manifest_path=args.relative_manifest.resolve(),
            source_routing_overlay_path=args.source_routing_overlay.resolve(),
            policy_overlay_path=args.policy_overlay.resolve(),
            finetune_raw_run=args.finetune_raw_run.resolve(),
            finetune_native3_run=args.finetune_native3_run.resolve(),
            tiled_raw_run=args.tiled_raw_run.resolve(),
            tiled_native3_run=args.tiled_native3_run.resolve(),
            seedless_ocr_overlay_path=(
                args.seedless_ocr_overlay.resolve()
                if args.seedless_ocr_overlay is not None
                else None
            ),
            output_root=output_root,
        )
        result_path = output_root / "glyph-refinement-mask-results.json"
        _atomic_json(result_path, result)
        if managed is not None:
            managed.complete(
                metadata={
                    "source_manifest_sha256": result["source_manifest"][
                        "manifest_sha256"
                    ],
                    "shortlist": result["shortlist"],
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
        return 0
    except BaseException as error:
        if managed is not None:
            managed.fail(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
