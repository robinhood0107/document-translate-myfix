#!/usr/bin/env python3
"""Export source-only product detector/OCR evidence for inpaint v3.4.

This exporter deliberately separates *where a product block is owned* from
*which pixels may later be edited*.  It opens only the source image and each
region's source-reviewed ownership mask.  Target, protection, candidate, and
evaluation artifacts are outside this module's input surface.

The normalized JSON is consumed by ``build_inpaint_seedless_ocr_overlay_v34``.
PaddleOCR-VL is generative and currently reports no calibrated OCR confidence,
so a positive confidence is accepted only when the exact RT-DETR ONNX proposal
that produced the product block can be recovered.  Otherwise the record is
kept with confidence zero and the downstream overlay remains
``information_limited``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Protocol, Sequence

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.detection.factory import DetectionEngineFactory  # noqa: E402
from modules.detection.processor import TextBlockDetector  # noqa: E402
from modules.ocr.common.result_contract import (  # noqa: E402
    OCR_STRATEGY_PADDLE_CROP,
    finalize_ocr_processing_contract,
    initialize_ocr_result_contract,
)
from modules.ocr.local_runtime import LocalOCRRuntimeManager  # noqa: E402
from modules.ocr.paddle_crop.engine import PaddleOCRVLEngine  # noqa: E402
from modules.ocr.paddle_crop.transport import (  # noqa: E402
    DEFAULT_PADDLE_DIRECT_SERVER_URL,
)
from modules.utils.download import ModelDownloader, ModelID  # noqa: E402
from modules.utils.textblock import TextBlock  # noqa: E402
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)
from scripts.build_inpaint_seedless_ocr_overlay_v34 import (  # noqa: E402
    TRACKED_RUNTIME_DEPENDENCIES,
)


FAMILY = "inpaint-source-ocr-evidence-v34"
CATEGORY = "40-inpaint-mask-render"
SOURCE_MANIFEST_SCHEMA_VERSION = "inpaint-factorized-source-manifest-v4"
SOURCE_EVIDENCE_SCHEMA_VERSION = "inpaint-source-ocr-block-evidence-v34"
SOURCE_EVIDENCE_SEAL_SCHEMA_VERSION = "inpaint-source-ocr-block-evidence-seal-v34"
SOURCE_EVIDENCE_RECEIPT_SCHEMA_VERSION = "inpaint-source-ocr-runtime-receipt-v34"

TRANSLATE_ACTION = "translate_inpaint"
REVIEW_ACTION = "review"
SEEDLESS_ROUTE_CLASS = "clean_translucent"
OCR_SUCCESS_STATUSES = frozenset({"ok", "ok_after_retry"})

def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _path_value(value: object, *, relative_to: Path) -> Path | None:
    raw: object = value
    if isinstance(value, Mapping):
        raw = value.get("path")
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text)
    return (path if path.is_absolute() else relative_to / path).resolve()


def _read_rgb(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        raise FileNotFoundError(path)
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _read_ownership(path: Path, shape: tuple[int, int]) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    value = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if value is None or value.size == 0:
        raise FileNotFoundError(path)
    if value.shape != shape:
        raise ValueError(f"ownership shape differs: {value.shape} != {shape}")
    unique = np.unique(value)
    if np.any((unique != 0) & (unique != 255)):
        raise ValueError("source ownership mask must be binary")
    return np.ascontiguousarray(value)


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _tracked_dependency_identity(*, require_clean: bool) -> dict[str, object]:
    head = _git_head()
    branch = _git("branch", "--show-current").stdout.strip()
    tracked_status = _git("status", "--porcelain", "--untracked-files=no")
    clean = tracked_status.returncode == 0 and not tracked_status.stdout.strip()
    dependencies: list[dict[str, object]] = []
    failures: list[str] = []
    for relative in TRACKED_RUNTIME_DEPENDENCIES:
        path = ROOT / relative
        tracked = _git("ls-files", "--error-unmatch", "--", relative)
        head_blob = _git("rev-parse", f"HEAD:{relative}")
        unchanged = _git("diff", "--quiet", "HEAD", "--", relative)
        record = {
            "path": relative,
            "tracked": tracked.returncode == 0,
            "unchanged_from_head": unchanged.returncode == 0,
            "head_blob_id": (
                head_blob.stdout.strip() if head_blob.returncode == 0 else ""
            ),
            "working_file_sha256": _file_sha256(path) if path.is_file() else "",
        }
        dependencies.append(record)
        if (
            not record["tracked"]
            or not record["unchanged_from_head"]
            or not record["head_blob_id"]
            or not _is_sha256(record["working_file_sha256"])
        ):
            failures.append(relative)
    payload: dict[str, object] = {
        "git_head": head,
        "git_branch": branch,
        "tracked_worktree_clean": clean,
        "dependency_count": len(dependencies),
        "dependencies": dependencies,
    }
    payload["dependency_inventory_sha256"] = _canonical_sha256(dependencies)
    if require_clean and (not head or not branch or not clean or failures):
        raise RuntimeError(
            "Official source OCR evidence requires a clean committed HEAD and "
            "unchanged tracked dependencies; invalid=" + ",".join(failures)
        )
    return payload


@dataclass(frozen=True, slots=True)
class SourceRegion:
    region_id: str
    ownership_path: Path
    ownership_file_sha256: str
    route_class: str
    source_reviewed: bool


@dataclass(frozen=True, slots=True)
class SourcePage:
    page_id: str
    source_path: Path
    source_sha256: str
    regions: tuple[SourceRegion, ...]


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    provider: str
    detector: Mapping[str, object]
    ocr: Mapping[str, object]
    code: Mapping[str, object] = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        return _canonical_sha256(
            {
                "provider": self.provider,
                "detector": dict(self.detector),
                "ocr": dict(self.ocr),
                "code": dict(self.code),
            }
        )

    @property
    def provider_tag(self) -> str:
        return f"{self.provider}:{self.sha256}"


@dataclass(slots=True)
class RuntimeBlock:
    xyxy: tuple[int, int, int, int]
    text_class: str
    detector_confidence: float = 0.0
    confidence_receipt: Mapping[str, object] = field(default_factory=dict)
    text: str = ""
    ocr_status: str = ""
    ocr_confidence: float = 0.0
    ocr_strategy: str = ""
    ocr_model_identity: str = ""
    ocr_runtime_identity: str = ""
    semantic_role: str = ""
    processing_action: str = ""
    processing_decision_source: str = ""
    processing_decision_reasons: tuple[str, ...] = ()
    native: Any = None


class SourceEvidenceRuntime(Protocol):
    identity: RuntimeIdentity

    def detect(self, image_rgb: np.ndarray) -> list[RuntimeBlock]: ...

    def recognize(
        self,
        image_rgb: np.ndarray,
        blocks: Sequence[RuntimeBlock],
    ) -> None: ...


def _normalized_route_class(raw: object) -> str:
    value = str(raw or "").strip().lower()
    if value in {"clean_translucent", "translucent"}:
        return SEEDLESS_ROUTE_CLASS
    return value or "ambiguous"


def _load_source_manifest(
    manifest_path: Path,
) -> tuple[dict[str, object], tuple[SourcePage, ...]]:
    """Load only source paths, identities, routes, and ownership masks."""

    manifest_path = manifest_path.resolve()
    payload = _read_json(manifest_path)
    seal_path = manifest_path.with_suffix(manifest_path.suffix + ".seal.json")
    seal = _read_json(seal_path)
    manifest_sha = _file_sha256(manifest_path)
    if payload.get("schema_version") != SOURCE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("v3.4 source OCR export requires a source-only v4 manifest")
    if seal.get("manifest_sha256") != manifest_sha:
        raise ValueError("source manifest seal SHA differs")
    if (
        payload.get("candidate_seen") is not False
        or payload.get("annotation_frozen_before_candidate") is not True
        or seal.get("candidate_generated") is not False
        or seal.get("candidate_seen") is not False
        or seal.get("annotation_frozen_before_candidate") is not True
    ):
        raise ValueError("source manifest is not frozen before candidate generation")
    inventory_sha = str(payload.get("page_inventory_sha256") or "").lower()
    if not _is_sha256(inventory_sha):
        raise ValueError("source manifest page inventory SHA is invalid")
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        raise ValueError("source manifest pages are missing")
    if payload.get("page_count") != len(raw_pages):
        raise ValueError("source manifest page count differs")

    pages: list[SourcePage] = []
    page_ids: set[str] = set()
    for raw_page in raw_pages:
        if not isinstance(raw_page, Mapping):
            raise ValueError("source manifest page is invalid")
        page_id = str(raw_page.get("page_id") or "").strip()
        source_sha = str(raw_page.get("source_sha256") or "").strip().lower()
        source_path = _path_value(raw_page.get("path"), relative_to=manifest_path.parent)
        if (
            not page_id
            or page_id in page_ids
            or not _is_sha256(source_sha)
            or source_path is None
        ):
            raise ValueError("source manifest page identity is invalid")
        page_ids.add(page_id)
        if (
            raw_page.get("candidate_seen") is not False
            or raw_page.get("annotation_frozen_before_candidate") is not True
        ):
            raise ValueError("source manifest page is not source-only")
        if not source_path.is_file() or _file_sha256(source_path) != source_sha:
            raise ValueError("source image SHA differs")

        raw_regions = raw_page.get("regions")
        if not isinstance(raw_regions, list):
            raise ValueError("source manifest regions are invalid")
        artifact_registry = raw_page.get("artifact_sha256")
        artifact_regions = (
            artifact_registry.get("regions", {})
            if isinstance(artifact_registry, Mapping)
            else {}
        )
        if not isinstance(artifact_regions, Mapping):
            artifact_regions = {}
        regions: list[SourceRegion] = []
        region_ids: set[str] = set()
        for raw_region in raw_regions:
            if not isinstance(raw_region, Mapping):
                raise ValueError("source manifest region is invalid")
            region_id = str(raw_region.get("region_id") or "").strip()
            ownership_path = _path_value(
                raw_region.get("ownership_mask"),
                relative_to=manifest_path.parent,
            )
            if not region_id or region_id in region_ids or ownership_path is None:
                raise ValueError("source manifest region identity is invalid")
            region_ids.add(region_id)
            ownership_sha = _file_sha256(ownership_path)
            declared = artifact_regions.get(region_id)
            declared_ownership_sha = (
                str(declared.get("ownership_mask") or "").lower()
                if isinstance(declared, Mapping)
                else ""
            )
            if declared_ownership_sha and declared_ownership_sha != ownership_sha:
                raise ValueError("source ownership mask SHA differs")
            regions.append(
                SourceRegion(
                    region_id=region_id,
                    ownership_path=ownership_path,
                    ownership_file_sha256=ownership_sha,
                    route_class=_normalized_route_class(
                        raw_region.get("bubble_route_class")
                    ),
                    source_reviewed=raw_region.get("source_reviewed") is True,
                )
            )
        pages.append(
            SourcePage(
                page_id=page_id,
                source_path=source_path,
                source_sha256=source_sha,
                regions=tuple(regions),
            )
        )
    source_inventory_sha = _canonical_sha256(_source_inventory_rows(pages))
    return (
        {
            "manifest_sha256": manifest_sha,
            "manifest_seal_sha256": _file_sha256(seal_path),
            "page_inventory_sha256": inventory_sha,
            "page_count": len(pages),
            "source_region_inventory_sha256": source_inventory_sha,
        },
        tuple(pages),
    )


def _bbox(block: RuntimeBlock, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    h, w = shape
    try:
        x1, y1, x2, y2 = [int(round(float(value))) for value in block.xyxy]
    except (TypeError, ValueError):
        return None
    x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _script_from_text(text: str) -> str:
    scripts: set[str] = set()
    for char in str(text or ""):
        code = ord(char)
        if 0x3040 <= code <= 0x30FF:
            scripts.add("Japanese")
        elif 0x4E00 <= code <= 0x9FFF:
            scripts.add("CJK")
        elif 0xAC00 <= code <= 0xD7AF:
            scripts.add("Korean")
        elif char.isalpha() and code < 0x0250:
            scripts.add("Latin")
    if not scripts:
        return ""
    if "Japanese" in scripts:
        scripts.discard("CJK")
    return next(iter(scripts)) if len(scripts) == 1 else "Mixed"


def _canonical_block_id(source_sha: str, block: RuntimeBlock) -> str:
    return "product-block-" + _canonical_sha256(
        {
            "source_sha256": source_sha,
            "xyxy": list(block.xyxy),
            "text_class": block.text_class,
            "confidence_receipt": dict(block.confidence_receipt),
        }
    )[:32]


def _block_owners(
    block: RuntimeBlock,
    ownership: Mapping[str, np.ndarray],
    shape: tuple[int, int],
) -> tuple[str, ...]:
    box = _bbox(block, shape)
    if box is None:
        return ()
    x1, y1, x2, y2 = box
    return tuple(
        sorted(
            region_id
            for region_id, mask in ownership.items()
            if np.any(mask[y1:y2, x1:x2] > 0)
        )
    )


def _owner_binding(
    *,
    page: SourcePage,
    region: SourceRegion,
    block: RuntimeBlock,
    canonical_block_id: str,
    owners: Sequence[str],
) -> dict[str, object]:
    return {
        "page_id": page.page_id,
        "source_sha256": page.source_sha256,
        "region_id": region.region_id,
        "ownership_mask_sha256": region.ownership_file_sha256,
        "canonical_block_id": canonical_block_id,
        "block_xyxy": list(block.xyxy),
        "block_text_class": block.text_class,
        "owner_region_ids": list(owners),
    }


def _region_record(
    *,
    page: SourcePage,
    region: SourceRegion,
    block: RuntimeBlock,
    owners: Sequence[str],
    owner_count: int,
    runtime_identity: RuntimeIdentity,
) -> dict[str, object]:
    canonical_id = _canonical_block_id(page.source_sha256, block)
    owner = _owner_binding(
        page=page,
        region=region,
        block=block,
        canonical_block_id=canonical_id,
        owners=owners,
    )
    text = str(block.text or "").strip()
    action = str(block.processing_action or REVIEW_ACTION).strip().lower()
    route = region.route_class
    detector_confidence = float(block.detector_confidence or 0.0)
    ocr_confidence = float(block.ocr_confidence or 0.0)
    if math.isfinite(detector_confidence) and 0.0 < detector_confidence <= 1.0:
        confidence = detector_confidence
        confidence_kind = "detector_block_confidence"
    elif math.isfinite(ocr_confidence) and 0.0 < ocr_confidence <= 1.0:
        confidence = ocr_confidence
        confidence_kind = "recognizer_confidence"
    else:
        confidence = 0.0
        confidence_kind = "unavailable"
    ocr_artifact = {
        "source_sha256": page.source_sha256,
        "canonical_block_id": canonical_id,
        "text": text,
        "status": block.ocr_status,
        "strategy": block.ocr_strategy,
        "model_identity": block.ocr_model_identity,
        "runtime_identity": block.ocr_runtime_identity,
    }
    confidence_artifact = {
        "canonical_block_id": canonical_id,
        "confidence": confidence,
        "confidence_kind": confidence_kind,
        "receipt": dict(block.confidence_receipt),
        "detector_runtime_sha256": _canonical_sha256(
            dict(runtime_identity.detector)
        ),
    }
    action_artifact = {
        "canonical_block_id": canonical_id,
        "semantic_role": block.semantic_role,
        "processing_action": action,
        "decision_source": block.processing_decision_source,
        "decision_reasons": list(block.processing_decision_reasons),
    }
    route_artifact = {
        "page_id": page.page_id,
        "region_id": region.region_id,
        "route_class": route,
        "source_reviewed": region.source_reviewed,
        "candidate_seen": False,
    }
    authoritative_ocr = bool(
        text
        and str(block.ocr_status or "").strip().lower() in OCR_SUCCESS_STATUSES
    )
    return {
        "region_id": region.region_id,
        "owner_region_id": region.region_id if owner_count == 1 else "",
        "owner_count": int(owner_count),
        "owner_binding_kind": (
            "canonical_block_region_id" if owner_count == 1 else "conflict"
        ),
        "canonical_block_id": canonical_id,
        "ocr_text": text,
        "ocr_script": _script_from_text(text),
        "provider": runtime_identity.provider_tag,
        "ocr_confidence": confidence,
        "confidence_kind": confidence_kind,
        "processing_action": action,
        "route_class": route,
        "authoritative_ocr": authoritative_ocr,
        "candidate_seen": False,
        "annotation_frozen_before_candidate": True,
        "provenance": {
            "ocr_artifact_sha256": _canonical_sha256(ocr_artifact),
            "confidence_artifact_sha256": _canonical_sha256(
                confidence_artifact
            ),
            "owner_binding_sha256": _canonical_sha256(owner),
            "action_artifact_sha256": _canonical_sha256(action_artifact),
            "route_artifact_sha256": _canonical_sha256(route_artifact),
        },
    }


def _load_reuse_records(
    evidence_path: Path | None,
    *,
    binding: Mapping[str, object],
    identity: RuntimeIdentity,
) -> dict[tuple[str, str], Mapping[str, object]]:
    if evidence_path is None:
        return {}
    evidence_path = evidence_path.resolve()
    seal_path = evidence_path.with_suffix(evidence_path.suffix + ".seal.json")
    receipt_path = evidence_path.with_suffix(evidence_path.suffix + ".receipt.json")
    try:
        evidence = _read_json(evidence_path)
        seal = _read_json(seal_path)
        receipt = _read_json(receipt_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    evidence_payload_sha = _canonical_sha256(evidence)
    receipt_payload_sha = _canonical_sha256(receipt)
    record_inventory_sha = _canonical_sha256(
        _record_inventory_rows(
            [
                page
                for page in evidence.get("pages", [])
                if isinstance(page, Mapping)
            ]
        )
    )
    if (
        seal.get("schema_version") != SOURCE_EVIDENCE_SEAL_SCHEMA_VERSION
        or seal.get("evidence_file_sha256") != _file_sha256(evidence_path)
        or seal.get("receipt_file_sha256") != _file_sha256(receipt_path)
        or seal.get("evidence_payload_sha256") != evidence_payload_sha
        or seal.get("receipt_payload_sha256") != receipt_payload_sha
        or seal.get("record_inventory_sha256") != record_inventory_sha
        or seal.get("source_region_inventory_sha256")
        != binding["source_region_inventory_sha256"]
        or seal.get("complete_page_set") is not True
        or seal.get("page_count") != binding["page_count"]
        or seal.get("source_manifest_sha256") != binding["manifest_sha256"]
        or seal.get("provider_identity_sha256") != identity.sha256
        or evidence.get("schema_version") != SOURCE_EVIDENCE_SCHEMA_VERSION
        or evidence.get("source_manifest_sha256") != binding["manifest_sha256"]
        or evidence.get("source_page_inventory_sha256")
        != binding["page_inventory_sha256"]
        or evidence.get("producer") != identity.provider_tag
        or evidence.get("candidate_seen") is not False
        or evidence.get("annotation_frozen_before_candidate") is not True
        or receipt.get("schema_version")
        != SOURCE_EVIDENCE_RECEIPT_SCHEMA_VERSION
        or receipt.get("evidence_file_sha256") != _file_sha256(evidence_path)
        or receipt.get("evidence_payload_sha256") != evidence_payload_sha
        or receipt.get("source_manifest_sha256") != binding["manifest_sha256"]
        or receipt.get("source_page_inventory_sha256")
        != binding["page_inventory_sha256"]
        or receipt.get("provider_identity_sha256") != identity.sha256
        or receipt.get("provider") != identity.provider_tag
        or receipt.get("source_region_inventory_sha256")
        != binding["source_region_inventory_sha256"]
        or receipt.get("record_inventory_sha256") != record_inventory_sha
        or receipt.get("candidate_seen") is not False
        or receipt.get("annotation_frozen_before_candidate") is not True
    ):
        return {}
    records: dict[tuple[str, str], Mapping[str, object]] = {}
    for raw_page in evidence.get("pages", []):
        if not isinstance(raw_page, Mapping):
            return {}
        page_id = str(raw_page.get("page_id") or "")
        for raw_region in raw_page.get("regions", []):
            if not isinstance(raw_region, Mapping):
                return {}
            region_id = str(raw_region.get("region_id") or "")
            identity_key = (page_id, region_id)
            if not page_id or not region_id or identity_key in records:
                return {}
            records[identity_key] = raw_region
    return records


def _cache_matches(
    cached: Mapping[str, object],
    current: Mapping[str, object],
    *,
    provider: str,
) -> bool:
    cached_provenance = cached.get("provenance")
    current_provenance = current.get("provenance")
    if not isinstance(cached_provenance, Mapping) or not isinstance(
        current_provenance, Mapping
    ):
        return False
    return bool(
        cached.get("provider") == provider
        and cached.get("region_id") == current.get("region_id")
        and cached_provenance.get("owner_binding_sha256")
        == current_provenance.get("owner_binding_sha256")
        and cached_provenance.get("confidence_artifact_sha256")
        == current_provenance.get("confidence_artifact_sha256")
        and cached_provenance.get("route_artifact_sha256")
        == current_provenance.get("route_artifact_sha256")
    )


def _source_inventory_rows(pages: Sequence[SourcePage]) -> list[dict[str, object]]:
    return [
        {
            "page_id": page.page_id,
            "source_sha256": page.source_sha256,
            "regions": [
                {
                    "region_id": region.region_id,
                    "ownership_mask_sha256": region.ownership_file_sha256,
                    "route_class": region.route_class,
                    "source_reviewed": region.source_reviewed,
                }
                for region in page.regions
            ],
        }
        for page in pages
    ]


def _record_inventory_rows(
    evidence_pages: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for page in evidence_pages:
        page_id = str(page.get("page_id") or "")
        source_sha = str(page.get("source_sha256") or "")
        raw_regions = page.get("regions")
        if not isinstance(raw_regions, list):
            continue
        for raw in raw_regions:
            if not isinstance(raw, Mapping):
                continue
            provenance = raw.get("provenance")
            rows.append(
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
    return sorted(
        rows,
        key=lambda row: (str(row["page_id"]), str(row["region_id"])),
    )


def build_source_product_evidence(
    manifest_path: Path,
    runtime: SourceEvidenceRuntime,
    *,
    reuse_evidence_path: Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    binding, pages = _load_source_manifest(manifest_path)
    source_inventory = _source_inventory_rows(pages)
    source_inventory_sha = _canonical_sha256(source_inventory)
    reused = _load_reuse_records(
        reuse_evidence_path,
        binding=binding,
        identity=runtime.identity,
    )
    output_pages: list[dict[str, object]] = []
    receipt_pages: list[dict[str, object]] = []
    totals: Counter[str] = Counter()

    for page in pages:
        image = _read_rgb(page.source_path)
        shape = image.shape[:2]
        ownership = {
            region.region_id: _read_ownership(region.ownership_path, shape)
            for region in page.regions
        }
        region_by_id = {region.region_id: region for region in page.regions}
        try:
            blocks = runtime.detect(image)
        except Exception as exc:
            output_pages.append(
                {
                    "page_id": page.page_id,
                    "source_sha256": page.source_sha256,
                    "candidate_seen": False,
                    "annotation_frozen_before_candidate": True,
                    "regions": [],
                }
            )
            receipt_pages.append(
                {
                    "page_id": page.page_id,
                    "source_sha256": page.source_sha256,
                    "status": "detector_failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "detected_block_count": 0,
                    "ocr_request_block_count": 0,
                    "cache_reuse_count": 0,
                    "source_regions": source_inventory[
                        len(receipt_pages)
                    ]["regions"],
                }
            )
            totals["detector_failed_page_count"] += 1
            continue

        owners_by_index = {
            index: _block_owners(block, ownership, shape)
            for index, block in enumerate(blocks)
        }
        blocks_by_region: dict[str, list[int]] = defaultdict(list)
        for index, owners in owners_by_index.items():
            for owner in owners:
                blocks_by_region[owner].append(index)

        final_records: dict[str, dict[str, object]] = {}
        pending: list[tuple[str, int]] = []
        cache_count = 0
        conflict_count = 0
        for region_id in sorted(region_by_id):
            region = region_by_id[region_id]
            indices = blocks_by_region.get(region_id, [])
            if not indices:
                totals["missing_region_count"] += 1
                continue
            block = blocks[indices[0]]
            owners = owners_by_index[indices[0]]
            owner_count = max(len(indices), len(owners))
            if len(indices) != 1 or len(owners) != 1 or owners[0] != region_id:
                conflict_count += 1
                block.processing_action = REVIEW_ACTION
                record = _region_record(
                    page=page,
                    region=region,
                    block=block,
                    owners=owners,
                    owner_count=max(2, owner_count),
                    runtime_identity=runtime.identity,
                )
                final_records[region_id] = record
                totals["ownership_conflict_region_count"] += 1
                continue

            pre_ocr_record = _region_record(
                page=page,
                region=region,
                block=block,
                owners=owners,
                owner_count=1,
                runtime_identity=runtime.identity,
            )
            cached = reused.get((page.page_id, region_id))
            if cached is not None and _cache_matches(
                cached,
                pre_ocr_record,
                provider=runtime.identity.provider_tag,
            ):
                final_records[region_id] = dict(cached)
                cache_count += 1
                totals["cache_reuse_region_count"] += 1
            else:
                pending.append((region_id, indices[0]))

        if pending:
            pending_blocks = [blocks[index] for _, index in pending]
            try:
                runtime.recognize(image, pending_blocks)
            except Exception as exc:
                for block in pending_blocks:
                    block.text = ""
                    block.ocr_status = "runtime_error"
                    block.processing_action = REVIEW_ACTION
                    block.processing_decision_source = "ocr_runtime_error"
                    block.processing_decision_reasons = (
                        f"{type(exc).__name__}: {exc}",
                    )
                totals["ocr_failed_page_count"] += 1
            for region_id, index in pending:
                block = blocks[index]
                final_records[region_id] = _region_record(
                    page=page,
                    region=region_by_id[region_id],
                    block=block,
                    owners=owners_by_index[index],
                    owner_count=1,
                    runtime_identity=runtime.identity,
                )

        ordered_records = [final_records[key] for key in sorted(final_records)]
        output_pages.append(
            {
                "page_id": page.page_id,
                "source_sha256": page.source_sha256,
                "candidate_seen": False,
                "annotation_frozen_before_candidate": True,
                "regions": ordered_records,
            }
        )
        receipt_pages.append(
            {
                "page_id": page.page_id,
                "source_sha256": page.source_sha256,
                "status": "completed",
                "detected_block_count": len(blocks),
                "exported_region_count": len(ordered_records),
                "missing_region_count": len(page.regions) - len(ordered_records),
                "ownership_conflict_region_count": conflict_count,
                "ocr_request_block_count": len(pending),
                "cache_reuse_count": cache_count,
                "source_regions": source_inventory[len(receipt_pages)][
                    "regions"
                ],
            }
        )
        totals["page_count"] += 1
        totals["detected_block_count"] += len(blocks)
        totals["exported_region_count"] += len(ordered_records)
        totals["ocr_request_block_count"] += len(pending)

    evidence: dict[str, object] = {
        "schema_version": SOURCE_EVIDENCE_SCHEMA_VERSION,
        "source_manifest_sha256": binding["manifest_sha256"],
        "source_page_inventory_sha256": binding["page_inventory_sha256"],
        "candidate_seen": False,
        "annotation_frozen_before_candidate": True,
        "producer": runtime.identity.provider_tag,
        "pages": output_pages,
    }
    evidence_payload_sha = _canonical_sha256(evidence)
    record_inventory = _record_inventory_rows(output_pages)
    record_inventory_sha = _canonical_sha256(record_inventory)
    receipt: dict[str, object] = {
        "schema_version": SOURCE_EVIDENCE_RECEIPT_SCHEMA_VERSION,
        "source_manifest_sha256": binding["manifest_sha256"],
        "source_manifest_seal_sha256": binding["manifest_seal_sha256"],
        "source_page_inventory_sha256": binding["page_inventory_sha256"],
        "candidate_seen": False,
        "annotation_frozen_before_candidate": True,
        "git_head": _git_head(),
        "provider": runtime.identity.provider_tag,
        "provider_identity_sha256": runtime.identity.sha256,
        "evidence_payload_sha256": evidence_payload_sha,
        "source_region_inventory_sha256": source_inventory_sha,
        "record_inventory_sha256": record_inventory_sha,
        "source_region_inventory": source_inventory,
        "record_inventory": record_inventory,
        "detector_identity": dict(runtime.identity.detector),
        "detector_identity_sha256": _canonical_sha256(
            dict(runtime.identity.detector)
        ),
        "ocr_identity": dict(runtime.identity.ocr),
        "ocr_identity_sha256": _canonical_sha256(dict(runtime.identity.ocr)),
        "tracked_dependency_identity": dict(runtime.identity.code),
        "tracked_dependency_identity_sha256": _canonical_sha256(
            dict(runtime.identity.code)
        ),
        "reuse_evidence_sha256": (
            _file_sha256(reuse_evidence_path.resolve())
            if reuse_evidence_path is not None and reuse_evidence_path.is_file()
            else None
        ),
        "summary": dict(sorted(totals.items())),
        "pages": receipt_pages,
    }
    return evidence, receipt


def write_source_product_evidence(
    manifest_path: Path,
    output_path: Path,
    runtime: SourceEvidenceRuntime,
    *,
    reuse_evidence_path: Path | None = None,
) -> tuple[Path, Path, Path, dict[str, object]]:
    output_path = output_path.resolve()
    receipt_path = output_path.with_suffix(output_path.suffix + ".receipt.json")
    seal_path = output_path.with_suffix(output_path.suffix + ".seal.json")
    for path in (output_path, receipt_path, seal_path):
        if path.exists():
            raise FileExistsError(f"source OCR evidence output must be fresh: {path}")
    evidence, receipt = build_source_product_evidence(
        manifest_path.resolve(),
        runtime,
        reuse_evidence_path=(
            reuse_evidence_path.resolve() if reuse_evidence_path is not None else None
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f".{output_path.name}.partial")
    temporary_receipt = receipt_path.with_name(f".{receipt_path.name}.partial")
    temporary_seal = seal_path.with_name(f".{seal_path.name}.partial")
    temporary_output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    receipt["evidence_file_sha256"] = _file_sha256(temporary_output)
    temporary_receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    seal = {
        "schema_version": SOURCE_EVIDENCE_SEAL_SCHEMA_VERSION,
        "evidence_file_sha256": _file_sha256(temporary_output),
        "receipt_file_sha256": _file_sha256(temporary_receipt),
        "evidence_payload_sha256": receipt["evidence_payload_sha256"],
        "receipt_payload_sha256": _canonical_sha256(receipt),
        "source_region_inventory_sha256": receipt[
            "source_region_inventory_sha256"
        ],
        "record_inventory_sha256": receipt["record_inventory_sha256"],
        "source_manifest_sha256": evidence["source_manifest_sha256"],
        "source_page_inventory_sha256": evidence[
            "source_page_inventory_sha256"
        ],
        "provider_identity_sha256": runtime.identity.sha256,
        "tracked_dependency_identity_sha256": receipt[
            "tracked_dependency_identity_sha256"
        ],
        "page_count": len(evidence["pages"]),
        "complete_page_set": True,
        "candidate_generated": False,
        "candidate_seen": False,
        "annotation_frozen_before_candidate": True,
    }
    temporary_seal.write_text(
        json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_output.replace(output_path)
    temporary_receipt.replace(receipt_path)
    temporary_seal.replace(seal_path)
    return output_path, receipt_path, seal_path, receipt


class _ProductSettings:
    def __init__(
        self,
        *,
        server_url: str,
        use_gpu: bool,
        max_new_tokens: int,
        parallel_workers: int,
    ) -> None:
        self.server_url = server_url
        self.use_gpu = bool(use_gpu)
        self.max_new_tokens = int(max_new_tokens)
        self.parallel_workers = int(parallel_workers)

    def get_tool_selection(self, tool_type: str) -> str:
        if tool_type == "detector":
            return "RT-DETR-v2"
        if tool_type == "ocr":
            return "PaddleOCR VL"
        if tool_type == "translator":
            return ""
        raise KeyError(tool_type)

    def is_gpu_enabled(self) -> bool:
        return self.use_gpu

    def get_paddleocr_vl_settings(self) -> dict[str, object]:
        return {
            "server_url": self.server_url,
            "prettify_markdown": False,
            "visualize": False,
            "max_new_tokens": self.max_new_tokens,
            "parallel_workers": self.parallel_workers,
        }

    def get_ocr_generic_settings(self) -> dict[str, object]:
        return {"paddleocr_vl_scheduler_mode": "fixed_area_desc"}


class ProductPaddleCropRuntime:
    """The current product RT-DETR + PaddleOCR-VL crop path."""

    def __init__(
        self,
        *,
        server_url: str = DEFAULT_PADDLE_DIRECT_SERVER_URL,
        source_language: str = "ja",
        use_gpu: bool = True,
        ensure_managed_runtime: bool = False,
        max_new_tokens: int = 1024,
        parallel_workers: int = 1,
    ) -> None:
        code_identity = _tracked_dependency_identity(require_clean=True)
        self.settings = _ProductSettings(
            server_url=server_url,
            use_gpu=use_gpu,
            max_new_tokens=max_new_tokens,
            parallel_workers=parallel_workers,
        )
        self.source_language = str(source_language or "").strip() or "ja"
        manager = LocalOCRRuntimeManager()
        if ensure_managed_runtime:
            manager.ensure_engine("PaddleOCR VL", self.settings)
        runtime_identity = manager.get_ocr_cache_identity(
            "PaddleOCR VL", self.settings
        )
        health = manager.probe_managed_engine(
            "PaddleOCR VL", self.settings, timeout_sec=3
        )
        if runtime_identity is None or health != "healthy":
            raise RuntimeError(
                "Pinned managed PaddleOCR-VL crop runtime is not healthy. "
                "Prepare/start the bundled Docker runtime before exporting."
            )

        self.detector = TextBlockDetector(self.settings)
        self.detector_engine = DetectionEngineFactory.create_engine(
            self.settings,
            "RT-DETR-v2",
            backend="onnx",
        )
        detector_path = Path(
            ModelDownloader.get_file_path(ModelID.RTDETR_V2_ONNX, "detector.onnx")
        ).resolve()
        detector_sha = _file_sha256(detector_path)
        expected_detector_sha = str(
            ModelDownloader.registry[ModelID.RTDETR_V2_ONNX].sha256[0] or ""
        ).lower()
        if detector_sha != expected_detector_sha:
            raise RuntimeError("RT-DETR ONNX asset SHA differs from product registry")
        session = getattr(self.detector_engine, "session", None)
        providers = (
            list(session.get_providers())
            if session is not None and callable(getattr(session, "get_providers", None))
            else []
        )
        slicer = self.detector_engine.image_slicer
        detector_identity = {
            "provider": "RT-DETR-v2-ONNX",
            "model_sha256": detector_sha,
            "expected_model_sha256": expected_detector_sha,
            "providers": providers,
            "confidence_threshold": float(self.detector_engine.confidence_threshold),
            "preprocess": {
                "resize": [640, 640],
                "scale": "uint8_to_float32_div_255",
                "layout": "NCHW_RGB",
                "slicer": {
                    "height_to_width_ratio_threshold": slicer.height_to_width_ratio_threshold,
                    "target_slice_ratio": slicer.target_slice_ratio,
                    "overlap_height_ratio": slicer.overlap_height_ratio,
                    "min_slice_height_ratio": slicer.min_slice_height_ratio,
                },
            },
            "code_sha256": _file_sha256(
                ROOT / "modules" / "detection" / "rtdetr_v2_onnx.py"
            ),
        }
        ocr_identity = {
            **dict(runtime_identity),
            "engine_class": "PaddleOCRVLEngine",
            "engine_code_sha256": _file_sha256(
                ROOT / "modules" / "ocr" / "paddle_crop" / "engine.py"
            ),
        }
        self.identity = RuntimeIdentity(
            provider="product-rtdetr-onnx+paddleocr-vl-crop",
            detector=detector_identity,
            ocr=ocr_identity,
            code=code_identity,
        )
        self.ocr = PaddleOCRVLEngine()
        self.ocr.initialize(self.settings)

    def _scored_single(
        self,
        image_rgb: np.ndarray,
        *,
        offset_y: int,
    ) -> list[dict[str, object]]:
        pil_image = Image.fromarray(image_rgb)
        resized = pil_image.resize((640, 640))
        data = np.asarray(resized, dtype=np.float32) / 255.0
        data = np.transpose(data, (2, 0, 1))[np.newaxis, ...]
        width, height = pil_image.size
        outputs = self.detector_engine.session.run(
            None,
            {
                "images": data,
                "orig_target_sizes": np.array([[width, height]], dtype=np.int64),
            },
        )
        labels, boxes, scores = outputs[:3]
        if labels.ndim == 2 and labels.shape[0] == 1:
            labels = labels[0]
        if boxes.ndim == 3 and boxes.shape[0] == 1:
            boxes = boxes[0]
        if scores.ndim == 2 and scores.shape[0] == 1:
            scores = scores[0]
        result: list[dict[str, object]] = []
        for label, box, score in zip(labels, boxes, scores):
            label_id = int(label)
            confidence = float(score)
            if (
                label_id not in {1, 2}
                or confidence < float(self.detector_engine.confidence_threshold)
            ):
                continue
            x1, y1, x2, y2 = [int(value) for value in box]
            result.append(
                {
                    "xyxy": [x1, y1 + offset_y, x2, y2 + offset_y],
                    "label_id": label_id,
                    "score": confidence,
                }
            )
        return result

    def _scored_proposals(self, image_rgb: np.ndarray) -> list[dict[str, object]]:
        slicer = self.detector_engine.image_slicer
        if not slicer.should_slice(image_rgb):
            return self._scored_single(image_rgb, offset_y=0)
        _width, slice_height, effective, _declared_count = (
            slicer.calculate_slice_params(image_rgb)
        )
        count = math.ceil(image_rgb.shape[0] / effective)
        proposals: list[dict[str, object]] = []
        for index in range(count):
            image_slice, start_y, _end_y = slicer.get_slice(
                image_rgb,
                index,
                effective,
                slice_height,
            )
            proposals.extend(self._scored_single(image_slice, offset_y=start_y))
        return proposals

    @staticmethod
    def _score_for_block(
        xyxy: tuple[int, int, int, int],
        proposals: Sequence[Mapping[str, object]],
    ) -> tuple[float, Mapping[str, object]]:
        x1, y1, x2, y2 = xyxy
        block_area = max(1, (x2 - x1) * (y2 - y1))
        matches: list[tuple[float, float, Mapping[str, object]]] = []
        for proposal in proposals:
            try:
                px1, py1, px2, py2 = [int(v) for v in proposal["xyxy"]]
                score = float(proposal["score"])
            except (KeyError, TypeError, ValueError):
                continue
            ix1, iy1 = max(x1, px1), max(y1, py1)
            ix2, iy2 = min(x2, px2), min(y2, py2)
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            proposal_area = max(1, (px2 - px1) * (py2 - py1))
            containment = intersection / float(min(block_area, proposal_area))
            if containment >= 0.5:
                matches.append((containment, score, proposal))
        if not matches:
            return 0.0, {}
        containment, score, proposal = max(matches, key=lambda item: (item[1], item[0]))
        return score, {
            "proposal": dict(proposal),
            "containment_ratio": containment,
            "scored_proposal_count": len(matches),
        }

    def detect(self, image_rgb: np.ndarray) -> list[RuntimeBlock]:
        native_blocks = self.detector.detect(image_rgb) or []
        proposals = self._scored_proposals(image_rgb)
        result: list[RuntimeBlock] = []
        for native in native_blocks:
            try:
                xyxy = tuple(int(float(value)) for value in native.xyxy)
            except (TypeError, ValueError):
                continue
            if len(xyxy) != 4:
                continue
            confidence, receipt = self._score_for_block(xyxy, proposals)
            result.append(
                RuntimeBlock(
                    xyxy=xyxy,
                    text_class=str(getattr(native, "text_class", "") or ""),
                    detector_confidence=confidence,
                    confidence_receipt={
                        **dict(receipt),
                        "model_sha256": self.identity.detector["model_sha256"],
                        "runtime_sha256": _canonical_sha256(
                            dict(self.identity.detector)
                        ),
                    },
                    native=native,
                )
            )
        return result

    def recognize(
        self,
        image_rgb: np.ndarray,
        blocks: Sequence[RuntimeBlock],
    ) -> None:
        native_blocks: list[TextBlock] = []
        by_native: dict[int, RuntimeBlock] = {}
        for block in blocks:
            native = block.native
            if not isinstance(native, TextBlock):
                raise TypeError("product runtime block lost its native TextBlock")
            native.source_lang = self.source_language
            initialize_ocr_result_contract(
                native,
                strategy=OCR_STRATEGY_PADDLE_CROP,
                model_identity=str(
                    self.identity.ocr.get("model_name")
                    or self.identity.ocr.get("model_file")
                    or PaddleOCRVLEngine.MODEL_IDENTITY
                ),
                runtime_identity=_canonical_sha256(dict(self.identity.ocr)),
            )
            native_blocks.append(native)
            by_native[id(native)] = block
        self.ocr.process_image(image_rgb, native_blocks)
        for native in native_blocks:
            finalize_ocr_processing_contract(native)
            block = by_native[id(native)]
            block.text = str(getattr(native, "text", "") or "")
            block.ocr_status = str(getattr(native, "ocr_status", "") or "")
            block.ocr_confidence = float(
                getattr(native, "ocr_confidence", 0.0) or 0.0
            )
            block.ocr_strategy = str(getattr(native, "ocr_strategy", "") or "")
            block.ocr_model_identity = str(
                getattr(native, "ocr_model_identity", "") or ""
            )
            block.ocr_runtime_identity = str(
                getattr(native, "ocr_runtime_identity", "") or ""
            )
            block.semantic_role = str(
                getattr(native, "semantic_role", "") or ""
            )
            block.processing_action = str(
                getattr(native, "processing_action", "") or ""
            )
            block.processing_decision_source = str(
                getattr(native, "processing_decision_source", "") or ""
            )
            block.processing_decision_reasons = tuple(
                str(value)
                for value in (
                    getattr(native, "processing_decision_reasons", []) or []
                )
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export source-only RT-DETR/PaddleOCR-VL evidence for the v3.4 "
            "conditional glyph segmenter."
        )
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reuse-evidence", type=Path)
    parser.add_argument("--source-language", default="ja")
    parser.add_argument(
        "--server-url",
        default=DEFAULT_PADDLE_DIRECT_SERVER_URL,
    )
    parser.add_argument("--cpu-detector", action="store_true")
    parser.add_argument("--ensure-managed-runtime", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--parallel-workers", type=int, default=1)
    args = parser.parse_args(argv)

    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        runtime = ProductPaddleCropRuntime(
            server_url=args.server_url,
            source_language=args.source_language,
            use_gpu=not args.cpu_detector,
            ensure_managed_runtime=args.ensure_managed_runtime,
            max_new_tokens=args.max_new_tokens,
            parallel_workers=args.parallel_workers,
        )
        output, receipt_path, seal, receipt = write_source_product_evidence(
            args.source_manifest,
            output_root / "source-product-evidence.json",
            runtime,
            reuse_evidence_path=args.reuse_evidence,
        )
        if managed is not None:
            managed.complete(
                metadata={
                    "source_manifest_sha256": receipt[
                        "source_manifest_sha256"
                    ],
                    "provider_identity_sha256": receipt[
                        "provider_identity_sha256"
                    ],
                    "evidence_file_sha256": receipt[
                        "evidence_file_sha256"
                    ],
                    "summary": receipt["summary"],
                }
            )
        print(
            json.dumps(
                {
                    "output": str(output),
                    "receipt": str(receipt_path),
                    "seal": str(seal),
                    "summary": receipt["summary"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        if managed is not None:
            managed.fail(exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
