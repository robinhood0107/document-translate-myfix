#!/usr/bin/env python3
"""Build a frozen, source-first visual manifest from a product debug export.

This builder does not run OCR, a detector, or an inpainter.  It verifies a
selected page set from the 130-page private source inventory against the actual
bytes emitted by ``export_inpaint_debug.py`` and writes only geometry/protection
evidence plus the frozen PR6 baseline.  Text/OCR rectangles are ownership
boundaries; they never create edit pixels here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePath
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "inpaint-glyph-source-visual-manifest-v34"
SOURCE_INVENTORY_PAGE_COUNT = 130
PAGE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TRANSLATE_ACTIONS = frozenset({"translate", "translate_inpaint"})
PRESERVE_ACTIONS = frozenset({"preserve"})
ABSTAIN_ACTIONS = frozenset({"review", "abstain", "skip"})
PRESERVE_ROLES = frozenset({"sfx", "onomatopoeia", "decorative", "decoration"})
TRANSLATE_ROLES = frozenset(
    {
        "dialogue_bubble",
        "dialogue_free",
        "narration",
        "caption",
        "text_bubble",
        "text_free",
        "ui_or_sign",
    }
)
FORBIDDEN_OUTPUT_FIELD_TOKENS = ("target", "score")

# name, export folder, filename suffix, optional debug-metadata pixel-count key
ROUTING_MASK_SPECS = (
    (
        "structure_protect",
        "routing_structure_masks",
        "_routing_structure.png",
        "routing_structure_protect_pixel_count",
    ),
    (
        "source_owned",
        "routing_source_owned_masks",
        "_routing_source_owned.png",
        "routing_source_owned_pixel_count",
    ),
    (
        "source_raw_owned",
        "routing_source_raw_owned_masks",
        "_routing_source_raw_owned.png",
        "routing_source_raw_owned_pixel_count",
    ),
    (
        "ownership_protect",
        "routing_ownership_protect_masks",
        "_routing_ownership_protect.png",
        "routing_ownership_protect_pixel_count",
    ),
    (
        "corner_protect",
        "protected_corner_masks",
        "_protected_corners.png",
        "protected_corner_mask_pixel_count",
    ),
    (
        "ambiguous_protect",
        "ambiguous_structure_masks",
        "_ambiguous_structure.png",
        None,
    ),
)


class VisualManifestError(ValueError):
    """A frozen source/debug contract was missing or changed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    header = json.dumps(
        {"shape": list(value.shape), "dtype": str(value.dtype)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256(header)
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VisualManifestError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise VisualManifestError(f"JSON root must be an object: {path}")
    return payload


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise VisualManifestError(f"missing product page metrics: {path}")
    rows: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise VisualManifestError(f"unreadable product page metrics: {path}") from exc
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VisualManifestError("invalid product page metrics JSONL") from exc
        if not isinstance(row, dict):
            raise VisualManifestError("product page metrics row must be an object")
        page_id = str(row.get("page_id") or "")
        if not PAGE_ID_RE.fullmatch(page_id) or page_id in rows:
            raise VisualManifestError("product page metrics contain an invalid or duplicate page id")
        rows[page_id] = row
    if not rows:
        raise VisualManifestError("product page metrics are empty")
    return rows


def _read_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise VisualManifestError(f"missing RGB artifact: {path}")
    try:
        with Image.open(path) as opened:
            return np.asarray(ImageOps.exif_transpose(opened).convert("RGB")).copy()
    except (OSError, ValueError) as exc:
        raise VisualManifestError(f"invalid RGB artifact: {path}") from exc


def _read_binary_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        raise VisualManifestError(f"missing mask artifact: {path}")
    try:
        with Image.open(path) as opened:
            raw = np.asarray(opened.convert("L")).copy()
    except (OSError, ValueError) as exc:
        raise VisualManifestError(f"invalid mask artifact: {path}") from exc
    if raw.shape != shape:
        raise VisualManifestError(f"mask shape mismatch: {path}")
    unique = set(int(value) for value in np.unique(raw))
    if not unique.issubset({0, 255}):
        raise VisualManifestError(f"mask is not binary 0/255: {path}")
    return np.where(raw > 0, 255, 0).astype(np.uint8)


def _mask_artifact(
    path: Path,
    mask: np.ndarray,
    *,
    debug_root: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    if (debug_root is None) == (output_root is None):
        raise AssertionError("exactly one artifact root is required")
    root = debug_root if debug_root is not None else output_root
    assert root is not None
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise VisualManifestError(f"artifact escaped its frozen root: {path}") from exc
    return {
        "relative_path": relative,
        "sha256": sha256_file(path),
        "pixel_sha256": pixel_sha256(mask),
        "size_bytes": path.stat().st_size,
        "pixel_count": int(np.count_nonzero(mask)),
        "width": int(mask.shape[1]),
        "height": int(mask.shape[0]),
    }


def _rgb_artifact(path: Path, image: np.ndarray, *, root: Path) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise VisualManifestError(f"artifact escaped product debug root: {path}") from exc
    return {
        "relative_path": relative,
        "sha256": sha256_file(path),
        "pixel_sha256": pixel_sha256(image),
        "size_bytes": path.stat().st_size,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
    }


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _source_inventory_identity(pages: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for page in pages:
        digest.update(
            (
                f"{page.get('inventory_number')}\t{page.get('page_id')}\t"
                f"{page.get('source_sha256')}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _validate_source_inventory(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    if payload.get("candidate_seen") is not False:
        raise VisualManifestError("source inventory is post-candidate")
    if payload.get("holdouts_inspected") not in (None, []):
        raise VisualManifestError("source inventory reports inspected holdouts")
    source_review = payload.get("source_only_review")
    if isinstance(source_review, Mapping) and source_review.get("candidate_seen") is not False:
        raise VisualManifestError("source review is post-candidate")
    pages = payload.get("pages")
    if not isinstance(pages, list) or len(pages) != SOURCE_INVENTORY_PAGE_COUNT:
        raise VisualManifestError("source inventory must contain exactly 130 pages")
    if int(payload.get("page_count", -1)) != SOURCE_INVENTORY_PAGE_COUNT:
        raise VisualManifestError("source inventory page_count must be 130")
    normalized: list[dict[str, Any]] = []
    page_ids: set[str] = set()
    inventory_numbers: set[int] = set()
    for raw_page in pages:
        if not isinstance(raw_page, dict):
            raise VisualManifestError("source inventory page must be an object")
        page = dict(raw_page)
        page_id = str(page.get("page_id") or "")
        if not PAGE_ID_RE.fullmatch(page_id) or page_id in page_ids:
            raise VisualManifestError("source inventory has an invalid or duplicate page id")
        try:
            inventory_number = int(page.get("inventory_number"))
        except (TypeError, ValueError) as exc:
            raise VisualManifestError("source inventory number is invalid") from exc
        if inventory_number in inventory_numbers:
            raise VisualManifestError("source inventory number is duplicated")
        source_sha = page.get("source_sha256")
        if not _valid_sha(source_sha):
            raise VisualManifestError(f"source SHA is invalid: {page_id}")
        if page.get("failure_class") is not None:
            raise VisualManifestError(f"source page was annotated after candidate review: {page_id}")
        for key in page:
            lowered = str(key).lower()
            if "target" in lowered or "score" in lowered:
                raise VisualManifestError(f"source page contains post-candidate field: {key}")
        page_ids.add(page_id)
        inventory_numbers.add(inventory_number)
        normalized.append(page)
    expected_identity = payload.get("source_inventory_sha256")
    actual_identity = _source_inventory_identity(normalized)
    if not _valid_sha(expected_identity) or expected_identity != actual_identity:
        raise VisualManifestError("source inventory identity SHA mismatch")
    return tuple(normalized)


def _select_pages(
    pages: Sequence[dict[str, Any]],
    *,
    selected_page_ids: Sequence[str] = (),
    selected_corpora: Sequence[str] = (),
    required_page_count: int,
) -> tuple[dict[str, Any], ...]:
    if bool(selected_page_ids) == bool(selected_corpora):
        raise VisualManifestError("select pages by page id or corpus, but not both")
    if required_page_count <= 0:
        raise VisualManifestError("required page count must be positive")
    if selected_page_ids:
        requested = tuple(str(value) for value in selected_page_ids)
        if len(requested) != len(set(requested)) or any(
            not PAGE_ID_RE.fullmatch(value) for value in requested
        ):
            raise VisualManifestError("selected page ids are invalid or duplicated")
        by_id = {str(page["page_id"]): page for page in pages}
        missing = sorted(set(requested) - set(by_id))
        if missing:
            raise VisualManifestError(f"selected page ids are absent: {missing}")
        selected = tuple(by_id[page_id] for page_id in requested)
    else:
        requested_corpora = tuple(str(value) for value in selected_corpora)
        if len(requested_corpora) != len(set(requested_corpora)) or any(
            not value for value in requested_corpora
        ):
            raise VisualManifestError("selected corpus ids are invalid or duplicated")
        selected = tuple(
            page for page in pages if str(page.get("corpus") or "") in requested_corpora
        )
    if len(selected) != required_page_count:
        raise VisualManifestError(
            f"selected page count differs: {len(selected)} != {required_page_count}"
        )
    return tuple(sorted(selected, key=lambda page: int(page["inventory_number"])))


def _safe_relative_source_path(value: object) -> Path:
    text = str(value or "")
    candidate = PurePath(text)
    if not text or candidate.is_absolute() or ".." in candidate.parts:
        raise VisualManifestError("source_relative_path is unsafe")
    return Path(*candidate.parts)


def _resolve_source_path(
    page: Mapping[str, Any],
    source_roots: Mapping[str, Path],
) -> Path:
    corpus = str(page.get("corpus") or "")
    reference_kind = str(page.get("source_reference_kind") or "")
    if corpus in source_roots:
        root = Path(source_roots[corpus])
    elif reference_kind == "repo_relative_private":
        root = ROOT
    else:
        raise VisualManifestError(f"source root is required for corpus: {corpus}")
    root = root.expanduser().resolve()
    relative = _safe_relative_source_path(page.get("source_relative_path"))
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise VisualManifestError("resolved source escaped its configured root") from exc
    if not path.is_file():
        raise VisualManifestError(f"source file is missing for page {page.get('page_id')}")
    return path


def _discover_debug_metadata(debug_root: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    for path in sorted(debug_root.rglob("*_debug.json")):
        if path.parent.name != "debug_metadata":
            continue
        payload = _read_json(path)
        page_id = str(payload.get("page_id") or "")
        if not PAGE_ID_RE.fullmatch(page_id) or page_id in discovered:
            raise VisualManifestError("debug export has an invalid or duplicate page id")
        discovered[page_id] = path.resolve()
    if not discovered:
        raise VisualManifestError("debug export contains no page metadata")
    return discovered


def _strict_bbox(
    value: object,
    shape: tuple[int, int],
    *,
    field: str,
) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        raise VisualManifestError(f"invalid owner boundary field: {field}")
    try:
        raw = tuple(int(round(float(item))) for item in value[:4])
    except (TypeError, ValueError, OverflowError) as exc:
        raise VisualManifestError(f"invalid owner boundary coordinates: {field}") from exc
    height, width = shape
    x1, y1, x2, y2 = raw
    x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
    y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
    if x2 <= x1 or y2 <= y1:
        raise VisualManifestError(f"empty owner boundary after clipping: {field}")
    return x1, y1, x2, y2


def _semantic_action(block: Mapping[str, Any]) -> tuple[str, str]:
    explicit = str(block.get("processing_action") or "").strip().lower()
    role = str(block.get("semantic_role") or "").strip().lower()
    text_class = str(block.get("text_class") or "").strip().lower()
    preserve_cue = role in PRESERVE_ROLES or text_class in PRESERVE_ROLES
    translate_cue = role in TRANSLATE_ROLES or text_class in TRANSLATE_ROLES
    if preserve_cue and translate_cue:
        return "abstain", "semantic_role_action_conflict"
    if explicit in TRANSLATE_ACTIONS:
        if preserve_cue:
            return "abstain", "semantic_role_action_conflict"
        return "translate", role or text_class or "text"
    if explicit in PRESERVE_ACTIONS:
        if translate_cue:
            return "abstain", "semantic_role_action_conflict"
        return "preserve", role or text_class or "preserve"
    if explicit in ABSTAIN_ACTIONS:
        return "abstain", role or text_class or "ambiguous"
    if explicit:
        return "abstain", "invalid_processing_action"
    inferred = role or text_class
    if inferred in PRESERVE_ROLES:
        return "preserve", inferred
    if inferred in TRANSLATE_ROLES:
        return "translate", inferred
    return "abstain", inferred or "missing_semantic_action"


def _require_same_source_pixels(
    source_rgb: np.ndarray, source_export: np.ndarray, *, page_id: str
) -> None:
    if not np.array_equal(source_export, source_rgb):
        raise VisualManifestError(
            f"{page_id}: product source pixels differ from sealed source"
        )


def _owner_regions_and_masks(
    blocks: Sequence[Mapping[str, Any]],
    shape: tuple[int, int],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    height, width = shape
    owner_count = np.zeros(shape, dtype=np.uint16)
    translate = np.zeros(shape, dtype=np.uint8)
    explicit_preserve = np.zeros(shape, dtype=np.uint8)
    explicit_abstain = np.zeros(shape, dtype=np.uint8)
    regions: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for ordinal, block in enumerate(blocks):
        try:
            index = int(block.get("index", ordinal))
        except (TypeError, ValueError) as exc:
            raise VisualManifestError("debug block index is invalid") from exc
        if index in seen_indices:
            raise VisualManifestError("debug block index is duplicated")
        seen_indices.add(index)
        text_class = str(block.get("text_class") or "").strip().lower()
        if text_class == "text_bubble":
            candidates = (
                "bubble_xyxy",
                "cleanup_roi_xyxy",
                "ctd_roi_xyxy",
                "mask_roi_xyxy",
                "xyxy",
            )
        else:
            candidates = (
                "text_free_erase_envelope_xyxy",
                "mask_actual_bbox",
                "mask_anchor_xyxy",
                "xyxy",
            )
        bbox = None
        boundary_source = ""
        for field in candidates:
            if block.get(field) is None:
                continue
            bbox = _strict_bbox(block.get(field), shape, field=field)
            boundary_source = field
            break
        if bbox is None:
            raise VisualManifestError(f"debug block has no owner boundary: {index}")
        action, role = _semantic_action(block)
        x1, y1, x2, y2 = bbox
        owner_count[y1:y2, x1:x2] += 1
        destination = {
            "translate": translate,
            "preserve": explicit_preserve,
            "abstain": explicit_abstain,
        }[action]
        destination[y1:y2, x1:x2] = 255
        canonical_block_id = str(block.get("canonical_block_id") or "")
        if len(canonical_block_id) > 256:
            raise VisualManifestError("canonical block id is too long")
        regions.append(
            {
                "region_id": f"block-{index:04d}",
                "block_index": index,
                "canonical_block_id": canonical_block_id,
                "boundary_xyxy": [x1, y1, x2, y2],
                "boundary_source": boundary_source,
                "text_class": text_class,
                "semantic_role": role,
                "semantic_action": action,
                "geometry_creates_edit_pixels": False,
            }
        )
    conflict = np.where(owner_count > 1, 255, 0).astype(np.uint8)
    owner_union = np.where(owner_count > 0, 255, 0).astype(np.uint8)
    authoritative = np.where(owner_count == 1, 255, 0).astype(np.uint8)
    preserve = np.where((explicit_preserve > 0) | (conflict > 0), 255, 0).astype(np.uint8)
    abstain = np.where((explicit_abstain > 0) | (conflict > 0), 255, 0).astype(np.uint8)
    translate = np.where((translate > 0) & (conflict == 0), 255, 0).astype(np.uint8)
    if owner_count.size != height * width:
        raise AssertionError("owner mask shape changed")
    return regions, {
        "owner_union": owner_union,
        "authoritative_owner": authoritative,
        "owner_conflict": conflict,
        "translate_owner": translate,
        "source_preserve": preserve,
        "source_abstain": abstain,
    }


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L").save(
        temporary,
        format="PNG",
        optimize=True,
    )
    temporary.replace(path)


def _verify_metadata_sha(
    value: object,
    actual: str,
    *,
    page_id: str,
    field: str,
) -> None:
    if not _valid_sha(value) or value != actual:
        raise VisualManifestError(f"{page_id}: {field} SHA mismatch")


def _validate_summary(summary: Mapping[str, Any], expected_count: int) -> None:
    if int(summary.get("failure_count", -1)) != 0 or summary.get("failures") not in (None, []):
        raise VisualManifestError("product debug export contains failures")
    if int(summary.get("success_count", -1)) != expected_count:
        raise VisualManifestError("product debug success count differs")
    image_count = summary.get("image_count", summary.get("total_images"))
    if int(image_count if image_count is not None else -1) != expected_count:
        raise VisualManifestError("product debug image count differs")
    if int(summary.get("required_gate_failure_count", 0) or 0) != 0:
        raise VisualManifestError("product debug required gate failed")


def _assert_no_target_or_score_fields(value: object, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if any(token in normalized for token in FORBIDDEN_OUTPUT_FIELD_TOKENS):
                raise VisualManifestError(f"forbidden output field at {path}.{key}")
            _assert_no_target_or_score_fields(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_target_or_score_fields(child, f"{path}[{index}]")


def _parse_source_roots(values: Iterable[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        corpus, separator, raw_path = str(value).partition("=")
        if not separator or not corpus or not raw_path or corpus in roots:
            raise VisualManifestError("--source-root must be a unique CORPUS=PATH value")
        roots[corpus] = Path(raw_path)
    return roots


def _page_set_sha(pages: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for page in pages:
        digest.update(
            f"{page['page_id']}\t{page['source_sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def build_visual_manifest(
    *,
    source_inventory_path: Path,
    product_debug_root: Path,
    output_dir: Path,
    source_roots: Mapping[str, Path],
    selected_page_ids: Sequence[str] = (),
    selected_corpora: Sequence[str] = (),
    required_page_count: int,
) -> dict[str, Any]:
    source_inventory_path = source_inventory_path.expanduser().resolve()
    product_debug_root = product_debug_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise VisualManifestError("output directory must be fresh")
    if not source_inventory_path.is_file():
        raise VisualManifestError("source inventory is missing")
    if not product_debug_root.is_dir():
        raise VisualManifestError("product debug root is missing")

    inventory = _read_json(source_inventory_path)
    all_pages = _validate_source_inventory(inventory)
    selected = _select_pages(
        all_pages,
        selected_page_ids=selected_page_ids,
        selected_corpora=selected_corpora,
        required_page_count=required_page_count,
    )
    selected_ids = {str(page["page_id"]) for page in selected}
    debug_metadata_paths = _discover_debug_metadata(product_debug_root)
    if set(debug_metadata_paths) != selected_ids:
        raise VisualManifestError("product debug inventory differs from selected source pages")

    summary_path = product_debug_root / "metrics" / "summary.json"
    pages_metrics_path = product_debug_root / "metrics" / "pages.jsonl"
    summary = _read_json(summary_path)
    _validate_summary(summary, len(selected))
    page_metrics = _read_jsonl(pages_metrics_path)
    if set(page_metrics) != selected_ids:
        raise VisualManifestError("product page metrics inventory differs")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.partial-", dir=output_dir.parent)
    )
    try:
        output_pages: list[dict[str, Any]] = []
        for page in selected:
            page_id = str(page["page_id"])
            source_path = _resolve_source_path(page, source_roots)
            source_sha = sha256_file(source_path)
            _verify_metadata_sha(
                page.get("source_sha256"),
                source_sha,
                page_id=page_id,
                field="inventory source",
            )
            if int(page.get("size_bytes", source_path.stat().st_size)) != source_path.stat().st_size:
                raise VisualManifestError(f"{page_id}: source size differs from inventory")
            source_rgb = _read_rgb(source_path)
            expected_shape = (int(page.get("height", -1)), int(page.get("width", -1)))
            if source_rgb.shape[:2] != expected_shape:
                raise VisualManifestError(f"{page_id}: source dimensions differ from inventory")

            metadata_path = debug_metadata_paths[page_id]
            metadata = _read_json(metadata_path)
            if str(metadata.get("page_id") or "") != page_id:
                raise VisualManifestError(f"{page_id}: debug metadata page id differs")
            if metadata.get("candidate_seen") is True:
                raise VisualManifestError(f"{page_id}: debug metadata is post-candidate")
            _verify_metadata_sha(
                metadata.get("source_sha256"),
                source_sha,
                page_id=page_id,
                field="debug source",
            )
            if int(metadata.get("source_size_bytes", source_path.stat().st_size)) != source_path.stat().st_size:
                raise VisualManifestError(f"{page_id}: debug source size differs")
            corpus_output = metadata_path.parent.parent.resolve()
            try:
                corpus_output.relative_to(product_debug_root)
            except ValueError as exc:
                raise VisualManifestError("debug page escaped the product root") from exc

            source_export_path = corpus_output / "source_images" / f"{page_id}_source.png"
            cleaned_path = corpus_output / "cleaned_images" / f"{page_id}_cleaned.png"
            final_mask_path = corpus_output / "final_masks" / f"{page_id}_final_mask.png"
            source_export = _read_rgb(source_export_path)
            cleaned = _read_rgb(cleaned_path)
            if source_export.shape[:2] != expected_shape or cleaned.shape[:2] != expected_shape:
                raise VisualManifestError(f"{page_id}: product RGB artifact shape differs")
            _require_same_source_pixels(source_rgb, source_export, page_id=page_id)
            final_mask = _read_binary_mask(final_mask_path, expected_shape)

            source_export_record = _rgb_artifact(
                source_export_path,
                source_export,
                root=product_debug_root,
            )
            cleaned_record = _rgb_artifact(cleaned_path, cleaned, root=product_debug_root)
            final_mask_record = _mask_artifact(
                final_mask_path,
                final_mask,
                debug_root=product_debug_root,
            )
            _verify_metadata_sha(
                metadata.get("source_pixel_sha256"),
                source_export_record["pixel_sha256"],
                page_id=page_id,
                field="debug source pixel",
            )
            _verify_metadata_sha(
                metadata.get("cleaned_sha256"),
                cleaned_record["sha256"],
                page_id=page_id,
                field="cleaned",
            )
            _verify_metadata_sha(
                metadata.get("cleaned_pixel_sha256"),
                cleaned_record["pixel_sha256"],
                page_id=page_id,
                field="cleaned pixel",
            )
            _verify_metadata_sha(
                metadata.get("final_mask_sha256"),
                final_mask_record["sha256"],
                page_id=page_id,
                field="final mask",
            )
            _verify_metadata_sha(
                metadata.get("final_mask_pixel_sha256"),
                final_mask_record["pixel_sha256"],
                page_id=page_id,
                field="final mask pixel",
            )
            if int(metadata.get("final_mask_pixel_count", -1)) != final_mask_record["pixel_count"]:
                raise VisualManifestError(f"{page_id}: final mask pixel count differs")

            metrics = page_metrics[page_id]
            for field, actual in (
                ("source_sha256", source_sha),
                ("cleaned_sha256", cleaned_record["sha256"]),
                ("cleaned_pixel_sha256", cleaned_record["pixel_sha256"]),
                ("final_mask_sha256", final_mask_record["sha256"]),
                ("final_mask_pixel_sha256", final_mask_record["pixel_sha256"]),
            ):
                _verify_metadata_sha(
                    metrics.get(field),
                    actual,
                    page_id=page_id,
                    field=f"page metrics {field}",
                )

            routing_arrays: dict[str, np.ndarray] = {}
            routing_records: dict[str, dict[str, Any]] = {}
            for name, folder, suffix, count_field in ROUTING_MASK_SPECS:
                artifact_path = corpus_output / folder / f"{page_id}{suffix}"
                mask = _read_binary_mask(artifact_path, expected_shape)
                record = _mask_artifact(
                    artifact_path,
                    mask,
                    debug_root=product_debug_root,
                )
                if count_field is not None and int(metadata.get(count_field, -1)) != record["pixel_count"]:
                    raise VisualManifestError(f"{page_id}: {name} pixel count differs")
                routing_arrays[name] = mask
                routing_records[name] = record

            blocks = metadata.get("blocks")
            if not isinstance(blocks, list) or any(not isinstance(item, dict) for item in blocks):
                raise VisualManifestError(f"{page_id}: debug blocks are invalid")
            if int(metadata.get("block_count", -1)) != len(blocks):
                raise VisualManifestError(f"{page_id}: debug block count differs")
            regions, generated_masks = _owner_regions_and_masks(blocks, expected_shape)
            source_protect = np.where(
                (routing_arrays["structure_protect"] > 0)
                | (routing_arrays["ownership_protect"] > 0)
                | (routing_arrays["corner_protect"] > 0)
                | (routing_arrays["ambiguous_protect"] > 0),
                255,
                0,
            ).astype(np.uint8)
            frozen_protect = np.where(
                (source_protect > 0)
                | (generated_masks["source_preserve"] > 0)
                | (generated_masks["source_abstain"] > 0),
                255,
                0,
            ).astype(np.uint8)
            generated_masks["source_protect"] = source_protect
            generated_masks["frozen_protect"] = frozen_protect

            generated_records: dict[str, dict[str, Any]] = {}
            page_output = temporary_root / "pages" / page_id
            for name, mask in generated_masks.items():
                mask_path = page_output / f"{name.replace('_', '-')}.png"
                _write_mask(mask_path, mask)
                generated_records[name] = _mask_artifact(
                    mask_path,
                    mask,
                    output_root=temporary_root,
                )

            metadata_sha = sha256_file(metadata_path)
            output_pages.append(
                {
                    "inventory_number": int(page["inventory_number"]),
                    "corpus": str(page.get("corpus") or ""),
                    "page_id": page_id,
                    "source": {
                        "path": str(source_path),
                        "sha256": source_sha,
                        "size_bytes": source_path.stat().st_size,
                        "width": int(source_rgb.shape[1]),
                        "height": int(source_rgb.shape[0]),
                        "product_export": source_export_record,
                    },
                    "debug": {
                        "metadata_relative_path": metadata_path.relative_to(product_debug_root).as_posix(),
                        "metadata_sha256": metadata_sha,
                        "block_count": len(blocks),
                    },
                    "owner_regions": regions,
                    "ownership": {
                        "geometry_creates_edit_pixels": False,
                        "geometry_generated_edit_pixel_count": 0,
                        "overlap_policy": "conflict_is_preserved_and_abstained",
                        "generated_masks": {
                            name: generated_records[name]
                            for name in (
                                "owner_union",
                                "authoritative_owner",
                                "owner_conflict",
                                "translate_owner",
                            )
                        },
                        "routing_source_owned": routing_records["source_owned"],
                        "routing_source_raw_owned": routing_records["source_raw_owned"],
                    },
                    "source_protection": {
                        "routing_masks": {
                            name: routing_records[name]
                            for name in (
                                "structure_protect",
                                "ownership_protect",
                                "corner_protect",
                                "ambiguous_protect",
                            )
                        },
                        "source_protect": generated_records["source_protect"],
                        "source_preserve": generated_records["source_preserve"],
                        "source_abstain": generated_records["source_abstain"],
                        "frozen_protect": generated_records["frozen_protect"],
                    },
                    "pr6_baseline": {
                        "cleaned": cleaned_record,
                        "final_mask": final_mask_record,
                        "edit_mask_origin": "product_debug_export_only",
                    },
                }
            )

        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "frozen": True,
            "private": True,
            "source_only_geometry": True,
            "research_candidate_seen": False,
            "post_candidate_annotation_allowed": False,
            "geometry_creates_edit_pixels": False,
            "contains_pr6_baseline": True,
            "source_inventory": {
                "path": str(source_inventory_path),
                "sha256": sha256_file(source_inventory_path),
                "source_inventory_sha256": str(inventory["source_inventory_sha256"]),
                "page_count": SOURCE_INVENTORY_PAGE_COUNT,
            },
            "product_debug": {
                "root": str(product_debug_root),
                "summary_relative_path": summary_path.relative_to(product_debug_root).as_posix(),
                "summary_sha256": sha256_file(summary_path),
                "pages_relative_path": pages_metrics_path.relative_to(product_debug_root).as_posix(),
                "pages_sha256": sha256_file(pages_metrics_path),
            },
            "page_set_sha256": _page_set_sha(selected),
            "page_count": len(output_pages),
            "page_ids": [str(page["page_id"]) for page in selected],
            "pages": output_pages,
        }
        _assert_no_target_or_score_fields(result)
        manifest_path = temporary_root / "visual-manifest-v34.json"
        manifest_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        seal = {
            "schema_version": "sealed-inpaint-glyph-source-visual-manifest-v34",
            "manifest_file": manifest_path.name,
            "manifest_sha256": sha256_file(manifest_path),
            "source_inventory_sha256": str(inventory["source_inventory_sha256"]),
            "page_set_sha256": result["page_set_sha256"],
            "page_count": len(output_pages),
            "frozen": True,
            "research_candidate_seen": False,
        }
        _assert_no_target_or_score_fields(seal)
        (temporary_root / "visual-manifest-v34.seal.json").write_text(
            json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_root, output_dir)
        return result
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal source ownership/protection and a PR6 debug baseline for visual review."
    )
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--product-debug-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--page-id", action="append", default=[])
    selection.add_argument("--corpus", action="append", default=[])
    parser.add_argument("--require-page-count", type=int, required=True)
    parser.add_argument(
        "--source-root",
        action="append",
        default=[],
        metavar="CORPUS=PATH",
        help="Root for source_relative_path entries; repeat for external corpora.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        roots = _parse_source_roots(args.source_root)
        build_visual_manifest(
            source_inventory_path=args.source_inventory,
            product_debug_root=args.product_debug_root,
            output_dir=args.output_dir,
            source_roots=roots,
            selected_page_ids=tuple(args.page_id),
            selected_corpora=tuple(args.corpus),
            required_page_count=int(args.require_page_count),
        )
    except VisualManifestError as exc:
        print(str(exc), file=os.sys.stderr)
        return 2
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
