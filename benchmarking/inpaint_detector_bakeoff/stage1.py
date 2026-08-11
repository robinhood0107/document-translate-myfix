from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from .contracts import CandidateMaskResult, Stage1Page, assert_disjoint_masks, binary_mask


@dataclass(frozen=True, slots=True)
class PageMasks:
    target: np.ndarray
    protected: np.ndarray
    ambiguous: np.ndarray
    ownership: np.ndarray
    claim_seed: np.ndarray
    existing_edit: np.ndarray


def _path_value(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        nested = value.get("path")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return None


def load_stage1_manifest(path: Path) -> list[Stage1Page]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise ValueError("mask-only manifest must contain a pages array")
    records: list[Stage1Page] = []
    for entry in pages:
        if not isinstance(entry, dict):
            raise ValueError("mask-only manifest page must be an object")
        target = _path_value(entry.get("target_text_mask"))
        if target is None:
            target = _path_value(entry.get("target_glyph_mask"))
        records.append(
            Stage1Page(
                page_id=str(entry.get("page_id") or "").strip(),
                source_image=str(entry.get("path") or "").strip(),
                target_text_mask=target,
                protected_structure_mask=_path_value(entry.get("protected_structure_mask")),
                ambiguous_structure_mask=_path_value(entry.get("ambiguous_structure_mask")),
                ownership_mask=_path_value(entry.get("ownership_mask")),
                claim_seed_mask=_path_value(entry.get("claim_seed_mask")),
                no_edit=str(entry.get("expected_edit") or "").strip().lower() == "none",
            )
        )
    if not all(record.page_id and record.source_image for record in records):
        raise ValueError("mask-only manifest contains an empty page id or source path")
    return records


def _read_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise FileNotFoundError(f"unable to read source image: {path}")
    return image


def _read_mask(path: str | None, shape: tuple[int, int]) -> np.ndarray:
    if not path:
        return np.zeros(shape, dtype=np.uint8)
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.size == 0:
        raise FileNotFoundError(f"unable to read evaluation mask: {path}")
    return binary_mask(mask, shape)


def load_page_masks(
    page: Stage1Page,
    shape: tuple[int, int],
    *,
    existing_edit_path: str | None = None,
) -> PageMasks:
    target = _read_mask(page.target_text_mask, shape)
    protected = _read_mask(page.protected_structure_mask, shape)
    ambiguous = _read_mask(page.ambiguous_structure_mask, shape)
    ownership = (
        _read_mask(page.ownership_mask, shape)
        if page.ownership_mask
        else np.full(shape, 255, dtype=np.uint8)
    )
    claim_seed = (
        _read_mask(page.claim_seed_mask, shape)
        if page.claim_seed_mask
        else np.full(shape, 255, dtype=np.uint8)
    )
    existing_edit = _read_mask(existing_edit_path, shape)
    assert_disjoint_masks(
        {
            "target_text_mask": target,
            "protected_structure_mask": protected,
            "ambiguous_structure_mask": ambiguous,
        }
    )
    return PageMasks(target, protected, ambiguous, ownership, claim_seed, existing_edit)


def _claim_components_owned_by_seed(
    claim: np.ndarray,
    claim_seed: np.ndarray,
) -> np.ndarray:
    normalized = binary_mask(claim, claim_seed.shape)
    count, labels = cv2.connectedComponents((normalized > 0).astype(np.uint8), 8)
    if count <= 1:
        return normalized
    selected = np.zeros_like(normalized)
    seed_labels = np.unique(labels[claim_seed > 0])
    seed_labels = seed_labels[seed_labels > 0]
    if seed_labels.size:
        selected[np.isin(labels, seed_labels)] = 255
    return selected


def positive_edit_from_claim(claim: np.ndarray, masks: PageMasks) -> np.ndarray:
    owned_claim = np.where(
        (binary_mask(claim, masks.target.shape) > 0) & (masks.ownership > 0),
        255,
        0,
    ).astype(np.uint8)
    edit = _claim_components_owned_by_seed(owned_claim, masks.claim_seed)
    edit = np.where(
        (edit > 0)
        & (masks.ownership > 0)
        & (masks.existing_edit == 0)
        & (masks.protected == 0)
        & (masks.ambiguous == 0),
        255,
        0,
    ).astype(np.uint8)
    return np.ascontiguousarray(edit)


def _component_coverages(target: np.ndarray, claim: np.ndarray) -> list[float]:
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (target > 0).astype(np.uint8),
        connectivity=8,
    )
    coverages: list[float] = []
    for index in range(1, component_count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        covered = int(np.count_nonzero((labels == index) & (claim > 0)))
        coverages.append(float(covered) / float(area))
    return coverages


def score_page(
    page: Stage1Page,
    result: CandidateMaskResult,
    masks: PageMasks,
    *,
    variant: str,
) -> tuple[dict[str, object], np.ndarray]:
    claim = result.mask_for_variant(variant)
    edit = positive_edit_from_claim(claim, masks)
    target_pixels = int(np.count_nonzero(masks.target))
    target_claimed = int(np.count_nonzero((claim > 0) & (masks.target > 0)))
    target_edit = int(np.count_nonzero((edit > 0) & (masks.target > 0)))
    component_coverages = _component_coverages(masks.target, edit)
    raw_claim_protected_overlap = int(
        np.count_nonzero((claim > 0) & (masks.protected > 0))
    )
    raw_claim_ambiguous_overlap = int(
        np.count_nonzero((claim > 0) & (masks.ambiguous > 0))
    )
    raw_claim_outside_ownership = int(
        np.count_nonzero((claim > 0) & (masks.ownership == 0))
    )
    protected_overlap = int(np.count_nonzero((edit > 0) & (masks.protected > 0)))
    ambiguous_overlap = int(np.count_nonzero((edit > 0) & (masks.ambiguous > 0)))
    ownership_leak = int(np.count_nonzero((edit > 0) & (masks.ownership == 0)))
    record = {
        "page_id": page.page_id,
        "candidate_id": result.candidate_id,
        "variant": variant,
        "claim_pixel_count": int(np.count_nonzero(claim)),
        "edit_pixel_count": int(np.count_nonzero(edit)),
        "target_pixel_count": target_pixels,
        "target_claimed_pixel_count": target_claimed,
        "target_edit_pixel_count": target_edit,
        "target_claim_coverage": (
            float(target_claimed) / float(target_pixels) if target_pixels else None
        ),
        "target_edit_coverage": (
            float(target_edit) / float(target_pixels) if target_pixels else None
        ),
        "component_coverages": component_coverages,
        "minimum_component_coverage": min(component_coverages) if component_coverages else None,
        "raw_claim_protected_overlap": raw_claim_protected_overlap,
        "raw_claim_ambiguous_overlap": raw_claim_ambiguous_overlap,
        "raw_claim_outside_ownership_pixel_count": raw_claim_outside_ownership,
        "protected_edit_overlap": protected_overlap,
        "ambiguous_edit_overlap": ambiguous_overlap,
        "ownership_leak_pixel_count": ownership_leak,
        "false_edit_pixel_count": int(np.count_nonzero(edit)) if page.no_edit else 0,
        "runtime": dict(result.runtime),
    }
    return record, edit


def summarize(records: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = list(records)
    target_pixels = sum(int(row["target_pixel_count"]) for row in rows)
    target_edit = sum(int(row["target_edit_pixel_count"]) for row in rows)
    components = [
        float(value)
        for row in rows
        for value in row.get("component_coverages", [])
    ]
    return {
        "page_count": len(rows),
        "target_pixel_count": target_pixels,
        "target_edit_pixel_count": target_edit,
        "aggregate_target_coverage": (
            float(target_edit) / float(target_pixels) if target_pixels else None
        ),
        "minimum_component_coverage": min(components) if components else None,
        "raw_claim_protected_overlap": sum(
            int(row["raw_claim_protected_overlap"]) for row in rows
        ),
        "raw_claim_ambiguous_overlap": sum(
            int(row["raw_claim_ambiguous_overlap"]) for row in rows
        ),
        "raw_claim_outside_ownership_pixel_count": sum(
            int(row["raw_claim_outside_ownership_pixel_count"]) for row in rows
        ),
        "protected_edit_overlap": sum(int(row["protected_edit_overlap"]) for row in rows),
        "ambiguous_edit_overlap": sum(int(row["ambiguous_edit_overlap"]) for row in rows),
        "ownership_leak_pixel_count": sum(
            int(row["ownership_leak_pixel_count"]) for row in rows
        ),
        "false_edit_pixel_count": sum(int(row["false_edit_pixel_count"]) for row in rows),
    }


def run_stage1(
    pages: Iterable[Stage1Page],
    infer: Callable[[np.ndarray], CandidateMaskResult],
    *,
    variant: str,
    existing_edit_paths: dict[str, str] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, np.ndarray]]:
    rows: list[dict[str, object]] = []
    edits: dict[str, np.ndarray] = {}
    for page in pages:
        image = _read_image(page.source_image)
        result = infer(image)
        masks = load_page_masks(
            page,
            image.shape[:2],
            existing_edit_path=(existing_edit_paths or {}).get(page.page_id),
        )
        row, edit = score_page(page, result, masks, variant=variant)
        rows.append(row)
        edits[page.page_id] = edit
    return rows, summarize(rows), edits
