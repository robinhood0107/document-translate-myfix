from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from modules.utils.inpaint_composite import normalize_edit_mask
from modules.utils.inpaint_evidence import (
    BlockInpaintEvidence,
    MaskPatch,
    combine_evidence_patches,
)
from modules.utils.mask_roi import normalize_xyxy, resolve_inpaint_text_xyxy


POSITIVE_EVIDENCE_RECOVERABLE_REASONS = frozenset(
    {
        "bubble_interior_cap_source_seed_unavailable",
        "bubble_interior_cap_source_seed_partially_suppressed",
        "bubble_protected_source_seed_unavailable",
        "bubble_residual_source_seed_unavailable",
        "line_art_source_seed_unavailable",
        "microtexture_source_seed_unavailable",
        "microtexture_source_seed_partially_suppressed",
        "text_prior_unavailable_source_seed_unavailable",
    }
)


@dataclass(slots=True)
class PositiveTextEvidenceResult:
    positive_claim: np.ndarray
    positive_edit: np.ndarray
    block_claim_patches: dict[int, MaskPatch]
    block_edit_patches: dict[int, MaskPatch]
    block_claim_providers: dict[int, tuple[str, ...]]


def _paint_box(mask: np.ndarray, raw_box) -> tuple[int, int, int, int] | None:
    box = normalize_xyxy(raw_box, mask.shape)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = 255
    return box


def _content_component_ownership(block, image_shape: tuple[int, ...]) -> np.ndarray:
    ownership = np.zeros(image_shape[:2], dtype=np.uint8)
    raw_boxes = getattr(block, "inpaint_bboxes", None)
    if raw_boxes is None:
        return ownership
    for raw_box in tuple(raw_boxes):
        _paint_box(ownership, raw_box)
    return ownership


def _patch_from_mask(mask: np.ndarray) -> MaskPatch | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    roi = (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )
    x1, y1, x2, y2 = roi
    return MaskPatch(roi, mask[y1:y2, x1:x2])


def _evidence_reason_by_index(
    routing_evidence: Iterable[BlockInpaintEvidence],
) -> dict[int, str]:
    return {
        int(item.block_index): str(item.skipped_reason or "")
        for item in routing_evidence
        if item.block_index is not None
    }


def build_detector_positive_text_evidence(
    blocks: Iterable,
    raw_pixel_claim: np.ndarray | None,
    routing_evidence: Iterable[BlockInpaintEvidence],
    *,
    image_shape: tuple[int, ...],
    existing_edit_mask: np.ndarray | None = None,
    protected_corner_mask: np.ndarray | None = None,
) -> PositiveTextEvidenceResult:
    """Build an edit mask whose every claimed pixel comes from the detector.

    Detector boxes and content-component boxes are ownership gates only. They
    never create edit pixels. A direct detector text box may widen ownership;
    a bubble-rescue block remains constrained to its existing content boxes.
    """

    raw_claim = normalize_edit_mask(raw_pixel_claim, image_shape)
    existing_edit = normalize_edit_mask(existing_edit_mask, image_shape)
    corner_protect = normalize_edit_mask(protected_corner_mask, image_shape)
    evidence_items = tuple(routing_evidence or ())
    structure_protect = combine_evidence_patches(
        evidence_items,
        "structure_protect",
        image_shape,
    )
    ownership_protect = combine_evidence_patches(
        evidence_items,
        "ownership_protect",
        image_shape,
    )
    reason_by_index = _evidence_reason_by_index(evidence_items)
    page_claim = np.zeros(image_shape[:2], dtype=np.uint8)
    page_edit = np.zeros(image_shape[:2], dtype=np.uint8)
    claim_patches: dict[int, MaskPatch] = {}
    edit_patches: dict[int, MaskPatch] = {}
    providers_by_index: dict[int, tuple[str, ...]] = {}

    for block_index, block in enumerate(tuple(blocks or ())):
        reason = reason_by_index.get(
            block_index,
            str(getattr(block, "_erase_skipped_reason", "") or ""),
        )
        if reason not in POSITIVE_EVIDENCE_RECOVERABLE_REASONS:
            continue
        anchor = resolve_inpaint_text_xyxy(block, image_shape)
        if anchor is None:
            continue

        prior = np.zeros(image_shape[:2], dtype=np.uint8)
        _paint_box(prior, anchor)
        content_ownership = _content_component_ownership(block, image_shape)
        ownership = content_ownership.copy()
        providers = ["content_component_ownership"]
        detector_provider = str(
            getattr(block, "detector_provider", "") or ""
        ).strip()
        if detector_provider:
            providers.append(f"block_detector:{detector_provider}")

        if str(getattr(block, "detector_origin", "") or "") == "direct_text":
            detector_box = normalize_xyxy(
                getattr(block, "detector_text_bbox", None),
                image_shape,
            )
            if detector_box is not None:
                x1, y1, x2, y2 = detector_box
                if np.any(content_ownership[y1:y2, x1:x2] > 0):
                    ownership[y1:y2, x1:x2] = 255
                    providers.append("rtdetr_raw_text_box")

        block_claim = np.where(
            (raw_claim > 0) & (prior > 0) & (ownership > 0),
            255,
            0,
        ).astype(np.uint8)
        block_edit = np.where(
            (block_claim > 0)
            & (existing_edit <= 0)
            & (structure_protect <= 0)
            & (ownership_protect <= 0)
            & (corner_protect <= 0),
            255,
            0,
        ).astype(np.uint8)
        claim_patch = _patch_from_mask(block_claim)
        if claim_patch is None:
            continue
        claim_patches[block_index] = claim_patch
        providers_by_index[block_index] = tuple(
            ["ctd_raw_fixed1280", *providers]
        )
        page_claim[block_claim > 0] = 255
        edit_patch = _patch_from_mask(block_edit)
        if edit_patch is not None:
            edit_patches[block_index] = edit_patch
            page_edit[block_edit > 0] = 255

    return PositiveTextEvidenceResult(
        positive_claim=np.ascontiguousarray(page_claim),
        positive_edit=np.ascontiguousarray(page_edit),
        block_claim_patches=claim_patches,
        block_edit_patches=edit_patches,
        block_claim_providers=providers_by_index,
    )
