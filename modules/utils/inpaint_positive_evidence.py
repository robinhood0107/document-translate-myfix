from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from modules.utils.bubble_silhouette import extract_bubble_interior_cap_crop
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

CLEAN_BUBBLE_BROAD_REASONS = frozenset(
    {
        "bubble_residual_source_seed_unavailable",
    }
)
MINIMUM_BROAD_BACKGROUND_SAMPLES = 32


@dataclass(slots=True)
class PositiveTextEvidenceResult:
    positive_claim: np.ndarray
    positive_edit: np.ndarray
    block_claim_patches: dict[int, MaskPatch]
    block_edit_patches: dict[int, MaskPatch]
    block_claim_providers: dict[int, tuple[str, ...]]
    narrow_edit: np.ndarray
    broad_edit: np.ndarray
    block_broad_edit_patches: dict[int, MaskPatch]
    block_bubble_interior_patches: dict[int, MaskPatch]
    block_route_decisions: dict[int, str]
    block_route_reasons: dict[int, tuple[str, ...]]


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


def _patch_to_page_mask(
    patch: MaskPatch | None,
    image_shape: tuple[int, ...],
) -> np.ndarray:
    page = np.zeros(image_shape[:2], dtype=np.uint8)
    if patch is None:
        return page
    x1, y1, x2, y2 = patch.xyxy
    if x1 < 0 or y1 < 0 or x2 > page.shape[1] or y2 > page.shape[0]:
        return page
    page[y1:y2, x1:x2] = patch.mask
    return page


def build_detector_positive_text_evidence(
    blocks: Iterable,
    raw_pixel_claim: np.ndarray | None,
    routing_evidence: Iterable[BlockInpaintEvidence],
    *,
    image_shape: tuple[int, ...],
    existing_edit_mask: np.ndarray | None = None,
    protected_corner_mask: np.ndarray | None = None,
    source_image: np.ndarray | None = None,
) -> PositiveTextEvidenceResult:
    """Build an edit mask whose every claimed pixel comes from the detector.

    Detector boxes and content-component boxes are ownership gates only. They
    never create edit pixels. A direct detector text box may widen ownership;
    a bubble-rescue block remains constrained to its existing content boxes.
    """

    block_items = tuple(blocks or ())
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
    source_owned = combine_evidence_patches(
        evidence_items,
        "source_owned",
        image_shape,
    )
    existing_owned = np.where(
        (existing_edit > 0) | (source_owned > 0),
        255,
        0,
    ).astype(np.uint8)
    reason_by_index = _evidence_reason_by_index(evidence_items)
    evidence_by_index = {
        int(item.block_index): item
        for item in evidence_items
        if item.block_index is not None
    }
    page_claim = np.zeros(image_shape[:2], dtype=np.uint8)
    page_edit = np.zeros(image_shape[:2], dtype=np.uint8)
    page_narrow_edit = np.zeros(image_shape[:2], dtype=np.uint8)
    page_broad_edit = np.zeros(image_shape[:2], dtype=np.uint8)
    claim_patches: dict[int, MaskPatch] = {}
    edit_patches: dict[int, MaskPatch] = {}
    providers_by_index: dict[int, tuple[str, ...]] = {}
    broad_edit_patches: dict[int, MaskPatch] = {}
    bubble_interior_patches: dict[int, MaskPatch] = {}
    route_decisions: dict[int, str] = {}
    route_reasons: dict[int, tuple[str, ...]] = {}

    for block_index, block in enumerate(block_items):
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
        # A recoverable block may reopen only pixels that have both explicit
        # detector evidence and source ownership.  The semantic anchor remains
        # an ownership gate; it must never reopen unrelated source pixels.
        block_existing_owned = np.where(
            (existing_owned > 0)
            & ~((source_owned > 0) & (block_claim > 0)),
            255,
            0,
        ).astype(np.uint8)
        block_evidence = evidence_by_index.get(block_index)
        narrow_edit = np.where(
            (block_claim > 0)
            & (block_existing_owned <= 0)
            & (structure_protect <= 0)
            & (ownership_protect <= 0)
            & (corner_protect <= 0),
            255,
            0,
        ).astype(np.uint8)
        block_edit = narrow_edit
        broad_edit = np.zeros(image_shape[:2], dtype=np.uint8)
        route_decision = "narrow"
        reasons: list[str] = []
        bubble_interior = _patch_to_page_mask(
            None if block_evidence is None else block_evidence.bubble_interior,
            image_shape,
        )
        bubble_roi = normalize_xyxy(
            getattr(block, "bubble_xyxy", None),
            image_shape,
        )
        if source_image is not None and bubble_roi is not None:
            x1, y1, x2, y2 = bubble_roi
            detector_seed_crop = block_claim[y1:y2, x1:x2]
            detector_seeded_interior = extract_bubble_interior_cap_crop(
                np.asarray(source_image)[y1:y2, x1:x2],
                detector_seed_crop,
                erode_px=0,
                min_area_ratio=0.0,
                max_area_ratio=1.0,
                min_seed_coverage=0.0,
                preserve_seed_after_erode=False,
            )
            if detector_seeded_interior is not None:
                bubble_interior.fill(0)
                bubble_interior[y1:y2, x1:x2] = normalize_edit_mask(
                    detector_seeded_interior,
                    detector_seed_crop.shape,
                )
        if reason not in CLEAN_BUBBLE_BROAD_REASONS:
            reasons.append("pr2_structure_or_ambiguity_veto")
        elif not np.any(bubble_interior):
            reasons.append("bubble_interior_missing")
        elif not np.any((block_claim > 0) & (bubble_interior > 0)):
            reasons.append("detector_seed_outside_bubble_interior")
        elif np.any(
            (bubble_interior > 0)
            & (
                (structure_protect > 0)
                | (ownership_protect > 0)
                | (corner_protect > 0)
            )
        ):
            reasons.append("bubble_interior_overlaps_exact_protection")
        else:
            broad_edit = np.where(
                (bubble_interior > 0)
                & (block_existing_owned <= 0)
                & (structure_protect <= 0)
                & (ownership_protect <= 0)
                & (corner_protect <= 0),
                255,
                0,
            ).astype(np.uint8)
            sample_mask = (
                (bubble_interior > 0)
                & (broad_edit <= 0)
                & (structure_protect <= 0)
                & (ownership_protect <= 0)
                & (corner_protect <= 0)
            )
            if int(np.count_nonzero(sample_mask)) < MINIMUM_BROAD_BACKGROUND_SAMPLES:
                broad_edit.fill(0)
                reasons.append("insufficient_roi_background_samples")
            elif np.any(broad_edit):
                block_edit = np.where(
                    (narrow_edit > 0) | (broad_edit > 0),
                    255,
                    0,
                ).astype(np.uint8)
                route_decision = "broad"
            else:
                reasons.append("post_protection_broad_edit_empty")
        claim_patch = _patch_from_mask(block_claim)
        if claim_patch is None:
            continue
        claim_patches[block_index] = claim_patch
        providers_by_index[block_index] = tuple(
            [
                "ctd_raw_fixed1280",
                *providers,
                *(["ballons_native_bubble_interior"] if route_decision == "broad" else []),
            ]
        )
        route_decisions[block_index] = route_decision
        route_reasons[block_index] = tuple(reasons)
        page_claim[block_claim > 0] = 255
        page_narrow_edit[narrow_edit > 0] = 255
        edit_patch = _patch_from_mask(block_edit)
        if edit_patch is not None:
            edit_patches[block_index] = edit_patch
            page_edit[block_edit > 0] = 255
        broad_patch = _patch_from_mask(broad_edit)
        if broad_patch is not None:
            broad_edit_patches[block_index] = broad_patch
            page_broad_edit[broad_edit > 0] = 255
        interior_patch = _patch_from_mask(bubble_interior)
        if interior_patch is not None:
            bubble_interior_patches[block_index] = interior_patch

    return PositiveTextEvidenceResult(
        positive_claim=np.ascontiguousarray(page_claim),
        positive_edit=np.ascontiguousarray(page_edit),
        block_claim_patches=claim_patches,
        block_edit_patches=edit_patches,
        block_claim_providers=providers_by_index,
        narrow_edit=np.ascontiguousarray(page_narrow_edit),
        broad_edit=np.ascontiguousarray(page_broad_edit),
        block_broad_edit_patches=broad_edit_patches,
        block_bubble_interior_patches=bubble_interior_patches,
        block_route_decisions=route_decisions,
        block_route_reasons=route_reasons,
    )
