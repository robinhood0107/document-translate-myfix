from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from .contracts import DetectorBox, binary_mask


@dataclass(frozen=True, slots=True)
class ProvenanceFusionResult:
    ownership: np.ndarray
    positive_claim: np.ndarray
    positive_edit: np.ndarray
    selected_raw_text_boxes: tuple[DetectorBox, ...]


@dataclass(frozen=True, slots=True)
class SourceEditReconciliation:
    verified_source_edit: np.ndarray
    positive_edit: np.ndarray
    replacement_edit: np.ndarray


@dataclass(frozen=True, slots=True)
class StructureGuardedReconciliation:
    """A source-mask replacement that never invents pixels from geometry."""

    verified_source_edit: np.ndarray
    positive_claim: np.ndarray
    positive_edit: np.ndarray
    replacement_edit: np.ndarray


def replace_guarded_regions_with_narrow_claim(
    existing_edit: np.ndarray,
    narrow_claim: np.ndarray,
    guarded_regions: np.ndarray,
    *,
    structure_protect: np.ndarray,
    ownership_protect: np.ndarray | None = None,
    corner_protect: np.ndarray | None = None,
) -> np.ndarray:
    """Undo broad edits only inside structure-risk blocks.

    Clean and ordinary blocks remain byte-for-byte unchanged.  Within a
    guarded block, the edit mask is rebuilt from the detector's narrow pixel
    claim after all exact production protection is subtracted.
    """

    existing = binary_mask(existing_edit)
    shape = existing.shape
    narrow = binary_mask(narrow_claim, shape)
    guarded = binary_mask(guarded_regions, shape)
    protected = binary_mask(structure_protect, shape)
    for optional in (ownership_protect, corner_protect):
        if optional is not None:
            protected = cv2.bitwise_or(protected, binary_mask(optional, shape))
    rebuilt = (narrow > 0) & (protected <= 0)
    result = np.where(
        guarded > 0,
        np.where(rebuilt, 255, 0),
        existing,
    ).astype(np.uint8)
    return np.ascontiguousarray(result)


def replace_guarded_expansion_halo_with_narrow_claim(
    existing_edit: np.ndarray,
    pre_expand_mask: np.ndarray,
    post_expand_mask: np.ndarray,
    narrow_claim: np.ndarray,
    guarded_regions: np.ndarray,
    *,
    structure_protect: np.ndarray,
    ownership_protect: np.ndarray | None = None,
    corner_protect: np.ndarray | None = None,
) -> np.ndarray:
    """Remove only the automatic expansion halo in structure-risk blocks."""

    existing = binary_mask(existing_edit)
    shape = existing.shape
    before = binary_mask(pre_expand_mask, shape)
    after = binary_mask(post_expand_mask, shape)
    narrow = binary_mask(narrow_claim, shape)
    guarded = binary_mask(guarded_regions, shape)
    protected = binary_mask(structure_protect, shape)
    for optional in (ownership_protect, corner_protect):
        if optional is not None:
            protected = cv2.bitwise_or(protected, binary_mask(optional, shape))
    expansion_halo = (after > 0) & (before <= 0) & (guarded > 0)
    result = existing.copy()
    result[expansion_halo] = 0
    recover = (narrow > 0) & (guarded > 0) & (protected <= 0)
    result[recover] = 255
    return np.ascontiguousarray(result)


def add_guarded_narrow_claim(
    existing_edit: np.ndarray,
    narrow_claim: np.ndarray,
    guarded_regions: np.ndarray,
    *,
    structure_protect: np.ndarray,
    ownership_protect: np.ndarray | None = None,
    corner_protect: np.ndarray | None = None,
) -> np.ndarray:
    """Add detector pixels in risk blocks without changing existing edits."""

    existing = binary_mask(existing_edit)
    shape = existing.shape
    narrow = binary_mask(narrow_claim, shape)
    guarded = binary_mask(guarded_regions, shape)
    protected = binary_mask(structure_protect, shape)
    for optional in (ownership_protect, corner_protect):
        if optional is not None:
            protected = cv2.bitwise_or(protected, binary_mask(optional, shape))
    result = existing.copy()
    addition = (narrow > 0) & (guarded > 0) & (protected <= 0)
    result[addition] = 255
    return np.ascontiguousarray(result)


NARROW_DETECTOR_RECOVERY_REASONS = frozenset(
    {
        "bubble_interior_cap_source_seed_unavailable",
        "bubble_interior_cap_source_seed_partially_suppressed",
        "bubble_protected_source_seed_unavailable",
        "bubble_residual_source_seed_unavailable",
        "line_art_intrusion",
        "line_art_source_seed_unavailable",
        "microtexture_intrusion",
        "microtexture_source_seed_unavailable",
        "microtexture_source_seed_partially_suppressed",
        "text_prior_unavailable_source_seed_unavailable",
    }
)


def detector_recovery_route(skipped_reason: str) -> str:
    """Return the most permissive route allowed by existing PR2 evidence.

    Structure or texture evidence vetoes only broad expansion.  A raw detector
    claim can still be used as a narrow edit after exact protection is removed.
    Unknown and successful routes remain closed in this recovery pass.
    """

    reason = str(skipped_reason or "").strip()
    if reason == "bubble_residual_source_seed_unavailable":
        return "broad"
    if reason in NARROW_DETECTOR_RECOVERY_REASONS:
        return "narrow"
    return "skip"


def reapply_exact_protection_after_expansion(
    expanded_mask: np.ndarray,
    *,
    structure_protect: np.ndarray,
    ownership_protect: np.ndarray | None = None,
    ambiguous_protect: np.ndarray | None = None,
    corner_protect: np.ndarray | None = None,
) -> np.ndarray:
    """Restore the mask-protection invariant after any mask expansion.

    Protection applied before dilation is not stable: dilation can reclaim the
    protected boundary.  This final subtraction is geometry-independent and
    therefore applies equally to raw, refined, and detector-expanded masks.
    """

    expanded = binary_mask(expanded_mask)
    shape = expanded.shape
    protected = binary_mask(structure_protect, shape)
    for optional in (ownership_protect, ambiguous_protect, corner_protect):
        if optional is not None:
            protected = cv2.bitwise_or(protected, binary_mask(optional, shape))
    return np.ascontiguousarray(
        np.where((expanded > 0) & (protected <= 0), 255, 0).astype(np.uint8)
    )


def reapply_source_protection_after_expansion(
    expanded_mask: np.ndarray,
    *,
    derived_structure_protect: np.ndarray,
    ownership_protect: np.ndarray | None = None,
    corner_protect: np.ndarray | None = None,
) -> np.ndarray:
    """Production-eligible final subtraction using source-only evidence.

    Evaluation annotations are deliberately absent from this interface.  They
    can score the returned mask but can never influence its pixels.
    """

    return reapply_exact_protection_after_expansion(
        expanded_mask,
        structure_protect=derived_structure_protect,
        ownership_protect=ownership_protect,
        corner_protect=corner_protect,
    )


def build_detector_verified_structure_protect(
    source_structure_proposal: np.ndarray,
    raw_pixel_claim: np.ndarray,
    *,
    claim_ownership: np.ndarray,
    corner_protect: np.ndarray | None = None,
) -> np.ndarray:
    """Keep structure unless an owned raw detector pixel proves it is text.

    The earlier source-cap policy removed protection around every existing edit
    pixel.  That assumes the expanded source mask is text evidence and can
    reopen adjacent halftone or line art.  Here only the detector's raw pixel
    output, clipped by authoritative ownership, may carve a hole in the
    structure proposal.  Corners remain protected unconditionally.

    Evaluation annotations are intentionally not accepted by this API.
    """

    proposal = binary_mask(source_structure_proposal)
    shape = proposal.shape
    claim = binary_mask(raw_pixel_claim, shape)
    ownership = binary_mask(claim_ownership, shape)
    verified_text = (claim > 0) & (ownership > 0)
    protected = (proposal > 0) & ~verified_text
    if corner_protect is not None:
        protected |= binary_mask(corner_protect, shape) > 0
    return np.ascontiguousarray(np.where(protected, 255, 0).astype(np.uint8))


def build_source_owned_expansion_cap(
    source_owned_mask: np.ndarray,
    *,
    final_dilate_size: int,
) -> np.ndarray:
    """Mirror the product's final expansion footprint for source-owned text."""

    source = binary_mask(source_owned_mask)
    size = max(0, int(final_dilate_size))
    if size <= 0 or not np.any(source):
        return np.ascontiguousarray(source)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return np.ascontiguousarray(
        np.where(cv2.dilate(source, kernel, iterations=1) > 0, 255, 0).astype(
            np.uint8
        )
    )


def build_post_expansion_protection_reentry(
    post_expand_mask: np.ndarray,
    structure_protect: np.ndarray,
    *,
    corner_protect: np.ndarray | None = None,
) -> np.ndarray:
    """Return only protected pixels reclaimed by the final expansion step."""

    expanded = binary_mask(post_expand_mask)
    shape = expanded.shape
    protect = binary_mask(structure_protect, shape)
    reentry = (expanded > 0) & (protect > 0)
    if corner_protect is not None:
        reentry |= (expanded > 0) & (binary_mask(corner_protect, shape) > 0)
    return np.ascontiguousarray(np.where(reentry, 255, 0).astype(np.uint8))


def build_source_protected_detector_candidate(
    expanded_source_edit: np.ndarray,
    raw_pixel_claim: np.ndarray,
    *,
    claim_ownership: np.ndarray,
    derived_structure_protect: np.ndarray,
    ownership_protect: np.ndarray | None = None,
    corner_protect: np.ndarray | None = None,
) -> StructureGuardedReconciliation:
    """Combine post-expansion protection with a narrow detector recovery.

    This is the source-only C15 mask contract.  It accepts no evaluation target
    or annotation mask.  The existing source edit is preserved everywhere
    except exact production protection, while added pixels must be claimed by
    the raw detector and authoritative ownership.
    """

    shape = binary_mask(expanded_source_edit).shape
    return reconcile_structure_guarded_source_edit(
        raw_pixel_claim,
        expanded_source_edit,
        ownership=claim_ownership,
        structure_protect=derived_structure_protect,
        ownership_protect=(
            ownership_protect
            if ownership_protect is not None
            else np.zeros(shape, dtype=np.uint8)
        ),
        ambiguous_protect=np.zeros(shape, dtype=np.uint8),
        corner_protect=corner_protect,
        allow_narrow_recovery=True,
    )


def reconcile_structure_guarded_source_edit(
    raw_pixel_claim: np.ndarray,
    existing_edit: np.ndarray,
    *,
    ownership: np.ndarray,
    structure_protect: np.ndarray,
    ownership_protect: np.ndarray,
    ambiguous_protect: np.ndarray,
    corner_protect: np.ndarray | None = None,
    allow_narrow_recovery: bool,
) -> StructureGuardedReconciliation:
    """Remove exact protection from source edits, then add detector pixels.

    This differs from the retired all-block C13 replacement: unprotected source
    components are retained in full, so an imperfect pixel detector cannot
    reopen anti-aliased glyph edges.  The only removed pixels are already owned
    by exact structure/ownership/ambiguous/corner evidence.  Added pixels remain
    a strict subset of the raw detector mask and authoritative ownership.
    """

    claim = binary_mask(raw_pixel_claim)
    shape = claim.shape
    existing = binary_mask(existing_edit, shape)
    owned = binary_mask(ownership, shape)
    structure = binary_mask(structure_protect, shape)
    ownership_guard = binary_mask(ownership_protect, shape)
    ambiguous = binary_mask(ambiguous_protect, shape)
    corner = (
        binary_mask(corner_protect, shape)
        if corner_protect is not None
        else np.zeros(shape, dtype=np.uint8)
    )
    protected = np.where(
        (structure > 0)
        | (ownership_guard > 0)
        | (ambiguous > 0)
        | (corner > 0),
        255,
        0,
    ).astype(np.uint8)
    verified = np.where(
        (existing > 0) & (protected <= 0),
        255,
        0,
    ).astype(np.uint8)
    positive_claim = np.where(
        (claim > 0) & (owned > 0),
        255,
        0,
    ).astype(np.uint8)
    positive_edit = np.where(
        (positive_claim > 0)
        & (existing <= 0)
        & (protected <= 0)
        & bool(allow_narrow_recovery),
        255,
        0,
    ).astype(np.uint8)
    replacement = cv2.bitwise_or(verified, positive_edit)
    return StructureGuardedReconciliation(
        verified_source_edit=np.ascontiguousarray(verified),
        positive_claim=np.ascontiguousarray(positive_claim),
        positive_edit=np.ascontiguousarray(positive_edit),
        replacement_edit=np.ascontiguousarray(replacement),
    )


def reconcile_source_edit(
    detector_claim: np.ndarray,
    existing_edit: np.ndarray,
    *,
    allow_positive_addition: bool,
    existing_ownership_evidence: np.ndarray | None = None,
) -> SourceEditReconciliation:
    """Keep detector-verified source edits and explicitly allowed additions."""

    claim = binary_mask(detector_claim)
    existing = binary_mask(existing_edit, claim.shape)
    verification = claim
    if existing_ownership_evidence is not None:
        ownership_evidence = binary_mask(existing_ownership_evidence, claim.shape)
        verification = cv2.bitwise_or(claim, ownership_evidence)
    component_count, labels = cv2.connectedComponents(
        (existing > 0).astype(np.uint8),
        connectivity=8,
    )
    verified = np.zeros_like(existing)
    if component_count > 1:
        touched = np.unique(labels[verification > 0])
        touched = touched[touched > 0]
        if touched.size:
            verified[np.isin(labels, touched)] = 255
    positive = np.where(
        (claim > 0) & (existing == 0) & bool(allow_positive_addition),
        255,
        0,
    ).astype(np.uint8)
    replacement = cv2.bitwise_or(verified, positive)
    return SourceEditReconciliation(
        verified_source_edit=np.ascontiguousarray(verified),
        positive_edit=np.ascontiguousarray(positive),
        replacement_edit=np.ascontiguousarray(replacement),
    )


def build_provenance_fusion(
    raw_pixel_claim: np.ndarray,
    *,
    required_skip_prior: np.ndarray,
    required_skip_seed: np.ndarray,
    content_component_ownership: np.ndarray,
    raw_detector_boxes: Iterable[DetectorBox],
    existing_edit: np.ndarray,
    structure_protect: np.ndarray,
    ambiguous_protect: np.ndarray,
    subtract_existing_edit: bool = True,
) -> ProvenanceFusionResult:
    """Fuse block provenance without turning a detector box into pixel claim.

    A directly detected RT-DETR text box may widen ownership for a required
    conservative skip. A bubble-rescue block, which has no raw text box, stays
    restricted to its existing content components. In both cases every output
    claim pixel must still originate in the raw pixel detector mask.
    """

    claim = binary_mask(raw_pixel_claim)
    shape = claim.shape
    prior = binary_mask(required_skip_prior, shape)
    seed = binary_mask(required_skip_seed, shape)
    content = binary_mask(content_component_ownership, shape)
    existing = binary_mask(existing_edit, shape)
    structure = binary_mask(structure_protect, shape)
    ambiguous = binary_mask(ambiguous_protect, shape)

    raw_text_ownership = np.zeros(shape, dtype=np.uint8)
    selected: list[DetectorBox] = []
    for original in raw_detector_boxes:
        if original.label == "bubble":
            continue
        box = original.clipped(shape)
        if box is None:
            continue
        x1, y1, x2, y2 = box.xyxy
        if not np.any(seed[y1:y2, x1:x2] > 0):
            continue
        raw_text_ownership[y1:y2, x1:x2] = 255
        selected.append(box)

    ownership = np.where(
        (prior > 0) & ((content > 0) | (raw_text_ownership > 0)),
        255,
        0,
    ).astype(np.uint8)
    positive_claim = np.where((claim > 0) & (ownership > 0), 255, 0).astype(np.uint8)
    positive_edit = np.where(
        (positive_claim > 0)
        & ((existing == 0) | (not subtract_existing_edit))
        & (structure == 0)
        & (ambiguous == 0),
        255,
        0,
    ).astype(np.uint8)
    return ProvenanceFusionResult(
        ownership=np.ascontiguousarray(ownership),
        positive_claim=np.ascontiguousarray(positive_claim),
        positive_edit=np.ascontiguousarray(positive_edit),
        selected_raw_text_boxes=tuple(selected),
    )
