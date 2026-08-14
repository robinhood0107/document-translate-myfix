from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np

from .contracts import binary_mask
from .semantic import PRESERVE, REVIEW, TRANSLATE, SemanticDecision


PROVIDER_FINETUNE = "finetune"
PROVIDER_TILED = "tiled"
PROVIDER_OR = "or"
PROVIDER_MODES = frozenset({PROVIDER_FINETUNE, PROVIDER_TILED, PROVIDER_OR})
ADMISSION_POLICIES = frozenset({"g1", "g2", "g3"})
EXPANSION_MODES = frozenset({"raw_core", "connected_halo"})


@dataclass(frozen=True, slots=True)
class RegionAdmissionEvidence:
    region_id: str
    ownership: np.ndarray
    semantic: SemanticDecision


@dataclass(frozen=True, slots=True)
class ComponentAdmissionRecord:
    component_id: int
    owner_region_id: str
    semantic_role: str
    semantic_action: str
    providers: tuple[str, ...]
    accepted: bool
    reason: str
    component_pixel_count: int
    raw_core_pixel_count: int
    halo_pixel_count: int
    accepted_pixel_count: int
    touches_pr6_edit: bool
    touches_source_raw_owned: bool


@dataclass(frozen=True, slots=True)
class ProposalRefinementResult:
    proposal: np.ndarray
    raw_core: np.ndarray
    connected_halo: np.ndarray
    accepted_claim: np.ndarray
    safe_addition: np.ndarray
    hard_protection: np.ndarray
    component_records: tuple[ComponentAdmissionRecord, ...]


def _union(*masks: np.ndarray) -> np.ndarray:
    if not masks:
        raise ValueError("mask union requires at least one input")
    shape = binary_mask(masks[0]).shape
    result = np.zeros(shape, dtype=np.uint8)
    for mask in masks:
        result = cv2.bitwise_or(result, binary_mask(mask, shape))
    return np.ascontiguousarray(result)


def _selected_provider_masks(
    provider_mode: str,
    *,
    finetune_raw: np.ndarray,
    finetune_native3: np.ndarray,
    tiled_raw: np.ndarray,
    tiled_native3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mode = str(provider_mode).strip().lower()
    if mode not in PROVIDER_MODES:
        raise ValueError(f"unsupported provider mode: {provider_mode}")
    shape = binary_mask(finetune_raw).shape
    fine_raw = binary_mask(finetune_raw, shape)
    fine_native3 = binary_mask(finetune_native3, shape)
    tile_raw = binary_mask(tiled_raw, shape)
    tile_native3 = binary_mask(tiled_native3, shape)
    if np.any((fine_raw > 0) & (fine_native3 == 0)):
        raise ValueError("finetune native3 mask must contain its raw core")
    if np.any((tile_raw > 0) & (tile_native3 == 0)):
        raise ValueError("tiled native3 mask must contain its raw core")
    if mode == PROVIDER_FINETUNE:
        return fine_raw, fine_native3
    if mode == PROVIDER_TILED:
        return tile_raw, tile_native3
    return _union(fine_raw, tile_raw), _union(fine_native3, tile_native3)


def _component_reason_for_protection(
    local_component: np.ndarray,
    slices: tuple[slice, slice],
    *,
    structure: np.ndarray,
    ambiguous: np.ndarray,
    corner: np.ndarray,
    preserve_action: np.ndarray,
    abstain_action: np.ndarray,
) -> str:
    checks = (
        ("exact_structure_contact", structure),
        ("exact_ambiguous_contact", ambiguous),
        ("exact_corner_contact", corner),
        ("explicit_preserve_contact", preserve_action),
        ("abstain_contact", abstain_action),
    )
    for reason, mask in checks:
        if np.any(local_component & (mask[slices] > 0)):
            return reason
    return ""


def refine_detector_proposal(
    *,
    finetune_raw: np.ndarray,
    finetune_native3: np.ndarray,
    tiled_raw: np.ndarray,
    tiled_native3: np.ndarray,
    provider_mode: str,
    expansion_mode: str,
    admission_policy: str,
    regions: tuple[RegionAdmissionEvidence, ...],
    pr6_existing_edit: np.ndarray,
    source_raw_owned: np.ndarray,
    structure_protect: np.ndarray,
    ambiguous_protect: np.ndarray,
    corner_protect: np.ndarray,
) -> ProposalRefinementResult:
    """Turn high-recall detector output into a conservative PR6 addition.

    Every accepted component needs one authoritative semantic owner.  Geometry
    is a gate only: returned pixels always remain a subset of the selected
    detector proposal.
    """

    mode = str(provider_mode).strip().lower()
    expansion = str(expansion_mode).strip().lower()
    policy = str(admission_policy).strip().lower()
    if expansion not in EXPANSION_MODES:
        raise ValueError(f"unsupported expansion mode: {expansion_mode}")
    if policy not in ADMISSION_POLICIES:
        raise ValueError(f"unsupported admission policy: {admission_policy}")
    if policy in {"g2", "g3"} and mode != PROVIDER_OR:
        raise ValueError(f"{policy} requires both detector providers")
    if not regions:
        raise ValueError("proposal refinement requires authoritative regions")

    raw_core, native3 = _selected_provider_masks(
        mode,
        finetune_raw=finetune_raw,
        finetune_native3=finetune_native3,
        tiled_raw=tiled_raw,
        tiled_native3=tiled_native3,
    )
    shape = raw_core.shape
    fine_raw = binary_mask(finetune_raw, shape)
    tile_raw = binary_mask(tiled_raw, shape)
    baseline = binary_mask(pr6_existing_edit, shape)
    source_seed = binary_mask(source_raw_owned, shape)
    structure = binary_mask(structure_protect, shape)
    ambiguous = binary_mask(ambiguous_protect, shape)
    corner = binary_mask(corner_protect, shape)
    normalized_regions = tuple(
        RegionAdmissionEvidence(
            str(region.region_id),
            binary_mask(region.ownership, shape),
            region.semantic,
        )
        for region in regions
    )
    if any(not region.region_id for region in normalized_regions):
        raise ValueError("authoritative region id must not be empty")
    if len({region.region_id for region in normalized_regions}) != len(
        normalized_regions
    ):
        raise ValueError("authoritative region ids must be unique")

    preserve_action = np.zeros(shape, dtype=np.uint8)
    abstain_action = np.zeros(shape, dtype=np.uint8)
    for region in normalized_regions:
        if region.semantic.action == PRESERVE:
            preserve_action[region.ownership > 0] = 255
        elif region.semantic.action != TRANSLATE or not region.semantic.available:
            abstain_action[region.ownership > 0] = 255
    hard_protection = _union(
        structure,
        ambiguous,
        corner,
        preserve_action,
        abstain_action,
    )

    proposal = raw_core if expansion == "raw_core" else native3
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (proposal > 0).astype(np.uint8),
        8,
        cv2.CV_32S,
    )
    accepted = np.zeros(shape, dtype=np.uint8)
    records: list[ComponentAdmissionRecord] = []
    for component_id in range(1, count):
        x, y, width, height, area = (
            int(value) for value in stats[component_id]
        )
        if area <= 0:
            continue
        slices = (slice(y, y + height), slice(x, x + width))
        local_component = labels[slices] == component_id
        local_raw = raw_core[slices] > 0
        raw_count = int(np.count_nonzero(local_component & local_raw))
        providers = tuple(
            provider
            for provider, mask in (
                (PROVIDER_FINETUNE, fine_raw),
                (PROVIDER_TILED, tile_raw),
            )
            if np.any(local_component & (mask[slices] > 0))
        )
        owner_matches = [
            region
            for region in normalized_regions
            if np.any(local_component & (region.ownership[slices] > 0))
        ]
        owner = owner_matches[0] if len(owner_matches) == 1 else None
        owner_id = "" if owner is None else owner.region_id
        semantic = (
            SemanticDecision("ambiguous", REVIEW, available=False)
            if owner is None
            else owner.semantic
        )
        touches_baseline = bool(
            np.any(local_component & (baseline[slices] > 0))
        )
        touches_source = bool(
            np.any(local_component & (source_seed[slices] > 0))
        )
        reason = ""
        if raw_count == 0:
            reason = "no_raw_seed"
        elif not owner_matches:
            reason = "ownership_missing"
        elif len(owner_matches) != 1:
            reason = "ownership_conflict"
        elif np.any(local_component & (owner.ownership[slices] == 0)):
            reason = "ownership_partial"
        elif not semantic.available or semantic.action == REVIEW:
            reason = "semantic_abstain"
        elif semantic.action == PRESERVE:
            reason = "semantic_preserve"
        elif semantic.action != TRANSLATE:
            reason = "semantic_abstain"
        else:
            reason = _component_reason_for_protection(
                local_component,
                slices,
                structure=structure,
                ambiguous=ambiguous,
                corner=corner,
                preserve_action=preserve_action,
                abstain_action=abstain_action,
            )
        if not reason and not touches_baseline:
            dual_raw = len(providers) == 2
            if policy == "g1":
                reason = "policy_no_existing_context"
            elif policy == "g2" and not dual_raw:
                reason = "policy_no_dual_support"
            elif policy == "g3" and not (dual_raw or touches_source):
                reason = "policy_no_source_seed"
        accepted_component = not reason
        if accepted_component:
            local_accepted = accepted[slices]
            local_accepted[local_component] = 255
            reason = "accepted"
        records.append(
            ComponentAdmissionRecord(
                component_id=component_id,
                owner_region_id=owner_id,
                semantic_role=semantic.role,
                semantic_action=semantic.action,
                providers=providers,
                accepted=accepted_component,
                reason=reason,
                component_pixel_count=area,
                raw_core_pixel_count=raw_count,
                halo_pixel_count=area - raw_count,
                accepted_pixel_count=area if accepted_component else 0,
                touches_pr6_edit=touches_baseline,
                touches_source_raw_owned=touches_source,
            )
        )

    safe_addition = np.where(
        (accepted > 0) & (baseline == 0) & (hard_protection == 0),
        255,
        0,
    ).astype(np.uint8)
    if np.any((safe_addition > 0) & (proposal == 0)):
        raise AssertionError("safe addition escaped detector proposal")
    if np.any((safe_addition > 0) & (hard_protection > 0)):
        raise AssertionError("safe addition overlaps exact hard protection")
    connected_halo = np.where(
        (accepted > 0) & (raw_core == 0),
        255,
        0,
    ).astype(np.uint8)
    return ProposalRefinementResult(
        proposal=np.ascontiguousarray(proposal),
        raw_core=np.ascontiguousarray(raw_core),
        connected_halo=np.ascontiguousarray(connected_halo),
        accepted_claim=np.ascontiguousarray(accepted),
        safe_addition=np.ascontiguousarray(safe_addition),
        hard_protection=np.ascontiguousarray(hard_protection),
        component_records=tuple(records),
    )
