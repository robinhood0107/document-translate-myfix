from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from .contracts import (
    BUBBLE_ROUTE_CLASSES,
    BubbleRouteDecision,
    CandidateMaskResult,
    DetectorBox,
    RoleCandidateSpec,
    Stage1Page,
    TargetInstance,
    assert_disjoint_masks,
    binary_mask,
    mask_sha256,
    tensor_sha256,
)


@dataclass(frozen=True, slots=True)
class PageMasks:
    target: np.ndarray
    protected: np.ndarray
    ambiguous: np.ndarray
    ownership: np.ndarray
    claim_seed: np.ndarray
    existing_edit: np.ndarray
    target_instances: tuple[tuple[str, np.ndarray], ...] = ()
    bubble_interior: np.ndarray | None = None
    corner: np.ndarray | None = None
    broad_ownership: np.ndarray | None = None


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
    schema_version = str(payload.get("schema_version") or "").strip()
    is_v3 = schema_version == "inpaint-detector-bakeoff-manifest-v3"
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
        expected_edit = str(entry.get("expected_edit") or "required").strip().lower()
        if expected_edit not in {"required", "none"}:
            raise ValueError(f"invalid expected_edit for {entry.get('page_id')}: {expected_edit}")
        route_class = str(entry.get("bubble_route_class") or "").strip().lower()
        target_instances: list[TargetInstance] = []
        raw_instances = entry.get("target_instances", [])
        if not isinstance(raw_instances, list):
            raise ValueError("target_instances must be an array")
        for instance in raw_instances:
            if not isinstance(instance, dict):
                raise ValueError("target instance must be an object")
            instance_id = str(instance.get("instance_id") or "").strip()
            instance_path = _path_value(
                instance.get("mask_path", instance.get("mask"))
            )
            if not instance_id or instance_path is None:
                raise ValueError("target instance requires instance_id and mask_path")
            target_instances.append(TargetInstance(instance_id, instance_path))
        if len({record.instance_id for record in target_instances}) != len(target_instances):
            raise ValueError("target instance ids must be unique within a page")
        if is_v3:
            required_fields = {
                "target_text_mask",
                "target_instances",
                "bubble_route_class",
                "bubble_interior_mask",
                "protected_structure_mask",
                "ambiguous_structure_mask",
                "ownership_mask",
                "expected_edit",
            }
            missing = sorted(required_fields.difference(entry))
            if missing:
                raise ValueError("manifest v3 page is missing fields: " + ", ".join(missing))
            if route_class not in BUBBLE_ROUTE_CLASSES:
                raise ValueError(f"invalid bubble_route_class: {route_class}")
            if expected_edit == "required" and (target is None or not target_instances):
                raise ValueError("required manifest v3 page needs target text and instances")
            if expected_edit == "none" and (target is not None or target_instances):
                raise ValueError("no-edit manifest v3 page cannot contain target instances")
        records.append(
            Stage1Page(
                page_id=str(entry.get("page_id") or "").strip(),
                source_image=str(entry.get("path") or "").strip(),
                target_text_mask=target,
                protected_structure_mask=_path_value(entry.get("protected_structure_mask")),
                ambiguous_structure_mask=_path_value(entry.get("ambiguous_structure_mask")),
                ownership_mask=_path_value(entry.get("ownership_mask")),
                claim_seed_mask=_path_value(entry.get("claim_seed_mask")),
                no_edit=expected_edit == "none",
                target_instances=tuple(target_instances),
                bubble_route_class=route_class or None,
                bubble_interior_mask=_path_value(entry.get("bubble_interior_mask")),
                corner_protect_mask=_path_value(entry.get("corner_protect_mask")),
                expected_edit=expected_edit,
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


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    target_instances = tuple(
        (record.instance_id, _read_mask(record.mask_path, shape))
        for record in page.target_instances
    )
    if target_instances:
        instance_union = np.zeros(shape, dtype=np.uint8)
        occupied = np.zeros(shape, dtype=np.uint8)
        for instance_id, instance_mask in target_instances:
            if not np.any(instance_mask):
                raise ValueError(f"target instance mask is empty: {instance_id}")
            if np.any((occupied > 0) & (instance_mask > 0)):
                raise ValueError(f"target instance masks overlap: {instance_id}")
            occupied[instance_mask > 0] = 255
            instance_union[instance_mask > 0] = 255
        if not np.array_equal(instance_union, target):
            raise ValueError("target_text_mask must equal the union of target_instances")
    bubble_interior = _read_mask(page.bubble_interior_mask, shape)
    corner = _read_mask(page.corner_protect_mask, shape)
    assert_disjoint_masks(
        {
            "target_text_mask": target,
            "protected_structure_mask": protected,
            "ambiguous_structure_mask": ambiguous,
        }
    )
    return PageMasks(
        target,
        protected,
        ambiguous,
        ownership,
        claim_seed,
        existing_edit,
        target_instances,
        bubble_interior,
        corner,
    )


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
    if masks.corner is not None:
        edit[masks.corner > 0] = 0
    return np.ascontiguousarray(edit)


def _components_touching_seed(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    normalized = binary_mask(mask, seed.shape)
    count, labels = cv2.connectedComponents((normalized > 0).astype(np.uint8), 8)
    if count <= 1 or not np.any(seed):
        return normalized if np.any(seed & normalized) else np.zeros_like(normalized)
    selected_labels = np.unique(labels[seed > 0])
    selected_labels = selected_labels[selected_labels > 0]
    if selected_labels.size == 0:
        return np.zeros_like(normalized)
    return np.where(np.isin(labels, selected_labels), 255, 0).astype(np.uint8)


def expand_detector_claim(
    expansion_id: str,
    *,
    seed: np.ndarray,
    raw: np.ndarray,
    refined: np.ndarray,
    dilated: np.ndarray,
    content_components: np.ndarray | None = None,
    bubble_interior: np.ndarray | None = None,
) -> np.ndarray:
    """Build one mask expansion without inventing pixels from OCR geometry."""

    shape = binary_mask(seed).shape
    normalized_seed = binary_mask(seed, shape)
    if not np.any(normalized_seed):
        return np.zeros(shape, dtype=np.uint8)
    expansion = str(expansion_id).strip().lower()
    if expansion == "raw":
        return binary_mask(raw, shape)
    if expansion == "refined":
        return binary_mask(refined, shape)
    if expansion in {"native3px", "dilated", "dilate3"}:
        return binary_mask(dilated, shape)
    if expansion == "content_component":
        if content_components is None:
            return np.zeros(shape, dtype=np.uint8)
        return _components_touching_seed(
            binary_mask(content_components, shape),
            normalized_seed,
        )
    if expansion == "bubble_interior":
        if bubble_interior is None:
            return np.zeros(shape, dtype=np.uint8)
        return _components_touching_seed(
            binary_mask(bubble_interior, shape),
            normalized_seed,
        )
    raise KeyError(f"unknown mask expansion: {expansion_id}")


def _subtract_exact_protection(claim: np.ndarray, masks: PageMasks) -> np.ndarray:
    edit = np.where(
        (binary_mask(claim, masks.target.shape) > 0)
        & (masks.existing_edit == 0)
        & (masks.protected == 0)
        & (masks.ambiguous == 0),
        255,
        0,
    ).astype(np.uint8)
    if masks.corner is not None:
        edit[masks.corner > 0] = 0
    return np.ascontiguousarray(edit)


def decide_bubble_route(
    router_id: str,
    *,
    narrow_claim: np.ndarray,
    broad_claim: np.ndarray,
    seed: np.ndarray,
    masks: PageMasks,
    ballons_clean: bool = False,
    pr2_clean: bool = False,
    segmentation_valid: bool = False,
    texture: bool = False,
    microtexture: bool = False,
    line_art: bool = False,
    ambiguous: bool = False,
    background_sample_count: int = 0,
    minimum_background_samples: int = 32,
    ballons_clean_mask: np.ndarray | None = None,
    pr2_clean_mask: np.ndarray | None = None,
    segmentation_valid_mask: np.ndarray | None = None,
    unsafe_signal_mask: np.ndarray | None = None,
) -> BubbleRouteDecision:
    """Choose narrow/broad without allowing a router to bypass exact protection."""

    normalized_router = str(router_id).strip().upper()
    if normalized_router not in {"R0", "R1", "R2", "R3", "R4"}:
        raise KeyError(f"unknown bubble router: {router_id}")
    shape = masks.target.shape
    seed_mask = binary_mask(seed, shape)
    narrow = binary_mask(narrow_claim, shape)
    broad = binary_mask(broad_claim, shape)
    interior = (
        binary_mask(masks.bubble_interior, shape)
        if masks.bubble_interior is not None
        else np.zeros(shape, dtype=np.uint8)
    )
    broad_ownership = (
        binary_mask(masks.broad_ownership, shape)
        if masks.broad_ownership is not None
        else binary_mask(masks.ownership, shape)
    )
    reasons: list[str] = []
    explicit_route_masks = any(
        value is not None
        for value in (
            ballons_clean_mask,
            pr2_clean_mask,
            segmentation_valid_mask,
            unsafe_signal_mask,
        )
    )
    full = np.full(shape, 255, np.uint8)
    ballons_allow = (
        binary_mask(ballons_clean_mask, shape)
        if ballons_clean_mask is not None
        else (full if ballons_clean else np.zeros(shape, np.uint8))
    )
    pr2_allow = (
        binary_mask(pr2_clean_mask, shape)
        if pr2_clean_mask is not None
        else (full if pr2_clean else np.zeros(shape, np.uint8))
    )
    segmentation_allow = (
        binary_mask(segmentation_valid_mask, shape)
        if segmentation_valid_mask is not None
        else (full if segmentation_valid else np.zeros(shape, np.uint8))
    )
    unsafe = (
        binary_mask(unsafe_signal_mask, shape)
        if unsafe_signal_mask is not None
        else np.zeros(shape, np.uint8)
    )
    route_allow = np.zeros(shape, np.uint8)
    if normalized_router == "R1":
        route_allow = ballons_allow
    elif normalized_router == "R2":
        route_allow = pr2_allow
    elif normalized_router == "R3":
        route_allow = cv2.bitwise_and(ballons_allow, pr2_allow)
    elif normalized_router == "R4":
        route_allow = segmentation_allow
    route_allow[unsafe > 0] = 0
    eligible_broad = np.where(
        (broad > 0) & (route_allow > 0), 255, 0
    ).astype(np.uint8)
    eligible_seed = np.where(
        (seed_mask > 0) & (route_allow > 0), 255, 0
    ).astype(np.uint8)
    if normalized_router == "R0":
        reasons.append("router_narrow_only")
    else:
        if not np.any(seed_mask):
            reasons.append("detector_seed_missing")
        if not np.any(interior):
            reasons.append("validated_bubble_interior_missing")
        if np.any(eligible_seed) and np.any(
            (eligible_seed > 0) & (masks.ownership == 0)
        ):
            reasons.append("seed_outside_ownership")
        if np.any(eligible_seed) and np.any((eligible_seed > 0) & (interior == 0)):
            reasons.append("seed_outside_bubble_interior")
        if np.any((eligible_broad > 0) & (broad_ownership == 0)):
            reasons.append("broad_outside_ownership")
        if np.any((eligible_broad > 0) & (interior == 0)):
            reasons.append("broad_outside_bubble_interior")
        if texture and not explicit_route_masks:
            reasons.append("texture_present")
        if microtexture and not explicit_route_masks:
            reasons.append("microtexture_present")
        if line_art and not explicit_route_masks:
            reasons.append("line_art_present")
        if ambiguous and not explicit_route_masks:
            reasons.append("ambiguous_structure_present")
        if np.any((eligible_broad > 0) & (masks.protected > 0)):
            reasons.append("broad_overlaps_structure_protect")
        if np.any((eligible_broad > 0) & (masks.ambiguous > 0)):
            reasons.append("broad_overlaps_ambiguous_protect")
        if masks.corner is not None and np.any(
            (eligible_broad > 0) & (masks.corner > 0)
        ):
            reasons.append("broad_overlaps_corner_protect")
        if int(background_sample_count) < int(minimum_background_samples):
            reasons.append("insufficient_roi_background_samples")
        if normalized_router == "R1" and not np.any(ballons_allow):
            reasons.append("ballons_not_clean")
        elif normalized_router == "R2" and not np.any(pr2_allow):
            reasons.append("pr2_not_clean")
        elif normalized_router == "R3" and not np.any(
            cv2.bitwise_and(ballons_allow, pr2_allow)
        ):
            reasons.append("clean_consensus_missing")
        elif normalized_router == "R4" and not np.any(segmentation_allow):
            reasons.append("segmentation_silhouette_invalid")

    allow_broad = normalized_router != "R0" and not reasons and np.any(eligible_broad)
    owned_narrow = np.where(
        (narrow > 0) & (masks.ownership > 0), 255, 0
    ).astype(np.uint8)
    owned_broad = np.where(
        (eligible_broad > 0) & (broad_ownership > 0), 255, 0
    ).astype(np.uint8)
    chosen = (
        cv2.bitwise_or(owned_narrow, owned_broad)
        if allow_broad
        else owned_narrow
    )
    edit = _subtract_exact_protection(chosen, masks)
    decision = "broad" if allow_broad else ("narrow" if np.any(edit) else "skip")
    if not allow_broad and not reasons and not np.any(broad):
        reasons.append("broad_candidate_empty")
    if not np.any(edit):
        reasons.append("post_protection_edit_empty")
    return BubbleRouteDecision(
        router_id=normalized_router,
        decision=decision,
        edit_mask=edit,
        interior_mask=interior,
        reasons=tuple(reasons),
    )


def _target_instances(masks: PageMasks) -> tuple[tuple[str, np.ndarray], ...]:
    if masks.target_instances:
        return masks.target_instances
    count, labels = cv2.connectedComponents((masks.target > 0).astype(np.uint8), 8)
    return tuple(
        (
            f"component-{index}",
            np.where(labels == index, 255, 0).astype(np.uint8),
        )
        for index in range(1, count)
    )


def _instance_scores(
    instances: tuple[tuple[str, np.ndarray], ...],
    mask: np.ndarray,
) -> list[dict[str, object]]:
    scores: list[dict[str, object]] = []
    for instance_id, target in instances:
        pixels = int(np.count_nonzero(target))
        covered = int(np.count_nonzero((target > 0) & (mask > 0)))
        scores.append(
            {
                "instance_id": instance_id,
                "target_pixel_count": pixels,
                "covered_pixel_count": covered,
                "coverage": float(covered) / float(pixels) if pixels else None,
                "seeded": covered > 0,
            }
        )
    return scores


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
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        component = labels[y : y + height, x : x + width] == index
        local_claim = claim[y : y + height, x : x + width] > 0
        covered = int(np.count_nonzero(component & local_claim))
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
    instances = _target_instances(masks)
    instance_seed_scores = _instance_scores(instances, claim)
    instance_edit_scores = _instance_scores(instances, edit)
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
        "target_instance_count": len(instances),
        "seeded_target_instance_count": sum(
            1 for value in instance_seed_scores if value["seeded"]
        ),
        "missed_target_instance_ids": [
            str(value["instance_id"])
            for value in instance_seed_scores
            if not value["seeded"]
        ],
        "target_instance_seed_recall": (
            float(sum(1 for value in instance_seed_scores if value["seeded"]))
            / float(len(instance_seed_scores))
            if instance_seed_scores
            else None
        ),
        "target_instance_seed_scores": instance_seed_scores,
        "target_instance_edit_scores": instance_edit_scores,
        "minimum_target_instance_edit_coverage": (
            min(float(value["coverage"]) for value in instance_edit_scores)
            if instance_edit_scores
            else None
        ),
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
    seed_scores = [
        value
        for row in rows
        for value in row.get("target_instance_seed_scores", [])
    ]
    edit_scores = [
        value
        for row in rows
        for value in row.get("target_instance_edit_scores", [])
    ]
    return {
        "page_count": len(rows),
        "target_pixel_count": target_pixels,
        "target_edit_pixel_count": target_edit,
        "aggregate_target_coverage": (
            float(target_edit) / float(target_pixels) if target_pixels else None
        ),
        "minimum_component_coverage": min(components) if components else None,
        "target_instance_count": len(seed_scores),
        "seeded_target_instance_count": sum(1 for value in seed_scores if value["seeded"]),
        "missed_target_instance_count": sum(
            1 for value in seed_scores if not value["seeded"]
        ),
        "target_instance_seed_recall": (
            float(sum(1 for value in seed_scores if value["seeded"]))
            / float(len(seed_scores))
            if seed_scores
            else None
        ),
        "minimum_target_instance_edit_coverage": (
            min(float(value["coverage"]) for value in edit_scores)
            if edit_scores
            else None
        ),
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


def detector_cache_key(spec: RoleCandidateSpec, source_sha256: str) -> str:
    return spec.cache_key(source_sha256)


def write_detector_cache(
    cache_root: Path,
    spec: RoleCandidateSpec,
    source_sha256: str,
    result: CandidateMaskResult,
) -> Path:
    key = detector_cache_key(spec, source_sha256)
    entry = cache_root / key
    entry.mkdir(parents=True, exist_ok=True)
    arrays_path = entry / "masks.npz"
    temporary_arrays = entry / ".masks.npz.partial"
    arrays = {
        "raw": result.raw_mask,
        "refined": result.refined_mask,
        "dilated": result.dilated_mask,
    }
    for name, tensor in sorted(result.stage_tensors.items()):
        arrays[f"stage__{name}"] = np.ascontiguousarray(tensor)
    with temporary_arrays.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary_arrays.replace(arrays_path)
    metadata = {
        "schema_version": "inpaint-detector-cache-v1",
        "cache_key": key,
        "cache_payload": spec.cache_payload(source_sha256),
        "candidate_id": result.candidate_id,
        "mask_sha256": {
            "raw": mask_sha256(result.raw_mask),
            "refined": mask_sha256(result.refined_mask),
            "dilated": mask_sha256(result.dilated_mask),
        },
        "stage_sha256": {
            name: tensor_sha256(tensor)
            for name, tensor in sorted(result.stage_tensors.items())
        },
        "runtime": dict(result.runtime),
        "boxes": [
            {
                "xyxy": list(box.xyxy),
                "label": box.label,
                "score": float(box.score),
                "provider": box.provider,
            }
            for box in result.boxes
        ],
    }
    metadata_path = entry / "metadata.json"
    temporary_metadata = entry / ".metadata.json.partial"
    temporary_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_metadata.replace(metadata_path)
    return entry


def read_detector_cache(
    cache_root: Path,
    spec: RoleCandidateSpec,
    source_sha256: str,
) -> CandidateMaskResult | None:
    key = detector_cache_key(spec, source_sha256)
    entry = cache_root / key
    metadata_path = entry / "metadata.json"
    arrays_path = entry / "masks.npz"
    if not metadata_path.exists() and not arrays_path.exists():
        return None
    if not metadata_path.is_file() or not arrays_path.is_file():
        raise ValueError(f"incomplete detector cache entry: {key}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("cache_key") != key:
        raise ValueError("detector cache key mismatch")
    if metadata.get("cache_payload") != spec.cache_payload(source_sha256):
        raise ValueError("detector cache provenance mismatch")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        stage_tensors = {
            name[len("stage__") :]: arrays[name]
            for name in arrays.files
            if name.startswith("stage__")
        }
        boxes = tuple(
            DetectorBox(
                tuple(int(value) for value in box["xyxy"]),
                str(box["label"]),
                float(box["score"]),
                str(box["provider"]),
            )
            for box in metadata.get("boxes", [])
        )
        recorded_runtime = metadata.get("runtime", {})
        if not isinstance(recorded_runtime, dict):
            raise ValueError("detector cache runtime metadata must be an object")
        result = CandidateMaskResult(
            candidate_id=str(metadata.get("candidate_id") or spec.candidate_id),
            raw_mask=arrays["raw"],
            refined_mask=arrays["refined"],
            dilated_mask=arrays["dilated"],
            boxes=boxes,
            stage_tensors=stage_tensors,
            runtime={**recorded_runtime, "cache_hit": True, "cache_key": key},
        )
    recorded = metadata.get("mask_sha256")
    actual = {
        "raw": mask_sha256(result.raw_mask),
        "refined": mask_sha256(result.refined_mask),
        "dilated": mask_sha256(result.dilated_mask),
    }
    if recorded != actual:
        raise ValueError("detector cache mask SHA mismatch")
    recorded_stage = metadata.get("stage_sha256", {})
    actual_stage = {
        name: tensor_sha256(tensor)
        for name, tensor in sorted(result.stage_tensors.items())
    }
    if recorded_stage != actual_stage:
        raise ValueError("detector cache stage tensor SHA mismatch")
    return result


def run_stage1(
    pages: Iterable[Stage1Page],
    infer: Callable[[np.ndarray], CandidateMaskResult],
    *,
    variant: str,
    existing_edit_paths: dict[str, str] | None = None,
    candidate_spec: RoleCandidateSpec | None = None,
    cache_root: Path | None = None,
    result_sink: Callable[[Stage1Page, CandidateMaskResult], None] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, np.ndarray]]:
    if (candidate_spec is None) != (cache_root is None):
        raise ValueError("candidate_spec and cache_root must be provided together")
    rows: list[dict[str, object]] = []
    edits: dict[str, np.ndarray] = {}
    for page in pages:
        image = _read_image(page.source_image)
        result = None
        source_sha256 = None
        if candidate_spec is not None and cache_root is not None:
            source_sha256 = _file_sha256(page.source_image)
            result = read_detector_cache(cache_root, candidate_spec, source_sha256)
        if result is None:
            result = infer(image)
            if candidate_spec is not None and cache_root is not None:
                assert source_sha256 is not None
                write_detector_cache(cache_root, candidate_spec, source_sha256, result)
        if result_sink is not None:
            result_sink(page, result)
        masks = load_page_masks(
            page,
            image.shape[:2],
            existing_edit_path=(existing_edit_paths or {}).get(page.page_id),
        )
        row, edit = score_page(page, result, masks, variant=variant)
        rows.append(row)
        edits[page.page_id] = edit
    return rows, summarize(rows), edits
