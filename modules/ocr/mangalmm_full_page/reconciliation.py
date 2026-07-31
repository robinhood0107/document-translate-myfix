"""MangaLMM region records and detector reconciliation policy."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from modules.utils.textblock import sort_textblock_rectangles


MANGALMM_NEAR_DUPLICATE_OVERLAP = 0.50
MANGALMM_COMPOUND_CONFLICT_OVERLAP = 0.20
MANGALMM_COMPOUND_MIN_PRECISION_COVER = 0.20
MANGALMM_DOMINANT_MIN_TEXT_CHARACTERS = 4
MANGALMM_DOMINANT_MIN_PRECISION_COVER = 0.80
MANGALMM_DOMINANT_MIN_OWNERSHIP_COVER = 0.80
MANGALMM_DOMINANT_MIN_OWNERSHIP_IOU = 0.12
MANGALMM_DOMINANT_MAX_CENTER_DISTANCE = 0.15
MANGALMM_DOMINANT_MAX_SECONDARY_IOU_RATIO = 0.35
MANGALMM_DOMINANT_MIN_CENTER_GAP = 0.15
MANGALMM_RECONCILIATION_SCHEMA_VERSION = 3


@dataclass(slots=True)
class OCRRegion:
    bbox_xyxy: list[int]
    bbox_xyxy_float: list[float]
    text: str
    unit_bbox_xyxy: list[int]
    unit_kind: str
    unit_resize_scale: float
    edge_distance: float
    normalized_text: str
    raw_text: str = ""
    response_bbox_2d: list[float] | None = None
    scale_x: float = 1.0
    scale_y: float = 1.0
    request_shape: list[int] | None = None
    resize_profile: str = "standard"


@dataclass(frozen=True)
class MangaCompoundDecision:
    accepted: bool
    ordered_items: tuple[dict[str, object], ...]
    duplicate_items: tuple[dict[str, object], ...]
    reason: str
    overlap_conflicts: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class MangaDominantRegionDecision:
    accepted: bool
    selected_item: dict[str, object] | None
    secondary_items: tuple[dict[str, object], ...]
    duplicate_items: tuple[dict[str, object], ...]
    reason: str


def _bbox(item: dict[str, object]) -> tuple[int, int, int, int]:
    region = item["region"]
    return tuple(int(value) for value in region.bbox_xyxy)


def _area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _overlap_of_smaller(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    intersection = max(
        0,
        min(left[2], right[2]) - max(left[0], right[0]),
    ) * max(
        0,
        min(left[3], right[3]) - max(left[1], right[1]),
    )
    smaller = min(_area(left), _area(right))
    return intersection / float(smaller) if smaller > 0 else 0.0


def _normalized_region_text(item: dict[str, object]) -> str:
    region = item["region"]
    return str(region.normalized_text or region.text or "").strip()


def _match_score(item: dict[str, object]) -> tuple[object, ...]:
    metrics = item.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    return (
        bool(metrics.get("center_in_precision", False)),
        float(metrics.get("precision_cover", 0.0) or 0.0),
        bool(metrics.get("center_in_ownership", False)),
        float(metrics.get("ownership_cover", 0.0) or 0.0),
        float(metrics.get("ownership_iou", 0.0) or 0.0),
    )


def _has_strong_precision_evidence(item: dict[str, object]) -> bool:
    metrics = item.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    return bool(metrics.get("center_in_precision", False)) or float(
        metrics.get("precision_cover", 0.0) or 0.0
    ) >= MANGALMM_COMPOUND_MIN_PRECISION_COVER


def _metric(item: dict[str, object], key: str) -> float:
    metrics = item.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    return float(metrics.get(key, 0.0) or 0.0)


def _semantic_character_count(item: dict[str, object]) -> int:
    return sum(
        1
        for character in _normalized_region_text(item)
        if unicodedata.category(character)[:1] in {"L", "N"}
    )


def _is_dominant_candidate(item: dict[str, object]) -> bool:
    metrics = item.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    return (
        _semantic_character_count(item)
        >= MANGALMM_DOMINANT_MIN_TEXT_CHARACTERS
        and bool(metrics.get("center_in_precision", False))
        and _metric(item, "precision_cover")
        >= MANGALMM_DOMINANT_MIN_PRECISION_COVER
        and _metric(item, "ownership_cover")
        >= MANGALMM_DOMINANT_MIN_OWNERSHIP_COVER
        and _metric(item, "ownership_iou")
        >= MANGALMM_DOMINANT_MIN_OWNERSHIP_IOU
        and _metric(item, "center_distance_norm")
        <= MANGALMM_DOMINANT_MAX_CENTER_DISTANCE
    )


def _collapse_near_duplicates(
    items: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    kept: list[dict[str, object]] = []
    duplicates: list[dict[str, object]] = []
    for item in items:
        duplicate_index = None
        for index, existing in enumerate(kept):
            if _normalized_region_text(item) != _normalized_region_text(
                existing
            ):
                continue
            if (
                _overlap_of_smaller(_bbox(item), _bbox(existing))
                < MANGALMM_NEAR_DUPLICATE_OVERLAP
            ):
                continue
            duplicate_index = index
            break
        if duplicate_index is None:
            kept.append(item)
            continue
        existing = kept[duplicate_index]
        if _match_score(item) > _match_score(existing):
            duplicates.append(existing)
            kept[duplicate_index] = item
        else:
            duplicates.append(item)
    return kept, duplicates


def _ordered_items(
    items: list[dict[str, object]],
    *,
    direction: str,
) -> list[dict[str, object]]:
    item_by_index = {str(index): item for index, item in enumerate(items)}
    ordering = sort_textblock_rectangles(
        [(_bbox(item), str(index)) for index, item in enumerate(items)],
        direction=(
            "ver_rtl"
            if str(direction or "").strip().lower() == "vertical"
            else "hor_ltr"
        ),
    )
    return [item_by_index[item_id] for _bbox_value, item_id in ordering]


def prepare_safe_detector_compound(
    items: list[dict[str, object]],
    *,
    allow_multi_region: bool,
    direction: str,
) -> MangaCompoundDecision:
    """Return a safe ordered N:1 compound or a fail-closed reason.

    Exact duplicates are removed by the engine before this boundary.  This
    policy additionally collapses near-identical geometry from the same raw
    text, then requires every surviving region to have strong text-box
    evidence and refuses overlapping distinct text.  It never splits one
    MangaLMM region across detector blocks.
    """

    kept, duplicates = _collapse_near_duplicates(list(items))
    if not kept:
        return MangaCompoundDecision(
            accepted=False,
            ordered_items=(),
            duplicate_items=tuple(duplicates),
            reason="no_regions_after_deduplication",
        )
    if len(kept) > 1 and not allow_multi_region:
        return MangaCompoundDecision(
            accepted=False,
            ordered_items=(),
            duplicate_items=tuple(duplicates),
            reason="missing_bubble_compound_boundary",
        )
    if any(not _has_strong_precision_evidence(item) for item in kept):
        return MangaCompoundDecision(
            accepted=False,
            ordered_items=(),
            duplicate_items=tuple(duplicates),
            reason="weak_precision_evidence",
        )

    conflicts: list[dict[str, object]] = []
    for left_index, left in enumerate(kept):
        for right_index in range(left_index + 1, len(kept)):
            right = kept[right_index]
            overlap = _overlap_of_smaller(_bbox(left), _bbox(right))
            if overlap < MANGALMM_COMPOUND_CONFLICT_OVERLAP:
                continue
            conflicts.append(
                {
                    "left_bbox_xyxy": list(_bbox(left)),
                    "right_bbox_xyxy": list(_bbox(right)),
                    "overlap_of_smaller": float(overlap),
                }
            )
    if conflicts:
        return MangaCompoundDecision(
            accepted=False,
            ordered_items=(),
            duplicate_items=tuple(duplicates),
            reason="overlapping_distinct_regions",
            overlap_conflicts=tuple(conflicts),
        )

    return MangaCompoundDecision(
        accepted=True,
        ordered_items=tuple(
            _ordered_items(kept, direction=direction)
        ),
        duplicate_items=tuple(duplicates),
        reason=(
            "near_duplicates_collapsed"
            if duplicates
            else "safe_ordered_compound"
        ),
    )


def prepare_dominant_detector_region(
    items: list[dict[str, object]],
    *,
    text_class: str,
    has_bubble: bool,
) -> MangaDominantRegionDecision:
    """Select one unambiguous main dialogue while preserving noise as shadow.

    MangaLMM can emit a correct long dialogue region plus a small, distant SFX
    region inside the same detected bubble.  The normal compound policy must
    reject overlapping or weak regions as a group.  This narrower recovery is
    allowed only for an explicit text bubble and only when exactly one region
    has strong text-box evidence while every secondary is geometrically much
    weaker and farther away.  It never combines or discards secondary text.
    """

    kept, duplicates = _collapse_near_duplicates(list(items))
    rejected = MangaDominantRegionDecision(
        accepted=False,
        selected_item=None,
        secondary_items=tuple(kept),
        duplicate_items=tuple(duplicates),
        reason="dominant_region_not_found",
    )
    if str(text_class or "") != "text_bubble" or not has_bubble:
        return MangaDominantRegionDecision(
            accepted=False,
            selected_item=None,
            secondary_items=tuple(kept),
            duplicate_items=tuple(duplicates),
            reason="dominant_region_requires_bubble",
        )
    if len(kept) < 2:
        return MangaDominantRegionDecision(
            accepted=False,
            selected_item=None,
            secondary_items=tuple(kept),
            duplicate_items=tuple(duplicates),
            reason="dominant_region_requires_secondary",
        )

    candidates = [item for item in kept if _is_dominant_candidate(item)]
    if len(candidates) != 1:
        return MangaDominantRegionDecision(
            accepted=False,
            selected_item=None,
            secondary_items=tuple(kept),
            duplicate_items=tuple(duplicates),
            reason=(
                "multiple_dominant_regions"
                if len(candidates) > 1
                else rejected.reason
            ),
        )

    selected = candidates[0]
    secondaries = [item for item in kept if item is not selected]
    selected_iou = _metric(selected, "ownership_iou")
    selected_distance = _metric(selected, "center_distance_norm")
    for secondary in secondaries:
        if (
            _metric(secondary, "ownership_iou")
            > selected_iou * MANGALMM_DOMINANT_MAX_SECONDARY_IOU_RATIO
        ):
            return MangaDominantRegionDecision(
                accepted=False,
                selected_item=None,
                secondary_items=tuple(kept),
                duplicate_items=tuple(duplicates),
                reason="secondary_region_not_weak_enough",
            )
        if (
            _metric(secondary, "center_distance_norm")
            < selected_distance + MANGALMM_DOMINANT_MIN_CENTER_GAP
        ):
            return MangaDominantRegionDecision(
                accepted=False,
                selected_item=None,
                secondary_items=tuple(kept),
                duplicate_items=tuple(duplicates),
                reason="secondary_region_not_distant_enough",
            )

    return MangaDominantRegionDecision(
        accepted=True,
        selected_item=selected,
        secondary_items=tuple(secondaries),
        duplicate_items=tuple(duplicates),
        reason="safe_dominant_dialogue_region",
    )


__all__ = [
    "MANGALMM_COMPOUND_CONFLICT_OVERLAP",
    "MANGALMM_COMPOUND_MIN_PRECISION_COVER",
    "MANGALMM_DOMINANT_MAX_CENTER_DISTANCE",
    "MANGALMM_DOMINANT_MAX_SECONDARY_IOU_RATIO",
    "MANGALMM_DOMINANT_MIN_CENTER_GAP",
    "MANGALMM_DOMINANT_MIN_OWNERSHIP_COVER",
    "MANGALMM_DOMINANT_MIN_OWNERSHIP_IOU",
    "MANGALMM_DOMINANT_MIN_PRECISION_COVER",
    "MANGALMM_DOMINANT_MIN_TEXT_CHARACTERS",
    "MANGALMM_NEAR_DUPLICATE_OVERLAP",
    "MANGALMM_RECONCILIATION_SCHEMA_VERSION",
    "MangaCompoundDecision",
    "MangaDominantRegionDecision",
    "OCRRegion",
    "prepare_dominant_detector_region",
    "prepare_safe_detector_compound",
]
