from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from modules.utils.textblock import ensure_text_block_id

from .response_parser import (
    PADDLE_SPOTTING_COORDINATE_SCALE,
    PaddleSpottingRegion,
)


@dataclass(frozen=True)
class MappedSpottingRegion:
    text: str
    normalized_points: tuple[
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
    ]
    points: tuple[
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
    ]
    bbox_xyxy: tuple[int, int, int, int]
    source_line: int


@dataclass(frozen=True)
class SpottingAssignment:
    region_index: int
    block_index: int
    region: MappedSpottingRegion
    region_coverage: float
    block_coverage: float
    iou: float
    center_inside: bool


@dataclass(frozen=True)
class SpottingGeometryResult:
    assignments: dict[int, tuple[SpottingAssignment, ...]]
    unmatched_regions: tuple[MappedSpottingRegion, ...]
    ambiguous_regions: tuple[dict[str, object], ...]
    ambiguous_block_indices: tuple[int, ...]
    block_diagnostics: dict[int, dict[str, object]]
    relation_components: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class _CandidateRelation:
    block_index: int
    region_coverage: float
    block_coverage: float
    iou: float
    center_inside: bool
    block_area: int


def _clip(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def map_normalized_region(
    region: PaddleSpottingRegion,
    *,
    image_width: int,
    image_height: int,
) -> MappedSpottingRegion:
    width = max(1, int(image_width))
    height = max(1, int(image_height))
    points = tuple(
        (
            _clip(
                round(
                    point[0]
                    / PADDLE_SPOTTING_COORDINATE_SCALE
                    * width
                ),
                0,
                width,
            ),
            _clip(
                round(
                    point[1]
                    / PADDLE_SPOTTING_COORDINATE_SCALE
                    * height
                ),
                0,
                height,
            ),
        )
        for point in region.normalized_points
    )
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    return MappedSpottingRegion(
        text=region.text,
        normalized_points=region.normalized_points,
        points=points,  # type: ignore[arg-type]
        bbox_xyxy=bbox,
        source_line=region.source_line,
    )


def _block_bbox(block: Any, field_name: str = "xyxy") -> tuple[int, int, int, int] | None:
    try:
        values = tuple(
            int(round(float(value)))
            for value in getattr(block, field_name, ())
        )
    except (TypeError, ValueError):
        return None
    if len(values) != 4:
        return None
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> int:
    return max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0,
        min(left[3], right[3]) - max(left[1], right[1]),
    )


def _center_inside(
    inner: tuple[int, int, int, int],
    outer: tuple[int, int, int, int],
) -> bool:
    center_x = (inner[0] + inner[2]) / 2.0
    center_y = (inner[1] + inner[3]) / 2.0
    return (
        outer[0] <= center_x <= outer[2]
        and outer[1] <= center_y <= outer[3]
    )


def _candidate_metrics(
    region: MappedSpottingRegion,
    block_box: tuple[int, int, int, int],
) -> tuple[float, float, float, bool]:
    region_area = _area(region.bbox_xyxy)
    block_area = _area(block_box)
    intersection = _intersection(region.bbox_xyxy, block_box)
    union = region_area + block_area - intersection
    return (
        intersection / region_area if region_area else 0.0,
        intersection / block_area if block_area else 0.0,
        intersection / union if union else 0.0,
        _center_inside(region.bbox_xyxy, block_box),
    )


def _is_ambiguous(
    best: _CandidateRelation,
    second: _CandidateRelation,
) -> bool:
    if best.block_coverage >= 0.50 and second.block_coverage >= 0.50:
        return True
    if best.center_inside and second.center_inside:
        return abs(best.region_coverage - second.region_coverage) < 0.10
    return (
        best.region_coverage >= 0.50
        and second.region_coverage >= 0.50
        and abs(best.region_coverage - second.region_coverage) < 0.05
        and abs(best.iou - second.iou) < 0.05
    )


def _candidate_sort_key(
    candidate: _CandidateRelation,
) -> tuple[bool, float, float, float, int]:
    return (
        candidate.center_inside,
        candidate.region_coverage,
        candidate.iou,
        candidate.block_coverage,
        -candidate.block_area,
    )


def _relation_type(region_count: int, block_count: int) -> str:
    if block_count == 0:
        return "unmatched_region"
    if region_count == 1 and block_count == 1:
        return "one_to_one"
    if region_count > 1 and block_count == 1:
        return "many_lines_to_one_block"
    if region_count == 1 and block_count > 1:
        return "one_region_to_many_blocks"
    return "many_to_many"


def _relation_components(
    candidate_map: dict[int, tuple[_CandidateRelation, ...]],
    *,
    region_count: int,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Return connected region/block components for global diagnostics.

    Detector blocks may accept multiple native Spotting lines, but a native
    line must never be copied to more than one block.  Building the complete
    bipartite graph first makes N:1 and 1:N relationships explicit before a
    local best edge is selected.
    """

    block_to_regions: dict[int, set[int]] = {}
    for region_index, candidates in candidate_map.items():
        for candidate in candidates:
            block_to_regions.setdefault(candidate.block_index, set()).add(
                region_index
            )

    pending_regions = set(range(region_count))
    components: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    while pending_regions:
        seed = min(pending_regions)
        region_indices = {seed}
        block_indices: set[int] = set()
        region_queue = [seed]
        while region_queue:
            region_index = region_queue.pop()
            for candidate in candidate_map.get(region_index, ()):
                block_index = candidate.block_index
                if block_index in block_indices:
                    continue
                block_indices.add(block_index)
                for linked_region in block_to_regions.get(block_index, ()):
                    if linked_region not in region_indices:
                        region_indices.add(linked_region)
                        region_queue.append(linked_region)
        pending_regions.difference_update(region_indices)
        components.append(
            (tuple(sorted(region_indices)), tuple(sorted(block_indices)))
        )
    return components


def _union_area(rectangles: Iterable[tuple[int, int, int, int]]) -> int:
    rects = [rect for rect in rectangles if _area(rect) > 0]
    if not rects:
        return 0
    xs = sorted({value for rect in rects for value in (rect[0], rect[2])})
    total = 0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (rect[1], rect[3])
            for rect in rects
            if rect[0] < right and rect[2] > left
        )
        if not intervals:
            continue
        merged_height = 0
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start > end:
                merged_height += max(0, end - start)
                start, end = next_start, next_end
            else:
                end = max(end, next_end)
        merged_height += max(0, end - start)
        total += (right - left) * merged_height
    return total


def _reading_order_key(
    assignment: SpottingAssignment,
    direction: str,
) -> tuple[float, float, int]:
    x1, y1, x2, y2 = assignment.region.bbox_xyxy
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    if str(direction or "").strip().lower() == "vertical":
        return (-center_x, center_y, assignment.region.source_line)
    return (center_y, center_x, assignment.region.source_line)


def assign_spotting_regions(
    regions: Iterable[PaddleSpottingRegion],
    blocks: list[Any],
    *,
    image_width: int,
    image_height: int,
) -> SpottingGeometryResult:
    mapped_regions = [
        map_normalized_region(
            region,
            image_width=image_width,
            image_height=image_height,
        )
        for region in regions
    ]
    block_boxes = [_block_bbox(block) for block in blocks]
    candidate_map: dict[int, tuple[_CandidateRelation, ...]] = {}
    for region_index, region in enumerate(mapped_regions):
        candidates: list[_CandidateRelation] = []
        for block_index, block_box in enumerate(block_boxes):
            if block_box is None:
                continue
            region_cover, block_cover, iou, center_inside = (
                _candidate_metrics(region, block_box)
            )
            if (
                region_cover < 0.50
                and block_cover < 0.50
                and not center_inside
            ):
                continue
            candidates.append(
                _CandidateRelation(
                    block_index=block_index,
                    region_coverage=region_cover,
                    block_coverage=block_cover,
                    iou=iou,
                    center_inside=center_inside,
                    block_area=_area(block_box),
                )
            )
        candidates.sort(key=_candidate_sort_key, reverse=True)
        candidate_map[region_index] = tuple(candidates)

    components = _relation_components(
        candidate_map,
        region_count=len(mapped_regions),
    )
    component_ids_by_block: dict[int, list[str]] = {
        index: [] for index in range(len(blocks))
    }
    for component_index, (_region_indices, block_indices) in enumerate(
        components
    ):
        component_id = f"component-{component_index:04d}"
        for block_index in block_indices:
            component_ids_by_block[block_index].append(component_id)

    proposed: dict[int, list[SpottingAssignment]] = {
        index: [] for index in range(len(blocks))
    }
    unmatched: list[MappedSpottingRegion] = []
    ambiguous: list[dict[str, object]] = []
    ambiguous_block_indices: set[int] = set()
    ambiguous_region_indices: set[int] = set()
    for region_index, region in enumerate(mapped_regions):
        candidates = candidate_map[region_index]
        if not candidates:
            unmatched.append(region)
            continue
        if len(candidates) > 1 and _is_ambiguous(
            candidates[0], candidates[1]
        ):
            candidate_indices = [
                candidate.block_index for candidate in candidates
            ]
            ambiguous_region_indices.add(region_index)
            ambiguous_block_indices.update(candidate_indices)
            ambiguous.append(
                {
                    "region_index": region_index,
                    "source_line": region.source_line,
                    "text": region.text,
                    "bbox_xyxy": list(region.bbox_xyxy),
                    "candidate_block_ids": [
                        ensure_text_block_id(blocks[block_index])
                        for block_index in candidate_indices
                    ],
                    "candidate_metrics": [
                        {
                            "block_id": ensure_text_block_id(
                                blocks[candidate.block_index]
                            ),
                            "region_coverage": candidate.region_coverage,
                            "block_coverage": candidate.block_coverage,
                            "iou": candidate.iou,
                            "center_inside": candidate.center_inside,
                        }
                        for candidate in candidates
                    ],
                    "reason": "one_spot_multiple_detector_blocks",
                    "processing_action": "review",
                }
            )
            continue
        best = candidates[0]
        proposed[best.block_index].append(
            SpottingAssignment(
                region_index=region_index,
                block_index=best.block_index,
                region=region,
                region_coverage=best.region_coverage,
                block_coverage=best.block_coverage,
                iou=best.iou,
                center_inside=best.center_inside,
            )
        )

    frozen_assignments: dict[int, tuple[SpottingAssignment, ...]] = {}
    block_diagnostics: dict[int, dict[str, object]] = {}
    ambiguous_regions_by_block: dict[int, set[int]] = {
        index: set() for index in range(len(blocks))
    }
    for region_index in ambiguous_region_indices:
        for candidate in candidate_map[region_index]:
            ambiguous_regions_by_block[candidate.block_index].add(region_index)
    for block_index, items in proposed.items():
        block_box = block_boxes[block_index]
        if block_index in ambiguous_block_indices:
            frozen_assignments[block_index] = ()
            block_diagnostics[block_index] = {
                "block_id": ensure_text_block_id(blocks[block_index]),
                "status": "ambiguous",
                "matched_region_count": 0,
                "deferred_region_indices": sorted(
                    {item.region_index for item in items}
                    | ambiguous_regions_by_block[block_index]
                ),
                "detector_coverage": 0.0,
                "component_ids": component_ids_by_block[block_index],
            }
            continue
        direction = str(getattr(blocks[block_index], "direction", "") or "")
        items.sort(key=lambda item: _reading_order_key(item, direction))
        frozen_assignments[block_index] = tuple(items)
        clipped_rectangles: list[tuple[int, int, int, int]] = []
        if block_box is not None:
            for item in items:
                region_box = item.region.bbox_xyxy
                clipped_rectangles.append(
                    (
                        max(block_box[0], region_box[0]),
                        max(block_box[1], region_box[1]),
                        min(block_box[2], region_box[2]),
                        min(block_box[3], region_box[3]),
                    )
                )
        detector_coverage = (
            _union_area(clipped_rectangles) / _area(block_box)
            if block_box is not None and _area(block_box)
            else 0.0
        )
        status = (
            "compound"
            if len(items) > 1
            else "matched"
            if items
            else "missing"
        )
        block_diagnostics[block_index] = {
            "block_id": ensure_text_block_id(blocks[block_index]),
            "status": status,
            "matched_region_count": len(items),
            "matched_region_indices": [
                item.region_index for item in items
            ],
            "detector_coverage": detector_coverage,
            "component_ids": component_ids_by_block[block_index],
        }

    relation_components: list[dict[str, object]] = []
    for component_index, (
        region_indices,
        block_indices,
    ) in enumerate(components):
        assigned_edges = sorted(
            (item.region_index, block_index)
            for block_index in block_indices
            for item in frozen_assignments.get(block_index, ())
            if item.region_index in region_indices
        )
        has_ambiguity = bool(
            ambiguous_region_indices.intersection(region_indices)
        )
        relation_components.append(
            {
                "component_id": f"component-{component_index:04d}",
                "relation_type": _relation_type(
                    len(region_indices), len(block_indices)
                ),
                "resolution": (
                    "review"
                    if has_ambiguity
                    else "unmatched"
                    if not block_indices
                    else "assigned"
                ),
                "region_indices": list(region_indices),
                "region_source_lines": [
                    mapped_regions[index].source_line
                    for index in region_indices
                ],
                "block_indices": list(block_indices),
                "block_ids": [
                    ensure_text_block_id(blocks[index])
                    for index in block_indices
                ],
                "assigned_edges": [list(edge) for edge in assigned_edges],
            }
        )
    return SpottingGeometryResult(
        assignments=frozen_assignments,
        unmatched_regions=tuple(unmatched),
        ambiguous_regions=tuple(ambiguous),
        ambiguous_block_indices=tuple(sorted(ambiguous_block_indices)),
        block_diagnostics=block_diagnostics,
        relation_components=tuple(relation_components),
    )
