from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from modules.utils.textblock import ensure_text_block_id

from .response_contract import (
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
    best: tuple[int, float, float, float, bool, int],
    second: tuple[int, float, float, float, bool, int],
) -> bool:
    (
        _best_index,
        best_cover,
        best_block_cover,
        best_iou,
        best_center,
        _best_area,
    ) = best
    (
        _second_index,
        second_cover,
        second_block_cover,
        second_iou,
        second_center,
        _second_area,
    ) = second
    if best_block_cover >= 0.50 and second_block_cover >= 0.50:
        return True
    if best_center and second_center:
        return abs(best_cover - second_cover) < 0.10
    return (
        best_cover >= 0.50
        and second_cover >= 0.50
        and abs(best_cover - second_cover) < 0.05
        and abs(best_iou - second_iou) < 0.05
    )


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
    assignments: dict[int, list[SpottingAssignment]] = {
        index: [] for index in range(len(blocks))
    }
    unmatched: list[MappedSpottingRegion] = []
    ambiguous: list[dict[str, object]] = []
    ambiguous_block_indices: set[int] = set()

    block_boxes = [_block_bbox(block) for block in blocks]
    for region in mapped_regions:
        candidates: list[tuple[int, float, float, float, bool, int]] = []
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
                (
                    block_index,
                    region_cover,
                    block_cover,
                    iou,
                    center_inside,
                    _area(block_box),
                )
            )
        if not candidates:
            unmatched.append(region)
            continue
        candidates.sort(
            key=lambda candidate: (
                candidate[4],
                candidate[1],
                candidate[3],
                candidate[2],
                -candidate[5],
            ),
            reverse=True,
        )
        if len(candidates) > 1 and _is_ambiguous(
            candidates[0],
            candidates[1],
        ):
            candidate_indices = [
                candidate[0] for candidate in candidates
            ]
            ambiguous_block_indices.update(candidate_indices)
            ambiguous.append(
                {
                    "text": region.text,
                    "bbox_xyxy": list(region.bbox_xyxy),
                    "candidate_block_ids": [
                        ensure_text_block_id(blocks[block_index])
                        for block_index in candidate_indices
                    ],
                    "reason": "one_spot_multiple_detector_blocks",
                }
            )
            continue
        (
            block_index,
            region_cover,
            block_cover,
            iou,
            center_inside,
            _block_area,
        ) = candidates[0]
        assignments[block_index].append(
            SpottingAssignment(
                block_index=block_index,
                region=region,
                region_coverage=region_cover,
                block_coverage=block_cover,
                iou=iou,
                center_inside=center_inside,
            )
        )

    frozen_assignments: dict[int, tuple[SpottingAssignment, ...]] = {}
    for block_index, items in assignments.items():
        if block_index in ambiguous_block_indices:
            frozen_assignments[block_index] = ()
            continue
        direction = str(getattr(blocks[block_index], "direction", "") or "")
        items.sort(key=lambda item: _reading_order_key(item, direction))
        frozen_assignments[block_index] = tuple(items)
    return SpottingGeometryResult(
        assignments=frozen_assignments,
        unmatched_regions=tuple(unmatched),
        ambiguous_regions=tuple(ambiguous),
        ambiguous_block_indices=tuple(sorted(ambiguous_block_indices)),
    )
