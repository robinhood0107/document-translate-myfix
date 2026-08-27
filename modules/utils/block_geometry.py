from __future__ import annotations

from collections.abc import Callable, Iterable
import copy
import math
from typing import Any

import numpy as np


BASE_BLOCK_BBOX_FIELDS = (
    "xyxy",
    "detector_text_bbox",
    "bubble_xyxy",
    "_render_original_xyxy",
    "_render_area_xyxy",
    "_render_bubble_xyxy",
    "bubble_panel_group_xyxy",
    "bubble_panel_render_xyxy",
)

RENDER_GEOMETRY_FIELDS = (
    "_render_original_xyxy",
    "_render_area_xyxy",
    "_render_bubble_xyxy",
    "bubble_panel_group_xyxy",
    "bubble_panel_render_xyxy",
)


def normalize_block_xyxy(
    value: object,
    image_shape: tuple[int, ...] | None = None,
) -> tuple[int, int, int, int] | None:
    """Normalize a bbox without retaining a reference to mutable block state."""
    if value is None:
        return None
    try:
        raw = list(value)  # type: ignore[arg-type]
        if len(raw) < 4:
            return None
        x1, y1, x2, y2 = [int(round(float(v))) for v in raw[:4]]
    except (TypeError, ValueError, OverflowError):
        return None

    if image_shape is not None:
        try:
            img_h, img_w = int(image_shape[0]), int(image_shape[1])
        except (IndexError, TypeError, ValueError):
            return None
        x1 = max(0, min(x1, img_w))
        x2 = max(0, min(x2, img_w))
        y1 = max(0, min(y1, img_h))
        y2 = max(0, min(y2, img_h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def copy_block_xyxy(
    value: object,
) -> tuple[int | float, int | float, int | float, int | float] | None:
    """Copy a valid bbox while preserving integral versus fractional values."""
    if value is None:
        return None
    try:
        original = list(value)[:4]  # type: ignore[arg-type]
        if len(original) != 4:
            return None
        converted = [float(item) for item in original]
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(item) for item in converted):
        return None
    if converted[2] <= converted[0] or converted[3] <= converted[1]:
        return None
    original_is_integral = all(
        isinstance(item, (int, np.integer))
        and not isinstance(item, (bool, np.bool_))
        for item in original
    )
    if original_is_integral:
        return tuple(int(item) for item in converted)  # type: ignore[return-value]
    return tuple(converted)  # type: ignore[return-value]


def legacy_adjust_xyxy(
    value: object,
    image_shape: tuple[int, ...],
    width_percentage: int,
    height_percentage: int,
) -> tuple[int, int, int, int] | None:
    """Mirror adjust_text_line_coordinates, including its int truncation."""
    adjusted = _legacy_adjust_xyxy_unbounded(
        value,
        width_percentage,
        height_percentage,
    )
    if adjusted is None:
        return None
    return normalize_block_xyxy(adjusted, image_shape)


def _legacy_adjust_xyxy_unbounded(
    value: object,
    width_percentage: int,
    height_percentage: int,
) -> tuple[float, float, float, float] | None:
    """Apply the legacy percentage adjustment without clipping to an image."""
    if value is None:
        return None
    try:
        raw = list(value)  # type: ignore[arg-type]
        if len(raw) < 4:
            return None
        x1, y1, x2, y2 = [float(item) for item in raw[:4]]
        width_offset = int((((x2 - x1) * width_percentage) / 100) / 2)
        height_offset = int((((y2 - y1) * height_percentage) / 100) / 2)
    except (TypeError, ValueError, OverflowError):
        return None
    adjusted = (
        x1 - width_offset,
        y1 - height_offset,
        x2 + width_offset,
        y2 + height_offset,
    )
    if normalize_block_xyxy(adjusted) is None:
        return None
    return adjusted


def _unbounded_render_geometry_relation(block: object) -> str | None:
    current = normalize_block_xyxy(getattr(block, "xyxy", None))
    render_area = normalize_block_xyxy(getattr(block, "_render_area_xyxy", None))
    if current is None or render_area is None:
        return None
    if current == render_area:
        return "render_area"
    contracted = normalize_block_xyxy(
        _legacy_adjust_xyxy_unbounded(render_area, -5, -5)
    )
    if contracted is not None and current == contracted:
        return "legacy_minus_five_percent"
    return None


def _transformable_render_geometry_relation(block: object) -> str | None:
    """Return a relation that a coordinate-only transform may preserve."""
    render_source = str(getattr(block, "_render_area_source", "") or "")
    if render_source not in {"text_bbox", "detected_bubble"}:
        return None
    if normalize_block_xyxy(getattr(block, "_render_original_xyxy", None)) is None:
        return None
    relation = _unbounded_render_geometry_relation(block)
    if relation is None or getattr(block, "text_class", "") != "text_bubble":
        return relation
    bubble = normalize_block_xyxy(getattr(block, "bubble_xyxy", None))
    render_bubble = normalize_block_xyxy(
        getattr(block, "_render_bubble_xyxy", None)
    )
    if bubble is None or render_bubble is None:
        if render_source == "text_bbox" and bubble is None and render_bubble is None:
            return relation
        return None
    if bubble != render_bubble:
        return None
    return relation


def render_geometry_relation(
    block: object,
    image_shape: tuple[int, ...],
) -> str | None:
    """Return how current xyxy relates to the stored render area."""
    current = normalize_block_xyxy(getattr(block, "xyxy", None), image_shape)
    render_area = normalize_block_xyxy(
        getattr(block, "_render_area_xyxy", None), image_shape
    )
    if current is None or render_area is None:
        return None
    if current == render_area:
        return "render_area"
    contracted = legacy_adjust_xyxy(render_area, image_shape, -5, -5)
    if contracted is not None and current == contracted:
        return "legacy_minus_five_percent"
    return None


def resolve_render_recompute_anchor_xyxy(
    block: object,
    image_shape: tuple[int, ...],
) -> tuple[int | float, int | float, int | float, int | float] | None:
    """Preserve the OCR anchor only while stored render geometry is coherent."""
    current = copy_block_xyxy(getattr(block, "xyxy", None))
    original = copy_block_xyxy(getattr(block, "_render_original_xyxy", None))
    if original is not None and render_geometry_is_coherent(block, image_shape):
        return original
    return current


def render_geometry_is_coherent(
    block: object,
    image_shape: tuple[int, ...],
) -> bool:
    render_source = str(getattr(block, "_render_area_source", "") or "")
    if render_source not in {"text_bbox", "detected_bubble"}:
        return False
    if render_geometry_relation(block, image_shape) is None:
        return False
    if normalize_block_xyxy(
        getattr(block, "_render_original_xyxy", None), image_shape
    ) is None:
        return False
    if getattr(block, "text_class", "") != "text_bubble":
        return True
    bubble = normalize_block_xyxy(getattr(block, "bubble_xyxy", None), image_shape)
    render_bubble = normalize_block_xyxy(
        getattr(block, "_render_bubble_xyxy", None), image_shape
    )
    if bubble is None or render_bubble is None:
        return (
            render_source == "text_bbox"
            and bubble is None
            and render_bubble is None
        )
    return bubble == render_bubble


def invalidate_render_geometry(block: object) -> None:
    """Drop render-only coordinates after a semantic or manual bbox change."""
    fields = set(RENDER_GEOMETRY_FIELDS)
    fields.update(
        field
        for field in iter_block_bbox_fields(block)
        if field.startswith("bubble_panel_") and field.endswith("_xyxy")
    )
    for field in fields:
        if hasattr(block, field):
            setattr(block, field, None)
    setattr(block, "_render_area_source", "text_bbox")


def iter_block_bbox_fields(block: object) -> tuple[str, ...]:
    fields = list(BASE_BLOCK_BBOX_FIELDS)
    try:
        dynamic_fields: Iterable[str] = vars(block).keys()
    except TypeError:
        dynamic_fields = ()
    for field in dynamic_fields:
        if (
            field.startswith("bubble_panel_")
            and field.endswith("_xyxy")
            and field not in fields
        ):
            fields.append(field)
    return tuple(fields)


def _replace_bbox_value(block: object, field: str, mapped: list[float]) -> None:
    original = getattr(block, field, None)
    try:
        original_values = list(original)[:4]
    except TypeError:
        original_values = []
    original_is_integral = len(original_values) == 4 and all(
        isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
        for value in original_values
    )
    mapped_is_integral = all(float(value).is_integer() for value in mapped)
    values: list[int] | list[float]
    if original_is_integral and mapped_is_integral:
        values = [int(value) for value in mapped]
    else:
        values = [float(value) for value in mapped]

    if isinstance(original, np.ndarray):
        dtype = original.dtype
        if not mapped_is_integral and np.issubdtype(dtype, np.integer):
            dtype = np.dtype(np.float64)
        setattr(block, field, np.asarray(values, dtype=dtype))
    elif isinstance(original, tuple):
        setattr(block, field, tuple(values))
    else:
        setattr(block, field, list(values))


def map_block_bbox_fields(
    block: object,
    mapper: Callable[[list[float]], Iterable[float] | None],
    *,
    fields: Iterable[str] | None = None,
) -> None:
    """Apply one coordinate transform to all block/render bbox fields."""
    mapped_fields = tuple(fields or iter_block_bbox_fields(block))
    preserve_relation = None
    if "xyxy" in mapped_fields and "_render_area_xyxy" in mapped_fields:
        preserve_relation = _transformable_render_geometry_relation(block)

    for field in mapped_fields:
        if not hasattr(block, field):
            continue
        value = getattr(block, field, None)
        if value is None:
            continue
        try:
            raw = [float(v) for v in list(value)[:4]]
        except (TypeError, ValueError, OverflowError):
            continue
        if len(raw) != 4:
            continue
        mapped_value = mapper(raw)
        if mapped_value is None:
            setattr(block, field, None)
            continue
        try:
            mapped = [float(v) for v in list(mapped_value)[:4]]
        except (TypeError, ValueError, OverflowError):
            continue
        if len(mapped) != 4:
            continue
        _replace_bbox_value(block, field, mapped)

    if preserve_relation == "render_area":
        try:
            repaired = [
                float(value)
                for value in list(getattr(block, "_render_area_xyxy", None))[:4]
            ]
        except (TypeError, ValueError, OverflowError):
            repaired = None
        if repaired is not None and normalize_block_xyxy(repaired) is None:
            repaired = None
    elif preserve_relation == "legacy_minus_five_percent":
        repaired = _legacy_adjust_xyxy_unbounded(
            getattr(block, "_render_area_xyxy", None),
            -5,
            -5,
        )
    else:
        repaired = None
    if repaired is not None:
        _replace_bbox_value(block, "xyxy", list(repaired))


def snapshot_block_bbox_fields(block: object) -> dict[str, tuple[bool, Any]]:
    snapshot: dict[str, tuple[bool, Any]] = {}
    for field in iter_block_bbox_fields(block):
        if not hasattr(block, field):
            snapshot[field] = (False, None)
            continue
        value = getattr(block, field, None)
        if value is None:
            snapshot[field] = (True, None)
            continue
        snapshot[field] = (True, copy.deepcopy(value))
    return snapshot


def restore_block_bbox_fields(
    block: object,
    snapshot: dict[str, tuple[bool, Any]],
) -> None:
    for field, (existed, value) in snapshot.items():
        if not existed:
            if hasattr(block, field):
                delattr(block, field)
            continue
        if value is None:
            setattr(block, field, None)
            continue
        setattr(block, field, copy.deepcopy(value))
