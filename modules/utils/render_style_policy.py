from __future__ import annotations

from typing import Iterable

from PySide6.QtGui import QColor

from modules.utils.image_utils import get_smart_text_color

VERTICAL_ALIGNMENT_TOP = "top"
VERTICAL_ALIGNMENT_CENTER = "center"
VERTICAL_ALIGNMENT_BOTTOM = "bottom"

VERTICAL_ALIGNMENT_BY_ID = {
    0: VERTICAL_ALIGNMENT_TOP,
    1: VERTICAL_ALIGNMENT_CENTER,
    2: VERTICAL_ALIGNMENT_BOTTOM,
}


def coerce_vertical_alignment(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    if normalized in {
        VERTICAL_ALIGNMENT_TOP,
        VERTICAL_ALIGNMENT_CENTER,
        VERTICAL_ALIGNMENT_BOTTOM,
    }:
        return normalized
    return VERTICAL_ALIGNMENT_TOP


def should_force_smart_value(
    force_specific: bool = False, force_all: bool = False
) -> bool:
    return bool(force_specific or force_all)


def resolve_render_text_color(
    detected_rgb: tuple | Iterable | None,
    setting_color: QColor,
    force_specific: bool = False,
    force_all: bool = False,
) -> QColor:
    if should_force_smart_value(force_specific, force_all):
        return QColor(setting_color)
    return get_smart_text_color(detected_rgb, setting_color)


def compute_vertical_aligned_y(
    source_y: float,
    source_height: float,
    rendered_height: float,
    vertical_alignment: str | None,
) -> float:
    alignment = coerce_vertical_alignment(vertical_alignment)
    slack = max(0.0, float(source_height) - float(rendered_height))
    if alignment == VERTICAL_ALIGNMENT_BOTTOM:
        return float(source_y) + slack
    if alignment == VERTICAL_ALIGNMENT_CENTER:
        return float(source_y) + (slack / 2.0)
    return float(source_y)


def build_rect_tuple(
    x: float,
    y: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    return (float(x), float(y), float(width), float(height))
