"""MangaLMM region records and detector reconciliation boundary."""

from __future__ import annotations

from dataclasses import dataclass


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


__all__ = ["OCRRegion"]
