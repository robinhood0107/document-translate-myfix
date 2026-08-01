"""Image sizing and attempt contracts for official MangaLMM full-page OCR."""

from __future__ import annotations

import math
from dataclasses import dataclass


MANGALMM_VISION_PATCH_SIZE = 14
MANGALMM_VISION_MERGE_SIZE = 2
MANGALMM_VISION_ALIGNMENT_FACTOR = (
    MANGALMM_VISION_PATCH_SIZE * MANGALMM_VISION_MERGE_SIZE
)
MANGALMM_OFFICIAL_MIN_PIXELS = 3_136
MANGALMM_OFFICIAL_MAX_PIXELS = 2_116_800
MANGALMM_MAX_ASPECT_RATIO = 200.0
MANGALMM_RESIZE_SCHEMA_VERSION = 2


@dataclass(slots=True)
class RequestUnit:
    bbox_xyxy: tuple[int, int, int, int]
    unit_kind: str


@dataclass(slots=True)
class ResizePlan:
    profile: str
    original_shape: tuple[int, int]
    request_shape: tuple[int, int]
    base_scale: float
    scale_x: float
    scale_y: float
    max_completion_tokens: int
    block_count: int
    small_block_ratio: float
    text_cover_ratio: float
    resize_schema_version: int = MANGALMM_RESIZE_SCHEMA_VERSION
    alignment_factor: int = MANGALMM_VISION_ALIGNMENT_FACTOR
    effective_max_pixels: int = MANGALMM_OFFICIAL_MAX_PIXELS


@dataclass(slots=True)
class AttemptSpec:
    index: int
    resize_plan: ResizePlan
    prompt_mode: str
    prompt_text: str
    attempt_kind: str
    repeat_penalty: float | None = None
    repeat_last_n: int | None = None


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def official_smart_resize(
    height: int,
    width: int,
    *,
    max_pixels: int,
    min_pixels: int = MANGALMM_OFFICIAL_MIN_PIXELS,
    factor: int = MANGALMM_VISION_ALIGNMENT_FACTOR,
) -> tuple[int, int]:
    """Mirror the Qwen2.5-VL smart_resize contract used by MangaLMM."""

    if height <= 0 or width <= 0:
        raise ValueError("MangaLMM image dimensions must be positive.")
    if max(height, width) / min(height, width) > MANGALMM_MAX_ASPECT_RATIO:
        raise ValueError(
            "MangaLMM image aspect ratio exceeds the official limit "
            f"of {MANGALMM_MAX_ASPECT_RATIO:g}."
        )
    max_pixels = max(int(min_pixels), int(max_pixels))
    resized_h = max(factor, round_by_factor(height, factor))
    resized_w = max(factor, round_by_factor(width, factor))
    if resized_h * resized_w > max_pixels:
        beta = math.sqrt((height * width) / float(max_pixels))
        resized_h = max(factor, floor_by_factor(height / beta, factor))
        resized_w = max(factor, floor_by_factor(width / beta, factor))
    elif resized_h * resized_w < min_pixels:
        beta = math.sqrt(float(min_pixels) / float(height * width))
        resized_h = ceil_by_factor(height * beta, factor)
        resized_w = ceil_by_factor(width * beta, factor)
    return int(resized_h), int(resized_w)


__all__ = [
    "AttemptSpec",
    "MANGALMM_MAX_ASPECT_RATIO",
    "MANGALMM_OFFICIAL_MAX_PIXELS",
    "MANGALMM_OFFICIAL_MIN_PIXELS",
    "MANGALMM_RESIZE_SCHEMA_VERSION",
    "MANGALMM_VISION_ALIGNMENT_FACTOR",
    "MANGALMM_VISION_MERGE_SIZE",
    "MANGALMM_VISION_PATCH_SIZE",
    "RequestUnit",
    "ResizePlan",
    "ceil_by_factor",
    "floor_by_factor",
    "official_smart_resize",
    "round_by_factor",
]
