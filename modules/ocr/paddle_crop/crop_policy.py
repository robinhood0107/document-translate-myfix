"""Image-region policy for Paddle detector-and-crop OCR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modules.utils.ocr_debug import (
    resolve_block_crop_bbox,
    set_block_ocr_crop_diagnostics,
)

from ..common.result_contract import (
    OCR_STRATEGY_PADDLE_CROP,
    initialize_ocr_result_contract,
)


PADDLE_CROP_OFFICIAL_IMAGE_MAX_PIXELS = 1_003_520


@dataclass(frozen=True, slots=True)
class PaddleCropPolicy:
    expansion_ratio: float = 0.03
    bubble_as_clamp: bool = True
    fallback_to_bubble: bool = True


def _coords_or_none(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        coords = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return coords if len(coords) == 4 else None


def resolve_paddle_crop_bbox(
    block: Any,
    image_shape: tuple[int, ...],
    *,
    model_identity: str,
    policy: PaddleCropPolicy,
) -> tuple[int, int, int, int] | None:
    """Resolve the current text-first, bubble-clamped crop without side effects elsewhere."""

    initialize_ocr_result_contract(
        block,
        strategy=OCR_STRATEGY_PADDLE_CROP,
        model_identity=str(
            getattr(block, "ocr_model_identity", "") or model_identity
        ),
    )
    bbox, crop_source = resolve_block_crop_bbox(
        block,
        image_shape,
        x_ratio=float(policy.expansion_ratio),
        y_ratio=float(policy.expansion_ratio),
        bubble_as_clamp=bool(policy.bubble_as_clamp),
        fallback_to_bubble=bool(policy.fallback_to_bubble),
    )
    set_block_ocr_crop_diagnostics(
        block,
        effective_crop_xyxy=bbox,
        crop_source=crop_source,
    )
    block.ocr_geometry_provenance = {
        "strategy": "text_first_bubble_clamp",
        "crop_source": crop_source,
        "text_bbox": _coords_or_none(getattr(block, "xyxy", None)),
        "bubble_bbox": _coords_or_none(
            getattr(block, "bubble_xyxy", None)
        ),
        "effective_crop_bbox": (
            [int(value) for value in bbox] if bbox is not None else None
        ),
        "text_expansion_ratio": float(policy.expansion_ratio),
    }
    return bbox


__all__ = [
    "PADDLE_CROP_OFFICIAL_IMAGE_MAX_PIXELS",
    "PaddleCropPolicy",
    "resolve_paddle_crop_bbox",
]
