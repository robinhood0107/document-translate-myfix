from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np

from .contracts import CandidateMaskResult, DetectorBox, binary_mask


@dataclass(frozen=True, slots=True)
class TiledInferenceSettings:
    """Source-space tiling for small text without page-specific coordinates."""

    tile_sizes: tuple[int, ...] = (768,)
    overlap: float = 0.2
    include_full_page: bool = True

    def __post_init__(self) -> None:
        if not self.tile_sizes or any(int(value) < 64 for value in self.tile_sizes):
            raise ValueError("tiled detector requires tile sizes of at least 64 pixels")
        if len(set(map(int, self.tile_sizes))) != len(self.tile_sizes):
            raise ValueError("tiled detector tile sizes must be unique")
        if not 0.0 <= float(self.overlap) < 0.5:
            raise ValueError("tiled detector overlap must be in [0, 0.5)")


def tile_origins(length: int, tile_size: int, overlap: float) -> tuple[int, ...]:
    if length < 1 or tile_size < 1:
        raise ValueError("tile dimensions must be positive")
    if tile_size >= length:
        return (0,)
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    origins = list(range(0, max(1, length - tile_size + 1), stride))
    final = length - tile_size
    if origins[-1] != final:
        origins.append(final)
    return tuple(origins)


class TiledCandidateReference:
    """Union full-page and overlapping crop detector outputs in source space."""

    def __init__(
        self,
        infer: Callable[[np.ndarray], CandidateMaskResult],
        settings: TiledInferenceSettings,
    ) -> None:
        self._infer = infer
        self.settings = settings

    def infer(self, image: np.ndarray) -> CandidateMaskResult:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("tiled detector expects a three-channel image")
        started = time.perf_counter()
        height, width = image.shape[:2]
        raw = np.zeros((height, width), dtype=np.uint8)
        refined = np.zeros_like(raw)
        dilated = np.zeros_like(raw)
        boxes: list[DetectorBox] = []
        child_seconds = 0.0
        calls = 0
        base_candidate_id = ""
        base_runtime: dict[str, object] = {}

        def merge(result: CandidateMaskResult, x1: int, y1: int) -> None:
            nonlocal child_seconds, calls, base_candidate_id, base_runtime
            crop_height, crop_width = result.shape
            x2, y2 = x1 + crop_width, y1 + crop_height
            raw[y1:y2, x1:x2] = np.maximum(
                raw[y1:y2, x1:x2], result.raw_mask
            )
            refined[y1:y2, x1:x2] = np.maximum(
                refined[y1:y2, x1:x2], result.refined_mask
            )
            dilated[y1:y2, x1:x2] = np.maximum(
                dilated[y1:y2, x1:x2], result.dilated_mask
            )
            for box in result.boxes:
                mapped = DetectorBox(
                    (
                        box.xyxy[0] + x1,
                        box.xyxy[1] + y1,
                        box.xyxy[2] + x1,
                        box.xyxy[3] + y1,
                    ),
                    box.label,
                    box.score,
                    box.provider,
                ).clipped((height, width))
                if mapped is not None:
                    boxes.append(mapped)
            calls += 1
            child_seconds += float(result.runtime.get("seconds") or 0.0)
            base_candidate_id = base_candidate_id or result.candidate_id
            if not base_runtime:
                base_runtime = dict(result.runtime)

        if self.settings.include_full_page:
            merge(self._infer(np.ascontiguousarray(image)), 0, 0)

        seen: set[tuple[int, int, int, int]] = set()
        for requested_size in self.settings.tile_sizes:
            tile_height = min(height, int(requested_size))
            tile_width = min(width, int(requested_size))
            if tile_height == height and tile_width == width:
                continue
            for y1 in tile_origins(height, tile_height, self.settings.overlap):
                for x1 in tile_origins(width, tile_width, self.settings.overlap):
                    roi = (x1, y1, x1 + tile_width, y1 + tile_height)
                    if roi in seen:
                        continue
                    seen.add(roi)
                    crop = np.ascontiguousarray(
                        image[y1 : y1 + tile_height, x1 : x1 + tile_width]
                    )
                    merge(self._infer(crop), x1, y1)

        if calls == 0:
            merge(self._infer(np.ascontiguousarray(image)), 0, 0)
        return CandidateMaskResult(
            candidate_id=f"{base_candidate_id or 'detector'}_tiled",
            raw_mask=binary_mask(raw),
            refined_mask=binary_mask(refined),
            dilated_mask=binary_mask(dilated),
            boxes=tuple(boxes),
            runtime={
                **base_runtime,
                "seconds": time.perf_counter() - started,
                "child_reported_seconds": child_seconds,
                "inference_call_count": calls,
                "full_page_included": self.settings.include_full_page,
                "tile_sizes": list(self.settings.tile_sizes),
                "tile_overlap": float(self.settings.overlap),
                "reference": "full-page plus overlapping source-space tiles",
            },
        )
