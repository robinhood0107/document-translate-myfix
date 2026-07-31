"""Strategy-neutral coordinate transforms.

OCR strategy modules own matching policy.  This module only performs explicit,
reversible scaling and clipping so that image resizing cannot silently change
the coordinate space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class ImageCoordinateTransform:
    original_width: int
    original_height: int
    request_width: int
    request_height: int

    def __post_init__(self) -> None:
        dimensions = (
            self.original_width,
            self.original_height,
            self.request_width,
            self.request_height,
        )
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("OCR coordinate dimensions must be positive.")

    @property
    def scale_x(self) -> float:
        return self.request_width / float(self.original_width)

    @property
    def scale_y(self) -> float:
        return self.request_height / float(self.original_height)

    def request_to_original_point(
        self,
        point: Iterable[float],
    ) -> tuple[float, float]:
        x, y = tuple(point)
        return (
            min(max(float(x) / self.scale_x, 0.0), float(self.original_width)),
            min(max(float(y) / self.scale_y, 0.0), float(self.original_height)),
        )


__all__ = ["ImageCoordinateTransform"]
