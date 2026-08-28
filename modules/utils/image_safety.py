from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO, Iterable

import psutil
from PIL import Image

MAX_IMAGE_PIXELS = 200_000_000
ESTIMATED_PEAK_BYTES_PER_PIXEL = 15


@dataclass(frozen=True)
class ImageResourcePlan:
    """Read-only image safety facts with no workflow-routing decision."""

    page_count: int = 0
    largest_pixels: int = 0
    largest_page_peak_bytes: int = 0
    hard_cap_passed: bool = True


class ImageResourceLimitError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        path: str = "",
        width: int = 0,
        height: int = 0,
        pixels: int = 0,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.pixels = int(pixels)


def configure_pillow_image_policy() -> None:
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    warnings.filterwarnings("error", category=Image.DecompressionBombWarning)


configure_pillow_image_policy()


def _validated_dimensions(
    width: int,
    height: int,
    *,
    path: str = "",
) -> tuple[int, int]:
    width = int(width)
    height = int(height)
    pixels = width * height
    if width <= 0 or height <= 0 or pixels > MAX_IMAGE_PIXELS:
        raise ImageResourceLimitError(
            f"Image exceeds the supported {MAX_IMAGE_PIXELS:,}-pixel limit.",
            path=path,
            width=width,
            height=height,
            pixels=pixels,
        )
    return width, height


def inspect_image_stream_dimensions(
    stream: BinaryIO,
    *,
    source_label: str = "",
) -> tuple[int, int]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(stream) as image:
                width, height = (int(image.width), int(image.height))
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise ImageResourceLimitError(
            f"Image exceeds the supported {MAX_IMAGE_PIXELS:,}-pixel limit.",
            path=source_label,
        ) from exc
    except (OSError, ValueError) as exc:
        raise ImageResourceLimitError(
            "Image dimensions could not be read safely.",
            path=source_label,
        ) from exc
    return _validated_dimensions(width, height, path=source_label)


@lru_cache(maxsize=512)
def _inspect_cached(path: str, size: int, mtime_ns: int) -> tuple[int, int]:
    del size, mtime_ns
    try:
        with open(path, "rb") as stream:
            return inspect_image_stream_dimensions(stream, source_label=path)
    except ImageResourceLimitError:
        raise
    except OSError as exc:
        raise ImageResourceLimitError(
            "Image dimensions could not be read safely.",
            path=path,
        ) from exc


def inspect_image_dimensions(path: str | os.PathLike[str]) -> tuple[int, int]:
    resolved = Path(path).resolve()
    try:
        stat = resolved.stat()
    except OSError as exc:
        raise ImageResourceLimitError("Image file is missing.", path=str(resolved)) from exc
    return _inspect_cached(str(resolved), int(stat.st_size), int(stat.st_mtime_ns))


def build_image_resource_plan(
    resources: Iterable[tuple[str, int, int]],
    *,
    available_memory_bytes: int | None = None,
) -> ImageResourcePlan:
    largest_pixels = 0
    largest_page_peak = 0
    page_count = 0
    available = int(
        psutil.virtual_memory().available
        if available_memory_bytes is None
        else available_memory_bytes
    )

    for path, width, height in resources:
        width, height = _validated_dimensions(width, height, path=str(path))
        pixels = int(width) * int(height)
        page_peak = pixels * ESTIMATED_PEAK_BYTES_PER_PIXEL
        if page_peak > int(available * 0.70):
            raise ImageResourceLimitError(
                "The image fits the pixel limit but requires too much currently available memory.",
                path=str(path),
                width=width,
                height=height,
                pixels=pixels,
            )
        largest_pixels = max(largest_pixels, pixels)
        largest_page_peak = max(largest_page_peak, page_peak)
        page_count += 1

    return ImageResourcePlan(
        page_count=page_count,
        largest_pixels=largest_pixels,
        largest_page_peak_bytes=largest_page_peak,
        hard_cap_passed=True,
    )
