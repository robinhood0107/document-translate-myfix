from __future__ import annotations

import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import psutil
from PIL import Image

MAX_IMAGE_PIXELS = 200_000_000
LARGE_PAGE_PIXELS = 50_000_000
ESTIMATED_PEAK_BYTES_PER_PIXEL = 15
MAX_RETAINED_BATCH_BYTES = 2 * 1024**3


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


@lru_cache(maxsize=512)
def _inspect_cached(path: str, size: int, mtime_ns: int) -> tuple[int, int]:
    del size, mtime_ns
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                width, height = (int(image.width), int(image.height))
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise ImageResourceLimitError(
            f"Image exceeds the supported {MAX_IMAGE_PIXELS:,}-pixel limit.",
            path=path,
        ) from exc
    except (OSError, ValueError) as exc:
        raise ImageResourceLimitError(
            "Image dimensions could not be read safely.",
            path=path,
        ) from exc
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


def inspect_image_dimensions(path: str | os.PathLike[str]) -> tuple[int, int]:
    resolved = Path(path).resolve()
    try:
        stat = resolved.stat()
    except OSError as exc:
        raise ImageResourceLimitError("Image file is missing.", path=str(resolved)) from exc
    return _inspect_cached(str(resolved), int(stat.st_size), int(stat.st_mtime_ns))


def build_image_memory_plan(paths: Iterable[str]) -> dict[str, int | bool]:
    total_estimated_peak = 0
    largest_pixels = 0
    page_count = 0
    available = int(psutil.virtual_memory().available)
    total_physical = int(psutil.virtual_memory().total)
    retention_budget = min(MAX_RETAINED_BATCH_BYTES, total_physical // 4)

    for path in paths:
        width, height = inspect_image_dimensions(path)
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
        total_estimated_peak += page_peak
        largest_pixels = max(largest_pixels, pixels)
        page_count += 1

    streaming = (
        largest_pixels > LARGE_PAGE_PIXELS
        or total_estimated_peak > retention_budget
    )
    return {
        "page_count": page_count,
        "largest_pixels": largest_pixels,
        "estimated_peak_bytes": total_estimated_peak,
        "retention_budget_bytes": retention_budget,
        "streaming": streaming,
    }
