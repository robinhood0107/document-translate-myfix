"""Strict parser for PaddleOCR-VL full-page Spotting output."""

from __future__ import annotations

import re
from dataclasses import dataclass


PADDLE_SPOTTING_RESPONSE_SCHEMA_VERSION = 2
PADDLE_SPOTTING_COORDINATE_SCALE = 1000
_LOCATION_TOKEN = re.compile(r"<\|LOC_(\d{1,4})\|>")
_NATIVE_LINE = re.compile(
    r"^(?P<text>.*?)(?P<coordinates>(?:<\|LOC_\d{1,4}\|>){8})$"
)
_SPECIAL_TOKEN_FRAGMENT = re.compile(r"<\|[^>]*\|>")
_END_TOKEN = re.compile(r"</s>")


class PaddleSpottingResponseContractError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code)


@dataclass(frozen=True)
class PaddleSpottingRegion:
    text: str
    normalized_points: tuple[
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
    ]
    source_line: int


@dataclass(frozen=True)
class PaddleSpottingParsedResponse:
    regions: tuple[PaddleSpottingRegion, ...]
    duplicate_region_count: int
    response_kind: str = "native_spotting_lines"


def _parse_line(line: str, line_number: int) -> PaddleSpottingRegion:
    native_line = _NATIVE_LINE.fullmatch(line)
    if native_line is None:
        raise PaddleSpottingResponseContractError(
            "invalid_native_line",
            (
                "PaddleOCR-VL Spotting line "
                f"{line_number} must end with exactly eight LOC tokens."
            ),
        )
    coordinate_text = native_line.group("coordinates")
    values = [
        int(value) for value in _LOCATION_TOKEN.findall(coordinate_text)
    ]
    if len(values) != 8:
        raise PaddleSpottingResponseContractError(
            "invalid_coordinate_count",
            (
                "PaddleOCR-VL Spotting line "
                f"{line_number} must contain exactly eight LOC coordinates."
            ),
        )
    if any(
        value < 0 or value > PADDLE_SPOTTING_COORDINATE_SCALE
        for value in values
    ):
        raise PaddleSpottingResponseContractError(
            "coordinate_out_of_range",
            (
                "PaddleOCR-VL Spotting line "
                f"{line_number} contains a coordinate outside 0..1000."
            ),
        )

    text = native_line.group("text").strip()
    if not text:
        raise PaddleSpottingResponseContractError(
            "empty_text",
            f"PaddleOCR-VL Spotting line {line_number} has no text.",
        )
    if _SPECIAL_TOKEN_FRAGMENT.search(text):
        raise PaddleSpottingResponseContractError(
            "unexpected_special_token",
            (
                "PaddleOCR-VL Spotting line "
                f"{line_number} contains a special token in OCR text."
            ),
        )

    points = tuple(
        (values[index], values[index + 1])
        for index in range(0, 8, 2)
    )
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    if min_x == max_x or min_y == max_y:
        raise PaddleSpottingResponseContractError(
            "degenerate_geometry",
            (
                "PaddleOCR-VL Spotting line "
                f"{line_number} describes an empty quadrilateral."
            ),
        )

    return PaddleSpottingRegion(
        text=text,
        normalized_points=points,  # type: ignore[arg-type]
        source_line=line_number,
    )


def parse_paddle_spotting_response(
    text: str,
) -> PaddleSpottingParsedResponse:
    raw = _END_TOKEN.sub("", str(text or "")).strip()
    if not raw:
        raise PaddleSpottingResponseContractError(
            "empty_response",
            "PaddleOCR-VL Spotting response is empty.",
        )

    regions: list[PaddleSpottingRegion] = []
    seen: set[tuple[object, ...]] = set()
    duplicate_count = 0
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        region = _parse_line(line, line_number)
        key = (region.text, region.normalized_points)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        regions.append(region)

    if not regions:
        raise PaddleSpottingResponseContractError(
            "no_regions",
            "PaddleOCR-VL Spotting response contains no valid regions.",
        )
    return PaddleSpottingParsedResponse(
        regions=tuple(regions),
        duplicate_region_count=duplicate_count,
    )
