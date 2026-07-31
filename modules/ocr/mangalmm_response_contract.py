from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any


MANGALMM_RESPONSE_SCHEMA_VERSION = 2
_COMPLETE_JSON_FENCE = re.compile(
    r"\A```(?:json)?[ \t]*\r?\n(?P<payload>[\s\S]*?)"
    r"\r?\n[ \t]*```[ \t]*\Z",
    flags=re.IGNORECASE,
)


class MangaLMMResponseContractError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code)


@dataclass(frozen=True)
class MangaLMMParsedResponse:
    regions: tuple[dict[str, object], ...]
    response_kind: str
    payload_type: str = "json_array"
    normalized_literal_control_count: int = 0
    normalized_bbox_order_count: int = 0


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MangaLMMResponseContractError(
                "duplicate_key",
                f"MangaLMM response contains a duplicate JSON key: {key}",
            )
        result[key] = value
    return result


def _unwrap_complete_json_fence(raw: str) -> tuple[str, str]:
    if not raw.startswith("```"):
        return raw, "json_array"
    match = _COMPLETE_JSON_FENCE.fullmatch(raw)
    if match is None:
        raise MangaLMMResponseContractError(
            "invalid_code_fence",
            "MangaLMM response contains an incomplete or wrapped JSON fence.",
        )
    return match.group("payload").strip(), "fenced_json_array"


def _escape_literal_controls_in_strings(payload_text: str) -> tuple[str, int]:
    """Normalize the pseudo-JSON emitted by the official MangaLMM corpus.

    MangaOCR ground truth and the upstream evaluator allow literal line breaks
    inside ``text_content`` strings.  They are invalid in RFC-compliant JSON,
    but their meaning is unambiguous while scanning a quoted string.  Only
    control characters inside strings are escaped; array structure, trailing
    content, duplicate keys, and all other validation remain strict.
    """

    escaped: list[str] = []
    in_string = False
    previous_was_escape = False
    normalized_count = 0
    replacements = {
        "\b": r"\b",
        "\f": r"\f",
        "\n": r"\n",
        "\r": r"\r",
        "\t": r"\t",
    }
    for character in payload_text:
        if not in_string:
            escaped.append(character)
            if character == '"':
                in_string = True
                previous_was_escape = False
            continue

        if previous_was_escape:
            escaped.append(character)
            previous_was_escape = False
            continue
        if character == "\\":
            escaped.append(character)
            previous_was_escape = True
            continue
        if character == '"':
            escaped.append(character)
            in_string = False
            continue
        if ord(character) < 0x20:
            escaped.append(
                replacements.get(character, f"\\u{ord(character):04x}")
            )
            normalized_count += 1
            continue
        escaped.append(character)
    return "".join(escaped), normalized_count


def _load_json_array(payload_text: str) -> list[Any]:
    try:
        payload = json.loads(
            payload_text,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except MangaLMMResponseContractError:
        raise
    except json.JSONDecodeError as exc:
        raise MangaLMMResponseContractError(
            "invalid_json",
            f"MangaLMM response is not one complete JSON value: {exc.msg}",
        ) from exc
    if not isinstance(payload, list):
        raise MangaLMMResponseContractError(
            "top_level_not_array",
            "MangaLMM response must be a top-level JSON array.",
        )
    return payload


def _decode_json_array(payload_text: str) -> tuple[list[Any], int]:
    try:
        return _load_json_array(payload_text), 0
    except MangaLMMResponseContractError as exc:
        if exc.code != "invalid_json":
            raise

    normalized, normalized_count = _escape_literal_controls_in_strings(
        payload_text
    )
    if normalized_count <= 0:
        return _load_json_array(payload_text), 0
    return _load_json_array(normalized), normalized_count


def _normalize_region(
    item: Any,
    index: int,
) -> tuple[dict[str, object], bool]:
    if not isinstance(item, dict):
        raise MangaLMMResponseContractError(
            "invalid_region_type",
            f"MangaLMM region {index} must be a JSON object.",
        )
    if "bbox_2d" not in item or "text_content" not in item:
        raise MangaLMMResponseContractError(
            "missing_region_field",
            f"MangaLMM region {index} must contain bbox_2d and text_content.",
        )

    bbox = item["bbox_2d"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise MangaLMMResponseContractError(
            "invalid_bbox",
            f"MangaLMM region {index} bbox_2d must contain four numbers.",
        )
    coords: list[float] = []
    for value in bbox:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MangaLMMResponseContractError(
                "invalid_bbox",
                f"MangaLMM region {index} bbox_2d must contain four numbers.",
            )
        parsed = float(value)
        if not math.isfinite(parsed):
            raise MangaLMMResponseContractError(
                "invalid_bbox",
                f"MangaLMM region {index} bbox_2d contains a non-finite number.",
            )
        coords.append(parsed)
    normalized_coords = [
        min(coords[0], coords[2]),
        min(coords[1], coords[3]),
        max(coords[0], coords[2]),
        max(coords[1], coords[3]),
    ]
    if (
        normalized_coords[2] <= normalized_coords[0]
        or normalized_coords[3] <= normalized_coords[1]
    ):
        raise MangaLMMResponseContractError(
            "invalid_bbox_order",
            f"MangaLMM region {index} bbox_2d is empty or reversed.",
        )
    bbox_order_normalized = normalized_coords != coords

    text_value = item["text_content"]
    if not isinstance(text_value, str):
        raise MangaLMMResponseContractError(
            "invalid_text_type",
            f"MangaLMM region {index} text_content must be a string.",
        )
    text = text_value.strip()
    if not text:
        raise MangaLMMResponseContractError(
            "empty_text",
            f"MangaLMM region {index} text_content is empty.",
        )
    region: dict[str, object] = {
        "bbox_2d": normalized_coords,
        "text_content": text,
        "raw_text_content": text_value.strip(),
    }
    if bbox_order_normalized:
        region["raw_bbox_2d"] = coords
        region["bbox_order_normalized"] = True
    return region, bbox_order_normalized


def parse_mangalmm_response(text: str) -> MangaLMMParsedResponse:
    raw = str(text or "").strip()
    if not raw:
        raise MangaLMMResponseContractError(
            "empty_response",
            "MangaLMM response is empty.",
        )
    payload_text, response_kind = _unwrap_complete_json_fence(raw)
    payload, normalized_literal_control_count = _decode_json_array(
        payload_text
    )
    normalized_regions = [
        _normalize_region(item, index)
        for index, item in enumerate(payload)
    ]
    regions = tuple(region for region, _normalized in normalized_regions)
    normalized_bbox_order_count = sum(
        1 for _region, normalized in normalized_regions if normalized
    )
    if normalized_literal_control_count:
        response_kind = f"{response_kind}_literal_controls_normalized"
    if normalized_bbox_order_count:
        response_kind = f"{response_kind}_bbox_order_normalized"
    return MangaLMMParsedResponse(
        regions=regions,
        response_kind=response_kind,
        normalized_literal_control_count=normalized_literal_control_count,
        normalized_bbox_order_count=normalized_bbox_order_count,
    )
