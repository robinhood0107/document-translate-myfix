from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any


MANGALMM_RESPONSE_SCHEMA_VERSION = 1
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


def _decode_json_array(payload_text: str) -> list[Any]:
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


def _normalize_region(item: Any, index: int) -> dict[str, object]:
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
    if coords[2] <= coords[0] or coords[3] <= coords[1]:
        raise MangaLMMResponseContractError(
            "invalid_bbox_order",
            f"MangaLMM region {index} bbox_2d is empty or reversed.",
        )

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
    return {
        "bbox_2d": coords,
        "text_content": text,
        "raw_text_content": text_value.strip(),
    }


def parse_mangalmm_response(text: str) -> MangaLMMParsedResponse:
    raw = str(text or "").strip()
    if not raw:
        raise MangaLMMResponseContractError(
            "empty_response",
            "MangaLMM response is empty.",
        )
    payload_text, response_kind = _unwrap_complete_json_fence(raw)
    payload = _decode_json_array(payload_text)
    regions = tuple(
        _normalize_region(item, index)
        for index, item in enumerate(payload)
    )
    return MangaLMMParsedResponse(
        regions=regions,
        response_kind=response_kind,
    )
