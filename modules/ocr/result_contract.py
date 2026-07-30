from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from modules.utils.textblock import ensure_text_block_id


OCR_STRATEGY_PADDLE_CROP = "paddle_crop"
OCR_STRATEGY_MANGALMM_FULL_PAGE = "mangalmm_full_page"

SEMANTIC_ROLE_DIALOGUE_BUBBLE = "dialogue_bubble"
SEMANTIC_ROLE_DIALOGUE_FREE = "dialogue_free"
SEMANTIC_ROLE_NARRATION = "narration"
SEMANTIC_ROLE_UI_OR_SIGN = "ui_or_sign"
SEMANTIC_ROLE_SFX = "sfx"
SEMANTIC_ROLE_DECORATIVE = "decorative"
SEMANTIC_ROLE_AMBIGUOUS = "ambiguous"

PROCESSING_ACTION_TRANSLATE_INPAINT = "translate_inpaint"
PROCESSING_ACTION_PRESERVE = "preserve"
PROCESSING_ACTION_REVIEW = "review"


def initialize_ocr_result_contract(
    block: Any,
    *,
    strategy: str = "",
    model_identity: str = "",
    runtime_identity: str = "",
) -> None:
    """Attach the common OCR provenance fields without changing product routing."""

    block_id = ensure_text_block_id(block)
    if strategy:
        block.ocr_strategy = str(strategy)
    elif not hasattr(block, "ocr_strategy"):
        block.ocr_strategy = ""
    if model_identity:
        block.ocr_model_identity = str(model_identity)
    elif not hasattr(block, "ocr_model_identity"):
        block.ocr_model_identity = ""
    if runtime_identity:
        block.ocr_runtime_identity = str(runtime_identity)
    elif not hasattr(block, "ocr_runtime_identity"):
        block.ocr_runtime_identity = ""
    if not isinstance(getattr(block, "ocr_geometry_provenance", None), dict):
        block.ocr_geometry_provenance = {}
    if not hasattr(block, "semantic_role"):
        block.semantic_role = ""
    if not hasattr(block, "processing_action"):
        block.processing_action = ""
    if not str(getattr(block, "canonical_block_id", "") or ""):
        block.canonical_block_id = block_id
    if not isinstance(getattr(block, "duplicate_alias_block_ids", None), list):
        block.duplicate_alias_block_ids = []
    block.duplicate_alias_count = len(block.duplicate_alias_block_ids)
    if not isinstance(getattr(block, "merge_split_diagnostics", None), dict):
        block.merge_split_diagnostics = {}


def _exact_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    try:
        values = np.asarray(value).reshape(-1)
    except Exception:
        return None
    if values.size != 4:
        return None
    try:
        numbers = tuple(float(item) for item in values)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(item) for item in numbers):
        return None
    x1, y1, x2, y2 = numbers
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _exact_duplicate_key(
    block: Any,
    *,
    source_identity: str,
) -> tuple[Any, ...] | None:
    text_bbox = _exact_bbox(getattr(block, "xyxy", None))
    if text_bbox is None:
        return None
    bubble_value = getattr(block, "bubble_xyxy", None)
    bubble_bbox = _exact_bbox(bubble_value)
    if bubble_value is not None and bubble_bbox is None:
        return None
    try:
        angle = float(getattr(block, "angle", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(angle):
        return None
    return (
        str(source_identity or ""),
        str(getattr(block, "text_class", "") or ""),
        text_bbox,
        bubble_bbox,
        angle,
        str(getattr(block, "direction", "") or ""),
    )


def canonicalize_exact_duplicate_blocks(
    blocks: Iterable[Any] | None,
    *,
    source_identity: str = "",
) -> tuple[list[Any], dict[str, Any]]:
    """Collapse only exact same-source detector duplicates before OCR.

    Blocks that merely share a bubble stay separate. A duplicate is removed only
    when its text geometry, bubble geometry, class, angle, direction, and page
    identity are all exactly equal.
    """

    block_list = list(blocks or [])
    canonical_blocks: list[Any] = []
    canonical_by_key: dict[tuple[Any, ...], Any] = {}
    groups: list[dict[str, Any]] = []
    group_by_canonical_id: dict[str, dict[str, Any]] = {}

    for block in block_list:
        try:
            initialize_ocr_result_contract(block)
        except (AttributeError, TypeError):
            canonical_blocks.append(block)
            continue
        block_id = ensure_text_block_id(block)
        key = _exact_duplicate_key(block, source_identity=source_identity)
        if key is None or key not in canonical_by_key:
            canonical_blocks.append(block)
            if key is not None:
                canonical_by_key[key] = block
            continue

        canonical = canonical_by_key[key]
        initialize_ocr_result_contract(canonical)
        canonical_id = ensure_text_block_id(canonical)
        aliases = canonical.duplicate_alias_block_ids
        if block_id not in aliases:
            aliases.append(block_id)
        canonical.duplicate_alias_count = len(aliases)
        canonical.merge_split_diagnostics = {
            **dict(canonical.merge_split_diagnostics),
            "exact_duplicate_canonicalization": "canonical",
            "duplicate_alias_count": canonical.duplicate_alias_count,
        }

        block.canonical_block_id = canonical_id
        block.merge_split_diagnostics = {
            **dict(block.merge_split_diagnostics),
            "exact_duplicate_canonicalization": "alias",
            "canonical_block_id": canonical_id,
        }

        group = group_by_canonical_id.get(canonical_id)
        if group is None:
            group = {
                "canonical_block_id": canonical_id,
                "alias_block_ids": aliases,
            }
            group_by_canonical_id[canonical_id] = group
            groups.append(group)

    return canonical_blocks, {
        "input_block_count": len(block_list),
        "canonical_block_count": len(canonical_blocks),
        "duplicate_alias_count": sum(
            int(getattr(block, "duplicate_alias_count", 0) or 0)
            for block in canonical_blocks
        ),
        "groups": groups,
    }
