"""Shared OCR result and destructive-processing contract."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from modules.utils.textblock import ensure_text_block_id


OCR_STRATEGY_PADDLE_CROP = "paddle_crop"
OCR_STRATEGY_PADDLE_SPOTTING = "paddle_spotting_full_page"
OCR_STRATEGY_MANGALMM_FULL_PAGE = "mangalmm_full_page"
OCR_PROCESSING_CONTRACT_SCHEMA_VERSION = 1

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

MASK_STRATEGY_BUBBLE_SAFE = "bubble_safe"
MASK_STRATEGY_GLYPH_ONLY = "glyph_only"
MASK_STRATEGY_GLYPH_ONLY_STRUCTURE_PROTECT = (
    "glyph_only_structure_protect"
)
MASK_STRATEGY_PRESERVE_ORIGINAL = "preserve_original"

VALID_SEMANTIC_ROLES = frozenset(
    {
        SEMANTIC_ROLE_DIALOGUE_BUBBLE,
        SEMANTIC_ROLE_DIALOGUE_FREE,
        SEMANTIC_ROLE_NARRATION,
        SEMANTIC_ROLE_UI_OR_SIGN,
        SEMANTIC_ROLE_SFX,
        SEMANTIC_ROLE_DECORATIVE,
        SEMANTIC_ROLE_AMBIGUOUS,
    }
)
VALID_PROCESSING_ACTIONS = frozenset(
    {
        PROCESSING_ACTION_TRANSLATE_INPAINT,
        PROCESSING_ACTION_PRESERVE,
        PROCESSING_ACTION_REVIEW,
    }
)
VALID_MASK_STRATEGIES = frozenset(
    {
        MASK_STRATEGY_BUBBLE_SAFE,
        MASK_STRATEGY_GLYPH_ONLY,
        MASK_STRATEGY_GLYPH_ONLY_STRUCTURE_PROTECT,
        MASK_STRATEGY_PRESERVE_ORIGINAL,
    }
)


def _normalized_choice(value: Any, choices: frozenset[str]) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in choices else ""


def _default_action_for_role(role: str) -> str:
    if role in {
        SEMANTIC_ROLE_DIALOGUE_BUBBLE,
        SEMANTIC_ROLE_DIALOGUE_FREE,
        SEMANTIC_ROLE_NARRATION,
    }:
        return PROCESSING_ACTION_TRANSLATE_INPAINT
    if role in {
        SEMANTIC_ROLE_SFX,
        SEMANTIC_ROLE_DECORATIVE,
    }:
        return PROCESSING_ACTION_PRESERVE
    return PROCESSING_ACTION_REVIEW


def _default_role_for_block(block: Any) -> str:
    role_hint = _normalized_choice(
        getattr(block, "semantic_role_hint", ""),
        VALID_SEMANTIC_ROLES,
    )
    if role_hint:
        return role_hint

    text_class = str(getattr(block, "text_class", "") or "").strip().lower()
    if (
        not text_class
        and str(getattr(block, "text", "") or "").strip()
    ):
        # Legacy/project-restored blocks may predate text_class. Preserve the
        # former translate-all behavior until an explicit semantic classifier
        # has supplied a stronger signal.
        return SEMANTIC_ROLE_DIALOGUE_FREE
    if text_class == "text_bubble":
        return SEMANTIC_ROLE_DIALOGUE_BUBBLE
    if text_class == "text_free":
        # Preserve current product recall. A later explicit SFX, decorative,
        # or UI signal may override this before destructive work begins.
        return SEMANTIC_ROLE_DIALOGUE_FREE
    if text_class in {"narration", "caption"}:
        return SEMANTIC_ROLE_NARRATION
    if text_class in {"ui", "sign", "ui_or_sign"}:
        return SEMANTIC_ROLE_UI_OR_SIGN
    if text_class in {"sfx", "onomatopoeia"}:
        return SEMANTIC_ROLE_SFX
    if text_class in {"decorative", "decoration"}:
        return SEMANTIC_ROLE_DECORATIVE
    return SEMANTIC_ROLE_AMBIGUOUS


def _default_mask_strategy(
    block: Any,
    *,
    semantic_role: str,
    processing_action: str,
) -> str:
    if processing_action != PROCESSING_ACTION_TRANSLATE_INPAINT:
        return MASK_STRATEGY_PRESERVE_ORIGINAL
    if any(
        bool(getattr(block, field_name, False))
        for field_name in (
            "bubble_panel_text_candidate",
            "bubble_transparency_risk",
            "background_structure_risk",
        )
    ):
        return MASK_STRATEGY_GLYPH_ONLY_STRUCTURE_PROTECT
    if (
        semantic_role == SEMANTIC_ROLE_DIALOGUE_FREE
        or str(getattr(block, "text_class", "") or "").strip().lower()
        == "text_free"
    ):
        return MASK_STRATEGY_GLYPH_ONLY
    return MASK_STRATEGY_BUBBLE_SAFE


def assign_ocr_processing_contract(
    block: Any,
    *,
    semantic_role: str,
    processing_action: str,
    decision_source: str,
    reasons: Iterable[str] = (),
) -> None:
    """Assign one validated semantic/action decision to a block.

    Callers must provide explicit evidence when overriding the conservative
    detector-class default. Unknown roles or actions fail closed as review.
    """

    initialize_ocr_result_contract(block)
    role = _normalized_choice(semantic_role, VALID_SEMANTIC_ROLES)
    action = _normalized_choice(
        processing_action,
        VALID_PROCESSING_ACTIONS,
    )
    normalized_reasons = sorted(
        {
            str(reason or "").strip()
            for reason in reasons
            if str(reason or "").strip()
        }
    )
    if not role:
        role = SEMANTIC_ROLE_AMBIGUOUS
        normalized_reasons.append("invalid_semantic_role")
    if not action:
        action = PROCESSING_ACTION_REVIEW
        normalized_reasons.append("invalid_processing_action")

    block.semantic_role = role
    block.processing_action = action
    block.processing_decision_source = str(decision_source or "unknown")
    block.processing_decision_reasons = sorted(set(normalized_reasons))
    block.mask_strategy = _default_mask_strategy(
        block,
        semantic_role=role,
        processing_action=action,
    )
    block.processing_contract_diagnostics = {
        "schema_version": OCR_PROCESSING_CONTRACT_SCHEMA_VERSION,
        "decision_source": block.processing_decision_source,
        "reasons": list(block.processing_decision_reasons),
        "semantic_role": role,
        "processing_action": action,
        "mask_strategy": block.mask_strategy,
    }


def finalize_ocr_processing_contract(block: Any) -> None:
    """Fill missing processing fields without overriding explicit decisions."""

    initialize_ocr_result_contract(block)
    role = _normalized_choice(
        getattr(block, "semantic_role", ""),
        VALID_SEMANTIC_ROLES,
    )
    action = _normalized_choice(
        getattr(block, "processing_action", ""),
        VALID_PROCESSING_ACTIONS,
    )
    source = str(
        getattr(block, "processing_decision_source", "") or ""
    ).strip()
    reasons = list(
        getattr(block, "processing_decision_reasons", []) or []
    )

    if not role:
        role = _default_role_for_block(block)
        reasons.append("detector_text_class_default")
    if not action:
        action_hint = _normalized_choice(
            getattr(block, "processing_action_hint", ""),
            VALID_PROCESSING_ACTIONS,
        )
        action = action_hint or _default_action_for_role(role)
        reasons.append(
            "processing_action_hint"
            if action_hint
            else "semantic_role_default"
        )
    assign_ocr_processing_contract(
        block,
        semantic_role=role,
        processing_action=action,
        decision_source=source or "ocr_processing_default",
        reasons=reasons,
    )


def finalize_ocr_processing_contracts(
    blocks: Iterable[Any] | None,
) -> dict[str, Any]:
    block_list = list(blocks or [])
    role_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    mask_strategy_counts: dict[str, int] = {}
    for block in block_list:
        finalize_ocr_processing_contract(block)
        role = str(getattr(block, "semantic_role", "") or "")
        action = str(getattr(block, "processing_action", "") or "")
        mask_strategy = str(getattr(block, "mask_strategy", "") or "")
        role_counts[role] = role_counts.get(role, 0) + 1
        action_counts[action] = action_counts.get(action, 0) + 1
        mask_strategy_counts[mask_strategy] = (
            mask_strategy_counts.get(mask_strategy, 0) + 1
        )
    return {
        "schema_version": OCR_PROCESSING_CONTRACT_SCHEMA_VERSION,
        "block_count": len(block_list),
        "semantic_role_counts": role_counts,
        "processing_action_counts": action_counts,
        "mask_strategy_counts": mask_strategy_counts,
    }


def select_translate_inpaint_blocks(
    blocks: Iterable[Any] | None,
) -> list[Any]:
    selected: list[Any] = []
    for block in list(blocks or []):
        finalize_ocr_processing_contract(block)
        if (
            getattr(block, "processing_action", "")
            == PROCESSING_ACTION_TRANSLATE_INPAINT
        ):
            selected.append(block)
    return selected


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
    if not hasattr(block, "processing_decision_source"):
        block.processing_decision_source = ""
    if not isinstance(
        getattr(block, "processing_decision_reasons", None),
        list,
    ):
        block.processing_decision_reasons = []
    if not isinstance(
        getattr(block, "processing_contract_diagnostics", None),
        dict,
    ):
        block.processing_contract_diagnostics = {}
    if not hasattr(block, "mask_strategy"):
        block.mask_strategy = ""
    if not hasattr(block, "mask_strategy_reason"):
        block.mask_strategy_reason = ""
    if not hasattr(block, "mask_actual_bbox"):
        block.mask_actual_bbox = None
    if not hasattr(block, "mask_actual_pixel_count"):
        block.mask_actual_pixel_count = 0
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
