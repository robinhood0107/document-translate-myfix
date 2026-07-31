from __future__ import annotations

import json
import os
import re

import imkit as imk
import numpy as np
from PIL import Image, ImageOps

from modules.ocr.common.result_contract import (
    PROCESSING_ACTION_PRESERVE,
    PROCESSING_ACTION_REVIEW,
    PROCESSING_ACTION_TRANSLATE_INPAINT,
    SEMANTIC_ROLE_AMBIGUOUS,
    SEMANTIC_ROLE_DIALOGUE_BUBBLE,
    SEMANTIC_ROLE_UI_OR_SIGN,
    assign_ocr_processing_contract,
    finalize_ocr_processing_contract,
)
from modules.utils.debug_artifacts import (
    atomic_debug_image,
    atomic_debug_json,
)


OCR_STATUS_OK = "ok"
OCR_STATUS_EMPTY_INITIAL = "empty_initial"
OCR_STATUS_OK_AFTER_RETRY = "ok_after_retry"
OCR_STATUS_EMPTY_AFTER_RETRY = "empty_after_retry"
OCR_EMPTY_REASON_LAYOUT_SCHEMA_LABELS = "PaddleOCR VL returned layout schema labels instead of OCR text."
OCR_EMPTY_REASON_TEXT_FREE_NO_VISUAL_EVIDENCE = (
    "PaddleOCR VL skipped text_free crop without enough visual text evidence."
)
OCR_EMPTY_REASON_NON_TEXT_RESPONSE = "PaddleOCR VL returned a non-text response."
OCR_EMPTY_REASON_EMBEDDED_UI_CLUSTER = (
    "OCR text is part of a dense embedded device/UI cluster and should be preserved."
)
OCR_EMPTY_REASON_SPOTTING_UNMATCHED = (
    "PaddleOCR VL Spotting did not map a native region to this detector block."
)
UI_PANEL_MODE_PRESERVE_ORIGINAL = "preserve_original"
UI_PANEL_MODE_PREVIEW = "ui_panel_mode_preview"
UI_PANEL_MODE_BUBBLE_PANEL_TEXT = "bubble_panel_text_candidate"
UI_PANEL_REVIEW_REASON_LAYOUT = "embedded_ui_panel_layout_review"
UI_PANEL_REVIEW_REASON_CLUSTER = "embedded_device_ui_cluster"
BUBBLE_PANEL_REVIEW_REASON = "bubble_panel_text_candidate"
DEFAULT_RETRY_CROP_X_RATIO = 0.06
DEFAULT_RETRY_CROP_Y_RATIO = 0.10

OCR_REJECTED_EMPTY_REASONS = frozenset(
    {
        OCR_EMPTY_REASON_LAYOUT_SCHEMA_LABELS,
        OCR_EMPTY_REASON_TEXT_FREE_NO_VISUAL_EVIDENCE,
        OCR_EMPTY_REASON_NON_TEXT_RESPONSE,
        OCR_EMPTY_REASON_EMBEDDED_UI_CLUSTER,
        OCR_EMPTY_REASON_SPOTTING_UNMATCHED,
    }
)

_EMBEDDED_UI_TOKEN_RE = re.compile(
    r"[@#]|[A-Za-z0-9]|"
    r"(?:\u30e6\u30fc\u30b6\u30fc|\u30e1\u30cb\u30e5\u30fc|\u30aa\u30d7\u30b7\u30e7\u30f3|"
    r"\u30a2\u30af\u30bb\u30b9|\u30c4\u30a4\u30fc\u30c8|\u30d5\u30a9\u30ed|"
    r"\u304a\u624b\u8efd|\u30aa\u30b9\u30b9\u30e1|\u8cfc\u5165|\u679a|\u5238|"
    r"\u30c7\u30fc\u30bf|\u6570\u636e|\u30d1\u30c3\u30af)"
)
_EMBEDDED_UI_PANEL_TOKEN_RE = re.compile(
    r"(?:\u30e6\u30fc\u30b6\u30fc|\u30e1\u30cb\u30e5\u30fc|\u30aa\u30d7\u30b7\u30e7\u30f3|"
    r"\u30a2\u30af\u30bb\u30b9|\u30c4\u30a4\u30fc\u30c8|\u30d5\u30a9\u30ed|"
    r"\u304a\u624b\u8efd|\u30aa\u30b9\u30b9\u30e1|\u8cfc\u5165|\u679a|\u5238|"
    r"\u30c7\u30fc\u30bf|\u6570\u636e|\u30d1\u30c3\u30af|\u8a18\u61b6)"
)
_EMBEDDED_UI_PUNCT_RE = re.compile(r"^[\s.()\[\]{}:;!?\-_/\\\u30fb\u3001\u3002]+")


def ensure_three_channel(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 1:
        channel = image[:, :, 0]
        return np.stack([channel] * 3, axis=-1).astype(np.uint8)
    return image.astype(np.uint8)


def _clip_bbox(x1: float, y1: float, x2: float, y2: float, image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    img_h, img_w = image_shape[:2]
    x1_i = max(0, min(int(np.floor(x1)), img_w))
    y1_i = max(0, min(int(np.floor(y1)), img_h))
    x2_i = max(0, min(int(np.ceil(x2)), img_w))
    y2_i = max(0, min(int(np.ceil(y2)), img_h))
    return x1_i, y1_i, x2_i, y2_i


def expand_bbox(
    xyxy,
    image_shape: tuple[int, ...],
    x_ratio: float = 0.0,
    y_ratio: float = 0.0,
    clamp_xyxy=None,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    x_pad = width * x_ratio
    y_pad = height * y_ratio
    expanded = _clip_bbox(x1 - x_pad, y1 - y_pad, x2 + x_pad, y2 + y_pad, image_shape)
    if clamp_xyxy is None:
        return expanded
    clamp_box = _clip_bbox(*[float(v) for v in clamp_xyxy], image_shape)
    ex1, ey1, ex2, ey2 = expanded
    cx1, cy1, cx2, cy2 = clamp_box
    return max(ex1, cx1), max(ey1, cy1), min(ex2, cx2), min(ey2, cy2)


def crop_bbox_image(
    image: np.ndarray,
    xyxy,
    auto_rotate_tall: bool = True,
) -> np.ndarray | None:
    x1, y1, x2, y2 = _clip_bbox(*[float(v) for v in xyxy], image.shape)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return None
    crop = ensure_three_channel(crop)
    if auto_rotate_tall:
        h, w = crop.shape[:2]
        if h > 0 and w > 0 and (h / float(w)) >= 1.5:
            crop = np.rot90(crop)
    return crop


def crop_block_image(
    image: np.ndarray,
    xyxy,
    x_ratio: float = 0.0,
    y_ratio: float = 0.0,
    auto_rotate_tall: bool = True,
    clamp_xyxy=None,
) -> np.ndarray | None:
    bbox = expand_bbox(xyxy, image.shape, x_ratio=x_ratio, y_ratio=y_ratio, clamp_xyxy=clamp_xyxy)
    return crop_bbox_image(image, bbox, auto_rotate_tall=auto_rotate_tall)


def resolve_block_crop_bbox(
    block,
    image_shape: tuple[int, ...],
    *,
    x_ratio: float = 0.0,
    y_ratio: float = 0.0,
    bubble_as_clamp: bool = True,
    fallback_to_bubble: bool = True,
) -> tuple[tuple[int, int, int, int] | None, str]:
    text_bbox = getattr(block, "xyxy", None)
    bubble_bbox = getattr(block, "bubble_xyxy", None)
    clamp_bbox = bubble_bbox if bubble_as_clamp and bubble_bbox is not None else None

    if text_bbox is not None:
        bbox = expand_bbox(
            text_bbox,
            image_shape,
            x_ratio=x_ratio,
            y_ratio=y_ratio,
            clamp_xyxy=clamp_bbox,
        )
        x1, y1, x2, y2 = bbox
        if x2 > x1 and y2 > y1:
            return bbox, "xyxy"

    if fallback_to_bubble and bubble_bbox is not None:
        bbox = expand_bbox(bubble_bbox, image_shape)
        x1, y1, x2, y2 = bbox
        if x2 > x1 and y2 > y1:
            return bbox, "bubble_fallback"

    return None, ""


def _otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    sum_total = np.dot(np.arange(256, dtype=np.float64), hist)
    sum_background = 0.0
    weight_background = 0.0
    max_variance = -1.0
    threshold = 0

    for idx in range(256):
        weight_background += hist[idx]
        if weight_background <= 0:
            continue

        weight_foreground = total - weight_background
        if weight_foreground <= 0:
            break

        sum_background += idx * hist[idx]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > max_variance:
            max_variance = variance
            threshold = idx

    return int(threshold)


def build_retry_crop_bbox(
    image_shape: tuple[int, ...],
    xyxy,
    *,
    clamp_xyxy=None,
    x_ratio: float = DEFAULT_RETRY_CROP_X_RATIO,
    y_ratio: float = DEFAULT_RETRY_CROP_Y_RATIO,
) -> tuple[int, int, int, int] | None:
    if xyxy is None:
        return None
    bbox = expand_bbox(image_shape=image_shape, xyxy=xyxy, x_ratio=x_ratio, y_ratio=y_ratio, clamp_xyxy=clamp_xyxy)
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    return bbox


def build_retry_crop_from_bbox(image: np.ndarray, bbox) -> np.ndarray | None:
    if bbox is None:
        return None
    crop = crop_bbox_image(image, bbox, auto_rotate_tall=True)
    if crop is None:
        return None
    gray = imk.to_gray(crop)
    contrasted = np.array(ImageOps.autocontrast(Image.fromarray(gray)), dtype=np.uint8)
    threshold = _otsu_threshold(contrasted)
    binary = np.where(contrasted > threshold, 255, 0).astype(np.uint8)
    return ensure_three_channel(binary)


def build_retry_crop(
    image: np.ndarray,
    xyxy,
    *,
    clamp_xyxy=None,
    x_ratio: float = DEFAULT_RETRY_CROP_X_RATIO,
    y_ratio: float = DEFAULT_RETRY_CROP_Y_RATIO,
) -> np.ndarray | None:
    bbox = build_retry_crop_bbox(
        image.shape,
        xyxy,
        clamp_xyxy=clamp_xyxy,
        x_ratio=x_ratio,
        y_ratio=y_ratio,
    )
    return build_retry_crop_from_bbox(image, bbox)


def set_block_ocr_diagnostics(
    block,
    *,
    text: str,
    confidence: float,
    status: str,
    empty_reason: str,
    attempt_count: int,
    raw_text: str | None = None,
    sanitized_text: str | None = None,
) -> None:
    final_text = text or ""
    block.text = final_text
    block.ocr_confidence = float(confidence or 0.0)
    block.ocr_status = status
    block.ocr_empty_reason = empty_reason or ""
    block.ocr_attempt_count = int(attempt_count or 0)
    block.ocr_raw_text = final_text if raw_text is None else str(raw_text or "")
    block.ocr_sanitized_text = final_text if sanitized_text is None else str(sanitized_text or "")


def set_block_ocr_crop_diagnostics(
    block,
    *,
    effective_crop_xyxy=None,
    retry_crop_xyxy=None,
    crop_source: str | None = None,
) -> None:
    if effective_crop_xyxy is not None:
        block.ocr_effective_crop_xyxy = [int(float(v)) for v in effective_crop_xyxy]
    elif not hasattr(block, "ocr_effective_crop_xyxy"):
        block.ocr_effective_crop_xyxy = None

    if retry_crop_xyxy is not None:
        block.ocr_retry_crop_xyxy = [int(float(v)) for v in retry_crop_xyxy]
    elif not hasattr(block, "ocr_retry_crop_xyxy"):
        block.ocr_retry_crop_xyxy = None

    if crop_source is not None:
        block.ocr_crop_source = crop_source
    elif not hasattr(block, "ocr_crop_source"):
        block.ocr_crop_source = ""


def is_block_ocr_empty(block) -> bool:
    return getattr(block, "ocr_status", "") in {
        OCR_STATUS_EMPTY_INITIAL,
        OCR_STATUS_EMPTY_AFTER_RETRY,
    }


def is_layout_schema_only_ocr_rejection(block) -> bool:
    return (
        is_block_ocr_empty(block)
        and not str(getattr(block, "text", "") or "").strip()
        and str(getattr(block, "ocr_empty_reason", "") or "") == OCR_EMPTY_REASON_LAYOUT_SCHEMA_LABELS
    )


def drop_layout_schema_only_ocr_blocks(blocks) -> tuple[list, list]:
    kept = []
    dropped = []
    for block in list(blocks or []):
        if is_layout_schema_only_ocr_rejection(block):
            dropped.append(block)
        else:
            kept.append(block)
    return kept, dropped


def is_rejected_empty_ocr_block(block) -> bool:
    return (
        is_block_ocr_empty(block)
        and not str(getattr(block, "text", "") or "").strip()
        and str(getattr(block, "ocr_empty_reason", "") or "") in OCR_REJECTED_EMPTY_REASONS
    )


def all_empty_blocks_are_rejected(blocks) -> bool:
    block_list = list(blocks or [])
    if not block_list:
        return False
    for block in block_list:
        if str(getattr(block, "text", "") or "").strip():
            return False
        if not is_rejected_empty_ocr_block(block):
            return False
    return True


def drop_rejected_empty_ocr_blocks(blocks) -> tuple[list, list]:
    kept = []
    dropped = []
    for block in list(blocks or []):
        if is_rejected_empty_ocr_block(block):
            assign_ocr_processing_contract(
                block,
                semantic_role=SEMANTIC_ROLE_AMBIGUOUS,
                processing_action=PROCESSING_ACTION_REVIEW,
                decision_source="ocr_rejected_empty",
                reasons=(
                    str(getattr(block, "ocr_empty_reason", "") or ""),
                    str(getattr(block, "ocr_reject_reason", "") or ""),
                ),
            )
            dropped.append(block)
        else:
            kept.append(block)
    return kept, dropped


def _block_text_class(block) -> str:
    value = getattr(block, "text_class", "")
    if not value:
        value = getattr(block, "class_name", "")
    return str(value or "").strip().lower()


def _block_text(block) -> str:
    return str(getattr(block, "text", "") or "").strip()


def _block_xyxy(block) -> tuple[float, float, float, float] | None:
    try:
        x1, y1, x2, y2 = [float(v) for v in getattr(block, "xyxy", ())]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _bbox_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _bbox_union(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _bbox_area(box: tuple[float, float, float, float] | None) -> float:
    if box is None:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _point_inside_expanded_bbox(
    point: tuple[float, float],
    box: tuple[float, float, float, float],
    padding: float,
) -> bool:
    x, y = point
    return (box[0] - padding) <= x <= (box[2] + padding) and (box[1] - padding) <= y <= (box[3] + padding)


def _block_bubble_xyxy(block) -> tuple[float, float, float, float] | None:
    try:
        x1, y1, x2, y2 = [float(v) for v in getattr(block, "bubble_xyxy", ())]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _is_bubble_protected_text_block(block) -> bool:
    if _block_text_class(block) != "text_bubble":
        return False
    text_box = _block_xyxy(block)
    bubble_box = _block_bubble_xyxy(block)
    if text_box is None or bubble_box is None:
        return False

    text_area = _bbox_area(text_box)
    bubble_area = _bbox_area(bubble_box)
    if text_area <= 0.0 or bubble_area <= 0.0:
        return False

    center_inside = _point_inside_expanded_bbox(_bbox_center(text_box), bubble_box, padding=1.0)
    contained_area = _bbox_area(
        (
            max(text_box[0], bubble_box[0]),
            max(text_box[1], bubble_box[1]),
            min(text_box[2], bubble_box[2]),
            min(text_box[3], bubble_box[3]),
        )
    )
    contained_ratio = contained_area / max(1.0, text_area)

    # Device/app UIs often contain words such as "menu", "access", or counts.
    # If the detector also found a containing speech/narration bubble, prefer
    # recall for dialogue and keep the block out of the embedded-UI seed set.
    return (center_inside or contained_ratio >= 0.80) and bubble_area >= text_area * 1.10


def is_embedded_ui_panel_layout_review_candidate(block) -> bool:
    if _block_text_class(block) != "text_bubble":
        return False
    text = _block_text(block)
    compact_text = re.sub(r"\s+", "", text)
    if len(compact_text) < 18 or not _EMBEDDED_UI_PANEL_TOKEN_RE.search(compact_text):
        return False

    text_box = _block_xyxy(block)
    bubble_box = _block_bubble_xyxy(block)
    if text_box is None or bubble_box is None:
        return False

    text_area = _bbox_area(text_box)
    bubble_area = _bbox_area(bubble_box)
    if text_area <= 0.0 or bubble_area <= 0.0:
        return False
    contained_area = _bbox_area(
        (
            max(text_box[0], bubble_box[0]),
            max(text_box[1], bubble_box[1]),
            min(text_box[2], bubble_box[2]),
            min(text_box[3], bubble_box[3]),
        )
    )
    contained_ratio = contained_area / max(1.0, text_area)
    return contained_ratio >= 0.70 and bubble_area >= text_area * 1.05


def is_bubble_panel_text_candidate(block) -> bool:
    if _block_text_class(block) != "text_bubble":
        return False
    text = _block_text(block)
    compact_text = re.sub(r"\s+", "", text)
    if len(compact_text) < 18 or not _EMBEDDED_UI_PANEL_TOKEN_RE.search(compact_text):
        return False

    text_box = _block_xyxy(block)
    bubble_box = _block_bubble_xyxy(block)
    if text_box is None or bubble_box is None:
        return False
    text_area = _bbox_area(text_box)
    bubble_area = _bbox_area(bubble_box)
    if text_area <= 0.0 or bubble_area <= 0.0:
        return False

    intersection = (
        max(text_box[0], bubble_box[0]),
        max(text_box[1], bubble_box[1]),
        min(text_box[2], bubble_box[2]),
        min(text_box[3], bubble_box[3]),
    )
    contained_ratio = _bbox_area(intersection) / max(1.0, text_area)
    center_inside = _point_inside_expanded_bbox(_bbox_center(text_box), bubble_box, padding=1.0)
    return center_inside and contained_ratio >= 0.80 and bubble_area >= text_area * 1.05


def mark_bubble_panel_text_candidate(block, *, group_id: str = "", member_indices: list[int] | None = None) -> None:
    block.bubble_panel_text_candidate = True
    block.ui_panel_mode = UI_PANEL_MODE_BUBBLE_PANEL_TEXT
    block.ui_panel_preview_path = str(getattr(block, "ui_panel_preview_path", "") or "")
    block.mask_decision = "review"
    block.mask_reject_reason = BUBBLE_PANEL_REVIEW_REASON
    block.bubble_panel_group_id = str(group_id or getattr(block, "bubble_panel_group_id", "") or "")
    block.bubble_panel_member_indices = list(member_indices or getattr(block, "bubble_panel_member_indices", []) or [])
    block.bubble_panel_merge_decision = str(getattr(block, "bubble_panel_merge_decision", "") or "candidate")
    block.bubble_merge_reocr_needed = bool(getattr(block, "bubble_merge_reocr_needed", False))
    assign_ocr_processing_contract(
        block,
        semantic_role=SEMANTIC_ROLE_DIALOGUE_BUBBLE,
        processing_action=PROCESSING_ACTION_TRANSLATE_INPAINT,
        decision_source="embedded_ui_bubble_dialogue",
        reasons=(BUBBLE_PANEL_REVIEW_REASON,),
    )


def _bubble_panel_group_key(block) -> tuple[int, int, int, int] | None:
    bubble = _block_bubble_xyxy(block)
    if bubble is None:
        return None
    return tuple(int(round(v / 8.0)) for v in bubble)


def _shrink_box(box: tuple[float, float, float, float], percent: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    dx = width * percent
    dy = height * percent
    return x1 + dx, y1 + dy, x2 - dx, y2 - dy


def group_bubble_panel_text_candidates(blocks) -> list[dict]:
    block_list = list(blocks or [])
    grouped: dict[tuple[int, int, int, int], list[tuple[int, object]]] = {}
    for idx, block in enumerate(block_list):
        if not is_bubble_panel_text_candidate(block):
            continue
        key = _bubble_panel_group_key(block)
        if key is None:
            continue
        grouped.setdefault(key, []).append((idx, block))

    groups: list[dict] = []
    for key, members in grouped.items():
        if not members:
            continue
        member_indices = [idx for idx, _ in members]
        bubble = _block_bubble_xyxy(members[0][1])
        if bubble is None:
            continue
        text_boxes = [_block_xyxy(block) for _, block in members]
        text_boxes = [box for box in text_boxes if box is not None]
        group_box = _bbox_union(text_boxes) if text_boxes else bubble
        safe_box = _shrink_box(bubble, 0.08)
        render_box = (
            max(safe_box[0], min(group_box[0], safe_box[2])),
            max(safe_box[1], min(group_box[1], safe_box[3])),
            min(safe_box[2], max(group_box[2], safe_box[0])),
            min(safe_box[3], max(group_box[3], safe_box[1])),
        )
        if _bbox_area(render_box) <= 0.0:
            render_box = safe_box
        group_id = "bubble_panel_" + "_".join(str(v) for v in key)
        requires_review = len(members) > 1
        for _member_offset, (idx, block) in enumerate(members):
            mark_bubble_panel_text_candidate(block, group_id=group_id, member_indices=member_indices)
            block.bubble_panel_group_id = group_id
            block.bubble_panel_member_indices = member_indices
            block.bubble_panel_merge_decision = (
                "merge_split_review"
                if requires_review
                else "group_primary"
            )
            block.bubble_merge_reocr_needed = requires_review
            block.bubble_panel_group_xyxy = [int(round(v)) for v in group_box]
            block.bubble_panel_render_xyxy = [int(round(v)) for v in render_box]
            block.merge_split_diagnostics = {
                **dict(
                    getattr(block, "merge_split_diagnostics", {}) or {}
                ),
                "bubble_panel_group_id": group_id,
                "member_indices": list(member_indices),
                "relationship": (
                    "n_to_1_review"
                    if requires_review
                    else "one_to_one"
                ),
                "automatic_merge": False,
                "automatic_split": False,
            }
            if requires_review:
                assign_ocr_processing_contract(
                    block,
                    semantic_role=SEMANTIC_ROLE_AMBIGUOUS,
                    processing_action=PROCESSING_ACTION_REVIEW,
                    decision_source="bubble_panel_merge_split_review",
                    reasons=("multiple_regions_share_bubble",),
                )
        groups.append(
            {
                "group_id": group_id,
                "member_indices": member_indices,
                "members": [block for _, block in members],
                "bubble_xyxy": [int(round(v)) for v in bubble],
                "group_xyxy": [int(round(v)) for v in group_box],
                "render_xyxy": [int(round(v)) for v in render_box],
                "requires_review": requires_review,
            }
        )
    return groups


def mark_ui_panel_review_candidate(block, *, reason: str, preview_path: str = "") -> None:
    block.ui_panel_mode = UI_PANEL_MODE_PRESERVE_ORIGINAL
    block.ui_panel_preview_path = str(preview_path or "")
    block.mask_decision = "review"
    block.mask_reject_reason = str(reason or UI_PANEL_REVIEW_REASON_LAYOUT)
    if reason == UI_PANEL_REVIEW_REASON_CLUSTER:
        assign_ocr_processing_contract(
            block,
            semantic_role=SEMANTIC_ROLE_UI_OR_SIGN,
            processing_action=PROCESSING_ACTION_PRESERVE,
            decision_source="embedded_ui_cluster",
            reasons=(reason,),
        )
    else:
        assign_ocr_processing_contract(
            block,
            semantic_role=SEMANTIC_ROLE_AMBIGUOUS,
            processing_action=PROCESSING_ACTION_REVIEW,
            decision_source="embedded_ui_layout_review",
            reasons=(reason or UI_PANEL_REVIEW_REASON_LAYOUT,),
        )


def split_inpaint_protected_ocr_blocks(blocks) -> tuple[list, list]:
    """Keep review-worthy embedded UI panels out of LaMa masks without dropping them."""
    inpaint_blocks: list = []
    protected_blocks: list = []
    block_list = list(blocks or [])
    group_bubble_panel_text_candidates(block_list)
    for block in block_list:
        finalize_ocr_processing_contract(block)
        if (
            getattr(block, "processing_action", "")
            != PROCESSING_ACTION_TRANSLATE_INPAINT
        ):
            block._inpaint_protected_reason = (
                "processing_action_"
                + str(getattr(block, "processing_action", "") or "review")
            )
            block.mask_decision = "review"
            protected_blocks.append(block)
            continue
        if is_bubble_panel_text_candidate(block):
            mark_bubble_panel_text_candidate(block)
            inpaint_blocks.append(block)
            continue
        if is_embedded_ui_panel_layout_review_candidate(block):
            block._inpaint_protected_reason = UI_PANEL_REVIEW_REASON_LAYOUT
            mark_ui_panel_review_candidate(block, reason=UI_PANEL_REVIEW_REASON_LAYOUT)
            protected_blocks.append(block)
        else:
            inpaint_blocks.append(block)
    return inpaint_blocks, protected_blocks


def _is_small_embedded_ui_label(
    box: tuple[float, float, float, float],
    image_shape: tuple[int, ...],
) -> bool:
    img_h, img_w = image_shape[:2]
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    horizontal_label = height <= max(120.0, img_h * 0.045) and width <= img_w * 0.32
    vertical_label = width <= max(150.0, img_w * 0.07) and height <= max(220.0, img_h * 0.075)
    return horizontal_label or vertical_label


def _is_embedded_ui_seed(block, image_shape: tuple[int, ...]) -> bool:
    text = _block_text(block)
    if not text:
        return False
    if _is_bubble_protected_text_block(block):
        return False
    box = _block_xyxy(block)
    if box is None:
        return False
    img_h, img_w = image_shape[:2]
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    cls = _block_text_class(block)
    has_ui_token = bool(_EMBEDDED_UI_TOKEN_RE.search(text))
    starts_like_ui_label = bool(_EMBEDDED_UI_PUNCT_RE.search(text))
    compact_long_text = len(text.replace(" ", "").replace("\n", "")) >= 24
    small_label = height <= max(110.0, img_h * 0.04) and width <= img_w * 0.28

    if cls == "text_free":
        return bool(has_ui_token and (small_label or starts_like_ui_label))

    return bool(has_ui_token and compact_long_text and width <= img_w * 0.22)


def drop_embedded_ui_ocr_blocks(
    blocks,
    image_shape: tuple[int, ...] | None,
    *,
    min_cluster_size: int = 4,
) -> tuple[list, list]:
    """Drop dense embedded phone/app UI labels before mask generation.

    These are real letters, but translating them in-place tends to destroy small
    device screenshots. Dropping before inpaint preserves the original UI instead
    of blanking it and rendering cramped Korean labels.
    """
    block_list = list(blocks or [])
    if not block_list or image_shape is None or len(image_shape) < 2:
        return block_list, []

    candidates: list[tuple[int, object, tuple[float, float, float, float]]] = []
    for idx, block in enumerate(block_list):
        box = _block_xyxy(block)
        if box is None:
            continue
        if _is_embedded_ui_seed(block, image_shape):
            candidates.append((idx, block, box))

    if len(candidates) < min_cluster_size:
        return block_list, []

    img_h, img_w = image_shape[:2]
    x_link = max(520.0, img_w * 0.22)
    y_link = max(420.0, img_h * 0.16)
    parent = list(range(len(candidates)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    centers = [_bbox_center(item[2]) for item in candidates]
    for i in range(len(candidates)):
        cx1, cy1 = centers[i]
        for j in range(i + 1, len(candidates)):
            cx2, cy2 = centers[j]
            if abs(cx1 - cx2) <= x_link and abs(cy1 - cy2) <= y_link:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(len(candidates)):
        clusters.setdefault(find(i), []).append(i)

    drop_indices: set[int] = set()
    qualified_boxes: list[tuple[float, float, float, float]] = []
    for member_indexes in clusters.values():
        if len(member_indexes) < min_cluster_size:
            continue
        boxes = [candidates[i][2] for i in member_indexes]
        qualified_boxes.append(_bbox_union(boxes))
        for i in member_indexes:
            drop_indices.add(candidates[i][0])

    row_y_tolerance = max(130.0, img_h * 0.055)
    row_min_width = img_w * 0.35
    for i, (_, _, seed_box) in enumerate(candidates):
        if not _is_small_embedded_ui_label(seed_box, image_shape):
            continue
        seed_center = _bbox_center(seed_box)
        row_members: list[int] = []
        for j, (_, _, other_box) in enumerate(candidates):
            if not _is_small_embedded_ui_label(other_box, image_shape):
                continue
            other_center = _bbox_center(other_box)
            if abs(seed_center[1] - other_center[1]) <= row_y_tolerance:
                row_members.append(j)
        if len(row_members) < min_cluster_size:
            continue
        row_box = _bbox_union([candidates[j][2] for j in row_members])
        if (row_box[2] - row_box[0]) < row_min_width:
            continue
        qualified_boxes.append(row_box)
        for j in row_members:
            drop_indices.add(candidates[j][0])

    if not qualified_boxes:
        return block_list, []

    padding = max(220.0, min(img_w, img_h) * 0.055)
    for idx, block in enumerate(block_list):
        if idx in drop_indices:
            continue
        if _block_text_class(block) != "text_free":
            continue
        box = _block_xyxy(block)
        if box is None:
            continue
        if not _is_small_embedded_ui_label(box, image_shape):
            continue
        center = _bbox_center(box)
        if any(_point_inside_expanded_bbox(center, cluster_box, padding) for cluster_box in qualified_boxes):
            drop_indices.add(idx)

    if not drop_indices:
        return block_list, []

    kept = []
    dropped = []
    for idx, block in enumerate(block_list):
        if idx in drop_indices:
            block.ocr_status = OCR_STATUS_EMPTY_AFTER_RETRY
            block.ocr_empty_reason = OCR_EMPTY_REASON_EMBEDDED_UI_CLUSTER
            block.ocr_reject_reason = UI_PANEL_REVIEW_REASON_CLUSTER
            mark_ui_panel_review_candidate(block, reason=UI_PANEL_REVIEW_REASON_CLUSTER)
            dropped.append(block)
        else:
            kept.append(block)
    return kept, dropped


def build_ocr_debug_payload(
    page: str,
    ocr_engine: str,
    source_lang: str,
    blk_list,
) -> dict:
    payload = {
        "page": page,
        "ocr_engine": ocr_engine or "",
        "source_lang": source_lang or "",
        "blocks": [],
    }
    for idx, blk in enumerate(blk_list or []):
        x1, y1, x2, y2 = [int(float(v)) for v in getattr(blk, "xyxy", (0, 0, 0, 0))]
        payload["blocks"].append(
            {
                "index": idx,
                "bbox": [x1, y1, x2, y2],
                "text": getattr(blk, "text", "") or "",
                "raw_text": getattr(blk, "ocr_raw_text", getattr(blk, "text", "")) or "",
                "sanitized_text": getattr(blk, "ocr_sanitized_text", getattr(blk, "text", "")) or "",
                "confidence": float(getattr(blk, "ocr_confidence", 0.0) or 0.0),
                "status": getattr(blk, "ocr_status", "") or "",
                "empty_reason": getattr(blk, "ocr_empty_reason", "") or "",
                "reject_reason": getattr(blk, "ocr_reject_reason", "") or "",
                "attempt_count": int(getattr(blk, "ocr_attempt_count", 0) or 0),
                "effective_crop_xyxy": getattr(blk, "ocr_effective_crop_xyxy", None),
                "retry_crop_xyxy": getattr(blk, "ocr_retry_crop_xyxy", None),
                "crop_source": getattr(blk, "ocr_crop_source", "") or "",
                "ocr_strategy": getattr(blk, "ocr_strategy", "") or "",
                "ocr_model_identity": getattr(
                    blk, "ocr_model_identity", ""
                )
                or "",
                "ocr_runtime_identity": getattr(
                    blk, "ocr_runtime_identity", ""
                )
                or "",
                "ocr_geometry_provenance": getattr(
                    blk, "ocr_geometry_provenance", {}
                )
                or {},
                "semantic_role": getattr(blk, "semantic_role", "") or "",
                "processing_action": getattr(
                    blk, "processing_action", ""
                )
                or "",
                "processing_decision_source": getattr(
                    blk, "processing_decision_source", ""
                )
                or "",
                "processing_decision_reasons": getattr(
                    blk, "processing_decision_reasons", []
                )
                or [],
                "processing_contract_diagnostics": getattr(
                    blk, "processing_contract_diagnostics", {}
                )
                or {},
                "canonical_block_id": getattr(
                    blk, "canonical_block_id", ""
                )
                or "",
                "duplicate_alias_block_ids": getattr(
                    blk, "duplicate_alias_block_ids", []
                )
                or [],
                "duplicate_alias_count": int(
                    getattr(blk, "duplicate_alias_count", 0) or 0
                ),
                "merge_split_diagnostics": getattr(
                    blk, "merge_split_diagnostics", {}
                )
                or {},
                "ocr_regions": getattr(blk, "ocr_regions", None),
                "ocr_crop_bbox": getattr(blk, "ocr_crop_bbox", None),
                "ocr_resize_scale": getattr(blk, "ocr_resize_scale", None),
                "ui_panel_mode": getattr(blk, "ui_panel_mode", "") or "",
                "ui_panel_preview_path": getattr(blk, "ui_panel_preview_path", "") or "",
                "mask_decision": getattr(blk, "mask_decision", "") or "",
                "mask_reject_reason": getattr(blk, "mask_reject_reason", "") or "",
                "mask_strategy": getattr(blk, "mask_strategy", "") or "",
                "mask_strategy_reason": getattr(
                    blk, "mask_strategy_reason", ""
                )
                or "",
                "mask_actual_bbox": getattr(
                    blk, "mask_actual_bbox", None
                ),
                "mask_actual_pixel_count": int(
                    getattr(blk, "mask_actual_pixel_count", 0) or 0
                ),
                "bubble_panel_text_candidate": bool(getattr(blk, "bubble_panel_text_candidate", False)),
                "bubble_panel_group_id": getattr(blk, "bubble_panel_group_id", "") or "",
                "bubble_panel_member_indices": getattr(blk, "bubble_panel_member_indices", []) or [],
                "bubble_panel_mask_pixel_count": int(getattr(blk, "bubble_panel_mask_pixel_count", 0) or 0),
                "bubble_panel_mask_source": getattr(blk, "bubble_panel_mask_source", "") or "",
                "bubble_panel_merge_decision": getattr(blk, "bubble_panel_merge_decision", "") or "",
                "bubble_merge_reocr_needed": bool(getattr(blk, "bubble_merge_reocr_needed", False)),
            }
        )
    return payload


def export_ocr_debug_artifacts(
    output_dir: str,
    page_base_name: str,
    image: np.ndarray,
    blk_list,
    ocr_engine: str,
    source_lang: str,
    *,
    flat_names: bool = False,
) -> dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    payload = build_ocr_debug_payload(page_base_name, ocr_engine, source_lang, blk_list)
    debug_path = os.path.join(
        output_dir,
        "ocr-debug.json"
        if flat_names
        else f"{page_base_name}_ocr_debug.json",
    )
    if flat_names:
        debug_path = atomic_debug_json(
            output_dir,
            "ocr-debug.json",
            payload,
        )
    else:
        with open(debug_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=4)

    written = {"ocr_debug": debug_path}
    for idx, blk in enumerate(blk_list or []):
        if getattr(blk, "ocr_status", "") != OCR_STATUS_EMPTY_AFTER_RETRY:
            continue
        retry_bbox = getattr(blk, "ocr_retry_crop_xyxy", None) or getattr(blk, "xyxy", (0, 0, 0, 0))
        retry_crop = build_retry_crop_from_bbox(image, retry_bbox)
        if retry_crop is None or retry_crop.size == 0:
            continue
        retry_path = os.path.join(
            output_dir,
            f"ocr-retry-block-{idx:04d}.png"
            if flat_names
            else f"{page_base_name}_block_{idx}_retry.png",
        )
        if flat_names:
            retry_path = atomic_debug_image(
                output_dir,
                os.path.basename(retry_path),
                retry_crop,
            )
        else:
            imk.write_image(retry_path, retry_crop)
        written[f"retry_crop_{idx}"] = retry_path
    return written
