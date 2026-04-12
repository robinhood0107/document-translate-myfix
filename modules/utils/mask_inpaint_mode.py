from __future__ import annotations

MASK_INPAINT_MODE_RTDETR_LEGACY_BBOX_SOURCE = "rtdetr_legacy_bbox_source_lama"
MASK_INPAINT_MODE_SOURCE_PARITY = "source_parity_ctd_lama"

DEPRECATED_MASK_INPAINT_MODE_RTDETR_SOURCE = "rtdetr_source_ctd_lama"

DEFAULT_MASK_INPAINT_MODE = MASK_INPAINT_MODE_RTDETR_LEGACY_BBOX_SOURCE

MASK_INPAINT_MODE_VALUES = {
    MASK_INPAINT_MODE_RTDETR_LEGACY_BBOX_SOURCE,
    MASK_INPAINT_MODE_SOURCE_PARITY,
}


_DEPRECATED_MODE_ALIASES = {
    DEPRECATED_MASK_INPAINT_MODE_RTDETR_SOURCE: MASK_INPAINT_MODE_RTDETR_LEGACY_BBOX_SOURCE,
}


def normalize_mask_inpaint_mode(value: str | None) -> str:
    raw = str(value or "").strip()
    raw = _DEPRECATED_MODE_ALIASES.get(raw, raw)
    if raw not in MASK_INPAINT_MODE_VALUES:
        return DEFAULT_MASK_INPAINT_MODE
    return raw


def uses_source_compat_mode(value: str | None) -> bool:
    return normalize_mask_inpaint_mode(value) in MASK_INPAINT_MODE_VALUES


def uses_source_parity_detector(value: str | None) -> bool:
    return normalize_mask_inpaint_mode(value) == MASK_INPAINT_MODE_SOURCE_PARITY


def uses_legacy_bbox_source_mode(value: str | None) -> bool:
    return normalize_mask_inpaint_mode(value) == MASK_INPAINT_MODE_RTDETR_LEGACY_BBOX_SOURCE
