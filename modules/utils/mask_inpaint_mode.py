from __future__ import annotations

MASK_INPAINT_MODE_RTDETR_LEGACY_BBOX_SOURCE = "rtdetr_legacy_bbox_source_lama"
DEFAULT_MASK_INPAINT_MODE = MASK_INPAINT_MODE_RTDETR_LEGACY_BBOX_SOURCE

MASK_INPAINT_MODE_VALUES = {
    MASK_INPAINT_MODE_RTDETR_LEGACY_BBOX_SOURCE,
}


_DEPRECATED_MODE_ALIASES = {
    "rtdetr_source_ctd_lama": MASK_INPAINT_MODE_RTDETR_LEGACY_BBOX_SOURCE,
    "source_parity_ctd_lama": MASK_INPAINT_MODE_RTDETR_LEGACY_BBOX_SOURCE,
}


def normalize_mask_inpaint_mode(value: str | None) -> str:
    raw = str(value or "").strip()
    raw = _DEPRECATED_MODE_ALIASES.get(raw, raw)
    if raw not in MASK_INPAINT_MODE_VALUES:
        return DEFAULT_MASK_INPAINT_MODE
    return raw
