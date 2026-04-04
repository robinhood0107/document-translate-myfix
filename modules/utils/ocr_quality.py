from __future__ import annotations

from typing import Iterable


def _compact_text(text: str) -> str:
    return "".join((text or "").split())


def summarize_ocr_quality(blocks: Iterable) -> dict:
    block_list = list(blocks or [])
    block_count = len(block_list)
    non_empty = 0
    single_char_like = 0

    for blk in block_list:
        compact = _compact_text(getattr(blk, "text", "") or "")
        if not compact:
            continue
        non_empty += 1
        if len(compact) <= 1:
            single_char_like += 1

    empty = max(0, block_count - non_empty)
    quality = {
        "block_count": block_count,
        "non_empty": non_empty,
        "empty": empty,
        "single_char_like": single_char_like,
    }
    low_quality, reason = is_low_quality_ocr(quality)
    quality["low_quality"] = low_quality
    quality["reason"] = reason
    return quality


def is_low_quality_ocr(quality: dict) -> tuple[bool, str]:
    block_count = int(quality.get("block_count", 0) or 0)
    non_empty = int(quality.get("non_empty", 0) or 0)
    empty = int(quality.get("empty", 0) or 0)
    single_char_like = int(quality.get("single_char_like", 0) or 0)

    if block_count <= 0:
        return True, "No text blocks available for OCR."
    if non_empty <= 0:
        return True, "OCR returned no text for any detected block."
    if block_count >= 3 and (non_empty / max(block_count, 1)) < 0.5:
        return True, f"OCR filled only {non_empty}/{block_count} detected blocks."
    if non_empty >= 3 and (single_char_like / max(non_empty, 1)) >= 0.7:
        return True, f"OCR returned too many suspiciously short results ({single_char_like}/{non_empty})."
    if block_count >= 5 and empty >= 4:
        return True, f"OCR left too many blocks empty ({empty}/{block_count})."
    return False, ""
