from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ERASE_MODE_TEXT_FREE_LAMA = "text_free_lama"
ERASE_MODE_BUBBLE_FLAT_FILL = "bubble_flat_fill"
ERASE_MODE_BUBBLE_TELEA = "bubble_telea"
ERASE_MODE_BUBBLE_LAMA_FALLBACK = "bubble_lama_fallback"
ERASE_MODE_BUBBLE_SKIPPED = "bubble_skipped"


@dataclass(slots=True)
class BubbleEraseBlockStats:
    mode: str
    edit_pixel_count: int = 0
    protect_pixel_count: int = 0
    skipped_reason: str = ""


def set_block_erase_metadata(block, stats: BubbleEraseBlockStats) -> None:
    block._erase_mode = str(stats.mode or "")
    block._erase_edit_pixel_count = int(stats.edit_pixel_count or 0)
    block._erase_protect_pixel_count = int(stats.protect_pixel_count or 0)
    block._erase_skipped_reason = str(stats.skipped_reason or "")


def mask_pixel_count(mask: np.ndarray | None) -> int:
    if mask is None:
        return 0
    return int(np.count_nonzero(np.asarray(mask) > 0))
