from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modules.utils.inpaint_composite import normalize_edit_mask


@dataclass(slots=True)
class MaskPatch:
    """A binary mask stored only for one half-open page ROI."""

    xyxy: tuple[int, int, int, int]
    mask: np.ndarray

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = [int(value) for value in self.xyxy]
        if x2 <= x1 or y2 <= y1:
            raise ValueError("inpaint_evidence_roi_invalid")
        normalized = normalize_edit_mask(
            self.mask,
            (y2 - y1, x2 - x1),
        )
        if normalized.shape != (y2 - y1, x2 - x1):
            raise ValueError("inpaint_evidence_mask_shape_invalid")
        self.xyxy = (x1, y1, x2, y2)
        self.mask = np.ascontiguousarray(normalized.copy())

    @property
    def pixel_count(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def roi(self) -> tuple[int, int, int, int]:
        """Compatibility alias for the evidence-first contract name."""

        return self.xyxy

    @property
    def local_mask(self) -> np.ndarray:
        """Compatibility alias for the sparse ROI-local mask payload."""

        return self.mask


@dataclass(slots=True)
class BlockInpaintEvidence:
    block_id: str
    block_index: int | None
    erase_mode: str = ""
    skipped_reason: str = ""
    source_raw_owned: MaskPatch | None = None
    source_owned: MaskPatch | None = None
    structure_protect: MaskPatch | None = None
    ownership_protect: MaskPatch | None = None
    positive_claim: MaskPatch | None = None
    positive_edit: MaskPatch | None = None
    claim_providers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.block_id = str(self.block_id or "")
        if self.block_index is not None:
            self.block_index = int(self.block_index)
        self.erase_mode = str(self.erase_mode or "")
        self.skipped_reason = str(self.skipped_reason or "")
        self.claim_providers = tuple(
            dict.fromkeys(
                str(provider).strip()
                for provider in self.claim_providers
                if str(provider).strip()
            )
        )


@dataclass(slots=True)
class SourceLamaBlockwiseResult:
    image: np.ndarray
    edit_mask: np.ndarray | None
    diagnostics: list[dict]
    evidence: tuple[BlockInpaintEvidence, ...] = ()


def mask_patch_from_page_mask(
    page_mask: np.ndarray | None,
    xyxy: tuple[int, int, int, int] | list[int] | None,
    image_shape: tuple[int, ...],
) -> MaskPatch | None:
    if page_mask is None or xyxy is None:
        return None
    height, width = image_shape[:2]
    x1, y1, x2, y2 = [int(value) for value in xyxy]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    normalized = normalize_edit_mask(page_mask, image_shape)
    return MaskPatch(
        xyxy=(x1, y1, x2, y2),
        mask=normalized[y1:y2, x1:x2],
    )


def combine_evidence_patches(
    evidence: tuple[BlockInpaintEvidence, ...] | list[BlockInpaintEvidence] | None,
    field_name: str,
    image_shape: tuple[int, ...],
) -> np.ndarray:
    combined = np.zeros(image_shape[:2], dtype=np.uint8)
    for item in tuple(evidence or ()):
        patch = getattr(item, field_name, None)
        if patch is None:
            continue
        x1, y1, x2, y2 = patch.xyxy
        if x2 > combined.shape[1] or y2 > combined.shape[0]:
            raise ValueError("inpaint_evidence_roi_out_of_bounds")
        combined[y1:y2, x1:x2] = np.where(
            (combined[y1:y2, x1:x2] > 0) | (patch.mask > 0),
            255,
            0,
        ).astype(np.uint8)
    return combined
