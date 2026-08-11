from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .contracts import DetectorBox, binary_mask


@dataclass(frozen=True, slots=True)
class ProvenanceFusionResult:
    ownership: np.ndarray
    positive_claim: np.ndarray
    positive_edit: np.ndarray
    selected_raw_text_boxes: tuple[DetectorBox, ...]


def build_provenance_fusion(
    raw_pixel_claim: np.ndarray,
    *,
    required_skip_prior: np.ndarray,
    required_skip_seed: np.ndarray,
    content_component_ownership: np.ndarray,
    raw_detector_boxes: Iterable[DetectorBox],
    existing_edit: np.ndarray,
    structure_protect: np.ndarray,
    ambiguous_protect: np.ndarray,
) -> ProvenanceFusionResult:
    """Fuse block provenance without turning a detector box into pixel claim.

    A directly detected RT-DETR text box may widen ownership for a required
    conservative skip. A bubble-rescue block, which has no raw text box, stays
    restricted to its existing content components. In both cases every output
    claim pixel must still originate in the raw pixel detector mask.
    """

    claim = binary_mask(raw_pixel_claim)
    shape = claim.shape
    prior = binary_mask(required_skip_prior, shape)
    seed = binary_mask(required_skip_seed, shape)
    content = binary_mask(content_component_ownership, shape)
    existing = binary_mask(existing_edit, shape)
    structure = binary_mask(structure_protect, shape)
    ambiguous = binary_mask(ambiguous_protect, shape)

    raw_text_ownership = np.zeros(shape, dtype=np.uint8)
    selected: list[DetectorBox] = []
    for original in raw_detector_boxes:
        if original.label == "bubble":
            continue
        box = original.clipped(shape)
        if box is None:
            continue
        x1, y1, x2, y2 = box.xyxy
        if not np.any(seed[y1:y2, x1:x2] > 0):
            continue
        raw_text_ownership[y1:y2, x1:x2] = 255
        selected.append(box)

    ownership = np.where(
        (prior > 0) & ((content > 0) | (raw_text_ownership > 0)),
        255,
        0,
    ).astype(np.uint8)
    positive_claim = np.where((claim > 0) & (ownership > 0), 255, 0).astype(np.uint8)
    positive_edit = np.where(
        (positive_claim > 0)
        & (existing == 0)
        & (structure == 0)
        & (ambiguous == 0),
        255,
        0,
    ).astype(np.uint8)
    return ProvenanceFusionResult(
        ownership=np.ascontiguousarray(ownership),
        positive_claim=np.ascontiguousarray(positive_claim),
        positive_edit=np.ascontiguousarray(positive_edit),
        selected_raw_text_boxes=tuple(selected),
    )
