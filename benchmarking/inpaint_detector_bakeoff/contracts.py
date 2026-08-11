from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Iterable, Mapping

import numpy as np


def binary_mask(mask: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError("detector mask must be a two-dimensional array")
    if shape is not None and tuple(array.shape) != tuple(shape):
        raise ValueError(f"detector mask shape mismatch: {array.shape} != {shape}")
    return np.ascontiguousarray(np.where(array > 0, 255, 0).astype(np.uint8))


def mask_sha256(mask: np.ndarray) -> str:
    normalized = binary_mask(mask)
    digest = hashlib.sha256()
    digest.update(str(normalized.shape).encode("ascii"))
    digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DetectorBox:
    xyxy: tuple[int, int, int, int]
    label: str
    score: float
    provider: str

    def clipped(self, shape: tuple[int, int]) -> "DetectorBox | None":
        height, width = shape
        x1, y1, x2, y2 = self.xyxy
        clipped = (
            max(0, min(width, int(x1))),
            max(0, min(height, int(y1))),
            max(0, min(width, int(x2))),
            max(0, min(height, int(y2))),
        )
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            return None
        return DetectorBox(clipped, self.label, float(self.score), self.provider)


@dataclass(slots=True)
class CandidateMaskResult:
    candidate_id: str
    raw_mask: np.ndarray
    refined_mask: np.ndarray
    dilated_mask: np.ndarray
    boxes: tuple[DetectorBox, ...] = ()
    stage_tensors: dict[str, np.ndarray] = field(default_factory=dict)
    runtime: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw = binary_mask(self.raw_mask)
        shape = raw.shape
        self.raw_mask = raw
        self.refined_mask = binary_mask(self.refined_mask, shape)
        self.dilated_mask = binary_mask(self.dilated_mask, shape)
        self.stage_tensors = {
            str(name): np.ascontiguousarray(np.asarray(value))
            for name, value in self.stage_tensors.items()
        }

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.raw_mask.shape)

    def mask_for_variant(self, variant: str) -> np.ndarray:
        normalized = str(variant).strip().lower()
        if normalized == "raw":
            return self.raw_mask
        if normalized == "refined":
            return self.refined_mask
        if normalized in {"dilated", "dilate3"}:
            return self.dilated_mask
        raise KeyError(f"unknown detector mask variant: {variant}")

    def parity_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "shape": list(self.shape),
            "raw_sha256": mask_sha256(self.raw_mask),
            "refined_sha256": mask_sha256(self.refined_mask),
            "dilated_sha256": mask_sha256(self.dilated_mask),
            "box_count": len(self.boxes),
            "stage_shapes": {
                key: list(value.shape) for key, value in sorted(self.stage_tensors.items())
            },
            "runtime": dict(self.runtime),
        }


@dataclass(frozen=True, slots=True)
class Stage1Page:
    page_id: str
    source_image: str
    target_text_mask: str | None
    protected_structure_mask: str | None
    ambiguous_structure_mask: str | None
    ownership_mask: str | None = None
    claim_seed_mask: str | None = None
    no_edit: bool = False


def union_masks(masks: Iterable[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    union = np.zeros(shape, dtype=np.uint8)
    for mask in masks:
        normalized = binary_mask(mask, shape)
        union[normalized > 0] = 255
    return union


def assert_disjoint_masks(masks: Mapping[str, np.ndarray]) -> None:
    names = list(masks)
    if not names:
        return
    shape = binary_mask(masks[names[0]]).shape
    normalized = {name: binary_mask(mask, shape) for name, mask in masks.items()}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if np.any((normalized[left] > 0) & (normalized[right] > 0)):
                raise ValueError(f"evaluation masks overlap: {left} and {right}")
