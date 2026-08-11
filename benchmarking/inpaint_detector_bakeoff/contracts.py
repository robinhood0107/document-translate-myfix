from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Iterable, Literal, Mapping

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


def tensor_sha256(tensor: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(tensor))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes(order="C"))
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
            "stage_dtypes": {
                key: str(value.dtype) for key, value in sorted(self.stage_tensors.items())
            },
            "stage_sha256": {
                key: tensor_sha256(value)
                for key, value in sorted(self.stage_tensors.items())
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
    target_instances: tuple["TargetInstance", ...] = ()
    bubble_route_class: str | None = None
    bubble_interior_mask: str | None = None
    corner_protect_mask: str | None = None
    expected_edit: str = "required"


@dataclass(frozen=True, slots=True)
class TargetInstance:
    instance_id: str
    mask_path: str


ROLE_NAMES = frozenset({"seed", "ownership", "silhouette", "router", "expansion", "fill"})
ROLE_STATES = frozenset(
    {
        "active",
        "family_complete",
        "pareto",
        "dominated",
        "information_limited",
        "blocked_asset",
    }
)
ROUTE_DECISIONS = frozenset({"narrow", "broad", "skip"})
BUBBLE_ROUTE_CLASSES = frozenset(
    {"clean_flat", "clean_gradient", "texture", "line_art", "ambiguous"}
)


@dataclass(frozen=True, slots=True)
class RoleCandidateSpec:
    """Immutable identity for one provider x role x output variant."""

    candidate_id: str
    provider: str
    role: str
    variant: str
    code_commit: str
    model_sha256: str
    runtime_provider: str
    preprocessing_contract_sha256: str
    status: str = "active"

    def __post_init__(self) -> None:
        values = {
            "candidate_id": self.candidate_id,
            "provider": self.provider,
            "role": self.role,
            "variant": self.variant,
            "code_commit": self.code_commit,
            "model_sha256": self.model_sha256,
            "runtime_provider": self.runtime_provider,
            "preprocessing_contract_sha256": self.preprocessing_contract_sha256,
        }
        empty = [name for name, value in values.items() if not str(value).strip()]
        if empty:
            raise ValueError("role candidate contains empty fields: " + ", ".join(empty))
        if self.role not in ROLE_NAMES:
            raise ValueError(f"unknown candidate role: {self.role}")
        if self.status not in ROLE_STATES:
            raise ValueError(f"unknown candidate state: {self.status}")

    def cache_payload(self, source_sha256: str) -> dict[str, str]:
        source = str(source_sha256).strip().lower()
        if not source:
            raise ValueError("source SHA is required for the detector cache")
        return {
            "source_sha256": source,
            "code_commit": self.code_commit,
            "model_sha256": self.model_sha256.lower(),
            "runtime_provider": self.runtime_provider,
            "preprocessing_contract_sha256": self.preprocessing_contract_sha256.lower(),
            "output_variant": self.variant,
        }

    def cache_key(self, source_sha256: str) -> str:
        payload = self.cache_payload(source_sha256)
        digest = hashlib.sha256()
        for key, value in sorted(payload.items()):
            digest.update(key.encode("utf-8"))
            digest.update(b"\0")
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()


@dataclass(slots=True)
class BubbleRouteDecision:
    router_id: str
    decision: Literal["narrow", "broad", "skip"]
    edit_mask: np.ndarray
    interior_mask: np.ndarray | None = None
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.router_id).strip():
            raise ValueError("router id must not be empty")
        if self.decision not in ROUTE_DECISIONS:
            raise ValueError(f"unknown route decision: {self.decision}")
        self.edit_mask = binary_mask(self.edit_mask)
        if self.interior_mask is not None:
            self.interior_mask = binary_mask(self.interior_mask, self.edit_mask.shape)
        self.reasons = tuple(str(value) for value in self.reasons)


@dataclass(frozen=True, slots=True)
class FactorizedRunRecord:
    run_id: str
    detector_id: str
    ownership_id: str
    silhouette_id: str
    router_id: str
    expansion_id: str
    fill_id: str
    oracle_only: bool
    status: str
    metrics: Mapping[str, object]
    closure_reason: str = ""

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("factorized run id must not be empty")
        if self.status not in ROLE_STATES:
            raise ValueError(f"unknown factorized run state: {self.status}")
        if self.oracle_only and self.status == "pareto":
            raise ValueError("oracle-only results cannot be product finalists")

    def as_record(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "detector_id": self.detector_id,
            "ownership_id": self.ownership_id,
            "silhouette_id": self.silhouette_id,
            "router_id": self.router_id,
            "expansion_id": self.expansion_id,
            "fill_id": self.fill_id,
            "oracle_only": self.oracle_only,
            "status": self.status,
            "metrics": dict(self.metrics),
            "closure_reason": self.closure_reason,
        }


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
