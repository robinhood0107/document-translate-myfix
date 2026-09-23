from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable, Iterable, Literal

import cv2
import numpy as np

from .contracts import binary_mask
from .stage2 import composite_positive_result, composite_replacement_result


IncrementalInpaintMode = Literal[
    "additive",
    "context_additive",
    "narrow_replacement",
    "conditional_segmenter",
]

INCREMENTAL_INPAINT_MODES = frozenset(
    {
        "additive",
        "context_additive",
        "narrow_replacement",
        "conditional_segmenter",
    }
)
_REPLACEMENT_MODES = frozenset(
    {"narrow_replacement", "conditional_segmenter"}
)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _readonly_exact_binary_mask(
    mask: np.ndarray,
    *,
    name: str,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Validate a stored plan mask without silently thresholding evidence."""

    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional binary mask")
    if shape is not None and tuple(array.shape) != tuple(shape):
        raise ValueError(f"{name} shape mismatch: {array.shape} != {shape}")
    if array.dtype.kind not in {"b", "i", "u"}:
        raise ValueError(f"{name} must use an integer or boolean binary dtype")
    if array.dtype.kind != "b" and np.any((array != 0) & (array != 255)):
        raise ValueError(f"{name} must contain only 0 and 255")
    normalized = binary_mask(array, shape).copy()
    normalized.setflags(write=False)
    return normalized


def _validated_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
        or any(value <= 0 for value in shape)
    ):
        raise ValueError("page mask shape must contain two positive integers")
    return shape


@dataclass(frozen=True, slots=True)
class IncrementalInpaintPlan:
    """One source-only proposal with separate generation and commit masks.

    ``generation_mask`` is the mask passed to the single page-level fill call.
    ``commit_mask`` is the exact subset allowed to reach the final image.
    ``existing_source_edit`` identifies inherited PR6 source edits for exclusion
    or restoration; it never creates a new edit by itself.
    """

    mode: IncrementalInpaintMode
    generation_mask: np.ndarray
    commit_mask: np.ndarray
    existing_source_edit: np.ndarray

    def __post_init__(self) -> None:
        normalized_mode = str(self.mode).strip().lower()
        if normalized_mode not in INCREMENTAL_INPAINT_MODES:
            raise ValueError(f"unknown incremental inpaint mode: {self.mode}")

        generation = _readonly_exact_binary_mask(
            self.generation_mask,
            name="generation_mask",
        )
        shape = tuple(generation.shape)
        commit = _readonly_exact_binary_mask(
            self.commit_mask,
            name="commit_mask",
            shape=shape,
        )
        existing = _readonly_exact_binary_mask(
            self.existing_source_edit,
            name="existing_source_edit",
            shape=shape,
        )
        if np.any((commit > 0) & (generation <= 0)):
            raise ValueError("commit_mask must be a subset of generation_mask")
        if normalized_mode == "additive" and np.any(
            (generation > 0) & (existing > 0)
        ):
            raise ValueError(
                "additive generation_mask must exclude existing_source_edit"
            )
        if normalized_mode == "context_additive" and np.any(
            (commit > 0) & (existing > 0)
        ):
            raise ValueError(
                "context_additive commit_mask must contain additions only"
            )
        if normalized_mode in _REPLACEMENT_MODES and not np.any(commit):
            if np.any(existing):
                raise ValueError(
                    "empty replacement commit must not restore existing source edits"
                )
        if normalized_mode in _REPLACEMENT_MODES and np.any(existing):
            commit_reach = cv2.dilate(
                commit,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                iterations=1,
            )
            (
                component_count,
                labels,
                stats,
                _centroids,
            ) = cv2.connectedComponentsWithStats(
                (existing > 0).astype(np.uint8), connectivity=8
            )
            for component_id in range(1, component_count):
                x, y, width, height, _area = stats[component_id]
                component_labels = labels[y : y + height, x : x + width]
                component_reach = commit_reach[y : y + height, x : x + width]
                if not np.any(
                    (component_labels == component_id) & (component_reach > 0)
                ):
                    raise ValueError(
                        "replacement source component must contact commit_mask"
                    )

        object.__setattr__(self, "mode", normalized_mode)
        object.__setattr__(self, "generation_mask", generation)
        object.__setattr__(self, "commit_mask", commit)
        object.__setattr__(self, "existing_source_edit", existing)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.generation_mask.shape)

    @property
    def replaces_existing_source_edit(self) -> bool:
        return self.mode in _REPLACEMENT_MODES

    @property
    def lama_request_count(self) -> int:
        return int(np.any(self.commit_mask))


@dataclass(frozen=True, slots=True)
class PageIncrementalInpaintPlan:
    """Union of all per-region plans for one page and one optional fill call."""

    plans: tuple[IncrementalInpaintPlan, ...]
    generation_mask: np.ndarray
    commit_mask: np.ndarray
    existing_source_edit: np.ndarray
    replacement_source_edit: np.ndarray

    def __post_init__(self) -> None:
        generation = _readonly_exact_binary_mask(
            self.generation_mask,
            name="page generation_mask",
        )
        shape = tuple(generation.shape)
        commit = _readonly_exact_binary_mask(
            self.commit_mask,
            name="page commit_mask",
            shape=shape,
        )
        existing = _readonly_exact_binary_mask(
            self.existing_source_edit,
            name="page existing_source_edit",
            shape=shape,
        )
        replacement = _readonly_exact_binary_mask(
            self.replacement_source_edit,
            name="page replacement_source_edit",
            shape=shape,
        )
        if np.any((commit > 0) & (generation <= 0)):
            raise ValueError("page commit_mask must be a subset of generation_mask")
        if np.any((replacement > 0) & (existing <= 0)):
            raise ValueError(
                "page replacement_source_edit must be a subset of existing_source_edit"
            )
        rows = tuple(self.plans)
        for plan in rows:
            if plan.shape != shape:
                raise ValueError("page plan shape mismatch")
        expected_generation = np.zeros(shape, dtype=np.uint8)
        expected_commit = np.zeros(shape, dtype=np.uint8)
        expected_existing = np.zeros(shape, dtype=np.uint8)
        expected_replacement = np.zeros(shape, dtype=np.uint8)
        for plan in rows:
            np.bitwise_or(
                expected_generation,
                plan.generation_mask,
                out=expected_generation,
            )
            np.bitwise_or(expected_commit, plan.commit_mask, out=expected_commit)
            np.bitwise_or(
                expected_existing,
                plan.existing_source_edit,
                out=expected_existing,
            )
            if plan.replaces_existing_source_edit:
                np.bitwise_or(
                    expected_replacement,
                    plan.existing_source_edit,
                    out=expected_replacement,
                )
        for name, actual, expected in (
            ("generation_mask", generation, expected_generation),
            ("commit_mask", commit, expected_commit),
            ("existing_source_edit", existing, expected_existing),
            ("replacement_source_edit", replacement, expected_replacement),
        ):
            if not np.array_equal(actual, expected):
                raise ValueError(f"page {name} must equal its per-plan union")

        object.__setattr__(self, "plans", rows)
        object.__setattr__(self, "generation_mask", generation)
        object.__setattr__(self, "commit_mask", commit)
        object.__setattr__(self, "existing_source_edit", existing)
        object.__setattr__(self, "replacement_source_edit", replacement)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.generation_mask.shape)

    @property
    def lama_request_count(self) -> int:
        """A page either needs no generated image or exactly one union request."""

        return int(np.any(self.commit_mask))

    def lama_request_masks(self) -> tuple[np.ndarray, ...]:
        if self.lama_request_count == 0:
            return ()
        return (self.generation_mask,)


@dataclass(frozen=True, slots=True)
class IncrementalInpaintReceipt:
    provider: str
    request_count: int
    source_sha256: str
    generation_mask_sha256: str
    generated_sha256: str | None

    def __post_init__(self) -> None:
        provider = str(self.provider).strip()
        if not provider:
            raise ValueError("incremental inpaint receipt provider is required")
        if self.request_count not in {0, 1}:
            raise ValueError("incremental inpaint receipt request_count must be 0 or 1")
        hashes = (self.source_sha256, self.generation_mask_sha256)
        if any(len(value) != 64 for value in hashes):
            raise ValueError("incremental inpaint receipt requires SHA-256 values")
        if self.request_count == 0 and self.generated_sha256 is not None:
            raise ValueError("empty incremental receipt cannot bind generated pixels")
        if self.request_count == 1 and (
            self.generated_sha256 is None or len(self.generated_sha256) != 64
        ):
            raise ValueError("executed incremental receipt must bind generated pixels")
        object.__setattr__(self, "provider", provider)


def execute_incremental_inpaint(
    source: np.ndarray,
    plan: PageIncrementalInpaintPlan,
    inpaint: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    provider: str,
) -> tuple[np.ndarray | None, IncrementalInpaintReceipt]:
    """Execute the exact page union zero or one time and bind a receipt."""

    source_array = np.asarray(source)
    if source_array.ndim != 3 or source_array.shape[:2] != plan.shape:
        raise ValueError("incremental execution source/plan shape mismatch")
    if source_array.dtype != np.uint8:
        raise ValueError("incremental execution source must use uint8 pixels")
    request_count = plan.lama_request_count
    source_hash = _array_sha256(source_array)
    generation_hash = _array_sha256(plan.generation_mask)
    if request_count == 0:
        return None, IncrementalInpaintReceipt(
            provider=provider,
            request_count=0,
            source_sha256=source_hash,
            generation_mask_sha256=generation_hash,
            generated_sha256=None,
        )
    generated = np.asarray(inpaint(source_array, plan.generation_mask))
    if generated.shape != source_array.shape or generated.dtype != source_array.dtype:
        raise ValueError("incremental inpaint callback returned incompatible pixels")
    generated = np.ascontiguousarray(generated)
    return generated, IncrementalInpaintReceipt(
        provider=provider,
        request_count=1,
        source_sha256=source_hash,
        generation_mask_sha256=generation_hash,
        generated_sha256=_array_sha256(generated),
    )


def union_incremental_plans(
    plans: Iterable[IncrementalInpaintPlan],
    *,
    shape: tuple[int, int] | None = None,
) -> PageIncrementalInpaintPlan:
    """Union region plans without creating one fill request per region."""

    rows = tuple(plans)
    if rows:
        resolved_shape = rows[0].shape
        if shape is not None and _validated_shape(shape) != resolved_shape:
            raise ValueError("explicit page mask shape does not match plan shape")
        if any(row.shape != resolved_shape for row in rows):
            raise ValueError("incremental plans must share one exact page shape")
    elif shape is not None:
        resolved_shape = _validated_shape(shape)
    else:
        raise ValueError("empty incremental plan union requires an explicit shape")

    generation = np.zeros(resolved_shape, dtype=np.uint8)
    commit = np.zeros(resolved_shape, dtype=np.uint8)
    existing = np.zeros(resolved_shape, dtype=np.uint8)
    replacement = np.zeros(resolved_shape, dtype=np.uint8)
    for row in rows:
        np.bitwise_or(generation, row.generation_mask, out=generation)
        np.bitwise_or(commit, row.commit_mask, out=commit)
        np.bitwise_or(existing, row.existing_source_edit, out=existing)
        if row.replaces_existing_source_edit:
            np.bitwise_or(
                replacement,
                row.existing_source_edit,
                out=replacement,
            )

    for row in rows:
        if row.mode == "additive" and np.any(
            (row.generation_mask > 0) & (existing > 0)
        ):
            raise ValueError(
                "page additive generation_mask must exclude all existing source edits"
            )
        if row.mode == "context_additive" and np.any(
            (row.commit_mask > 0) & (existing > 0)
        ):
            raise ValueError(
                "page context_additive commit_mask must contain additions only"
            )

    return PageIncrementalInpaintPlan(
        plans=rows,
        generation_mask=generation,
        commit_mask=commit,
        existing_source_edit=existing,
        replacement_source_edit=replacement,
    )


def composite_incremental_result(
    original: np.ndarray,
    baseline: np.ndarray,
    generated: np.ndarray | None,
    baseline_mask: np.ndarray,
    plan: PageIncrementalInpaintPlan,
    receipt: IncrementalInpaintReceipt,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one page result under additive or replacement exact-mask rules."""

    source = np.asarray(original)
    prior = np.asarray(baseline)
    if source.ndim != 3 or source.shape != prior.shape:
        raise ValueError("incremental composite source/baseline shape mismatch")
    if source.shape[:2] != plan.shape:
        raise ValueError("incremental composite image/plan shape mismatch")
    if source.dtype != prior.dtype:
        raise ValueError("incremental composite source/baseline dtype mismatch")
    prior_mask = _readonly_exact_binary_mask(
        baseline_mask,
        name="baseline_mask",
        shape=plan.shape,
    )
    if np.any((plan.replacement_source_edit > 0) & (prior_mask <= 0)):
        raise ValueError(
            "replacement_source_edit must be a subset of baseline_mask"
        )
    if receipt.request_count != plan.lama_request_count:
        raise ValueError("incremental receipt request count differs from the plan")
    if receipt.source_sha256 != _array_sha256(source):
        raise ValueError("incremental receipt source SHA differs")
    if receipt.generation_mask_sha256 != _array_sha256(plan.generation_mask):
        raise ValueError("incremental receipt generation-mask SHA differs")

    if plan.lama_request_count == 0:
        if np.any(plan.replacement_source_edit):
            raise AssertionError(
                "empty incremental plan cannot contain replacement source edits"
            )
        return np.ascontiguousarray(prior.copy()), np.ascontiguousarray(
            prior_mask.copy()
        )
    else:
        if generated is None:
            raise ValueError("incremental composite requires the generated page image")
        fill_result = np.asarray(generated)
        if fill_result.shape != source.shape:
            raise ValueError("incremental composite generated image shape mismatch")
        if fill_result.dtype != source.dtype:
            raise ValueError("incremental composite generated image dtype mismatch")
        if receipt.generated_sha256 != _array_sha256(fill_result):
            raise ValueError("incremental receipt generated-image SHA differs")

    if np.any(plan.replacement_source_edit):
        return composite_replacement_result(
            source,
            prior,
            fill_result,
            plan.commit_mask,
            prior_mask,
            plan.replacement_source_edit,
        )
    return composite_positive_result(
        prior,
        fill_result,
        plan.commit_mask,
        prior_mask,
    )


__all__ = [
    "INCREMENTAL_INPAINT_MODES",
    "IncrementalInpaintMode",
    "IncrementalInpaintPlan",
    "IncrementalInpaintReceipt",
    "PageIncrementalInpaintPlan",
    "composite_incremental_result",
    "execute_incremental_inpaint",
    "union_incremental_plans",
]
