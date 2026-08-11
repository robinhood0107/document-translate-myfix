from __future__ import annotations

import cv2
import numpy as np

from .contracts import binary_mask
from .stage1 import PageMasks


def composite_positive_result(
    baseline: np.ndarray,
    generated: np.ndarray,
    positive_edit: np.ndarray,
    baseline_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Composite only detector-positive pixels onto an immutable baseline."""

    left = np.asarray(baseline)
    right = np.asarray(generated)
    if left.shape != right.shape:
        raise ValueError("positive composite image shape mismatch")
    edit = binary_mask(positive_edit, left.shape[:2])
    existing = binary_mask(baseline_mask, left.shape[:2])
    candidate = left.copy()
    candidate[edit > 0] = right[edit > 0]
    final_mask = np.where((existing > 0) | (edit > 0), 255, 0).astype(np.uint8)
    return np.ascontiguousarray(candidate), np.ascontiguousarray(final_mask)


def composite_replacement_result(
    original: np.ndarray,
    baseline: np.ndarray,
    generated: np.ndarray,
    replacement_edit: np.ndarray,
    baseline_mask: np.ndarray,
    existing_source_edit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace source-owned edits while retaining non-source baseline edits."""

    source = np.asarray(original)
    prior = np.asarray(baseline)
    replacement = np.asarray(generated)
    if source.shape != prior.shape or source.shape != replacement.shape:
        raise ValueError("replacement composite image shape mismatch")
    shape = source.shape[:2]
    edit = binary_mask(replacement_edit, shape)
    prior_mask = binary_mask(baseline_mask, shape)
    source_edit = binary_mask(existing_source_edit, shape)
    safe_prior = np.where(
        (prior_mask > 0) & (source_edit == 0),
        255,
        0,
    ).astype(np.uint8)
    candidate = source.copy()
    candidate[safe_prior > 0] = prior[safe_prior > 0]
    candidate[edit > 0] = replacement[edit > 0]
    final_mask = cv2.bitwise_or(safe_prior, edit)
    return np.ascontiguousarray(candidate), np.ascontiguousarray(final_mask)


def changed_mask(source: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    left = np.asarray(source)
    right = np.asarray(candidate)
    if left.shape != right.shape:
        raise ValueError("stage2 image shape mismatch")
    changed = np.any(left[:, :, :3] != right[:, :, :3], axis=2)
    return np.where(changed, 255, 0).astype(np.uint8)


def _component_coverages(target: np.ndarray, mask: np.ndarray) -> list[float]:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (target > 0).astype(np.uint8),
        connectivity=8,
    )
    values: list[float] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area > 0:
            x = int(stats[index, cv2.CC_STAT_LEFT])
            y = int(stats[index, cv2.CC_STAT_TOP])
            width = int(stats[index, cv2.CC_STAT_WIDTH])
            height = int(stats[index, cv2.CC_STAT_HEIGHT])
            component = labels[y : y + height, x : x + width] == index
            local_mask = mask[y : y + height, x : x + width] > 0
            values.append(
                float(np.count_nonzero(component & local_mask)) / float(area)
            )
    return values


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image)[:, :, :3], cv2.COLOR_BGR2GRAY)


def residue_score(
    source: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
) -> tuple[float | None, float, int]:
    normalized = binary_mask(target, source.shape[:2])
    source_gray = _gray(source)
    candidate_gray = _gray(candidate)
    source_background = cv2.GaussianBlur(source_gray, (15, 15), 0)
    candidate_background = cv2.GaussianBlur(candidate_gray, (15, 15), 0)
    source_contrast = np.abs(
        source_gray.astype(np.int16) - source_background.astype(np.int16)
    )
    candidate_contrast = np.abs(
        candidate_gray.astype(np.int16) - candidate_background.astype(np.int16)
    )
    residue_source = (normalized > 0) & (source_contrast >= 8)
    source_count = int(np.count_nonzero(residue_source))
    if source_count <= 0:
        return None, 0.0, 0
    ratios = np.minimum(
        candidate_contrast[residue_source].astype(np.float32)
        / np.maximum(source_contrast[residue_source].astype(np.float32), 1.0),
        1.0,
    )
    score = float(np.mean(ratios))
    return score, float(np.sum(ratios)), source_count


def score_stage2_page(
    source: np.ndarray,
    candidate: np.ndarray,
    detector_mask: np.ndarray,
    masks: PageMasks,
    *,
    baseline: np.ndarray | None = None,
) -> tuple[dict[str, object], np.ndarray]:
    detector = binary_mask(detector_mask, source.shape[:2])
    changed = changed_mask(source, candidate)
    target_pixels = int(np.count_nonzero(masks.target))
    target_coverages = _component_coverages(masks.target, detector)
    score, score_sum, score_count = residue_score(
        source,
        candidate,
        masks.target,
    )
    baseline_score = None
    if baseline is not None:
        baseline_score, _baseline_sum, _baseline_count = residue_score(
            source,
            baseline,
            masks.target,
        )
    record = {
        "detector_mask_pixel_count": int(np.count_nonzero(detector)),
        "changed_pixel_count": int(np.count_nonzero(changed)),
        "changed_outside_detector_mask_pixel_count": int(
            np.count_nonzero((changed > 0) & (detector == 0))
        ),
        "target_pixel_count": target_pixels,
        "target_detector_covered_pixel_count": int(
            np.count_nonzero((masks.target > 0) & (detector > 0))
        ),
        "target_detector_coverage": (
            float(np.count_nonzero((masks.target > 0) & (detector > 0)))
            / float(target_pixels)
            if target_pixels
            else None
        ),
        "target_component_coverages": target_coverages,
        "minimum_target_component_coverage": (
            min(target_coverages) if target_coverages else None
        ),
        "protected_changed_pixel_count": int(
            np.count_nonzero((masks.protected > 0) & (changed > 0))
        ),
        "ambiguous_changed_pixel_count": int(
            np.count_nonzero((masks.ambiguous > 0) & (changed > 0))
        ),
        "residue_score": score,
        "residue_score_sum": score_sum,
        "residue_source_contrast_pixel_count": score_count,
        "baseline_residue_score": baseline_score,
        "residue_score_delta_from_baseline": (
            float(score) - float(baseline_score)
            if score is not None and baseline_score is not None
            else None
        ),
    }
    return record, changed
