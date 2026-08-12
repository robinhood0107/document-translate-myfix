from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class PairedTargetProposal:
    core_mask: np.ndarray
    extent_mask: np.ndarray
    instance_masks: tuple[np.ndarray, ...]
    delta_threshold: int
    delta_median: float
    delta_mad: float


@dataclass(frozen=True, slots=True)
class SourceExtentFeatures:
    contrast: np.ndarray
    edge_support: np.ndarray


def build_source_extent_features(source_bgr: np.ndarray) -> SourceExtentFeatures:
    source = np.asarray(source_bgr)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("source must be a BGR image")
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    local_background = cv2.medianBlur(gray, 15)
    contrast = cv2.absdiff(gray, local_background)
    edge_support = cv2.dilate(
        cv2.Canny(gray, 40, 120), np.ones((3, 3), np.uint8), iterations=1
    )
    return SourceExtentFeatures(contrast=contrast, edge_support=edge_support)


def source_extent_variants(
    source_bgr: np.ndarray,
    location_seed: np.ndarray,
    *,
    features: SourceExtentFeatures | None = None,
) -> dict[str, np.ndarray]:
    """Build candidate-blind annotation extents around an independent seed.

    These masks are review aids, never detector outputs.  They use only the
    source pixels and a bounded location seed.  Review must explicitly choose a
    variant (or reject all variants) before it can become target annotation.
    """

    source = np.asarray(source_bgr)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("source must be a BGR image")
    seed = np.where(np.asarray(location_seed) > 0, 255, 0).astype(np.uint8)
    if seed.shape != source.shape[:2]:
        raise ValueError("location seed shape mismatch")
    if not np.any(seed):
        empty = np.zeros(seed.shape, np.uint8)
        return {
            key: empty.copy()
            for key in (
                "strict",
                "balanced",
                "edge_supported",
                "location_dilate1",
                "location_dilate2",
            )
        }

    prepared = features or build_source_extent_features(source)
    contrast = np.asarray(prepared.contrast)
    edge_support = np.asarray(prepared.edge_support)
    if contrast.shape != seed.shape or edge_support.shape != seed.shape:
        raise ValueError("source extent feature shape mismatch")
    neighborhood = cv2.dilate(seed, np.ones((9, 9), np.uint8), iterations=1)

    def connected_support(support: np.ndarray) -> np.ndarray:
        candidate = np.where(
            (neighborhood > 0) & ((support > 0) | (seed > 0)), 255, 0
        ).astype(np.uint8)
        count, labels = cv2.connectedComponents((candidate > 0).astype(np.uint8), 8)
        if count <= 1:
            return candidate
        selected = np.unique(labels[seed > 0])
        selected = selected[selected > 0]
        if selected.size == 0:
            return np.zeros(seed.shape, np.uint8)
        return np.where(np.isin(labels, selected), 255, 0).astype(np.uint8)

    return {
        "strict": connected_support(np.where(contrast >= 12, 255, 0).astype(np.uint8)),
        "balanced": connected_support(np.where(contrast >= 7, 255, 0).astype(np.uint8)),
        "edge_supported": connected_support(
            np.where((contrast >= 4) & (edge_support > 0), 255, 0).astype(np.uint8)
        ),
        # Annotation-only extents. They do not inspect any remote source pixel,
        # so they cannot bridge from a paired location cue into unrelated art.
        "location_dilate1": cv2.dilate(
            seed, np.ones((3, 3), np.uint8), iterations=1
        ),
        "location_dilate2": cv2.dilate(
            seed, np.ones((5, 5), np.uint8), iterations=1
        ),
    }


def _binary_components(mask: np.ndarray, *, minimum_area: int) -> list[np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8, cv2.CV_32S
    )
    output: list[np.ndarray] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        component = np.zeros(mask.shape, np.uint8)
        component[labels == label] = 255
        output.append(component)
    return output


def paired_old_text_proposal(
    source_bgr: np.ndarray,
    paired_bgr: np.ndarray,
) -> PairedTargetProposal:
    """Propose source-only strokes removed by a human-edited paired page.

    The paired page is only a location proposal. Its pixels are never returned as
    a clean background or candidate image. A source pixel must both change in the
    pair and carry more local contrast in the source than in the paired page.
    """

    source = np.asarray(source_bgr)
    paired = np.asarray(paired_bgr)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("source must be a BGR image")
    if paired.shape != source.shape:
        raise ValueError("paired proposal image shape mismatch")

    delta = np.max(
        np.abs(source.astype(np.int16) - paired.astype(np.int16)), axis=2
    )
    delta_median = float(np.median(delta))
    delta_mad = float(np.median(np.abs(delta.astype(np.float32) - delta_median)))
    delta_threshold = max(
        20,
        int(np.ceil(delta_median + 8.0 * max(1.0, delta_mad))),
    )

    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    paired_gray = cv2.cvtColor(paired, cv2.COLOR_BGR2GRAY)
    source_local = cv2.absdiff(
        source_gray, cv2.GaussianBlur(source_gray, (0, 0), 3.0)
    )
    paired_local = cv2.absdiff(
        paired_gray, cv2.GaussianBlur(paired_gray, (0, 0), 3.0)
    )
    source_local_i16 = source_local.astype(np.int16)
    paired_local_i16 = paired_local.astype(np.int16)
    source_edges = cv2.Canny(source_gray, 40, 120) > 0
    paired_edges = cv2.Canny(paired_gray, 40, 120) > 0
    paired_edge_distance = cv2.distanceTransform(
        np.where(paired_edges, 0, 1).astype(np.uint8), cv2.DIST_L2, 3
    )
    source_only_edges = source_edges & (paired_edge_distance >= 1.5)
    core = (
        (delta >= delta_threshold)
        & source_only_edges
        & (source_local_i16 >= 8)
        & (source_local_i16 >= paired_local_i16 + 2)
    )

    neighborhood = cv2.dilate(
        core.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1
    )
    source_stroke_support = cv2.dilate(
        source_edges.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
    )
    extent = (
        (neighborhood > 0)
        & (source_stroke_support > 0)
        & (delta >= max(6, delta_threshold // 3))
        & (source_local_i16 >= 3)
        & (source_local_i16 + 3 >= paired_local_i16)
    ).astype(np.uint8)
    extent = cv2.morphologyEx(
        extent, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    kept = np.zeros(extent.shape, np.uint8)
    for component in _binary_components(extent, minimum_area=4):
        kept[component > 0] = 255

    grouped = cv2.dilate(
        (kept > 0).astype(np.uint8), np.ones((11, 11), np.uint8), iterations=1
    )
    instances: list[np.ndarray] = []
    for group in _binary_components(grouped, minimum_area=16):
        instance = np.where((group > 0) & (kept > 0), 255, 0).astype(np.uint8)
        pixels = int(np.count_nonzero(instance))
        if pixels < 8:
            continue
        x, y, width, height = cv2.boundingRect((instance > 0).astype(np.uint8))
        if width > source.shape[1] // 2 or height > source.shape[0] // 2:
            continue
        aspect = float(max(width, height)) / float(max(1, min(width, height)))
        if aspect > 14.0 and pixels < 64:
            continue
        instances.append(instance)
    kept = np.zeros(extent.shape, np.uint8)
    for instance in instances:
        kept[instance > 0] = 255
    instances.sort(
        key=lambda value: tuple(
            int(item)
            for item in cv2.boundingRect((value > 0).astype(np.uint8))[:2][::-1]
        )
    )
    return PairedTargetProposal(
        core_mask=np.where(core, 255, 0).astype(np.uint8),
        extent_mask=kept,
        instance_masks=tuple(instances),
        delta_threshold=delta_threshold,
        delta_median=delta_median,
        delta_mad=delta_mad,
    )
