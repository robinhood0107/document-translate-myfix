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
    core = (
        (delta >= delta_threshold)
        & (source_local_i16 >= 8)
        & (source_local_i16 >= paired_local_i16 + 4)
    )

    neighborhood = cv2.dilate(
        core.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2
    )
    extent = (
        (neighborhood > 0)
        & (delta >= max(6, delta_threshold // 3))
        & (source_local_i16 >= 3)
        & (source_local_i16 + 2 >= paired_local_i16)
    ).astype(np.uint8)
    extent = cv2.morphologyEx(
        extent, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    kept = np.zeros(extent.shape, np.uint8)
    for component in _binary_components(extent, minimum_area=4):
        kept[component > 0] = 255

    grouped = cv2.dilate(
        (kept > 0).astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1
    )
    instances: list[np.ndarray] = []
    for group in _binary_components(grouped, minimum_area=16):
        instance = np.where((group > 0) & (kept > 0), 255, 0).astype(np.uint8)
        if np.count_nonzero(instance) >= 4:
            instances.append(instance)
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
