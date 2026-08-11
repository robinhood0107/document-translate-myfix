from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from modules.utils.bubble_silhouette import extract_bubble_interior_cap_crop
from modules.utils.inpaint_composite import (
    composite_with_edit_mask,
    count_changed_outside_edit_mask,
    normalize_edit_mask,
)
from modules.utils.mask_roi import (
    build_text_prior_mask,
    normalize_xyxy,
    resolve_inpaint_text_xyxy,
)


ERASE_MODE_TEXT_FREE_LAMA = "text_free_lama"
ERASE_MODE_BUBBLE_FLAT_FILL = "bubble_flat_fill"
ERASE_MODE_BUBBLE_GRADIENT_FILL = "bubble_gradient_fill"
ERASE_MODE_BUBBLE_TELEA = "bubble_telea"
ERASE_MODE_BUBBLE_LAMA_FALLBACK = "bubble_lama_fallback"
ERASE_MODE_BUBBLE_SKIPPED = "bubble_skipped"


@dataclass(slots=True)
class BubbleEraseBlockStats:
    mode: str
    edit_pixel_count: int = 0
    protect_pixel_count: int = 0
    skipped_reason: str = ""


@dataclass(slots=True)
class BubbleEraseResult:
    image: np.ndarray
    edit_mask: np.ndarray
    fallback_mask: np.ndarray
    expanded_bubble_mask: np.ndarray
    stats: dict


@dataclass(slots=True)
class BubbleLineArtContext:
    interior_cap: np.ndarray | None
    source_seed_mask: np.ndarray
    source_glyph_mask: np.ndarray
    source_residual_seed_mask: np.ndarray
    line_protect_mask: np.ndarray
    texture_field_detected: bool
    ambiguous_structure_near_source: bool


def set_block_erase_metadata(block, stats: BubbleEraseBlockStats) -> None:
    block._erase_mode = str(stats.mode or "")
    block._erase_edit_pixel_count = int(stats.edit_pixel_count or 0)
    block._erase_protect_pixel_count = int(stats.protect_pixel_count or 0)
    block._erase_skipped_reason = str(stats.skipped_reason or "")


def mask_pixel_count(mask: np.ndarray | None) -> int:
    if mask is None:
        return 0
    return int(np.count_nonzero(np.asarray(mask) > 0))


def _bubble_border_protect_mask(shape: tuple[int, int], width: int = 3) -> np.ndarray:
    h, w = shape
    protect = np.zeros((h, w), dtype=np.uint8)
    if h <= 0 or w <= 0:
        return protect
    cv2.rectangle(protect, (0, 0), (w - 1, h - 1), 255, thickness=max(1, int(width)))
    return protect


def _extract_bubble_interior_cap_mask(
    crop: np.ndarray,
    seed_mask: np.ndarray,
    *,
    min_seed_coverage: float,
    max_area_ratio: float,
) -> np.ndarray | None:
    if crop.size == 0 or seed_mask.size == 0:
        return None
    seed = np.asarray(seed_mask)
    if seed.ndim == 3:
        seed = seed[:, :, 0]
    if seed.shape[:2] != crop.shape[:2]:
        return None
    seed = np.where(seed > 0, 255, 0).astype(np.uint8)
    seed_pixel_count = int(np.count_nonzero(seed))
    if seed_pixel_count <= 0:
        return None
    cap = extract_bubble_interior_cap_crop(
        crop,
        seed,
        erode_px=1,
        min_area_ratio=0.20,
        max_area_ratio=float(max_area_ratio),
        min_seed_coverage=float(min_seed_coverage),
        preserve_seed_after_erode=False,
        erode_below_area_ratio=0.995,
        erode_shape=cv2.MORPH_RECT,
    )
    if cap is None:
        return None
    cap_array = np.asarray(cap)
    if cap_array.ndim == 3:
        cap_array = cap_array[:, :, 0]
    if cap_array.shape[:2] != seed.shape[:2]:
        return None
    cap_mask = np.where(cap_array > 0, 255, 0).astype(np.uint8)
    if not np.any(cap_mask):
        return None
    post_erode_seed_coverage = float(
        np.count_nonzero((cap_mask > 0) & (seed > 0))
    ) / float(seed_pixel_count)
    if post_erode_seed_coverage < float(min_seed_coverage):
        return None
    return cap_mask


def _validated_bubble_interior_cap_mask(
    crop: np.ndarray,
    seed_mask: np.ndarray,
) -> np.ndarray | None:
    return _extract_bubble_interior_cap_mask(
        crop,
        seed_mask,
        min_seed_coverage=0.98,
        max_area_ratio=0.995,
    )


def _bubble_interior_cap_mask(crop: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    if crop.size == 0 or seed_mask.size == 0:
        return np.zeros(seed_mask.shape, dtype=np.uint8)
    cap = _extract_bubble_interior_cap_mask(
        crop,
        seed_mask,
        min_seed_coverage=0.0,
        max_area_ratio=1.0,
    )
    if cap is None:
        return np.full(seed_mask.shape, 255, dtype=np.uint8)
    return cap


def _line_art_protect_mask(
    crop: np.ndarray,
    *,
    interior_cap: np.ndarray,
    source_seed_mask: np.ndarray,
    source_glyph_mask: np.ndarray,
    text_prior_mask: np.ndarray | None = None,
) -> np.ndarray:
    if crop.size == 0:
        return np.zeros(crop.shape[:2], dtype=np.uint8)
    gray = _to_gray(crop)
    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((h, w), dtype=np.uint8)
    cap = normalize_edit_mask(interior_cap, crop.shape)
    if not np.any(cap):
        return np.zeros((h, w), dtype=np.uint8)
    analysis_region = cv2.erode(
        cap,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    glyphs = normalize_edit_mask(source_glyph_mask, crop.shape)
    source_seed = normalize_edit_mask(source_seed_mask, crop.shape)
    text_prior = (
        None
        if text_prior_mask is None
        else normalize_edit_mask(text_prior_mask, crop.shape)
    )
    text_prior_available = bool(
        text_prior is not None and np.any(text_prior)
    )
    glyph_exclusion = cv2.dilate(
        glyphs,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    source_gap_override = _source_gap_edge_override_mask(
        gray,
        source_seed,
        analysis_region,
    )
    trusted_line_region = np.where(
        (analysis_region > 0)
        & ((glyph_exclusion <= 0) | (source_gap_override > 0)),
        255,
        0,
    ).astype(np.uint8)
    edges = cv2.Canny(gray, 40, 120)
    edges = np.where(
        (edges > 0)
        & (analysis_region > 0),
        255,
        0,
    ).astype(np.uint8)
    roi_short_side = min(h, w)
    # A fixed 28 px floor misses real structure in small bubbles, while a large
    # ROI fraction misses short panel details inside otherwise large bubbles.
    # Candidate continuity and source-glyph exclusion below provide the
    # semantic rejection; this length is only the Hough proposal floor.
    min_line_span = max(8, int(round(roi_short_side * 0.18)))
    min_hough_length = min_line_span - 1
    max_line_gap = max(2, min(6, int(round(roi_short_side * 0.075))))
    edge_without_glyphs = np.where(
        (edges > 0) & (trusted_line_region > 0),
        255,
        0,
    ).astype(np.uint8)
    protect = _elongated_gap_line_mask(
        source_gap_override,
        analysis_region,
        min_line_span=min_line_span,
    )
    if min_line_span <= roi_short_side:
        diagonal = np.eye(min_line_span, dtype=np.uint8)
        short_line_support = np.zeros((h, w), dtype=np.uint8)
        for kernel in (
            np.ones((1, min_line_span), dtype=np.uint8),
            np.ones((min_line_span, 1), dtype=np.uint8),
            diagonal,
            np.fliplr(diagonal),
        ):
            opened = cv2.morphologyEx(
                edge_without_glyphs,
                cv2.MORPH_OPEN,
                kernel,
            )
            short_line_support = np.where(
                (short_line_support > 0) | (opened > 0),
                255,
                0,
            ).astype(np.uint8)
        if np.any(short_line_support):
            filtered_short_line_support = np.zeros((h, w), dtype=np.uint8)
            (
                component_count,
                component_labels,
                component_stats,
                _component_centroids,
            ) = cv2.connectedComponentsWithStats(
                (short_line_support > 0).astype(np.uint8),
                8,
                cv2.CV_32S,
            )
            for component_index in range(1, component_count):
                x, y, width, height, area = [
                    int(value) for value in component_stats[component_index]
                ]
                if area <= 0 or width <= 0 or height <= 0:
                    continue
                margin = 2
                crop_x1 = max(0, x - margin)
                crop_y1 = max(0, y - margin)
                crop_x2 = min(w, x + width + margin)
                crop_y2 = min(h, y + height + margin)
                component = np.where(
                    component_labels[
                        crop_y1:crop_y2,
                        crop_x1:crop_x2,
                    ]
                    == component_index,
                    255,
                    0,
                ).astype(np.uint8)
                source_seed_crop = source_seed[
                    crop_y1:crop_y2,
                    crop_x1:crop_x2,
                ]
                source_gap_crop = source_gap_override[
                    crop_y1:crop_y2,
                    crop_x1:crop_x2,
                ]
                analysis_crop = analysis_region[
                    crop_y1:crop_y2,
                    crop_x1:crop_x2,
                ]
                text_prior_crop = (
                    None
                    if text_prior is None
                    else text_prior[
                        crop_y1:crop_y2,
                        crop_x1:crop_x2,
                    ]
                )
                if _line_candidate_has_source_gap(
                    component,
                    source_seed_crop,
                    analysis_crop,
                    min_support=min_line_span,
                ) and (
                    _line_candidate_has_trusted_source_gap(
                        component,
                        source_gap_crop,
                        analysis_crop,
                        min_support=min_line_span,
                    )
                    or _line_candidate_has_outside_text_support(
                        component,
                        text_prior_crop,
                        analysis_crop,
                        min_support=min_line_span,
                        text_prior_available=text_prior_available,
                    )
                ):
                    output_crop = filtered_short_line_support[
                        crop_y1:crop_y2,
                        crop_x1:crop_x2,
                    ]
                    output_crop[component > 0] = 255
            short_line_support = filtered_short_line_support
        if np.any(short_line_support):
            expanded_short_line_support = cv2.dilate(
                short_line_support,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
            protect = np.where(
                (protect > 0) | (expanded_short_line_support > 0),
                255,
                0,
            ).astype(np.uint8)

    hough_candidates: set[tuple[int, int, int, int]] = set()
    for proposal_edges in (edge_without_glyphs, edges):
        lines = cv2.HoughLinesP(
            proposal_edges,
            1,
            np.pi / 180.0,
            threshold=max(6, min_line_span // 2),
            minLineLength=min_hough_length,
            maxLineGap=max_line_gap,
        )
        if lines is None:
            continue
        for line in lines[:, 0, :]:
            endpoints = tuple(int(value) for value in line)
            reverse = (
                endpoints[2],
                endpoints[3],
                endpoints[0],
                endpoints[1],
            )
            hough_candidates.add(min(endpoints, reverse))

    for x1, y1, x2, y2 in sorted(hough_candidates):
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_hough_length:
            continue
        validation_margin = 2
        crop_x1 = max(0, min(x1, x2) - validation_margin)
        crop_y1 = max(0, min(y1, y2) - validation_margin)
        crop_x2 = min(w, max(x1, x2) + validation_margin + 1)
        crop_y2 = min(h, max(y1, y2) + validation_margin + 1)
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            continue
        candidate = np.zeros(
            (crop_y2 - crop_y1, crop_x2 - crop_x1),
            dtype=np.uint8,
        )
        cv2.line(
            candidate,
            (x1 - crop_x1, y1 - crop_y1),
            (x2 - crop_x1, y2 - crop_y1),
            255,
            thickness=1,
        )
        trusted_line_crop = trusted_line_region[
            crop_y1:crop_y2,
            crop_x1:crop_x2,
        ]
        edge_crop = edges[crop_y1:crop_y2, crop_x1:crop_x2]
        analysis_crop = analysis_region[
            crop_y1:crop_y2,
            crop_x1:crop_x2,
        ]
        source_seed_crop = source_seed[
            crop_y1:crop_y2,
            crop_x1:crop_x2,
        ]
        source_gap_crop = source_gap_override[
            crop_y1:crop_y2,
            crop_x1:crop_x2,
        ]
        text_prior_crop = (
            None
            if text_prior is None
            else text_prior[crop_y1:crop_y2, crop_x1:crop_x2]
        )
        outside_glyph_support = np.count_nonzero(
            (candidate > 0)
            & (trusted_line_crop > 0)
        )
        if outside_glyph_support < min_line_span:
            continue
        edge_support = np.count_nonzero(
            (candidate > 0)
            & (edge_crop > 0)
            & (trusted_line_crop > 0)
        )
        if edge_support < max(
            6,
            int(np.ceil(outside_glyph_support * 0.55)),
        ):
            continue
        if not _line_candidate_has_source_gap(
            candidate,
            source_seed_crop,
            analysis_crop,
            min_support=min_line_span,
        ):
            continue
        if not (
            _line_candidate_has_trusted_source_gap(
                candidate,
                source_gap_crop,
                analysis_crop,
                min_support=min_line_span,
            )
            or _line_candidate_has_outside_text_support(
                candidate,
                text_prior_crop,
                analysis_crop,
                min_support=min_line_span,
                text_prior_available=text_prior_available,
            )
        ):
            continue
        extension = max(3, int(max_line_gap))
        direction_x = float(x2 - x1) / max(length, 1.0)
        direction_y = float(y2 - y1) / max(length, 1.0)
        extended_start = (
            int(round(x1 - direction_x * extension)),
            int(round(y1 - direction_y * extension)),
        )
        extended_end = (
            int(round(x2 + direction_x * extension)),
            int(round(y2 + direction_y * extension)),
        )
        cv2.line(
            protect,
            extended_start,
            extended_end,
            255,
            thickness=6,
            lineType=cv2.LINE_AA,
        )
    if np.any(protect) and np.any(source_gap_override):
        nearby_protect = cv2.dilate(
            protect,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        gap_count, gap_labels, gap_stats, _gap_centroids = (
            cv2.connectedComponentsWithStats(
                (source_gap_override > 0).astype(np.uint8),
                8,
                cv2.CV_32S,
            )
        )
        for gap_index in range(1, gap_count):
            x, y, width, height, area = [
                int(value) for value in gap_stats[gap_index]
            ]
            if area <= 0 or width <= 0 or height <= 0:
                continue
            gap_component = (
                gap_labels[y:y + height, x:x + width] == gap_index
            )
            if np.any(
                gap_component
                & (nearby_protect[y:y + height, x:x + width] > 0)
            ):
                protect_crop = protect[y:y + height, x:x + width]
                protect_crop[gap_component] = 255
    return np.where(
        (protect > 0)
        & (analysis_region > 0),
        255,
        0,
    ).astype(np.uint8)


def _ambiguous_structure_near_source(
    crop: np.ndarray,
    *,
    interior_cap: np.ndarray,
    source_seed_mask: np.ndarray,
) -> bool:
    """Select a source-only fallback when no text prior can classify a line."""
    if crop.size == 0:
        return False
    gray = _to_gray(crop)
    cap = normalize_edit_mask(interior_cap, crop.shape)
    source_seed = normalize_edit_mask(source_seed_mask, crop.shape)
    if not np.any(cap) or not np.any(source_seed):
        return False
    analysis_region = cv2.erode(
        cap,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    background_pixels = gray[
        (analysis_region > 0) & (source_seed <= 0)
    ]
    if background_pixels.size < 16:
        return False
    background_median = float(np.median(background_pixels))
    background_mad = float(
        np.median(
            np.abs(
                background_pixels.astype(np.float32) - background_median
            )
        )
    )
    contrast_threshold = max(18.0, 2.5 * 1.4826 * background_mad)
    contrast = np.where(
        (analysis_region > 0)
        & (source_seed <= 0)
        & (
            np.abs(gray.astype(np.float32) - background_median)
            >= contrast_threshold
        ),
        255,
        0,
    ).astype(np.uint8)
    source_edge_exclusion = cv2.dilate(
        source_seed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    contrast[source_edge_exclusion > 0] = 0
    if not np.any(contrast):
        return False
    roi_short_side = min(gray.shape[:2])
    neighborhood_radius = max(6, int(round(roi_short_side * 0.10)))
    source_neighborhood = cv2.dilate(
        source_seed,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * neighborhood_radius + 1, 2 * neighborhood_radius + 1),
        ),
        iterations=1,
    )
    component_count, labels, stats, _centroids = (
        cv2.connectedComponentsWithStats(
            (contrast > 0).astype(np.uint8),
            8,
            cv2.CV_32S,
        )
    )
    candidate_points: list[np.ndarray] = []
    for component_index in range(1, component_count):
        x, y, width, height, area = stats[component_index]
        if int(area) <= 0 or width <= 0 or height <= 0:
            continue
        component = labels[y:y + height, x:x + width] == component_index
        neighborhood_crop = source_neighborhood[
            y:y + height,
            x:x + width,
        ]
        if not np.any(component & (neighborhood_crop > 0)):
            continue
        yy, xx = np.where(component)
        candidate_points.append(
            np.column_stack((xx + x, yy + y)).astype(np.float32)
        )
    if not candidate_points:
        return False

    def has_ambiguous_extent(points: np.ndarray) -> bool:
        if points.shape[0] < 6:
            return False
        _center, size, _angle = cv2.minAreaRect(points)
        major_extent = max(float(size[0]), float(size[1])) + 1.0
        return bool(
            major_extent >= max(8.0, float(roi_short_side) * 0.14)
        )

    candidate_points.sort(key=lambda points: points.shape[0], reverse=True)
    candidate_points = candidate_points[:32]
    if any(has_ambiguous_extent(points) for points in candidate_points):
        return True
    for left_index, left_points in enumerate(candidate_points):
        for right_points in candidate_points[left_index + 1:]:
            if has_ambiguous_extent(
                np.concatenate((left_points, right_points), axis=0)
            ):
                return True
    return False


def _compact_texture_field_detected(
    crop: np.ndarray,
    *,
    analysis_cap: np.ndarray,
    source_seed_mask: np.ndarray,
    text_prior_mask: np.ndarray | None,
) -> bool:
    """Fail closed when a bubble contains a distributed compact texture field."""
    if crop.size == 0:
        return False
    gray = _to_gray(crop)
    analysis = normalize_edit_mask(analysis_cap, crop.shape)
    source_seed = normalize_edit_mask(source_seed_mask, crop.shape)
    background_region = (analysis > 0) & (source_seed <= 0)
    background_pixels = gray[background_region]
    if background_pixels.size < 64:
        return False
    background_median = float(np.median(background_pixels))
    background_mad = float(
        np.median(
            np.abs(
                background_pixels.astype(np.float32) - background_median
            )
        )
    )
    contrast_threshold = max(18.0, 2.5 * 1.4826 * background_mad)
    short_side = min(gray.shape[:2])
    local_window = max(9, min(31, int(round(short_side / 8.0))))
    if local_window % 2 == 0:
        local_window += 1
    local_background = cv2.medianBlur(gray, local_window)
    local_contrast = np.abs(
        gray.astype(np.float32) - local_background.astype(np.float32)
    )
    contrast = np.where(
        background_region
        & (
            (
                np.abs(gray.astype(np.float32) - background_median)
                >= contrast_threshold
            )
            | (local_contrast >= 18.0)
        ),
        255,
        0,
    ).astype(np.uint8)
    # A large OCR prior can cover the entire bubble. It is therefore not
    # evidence that a repeated field is text. PR2 keeps structure first and
    # only routes the source-owned seed; PR3 may expand residue from stronger
    # OCR/CTD evidence later.
    _ = text_prior_mask
    if int(np.count_nonzero(contrast)) < 12:
        return False
    analysis_y, analysis_x = np.nonzero(analysis > 0)
    if analysis_x.size <= 0:
        return False
    analysis_x1 = float(analysis_x.min())
    analysis_y1 = float(analysis_y.min())
    analysis_width = float(analysis_x.max() - analysis_x.min() + 1)
    analysis_height = float(analysis_y.max() - analysis_y.min() + 1)
    analysis_distance = cv2.distanceTransform(
        (analysis > 0).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )

    def _points_are_distributed(points: np.ndarray) -> bool:
        if points.size <= 0:
            return False
        if not (
            float(np.ptp(points[:, 0]) + 1.0)
            >= analysis_width * 0.20
            and float(np.ptp(points[:, 1]) + 1.0)
            >= analysis_height * 0.20
        ):
            return False
        x_bins = np.clip(
            np.floor(
                (points[:, 0] - analysis_x1)
                * 4.0
                / max(1.0, analysis_width)
            ).astype(np.int32),
            0,
            3,
        )
        y_bins = np.clip(
            np.floor(
                (points[:, 1] - analysis_y1)
                * 4.0
                / max(1.0, analysis_height)
            ).astype(np.int32),
            0,
            3,
        )
        x_bin_count = int(np.unique(x_bins).size)
        y_bin_count = int(np.unique(y_bins).size)
        return bool(
            min(x_bin_count, y_bin_count) >= 2
            and max(x_bin_count, y_bin_count) >= 3
        )

    def _is_distributed_compact_field(proposal: np.ndarray) -> bool:
        component_count, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                (proposal > 0).astype(np.uint8),
                4,
                cv2.CV_32S,
            )
        )
        compact_centroids: list[np.ndarray] = []
        interior_centroids: list[np.ndarray] = []
        for component_index in range(1, component_count):
            _x, _y, component_width, component_height, area = [
                int(value) for value in stats[component_index]
            ]
            if (
                area < 2
                or area > 144
                or component_width <= 0
                or component_height <= 0
                or component_width > 12
                or component_height > 12
            ):
                continue
            aspect = float(max(component_width, component_height)) / float(
                max(1, min(component_width, component_height))
            )
            fill_ratio = float(area) / float(
                max(1, component_width * component_height)
            )
            if aspect > 3.0 or fill_ratio < 0.20:
                continue
            component_region = (
                labels[
                    _y:_y + component_height,
                    _x:_x + component_width,
                ]
                == component_index
            )
            component_pixels = gray[
                _y:_y + component_height,
                _x:_x + component_width,
            ][component_region]
            component_local_contrast = local_contrast[
                _y:_y + component_height,
                _x:_x + component_width,
            ][component_region]
            if (
                component_pixels.size <= 0
                or component_local_contrast.size <= 0
            ):
                continue
            component_delta = abs(
                float(np.median(component_pixels))
                - background_median
            )
            if (
                component_delta < 18.0
                and float(np.median(component_local_contrast)) < 18.0
            ):
                continue
            center_x = int(
                np.clip(
                    round(float(centroids[component_index, 0])),
                    0,
                    max(0, analysis.shape[1] - 1),
                )
            )
            center_y = int(
                np.clip(
                    round(float(centroids[component_index, 1])),
                    0,
                    max(0, analysis.shape[0] - 1),
                )
            )
            compact_centroids.append(centroids[component_index])
            if float(analysis_distance[center_y, center_x]) >= 3.0:
                interior_centroids.append(centroids[component_index])
        return bool(
            (
                len(interior_centroids) >= 6
                and _points_are_distributed(
                    np.asarray(interior_centroids, dtype=np.float32)
                )
            )
            or (
                len(compact_centroids) >= 10
                and _points_are_distributed(
                    np.asarray(compact_centroids, dtype=np.float32)
                )
            )
        )

    binary_contrast = (contrast > 0).astype(np.uint8)
    proposals = [binary_contrast]
    for kernel in (
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    ):
        opened = cv2.morphologyEx(
            binary_contrast,
            cv2.MORPH_OPEN,
            kernel,
        )
        if np.any(opened):
            proposals.append(opened)
    if any(
        _is_distributed_compact_field(proposal)
        for proposal in proposals
    ):
        return True

    interior_contrast = np.where(
        (binary_contrast > 0) & (analysis_distance >= 3.0),
        255,
        0,
    ).astype(np.uint8)
    contrast_y, contrast_x = np.nonzero(interior_contrast > 0)
    if contrast_x.size < 128:
        return False
    if not (
        float(np.ptp(contrast_x) + 1.0) >= analysis_width * 0.20
        and float(np.ptp(contrast_y) + 1.0) >= analysis_height * 0.20
    ):
        return False
    x_bins = np.clip(
        np.floor(
            (contrast_x.astype(np.float32) - analysis_x1)
            * 4.0
            / max(1.0, analysis_width)
        ).astype(np.int32),
        0,
        3,
    )
    y_bins = np.clip(
        np.floor(
            (contrast_y.astype(np.float32) - analysis_y1)
            * 4.0
            / max(1.0, analysis_height)
        ).astype(np.int32),
        0,
        3,
    )
    occupied_bins = np.zeros((4, 4), dtype=np.int32)
    np.add.at(occupied_bins, (y_bins, x_bins), 1)
    occupied_y, occupied_x = np.nonzero(occupied_bins >= 4)
    return bool(
        occupied_x.size >= 6
        and np.unique(occupied_x).size >= 2
        and np.unique(occupied_y).size >= 2
    )


def _line_candidate_has_outside_text_support(
    candidate: np.ndarray,
    text_prior_mask: np.ndarray | None,
    analysis_region: np.ndarray,
    *,
    min_support: int,
    text_prior_available: bool | None = None,
) -> bool:
    """Reject full-edge proposals that only describe text inside its prior."""
    if text_prior_mask is None:
        return False
    text_prior = normalize_edit_mask(
        text_prior_mask,
        analysis_region.shape,
    )
    prior_available = (
        bool(np.any(text_prior))
        if text_prior_available is None
        else bool(text_prior_available)
    )
    if not prior_available:
        return False
    line = normalize_edit_mask(candidate, analysis_region.shape)
    if not np.any(line):
        return False
    outside_line = np.where(
        (line > 0)
        & (analysis_region > 0)
        & (text_prior <= 0),
        255,
        0,
    ).astype(np.uint8)
    ys, xs = np.where(outside_line > 0)
    if xs.size <= 0:
        return False
    points = np.column_stack((xs, ys)).astype(np.float32)
    _center, size, _angle = cv2.minAreaRect(points)
    longitudinal_extent = max(float(size[0]), float(size[1])) + 1.0
    return longitudinal_extent >= max(
        3.0,
        float(np.ceil(min_support * 0.35)),
    )


def _line_candidate_has_trusted_source_gap(
    candidate: np.ndarray,
    source_gap_override: np.ndarray,
    analysis_region: np.ndarray,
    *,
    min_support: int,
) -> bool:
    """Accept an internal line only when the upstream seed carved its ink."""
    line = normalize_edit_mask(candidate, analysis_region.shape)
    if not np.any(line):
        return False
    corridor = cv2.dilate(
        line,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    source_gap = normalize_edit_mask(
        source_gap_override,
        analysis_region.shape,
    )
    gap_support = np.count_nonzero(
        (corridor > 0)
        & (source_gap > 0)
        & (analysis_region > 0)
    )
    return gap_support >= max(3, int(np.ceil(min_support * 0.25)))


def _line_candidate_has_source_gap(
    candidate: np.ndarray,
    source_seed_mask: np.ndarray,
    analysis_region: np.ndarray,
    *,
    min_support: int,
) -> bool:
    """Require line support in pixels the upstream erase mask did not claim.

    CTD text regions can be dense rectangles after dilation.  Treating the
    whole region as glyph support hides protected structure holes, while
    treating every edge in the rectangle as structure mistakes long glyph
    strokes for art.  The upstream keep-lines policy already removes protected
    structure from the erase seed, so a narrow corridor around a real line must
    contain enough unclaimed pixels.
    """
    line = normalize_edit_mask(candidate, analysis_region.shape)
    if not np.any(line):
        return False
    corridor = cv2.dilate(
        line,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    source_seed = normalize_edit_mask(source_seed_mask, analysis_region.shape)
    gap_support = np.count_nonzero(
        (corridor > 0)
        & (analysis_region > 0)
        & (source_seed <= 0)
    )
    return gap_support >= max(1, int(min_support))


def _source_gap_edge_override_mask(
    gray: np.ndarray,
    source_seed_mask: np.ndarray,
    analysis_region: np.ndarray,
) -> np.ndarray:
    """Expose only contrast-bearing keep-lines gaps in dense erase regions."""
    source_seed = np.where(source_seed_mask > 0, 255, 0).astype(np.uint8)
    if not np.any(source_seed):
        return np.zeros_like(source_seed, dtype=np.uint8)
    group_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    grouped_seed = cv2.dilate(source_seed, group_kernel, iterations=1)
    group_count, group_labels, group_stats, _group_centroids = (
        cv2.connectedComponentsWithStats(
            (grouped_seed > 0).astype(np.uint8),
            8,
            cv2.CV_32S,
        )
    )
    narrow_gaps = np.zeros_like(source_seed, dtype=np.uint8)
    for group_index in range(1, group_count):
        group_x, group_y, group_width, group_height, group_area = [
            int(value) for value in group_stats[group_index]
        ]
        if group_area <= 0 or group_width <= 0 or group_height <= 0:
            continue
        grouped_source = (
            source_seed[
                group_y:group_y + group_height,
                group_x:group_x + group_width,
            ]
            > 0
        ) & (
            group_labels[
                group_y:group_y + group_height,
                group_x:group_x + group_width,
            ]
            == group_index
        )
        ys, xs = np.where(grouped_source)
        if xs.size <= 0:
            continue
        local_x_min = int(xs.min())
        local_y_min = int(ys.min())
        x = group_x + local_x_min
        y = group_y + local_y_min
        w = int(xs.max()) - local_x_min + 1
        h = int(ys.max()) - local_y_min + 1
        area = int(xs.size)
        bbox_area = max(1, int(w) * int(h))
        fill_ratio = float(area) / float(bbox_area)
        if bbox_area < 64 or fill_ratio < 0.78:
            continue
        component_gaps = np.where(
            source_seed[y:y + h, x:x + w] <= 0,
            255,
            0,
        ).astype(np.uint8)
        narrow_gaps[y:y + h, x:x + w] = np.where(
            (narrow_gaps[y:y + h, x:x + w] > 0)
            | (component_gaps > 0),
            255,
            0,
        ).astype(np.uint8)
    if not np.any(narrow_gaps):
        return narrow_gaps

    analysis = normalize_edit_mask(analysis_region, source_seed.shape)
    background_region = (
        (analysis > 0)
        & (source_seed <= 0)
        & (narrow_gaps <= 0)
    )
    background_pixels = np.asarray(gray)[background_region]
    if background_pixels.size < 16:
        return np.zeros_like(source_seed, dtype=np.uint8)
    background_median = float(np.median(background_pixels))
    background_mad = float(
        np.median(np.abs(background_pixels.astype(np.float32) - background_median))
    )
    contrast_threshold = max(18.0, 2.5 * 1.4826 * background_mad)
    contrast_from_background = np.abs(
        np.asarray(gray, dtype=np.float32) - background_median
    )
    contrast_gaps = np.where(
        (narrow_gaps > 0)
        & (analysis > 0)
        & (contrast_from_background >= contrast_threshold),
        255,
        0,
    ).astype(np.uint8)
    if not np.any(contrast_gaps):
        return contrast_gaps
    return cv2.dilate(
        contrast_gaps,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )


def _elongated_gap_line_mask(
    source_gap_override: np.ndarray,
    analysis_region: np.ndarray,
    *,
    min_line_span: int,
) -> np.ndarray:
    """Keep rotation-independent elongated ink gaps at the Hough span floor."""
    gap_mask = normalize_edit_mask(
        source_gap_override,
        analysis_region.shape,
    )
    output = np.zeros_like(gap_mask, dtype=np.uint8)
    (
        component_count,
        component_labels,
        component_stats,
        _component_centroids,
    ) = cv2.connectedComponentsWithStats(
        (gap_mask > 0).astype(np.uint8),
        8,
        cv2.CV_32S,
    )
    for component_index in range(1, component_count):
        x, y, width, height, area = [
            int(value) for value in component_stats[component_index]
        ]
        if area < 3 or width <= 0 or height <= 0:
            continue
        component = (
            component_labels[y:y + height, x:x + width]
            == component_index
        )
        ys, xs = np.where(component)
        if xs.size < 3:
            continue
        points = np.column_stack((xs, ys)).astype(np.float32)
        _center, size, _angle = cv2.minAreaRect(points)
        major_extent = max(float(size[0]), float(size[1])) + 1.0
        minor_extent = min(float(size[0]), float(size[1])) + 1.0
        if major_extent < float(max(1, int(min_line_span) - 1)):
            continue
        if major_extent / max(1.0, minor_extent) < 2.0:
            continue
        output_crop = output[y:y + height, x:x + width]
        analysis_crop = analysis_region[y:y + height, x:x + width]
        output_crop[component & (analysis_crop > 0)] = 255
    return output


def _mask_near_line_art(
    edit_mask_crop: np.ndarray,
    line_protect_mask: np.ndarray,
) -> bool:
    mask_crop = normalize_edit_mask(edit_mask_crop, line_protect_mask.shape)
    line_mask = normalize_edit_mask(line_protect_mask, line_protect_mask.shape)
    if not np.any(mask_crop) or not np.any(line_mask):
        return False
    near_mask = cv2.dilate(mask_crop, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    overlap = np.count_nonzero((line_mask > 0) & (near_mask > 0))
    return overlap >= max(8, int(np.count_nonzero(mask_crop) * 0.002))


def _build_bubble_line_art_context(
    crop: np.ndarray,
    seed_mask: np.ndarray,
    *,
    text_prior_mask: np.ndarray | None = None,
) -> BubbleLineArtContext:
    interior_cap = _validated_bubble_interior_cap_mask(crop, seed_mask)
    line_protect_mask = np.zeros(seed_mask.shape, dtype=np.uint8)
    normalized_seed = normalize_edit_mask(seed_mask, crop.shape)
    source_residual_seed_mask = _non_boxy_seed_mask(normalized_seed)
    source_glyph_mask = normalized_seed
    texture_analysis_cap = interior_cap
    if texture_analysis_cap is None:
        texture_analysis_cap = np.zeros(normalized_seed.shape, dtype=np.uint8)
        if normalized_seed.shape[0] > 6 and normalized_seed.shape[1] > 6:
            texture_analysis_cap[3:-3, 3:-3] = 255
    texture_field_detected = _compact_texture_field_detected(
        crop,
        analysis_cap=texture_analysis_cap,
        source_seed_mask=normalized_seed,
        text_prior_mask=text_prior_mask,
    )
    normalized_text_prior = normalize_edit_mask(
        text_prior_mask,
        crop.shape,
    )
    ambiguous_structure_near_source = bool(
        interior_cap is not None
        and not np.any(normalized_text_prior)
        and _ambiguous_structure_near_source(
            crop,
            interior_cap=interior_cap,
            source_seed_mask=normalized_seed,
        )
    )
    if interior_cap is not None and not texture_field_detected:
        line_protect_mask = _line_art_protect_mask(
            crop,
            interior_cap=interior_cap,
            source_seed_mask=normalized_seed,
            source_glyph_mask=source_glyph_mask,
            text_prior_mask=text_prior_mask,
        )
    return BubbleLineArtContext(
        interior_cap=interior_cap,
        source_seed_mask=normalized_seed,
        source_glyph_mask=source_glyph_mask,
        source_residual_seed_mask=source_residual_seed_mask,
        line_protect_mask=line_protect_mask,
        texture_field_detected=texture_field_detected,
        ambiguous_structure_near_source=ambiguous_structure_near_source,
    )


def _rule_like_component(
    width: int,
    height: int,
    area: int,
) -> bool:
    long_side = max(int(width), int(height))
    short_side = min(int(width), int(height))
    if long_side < 16 or short_side > 5:
        return False
    aspect = float(long_side) / float(max(1, short_side))
    fill_ratio = float(area) / float(max(1, int(width) * int(height)))
    return aspect >= 10.0 and fill_ratio >= 0.30


def _component_filtered_mask(
    candidate: np.ndarray,
    seed_gate: np.ndarray,
    *,
    max_bbox_ratio: float = 0.22,
    min_area: int = 3,
) -> np.ndarray:
    if candidate.size == 0 or not np.any(candidate) or not np.any(seed_gate):
        return np.zeros_like(candidate, dtype=np.uint8)
    roi_h, roi_w = candidate.shape[:2]
    roi_area = max(1, roi_h * roi_w)
    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (candidate > 0).astype(np.uint8),
        8,
        cv2.CV_32S,
    )
    output = np.zeros_like(candidate, dtype=np.uint8)
    for label_idx in range(1, labels_count):
        x, y, w, h, area = stats[label_idx]
        if int(area) < int(min_area) or w <= 0 or h <= 0:
            continue
        if int(w * h) > int(round(roi_area * max_bbox_ratio)):
            continue
        if _rule_like_component(int(w), int(h), int(area)):
            continue
        component = labels[y:y + h, x:x + w] == label_idx
        if not np.any(seed_gate[y:y + h, x:x + w][component] > 0):
            continue
        output[y:y + h, x:x + w][component] = 255
    return np.where(output > 0, 255, 0).astype(np.uint8)


def _non_boxy_seed_mask(seed: np.ndarray) -> np.ndarray:
    if seed.size == 0 or not np.any(seed):
        return np.zeros_like(seed, dtype=np.uint8)
    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (seed > 0).astype(np.uint8),
        8,
        cv2.CV_32S,
    )
    output = np.zeros_like(seed, dtype=np.uint8)
    for label_idx in range(1, labels_count):
        x, y, w, h, area = stats[label_idx]
        if int(area) <= 0 or w <= 0 or h <= 0:
            continue
        bbox_area = max(1, int(w) * int(h))
        fill_ratio = float(area) / float(bbox_area)
        # Small CTD rescue boxes remain a useful conservative source seed.
        # Larger filled rectangles are layout geometry, not glyph evidence.
        if bbox_area >= 256 and fill_ratio >= 0.78:
            continue
        component = labels[y:y + h, x:x + w] == label_idx
        output[y:y + h, x:x + w][component] = 255
    return np.where(output > 0, 255, 0).astype(np.uint8)


def _capless_safe_source_seed_mask(seed: np.ndarray) -> np.ndarray:
    """Keep only compact source-owned glyph evidence without image expansion."""

    candidate = _non_boxy_seed_mask(seed)
    if candidate.size == 0 or not np.any(candidate):
        return np.zeros_like(candidate, dtype=np.uint8)
    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (candidate > 0).astype(np.uint8),
        8,
        cv2.CV_32S,
    )
    compact_centroids: list[np.ndarray] = []
    compact_label_indices: list[int] = []
    compact_descriptors: list[tuple[str, int, int, int, bytes]] = []
    boundary_compact_label_indices: set[int] = set()
    interior_compact_centroids: list[np.ndarray] = []
    interior_compact_label_indices: list[int] = []
    interior_compact_descriptors: list[
        tuple[str, int, int, int, bytes]
    ] = []
    dash_components: list[
        tuple[int, np.ndarray, np.ndarray, float]
    ] = []
    roi_height, roi_width = candidate.shape[:2]
    for label_idx in range(1, labels_count):
        x, y, w, h, area = stats[label_idx]
        if int(area) <= 0 or w <= 0 or h <= 0:
            continue
        long_side = max(int(w), int(h))
        short_side = min(int(w), int(h))
        aspect = float(long_side) / float(max(1, short_side))
        fill_ratio = float(area) / float(max(1, int(w) * int(h)))
        component = labels[y:y + h, x:x + w] == label_idx
        yy, xx = np.where(component)
        points = np.column_stack((xx, yy)).astype(np.float32)
        _center, oriented_size, _angle = cv2.minAreaRect(points)
        oriented_major = max(
            float(oriented_size[0]),
            float(oriented_size[1]),
        ) + 1.0
        oriented_minor = min(
            float(oriented_size[0]),
            float(oriented_size[1]),
        ) + 1.0
        if (
            2 <= int(area) <= 144
            and int(w) <= 12
            and int(h) <= 12
            and aspect <= 3.0
            and fill_ratio >= 0.20
        ):
            compact_centroids.append(_centroids[label_idx])
            compact_label_indices.append(int(label_idx))
            normalized_shape = np.zeros((12, 12), dtype=np.uint8)
            scale = min(10.0 / float(w), 10.0 / float(h))
            normalized_width = max(1, int(round(float(w) * scale)))
            normalized_height = max(1, int(round(float(h) * scale)))
            resized_shape = cv2.resize(
                component.astype(np.uint8),
                (normalized_width, normalized_height),
                interpolation=cv2.INTER_NEAREST,
            )
            normalized_x = (12 - normalized_width) // 2
            normalized_y = (12 - normalized_height) // 2
            normalized_shape[
                normalized_y:normalized_y + normalized_height,
                normalized_x:normalized_x + normalized_width,
            ] = resized_shape
            aspect_bin = int(
                round(
                    float(
                        np.log(
                            max(1e-6, float(w) / float(max(1, h)))
                        )
                    )
                    / 0.08
                )
            )
            size_bin = int(
                round(
                    float(np.log2(max(1.0, float(area))))
                )
            )
            if fill_ratio >= 0.60:
                solid_aspect_class = (
                    0
                    if aspect <= 1.75
                    else int(round(float(np.log(aspect)) / 0.35))
                )
                shape_descriptor = (
                    "solid",
                    solid_aspect_class,
                    size_bin,
                    0,
                    b"",
                )
            else:
                shape_descriptor = (
                    "shape",
                    aspect_bin,
                    size_bin,
                    int(round(fill_ratio / 0.10)),
                    normalized_shape.tobytes(),
                )
            compact_descriptors.append(shape_descriptor)
            if (
                int(x) < 3
                or int(y) < 3
                or int(x + w) > int(roi_width - 3)
                or int(y + h) > int(roi_height - 3)
            ):
                boundary_compact_label_indices.add(int(label_idx))
            center_x = float(_centroids[label_idx, 0])
            center_y = float(_centroids[label_idx, 1])
            if (
                3.0 <= center_x < float(max(3, roi_width - 3))
                and 3.0 <= center_y < float(max(3, roi_height - 3))
            ):
                interior_compact_centroids.append(_centroids[label_idx])
                interior_compact_label_indices.append(int(label_idx))
                interior_compact_descriptors.append(shape_descriptor)
        if (
            oriented_major >= 8.0
            and oriented_major / max(1.0, oriented_minor) >= 2.0
            and fill_ratio >= 0.20
        ):
            covariance = np.cov(points, rowvar=False, bias=True)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            major_direction = eigenvectors[:, int(np.argmax(eigenvalues))]
            dash_components.append(
                (
                    int(label_idx),
                    _centroids[label_idx],
                    major_direction.astype(np.float32),
                    float(oriented_minor),
                )
            )

    def compact_points_are_distributed(points: np.ndarray) -> bool:
        x_span = float(np.ptp(points[:, 0]) + 1.0)
        y_span = float(np.ptp(points[:, 1]) + 1.0)
        x_bins = np.unique(
            np.clip(
                np.floor(points[:, 0] * 4.0 / max(1, roi_width)),
                0,
                3,
            ).astype(np.int32)
        ).size
        y_bins = np.unique(
            np.clip(
                np.floor(points[:, 1] * 4.0 / max(1, roi_height)),
                0,
                3,
            ).astype(np.int32)
        ).size
        return bool(
            x_span >= float(roi_width) * 0.20
            and y_span >= float(roi_height) * 0.20
            and min(int(x_bins), int(y_bins)) >= 2
            and max(int(x_bins), int(y_bins)) >= 3
        )

    def repeated_shape_points(
        centroids: list[np.ndarray],
        label_indices: list[int],
        descriptors: list[tuple[str, int, int, int, bytes]],
        *,
        minimum_count: int,
    ) -> tuple[
        np.ndarray,
        list[int],
        set[tuple[str, int, int, int, bytes]],
    ] | None:
        descriptor_groups: dict[
            tuple[str, int, int, int, bytes],
            list[int],
        ] = {}
        for index, descriptor in enumerate(descriptors):
            descriptor_groups.setdefault(descriptor, []).append(index)
        ranked_groups = sorted(
            descriptor_groups.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
        if not ranked_groups:
            return None
        half = int(np.ceil(minimum_count / 2.0))
        eligible_groups = [
            item for item in ranked_groups if len(item[1]) >= half
        ]
        eligible_count = sum(
            len(item[1]) for item in eligible_groups
        )
        if eligible_count < minimum_count:
            dominant_descriptor, dominant_indices = ranked_groups[0]
            dominant_minimum = max(
                4,
                int(np.ceil(float(minimum_count) * (2.0 / 3.0))),
            )
            if (
                minimum_count <= 6
                and dominant_descriptor[0] == "solid"
                and len(dominant_indices) >= dominant_minimum
            ):
                dominant_points = np.asarray(
                    [centroids[index] for index in dominant_indices],
                    dtype=np.float32,
                )
                dominant_distances = np.linalg.norm(
                    dominant_points[:, None, :]
                    - dominant_points[None, :, :],
                    axis=2,
                )
                np.fill_diagonal(dominant_distances, np.inf)
                dominant_spacing = float(
                    np.median(np.min(dominant_distances, axis=1))
                )

                def spatial_continuation_error(
                    item: tuple[
                        tuple[str, int, int, int, bytes],
                        list[int],
                    ],
                ) -> float:
                    if item[0] == dominant_descriptor:
                        return 0.0
                    candidate_points = np.asarray(
                        [centroids[index] for index in item[1]],
                        dtype=np.float32,
                    )
                    nearest = np.min(
                        np.linalg.norm(
                            candidate_points[:, None, :]
                            - dominant_points[None, :, :],
                            axis=2,
                        ),
                        axis=1,
                    )
                    return float(
                        np.mean(
                            np.abs(nearest - dominant_spacing)
                            / max(1.0, dominant_spacing)
                        )
                    )

                compatible_groups = [
                    item
                    for item in ranked_groups
                    if item[0][0] == "solid"
                    and item[0][1] == dominant_descriptor[1]
                    and abs(item[0][2] - dominant_descriptor[2]) <= 1
                ]
                compatible_groups.sort(
                    key=lambda item: (
                        0 if item[0] == dominant_descriptor else 1,
                        spatial_continuation_error(item),
                        -len(item[1]),
                        abs(item[0][2] - dominant_descriptor[2]),
                        item[0],
                    )
                )
                eligible_groups = []
                selected_count = 0
                for item in compatible_groups:
                    eligible_groups.append(item)
                    selected_count += len(item[1])
                    if selected_count >= minimum_count:
                        break
        if sum(len(item[1]) for item in eligible_groups) < minimum_count:
            return None
        selected_indices = [
            index
            for _descriptor, group_indices in eligible_groups
            for index in group_indices
        ]

        spatial_refinement_limit = 64
        selected_points = np.asarray(
            [centroids[index] for index in selected_indices],
            dtype=np.float32,
        )
        if (
            2 <= selected_points.shape[0] <= spatial_refinement_limit
            and minimum_count <= 6
        ):
            seed_distances = np.linalg.norm(
                selected_points[:, None, :]
                - selected_points[None, :, :],
                axis=2,
            )
            np.fill_diagonal(seed_distances, np.inf)
            seed_spacing = float(
                np.median(np.min(seed_distances, axis=1))
            )
            seed_radius = max(4.0, seed_spacing * 2.05)
            remaining = set(selected_indices)
            spatial_groups: list[list[int]] = []
            while remaining:
                group = [remaining.pop()]
                pending = list(group)
                while pending:
                    current_index = pending.pop()
                    current_point = centroids[current_index]
                    connected = [
                        candidate_index
                        for candidate_index in remaining
                        if float(
                            np.linalg.norm(
                                centroids[candidate_index]
                                - current_point
                            )
                        ) <= seed_radius
                    ]
                    for candidate_index in connected:
                        remaining.remove(candidate_index)
                        group.append(candidate_index)
                        pending.append(candidate_index)
                spatial_groups.append(group)
            qualifying_groups = [
                group
                for group in spatial_groups
                if len(group) >= minimum_count
            ]
            pruned_groups: list[list[int]] = []
            for group in qualifying_groups:
                group_points = np.asarray(
                    [centroids[index] for index in group],
                    dtype=np.float32,
                )
                adjacency = (
                    np.linalg.norm(
                        group_points[:, None, :]
                        - group_points[None, :, :],
                        axis=2,
                    ) <= seed_radius
                )
                np.fill_diagonal(adjacency, False)
                active = np.ones(len(group), dtype=bool)
                while np.any(active):
                    active_indices = np.flatnonzero(active)
                    degrees = np.sum(
                        adjacency[
                            np.ix_(active_indices, active_indices)
                        ],
                        axis=1,
                    )
                    remove_indices = active_indices[degrees < 2]
                    if remove_indices.size == 0:
                        break
                    active[remove_indices] = False
                pruned_group = [
                    index
                    for index, keep in zip(group, active)
                    if bool(keep)
                ]
                if len(pruned_group) >= minimum_count:
                    pruned_points = np.asarray(
                        [centroids[index] for index in pruned_group],
                        dtype=np.float32,
                    )
                    centered_points = pruned_points - np.mean(
                        pruned_points,
                        axis=0,
                    )
                    covariance = np.cov(
                        centered_points,
                        rowvar=False,
                        bias=True,
                    )
                    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                    major_direction = eigenvectors[
                        :, int(np.argmax(eigenvalues))
                    ].astype(np.float32)
                    orthogonal_direction = np.asarray(
                        [
                            -float(major_direction[1]),
                            float(major_direction[0]),
                        ],
                        dtype=np.float32,
                    )
                    directional_indices: list[int] = []
                    for point_index, point in enumerate(pruned_points):
                        deltas = pruned_points - point
                        distances = np.linalg.norm(deltas, axis=1)
                        neighbor_mask = (
                            (distances > 1e-6)
                            & (distances <= seed_radius)
                        )
                        if not np.any(neighbor_mask):
                            continue
                        directions = (
                            deltas[neighbor_mask]
                            / distances[neighbor_mask, None]
                        )
                        has_major_support = bool(
                            np.any(
                                np.abs(directions @ major_direction)
                                >= 0.90
                            )
                        )
                        has_orthogonal_support = bool(
                            np.any(
                                np.abs(directions @ orthogonal_direction)
                                >= 0.90
                            )
                        )
                        if has_major_support and has_orthogonal_support:
                            directional_indices.append(point_index)
                    if len(directional_indices) >= minimum_count:
                        pruned_group = [
                            pruned_group[index]
                            for index in directional_indices
                        ]
                if (
                    len(pruned_group) >= minimum_count
                    and compact_points_are_distributed(
                        np.asarray(
                            [centroids[index] for index in pruned_group],
                            dtype=np.float32,
                        )
                    )
                ):
                    pruned_groups.append(pruned_group)
            qualifying_groups = pruned_groups
            if not qualifying_groups:
                return None
            selected_indices = max(
                qualifying_groups,
                key=lambda group: (len(group), tuple(sorted(group))),
            )

        selected_descriptors = {
            descriptors[index] for index in selected_indices
        }

        def descriptor_is_compatible(
            descriptor: tuple[str, int, int, int, bytes],
        ) -> bool:
            if descriptor in selected_descriptors:
                return True
            if descriptor[0] != "solid":
                return False
            return any(
                selected[0] == "solid"
                and descriptor[1] == selected[1]
                and abs(descriptor[2] - selected[2]) <= 1
                for selected in selected_descriptors
            )

        selected_points = np.asarray(
            [centroids[index] for index in selected_indices],
            dtype=np.float32,
        )
        if 2 <= selected_points.shape[0] <= spatial_refinement_limit:
            selected_distances = np.linalg.norm(
                selected_points[:, None, :]
                - selected_points[None, :, :],
                axis=2,
            )
            np.fill_diagonal(selected_distances, np.inf)
            repeated_spacing = float(
                np.median(np.min(selected_distances, axis=1))
            )
            connection_radius = max(4.0, repeated_spacing * 1.55)
            compatible_indices = [
                index
                for index, descriptor in enumerate(descriptors)
                if descriptor_is_compatible(descriptor)
            ]
            cell_size = connection_radius
            spatial_cells: dict[tuple[int, int], list[int]] = {}
            for index in compatible_indices:
                point = centroids[index]
                cell = (
                    int(np.floor(float(point[0]) / cell_size)),
                    int(np.floor(float(point[1]) / cell_size)),
                )
                spatial_cells.setdefault(cell, []).append(index)

            selected_set = set(selected_indices)
            pending = list(selected_indices)
            while pending:
                current_index = pending.pop()
                current_point = centroids[current_index]
                current_cell = (
                    int(
                        np.floor(float(current_point[0]) / cell_size)
                    ),
                    int(
                        np.floor(float(current_point[1]) / cell_size)
                    ),
                )
                for offset_y in (-1, 0, 1):
                    for offset_x in (-1, 0, 1):
                        for candidate_index in spatial_cells.get(
                            (
                                current_cell[0] + offset_x,
                                current_cell[1] + offset_y,
                            ),
                            (),
                        ):
                            if candidate_index in selected_set:
                                continue
                            candidate_point = centroids[candidate_index]
                            if float(
                                np.linalg.norm(
                                    candidate_point - current_point
                                )
                            ) > connection_radius:
                                continue
                            selected_set.add(candidate_index)
                            pending.append(candidate_index)
            selected_indices = sorted(selected_set)
            selected_descriptors.update(
                descriptors[index] for index in selected_indices
            )
        return (
            np.asarray(
                [centroids[index] for index in selected_indices],
                dtype=np.float32,
            ),
            [label_indices[index] for index in selected_indices],
            selected_descriptors,
        )

    interior_repeated = repeated_shape_points(
        interior_compact_centroids,
        interior_compact_label_indices,
        interior_compact_descriptors,
        minimum_count=6,
    )
    boundary_repeated = repeated_shape_points(
        compact_centroids,
        compact_label_indices,
        compact_descriptors,
        minimum_count=10,
    )
    suppressed_label_indices: set[int] = set()
    if (
        interior_repeated is not None
        and compact_points_are_distributed(interior_repeated[0])
    ):
        matching_descriptors = interior_repeated[2]
        interior_points = interior_repeated[0]
        def matches_repeated_descriptor(
            descriptor: tuple[str, int, int, int, bytes],
        ) -> bool:
            return descriptor in matching_descriptors or (
                descriptor[0] == "solid"
                and any(
                    selected[0] == "solid"
                    and descriptor[1] == selected[1]
                    and abs(descriptor[2] - selected[2]) <= 1
                    for selected in matching_descriptors
                )
            )

        continuation_labels = set(interior_repeated[1])
        if (
            len(interior_points) > 64
            or len(compact_centroids) > 64
        ):
            continuation_labels.update(
                label_index
                for label_index, descriptor in zip(
                    compact_label_indices,
                    compact_descriptors,
                )
                if matches_repeated_descriptor(descriptor)
            )
        else:
            interior_distances = np.linalg.norm(
                interior_points[:, None, :]
                - interior_points[None, :, :],
                axis=2,
            )
            np.fill_diagonal(interior_distances, np.inf)
            interior_spacing = float(
                np.median(np.min(interior_distances, axis=1))
            )
            continuation_radius = max(4.0, interior_spacing * 1.55)
            continuation_points = list(interior_points)
            changed = True
            while changed:
                changed = False
                current_points = np.asarray(
                    continuation_points,
                    dtype=np.float32,
                )
                for centroid, label_index, descriptor in zip(
                    compact_centroids,
                    compact_label_indices,
                    compact_descriptors,
                ):
                    if label_index in continuation_labels:
                        continue
                    if not matches_repeated_descriptor(descriptor):
                        continue
                    if float(
                        np.min(
                            np.linalg.norm(
                                current_points - centroid,
                                axis=1,
                            )
                        )
                    ) > continuation_radius:
                        continue
                    continuation_labels.add(label_index)
                    continuation_points.append(centroid)
                    changed = True
        suppressed_label_indices.update(continuation_labels)
    if (
        boundary_repeated is not None
        and compact_points_are_distributed(boundary_repeated[0])
    ):
        boundary_labels = set(boundary_repeated[1])
        boundary_seeds = boundary_labels & boundary_compact_label_indices
        if boundary_seeds:
            label_to_point = {
                label_index: centroid
                for label_index, centroid in zip(
                    compact_label_indices,
                    compact_centroids,
                )
                if label_index in boundary_labels
            }
            label_to_descriptor = {
                label_index: descriptor
                for label_index, descriptor in zip(
                    compact_label_indices,
                    compact_descriptors,
                )
                if label_index in boundary_labels
            }
            selected_boundary_labels: set[int] = set()
            seed_descriptors = {
                label_to_descriptor[label] for label in boundary_seeds
            }
            for seed_descriptor in seed_descriptors:
                descriptor_labels = {
                    label
                    for label in boundary_labels
                    if label_to_descriptor[label] == seed_descriptor
                }
                descriptor_seeds = boundary_seeds & descriptor_labels
                if len(descriptor_labels) > 64:
                    selected_boundary_labels.update(descriptor_labels)
                    continue
                descriptor_points = np.asarray(
                    [label_to_point[label] for label in descriptor_labels],
                    dtype=np.float32,
                )
                if descriptor_points.shape[0] >= 2:
                    descriptor_distances = np.linalg.norm(
                        descriptor_points[:, None, :]
                        - descriptor_points[None, :, :],
                        axis=2,
                    )
                    np.fill_diagonal(descriptor_distances, np.inf)
                    descriptor_spacing = float(
                        np.median(np.min(descriptor_distances, axis=1))
                    )
                else:
                    descriptor_spacing = 4.0
                boundary_radius = max(4.0, descriptor_spacing * 2.05)
                descriptor_seed_points = np.asarray(
                    [label_to_point[label] for label in descriptor_seeds],
                    dtype=np.float32,
                )
                lattice_direction: np.ndarray | None = None
                lattice_origin = 0.0
                lattice_spacing = 0.0
                orthogonal_direction: np.ndarray | None = None
                orthogonal_seed_projections = np.empty(
                    (0,),
                    dtype=np.float32,
                )
                orthogonal_spacing = 0.0
                repeated_row_phase_labels: set[int] = set()
                if descriptor_seed_points.shape[0] >= 3:
                    centered_seed_points = descriptor_seed_points - np.mean(
                        descriptor_seed_points,
                        axis=0,
                    )
                    covariance = np.cov(
                        centered_seed_points,
                        rowvar=False,
                        bias=True,
                    )
                    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                    lattice_direction = eigenvectors[
                        :, int(np.argmax(eigenvalues))
                    ].astype(np.float32)
                    seed_projections = np.sort(
                        descriptor_seed_points @ lattice_direction
                    )
                    projection_steps = np.diff(seed_projections)
                    projection_steps = projection_steps[
                        projection_steps > 2.0
                    ]
                    if projection_steps.size > 0:
                        lattice_spacing = float(np.median(projection_steps))
                        lattice_origin = float(seed_projections[0])
                        orthogonal_direction = np.asarray(
                            [
                                -float(lattice_direction[1]),
                                float(lattice_direction[0]),
                            ],
                            dtype=np.float32,
                        )
                        orthogonal_seed_projections = (
                            descriptor_seed_points @ orthogonal_direction
                        )
                        candidate_orthogonal_projections = (
                            descriptor_points @ orthogonal_direction
                        )
                        orthogonal_offsets = np.asarray(
                            [
                                float(
                                    np.min(
                                        np.abs(
                                            orthogonal_seed_projections
                                            - projection
                                        )
                                    )
                                )
                                for projection in (
                                    candidate_orthogonal_projections
                                )
                            ],
                            dtype=np.float32,
                        )
                        positive_offsets = np.sort(
                            orthogonal_offsets[orthogonal_offsets > 2.0]
                        )
                        offset_groups: list[list[float]] = []
                        for offset in positive_offsets:
                            value = float(offset)
                            if (
                                not offset_groups
                                or abs(
                                    value
                                    - float(np.mean(offset_groups[-1]))
                                )
                                > 3.0
                            ):
                                offset_groups.append([value])
                            else:
                                offset_groups[-1].append(value)
                        repeated_offset_groups = [
                            group
                            for group in offset_groups
                            if len(group) >= 2
                        ]
                        if repeated_offset_groups:
                            best_offset_group = min(
                                repeated_offset_groups,
                                key=lambda group: (
                                    -len(group),
                                    float(np.mean(group)),
                                ),
                            )
                            orthogonal_spacing = float(
                                np.mean(best_offset_group)
                            )
                        primary_phase_tolerance = max(
                            2.5,
                            lattice_spacing * 0.22,
                        )
                        descriptor_projection_by_label = {
                            label: float(
                                label_to_point[label] @ lattice_direction
                            )
                            for label in descriptor_labels
                        }
                        descriptor_orthogonal_by_label = {
                            label: float(
                                label_to_point[label]
                                @ orthogonal_direction
                            )
                            for label in descriptor_labels
                        }
                        for label in descriptor_labels:
                            phase = (
                                descriptor_projection_by_label[label]
                                - lattice_origin
                            ) % lattice_spacing
                            supporting_labels = 0
                            for candidate_label in descriptor_labels:
                                if abs(
                                    descriptor_orthogonal_by_label[
                                        candidate_label
                                    ]
                                    - descriptor_orthogonal_by_label[label]
                                ) > 3.0:
                                    continue
                                candidate_phase = (
                                    descriptor_projection_by_label[
                                        candidate_label
                                    ]
                                    - lattice_origin
                                ) % lattice_spacing
                                phase_delta = abs(candidate_phase - phase)
                                circular_delta = min(
                                    phase_delta,
                                    lattice_spacing - phase_delta,
                                )
                                if circular_delta <= primary_phase_tolerance:
                                    supporting_labels += 1
                            if supporting_labels >= 2:
                                repeated_row_phase_labels.add(label)

                def follows_boundary_lattice(label_index: int) -> bool:
                    if (
                        lattice_direction is None
                        or lattice_spacing <= 0.0
                    ):
                        return True
                    projection = float(
                        label_to_point[label_index] @ lattice_direction
                    )
                    phase = (projection - lattice_origin) % lattice_spacing
                    phase_error = min(phase, lattice_spacing - phase)
                    if (
                        phase_error
                        > max(2.5, lattice_spacing * 0.22)
                        and label_index not in repeated_row_phase_labels
                    ):
                        return False
                    if (
                        orthogonal_direction is None
                        or orthogonal_spacing <= 0.0
                    ):
                        return True
                    orthogonal_projection = float(
                        label_to_point[label_index]
                        @ orthogonal_direction
                    )
                    orthogonal_offset = float(
                        np.min(
                            np.abs(
                                orthogonal_seed_projections
                                - orthogonal_projection
                            )
                        )
                    )
                    if orthogonal_offset <= 2.0:
                        return True
                    orthogonal_phase = (
                        orthogonal_offset % orthogonal_spacing
                    )
                    orthogonal_phase_error = min(
                        orthogonal_phase,
                        orthogonal_spacing - orthogonal_phase,
                    )
                    return orthogonal_phase_error <= max(
                        2.5,
                        orthogonal_spacing * 0.18,
                    )

                selected_descriptor_labels = set(descriptor_seeds)
                pending_boundary_labels = list(descriptor_seeds)
                while pending_boundary_labels:
                    current_label = pending_boundary_labels.pop()
                    current_point = label_to_point[current_label]
                    for candidate_label in (
                        descriptor_labels - selected_descriptor_labels
                    ):
                        if not follows_boundary_lattice(candidate_label):
                            continue
                        if float(
                            np.linalg.norm(
                                label_to_point[candidate_label]
                                - current_point
                            )
                        ) > boundary_radius:
                            continue
                        selected_descriptor_labels.add(candidate_label)
                        pending_boundary_labels.append(candidate_label)
                selected_boundary_labels.update(selected_descriptor_labels)
            suppressed_label_indices.update(selected_boundary_labels)
    active_dash_components = [
        component
        for component in dash_components
        if component[0] not in suppressed_label_indices
    ]
    if len(active_dash_components) >= 3:
        orientation_bin_count = 12
        orientation_bins: list[
            list[tuple[int, np.ndarray, np.ndarray, float]]
        ] = [[] for _index in range(orientation_bin_count)]
        for component in active_dash_components:
            direction = component[2]
            angle = float(
                np.arctan2(float(direction[1]), float(direction[0]))
            ) % float(np.pi)
            bin_index = int(
                np.floor(
                    angle * float(orientation_bin_count) / float(np.pi)
                )
            ) % orientation_bin_count
            orientation_bins[bin_index].append(component)

        seen_orientation_groups: set[tuple[int, ...]] = set()
        bin_width = float(np.pi) / float(orientation_bin_count)
        for center_bin in range(orientation_bin_count):
            candidates = [
                component
                for offset in (-1, 0, 1)
                for component in orientation_bins[
                    (center_bin + offset) % orientation_bin_count
                ]
            ]
            if len(candidates) < 3:
                continue
            center_angle = (float(center_bin) + 0.5) * bin_width
            center_direction = np.asarray(
                (np.cos(center_angle), np.sin(center_angle)),
                dtype=np.float32,
            )
            orientation_cluster = [
                component
                for component in candidates
                if abs(float(np.dot(component[2], center_direction)))
                >= 0.90
            ]
            group_key = tuple(
                sorted(component[0] for component in orientation_cluster)
            )
            if len(group_key) < 3 or group_key in seen_orientation_groups:
                continue
            seen_orientation_groups.add(group_key)
            perpendicular_direction = np.asarray(
                (-float(center_direction[1]), float(center_direction[0])),
                dtype=np.float32,
            )
            projected_components = [
                (
                    float(np.dot(component[1], perpendicular_direction)),
                    component,
                )
                for component in orientation_cluster
            ]
            projected_components.sort(key=lambda item: item[0])
            collinear_bands: list[
                list[tuple[int, np.ndarray, np.ndarray, float]]
            ] = []
            for projection, component in projected_components:
                if not collinear_bands:
                    collinear_bands.append([component])
                    previous_projection = projection
                    continue
                previous_component = collinear_bands[-1][-1]
                band_tolerance = max(
                    4.0,
                    1.5 * max(previous_component[3], component[3]),
                )
                if projection - previous_projection <= band_tolerance:
                    collinear_bands[-1].append(component)
                else:
                    collinear_bands.append([component])
                previous_projection = projection

            for collinear_band in collinear_bands:
                if len(collinear_band) < 3:
                    continue
                points = np.asarray(
                    [component[1] for component in collinear_band],
                    dtype=np.float32,
                )
                centered_points = points - np.mean(
                    points,
                    axis=0,
                    keepdims=True,
                )
                covariance = np.cov(
                    centered_points,
                    rowvar=False,
                    bias=True,
                )
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                group_direction = eigenvectors[:, int(np.argmax(eigenvalues))]
                aligned_components = [
                    component
                    for component in collinear_band
                    if abs(float(np.dot(component[2], group_direction)))
                    >= 0.90
                ]
                if len(aligned_components) < 3:
                    continue
                aligned_points = np.asarray(
                    [component[1] for component in aligned_components],
                    dtype=np.float32,
                )
                _center, size, _angle = cv2.minAreaRect(aligned_points)
                major_extent = max(float(size[0]), float(size[1])) + 1.0
                minor_extent = min(float(size[0]), float(size[1])) + 1.0
                if (
                    major_extent >= 24.0
                    and major_extent / max(1.0, minor_extent) >= 4.0
                ):
                    suppressed_label_indices.update(
                        component[0] for component in aligned_components
                    )

    output = np.zeros_like(candidate, dtype=np.uint8)
    for label_idx in range(1, labels_count):
        x, y, w, h, area = stats[label_idx]
        if int(area) <= 0 or w <= 0 or h <= 0:
            continue
        if int(label_idx) in suppressed_label_indices:
            continue
        if _rule_like_component(int(w), int(h), int(area)):
            continue
        component = labels[y:y + h, x:x + w] == label_idx
        yy, xx = np.where(component)
        points = np.column_stack((xx, yy)).astype(np.float32)
        _center, size, _angle = cv2.minAreaRect(points)
        major_extent = max(float(size[0]), float(size[1])) + 1.0
        minor_extent = min(float(size[0]), float(size[1])) + 1.0
        bbox_area = max(1, int(w) * int(h))
        if (
            bbox_area >= 256
            or (
                major_extent >= 24.0
                and major_extent / max(1.0, minor_extent) >= 2.0
            )
        ):
            continue
        output[y:y + h, x:x + w][component] = 255
    return np.where(output > 0, 255, 0).astype(np.uint8)


def build_bubble_residual_edit_mask(
    image_rgb: np.ndarray,
    source_mask: np.ndarray,
    block,
    *,
    seed_dilate_px: int = 8,
    final_dilate_px: int = 4,
    protect_line_art: bool = True,
    line_art_context: BubbleLineArtContext | None = None,
) -> tuple[np.ndarray, BubbleEraseBlockStats]:
    edit_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    if getattr(block, "text_class", "") != "text_bubble":
        return edit_mask, BubbleEraseBlockStats(mode=ERASE_MODE_BUBBLE_SKIPPED, skipped_reason="not_text_bubble")

    bubble_roi = normalize_xyxy(getattr(block, "bubble_xyxy", None), image_rgb.shape)
    if bubble_roi is None:
        return edit_mask, BubbleEraseBlockStats(mode=ERASE_MODE_BUBBLE_SKIPPED, skipped_reason="missing_bubble_roi")

    x1, y1, x2, y2 = bubble_roi
    source = normalize_edit_mask(source_mask, image_rgb.shape)
    seed = source[y1:y2, x1:x2]
    if not np.any(seed):
        return edit_mask, BubbleEraseBlockStats(mode=ERASE_MODE_BUBBLE_SKIPPED, skipped_reason="empty_seed")

    crop = image_rgb[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop.astype(np.uint8)
    prior = build_text_prior_mask(image_rgb, block, bubble_roi, dilate_iterations=2)
    if not np.any(prior):
        prior = seed.copy()

    px = max(1, int(seed_dilate_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1), (px, px))
    seed_gate = cv2.dilate(seed, kernel, iterations=1)
    candidate_gate = np.where((seed_gate > 0) & (prior > 0), 255, 0).astype(np.uint8)
    context = line_art_context or _build_bubble_line_art_context(
        crop,
        seed,
        text_prior_mask=prior,
    )
    interior_cap = (
        context.interior_cap
        if context.interior_cap is not None
        else np.full(seed.shape, 255, dtype=np.uint8)
    )
    line_protect = (
        np.where(
            context.line_protect_mask > 0,
            255,
            0,
        ).astype(np.uint8)
        if bool(protect_line_art)
        else np.zeros_like(seed, dtype=np.uint8)
    )
    protect = np.where(
        (_bubble_border_protect_mask(seed.shape, width=3) > 0)
        | (interior_cap <= 0)
        | (line_protect > 0),
        255,
        0,
    ).astype(np.uint8)

    safe_bg = gray[(candidate_gate <= 0) & (protect <= 0)]
    if safe_bg.size == 0:
        safe_bg = gray[protect <= 0]
    if safe_bg.size == 0:
        safe_bg = gray.reshape(-1)
    bg_median = float(np.median(safe_bg))
    bg_std = float(np.std(safe_bg))
    bright_threshold = min(245.0, bg_median + max(18.0, bg_std * 1.25))
    dark_threshold = max(10.0, bg_median - max(24.0, bg_std * 1.50))

    bright = np.where(gray >= bright_threshold, 255, 0).astype(np.uint8)
    dark = np.where(gray <= dark_threshold, 255, 0).astype(np.uint8)
    candidates = np.where(((bright > 0) | (dark > 0)) & (candidate_gate > 0) & (protect <= 0), 255, 0).astype(np.uint8)
    residual = _component_filtered_mask(candidates, seed_gate)
    source_glyphs = context.source_residual_seed_mask
    prior_candidates = np.where(((bright > 0) | (dark > 0)) & (prior > 0) & (protect <= 0), 255, 0).astype(np.uint8)
    if np.any(prior_candidates):
        prior_candidates = cv2.morphologyEx(
            prior_candidates,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), np.uint8),
            iterations=1,
        )
    orphan_glyphs = _component_filtered_mask(
        prior_candidates,
        prior,
        max_bbox_ratio=0.45,
        min_area=4,
    )
    merged = np.where(
        ((source_glyphs > 0) | (residual > 0) | (orphan_glyphs > 0)) & (protect <= 0),
        255,
        0,
    ).astype(np.uint8)
    if np.any(merged):
        final_px = max(1, int(final_dilate_px))
        merged = cv2.dilate(
            merged,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * final_px + 1, 2 * final_px + 1), (final_px, final_px)),
            iterations=1,
        )
        merged = np.where((merged > 0) & (protect <= 0), 255, 0).astype(np.uint8)

    edit_mask[y1:y2, x1:x2] = merged
    stats = BubbleEraseBlockStats(
        mode=ERASE_MODE_BUBBLE_TELEA,
        edit_pixel_count=mask_pixel_count(merged),
        protect_pixel_count=mask_pixel_count(protect),
    )
    return edit_mask, stats


def _ring_mask(mask: np.ndarray, *, radius: int) -> np.ndarray:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not np.any(binary):
        return np.zeros_like(binary, dtype=np.uint8)
    px = max(1, int(radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1), (px, px))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    return np.where((dilated > 0) & (binary <= 0), 255, 0).astype(np.uint8)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8)
    if image.shape[2] >= 3:
        return cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    return image[:, :, 0].astype(np.uint8)


def _should_use_flat_fill(
    image_rgb: np.ndarray,
    edit_mask: np.ndarray,
    *,
    background_exclude_mask: np.ndarray | None = None,
) -> bool:
    ring = _ring_mask(edit_mask, radius=6)
    excluded = normalize_edit_mask(background_exclude_mask, image_rgb.shape)
    ring = np.where((ring > 0) & (excluded <= 0), 255, 0).astype(np.uint8)
    if not np.any(ring):
        return False
    gray = _to_gray(image_rgb)
    ring_pixels = gray[ring > 0]
    if ring_pixels.size == 0:
        return False
    spread = float(np.percentile(ring_pixels, 95) - np.percentile(ring_pixels, 5))
    std = float(np.std(ring_pixels))
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges[ring > 0])) / float(max(1, ring_pixels.size))
    return std <= 12.0 and spread <= 36.0 and edge_density <= 0.08


def _local_median_fill(
    image_rgb: np.ndarray,
    edit_mask: np.ndarray,
    *,
    sample_roi: tuple[int, int, int, int] | None = None,
    background_exclude_mask: np.ndarray | None = None,
) -> np.ndarray:
    mask = normalize_edit_mask(edit_mask, image_rgb.shape)
    excluded = normalize_edit_mask(background_exclude_mask, image_rgb.shape)
    output = np.asarray(image_rgb).copy()
    if not np.any(mask):
        return output
    normalized_sample_roi = normalize_xyxy(sample_roi, image_rgb.shape)
    local_sample_region = None
    if normalized_sample_roi is not None:
        x1, y1, x2, y2 = normalized_sample_roi
        local_sample_region = np.zeros(mask.shape, dtype=bool)
        local_crop = (
            (mask[y1:y2, x1:x2] <= 0)
            & (excluded[y1:y2, x1:x2] <= 0)
            & (
                _bubble_border_protect_mask(
                    (y2 - y1, x2 - x1),
                    width=4,
                )
                <= 0
            )
        )
        local_sample_region[y1:y2, x1:x2] = local_crop

    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        8,
        cv2.CV_32S,
    )
    for label_idx in range(1, labels_count):
        x, y, w, h, area = stats[label_idx]
        if int(area) <= 0 or w <= 0 or h <= 0:
            continue
        component = np.zeros_like(mask, dtype=np.uint8)
        component[y:y + h, x:x + w][labels[y:y + h, x:x + w] == label_idx] = 255
        ring = _ring_mask(component, radius=5)
        ring_region = (ring > 0) & (excluded <= 0)
        if local_sample_region is not None:
            ring_region &= local_sample_region
        ring_pixels = output[ring_region]
        if ring_pixels.size == 0:
            if local_sample_region is not None:
                ring_pixels = output[local_sample_region]
            else:
                ring_pixels = output[(mask <= 0) & (excluded <= 0)]
        if ring_pixels.size == 0:
            continue
        fill_value = np.median(ring_pixels, axis=0)
        if output.ndim == 2:
            output[component > 0] = np.uint8(np.clip(round(float(fill_value)), 0, 255))
        else:
            output[component > 0] = np.clip(np.round(fill_value), 0, 255).astype(output.dtype)
    return composite_with_edit_mask(image_rgb, output, mask)


def _trimmed_background_pixels(
    image_rgb: np.ndarray,
    edit_mask: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    background_exclude_mask: np.ndarray | None = None,
) -> np.ndarray:
    x1, y1, x2, y2 = roi
    crop = np.asarray(image_rgb)[y1:y2, x1:x2]
    mask_crop = normalize_edit_mask(edit_mask, image_rgb.shape)[y1:y2, x1:x2]
    excluded_crop = normalize_edit_mask(
        background_exclude_mask,
        image_rgb.shape,
    )[y1:y2, x1:x2]
    protect = _bubble_border_protect_mask(mask_crop.shape, width=4)
    bg_pixels = crop[
        (mask_crop <= 0)
        & (protect <= 0)
        & (excluded_crop <= 0)
    ]
    if bg_pixels.size == 0:
        return bg_pixels
    gray = _to_gray(bg_pixels.reshape((-1, 1, bg_pixels.shape[-1]))).reshape(-1) if crop.ndim == 3 else bg_pixels.reshape(-1)
    low = float(np.percentile(gray, 15))
    high = float(np.percentile(gray, 85))
    keep = (gray >= low) & (gray <= high)
    if np.count_nonzero(keep) < 8:
        return bg_pixels
    return bg_pixels[keep]


def _bubble_roi_background_metrics(
    image_rgb: np.ndarray,
    edit_mask: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    background_exclude_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    x1, y1, x2, y2 = roi
    crop = np.asarray(image_rgb)[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((0,), dtype=np.uint8), np.zeros((0, 0), dtype=bool), 1.0
    mask_crop = normalize_edit_mask(edit_mask, image_rgb.shape)[y1:y2, x1:x2]
    excluded_crop = normalize_edit_mask(
        background_exclude_mask,
        image_rgb.shape,
    )[y1:y2, x1:x2]
    protect = _bubble_border_protect_mask(mask_crop.shape, width=4)
    gray = _to_gray(crop)
    bg_region = (
        (mask_crop <= 0)
        & (protect <= 0)
        & (excluded_crop <= 0)
    )
    bg_pixels = gray[bg_region]
    if bg_pixels.size == 0:
        return bg_pixels, bg_region, 1.0
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges[bg_region])) / float(max(1, int(np.count_nonzero(bg_region))))
    horizontal_pairs = bg_region[:, 1:] & bg_region[:, :-1]
    vertical_pairs = bg_region[1:, :] & bg_region[:-1, :]
    horizontal_jumps = np.abs(gray[:, 1:].astype(np.int16) - gray[:, :-1].astype(np.int16)) >= 18
    vertical_jumps = np.abs(gray[1:, :].astype(np.int16) - gray[:-1, :].astype(np.int16)) >= 18
    pair_count = int(np.count_nonzero(horizontal_pairs)) + int(np.count_nonzero(vertical_pairs))
    jump_count = int(np.count_nonzero(horizontal_jumps & horizontal_pairs)) + int(np.count_nonzero(vertical_jumps & vertical_pairs))
    texture_density = float(jump_count) / float(max(1, pair_count))
    edge_density = max(edge_density, texture_density)
    return bg_pixels, bg_region, edge_density


def _should_use_bubble_roi_flat_fill(
    image_rgb: np.ndarray,
    edit_mask: np.ndarray,
    bubble_roi: tuple[int, int, int, int] | None,
    *,
    background_exclude_mask: np.ndarray | None = None,
) -> bool:
    if bubble_roi is None:
        return False
    bg_pixels = _trimmed_background_pixels(
        image_rgb,
        edit_mask,
        bubble_roi,
        background_exclude_mask=background_exclude_mask,
    )
    if bg_pixels.size == 0:
        return False
    gray = (
        _to_gray(bg_pixels.reshape((-1, 1, bg_pixels.shape[-1]))).reshape(-1)
        if bg_pixels.ndim == 2
        else bg_pixels.reshape(-1)
    )
    if gray.size == 0:
        return False
    iqr = float(np.percentile(gray, 75) - np.percentile(gray, 25))
    spread = float(np.percentile(gray, 90) - np.percentile(gray, 10))
    _bg_pixels, _bg_region, edge_density = _bubble_roi_background_metrics(
        image_rgb,
        edit_mask,
        bubble_roi,
        background_exclude_mask=background_exclude_mask,
    )
    return iqr <= 10.0 and spread <= 28.0 and edge_density <= 0.08


def _should_use_bubble_roi_gradient_fill(
    image_rgb: np.ndarray,
    edit_mask: np.ndarray,
    bubble_roi: tuple[int, int, int, int] | None,
    *,
    background_exclude_mask: np.ndarray | None = None,
) -> bool:
    if bubble_roi is None:
        return False
    bg_pixels, bg_region, edge_density = _bubble_roi_background_metrics(
        image_rgb,
        edit_mask,
        bubble_roi,
        background_exclude_mask=background_exclude_mask,
    )
    if bg_pixels.size < 64 or np.count_nonzero(bg_region) < 64:
        return False
    return edge_density <= 0.09


def _bubble_roi_median_fill(
    image_rgb: np.ndarray,
    edit_mask: np.ndarray,
    bubble_roi: tuple[int, int, int, int],
    *,
    background_exclude_mask: np.ndarray | None = None,
) -> np.ndarray:
    mask = normalize_edit_mask(edit_mask, image_rgb.shape)
    output = np.asarray(image_rgb).copy()
    bg_pixels = _trimmed_background_pixels(
        image_rgb,
        mask,
        bubble_roi,
        background_exclude_mask=background_exclude_mask,
    )
    if bg_pixels.size == 0:
        return _local_median_fill(
            image_rgb,
            mask,
            sample_roi=bubble_roi,
            background_exclude_mask=background_exclude_mask,
        )
    fill_value = np.median(bg_pixels, axis=0)
    if output.ndim == 2:
        output[mask > 0] = np.uint8(np.clip(round(float(fill_value)), 0, 255))
    else:
        output[mask > 0] = np.clip(np.round(fill_value), 0, 255).astype(output.dtype)
    return composite_with_edit_mask(image_rgb, output, mask)


def _bubble_roi_gradient_fill(
    image_rgb: np.ndarray,
    edit_mask: np.ndarray,
    bubble_roi: tuple[int, int, int, int],
    *,
    background_exclude_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mask = normalize_edit_mask(edit_mask, image_rgb.shape)
    output = np.asarray(image_rgb).copy()
    filled_mask = np.zeros(mask.shape, dtype=np.uint8)
    x1, y1, x2, y2 = bubble_roi
    crop = output[y1:y2, x1:x2].copy()
    mask_crop = mask[y1:y2, x1:x2]
    excluded_crop = normalize_edit_mask(
        background_exclude_mask,
        image_rgb.shape,
    )[y1:y2, x1:x2]
    if crop.size == 0 or not np.any(mask_crop):
        return composite_with_edit_mask(image_rgb, output, mask), filled_mask

    protect = _bubble_border_protect_mask(mask_crop.shape, width=5)
    gray = _to_gray(crop)

    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (mask_crop > 0).astype(np.uint8),
        8,
        cv2.CV_32S,
    )
    if labels_count <= 1:
        return composite_with_edit_mask(image_rgb, output, mask), filled_mask

    for label_idx in range(1, labels_count):
        x, y, w, h, area = stats[label_idx]
        if int(area) <= 0 or w <= 0 or h <= 0:
            continue
        component = np.zeros_like(mask_crop, dtype=np.uint8)
        component[y:y + h, x:x + w][labels[y:y + h, x:x + w] == label_idx] = 255
        fit_region = np.zeros_like(mask_crop, dtype=bool)
        for radius in (10, 16, 24, 32):
            ring = _ring_mask(component, radius=radius)
            candidate_region = (
                (ring > 0)
                & (mask_crop <= 0)
                & (protect <= 0)
                & (excluded_crop <= 0)
            )
            candidate_values = gray[candidate_region]
            if candidate_values.size >= max(32, min(256, int(area) // 2)):
                low = float(np.percentile(candidate_values, 15))
                high = float(np.percentile(candidate_values, 85))
                fit_region = candidate_region & (gray >= low) & (gray <= high)
                if np.count_nonzero(fit_region) >= 32:
                    break
                fit_region = candidate_region
                break
        yy, xx = np.nonzero(fit_region)
        if yy.size < 32:
            continue
        if yy.size > 3000:
            sample_indices = np.linspace(0, yy.size - 1, 3000).astype(np.int32)
            yy = yy[sample_indices]
            xx = xx[sample_indices]

        target_y, target_x = np.nonzero(component > 0)
        design = np.stack(
            [
                xx.astype(np.float64),
                yy.astype(np.float64),
                np.ones_like(xx, dtype=np.float64),
            ],
            axis=1,
        )
        target_design = np.stack(
            [
                target_x.astype(np.float64),
                target_y.astype(np.float64),
                np.ones_like(target_x, dtype=np.float64),
            ],
            axis=1,
        )
        if crop.ndim == 2:
            coeffs, *_ = np.linalg.lstsq(design, crop[yy, xx].astype(np.float64), rcond=None)
            crop[target_y, target_x] = np.clip(np.round(target_design @ coeffs), 0, 255).astype(crop.dtype)
        else:
            channel_count = min(3, crop.shape[2])
            for channel in range(channel_count):
                coeffs, *_ = np.linalg.lstsq(design, crop[yy, xx, channel].astype(np.float64), rcond=None)
                crop[target_y, target_x, channel] = np.clip(np.round(target_design @ coeffs), 0, 255).astype(crop.dtype)
        filled_mask[y1:y2, x1:x2][component > 0] = 255
    output[y1:y2, x1:x2] = crop
    return composite_with_edit_mask(image_rgb, output, mask), filled_mask


def _telea_fill(
    image_rgb: np.ndarray,
    edit_mask: np.ndarray,
    *,
    radius: int = 3,
    bubble_roi: tuple[int, int, int, int] | None = None,
    background_exclude_mask: np.ndarray | None = None,
) -> np.ndarray:
    mask = normalize_edit_mask(edit_mask, image_rgb.shape)
    if not np.any(mask):
        return np.asarray(image_rgb).copy()
    image = np.asarray(image_rgb)
    excluded = normalize_edit_mask(background_exclude_mask, image_rgb.shape)
    sampling_image = image.copy()
    normalized_bubble_roi = normalize_xyxy(bubble_roi, image_rgb.shape)
    if normalized_bubble_roi is not None:
        x1, y1, x2, y2 = normalized_bubble_roi
        mask_crop = mask[y1:y2, x1:x2]
        excluded_crop = excluded[y1:y2, x1:x2]
        sampling_crop = sampling_image[y1:y2, x1:x2]
        if np.any(excluded_crop):
            border = _bubble_border_protect_mask(mask_crop.shape, width=4)
            local_samples = sampling_crop[
                (mask_crop <= 0)
                & (excluded_crop <= 0)
                & (border <= 0)
            ]
            if local_samples.size == 0:
                return image.copy()
            local_fill = np.median(local_samples, axis=0)
            if sampling_crop.ndim == 2:
                sampling_crop[excluded_crop > 0] = np.uint8(
                    np.clip(round(float(local_fill)), 0, 255)
                )
            else:
                sampling_crop[excluded_crop > 0] = np.clip(
                    np.round(local_fill),
                    0,
                    255,
                ).astype(sampling_crop.dtype)
        if image.ndim == 3 and image.shape[2] > 3:
            filled_crop = sampling_crop.copy()
            filled_crop[:, :, :3] = cv2.inpaint(
                sampling_crop[:, :, :3],
                mask_crop,
                max(1, int(radius)),
                cv2.INPAINT_TELEA,
            )
        else:
            filled_crop = cv2.inpaint(
                sampling_crop,
                mask_crop,
                max(1, int(radius)),
                cv2.INPAINT_TELEA,
            )
        filled = image.copy()
        filled[y1:y2, x1:x2] = filled_crop
        return composite_with_edit_mask(image_rgb, filled, mask)
    if np.any(excluded):
        sampling_exclusions = np.where(
            (excluded > 0) | (mask > 0),
            255,
            0,
        ).astype(np.uint8)
        sampling_image = _local_median_fill(
            image,
            excluded,
            background_exclude_mask=sampling_exclusions,
        )
    if image.ndim == 3 and image.shape[2] > 3:
        rgb = sampling_image[:, :, :3]
        filled_rgb = cv2.inpaint(rgb, mask, max(1, int(radius)), cv2.INPAINT_TELEA)
        filled = image.copy()
        filled[:, :, :3] = filled_rgb
    else:
        filled = cv2.inpaint(
            sampling_image,
            mask,
            max(1, int(radius)),
            cv2.INPAINT_TELEA,
        )
    return composite_with_edit_mask(image_rgb, filled, mask)


def _fill_bubble_mask(
    image_rgb: np.ndarray,
    edit_mask: np.ndarray,
    *,
    bubble_roi: tuple[int, int, int, int] | None = None,
    background_exclude_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    mask = normalize_edit_mask(edit_mask, image_rgb.shape)
    excluded = normalize_edit_mask(background_exclude_mask, image_rgb.shape)
    usable_background = (mask <= 0) & (excluded <= 0)
    normalized_bubble_roi = normalize_xyxy(bubble_roi, image_rgb.shape)
    if normalized_bubble_roi is not None:
        x1, y1, x2, y2 = normalized_bubble_roi
        mask_crop = mask[y1:y2, x1:x2]
        excluded_crop = excluded[y1:y2, x1:x2]
        usable_crop = usable_background[y1:y2, x1:x2]
        usable_crop = usable_crop & (
            _bubble_border_protect_mask(usable_crop.shape, width=4) <= 0
        )
        has_usable_background = bool(np.any(usable_crop))
        if has_usable_background:
            component_count, component_labels, component_stats, centroids = (
                cv2.connectedComponentsWithStats(
                    (mask_crop > 0).astype(np.uint8),
                    8,
                    cv2.CV_32S,
                )
            )
            ring_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (21, 21),
                (10, 10),
            )
            for component_index in range(1, component_count):
                component_x, component_y, component_width, component_height, area = (
                    int(value) for value in component_stats[component_index]
                )
                if area <= 0 or component_width <= 0 or component_height <= 0:
                    continue
                sample_x1 = max(0, component_x - 10)
                sample_y1 = max(0, component_y - 10)
                sample_x2 = min(
                    mask_crop.shape[1],
                    component_x + component_width + 10,
                )
                sample_y2 = min(
                    mask_crop.shape[0],
                    component_y + component_height + 10,
                )
                component = np.where(
                    component_labels[
                        sample_y1:sample_y2,
                        sample_x1:sample_x2,
                    ]
                    == component_index,
                    255,
                    0,
                ).astype(np.uint8)
                local_ring = np.where(
                    (cv2.dilate(component, ring_kernel, iterations=1) > 0)
                    & (component <= 0),
                    255,
                    0,
                ).astype(np.uint8)
                local_samples = (
                    (local_ring > 0)
                    & usable_crop[
                        sample_y1:sample_y2,
                        sample_x1:sample_x2,
                    ]
                )
                sample_y, sample_x = np.where(local_samples)
                required_sample_count = max(
                    32,
                    min(256, area // 2),
                )
                if sample_y.size < required_sample_count:
                    has_usable_background = False
                    break
                sample_x = sample_x + sample_x1
                sample_y = sample_y + sample_y1
                center_x = float(centroids[component_index, 0])
                center_y = float(centroids[component_index, 1])
                if not (
                    (
                        np.any(sample_x < center_x)
                        and np.any(sample_x > center_x)
                    )
                    or (
                        np.any(sample_y < center_y)
                        and np.any(sample_y > center_y)
                    )
                ):
                    has_usable_background = False
                    break
    else:
        has_usable_background = bool(np.any(usable_background))
    if np.any(mask) and not has_usable_background:
        return (
            np.asarray(image_rgb).copy(),
            ERASE_MODE_BUBBLE_LAMA_FALLBACK,
        )
    if _should_use_bubble_roi_flat_fill(
        image_rgb,
        edit_mask,
        bubble_roi,
        background_exclude_mask=background_exclude_mask,
    ):
        return (
            _bubble_roi_median_fill(
                image_rgb,
                edit_mask,
                bubble_roi,
                background_exclude_mask=background_exclude_mask,
            ),
            ERASE_MODE_BUBBLE_FLAT_FILL,
        )
    if _should_use_bubble_roi_gradient_fill(
        image_rgb,
        edit_mask,
        bubble_roi,
        background_exclude_mask=background_exclude_mask,
    ):
        gradient_filled, gradient_filled_mask = _bubble_roi_gradient_fill(
            image_rgb,
            edit_mask,
            bubble_roi,
            background_exclude_mask=background_exclude_mask,
        )
        if np.all(gradient_filled_mask[mask > 0] > 0):
            return (
                gradient_filled,
                ERASE_MODE_BUBBLE_GRADIENT_FILL,
            )
    if _should_use_flat_fill(
        image_rgb,
        edit_mask,
        background_exclude_mask=background_exclude_mask,
    ):
        return (
            _local_median_fill(
                image_rgb,
                edit_mask,
                sample_roi=bubble_roi,
                background_exclude_mask=background_exclude_mask,
            ),
            ERASE_MODE_BUBBLE_FLAT_FILL,
        )
    return (
        _telea_fill(
            image_rgb,
            edit_mask,
            radius=2,
            bubble_roi=bubble_roi,
            background_exclude_mask=background_exclude_mask,
        ),
        ERASE_MODE_BUBBLE_TELEA,
    )


def fill_bubble_edit_mask(
    image_rgb: np.ndarray,
    edit_mask: np.ndarray,
    *,
    bubble_roi: tuple[int, int, int, int] | None = None,
    background_exclude_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    return _fill_bubble_mask(
        image_rgb,
        edit_mask,
        bubble_roi=bubble_roi,
        background_exclude_mask=background_exclude_mask,
    )


def erase_text_bubble_regions(
    original_image: np.ndarray,
    current_cleaned: np.ndarray,
    source_mask: np.ndarray,
    blocks: list,
    config=None,
    *,
    protected_edit_mask: np.ndarray | None = None,
) -> BubbleEraseResult:
    if original_image is None or current_cleaned is None:
        empty_shape = (0, 0) if original_image is None else original_image.shape[:2]
        empty_mask = np.zeros(empty_shape, dtype=np.uint8)
        return BubbleEraseResult(
            image=current_cleaned,
            edit_mask=empty_mask,
            fallback_mask=empty_mask.copy(),
            expanded_bubble_mask=empty_mask.copy(),
            stats={
                "applied": False,
                "block_count": 0,
                "applied_block_count": 0,
                "fallback_block_count": 0,
                "edit_pixel_count": 0,
                "fallback_pixel_count": 0,
                "changed_outside_edit_mask_pixel_count": 0,
                "blocks": [],
            },
        )

    result = np.asarray(current_cleaned).copy()
    source = normalize_edit_mask(source_mask, original_image.shape)
    protected = normalize_edit_mask(protected_edit_mask, original_image.shape)
    union_edit_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    fallback_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    bubble_roi_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    block_entries: list[dict] = []
    applied_blocks = 0
    fallback_blocks = 0

    for index, block in enumerate(list(blocks or [])):
        block_started = perf_counter()
        if getattr(block, "text_class", "") != "text_bubble":
            if getattr(block, "text_class", "") == "text_free":
                set_block_erase_metadata(block, BubbleEraseBlockStats(mode=ERASE_MODE_TEXT_FREE_LAMA))
            continue

        bubble_roi = normalize_xyxy(getattr(block, "bubble_xyxy", None), original_image.shape)
        line_art_context = None
        source_crop = None
        trusted_source_crop = None
        trusted_text_prior_available = False
        text_anchor_overlaps_bubble = False
        if bubble_roi is not None:
            x1, y1, x2, y2 = bubble_roi
            bubble_roi_mask[y1:y2, x1:x2] = 255
            source_crop = source[y1:y2, x1:x2]
            semantic_text_prior = build_text_prior_mask(
                original_image,
                block,
                bubble_roi,
                dilate_iterations=2,
            )
            text_anchor = resolve_inpaint_text_xyxy(
                block,
                original_image.shape,
            )
            if text_anchor is not None:
                ax1, ay1, ax2, ay2 = text_anchor
                clipped_x1 = max(x1, ax1)
                clipped_y1 = max(y1, ay1)
                clipped_x2 = min(x2, ax2)
                clipped_y2 = min(y2, ay2)
                if clipped_x2 > clipped_x1 and clipped_y2 > clipped_y1:
                    text_anchor_overlaps_bubble = True
                    semantic_text_prior[
                        clipped_y1 - y1:clipped_y2 - y1,
                        clipped_x1 - x1:clipped_x2 - x1,
                    ] = 255
            trusted_text_prior_available = bool(
                text_anchor is not None
                and text_anchor_overlaps_bubble
                and np.any(semantic_text_prior)
                and int(
                    getattr(block, "mask_actual_pixel_count", 0) or 0
                ) > 0
            )
            context_source_crop = source_crop
            if trusted_text_prior_available:
                context_source_crop = np.where(
                    (source_crop > 0) & (semantic_text_prior > 0),
                    255,
                    0,
                ).astype(np.uint8)
                trusted_source_crop = context_source_crop.copy()
            line_art_context = _build_bubble_line_art_context(
                np.asarray(original_image)[y1:y2, x1:x2],
                context_source_crop,
                text_prior_mask=semantic_text_prior,
            )

        edit_mask, mask_stats = build_bubble_residual_edit_mask(
            original_image,
            source,
            block,
            line_art_context=line_art_context,
        )
        raw_edit_mask = edit_mask.copy()
        edit_mask = np.where(
            (edit_mask > 0) & (protected <= 0),
            255,
            0,
        ).astype(np.uint8)
        cap_unavailable = (
            line_art_context is None
            or line_art_context.interior_cap is None
        )
        capless_safe_source_crop = None
        capless_source_partially_suppressed = False
        if (
            cap_unavailable
            and line_art_context is not None
            and bubble_roi is not None
        ):
            if trusted_text_prior_available:
                capless_safe_source_crop = trusted_source_crop
            else:
                capless_safe_source_crop = _capless_safe_source_seed_mask(
                    line_art_context.source_residual_seed_mask
                )
                capless_source_partially_suppressed = bool(
                    np.any(capless_safe_source_crop)
                    and np.any(
                        (line_art_context.source_residual_seed_mask > 0)
                        & (capless_safe_source_crop <= 0)
                    )
                )
        texture_intrusion = bool(
            line_art_context is not None
            and line_art_context.texture_field_detected
        )
        ambiguous_structure = bool(
            line_art_context is not None
            and line_art_context.ambiguous_structure_near_source
        )
        if (texture_intrusion or ambiguous_structure) and bubble_roi is not None:
            safe_source_crop = line_art_context.source_residual_seed_mask
            if (
                trusted_text_prior_available
                and trusted_source_crop is not None
            ):
                safe_source_crop = trusted_source_crop
            elif capless_safe_source_crop is not None:
                safe_source_crop = capless_safe_source_crop
            texture_fallback_mask = np.zeros(
                original_image.shape[:2],
                dtype=np.uint8,
            )
            texture_fallback_mask[y1:y2, x1:x2] = np.where(
                (safe_source_crop > 0)
                & (protected[y1:y2, x1:x2] <= 0),
                255,
                0,
            ).astype(np.uint8)
            texture_fallback_pixels = mask_pixel_count(
                texture_fallback_mask
            )
            if texture_fallback_pixels > 0:
                fallback_blocks += 1
                fallback_mask = np.where(
                    (fallback_mask > 0)
                    | (texture_fallback_mask > 0),
                    255,
                    0,
                ).astype(np.uint8)
                texture_reason = (
                    "microtexture_source_seed_partially_suppressed"
                    if (
                        texture_intrusion
                        and capless_source_partially_suppressed
                    )
                    else (
                        "microtexture_intrusion"
                        if texture_intrusion
                        else "text_prior_unavailable_structure_ambiguous"
                    )
                )
            else:
                texture_reason = (
                    "microtexture_source_seed_unavailable"
                    if texture_intrusion
                    else "text_prior_unavailable_source_seed_unavailable"
                )
            block_stats = BubbleEraseBlockStats(
                mode=(
                    ERASE_MODE_BUBBLE_LAMA_FALLBACK
                    if texture_fallback_pixels > 0
                    else ERASE_MODE_BUBBLE_SKIPPED
                ),
                edit_pixel_count=texture_fallback_pixels,
                protect_pixel_count=mask_stats.protect_pixel_count,
                skipped_reason=texture_reason,
            )
            set_block_erase_metadata(block, block_stats)
            elapsed_seconds = float(perf_counter() - block_started)
            block_entries.append(
                {
                    "index": index,
                    "mode": block_stats.mode,
                    "edit_pixel_count": block_stats.edit_pixel_count,
                    "protect_pixel_count": block_stats.protect_pixel_count,
                    "skipped_reason": block_stats.skipped_reason,
                    "elapsed_seconds": elapsed_seconds,
                }
            )
            continue
        line_art_intrusion = False
        if line_art_context is not None and bubble_roi is not None:
            edit_crop = edit_mask[y1:y2, x1:x2]
            line_art_intrusion = (
                _mask_near_line_art(
                    source_crop,
                    line_art_context.line_protect_mask,
                )
                or _mask_near_line_art(
                    edit_crop,
                    line_art_context.line_protect_mask,
                )
            )
        if cap_unavailable or line_art_intrusion:
            if cap_unavailable:
                fallback_edit_mask = np.zeros(
                    original_image.shape[:2],
                    dtype=np.uint8,
                )
                fallback_stats = mask_stats
                if capless_safe_source_crop is not None:
                    fallback_edit_mask[y1:y2, x1:x2] = (
                        capless_safe_source_crop
                    )
            else:
                fallback_edit_mask, fallback_stats = (
                    build_bubble_residual_edit_mask(
                        original_image,
                        source,
                        block,
                        protect_line_art=False,
                        line_art_context=line_art_context,
                    )
                )
            if (
                not cap_unavailable
                and line_art_context is not None
                and bubble_roi is not None
            ):
                fallback_crop = fallback_edit_mask[y1:y2, x1:x2]
                fallback_protect = np.where(
                    line_art_context.line_protect_mask > 0,
                    255,
                    0,
                ).astype(np.uint8)
                fallback_edit_mask[y1:y2, x1:x2] = np.where(
                    (fallback_crop > 0)
                    & (
                        (fallback_protect <= 0)
                        | (line_art_context.source_residual_seed_mask > 0)
                    ),
                    255,
                    0,
                ).astype(np.uint8)
            fallback_edit_mask = np.where(
                (fallback_edit_mask > 0) & (protected <= 0),
                255,
                0,
            ).astype(np.uint8)
            if (
                not cap_unavailable
                and not np.any(fallback_edit_mask)
                and np.any(edit_mask)
            ):
                fallback_edit_mask = edit_mask
            if (
                not np.any(fallback_edit_mask)
                and bubble_roi is not None
                and source_crop is not None
                and not cap_unavailable
            ):
                safe_source_crop = line_art_context.source_residual_seed_mask
                fallback_edit_mask[y1:y2, x1:x2] = np.where(
                    (safe_source_crop > 0)
                    & (protected[y1:y2, x1:x2] <= 0),
                    255,
                    0,
                ).astype(np.uint8)
            if np.any(fallback_edit_mask):
                fallback_blocks += 1
                fallback_mask = np.where((fallback_mask > 0) | (fallback_edit_mask > 0), 255, 0).astype(np.uint8)
                fallback_reason = (
                    "bubble_interior_cap_source_seed_partially_suppressed"
                    if capless_source_partially_suppressed
                    else (
                        "bubble_interior_cap_unavailable"
                        if cap_unavailable
                        else "line_art_intrusion"
                    )
                )
                block_stats = BubbleEraseBlockStats(
                    mode=ERASE_MODE_BUBBLE_LAMA_FALLBACK,
                    edit_pixel_count=mask_pixel_count(fallback_edit_mask),
                    protect_pixel_count=max(mask_stats.protect_pixel_count, fallback_stats.protect_pixel_count),
                    skipped_reason=fallback_reason,
                )
                set_block_erase_metadata(block, block_stats)
                elapsed_seconds = float(perf_counter() - block_started)
                block_entries.append(
                    {
                        "index": index,
                        "mode": block_stats.mode,
                        "edit_pixel_count": block_stats.edit_pixel_count,
                        "protect_pixel_count": block_stats.protect_pixel_count,
                        "skipped_reason": block_stats.skipped_reason,
                        "elapsed_seconds": elapsed_seconds,
                    }
                )
                continue
            if bubble_roi is not None:
                fallback_reason = (
                    "bubble_interior_cap_source_seed_unavailable"
                    if cap_unavailable
                    else "line_art_source_seed_unavailable"
                )
                block_stats = BubbleEraseBlockStats(
                    mode=ERASE_MODE_BUBBLE_SKIPPED,
                    edit_pixel_count=0,
                    protect_pixel_count=max(
                        mask_stats.protect_pixel_count,
                        fallback_stats.protect_pixel_count,
                    ),
                    skipped_reason=fallback_reason,
                )
                set_block_erase_metadata(block, block_stats)
                elapsed_seconds = float(perf_counter() - block_started)
                block_entries.append(
                    {
                        "index": index,
                        "mode": block_stats.mode,
                        "edit_pixel_count": 0,
                        "protect_pixel_count": block_stats.protect_pixel_count,
                        "skipped_reason": block_stats.skipped_reason,
                        "elapsed_seconds": elapsed_seconds,
                    }
                )
                continue

        if np.any(raw_edit_mask) and not np.any(edit_mask):
            source_pixel_count = (
                0
                if source_crop is None
                else mask_pixel_count(source_crop)
            )
            suppressed_pixel_count = int(
                np.count_nonzero(
                    (raw_edit_mask > 0) & (protected > 0)
                )
            )
            skipped_reason = (
                "bubble_protected_source_seed_unavailable"
                if source_pixel_count > 0
                else "empty_seed"
            )
            block_stats = BubbleEraseBlockStats(
                mode=ERASE_MODE_BUBBLE_SKIPPED,
                edit_pixel_count=0,
                protect_pixel_count=max(
                    mask_stats.protect_pixel_count,
                    suppressed_pixel_count,
                ),
                skipped_reason=skipped_reason,
            )
            set_block_erase_metadata(block, block_stats)
            elapsed_seconds = float(perf_counter() - block_started)
            block_entries.append(
                {
                    "index": index,
                    "mode": block_stats.mode,
                    "edit_pixel_count": 0,
                    "protect_pixel_count": block_stats.protect_pixel_count,
                    "skipped_reason": block_stats.skipped_reason,
                    "elapsed_seconds": elapsed_seconds,
                }
            )
            continue

        if not np.any(edit_mask):
            source_pixel_count = (
                0
                if source_crop is None
                else mask_pixel_count(source_crop)
            )
            block_stats = mask_stats
            if source_pixel_count > 0:
                block_stats = BubbleEraseBlockStats(
                    mode=ERASE_MODE_BUBBLE_SKIPPED,
                    edit_pixel_count=0,
                    protect_pixel_count=mask_stats.protect_pixel_count,
                    skipped_reason="bubble_residual_source_seed_unavailable",
                )
            set_block_erase_metadata(block, block_stats)
            elapsed_seconds = float(perf_counter() - block_started)
            block_entries.append(
                {
                    "index": index,
                    "mode": block_stats.mode,
                    "edit_pixel_count": 0,
                    "protect_pixel_count": block_stats.protect_pixel_count,
                    "skipped_reason": block_stats.skipped_reason,
                    "elapsed_seconds": elapsed_seconds,
                }
            )
            continue

        filled, mode = _fill_bubble_mask(
            original_image,
            edit_mask,
            bubble_roi=bubble_roi,
            background_exclude_mask=protected,
        )
        if mode == ERASE_MODE_BUBBLE_LAMA_FALLBACK:
            fallback_edit_mask = np.where(
                (edit_mask > 0) & (protected <= 0),
                255,
                0,
            ).astype(np.uint8)
            if np.any(fallback_edit_mask):
                fallback_blocks += 1
                fallback_mask = np.where(
                    (fallback_mask > 0) | (fallback_edit_mask > 0),
                    255,
                    0,
                ).astype(np.uint8)
            block_stats = BubbleEraseBlockStats(
                mode=ERASE_MODE_BUBBLE_LAMA_FALLBACK,
                edit_pixel_count=mask_pixel_count(fallback_edit_mask),
                protect_pixel_count=mask_stats.protect_pixel_count,
                skipped_reason="bubble_background_sample_unavailable",
            )
            set_block_erase_metadata(block, block_stats)
            elapsed_seconds = float(perf_counter() - block_started)
            block_entries.append(
                {
                    "index": index,
                    "mode": block_stats.mode,
                    "edit_pixel_count": block_stats.edit_pixel_count,
                    "protect_pixel_count": block_stats.protect_pixel_count,
                    "skipped_reason": block_stats.skipped_reason,
                    "elapsed_seconds": elapsed_seconds,
                }
            )
            continue
        result = composite_with_edit_mask(result, filled, edit_mask)
        union_edit_mask = np.where((union_edit_mask > 0) | (edit_mask > 0), 255, 0).astype(np.uint8)
        applied_blocks += 1
        block_stats = BubbleEraseBlockStats(
            mode=mode,
            edit_pixel_count=mask_pixel_count(edit_mask),
            protect_pixel_count=mask_stats.protect_pixel_count,
        )
        set_block_erase_metadata(block, block_stats)
        elapsed_seconds = float(perf_counter() - block_started)
        block_entries.append(
            {
                "index": index,
                "mode": mode,
                "edit_pixel_count": block_stats.edit_pixel_count,
                "protect_pixel_count": block_stats.protect_pixel_count,
                "skipped_reason": "",
                "elapsed_seconds": elapsed_seconds,
            }
        )

    result = composite_with_edit_mask(current_cleaned, result, union_edit_mask)
    outside_changed = count_changed_outside_edit_mask(current_cleaned, result, union_edit_mask)
    return BubbleEraseResult(
        image=result,
        edit_mask=union_edit_mask,
        fallback_mask=fallback_mask,
        expanded_bubble_mask=bubble_roi_mask,
        stats={
            "applied": bool(applied_blocks or fallback_blocks),
            "block_count": len(list(blocks or [])),
            "applied_block_count": applied_blocks,
            "fallback_block_count": fallback_blocks,
            "edit_pixel_count": mask_pixel_count(union_edit_mask),
            "fallback_pixel_count": mask_pixel_count(fallback_mask),
            "changed_outside_edit_mask_pixel_count": int(outside_changed),
            "blocks": block_entries,
        },
    )
