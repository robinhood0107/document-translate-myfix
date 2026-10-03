from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from types import MappingProxyType
from typing import Mapping

import cv2
import numpy as np

from .contracts import binary_mask, mask_sha256, tensor_sha256
from .semantic import TRANSLATE


SEEDLESS_ALLOWED_ROUTES = frozenset(
    {"clean", "translucent", "clean_translucent"}
)


@dataclass(frozen=True, slots=True)
class GlyphComponentRecord:
    ownership_component_id: int
    candidate_component_id: int
    bbox_xyxy: tuple[int, int, int, int]
    polarity: str
    accepted: bool
    reason: str
    detector_seed_pixel_count: int
    core_pixel_count: int
    effect_pixel_count: int
    protect_overlap_pixel_count: int
    protect_adjacency_pixel_count: int
    protect_contact_pixel_count: int


@dataclass(frozen=True, slots=True)
class GlyphOwnerRecord:
    ownership_component_id: int
    bbox_xyxy: tuple[int, int, int, int]
    ownership_pixel_count: int
    detector_seed_pixel_count: int
    accounted_seed_pixel_count: int
    candidate_component_count: int
    accepted_component_count: int
    rejected_component_count: int
    status: str
    reason: str


@dataclass(frozen=True, slots=True)
class GlyphRefinementResult:
    owned_detector_seed: np.ndarray
    effect_support: np.ndarray
    candidate_mask: np.ndarray
    glyph_core: np.ndarray
    glyph_effect: np.ndarray
    refined_mask: np.ndarray
    rejected_mask: np.ndarray
    hard_protect: np.ndarray
    component_records: tuple[GlyphComponentRecord, ...]
    owner_records: tuple[GlyphOwnerRecord, ...]
    provenance: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class SeedlessRoiEvidence:
    """Source/owner-bound OCR authority for the detector-zero B3 route."""

    owner_region_id: str
    owner_count: int
    authoritative_ocr: bool
    ocr_text: str
    ocr_script: str
    ocr_confidence: float | None
    ocr_provider: str
    processing_action: str
    route_class: str
    source_sha256: str
    ownership_mask_sha256: str
    hard_protect_mask_sha256: str
    ocr_provider_sha256: str
    seal_sha256: str

    @classmethod
    def seal(
        cls,
        source_image: np.ndarray,
        *,
        ocr_ownership: np.ndarray,
        hard_protect: np.ndarray,
        owner_region_id: str,
        owner_count: int,
        authoritative_ocr: bool,
        ocr_text: str,
        ocr_script: str,
        ocr_confidence: float | None,
        ocr_provider: str,
        processing_action: str,
        route_class: str,
    ) -> "SeedlessRoiEvidence":
        shape = _source_gray(source_image).shape
        ownership = binary_mask(ocr_ownership, shape)
        protect = binary_mask(hard_protect, shape)
        provider = str(ocr_provider).strip()
        payload: dict[str, object] = {
            "owner_region_id": str(owner_region_id).strip(),
            "owner_count": int(owner_count),
            "authoritative_ocr": authoritative_ocr is True,
            "ocr_text": str(ocr_text).strip(),
            "ocr_script": str(ocr_script).strip(),
            "ocr_confidence": ocr_confidence,
            "ocr_provider": provider,
            "processing_action": str(processing_action).strip().lower(),
            "route_class": str(route_class).strip().lower(),
            "source_sha256": tensor_sha256(np.asarray(source_image)),
            "ownership_mask_sha256": mask_sha256(ownership),
            "hard_protect_mask_sha256": mask_sha256(protect),
            "ocr_provider_sha256": _text_sha256(provider),
        }
        return cls(**payload, seal_sha256=_evidence_seal(payload))

    def payload(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "seal_sha256"
        }


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _evidence_seal(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_gray(source_image: np.ndarray) -> np.ndarray:
    source = np.asarray(source_image)
    if source.dtype != np.uint8:
        raise ValueError("source image must use uint8 pixels")
    if source.ndim == 2:
        return np.ascontiguousarray(source)
    if source.ndim != 3:
        raise ValueError("source image must be grayscale, BGR, or BGRA")
    if source.shape[2] == 3:
        return cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    if source.shape[2] == 4:
        return cv2.cvtColor(source, cv2.COLOR_BGRA2GRAY)
    raise ValueError("source image must be grayscale, BGR, or BGRA")


def _source_lab_l(source_image: np.ndarray) -> np.ndarray:
    source = np.asarray(source_image)
    if source.ndim == 2:
        return np.ascontiguousarray(source.astype(np.float32))
    return np.ascontiguousarray(
        cv2.cvtColor(source[..., :3], cv2.COLOR_BGR2LAB)[..., 0].astype(
            np.float32
        )
    )


def _ellipse(radius: int) -> np.ndarray:
    normalized = max(0, int(radius))
    size = (normalized * 2) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    normalized = binary_mask(mask)
    if radius <= 0:
        return normalized
    return np.ascontiguousarray(
        cv2.dilate(normalized, _ellipse(radius), iterations=1)
    )


def _protect_adjacency(protect: np.ndarray) -> np.ndarray:
    expanded = _dilate(protect, 1)
    return np.ascontiguousarray(
        np.where((expanded > 0) & (protect == 0), 255, 0).astype(np.uint8)
    )


def _normalized_local_background(
    luminance: np.ndarray,
    ownership: np.ndarray,
    *,
    sigma: float,
) -> np.ndarray:
    weights = (ownership > 0).astype(np.float32)
    weighted_source = luminance.astype(np.float32) * weights
    blurred_weights = cv2.GaussianBlur(
        weights,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    )
    blurred_source = cv2.GaussianBlur(
        weighted_source,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    )
    return np.divide(
        blurred_source,
        np.maximum(blurred_weights, np.float32(1e-4)),
        dtype=np.float32,
    )


def _noise_floor(
    residual: np.ndarray,
    ownership: np.ndarray,
    detector_seed: np.ndarray | None = None,
) -> float:
    background = ownership > 0
    if detector_seed is not None and np.any(detector_seed):
        background &= _dilate(detector_seed, 4) == 0
    values = residual[background]
    if values.size < 16:
        values = residual[ownership > 0]
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - median)))


def _polarity_name(
    component: np.ndarray,
    bright: np.ndarray,
    dark: np.ndarray,
) -> str:
    bright_count = int(np.count_nonzero(component & bright))
    dark_count = int(np.count_nonzero(component & dark))
    if bright_count and dark_count:
        return "mixed"
    if bright_count:
        return "bright"
    if dark_count:
        return "dark"
    return "low_contrast"


def _readonly_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    result = binary_mask(mask, shape)
    result.setflags(write=False)
    return result


def _build_result(
    *,
    owned_detector_seed: np.ndarray,
    effect_support: np.ndarray,
    candidate_mask: np.ndarray,
    glyph_core: np.ndarray,
    glyph_effect: np.ndarray,
    rejected_mask: np.ndarray,
    hard_protect: np.ndarray,
    component_records: tuple[GlyphComponentRecord, ...],
    owner_records: tuple[GlyphOwnerRecord, ...],
    provenance: Mapping[str, object],
) -> GlyphRefinementResult:
    shape = binary_mask(owned_detector_seed).shape
    seed = _readonly_mask(owned_detector_seed, shape)
    support = _readonly_mask(effect_support, shape)
    candidate = _readonly_mask(candidate_mask, shape)
    core = _readonly_mask(glyph_core, shape)
    effect = _readonly_mask(glyph_effect, shape)
    refined = _readonly_mask(cv2.bitwise_or(core, effect), shape)
    rejected = _readonly_mask(rejected_mask, shape)
    protect = _readonly_mask(hard_protect, shape)
    if np.any((effect > 0) & (support == 0)):
        raise AssertionError("glyph effect escaped explicit source support")
    if np.any((core > 0) & (effect > 0)):
        raise AssertionError("glyph core and effect masks overlap")
    hashes = {
        "owned_detector_seed_sha256": mask_sha256(seed),
        "effect_support_sha256": mask_sha256(support),
        "candidate_mask_sha256": mask_sha256(candidate),
        "glyph_core_sha256": mask_sha256(core),
        "glyph_effect_sha256": mask_sha256(effect),
        "refined_mask_sha256": mask_sha256(refined),
        "rejected_mask_sha256": mask_sha256(rejected),
        "hard_protect_sha256": mask_sha256(protect),
    }
    return GlyphRefinementResult(
        owned_detector_seed=seed,
        effect_support=support,
        candidate_mask=candidate,
        glyph_core=core,
        glyph_effect=effect,
        refined_mask=refined,
        rejected_mask=rejected,
        hard_protect=protect,
        component_records=component_records,
        owner_records=owner_records,
        provenance=MappingProxyType({**dict(provenance), **hashes}),
    )


def extract_roi_local_glyph_masks(
    source_image: np.ndarray,
    *,
    detector_seed: np.ndarray,
    ocr_ownership: np.ndarray,
    hard_protect: np.ndarray,
    detector_provider: str,
    ownership_provider: str,
    effect_support: np.ndarray | None = None,
    effect_support_provider: str = "",
    max_core_expansion_px: int = 4,
    max_effect_expansion_px: int = 2,
) -> GlyphRefinementResult:
    """Extract detector-seeded glyph pixels using ROI-local source cues.

    OCR geometry owns and bounds the search but never creates a pixel. Every
    output component contains a detector pixel. Outline/glow pixels also need
    explicit source-derived ``effect_support``. A component that overlaps or
    is 8-neighbor adjacent to protection is rejected whole.
    """

    gray = _source_gray(source_image)
    shape = gray.shape
    detector = binary_mask(detector_seed, shape)
    ownership = binary_mask(ocr_ownership, shape)
    protect = binary_mask(hard_protect, shape)
    protect_adjacent = _protect_adjacency(protect)
    detector_name = str(detector_provider).strip()
    ownership_name = str(ownership_provider).strip()
    if not detector_name or not ownership_name:
        raise ValueError("detector and ownership providers must not be empty")
    if effect_support is None:
        support = np.zeros(shape, dtype=np.uint8)
        support_name = "none"
    else:
        support = binary_mask(effect_support, shape)
        support_name = str(effect_support_provider).strip()
        if not support_name:
            raise ValueError(
                "effect support provider is required with an effect mask"
            )
    core_radius = int(max_core_expansion_px)
    effect_radius = int(max_effect_expansion_px)
    if core_radius < 0 or core_radius > 16:
        raise ValueError("max core expansion must be between 0 and 16 pixels")
    if effect_radius < 0 or effect_radius > 8:
        raise ValueError("max effect expansion must be between 0 and 8 pixels")

    owned_seed = np.where(
        (detector > 0) & (ownership > 0), 255, 0
    ).astype(np.uint8)
    discarded_seed_count = int(
        np.count_nonzero((detector > 0) & (ownership == 0))
    )
    candidate_page = np.zeros(shape, dtype=np.uint8)
    core_page = np.zeros(shape, dtype=np.uint8)
    effect_page = np.zeros(shape, dtype=np.uint8)
    rejected_page = np.zeros(shape, dtype=np.uint8)
    accounted_seed_page = np.zeros(shape, dtype=np.uint8)
    records: list[GlyphComponentRecord] = []
    owner_records: list[GlyphOwnerRecord] = []
    page_pixels = shape[0] * shape[1]
    scanned_pixels = page_pixels
    owner_count, owner_labels, owner_stats, _ = cv2.connectedComponentsWithStats(
        (ownership > 0).astype(np.uint8), 8, cv2.CV_32S
    )
    next_candidate_id = 1
    for owner_id in range(1, owner_count):
        x, y, width, height, area = (
            int(value) for value in owner_stats[owner_id]
        )
        if area <= 0:
            continue
        scanned_pixels += width * height
        owner_slice = (slice(y, y + height), slice(x, x + width))
        local_owner = owner_labels[owner_slice] == owner_id
        local_seed = (owned_seed[owner_slice] > 0) & local_owner
        owner_seed_count = int(np.count_nonzero(local_seed))
        record_start = len(records)
        owner_status = "skipped"
        owner_reason = "no_detector_seed"
        if owner_seed_count:
            local_gray = gray[owner_slice]
            owner_u8 = np.where(local_owner, 255, 0).astype(np.uint8)
            seed_u8 = np.where(local_seed, 255, 0).astype(np.uint8)
            sigma = max(2.0, min(6.0, min(width, height) / 12.0))
            background = _normalized_local_background(
                local_gray, owner_u8, sigma=sigma
            )
            residual = local_gray.astype(np.float32) - background
            noise = _noise_floor(residual, owner_u8, seed_u8)
            strength = float(
                np.percentile(np.abs(residual[local_seed]), 75.0)
            )
            if strength < 1.0:
                owner_status = "information_limited"
                owner_reason = "seed_contrast_unavailable"
            else:
                core_threshold = max(
                    2.0, (noise * 2.5) + 0.5, min(18.0, strength * 0.35)
                )
                effect_threshold = max(
                    1.0, (noise * 1.25) + 0.25, min(7.0, strength * 0.10)
                )
                bright_supported = bool(
                    np.any(local_seed & (residual >= effect_threshold))
                )
                dark_supported = bool(
                    np.any(local_seed & (residual <= -effect_threshold))
                )
                if not (bright_supported or dark_supported):
                    owner_status = "information_limited"
                    owner_reason = "seed_polarity_unavailable"
                else:
                    reach = _dilate(seed_u8, core_radius) > 0
                    raw_core = local_owner & reach & (
                        ((residual >= core_threshold) & bright_supported)
                        | ((residual <= -core_threshold) & dark_supported)
                    )
                    core_count, core_labels, core_stats, _ = (
                        cv2.connectedComponentsWithStats(
                            raw_core.astype(np.uint8), 8, cv2.CV_32S
                        )
                    )
                    seeded_core = np.zeros(raw_core.shape, dtype=bool)
                    for component_id in range(1, core_count):
                        cx, cy, cw, ch, component_area = (
                            int(value) for value in core_stats[component_id]
                        )
                        if component_area <= 0:
                            continue
                        scanned_pixels += cw * ch
                        component_slice = (
                            slice(cy, cy + ch),
                            slice(cx, cx + cw),
                        )
                        component = (
                            core_labels[component_slice] == component_id
                        )
                        if np.any(component & local_seed[component_slice]):
                            seeded_core[component_slice][component] = True
                    if not np.any(seeded_core):
                        owner_status = "information_limited"
                        owner_reason = "core_candidate_unavailable"
                    else:
                        effect_pool = (
                            local_owner
                            & (_dilate(
                                np.where(seeded_core, 255, 0).astype(np.uint8),
                                effect_radius,
                            ) > 0)
                            & (np.abs(residual) >= effect_threshold)
                            & (support[owner_slice] > 0)
                            & ~seeded_core
                        )
                        local_candidate = seeded_core | effect_pool
                        count, labels, stats, _ = cv2.connectedComponentsWithStats(
                            local_candidate.astype(np.uint8), 8, cv2.CV_32S
                        )
                        for component_id in range(1, count):
                            cx, cy, cw, ch, component_area = (
                                int(value) for value in stats[component_id]
                            )
                            if component_area <= 0:
                                continue
                            scanned_pixels += cw * ch
                            local_slice = (
                                slice(cy, cy + ch),
                                slice(cx, cx + cw),
                            )
                            global_slice = (
                                slice(y + cy, y + cy + ch),
                                slice(x + cx, x + cx + cw),
                            )
                            component = labels[local_slice] == component_id
                            seed_view = local_seed[local_slice]
                            core_view = seeded_core[local_slice]
                            seed_count = int(
                                np.count_nonzero(component & seed_view)
                            )
                            core_pixels = int(
                                np.count_nonzero(component & core_view)
                            )
                            overlap = int(
                                np.count_nonzero(
                                    component & (protect[global_slice] > 0)
                                )
                            )
                            adjacency = int(
                                np.count_nonzero(
                                    component
                                    & (protect_adjacent[global_slice] > 0)
                                )
                            )
                            accepted = True
                            reason = "accepted"
                            if seed_count == 0:
                                accepted = False
                                reason = "disconnected_from_detector_seed"
                            elif overlap:
                                accepted = False
                                reason = "exact_protect_overlap"
                            elif adjacency:
                                accepted = False
                                reason = "exact_protect_adjacency"
                            candidate_page[global_slice][component] = 255
                            if accepted:
                                core_page[global_slice][
                                    component & core_view
                                ] = 255
                                effect_page[global_slice][
                                    component & ~core_view
                                ] = 255
                            else:
                                rejected_page[global_slice][component] = 255
                            accounted_seed_page[global_slice][
                                component & seed_view
                            ] = 255
                            records.append(
                                GlyphComponentRecord(
                                    ownership_component_id=owner_id,
                                    candidate_component_id=next_candidate_id,
                                    bbox_xyxy=(
                                        x + cx,
                                        y + cy,
                                        x + cx + cw,
                                        y + cy + ch,
                                    ),
                                    polarity=_polarity_name(
                                        component,
                                        residual[local_slice]
                                        >= core_threshold,
                                        residual[local_slice]
                                        <= -core_threshold,
                                    ),
                                    accepted=accepted,
                                    reason=reason,
                                    detector_seed_pixel_count=seed_count,
                                    core_pixel_count=core_pixels,
                                    effect_pixel_count=(
                                        component_area - core_pixels
                                    ),
                                    protect_overlap_pixel_count=overlap,
                                    protect_adjacency_pixel_count=adjacency,
                                    protect_contact_pixel_count=(
                                        overlap + adjacency
                                    ),
                                )
                            )
                            next_candidate_id += 1
                        owner_status = "completed"
                        owner_reason = (
                            "candidate_components_evaluated"
                            if len(records) > record_start
                            else "candidate_component_unavailable"
                        )
        owner_rows = records[record_start:]
        accounted_count = int(
            np.count_nonzero(
                (accounted_seed_page[owner_slice] > 0) & local_seed
            )
        )
        owner_records.append(
            GlyphOwnerRecord(
                ownership_component_id=owner_id,
                bbox_xyxy=(x, y, x + width, y + height),
                ownership_pixel_count=area,
                detector_seed_pixel_count=owner_seed_count,
                accounted_seed_pixel_count=accounted_count,
                candidate_component_count=len(owner_rows),
                accepted_component_count=sum(row.accepted for row in owner_rows),
                rejected_component_count=sum(
                    not row.accepted for row in owner_rows
                ),
                status=owner_status,
                reason=owner_reason,
            )
        )

    refined = cv2.bitwise_or(core_page, effect_page)
    if np.any((refined > 0) & (ownership == 0)):
        raise AssertionError("glyph refinement escaped OCR ownership")
    if np.any((refined > 0) & ((protect > 0) | (protect_adjacent > 0))):
        raise AssertionError("glyph refinement contacts exact protection")
    if not np.any(owned_seed):
        status = "no_owned_detector_seed"
    elif np.any(refined):
        status = "completed"
    else:
        status = "completed_no_refined_pixels"
    provenance: dict[str, object] = {
        "schema_version": "glyph-refinement-v2",
        "status": status,
        "seed_mode": "detector_owned",
        "detector_provider": detector_name,
        "ownership_provider": ownership_name,
        "effect_support_provider": support_name,
        "max_core_expansion_px": core_radius,
        "max_effect_expansion_px": effect_radius,
        "owned_detector_seed_pixel_count": int(np.count_nonzero(owned_seed)),
        "discarded_detector_seed_pixel_count": discarded_seed_count,
        "unaccounted_seed_pixel_count": int(
            np.count_nonzero(
                (owned_seed > 0) & (accounted_seed_page == 0)
            )
        ),
        "candidate_pixel_count": int(np.count_nonzero(candidate_page)),
        "glyph_core_pixel_count": int(np.count_nonzero(core_page)),
        "glyph_effect_pixel_count": int(np.count_nonzero(effect_page)),
        "refined_pixel_count": int(np.count_nonzero(refined)),
        "accepted_component_count": sum(row.accepted for row in records),
        "rejected_component_count": sum(not row.accepted for row in records),
        "owner_component_count": len(owner_records),
        "silent_owner_count": sum(
            row.status != "completed" for row in owner_records
        ),
        "page_pixel_count": page_pixels,
        "component_bbox_scanned_pixel_count": scanned_pixels,
        "component_bbox_scan_ratio": scanned_pixels / max(1, page_pixels),
        "output_subset_of_ownership": bool(
            not np.any((refined > 0) & (ownership == 0))
        ),
        "effect_subset_of_support": bool(
            not np.any((effect_page > 0) & (support == 0))
        ),
        "effect_disjoint_from_core": bool(
            not np.any((core_page > 0) & (effect_page > 0))
        ),
        "no_seed_no_expansion": bool(
            np.any(owned_seed) or not np.any(refined)
        ),
    }
    return _build_result(
        owned_detector_seed=owned_seed,
        effect_support=support,
        candidate_mask=candidate_page,
        glyph_core=core_page,
        glyph_effect=effect_page,
        rejected_mask=rejected_page,
        hard_protect=protect,
        component_records=tuple(records),
        owner_records=tuple(owner_records),
        provenance=provenance,
    )


def _shifted_binary_correlation(
    values: np.ndarray,
    ownership: np.ndarray,
    *,
    dy: int,
    dx: int,
) -> float:
    height, width = values.shape
    y0_a = max(0, -dy)
    y1_a = min(height, height - dy)
    x0_a = max(0, -dx)
    x1_a = min(width, width - dx)
    if y1_a <= y0_a or x1_a <= x0_a:
        return 0.0
    y0_b, y1_b = y0_a + dy, y1_a + dy
    x0_b, x1_b = x0_a + dx, x1_a + dx
    owner_a = ownership[y0_a:y1_a, x0_a:x1_a]
    owner_b = ownership[y0_b:y1_b, x0_b:x1_b]
    valid = owner_a & owner_b
    if int(np.count_nonzero(valid)) < 16:
        return 0.0
    first = values[y0_a:y1_a, x0_a:x1_a][valid].astype(np.float32)
    second = values[y0_b:y1_b, x0_b:x1_b][valid].astype(np.float32)
    first_mean = float(np.mean(first))
    second_mean = float(np.mean(second))
    first_centered = first - first_mean
    second_centered = second - second_mean
    denominator = float(
        np.sqrt(
            np.mean(first_centered * first_centered)
            * np.mean(second_centered * second_centered)
        )
    )
    if denominator <= 1e-8:
        return 0.0
    return float(
        np.clip(
            np.mean(first_centered * second_centered) / denominator,
            -1.0,
            1.0,
        )
    )


def _seedless_periodicity_metrics(
    candidate: np.ndarray,
    ownership: np.ndarray,
) -> dict[str, object]:
    height, width = candidate.shape
    max_lag = min(16, (min(height, width) - 1) // 2)
    if max_lag < 6:
        return {
            "source_roi_periodicity_peak_correlation": 0.0,
            "source_roi_periodicity_pair_score": 0.0,
            "source_roi_periodicity_harmonic_correlation": 0.0,
            "source_roi_periodicity_lag_px": 0,
            "source_roi_periodicity_direction": "none",
        }
    directions = (
        ("horizontal", 0, 1),
        ("vertical", 1, 0),
        ("diagonal_down", 1, 1),
        ("diagonal_up", 1, -1),
    )
    peak = 0.0
    best_pair = 0.0
    best_harmonic = 0.0
    best_lag = 0
    best_direction = "none"
    for direction, unit_y, unit_x in directions:
        correlations: dict[int, float] = {}
        for lag in range(3, max_lag + 1):
            correlation = max(
                0.0,
                _shifted_binary_correlation(
                    candidate,
                    ownership,
                    dy=unit_y * lag,
                    dx=unit_x * lag,
                ),
            )
            correlations[lag] = correlation
            peak = max(peak, correlation)
        for lag in range(3, (max_lag // 2) + 1):
            first = correlations[lag]
            harmonic = correlations[lag * 2]
            pair_score = min(first, harmonic)
            if pair_score > best_pair:
                best_pair = pair_score
                best_harmonic = harmonic
                best_lag = lag
                best_direction = direction
    return {
        "source_roi_periodicity_peak_correlation": peak,
        "source_roi_periodicity_pair_score": best_pair,
        "source_roi_periodicity_harmonic_correlation": best_harmonic,
        "source_roi_periodicity_lag_px": best_lag,
        "source_roi_periodicity_direction": best_direction,
    }


def _seedless_roi_texture_diagnostics(
    *,
    candidate: np.ndarray,
    ownership: np.ndarray,
    component_stats: np.ndarray,
    owner_area: int,
    sobel_x: np.ndarray,
    sobel_y: np.ndarray,
    gradient: np.ndarray,
) -> dict[str, object]:
    candidate_bool = candidate.astype(bool)
    owner_bool = ownership.astype(bool)
    candidate_pixels = int(np.count_nonzero(candidate_bool))
    component_count = max(0, int(component_stats.shape[0]) - 1)
    areas = component_stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    widths = component_stats[1:, cv2.CC_STAT_WIDTH]
    heights = component_stats[1:, cv2.CC_STAT_HEIGHT]
    micro_area_limit = max(4, min(18, int(round(owner_area * 0.0025))))
    micro = (
        (areas <= micro_area_limit)
        & (widths <= 8)
        & (heights <= 8)
    )
    micro_count = int(np.count_nonzero(micro))
    micro_fraction = micro_count / max(1, component_count)
    micro_areas = areas[micro]
    micro_area_cv = (
        float(np.std(micro_areas) / max(1e-6, float(np.mean(micro_areas))))
        if micro_areas.size >= 2
        else 1.0
    )
    candidate_ratio = candidate_pixels / max(1, owner_area)
    component_density = (component_count * 1000.0) / max(1, owner_area)

    transition_count = 0
    transition_pairs = 0
    if candidate.shape[1] > 1:
        valid = owner_bool[:, 1:] & owner_bool[:, :-1]
        transition_count += int(
            np.count_nonzero(
                (candidate_bool[:, 1:] != candidate_bool[:, :-1]) & valid
            )
        )
        transition_pairs += int(np.count_nonzero(valid))
    if candidate.shape[0] > 1:
        valid = owner_bool[1:, :] & owner_bool[:-1, :]
        transition_count += int(
            np.count_nonzero(
                (candidate_bool[1:, :] != candidate_bool[:-1, :]) & valid
            )
        )
        transition_pairs += int(np.count_nonzero(valid))
    transition_ratio = transition_count / max(1, transition_pairs)

    orientation_mask = candidate_bool & owner_bool & (gradient > 0)
    orientation_weights = gradient[orientation_mask].astype(np.float64)
    if orientation_weights.size:
        orientations = np.mod(
            np.arctan2(
                sobel_y[orientation_mask],
                sobel_x[orientation_mask],
            ),
            np.pi,
        )
        histogram, _ = np.histogram(
            orientations,
            bins=12,
            range=(0.0, np.pi),
            weights=orientation_weights,
        )
        orientation_concentration = float(
            np.max(histogram) / max(1e-6, float(np.sum(histogram)))
        )
    else:
        orientation_concentration = 0.0

    periodicity = _seedless_periodicity_metrics(
        candidate_bool,
        owner_bool,
    )
    pair_score = float(periodicity["source_roi_periodicity_pair_score"])
    cue_saturation_threshold = 0.55
    veto_reason = "none"
    if candidate_ratio >= cue_saturation_threshold:
        if pair_score >= 0.30:
            veto_reason = "periodic_source_cue_saturation"
        elif orientation_concentration >= 0.62:
            veto_reason = "oriented_source_cue_saturation"
        else:
            veto_reason = "textured_carrier_source_cue_saturation"
    elif (
        micro_count >= 8
        and micro_fraction >= 0.65
        and micro_area_cv <= 0.65
        and pair_score >= 0.30
    ):
        veto_reason = "regular_micro_components"
    elif (
        component_count >= 4
        and orientation_concentration >= 0.62
        and pair_score >= 0.25
        and candidate_ratio >= 0.02
    ):
        veto_reason = "periodic_hatching"
    elif (
        micro_count >= 12
        and micro_fraction >= 0.70
        and component_density >= 3.0
        and transition_ratio >= 0.12
    ):
        veto_reason = "dense_microtexture_components"
    elif (
        component_count >= 12
        and component_density >= 3.0
        and candidate_ratio >= 0.05
        and transition_ratio >= 0.18
    ):
        veto_reason = "textured_carrier_high_frequency"

    return {
        "source_roi_texture_metric_version": "microtexture-periodicity-v1",
        "source_roi_microtexture_vetoed": veto_reason != "none",
        "source_roi_microtexture_veto_reason": veto_reason,
        "source_roi_candidate_component_count": component_count,
        "source_roi_candidate_pixel_ratio": candidate_ratio,
        "source_roi_cue_saturation_threshold": cue_saturation_threshold,
        "source_roi_component_density_per_kpixel": component_density,
        "source_roi_micro_component_area_limit": micro_area_limit,
        "source_roi_micro_component_count": micro_count,
        "source_roi_micro_component_fraction": micro_fraction,
        "source_roi_micro_component_area_cv": micro_area_cv,
        "source_roi_transition_ratio": transition_ratio,
        "source_roi_gradient_orientation_concentration": (
            orientation_concentration
        ),
        **periodicity,
    }


def _seedless_empty_result(
    *,
    shape: tuple[int, int],
    protect: np.ndarray,
    evidence: SeedlessRoiEvidence,
    status: str,
    reason: str,
    owner_count: int,
    source_sha256: str,
    scanned_pixels: int,
    candidate_mask: np.ndarray | None = None,
    rejected_mask: np.ndarray | None = None,
    component_records: tuple[GlyphComponentRecord, ...] = (),
    owner_records: tuple[GlyphOwnerRecord, ...] = (),
    texture_diagnostics: Mapping[str, object] | None = None,
) -> GlyphRefinementResult:
    zero = np.zeros(shape, dtype=np.uint8)
    candidate = zero if candidate_mask is None else candidate_mask
    rejected = zero if rejected_mask is None else rejected_mask
    page_pixels = shape[0] * shape[1]
    diagnostics: dict[str, object] = {
        "source_roi_texture_metric_version": "microtexture-periodicity-v1",
        "source_roi_microtexture_vetoed": False,
        "source_roi_microtexture_veto_reason": "not_evaluated",
    }
    if texture_diagnostics is not None:
        diagnostics.update(texture_diagnostics)
    return _build_result(
        owned_detector_seed=zero,
        effect_support=zero,
        candidate_mask=candidate,
        glyph_core=zero,
        glyph_effect=zero,
        rejected_mask=rejected,
        hard_protect=protect,
        component_records=component_records,
        owner_records=owner_records,
        provenance={
            "schema_version": "seedless-roi-glyph-v1",
            "status": status,
            "reason": reason,
            "seed_mode": "authoritative_ocr_seedless",
            "detector_provider": "none",
            "ownership_provider": evidence.ocr_provider,
            "source_image_sha256": source_sha256,
            "evidence_seal_sha256": evidence.seal_sha256,
            "owner_component_count": owner_count,
            "candidate_pixel_count": int(np.count_nonzero(candidate)),
            "refined_pixel_count": 0,
            "page_pixel_count": page_pixels,
            "component_bbox_scanned_pixel_count": scanned_pixels,
            "component_bbox_scan_ratio": scanned_pixels / max(1, page_pixels),
            "output_subset_of_ownership": True,
            "effect_subset_of_support": True,
            "effect_disjoint_from_core": True,
            "no_seed_no_expansion": True,
            "information_limited": status == "information_limited",
            **diagnostics,
        },
    )


def _seedless_evidence_failure(
    evidence: SeedlessRoiEvidence,
    *,
    source_sha256: str,
    ownership_sha256: str,
    protect_sha256: str,
) -> tuple[str, str] | None:
    if evidence.seal_sha256 != _evidence_seal(evidence.payload()):
        return "rejected", "evidence_seal_mismatch"
    if evidence.source_sha256 != source_sha256:
        return "rejected", "source_sha256_mismatch"
    if evidence.ownership_mask_sha256 != ownership_sha256:
        return "rejected", "ownership_mask_sha256_mismatch"
    if evidence.hard_protect_mask_sha256 != protect_sha256:
        return "rejected", "hard_protect_mask_sha256_mismatch"
    if evidence.ocr_provider_sha256 != _text_sha256(evidence.ocr_provider):
        return "rejected", "ocr_provider_sha256_mismatch"
    if evidence.authoritative_ocr is not True:
        return "information_limited", "authoritative_ocr_missing"
    if not evidence.owner_region_id.strip():
        return "information_limited", "owner_region_id_missing"
    if not evidence.ocr_provider.strip():
        return "information_limited", "ocr_provider_missing"
    if not evidence.ocr_text.strip():
        return "information_limited", "ocr_text_missing"
    if not evidence.ocr_script.strip():
        return "information_limited", "ocr_script_missing"
    confidence = evidence.ocr_confidence
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not math.isfinite(float(confidence))
        or not 0.0 < float(confidence) <= 1.0
    ):
        return "information_limited", "ocr_confidence_missing_or_invalid"
    if evidence.owner_count != 1:
        return "rejected", "ownership_conflict"
    if evidence.processing_action != TRANSLATE:
        return "rejected", "processing_action_not_translate"
    if evidence.route_class not in SEEDLESS_ALLOWED_ROUTES:
        return "rejected", "route_not_seedless_safe"
    return None


def extract_seedless_roi_glyph_masks(
    source_image: np.ndarray,
    *,
    ocr_ownership: np.ndarray,
    hard_protect: np.ndarray,
    evidence: SeedlessRoiEvidence,
) -> GlyphRefinementResult:
    """Create B3 pixels from two source cues, never from an OCR bbox."""

    gray = _source_gray(source_image)
    lab_l = _source_lab_l(source_image)
    shape = gray.shape
    ownership = binary_mask(ocr_ownership, shape)
    protect = binary_mask(hard_protect, shape)
    protect_adjacent = _protect_adjacency(protect)
    source_sha = tensor_sha256(np.asarray(source_image))
    owner_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (ownership > 0).astype(np.uint8), 8, cv2.CV_32S
    )
    actual_owner_count = owner_count - 1
    scanned_pixels = shape[0] * shape[1]
    failure = _seedless_evidence_failure(
        evidence,
        source_sha256=source_sha,
        ownership_sha256=mask_sha256(ownership),
        protect_sha256=mask_sha256(protect),
    )
    if failure is not None:
        return _seedless_empty_result(
            shape=shape,
            protect=protect,
            evidence=evidence,
            status=failure[0],
            reason=failure[1],
            owner_count=actual_owner_count,
            source_sha256=source_sha,
            scanned_pixels=scanned_pixels,
        )
    if actual_owner_count != 1:
        return _seedless_empty_result(
            shape=shape,
            protect=protect,
            evidence=evidence,
            status="rejected",
            reason="runtime_ownership_conflict",
            owner_count=actual_owner_count,
            source_sha256=source_sha,
            scanned_pixels=scanned_pixels,
        )

    x, y, width, height, owner_area = (int(value) for value in stats[1])
    scanned_pixels += width * height
    owner_slice = (slice(y, y + height), slice(x, x + width))
    local_owner = labels[owner_slice] == 1
    owner_u8 = np.where(local_owner, 255, 0).astype(np.uint8)
    local_l = lab_l[owner_slice]
    sigma_small = max(1.5, min(3.5, min(width, height) / 18.0))
    sigma_large = max(3.5, min(8.0, min(width, height) / 8.0))
    background_small = _normalized_local_background(
        local_l, owner_u8, sigma=sigma_small
    )
    background_large = _normalized_local_background(
        local_l, owner_u8, sigma=sigma_large
    )
    residual = np.maximum(
        np.abs(local_l - background_small),
        np.abs(local_l - background_large),
    )
    residual_threshold = max(2.0, _noise_floor(residual, owner_u8) * 1.5)
    lab_cue = residual >= residual_threshold

    sobel_x = cv2.Sobel(local_l, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(local_l, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobel_x, sobel_y)
    gradient_threshold = max(4.0, _noise_floor(gradient, owner_u8) * 1.5)
    local_l_u8 = np.clip(local_l, 0, 255).astype(np.uint8)
    stroke_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    stroke = np.maximum(
        cv2.morphologyEx(
            local_l_u8, cv2.MORPH_TOPHAT, stroke_kernel
        ).astype(np.float32),
        cv2.morphologyEx(
            local_l_u8, cv2.MORPH_BLACKHAT, stroke_kernel
        ).astype(np.float32),
    )
    stroke_threshold = max(2.0, _noise_floor(stroke, owner_u8) * 1.5)
    raw_shape_cue = (gradient >= gradient_threshold) | (
        stroke >= stroke_threshold
    )
    shape_cue = _dilate(
        np.where(raw_shape_cue, 255, 0).astype(np.uint8), 2
    ) > 0
    owner_interior = cv2.erode(owner_u8, _ellipse(1), iterations=1) > 0
    local_candidate = local_owner & owner_interior & lab_cue & shape_cue

    candidate_page = np.zeros(shape, dtype=np.uint8)
    core_page = np.zeros(shape, dtype=np.uint8)
    rejected_page = np.zeros(shape, dtype=np.uint8)
    records: list[GlyphComponentRecord] = []
    count, component_labels, component_stats, _ = (
        cv2.connectedComponentsWithStats(
            local_candidate.astype(np.uint8), 8, cv2.CV_32S
        )
    )
    texture_diagnostics = _seedless_roi_texture_diagnostics(
        candidate=local_candidate,
        ownership=local_owner,
        component_stats=component_stats,
        owner_area=owner_area,
        sobel_x=sobel_x,
        sobel_y=sobel_y,
        gradient=gradient,
    )
    texture_vetoed = bool(
        texture_diagnostics["source_roi_microtexture_vetoed"]
    )
    for component_id in range(1, count):
        cx, cy, cw, ch, component_area = (
            int(value) for value in component_stats[component_id]
        )
        if component_area <= 0:
            continue
        scanned_pixels += cw * ch
        local_slice = (slice(cy, cy + ch), slice(cx, cx + cw))
        global_slice = (
            slice(y + cy, y + cy + ch),
            slice(x + cx, x + cx + cw),
        )
        component = component_labels[local_slice] == component_id
        overlap = int(
            np.count_nonzero(component & (protect[global_slice] > 0))
        )
        adjacency = int(
            np.count_nonzero(
                component & (protect_adjacent[global_slice] > 0)
            )
        )
        line_like = cw >= max(20, ch * 8) or ch >= max(20, cw * 8)
        oversized = component_area > max(32, int(owner_area * 0.20))
        accepted = True
        reason = "accepted_seedless_source_intersection"
        if component_area < 2:
            accepted = False
            reason = "source_component_too_small"
        elif line_like:
            accepted = False
            reason = "source_component_line_like"
        elif oversized:
            accepted = False
            reason = "source_component_oversized"
        elif overlap:
            accepted = False
            reason = "exact_protect_overlap"
        elif adjacency:
            accepted = False
            reason = "exact_protect_adjacency"
        elif texture_vetoed:
            accepted = False
            reason = "source_roi_microtexture_veto"
        candidate_page[global_slice][component] = 255
        if accepted:
            core_page[global_slice][component] = 255
        else:
            rejected_page[global_slice][component] = 255
        signed_residual = local_l[local_slice] - background_large[local_slice]
        records.append(
            GlyphComponentRecord(
                ownership_component_id=1,
                candidate_component_id=component_id,
                bbox_xyxy=(
                    x + cx,
                    y + cy,
                    x + cx + cw,
                    y + cy + ch,
                ),
                polarity=_polarity_name(
                    component,
                    signed_residual >= residual_threshold,
                    signed_residual <= -residual_threshold,
                ),
                accepted=accepted,
                reason=reason,
                detector_seed_pixel_count=0,
                core_pixel_count=component_area,
                effect_pixel_count=0,
                protect_overlap_pixel_count=overlap,
                protect_adjacency_pixel_count=adjacency,
                protect_contact_pixel_count=overlap + adjacency,
            )
        )

    owner_record = GlyphOwnerRecord(
        ownership_component_id=1,
        bbox_xyxy=(x, y, x + width, y + height),
        ownership_pixel_count=owner_area,
        detector_seed_pixel_count=0,
        accounted_seed_pixel_count=0,
        candidate_component_count=len(records),
        accepted_component_count=sum(row.accepted for row in records),
        rejected_component_count=sum(not row.accepted for row in records),
        status=(
            "rejected"
            if texture_vetoed
            else "completed" if np.any(core_page) else "information_limited"
        ),
        reason=(
            "source_roi_microtexture_veto"
            if texture_vetoed
            else (
                "source_cues_intersected"
                if np.any(core_page)
                else "source_local_cue_absent"
            )
        ),
    )
    if any(row.protect_contact_pixel_count for row in records):
        return _seedless_empty_result(
            shape=shape,
            protect=protect,
            evidence=evidence,
            status="rejected",
            reason="exact_protect_contact",
            owner_count=1,
            source_sha256=source_sha,
            scanned_pixels=scanned_pixels,
            candidate_mask=candidate_page,
            rejected_mask=cv2.bitwise_or(rejected_page, core_page),
            component_records=tuple(records),
            owner_records=(owner_record,),
            texture_diagnostics=texture_diagnostics,
        )
    if texture_vetoed:
        return _seedless_empty_result(
            shape=shape,
            protect=protect,
            evidence=evidence,
            status="rejected",
            reason="source_roi_microtexture_veto",
            owner_count=1,
            source_sha256=source_sha,
            scanned_pixels=scanned_pixels,
            candidate_mask=candidate_page,
            rejected_mask=rejected_page,
            component_records=tuple(records),
            owner_records=(owner_record,),
            texture_diagnostics=texture_diagnostics,
        )
    if not np.any(core_page):
        return _seedless_empty_result(
            shape=shape,
            protect=protect,
            evidence=evidence,
            status="information_limited",
            reason="source_local_cue_absent",
            owner_count=1,
            source_sha256=source_sha,
            scanned_pixels=scanned_pixels,
            candidate_mask=candidate_page,
            rejected_mask=rejected_page,
            component_records=tuple(records),
            owner_records=(owner_record,),
            texture_diagnostics=texture_diagnostics,
        )

    zero = np.zeros(shape, dtype=np.uint8)
    page_pixels = shape[0] * shape[1]
    return _build_result(
        owned_detector_seed=zero,
        effect_support=zero,
        candidate_mask=candidate_page,
        glyph_core=core_page,
        glyph_effect=zero,
        rejected_mask=rejected_page,
        hard_protect=protect,
        component_records=tuple(records),
        owner_records=(owner_record,),
        provenance={
            "schema_version": "seedless-roi-glyph-v1",
            "status": "completed",
            "reason": "authoritative_ocr_and_two_source_cues",
            "seed_mode": "authoritative_ocr_seedless",
            "detector_provider": "none",
            "ownership_provider": evidence.ocr_provider,
            "source_image_sha256": source_sha,
            "evidence_seal_sha256": evidence.seal_sha256,
            "owner_component_count": 1,
            "candidate_pixel_count": int(np.count_nonzero(candidate_page)),
            "glyph_core_pixel_count": int(np.count_nonzero(core_page)),
            "glyph_effect_pixel_count": 0,
            "refined_pixel_count": int(np.count_nonzero(core_page)),
            "page_pixel_count": page_pixels,
            "component_bbox_scanned_pixel_count": scanned_pixels,
            "component_bbox_scan_ratio": scanned_pixels / max(1, page_pixels),
            "output_subset_of_ownership": bool(
                not np.any((core_page > 0) & (ownership == 0))
            ),
            "effect_subset_of_support": True,
            "effect_disjoint_from_core": True,
            "no_seed_no_expansion": False,
            "information_limited": False,
            "lab_multiscale_cue_pixel_count": int(np.count_nonzero(lab_cue)),
            "stroke_outline_cue_pixel_count": int(np.count_nonzero(shape_cue)),
            "source_cue_intersection_pixel_count": int(
                np.count_nonzero(local_candidate)
            ),
            **texture_diagnostics,
        },
    )
