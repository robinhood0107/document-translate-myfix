#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.contracts import (  # noqa: E402
    CombinationClosureRecord,
    FactorizedRunRecord,
    binary_mask,
)
from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    PageMasks,
    RegionMasks,
    broad_route_false_positive_pixels,
    decide_bubble_route,
    expand_detector_claim,
    load_page_masks,
    load_stage1_manifest,
)
from benchmarking.inpaint_detector_bakeoff.stage2 import (  # noqa: E402
    assert_complete_closure_ledger,
    build_combination_closure_ledger,
    build_factorized_matrix,
    composite_positive_result,
    fill_factorized_mask,
    reconstruction_error,
    score_stage2_page,
    select_pareto_records,
)
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


def _route_fill_backend(
    fill_id: str,
    route_decision: str,
    bubble_route_class: str | None = None,
) -> str:
    normalized = str(fill_id).strip().lower()
    if normalized == "narrow_lama_broad_flat":
        return "robust_flat_median" if route_decision == "broad" else "current_lama"
    if normalized == "conditional_hybrid":
        if route_decision != "broad":
            return "current_lama"
        if bubble_route_class == "clean_flat":
            return "robust_flat_median"
        if bubble_route_class == "clean_gradient":
            return "planar_gradient"
        return "current_lama"
    return normalized


FAMILY = "inpaint-factorized-v3"
CATEGORY = "40-inpaint-mask-render"


@lru_cache(maxsize=None)
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write image: {path}")


def _path_value(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        nested = value.get("path")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return None


def _read_image(
    path: str | Path,
    cache: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    key = str(path)
    if cache is not None and key in cache:
        return cache[key]
    image = cv2.imdecode(np.fromfile(key, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise FileNotFoundError(path)
    if cache is not None:
        cache[key] = image
    return image


def _read_mask(
    path: object,
    shape: tuple[int, int],
    cache: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    value = _path_value(path)
    if value is None:
        return np.zeros(shape, dtype=np.uint8)
    if cache is not None and value in cache:
        mask = cache[value]
        if mask.shape != shape:
            raise ValueError(f"mask shape mismatch: {mask.shape} != {shape}")
        return mask
    mask = cv2.imdecode(np.fromfile(value, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.size == 0:
        raise FileNotFoundError(value)
    normalized = binary_mask(mask, shape)
    if cache is not None:
        cache[value] = normalized
    return normalized


def _sparse_mask_indices(
    path: str,
    shape: tuple[int, int],
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    cached = cache.get(path)
    if cached is not None:
        return cached
    mask = _read_mask(path, shape, None)
    indices = np.flatnonzero(mask.reshape(-1)).astype(np.int64, copy=False)
    if indices.size == 0:
        raise ValueError(f"target instance mask is empty: {path}")
    cache[path] = indices
    return indices


def _page_artifact(
    family: dict[str, object],
    page_id: str,
    key: str,
    fallback_key: str | None = None,
) -> object:
    pages = family.get("pages")
    if not isinstance(pages, dict):
        raise ValueError("matrix family must contain a pages object")
    page = pages.get(page_id)
    if not isinstance(page, dict):
        raise ValueError(f"matrix family is missing page {page_id}")
    if key not in page and fallback_key is not None:
        key = fallback_key
    if key not in page:
        raise ValueError(f"matrix page {page_id} is missing artifact {key}")
    return page[key]


def _entries(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(entry.get("page_id") or ""): entry
        for entry in payload.get("pages", [])
        if isinstance(entry, dict)
    }


def _with_candidate_ownership(
    masks: PageMasks,
    ownership: np.ndarray,
    broad_ownership: np.ndarray,
    interior: np.ndarray,
) -> PageMasks:
    return PageMasks(
        masks.target,
        masks.protected,
        masks.ambiguous,
        ownership,
        masks.claim_seed,
        masks.existing_edit,
        masks.target_instances,
        interior,
        masks.corner,
        broad_ownership,
        masks.preserve,
        masks.regions,
    )


def _annotation_masks(
    page,
    entry: dict[str, object],
    shape: tuple[int, int],
    cache: dict[str, np.ndarray],
    *,
    sparse_evidence: bool = False,
) -> PageMasks:
    target = _read_mask(page.target_text_mask, shape, cache)
    protected = _read_mask(page.protected_structure_mask, shape, cache)
    ambiguous = _read_mask(page.ambiguous_structure_mask, shape, cache)
    ownership = (
        _read_mask(page.ownership_mask, shape, cache)
        if page.ownership_mask
        else np.full(shape, 255, np.uint8)
    )
    claim_seed = (
        _read_mask(page.claim_seed_mask, shape, cache)
        if page.claim_seed_mask
        else np.full(shape, 255, np.uint8)
    )
    existing_path = _path_value(
        entry.get("existing_source_edit_mask", entry.get("baseline_mask"))
    )
    existing = _read_mask(existing_path, shape, cache)
    interior = _read_mask(page.bubble_interior_mask, shape, cache)
    corner = _read_mask(page.corner_protect_mask, shape, cache)
    preserve = _read_mask(page.preserve_mask, shape, cache)
    instances = () if sparse_evidence else tuple(
        (record.instance_id, _read_mask(record.mask_path, shape, cache))
        for record in page.target_instances
        if record.priority == "required"
    )
    # Region and target-instance artifacts are full-page PNGs in the sealed
    # source manifest.  Keeping hundreds of those arrays resident turns E1
    # evaluation into multi-gigabyte work.  The full helper contract remains
    # the default; the matrix runner opts into sparse streaming explicitly.
    regions = () if sparse_evidence else tuple(
        RegionMasks(
            region.region_id,
            region.bubble_route_class,
            _read_mask(region.bubble_interior_mask, shape, cache),
            _read_mask(region.ownership_mask, shape, cache),
            _read_mask(region.protected_structure_mask, shape, cache),
            _read_mask(region.ambiguous_structure_mask, shape, cache),
            _read_mask(region.corner_protect_mask, shape, cache),
        )
        for region in page.regions
    )
    return PageMasks(
        target,
        protected,
        ambiguous,
        ownership,
        claim_seed,
        existing,
        instances,
        interior,
        corner,
        ownership,
        preserve,
        regions,
    )


def _component_coverages(target: np.ndarray, mask: np.ndarray) -> list[float]:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (target > 0).astype(np.uint8), 8
    )
    values: list[float] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if area <= 0:
            continue
        component = labels[y:y + height, x:x + width] == index
        local = mask[y:y + height, x:x + width] > 0
        values.append(float(np.count_nonzero(component & local)) / float(area))
    return values


def _score_cached_mask_only(
    page_id: str,
    source: np.ndarray,
    baseline: np.ndarray,
    final_mask: np.ndarray,
    masks: PageMasks,
    cache: dict[str, dict[str, object]],
) -> tuple[dict[str, object], np.ndarray]:
    cached = cache.get(page_id)
    if cached is None:
        invariant, _changed = score_stage2_page(
            source,
            baseline,
            np.zeros(source.shape[:2], np.uint8),
            masks,
            baseline=baseline,
        )
        cache[page_id] = invariant
    else:
        invariant = cached
    # A mask-only run never changes pixels.  Do not retain one full-page zero
    # array per corpus page merely to report that invariant across candidates.
    changed = np.zeros(source.shape[:2], np.uint8)
    detector = binary_mask(final_mask, source.shape[:2])
    metrics = dict(invariant)
    covered = int(np.count_nonzero((masks.target > 0) & (detector > 0)))
    target_pixels = int(np.count_nonzero(masks.target))
    coverages = _component_coverages(masks.target, detector)
    metrics.update(
        {
            "detector_mask_pixel_count": int(np.count_nonzero(detector)),
            "changed_outside_detector_mask_pixel_count": int(
                np.count_nonzero((changed > 0) & (detector == 0))
            ),
            "target_detector_covered_pixel_count": covered,
            "target_detector_coverage": (
                float(covered) / float(target_pixels) if target_pixels else None
            ),
            "target_component_coverages": coverages,
            "minimum_target_component_coverage": (
                min(coverages) if coverages else None
            ),
        }
    )
    return metrics, changed


class _LamaPool:
    def __init__(self, *, device: str, precision: str, inpaint_size: int) -> None:
        self.device = device
        self.precision = precision
        self.inpaint_size = int(inpaint_size)
        self._model: Any | None = None
        self.call_count = 0

    def fill(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self._model is None:
            from modules.inpainting.source_lama_blockwise import SourceLaMaLarge

            self._model = SourceLaMaLarge(
                device=self.device,
                precision=self.precision,
                inpaint_size=self.inpaint_size,
            )
            self._model.ensure_loaded()
        generated_rgb = self._model.memory_safe_inpaint(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            mask,
        )
        self.call_count += 1
        return cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2BGR)


def _run_combination(
    combination: dict[str, str],
    *,
    matrix: dict[str, object],
    pages,
    entries: dict[str, dict[str, object]],
    output_root: Path,
    lama_pool: _LamaPool,
    instance_index_cache: dict[str, np.ndarray],
    mask_only_score_cache: dict[str, dict[str, object]],
) -> tuple[FactorizedRunRecord, list[dict[str, object]]]:
    families = matrix.get("families")
    if not isinstance(families, dict):
        raise ValueError("matrix spec must contain families")
    detector = families["detector"][combination["detector"]]
    ownership_family = families["ownership"][combination["ownership"]]
    silhouette = families["silhouette"][combination["silhouette"]]
    router = families["router"][combination["router"]]
    expansion_id = combination["expansion"]
    fill_id = combination["fill"]
    if not all(
        isinstance(value, dict)
        for value in (detector, ownership_family, silhouette, router)
    ):
        raise ValueError("matrix family entries must be objects")
    run_id = "__".join(combination[role] for role in combination)
    run_root = output_root / "runs" / run_id
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    total_target = 0
    total_target_edit = 0
    seed_total = 0
    seed_hits = 0
    instance_coverages: list[float] = []
    residue_sum = 0.0
    residue_count = 0
    baseline_sum = 0.0
    baseline_count = 0
    reconstruction_values: list[float] = []
    protected_changed = 0
    protected_overlap = 0
    preserve_overlap = 0
    ambiguous_changed = 0
    ambiguous_overlap = 0
    outside_changed = 0
    no_edit_false = 0
    broad_false = 0
    required_skips = 0
    page_residue_worsened = 0
    calls_before = lama_pool.call_count

    for page in pages:
        # E1 contains ~400 MiB for each full corpus mask.  A process-wide
        # artifact cache held several detector/ownership/silhouette families
        # plus annotations simultaneously and grew past 8 GiB.  Keep only one
        # page's decoded images and masks alive; the OS file cache still makes
        # repeated combinations inexpensive without retaining NumPy arrays.
        image_cache: dict[str, np.ndarray] = {}
        mask_cache: dict[str, np.ndarray] = {}
        entry = entries[page.page_id]
        if fill_id == "mask_only":
            height = int(entry.get("height") or 0)
            width = int(entry.get("width") or 0)
            if height <= 0 or width <= 0:
                source = _read_image(page.source_image, image_cache)
                shape = source.shape[:2]
            else:
                shape = (height, width)
                source = np.zeros((height, width, 3), np.uint8)
        else:
            source = _read_image(page.source_image, image_cache)
            shape = source.shape[:2]
        annotation = _annotation_masks(
            page, entry, shape, mask_cache, sparse_evidence=True
        )
        ownership = _read_mask(
            _page_artifact(ownership_family, page.page_id, "mask"), shape, mask_cache
        )
        broad_ownership = _read_mask(
            _page_artifact(
                ownership_family, page.page_id, "broad_mask", "mask"
            ),
            shape,
            mask_cache,
        )
        interior = _read_mask(
            _page_artifact(silhouette, page.page_id, "interior"), shape, mask_cache
        )
        masks = _with_candidate_ownership(
            annotation, ownership, broad_ownership, interior
        )
        raw = _read_mask(
            _page_artifact(detector, page.page_id, "raw"), shape, mask_cache
        )
        refined = _read_mask(
            _page_artifact(detector, page.page_id, "refined"), shape, mask_cache
        )
        dilated = _read_mask(
            _page_artifact(detector, page.page_id, "dilated"), shape, mask_cache
        )
        seed_variant = str(detector.get("seed_variant") or "raw")
        detector_seed = {
            "raw": raw,
            "refined": refined,
            "dilated": dilated,
        }[seed_variant]
        route_seed = cv2.bitwise_and(detector_seed, ownership)
        content = None
        if expansion_id == "content_component":
            content = _read_mask(
                _page_artifact(ownership_family, page.page_id, "content_components"),
                shape,
                mask_cache,
            )
        broad = expand_detector_claim(
            expansion_id,
            seed=route_seed,
            raw=raw,
            refined=refined,
            dilated=dilated,
            content_components=content,
            bubble_interior=interior,
        )
        router_pages = router.get("pages", {})
        evidence = router_pages.get(page.page_id, {}) if isinstance(router_pages, dict) else {}
        if not isinstance(evidence, dict):
            raise ValueError("router page evidence must be an object")
        route_masks = {
            key: (
                _read_mask(evidence.get(path_key), shape, mask_cache)
                if evidence.get(path_key)
                else None
            )
            for key, path_key in (
                ("ballons_clean_mask", "ballons_clean_mask"),
                ("pr2_clean_mask", "pr2_clean_mask"),
                ("segmentation_valid_mask", "segmentation_valid_mask"),
                ("unsafe_signal_mask", "unsafe_signal_mask"),
            )
        }
        exclude = cv2.bitwise_or(masks.protected, masks.ambiguous)
        if masks.corner is not None:
            exclude = cv2.bitwise_or(exclude, masks.corner)
        if masks.preserve is not None:
            exclude = cv2.bitwise_or(exclude, masks.preserve)
        background_samples = int(
            np.count_nonzero(
                (interior > 0)
                & (route_seed == 0)
                & (exclude == 0)
                & (
                    (masks.broad_ownership > 0)
                    if masks.broad_ownership is not None
                    else (masks.ownership > 0)
                )
            )
        )
        decision = decide_bubble_route(
            str(router.get("algorithm") or combination["router"]),
            narrow_claim=detector_seed,
            broad_claim=broad,
            seed=route_seed,
            masks=masks,
            ballons_clean=bool(evidence.get("ballons_clean", False)),
            pr2_clean=bool(evidence.get("pr2_clean", False)),
            segmentation_valid=bool(evidence.get("segmentation_valid", False)),
            texture=bool(evidence.get("texture", False)),
            microtexture=bool(evidence.get("microtexture", False)),
            line_art=bool(evidence.get("line_art", False)),
            ambiguous=bool(evidence.get("ambiguous", False)),
            background_sample_count=background_samples,
            minimum_background_samples=int(router.get("minimum_background_samples", 32)),
            **route_masks,
        )
        clean_annotation = page.bubble_route_class in {"clean_flat", "clean_gradient"}
        broad_only = cv2.bitwise_and(
            decision.edit_mask,
            cv2.bitwise_not(detector_seed),
        )
        if page.regions:
            source_clean = route_masks.get("pr2_clean_mask")
            if source_clean is None:
                source_clean = np.zeros(shape, np.uint8)
            broad_route_false_pixels = int(
                np.count_nonzero((broad_only > 0) & (source_clean == 0))
            )
            broad_route_false = broad_route_false_pixels > 0
        else:
            broad_route_false_pixels = int(np.count_nonzero(broad_only)) if (
                decision.decision == "broad" and not clean_annotation
            ) else 0
            broad_route_false = broad_route_false_pixels > 0
        broad_false += broad_route_false_pixels
        if decision.decision == "skip" and not page.no_edit:
            required_skips += 1

        seed_scores = []
        edit_scores = []
        flat_seed = detector_seed.reshape(-1)
        flat_edit = decision.edit_mask.reshape(-1)
        for instance_record in page.target_instances:
            if instance_record.priority != "required":
                continue
            instance_id = instance_record.instance_id
            indices = _sparse_mask_indices(
                instance_record.mask_path, shape, instance_index_cache
            )
            pixels = int(indices.size)
            seed_covered = int(np.count_nonzero(flat_seed[indices]))
            edit_covered = int(np.count_nonzero(flat_edit[indices]))
            seed_scores.append(
                {"instance_id": instance_id, "seeded": seed_covered > 0}
            )
            coverage = float(edit_covered) / float(pixels) if pixels else 0.0
            edit_scores.append({"instance_id": instance_id, "coverage": coverage})
            seed_total += 1
            seed_hits += int(seed_covered > 0)
            instance_coverages.append(coverage)

        baseline_path = _path_value(entry.get("baseline"))
        baseline_mask_path = _path_value(entry.get("baseline_mask"))
        baseline = (
            _read_image(baseline_path, image_cache) if baseline_path else source.copy()
        )
        baseline_mask = (
            _read_mask(baseline_mask_path, shape, mask_cache)
            if baseline_mask_path
            else np.zeros(shape, dtype=np.uint8)
        )
        if fill_id == "mask_only":
            candidate = baseline.copy()
            final_mask = cv2.bitwise_or(baseline_mask, decision.edit_mask)
            fill_diagnostics = {"backend": fill_id, "applied": False}
        else:
            selected_fill = _route_fill_backend(
                fill_id,
                decision.decision,
                page.bubble_route_class,
            )
            callback = (
                lama_pool.fill
                if selected_fill in {"current_lama", "ballons_lama"}
                else None
            )
            generated, fill_diagnostics = fill_factorized_mask(
                source,
                decision.edit_mask,
                backend=selected_fill,
                interior_mask=interior,
                background_exclude_mask=exclude,
                lama_fill=callback,
            )
            if selected_fill != fill_id:
                fill_diagnostics["requested_backend"] = fill_id
            candidate, final_mask = composite_positive_result(
                baseline,
                generated,
                decision.edit_mask,
                baseline_mask,
            )
        if fill_id == "mask_only":
            metrics, changed = _score_cached_mask_only(
                page.page_id,
                source,
                baseline,
                final_mask,
                masks,
                mask_only_score_cache,
            )
        else:
            metrics, changed = score_stage2_page(
                source,
                candidate,
                final_mask,
                masks,
                baseline=baseline,
            )
        target_pixels = int(np.count_nonzero(masks.target))
        target_edit = int(
            np.count_nonzero((masks.target > 0) & (decision.edit_mask > 0))
        )
        total_target += target_pixels
        total_target_edit += target_edit
        protected_changed += int(metrics["protected_changed_pixel_count"])
        protected_overlap += int(
            np.count_nonzero((decision.edit_mask > 0) & (masks.protected > 0))
        )
        if masks.preserve is not None:
            preserve_overlap += int(
                np.count_nonzero((decision.edit_mask > 0) & (masks.preserve > 0))
            )
        ambiguous_changed += int(metrics["ambiguous_changed_pixel_count"])
        ambiguous_overlap += int(
            np.count_nonzero((decision.edit_mask > 0) & (masks.ambiguous > 0))
        )
        outside_changed += int(metrics["changed_outside_detector_mask_pixel_count"])
        no_edit_false += int(page.no_edit) * int(np.count_nonzero(decision.edit_mask))
        residue_value = metrics.get("residue_score")
        residue_pixels = int(metrics.get("residue_source_contrast_pixel_count") or 0)
        if residue_value is not None and residue_pixels:
            residue_sum += float(residue_value) * residue_pixels
            residue_count += residue_pixels
        baseline_value = metrics.get("baseline_residue_score")
        if baseline_value is not None and residue_pixels:
            baseline_sum += float(baseline_value) * residue_pixels
            baseline_count += residue_pixels
        if (
            residue_value is not None
            and baseline_value is not None
            and float(residue_value) > float(baseline_value) + 1e-12
        ):
            page_residue_worsened += 1
        known_background = _path_value(entry.get("known_background"))
        reconstruction = None
        if known_background and fill_id != "mask_only":
            reconstruction = reconstruction_error(
                candidate,
                _read_image(known_background, image_cache),
                decision.edit_mask,
            )
            if reconstruction is not None:
                reconstruction_values.append(reconstruction)
        row = {
            "page_id": page.page_id,
            "route_decision": decision.decision,
            "route_reasons": list(decision.reasons),
            "broad_route_false_positive": broad_route_false,
            "broad_route_false_positive_pixel_count": broad_route_false_pixels,
            "target_instance_seed_scores": seed_scores,
            "target_instance_edit_scores": edit_scores,
            "edit_pixel_count": int(np.count_nonzero(decision.edit_mask)),
            "fill": fill_diagnostics,
            "reconstruction_mse": reconstruction,
            **metrics,
        }
        rows.append(row)
        retain_artifacts = bool(matrix.get("retain_page_artifacts", False))
        required_only = bool(
            matrix.get("retain_required_page_artifacts_only", False)
        )
        if retain_artifacts and (not required_only or not page.no_edit):
            _write_image(run_root / "edit_masks" / f"{page.page_id}.png", decision.edit_mask)
            _write_image(run_root / "candidate_images" / f"{page.page_id}.png", candidate)
            _write_image(run_root / "changed_masks" / f"{page.page_id}.png", changed)

    metrics = {
        "page_count": len(rows),
        "target_extent_independent": all(
            page.target_extent_independent for page in pages
        ),
        "target_mask_provenance": sorted(
            {page.target_mask_provenance for page in pages}
        ),
        "target_instance_seed_recall": (
            float(seed_hits) / float(seed_total) if seed_total else None
        ),
        "missed_target_instance_count": seed_total - seed_hits,
        "aggregate_target_coverage": (
            float(total_target_edit) / float(total_target) if total_target else None
        ),
        "minimum_target_instance_coverage": (
            min(instance_coverages) if instance_coverages else None
        ),
        "protected_structure_overlap": protected_overlap,
        "protected_structure_changed": protected_changed,
        "preserve_edit_overlap": preserve_overlap,
        "ambiguous_structure_overlap": ambiguous_overlap,
        "ambiguous_structure_changed": ambiguous_changed,
        "outside_final_changed": outside_changed,
        "broad_route_false_positive": broad_false,
        "no_edit_false_edit": no_edit_false,
        "required_skip_count": required_skips,
        "page_residue_worsened_count": page_residue_worsened,
        "aggregate_residue_score": (
            residue_sum / float(residue_count) if residue_count else None
        ),
        "baseline_aggregate_residue_score": (
            baseline_sum / float(baseline_count) if baseline_count else None
        ),
        "reconstruction_mse": (
            float(np.mean(reconstruction_values)) if reconstruction_values else None
        ),
        "runtime_seconds": time.perf_counter() - started,
        "positive_lama_inference_count": lama_pool.call_count - calls_before,
        "residue_gate_applicable": fill_id != "mask_only",
    }
    oracle_only_ids = set(matrix.get("oracle_only", []))
    oracle_only = any(value in oracle_only_ids for value in combination.values())
    record = FactorizedRunRecord(
        run_id=run_id,
        detector_id=combination["detector"],
        ownership_id=combination["ownership"],
        silhouette_id=combination["silhouette"],
        router_id=combination["router"],
        expansion_id=combination["expansion"],
        fill_id=fill_id,
        oracle_only=oracle_only,
        status="active",
        metrics=metrics,
    )
    return record, rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the v3 role-factorized control/single/pairwise matrix. "
            "Detector masks and route evidence must be source-only artifacts."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--inpaint-size", type=int, default=1536)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--limit", type=int)
    return parser


def _declared_combinations(
    matrix: dict[str, object],
    axes: dict[str, list[str]],
    controls: dict[str, str],
) -> list[dict[str, str]]:
    combinations = (
        build_factorized_matrix(axes, controls)
        if bool(matrix.get("factorized", True))
        else [dict(controls)]
    )
    explicit = matrix.get("explicit_combinations", [])
    if not isinstance(explicit, list):
        raise ValueError("explicit_combinations must be a list")
    seen = {tuple(sorted(record.items())) for record in combinations}
    for raw in explicit:
        if not isinstance(raw, dict):
            raise ValueError("explicit combination must be an object")
        record = dict(controls)
        for role, value in raw.items():
            if role not in controls or not isinstance(value, str) or not value:
                raise ValueError(f"invalid explicit combination field: {role}")
            if value not in axes[role]:
                raise ValueError(
                    f"explicit combination value is outside axis: {role}={value}"
                )
            record[role] = value
        key = tuple(sorted(record.items()))
        if key not in seen:
            seen.add(key)
            combinations.append(record)
    return combinations


def _logical_id(selection: dict[str, str]) -> str:
    return "__".join(selection[role] for role in selection)


def _content_value(value: object) -> object:
    """Replace artifact paths with bytes SHA while retaining algorithm settings."""

    if isinstance(value, dict):
        return {
            str(key): _content_value(nested)
            for key, nested in sorted(value.items())
            if key not in {"provider", "candidate_id", "status"}
        }
    if isinstance(value, list):
        return [_content_value(nested) for nested in value]
    if isinstance(value, str):
        path = Path(value)
        if path.is_file():
            return {"artifact_sha256": _sha256(path)}
        return value
    return value


def _combination_content_sha256(
    combination: dict[str, str],
    *,
    matrix: dict[str, object],
    manifest_sha256: str,
) -> str:
    families = matrix.get("families")
    if not isinstance(families, dict):
        raise ValueError("matrix spec must contain families")
    payload: dict[str, object] = {
        "manifest_sha256": manifest_sha256,
        "expansion": combination["expansion"],
        "fill": combination["fill"],
    }
    for role in ("detector", "ownership", "silhouette", "router"):
        role_families = families.get(role)
        if not isinstance(role_families, dict):
            raise ValueError(f"matrix families lacks role: {role}")
        selected = role_families.get(combination[role])
        if not isinstance(selected, dict):
            raise ValueError(f"matrix lacks selected family: {role}/{combination[role]}")
        payload[role] = _content_value(selected)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepare_closure_ledger(
    combinations: list[dict[str, str]],
    *,
    matrix: dict[str, object],
    manifest_sha256: str,
) -> tuple[list[CombinationClosureRecord], list[dict[str, str]]]:
    families = matrix.get("families")
    family_metadata = families if isinstance(families, dict) else None
    oracle_only = matrix.get("oracle_only", [])
    initial: list[CombinationClosureRecord] = []
    executable: list[dict[str, str]] = []
    content_owner: dict[str, str] = {}
    for combination in combinations:
        stage = "stage1" if combination["fill"] == "mask_only" else "product"
        row = build_combination_closure_ledger(
            [combination],
            stage=stage,
            family_metadata=family_metadata,
            oracle_only_ids=oracle_only if isinstance(oracle_only, list) else (),
        )[0]
        if row.closure_state != "executed":
            initial.append(row)
            continue
        content_sha = _combination_content_sha256(
            combination,
            matrix=matrix,
            manifest_sha256=manifest_sha256,
        )
        owner = content_owner.get(content_sha)
        if owner is not None:
            initial.append(
                CombinationClosureRecord(
                    logical_id=row.logical_id,
                    selection=row.selection,
                    closure_state="reused_by_sha",
                    content_sha256=content_sha,
                    reused_from=owner,
                )
            )
            continue
        content_owner[content_sha] = row.logical_id
        initial.append(
            CombinationClosureRecord(
                logical_id=row.logical_id,
                selection=row.selection,
                closure_state="executed",
                content_sha256=content_sha,
            )
        )
        executable.append(combination)
    assert_complete_closure_ledger(combinations, initial)
    return initial, executable


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest.resolve()
    matrix_path = args.matrix.resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("schema_version") != "inpaint-factorized-matrix-v3":
        raise ValueError("unsupported factorized matrix schema")
    axes = matrix.get("axes")
    controls = matrix.get("controls")
    if not isinstance(axes, dict) or not isinstance(controls, dict):
        raise ValueError("factorized matrix requires axes and controls")
    all_combinations = _declared_combinations(matrix, axes, controls)
    manifest_sha256 = _sha256(manifest_path)
    closure_ledger, physical_combinations = _prepare_closure_ledger(
        all_combinations,
        matrix=matrix,
        manifest_sha256=manifest_sha256,
    )
    if args.start_index < 1:
        raise ValueError("--start-index must be at least 1")
    indexed_combinations = list(enumerate(physical_combinations, start=1))[
        args.start_index - 1:
    ]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        indexed_combinations = indexed_combinations[:args.limit]
    combinations_to_run = [combination for _index, combination in indexed_combinations]
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        pages = load_stage1_manifest(manifest_path)
        entries = _entries(manifest_path)
        lama_pool = _LamaPool(
            device=args.device,
            precision=args.precision,
            inpaint_size=args.inpaint_size,
        )
        records: list[FactorizedRunRecord] = []
        page_rows: dict[str, list[dict[str, object]]] = {}
        instance_index_cache: dict[str, np.ndarray] = {}
        mask_only_score_cache: dict[str, dict[str, object]] = {}
        for combination_index, combination in enumerate(combinations_to_run, start=1):
            print(
                f"factorized-v3 {combination_index}/{len(combinations_to_run)} "
                f"{combination}",
                flush=True,
            )
            record, rows = _run_combination(
                combination,
                matrix=matrix,
                pages=pages,
                entries=entries,
                output_root=output_root,
                lama_pool=lama_pool,
                instance_index_cache=instance_index_cache,
                mask_only_score_cache=mask_only_score_cache,
            )
            records.append(record)
            page_rows[record.run_id] = rows
            gc.collect()
        ranked = select_pareto_records(records)
        result = {
            "schema_version": "inpaint-factorized-results-v3",
            "manifest_sha256": manifest_sha256,
            "matrix_sha256": _sha256(matrix_path),
            "logical_combination_count": len(all_combinations),
            "physical_combination_count": len(physical_combinations),
            "combination_count": len(ranked),
            "closure_ledger": [row.as_record() for row in closure_ledger],
            "positive_lama_inference_count": lama_pool.call_count,
            "runs": [record.as_record() for record in ranked],
            "pages": page_rows,
        }
        _write_json(output_root / "factorized-results.json", result)
        if managed is not None:
            managed.complete(
                metadata={
                    "manifest_sha256": result["manifest_sha256"],
                    "matrix_sha256": result["matrix_sha256"],
                    "combination_count": result["combination_count"],
                    "positive_lama_inference_count": result[
                        "positive_lama_inference_count"
                    ],
                }
            )
            mismatches = managed.verify()
            if mismatches:
                raise RuntimeError("managed artifact verification failed: " + "; ".join(mismatches))
            print(managed.run_root)
        else:
            print(output_root)
        return 0
    except BaseException as error:
        if managed is not None:
            managed.fail(
                error,
                metadata={"manifest": manifest_path.name, "matrix": matrix_path.name},
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
