#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.silhouette import (  # noqa: E402
    ballons_native_clean_background,
    extract_pr2_validated_interior,
)
from modules.detection.processor import TextBlockDetector  # noqa: E402
from modules.rendering.render import get_best_render_area  # noqa: E402
from modules.utils.image_utils import generate_mask  # noqa: E402
from modules.utils.mask_roi import build_text_prior_mask, resolve_block_ctd_roi  # noqa: E402
from scripts.export_inpaint_debug import _SettingsStub  # noqa: E402
from scripts.validation_artifact_harness import (  # noqa: E402
    default_archive_root,
    select_managed_output_directory,
)


SCHEMA_VERSION = "inpaint-factorized-source-proposals-v4"
DECISIONS_SCHEMA_VERSION = "inpaint-factorized-source-decisions-v4"
FAMILY = "inpaint-factorized-source-proposals-v4"
CATEGORY = "40-inpaint-mask-render"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise FileNotFoundError(path)
    return image


def _write_image(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, np.asarray(image))
    if not ok:
        raise OSError(f"failed to encode image: {path}")
    encoded.tofile(path)
    return str(path.resolve())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _binary(value: object, shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.zeros(shape, np.uint8)
    array = np.asarray(value)
    if array.ndim == 3:
        array = array[..., 0]
    if tuple(array.shape) != tuple(shape):
        raise ValueError(f"mask shape mismatch: {array.shape} != {shape}")
    return np.where(array > 0, 255, 0).astype(np.uint8)


def _paired_change_proposal(source: np.ndarray, paired: np.ndarray) -> np.ndarray:
    if paired.shape != source.shape:
        raise ValueError("paired proposal image shape mismatch")
    delta = np.max(
        np.abs(source.astype(np.int16) - paired.astype(np.int16)), axis=2
    )
    changed = np.where(delta >= 12, 255, 0).astype(np.uint8)
    return cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def propose_semantic_contract(
    text_class: str,
    *,
    paired_change_contact: bool,
) -> tuple[str, str, str]:
    if str(text_class or "") == "text_bubble":
        return "dialogue_bubble", "translate_inpaint", "required"
    if str(text_class or "") == "text_free" and paired_change_contact:
        return "dialogue_free", "translate_inpaint", "required"
    if str(text_class or "") == "text_free":
        return "ambiguous", "review", "ambiguous"
    return "ambiguous", "review", "ambiguous"


def _connected_instances(mask: np.ndarray) -> list[np.ndarray]:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8, cv2.CV_32S
    )
    instances: list[np.ndarray] = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if area <= 0:
            continue
        component = np.zeros_like(mask, np.uint8)
        local = labels[y:y + height, x:x + width] == label
        component[y:y + height, x:x + width][local] = 255
        instances.append(component)
    return instances


def _region_route_class(
    image_bgr: np.ndarray,
    seed: np.ndarray,
    interior: np.ndarray,
    text_class: str,
) -> str:
    if str(text_class or "") != "text_bubble" or not np.any(interior):
        return "ambiguous"
    if ballons_native_clean_background(image_bgr, seed):
        return "clean_flat"
    return "ambiguous"


def _contact_sheets(
    rows: list[tuple[str, Path, Path]],
    output_dir: Path,
    *,
    rows_per_sheet: int = 12,
) -> list[str]:
    paths: list[str] = []
    thumb = (320, 440)
    for sheet_index, start in enumerate(range(0, len(rows), rows_per_sheet), 1):
        group = rows[start:start + rows_per_sheet]
        canvas = Image.new(
            "RGB", (thumb[0] * 2, (thumb[1] + 34) * len(group)), "white"
        )
        draw = ImageDraw.Draw(canvas)
        for row, (page_id, source_path, overlay_path) in enumerate(group):
            for column, (label, path) in enumerate(
                (("SOURCE", source_path), ("SOURCE-ONLY PROPOSAL", overlay_path))
            ):
                image = Image.open(path).convert("RGB")
                image.thumbnail(thumb, Image.Resampling.LANCZOS)
                x = column * thumb[0] + (thumb[0] - image.width) // 2
                y = row * (thumb[1] + 34) + 20
                canvas.paste(image, (x, y))
                draw.text(
                    (column * thumb[0] + 4, y - 18),
                    f"{page_id} {label}",
                    fill="black",
                )
        path = output_dir / "review" / f"source-proposals-{sheet_index:02d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, quality=92)
        paths.append(str(path.resolve()))
    return paths


def build_source_proposals(
    index_path: Path,
    output_dir: Path,
    *,
    use_gpu: bool,
    page_ids: frozenset[str] = frozenset(),
    limit: int = 0,
) -> dict[str, Any]:
    source_index = _read_json(index_path)
    if source_index.get("schema_version") != "inpaint-development-source-index-v4":
        raise ValueError("unsupported development source index")
    pages = source_index.get("pages")
    if not isinstance(pages, list):
        raise ValueError("source index must contain pages")
    selected_pages = [
        page
        for page in pages
        if not page_ids or str(page.get("page_id") or "") in page_ids
    ]
    if limit > 0:
        selected_pages = selected_pages[:limit]
    if page_ids and len(selected_pages) != len(page_ids):
        found = {str(page.get("page_id") or "") for page in selected_pages}
        raise ValueError(f"unknown requested page ids: {sorted(page_ids - found)}")
    settings = _SettingsStub(inpainter="lama_large_512px", use_gpu=use_gpu)
    detector = TextBlockDetector(settings)
    started = perf_counter()
    source_pages: list[dict[str, Any]] = []
    proposal_pages: list[dict[str, Any]] = []
    review_rows: list[tuple[str, Path, Path]] = []
    block_count = 0
    instance_count = 0
    page_timings: list[dict[str, object]] = []
    for page_number, page in enumerate(selected_pages, 1):
        if not isinstance(page, dict):
            raise ValueError("source index page must be an object")
        page_id = str(page.get("page_id") or "").strip()
        source_path = Path(str(page.get("path") or ""))
        if _sha256(source_path) != str(page.get("source_sha256") or "").lower():
            raise ValueError(f"source SHA mismatch: {page_id}")
        source_bgr = _read_image(source_path)
        source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
        shape = source_bgr.shape[:2]
        paired_record = page.get("paired_reference")
        paired_change = np.zeros(shape, np.uint8)
        if paired_record is not None:
            if not isinstance(paired_record, dict) or paired_record.get("proposal_only") is not True:
                raise ValueError(f"invalid paired proposal: {page_id}")
            paired_path = Path(str(paired_record.get("path") or ""))
            if _sha256(paired_path) != str(paired_record.get("reference_sha256") or "").lower():
                raise ValueError(f"paired reference SHA mismatch: {page_id}")
            paired_change = _paired_change_proposal(source_bgr, _read_image(paired_path))

        page_started = perf_counter()
        print(
            f"source-proposal {page_number}/{len(selected_pages)} {page_id}: detect",
            flush=True,
        )
        detect_started = perf_counter()
        blocks = detector.detect(source_rgb) or []
        detect_seconds = perf_counter() - detect_started
        get_best_render_area(blocks, source_rgb)
        mask_started = perf_counter()
        details = generate_mask(
            source_rgb,
            blocks,
            settings=settings.get_mask_refiner_settings(),
            return_details=True,
        )
        mask_seconds = perf_counter() - mask_started
        raw = _binary(details.get("raw_mask"), shape)
        refined = _binary(details.get("refined_mask"), shape)
        final = _binary(details.get("final_mask"), shape)
        protect = _binary(details.get("protect_mask"), shape)
        page_dir = output_dir / "pages" / page_id
        ownership_union = np.zeros(shape, np.uint8)
        interior_union = np.zeros(shape, np.uint8)
        required_union = np.zeros(shape, np.uint8)
        preserve_union = np.zeros(shape, np.uint8)
        ambiguous_union = np.zeros(shape, np.uint8)
        assigned = np.zeros(shape, np.uint8)
        target_instances: list[dict[str, Any]] = []
        regions: list[dict[str, Any]] = []

        for block_index, block in enumerate(blocks):
            roi = resolve_block_ctd_roi(block, source_rgb.shape)
            if roi is None:
                continue
            x1, y1, x2, y2 = roi
            ownership = np.zeros(shape, np.uint8)
            ownership[y1:y2, x1:x2] = build_text_prior_mask(
                source_rgb, block, roi, dilate_iterations=2
            )
            if not np.any(ownership):
                continue
            region_id = f"region-{block_index:03d}"
            ownership_union[ownership > 0] = 255
            seed = np.where(
                (raw > 0) & (ownership > 0) & (assigned == 0), 255, 0
            ).astype(np.uint8)
            paired_contact = bool(np.any((paired_change > 0) & (ownership > 0)))
            role, action, priority = propose_semantic_contract(
                str(getattr(block, "text_class", "") or ""),
                paired_change_contact=paired_contact,
            )
            local_seed = seed[y1:y2, x1:x2]
            local_image = source_bgr[y1:y2, x1:x2]
            local_interior = extract_pr2_validated_interior(local_image, local_seed)
            interior = np.zeros(shape, np.uint8)
            if local_interior is not None:
                interior[y1:y2, x1:x2] = local_interior
            interior_union[interior > 0] = 255
            route_class = _region_route_class(
                local_image,
                local_seed,
                interior[y1:y2, x1:x2],
                str(getattr(block, "text_class", "") or ""),
            )
            region_protected = np.where(
                (protect > 0) & (ownership > 0) & (seed == 0), 255, 0
            ).astype(np.uint8)
            region_ambiguous = np.zeros(shape, np.uint8)
            if priority == "ambiguous":
                region_ambiguous[seed > 0] = 255
            region_dir = page_dir / "regions" / region_id
            region_paths = {
                "bubble_interior_mask": _write_image(region_dir / "bubble-interior.png", interior),
                "ownership_mask": _write_image(region_dir / "ownership.png", ownership),
                "protected_structure_mask": _write_image(region_dir / "protected.png", region_protected),
                "ambiguous_structure_mask": _write_image(region_dir / "ambiguous.png", region_ambiguous),
                "corner_protect_mask": _write_image(region_dir / "corner.png", np.zeros(shape, np.uint8)),
            }
            regions.append(
                {
                    "region_id": region_id,
                    "bubble_route_class": route_class,
                    **region_paths,
                    "proposal": {
                        "text_class": str(getattr(block, "text_class", "") or ""),
                        "paired_change_contact": paired_contact,
                        "candidate_seen": False,
                    },
                }
            )
            if np.any(seed):
                component = seed
                instance_id = f"{region_id}-instance-000"
                instance_path = _write_image(
                    page_dir / "instances" / f"{instance_id}.png", component
                )
                target_instances.append(
                    {
                        "instance_id": instance_id,
                        "region_id": region_id,
                        "mask_path": instance_path,
                        "semantic_role": role,
                        "processing_action": action,
                        "priority": priority,
                    }
                )
                if priority == "required":
                    required_union[component > 0] = 255
                elif priority == "optional":
                    preserve_union[component > 0] = 255
                else:
                    ambiguous_union[component > 0] = 255
                assigned[component > 0] = 255
                instance_count += 1

        unowned = np.where((raw > 0) & (assigned == 0), 255, 0).astype(np.uint8)
        if np.any(unowned):
            region_id = "region-unowned"
            unowned_ownership = cv2.dilate(
                unowned, np.ones((5, 5), np.uint8), iterations=1
            )
            ownership_union[unowned_ownership > 0] = 255
            regions.append(
                {
                    "region_id": region_id,
                    "bubble_route_class": "ambiguous",
                    "bubble_interior_mask": _write_image(page_dir / "regions" / region_id / "bubble-interior.png", np.zeros(shape, np.uint8)),
                    "ownership_mask": _write_image(page_dir / "regions" / region_id / "ownership.png", unowned_ownership),
                    "protected_structure_mask": _write_image(page_dir / "regions" / region_id / "protected.png", np.zeros(shape, np.uint8)),
                    "ambiguous_structure_mask": _write_image(page_dir / "regions" / region_id / "ambiguous.png", unowned),
                    "corner_protect_mask": _write_image(page_dir / "regions" / region_id / "corner.png", np.zeros(shape, np.uint8)),
                    "proposal": {"blockless": True, "candidate_seen": False},
                }
            )
            for component_index, component in enumerate(_connected_instances(unowned)):
                instance_id = f"{region_id}-instance-{component_index:03d}"
                target_instances.append(
                    {
                        "instance_id": instance_id,
                        "region_id": region_id,
                        "mask_path": _write_image(page_dir / "instances" / f"{instance_id}.png", component),
                        "semantic_role": "ambiguous",
                        "processing_action": "review",
                        "priority": "ambiguous",
                    }
                )
                ambiguous_union[component > 0] = 255
                instance_count += 1

        protected = np.where(
            (protect > 0)
            & (required_union == 0)
            & (preserve_union == 0)
            & (ambiguous_union == 0),
            255,
            0,
        ).astype(np.uint8)
        ambiguous_union = np.where(
            (ambiguous_union > 0) & (required_union == 0) & (preserve_union == 0),
            255,
            0,
        ).astype(np.uint8)
        masks = {
            "target_text_mask": _write_image(page_dir / "target-text.png", required_union),
            "preserve_mask": _write_image(page_dir / "preserve.png", preserve_union),
            "protected_structure_mask": _write_image(page_dir / "protected-structure.png", protected),
            "ambiguous_structure_mask": _write_image(page_dir / "ambiguous-structure.png", ambiguous_union),
            "ownership_mask": _write_image(page_dir / "ownership.png", ownership_union),
            "claim_seed_mask": _write_image(page_dir / "claim-seed.png", raw),
            "bubble_interior_mask": _write_image(page_dir / "bubble-interior.png", interior_union),
            "corner_protect_mask": _write_image(page_dir / "corner-protect.png", np.zeros(shape, np.uint8)),
            "raw_mask": _write_image(page_dir / "raw.png", raw),
            "refined_mask": _write_image(page_dir / "refined.png", refined),
            "final_mask": _write_image(page_dir / "final.png", final),
            "paired_change_proposal": _write_image(page_dir / "paired-change-proposal.png", paired_change),
        }
        overlay = source_bgr.copy()
        colors = (
            (required_union, np.array([40, 40, 255], np.uint8)),
            (preserve_union, np.array([255, 180, 40], np.uint8)),
            (protected, np.array([40, 220, 40], np.uint8)),
            (ambiguous_union, np.array([255, 80, 180], np.uint8)),
        )
        tinted = overlay.copy()
        for mask, color in colors:
            tinted[mask > 0] = color
        overlay = cv2.addWeighted(overlay, 0.58, tinted, 0.42, 0)
        overlay_path = Path(_write_image(page_dir / "source-only-overlay.jpg", overlay))
        review_rows.append((page_id, source_path, overlay_path))
        page_timings.append(
            {
                "page_id": page_id,
                "detect_seconds": detect_seconds,
                "mask_seconds": mask_seconds,
                "total_seconds": perf_counter() - page_started,
            }
        )
        print(
            f"source-proposal {page_number}/{len(selected_pages)} {page_id}: "
            f"done detect={detect_seconds:.3f}s mask={mask_seconds:.3f}s",
            flush=True,
        )
        source_entry = {
            "page_id": page_id,
            "path": str(source_path.resolve()),
            "source_sha256": str(page.get("source_sha256") or ""),
            "width": int(shape[1]),
            "height": int(shape[0]),
            "expected_edit": "required" if np.any(required_union) else "none",
            "paired_reference": paired_record,
        }
        source_pages.append(source_entry)
        proposal_pages.append(
            {
                "page_id": page_id,
                "expected_edit": source_entry["expected_edit"],
                **{key: value for key, value in masks.items() if key in {
                    "target_text_mask", "preserve_mask", "protected_structure_mask",
                    "ambiguous_structure_mask", "ownership_mask", "claim_seed_mask",
                    "bubble_interior_mask", "corner_protect_mask",
                }},
                "target_instances": target_instances,
                "target_mask_provenance": "current_ctd_raw_components",
                "target_extent_independent": False,
                "target_inventory_independent": False,
                "target_review_complete": False,
                "regions": regions,
                "reviewed_source_only": False,
                "candidate_seen": False,
            }
        )
        block_count += len(blocks)

    source_payload = {
        "schema_version": SCHEMA_VERSION,
        "corpus_id": "e1-development-v4",
        "split_role": "development_source_only",
        "source_index_sha256": _sha256(index_path),
        "candidate_images_generated": False,
        "inpainting_invoked": False,
        "pages": source_pages,
    }
    decisions_payload = {
        "schema_version": DECISIONS_SCHEMA_VERSION,
        "corpus_id": "e1-development-v4",
        "candidate_seen": False,
        "review_complete": False,
        "pages": proposal_pages,
    }
    _write_json(output_dir / "source-manifest-v4-proposal.json", source_payload)
    _write_json(output_dir / "source-decisions-v4-proposal.json", decisions_payload)
    review_paths = _contact_sheets(review_rows, output_dir)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "candidate_images_generated": False,
        "inpainting_invoked": False,
        "page_count": len(source_pages),
        "block_count": block_count,
        "instance_count": instance_count,
        "elapsed_seconds": perf_counter() - started,
        "detector_engine": detector.last_engine_name,
        "detector_device": detector.last_device,
        "review_sheets": review_paths,
        "page_timings": page_timings,
    }
    _write_json(output_dir / "source-proposal-summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate candidate-blind source-only v4 annotation proposals."
    )
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--page-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    output_dir, managed_run = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    archive = default_archive_root().resolve()
    try:
        output_dir.resolve().relative_to(archive)
    except ValueError as exc:
        raise ValueError("source proposal output must stay in the private archive") from exc
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        summary = build_source_proposals(
            args.source_index.resolve(),
            output_dir,
            use_gpu=not args.cpu,
            page_ids=frozenset(str(value) for value in args.page_id),
            limit=max(0, int(args.limit)),
        )
    except BaseException as exc:
        if managed_run is not None:
            managed_run.fail(exc)
        raise
    if managed_run is not None:
        managed_run.complete(metadata=summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
