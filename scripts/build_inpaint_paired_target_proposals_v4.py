#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.paired_target import (  # noqa: E402
    paired_old_text_proposal,
    source_extent_variants,
)
from scripts.validation_artifact_harness import (  # noqa: E402
    default_archive_root,
    select_managed_output_directory,
)


FAMILY = "inpaint-paired-target-proposals-v4"
CATEGORY = "40-inpaint-mask-render"
SCHEMA_VERSION = "inpaint-paired-target-proposals-v4"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_image(path: Path) -> np.ndarray:
    value = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if value is None or value.size == 0:
        raise FileNotFoundError(path)
    return value


def _write_image(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", np.asarray(value))
    if not ok:
        raise OSError(f"unable to encode image: {path}")
    encoded.tofile(path)
    return str(path.resolve())


def _overlay(source: np.ndarray, mask: np.ndarray) -> np.ndarray:
    tint = source.copy()
    tint[mask > 0] = np.array([30, 30, 250], np.uint8)
    return cv2.addWeighted(source, 0.55, tint, 0.45, 0)


def _contact_sheets(
    rows: list[tuple[str, Path, Path, Path]],
    output_dir: Path,
    *,
    rows_per_sheet: int = 8,
) -> list[str]:
    cell = (300, 420)
    columns = ("SOURCE", "PAIRED LOCATION ONLY", "PROPOSED SOURCE EXTENT")
    paths: list[str] = []
    for sheet_index, start in enumerate(range(0, len(rows), rows_per_sheet), 1):
        group = rows[start : start + rows_per_sheet]
        canvas = Image.new(
            "RGB", (cell[0] * len(columns), (cell[1] + 30) * len(group)), "white"
        )
        draw = ImageDraw.Draw(canvas)
        for row_index, (page_id, source_path, paired_path, overlay_path) in enumerate(group):
            for column_index, (label, path) in enumerate(
                zip(columns, (source_path, paired_path, overlay_path))
            ):
                image = Image.open(path).convert("RGB")
                image.thumbnail(cell, Image.Resampling.LANCZOS)
                x = column_index * cell[0] + (cell[0] - image.width) // 2
                y = row_index * (cell[1] + 30) + 24
                canvas.paste(image, (x, y))
                draw.text((column_index * cell[0] + 4, y - 18), label, fill="black")
            draw.text((4, row_index * (cell[1] + 30) + 4), page_id, fill="black")
        path = output_dir / "review" / f"paired-target-review-{sheet_index:02d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, quality=94)
        paths.append(str(path.resolve()))
    return paths


def _variant_contact_sheets(
    rows: list[tuple[str, Path, dict[str, Path]]],
    output_dir: Path,
    *,
    rows_per_sheet: int = 8,
) -> list[str]:
    cell = (260, 380)
    columns = ("SOURCE", "STRICT", "BALANCED", "EDGE SUPPORTED")
    paths: list[str] = []
    for sheet_index, start in enumerate(range(0, len(rows), rows_per_sheet), 1):
        group = rows[start : start + rows_per_sheet]
        canvas = Image.new(
            "RGB", (cell[0] * len(columns), (cell[1] + 30) * len(group)), "white"
        )
        draw = ImageDraw.Draw(canvas)
        for row_index, (page_id, source_path, variants) in enumerate(group):
            image_paths = (
                source_path,
                variants["strict"],
                variants["balanced"],
                variants["edge_supported"],
            )
            for column_index, (label, path) in enumerate(zip(columns, image_paths)):
                image = Image.open(path).convert("RGB")
                image.thumbnail(cell, Image.Resampling.LANCZOS)
                x = column_index * cell[0] + (cell[0] - image.width) // 2
                y = row_index * (cell[1] + 30) + 24
                canvas.paste(image, (x, y))
                draw.text((column_index * cell[0] + 4, y - 18), label, fill="black")
            draw.text((4, row_index * (cell[1] + 30) + 4), page_id, fill="black")
        path = output_dir / "review" / f"source-extent-variants-{sheet_index:02d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, quality=94)
        paths.append(str(path.resolve()))
    return paths


def build_paired_target_proposals(
    source_index_path: Path,
    output_dir: Path,
    *,
    page_ids: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    source_index = _read_json(source_index_path)
    if source_index.get("schema_version") != "inpaint-development-source-index-v4":
        raise ValueError("unsupported development source index")
    pages: list[dict[str, Any]] = []
    review_rows: list[tuple[str, Path, Path, Path]] = []
    variant_review_rows: list[tuple[str, Path, dict[str, Path]]] = []
    paired_page_count = 0
    unpaired_page_count = 0
    instance_count = 0
    for raw_page in source_index.get("pages", []):
        if not isinstance(raw_page, dict):
            raise ValueError("source index page must be an object")
        page_id = str(raw_page.get("page_id") or "").strip()
        if page_ids and page_id not in page_ids:
            continue
        source_path = Path(str(raw_page.get("path") or ""))
        source_sha = str(raw_page.get("source_sha256") or "").lower()
        if _sha256(source_path) != source_sha:
            raise ValueError(f"source SHA mismatch: {page_id}")
        paired = raw_page.get("paired_reference")
        if paired is None:
            unpaired_page_count += 1
            pages.append(
                {
                    "page_id": page_id,
                    "source_sha256": source_sha,
                    "status": "source_inventory_review_required",
                    "candidate_seen": False,
                    "target_mask_provenance": "unpaired_source_review_pending",
                    "target_extent_independent": False,
                    "target_inventory_independent": False,
                    "target_review_complete": False,
                    "instances": [],
                }
            )
            continue
        if not isinstance(paired, dict) or paired.get("proposal_only") is not True:
            raise ValueError(f"invalid paired proposal: {page_id}")
        paired_path = Path(str(paired.get("path") or ""))
        reference_sha = str(paired.get("reference_sha256") or "").lower()
        if _sha256(paired_path) != reference_sha:
            raise ValueError(f"paired reference SHA mismatch: {page_id}")
        source = _read_image(source_path)
        reference = _read_image(paired_path)
        proposal = paired_old_text_proposal(source, reference)
        page_dir = output_dir / "pages" / page_id
        core_path = _write_image(page_dir / "paired-old-text-core.png", proposal.core_mask)
        extent_path = _write_image(page_dir / "paired-old-text-extent.png", proposal.extent_mask)
        instances: list[dict[str, object]] = []
        for index, mask in enumerate(proposal.instance_masks):
            instance_id = f"paired-instance-{index:04d}"
            instances.append(
                {
                    "instance_id": instance_id,
                    "mask_path": _write_image(page_dir / "instances" / f"{instance_id}.png", mask),
                    "review_status": "pending",
                }
            )
        overlay_path = Path(
            _write_image(page_dir / "source-proposal-overlay.jpg", _overlay(source, proposal.extent_mask))
        )
        variant_masks = source_extent_variants(source, proposal.extent_mask)
        variant_paths: dict[str, Path] = {}
        variant_mask_paths: dict[str, str] = {}
        for variant_id, variant_mask in variant_masks.items():
            variant_mask_paths[variant_id] = _write_image(
                page_dir / "extent-variants" / f"{variant_id}.png", variant_mask
            )
            variant_paths[variant_id] = Path(
                _write_image(
                    page_dir / "extent-variants" / f"{variant_id}-overlay.jpg",
                    _overlay(source, variant_mask),
                )
            )
        review_rows.append((page_id, source_path, paired_path, overlay_path))
        variant_review_rows.append((page_id, source_path, variant_paths))
        paired_page_count += 1
        instance_count += len(instances)
        pages.append(
            {
                "page_id": page_id,
                "source_sha256": source_sha,
                "paired_reference_sha256": reference_sha,
                "status": "source_review_required",
                "candidate_seen": False,
                "paired_reference_used_as": "location_proposal_only",
                "target_mask_provenance": "paired_removed_source_contrast_proposal",
                "target_extent_independent": True,
                "target_inventory_independent": True,
                "target_review_complete": False,
                "target_extent_selected": None,
                "target_extent_variants": variant_mask_paths,
                "core_mask": core_path,
                "target_text_mask": extent_path,
                "instances": instances,
                "diagnostics": {
                    "delta_threshold": proposal.delta_threshold,
                    "delta_median": proposal.delta_median,
                    "delta_mad": proposal.delta_mad,
                    "core_pixel_count": int(np.count_nonzero(proposal.core_mask)),
                    "target_pixel_count": int(np.count_nonzero(proposal.extent_mask)),
                    "instance_count": len(instances),
                },
            }
        )
    sheets = _contact_sheets(review_rows, output_dir)
    variant_sheets = _variant_contact_sheets(variant_review_rows, output_dir)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_index_sha256": _sha256(source_index_path),
        "candidate_seen": False,
        "paired_reference_is_clean_background": False,
        "review_complete": False,
        "paired_page_count": paired_page_count,
        "unpaired_page_count": unpaired_page_count,
        "proposed_instance_count": instance_count,
        "pages": pages,
        "review_sheets": sheets,
        "variant_review_sheets": variant_sheets,
    }
    path = output_dir / "paired-target-proposals.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build candidate-blind old-text proposals from paired human edits."
    )
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--page-id", action="append", default=[])
    args = parser.parse_args(argv)
    output_dir, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    archive = default_archive_root().resolve()
    try:
        output_dir.resolve().relative_to(archive)
    except ValueError as exc:
        raise ValueError("paired target output must stay in the private archive") from exc
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        payload = build_paired_target_proposals(
            args.source_index.resolve(),
            output_dir.resolve(),
            page_ids=frozenset(str(value) for value in args.page_id),
        )
        if managed is not None:
            managed.complete(
                metadata={
                    "source_index_sha256": payload["source_index_sha256"],
                    "paired_page_count": payload["paired_page_count"],
                    "unpaired_page_count": payload["unpaired_page_count"],
                    "proposed_instance_count": payload["proposed_instance_count"],
                }
            )
    except BaseException as error:
        if managed is not None:
            managed.fail(error)
        raise
    print(
        json.dumps(
            {
                "paired_pages": payload["paired_page_count"],
                "unpaired_pages": payload["unpaired_page_count"],
                "instances": payload["proposed_instance_count"],
                "review_sheets": len(payload["review_sheets"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
