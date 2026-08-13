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
    SourceExtentFeatures,
    build_source_extent_features,
    source_extent_variants,
)
from scripts.validation_artifact_harness import default_archive_root  # noqa: E402


SCHEMA_VERSION = "inpaint-independent-target-review-ledger-v4"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    value = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if value is None or value.size == 0:
        raise FileNotFoundError(path)
    return value


def _write_image(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix or ".png", np.asarray(value))
    if not success:
        raise OSError(f"unable to encode image: {path}")
    encoded.tofile(path)
    return str(path.resolve())


def _mask(path: object, shape: tuple[int, int]) -> np.ndarray:
    value = _read_image(Path(str(path or "")), cv2.IMREAD_GRAYSCALE)
    if value.shape != shape:
        raise ValueError("review mask shape mismatch")
    return np.where(value > 0, 255, 0).astype(np.uint8)


def _crop_bounds(mask: np.ndarray, pad: int = 28) -> tuple[int, int, int, int]:
    yy, xx = np.nonzero(mask > 0)
    if yy.size == 0:
        raise ValueError("review item mask is empty")
    height, width = mask.shape
    return (
        max(0, int(xx.min()) - pad),
        max(0, int(yy.min()) - pad),
        min(width, int(xx.max()) + pad + 1),
        min(height, int(yy.max()) + pad + 1),
    )


def _overlay(source: np.ndarray, mask: np.ndarray) -> np.ndarray:
    tint = source.copy()
    tint[mask > 0] = np.array([30, 30, 250], np.uint8)
    return cv2.addWeighted(source, 0.55, tint, 0.45, 0)


def _grouped_components(mask: np.ndarray, radius: int = 12) -> list[np.ndarray]:
    grouped = cv2.dilate(
        (mask > 0).astype(np.uint8),
        np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, 8)
    output: list[np.ndarray] = []
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) <= 0:
            continue
        component = np.where((labels == index) & (mask > 0), 255, 0).astype(np.uint8)
        if np.count_nonzero(component) >= 8:
            output.append(component)
    return output


def _region_role(page: dict[str, Any], region_id: str) -> str:
    for instance in page.get("target_instances", []):
        if str(instance.get("region_id") or "") != region_id:
            continue
        if str(instance.get("priority") or "") == "required":
            return str(instance.get("semantic_role") or "dialogue_bubble")
    return "review"


def _instance_role(instance: dict[str, Any]) -> str:
    role = str(instance.get("semantic_role") or "").strip()
    return role or "review"


def _write_review_item(
    output_dir: Path,
    source: np.ndarray,
    *,
    page_id: str,
    review_id: str,
    location: np.ndarray,
    region_id: str,
    role_proposal: str,
    inventory_source: str,
    extent_features: SourceExtentFeatures,
    source_instance_id: str | None = None,
    source_priority_proposal: str | None = None,
) -> dict[str, Any]:
    item_dir = output_dir / "pages" / page_id / review_id
    location_path = _write_image(item_dir / "location-seed.png", location)
    variants = source_extent_variants(source, location, features=extent_features)
    variant_paths = {
        variant_id: _write_image(item_dir / f"extent-{variant_id}.png", mask)
        for variant_id, mask in variants.items()
    }
    row = {
        "review_id": review_id,
        "page_id": page_id,
        "region_id": region_id,
        "semantic_role_proposal": role_proposal,
        "inventory_source": inventory_source,
        "location_seed": location_path,
        "extent_variants": variant_paths,
        "selected_extent": None,
        "semantic_decision": None,
        "review_status": "pending",
    }
    if source_instance_id:
        row["source_instance_id"] = source_instance_id
    if source_priority_proposal:
        row["source_priority_proposal"] = source_priority_proposal
    return row


def _review_sheets(
    rows: list[dict[str, Any]],
    sources: dict[str, Path],
    output_dir: Path,
    *,
    rows_per_sheet: int = 8,
) -> list[str]:
    cell = (260, 210)
    columns = ("SOURCE", "LOCATION", "STRICT", "BALANCED", "EDGE", "DILATE1", "DILATE2")
    paths: list[str] = []
    for sheet_index, start in enumerate(range(0, len(rows), rows_per_sheet), 1):
        group = rows[start : start + rows_per_sheet]
        canvas = Image.new(
            "RGB", (cell[0] * len(columns), (cell[1] + 42) * len(group)), "white"
        )
        draw = ImageDraw.Draw(canvas)
        for row_index, row in enumerate(group):
            source = _read_image(sources[row["page_id"]])
            location = _read_image(Path(row["location_seed"]), cv2.IMREAD_GRAYSCALE)
            x1, y1, x2, y2 = _crop_bounds(location)
            masks = [
                location,
                *(
                    _read_image(Path(row["extent_variants"][key]), cv2.IMREAD_GRAYSCALE)
                    for key in (
                        "strict",
                        "balanced",
                        "edge_supported",
                        "location_dilate1",
                        "location_dilate2",
                    )
                ),
            ]
            images = [source[y1:y2, x1:x2]] + [
                _overlay(source, mask)[y1:y2, x1:x2] for mask in masks
            ]
            for column_index, (label, value) in enumerate(zip(columns, images)):
                image = Image.fromarray(cv2.cvtColor(value, cv2.COLOR_BGR2RGB))
                image.thumbnail(cell, Image.Resampling.LANCZOS)
                x = column_index * cell[0] + (cell[0] - image.width) // 2
                y = row_index * (cell[1] + 42) + 30
                canvas.paste(image, (x, y))
                draw.text((column_index * cell[0] + 4, y - 18), label, fill="black")
            draw.text(
                (4, row_index * (cell[1] + 42) + 5),
                f"{row['review_id']} {row['page_id']} {row['semantic_role_proposal']}",
                fill="black",
            )
        path = output_dir / "review" / f"independent-target-review-{sheet_index:03d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, quality=94)
        paths.append(str(path.resolve()))
    return paths


def _full_page_inventory_sheets(
    rows: list[dict[str, Any]],
    semantic_pages: dict[str, dict[str, Any]],
    output_dir: Path,
    *,
    rows_per_sheet: int = 4,
) -> list[str]:
    cell = (480, 650)
    columns = ("SOURCE INVENTORY REVIEW", "KNOWN LOCATION AID ONLY")
    paths: list[str] = []
    for sheet_index, start in enumerate(range(0, len(rows), rows_per_sheet), 1):
        group = rows[start : start + rows_per_sheet]
        canvas = Image.new(
            "RGB", (cell[0] * len(columns), (cell[1] + 34) * len(group)), "white"
        )
        draw = ImageDraw.Draw(canvas)
        for row_index, row in enumerate(group):
            source = _read_image(Path(row["source_path"]))
            semantic = semantic_pages[row["page_id"]]
            target_path = semantic.get("target_text_mask")
            aid = np.zeros(source.shape[:2], np.uint8)
            if target_path:
                aid = _mask(target_path, source.shape[:2])
            overlay = _overlay(source, aid)
            for column_index, (label, value) in enumerate(
                zip(columns, (source, overlay))
            ):
                image = Image.fromarray(cv2.cvtColor(value, cv2.COLOR_BGR2RGB))
                image.thumbnail(cell, Image.Resampling.LANCZOS)
                x = column_index * cell[0] + (cell[0] - image.width) // 2
                y = row_index * (cell[1] + 34) + 26
                canvas.paste(image, (x, y))
                draw.text((column_index * cell[0] + 4, y - 18), label, fill="black")
            draw.text((4, row_index * (cell[1] + 34) + 4), row["page_id"], fill="black")
        path = output_dir / "review" / f"full-page-inventory-{sheet_index:02d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, quality=94)
        paths.append(str(path.resolve()))
    return paths


def build_independent_target_review(
    source_index_path: Path,
    semantic_manifest_path: Path,
    paired_proposals_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_index = _read_json(source_index_path)
    semantic_manifest = _read_json(semantic_manifest_path)
    paired_proposals = _read_json(paired_proposals_path)
    if paired_proposals.get("candidate_seen") is not False:
        raise ValueError("paired proposals must be candidate blind")
    sources = {str(page["page_id"]): page for page in source_index.get("pages", [])}
    semantics = {str(page["page_id"]): page for page in semantic_manifest.get("pages", [])}
    proposals = {str(page["page_id"]): page for page in paired_proposals.get("pages", [])}
    if set(sources) != set(semantics) or set(sources) != set(proposals):
        raise ValueError("independent review page sets differ")

    rows: list[dict[str, Any]] = []
    full_page_pending: list[dict[str, Any]] = []
    source_paths: dict[str, Path] = {}
    for page_id, source_record in sources.items():
        source_path = Path(str(source_record.get("path") or ""))
        source_paths[page_id] = source_path
        source = _read_image(source_path)
        shape = source.shape[:2]
        extent_features = build_source_extent_features(source)
        proposal_record = proposals[page_id]
        proposal_path = proposal_record.get("target_text_mask")
        if not proposal_path:
            full_page_pending.append(
                {
                    "page_id": page_id,
                    "source_path": str(source_path.resolve()),
                    "review_status": "full_page_inventory_pending",
                }
            )
            semantic_page = semantics[page_id]
            page_rows: list[dict[str, Any]] = []
            for instance in semantic_page.get("target_instances", []):
                instance_path = instance.get("mask_path")
                if not instance_path:
                    continue
                location = _mask(instance_path, shape)
                if not np.any(location):
                    continue
                review_id = f"review-{len(rows) + len(page_rows):04d}"
                page_rows.append(
                    _write_review_item(
                        output_dir,
                        source,
                        page_id=page_id,
                        review_id=review_id,
                        location=location,
                        region_id=str(instance.get("region_id") or "region-page-review"),
                        role_proposal=_instance_role(instance),
                        inventory_source="source_manifest_location_aid_only",
                        extent_features=extent_features,
                        source_instance_id=str(instance.get("instance_id") or ""),
                        source_priority_proposal=str(instance.get("priority") or ""),
                    )
                )
            rows.extend(page_rows)
            continue
        proposal = _mask(proposal_path, shape)
        assigned = np.zeros(shape, np.uint8)
        semantic_page = semantics[page_id]
        page_rows: list[dict[str, Any]] = []
        for region in semantic_page.get("regions", []):
            region_id = str(region.get("region_id") or "")
            ownership = _mask(region.get("ownership_mask"), shape)
            location = np.where(
                (proposal > 0) & (ownership > 0) & (assigned == 0), 255, 0
            ).astype(np.uint8)
            if not np.any(location):
                continue
            assigned[ownership > 0] = 255
            review_id = f"review-{len(rows) + len(page_rows):04d}"
            page_rows.append(
                _write_review_item(
                    output_dir,
                    source,
                    page_id=page_id,
                    review_id=review_id,
                    location=location,
                    region_id=region_id,
                    role_proposal=_region_role(semantic_page, region_id),
                    inventory_source="paired_location_with_human_semantic_region",
                    extent_features=extent_features,
                )
            )
        unowned = np.where((proposal > 0) & (assigned == 0), 255, 0).astype(np.uint8)
        for component in _grouped_components(unowned):
            review_id = f"review-{len(rows) + len(page_rows):04d}"
            page_rows.append(
                _write_review_item(
                    output_dir,
                    source,
                    page_id=page_id,
                    review_id=review_id,
                    location=component,
                    region_id="region-unowned-review",
                    role_proposal="review",
                    inventory_source="paired_location_outside_known_ownership",
                    extent_features=extent_features,
                )
            )
        rows.extend(page_rows)

    sheets = _review_sheets(rows, source_paths, output_dir)
    full_page_sheets = _full_page_inventory_sheets(
        full_page_pending, semantics, output_dir
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_index": str(source_index_path.resolve()),
        "source_index_sha256": _sha256(source_index_path),
        "semantic_manifest": str(semantic_manifest_path.resolve()),
        "semantic_manifest_sha256": _sha256(semantic_manifest_path),
        "candidate_seen": False,
        "review_complete": False,
        "target_extent_independent": True,
        "target_inventory_independent": False,
        "paired_reference_used_as": "location_proposal_only",
        "review_row_count": len(rows),
        "full_page_inventory_pending_count": len(full_page_pending),
        "rows": rows,
        "full_page_inventory_pending": full_page_pending,
        "review_sheets": sheets,
        "full_page_inventory_sheets": full_page_sheets,
    }
    output_path = output_dir / "independent-target-review-ledger.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build candidate-blind target extent review sheets.")
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--semantic-manifest", type=Path, required=True)
    parser.add_argument("--paired-proposals", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    try:
        output_dir.relative_to(default_archive_root().resolve())
    except ValueError as exc:
        raise ValueError("review output must stay in the private archive") from exc
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_independent_target_review(
        args.source_index.resolve(),
        args.semantic_manifest.resolve(),
        args.paired_proposals.resolve(),
        output_dir,
    )
    print(
        json.dumps(
            {
                "rows": payload["review_row_count"],
                "full_page_pending": payload["full_page_inventory_pending_count"],
                "sheets": len(payload["review_sheets"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
