#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from scripts.validation_artifact_harness import default_archive_root  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None or image.size == 0:
        raise FileNotFoundError(path)
    return image


def _crop_bounds(mask: np.ndarray, *, pad: int = 28) -> tuple[int, int, int, int]:
    yy, xx = np.nonzero(mask > 0)
    if yy.size == 0:
        raise ValueError("review mask is empty")
    height, width = mask.shape
    return (
        max(0, int(xx.min()) - pad),
        max(0, int(yy.min()) - pad),
        min(width, int(xx.max()) + pad + 1),
        min(height, int(yy.max()) + pad + 1),
    )


def _resize_for_cell(image: Image.Image, cell: tuple[int, int]) -> Image.Image:
    value = image.copy()
    value.thumbnail(cell, Image.Resampling.LANCZOS)
    return value


def build_review_sheets(
    source_manifest: Path,
    decisions_path: Path,
    output_dir: Path,
    *,
    rows_per_sheet: int = 10,
) -> dict[str, Any]:
    sources = {
        str(page["page_id"]): page
        for page in _read_json(source_manifest).get("pages", [])
    }
    decisions = _read_json(decisions_path)
    review_rows: list[dict[str, Any]] = []
    for page in decisions.get("pages", []):
        page_id = str(page.get("page_id") or "")
        source = sources.get(page_id)
        if source is None:
            raise ValueError(f"source page missing: {page_id}")
        regions = {
            str(region.get("region_id") or ""): region
            for region in page.get("regions", [])
        }
        paired = source.get("paired_reference")
        for instance in page.get("target_instances", []):
            region = regions[str(instance.get("region_id") or "")]
            proposal = region.get("proposal") or {}
            priority = str(instance.get("priority") or "")
            needs_review = priority == "ambiguous" or (
                page_id.startswith("elven-")
                and priority == "required"
                and str(proposal.get("text_class") or "") == "text_bubble"
                and not bool(proposal.get("paired_change_contact"))
            )
            if not needs_review:
                continue
            review_rows.append(
                {
                    "review_id": f"review-{len(review_rows):04d}",
                    "page_id": page_id,
                    "instance_id": str(instance.get("instance_id") or ""),
                    "region_id": str(instance.get("region_id") or ""),
                    "priority_proposal": priority,
                    "semantic_role_proposal": str(instance.get("semantic_role") or ""),
                    "source_path": str(source.get("path") or ""),
                    "paired_path": str(paired.get("path") or "") if isinstance(paired, dict) else "",
                    "mask_path": str(instance.get("mask_path") or ""),
                    "reason": (
                        "ambiguous_semantic"
                        if priority == "ambiguous"
                        else "required_without_paired_change_contact"
                    ),
                }
            )

    cell = (300, 210)
    columns = ("SOURCE", "PAIRED PROPOSAL", "MASK OVERLAY")
    sheet_paths: list[str] = []
    for sheet_index, start in enumerate(range(0, len(review_rows), rows_per_sheet), 1):
        group = review_rows[start:start + rows_per_sheet]
        canvas = Image.new(
            "RGB",
            (cell[0] * len(columns), (cell[1] + 40) * len(group)),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        for row_index, record in enumerate(group):
            source_bgr = _read_image(Path(record["source_path"]))
            mask = _read_image(Path(record["mask_path"]), cv2.IMREAD_GRAYSCALE)
            x1, y1, x2, y2 = _crop_bounds(mask)
            source_crop = source_bgr[y1:y2, x1:x2]
            paired_path = Path(record["paired_path"]) if record["paired_path"] else None
            paired_crop = (
                _read_image(paired_path)[y1:y2, x1:x2]
                if paired_path is not None
                else np.full_like(source_crop, 235)
            )
            local_mask = mask[y1:y2, x1:x2]
            overlay = source_crop.copy()
            tint = overlay.copy()
            tint[local_mask > 0] = np.array([30, 30, 250], np.uint8)
            overlay = cv2.addWeighted(overlay, 0.5, tint, 0.5, 0)
            images = (source_crop, paired_crop, overlay)
            for column_index, (label, image_bgr) in enumerate(zip(columns, images)):
                image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
                image = _resize_for_cell(image, cell)
                x = column_index * cell[0] + (cell[0] - image.width) // 2
                y = row_index * (cell[1] + 40) + 28
                canvas.paste(image, (x, y))
                draw.text(
                    (column_index * cell[0] + 4, y - 18),
                    f"{record['review_id']} {label}",
                    fill="black",
                )
            draw.text(
                (4, row_index * (cell[1] + 40) + 5),
                f"{record['page_id']} {record['instance_id']} {record['reason']}",
                fill="black",
            )
        path = output_dir / f"semantic-review-{sheet_index:02d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, quality=94)
        sheet_paths.append(str(path.resolve()))

    payload = {
        "schema_version": "inpaint-source-review-ledger-v4",
        "candidate_seen": False,
        "review_complete": False,
        "review_row_count": len(review_rows),
        "rows": review_rows,
        "sheets": sheet_paths,
    }
    ledger = output_dir / "semantic-review-ledger.json"
    ledger.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build source-only semantic review sheets.")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    archive = default_archive_root().resolve()
    output_dir = args.output_dir.resolve()
    try:
        output_dir.relative_to(archive)
    except ValueError as exc:
        raise ValueError("review output must stay in the private archive") from exc
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    payload = build_review_sheets(
        args.source_manifest.resolve(),
        args.decisions.resolve(),
        output_dir,
    )
    print(json.dumps({"rows": payload["review_row_count"], "sheets": len(payload["sheets"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
