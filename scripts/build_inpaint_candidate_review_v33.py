#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import hashlib
import sys
from typing import Mapping

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.contracts import binary_mask  # noqa: E402
from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    _resolve_manifest_artifact,
    load_page_masks,
    load_stage1_manifest,
    validate_source_only_manifest_v4,
)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"review JSON root must be an object: {path}")
    return value


def _read_image(path: str | Path, flags: int) -> np.ndarray:
    result = cv2.imdecode(np.fromfile(Path(path), dtype=np.uint8), flags)
    if result is None or result.size == 0:
        raise FileNotFoundError(path)
    return result


def _read_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    result = _read_image(path, cv2.IMREAD_GRAYSCALE)
    if result.shape != shape:
        raise ValueError("review mask shape differs")
    return binary_mask(result, shape)


def _result_artifacts(run_root: Path) -> dict[tuple[str, str], Path]:
    result = _read_json(run_root / "stage2-results.json")
    binding = result.get("output_inventory")
    if not isinstance(binding, Mapping):
        raise ValueError("review stage2 result lacks output inventory")
    relative = str(binding.get("relative_path") or "")
    if not relative or Path(relative).is_absolute():
        raise ValueError("review stage2 inventory path must be relative")
    inventory_path = (run_root / relative).resolve()
    if not inventory_path.is_file() or binding.get("artifact_sha256") != (
        hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    ):
        raise ValueError("review stage2 inventory file SHA differs")
    inventory = _read_json(inventory_path)
    records = inventory.get("records")
    if not isinstance(records, list):
        raise ValueError("review stage2 inventory lacks records")
    result_paths: dict[tuple[str, str], Path] = {}
    for value in records:
        if not isinstance(value, Mapping):
            raise ValueError("review stage2 inventory record is invalid")
        page_id = str(value.get("page_id") or "")
        role = str(value.get("role") or "")
        relative_path = str(value.get("relative_path") or "")
        if not page_id or not role or not relative_path or Path(relative_path).is_absolute():
            raise ValueError("review stage2 artifact identity is invalid")
        path = (run_root / relative_path).resolve()
        try:
            path.relative_to(run_root.resolve())
        except ValueError as error:
            raise ValueError("review stage2 artifact escapes its run") from error
        if not path.is_file() or (page_id, role) in result_paths:
            raise ValueError("review stage2 artifact is missing or duplicate")
        if value.get("file_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("review stage2 artifact file SHA differs")
        result_paths[(page_id, role)] = path
    return result_paths


def _overlay(
    source: np.ndarray,
    addition: np.ndarray,
    protect: np.ndarray,
) -> np.ndarray:
    result = source.copy()
    for mask, color in ((addition, (0, 220, 0)), (protect, (0, 0, 255))):
        selected = mask > 0
        if np.any(selected):
            tint = np.zeros_like(result)
            tint[:] = color
            result[selected] = np.round(
                result[selected].astype(np.float32) * 0.4
                + tint[selected].astype(np.float32) * 0.6
            ).astype(np.uint8)
    return result


def _thumbnail(value: np.ndarray, size: tuple[int, int]) -> Image.Image:
    rgb = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    image.thumbnail(size, Image.Resampling.LANCZOS)
    return image


def _difference_boxes(mask: np.ndarray, *, limit: int = 12) -> list[tuple[int, int, int, int]]:
    if not np.any(mask):
        return []
    joined = cv2.dilate(
        binary_mask(mask),
        cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31)),
        iterations=1,
    )
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (joined > 0).astype(np.uint8), 8, cv2.CV_32S
    )
    height, width = mask.shape
    boxes: list[tuple[int, int, int, int, int]] = []
    for index in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[index])
        if area <= 0:
            continue
        margin = 40
        boxes.append(
            (
                max(0, x - margin),
                max(0, y - margin),
                min(width, x + box_width + margin),
                min(height, y + box_height + margin),
                area,
            )
        )
    boxes.sort(key=lambda value: (-value[4], value[1], value[0]))
    return [value[:4] for value in boxes[:limit]]


def build_page_panel(
    *,
    page_label: str,
    source: np.ndarray,
    baseline: np.ndarray,
    candidate_a: np.ndarray,
    candidate_b: np.ndarray | None,
    addition: np.ndarray,
    protect: np.ndarray,
) -> Image.Image:
    overlay = _overlay(source, addition, protect)
    values = [source, baseline, candidate_a]
    labels = ["SOURCE", "PR6 CONTROL", "FINALIST A"]
    if candidate_b is not None:
        values.append(candidate_b)
        labels.append("FINALIST B")
    values.append(overlay)
    labels.append("GREEN NEW EDIT / RED PROTECT")
    column_width = 360
    full_height = 520
    header_height = 48
    full = Image.new(
        "RGB",
        (column_width * len(values), full_height + header_height),
        "white",
    )
    draw = ImageDraw.Draw(full)
    draw.text((8, 5), page_label, fill="black")
    for index, (value, label) in enumerate(zip(values, labels)):
        image = _thumbnail(value, (column_width - 12, full_height - 16))
        x = index * column_width + (column_width - image.width) // 2
        y = header_height + (full_height - image.height) // 2
        full.paste(image, (x, y))
        draw.text((index * column_width + 8, 26), label, fill="black")

    difference = changed_union(baseline, candidate_a, candidate_b)
    boxes = _difference_boxes(difference)
    if not boxes:
        return full
    crop_width = 300
    crop_height = 260
    crop_label_height = 28
    crop_rows: list[Image.Image] = []
    for number, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        row = Image.new(
            "RGB",
            (crop_width * len(values), crop_height + crop_label_height),
            "white",
        )
        row_draw = ImageDraw.Draw(row)
        row_draw.text((8, 7), f"CHANGE {number}: ({x1},{y1})-({x2},{y2})", fill="black")
        for index, value in enumerate(values):
            image = _thumbnail(value[y1:y2, x1:x2], (crop_width - 10, crop_height - 8))
            x = index * crop_width + (crop_width - image.width) // 2
            y = crop_label_height + (crop_height - image.height) // 2
            row.paste(image, (x, y))
        crop_rows.append(row)
    width = max(full.width, max(row.width for row in crop_rows))
    height = full.height + sum(row.height for row in crop_rows)
    result = Image.new("RGB", (width, height), "white")
    result.paste(full, (0, 0))
    cursor = full.height
    for row in crop_rows:
        result.paste(row, (0, cursor))
        cursor += row.height
    return result


def changed_union(
    baseline: np.ndarray,
    candidate_a: np.ndarray,
    candidate_b: np.ndarray | None,
) -> np.ndarray:
    changed = np.any(baseline != candidate_a, axis=2)
    if candidate_b is not None:
        changed |= np.any(baseline != candidate_b, axis=2)
    return np.where(changed, 255, 0).astype(np.uint8)


def build_review(
    *,
    relative_manifest_path: Path,
    candidate_a_run: Path,
    candidate_b_run: Path | None,
    page_id_prefix: str,
    output_root: Path,
) -> dict[str, object]:
    validate_source_only_manifest_v4(relative_manifest_path)
    pages = [
        page
        for page in load_stage1_manifest(relative_manifest_path)
        if page.page_id.startswith(page_id_prefix)
    ]
    if not pages:
        raise ValueError("review page prefix selected no pages")
    entries_payload = _read_json(relative_manifest_path)
    entries = {
        str(row.get("page_id") or ""): row
        for row in entries_payload.get("pages", [])
        if isinstance(row, Mapping)
    }
    artifacts_a = _result_artifacts(candidate_a_run.resolve())
    artifacts_b = (
        _result_artifacts(candidate_b_run.resolve())
        if candidate_b_run is not None
        else {}
    )
    output_root.mkdir(parents=True, exist_ok=False)
    index_cells: list[Image.Image] = []
    records: list[dict[str, object]] = []
    for number, page in enumerate(pages, start=1):
        entry = entries[page.page_id]
        source = _read_image(page.source_image, cv2.IMREAD_COLOR)
        shape = source.shape[:2]
        baseline_path = _resolve_manifest_artifact(
            relative_manifest_path, entry.get("baseline")
        )
        if baseline_path is None:
            raise ValueError("review baseline path is missing")
        baseline = _read_image(baseline_path, cv2.IMREAD_COLOR)
        candidate_a = _read_image(
            artifacts_a[(page.page_id, "candidate_image")], cv2.IMREAD_COLOR
        )
        candidate_b = (
            _read_image(
                artifacts_b[(page.page_id, "candidate_image")], cv2.IMREAD_COLOR
            )
            if candidate_b_run is not None
            else None
        )
        changed = changed_union(baseline, candidate_a, candidate_b)
        addition = _read_mask(
            artifacts_a[(page.page_id, "safe_addition")], shape
        )
        if candidate_b_run is not None:
            addition = cv2.bitwise_or(
                addition,
                _read_mask(
                    artifacts_b[(page.page_id, "safe_addition")], shape
                ),
            )
        masks = load_page_masks(page, shape, strict_binary=True)
        panel = build_page_panel(
            page_label=f"{number:02d}  {Path(page.source_image).name}  [{page.page_id}]",
            source=source,
            baseline=baseline,
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            addition=addition,
            protect=cv2.bitwise_or(masks.protected, masks.ambiguous),
        )
        panel_path = output_root / "pages" / f"{number:02d}-{page.page_id}.png"
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        panel.save(panel_path)
        source_thumb = _thumbnail(source, (260, 360))
        cell = Image.new("RGB", (280, 410), "white")
        cell.paste(source_thumb, ((280 - source_thumb.width) // 2, 28))
        ImageDraw.Draw(cell).text(
            (8, 6),
            f"{number:02d} {Path(page.source_image).name}",
            fill="black",
        )
        index_cells.append(cell)
        records.append(
            {
                "number": number,
                "page_id": page.page_id,
                "source_file_name": Path(page.source_image).name,
                "difference_pixel_count": int(np.count_nonzero(changed)),
                "panel": panel_path.relative_to(output_root).as_posix(),
            }
        )
    columns = 4
    rows = (len(index_cells) + columns - 1) // columns
    index = Image.new("RGB", (columns * 280, rows * 410), "white")
    for index_value, cell in enumerate(index_cells):
        index.paste(cell, ((index_value % columns) * 280, (index_value // columns) * 410))
    index_path = output_root / "00-index.png"
    index.save(index_path)
    ledger = {
        "schema_version": "inpaint-candidate-review-v33",
        "page_id_prefix": page_id_prefix,
        "page_count": len(records),
        "candidate_a_run": str(candidate_a_run.resolve()),
        "candidate_b_run": str(candidate_b_run.resolve()) if candidate_b_run else None,
        "legend": {
            "green": "new candidate difference from PR6",
            "red": "sealed structure or ambiguous protection",
        },
        "pages": records,
    }
    (output_root / "review-ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return ledger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a readable per-page comparison for v3.3 finalists."
    )
    parser.add_argument("--relative-manifest", type=Path, required=True)
    parser.add_argument("--candidate-a-run", type=Path, required=True)
    parser.add_argument("--candidate-b-run", type=Path)
    parser.add_argument("--page-id-prefix", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    build_review(
        relative_manifest_path=args.relative_manifest.resolve(),
        candidate_a_run=args.candidate_a_run.resolve(),
        candidate_b_run=(
            args.candidate_b_run.resolve() if args.candidate_b_run else None
        ),
        page_id_prefix=args.page_id_prefix,
        output_root=args.output_dir.resolve(),
    )
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
