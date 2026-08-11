#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-v3-four-case-contact-sheet"
CATEGORY = "40-inpaint-mask-render"
ROW_KINDS = ("small_text", "clean_bubble", "halftone", "line_adjacent")
HEADERS = ("source", "rewritten PR3", "candidate 1", "candidate 2", "edit / protect")


def _read_bgr(path: object) -> np.ndarray:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("contact sheet image path must be a non-empty string")
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise FileNotFoundError(path)
    return image


def _read_mask(path: object, shape: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(shape, np.uint8)
    if not isinstance(path, str) or not path.strip():
        raise ValueError("contact sheet mask path must be null or a non-empty string")
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.size == 0:
        raise FileNotFoundError(path)
    if mask.shape != shape:
        raise ValueError(f"contact sheet mask shape mismatch: {mask.shape} != {shape}")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _crop(image: np.ndarray, xyxy: object) -> np.ndarray:
    if not isinstance(xyxy, list) or len(xyxy) != 4:
        raise ValueError("contact sheet row requires crop_xyxy")
    x1, y1, x2, y2 = (int(value) for value in xyxy)
    height, width = image.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid contact sheet crop: {xyxy}")
    return np.ascontiguousarray(image[y1:y2, x1:x2])


def _overlay(source: np.ndarray, edit: np.ndarray, protect: np.ndarray) -> np.ndarray:
    output = source.copy()
    edit_pixels = edit > 0
    protect_pixels = protect > 0
    if np.any(edit_pixels):
        green = np.zeros_like(output)
        green[:] = (40, 220, 40)
        output[edit_pixels] = np.rint(
            output[edit_pixels].astype(np.float32) * 0.35
            + green[edit_pixels].astype(np.float32) * 0.65
        ).astype(np.uint8)
    if np.any(protect_pixels):
        red = np.zeros_like(output)
        red[:] = (40, 40, 240)
        output[protect_pixels] = np.rint(
            output[protect_pixels].astype(np.float32) * 0.35
            + red[protect_pixels].astype(np.float32) * 0.65
        ).astype(np.uint8)
    return output


def build_contact_sheet(spec: dict[str, object]) -> Image.Image:
    rows = spec.get("rows")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("contact sheet requires exactly four rows")
    kinds = tuple(str(row.get("kind")) for row in rows if isinstance(row, dict))
    if kinds != ROW_KINDS:
        raise ValueError(f"contact sheet rows must be ordered as {ROW_KINDS}")
    cell_width = int(spec.get("cell_width", 360))
    cell_height = int(spec.get("cell_height", 260))
    header_height = 34
    row_label_height = 28
    canvas = Image.new(
        "RGB",
        (cell_width * len(HEADERS), header_height + (cell_height + row_label_height) * 4),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, header in enumerate(HEADERS):
        draw.text((column * cell_width + 8, 10), header, fill="black", font=font)

    for row_index, row in enumerate(rows):
        assert isinstance(row, dict)
        source = _read_bgr(row.get("source"))
        shape = source.shape[:2]
        images = [
            source,
            _read_bgr(row.get("control")),
            _read_bgr(row.get("candidate_1")),
            _read_bgr(row.get("candidate_2")),
        ]
        if any(image.shape != source.shape for image in images):
            raise ValueError("contact sheet row images must have identical shapes")
        edit = _read_mask(row.get("edit_mask"), shape)
        protect = _read_mask(row.get("protect_mask"), shape)
        images.append(_overlay(source, edit, protect))
        y_base = header_height + row_index * (cell_height + row_label_height)
        draw.text(
            (8, y_base + cell_height + 7),
            str(row.get("label") or row["kind"]),
            fill="black",
            font=font,
        )
        for column, image in enumerate(images):
            crop = _crop(image, row.get("crop_xyxy"))
            rgb = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            rgb.thumbnail((cell_width - 8, cell_height - 8), Image.Resampling.LANCZOS)
            x = column * cell_width + (cell_width - rgb.width) // 2
            y = y_base + (cell_height - rgb.height) // 2
            canvas.paste(rgb, (x, y))
    return canvas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the single four-case inpaint v3 human review sheet."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != "inpaint-v3-contact-sheet-v1":
        raise ValueError("unsupported contact sheet schema")
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        sheet = build_contact_sheet(spec)
        output = output_root / "inpaint-v3-four-case-contact-sheet.png"
        sheet.save(output, format="PNG", optimize=False)
        if managed is not None:
            managed.complete(metadata={"spec": spec_path.name, "row_count": 4})
            mismatches = managed.verify()
            if mismatches:
                raise RuntimeError("managed artifact verification failed: " + "; ".join(mismatches))
            print(managed.run_root)
        else:
            print(output_root)
        return 0
    except BaseException as error:
        if managed is not None:
            managed.fail(error, metadata={"spec": spec_path.name})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
