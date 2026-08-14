#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_inpaint_v3_contact_sheet import (  # noqa: E402
    _crop,
    _overlay,
    _read_bgr,
    _read_mask,
)
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-v32-three-case-contact-sheet"
CATEGORY = "40-inpaint-mask-render"
ROW_IDS = ("japan-i_102", "japan-p_015", "japan-096")


def build_three_case_sheet(spec: dict[str, object]) -> Image.Image:
    rows = spec.get("rows")
    if not isinstance(rows, list) or len(rows) != 3 or any(
        not isinstance(row, dict) for row in rows
    ):
        raise ValueError("v3.2 contact sheet requires exactly three rows")
    if tuple(str(row.get("case_id") or "") for row in rows) != ROW_IDS:
        raise ValueError(f"v3.2 contact sheet rows must be ordered as {ROW_IDS}")
    balanced_available = spec.get("balanced_available") is True
    headers = ["source", "PR6", "fill-only"]
    if balanced_available:
        headers.append("balanced")
    headers.append("edit / protect")
    cell_width = int(spec.get("cell_width", 360))
    cell_height = int(spec.get("cell_height", 280))
    header_height = 34
    label_height = 30
    canvas = Image.new(
        "RGB",
        (
            cell_width * len(headers),
            header_height + (cell_height + label_height) * len(rows),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, header in enumerate(headers):
        draw.text((column * cell_width + 8, 10), header, fill="black", font=font)
    for row_index, row in enumerate(rows):
        source = _read_bgr(row.get("source"))
        images = [
            source,
            _read_bgr(row.get("control")),
            _read_bgr(row.get("fill_only")),
        ]
        if balanced_available:
            images.append(_read_bgr(row.get("balanced")))
        if any(image.shape != source.shape for image in images):
            raise ValueError("v3.2 contact sheet row image shapes differ")
        edit = _read_mask(row.get("edit_mask"), source.shape[:2])
        protect = _read_mask(row.get("protect_mask"), source.shape[:2])
        images.append(_overlay(source, edit, protect))
        y_base = header_height + row_index * (cell_height + label_height)
        for column, image in enumerate(images):
            crop = _crop(image, row.get("crop_xyxy"))
            rendered = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            rendered.thumbnail(
                (cell_width - 8, cell_height - 8), Image.Resampling.LANCZOS
            )
            canvas.paste(
                rendered,
                (
                    column * cell_width + (cell_width - rendered.width) // 2,
                    y_base + (cell_height - rendered.height) // 2,
                ),
            )
        draw.text(
            (8, y_base + cell_height + 8),
            str(row.get("label") or row["case_id"]),
            fill="black",
            font=font,
        )
    return canvas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the single three-case v3.2 product review sheet."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or spec.get("schema_version") != (
        "inpaint-v32-three-case-contact-sheet-v1"
    ):
        raise ValueError("unsupported v3.2 contact sheet schema")
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        sheet = build_three_case_sheet(spec)
        output = output_root / "inpaint-v32-three-case-contact-sheet.png"
        sheet.save(output, format="PNG", optimize=False)
        if managed is not None:
            managed.complete(
                metadata={
                    "spec": spec_path.name,
                    "row_count": 3,
                    "balanced_available": spec.get("balanced_available") is True,
                }
            )
            mismatches = managed.verify()
            if mismatches:
                raise RuntimeError(
                    "managed artifact verification failed: " + "; ".join(mismatches)
                )
            print(managed.run_root)
        else:
            print(output)
        return 0
    except BaseException as error:
        if managed is not None:
            managed.fail(error, metadata={"spec": spec_path.name})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
