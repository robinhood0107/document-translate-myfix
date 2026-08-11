#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.ownership import (  # noqa: E402
    build_existing_ownership_mask,
)
from benchmarking.inpaint_detector_bakeoff.stage1 import load_stage1_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--debug-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scope",
        choices=(
            "region",
            "text_prior",
            "content_components",
            "content_prior",
            "required_skip_text_prior",
            "required_skip_components",
        ),
        default="region",
        help="Use broad owned regions or semantic text-prior anchors for gating.",
    )
    args = parser.parse_args(argv)

    pages = load_stage1_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for page in pages:
        image = cv2.imread(page.source_image, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise FileNotFoundError(f"source image is unreadable: {page.page_id}")
        debug_path = args.debug_root / f"{page.page_id}_debug.json"
        payload = json.loads(debug_path.read_text(encoding="utf-8"))
        blocks = payload.get("blocks")
        if not isinstance(blocks, list):
            raise ValueError(f"debug metadata has no blocks: {page.page_id}")
        ownership = build_existing_ownership_mask(
            blocks,
            image.shape[:2],
            scope=args.scope,
        )
        output_path = args.output_dir / f"{page.page_id}_ownership.png"
        if not cv2.imwrite(str(output_path), ownership):
            raise OSError(f"unable to write ownership mask: {page.page_id}")
        records.append(
            {
                "page_id": page.page_id,
                "block_count": len(blocks),
                "ownership_pixel_count": int(cv2.countNonZero(ownership)),
                "path": str(output_path),
            }
        )
    (args.output_dir / "ownership-index.json").write_text(
        json.dumps(
            {
                "schema_version": "inpaint-bakeoff-ownership-v1",
                "scope": args.scope,
                "pages": records,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
