#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-positive-mask-parity-v1"
CATEGORY = "40-inpaint-mask-render"


def _mask_map(root: Path, directory: str) -> dict[str, Path]:
    mask_root = root / directory
    return {path.name: path for path in sorted(mask_root.glob("*.png"))}


def _compare_kind(reference: Path, candidate: Path, directory: str):
    left = _mask_map(reference, directory)
    right = _mask_map(candidate, directory)
    if not left or not right:
        raise ValueError(f"{directory} contains no parity masks")
    if left.keys() != right.keys():
        raise ValueError(
            f"{directory} file mismatch: "
            f"reference_only={sorted(left.keys() - right.keys())!r}, "
            f"candidate_only={sorted(right.keys() - left.keys())!r}"
        )
    rows: list[dict[str, object]] = []
    suffix = f"_{directory.removesuffix('_masks')}.png"
    for name in left:
        first = cv2.imread(str(left[name]), cv2.IMREAD_GRAYSCALE)
        second = cv2.imread(str(right[name]), cv2.IMREAD_GRAYSCALE)
        if first is None or second is None or first.shape != second.shape:
            raise ValueError(f"invalid parity masks for {name}")
        rows.append(
            {
                "page_id": name.removesuffix(suffix),
                "file": name,
                "reference_pixel_count": int(np.count_nonzero(first)),
                "candidate_pixel_count": int(np.count_nonzero(second)),
                "xor_pixel_count": int(np.count_nonzero(first != second)),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Require exact final positive-edit parity between Stage 1 runs."
    )
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reference = args.reference_root.resolve()
    candidate = args.candidate_root.resolve()
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        edits = _compare_kind(
            reference,
            candidate,
            "positive_edit_masks",
        )
        claims = _compare_kind(
            reference,
            candidate,
            "positive_claim_masks",
        )
        payload = {
            "schema_version": "inpaint-positive-mask-parity-v1",
            "reference_root": str(reference),
            "candidate_root": str(candidate),
            "final_edit_xor_pixel_count": sum(
                int(row["xor_pixel_count"]) for row in edits
            ),
            "claim_xor_pixel_count": sum(
                int(row["xor_pixel_count"]) for row in claims
            ),
            "edit_pages": edits,
            "claim_pages": claims,
        }
        result_path = output_root / "parity-results.json"
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if managed is not None:
            managed.complete(
                metadata={
                    "final_edit_xor_pixel_count": payload[
                        "final_edit_xor_pixel_count"
                    ],
                    "claim_xor_pixel_count": payload["claim_xor_pixel_count"],
                }
            )
        print(output_root)
        exact = (
            payload["final_edit_xor_pixel_count"] == 0
            and payload["claim_xor_pixel_count"] == 0
        )
        return 0 if exact else 1
    except Exception as error:
        if managed is not None:
            managed.fail(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
