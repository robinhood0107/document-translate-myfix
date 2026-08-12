#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.contracts import binary_mask  # noqa: E402
from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    load_stage1_manifest,
)
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-silhouette-consensus-v4"
CATEGORY = "40-inpaint-mask-render"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    value = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if value is None or value.size == 0:
        raise FileNotFoundError(path)
    return binary_mask(value, shape)


def _write_mask(path: Path, mask: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = cv2.imencode(".png", binary_mask(mask))[1]
    encoded.tofile(str(path))
    return str(path.resolve())


def consensus_masks(
    masks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if set(masks) != {"ballons", "pr2", "ctbd", "manga109"}:
        raise ValueError("silhouette consensus requires all four named inputs")
    shape = next(iter(masks.values())).shape
    normalized = {name: binary_mask(mask, shape) for name, mask in masks.items()}
    stack = np.stack([mask > 0 for mask in normalized.values()], axis=0)
    return {
        "ballons_pr2_union": np.where(
            (normalized["ballons"] > 0) | (normalized["pr2"] > 0), 255, 0
        ).astype(np.uint8),
        "ballons_pr2_intersection": np.where(
            (normalized["ballons"] > 0) & (normalized["pr2"] > 0), 255, 0
        ).astype(np.uint8),
        "two_of_four_consensus": np.where(np.count_nonzero(stack, axis=0) >= 2, 255, 0)
        .astype(np.uint8),
        "three_of_four_consensus": np.where(np.count_nonzero(stack, axis=0) >= 3, 255, 0)
        .astype(np.uint8),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export source-only union/intersection/consensus silhouettes."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ballons-template", required=True)
    parser.add_argument("--pr2-template", required=True)
    parser.add_argument("--ctbd-template", required=True)
    parser.add_argument("--manga109-template", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    manifest_path = args.manifest.resolve()
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        rows: list[dict[str, object]] = []
        templates = {
            "ballons": args.ballons_template,
            "pr2": args.pr2_template,
            "ctbd": args.ctbd_template,
            "manga109": args.manga109_template,
        }
        for page in load_stage1_manifest(manifest_path):
            shape = None
            inputs: dict[str, np.ndarray] = {}
            for name, template in templates.items():
                path = Path(str(template).format(page_id=page.page_id))
                if shape is None:
                    value = cv2.imdecode(
                        np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
                    )
                    if value is None or value.size == 0:
                        raise FileNotFoundError(path)
                    shape = value.shape
                    inputs[name] = binary_mask(value)
                else:
                    inputs[name] = _read_mask(path, shape)
            assert shape is not None
            products = consensus_masks(inputs)
            paths = {
                name: _write_mask(output_root / name / f"{page.page_id}.png", mask)
                for name, mask in products.items()
            }
            rows.append(
                {
                    "page_id": page.page_id,
                    "paths": paths,
                    "pixel_counts": {
                        name: int(np.count_nonzero(mask))
                        for name, mask in products.items()
                    },
                }
            )
        payload = {
            "schema_version": "inpaint-silhouette-consensus-v4",
            "manifest_sha256": _sha256(manifest_path),
            "input_templates": templates,
            "pages": rows,
        }
        index = output_root / "silhouette-consensus-index.json"
        index.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if managed is not None:
            managed.complete(
                metadata={
                    "manifest_sha256": payload["manifest_sha256"],
                    "page_count": len(rows),
                }
            )
            mismatches = managed.verify()
            if mismatches:
                raise RuntimeError(
                    "managed artifact verification failed: " + "; ".join(mismatches)
                )
            print(managed.run_root)
        else:
            print(output_root)
        return 0
    except BaseException as error:
        if managed is not None:
            managed.fail(error, metadata={"manifest": manifest_path.name})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
