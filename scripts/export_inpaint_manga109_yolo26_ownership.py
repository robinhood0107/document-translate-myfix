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

from benchmarking.inpaint_detector_bakeoff.manga109_yolo26 import (  # noqa: E402
    Manga109YOLO26OwnershipReference,
    Manga109YOLO26Settings,
)
from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    load_stage1_manifest,
)
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-manga109-yolo26-evidence-v3"
CATEGORY = "40-inpaint-mask-render"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Manga109 YOLO26 text masks as ownership evidence only.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest.resolve()
    model_path = args.model.resolve()
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        reference = Manga109YOLO26OwnershipReference(
            model_path,
            Manga109YOLO26Settings(
                image_size=int(args.image_size),
                confidence=float(args.confidence),
                iou=float(args.iou),
                device=str(args.device),
            ),
        )
        rows: list[dict[str, object]] = []
        for page in load_stage1_manifest(manifest_path):
            image = cv2.imdecode(
                np.fromfile(page.source_image, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is None or image.size == 0:
                raise FileNotFoundError(page.source_image)
            result, balloon = reference.infer_evidence(image)
            path = output_root / f"{page.page_id}_ownership.png"
            balloon_path = output_root / f"{page.page_id}_balloon.png"
            if not cv2.imwrite(str(path), result.raw_mask):
                raise OSError(f"failed to write ownership mask: {path}")
            if not cv2.imwrite(str(balloon_path), balloon.raw_mask):
                raise OSError(f"failed to write balloon mask: {balloon_path}")
            rows.append(
                {
                    "page_id": page.page_id,
                    "pixel_count": int(cv2.countNonZero(result.raw_mask)),
                    "mask_sha256": hashlib.sha256(
                        result.raw_mask.tobytes(order="C")
                    ).hexdigest(),
                    "path": str(path),
                    "balloon_pixel_count": int(
                        cv2.countNonZero(balloon.raw_mask)
                    ),
                    "balloon_mask_sha256": hashlib.sha256(
                        balloon.raw_mask.tobytes(order="C")
                    ).hexdigest(),
                    "balloon_path": str(balloon_path),
                    "runtime": dict(result.runtime),
                }
            )
        payload = {
            "schema_version": "inpaint-manga109-yolo26-evidence-v2",
            "manifest_sha256": _sha256(manifest_path),
            "model_sha256": _sha256(model_path),
            "model_size_bytes": model_path.stat().st_size,
            "settings": {
                "device": str(args.device),
                "image_size": int(args.image_size),
                "confidence": float(args.confidence),
                "iou": float(args.iou),
                "retina_masks": True,
                "text_class_id": 1,
                "balloon_class_id": 2,
            },
            "pages": rows,
        }
        index_path = output_root / "ownership-index.json"
        index_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if managed is not None:
            managed.complete(
                metadata={
                    "manifest_sha256": payload["manifest_sha256"],
                    "model_sha256": payload["model_sha256"],
                    "page_count": len(rows),
                }
            )
            mismatches = managed.verify()
            if mismatches:
                raise RuntimeError(
                    "managed artifact verification failed: " + "; ".join(mismatches)
                )
            print(str(managed.run_root))
        else:
            print(str(output_root))
        return 0
    except BaseException as error:
        if managed is not None:
            managed.fail(error, metadata={"manifest": manifest_path.name})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
