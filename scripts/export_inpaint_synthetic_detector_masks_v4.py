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

from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    load_stage1_manifest,
)
from benchmarking.inpaint_detector_bakeoff.synthetic_detector import (  # noqa: E402
    CTDSyntheticFineTuneReference,
)
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-ctd-synthetic-finetune-export-v4"
CATEGORY = "40-inpaint-mask-render"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export synthetic-fine-tuned CTD masks for a sealed manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detect-size", type=int, default=1280)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = args.manifest.resolve()
    base_model = args.base_model.resolve()
    checkpoint = args.checkpoint.resolve()
    output, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output.mkdir(parents=True, exist_ok=True)
    try:
        detector = CTDSyntheticFineTuneReference(
            base_model,
            checkpoint,
            device=str(args.device),
            detect_size=int(args.detect_size),
        )
        rows = []
        for page in load_stage1_manifest(manifest):
            image = cv2.imdecode(
                np.fromfile(page.source_image, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is None or image.size == 0:
                raise FileNotFoundError(page.source_image)
            result = detector.infer(image)
            path = output / f"{page.page_id}_ctd_synthetic_finetune_raw.png"
            if not cv2.imwrite(str(path), result.raw_mask):
                raise OSError(path)
            rows.append(
                {
                    "page_id": page.page_id,
                    "path": str(path),
                    "pixel_count": int(cv2.countNonZero(result.raw_mask)),
                    "mask_sha256": hashlib.sha256(
                        result.raw_mask.tobytes(order="C")
                    ).hexdigest(),
                    "runtime": dict(result.runtime),
                }
            )
        payload = {
            "schema_version": "inpaint-ctd-synthetic-finetune-export-v4",
            "manifest_sha256": _sha256(manifest),
            "base_model_sha256": _sha256(base_model),
            "checkpoint_sha256": _sha256(checkpoint),
            "device": str(args.device),
            "detect_size": int(args.detect_size),
            "pages": rows,
        }
        (output / "synthetic-detector-index.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if managed is not None:
            managed.complete(
                metadata={
                    "manifest_sha256": payload["manifest_sha256"],
                    "base_model_sha256": payload["base_model_sha256"],
                    "checkpoint_sha256": payload["checkpoint_sha256"],
                    "page_count": len(rows),
                    "device": payload["device"],
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
            managed.fail(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
