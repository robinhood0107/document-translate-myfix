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

from benchmarking.inpaint_detector_bakeoff.ballons_ysg import (  # noqa: E402
    BallonsYSGReference,
)
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-ysg-ownership-v2"
CATEGORY = "40-inpaint-mask-render"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        detector = BallonsYSGReference(str(args.model), device=args.device)
        rows = []
        for page in manifest.get("pages", []):
            page_id = str(page.get("page_id") or "").strip()
            image = cv2.imdecode(
                np.fromfile(str(page.get("path") or ""), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if not page_id or image is None or image.size == 0:
                raise ValueError(f"invalid YSG source page: {page_id}")
            result = detector.infer(image)
            path = output_root / f"{page_id}_ysg_ownership.png"
            if not cv2.imwrite(str(path), result.raw_mask):
                raise OSError(path)
            rows.append(
                {
                    "page_id": page_id,
                    "path": str(path),
                    "pixel_count": int(cv2.countNonZero(result.raw_mask)),
                    "boxes": [
                        {
                            "xyxy": list(box.xyxy),
                            "label": box.label,
                            "provider": box.provider,
                        }
                        for box in result.boxes
                    ],
                    "runtime": result.runtime,
                }
            )
        payload = {
            "schema_version": "inpaint-ysg-ownership-v2",
            "manifest_sha256": _sha256(args.manifest),
            "model_sha256": _sha256(args.model),
            "device": args.device,
            "pages": rows,
        }
        (output_root / "ysg-ownership-index.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if managed is not None:
            managed.complete(
                metadata={
                    "manifest_sha256": payload["manifest_sha256"],
                    "model_sha256": payload["model_sha256"],
                    "page_count": len(rows),
                    "device": args.device,
                }
            )
            mismatches = managed.verify()
            if mismatches:
                raise RuntimeError("managed artifact verification failed: " + "; ".join(mismatches))
            print(str(managed.run_root))
        else:
            print(str(output_root))
        return 0
    except BaseException as error:
        if managed is not None:
            managed.fail(error, metadata={"manifest": args.manifest.name})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
