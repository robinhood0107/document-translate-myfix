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

from benchmarking.inpaint_detector_bakeoff.ballons_ctd import (  # noqa: E402
    BallonsCTDFullPageReference,
)
from benchmarking.inpaint_detector_bakeoff.reference_probe import (  # noqa: E402
    load_ballons_ctd_inference_reference,
    reference_ctd_raw_mask,
)
from modules.masking.ctd_refiner import CTDRefinerSettings  # noqa: E402
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synthetic_rgb() -> np.ndarray:
    image = np.full((257, 383, 3), 238, dtype=np.uint8)
    cv2.ellipse(image, (122, 112), (78, 53), 0, 0, 360, (252, 252, 252), -1)
    cv2.putText(image, "ABC", (68, 124), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (18, 18, 18), 3)
    cv2.putText(image, "SFX", (242, 201), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (38, 38, 38), 2)
    return image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ballons-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--detect-size", type=int, default=1280)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    output_root, managed = select_managed_output_directory(
        family="inpaint-ctd-reference-parity-v2",
        category="40-inpaint-mask-render",
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        if args.image is None:
            image_rgb = _synthetic_rgb()
        else:
            image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
            if image_bgr is None or image_bgr.size == 0:
                raise FileNotFoundError("parity source image is unreadable")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        adapter = BallonsCTDFullPageReference(
            CTDRefinerSettings(
                detect_size=int(args.detect_size),
                det_rearrange_max_batches=4,
                device=args.device,
                mask_dilate_size=0,
            ),
            dilate_size=3,
        )
        port = adapter.infer(image_rgb)
        reference_module = load_ballons_ctd_inference_reference(args.ballons_root)
        reference = reference_ctd_raw_mask(
            reference_module,
            adapter.refiner.net,
            image_rgb,
            int(args.detect_size),
        )
        reference = np.where(reference > 0, 255, 0).astype(np.uint8)
        xor = cv2.bitwise_xor(reference, port.raw_mask)
        result = {
            "schema_version": "inpaint-ctd-reference-parity-v1",
            "reference_source_sha256": _sha256(
                args.ballons_root
                / "ballontranslator"
                / "modules"
                / "textdetector"
                / "ctd"
                / "inference.py"
            ),
            "model_sha256": _sha256(args.model),
            "backend": port.runtime.get("backend"),
            "device": args.device,
            "binary_mask_xor_pixel_count": int(np.count_nonzero(xor)),
            "passed": not np.any(xor),
        }
        (output_root / "parity.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if not cv2.imwrite(str(output_root / "binary-mask-xor.png"), xor):
            raise OSError("failed to write CTD XOR mask")
        if managed is not None:
            if result["passed"]:
                managed.complete(metadata=result)
            else:
                managed.fail("CTD reference parity mismatch", metadata=result)
            print(str(managed.run_root))
        else:
            print(str(output_root))
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    except BaseException as error:
        if managed is not None:
            managed.fail(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
