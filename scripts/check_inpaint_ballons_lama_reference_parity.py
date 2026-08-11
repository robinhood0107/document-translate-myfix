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

from benchmarking.inpaint_detector_bakeoff.reference_probe import (  # noqa: E402
    load_ballons_lama_runtime_reference,
)
from modules.inpainting.source_lama_blockwise import SourceLaMaLarge  # noqa: E402
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


def _pixel_sha256(image: np.ndarray) -> str:
    value = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    height, width = 257, 383
    y, x = np.mgrid[0:height, 0:width]
    image = np.stack(
        (
            (x * 3 + y * 2 + 17) % 256,
            (x * 5 + y * 7 + 29) % 256,
            (x * 11 + y * 13 + 43) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (42, 37), (126, 104), 255, -1)
    cv2.circle(mask, (252, 164), 31, 255, -1)
    cv2.line(mask, (165, 219), (338, 203), 255, 5)
    return image, mask


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ballons-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    args = parser.parse_args(argv)
    output, managed = select_managed_output_directory(
        family="inpaint-ballons-lama-reference-parity-v2",
        category="40-inpaint-mask-render",
        explicit_output_directory=args.output_dir,
    )
    output.mkdir(parents=True, exist_ok=True)
    try:
        source = SourceLaMaLarge(
            device=args.device,
            precision=args.precision,
            inpaint_size=1536,
        )
        source.ensure_loaded()
        reference_module = load_ballons_lama_runtime_reference(args.ballons_root)
        reference = reference_module.LamaLarge()
        reference.model = source.model
        reference.device = args.device
        reference.precision = args.precision
        reference.inpaint_size = 1536
        image, mask = _fixture()
        expected = reference._inpaint(image.copy(), mask.copy())
        actual = source._inpaint(image.copy(), mask.copy())
        mismatch = np.any(expected != actual, axis=2)
        record = {
            "schema_version": "inpaint-ballons-lama-reference-parity-v1",
            "device": args.device,
            "precision": args.precision,
            "shape": list(image.shape),
            "mask_pixel_count": int(np.count_nonzero(mask)),
            "reference_pixel_sha256": _pixel_sha256(expected),
            "source_parity_pixel_sha256": _pixel_sha256(actual),
            "mismatch_pixel_count": int(np.count_nonzero(mismatch)),
            "maximum_channel_delta": int(
                np.max(np.abs(expected.astype(np.int16) - actual.astype(np.int16)))
            ),
        }
        (output / "parity.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if managed is not None:
            managed.complete(metadata=record)
            mismatches = managed.verify()
            if mismatches:
                raise RuntimeError("managed artifact verification failed: " + "; ".join(mismatches))
            print(str(managed.run_root))
        else:
            print(str(output))
        print(json.dumps(record, sort_keys=True))
        return 0 if record["mismatch_pixel_count"] == 0 else 1
    except BaseException as error:
        if managed is not None:
            managed.fail(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
