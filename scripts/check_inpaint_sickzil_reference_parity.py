#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.sickzil import (  # noqa: E402
    SickZilSegmentationReference,
)
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-sickzil-reference-parity-v2"
CATEGORY = "40-inpaint-mask-render"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_original_core(reference_root: Path, snet_path: Path, cnet_path: Path):
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
    pyqt = types.ModuleType("PyQt5")
    qtgui = types.ModuleType("PyQt5.QtGui")
    qtgui.QImage = type("QImage", (), {})
    pyqt.QtGui = qtgui
    sys.modules.setdefault("PyQt5", pyqt)
    sys.modules.setdefault("PyQt5.QtGui", qtgui)
    source_root = reference_root / "src"
    sys.path.insert(0, str(source_root))
    consts = importlib.import_module("consts")
    consts.SNETPATH = str(snet_path)
    consts.CNETPATH = str(cnet_path)
    return importlib.import_module("core")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--snet", type=Path, required=True)
    parser.add_argument("--cnet", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args(argv)
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
    )
    detector = None
    try:
        image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise FileNotFoundError(args.image)
        original = _load_original_core(
            args.reference_root.resolve(),
            args.snet.resolve(),
            args.cnet.resolve(),
        )
        original_mask = original.segmap(image)
        detector = SickZilSegmentationReference(str(args.snet.resolve()))
        port_mask = detector.infer(image).raw_mask
        original_binary = np.where(original_mask[..., 0] > 0, 255, 0).astype(np.uint8)
        xor = cv2.bitwise_xor(original_binary, port_mask)
        result = {
            "schema_version": "inpaint-sickzil-reference-parity-v2",
            "reference_commit": "6e6d31870a2028f9d1ea9a402c20a79fcbde04ab",
            "snet_sha256": _sha256(args.snet),
            "cnet_sha256": _sha256(args.cnet),
            "source_image_sha256": _sha256(args.image),
            "shape": list(port_mask.shape),
            "binary_mask_xor_pixel_count": int(cv2.countNonZero(xor)),
            "passed": int(cv2.countNonZero(xor)) == 0,
        }
        (output_root / "sickzil-parity.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        cv2.imwrite(str(output_root / "original-mask.png"), original_binary)
        cv2.imwrite(str(output_root / "ported-mask.png"), port_mask)
        cv2.imwrite(str(output_root / "xor.png"), xor)
        if not result["passed"]:
            raise RuntimeError(f"SickZil parity failed: {result}")
        managed.complete(metadata=result)
        mismatches = managed.verify()
        if mismatches:
            raise RuntimeError("managed artifact verification failed: " + "; ".join(mismatches))
        print(str(managed.run_root))
        return 0
    except BaseException as error:
        managed.fail(error, metadata={"model": args.snet.name})
        raise
    finally:
        if detector is not None:
            detector.close()


if __name__ == "__main__":
    raise SystemExit(main())
