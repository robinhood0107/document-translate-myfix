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

from benchmarking.inpaint_detector_bakeoff.ballons_ctbd import (  # noqa: E402
    BallonsCTBDReference,
    _build_content_mask,
)
from benchmarking.inpaint_detector_bakeoff.contracts import DetectorBox  # noqa: E402
from benchmarking.inpaint_detector_bakeoff.reference_probe import (  # noqa: E402
    load_ballons_ctbd_reference,
    make_ballons_ctbd_detector,
    reference_ctbd_single_image,
)
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synthetic_image() -> np.ndarray:
    image = np.full((192, 256, 3), 232, dtype=np.uint8)
    cv2.ellipse(image, (80, 88), (58, 42), 0, 0, 360, (248, 248, 248), -1)
    cv2.putText(image, "ABC", (43, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (18, 18, 18), 2)
    cv2.putText(image, "SFX", (155, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (35, 35, 35), 2)
    return image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ballons-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    output_root, managed = select_managed_output_directory(
        family="inpaint-detector-reference-parity-v2",
        category="40-inpaint-mask-render",
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        import onnxruntime as ort

        image = (
            cv2.imread(str(args.image), cv2.IMREAD_COLOR)
            if args.image is not None
            else _synthetic_image()
        )
        if image is None or image.size == 0:
            raise FileNotFoundError("parity source image is unreadable")

        providers = [args.provider]
        if args.provider != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")
        session = ort.InferenceSession(str(args.model), providers=providers)

        original_module = load_ballons_ctbd_reference(args.ballons_root)
        original_detector = make_ballons_ctbd_detector(original_module, session)
        ref_bubbles, ref_texts, ref_mask, ref_blocks = reference_ctbd_single_image(
            original_module,
            original_detector,
            image,
        )

        port = BallonsCTBDReference(str(args.model), providers)
        port_bubbles_array, port_texts_array = port._detect_single(image)
        port_bubbles = tuple(
            DetectorBox(tuple(map(int, box)), "bubble", 1.0, "port")
            for box in np.asarray(port_bubbles_array).reshape(-1, 4)
        )
        port_texts = tuple(
            DetectorBox(tuple(map(int, box)), "text", 1.0, "port")
            for box in np.asarray(port_texts_array).reshape(-1, 4)
        )
        port_mask, port_blocks = _build_content_mask(
            image,
            port_texts,
            port_bubbles,
            port.settings,
        )

        bubbles_equal = np.array_equal(ref_bubbles, port_bubbles_array)
        texts_equal = np.array_equal(ref_texts, port_texts_array)
        mask_xor = cv2.bitwise_xor(
            np.where(ref_mask > 0, 255, 0).astype(np.uint8),
            np.where(port_mask > 0, 255, 0).astype(np.uint8),
        )
        port_block_records = [
            {"xyxy": block.xyxy, "text_class": block.label}
            for block in port_blocks
        ]
        result = {
            "schema_version": "inpaint-detector-reference-parity-v1",
            "reference_source_sha256": _sha256(
                args.ballons_root
                / "ballontranslator"
                / "modules"
                / "textdetector"
                / "detector_ctbd.py"
            ),
            "model_sha256": _sha256(args.model),
            "provider": session.get_providers(),
            "bubble_boxes_equal": bubbles_equal,
            "text_boxes_equal": texts_equal,
            "binary_mask_xor_pixel_count": int(np.count_nonzero(mask_xor)),
            "reference_block_count": len(ref_blocks),
            "ported_block_count": len(port_block_records),
            "passed": bool(bubbles_equal and texts_equal and not np.any(mask_xor)),
        }
        (output_root / "parity.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if not cv2.imwrite(str(output_root / "binary-mask-xor.png"), mask_xor):
            raise OSError("failed to write parity XOR mask")

        if managed is not None:
            if result["passed"]:
                managed.complete(metadata=result)
            else:
                managed.fail("detector reference parity mismatch", metadata=result)
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
