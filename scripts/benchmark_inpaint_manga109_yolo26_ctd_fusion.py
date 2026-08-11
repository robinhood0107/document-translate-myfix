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

from benchmarking.inpaint_detector_bakeoff.contracts import (  # noqa: E402
    CandidateMaskResult,
    binary_mask,
)
from benchmarking.inpaint_detector_bakeoff.fixed_ctd_onnx import (  # noqa: E402
    FixedSizeCTDONNXReference,
)
from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    PageMasks,
    load_page_masks,
    load_stage1_manifest,
    score_page,
    summarize,
)
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-manga109-yolo26-ctd-fusion-v2"
CATEGORY = "40-inpaint-mask-render"
CANDIDATE_ID = "manga109-yolo26-text-ownership-plus-ctd-raw"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score CTD raw pixels gated by Manga109 YOLO26 text ownership.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ownership-root", type=Path, required=True)
    parser.add_argument("--ownership-model-sha256", required=True)
    parser.add_argument("--ctd-model", type=Path, required=True)
    parser.add_argument("--ctd-provider", default="CUDAExecutionProvider")
    parser.add_argument("--detect-size", type=int, default=1280)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest.resolve()
    ownership_root = args.ownership_root.resolve()
    ctd_model = args.ctd_model.resolve()
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        providers = [args.ctd_provider]
        if args.ctd_provider != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")
        ctd = FixedSizeCTDONNXReference(
            ctd_model,
            providers=providers,
            detect_size=int(args.detect_size),
        )
        edit_root = output_root / "positive_edit_masks"
        claim_root = output_root / "positive_claim_masks"
        raw_root = output_root / "ctd_raw_masks"
        for directory in (edit_root, claim_root, raw_root):
            directory.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, object]] = []
        for page in load_stage1_manifest(manifest_path):
            image = cv2.imread(page.source_image, cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise FileNotFoundError(page.source_image)
            shape = image.shape[:2]
            evaluation = load_page_masks(page, shape)
            ownership_path = ownership_root / f"{page.page_id}_ownership.png"
            ownership_raw = cv2.imread(str(ownership_path), cv2.IMREAD_GRAYSCALE)
            if ownership_raw is None or ownership_raw.size == 0:
                raise FileNotFoundError(ownership_path)
            ownership = binary_mask(ownership_raw, shape)
            ctd_result = ctd.infer(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            claim = np.where(
                (ctd_result.raw_mask > 0) & (ownership > 0),
                255,
                0,
            ).astype(np.uint8)
            scoring_masks = PageMasks(
                evaluation.target,
                evaluation.protected,
                evaluation.ambiguous,
                ownership,
                np.full(shape, 255, dtype=np.uint8),
                np.zeros(shape, dtype=np.uint8),
            )
            candidate = CandidateMaskResult(
                CANDIDATE_ID,
                claim,
                claim,
                claim,
                runtime={
                    "ctd": dict(ctd_result.runtime),
                    "ownership_provider": "manga109-yolo26-ultralytics",
                    "ownership_model_sha256": args.ownership_model_sha256,
                },
            )
            row, edit = score_page(page, candidate, scoring_masks, variant="raw")
            rows.append(row)
            for path, mask in (
                (edit_root / f"{page.page_id}_positive_edit.png", edit),
                (claim_root / f"{page.page_id}_positive_claim.png", claim),
                (raw_root / f"{page.page_id}_ctd_raw.png", ctd_result.raw_mask),
            ):
                if not cv2.imwrite(str(path), mask):
                    raise OSError(f"failed to write mask: {path}")

        payload = {
            "schema_version": "inpaint-manga109-yolo26-ctd-stage1-v1",
            "candidate": CANDIDATE_ID,
            "manifest_sha256": _sha256(manifest_path),
            "models": {
                "ownership": {
                    "sha256": args.ownership_model_sha256,
                    "runtime": "ultralytics-python-reference",
                },
                "ctd": {
                    "sha256": _sha256(ctd_model),
                    "provider": args.ctd_provider,
                    "providers": list(ctd.providers),
                },
            },
            "summary": summarize(rows),
            "pages": rows,
        }
        result_path = output_root / "stage1-results.json"
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if managed is not None:
            managed.complete(
                metadata={
                    "candidate": CANDIDATE_ID,
                    "manifest_sha256": payload["manifest_sha256"],
                    "summary": payload["summary"],
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
            managed.fail(error, metadata={"candidate": CANDIDATE_ID})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
