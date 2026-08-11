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
    CTBDSettings,
)
from benchmarking.inpaint_detector_bakeoff.ballons_ctd import (  # noqa: E402
    BallonsCTDOriginalReference,
)
from benchmarking.inpaint_detector_bakeoff.contracts import (  # noqa: E402
    CandidateMaskResult,
)
from benchmarking.inpaint_detector_bakeoff.fixed_ctd_onnx import (  # noqa: E402
    FixedSizeCTDONNXReference,
)
from benchmarking.inpaint_detector_bakeoff.provenance_fusion import (  # noqa: E402
    build_provenance_fusion,
)
from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    PageMasks,
    load_page_masks,
    load_stage1_manifest,
    score_page,
    summarize,
)
from scripts.benchmark_inpaint_detector_bakeoff import _existing_edit_paths  # noqa: E402
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-detector-provenance-fusion-v2"
CATEGORY = "40-inpaint-mask-render"
CANDIDATE_ID = "rtdetr-raw-text-provenance-plus-ctd-raw"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _page_map(path: Path):
    return {page.page_id: page for page in load_stage1_manifest(path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score RT-DETR raw-text provenance plus CTD raw pixel claim without "
            "creating candidate images."
        )
    )
    parser.add_argument("--routing-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--content-component-root", type=Path, required=True)
    parser.add_argument("--ballons-root", type=Path)
    parser.add_argument("--ctd-model", type=Path, required=True)
    parser.add_argument("--rtdetr-model", type=Path, required=True)
    parser.add_argument("--ctd-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--ctd-runtime",
        choices=("ballons-python", "onnxruntime"),
        default="ballons-python",
    )
    parser.add_argument("--ctd-provider", default="CUDAExecutionProvider")
    parser.add_argument("--rtdetr-provider", default="CPUExecutionProvider")
    parser.add_argument("--detect-size", type=int, default=1280)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    routing_manifest = args.routing_manifest.resolve()
    evaluation_manifest = args.evaluation_manifest.resolve()
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        routing_pages = load_stage1_manifest(routing_manifest)
        evaluation_pages = _page_map(evaluation_manifest)
        existing_paths = _existing_edit_paths(routing_manifest)
        if args.ctd_runtime == "ballons-python":
            if args.ballons_root is None:
                raise ValueError("--ballons-root is required for ballons-python")
            ctd = BallonsCTDOriginalReference(
                ballons_root=str(args.ballons_root.resolve()),
                model_path=str(args.ctd_model.resolve()),
                device=args.ctd_device,
                detect_size=int(args.detect_size),
                dilate_size=3,
            )
        else:
            ctd_providers = [args.ctd_provider]
            if args.ctd_provider != "CPUExecutionProvider":
                ctd_providers.append("CPUExecutionProvider")
            ctd = FixedSizeCTDONNXReference(
                args.ctd_model,
                providers=ctd_providers,
                detect_size=int(args.detect_size),
            )
        providers = [args.rtdetr_provider]
        if args.rtdetr_provider != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")
        rtdetr = BallonsCTBDReference(
            str(args.rtdetr_model.resolve()),
            providers,
            CTBDSettings(confidence_threshold=0.3, inpaint_mask_dilate=4),
        )

        edit_root = output_root / "positive_edit_masks"
        claim_root = output_root / "positive_claim_masks"
        raw_claim_root = output_root / "raw_pixel_claim_masks"
        ownership_root = output_root / "provenance_ownership_masks"
        for directory in (edit_root, claim_root, raw_claim_root, ownership_root):
            directory.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, object]] = []
        for routing_page in routing_pages:
            evaluation_page = evaluation_pages.get(routing_page.page_id)
            if evaluation_page is None:
                raise ValueError(
                    f"evaluation manifest is missing page {routing_page.page_id}"
                )
            image = cv2.imread(routing_page.source_image, cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise FileNotFoundError(routing_page.source_image)
            shape = image.shape[:2]
            routing_masks = load_page_masks(
                routing_page,
                shape,
                existing_edit_path=existing_paths.get(routing_page.page_id),
            )
            evaluation_masks = load_page_masks(evaluation_page, shape)
            content_path = (
                args.content_component_root
                / f"{routing_page.page_id}_ownership.png"
            )
            content_components = cv2.imread(str(content_path), cv2.IMREAD_GRAYSCALE)
            if content_components is None or content_components.size == 0:
                raise FileNotFoundError(content_path)

            ctd_result = ctd.infer(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            rtdetr_result = rtdetr.infer(image)
            fusion = build_provenance_fusion(
                ctd_result.raw_mask,
                required_skip_prior=routing_masks.ownership,
                required_skip_seed=routing_masks.claim_seed,
                content_component_ownership=content_components,
                raw_detector_boxes=rtdetr_result.boxes,
                existing_edit=routing_masks.existing_edit,
                structure_protect=evaluation_masks.protected,
                ambiguous_protect=evaluation_masks.ambiguous,
            )
            scoring_masks = PageMasks(
                evaluation_masks.target,
                evaluation_masks.protected,
                evaluation_masks.ambiguous,
                fusion.ownership,
                np.full(shape, 255, dtype=np.uint8),
                routing_masks.existing_edit,
            )
            candidate = CandidateMaskResult(
                CANDIDATE_ID,
                fusion.positive_claim,
                fusion.positive_claim,
                fusion.positive_claim,
                runtime={
                    "ctd": dict(ctd_result.runtime),
                    "rtdetr": dict(rtdetr_result.runtime),
                    "selected_raw_text_box_count": len(
                        fusion.selected_raw_text_boxes
                    ),
                    "rtdetr_providers": rtdetr.providers,
                },
            )
            row, edit = score_page(
                evaluation_page,
                candidate,
                scoring_masks,
                variant="raw",
            )
            row["selected_raw_text_boxes"] = [
                list(box.xyxy) for box in fusion.selected_raw_text_boxes
            ]
            rows.append(row)
            outputs = (
                (edit_root / f"{routing_page.page_id}_positive_edit.png", edit),
                (
                    claim_root / f"{routing_page.page_id}_positive_claim.png",
                    fusion.positive_claim,
                ),
                (
                    raw_claim_root / f"{routing_page.page_id}_raw_pixel_claim.png",
                    ctd_result.raw_mask,
                ),
                (
                    ownership_root
                    / f"{routing_page.page_id}_provenance_ownership.png",
                    fusion.ownership,
                ),
            )
            for path, mask in outputs:
                if not cv2.imwrite(str(path), mask):
                    raise OSError(f"failed to write mask: {path}")

        payload = {
            "schema_version": "inpaint-detector-provenance-fusion-stage1-v1",
            "candidate": CANDIDATE_ID,
            "routing_manifest_sha256": _sha256(routing_manifest),
            "evaluation_manifest_sha256": _sha256(evaluation_manifest),
            "models": {
                "ctd": {
                    "name": args.ctd_model.name,
                    "sha256": _sha256(args.ctd_model),
                    "device": args.ctd_device,
                    "runtime": args.ctd_runtime,
                    "provider": args.ctd_provider
                    if args.ctd_runtime == "onnxruntime"
                    else None,
                    "providers": list(getattr(ctd, "providers", ())),
                },
                "rtdetr": {
                    "name": args.rtdetr_model.name,
                    "sha256": _sha256(args.rtdetr_model),
                    "providers": rtdetr.providers,
                },
            },
            "summary": summarize(rows),
            "pages": rows,
        }
        _write_json(output_root / "stage1-results.json", payload)
        if managed is not None:
            managed.complete(
                metadata={
                    "candidate": CANDIDATE_ID,
                    "routing_manifest_sha256": payload[
                        "routing_manifest_sha256"
                    ],
                    "evaluation_manifest_sha256": payload[
                        "evaluation_manifest_sha256"
                    ],
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
