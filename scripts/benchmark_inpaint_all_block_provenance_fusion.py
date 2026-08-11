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
from benchmarking.inpaint_detector_bakeoff.contracts import (  # noqa: E402
    CandidateMaskResult,
    binary_mask,
)
from benchmarking.inpaint_detector_bakeoff.fixed_ctd_onnx import (  # noqa: E402
    FixedSizeCTDONNXReference,
)
from benchmarking.inpaint_detector_bakeoff.provenance_fusion import (  # noqa: E402
    build_provenance_fusion,
    reconcile_source_edit,
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
from scripts.benchmark_inpaint_detector_bakeoff import (  # noqa: E402
    _existing_edit_paths,
)


FAMILY = "inpaint-all-block-provenance-fusion-v2"
CATEGORY = "40-inpaint-mask-render"
CANDIDATE_ID = "rtdetr-all-block-provenance-plus-ctd-raw"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None or value.size == 0:
        raise FileNotFoundError(path)
    return binary_mask(value, shape)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score C11 RT-DETR provenance plus CTD raw over every existing block "
            "without creating candidate images."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--content-component-root", type=Path, required=True)
    parser.add_argument("--ctd-model", type=Path, required=True)
    parser.add_argument("--rtdetr-model", type=Path, required=True)
    parser.add_argument("--ctd-provider", default="CUDAExecutionProvider")
    parser.add_argument("--rtdetr-provider", default="CPUExecutionProvider")
    parser.add_argument("--detect-size", type=int, default=1280)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest.resolve()
    prior_root = args.prior_root.resolve()
    content_root = args.content_component_root.resolve()
    ctd_model = args.ctd_model.resolve()
    rtdetr_model = args.rtdetr_model.resolve()
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        ctd_providers = [args.ctd_provider]
        if args.ctd_provider != "CPUExecutionProvider":
            ctd_providers.append("CPUExecutionProvider")
        ctd = FixedSizeCTDONNXReference(
            ctd_model,
            providers=ctd_providers,
            detect_size=int(args.detect_size),
        )
        rtdetr_providers = [args.rtdetr_provider]
        if args.rtdetr_provider != "CPUExecutionProvider":
            rtdetr_providers.append("CPUExecutionProvider")
        rtdetr = BallonsCTBDReference(
            str(rtdetr_model),
            rtdetr_providers,
            CTBDSettings(confidence_threshold=0.3, inpaint_mask_dilate=4),
        )

        claim_root = output_root / "positive_claim_masks"
        edit_root = output_root / "positive_edit_masks"
        verified_root = output_root / "verified_source_edit_masks"
        replacement_root = output_root / "replacement_edit_masks"
        ownership_root = output_root / "provenance_ownership_masks"
        for directory in (
            claim_root,
            edit_root,
            verified_root,
            replacement_root,
            ownership_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, object]] = []
        existing_paths = _existing_edit_paths(manifest_path)
        for page in load_stage1_manifest(manifest_path):
            image = cv2.imread(page.source_image, cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise FileNotFoundError(page.source_image)
            shape = image.shape[:2]
            evaluation = load_page_masks(
                page,
                shape,
                existing_edit_path=existing_paths.get(page.page_id),
            )
            prior = _read_mask(prior_root / f"{page.page_id}_ownership.png", shape)
            content = _read_mask(
                content_root / f"{page.page_id}_ownership.png",
                shape,
            )
            ctd_result = ctd.infer(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            rtdetr_result = rtdetr.infer(image)
            fusion = build_provenance_fusion(
                ctd_result.raw_mask,
                required_skip_prior=prior,
                required_skip_seed=prior,
                content_component_ownership=content,
                raw_detector_boxes=rtdetr_result.boxes,
                existing_edit=evaluation.existing_edit,
                structure_protect=evaluation.protected,
                ambiguous_protect=evaluation.ambiguous,
                subtract_existing_edit=False,
            )
            raw_text_ownership = np.zeros(shape, dtype=np.uint8)
            for box in fusion.selected_raw_text_boxes:
                x1, y1, x2, y2 = box.xyxy
                raw_text_ownership[y1:y2, x1:x2] = prior[y1:y2, x1:x2]
            reconciliation = reconcile_source_edit(
                fusion.positive_edit,
                evaluation.existing_edit,
                allow_positive_addition=not page.no_edit,
                existing_ownership_evidence=raw_text_ownership,
            )
            scoring_masks = PageMasks(
                evaluation.target,
                evaluation.protected,
                evaluation.ambiguous,
                fusion.ownership,
                np.full(shape, 255, dtype=np.uint8),
                np.zeros(shape, dtype=np.uint8),
            )
            candidate = CandidateMaskResult(
                CANDIDATE_ID,
                reconciliation.replacement_edit,
                reconciliation.replacement_edit,
                reconciliation.replacement_edit,
                runtime={
                    "ctd": dict(ctd_result.runtime),
                    "rtdetr_providers": list(rtdetr.providers),
                    "selected_raw_text_box_count": len(
                        fusion.selected_raw_text_boxes
                    ),
                },
            )
            row, edit = score_page(page, candidate, scoring_masks, variant="raw")
            row.update(
                {
                    "raw_claim_protected_overlap": int(
                        np.count_nonzero(
                            (fusion.positive_claim > 0)
                            & (evaluation.protected > 0)
                        )
                    ),
                    "raw_claim_ambiguous_overlap": int(
                        np.count_nonzero(
                            (fusion.positive_claim > 0)
                            & (evaluation.ambiguous > 0)
                        )
                    ),
                    "raw_claim_outside_ownership_pixel_count": int(
                        np.count_nonzero(
                            (fusion.positive_claim > 0)
                            & (fusion.ownership == 0)
                        )
                    ),
                }
            )
            effective_verified = np.where(
                (edit > 0) & (evaluation.existing_edit > 0),
                255,
                0,
            ).astype(np.uint8)
            effective_positive = np.where(
                (edit > 0) & (evaluation.existing_edit == 0),
                255,
                0,
            ).astype(np.uint8)
            row.update(
                {
                    "raw_detector_claim_pixel_count": int(
                        np.count_nonzero(fusion.positive_claim)
                    ),
                    "existing_source_edit_pixel_count": int(
                        np.count_nonzero(evaluation.existing_edit)
                    ),
                    "verified_source_edit_pixel_count": int(
                        np.count_nonzero(effective_verified)
                    ),
                    "positive_addition_pixel_count": int(
                        np.count_nonzero(effective_positive)
                    ),
                    "dropped_existing_edit_pixel_count": int(
                        np.count_nonzero(
                            (evaluation.existing_edit > 0)
                            & (edit == 0)
                        )
                    ),
                    "replacement_edit_pixel_count": int(
                        np.count_nonzero(edit)
                    ),
                }
            )
            row["false_edit_pixel_count"] = (
                int(np.count_nonzero(effective_positive))
                if page.no_edit
                else 0
            )
            row["selected_raw_text_boxes"] = [
                list(box.xyxy) for box in fusion.selected_raw_text_boxes
            ]
            rows.append(row)
            for path, mask in (
                (claim_root / f"{page.page_id}_positive_claim.png", fusion.positive_claim),
                (
                    edit_root / f"{page.page_id}_positive_edit.png",
                    effective_positive,
                ),
                (
                    verified_root / f"{page.page_id}_verified_source_edit.png",
                    effective_verified,
                ),
                (
                    replacement_root / f"{page.page_id}_replacement_edit.png",
                    edit,
                ),
                (
                    ownership_root / f"{page.page_id}_provenance_ownership.png",
                    fusion.ownership,
                ),
            ):
                if not cv2.imwrite(str(path), mask):
                    raise OSError(f"failed to write mask: {path}")

        payload = {
            "schema_version": "inpaint-all-block-provenance-stage1-v1",
            "candidate": CANDIDATE_ID,
            "manifest_sha256": _sha256(manifest_path),
            "models": {
                "ctd": {
                    "sha256": _sha256(ctd_model),
                    "providers": list(ctd.providers),
                },
                "rtdetr": {
                    "sha256": _sha256(rtdetr_model),
                    "providers": list(rtdetr.providers),
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
