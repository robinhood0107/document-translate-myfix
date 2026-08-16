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

import imkit as imk  # noqa: E402

from benchmarking.inpaint_detector_bakeoff.contracts import (  # noqa: E402
    CandidateMaskResult,
    binary_mask,
)
from benchmarking.inpaint_detector_bakeoff.provenance_fusion import (  # noqa: E402
    StructureGuardedReconciliation,
    add_guarded_narrow_claim,
    build_detector_verified_structure_protect,
    build_post_expansion_protection_reentry,
    build_source_owned_expansion_cap,
    build_source_protected_detector_candidate,
    detector_recovery_route,
    replace_guarded_regions_with_narrow_claim,
    replace_guarded_expansion_halo_with_narrow_claim,
)
from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    PageMasks,
    load_page_masks,
    load_stage1_manifest,
    score_page,
    summarize,
)
from modules.detection.processor import TextBlockDetector  # noqa: E402
from modules.rendering.render import get_best_render_area  # noqa: E402
from modules.utils.image_utils import generate_mask  # noqa: E402
from scripts.benchmark_inpaint_detector_bakeoff import (  # noqa: E402
    _existing_edit_paths,
)
from scripts.export_inpaint_debug import _SettingsStub  # noqa: E402
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-source-protection-reapply-v3"
CATEGORY = "40-inpaint-mask-render"
SOURCE_EVIDENCE_KINDS = (
    "raw",
    "pre_expand",
    "post_expand",
    "final",
    "protect",
    "corner",
)
CANDIDATE_IDS = {
    "source_cap": "c15-post-expansion-protect-plus-c11-narrow",
    "source_expansion_cap": "c18-product-expansion-matched-protect",
    "accepted_expansion_cap": "c19-accepted-seed-final-protect",
    "expansion_reentry_only": "c21-expansion-reentry-protect",
    "guarded_narrow": "c14-structure-risk-narrow-claim",
    "guarded_halo_narrow": "c22-structure-risk-halo-narrow",
    "guarded_narrow_add": "c23-structure-risk-narrow-addition",
    "detector_verified": "c17-detector-verified-final-protect",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.size == 0:
        raise FileNotFoundError(path)
    return binary_mask(mask, shape)


def _optional_mask(value, shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=np.uint8)
    return binary_mask(value, shape)


def _write_source_evidence_contract(
    root: Path,
    *,
    manifest_sha256: str,
    page_ids: list[str],
) -> None:
    mask_sha256: dict[str, str] = {}
    for page_id in page_ids:
        for mask_kind in SOURCE_EVIDENCE_KINDS:
            relative = f"{page_id}_{mask_kind}.png"
            path = root / relative
            if not path.is_file():
                raise ValueError(f"source evidence cache mask is missing: {relative}")
            mask_sha256[relative] = _sha256(path)
    payload = {
        "schema_version": "inpaint-source-evidence-cache-v1",
        "manifest_sha256": manifest_sha256,
        "page_ids": page_ids,
        "mask_kinds": list(SOURCE_EVIDENCE_KINDS),
        "mask_sha256": mask_sha256,
    }
    (root / "source-evidence-contract.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _validate_source_evidence_contract(
    root: Path,
    *,
    manifest_sha256: str,
    page_ids: list[str],
) -> None:
    contract_path = root / "source-evidence-contract.json"
    if not contract_path.is_file():
        raise ValueError("source evidence cache contract is missing")
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "inpaint-source-evidence-cache-v1":
        raise ValueError("source evidence cache schema mismatch")
    if payload.get("manifest_sha256") != manifest_sha256:
        raise ValueError("source evidence cache manifest SHA mismatch")
    if payload.get("page_ids") != page_ids:
        raise ValueError("source evidence cache page order mismatch")
    expected_masks = {
        f"{page_id}_{mask_kind}.png"
        for page_id in page_ids
        for mask_kind in SOURCE_EVIDENCE_KINDS
    }
    recorded = payload.get("mask_sha256")
    if not isinstance(recorded, dict) or set(recorded) != expected_masks:
        raise ValueError("source evidence cache mask inventory mismatch")
    for relative in sorted(expected_masks):
        path = root / relative
        if not path.is_file() or recorded.get(relative) != _sha256(path):
            raise ValueError(f"source evidence cache mask SHA mismatch: {relative}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reapply source-only structure protection after CTD expansion, "
            "then add only C11 narrow detector pixels."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--claim-root", type=Path, required=True)
    parser.add_argument(
        "--structure-claim-root",
        type=Path,
        help=(
            "Optional full detector-claim directory used only to disprove a "
            "structure proposal; it never creates positive edit pixels."
        ),
    )
    parser.add_argument(
        "--debug-metadata-root",
        type=Path,
        help="Product debug metadata used to identify structure-risk blocks.",
    )
    parser.add_argument("--ownership-root", type=Path, required=True)
    parser.add_argument(
        "--source-evidence-root",
        type=Path,
        help=(
            "Optional cache created by an earlier run. When provided, raw, "
            "pre-expand, post-expand, protect, and corner masks are loaded "
            "instead of rerunning the product detector."
        ),
    )
    parser.add_argument(
        "--base-mask-root",
        type=Path,
        help=(
            "Optional final-mask directory to audit as the immutable product "
            "candidate instead of regenerating the current branch mask."
        ),
    )
    parser.add_argument("--source-cap-size", type=int, default=8)
    parser.add_argument(
        "--structure-policy",
        choices=tuple(CANDIDATE_IDS),
        default="source_cap",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = args.manifest.resolve()
    claim_root = args.claim_root.resolve()
    structure_claim_root = (
        args.structure_claim_root.resolve() if args.structure_claim_root else None
    )
    ownership_root = args.ownership_root.resolve()
    source_evidence_root = (
        args.source_evidence_root.resolve() if args.source_evidence_root else None
    )
    base_mask_root = args.base_mask_root.resolve() if args.base_mask_root else None
    debug_metadata_root = (
        args.debug_metadata_root.resolve() if args.debug_metadata_root else None
    )
    candidate_id = CANDIDATE_IDS[args.structure_policy]
    output, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output.mkdir(parents=True, exist_ok=True)
    try:
        pages = list(load_stage1_manifest(manifest))
        manifest_sha256 = _sha256(manifest)
        page_ids = [page.page_id for page in pages]
        if source_evidence_root is not None:
            _validate_source_evidence_contract(
                source_evidence_root,
                manifest_sha256=manifest_sha256,
                page_ids=page_ids,
            )
        settings = _SettingsStub(
            inpainter="lama_large_512px",
            use_gpu=args.device == "cuda",
        )
        detector = TextBlockDetector(settings) if source_evidence_root is None else None
        existing_paths = _existing_edit_paths(manifest)
        rows: list[dict[str, object]] = []
        mask_root = output / "replacement_edit_masks"
        protect_root = output / "derived_structure_masks"
        source_cache_root = output / "source_evidence_masks"
        mask_root.mkdir(parents=True, exist_ok=True)
        protect_root.mkdir(parents=True, exist_ok=True)
        source_cache_root.mkdir(parents=True, exist_ok=True)

        for page in pages:
            # OpenCV's Windows filename loader cannot reliably open non-ASCII
            # source paths.  imkit decodes the bytes after a Unicode-safe file
            # read, which is the same path contract used by the product
            # exporter.
            image = imk.read_image(page.source_image)
            if image is None or image.size == 0:
                raise FileNotFoundError(page.source_image)
            shape = image.shape[:2]
            evaluation = load_page_masks(
                page,
                shape,
                existing_edit_path=existing_paths.get(page.page_id),
            )
            if source_evidence_root is None:
                if detector is None:
                    raise RuntimeError("detector cache state is inconsistent")
                blocks = detector.detect(image) or []
                get_best_render_area(blocks, image)
                details = generate_mask(
                    image,
                    blocks,
                    settings=settings.get_mask_refiner_settings(),
                    return_details=True,
                )
                raw_source = _optional_mask(details.get("raw_mask"), shape)
                accepted_source = _optional_mask(
                    details.get("final_mask_pre_expand"), shape
                )
                post_expand_source = _optional_mask(
                    details.get("final_mask_post_expand"), shape
                )
                generated_final = _optional_mask(details.get("final_mask"), shape)
                protect = _optional_mask(details.get("protect_mask"), shape)
                corner = _optional_mask(details.get("protected_corner_mask"), shape)
                cache_masks = {
                    "raw": raw_source,
                    "pre_expand": accepted_source,
                    "post_expand": post_expand_source,
                    "final": generated_final,
                    "protect": protect,
                    "corner": corner,
                }
                for cache_name, cache_mask in cache_masks.items():
                    cv2.imwrite(
                        str(
                            source_cache_root
                            / f"{page.page_id}_{cache_name}.png"
                        ),
                        cache_mask,
                    )
            else:
                cache_masks = {
                    cache_name: _read_mask(
                        source_evidence_root
                        / f"{page.page_id}_{cache_name}.png",
                        shape,
                    )
                    for cache_name in SOURCE_EVIDENCE_KINDS
                }
                raw_source = cache_masks["raw"]
                accepted_source = cache_masks["pre_expand"]
                post_expand_source = cache_masks["post_expand"]
                generated_final = cache_masks["final"]
                protect = cache_masks["protect"]
                corner = cache_masks["corner"]
            expanded = (
                _read_mask(
                    base_mask_root / f"{page.page_id}_final_mask.png",
                    shape,
                )
                if base_mask_root is not None
                else generated_final
            )
            claim = _read_mask(
                claim_root / f"{page.page_id}_raw_pixel_claim.png",
                shape,
            )
            structure_claim = (
                _read_mask(
                    structure_claim_root
                    / f"{page.page_id}_positive_claim_raw.png",
                    shape,
                )
                if structure_claim_root is not None
                else claim
            )
            ownership = _read_mask(
                ownership_root / f"{page.page_id}_provenance_ownership.png",
                shape,
            )
            guarded_regions = np.zeros(shape, dtype=np.uint8)
            if args.structure_policy in {
                "guarded_narrow",
                "guarded_halo_narrow",
                "guarded_narrow_add",
            }:
                if debug_metadata_root is None:
                    raise ValueError("guarded_narrow requires --debug-metadata-root")
                metadata_path = (
                    debug_metadata_root / f"{page.page_id}_debug.json"
                )
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                for block in metadata.get("blocks", []):
                    if detector_recovery_route(
                        block.get("erase_skipped_reason", "")
                    ) != "narrow":
                        continue
                    roi = block.get("bubble_xyxy") or block.get("mask_actual_bbox")
                    if not isinstance(roi, list) or len(roi) != 4:
                        continue
                    x1, y1, x2, y2 = (int(value) for value in roi)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(shape[1], x2), min(shape[0], y2)
                    if x2 > x1 and y2 > y1:
                        guarded_regions[y1:y2, x1:x2] = 255
                derived = protect
            elif args.structure_policy == "expansion_reentry_only":
                derived = build_post_expansion_protection_reentry(
                    post_expand_source,
                    protect,
                    corner_protect=corner,
                )
            elif args.structure_policy in {
                "source_cap",
                "source_expansion_cap",
                "accepted_expansion_cap",
            }:
                source_cap = (
                    imk.dilate(
                        raw_source,
                        np.ones((5, 5), dtype=np.uint8),
                        iterations=1,
                    )
                    if args.structure_policy == "source_cap"
                    else build_source_owned_expansion_cap(
                        (
                            accepted_source
                            if args.structure_policy == "accepted_expansion_cap"
                            else raw_source
                        ),
                        final_dilate_size=args.source_cap_size,
                    )
                )
                derived = np.where(
                    ((protect > 0) & (source_cap <= 0)) | (corner > 0),
                    255,
                    0,
                ).astype(np.uint8)
            else:
                derived = build_detector_verified_structure_protect(
                    protect,
                    structure_claim,
                    claim_ownership=ownership,
                    corner_protect=corner,
                )
            if args.structure_policy in {
                "guarded_narrow",
                "guarded_halo_narrow",
                "guarded_narrow_add",
            }:
                positive_claim = np.where(
                    (claim > 0) & (ownership > 0) & (guarded_regions > 0),
                    255,
                    0,
                ).astype(np.uint8)
                replacement = (
                    add_guarded_narrow_claim(
                        expanded,
                        positive_claim,
                        guarded_regions,
                        structure_protect=derived,
                        corner_protect=corner,
                    )
                    if args.structure_policy == "guarded_narrow_add"
                    else
                    replace_guarded_regions_with_narrow_claim(
                        expanded,
                        positive_claim,
                        guarded_regions,
                        structure_protect=derived,
                        corner_protect=corner,
                    )
                    if args.structure_policy == "guarded_narrow"
                    else replace_guarded_expansion_halo_with_narrow_claim(
                        expanded,
                        accepted_source,
                        post_expand_source,
                        positive_claim,
                        guarded_regions,
                        structure_protect=derived,
                        corner_protect=corner,
                    )
                )
                verified = np.where(
                    (replacement > 0) & (expanded > 0), 255, 0
                ).astype(np.uint8)
                positive_edit = np.where(
                    (replacement > 0) & (expanded <= 0), 255, 0
                ).astype(np.uint8)
                reconciled = StructureGuardedReconciliation(
                    verified_source_edit=np.ascontiguousarray(verified),
                    positive_claim=np.ascontiguousarray(positive_claim),
                    positive_edit=np.ascontiguousarray(positive_edit),
                    replacement_edit=np.ascontiguousarray(replacement),
                )
            else:
                reconciled = build_source_protected_detector_candidate(
                    expanded,
                    claim,
                    claim_ownership=ownership,
                    derived_structure_protect=derived,
                    corner_protect=corner,
                )
            scoring = PageMasks(
                evaluation.target,
                evaluation.protected,
                evaluation.ambiguous,
                np.full(shape, 255, dtype=np.uint8),
                np.full(shape, 255, dtype=np.uint8),
                np.zeros(shape, dtype=np.uint8),
                evaluation.target_instances,
                evaluation.bubble_interior,
                evaluation.corner,
            )
            candidate = CandidateMaskResult(
                candidate_id,
                reconciled.replacement_edit,
                reconciled.replacement_edit,
                reconciled.replacement_edit,
                runtime={
                    "source_only": True,
                    "annotation_used_for_mask": False,
                    "device": args.device,
                    "structure_policy": args.structure_policy,
                    "source_cap_size": args.source_cap_size,
                    "guarded_region_pixel_count": int(
                        np.count_nonzero(guarded_regions)
                    ),
                    "base_mask_root": str(base_mask_root or "generated"),
                    "source_evidence_root": str(
                        source_evidence_root or source_cache_root
                    ),
                    "structure_claim_root": str(
                        structure_claim_root or claim_root
                    ),
                },
            )
            row, _ = score_page(page, candidate, scoring, variant="raw")
            source_only_edit = reconciled.replacement_edit
            row.update(
                {
                    "expected_edit": page.expected_edit,
                    "existing_edit_pixel_count": int(np.count_nonzero(expanded)),
                    "derived_structure_pixel_count": int(np.count_nonzero(derived)),
                    "removed_existing_pixel_count": int(
                        np.count_nonzero(
                            (expanded > 0) & (source_only_edit <= 0)
                        )
                    ),
                    "source_only_protected_overlap_pixel_count": int(
                        np.count_nonzero(
                            (source_only_edit > 0) & (evaluation.protected > 0)
                        )
                    ),
                    "source_only_ambiguous_overlap_pixel_count": int(
                        np.count_nonzero(
                            (source_only_edit > 0) & (evaluation.ambiguous > 0)
                        )
                    ),
                    "positive_claim_pixel_count": int(
                        np.count_nonzero(reconciled.positive_claim)
                    ),
                    "positive_edit_pixel_count": int(
                        np.count_nonzero(reconciled.positive_edit)
                    ),
                    "false_edit_pixel_count": (
                        int(np.count_nonzero(reconciled.positive_edit))
                        if page.no_edit
                        else 0
                    ),
                }
            )
            rows.append(row)
            cv2.imwrite(
                str(mask_root / f"{page.page_id}_replacement_edit.png"),
                source_only_edit,
            )
            cv2.imwrite(
                str(protect_root / f"{page.page_id}_derived_structure.png"),
                derived,
            )

        if source_evidence_root is None:
            _write_source_evidence_contract(
                source_cache_root,
                manifest_sha256=manifest_sha256,
                page_ids=page_ids,
            )
        result = {
            "schema_version": "inpaint-source-protection-reapply-v3",
            "candidate_id": candidate_id,
            "manifest_sha256": manifest_sha256,
            "mask_contract": {
                "source_cap": (
                    "(expanded_source_edit - protect outside dilated raw source "
                    "- corner) union owned c11 raw claim"
                ),
                "detector_verified": (
                    "(final_product_edit - protect outside owned c11 raw claim "
                    "- corner) union owned c11 raw claim"
                ),
                "source_expansion_cap": (
                    "(final_product_edit - protect outside product-matched "
                    "source expansion cap - corner) union owned c11 raw claim"
                ),
                "accepted_expansion_cap": (
                    "(final_product_edit - protect outside product-matched "
                    "accepted pre-expand seed - corner) union owned c11 raw claim"
                ),
                "expansion_reentry_only": (
                    "final_product_edit - (post_expand_source intersect "
                    "structure_or_corner_protect)"
                ),
                "guarded_narrow": (
                    "existing product edit outside structure-risk block union "
                    "owned c11 raw pixel claim inside risk block minus protect"
                ),
                "guarded_halo_narrow": (
                    "existing product edit minus final expansion halo inside "
                    "structure-risk blocks union owned narrow detector claim"
                ),
                "guarded_narrow_add": (
                    "existing product edit union owned narrow detector pixels "
                    "inside structure-risk blocks minus exact protection"
                ),
            }[args.structure_policy],
            "structure_policy": args.structure_policy,
            "source_cap_size": args.source_cap_size,
            "base_mask_root": str(base_mask_root or "generated"),
            "source_evidence_root": str(source_evidence_root or source_cache_root),
            "structure_claim_root": str(structure_claim_root or claim_root),
            "annotation_used_for_mask": False,
            "summary": summarize(rows),
            "pages": rows,
        }
        result_path = output / "stage1-results.json"
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if managed is not None:
            managed.complete(
                metadata={
                    "candidate_id": candidate_id,
                    "manifest_sha256": result["manifest_sha256"],
                }
            )
        return 0
    except Exception as exc:
        if managed is not None:
            managed.fail(exc, metadata={"manifest": manifest.name})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
