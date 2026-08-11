#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.contracts import binary_mask  # noqa: E402
from benchmarking.inpaint_detector_bakeoff.silhouette import (  # noqa: E402
    ballons_native_clean_background,
    extract_ballons_native_interior,
    extract_pr2_validated_interior,
)
from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    load_stage1_manifest,
)
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-silhouette-router-v3"
CATEGORY = "40-inpaint-mask-render"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask):
        raise OSError(f"failed to write mask: {path}")


def _format_path(template: str, page_id: str) -> Path:
    return Path(str(template).format(page_id=page_id))


def _read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.size == 0:
        raise FileNotFoundError(path)
    return binary_mask(mask, shape)


def _overlap(mask: np.ndarray, xyxy: object) -> int:
    if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
        return 0
    height, width = mask.shape
    x1, y1, x2, y2 = (int(value) for value in xyxy)
    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return 0
    return int(np.count_nonzero(mask[y1:y2, x1:x2]))


def _place_crop(
    destination: np.ndarray,
    local: np.ndarray | None,
    xyxy: list[int],
) -> None:
    if local is None:
        return
    x1, y1, x2, y2 = xyxy
    if local.shape != (y2 - y1, x2 - x1):
        raise ValueError("silhouette crop shape mismatch")
    destination[y1:y2, x1:x2][local > 0] = 255


def _selected_blocks(metadata: dict[str, Any], seed: np.ndarray) -> list[dict[str, Any]]:
    blocks = metadata.get("blocks", [])
    if not isinstance(blocks, list):
        raise ValueError("block metadata must contain blocks")
    return [
        block
        for block in blocks
        if isinstance(block, dict)
        and block.get("bubble_xyxy") is not None
        and _overlap(seed, block.get("bubble_xyxy")) > 0
    ]


def _ctbd_silhouette(metadata: dict[str, Any], seed: np.ndarray) -> np.ndarray:
    output = np.zeros_like(seed)
    boxes = metadata.get("boxes", [])
    if not isinstance(boxes, list):
        return output
    for record in boxes:
        if not isinstance(record, dict) or record.get("label") != "bubble":
            continue
        xyxy = record.get("xyxy")
        if _overlap(seed, xyxy) <= 0:
            continue
        x1, y1, x2, y2 = (int(value) for value in xyxy)
        x1, x2 = max(0, x1), min(output.shape[1], x2)
        y1, y2 = max(0, y1), min(output.shape[0], y2)
        if x2 > x1 and y2 > y1:
            output[y1:y2, x1:x2] = 255
    return output


def export_candidates(
    manifest_path: Path,
    *,
    block_metadata_template: str,
    seed_template: str,
    ctbd_metadata_template: str,
    output_root: Path,
) -> dict[str, Any]:
    pages = load_stage1_manifest(manifest_path)
    evidence: dict[str, dict[str, Any]] = {}
    index_pages: list[dict[str, Any]] = []
    for page in pages:
        image = cv2.imread(page.source_image, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise FileNotFoundError(page.source_image)
        shape = image.shape[:2]
        seed = _read_mask(_format_path(seed_template, page.page_id), shape)
        block_metadata = _read_json(
            _format_path(block_metadata_template, page.page_id)
        )
        ctbd_metadata = _read_json(
            _format_path(ctbd_metadata_template, page.page_id)
        )
        native_page = np.zeros(shape, np.uint8)
        pr2_page = np.zeros(shape, np.uint8)
        ballons_clean_page = np.zeros(shape, np.uint8)
        pr2_clean_page = np.zeros(shape, np.uint8)
        unsafe_page = np.zeros(shape, np.uint8)
        selected = _selected_blocks(block_metadata, seed)
        ballons_clean = False
        pr2_clean = False
        reasons: list[str] = []
        for block in selected:
            x1, y1, x2, y2 = (int(value) for value in block["bubble_xyxy"])
            x1, x2 = max(0, x1), min(shape[1], x2)
            y1, y2 = max(0, y1), min(shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]
            local_seed = seed[y1:y2, x1:x2]
            native = extract_ballons_native_interior(crop, local_seed)
            validated = extract_pr2_validated_interior(crop, local_seed)
            _place_crop(native_page, native, [x1, y1, x2, y2])
            _place_crop(pr2_page, validated, [x1, y1, x2, y2])
            local_ballons_clean = ballons_native_clean_background(crop, local_seed)
            reason = str(block.get("erase_skipped_reason") or "")
            mode = str(block.get("erase_mode") or "")
            reasons.append(reason)
            local_pr2_clean = mode in {
                "bubble_flat_fill",
                "bubble_gradient_fill",
                "bubble_telea",
            } and not any(
                token in reason
                for token in ("texture", "line_art", "ambiguous", "cap_unavailable")
            )
            if local_ballons_clean and native is not None:
                _place_crop(ballons_clean_page, native, [x1, y1, x2, y2])
                ballons_clean = True
            if local_pr2_clean and validated is not None:
                _place_crop(pr2_clean_page, validated, [x1, y1, x2, y2])
                pr2_clean = True
            if any(
                token in reason
                for token in ("texture", "line_art", "ambiguous", "cap_unavailable")
            ):
                unsafe_page[y1:y2, x1:x2] = 255
        ctbd_page = _ctbd_silhouette(ctbd_metadata, seed)
        segmentation_page = cv2.bitwise_or(
            cv2.bitwise_or(native_page, pr2_page), ctbd_page
        )
        paths = {
            "ballons_native": output_root / "ballons_native" / f"{page.page_id}.png",
            "pr2_validated": output_root / "pr2_validated" / f"{page.page_id}.png",
            "ctbd_bubble": output_root / "ctbd_bubble" / f"{page.page_id}.png",
        }
        _write_mask(paths["ballons_native"], native_page)
        _write_mask(paths["pr2_validated"], pr2_page)
        _write_mask(paths["ctbd_bubble"], ctbd_page)
        route_mask_paths = {
            "ballons_clean_mask": output_root / "route_masks" / "ballons_clean" / f"{page.page_id}.png",
            "pr2_clean_mask": output_root / "route_masks" / "pr2_clean" / f"{page.page_id}.png",
            "segmentation_valid_mask": output_root / "route_masks" / "segmentation_valid" / f"{page.page_id}.png",
            "unsafe_signal_mask": output_root / "route_masks" / "unsafe" / f"{page.page_id}.png",
        }
        _write_mask(route_mask_paths["ballons_clean_mask"], ballons_clean_page)
        _write_mask(route_mask_paths["pr2_clean_mask"], pr2_clean_page)
        _write_mask(route_mask_paths["segmentation_valid_mask"], segmentation_page)
        _write_mask(route_mask_paths["unsafe_signal_mask"], unsafe_page)
        joined_reason = " ".join(reasons)
        evidence[page.page_id] = {
            "ballons_clean": bool(ballons_clean),
            "pr2_clean": bool(pr2_clean),
            "segmentation_valid": bool(
                np.any(native_page) or np.any(pr2_page) or np.any(ctbd_page)
            ),
            "texture": "texture" in joined_reason,
            "microtexture": "microtexture" in joined_reason,
            "line_art": "line_art" in joined_reason,
            "ambiguous": "ambiguous" in joined_reason,
            "source_only_reason_codes": reasons,
            **{key: str(path.resolve()) for key, path in route_mask_paths.items()},
        }
        index_pages.append(
            {
                "page_id": page.page_id,
                "seed_pixel_count": int(np.count_nonzero(seed)),
                "selected_block_count": len(selected),
                "ballons_native_pixel_count": int(np.count_nonzero(native_page)),
                "pr2_validated_pixel_count": int(np.count_nonzero(pr2_page)),
                "ctbd_bubble_pixel_count": int(np.count_nonzero(ctbd_page)),
            }
        )
    payload = {
        "schema_version": "inpaint-silhouette-router-evidence-v3",
        "pages": evidence,
        "index": index_pages,
    }
    _write_json(output_root / "router-evidence.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export source-only Ballons/PR2/CTBD silhouettes and router evidence."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--block-metadata-template", required=True)
    parser.add_argument("--seed-template", required=True)
    parser.add_argument("--ctbd-metadata-template", required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        payload = export_candidates(
            args.manifest.resolve(),
            block_metadata_template=args.block_metadata_template,
            seed_template=args.seed_template,
            ctbd_metadata_template=args.ctbd_metadata_template,
            output_root=output_root,
        )
        if managed is not None:
            managed.complete(metadata={"page_count": len(payload["index"])})
            mismatches = managed.verify()
            if mismatches:
                raise RuntimeError("managed artifact verification failed: " + "; ".join(mismatches))
            print(managed.run_root)
        else:
            print(output_root)
        return 0
    except BaseException as error:
        if managed is not None:
            managed.fail(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
