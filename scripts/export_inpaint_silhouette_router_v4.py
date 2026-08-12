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
)
from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    load_stage1_manifest,
)
from modules.source_parity_vendor.utils.imgproc_utils import enlarge_window  # noqa: E402
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-silhouette-router-v4"
CATEGORY = "40-inpaint-mask-render"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_image(path: str | Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise FileNotFoundError(path)
    return image


def _read_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.size == 0:
        raise FileNotFoundError(path)
    return binary_mask(mask, shape)


def _write_mask(path: Path, mask: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = cv2.imencode(".png", binary_mask(mask))[1]
    encoded.tofile(str(path))
    return str(path.resolve())


def _format(template: str, page_id: str) -> Path:
    return Path(str(template).format(page_id=page_id))


def _clipped_box(value: object, shape: tuple[int, int]) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    height, width = shape
    x1, y1, x2, y2 = (int(item) for item in value)
    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def _overlap(mask: np.ndarray, xyxy: object) -> int:
    clipped = _clipped_box(xyxy, mask.shape)
    if clipped is None:
        return 0
    x1, y1, x2, y2 = clipped
    return int(np.count_nonzero(mask[y1:y2, x1:x2]))


def _place(destination: np.ndarray, local: np.ndarray | None, xyxy: list[int]) -> None:
    if local is None:
        return
    x1, y1, x2, y2 = xyxy
    normalized = binary_mask(local, (y2 - y1, x2 - x1))
    destination[y1:y2, x1:x2][normalized > 0] = 255


def _region_instance_seed(
    page_record: dict[str, Any], region_id: str, shape: tuple[int, int]
) -> np.ndarray:
    seed = np.zeros(shape, np.uint8)
    for record in page_record.get("target_instances", []):
        if not isinstance(record, dict) or record.get("region_id") != region_id:
            continue
        if str(record.get("priority") or "required") == "optional":
            continue
        path = record.get("mask_path", record.get("mask"))
        if isinstance(path, str) and path:
            seed[_read_mask(path, shape) > 0] = 255
    return seed


def _ballons_silhouette(
    image: np.ndarray,
    ballons_seed: np.ndarray,
    metadata: dict[str, Any],
    source_regions: list[tuple[np.ndarray, str]],
) -> tuple[np.ndarray, np.ndarray, int]:
    shape = image.shape[:2]
    interior = np.zeros(shape, np.uint8)
    clean = np.zeros(shape, np.uint8)
    selected = 0
    for record in metadata.get("boxes", []):
        if not isinstance(record, dict) or record.get("label") != "text":
            continue
        xyxy = _clipped_box(record.get("xyxy"), shape)
        if xyxy is None or _overlap(ballons_seed, xyxy) <= 0:
            continue
        if not any(_overlap(ownership, xyxy) > 0 for ownership, _route in source_regions):
            continue
        expanded = enlarge_window(xyxy, shape[1], shape[0], ratio=1.7)
        expanded = _clipped_box(expanded, shape)
        if expanded is None:
            continue
        x1, y1, x2, y2 = expanded
        crop = image[y1:y2, x1:x2]
        seed = ballons_seed[y1:y2, x1:x2]
        if not np.any(seed):
            continue
        local = extract_ballons_native_interior(crop, seed)
        _place(interior, local, expanded)
        if local is not None and ballons_native_clean_background(crop, seed):
            _place(clean, local, expanded)
        selected += 1
    return interior, clean, selected


def _ctbd_silhouette(
    metadata: dict[str, Any], source_regions: list[tuple[np.ndarray, str]], shape: tuple[int, int]
) -> np.ndarray:
    output = np.zeros(shape, np.uint8)
    for record in metadata.get("boxes", []):
        if not isinstance(record, dict) or record.get("label") != "bubble":
            continue
        xyxy = _clipped_box(record.get("xyxy"), shape)
        if xyxy is None or not any(
            _overlap(ownership, xyxy) > 0 for ownership, _route in source_regions
        ):
            continue
        x1, y1, x2, y2 = xyxy
        output[y1:y2, x1:x2] = 255
    return output


def export_candidates(
    manifest_path: Path,
    *,
    ballons_mask_template: str,
    ballons_metadata_template: str,
    ctbd_metadata_template: str,
    output_root: Path,
) -> dict[str, Any]:
    payload = _read_json(manifest_path)
    page_records = {
        str(page.get("page_id") or ""): page
        for page in payload.get("pages", [])
        if isinstance(page, dict)
    }
    pages = load_stage1_manifest(manifest_path)
    evidence: dict[str, dict[str, Any]] = {}
    index: list[dict[str, Any]] = []
    for page in pages:
        source = _read_image(page.source_image)
        shape = source.shape[:2]
        raw_page = page_records[page.page_id]
        source_regions: list[tuple[np.ndarray, str]] = []
        pr2_interior = np.zeros(shape, np.uint8)
        pr2_clean = np.zeros(shape, np.uint8)
        unsafe = np.zeros(shape, np.uint8)
        source_seed = np.zeros(shape, np.uint8)
        for region in raw_page.get("regions", []):
            if not isinstance(region, dict):
                continue
            region_id = str(region.get("region_id") or "")
            ownership = _read_mask(str(region["ownership_mask"]), shape)
            route = str(region.get("bubble_route_class") or "ambiguous")
            source_regions.append((ownership, route))
            source_seed[_region_instance_seed(raw_page, region_id, shape) > 0] = 255
            local_interior = _read_mask(str(region["bubble_interior_mask"]), shape)
            pr2_interior[local_interior > 0] = 255
            if route in {"clean_flat", "clean_gradient"}:
                pr2_clean[local_interior > 0] = 255
            else:
                unsafe[ownership > 0] = 255

        ballons_seed = _read_mask(_format(ballons_mask_template, page.page_id), shape)
        ballons_metadata = _read_json(
            _format(ballons_metadata_template, page.page_id)
        )
        ctbd_metadata = _read_json(_format(ctbd_metadata_template, page.page_id))
        ballons_interior, ballons_clean, selected = _ballons_silhouette(
            source, ballons_seed, ballons_metadata, source_regions
        )
        ctbd_interior = _ctbd_silhouette(ctbd_metadata, source_regions, shape)
        segmentation_valid = cv2.bitwise_or(
            cv2.bitwise_or(ballons_interior, pr2_interior), ctbd_interior
        )
        paths = {
            "ballons_native": _write_mask(
                output_root / "silhouettes" / "ballons-native" / f"{page.page_id}.png",
                ballons_interior,
            ),
            "pr2_validated": _write_mask(
                output_root / "silhouettes" / "pr2-validated" / f"{page.page_id}.png",
                pr2_interior,
            ),
            "ctbd_bubble": _write_mask(
                output_root / "silhouettes" / "ctbd-bubble" / f"{page.page_id}.png",
                ctbd_interior,
            ),
            "ballons_clean_mask": _write_mask(
                output_root / "routes" / "ballons-clean" / f"{page.page_id}.png",
                ballons_clean,
            ),
            "pr2_clean_mask": _write_mask(
                output_root / "routes" / "pr2-clean" / f"{page.page_id}.png",
                pr2_clean,
            ),
            "segmentation_valid_mask": _write_mask(
                output_root / "routes" / "segmentation-valid" / f"{page.page_id}.png",
                segmentation_valid,
            ),
            "unsafe_signal_mask": _write_mask(
                output_root / "routes" / "unsafe" / f"{page.page_id}.png", unsafe
            ),
        }
        evidence[page.page_id] = {
            "ballons_clean": bool(np.any(ballons_clean)),
            "pr2_clean": bool(np.any(pr2_clean)),
            "segmentation_valid": bool(np.any(segmentation_valid)),
            "texture": False,
            "microtexture": False,
            "line_art": False,
            "ambiguous": bool(np.any(unsafe)),
            **{key: paths[key] for key in (
                "ballons_clean_mask", "pr2_clean_mask",
                "segmentation_valid_mask", "unsafe_signal_mask",
            )},
        }
        index.append(
            {
                "page_id": page.page_id,
                "source_seed_pixel_count": int(np.count_nonzero(source_seed)),
                "ballons_seed_pixel_count": int(np.count_nonzero(ballons_seed)),
                "selected_ballons_block_count": selected,
                **{
                    f"{name}_pixel_count": int(np.count_nonzero(mask))
                    for name, mask in (
                        ("ballons_native", ballons_interior),
                        ("pr2_validated", pr2_interior),
                        ("ctbd_bubble", ctbd_interior),
                    )
                },
            }
        )
    result = {
        "schema_version": "inpaint-silhouette-router-evidence-v4",
        "manifest": str(manifest_path.resolve()),
        "pages": evidence,
        "index": index,
    }
    _write_json(output_root / "router-evidence.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export E1 source-only Ballons, PR2, and CTBD route evidence."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ballons-mask-template", required=True)
    parser.add_argument("--ballons-metadata-template", required=True)
    parser.add_argument("--ctbd-metadata-template", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output_root, managed = select_managed_output_directory(
        family=FAMILY, category=CATEGORY, explicit_output_directory=args.output_dir
    )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        result = export_candidates(
            args.manifest.resolve(),
            ballons_mask_template=args.ballons_mask_template,
            ballons_metadata_template=args.ballons_metadata_template,
            ctbd_metadata_template=args.ctbd_metadata_template,
            output_root=output_root,
        )
        if managed is not None:
            managed.complete(metadata={"page_count": len(result["index"])})
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
