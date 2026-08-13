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

from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    manifest_page_artifact_sha256,
    source_manifest_page_inventory_sha256,
    validate_source_only_manifest_v4,
)


SHAPE = (256, 320)
SCHEMA = "inpaint-factorized-source-manifest-v4"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_image(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", np.asarray(value))
    if not success:
        raise OSError(path)
    encoded.tofile(path)
    return str(path.resolve())


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _shift_mask(mask: np.ndarray, delta_x: int, delta_y: int) -> np.ndarray:
    return cv2.warpAffine(
        mask,
        np.array(((1.0, 0.0, float(delta_x)), (0.0, 1.0, float(delta_y)))),
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _base_scene(background: str) -> tuple[np.ndarray, np.ndarray]:
    height, width = SHAPE
    image = np.full((height, width, 3), 150, np.uint8)
    interior = np.zeros(SHAPE, np.uint8)
    cv2.ellipse(interior, (160, 128), (118, 92), 0, 0, 360, 255, -1)
    if background == "gradient":
        gradient = np.linspace(225, 248, height, dtype=np.uint8)
        field = np.repeat(gradient[:, None], width, axis=1)
        image[interior > 0] = np.repeat(field[:, :, None], 3, axis=2)[interior > 0]
    else:
        image[interior > 0] = (242, 242, 242)
    if background == "paper_noise":
        y, x = np.indices(SHAPE)
        noise = ((x * 13 + y * 7) % 9 - 4).astype(np.int16)
        adjusted = np.clip(image.astype(np.int16) + noise[:, :, None], 0, 255)
        image[interior > 0] = adjusted.astype(np.uint8)[interior > 0]
    elif background == "halftone":
        for y in range(42, 216, 8):
            for x in range(46, 278, 8):
                if interior[y, x]:
                    cv2.circle(image, (x, y), 2, (70, 70, 70), -1)
    elif background == "hatching":
        for offset in range(-220, 320, 9):
            cv2.line(image, (offset, 35), (offset + 180, 215), (75, 75, 75), 2)
        image[interior == 0] = 150
    cv2.ellipse(image, (160, 128), (122, 96), 0, 0, 360, (24, 24, 24), 3)
    return image, interior


def _glyph_mask(kind: str) -> np.ndarray:
    mask = np.zeros(SHAPE, np.uint8)
    if kind == "vertical":
        for y in (82, 118, 154):
            cv2.rectangle(mask, (154, y), (166, y + 25), 255, -1)
            cv2.line(mask, (148, y + 12), (172, y + 12), 255, 4)
    elif kind == "crop_edge":
        cv2.putText(mask, "TXT", (-12, 142), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 4, cv2.LINE_AA)
    elif kind == "small":
        for x in (128, 152, 176):
            cv2.rectangle(mask, (x, 112), (x + 12, 132), 255, 2)
            cv2.line(mask, (x + 2, 122), (x + 10, 122), 255, 2)
    else:
        cv2.putText(mask, "TXT", (104, 142), cv2.FONT_HERSHEY_SIMPLEX, 1.25, 255, 4, cv2.LINE_AA)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _paint_text(image: np.ndarray, target: np.ndarray, style: str) -> None:
    if style == "bright":
        image[target > 0] = (248, 248, 248)
        return
    if style in {"outline", "shadow", "glow"}:
        radius = {"outline": 2, "shadow": 3, "glow": 5}[style]
        halo = cv2.dilate(target, np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8))
        if style == "shadow":
            shifted = _shift_mask(halo, 3, 3)
            image[(shifted > 0) & (target == 0)] = (85, 85, 85)
        elif style == "glow":
            image[(halo > 0) & (target == 0)] = (210, 210, 210)
        else:
            image[(halo > 0) & (target == 0)] = (22, 22, 22)
        image[target > 0] = (245, 245, 245)
        target[halo > 0] = 255
        return
    image[target > 0] = (24, 24, 24)


def _fixture(
    root: Path,
    fixture_id: str,
    *,
    background: str = "flat",
    glyph: str = "normal",
    style: str = "dark",
    route: str = "clean_flat",
    seed_mode: str = "full",
    silhouette_mode: str = "valid",
    ownership_mode: str = "valid",
    preserve: bool = False,
    line_art: bool = False,
) -> dict[str, object]:
    image, interior = _base_scene(background)
    target = _glyph_mask(glyph)
    protected = np.zeros(SHAPE, np.uint8)
    ambiguous = np.zeros(SHAPE, np.uint8)
    if line_art:
        cv2.line(image, (72, 151), (248, 151), (20, 20, 20), 4)
        cv2.line(protected, (72, 151), (248, 151), 255, 7)
        target[protected > 0] = 0
    _paint_text(image, target, style)
    corner = cv2.subtract(cv2.dilate(interior, np.ones((9, 9), np.uint8)), interior)
    if silhouette_mode == "under":
        silhouette = cv2.erode(interior, np.ones((31, 31), np.uint8))
    elif silhouette_mode == "over":
        silhouette = cv2.dilate(interior, np.ones((35, 35), np.uint8))
    elif silhouette_mode == "empty":
        silhouette = np.zeros(SHAPE, np.uint8)
    else:
        silhouette = interior.copy()
    if ownership_mode == "conflict":
        ownership = interior.copy()
    elif ownership_mode == "unowned":
        ownership = np.full(SHAPE, 255, np.uint8)
    else:
        ownership = interior.copy()
    if glyph == "crop_edge":
        ownership[target > 0] = 255
    if seed_mode == "partial":
        seed = np.zeros(SHAPE, np.uint8)
        seed[:, :160] = target[:, :160]
    elif seed_mode == "missed":
        seed = np.zeros(SHAPE, np.uint8)
    else:
        seed = target.copy()
    zero = np.zeros(SHAPE, np.uint8)
    page_root = root / fixture_id
    paths = {
        "source": _write_image(page_root / "source.png", image),
        "target": _write_image(page_root / "target.png", zero if preserve else target),
        "preserve": _write_image(page_root / "preserve.png", target if preserve else zero),
        "interior": _write_image(page_root / "interior.png", silhouette),
        "ownership": _write_image(page_root / "ownership.png", ownership),
        "protected": _write_image(page_root / "protected.png", protected),
        "ambiguous": _write_image(page_root / "ambiguous.png", ambiguous),
        "corner": _write_image(page_root / "corner.png", corner),
        "seed": _write_image(page_root / "seed.png", seed),
        "zero": _write_image(page_root / "zero.png", zero),
    }
    priority = "optional" if preserve else "required"
    action = "preserve" if preserve else "translate_inpaint"
    semantic_role = "sfx" if preserve else ("dialogue_free" if ownership_mode == "unowned" else "dialogue_bubble")
    instance_mask = paths["preserve"] if preserve else paths["target"]
    region = {
        "region_id": "region-0",
        "bubble_route_class": route,
        "bubble_interior_mask": paths["interior"],
        "ownership_mask": paths["ownership"],
        "protected_structure_mask": paths["protected"],
        "ambiguous_structure_mask": paths["ambiguous"],
        "corner_protect_mask": paths["corner"],
    }
    regions = [region]
    if ownership_mode == "conflict":
        regions.append(
            {
                **region,
                "region_id": "region-overlap",
                "bubble_route_class": "line_art",
            }
        )
    return {
        "page_id": f"synthetic-{fixture_id}",
        "path": paths["source"],
        "width": SHAPE[1],
        "height": SHAPE[0],
        "expected_edit": "none" if preserve else "required",
        "target_text_mask": None if preserve else paths["target"],
        "preserve_mask": paths["preserve"],
        "target_instances": [
            {
                "instance_id": f"{fixture_id}-text",
                "region_id": "region-0",
                "mask_path": instance_mask,
                "semantic_role": semantic_role,
                "processing_action": action,
                "priority": priority,
                "source_reviewed": True,
            }
        ],
        "regions": regions,
        "protected_structure_mask": paths["protected"],
        "ambiguous_structure_mask": paths["ambiguous"],
        "ownership_mask": paths["ownership"],
        "claim_seed_mask": paths["seed"],
        "bubble_interior_mask": paths["interior"],
        "corner_protect_mask": paths["corner"],
        "existing_source_edit_mask": paths["zero"],
        "baseline": paths["source"],
        "baseline_mask": paths["zero"],
        "target_mask_provenance": "synthetic_ground_truth_v4",
        "target_extent_independent": True,
        "target_inventory_independent": True,
        "target_review_complete": True,
        "synthetic_seed_mode": seed_mode,
        "synthetic_silhouette_mode": silhouette_mode,
        "synthetic_ownership_mode": ownership_mode,
    }


def build_synthetic_manifest(output_root: Path) -> dict[str, object]:
    specs = (
        ("small-cjk-dark", dict(glyph="small")),
        ("small-cjk-bright", dict(glyph="small", style="bright", background="halftone", route="texture")),
        ("vertical-outline", dict(glyph="vertical", style="outline", background="gradient", route="clean_gradient")),
        ("shadow", dict(style="shadow", background="gradient", route="clean_gradient")),
        ("glow", dict(style="glow")),
        ("crop-edge", dict(glyph="crop_edge")),
        ("paper-noise", dict(background="paper_noise")),
        ("halftone", dict(background="halftone", route="texture")),
        ("hatching", dict(background="hatching", route="texture")),
        ("line-art", dict(line_art=True, route="line_art")),
        ("partial-detection", dict(seed_mode="partial")),
        ("complete-miss", dict(seed_mode="missed")),
        ("silhouette-under", dict(silhouette_mode="under")),
        ("silhouette-over", dict(silhouette_mode="over")),
        ("silhouette-empty", dict(silhouette_mode="empty", route="ambiguous")),
        ("ownership-conflict", dict(ownership_mode="conflict", route="ambiguous")),
        ("unowned-meaningful", dict(ownership_mode="unowned", route="ambiguous")),
        ("preserve-sfx", dict(preserve=True, ownership_mode="unowned", route="ambiguous")),
    )
    pages = [_fixture(output_root, fixture_id, **options) for fixture_id, options in specs]
    manifest_path = output_root / "synthetic-inpaint-generalization-v4.json"
    for page in pages:
        page["annotation_basis"] = "synthetic_known_ground_truth_v4"
        page["annotation_frozen_before_candidate"] = True
        page["candidate_seen"] = False
        page["artifact_sha256"] = manifest_page_artifact_sha256(
            manifest_path,
            page,
        )
        page["source_sha256"] = page["artifact_sha256"]["path"]
    page_ids = sorted(str(page["page_id"]) for page in pages)
    return {
        "schema_version": SCHEMA,
        "corpus_id": "synthetic-inpaint-generalization-v4",
        "split_role": "synthetic_known_ground_truth",
        "annotation_frozen_before_candidate": True,
        "candidate_seen": False,
        "target_extent_independent": True,
        "target_inventory_independent": True,
        "target_review_complete": True,
        "page_count": len(page_ids),
        "page_ids": page_ids,
        "page_inventory_sha256": source_manifest_page_inventory_sha256(pages),
        "pages": pages,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the neutral v4 inpaint detector/router challenge corpus."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(
            "synthetic v4 output directory must be fresh and absent: " f"{output}"
        )
    output.mkdir(parents=True)
    manifest_path = output / "synthetic-inpaint-generalization-v4.json"
    _write_json(manifest_path, build_synthetic_manifest(output))
    _write_json(
        manifest_path.with_suffix(manifest_path.suffix + ".seal.json"),
        {
            "schema_version": "inpaint-factorized-manifest-seal-v4-synthetic",
            "manifest": manifest_path.name,
            "manifest_sha256": _sha256(manifest_path),
            "annotation_frozen_before_candidate": True,
            "candidate_seen": False,
            "candidate_generated": False,
        },
    )
    validate_source_only_manifest_v4(manifest_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
