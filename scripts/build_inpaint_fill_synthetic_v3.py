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

from scripts.build_inpaint_factorized_manifest_v3 import (  # noqa: E402
    attach_artifact_hashes,
    validate_manifest,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_image(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(path)
    return str(path.resolve())


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _bubble_truth(kind: str, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    truth = np.full((height, width, 3), 150, np.uint8)
    interior = np.zeros(shape, np.uint8)
    cv2.ellipse(interior, (width // 2, height // 2), (104, 82), 0, 0, 360, 255, -1)
    if kind == "halftone":
        for y in range(8, height - 8, 8):
            for x in range(8, width - 8, 8):
                if interior[y, x] == 0:
                    cv2.circle(truth, (x, y), 2, (65, 65, 65), -1)
    if kind == "clean_gradient":
        rows = np.linspace(228, 248, height, dtype=np.uint8)
        gradient = np.repeat(rows[:, None], width, axis=1)
        for channel in range(3):
            truth[:, :, channel][interior > 0] = gradient[interior > 0]
    else:
        truth[interior > 0] = (244, 244, 244)
    cv2.ellipse(truth, (width // 2, height // 2), (106, 84), 0, 0, 360, (25, 25, 25), 3)
    return truth, interior


def build_synthetic_manifest(output_root: Path) -> dict[str, object]:
    shape = (256, 320)
    pages: list[dict[str, object]] = []
    for kind, route_class in (
        ("clean_flat", "clean_flat"),
        ("clean_gradient", "clean_gradient"),
        ("halftone", "texture"),
        ("line_art", "line_art"),
    ):
        truth, interior = _bubble_truth(kind, shape)
        protected = np.zeros(shape, np.uint8)
        if kind == "line_art":
            cv2.line(truth, (92, 168), (228, 168), (25, 25, 25), 4)
            cv2.line(protected, (92, 168), (228, 168), 255, 6)
        text = np.zeros(shape, np.uint8)
        cv2.putText(
            text,
            "SFX",
            (104, 142),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.25,
            255,
            4,
            cv2.LINE_AA,
        )
        text[protected > 0] = 0
        source = truth.copy()
        alpha = text.astype(np.float32) / 255.0
        source = np.rint(
            source.astype(np.float32) * (1.0 - alpha[:, :, None])
            + 25.0 * alpha[:, :, None]
        ).astype(np.uint8)
        zero = np.zeros(shape, np.uint8)
        page_root = output_root / kind
        source_path = _write_image(page_root / "source.png", source)
        truth_path = _write_image(page_root / "known-background.png", truth)
        target_path = _write_image(page_root / "target.png", text)
        interior_path = _write_image(page_root / "interior.png", interior)
        protected_path = _write_image(page_root / "protected.png", protected)
        zero_path = _write_image(page_root / "zero.png", zero)
        pages.append(
            {
                "page_id": f"synthetic-{kind}",
                "path": source_path,
                "target_text_mask": target_path,
                "target_instances": [
                    {"instance_id": f"{kind}-text", "mask_path": target_path}
                ],
                "bubble_route_class": route_class,
                "bubble_interior_mask": interior_path,
                "protected_structure_mask": protected_path,
                "ambiguous_structure_mask": zero_path,
                "ownership_mask": interior_path,
                "claim_seed_mask": interior_path,
                "corner_protect_mask": zero_path,
                "existing_source_edit_mask": zero_path,
                "baseline": source_path,
                "baseline_mask": zero_path,
                "known_background": truth_path,
                "expected_edit": "required",
                "annotation_basis": "synthetic_known_background_v3",
                "target_mask_provenance": "synthetic_ground_truth",
                "target_extent_independent": True,
            }
        )
    payload: dict[str, object] = {
        "schema_version": "inpaint-detector-bakeoff-manifest-v3",
        "corpus_id": "synthetic-fill-v3",
        "split_role": "synthetic_known_background",
        "annotation_frozen_before_candidate": True,
        "pages": pages,
    }
    attach_artifact_hashes(payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build neutral known-background fill fixtures for inpaint v3."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = args.output_dir.resolve()
    manifest = output_root / "synthetic-fill-manifest-v3.json"
    _write_json(manifest, build_synthetic_manifest(output_root))
    validate_manifest(manifest)
    _write_json(
        manifest.with_suffix(manifest.suffix + ".seal.json"),
        {
            "schema_version": "inpaint-factorized-manifest-seal-v3",
            "manifest": manifest.name,
            "manifest_sha256": _sha256(manifest),
            "candidate_generated": False,
        },
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
