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

from scripts.validation_artifact_harness import select_managed_output_directory  # noqa: E402


FAMILY = "inpaint-bakeoff-existing-edit-v2"
CATEGORY = "40-inpaint-mask-render"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path, flags: int) -> np.ndarray:
    value = cv2.imread(str(path), flags)
    if value is None or value.size == 0:
        raise FileNotFoundError(path)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the sealed baseline ownership mask while reopening only source-only "
            "annotated residual targets for detector bake-off."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest.resolve()
    baseline_root = args.baseline_run.resolve()
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        records: list[dict[str, object]] = []
        for page in payload.get("pages", []):
            page_id = str(page.get("page_id") or "").strip()
            if not page_id:
                raise ValueError("manifest contains an empty page id")
            source_path = baseline_root / "source_images" / f"{page_id}_source.png"
            cleaned_path = baseline_root / "cleaned_images" / f"{page_id}_cleaned.png"
            final_mask_path = baseline_root / "final_masks" / f"{page_id}_final_mask.png"
            source = _read(source_path, cv2.IMREAD_COLOR)
            cleaned = _read(cleaned_path, cv2.IMREAD_COLOR)
            final_mask = _read(final_mask_path, cv2.IMREAD_GRAYSCALE)
            if source.shape != cleaned.shape or source.shape[:2] != final_mask.shape:
                raise ValueError(f"baseline shape mismatch for {page_id}")
            target_value = page.get("target_text_mask", page.get("target_glyph_mask"))
            if isinstance(target_value, dict):
                target_value = target_value.get("path")
            if isinstance(target_value, str) and target_value.strip():
                target = _read(Path(target_value), cv2.IMREAD_GRAYSCALE)
                if target.shape != final_mask.shape:
                    raise ValueError(f"target mask shape mismatch for {page_id}")
            else:
                target = np.zeros_like(final_mask)
            changed = np.any(source != cleaned, axis=2)
            exact = np.where((final_mask > 0) & (target == 0), 255, 0).astype(np.uint8)
            output_path = output_root / f"{page_id}_existing_edit.png"
            if not cv2.imwrite(str(output_path), exact):
                raise OSError(f"failed to write {output_path}")
            records.append(
                {
                    "page_id": page_id,
                    "source_sha256": _sha256(source_path),
                    "cleaned_sha256": _sha256(cleaned_path),
                    "final_mask_sha256": _sha256(final_mask_path),
                    "requested_final_mask_pixel_count": int(np.count_nonzero(final_mask)),
                    "source_only_residual_target_pixel_count": int(np.count_nonzero(target)),
                    "changed_in_final_mask_pixel_count": int(
                        np.count_nonzero(changed & (final_mask > 0))
                    ),
                    "baseline_owned_excluding_residual_pixel_count": int(
                        np.count_nonzero(exact)
                    ),
                    "path": str(output_path),
                }
            )
        index = {
            "schema_version": "inpaint-bakeoff-existing-edit-v2",
            "manifest_sha256": _sha256(manifest_path),
            "baseline_run": str(baseline_root),
            "pages": records,
        }
        index_path = output_root / "existing-edit-index.json"
        index_path.write_text(
            json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if managed is not None:
            managed.complete(
                metadata={
                    "manifest_sha256": index["manifest_sha256"],
                    "page_count": len(records),
                    "baseline_run": baseline_root.name,
                }
            )
            mismatches = managed.verify()
            if mismatches:
                raise RuntimeError("managed artifact verification failed: " + "; ".join(mismatches))
            print(str(managed.run_root))
        else:
            print(str(output_root))
        return 0
    except BaseException as error:
        if managed is not None:
            managed.fail(error, metadata={"manifest": manifest_path.name})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
