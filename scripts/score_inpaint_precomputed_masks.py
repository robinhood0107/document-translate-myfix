#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.contracts import CandidateMaskResult  # noqa: E402
from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    load_page_masks,
    load_stage1_manifest,
    score_page,
    summarize,
)
from scripts.benchmark_inpaint_detector_bakeoff import _existing_edit_paths  # noqa: E402
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-detector-bakeoff-v2"
CATEGORY = "40-inpaint-mask-render"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        pages = load_stage1_manifest(args.manifest)
        index = json.loads(args.index.read_text(encoding="utf-8"))
        precomputed = {
            str(row.get("page_id") or ""): row
            for row in index.get("pages", [])
            if isinstance(row, dict)
        }
        existing = _existing_edit_paths(args.manifest)
        rows = []
        edit_root = output_root / "positive_edit_masks"
        edit_root.mkdir(parents=True, exist_ok=True)
        for page in pages:
            image = cv2.imread(page.source_image, cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise FileNotFoundError(page.source_image)
            source = precomputed.get(page.page_id)
            if source is None:
                raise ValueError(f"precomputed index is missing {page.page_id}")
            mask = cv2.imread(str(source.get("path") or ""), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.size == 0:
                raise FileNotFoundError(source.get("path"))
            result = CandidateMaskResult(
                args.candidate_id,
                mask,
                mask,
                mask,
                runtime=dict(source.get("runtime") or {}),
            )
            masks = load_page_masks(
                page,
                image.shape[:2],
                existing_edit_path=existing.get(page.page_id),
            )
            row, edit = score_page(page, result, masks, variant="raw")
            rows.append(row)
            if not cv2.imwrite(str(edit_root / f"{page.page_id}_positive_edit.png"), edit):
                raise OSError(page.page_id)
        payload = {
            "schema_version": "inpaint-detector-bakeoff-stage1-v1",
            "candidate": args.candidate_id,
            "variant": "raw",
            "manifest_sha256": _sha256(args.manifest),
            "source_index_sha256": _sha256(args.index),
            "summary": summarize(rows),
            "pages": rows,
        }
        (output_root / "stage1-results.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if managed is not None:
            managed.complete(
                metadata={
                    "candidate": args.candidate_id,
                    "manifest_sha256": payload["manifest_sha256"],
                    "source_index_sha256": payload["source_index_sha256"],
                    "summary": payload["summary"],
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
            managed.fail(error, metadata={"candidate": args.candidate_id})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
