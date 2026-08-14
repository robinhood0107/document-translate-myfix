#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    validate_source_only_manifest_v4,
)


COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _read_jsonl(path: Path) -> dict[str, Mapping[str, object]]:
    rows: dict[str, Mapping[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError("baseline metrics row must be an object")
        page_id = str(row.get("page_id") or "")
        if not page_id or page_id in rows:
            raise ValueError("baseline metrics contain an empty or duplicate page id")
        rows[page_id] = row
    if not rows:
        raise ValueError("baseline metrics are empty")
    return rows


def seal_product_baseline(
    *,
    source_manifest_path: Path,
    corpus_artifact_dir: Path,
    metrics_pages_path: Path,
    product_commit: str,
) -> dict[str, object]:
    if not COMMIT_RE.fullmatch(product_commit):
        raise ValueError("product commit must be a full lowercase Git SHA")
    validate_source_only_manifest_v4(source_manifest_path)
    source = _read_json(source_manifest_path)
    pages = source.get("pages")
    if not isinstance(pages, list) or not pages or any(
        not isinstance(page, Mapping) for page in pages
    ):
        raise ValueError("source manifest requires pages")
    metrics = _read_jsonl(metrics_pages_path)
    page_ids = [str(page.get("page_id") or "") for page in pages]
    if any(not page_id for page_id in page_ids) or len(page_ids) != len(set(page_ids)):
        raise ValueError("source manifest contains invalid page ids")
    if set(page_ids) != set(metrics):
        raise ValueError("source manifest and baseline metrics page inventories differ")
    output_pages: list[dict[str, object]] = []
    for page in pages:
        page_id = str(page["page_id"])
        source_path = Path(str(page.get("path") or ""))
        if not source_path.is_absolute():
            source_path = source_manifest_path.parent / source_path
        source_path = source_path.resolve()
        baseline_image = corpus_artifact_dir / "cleaned_images" / f"{page_id}_cleaned.png"
        baseline_mask = corpus_artifact_dir / "final_masks" / f"{page_id}_final_mask.png"
        for artifact in (source_path, baseline_image, baseline_mask):
            if not artifact.is_file():
                raise FileNotFoundError(artifact)
        row = metrics[page_id]
        source_sha = _sha256(source_path)
        if str(row.get("source_sha256") or "") != source_sha:
            raise ValueError(f"baseline metric source SHA differs: {page_id}")
        if str(row.get("cleaned_sha256") or "") != _sha256(baseline_image):
            raise ValueError(f"baseline cleaned SHA differs: {page_id}")
        if str(row.get("final_mask_sha256") or "") != _sha256(baseline_mask):
            raise ValueError(f"baseline final-mask SHA differs: {page_id}")
        output_pages.append(
            {
                "page_id": page_id,
                "source_path": str(source_path.resolve()),
                "source_sha256": source_sha,
                "baseline_image": str(baseline_image.resolve()),
                "baseline_image_sha256": _sha256(baseline_image),
                "baseline_mask": str(baseline_mask.resolve()),
                "baseline_mask_sha256": _sha256(baseline_mask),
            }
        )
    return {
        "schema_version": "inpaint-product-baseline-manifest-v32",
        "source_manifest_sha256": _sha256(source_manifest_path),
        "metrics_pages_sha256": _sha256(metrics_pages_path),
        "product_commit": product_commit,
        "page_count": len(output_pages),
        "page_ids": sorted(page_ids),
        "pages": sorted(output_pages, key=lambda row: str(row["page_id"])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seal PR6 cleaned images and final masks as a v3.2 baseline."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--corpus-artifact-dir", type=Path, required=True)
    parser.add_argument("--metrics-pages", type=Path, required=True)
    parser.add_argument("--product-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("product baseline manifest output must be fresh")
    payload = seal_product_baseline(
        source_manifest_path=args.source_manifest.resolve(),
        corpus_artifact_dir=args.corpus_artifact_dir.resolve(),
        metrics_pages_path=args.metrics_pages.resolve(),
        product_commit=str(args.product_commit),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
