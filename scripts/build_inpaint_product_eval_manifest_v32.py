#!/usr/bin/env python3
"""Build a lossless PR6 evaluation view of a sealed source-only v4 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    load_stage1_manifest,
    validate_source_only_manifest_v4,
)


COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
PRODUCT_SCHEMA_VERSION = 2
PRODUCT_SPLIT_ROLE = "tuning"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_manifest_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        {key: value for key, value in payload.items() if key != "manifest_sha256"},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _decision_sha256(pages: list[Mapping[str, object]]) -> str:
    encoded = json.dumps(
        [
            {
                "expected_edit": str(page["expected_edit"]),
                "page_id": str(page["page_id"]),
            }
            for page in sorted(pages, key=lambda value: str(value["page_id"]))
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_image(path: Path, flags: int) -> np.ndarray:
    value = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if value is None:
        raise ValueError(f"unable to decode source-only artifact: {path.name}")
    return value


def _write_zero_mask(path: Path, shape: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("product evaluation zero-mask output must be fresh")
    encoded, buffer = cv2.imencode(".png", np.zeros(shape, dtype=np.uint8))
    if not encoded:
        raise RuntimeError("failed to encode product evaluation zero mask")
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_bytes(buffer.tobytes())
    temporary.replace(path)


def _reference(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def build_product_eval_manifest(
    *,
    source_manifest_path: Path,
    output_path: Path,
    source_lock_git_sha: str,
) -> tuple[dict[str, object], dict[str, object]]:
    if not COMMIT_RE.fullmatch(source_lock_git_sha):
        raise ValueError("source lock must be a full lowercase Git SHA")
    if output_path.exists():
        raise FileExistsError("product evaluation manifest output must be fresh")
    source_binding = validate_source_only_manifest_v4(source_manifest_path)
    source_payload: dict[str, Any] = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    source_pages = load_stage1_manifest(source_manifest_path)
    declared_pages = source_payload.get("pages")
    if not isinstance(declared_pages, list) or len(declared_pages) != len(source_pages):
        raise ValueError("source manifest page inventory differs after validation")
    declared_by_id = {
        str(page.get("page_id") or ""): page
        for page in declared_pages
        if isinstance(page, Mapping)
    }
    if set(declared_by_id) != {page.page_id for page in source_pages}:
        raise ValueError("source manifest page IDs differ after validation")

    zero_dir = output_path.parent / f"{output_path.stem}-zero-targets"
    if zero_dir.exists():
        raise FileExistsError("product evaluation zero-mask directory must be fresh")
    product_pages: list[dict[str, object]] = []
    derived_zero_masks: list[dict[str, object]] = []
    try:
        for page in source_pages:
            declared = declared_by_id[page.page_id]
            source = Path(page.source_image).resolve()
            source_image = _read_image(source, cv2.IMREAD_COLOR)
            height, width = source_image.shape[:2]
            if int(declared.get("width") or 0) != width or int(
                declared.get("height") or 0
            ) != height:
                raise ValueError(f"source dimensions differ: {page.page_id}")
            if page.expected_edit == "required":
                if page.target_text_mask is None:
                    raise ValueError(f"required page lacks a target mask: {page.page_id}")
                target = Path(page.target_text_mask).resolve()
            else:
                if page.target_text_mask is not None:
                    raise ValueError(f"no-edit page unexpectedly has a target: {page.page_id}")
                target = zero_dir / f"{page.page_id}_target_text_zero.png"
                _write_zero_mask(target, (height, width))
                derived_zero_masks.append(
                    {
                        "page_id": page.page_id,
                        "path": str(target.resolve()),
                        "sha256": _sha256(target),
                    }
                )
            protected = Path(str(page.protected_structure_mask or "")).resolve()
            ambiguous = Path(str(page.ambiguous_structure_mask or "")).resolve()
            product_pages.append(
                {
                    "page_id": page.page_id,
                    "path": str(source),
                    "sha256": _sha256(source),
                    "size_bytes": source.stat().st_size,
                    "width": width,
                    "height": height,
                    "expected_edit": page.expected_edit,
                    "target_text_mask": _reference(target),
                    "protected_structure_mask": _reference(protected),
                    "ambiguous_structure_mask": _reference(ambiguous),
                }
            )
    except Exception:
        for record in derived_zero_masks:
            Path(str(record["path"])).unlink(missing_ok=True)
        if zero_dir.exists() and not any(zero_dir.iterdir()):
            zero_dir.rmdir()
        raise

    product_pages.sort(key=lambda value: str(value["page_id"]))
    manifest: dict[str, object] = {
        "schema_version": PRODUCT_SCHEMA_VERSION,
        "corpus_id": "e1-v32-product-baseline",
        "split_role": PRODUCT_SPLIT_ROLE,
        "source_lock_git_sha": source_lock_git_sha,
        "expected_count": len(product_pages),
        "pages": product_pages,
        "parent_manifest_sha256": str(source_binding["manifest_sha256"]),
        "expected_edit_basis": "source-only-review",
        "expected_edit_decisions_sha256": _decision_sha256(product_pages),
    }
    manifest["manifest_sha256"] = _canonical_manifest_sha256(manifest)
    provenance = {
        "schema_version": "inpaint-product-eval-manifest-provenance-v32",
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": str(source_binding["manifest_sha256"]),
        "source_page_inventory_sha256": str(
            source_binding["page_inventory_sha256"]
        ),
        "product_manifest_sha256": str(manifest["manifest_sha256"]),
        "expected_edit_decisions_sha256": str(
            manifest["expected_edit_decisions_sha256"]
        ),
        "page_count": len(product_pages),
        "required_page_count": sum(
            page["expected_edit"] == "required" for page in product_pages
        ),
        "no_edit_page_count": sum(
            page["expected_edit"] == "none" for page in product_pages
        ),
        "derived_zero_target_masks": derived_zero_masks,
        "annotation_transform": "none",
        "candidate_seen": False,
        "annotation_frozen_before_candidate": True,
    }
    return manifest, provenance


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a lossless PR6 evaluation view of source-only v4 evidence."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-lock-git-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.source_manifest.resolve()
    output = args.output.resolve()
    provenance_path = output.with_suffix(output.suffix + ".provenance.json")
    if provenance_path.exists():
        raise FileExistsError("product evaluation provenance output must be fresh")
    manifest, provenance = build_product_eval_manifest(
        source_manifest_path=source,
        output_path=output,
        source_lock_git_sha=str(args.source_lock_git_sha),
    )
    _write_json(output, manifest)
    provenance["product_manifest_file_sha256"] = _sha256(output)
    _write_json(provenance_path, provenance)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
