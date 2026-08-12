#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validation_artifact_harness import (  # noqa: E402
    default_archive_root,
    select_managed_output_directory,
)


SCHEMA_VERSION = "inpaint-development-source-index-v4"
FAMILY = "inpaint-development-source-index-v4"
CATEGORY = "40-inpaint-mask-render"
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_source_set(value: str) -> tuple[str, Path, Path | None]:
    parts = value.split("::")
    if len(parts) not in {2, 3}:
        raise ValueError("source set must be ID::SOURCE_DIR[::PAIRED_DIR]")
    set_id = parts[0].strip()
    source_dir = Path(parts[1]).resolve()
    paired_dir = Path(parts[2]).resolve() if len(parts) == 3 else None
    if not set_id or not set_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"invalid source set id: {set_id!r}")
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    if paired_dir is not None and not paired_dir.is_dir():
        raise FileNotFoundError(paired_dir)
    return set_id, source_dir, paired_dir


def _indexed_images(directory: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in indexed:
            raise ValueError(f"duplicate image stem in {directory}: {path.stem}")
        indexed[path.stem] = path
    return indexed


def _read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise FileNotFoundError(path)
    return image


def build_source_index(source_sets: list[str]) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    source_sha_to_page: dict[str, str] = {}
    for raw_spec in source_sets:
        set_id, source_dir, paired_dir = _parse_source_set(raw_spec)
        sources = _indexed_images(source_dir)
        paired = _indexed_images(paired_dir) if paired_dir is not None else {}
        if paired_dir is not None and set(paired) != set(sources):
            missing = sorted(set(sources) - set(paired))
            extra = sorted(set(paired) - set(sources))
            raise ValueError(
                f"paired set mismatch for {set_id}: missing={missing}, extra={extra}"
            )
        for stem, source_path in sources.items():
            image = _read_image(source_path)
            source_sha = _sha256(source_path)
            page_id = f"{set_id}-{stem}"
            if source_sha in source_sha_to_page:
                raise ValueError(
                    f"duplicate source image SHA: {page_id} and {source_sha_to_page[source_sha]}"
                )
            source_sha_to_page[source_sha] = page_id
            entry: dict[str, Any] = {
                "page_id": page_id,
                "set_id": set_id,
                "path": str(source_path),
                "source_sha256": source_sha,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "expected_edit": "required",
            }
            if stem in paired:
                paired_path = paired[stem]
                paired_image = _read_image(paired_path)
                if paired_image.shape[:2] != image.shape[:2]:
                    raise ValueError(f"paired image shape mismatch: {page_id}")
                entry["paired_reference"] = {
                    "path": str(paired_path),
                    "source_sha256": source_sha,
                    "reference_sha256": _sha256(paired_path),
                    "proposal_only": True,
                }
            pages.append(entry)
    return {
        "schema_version": SCHEMA_VERSION,
        "corpus_id": "e1-development-v4",
        "split_role": "development_source_only",
        "candidate_images_generated": False,
        "inpainting_invoked": False,
        "pages": pages,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a source-only development index with proposal-only pairs."
    )
    parser.add_argument(
        "--source-set",
        action="append",
        required=True,
        help="ID::SOURCE_DIR[::PAIRED_DIR]; repeat for each source family",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    output_dir, managed_run = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    archive = default_archive_root().resolve()
    try:
        output_dir.resolve().relative_to(archive)
    except ValueError as exc:
        raise ValueError("source index output must stay in the private archive") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "source-index-v4.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    payload = build_source_index(list(args.source_set))
    _write_json(output_path, payload)
    if managed_run is not None:
        managed_run.complete(
            metadata={"page_count": len(payload["pages"]), "schema_version": SCHEMA_VERSION}
        )
    print(json.dumps({"output": str(output_path), "page_count": len(payload["pages"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
