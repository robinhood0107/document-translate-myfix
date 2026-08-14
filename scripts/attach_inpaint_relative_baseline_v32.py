#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    SOURCE_ONLY_MANIFEST_SCHEMA_V4,
    manifest_page_artifact_sha256,
    source_manifest_page_inventory_sha256,
    validate_source_only_manifest_v4,
)


SEAL_SCHEMA_VERSION = "inpaint-factorized-manifest-seal-v4"


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


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _page_map(payload: Mapping[str, object], label: str) -> dict[str, Mapping[str, object]]:
    pages = payload.get("pages")
    if not isinstance(pages, list) or not pages or any(
        not isinstance(page, Mapping) for page in pages
    ):
        raise ValueError(f"{label} requires non-empty page records")
    result: dict[str, Mapping[str, object]] = {}
    for page in pages:
        page_id = str(page.get("page_id") or "")
        if not page_id or page_id in result:
            raise ValueError(f"{label} contains an empty or duplicate page id")
        result[page_id] = page
    return result


def _required_path(
    page: Mapping[str, object],
    field: str,
    *,
    base_dir: Path,
) -> Path:
    value = page.get(field)
    if isinstance(value, Mapping):
        value = value.get("path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"baseline page lacks {field}")
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def attach_relative_baseline(
    source_manifest_path: Path,
    baseline_manifest_path: Path,
    output_path: Path,
) -> dict[str, object]:
    source_binding = validate_source_only_manifest_v4(source_manifest_path)
    source = _read_json(source_manifest_path)
    baseline = _read_json(baseline_manifest_path)
    if baseline.get("schema_version") != "inpaint-product-baseline-manifest-v32":
        raise ValueError("relative baseline requires product baseline manifest v32")
    if str(baseline.get("source_manifest_sha256") or "") != str(
        source_binding["manifest_sha256"]
    ):
        raise ValueError("baseline manifest is bound to a different source manifest")
    source_pages = _page_map(source, "source manifest")
    baseline_pages = _page_map(baseline, "baseline manifest")
    if set(source_pages) != set(baseline_pages):
        raise ValueError("source and baseline page inventories differ")

    pages: list[dict[str, object]] = []
    for page_id in sorted(source_pages):
        source_page = source_pages[page_id]
        baseline_page = baseline_pages[page_id]
        source_path = _required_path(
            source_page,
            "path",
            base_dir=source_manifest_path.parent,
        )
        baseline_source = _required_path(
            baseline_page,
            "source_path",
            base_dir=baseline_manifest_path.parent,
        )
        if _sha256(source_path) != _sha256(baseline_source):
            raise ValueError(f"baseline source bytes differ: {page_id}")
        declared_source_sha = str(baseline_page.get("source_sha256") or "")
        if declared_source_sha != _sha256(source_path):
            raise ValueError(f"baseline source SHA differs: {page_id}")
        baseline_image = _required_path(
            baseline_page,
            "baseline_image",
            base_dir=baseline_manifest_path.parent,
        )
        baseline_mask = _required_path(
            baseline_page,
            "baseline_mask",
            base_dir=baseline_manifest_path.parent,
        )
        for field, artifact in (
            ("baseline_image_sha256", baseline_image),
            ("baseline_mask_sha256", baseline_mask),
        ):
            if str(baseline_page.get(field) or "") != _sha256(artifact):
                raise ValueError(f"baseline artifact SHA differs: {page_id}/{field}")

        page = dict(source_page)
        page["baseline"] = str(baseline_image)
        page["baseline_mask"] = str(baseline_mask)
        page["existing_source_edit_mask"] = str(baseline_mask)
        page["candidate_seen"] = False
        page["annotation_frozen_before_candidate"] = True
        page["artifact_sha256"] = manifest_page_artifact_sha256(output_path, page)
        pages.append(page)

    payload: dict[str, object] = dict(source)
    payload.update(
        {
            "schema_version": SOURCE_ONLY_MANIFEST_SCHEMA_V4,
            "corpus_id": f"{source['corpus_id']}-relative-v32",
            "split_role": "development_source_only",
            "source_annotation_manifest_sha256": source_binding[
                "manifest_sha256"
            ],
            "baseline_manifest_sha256": _sha256(baseline_manifest_path),
            "annotation_frozen_before_candidate": True,
            "candidate_seen": False,
            "pages": pages,
            "page_count": len(pages),
            "page_ids": sorted(source_pages),
            "page_inventory_sha256": source_manifest_page_inventory_sha256(pages),
        }
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Attach a hash-bound PR6 baseline to frozen E1 annotations."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.source_manifest.resolve()
    baseline = args.baseline_manifest.resolve()
    output = args.output.resolve()
    if output.exists() or output.with_suffix(output.suffix + ".seal.json").exists():
        raise FileExistsError("relative manifest output must be fresh")
    payload = attach_relative_baseline(source, baseline, output)
    _write_json(output, payload)
    _write_json(
        output.with_suffix(output.suffix + ".seal.json"),
        {
            "schema_version": SEAL_SCHEMA_VERSION,
            "manifest": output.name,
            "manifest_sha256": _sha256(output),
            "source_manifest_sha256": _sha256(source),
            "baseline_manifest_sha256": _sha256(baseline),
            "annotation_frozen_before_candidate": True,
            "candidate_seen": False,
            "candidate_generated": False,
        },
    )
    validate_source_only_manifest_v4(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
