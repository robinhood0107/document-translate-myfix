#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    load_stage1_manifest,
    validate_source_only_manifest_v4,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _required_path(
    page: Mapping[str, object],
    field: str,
    *,
    base_dir: Path,
) -> str:
    value = page.get(field)
    if isinstance(value, Mapping):
        value = value.get("path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"relative matrix page lacks {field}")
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"relative matrix page lacks {field}")
    return str(path)


def build_relative_matrix(manifest_path: Path) -> dict[str, object]:
    validate_source_only_manifest_v4(manifest_path)
    payload = _read_json(manifest_path)
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list) or any(
        not isinstance(page, Mapping) for page in raw_pages
    ):
        raise ValueError("relative matrix manifest lacks pages")
    parsed = {page.page_id: page for page in load_stage1_manifest(manifest_path)}
    entries = {str(page.get("page_id") or ""): page for page in raw_pages}
    if set(parsed) != set(entries):
        raise ValueError("relative matrix page inventory differs")

    detector_pages: dict[str, dict[str, str]] = {}
    ownership_pages: dict[str, dict[str, str]] = {}
    silhouette_pages: dict[str, dict[str, str]] = {}
    router_pages: dict[str, dict[str, object]] = {}
    for page_id in sorted(entries):
        entry = entries[page_id]
        baseline_mask = _required_path(
            entry,
            "baseline_mask",
            base_dir=manifest_path.parent,
        )
        detector_pages[page_id] = {
            "raw": baseline_mask,
            "refined": baseline_mask,
            "dilated": baseline_mask,
        }
        ownership_pages[page_id] = {
            "mask": baseline_mask,
            "broad_mask": baseline_mask,
            "content_components": baseline_mask,
        }
        interior = parsed[page_id].bubble_interior_mask
        if not interior or not Path(interior).is_file():
            raise ValueError(f"relative matrix page lacks bubble interior: {page_id}")
        silhouette_pages[page_id] = {"interior": str(Path(interior).resolve())}
        router_pages[page_id] = {}

    base = {
        "detector": "pr6_baseline_edit",
        "ownership": "pr6_baseline_exact_ownership",
        "silhouette": "source_validated_interior",
        "router": "control_r0",
        "expansion": "raw",
    }
    combinations = [
        {**base, "fill": "mask_only"},
        {**base, "fill": "conditional_refill_existing"},
    ]
    return {
        "schema_version": "inpaint-factorized-matrix-v3",
        "relative_product_matrix": True,
        "factorized": False,
        "controls": combinations[0],
        "axes": {
            "detector": ["pr6_baseline_edit"],
            "ownership": ["pr6_baseline_exact_ownership"],
            "silhouette": ["source_validated_interior"],
            "router": ["control_r0"],
            "expansion": ["raw"],
            "fill": ["mask_only", "conditional_refill_existing"],
        },
        "families": {
            "detector": {
                "pr6_baseline_edit": {
                    "provider": "sealed_pr6_final_mask",
                    "seed_variant": "raw",
                    "source_seed_available": True,
                    "pages": detector_pages,
                }
            },
            "ownership": {
                "pr6_baseline_exact_ownership": {
                    "provider": "sealed_pr6_final_mask",
                    "pages": ownership_pages,
                }
            },
            "silhouette": {
                "source_validated_interior": {
                    "provider": "frozen_source_annotation",
                    "pages": silhouette_pages,
                }
            },
            "router": {
                "control_r0": {
                    "algorithm": "R0",
                    "pages": router_pages,
                }
            },
        },
        "oracle_only": [],
        "explicit_combinations": combinations,
        "retain_page_artifacts": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the frozen PR6 control and fill-only v3.2 matrix."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("relative matrix output must be fresh")
    payload = build_relative_matrix(args.manifest.resolve())
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
