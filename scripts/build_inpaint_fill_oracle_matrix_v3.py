#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "inpaint-factorized-matrix-v3"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be an object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def build_fill_matrix(manifest_path: Path) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    pages = manifest.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("fill manifest must contain pages")
    detector_pages: dict[str, dict[str, str]] = {}
    ownership_pages: dict[str, dict[str, str]] = {}
    silhouette_pages: dict[str, dict[str, str]] = {}
    router_pages: dict[str, dict[str, object]] = {}
    for page in pages:
        if not isinstance(page, dict):
            raise ValueError("fill manifest page must be an object")
        page_id = str(page.get("page_id") or "").strip()
        target = str(page.get("target_text_mask") or "").strip()
        ownership = str(page.get("ownership_mask") or "").strip()
        interior = str(page.get("bubble_interior_mask") or "").strip()
        zero = str(page.get("ambiguous_structure_mask") or "").strip()
        route_class = str(page.get("bubble_route_class") or "").strip()
        if not all((page_id, target, ownership, interior, zero, route_class)):
            raise ValueError("fill manifest page lacks required evidence")
        detector_pages[page_id] = {
            "raw": target,
            "refined": target,
            "dilated": target,
        }
        ownership_pages[page_id] = {
            "mask": ownership,
            "broad_mask": interior,
            "content_components": target,
        }
        silhouette_pages[page_id] = {"interior": interior}
        clean = route_class in {"clean_flat", "clean_gradient"}
        router_pages[page_id] = {
            "ballons_clean_mask": interior if clean else zero,
            "pr2_clean_mask": interior if clean else zero,
            "segmentation_valid_mask": interior,
        }

    fills = (
        "robust_flat_median",
        "planar_gradient",
        "telea",
        "current_lama",
        "ballons_lama",
        "conditional_hybrid",
    )
    explicit: list[dict[str, str]] = []
    for fill in fills[:-1]:
        explicit.append(
            {
                "detector": "annotation_target_oracle",
                "ownership": "annotation_ownership_oracle",
                "silhouette": "annotation_interior_oracle",
                "router": "control_r0",
                "expansion": "raw",
                "fill": fill,
            }
        )
    for fill in fills:
        explicit.append(
            {
                "detector": "annotation_target_oracle",
                "ownership": "annotation_ownership_oracle",
                "silhouette": "annotation_interior_oracle",
                "router": "clean_route_oracle",
                "expansion": "bubble_interior",
                "fill": fill,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "oracle_experiment": True,
        "factorized": False,
        "controls": explicit[0],
        "axes": {
            "detector": ["annotation_target_oracle"],
            "ownership": ["annotation_ownership_oracle"],
            "silhouette": ["annotation_interior_oracle"],
            "router": ["control_r0", "clean_route_oracle"],
            "expansion": ["raw", "bubble_interior"],
            "fill": list(fills),
        },
        "families": {
            "detector": {
                "annotation_target_oracle": {
                    "provider": "synthetic_ground_truth",
                    "seed_variant": "raw",
                    "pages": detector_pages,
                }
            },
            "ownership": {
                "annotation_ownership_oracle": {
                    "provider": "synthetic_ground_truth",
                    "pages": ownership_pages,
                }
            },
            "silhouette": {
                "annotation_interior_oracle": {
                    "provider": "synthetic_ground_truth",
                    "pages": silhouette_pages,
                }
            },
            "router": {
                "control_r0": {
                    "algorithm": "R0",
                    "pages": {page_id: {} for page_id in detector_pages},
                },
                "clean_route_oracle": {
                    "algorithm": "R3",
                    "pages": router_pages,
                },
            },
        },
        "oracle_only": [
            "annotation_target_oracle",
            "annotation_ownership_oracle",
            "annotation_interior_oracle",
            "clean_route_oracle",
        ],
        "explicit_combinations": explicit,
        "retain_page_artifacts": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build narrow and clean-bubble broad oracle fill comparisons."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build_fill_matrix(args.manifest.resolve())
    _write_json(args.output.resolve(), payload)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
