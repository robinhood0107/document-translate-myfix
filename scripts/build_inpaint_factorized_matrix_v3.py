#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    load_stage1_manifest,
)


SCHEMA_VERSION = "inpaint-factorized-matrix-v3"
REGISTRY_SCHEMA_VERSION = "inpaint-factorized-family-registry-v3"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _artifact_path(template: object, page_id: str) -> str:
    if not isinstance(template, str) or not template.strip():
        raise ValueError("artifact template must be a non-empty string")
    return template.format(page_id=page_id)


def _require_file(path: str, *, family_id: str, page_id: str, key: str) -> str:
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(
            f"{family_id}/{page_id}/{key} artifact does not exist: {artifact}"
        )
    return str(artifact.resolve())


def _family_pages(
    family_id: str,
    spec: dict[str, Any],
    page_ids: tuple[str, ...],
    keys: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    templates = spec.get("templates")
    if not isinstance(templates, dict):
        raise ValueError(f"family {family_id} must contain templates")
    pages: dict[str, dict[str, str]] = {}
    for page_id in page_ids:
        page: dict[str, str] = {}
        for key in keys:
            template = templates.get(key)
            if template is None:
                if key in {"refined", "dilated"}:
                    template = templates.get("raw")
                elif key == "content_components":
                    template = templates.get("mask")
                elif key == "broad_mask":
                    template = templates.get("mask")
            path = _artifact_path(template, page_id)
            page[key] = _require_file(
                path, family_id=family_id, page_id=page_id, key=key
            )
        pages[page_id] = page
    return pages


def _manifest_control_families(
    manifest_path: Path,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    payload = _read_json(manifest_path)
    entries = {
        str(entry.get("page_id") or ""): entry
        for entry in payload.get("pages", [])
        if isinstance(entry, dict)
    }
    pages = load_stage1_manifest(manifest_path)
    page_ids = tuple(page.page_id for page in pages)
    detector_pages: dict[str, dict[str, Any]] = {}
    annotation_target_pages: dict[str, dict[str, Any]] = {}
    ownership_pages: dict[str, dict[str, Any]] = {}
    silhouette_pages: dict[str, dict[str, Any]] = {}
    annotation_silhouette_pages: dict[str, dict[str, Any]] = {}
    for page in pages:
        entry = entries[page.page_id]
        existing = (
            entry.get("existing_source_edit_mask")
            or entry.get("baseline_mask")
            or entry.get("claim_seed_mask")
        )
        if not isinstance(existing, str) or not existing:
            raise ValueError(f"page {page.page_id} has no control edit or seed mask")
        if not page.ownership_mask or not page.bubble_interior_mask:
            raise ValueError(f"page {page.page_id} has incomplete v3 control evidence")
        detector_pages[page.page_id] = {
            "raw": existing,
            "refined": existing,
            "dilated": existing,
        }
        annotation_target_pages[page.page_id] = {
            "raw": page.target_text_mask,
            "refined": page.target_text_mask,
            "dilated": page.target_text_mask,
        }
        ownership_pages[page.page_id] = {
            "mask": page.ownership_mask,
            "broad_mask": page.ownership_mask,
            "content_components": page.claim_seed_mask or page.ownership_mask,
        }
        silhouette_pages[page.page_id] = {"interior": None}
        annotation_silhouette_pages[page.page_id] = {
            "interior": page.bubble_interior_mask
        }
    return page_ids, {
        "detector": {
            "control_source_edit": {
                "seed_variant": "raw",
                "provider": "existing_source_edit",
                "pages": detector_pages,
            },
            "annotation_target_oracle": {
                "seed_variant": "raw",
                "provider": "source_only_annotation",
                "pages": annotation_target_pages,
            }
        },
        "ownership": {
            "control_text_prior": {
                "provider": "manifest_v3",
                "pages": ownership_pages,
            }
        },
        "silhouette": {
            "control_empty_silhouette": {
                "provider": "none",
                "pages": silhouette_pages,
            },
            "annotation_interior_oracle": {
                "provider": "source_only_annotation",
                "pages": annotation_silhouette_pages,
            }
        },
        "router": {
            "control_r0": {
                "algorithm": "R0",
                "pages": {page_id: {} for page_id in page_ids},
            }
        },
    }


def build_matrix(manifest_path: Path, registry_path: Path) -> dict[str, Any]:
    page_ids, families = _manifest_control_families(manifest_path)
    registry = _read_json(registry_path)
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ValueError("unsupported factorized family registry schema")
    source_families = registry.get("families")
    if not isinstance(source_families, dict):
        raise ValueError("family registry must contain families")
    definitions = {
        "detector": ("raw", "refined", "dilated"),
        "ownership": ("mask", "content_components", "broad_mask"),
        "silhouette": ("interior",),
    }
    for role, keys in definitions.items():
        role_specs = source_families.get(role, {})
        if not isinstance(role_specs, dict):
            raise ValueError(f"registry role {role} must be an object")
        target = families[role]
        for family_id, raw_spec in role_specs.items():
            if family_id in target:
                raise ValueError(f"duplicate family id: {family_id}")
            if not isinstance(raw_spec, dict):
                raise ValueError(f"family {family_id} must be an object")
            spec = {key: value for key, value in raw_spec.items() if key != "templates"}
            spec["pages"] = _family_pages(family_id, raw_spec, page_ids, keys)
            target[family_id] = spec
    router_specs = source_families.get("router", {})
    if not isinstance(router_specs, dict):
        raise ValueError("registry role router must be an object")
    for family_id, raw_spec in router_specs.items():
        if not isinstance(raw_spec, dict):
            raise ValueError(f"router {family_id} must be an object")
        evidence: dict[str, Any] = {}
        evidence_path = raw_spec.get("evidence")
        if isinstance(evidence_path, str) and evidence_path:
            evidence_payload = _read_json(Path(evidence_path))
            raw_pages = evidence_payload.get("pages", evidence_payload)
            if not isinstance(raw_pages, dict):
                raise ValueError(f"router evidence {family_id} must contain pages")
            evidence = raw_pages
        pages = {}
        for page_id in page_ids:
            value = evidence.get(page_id, {})
            if not isinstance(value, dict):
                raise ValueError(f"router evidence {family_id}/{page_id} must be an object")
            pages[page_id] = value
        families["router"][family_id] = {
            key: value for key, value in raw_spec.items() if key != "evidence"
        }
        families["router"][family_id]["pages"] = pages
    settings = registry.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("registry settings must be an object")
    controls = {
        "detector": "control_source_edit",
        "ownership": "control_text_prior",
        "silhouette": "control_empty_silhouette",
        "router": "control_r0",
        "expansion": "raw",
        "fill": "mask_only",
    }
    explicit = list(registry.get("explicit_combinations", []))
    compatible_sets = registry.get("compatible_sets", [])
    if not isinstance(compatible_sets, list):
        raise ValueError("compatible_sets must be a list")
    for compatible in compatible_sets:
        if not isinstance(compatible, dict):
            raise ValueError("compatible set must be an object")
        base = compatible.get("base", {})
        vary = compatible.get("vary", {})
        if not isinstance(base, dict) or not isinstance(vary, dict):
            raise ValueError("compatible set base and vary must be objects")
        roles = tuple(vary)
        values = []
        for role in roles:
            choices = vary[role]
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"compatible set role {role} needs choices")
            values.append(choices)
        for selection in product(*values):
            explicit.append(
                {
                    **base,
                    **dict(zip(roles, selection)),
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest": str(manifest_path.resolve()),
        "controls": controls,
        "axes": {
            "detector": list(families["detector"]),
            "ownership": list(families["ownership"]),
            "silhouette": list(families["silhouette"]),
            "router": list(families["router"]),
            "expansion": list(registry.get("axes", {}).get("expansion", ["raw"])),
            "fill": list(registry.get("axes", {}).get("fill", ["mask_only"])),
        },
        "families": families,
        "oracle_only": [
            "annotation_target_oracle",
            "annotation_interior_oracle",
            *list(registry.get("oracle_only", [])),
        ],
        "factorized": bool(settings.get("factorized", True)),
        "explicit_combinations": explicit,
        "retain_page_artifacts": bool(settings.get("retain_page_artifacts", False)),
        "retain_required_page_artifacts_only": bool(
            settings.get("retain_required_page_artifacts_only", False)
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a validated v3 detector/ownership/silhouette matrix."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_matrix(args.manifest.resolve(), args.registry.resolve())
    _write_json(args.output.resolve(), payload)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
