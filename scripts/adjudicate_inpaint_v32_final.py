#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-v32-final-adjudication"
CATEGORY = "40-inpaint-mask-render"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adjudicate_final_selection(
    *,
    balanced_preflight_path: Path,
    fill_preflight_path: Path,
) -> dict[str, object]:
    balanced = _read_json(balanced_preflight_path)
    fill = _read_json(fill_preflight_path)
    if balanced.get("schema_version") != "inpaint-balanced-preflight-adjudication-v32":
        raise ValueError("final selection requires balanced preflight v32")
    if fill.get("schema_version") != "inpaint-fill-preflight-adjudication-v32":
        raise ValueError("final selection requires fill preflight v32")
    balanced_manifest = str(balanced.get("manifest_sha256") or "")
    source_annotation_sha = str(
        fill.get("source_annotation_manifest_sha256") or ""
    )
    if not balanced_manifest or balanced_manifest != source_annotation_sha:
        raise ValueError("balanced and fill preflights use different E1 annotations")
    balanced_admitted = balanced.get("balanced_candidate_admitted") is True
    fill_admitted = fill.get("fill_candidate_admitted") is True
    if balanced_admitted:
        selected = "balanced"
    elif fill_admitted:
        selected = "fill_only"
    else:
        selected = "current_pr6"
    finalist_available = selected in {"balanced", "fill_only"}
    return {
        "schema_version": "inpaint-v32-final-adjudication-v1",
        "source_annotation_manifest_sha256": source_annotation_sha,
        "balanced_preflight_sha256": _sha256(balanced_preflight_path),
        "fill_preflight_sha256": _sha256(fill_preflight_path),
        "balanced_candidate_admitted": balanced_admitted,
        "fill_candidate_admitted": fill_admitted,
        "selected_candidate": selected,
        "relative_product_pass": finalist_available,
        "product_stack_action": (
            "promote_finalist" if finalist_available else "keep_current_pr6_draft"
        ),
        "onnx_promotion_authorized": selected == "balanced",
        "product_pr_rebase_authorized": finalist_available,
        "a5_authorized": finalist_available,
        "a5_state": "available_after_product_freeze" if finalist_available else "unavailable",
        "selection_reasons": {
            "balanced": list(balanced.get("gate_failures") or []),
            "fill_only": list(fill.get("gate_failures") or []),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply the automatic v3.2 B2, B1, then B0 selection rule."
    )
    parser.add_argument("--balanced-preflight", type=Path, required=True)
    parser.add_argument("--fill-preflight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    payload = adjudicate_final_selection(
        balanced_preflight_path=args.balanced_preflight.resolve(),
        fill_preflight_path=args.fill_preflight.resolve(),
    )
    output = output_root / "final-adjudication-v32.json"
    temporary = output.with_name(f".{output.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(output)
    if managed is not None:
        managed.complete(
            metadata={
                "selected_candidate": payload["selected_candidate"],
                "a5_authorized": payload["a5_authorized"],
            }
        )
        mismatches = managed.verify()
        if mismatches:
            raise RuntimeError(
                "managed artifact verification failed: " + "; ".join(mismatches)
            )
        print(managed.run_root)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
