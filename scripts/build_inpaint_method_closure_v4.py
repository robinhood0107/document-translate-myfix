#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.method_closure import (  # noqa: E402
    build_method_family_closure,
    evidence_from_records,
    requirements_from_registry,
)
from benchmarking.inpaint_detector_bakeoff.evidence_ledger import (  # noqa: E402
    registry_evidence_adapter_gaps,
    sha256_file,
)
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-method-family-closure-v4"
CATEGORY = "40-inpaint-mask-render"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def build_closure(registry_path: Path, evidence_path: Path) -> dict[str, object]:
    registry = _read_json(registry_path)
    if registry.get("schema_version") != "inpaint-method-family-registry-v4":
        raise ValueError("unsupported method family registry schema")
    evidence_payload = _read_json(evidence_path)
    if evidence_payload.get("schema_version") != "inpaint-method-family-evidence-v4":
        raise ValueError("unsupported method family evidence schema")
    records = evidence_payload.get("evidence")
    if not isinstance(records, list):
        raise ValueError("method evidence must contain an evidence list")
    if any(not isinstance(row, dict) for row in records):
        raise ValueError("method evidence rows must be objects")
    scope_manifests = evidence_payload.get("scope_manifests")
    if not isinstance(scope_manifests, dict):
        raise ValueError("method evidence must contain canonical scope_manifests")
    requirements = requirements_from_registry(registry)
    result = build_method_family_closure(
        requirements,
        evidence_from_records(records),
        scope_manifests=scope_manifests,
    )
    result["registry_sha256"] = sha256_file(registry_path)
    result["evidence_sha256"] = sha256_file(evidence_path)
    result["scope_manifests"] = scope_manifests
    gaps = registry_evidence_adapter_gaps(requirements)
    result["evidence_adapter_gap_count"] = len(gaps)
    result["evidence_adapter_gaps"] = list(gaps)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Account for every registered inpaint method-family variant."
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output.mkdir(parents=True, exist_ok=True)
    try:
        payload = build_closure(args.registry.resolve(), args.evidence.resolve())
        (output / "method-family-closure.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if managed is not None:
            managed.complete(
                metadata={
                    "family_count": payload["family_count"],
                    "required_variant_count": payload["required_variant_count"],
                    "unaccounted_variant_count": payload["unaccounted_variant_count"],
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
    except BaseException as error:
        if managed is not None:
            managed.fail(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
