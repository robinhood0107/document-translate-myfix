#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.validation_artifact_harness import select_managed_output_directory


FAMILY = "inpaint-balanced-preflight-v32"
CATEGORY = "40-inpaint-mask-render"


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


def _one(rows: object, field: str, value: str, label: str) -> Mapping[str, object]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} lacks records")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and str(row.get(field) or "") == value
    ]
    if len(matches) != 1:
        raise ValueError(f"{label} record must resolve exactly once: {value}")
    return matches[0]


def adjudicate_balanced_preflight(
    *,
    fusion_path: Path,
    fusion_run_id: str,
    semantic_path: Path,
    semantic_policy_id: str,
) -> dict[str, object]:
    fusion = _read_json(fusion_path)
    semantic = _read_json(semantic_path)
    if fusion.get("schema_version") != "inpaint-detector-fusion-results-v4":
        raise ValueError("balanced preflight requires detector fusion results v4")
    if semantic.get("schema_version") != "inpaint-semantic-policy-results-v4":
        raise ValueError("balanced preflight requires semantic policy results v4")
    manifest_sha = str(fusion.get("manifest_sha256") or "")
    if not manifest_sha or semantic.get("manifest_sha256") != manifest_sha:
        raise ValueError("balanced preflight artifacts use different manifests")
    fusion_run = _one(fusion.get("runs"), "run_id", fusion_run_id, "fusion")
    policy = _one(
        semantic.get("policies"),
        "policy_id",
        semantic_policy_id,
        "semantic policy",
    )
    fusion_metrics = fusion_run.get("metrics")
    semantic_metrics = policy.get("metrics")
    if not isinstance(fusion_metrics, Mapping) or not isinstance(
        semantic_metrics, Mapping
    ):
        raise ValueError("balanced preflight records lack metrics")
    failures: list[str] = []
    seed_admitted = fusion_run.get("seed_admitted") is True
    if not seed_admitted:
        failures.append("seed_not_admitted")
    strict_seed = (
        int(fusion_metrics.get("missed_target_instance_count", -1)) == 0
        and float(fusion_metrics.get("target_instance_seed_recall", -1.0)) >= 1.0
    )
    if policy.get("oracle_only") is True:
        failures.append("semantic_oracle_forbidden")
    if policy.get("status") in {"blocked_asset", "information_limited"}:
        failures.append("semantic_provider_unavailable")
    for field in (
        "preserve_destructive_count",
        "ambiguous_destructive_count",
        "no_edit_false_translate_page_count",
        "unavailable_instance_count",
    ):
        value = semantic_metrics.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value != 0:
            failures.append(f"semantic_gate_nonzero:{field}")
    required_recall = semantic_metrics.get("required_translate_recall")
    if not isinstance(required_recall, (int, float)) or isinstance(
        required_recall, bool
    ) or float(required_recall) <= 0.0:
        failures.append("semantic_required_recall_zero")
    return {
        "schema_version": "inpaint-balanced-preflight-adjudication-v32",
        "manifest_sha256": manifest_sha,
        "fusion_artifact_sha256": _sha256(fusion_path),
        "fusion_run_id": fusion_run_id,
        "semantic_artifact_sha256": _sha256(semantic_path),
        "semantic_provider": semantic_policy_id,
        "strict_seed_eligible": strict_seed,
        "seed_admitted": seed_admitted,
        "semantic_safe": not any(
            failure.startswith("semantic_") for failure in failures
        ),
        "balanced_candidate_admitted": not failures,
        "gate_failures": sorted(set(failures)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail closed before spending CUDA on a balanced v3.2 candidate."
    )
    parser.add_argument("--fusion-result", type=Path, required=True)
    parser.add_argument("--fusion-run", required=True)
    parser.add_argument("--semantic-result", type=Path, required=True)
    parser.add_argument("--semantic-policy", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    payload = adjudicate_balanced_preflight(
        fusion_path=args.fusion_result.resolve(),
        fusion_run_id=str(args.fusion_run),
        semantic_path=args.semantic_result.resolve(),
        semantic_policy_id=str(args.semantic_policy),
    )
    output = output_root / "balanced-preflight-adjudication.json"
    temporary = output.with_name(f".{output.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(output)
    if managed is not None:
        managed.complete(
            metadata={
                "manifest_sha256": payload["manifest_sha256"],
                "balanced_candidate_admitted": payload[
                    "balanced_candidate_admitted"
                ],
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
    return 0 if payload["balanced_candidate_admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
