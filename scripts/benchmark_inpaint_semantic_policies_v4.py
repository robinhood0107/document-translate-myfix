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

from benchmarking.inpaint_detector_bakeoff.semantic import (  # noqa: E402
    REVIEW,
    TRANSLATE,
    SemanticDecision,
    consensus_decision,
    default_detector_decision,
    explicit_decision,
)
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-semantic-policy-v4"
CATEGORY = "40-inpaint-mask-render"
POLICIES = (
    "current_default",
    "detector_explicit_role",
    "ocr_semantic_hint",
    "explicit_role_consensus",
    "human_oracle",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be an object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decision(
    policy: str,
    instance: dict[str, Any],
    region: dict[str, Any],
) -> SemanticDecision:
    if policy == "current_default":
        return default_detector_decision(region)
    detector = explicit_decision(
        region,
        role_key="semantic_role",
        action_key="processing_action",
    )
    if policy == "detector_explicit_role":
        return detector
    ocr = explicit_decision(
        region,
        role_key="semantic_role_hint",
        action_key="processing_action_hint",
    )
    if policy == "ocr_semantic_hint":
        return ocr
    if policy == "explicit_role_consensus":
        return consensus_decision(detector, ocr)
    if policy == "human_oracle":
        return SemanticDecision(
            str(instance.get("semantic_role") or "ambiguous"),
            str(instance.get("processing_action") or REVIEW),
        )
    raise KeyError(policy)


def score_semantic_policies(manifest_path: Path) -> dict[str, object]:
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != "inpaint-factorized-source-manifest-v4":
        raise ValueError("semantic policy scoring requires manifest v4")
    pages = manifest.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("manifest v4 requires pages")
    output: list[dict[str, object]] = []
    for policy in POLICIES:
        total = role_exact = action_exact = 0
        required = required_translate = 0
        preserve = preserve_destructive = 0
        ambiguous = ambiguous_destructive = 0
        unavailable = 0
        reasons: dict[str, int] = {}
        for page in pages:
            if not isinstance(page, dict):
                raise ValueError("manifest page must be an object")
            raw_regions = page.get("regions")
            raw_instances = page.get("target_instances")
            if not isinstance(raw_regions, list) or not isinstance(raw_instances, list):
                raise ValueError("manifest page lacks regions or target_instances")
            regions = {
                str(region.get("region_id") or ""): region
                for region in raw_regions
                if isinstance(region, dict)
            }
            for raw_instance in raw_instances:
                if not isinstance(raw_instance, dict):
                    raise ValueError("target instance must be an object")
                region_id = str(raw_instance.get("region_id") or "")
                region = regions.get(region_id)
                if region is None:
                    raise ValueError(f"target instance references missing region: {region_id}")
                predicted = _decision(policy, raw_instance, region)
                truth_role = str(raw_instance.get("semantic_role") or "ambiguous")
                truth_action = str(raw_instance.get("processing_action") or REVIEW)
                priority = str(raw_instance.get("priority") or "ambiguous")
                total += 1
                role_exact += int(predicted.role == truth_role)
                action_exact += int(predicted.action == truth_action)
                unavailable += int(not predicted.available)
                if predicted.reason:
                    reasons[predicted.reason] = reasons.get(predicted.reason, 0) + 1
                if priority == "required":
                    required += 1
                    required_translate += int(predicted.action == TRANSLATE)
                elif priority == "optional":
                    preserve += 1
                    preserve_destructive += int(predicted.action == TRANSLATE)
                elif priority == "ambiguous":
                    ambiguous += 1
                    ambiguous_destructive += int(predicted.action == TRANSLATE)
        blocked = policy != "human_oracle" and unavailable > 0
        hard_pass = (
            not blocked
            and required_translate == required
            and preserve_destructive == 0
            and ambiguous_destructive == 0
        )
        output.append(
            {
                "policy_id": policy,
                "oracle_only": policy == "human_oracle",
                "status": (
                    "blocked_asset"
                    if blocked
                    else ("family_complete" if hard_pass else "dominated")
                ),
                "closure_reason": (
                    "semantic_evidence_missing"
                    if blocked
                    else ("" if hard_pass else "semantic_hard_gate_failed")
                ),
                "metrics": {
                    "instance_count": total,
                    "role_exact_accuracy": role_exact / total if total else None,
                    "action_exact_accuracy": action_exact / total if total else None,
                    "required_instance_count": required,
                    "required_translate_recall": (
                        required_translate / required if required else None
                    ),
                    "preserve_instance_count": preserve,
                    "preserve_destructive_count": preserve_destructive,
                    "ambiguous_instance_count": ambiguous,
                    "ambiguous_destructive_count": ambiguous_destructive,
                    "unavailable_instance_count": unavailable,
                    "reason_counts": dict(sorted(reasons.items())),
                },
            }
        )
    return {
        "schema_version": "inpaint-semantic-policy-results-v4",
        "manifest_sha256": _sha256(manifest_path),
        "policy_count": len(POLICIES),
        "unaccounted_policy_count": 0,
        "policies": output,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score source-only semantic routing policies on manifest v4."
    )
    parser.add_argument("--manifest", type=Path, required=True)
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
        payload = score_semantic_policies(args.manifest.resolve())
        result_path = output / "semantic-policy-results.json"
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if managed is not None:
            managed.complete(
                metadata={
                    "manifest_sha256": payload["manifest_sha256"],
                    "policy_count": payload["policy_count"],
                }
            )
            mismatches = managed.verify()
            if mismatches:
                raise RuntimeError("managed artifact verification failed: " + "; ".join(mismatches))
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
