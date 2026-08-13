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
    ocr_provenance_decision,
)
from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    validate_source_only_manifest_v4,
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
    "ocr_provenance_verifier",
    "explicit_role_consensus",
    "human_oracle",
)


def aggregate_semantic_page_statistics(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    if not rows:
        raise ValueError("semantic aggregation requires page statistics")
    page_ids = [str(row.get("page_id") or "") for row in rows]
    if any(not value for value in page_ids) or len(page_ids) != len(set(page_ids)):
        raise ValueError("semantic page statistics require unique page IDs")
    decisions: list[dict[str, object]] = []
    no_edit_pages = no_edit_false_pages = no_edit_false_regions = 0
    for row in rows:
        values = row.get("decisions")
        if not isinstance(values, list) or any(
            not isinstance(value, dict) for value in values
        ):
            raise ValueError("semantic page statistics require decisions")
        decisions.extend(values)
        no_edit = row.get("no_edit")
        region_decisions = row.get("region_decisions")
        if not isinstance(no_edit, bool) or not isinstance(region_decisions, list) or any(
            not isinstance(value, dict) for value in region_decisions
        ):
            raise ValueError("semantic page statistics require region decisions")
        if no_edit:
            no_edit_pages += 1
            translated = sum(
                int(value.get("predicted_action") == TRANSLATE)
                for value in region_decisions
            )
            no_edit_false_regions += translated
            no_edit_false_pages += int(translated > 0)
    required = required_translate = preserve = preserve_destructive = 0
    ambiguous = ambiguous_destructive = unavailable = 0
    role_exact = action_exact = 0
    reasons: dict[str, int] = {}
    for decision in decisions:
        for field in (
            "instance_id",
            "truth_role",
            "truth_action",
            "priority",
            "predicted_role",
            "predicted_action",
        ):
            if not str(decision.get(field) or "").strip():
                raise ValueError(f"semantic decision lacks {field}")
        if not isinstance(decision.get("available"), bool):
            raise ValueError("semantic decision availability must be boolean")
        role_exact += int(decision["predicted_role"] == decision["truth_role"])
        action_exact += int(decision["predicted_action"] == decision["truth_action"])
        unavailable += int(not decision["available"])
        reason = str(decision.get("reason") or "")
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
        priority = str(decision["priority"])
        if priority == "required":
            required += 1
            required_translate += int(decision["predicted_action"] == TRANSLATE)
        elif priority == "optional":
            preserve += 1
            preserve_destructive += int(
                decision["predicted_action"] == TRANSLATE
            )
        elif priority == "ambiguous":
            ambiguous += 1
            ambiguous_destructive += int(
                decision["predicted_action"] == TRANSLATE
            )
        else:
            raise ValueError("semantic decision priority is invalid")
    total = len(decisions)
    return {
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
        "no_edit_page_count": no_edit_pages,
        "no_edit_false_translate_page_count": no_edit_false_pages,
        "no_edit_false_translate_region_count": no_edit_false_regions,
        "reason_counts": dict(sorted(reasons.items())),
    }


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


def _logical_inventory_sha256(policy_ids: tuple[str, ...]) -> str:
    encoded = json.dumps(
        sorted(policy_ids), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    if policy == "ocr_provenance_verifier":
        return ocr_provenance_decision(region)
    if policy == "explicit_role_consensus":
        return consensus_decision(detector, ocr)
    if policy == "human_oracle":
        return SemanticDecision(
            str(instance.get("semantic_role") or "ambiguous"),
            str(instance.get("processing_action") or REVIEW),
        )
    raise KeyError(policy)


def score_semantic_policies(manifest_path: Path) -> dict[str, object]:
    validate_source_only_manifest_v4(manifest_path)
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != "inpaint-factorized-source-manifest-v4":
        raise ValueError("semantic policy scoring requires manifest v4")
    pages = manifest.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("manifest v4 requires pages")
    page_ids = [
        str(page.get("page_id") or "")
        for page in pages
        if isinstance(page, dict)
    ]
    if len(page_ids) != len(pages) or any(not page_id for page_id in page_ids):
        raise ValueError("semantic manifest contains an invalid page ID")
    if len(page_ids) != len(set(page_ids)):
        raise ValueError("semantic manifest contains duplicate page IDs")
    output: list[dict[str, object]] = []
    page_statistics: dict[str, list[dict[str, object]]] = {}
    for policy in POLICIES:
        policy_pages: list[dict[str, object]] = []
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
            decisions: list[dict[str, object]] = []
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
                decisions.append(
                    {
                        "instance_id": str(raw_instance.get("instance_id") or ""),
                        "truth_role": truth_role,
                        "truth_action": truth_action,
                        "priority": priority,
                        "predicted_role": predicted.role,
                        "predicted_action": predicted.action,
                        "available": predicted.available,
                        "reason": predicted.reason,
                    }
                )
            required_regions = {
                str(value.get("region_id") or "")
                for value in raw_instances
                if isinstance(value, dict) and value.get("priority") == "required"
            }
            no_edit = not required_regions
            region_truth: dict[str, dict[str, Any]] = {}
            for value in raw_instances:
                if not isinstance(value, dict):
                    continue
                region_id = str(value.get("region_id") or "")
                current = region_truth.get(region_id)
                if current is None or value.get("priority") == "required":
                    region_truth[region_id] = value
            region_decisions: list[dict[str, object]] = []
            for region_id, region in sorted(regions.items()):
                instance = region_truth.get(region_id, {})
                predicted = _decision(policy, instance, region)
                region_decisions.append(
                    {
                        "region_id": region_id,
                        "predicted_role": predicted.role,
                        "predicted_action": predicted.action,
                        "available": predicted.available,
                        "reason": predicted.reason,
                    }
                )
            policy_pages.append(
                {
                    "page_id": str(page.get("page_id") or ""),
                    "no_edit": no_edit,
                    "decisions": decisions,
                    "region_decisions": region_decisions,
                }
            )
        metrics = aggregate_semantic_page_statistics(policy_pages)
        required = int(metrics["required_instance_count"])
        unavailable = int(metrics["unavailable_instance_count"])
        preserve_destructive = int(metrics["preserve_destructive_count"])
        ambiguous_destructive = int(metrics["ambiguous_destructive_count"])
        required_recall = metrics["required_translate_recall"]
        blocked = policy != "human_oracle" and unavailable > 0
        hard_pass = (
            not blocked
            and (required == 0 or required_recall == 1.0)
            and preserve_destructive == 0
            and ambiguous_destructive == 0
            and int(metrics["no_edit_false_translate_page_count"]) == 0
        )
        page_statistics[policy] = policy_pages
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
                "metrics": metrics,
            }
        )
    return {
        "schema_version": "inpaint-semantic-policy-results-v4",
        "manifest_sha256": _sha256(manifest_path),
        "page_ids": page_ids,
        "policy_count": len(POLICIES),
        "unaccounted_policy_count": 0,
        "logical_inventory_sha256": _logical_inventory_sha256(POLICIES),
        "policies": output,
        "pages": page_statistics,
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
