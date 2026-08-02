from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


review = _load(
    "benchmark_translation_semantic_review_test",
    SCRIPTS / "benchmark_translation_semantic_review.py",
)


def _run(key: str, translations: list[str], *, finish_reason: str = "stop") -> dict[str, object]:
    rows = [
        {
            "choices": [
                {
                    "index": 0,
                    "content": json.dumps({"translation": text}, ensure_ascii=False),
                    "finish_reason": finish_reason,
                }
            ]
        }
        for text in translations
    ]
    return {
        "candidate": {"key": key, "cache_type_v": "f16"},
        "runtime": {"image_id": "sha256:test", "model": "model"},
        "request_ledger": {"sha256": "request-ledger"},
        "response_ledger": {"rows": rows, "sha256": review.canonical_sha256(rows)},
    }


def _approved_template(
    template: dict[str, object],
    *,
    page_checked: bool = False,
) -> dict[str, object]:
    approval = copy.deepcopy(template)
    approval["decision"] = "PASS"
    scope = review.TEXT_PLUS_PAGE_SCOPE if page_checked else review.TEXT_ONLY_SCOPE
    for comparison in approval["comparisons"]:
        comparison["reviewed_count"] = len(comparison["items"])
        comparison["unresolved_count"] = 0
        comparison["semantic_reject_count"] = 0
        for item in comparison["items"]:
            item.update(
                {
                    "decision": "PASS",
                    "review_scope": scope,
                    "page_checked": page_checked,
                    "requires_user_confirmation": False,
                    "classification": "style_or_register",
                    "rejection_category": "",
                    "attestations": {
                        name: True for name in review._PASS_ATTESTATIONS
                    },
                }
            )
    return approval


def test_raw_non_exact_difference_requires_private_semantic_review() -> None:
    binding = review.build_replay_comparison(
        baseline=_run("shipping-f16", ["기준", "동일"]),
        candidate=_run("candidate", ["후보", "동일"]),
    )

    result = review.evaluate_semantic_review(
        protocol_version="test-v2",
        stage="fixed-seed-structural",
        comparisons=[binding],
        approval=None,
    )

    assert result["status"] == "REVIEW_REQUIRED"
    assert result["mismatch_count"] == 1
    template = result["template"]
    assert template["review_method"] == review.TEXT_FIRST_REVIEW_METHOD
    assert template["comparisons"][0]["mismatch_indices"] == [0]
    assert "기준" not in json.dumps(template, ensure_ascii=False)
    assert "후보" not in json.dumps(template, ensure_ascii=False)


def test_text_only_semantic_pass_unlocks_raw_non_exact_difference() -> None:
    binding = review.build_replay_comparison(
        baseline=_run("shipping-f16", ["기준"]),
        candidate=_run("candidate", ["후보"]),
    )
    pending = review.evaluate_semantic_review(
        protocol_version="test-v2",
        stage="fixed-seed-structural",
        comparisons=[binding],
        approval=None,
    )

    approved = review.evaluate_semantic_review(
        protocol_version="test-v2",
        stage="fixed-seed-structural",
        comparisons=[binding],
        approval=_approved_template(pending["template"]),
    )

    assert approved["status"] == "PASS"
    assert approved["mismatch_count"] == 1


def test_target_page_is_optional_not_a_required_semantic_review_step() -> None:
    binding = review.build_replay_comparison(
        baseline=_run("shipping-f16", ["기준"]),
        candidate=_run("candidate", ["후보"]),
    )
    pending = review.evaluate_semantic_review(
        protocol_version="test-v2",
        stage="fixed-seed-structural",
        comparisons=[binding],
        approval=None,
    )

    approved = review.evaluate_semantic_review(
        protocol_version="test-v2",
        stage="fixed-seed-structural",
        comparisons=[binding],
        approval=_approved_template(pending["template"], page_checked=True),
    )

    assert approved["status"] == "PASS"


def test_ambiguous_item_stays_review_required_for_user_confirmation() -> None:
    binding = review.build_replay_comparison(
        baseline=_run("shipping-f16", ["기준"]),
        candidate=_run("candidate", ["후보"]),
    )
    pending = review.evaluate_semantic_review(
        protocol_version="test-v2",
        stage="fixed-seed-structural",
        comparisons=[binding],
        approval=None,
    )

    unresolved = review.evaluate_semantic_review(
        protocol_version="test-v2",
        stage="fixed-seed-structural",
        comparisons=[binding],
        approval=pending["template"],
    )

    assert unresolved["status"] == "REVIEW_REQUIRED"
    assert unresolved["unresolved_count"] == 1


@pytest.mark.parametrize(
    "category",
    [
        "censorship_or_deletion",
        "speaker_target_or_action_change",
        "relationship_or_number_change",
    ],
)
def test_censorship_deletion_or_semantic_reversal_is_rejected(category: str) -> None:
    binding = review.build_replay_comparison(
        baseline=_run("shipping-f16", ["기준"]),
        candidate=_run("candidate", ["후보"]),
    )
    pending = review.evaluate_semantic_review(
        protocol_version="test-v2",
        stage="fixed-seed-structural",
        comparisons=[binding],
        approval=None,
    )
    rejected = copy.deepcopy(pending["template"])
    rejected["decision"] = "REJECT"
    comparison = rejected["comparisons"][0]
    comparison["reviewed_count"] = 1
    comparison["unresolved_count"] = 0
    comparison["semantic_reject_count"] = 1
    comparison["items"][0].update(
        {
            "decision": "REJECT",
            "requires_user_confirmation": False,
            "rejection_category": category,
        }
    )

    result = review.evaluate_semantic_review(
        protocol_version="test-v2",
        stage="fixed-seed-structural",
        comparisons=[binding],
        approval=rejected,
    )

    assert result["status"] == "REJECT"
    assert result["semantic_reject_count"] == 1


def test_semantic_approval_rejects_missing_or_stale_review_rows() -> None:
    binding = review.build_replay_comparison(
        baseline=_run("shipping-f16", ["기준", "동일"]),
        candidate=_run("candidate", ["후보", "다름"]),
    )
    pending = review.evaluate_semantic_review(
        protocol_version="test-v2",
        stage="fixed-seed-structural",
        comparisons=[binding],
        approval=None,
    )
    approval = _approved_template(pending["template"])
    approval["comparisons"][0]["items"].pop()

    with pytest.raises(review.SemanticReviewError, match="mismatch indexes"):
        review.evaluate_semantic_review(
            protocol_version="test-v2",
            stage="fixed-seed-structural",
            comparisons=[binding],
            approval=approval,
        )


def test_semantic_pass_requires_all_sensitive_and_context_attestations() -> None:
    binding = review.build_replay_comparison(
        baseline=_run("shipping-f16", ["기준"]),
        candidate=_run("candidate", ["후보"]),
    )
    pending = review.evaluate_semantic_review(
        protocol_version="test-v2",
        stage="fixed-seed-structural",
        comparisons=[binding],
        approval=None,
    )
    approval = _approved_template(pending["template"])
    approval["comparisons"][0]["items"][0]["attestations"]["no_censorship"] = False

    with pytest.raises(review.SemanticReviewError, match="text-context attestation"):
        review.evaluate_semantic_review(
            protocol_version="test-v2",
            stage="fixed-seed-structural",
            comparisons=[binding],
            approval=approval,
        )


def test_hard_contract_failure_cannot_be_semantically_approved() -> None:
    with pytest.raises(review.SemanticReviewError, match="finish with stop"):
        review.build_replay_comparison(
            baseline=_run("shipping-f16", ["기준"]),
            candidate=_run("candidate", ["후보"], finish_reason="length"),
        )


def test_complete_json_after_a_reasoning_prefix_still_satisfies_schema_contract() -> None:
    response = {
        "choices": [
            {
                "index": 0,
                "content": "<reasoning>internal</reasoning>{\"translation\":\"후보\"}",
                "finish_reason": "stop",
            }
        ]
    }

    assert review.validate_translation_response(response, index=0).endswith("}")
