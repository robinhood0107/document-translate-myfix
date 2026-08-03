from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import benchmark_gemma_sampler_quality_v2 as sampler_cli  # noqa: E402
from scripts import validation_artifact_harness as harness  # noqa: E402
from scripts.gemma_sampler_quality_v2 import (  # noqa: E402
    campaign,
    corpus,
    execution,
    final_analysis,
    incremental,
    judgment,
    protocol,
    report,
    runtime,
    storage,
)
from scripts.gemma_sampler_quality_v2.storage import RunStore, read_json  # noqa: E402
from scripts.gemma_sampler_quality_v2.review import render_private_review_html  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _source_manifest(archive: Path) -> Path:
    snapshot = archive / "snapshots" / "safe-snapshot.json"
    _write_json(
        snapshot,
        {
            "pages": [
                {
                    "blocks": [
                        {"text": "source-a", "translation": "old-a"},
                        {"text": "context-a"},
                        {"text": "source-a", "translation": "old-b"},
                        {"text": "context-a"},
                    ]
                }
            ]
        },
    )
    manifest = archive / "source-manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": corpus.SOURCE_MANIFEST_SCHEMA_VERSION,
            "sources": [
                {
                    "source_id": "synthetic-source",
                    "snapshot_path": "snapshots/safe-snapshot.json",
                    "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                    "language": "ja-ko",
                    "expected_occurrences": 4,
                }
            ],
        },
    )
    return manifest


def _canonical_answers(reference: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(case["case_id"]): {
            "canonical_translation": f"translation-{index}",
            "required_meaning": ["meaning"],
            "prohibited_changes": ["number_change"],
            "confidence": "high",
            "terminology_basis": [],
            "acceptable_alternatives": [],
        }
        for index, case in enumerate(reference["cases"])
    }


def test_corpus_identity_uses_source_and_next_adjacent_context_only(tmp_path: Path) -> None:
    archive = tmp_path / "private-archive"
    archive.mkdir()
    draft = corpus.build_reference_draft(
        _source_manifest(archive),
        archive_root=archive,
        strict_counts=False,
    )

    matching = [
        case
        for case in draft["cases"]
        if case["source_text"] == "source-a" and case["context_after_text"] == "context-a"
    ]
    assert len(matching) == 1
    assert len(matching[0]["provenance"]) == 2
    assert matching[0]["canonical_translation"] == ""
    assert matching[0]["provenance"][0]["prior_outputs"]


def test_blind_review_hides_first_answer_and_flags_every_difference(tmp_path: Path) -> None:
    archive = tmp_path / "private-archive"
    archive.mkdir()
    draft = corpus.build_reference_draft(
        _source_manifest(archive),
        archive_root=archive,
        strict_counts=False,
    )
    canonical = corpus.apply_canonical_answers(draft, _canonical_answers(draft))
    packet = corpus.build_blind_review_packet(canonical)
    serialized = json.dumps(packet, ensure_ascii=False)
    assert "canonical_translation" not in serialized
    decisions = {
        row["blind_id"]: {
            "independent_translation": "different-second-pass",
            "required_meaning": ["meaning"],
            "confidence": "high",
            "ocr_damaged": False,
        }
        for row in packet["rows"]
    }
    resolved = corpus.apply_blind_review(canonical, packet, decisions)
    assert all("BLIND_TRANSLATION_DIFFERENCE" in case["flags"] for case in resolved["cases"])
    assert corpus.reference_summary(resolved)["flagged_case_count"] == len(resolved["cases"])


def test_matrix_counts_exclude_temperature_zero_and_reuse_existing_rows() -> None:
    assert protocol.expected_response_counts() == {
        "temperature": 9560,
        "joint_top_p_top_k": 55448,
        "min_p": 11472,
        "total_new": 76480,
    }
    assert all(arm.sampler.temperature > 0.0 for arm in protocol.temperature_arms())
    joint = protocol.joint_top_p_top_k_arms((0.4, 0.7))
    assert len(joint) == 60
    assert len(protocol.new_arms(joint)) == 58
    min_p = protocol.min_p_arms(
        (
            protocol.SamplerTuple(0.4, 0.95, 64, 0.0),
            protocol.SamplerTuple(0.7, 0.90, 128, 0.0),
            protocol.SamplerTuple(0.7, 1.00, 0, 0.0),
        )
    )
    assert len(min_p) == 15
    assert len(protocol.new_arms(min_p)) == 12
    assert protocol.filters_disabled(top_k=0, top_p=1.0)
    with pytest.raises(protocol.ProtocolError):
        protocol.SamplerTuple(0.0, 0.95, 64, 0.0)


def test_single_campaign_matrix_is_fixed_and_reuses_only_approved_rows() -> None:
    plan = protocol.campaign_plan()

    assert len(plan.temperature_arms) == 10
    assert len(plan.joint_arms) == 120
    assert len(plan.new_joint_arms) == 110
    assert len(plan.min_p_arms) == 30
    assert len(plan.new_min_p_arms) == 20
    assert len(plan.sampler_keys) == 140
    assert all(
        arm.sampler.top_p == 1.0 and arm.sampler.top_k == 0 and arm.sampler.min_p == 0.0
        for arm in plan.joint_arms[:10]
    )
    assert [arm.sampler.temperature for arm in plan.joint_arms[:10]] == list(protocol.TEMPERATURE_VALUES)
    assert protocol.campaign_response_counts() == {
        "r6_reused": 9560,
        "joint_new": 105160,
        "min_p_new": 19120,
        "total_new": 124280,
        "total_evidence": 133840,
    }
    assert plan.sha256 == protocol.campaign_plan().sha256


def test_atomic_write_retries_a_transient_permission_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    destination = tmp_path / "progress.json"
    original_replace = storage.os.replace
    attempts = 0

    def replace_after_one_lock(source: str | Path, target: str | Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("reader still holds the old progress file")
        original_replace(source, target)

    monkeypatch.setattr(storage.os, "replace", replace_after_one_lock)
    monkeypatch.setattr(storage.time, "sleep", lambda _seconds: None)

    storage.atomic_write_json(destination, {"state": "RUNNING"})

    assert attempts == 2
    assert read_json(destination) == {"state": "RUNNING"}
    assert not list(tmp_path.glob(".progress.json.partial-*"))


def test_atomic_write_survives_a_windows_reader_without_delete_share(tmp_path: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows file-share semantics are required for this regression test.")
    destination = tmp_path / "progress.json"
    destination.write_text('{"state":"old"}\n', encoding="utf-8")
    lock_script = (
        "$lock = [System.IO.File]::Open($env:GEMMA_MONITOR_LOCK_TARGET, [System.IO.FileMode]::Open, "
        "[System.IO.FileAccess]::Read, [System.IO.FileShare]::Read); "
        "[Console]::Out.WriteLine('LOCKED'); Start-Sleep -Milliseconds 700; $lock.Dispose()"
    )
    child_environment = dict(os.environ)
    child_environment["GEMMA_MONITOR_LOCK_TARGET"] = str(destination)
    process = subprocess.Popen(
        ["powershell.exe", "-NoProfile", "-Command", lock_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=child_environment,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "LOCKED"
        started = time.monotonic()
        storage.atomic_write_json(destination, {"state": "RUNNING"})
        elapsed = time.monotonic() - started
        _stdout, stderr = process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    assert process.returncode == 0, stderr
    assert elapsed >= 0.3
    assert read_json(destination) == {"state": "RUNNING"}


def test_run_phase_reopens_only_the_known_progress_replace_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "banchmark_result_log"
    archive_root.mkdir()
    monkeypatch.setattr(harness, "default_archive_root", lambda: archive_root)
    monkeypatch.setattr(harness, "_is_ignored_by_git", lambda _path: True)
    run = harness.ManagedArtifactRun.create(
        family=sampler_cli.FAMILY,
        category=sampler_cli.CATEGORY,
        run_id="recover-known-progress-lock",
    )
    run.fail(
        PermissionError(
            "[WinError 5] Access is denied: "
            "'C:\\private\\run\\.progress.json.partial-1-a' -> "
            "'C:\\private\\run\\progress.json'"
        ),
        metadata={"command": "run-phase"},
    )

    reopened = sampler_cli._open_run(
        SimpleNamespace(resume_run=str(run.run_root), phase="temperature")
    )

    manifest = json.loads(reopened.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "running"
    assert manifest["metadata"]["state"] == "RECOVERED_ATOMIC_REPLACE"


def test_pinned_filter_contract_requires_image_build_and_exact_payload() -> None:
    protocol.assert_pinned_sampler_contract(
        image_ref=protocol.PINNED_LLAMA_CPP_IMAGE,
        binary_version="llama.cpp b10133",
        payload={"top_k": 0, "top_p": 1.0},
    )
    protocol.assert_pinned_sampler_contract(
        image_ref=protocol.PINNED_LLAMA_CPP_IMAGE,
        binary_version="version: 10133 (ff067f76d)",
        payload={"top_k": 0, "top_p": 1.0},
    )
    with pytest.raises(protocol.ProtocolError):
        protocol.assert_pinned_sampler_contract(
            image_ref="other",
            binary_version="llama.cpp b10133",
            payload={"top_k": 0, "top_p": 1.0},
        )
    with pytest.raises(protocol.ProtocolError, match="binary revision"):
        protocol.assert_pinned_sampler_contract(
            image_ref=protocol.PINNED_LLAMA_CPP_IMAGE,
            binary_version="version: 10134 (ff067f76d)",
            payload={"top_k": 0, "top_p": 1.0},
        )


@pytest.mark.parametrize(
    ("content", "category"),
    [
        ('{"translation":"나Please세"}', "mixed_token_corruption"),
        ('{"translation":"나please세"}', "mixed_token_corruption"),
        ('{"translation":"나I세"}', "mixed_token_corruption"),
        ('{"translation":"   "}', "censorship_or_deletion"),
    ],
)
def test_visible_translation_damage_remains_catastrophic(content: str, category: str) -> None:
    verdict = judgment.validate_response_envelope(
        {"choices": [{"index": 0, "content": content, "finish_reason": "stop"}]}
    )
    assert verdict.status == "CATASTROPHIC"
    assert verdict.category == category


def test_channel_framing_and_hidden_thought_are_ignored_for_translation_quality() -> None:
    content = '<|CHANNEL>THOUGHT\nactual hidden thought\n<channel|>{"translation":"<|channel>thought\\ninner thought\\n<channel|>normal"} trailing'
    cleaned, sanitized = judgment._quality_channel_sanitize(content)
    verdict = judgment.validate_response_envelope(
        {"choices": [{"index": 0, "message": {"content": content}, "finish_reason": "stop"}]}
    )

    assert sanitized is True
    assert cleaned == '{"translation":"normal"} trailing'
    assert verdict.status == "VALID"
    assert verdict.translation == "normal"
    assert verdict.payload()["schema_version"] == judgment.RESPONSE_VALIDATION_SCHEMA_VERSION
    assert verdict.payload()["sanitized_channel_tokens"] is True
    assert "non_translation_envelope" in verdict.payload()["transport_diagnostics"]


def test_channel_payload_inside_json_is_not_counted_as_visible_translation_damage() -> None:
    content = '{"translation":"<|channel>thought\\nactual visible text\\n<channel|>normal"}'
    verdict = judgment.validate_response_envelope(
        {"choices": [{"index": 0, "content": content, "finish_reason": "stop"}]}
    )

    assert verdict.status == "VALID"
    assert verdict.translation == "normal"
    assert verdict.payload()["sanitized_channel_tokens"] is True


@pytest.mark.parametrize(
    "content",
    [
        '<|channel>thought\nactual thought text\n<channel|>{"translation":"normal"}',
        '<|CHANNEL>THOUGHT\n<channel|>{"translation":"normal"}',
        '{"translation":"normal","extra":"x"}',
        '{"translation":"normal"} trailing text',
    ],
)
def test_transport_shape_does_not_change_extractable_translation_quality(content: str) -> None:
    verdict = judgment.validate_response_envelope(
        {"choices": [{"index": 0, "content": content, "finish_reason": "length"}]}
    )

    assert verdict.status == "VALID"
    assert verdict.translation == "normal"
    assert "finish_reason" in verdict.payload()["transport_diagnostics"]


def test_missing_translation_is_unjudged_not_a_semantic_failure() -> None:
    verdict = judgment.validate_response_envelope(
        {"choices": [{"index": 0, "content": "analysis only", "finish_reason": "stop"}]}
    )

    assert verdict.status == "UNJUDGED"
    assert verdict.category == "translation_unavailable"


def test_unjudged_translation_blocks_automatic_tuning_winner() -> None:
    assert sampler_cli._rank_outcome(
        "tuning",
        [{"sampler_key": "t0.70-p0.95-k64-m0.00", "unjudged": 1}],
    ) == {"status": "INSUFFICIENT_TRANSLATION_EVIDENCE"}
    assert sampler_cli._rank_outcome(
        "tuning",
        [{"sampler_key": "t0.70-p0.95-k64-m0.00", "unjudged": 0}],
    ) == {
        "status": "PROVISIONAL_WINNER",
        "provisional_sampler_key": "t0.70-p0.95-k64-m0.00",
    }


def test_blind_judgment_requires_current_in_memory_quality_view() -> None:
    reference = {
        "cases": [
            {
                "case_id": "case-safe",
                "split": "tuning",
                "language": "ja-ko",
                "source_text": "source",
                "context_after_text": "context",
                "canonical_translation": "candidate",
                "required_meaning": ["meaning"],
                "prohibited_changes": ["number_change"],
            }
        ]
    }
    stale_validation = {
        "status": "VALID",
        "category": "",
        "translation_sha256": hashlib.sha256(b"candidate").hexdigest(),
        "message": "",
    }
    records = [
        {
            "case_id": "case-safe",
            "split": "tuning",
            "logical_slot": "slot-safe",
            "sampler": protocol.SamplerTuple(0.7, 0.95, 64, 0.0).payload(),
            "response_validation": stale_validation,
            "translation": "candidate",
        }
    ]

    with pytest.raises(judgment.JudgmentError, match="obsolete contract"):
        judgment.build_blind_judgment_packet(reference, records)


def test_legacy_raw_records_are_rejudged_without_mutating_the_private_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution, "CORPUS_CASE_COUNT", 1)
    case = {
        "case_id": "case-safe",
        "split": "tuning",
        "language": "ja-ko",
        "source_text": "source",
        "context_after_text": "context",
        "canonical_translation": "candidate",
        "required_meaning": ["meaning"],
        "prohibited_changes": ["number_change"],
        "review_status": "APPROVED",
    }
    reference = {
        "schema_version": corpus.REFERENCE_SCHEMA_VERSION,
        "state": "FROZEN",
        "case_identity": "language+source_text+context_after_text",
        "cases": [case],
    }
    reference["reference_sha256"] = protocol.canonical_sha256(
        {
            "schema_version": reference["schema_version"],
            "case_identity": reference["case_identity"],
            "cases": reference["cases"],
        }
    )
    legacy_root = tmp_path / "legacy-artifacts"
    legacy_root.mkdir()
    legacy_store = RunStore(legacy_root)
    assert legacy_store.record_case_if_first(
        phase="temperature",
        arm="temperature-t0.70-p0.95-k64-m0.00",
        run="seed-20260802-forward",
        case_id="case-safe",
        logical_slot="legacy-slot",
        payload={
            "status": "complete",
            "recorded_utc": "2026-08-03T00:00:00Z",
            "phase": "temperature",
            "arm_key": "temperature-t0.70-p0.95-k64-m0.00",
            "case_id": "case-safe",
            "logical_slot": "legacy-slot",
            "split": "tuning",
            "reference_sha256": reference["reference_sha256"],
            "runtime_fingerprint": "router-fingerprint",
            "request_identity": {"case_id": "case-safe"},
            "response_validation": {"schema_version": "gemma-sampler-response-validation-v2"},
            "translation": "old value",
            "response": {
                "choices": [
                    {
                        "index": 0,
                        "content": '<|channel>thought\nprivate wrapper\n<channel|>{"translation":"candidate"} trailing',
                        "finish_reason": "length",
                    }
                ]
            },
        },
    )
    case_path = legacy_root / next(iter(legacy_store.completed_index().values()))["path"]
    before = case_path.read_bytes()

    records = list(execution.iter_compatible_completed_records((RunStore(legacy_root),), reference=reference))

    assert case_path.read_bytes() == before
    assert records[0]["translation"] == "candidate"
    assert records[0]["response_validation"]["schema_version"] == judgment.RESPONSE_VALIDATION_SCHEMA_VERSION


def test_blind_cluster_judgment_propagates_to_each_actual_response() -> None:
    reference = {
        "cases": [
            {
                "case_id": "case-safe",
                "split": "tuning",
                "language": "ja-ko",
                "source_text": "source",
                "context_after_text": "context",
                "canonical_translation": "canonical",
                "required_meaning": ["meaning"],
                "prohibited_changes": ["number_change"],
            }
        ]
    }
    raw = {"choices": [{"index": 0, "content": '{"translation":"candidate"}', "finish_reason": "stop"}]}
    response_verdict = judgment.validate_response_envelope(raw)
    records = [
        {
            "case_id": "case-safe",
            "split": "tuning",
            "logical_slot": f"slot-{index}",
            "sampler": protocol.SamplerTuple(0.7, 0.95, 64, 0.0).payload(),
            "response_validation": response_verdict.payload(),
            "translation": response_verdict.translation,
            "latency_ms": 10.0,
            "completion_tokens": 4,
        }
        for index in range(2)
    ]
    packet = judgment.build_blind_judgment_packet(reference, records)
    assert packet["arm_and_seed_hidden"] is True
    assert len(packet["rows"]) == 1
    cluster = packet["rows"][0]["cluster_id"]
    decisions = {
        cluster: {"decision": "MAJOR", "category": "semantic_change", "naturalness": 2}
    }
    verdicts = judgment.bind_cluster_verdicts_to_records(records, packet, decisions)
    assert {verdicts["slot-0"]["decision"], verdicts["slot-1"]["decision"]} == {"MAJOR"}
    ranked = judgment.rank_sampler_results(records, verdicts)
    assert ranked[0]["major"] == 2
    assert ranked[0]["unique_error_cases"] == 1


def _incremental_fixture() -> tuple[dict[str, object], list[dict[str, object]]]:
    cases = [
        {
            "case_id": "case-tuning",
            "split": "tuning",
            "language": "ja-ko",
            "source_text": "source tuning",
            "context_after_text": "context tuning",
            "canonical_translation": "정답 A",
            "required_meaning": ["meaning A"],
            "prohibited_changes": ["number_change"],
        },
        {
            "case_id": "case-holdout",
            "split": "holdout",
            "language": "en-ko",
            "source_text": "source holdout",
            "context_after_text": "context holdout",
            "canonical_translation": "정답 B",
            "required_meaning": ["meaning B"],
            "prohibited_changes": ["identity_change"],
        },
    ]
    reference: dict[str, object] = {
        "reference_sha256": "a" * 64,
        "cases": cases,
    }
    records: list[dict[str, object]] = []
    for index, (case, translation) in enumerate(zip(cases, ("후보 A", "후보 B"), strict=True)):
        verdict = judgment.validate_response_envelope(
            {"choices": [{"index": 0, "content": json.dumps({"translation": translation}, ensure_ascii=False), "finish_reason": "stop"}]}
        )
        records.append(
            {
                "case_id": case["case_id"],
                "split": case["split"],
                "logical_slot": f"slot-{index}",
                "sampler": protocol.SamplerTuple(0.3, 0.95, 64, 0.0).payload(),
                "response_validation": verdict.payload(),
                "translation": verdict.translation,
                "latency_ms": 10.0 + index,
                "completion_tokens": 4 + index,
            }
        )
    return reference, records


def test_incremental_blind_batches_cover_all_478_splits_and_reuse_stable_clusters() -> None:
    reference, records = _incremental_fixture()
    ledger = incremental.new_incremental_ledger(reference)
    packet = incremental.build_incremental_judgment_packet(
        reference,
        records,
        ledger,
        batch_size=1,
    )

    assert packet["scope"] == "all-478"
    assert packet["pending_total_cluster_count"] == 2
    assert packet["batch_cluster_count"] == 1
    serialized = json.dumps(packet, ensure_ascii=False)
    assert "logical_slot" not in serialized
    assert "sampler_key" not in serialized
    assert "top_p" not in serialized

    cluster_id = packet["rows"][0]["cluster_id"]
    ledger = incremental.mark_pending_batch(
        ledger,
        reference=reference,
        packet=packet,
        packet_file="judgment-batch-0001.json",
    )
    ledger = incremental.apply_incremental_judgments(
        ledger,
        reference=reference,
        packet=packet,
        decisions={
            cluster_id: {
                "decision": "PASS",
                "category": "faithful_translation",
                "naturalness": 5,
            }
        },
        applied_utc="2026-08-04T00:00:00Z",
    )
    refreshed = incremental.build_incremental_judgment_packet(
        reference,
        records,
        ledger,
        batch_size=10,
    )

    assert refreshed["reused_judged_cluster_count"] == 1
    assert refreshed["pending_total_cluster_count"] == 1
    assert refreshed["rows"][0]["cluster_id"] != cluster_id


def test_incremental_ledger_imports_prior_blind_decisions_and_final_rank_uses_all_cases() -> None:
    reference, records = _incremental_fixture()
    prior_packet = judgment.build_blind_judgment_packet(reference, records[:1], scope="tuning")
    prior_cluster = prior_packet["rows"][0]["cluster_id"]
    ledger = incremental.seed_ledger_from_completed_packet(
        incremental.new_incremental_ledger(reference),
        reference=reference,
        packet=prior_packet,
        decisions={
            prior_cluster: {
                "decision": "PASS",
                "category": "faithful_translation",
                "naturalness": 5,
            }
        },
        source_label="prior-temperature",
    )
    packet = incremental.build_incremental_judgment_packet(reference, records, ledger, batch_size=10)
    remaining_cluster = packet["rows"][0]["cluster_id"]
    ledger = incremental.mark_pending_batch(
        ledger,
        reference=reference,
        packet=packet,
        packet_file="judgment-batch-0001.json",
    )
    ledger = incremental.apply_incremental_judgments(
        ledger,
        reference=reference,
        packet=packet,
        decisions={
            remaining_cluster: {
                "decision": "MINOR",
                "category": "naturalness_grammar",
                "naturalness": 3,
            }
        },
        applied_utc="2026-08-04T00:01:00Z",
    )
    verdicts = incremental.bind_incremental_ledger_to_records(reference, records, ledger)
    ranked = judgment.rank_sampler_results(records, verdicts, scope="all")

    assert len(ledger["imports"]) == 1
    assert ranked[0]["response_count"] == 2
    assert ranked[0]["minor"] == 1
    assert ranked[0]["unique_error_cases"] == 1


def test_incremental_ledger_rejects_reusable_packet_from_changed_reference() -> None:
    reference, records = _incremental_fixture()
    prior_packet = judgment.build_blind_judgment_packet(reference, records[:1], scope="tuning")
    prior_cluster = prior_packet["rows"][0]["cluster_id"]
    prior_packet["rows"][0]["canonical_translation"] = "바뀐 정답"

    with pytest.raises(judgment.JudgmentError, match="does not match the frozen reference"):
        incremental.seed_ledger_from_completed_packet(
            incremental.new_incremental_ledger(reference),
            reference=reference,
            packet=prior_packet,
            decisions={
                prior_cluster: {
                    "decision": "PASS",
                    "category": "faithful_translation",
                    "naturalness": 5,
                }
            },
            source_label="different-reference",
        )


def test_incremental_ledger_fails_closed_when_semantic_rule_version_changes() -> None:
    reference, _records = _incremental_fixture()
    ledger = incremental.new_incremental_ledger(reference)
    ledger["rule_version"] = "different-rule"

    with pytest.raises(judgment.JudgmentError, match="different semantic rule"):
        incremental.validate_incremental_ledger(ledger, reference=reference)


def test_incremental_packet_checks_identity_keys_without_rejecting_candidate_words() -> None:
    reference, records = _incremental_fixture()
    response_verdict = judgment.validate_response_envelope(
        {
            "choices": [
                {
                    "index": 0,
                    "content": json.dumps(
                        {"translation": "logical_slot sampler_key는 번역문의 일반 문자열이다"},
                        ensure_ascii=False,
                    ),
                    "finish_reason": "stop",
                }
            ]
        }
    )
    records[0]["translation"] = response_verdict.translation
    records[0]["response_validation"] = response_verdict.payload()
    packet = incremental.build_incremental_judgment_packet(
        reference,
        records,
        incremental.new_incremental_ledger(reference),
        batch_size=10,
    )

    incremental.validate_incremental_packet(packet, reference=reference)

    packet["rows"][0]["sampler_key"] = "secret-arm"
    packet["packet_sha256"] = protocol.canonical_sha256(
        {key: value for key, value in packet.items() if key != "packet_sha256"}
    )
    with pytest.raises(judgment.JudgmentError, match="leaked sampler execution identity"):
        incremental.validate_incremental_packet(packet, reference=reference)


def test_incremental_command_error_preserves_running_run_for_resume() -> None:
    checkpoints: list[dict[str, object]] = []

    class FakeRun:
        def checkpoint(self, *, metadata: dict[str, object]) -> None:
            checkpoints.append(metadata)

        def fail(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("incremental errors must not close the managed run")

    sampler_cli._checkpoint_incremental_error(
        FakeRun(),  # type: ignore[arg-type]
        command="apply-incremental-judgment",
        error=RuntimeError("checkpoint race"),
    )

    assert checkpoints == [
        {
            "command": "apply-incremental-judgment",
            "state": "ERROR_REQUIRES_INSPECTION",
            "error_type": "RuntimeError",
            "error_message": "checkpoint race",
        }
    ]


def test_live_snapshot_index_does_not_recover_or_rewrite_in_flight_case(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    store = storage.RunStore(root)
    destination = store.case_path(
        phase="joint_top_p_top_k",
        arm="joint-arm",
        run="seed-run",
        case_id="case-safe",
    )
    storage.atomic_write_json(
        destination,
        {
            "status": "complete",
            "recorded_utc": "2026-08-04T00:00:00Z",
            "phase": "joint_top_p_top_k",
            "arm_key": "joint-arm",
            "case_id": "case-safe",
            "logical_slot": "slot-safe",
        },
    )

    assert list(store.iter_snapshot_completion_entries()) == []
    assert not store.completion_index_path.exists()
    assert store.completed_count() == 1
    assert store.completion_index_path.exists()


def _final_analysis_fixture() -> tuple[
    dict[str, object],
    protocol.CampaignPlan,
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    cases = [
        {
            "case_id": "case-name-age",
            "split": "tuning",
            "language": "ja-ko",
            "source_text": "private source A",
            "context_after_text": "private context A",
            "canonical_translation": "본명 나나세 아야카 25세",
            "required_meaning": ["name and age"],
            "prohibited_changes": ["name_masking"],
        },
        {
            "case_id": "case-meaning",
            "split": "holdout",
            "language": "en-ko",
            "source_text": "private source B",
            "context_after_text": "private context B",
            "canonical_translation": "정답 B",
            "required_meaning": ["meaning B"],
            "prohibited_changes": ["meaning_change"],
        },
    ]
    reference: dict[str, object] = {
        "reference_sha256": "f" * 64,
        "cases": cases,
    }
    baseline = protocol.SamplerArm(
        "temperature",
        protocol.SamplerTuple(0.7, 0.95, 64, 0.0),
    )
    joint = protocol.SamplerArm(
        "joint_top_p_top_k",
        protocol.SamplerTuple(0.7, 1.0, 0, 0.0),
    )
    minimum = protocol.SamplerArm(
        "min_p",
        protocol.SamplerTuple(0.7, 1.0, 0, 0.05),
    )
    plan = protocol.CampaignPlan(
        temperature_arms=(baseline,),
        joint_arms=(joint,),
        min_p_arms=(minimum,),
    )
    translations = {
        baseline.sampler.key: {
            (20260802, "case-name-age"): "본명 나Please세 아야카 25세",
            (20260803, "case-name-age"): "본명 나나세 아야카 25세",
            (20260802, "case-meaning"): "정답 B",
            (20260803, "case-meaning"): "정답 B",
        },
        joint.sampler.key: {
            (20260802, "case-name-age"): "본명 나나세 아야카 25세",
            (20260803, "case-name-age"): "본명 나나세 아야카 25세",
            (20260802, "case-meaning"): "자연스러운 번역",
            (20260803, "case-meaning"): "자연스러운 번역",
        },
        minimum.sampler.key: {
            (20260802, "case-name-age"): "본명 나나세 아야카 25세",
            (20260803, "case-name-age"): "본명 나나세 아야카 25세",
            (20260802, "case-meaning"): "반대 번역",
            (20260803, "case-meaning"): "반대 번역",
        },
    }

    def records_for(arms: tuple[protocol.SamplerArm, ...]) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for arm in arms:
            for seed in protocol.SEEDS:
                for case in cases:
                    case_id = str(case["case_id"])
                    translation = translations[arm.sampler.key][(seed, case_id)]
                    response = {
                        "choices": [
                            {
                                "index": 0,
                                "content": json.dumps(
                                    {"translation": translation},
                                    ensure_ascii=False,
                                ),
                                "finish_reason": "stop",
                            }
                        ]
                    }
                    verdict = judgment.validate_response_envelope(response)
                    records.append(
                        {
                            "status": "complete",
                            "phase": arm.phase,
                            "arm_key": arm.key,
                            "logical_slot": f"{arm.sampler.key}|{seed}|{case_id}",
                            "case_id": case_id,
                            "split": case["split"],
                            "sampler": arm.sampler.payload(),
                            "seed": seed,
                            "reference_sha256": reference["reference_sha256"],
                            "runtime_fingerprint": "runtime-one",
                            "request_identity": {"case_id": case_id, "fixed": True},
                            "response": response,
                            "response_validation": verdict.payload(),
                            "translation": verdict.translation if verdict.status == "VALID" else "",
                            "latency_ms": 10.0,
                            "completion_tokens": 4,
                        }
                    )
        return records

    r6_records = records_for((baseline,))
    campaign_records = records_for((joint, minimum))
    all_records = r6_records + campaign_records
    ledger = incremental.new_incremental_ledger(reference)
    packet = incremental.build_incremental_judgment_packet(reference, all_records, ledger, batch_size=10)
    decisions = {
        str(row["cluster_id"]): {
            "decision": "MAJOR" if row["candidate_translation"] == "반대 번역" else "PASS",
            "category": (
                "semantic_reversal"
                if row["candidate_translation"] == "반대 번역"
                else "faithful_translation"
            ),
            "naturalness": 2 if row["candidate_translation"] == "반대 번역" else 5,
        }
        for row in packet["rows"]
    }
    ledger = incremental.mark_pending_batch(
        ledger,
        reference=reference,
        packet=packet,
        packet_file="judgment-batch-0001.json",
    )
    ledger = incremental.apply_incremental_judgments(
        ledger,
        reference=reference,
        packet=packet,
        decisions=decisions,
        applied_utc="2026-08-04T00:00:00Z",
    )
    return reference, plan, r6_records, campaign_records, ledger


def test_final_campaign_analysis_requires_exact_matrix_and_applies_name_age_gate() -> None:
    reference, plan, r6_records, campaign_records, ledger = _final_analysis_fixture()
    per_arm = 2 * len(protocol.SEEDS)
    status = {
        "state": "WAITING_FOR_FINAL_JUDGMENT",
        "reference_sha256": reference["reference_sha256"],
        "campaign_plan_sha256": plan.sha256,
        "reused_logical_slots": per_arm,
        "completed_new_logical_slots": per_arm * 2,
        "expected_evidence_logical_slots": per_arm * 3,
        "sampler_keys": list(plan.sampler_keys),
        "seeds": list(protocol.SEEDS),
        "r6_runtime_fingerprint": "runtime-one",
        "runtime_fingerprint": "runtime-one",
    }
    plan_artifact = {
        "campaign_plan_sha256": plan.sha256,
        "plan": plan.payload(),
        "reference_sha256": reference["reference_sha256"],
    }
    evidence = final_analysis.validate_final_campaign_evidence(
        reference=reference,
        r6_records=r6_records,
        campaign_records=campaign_records,
        r6_manifest={"status": "completed"},
        campaign_manifest={"status": "completed"},
        campaign_status=status,
        campaign_plan_artifact=plan_artifact,
        plan=plan,
    )
    gates = {
        "schema_version": final_analysis.FINAL_GATE_SCHEMA_VERSION,
        "reference_sha256": reference["reference_sha256"],
        "gates": [
            {
                "gate_id": "required-name-age-v1",
                "case_id": "case-name-age",
                "required_substrings": ["나나세 아야카", "25"],
            }
        ],
    }
    public, private = final_analysis.build_final_campaign_analysis(
        reference=reference,
        records=r6_records + campaign_records,
        ledger=ledger,
        gates=gates,
        evidence_summary=evidence,
        plan=plan,
    )

    assert evidence["total_response_count"] == 12
    assert len(public["ranked_arms"]) == 3
    assert public["provisional_candidate_sampler_key"] == "t0.70-p1.00-k0-m0.00"
    assert public["required_gates"][0]["excluded_sampler_keys"] == [
        "t0.70-p0.95-k64-m0.00"
    ]
    assert public["product_promotion_allowed"] is False
    assert len(public["seed_rows"]) == 3
    assert {item["decision"] for item in private["error_clusters"]} == {
        "CATASTROPHIC",
        "MAJOR",
    }
    catastrophic = next(
        item for item in private["error_clusters"] if item["decision"] == "CATASTROPHIC"
    )
    assert catastrophic["candidate_translation"] == ""
    assert catastrophic["raw_response"] is not None


def test_final_campaign_evidence_rejects_one_missing_sampler_seed_case() -> None:
    reference, plan, r6_records, campaign_records, _ledger = _final_analysis_fixture()
    per_arm = 2 * len(protocol.SEEDS)

    with pytest.raises(judgment.JudgmentError, match="Final response counts"):
        final_analysis.validate_final_campaign_evidence(
            reference=reference,
            r6_records=r6_records,
            campaign_records=campaign_records[:-1],
            r6_manifest={"status": "completed"},
            campaign_manifest={"status": "completed"},
            campaign_status={
                "state": "WAITING_FOR_FINAL_JUDGMENT",
                "reference_sha256": reference["reference_sha256"],
                "campaign_plan_sha256": plan.sha256,
                "reused_logical_slots": per_arm,
                "completed_new_logical_slots": per_arm * 2,
                "expected_evidence_logical_slots": per_arm * 3,
                "sampler_keys": list(plan.sampler_keys),
                "seeds": list(protocol.SEEDS),
            },
            campaign_plan_artifact={
                "campaign_plan_sha256": plan.sha256,
                "plan": plan.payload(),
                "reference_sha256": reference["reference_sha256"],
            },
            plan=plan,
        )


def test_automatic_blind_verdicts_hide_sampler_and_logical_slot_identity() -> None:
    reference = {
        "cases": [
            {
                "case_id": "case-safe",
                "split": "tuning",
                "language": "ja-ko",
                "source_text": "source",
                "context_after_text": "context",
                "canonical_translation": "canonical",
                "required_meaning": ["meaning"],
                "prohibited_changes": ["number_change"],
            }
        ]
    }
    packet = judgment.build_blind_judgment_packet(
        reference,
        [
            {
                "case_id": "case-safe",
                "split": "tuning",
                "logical_slot": "temperature|t0.70-p0.95-k64-m0.00|seed-20260802|case-safe",
                "sampler": protocol.SamplerTuple(0.7, 0.95, 64, 0.0).payload(),
                "response_validation": {
                    "schema_version": judgment.RESPONSE_VALIDATION_SCHEMA_VERSION,
                    "status": "CATASTROPHIC",
                    "category": "mixed_token_corruption",
                },
            }
        ],
    )
    serialized = json.dumps(packet, ensure_ascii=False)
    assert "logical_slot" not in serialized
    assert "seed-20260802" not in serialized
    assert packet["automatic_verdicts"][0]["occurrence_count"] == 1


def test_holdout_cannot_open_without_tuning_provisional_winner() -> None:
    with pytest.raises(judgment.JudgmentError):
        judgment.open_holdout_packet(
            {"cases": []},
            [],
            provisional_sampler=protocol.SamplerTuple(0.7, 0.95, 64, 0.0),
            baseline_sampler=protocol.SamplerTuple(0.7, 0.95, 64, 0.0),
            tuning_report={"scope": "tuning", "status": "WAITING_FOR_JUDGMENT"},
        )


def test_private_run_store_keeps_first_complete_response_only(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    store = RunStore(root)
    payload = {
        "status": "complete",
        "case_id": "case-safe",
        "phase": "temperature",
        "arm_key": "temperature-t0.10-p0.95-k64-m0.00",
        "logical_slot": "slot-safe",
        "recorded_utc": "2026-08-03T00:00:00Z",
    }
    assert store.record_case_if_first(
        phase="temperature",
        arm="temperature-t0.10-p0.95-k64-m0.00",
        run="seed-20260802-forward",
        case_id="case-safe",
        logical_slot="slot-safe",
        payload=payload,
    )
    assert not store.record_case_if_first(
        phase="temperature",
        arm="temperature-t0.10-p0.95-k64-m0.00",
        run="seed-20260802-forward",
        case_id="case-safe",
        logical_slot="slot-safe",
        payload={"status": "complete", "case_id": "changed"},
    )
    index = store.completed_index()
    record = read_json(root / index["slot-safe"]["path"])
    assert record["case_id"] == "case-safe"
    store.completion_index_path.unlink()
    recovered = RunStore(root)
    assert recovered.is_completed("slot-safe")
    assert recovered.completed_count() == 1
    store.bind_request_contract(case_id="case-safe", identity={"prompt_sha256": "a" * 64})
    store.bind_request_contract(case_id="case-safe", identity={"prompt_sha256": "a" * 64})
    with pytest.raises(Exception, match="request contract changed"):
        store.bind_request_contract(case_id="case-safe", identity={"prompt_sha256": "b" * 64})


def test_completion_index_and_completed_marker_must_match_private_records(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    store = RunStore(root)
    sampler = protocol.SamplerTuple(0.1, 0.95, 64, 0.0)
    cases = [{"case_id": "case-safe"}]
    selection = execution.PhaseSelection(
        "temperature",
        (protocol.SamplerArm("temperature", sampler),),
    )
    expected_slots = execution._expected_phase_slots(selection, cases)
    status = {
        "state": "WAITING_FOR_JUDGMENT",
        "phase": "temperature",
        "reference_sha256": "a" * 64,
        "arms": [arm.payload() for arm in selection.arms],
        "expected_logical_slots": len(expected_slots),
        "completed_logical_slots": len(expected_slots),
    }
    with pytest.raises(execution.ExecutionError, match="stored logical slots"):
        execution._assert_completed_phase_status(
            status,
            selection=selection,
            reference_sha256="a" * 64,
            expected_slots=expected_slots,
            store=store,
        )

    slot = next(iter(expected_slots))
    payload = {
        "status": "complete",
        "recorded_utc": "2026-08-03T00:00:00Z",
        "case_id": "case-safe",
        "phase": "temperature",
        "arm_key": f"temperature-{sampler.key}",
        "logical_slot": slot,
    }
    store.record_case_if_first(
        phase="temperature",
        arm=f"temperature-{sampler.key}",
        run="seed-20260802-forward",
        case_id="case-safe",
        logical_slot=slot,
        payload=payload,
    )
    # One seed is still missing, so a marker that claims completion is refused.
    with pytest.raises(execution.ExecutionError, match="stored logical slots"):
        execution._assert_completed_phase_status(
            status,
            selection=selection,
            reference_sha256="a" * 64,
            expected_slots=expected_slots,
            store=store,
        )

    index = store.completed_index()
    case_path = root / index[slot]["path"]
    corrupted = read_json(case_path)
    corrupted["logical_slot"] = "wrong-slot"
    _write_json(case_path, corrupted)
    with pytest.raises(execution.ExecutionError, match="identities disagree"):
        list(execution.iter_completed_records(RunStore(root)))


def test_frozen_reference_hash_and_deterministic_user_sample_are_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution, "CORPUS_CASE_COUNT", 1)
    one_case = {
        "case_id": "case-safe",
        "split": "tuning",
        "language": "ja-ko",
        "source_text": "source",
        "context_after_text": "context",
        "canonical_translation": "canonical",
        "required_meaning": ["meaning"],
        "prohibited_changes": ["number_change"],
        "review_status": "APPROVED",
    }
    frozen = {
        "schema_version": corpus.REFERENCE_SCHEMA_VERSION,
        "state": "FROZEN",
        "case_identity": "language+source_text+context_after_text",
        "cases": [one_case],
    }
    frozen["reference_sha256"] = protocol.canonical_sha256(
        {
            "schema_version": frozen["schema_version"],
            "case_identity": frozen["case_identity"],
            "cases": frozen["cases"],
        }
    )
    assert execution._case_list(frozen)[0]["case_id"] == "case-safe"
    tampered = json.loads(json.dumps(frozen))
    tampered["cases"][0]["source_text"] = "changed"
    with pytest.raises(execution.ExecutionError, match="hash"):
        execution._case_list(tampered)

    monkeypatch.setattr(corpus, "CORPUS_CASE_COUNT", 24)
    reference = {
        "schema_version": corpus.REFERENCE_SCHEMA_VERSION,
        "state": "PENDING_USER_SAMPLE",
        "case_identity": "language+source_text+context_after_text",
        "cases": [
            {
                "case_id": f"case-{index:02d}",
                "split": "tuning",
                "language": "ja-ko",
                "source_text": f"source-{index}",
                "context_after_text": "context",
                "canonical_translation": f"canonical-{index}",
                "required_meaning": ["meaning"],
                "prohibited_changes": ["number_change"],
                "review_status": "PENDING_USER_SAMPLE",
                "flags": [],
            }
            for index in range(24)
        ],
    }
    sample = corpus.select_user_sample(reference)
    for row in sample["rows"]:
        row["decision"] = "PASS"
    approval = {"approved": True, "sample_sha256": sample["sample_sha256"]}
    assert corpus.freeze_reference(reference, user_sample=sample, user_approval=approval)["state"] == "FROZEN"

    substituted = json.loads(json.dumps(sample))
    substituted["rows"][0], substituted["rows"][1] = substituted["rows"][1], substituted["rows"][0]
    substituted["sample_sha256"] = protocol.canonical_sha256(
        [row["case_id"] for row in substituted["rows"]]
    )
    with pytest.raises(corpus.CorpusError, match="cases or order"):
        corpus.freeze_reference(
            reference,
            user_sample=substituted,
            user_approval={"approved": True, "sample_sha256": substituted["sample_sha256"]},
        )

    altered_content = json.loads(json.dumps(sample))
    altered_content["rows"][0]["canonical_translation"] = "not-user-reviewed"
    with pytest.raises(corpus.CorpusError, match="content changed"):
        corpus.freeze_reference(
            reference,
            user_sample=altered_content,
            user_approval={"approved": True, "sample_sha256": altered_content["sample_sha256"]},
        )


def test_phase_report_exposes_tuning_scope_for_holdout_gate() -> None:
    provisional = protocol.SamplerTuple(0.7, 0.95, 64, 0.0)
    tuning_report = report.build_phase_report(
        phase_status={
            "state": "WAITING_FOR_JUDGMENT",
            "phase": "temperature",
            "reference_sha256": "a" * 64,
            "expected_logical_slots": 2,
            "completed_logical_slots": 2,
        },
        ranked=[],
        scope="tuning",
    )
    assert tuning_report["schema_version"] == "gemma-sampler-report-v3"
    tuning_report.update(
        {
            "status": "PROVISIONAL_WINNER",
            "provisional_sampler_key": provisional.key,
        }
    )
    packet = judgment.open_holdout_packet(
        {
            "reference_sha256": "a" * 64,
            "cases": [
                {
                    "case_id": "case-safe",
                    "split": "holdout",
                    "language": "ja-ko",
                    "source_text": "source",
                    "context_after_text": "context",
                    "canonical_translation": "canonical",
                    "required_meaning": ["meaning"],
                    "prohibited_changes": ["number_change"],
                }
            ],
        },
        [],
        provisional_sampler=provisional,
        baseline_sampler=provisional,
        tuning_report=tuning_report,
    )
    assert tuning_report["scope"] == "tuning"
    assert packet["scope"] == "holdout"


def test_campaign_preflight_reuses_complete_r6_without_starting_a_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution, "CORPUS_CASE_COUNT", 1)
    case = {
        "case_id": "case-safe",
        "split": "tuning",
        "language": "ja-ko",
        "source_text": "source",
        "context_after_text": "context",
        "canonical_translation": "canonical",
        "required_meaning": ["meaning"],
        "prohibited_changes": ["number_change"],
        "review_status": "APPROVED",
    }
    reference = {
        "schema_version": corpus.REFERENCE_SCHEMA_VERSION,
        "state": "FROZEN",
        "case_identity": "language+source_text+context_after_text",
        "cases": [case],
    }
    reference["reference_sha256"] = protocol.canonical_sha256(
        {
            "schema_version": reference["schema_version"],
            "case_identity": reference["case_identity"],
            "cases": reference["cases"],
        }
    )
    request_identity = {"case_id": "case-safe", "request_sha256_without_sampler_or_seed": "a" * 64}
    monkeypatch.setattr(
        campaign,
        "_campaign_request_identities",
        lambda **_kwargs: ({"Japanese": object()}, {"case-safe": request_identity}),
    )

    run_root = tmp_path / "r6-run"
    artifact_root = run_root / "artifacts"
    artifact_root.mkdir(parents=True)
    _write_json(run_root / "artifact-manifest.json", {"status": "completed"})
    r6_store = RunStore(artifact_root)
    plan = protocol.campaign_plan()
    for arm in plan.temperature_arms:
        for seed, order in zip(protocol.SEEDS, ("forward", "reverse"), strict=True):
            slot = f"temperature|{arm.sampler.key}|seed-{seed}|case-safe"
            r6_store.record_case_if_first(
                phase="temperature",
                arm=arm.key,
                run=f"seed-{seed}-{order}",
                case_id="case-safe",
                logical_slot=slot,
                payload={
                    "status": "complete",
                    "recorded_utc": "2026-08-03T00:00:00Z",
                    "phase": "temperature",
                    "arm_key": arm.key,
                    "sampler": arm.sampler.payload(),
                    "seed": seed,
                    "case_id": "case-safe",
                    "split": "tuning",
                    "reference_sha256": reference["reference_sha256"],
                    "runtime_fingerprint": "router-fingerprint",
                    "request_identity": request_identity,
                    "response_validation": {
                        "schema_version": judgment.RESPONSE_VALIDATION_SCHEMA_VERSION,
                        "status": "VALID",
                    },
                    "logical_slot": slot,
                },
            )
    expected = len(plan.temperature_arms) * len(protocol.SEEDS)
    r6_store.write_phase_status(
        "temperature",
        {
            "state": "WAITING_FOR_JUDGMENT",
            "phase": "temperature",
            "reference_sha256": reference["reference_sha256"],
            "arms": [arm.payload() for arm in plan.temperature_arms],
            "expected_logical_slots": expected,
            "completed_logical_slots": expected,
        },
    )
    _write_json(
        artifact_root / "runtime-contract.json",
        {
            "fingerprint": "router-fingerprint",
            "image_ref": protocol.PINNED_LLAMA_CPP_IMAGE,
            "binary_version": "version: 10133 (ff067f76d)",
            "command_sha256": "b" * 64,
            "preset_sha256": "c" * 64,
            "effective_context_size": "4096",
            "fixed_request_contract_sha256": protocol.canonical_sha256(
                protocol.fixed_request_contract_payload()
            ),
        },
    )

    summary = campaign.campaign_preflight_summary(reference=reference, r6_store=r6_store)

    assert summary["state"] == "READY_TO_RUN"
    assert summary["response_counts"]["total_new"] == 124280
    assert summary["case_count"] == 1


def test_campaign_runs_one_runtime_session_then_waits_for_final_judgment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution, "CORPUS_CASE_COUNT", 1)
    monkeypatch.setattr(protocol, "TEMPERATURE_VALUES", (0.1,))
    scaled_counts = {
        "r6_reused": 2,
        "joint_new": 22,
        "min_p_new": 4,
        "total_new": 26,
        "total_evidence": 28,
    }
    monkeypatch.setattr(campaign, "campaign_response_counts", lambda: dict(scaled_counts))
    case = {
        "case_id": "case-safe",
        "split": "tuning",
        "language": "ja-ko",
        "source_text": "source",
        "context_after_text": "context",
        "canonical_translation": "canonical",
        "required_meaning": ["meaning"],
        "prohibited_changes": ["number_change"],
        "review_status": "APPROVED",
    }
    reference = {
        "schema_version": corpus.REFERENCE_SCHEMA_VERSION,
        "state": "FROZEN",
        "case_identity": "language+source_text+context_after_text",
        "cases": [case],
    }
    reference["reference_sha256"] = protocol.canonical_sha256(
        {
            "schema_version": reference["schema_version"],
            "case_identity": reference["case_identity"],
            "cases": reference["cases"],
        }
    )
    request_identity = {"case_id": "case-safe", "request_sha256_without_sampler_or_seed": "a" * 64}
    engine = object()
    monkeypatch.setattr(
        campaign,
        "_campaign_request_identities",
        lambda **_kwargs: ({"Japanese": engine}, {"case-safe": request_identity}),
    )
    monkeypatch.setattr(
        execution,
        "_case_request",
        lambda _engine, _case, sampler, seed: (
            {"sampler": sampler.payload(), "seed": seed},
            request_identity,
        ),
    )

    r6_run = tmp_path / "r6-run"
    r6_artifacts = r6_run / "artifacts"
    r6_artifacts.mkdir(parents=True)
    _write_json(r6_run / "artifact-manifest.json", {"status": "completed"})
    r6_store = RunStore(r6_artifacts)
    plan = protocol.campaign_plan()
    for seed, order in zip(protocol.SEEDS, ("forward", "reverse"), strict=True):
        arm = plan.temperature_arms[0]
        slot = f"temperature|{arm.sampler.key}|seed-{seed}|case-safe"
        r6_store.record_case_if_first(
            phase="temperature",
            arm=arm.key,
            run=f"seed-{seed}-{order}",
            case_id="case-safe",
            logical_slot=slot,
            payload={
                "status": "complete",
                "recorded_utc": "2026-08-03T00:00:00Z",
                "phase": "temperature",
                "arm_key": arm.key,
                "sampler": arm.sampler.payload(),
                "seed": seed,
                "case_id": "case-safe",
                "split": "tuning",
                "reference_sha256": reference["reference_sha256"],
                "runtime_fingerprint": "router-fingerprint",
                "request_identity": request_identity,
                "response_validation": {"schema_version": judgment.RESPONSE_VALIDATION_SCHEMA_VERSION},
                "logical_slot": slot,
            },
        )
    r6_store.write_phase_status(
        "temperature",
        {
            "state": "WAITING_FOR_JUDGMENT",
            "phase": "temperature",
            "reference_sha256": reference["reference_sha256"],
            "arms": [arm.payload() for arm in plan.temperature_arms],
            "expected_logical_slots": 2,
            "completed_logical_slots": 2,
        },
    )
    _write_json(
        r6_artifacts / "runtime-contract.json",
        {
            "fingerprint": "router-fingerprint",
            "image_ref": protocol.PINNED_LLAMA_CPP_IMAGE,
            "binary_version": "version: 10133 (ff067f76d)",
            "command_sha256": "b" * 64,
            "preset_sha256": "c" * 64,
            "effective_context_size": "4096",
            "fixed_request_contract_sha256": protocol.canonical_sha256(
                protocol.fixed_request_contract_payload()
            ),
        },
    )

    class FakeRuntime:
        def __init__(self) -> None:
            self.contract = _router_runtime_contract(gemma_context_size="4096")
            self.started = 0
            self.closed = 0

        def start(self) -> object:
            self.started += 1
            return self.contract

        def request(self, _payload: object, *, timeout_sec: float) -> runtime.ReplayResponse:
            assert timeout_sec == 1.0
            return runtime.ReplayResponse(
                envelope={
                    "choices": [
                        {"index": 0, "finish_reason": "stop", "message": {"content": '{"translation":"ok"}'}}
                    ]
                },
                latency_ms=1.0,
                completion_tokens=1,
            )

        def close(self) -> None:
            self.closed += 1

    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    fake_runtime = FakeRuntime()
    status = campaign.execute_campaign(
        store=RunStore(campaign_root),
        reference=reference,
        r6_store=r6_store,
        timeout_sec=1.0,
        max_attempts=1,
        runtime=fake_runtime,
    )

    assert status["state"] == "WAITING_FOR_FINAL_JUDGMENT"
    assert status["completed_new_logical_slots"] == 26
    assert fake_runtime.started == 1
    assert fake_runtime.closed == 1
    records = list(execution.iter_completed_records(RunStore(campaign_root)))
    assert len(records) == 26
    assert {record["reference_sha256"] for record in records} == {reference["reference_sha256"]}


def test_campaign_progress_excludes_retry_backoff_from_active_elapsed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    store = RunStore(artifact_root)
    validation = campaign.CampaignReuseValidation(
        plan=protocol.campaign_plan(),
        cases=(),
        engines={},
        request_identities={},
        r6_records={},
        r6_contract={},
    )
    moments = iter((100.0, 105.0))
    monkeypatch.setattr(campaign.time, "monotonic", lambda: next(moments))

    progress = campaign._CampaignProgress(store=store, validation=validation)
    progress.add_backoff(2.0)
    progress.write(state="RUNNING_JOINT")

    payload = read_json(store.progress_path)
    assert payload["active_elapsed_seconds"] == 3.0
    assert payload["backoff_elapsed_seconds"] == 2.0


def test_reused_phase_rows_require_matching_reference_runtime_and_request_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution, "CORPUS_CASE_COUNT", 1)
    case = {
        "case_id": "case-safe",
        "split": "tuning",
        "language": "ja-ko",
        "source_text": "source",
        "context_after_text": "context",
        "canonical_translation": "canonical",
        "required_meaning": ["meaning"],
        "prohibited_changes": ["number_change"],
        "review_status": "APPROVED",
    }
    reference = {
        "schema_version": corpus.REFERENCE_SCHEMA_VERSION,
        "state": "FROZEN",
        "case_identity": "language+source_text+context_after_text",
        "cases": [case],
    }
    reference["reference_sha256"] = protocol.canonical_sha256(
        {
            "schema_version": reference["schema_version"],
            "case_identity": reference["case_identity"],
            "cases": reference["cases"],
        }
    )
    sampler = protocol.SamplerTuple(0.7, 0.95, 64, 0.0)
    prior_root = tmp_path / "prior"
    prior_root.mkdir()
    prior = RunStore(prior_root)
    request_identity = {"case_id": "case-safe", "request_sha256_without_sampler_or_seed": "a" * 64}
    for seed, order in zip(protocol.SEEDS, ("forward", "reverse"), strict=True):
        slot = f"temperature|{sampler.key}|seed-{seed}|case-safe"
        record = {
            "status": "complete",
            "recorded_utc": "2026-08-03T00:00:00Z",
            "phase": "temperature",
            "arm_key": f"temperature-{sampler.key}",
            "sampler": sampler.payload(),
            "seed": seed,
            "case_id": "case-safe",
            "split": "tuning",
            "reference_sha256": reference["reference_sha256"],
            "runtime_fingerprint": "router-fingerprint",
            "request_identity": request_identity,
            "response_validation": {
                "schema_version": judgment.RESPONSE_VALIDATION_SCHEMA_VERSION,
                "status": "VALID",
                "category": "",
                "translation_sha256": "a" * 64,
                "message": "",
                "sanitized_channel_tokens": False,
            },
            "logical_slot": slot,
        }
        prior.record_case_if_first(
            phase="temperature",
            arm=f"temperature-{sampler.key}",
            run=f"seed-{seed}-{order}",
            case_id="case-safe",
            logical_slot=slot,
            payload=record,
        )

    records = execution.collect_completed_records((prior,), reference=reference)
    phase_status = {
        "state": "WAITING_FOR_JUDGMENT",
        "phase": "temperature",
        "reference_sha256": reference["reference_sha256"],
        "arms": [protocol.SamplerArm("temperature", sampler).payload()],
    }
    assert execution.sampler_keys_from_phase_status(phase_status, reference=reference) == (sampler.key,)
    assert len(
        execution.complete_scope_records(
            records,
            reference=reference,
            scope="tuning",
            sampler_keys=(sampler.key,),
        )
    ) == 2
    reused = execution._required_reused_records(
        records,
        reused_arms=(protocol.SamplerArm("joint_top_p_top_k", sampler, reused=True),),
        cases=execution._case_list(reference),
    )
    assert len(reused) == 2
    execution._assert_reused_contracts(
        reused,
        runtime_fingerprint="router-fingerprint",
        request_identities={"case-safe": request_identity},
    )
    with pytest.raises(execution.ExecutionError, match="runtime fingerprint"):
        execution._assert_reused_contracts(
            reused,
            runtime_fingerprint="different-router-fingerprint",
            request_identities={"case-safe": request_identity},
        )

    first_path = prior.root / next(iter(prior.completed_index().values()))["path"]
    stale_record = read_json(first_path)
    stale_record["response_validation"].pop("schema_version")
    _write_json(first_path, stale_record)
    with pytest.raises(execution.ExecutionError, match="lacks raw Router output"):
        list(execution.iter_compatible_completed_records((RunStore(prior_root),), reference=reference))


def test_complete_scope_records_keeps_holdout_sealed_without_weakening_tuning_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution, "CORPUS_CASE_COUNT", 2)
    cases = [
        {
            "case_id": "case-holdout",
            "split": "holdout",
            "language": "ja-ko",
            "source_text": "holdout source",
            "context_after_text": "",
            "canonical_translation": "holdout canonical",
            "required_meaning": ["meaning"],
            "prohibited_changes": ["number_change"],
            "review_status": "APPROVED",
        },
        {
            "case_id": "case-tuning",
            "split": "tuning",
            "language": "ja-ko",
            "source_text": "tuning source",
            "context_after_text": "",
            "canonical_translation": "tuning canonical",
            "required_meaning": ["meaning"],
            "prohibited_changes": ["number_change"],
            "review_status": "APPROVED",
        },
    ]
    reference = {
        "schema_version": corpus.REFERENCE_SCHEMA_VERSION,
        "state": "FROZEN",
        "case_identity": "language+source_text+context_after_text",
        "cases": cases,
    }
    reference["reference_sha256"] = protocol.canonical_sha256(
        {
            "schema_version": reference["schema_version"],
            "case_identity": reference["case_identity"],
            "cases": reference["cases"],
        }
    )
    sampler = protocol.SamplerTuple(0.7, 0.95, 64, 0.0)
    records = [
        {"case_id": str(case["case_id"]), "seed": seed, "sampler": sampler.payload()}
        for case in cases
        for seed in protocol.SEEDS
    ]

    selected = execution.complete_scope_records(
        records,
        reference=reference,
        scope="tuning",
        sampler_keys=(sampler.key,),
    )
    assert len(selected) == len(protocol.SEEDS)
    assert {record["case_id"] for record in selected} == {"case-tuning"}

    with pytest.raises(execution.ExecutionError, match="complete sampler matrix"):
        execution.complete_scope_records(
            [*records, dict(records[2])],
            reference=reference,
            scope="tuning",
            sampler_keys=(sampler.key,),
        )
    with pytest.raises(execution.ExecutionError, match="every selected case and seed"):
        execution.complete_scope_records(
            [record for record in records if record["case_id"] != "case-tuning" or record["seed"] != protocol.SEEDS[0]],
            reference=reference,
            scope="tuning",
            sampler_keys=(sampler.key,),
        )


def test_execution_phase_selection_requires_two_temps_or_three_tuples() -> None:
    with pytest.raises(protocol.ProtocolError):
        execution.select_phase("joint_top_p_top_k", selected_temperatures=(0.7,))
    with pytest.raises(protocol.ProtocolError):
        execution.select_phase(
            "min_p",
            selected_tuples=(protocol.SamplerTuple(0.7, 0.95, 64, 0.0),),
        )
    with pytest.raises(execution.ExecutionError):
        execution.parse_sampler_tuple("t0.00-p0.95-k64-m0.00")
    assert execution._source_language_name({"language": "en-ko"}) == "English"
    with pytest.raises(execution.ExecutionError, match="unsupported source-language"):
        execution._source_language_name({"language": "unknown"})


def _router_runtime_contract(*, gemma_context_size: str) -> SimpleNamespace:
    model = SimpleNamespace(
        model_sha256="a" * 64,
        ready_manifest_sha256="b" * 64,
        runtime_options={"LLAMA_CTX_SIZE": gemma_context_size},
    )
    return SimpleNamespace(
        effective_environment={},
        fingerprint="router-fingerprint",
        image_ref=protocol.PINNED_LLAMA_CPP_IMAGE,
        image_id="sha256:image",
        repo_digest="ghcr.io/ggml-org/llama.cpp@sha256:image",
        binary_version="llama.cpp b10133",
        command_sha256="c" * 64,
        preset_sha256="d" * 64,
        ocr_model=model,
        gemma_model=model,
    )


def test_runtime_contract_uses_router_gemma_model_context_not_container_environment() -> None:
    contract = _router_runtime_contract(gemma_context_size="4096")
    contract.effective_environment = {"LLAMA_CTX_SIZE": "8192"}

    payload = execution._assert_runtime_contract(SimpleNamespace(contract=contract))

    assert payload["effective_context_size"] == "4096"


def test_runtime_contract_rejects_router_gemma_model_context_drift() -> None:
    contract = _router_runtime_contract(gemma_context_size="8192")

    with pytest.raises(execution.ExecutionError, match="Gemma model context size"):
        execution._assert_runtime_contract(SimpleNamespace(contract=contract))


def test_rank_report_never_serializes_missing_usage_as_nonstandard_infinity() -> None:
    ranked = judgment.rank_sampler_results(
        [
            {
                "case_id": "case-safe",
                "split": "tuning",
                "logical_slot": "slot-safe",
                "sampler": protocol.SamplerTuple(0.7, 0.95, 64, 0.0).payload(),
                "latency_ms": 10.0,
                "completion_tokens": None,
            }
        ],
        {"slot-safe": {"decision": "PASS", "naturalness": 5}},
    )
    summary = judgment.public_rank_summary(ranked, scope="tuning", reference_sha256="a" * 64)
    assert summary["schema_version"] == judgment.JUDGMENT_SCHEMA_VERSION
    assert summary["rows"][0]["unjudged"] == 0
    assert summary["rows"][0]["completion_tokens_mean"] is None
    assert "Infinity" not in json.dumps(summary, allow_nan=False)


def test_startup_cleanup_failure_is_returned_to_the_calling_failure_path() -> None:
    class FailingCoordinator:
        def finish(self, **_kwargs: object) -> None:
            raise RuntimeError("release verification failed")

    replay = object.__new__(runtime.RouterGemmaReplayRuntime)
    replay.coordinator = FailingCoordinator()
    replay.arbiter = object()
    replay._started = True
    replay.spec = object()
    replay.contract = object()

    cleanup_error = replay._terminal_cleanup_after_start_failure()
    assert isinstance(cleanup_error, RuntimeError)
    assert not replay._started
    assert replay.spec is None
    assert replay.contract is None


def test_startup_failure_keeps_setup_cause_after_successful_terminal_cleanup() -> None:
    class FailingPrepareCoordinator:
        def classify_pair(self, *_args: object) -> SimpleNamespace:
            return SimpleNamespace(kind=SimpleNamespace(value="crop"))

        def prepare(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("compose preparation root cause")

        def finish(self, **_kwargs: object) -> None:
            return None

    replay = object.__new__(runtime.RouterGemmaReplayRuntime)
    replay.settings = runtime.RouterLabSettings()
    replay.coordinator = FailingPrepareCoordinator()
    replay.ocr_manager = SimpleNamespace(_router_runtime_spec=lambda *_args: object())
    replay.gemma_manager = SimpleNamespace(set_router_spec=lambda *_args: None)
    replay.arbiter = object()
    replay.adapter = object()
    replay.spec = None
    replay.contract = None
    replay._started = False

    with pytest.raises(runtime.RuntimeErrorV2, match="compose preparation root cause") as raised:
        replay.start()

    assert "terminal cleanup/GPU return verification passed" in str(raised.value)
    assert isinstance(raised.value.__cause__, RuntimeError)


def test_only_cuda13_bat_can_start_or_resume_the_campaign() -> None:
    bat = (ROOT / "scripts" / "benchmark_gemma_sampler_quality_v2.bat").read_text(encoding="utf-8")
    cuda13 = (ROOT / "scripts" / "benchmark_gemma_sampler_quality_v2_cuda13.bat").read_text(
        encoding="utf-8"
    )
    assert "Read-only campaign preflight only" in bat
    assert "run-campaign" not in bat
    assert "verify-campaign" in bat
    assert ".venv-win\\Scripts\\python.exe" in bat
    assert "SAMPLER_EXIT%\"==\"75" in cuda13
    assert "goto retry" in cuda13
    assert "run-campaign" in cuda13
    assert "SAMPLER_LAUNCHED_BY_EXE" in cuda13
    assert "SAMPLER_PHASE" not in cuda13
    assert ".venv-win-cuda13\\Scripts\\python.exe" in cuda13


def test_cuda13_bat_launches_the_read_only_monitor_and_persists_runner_logs() -> None:
    bat = (ROOT / "scripts" / "benchmark_gemma_sampler_quality_v2_cuda13.bat").read_text(
        encoding="utf-8"
    )

    assert "SAMPLER_NO_MONITOR" in bat
    assert "SAMPLER_VERIFY_ONLY" in bat
    assert "build_gemma_sampler_monitor.bat --if-stale" in bat
    assert 'start "Gemma Sampler Monitor"' in bat
    assert '"%SAMPLER_MONITOR_EXE%" --run-root "%SAMPLER_RUN_ROOT%"' in bat
    assert "--exit-on-completion" in bat
    assert "SAMPLER_LOG_DIR" in bat
    assert '>> "%SAMPLER_LOG%" 2>&1' in bat


def test_monitor_builder_uses_scoop_fallback_and_private_executable_output() -> None:
    builder = (ROOT / "scripts" / "build_gemma_sampler_monitor.bat").read_text(encoding="utf-8")

    assert "%USERPROFILE%\\scoop\\shims\\go.exe" in builder
    assert "%USERPROFILE%\\scoop\\apps\\go\\current\\bin\\go.exe" in builder
    assert "gemma-monitor.exe" in builder
    assert "gemma-sampler-launcher.exe" in builder
    assert "--if-stale" in builder
    assert "--monitor-only-if-stale" in builder
    assert "build -trimpath" in builder


def test_monitor_source_uses_bubble_tea_alt_screen_and_bounded_snapshot_reads() -> None:
    monitor = (ROOT / "scripts" / "gemma_sampler_monitor" / "main.go").read_text(encoding="utf-8")
    snapshot = (ROOT / "scripts" / "gemma_sampler_monitor" / "snapshot.go").read_text(encoding="utf-8")

    assert "tea.WithAltScreen()" in monitor
    assert "runner·Docker·GPU 작업은 계속됩니다" in monitor
    launcher = (ROOT / "scripts" / "gemma_sampler_monitor" / "launcher.go").read_text(encoding="utf-8")
    assert "gemma-sampler-launcher.exe" in launcher
    assert "--monitor-only-if-stale" in launcher
    assert "windowsCampaignWorkerAlive" in launcher
    assert "활성 시간 / ACTIVE:" in monitor
    assert "runner log:" in monitor
    assert "readSharedFile" in snapshot
    assert "file.Close()" in snapshot
    assert "Do not tail or retain a handle" in snapshot


def test_private_review_html_escapes_text_and_never_adds_source_paths() -> None:
    document = render_private_review_html(
        {
            "rows": [
                {
                    "case_id": "case-safe",
                    "source_text": "<script>alert(1)</script>",
                    "canonical_translation": "safe",
                }
            ]
        },
        title="Private review",
    )
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in document
    assert "C:\\" not in document
