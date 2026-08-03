from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gemma_sampler_quality_v2 import corpus, execution, judgment, protocol, report, runtime  # noqa: E402
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


def test_pinned_filter_contract_requires_image_build_and_exact_payload() -> None:
    protocol.assert_pinned_sampler_contract(
        image_ref=protocol.PINNED_LLAMA_CPP_IMAGE,
        binary_version="llama.cpp b10133",
        payload={"top_k": 0, "top_p": 1.0},
    )
    with pytest.raises(protocol.ProtocolError):
        protocol.assert_pinned_sampler_contract(
            image_ref="other",
            binary_version="llama.cpp b10133",
            payload={"top_k": 0, "top_p": 1.0},
        )


@pytest.mark.parametrize(
    ("content", "category"),
    [
        ('{"translation":"나Please세"}', "mixed_token_corruption"),
        ('{"translation":"나please세"}', "mixed_token_corruption"),
        ('{"translation":"나I세"}', "mixed_token_corruption"),
        ('{"translation":"   "}', "censorship_or_deletion"),
        ('<|channel>thought<channel|>{"translation":"normal"}', "raw_channel_or_reasoning_leak"),
        ('{"translation":"normal","extra":"x"}', "json_schema_order_count_or_finish"),
    ],
)
def test_raw_bad_outputs_are_catastrophic_before_product_sanitizing(content: str, category: str) -> None:
    verdict = judgment.validate_response_envelope(
        {"choices": [{"index": 0, "content": content, "finish_reason": "stop"}]}
    )
    assert verdict.status == "CATASTROPHIC"
    assert verdict.category == category


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


def test_windows_bats_resume_transient_worker_exit_and_keep_cuda_pair() -> None:
    bat = (ROOT / "scripts" / "benchmark_gemma_sampler_quality_v2.bat").read_text(encoding="utf-8")
    cuda13 = (ROOT / "scripts" / "benchmark_gemma_sampler_quality_v2_cuda13.bat").read_text(
        encoding="utf-8"
    )
    assert "SAMPLER_EXIT%\"==\"75" in bat
    assert "goto retry" in bat
    assert "SAMPLER_PRIOR_RESPONSE_RUN" in bat
    assert ".venv-win\\Scripts\\python.exe" in bat
    assert ".venv-win-cuda13\\Scripts\\python.exe" in cuda13


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
