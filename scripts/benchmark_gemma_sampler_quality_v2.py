#!/usr/bin/env python3
"""Private, resumable Gemma sampler-quality v2 benchmark CLI.

Tracked code contains protocol and machinery only.  This command refuses every
raw corpus, response, judgment, and report path outside the ignored managed
``banchmark_result_log/`` archive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import validation_artifact_harness as harness  # noqa: E402
from scripts.gemma_sampler_quality_v2.corpus import (  # noqa: E402
    CorpusError,
    apply_blind_review,
    apply_canonical_answers,
    apply_resolutions,
    build_blind_review_packet,
    build_reference_draft,
    build_resolution_packet,
    freeze_reference,
    reference_summary,
    select_user_sample,
)
from scripts.gemma_sampler_quality_v2.execution import (  # noqa: E402
    ExecutionError,
    ResumeRequired,
    RunStore,
    collect_completed_records,
    complete_scope_records,
    execute_phase,
    load_frozen_reference,
    parse_sampler_tuple,
    sampler_keys_from_phase_status,
    select_phase,
)
from scripts.gemma_sampler_quality_v2.final_analysis import (  # noqa: E402
    build_final_campaign_analysis,
    validate_final_campaign_evidence,
)
from scripts.gemma_sampler_quality_v2.campaign import (  # noqa: E402
    campaign_preflight_summary,
    execute_campaign,
)
from scripts.gemma_sampler_quality_v2.judgment import (  # noqa: E402
    JudgmentError,
    bind_cluster_verdicts_to_records,
    build_blind_judgment_packet,
    open_holdout_packet,
    rank_sampler_results,
)
from scripts.gemma_sampler_quality_v2.incremental import (  # noqa: E402
    apply_incremental_judgments,
    build_incremental_judgment_packet,
    mark_pending_batch,
    new_incremental_ledger,
    seed_ledger_from_completed_packet,
    validate_incremental_ledger,
    validate_incremental_packet,
)
from scripts.gemma_sampler_quality_v2.report import (  # noqa: E402
    build_phase_report,
    render_public_markdown,
)
from scripts.gemma_sampler_quality_v2.review import ReviewBoardError, render_private_review_html  # noqa: E402
from scripts.gemma_sampler_quality_v2.storage import (  # noqa: E402
    StorageError,
    atomic_write_json,
    read_json,
    utc_now,
)
from scripts.gemma_sampler_quality_v2.protocol import ProtocolError  # noqa: E402


FAMILY = "gemma-sampler-quality-v2"
CATEGORY = "10-gemma-translation"
EXIT_RESUME = 75
INCREMENTAL_LEDGER_FILE = "incremental-judgment-ledger.json"


def _private_path(value: str | Path) -> Path:
    archive_root = harness.default_archive_root().resolve()
    path = Path(value).resolve()
    try:
        path.relative_to(archive_root)
    except ValueError as exc:
        raise ExecutionError("Sampler v2 raw artifacts must remain in the private validation archive.") from exc
    return path


def _load_object(path: str | Path) -> dict[str, Any]:
    value = read_json(_private_path(path))
    if not isinstance(value, Mapping):
        raise ExecutionError("Private sampler input must be a JSON object.")
    return dict(value)


def _decision_map(path: str | Path, *, collection_keys: tuple[str, ...], id_key: str) -> dict[str, dict[str, Any]]:
    payload = _load_object(path)
    candidate: Any = payload
    for key in collection_keys:
        if key in payload:
            candidate = payload[key]
            break
    if isinstance(candidate, Mapping) and all(isinstance(value, Mapping) for value in candidate.values()):
        return {str(key): dict(value) for key, value in candidate.items()}
    if not isinstance(candidate, list):
        raise ExecutionError("Private decision input has no valid decision collection.")
    result: dict[str, dict[str, Any]] = {}
    for row in candidate:
        if not isinstance(row, Mapping):
            raise ExecutionError("Private decision input contains an invalid row.")
        identity = str(row.get(id_key) or "")
        if not identity or identity in result:
            raise ExecutionError("Private decision input has duplicate or empty identities.")
        result[identity] = dict(row)
    return result


def _open_run(args: argparse.Namespace) -> harness.ManagedArtifactRun:
    resume = str(getattr(args, "resume_run", "") or "").strip()
    if resume:
        run_root = _private_path(resume)
        try:
            return harness.ManagedArtifactRun.resume(run_root)
        except harness.ArtifactHarnessError:
            command = "run-campaign" if bool(getattr(args, "campaign", False)) else "run-phase"
            if command != "run-campaign" and not str(getattr(args, "phase", "") or "").strip():
                raise
            return harness.ManagedArtifactRun.recover_failed_atomic_replace(
                run_root,
                command=command,
                target_file_name="progress.json",
            )
    return harness.ManagedArtifactRun.create(
        family=FAMILY,
        category=CATEGORY,
        run_id=(str(getattr(args, "run_id", "") or "").strip() or None),
    )


def _finish_run(run: harness.ManagedArtifactRun, *, command: str, summary: Mapping[str, Any]) -> None:
    safe = {"command": command, "summary": dict(summary)}
    run.complete(metadata=safe)
    print(json.dumps({"run_root": str(run.run_root), "summary": safe["summary"]}, ensure_ascii=False))


def _write_transform(
    args: argparse.Namespace,
    *,
    command: str,
    filename: str,
    action: Callable[[], Mapping[str, Any]],
    summary: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> int:
    run = _open_run(args)
    try:
        output = dict(action())
        atomic_write_json(run.artifact_root / filename, output)
        _finish_run(run, command=command, summary=summary(output))
        return 0
    except BaseException as exc:
        run.fail(exc, metadata={"command": command})
        raise


def command_build_reference_draft(args: argparse.Namespace) -> int:
    source_manifest = _private_path(args.source_manifest)
    return _write_transform(
        args,
        command="build-reference-draft",
        filename="reference-draft.json",
        action=lambda: build_reference_draft(source_manifest, archive_root=harness.default_archive_root()),
        summary=reference_summary,
    )


def command_apply_canonical(args: argparse.Namespace) -> int:
    reference = _load_object(args.reference)
    answer_payload = _load_object(args.answers)
    answers = _decision_map(args.answers, collection_keys=("answers", "rows"), id_key="case_id")
    defaults = answer_payload.get("defaults")
    if defaults is not None and not isinstance(defaults, Mapping):
        raise ExecutionError("Canonical answer defaults must be an object.")
    if isinstance(defaults, Mapping):
        answers = {
            case_id: {**dict(defaults), **answer}
            for case_id, answer in answers.items()
        }
    return _write_transform(
        args,
        command="apply-canonical",
        filename="reference-canonical.json",
        action=lambda: apply_canonical_answers(reference, answers),
        summary=reference_summary,
    )


def command_build_blind_review(args: argparse.Namespace) -> int:
    reference = _load_object(args.reference)
    return _write_transform(
        args,
        command="build-blind-review",
        filename="blind-review.json",
        action=lambda: build_blind_review_packet(reference),
        summary=lambda result: {
            "schema_version": result.get("schema_version"),
            "reference_case_count": result.get("reference_case_count"),
        },
    )


def command_apply_blind_review(args: argparse.Namespace) -> int:
    reference = _load_object(args.reference)
    packet = _load_object(args.packet)
    decisions = _decision_map(args.decisions, collection_keys=("decisions", "rows"), id_key="blind_id")
    return _write_transform(
        args,
        command="apply-blind-review",
        filename="reference-pending-resolution.json",
        action=lambda: apply_blind_review(reference, packet, decisions),
        summary=reference_summary,
    )


def command_build_resolution_packet(args: argparse.Namespace) -> int:
    reference = _load_object(args.reference)
    return _write_transform(
        args,
        command="build-resolution-packet",
        filename="reference-resolution-packet.json",
        action=lambda: build_resolution_packet(reference),
        summary=lambda result: {"flagged_case_count": result.get("flagged_case_count", 0)},
    )


def command_apply_resolutions(args: argparse.Namespace) -> int:
    reference = _load_object(args.reference)
    resolutions = _decision_map(args.resolutions, collection_keys=("resolutions", "rows"), id_key="case_id")
    return _write_transform(
        args,
        command="apply-resolutions",
        filename="reference-pending-user-sample.json",
        action=lambda: apply_resolutions(reference, resolutions),
        summary=reference_summary,
    )


def command_build_user_sample(args: argparse.Namespace) -> int:
    reference = _load_object(args.reference)
    return _write_transform(
        args,
        command="build-user-sample",
        filename="reference-user-sample.json",
        action=lambda: select_user_sample(reference),
        summary=lambda result: {
            "sample_size": result.get("sample_size", 0),
            "sample_sha256": result.get("sample_sha256", ""),
        },
    )


def command_freeze_reference(args: argparse.Namespace) -> int:
    reference = _load_object(args.reference)
    user_sample = _load_object(args.user_sample)
    user_approval = _load_object(args.user_approval)
    return _write_transform(
        args,
        command="freeze-reference",
        filename="reference-frozen.json",
        action=lambda: freeze_reference(
            reference,
            user_sample=user_sample,
            user_approval=user_approval,
        ),
        summary=reference_summary,
    )


def command_run_phase(args: argparse.Namespace) -> int:
    run = _open_run(args)
    try:
        reference = load_frozen_reference(_private_path(args.reference))
        selected_temperatures = tuple(float(value) for value in args.selected_temperature)
        selected_tuples = tuple(parse_sampler_tuple(value) for value in args.selected_tuple)
        selection = select_phase(
            args.phase,
            selected_temperatures=selected_temperatures,
            selected_tuples=selected_tuples,
        )
        status = execute_phase(
            store=RunStore(run.artifact_root),
            reference=reference,
            selection=selection,
            timeout_sec=float(args.timeout_sec),
            max_attempts=int(args.max_attempts),
            prior_stores=_response_stores(args.prior_response_run, required=False),
        )
    except ResumeRequired as exc:
        run.checkpoint(
            metadata={
                "command": "run-phase",
                "state": "RESUME_REQUIRED",
                "detail": str(exc)[:512],
            }
        )
        print(f"RESUME_RUN={run.run_root}")
        return EXIT_RESUME
    except BaseException as exc:
        run.fail(exc, metadata={"command": "run-phase"})
        raise
    _finish_run(
        run,
        command="run-phase",
        summary={
            "phase": status.get("phase"),
            "state": status.get("state"),
            "completed_logical_slots": status.get("completed_logical_slots"),
            "expected_logical_slots": status.get("expected_logical_slots"),
            "reference_sha256": status.get("reference_sha256"),
        },
    )
    return 0


def _campaign_family_root() -> Path:
    return harness.default_archive_root() / "managed-runs" / CATEGORY / FAMILY


def _discover_frozen_reference_path() -> Path:
    root = _campaign_family_root()
    candidates: list[Path] = []
    for candidate in sorted(root.glob("*/artifacts/reference-frozen.json")):
        try:
            reference = load_frozen_reference(candidate)
        except (ExecutionError, StorageError):
            continue
        if reference.get("state") == "FROZEN":
            candidates.append(candidate)
    if len(candidates) != 1:
        raise ExecutionError(
            "Campaign requires exactly one user-approved frozen reference in the managed private archive."
        )
    return candidates[0]


def _discover_r6_run_root(*, reference: Mapping[str, Any]) -> Path:
    root = _campaign_family_root()
    candidates: list[Path] = []
    expected_reference = str(reference.get("reference_sha256") or "")
    for candidate in sorted(root.glob("*/artifacts/phase-status/temperature.json")):
        try:
            status = _load_object(candidate)
        except (ExecutionError, StorageError):
            continue
        if (
            status.get("state") == "WAITING_FOR_JUDGMENT"
            and status.get("phase") == "temperature"
            and status.get("expected_logical_slots") == 9560
            and status.get("completed_logical_slots") == 9560
            and str(status.get("reference_sha256") or "") == expected_reference
        ):
            candidates.append(candidate.parents[2])
    if len(candidates) != 1:
        raise ExecutionError(
            "Campaign requires exactly one complete r6 temperature run matching the frozen reference."
        )
    return candidates[0]


def _resolve_campaign_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], Path, RunStore]:
    reference_value = str(getattr(args, "reference", "") or "").strip()
    reference_path = _private_path(reference_value) if reference_value else _discover_frozen_reference_path()
    reference = load_frozen_reference(reference_path)
    r6_value = str(getattr(args, "r6_run", "") or "").strip()
    r6_root = _private_path(r6_value) if r6_value else _discover_r6_run_root(reference=reference)
    return reference, r6_root, _response_store(str(r6_root))


def command_verify_campaign(args: argparse.Namespace) -> int:
    reference, r6_root, r6_store = _resolve_campaign_inputs(args)
    summary = campaign_preflight_summary(reference=reference, r6_store=r6_store)
    summary["r6_run_id"] = r6_root.name
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def command_run_campaign(args: argparse.Namespace) -> int:
    run = _open_run(args)
    try:
        reference, r6_root, r6_store = _resolve_campaign_inputs(args)
        status = execute_campaign(
            store=RunStore(run.artifact_root),
            reference=reference,
            r6_store=r6_store,
            timeout_sec=float(args.timeout_sec),
            max_attempts=int(args.max_attempts),
        )
    except ResumeRequired as exc:
        run.checkpoint(
            metadata={
                "command": "run-campaign",
                "state": "RESUME_REQUIRED",
                "detail": str(exc)[:512],
            }
        )
        print(f"RESUME_RUN={run.run_root}")
        return EXIT_RESUME
    except BaseException as exc:
        run.fail(exc, metadata={"command": "run-campaign"})
        raise
    _finish_run(
        run,
        command="run-campaign",
        summary={
            "state": status.get("state"),
            "completed_new_logical_slots": status.get("completed_new_logical_slots"),
            "expected_new_logical_slots": status.get("expected_new_logical_slots"),
            "expected_evidence_logical_slots": status.get("expected_evidence_logical_slots"),
            "reference_sha256": status.get("reference_sha256"),
            "r6_run_id": r6_root.name,
        },
    )
    return 0


def _response_store(run_root: str) -> RunStore:
    root = _private_path(run_root)
    artifacts = root / harness.ARTIFACT_DIRECTORY_NAME
    if not artifacts.is_dir():
        raise ExecutionError("Private response run has no managed artifacts directory.")
    return RunStore(artifacts)


def _response_stores(run_roots: list[str], *, required: bool = True) -> tuple[RunStore, ...]:
    if required and not run_roots:
        raise ExecutionError("At least one private response run is required.")
    stores: list[RunStore] = []
    seen_roots: set[Path] = set()
    for run_root in run_roots:
        store = _response_store(run_root)
        if store.root in seen_roots:
            raise ExecutionError("A private response run was supplied more than once.")
        seen_roots.add(store.root)
        stores.append(store)
    return tuple(stores)


def _completed_phase_status(path: str | Path, *, reference: Mapping[str, Any]) -> dict[str, Any]:
    status = _load_object(path)
    if status.get("state") != "WAITING_FOR_JUDGMENT":
        raise ExecutionError("Sampler rank requires a completed WAITING_FOR_JUDGMENT phase status.")
    if str(status.get("reference_sha256") or "") != str(reference.get("reference_sha256") or ""):
        raise ExecutionError("Sampler phase status belongs to a different frozen reference.")
    expected = status.get("expected_logical_slots")
    completed = status.get("completed_logical_slots")
    if (
        isinstance(expected, bool)
        or not isinstance(expected, int)
        or expected <= 0
        or isinstance(completed, bool)
        or not isinstance(completed, int)
        or completed != expected
    ):
        raise ExecutionError("Sampler phase status does not prove a complete response matrix.")
    return status


def _rank_outcome(scope: str, ranked: list[Mapping[str, Any]]) -> dict[str, str]:
    """Choose a next-step state without treating transport noise as quality loss."""

    if scope == "tuning":
        if not ranked:
            return {"status": "NO_RESULT"}
        if any(int(row.get("unjudged") or 0) for row in ranked):
            return {"status": "INSUFFICIENT_TRANSLATION_EVIDENCE"}
        return {
            "status": "PROVISIONAL_WINNER",
            "provisional_sampler_key": str(ranked[0].get("sampler_key") or ""),
        }
    return {"status": "HOLDOUT_REVIEWED"}


def command_build_judgment_packet(args: argparse.Namespace) -> int:
    reference = load_frozen_reference(_private_path(args.reference))
    all_records = collect_completed_records(
        _response_stores(args.response_run),
        reference=reference,
    )
    scope = str(args.scope)
    if scope == "holdout":
        tuning_report = _load_object(args.tuning_report)
        provisional = parse_sampler_tuple(args.provisional_tuple)
        baseline = parse_sampler_tuple(args.baseline_tuple)
        records = complete_scope_records(
            all_records,
            reference=reference,
            scope="holdout",
            sampler_keys=tuple(dict.fromkeys((provisional.key, baseline.key))),
        )
        action = lambda: open_holdout_packet(
            reference,
            records,
            provisional_sampler=provisional,
            baseline_sampler=baseline,
            tuning_report=tuning_report,
        )
    else:
        if not str(args.phase_status or "").strip():
            raise ExecutionError("Tuning judgment requires its completed phase status.")
        phase_status = _completed_phase_status(args.phase_status, reference=reference)
        sampler_keys = sampler_keys_from_phase_status(phase_status, reference=reference)
        records = complete_scope_records(
            all_records,
            reference=reference,
            scope="tuning",
            sampler_keys=sampler_keys,
        )
        action = lambda: build_blind_judgment_packet(
            reference,
            records,
            scope="tuning",
            allowed_sampler_keys=sampler_keys,
        )
    return _write_transform(
        args,
        command="build-judgment-packet",
        filename=f"judgment-{scope}-packet.json",
        action=action,
        summary=lambda result: {
            "scope": result.get("scope"),
            "pending_cluster_count": result.get("pending_cluster_count"),
            "automatic_verdict_count": len(result.get("automatic_verdicts") or []),
        },
    )


def _incremental_ledger(run: harness.ManagedArtifactRun, reference: Mapping[str, Any]) -> dict[str, Any]:
    path = run.artifact_root / INCREMENTAL_LEDGER_FILE
    if not path.exists():
        return new_incremental_ledger(reference)
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise ExecutionError("Incremental judgment ledger is not a JSON object.")
    ledger = dict(value)
    validate_incremental_ledger(ledger, reference=reference)
    return ledger


def _incremental_summary(
    *,
    run: harness.ManagedArtifactRun,
    ledger: Mapping[str, Any],
    packet: Mapping[str, Any] | None,
) -> dict[str, Any]:
    pending = ledger.get("pending_batch")
    return {
        "run_root": str(run.run_root),
        "judged_cluster_count": len(ledger.get("verdicts") or {}),
        "applied_batch_count": len(ledger.get("applied_batches") or []),
        "pending_batch": dict(pending) if isinstance(pending, Mapping) else None,
        "observed_response_count": int(packet.get("observed_response_count") or 0) if packet else 0,
        "pending_total_cluster_count": int(packet.get("pending_total_cluster_count") or 0) if packet else 0,
        "batch_cluster_count": int(packet.get("batch_cluster_count") or 0) if packet else 0,
        "unjudged_response_count": int(packet.get("unjudged_response_count") or 0) if packet else 0,
    }


def _checkpoint_incremental_error(
    run: harness.ManagedArtifactRun,
    *,
    command: str,
    error: BaseException,
) -> None:
    """Keep a durable judgment ledger resumable after an operator-visible error."""

    try:
        run.checkpoint(
            metadata={
                "command": command,
                "state": "ERROR_REQUIRES_INSPECTION",
                "error_type": type(error).__name__,
                "error_message": str(error)[:4096],
            }
        )
    except BaseException:
        # Preserve the original exception. The previous running manifest remains
        # resumable even when this best-effort diagnostic checkpoint cannot land.
        pass


def command_refresh_incremental_judgment(args: argparse.Namespace) -> int:
    """Create or refresh one stable blind batch without closing the live campaign."""

    run = _open_run(args)
    try:
        reference = load_frozen_reference(_private_path(args.reference))
        ledger = _incremental_ledger(run, reference)
        reuse_packet = str(args.reuse_packet or "").strip()
        reuse_decisions = str(args.reuse_decisions or "").strip()
        if bool(reuse_packet) != bool(reuse_decisions):
            raise ExecutionError("Reusable packet and decisions must be supplied together.")
        if reuse_packet:
            completed_packet = _load_object(reuse_packet)
            completed_decisions = _decision_map(
                reuse_decisions,
                collection_keys=("decisions", "rows"),
                id_key="cluster_id",
            )
            ledger = seed_ledger_from_completed_packet(
                ledger,
                reference=reference,
                packet=completed_packet,
                decisions=completed_decisions,
                source_label="temperature-r6-semantic-v4",
            )

        packet: dict[str, Any] | None = None
        pending = ledger.get("pending_batch")
        if isinstance(pending, Mapping):
            packet_file = str(pending.get("packet_file") or "")
            if Path(packet_file).name != packet_file:
                raise ExecutionError("Incremental judgment pending packet path is invalid.")
            value = read_json(run.artifact_root / packet_file)
            if not isinstance(value, Mapping):
                raise ExecutionError("Incremental judgment pending packet is unreadable.")
            packet = dict(value)
            validate_incremental_packet(packet, reference=reference)
        else:
            records = collect_completed_records(
                _response_stores(args.response_run),
                reference=reference,
                snapshot=True,
            )
            packet = build_incremental_judgment_packet(
                reference,
                records,
                ledger,
                batch_size=int(args.batch_size),
            )
            if packet.get("rows"):
                batch_number = ledger.get("next_batch_number")
                if isinstance(batch_number, bool) or not isinstance(batch_number, int) or batch_number <= 0:
                    raise ExecutionError("Incremental judgment ledger batch counter is invalid.")
                packet_file = f"judgment-batch-{batch_number:04d}.json"
                atomic_write_json(run.artifact_root / packet_file, packet)
                ledger = mark_pending_batch(
                    ledger,
                    reference=reference,
                    packet=packet,
                    packet_file=packet_file,
                )

        atomic_write_json(run.artifact_root / INCREMENTAL_LEDGER_FILE, ledger)
        summary = _incremental_summary(run=run, ledger=ledger, packet=packet)
        run.checkpoint(metadata={"command": "refresh-incremental-judgment", "summary": summary})
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    except BaseException as exc:
        _checkpoint_incremental_error(
            run,
            command="refresh-incremental-judgment",
            error=exc,
        )
        raise


def command_apply_incremental_judgment(args: argparse.Namespace) -> int:
    """Apply one complete blind decision batch and leave the ledger resumable."""

    if not str(args.resume_run or "").strip():
        raise ExecutionError("Applying incremental judgments requires the existing ledger run.")
    run = _open_run(args)
    try:
        reference = load_frozen_reference(_private_path(args.reference))
        ledger = _incremental_ledger(run, reference)
        pending = ledger.get("pending_batch")
        if not isinstance(pending, Mapping):
            raise ExecutionError("Incremental judgment ledger has no pending blind batch.")
        packet_file = str(pending.get("packet_file") or "")
        if Path(packet_file).name != packet_file:
            raise ExecutionError("Incremental judgment pending packet path is invalid.")
        packet_value = read_json(run.artifact_root / packet_file)
        if not isinstance(packet_value, Mapping):
            raise ExecutionError("Incremental judgment pending packet is unreadable.")
        packet = dict(packet_value)
        decisions = _decision_map(
            args.decisions,
            collection_keys=("decisions", "rows"),
            id_key="cluster_id",
        )
        ledger = apply_incremental_judgments(
            ledger,
            reference=reference,
            packet=packet,
            decisions=decisions,
            applied_utc=utc_now(),
        )
        decision_file = packet_file.removesuffix(".json") + "-decisions.json"
        atomic_write_json(
            run.artifact_root / decision_file,
            {
                "schema_version": packet.get("schema_version"),
                "rule_version": packet.get("rule_version"),
                "packet_id": packet.get("packet_id"),
                "packet_sha256": packet.get("packet_sha256"),
                "decisions": decisions,
            },
        )
        atomic_write_json(run.artifact_root / INCREMENTAL_LEDGER_FILE, ledger)
        summary = _incremental_summary(run=run, ledger=ledger, packet=None)
        summary["applied_cluster_count"] = len(decisions)
        summary["decision_file"] = decision_file
        run.checkpoint(metadata={"command": "apply-incremental-judgment", "summary": summary})
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    except BaseException as exc:
        _checkpoint_incremental_error(
            run,
            command="apply-incremental-judgment",
            error=exc,
        )
        raise


def command_analyze_final_campaign(args: argparse.Namespace) -> int:
    """Require terminal cleanup evidence and analyze every sealed response."""

    run = _open_run(args)
    try:
        reference = load_frozen_reference(_private_path(args.reference))
        r6_root = _private_path(args.r6_run)
        campaign_root = _private_path(args.campaign_run)
        ledger_root = _private_path(args.ledger_run)
        r6_store = _response_store(str(r6_root))
        campaign_store = _response_store(str(campaign_root))
        r6_records = collect_completed_records((r6_store,), reference=reference)
        campaign_records = collect_completed_records((campaign_store,), reference=reference)
        ledger = _load_object(
            ledger_root / harness.ARTIFACT_DIRECTORY_NAME / INCREMENTAL_LEDGER_FILE
        )
        gates = _load_object(args.gate_manifest)
        evidence = validate_final_campaign_evidence(
            reference=reference,
            r6_records=r6_records,
            campaign_records=campaign_records,
            r6_manifest=_load_object(r6_root / harness.MANIFEST_FILE_NAME),
            campaign_manifest=_load_object(campaign_root / harness.MANIFEST_FILE_NAME),
            campaign_status=_load_object(
                campaign_root / harness.ARTIFACT_DIRECTORY_NAME / "campaign-status.json"
            ),
            campaign_plan_artifact=_load_object(
                campaign_root / harness.ARTIFACT_DIRECTORY_NAME / "campaign-plan.json"
            ),
        )
        public, private = build_final_campaign_analysis(
            reference=reference,
            records=tuple(r6_records) + tuple(campaign_records),
            ledger=ledger,
            gates=gates,
            evidence_summary=evidence,
        )
        atomic_write_json(run.artifact_root / "final-campaign-analysis-public.json", public)
        atomic_write_json(run.artifact_root / "final-campaign-analysis-private.json", private)
        _finish_run(
            run,
            command="analyze-final-campaign",
            summary={
                "state": public.get("state"),
                "analysis_sha256": public.get("analysis_sha256"),
                "sampler_count": evidence.get("sampler_count"),
                "total_response_count": evidence.get("total_response_count"),
                "provisional_candidate_sampler_key": public.get(
                    "provisional_candidate_sampler_key"
                ),
                "product_promotion_allowed": False,
            },
        )
        return 0
    except BaseException as exc:
        run.fail(exc, metadata={"command": "analyze-final-campaign"})
        raise


def command_rank(args: argparse.Namespace) -> int:
    reference = load_frozen_reference(_private_path(args.reference))
    all_records = collect_completed_records(
        _response_stores(args.response_run),
        reference=reference,
    )
    packet = _load_object(args.packet)
    if packet.get("scope") != args.scope:
        raise ExecutionError("Blind judgment packet scope does not match the requested rank scope.")
    decisions = _decision_map(args.decisions, collection_keys=("decisions", "rows"), id_key="cluster_id")
    phase_status = _completed_phase_status(args.phase_status, reference=reference)
    if args.scope == "tuning":
        sampler_keys = sampler_keys_from_phase_status(phase_status, reference=reference)
    else:
        if not str(args.provisional_tuple or "").strip() or not str(args.baseline_tuple or "").strip():
            raise ExecutionError("Holdout rank requires the provisional and baseline sampler tuples.")
        sampler_keys = tuple(
            dict.fromkeys(
                (
                    parse_sampler_tuple(args.provisional_tuple).key,
                    parse_sampler_tuple(args.baseline_tuple).key,
                )
            )
        )
    records = complete_scope_records(
        all_records,
        reference=reference,
        scope=args.scope,
        sampler_keys=sampler_keys,
    )
    verdicts = bind_cluster_verdicts_to_records(records, packet, decisions)
    ranked = rank_sampler_results(records, verdicts, scope=args.scope)
    report = build_phase_report(phase_status=phase_status, ranked=ranked, scope=args.scope)
    report.update(_rank_outcome(args.scope, ranked))
    return _write_transform(
        args,
        command="rank",
        filename=f"report-{args.scope}.json",
        action=lambda: report,
        summary=lambda result: {
            "scope": result.get("rank", {}).get("scope") if isinstance(result.get("rank"), Mapping) else "",
            "status": result.get("status"),
            "report_sha256": result.get("report_sha256"),
            "row_count": len(result.get("rank", {}).get("rows") or []) if isinstance(result.get("rank"), Mapping) else 0,
        },
    )


def command_render_public_summary(args: argparse.Namespace) -> int:
    """Render a sanitized report *inside* a managed run for manual publication review."""

    report = _load_object(args.report)
    run = _open_run(args)
    try:
        destination = run.artifact_root / "latest-report-ko.md"
        destination.write_text(render_public_markdown(report), encoding="utf-8", newline="\n")
        _finish_run(
            run,
            command="render-public-summary",
            summary={"report_sha256": report.get("report_sha256", "")},
        )
        return 0
    except BaseException as exc:
        run.fail(exc, metadata={"command": "render-public-summary"})
        raise


def command_render_private_review(args: argparse.Namespace) -> int:
    packet = _load_object(args.packet)
    run = _open_run(args)
    try:
        destination = run.artifact_root / "reference-review.html"
        destination.write_text(
            render_private_review_html(packet, title=str(args.title)),
            encoding="utf-8",
            newline="\n",
        )
        _finish_run(
            run,
            command="render-private-review",
            summary={"row_count": len(packet.get("rows") or [])},
        )
        return 0
    except BaseException as exc:
        run.fail(exc, metadata={"command": "render-private-review"})
        raise


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", default="", help="Optional safe managed-run id.")
    parser.add_argument("--resume-run", default="", help="Interrupted private managed run root.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    draft = subparsers.add_parser("build-reference-draft")
    draft.add_argument("--source-manifest", required=True)
    _add_run_options(draft)
    draft.set_defaults(handler=command_build_reference_draft)

    canonical = subparsers.add_parser("apply-canonical")
    canonical.add_argument("--reference", required=True)
    canonical.add_argument("--answers", required=True)
    _add_run_options(canonical)
    canonical.set_defaults(handler=command_apply_canonical)

    blind = subparsers.add_parser("build-blind-review")
    blind.add_argument("--reference", required=True)
    _add_run_options(blind)
    blind.set_defaults(handler=command_build_blind_review)

    apply_blind = subparsers.add_parser("apply-blind-review")
    apply_blind.add_argument("--reference", required=True)
    apply_blind.add_argument("--packet", required=True)
    apply_blind.add_argument("--decisions", required=True)
    _add_run_options(apply_blind)
    apply_blind.set_defaults(handler=command_apply_blind_review)

    resolution = subparsers.add_parser("build-resolution-packet")
    resolution.add_argument("--reference", required=True)
    _add_run_options(resolution)
    resolution.set_defaults(handler=command_build_resolution_packet)

    apply_resolution = subparsers.add_parser("apply-resolutions")
    apply_resolution.add_argument("--reference", required=True)
    apply_resolution.add_argument("--resolutions", required=True)
    _add_run_options(apply_resolution)
    apply_resolution.set_defaults(handler=command_apply_resolutions)

    sample = subparsers.add_parser("build-user-sample")
    sample.add_argument("--reference", required=True)
    _add_run_options(sample)
    sample.set_defaults(handler=command_build_user_sample)

    freeze = subparsers.add_parser("freeze-reference")
    freeze.add_argument("--reference", required=True)
    freeze.add_argument("--user-sample", required=True)
    freeze.add_argument("--user-approval", required=True)
    _add_run_options(freeze)
    freeze.set_defaults(handler=command_freeze_reference)

    run = subparsers.add_parser("run-phase")
    run.add_argument("--reference", required=True)
    run.add_argument("--phase", required=True, choices=("temperature", "joint_top_p_top_k", "min_p"))
    run.add_argument("--selected-temperature", action="append", default=[])
    run.add_argument("--selected-tuple", action="append", default=[])
    run.add_argument("--timeout-sec", type=float, default=180.0)
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument(
        "--prior-response-run",
        action="append",
        default=[],
        help="Private completed response run used only for declared reused arms.",
    )
    _add_run_options(run)
    run.set_defaults(handler=command_run_phase)

    verify_campaign = subparsers.add_parser("verify-campaign")
    verify_campaign.add_argument(
        "--reference",
        default="",
        help="Optional private frozen reference override; default discovery requires exactly one.",
    )
    verify_campaign.add_argument(
        "--r6-run",
        default="",
        help="Optional private r6 managed-run override; default discovery requires exactly one.",
    )
    verify_campaign.set_defaults(handler=command_verify_campaign)

    campaign = subparsers.add_parser("run-campaign")
    campaign.add_argument(
        "--reference",
        default="",
        help="Optional private frozen reference override; default discovery requires exactly one.",
    )
    campaign.add_argument(
        "--r6-run",
        default="",
        help="Optional private r6 managed-run override; default discovery requires exactly one.",
    )
    campaign.add_argument("--timeout-sec", type=float, default=180.0)
    campaign.add_argument("--max-attempts", type=int, default=3)
    _add_run_options(campaign)
    campaign.set_defaults(handler=command_run_campaign, campaign=True)

    judgment = subparsers.add_parser("build-judgment-packet")
    judgment.add_argument("--reference", required=True)
    judgment.add_argument("--response-run", action="append", required=True)
    judgment.add_argument("--scope", choices=("tuning", "holdout"), default="tuning")
    judgment.add_argument("--phase-status", default="")
    judgment.add_argument("--tuning-report", default="")
    judgment.add_argument("--provisional-tuple", default="")
    judgment.add_argument("--baseline-tuple", default="")
    _add_run_options(judgment)
    judgment.set_defaults(handler=command_build_judgment_packet)

    incremental = subparsers.add_parser("refresh-incremental-judgment")
    incremental.add_argument("--reference", required=True)
    incremental.add_argument("--response-run", action="append", required=True)
    incremental.add_argument("--reuse-packet", default="")
    incremental.add_argument("--reuse-decisions", default="")
    incremental.add_argument("--batch-size", type=int, default=100)
    _add_run_options(incremental)
    incremental.set_defaults(handler=command_refresh_incremental_judgment)

    apply_incremental = subparsers.add_parser("apply-incremental-judgment")
    apply_incremental.add_argument("--reference", required=True)
    apply_incremental.add_argument("--decisions", required=True)
    _add_run_options(apply_incremental)
    apply_incremental.set_defaults(handler=command_apply_incremental_judgment)

    final_analysis = subparsers.add_parser("analyze-final-campaign")
    final_analysis.add_argument("--reference", required=True)
    final_analysis.add_argument("--r6-run", required=True)
    final_analysis.add_argument("--campaign-run", required=True)
    final_analysis.add_argument("--ledger-run", required=True)
    final_analysis.add_argument("--gate-manifest", required=True)
    _add_run_options(final_analysis)
    final_analysis.set_defaults(handler=command_analyze_final_campaign)

    rank = subparsers.add_parser("rank")
    rank.add_argument("--reference", required=True)
    rank.add_argument("--response-run", action="append", required=True)
    rank.add_argument("--packet", required=True)
    rank.add_argument("--decisions", required=True)
    rank.add_argument("--phase-status", required=True)
    rank.add_argument("--scope", choices=("tuning", "holdout"), required=True)
    rank.add_argument("--provisional-tuple", default="")
    rank.add_argument("--baseline-tuple", default="")
    _add_run_options(rank)
    rank.set_defaults(handler=command_rank)

    render = subparsers.add_parser("render-public-summary")
    render.add_argument("--report", required=True)
    _add_run_options(render)
    render.set_defaults(handler=command_render_public_summary)

    private_review = subparsers.add_parser("render-private-review")
    private_review.add_argument("--packet", required=True)
    private_review.add_argument("--title", default="Gemma sampler quality v2 private review")
    _add_run_options(private_review)
    private_review.set_defaults(handler=command_render_private_review)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        CorpusError,
        ExecutionError,
        JudgmentError,
        ProtocolError,
        ReviewBoardError,
        StorageError,
        harness.ArtifactHarnessError,
    ) as exc:
        print(f"[GEMMA-SAMPLER-V2] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
