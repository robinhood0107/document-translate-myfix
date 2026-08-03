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
from scripts.gemma_sampler_quality_v2.judgment import (  # noqa: E402
    JudgmentError,
    bind_cluster_verdicts_to_records,
    build_blind_judgment_packet,
    open_holdout_packet,
    rank_sampler_results,
)
from scripts.gemma_sampler_quality_v2.report import (  # noqa: E402
    build_phase_report,
    render_public_markdown,
)
from scripts.gemma_sampler_quality_v2.review import ReviewBoardError, render_private_review_html  # noqa: E402
from scripts.gemma_sampler_quality_v2.storage import StorageError, atomic_write_json, read_json  # noqa: E402
from scripts.gemma_sampler_quality_v2.protocol import ProtocolError  # noqa: E402


FAMILY = "gemma-sampler-quality-v2"
CATEGORY = "10-gemma-translation"
EXIT_RESUME = 75


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
            if not str(getattr(args, "phase", "") or "").strip():
                raise
            return harness.ManagedArtifactRun.recover_failed_atomic_replace(
                run_root,
                command="run-phase",
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
