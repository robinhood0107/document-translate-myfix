"""One-shot, resumable execution for the approved Gemma sampler v2 campaign.

This module deliberately leaves the historical phase runner intact.  The
single campaign has one immutable matrix, reads r6 as provenance only, keeps
Gemma loaded for its complete invocation, then requires terminal Router/GPU
cleanup before it exposes a final-judgment state.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from . import execution
from .corpus import reference_summary
from .judgment import ResponseVerdict, validate_response_envelope
from .protocol import (
    CAMPAIGN_PROTOCOL_VERSION,
    CampaignPlan,
    SEEDS,
    SamplerArm,
    canonical_sha256,
    campaign_plan,
    campaign_response_counts,
    fixed_request_contract_payload,
    new_arms,
)
from .runtime import RouterGemmaReplayRuntime, RuntimeErrorV2, TransientReplayError
from .storage import RunStore, atomic_write_json, read_json, utc_now


CAMPAIGN_EXECUTION_SCHEMA_VERSION = "gemma-sampler-campaign-execution-v1"
BASELINE_ETA_SECONDS = 102_544
BASELINE_ETA_LOW_SECONDS = 98_400
BASELINE_ETA_HIGH_SECONDS = 119_340


@dataclass(frozen=True)
class CampaignReuseValidation:
    """Private evidence that r6 can be reused without a new GPU request."""

    plan: CampaignPlan
    cases: tuple[dict[str, Any], ...]
    engines: Mapping[str, Any]
    request_identities: Mapping[str, Mapping[str, Any]]
    r6_records: Mapping[tuple[str, int, str], Mapping[str, Any]]
    r6_contract: Mapping[str, Any]


def _read_mapping(path: Path, *, message: str) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise execution.ExecutionError(message)
    return dict(value)


def _r6_runtime_contract(store: RunStore) -> dict[str, Any]:
    contract = _read_mapping(
        store.root / "runtime-contract.json",
        message="r6 has no readable Router runtime contract.",
    )
    required = {
        "fingerprint",
        "image_ref",
        "binary_version",
        "command_sha256",
        "preset_sha256",
        "effective_context_size",
        "fixed_request_contract_sha256",
    }
    if any(not str(contract.get(key) or "") for key in required):
        raise execution.ExecutionError("r6 Router runtime contract is incomplete.")
    if str(contract.get("effective_context_size") or "") != "4096":
        raise execution.ExecutionError("r6 Router context size differs from the fixed campaign contract.")
    if str(contract.get("fixed_request_contract_sha256") or "") != canonical_sha256(
        fixed_request_contract_payload()
    ):
        raise execution.ExecutionError("r6 fixed request contract differs from the campaign contract.")
    return contract


def _assert_r6_manifest_completed(store: RunStore) -> None:
    manifest_path = store.root.parent / "artifact-manifest.json"
    manifest = _read_mapping(manifest_path, message="r6 managed artifact manifest is unreadable.")
    if str(manifest.get("status") or "") != "completed":
        raise execution.ExecutionError("r6 managed artifact manifest is not completed.")


def _campaign_request_identities(
    *,
    store: RunStore | None,
    cases: Sequence[Mapping[str, Any]],
    plan: CampaignPlan,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    engines: dict[str, Any] = {}
    identities: dict[str, Mapping[str, Any]] = {}
    identity_sampler = plan.temperature_arms[0].sampler
    for case in cases:
        source_language = execution._source_language_name(case)
        engine = engines.get(source_language)
        if engine is None:
            engine = execution._build_engine(source_language)
            engines[source_language] = engine
        _payload, identity = execution._case_request(
            engine,
            case,
            identity_sampler,
            SEEDS[0],
        )
        case_id = str(case.get("case_id") or "")
        if store is not None:
            store.bind_request_contract(case_id=case_id, identity=identity)
        identities[case_id] = identity
    return engines, identities


def validate_r6_reuse(
    *,
    reference: Mapping[str, Any],
    r6_store: RunStore,
    campaign_store: RunStore | None = None,
) -> CampaignReuseValidation:
    """Validate all r6 provenance before Docker is touched.

    This is intentionally callable by the BAT's verify-only command.  It
    never writes to the r6 run and it does not start Docker or the Router.
    """

    plan = campaign_plan()
    cases = tuple(execution._case_list(reference))
    reference_sha256 = str(reference.get("reference_sha256") or "")
    r6_selection = execution.PhaseSelection("temperature", plan.temperature_arms)
    expected_slots = execution._expected_phase_slots(r6_selection, cases)
    status = _read_mapping(
        r6_store.phase_status_path("temperature"),
        message="r6 temperature status is unreadable.",
    )
    execution._assert_completed_phase_status(
        status,
        selection=r6_selection,
        reference_sha256=reference_sha256,
        expected_slots=expected_slots,
        store=r6_store,
    )
    _assert_r6_manifest_completed(r6_store)
    r6_contract = _r6_runtime_contract(r6_store)
    engines, request_identities = _campaign_request_identities(
        store=campaign_store,
        cases=cases,
        plan=plan,
    )
    r6_records = execution._required_reused_records(
        execution.iter_compatible_completed_records((r6_store,), reference=reference),
        reused_arms=plan.temperature_arms,
        cases=cases,
    )
    execution._assert_reused_contracts(
        r6_records,
        runtime_fingerprint=str(r6_contract["fingerprint"]),
        request_identities=request_identities,
    )
    return CampaignReuseValidation(
        plan=plan,
        cases=cases,
        engines=engines,
        request_identities=request_identities,
        r6_records=r6_records,
        r6_contract=r6_contract,
    )


def campaign_preflight_summary(
    *,
    reference: Mapping[str, Any],
    r6_store: RunStore,
) -> dict[str, Any]:
    """Read-only readiness proof used by the CUDA13 BAT and launcher checks."""

    validation = validate_r6_reuse(reference=reference, r6_store=r6_store)
    counts = campaign_response_counts()
    return {
        "schema_version": CAMPAIGN_EXECUTION_SCHEMA_VERSION,
        "state": "READY_TO_RUN",
        "campaign_plan_sha256": validation.plan.sha256,
        "reference_sha256": str(reference.get("reference_sha256") or ""),
        "r6_runtime_fingerprint": str(validation.r6_contract.get("fingerprint") or ""),
        "case_count": len(validation.cases),
        "seeds": list(SEEDS),
        "response_counts": counts,
        "initial_eta_seconds": BASELINE_ETA_SECONDS,
        "initial_eta_low_seconds": BASELINE_ETA_LOW_SECONDS,
        "initial_eta_high_seconds": BASELINE_ETA_HIGH_SECONDS,
    }


def _campaign_plan_payload(
    *,
    validation: CampaignReuseValidation,
    reference: Mapping[str, Any],
    r6_store: RunStore,
) -> dict[str, Any]:
    return {
        "schema_version": CAMPAIGN_EXECUTION_SCHEMA_VERSION,
        "campaign_plan_sha256": validation.plan.sha256,
        "plan": validation.plan.payload(),
        "reference_sha256": str(reference.get("reference_sha256") or ""),
        "r6": {
            "run_id": r6_store.root.parent.name,
            "runtime_fingerprint": str(validation.r6_contract.get("fingerprint") or ""),
            "fixed_request_contract_sha256": str(
                validation.r6_contract.get("fixed_request_contract_sha256") or ""
            ),
        },
        "response_counts": campaign_response_counts(),
    }


def _assert_or_write_campaign_plan(
    *,
    store: RunStore,
    payload: Mapping[str, Any],
) -> None:
    destination = store.root / "campaign-plan.json"
    if destination.exists():
        previous = _read_mapping(destination, message="Campaign plan artifact is unreadable.")
        if previous != dict(payload):
            raise execution.ExecutionError("Resumed campaign plan differs from the sealed single-campaign plan.")
        return
    atomic_write_json(destination, dict(payload))


def _existing_progress(store: RunStore) -> dict[str, Any]:
    if not store.progress_path.exists():
        return {}
    value = read_json(store.progress_path)
    return dict(value) if isinstance(value, Mapping) else {}


def _integer(value: Any) -> int:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


class _CampaignProgress:
    def __init__(self, *, store: RunStore, validation: CampaignReuseValidation) -> None:
        self.store = store
        self.validation = validation
        self.counts = campaign_response_counts()
        previous = _existing_progress(store)
        raw_attempts = previous.get("attempt_counts")
        raw_attempts = raw_attempts if isinstance(raw_attempts, Mapping) else {}
        self.attempt_counts = {
            "valid": store.completed_count(),
            "retry": _integer(raw_attempts.get("retry")),
            "timeout": _integer(raw_attempts.get("timeout")),
            "indeterminate": _integer(raw_attempts.get("indeterminate")),
        }
        self.previous_active_seconds = float(previous.get("active_elapsed_seconds") or 0.0)
        self.backoff_seconds = float(previous.get("backoff_elapsed_seconds") or 0.0)
        self.previous_backoff_seconds = self.backoff_seconds
        self.started_at = time.monotonic()

    def record(self, outcome: str) -> None:
        if outcome in self.attempt_counts:
            self.attempt_counts[outcome] += 1

    def add_backoff(self, seconds: float) -> None:
        self.backoff_seconds += max(0.0, float(seconds))

    def write(
        self,
        *,
        state: str,
        phase: str = "",
        stage_completed: int = 0,
        stage_expected: int = 0,
        arm: SamplerArm | None = None,
        seed: int | None = None,
        case_position: int | None = None,
        detail: str = "",
    ) -> None:
        elapsed_since_start = max(0.0, time.monotonic() - self.started_at)
        backoff_since_start = max(0.0, self.backoff_seconds - self.previous_backoff_seconds)
        elapsed = self.previous_active_seconds + max(0.0, elapsed_since_start - backoff_since_start)
        payload: dict[str, Any] = {
            "schema_version": CAMPAIGN_EXECUTION_SCHEMA_VERSION,
            "state": state,
            "phase": phase,
            # Keep the legacy key populated so older monitors show sensible
            # numbers when attached to a new campaign.
            "completed_logical_slots": self.store.completed_count(),
            "campaign_completed_logical_slots": self.store.completed_count(),
            "campaign_expected_logical_slots": self.counts["total_new"],
            "reused_logical_slots": self.counts["r6_reused"],
            "evidence_expected_logical_slots": self.counts["total_evidence"],
            "stage_completed_logical_slots": stage_completed,
            "stage_expected_logical_slots": stage_expected,
            "case_count": len(self.validation.cases),
            "attempt_counts": dict(self.attempt_counts),
            "active_elapsed_seconds": round(elapsed, 3),
            "backoff_elapsed_seconds": round(self.backoff_seconds, 3),
            "initial_eta_seconds": BASELINE_ETA_SECONDS,
            "initial_eta_low_seconds": BASELINE_ETA_LOW_SECONDS,
            "initial_eta_high_seconds": BASELINE_ETA_HIGH_SECONDS,
            "updated_utc": utc_now(),
            "worker_pid": os.getpid(),
        }
        if arm is not None:
            payload["current_sampler"] = arm.sampler.payload()
            payload["current_arm_key"] = arm.key
        if seed is not None:
            payload["current_seed"] = int(seed)
        if case_position is not None:
            payload["current_case_position"] = int(case_position)
        if detail:
            payload["detail"] = detail[:512]
        self.store.update_progress(payload)


def _expected_stage_slots(
    *,
    phase: str,
    arms: Sequence[SamplerArm],
    cases: Sequence[Mapping[str, Any]],
) -> set[str]:
    selection = execution.PhaseSelection(phase, tuple(arms))
    return execution._expected_phase_slots(selection, cases)


def _assert_campaign_slots(
    *,
    store: RunStore,
    expected_slots: set[str],
) -> set[str]:
    # The fsynced completion index is the resumable-run authority.  Reading
    # every raw response for every completion would turn a 124k-slot campaign
    # into quadratic disk I/O, so full raw-envelope validation stays in the
    # later judgment path where it is required.
    observed = set(store.completed_index())
    if not observed.issubset(expected_slots):
        raise execution.ExecutionError("Campaign store contains a response outside the sealed new matrix.")
    return observed


def _write_runtime_contract(store: RunStore, contract: Mapping[str, Any]) -> None:
    destination = store.root / "runtime-contract.json"
    if destination.exists():
        previous = _read_mapping(destination, message="Campaign runtime contract is unreadable.")
        if previous.get("fingerprint") != contract.get("fingerprint"):
            raise execution.ExecutionError("Resumed campaign has a different Router runtime fingerprint.")
        return
    atomic_write_json(destination, dict(contract))


def _run_stage(
    *,
    store: RunStore,
    runtime: RouterGemmaReplayRuntime,
    validation: CampaignReuseValidation,
    phase: str,
    arms: Sequence[SamplerArm],
    timeout_sec: float,
    max_attempts: int,
    runtime_contract: Mapping[str, Any],
    reference_sha256: str,
    progress: _CampaignProgress,
) -> None:
    execution_arms = tuple(new_arms(arms))
    expected_stage_slots = _expected_stage_slots(
        phase=phase,
        arms=execution_arms,
        cases=validation.cases,
    )
    stage_expected = len(expected_stage_slots)
    completed_stage_slots = {
        slot for slot in store.completed_index() if slot in expected_stage_slots
    }
    phase_state = "RUNNING_JOINT" if phase == "joint_top_p_top_k" else "RUNNING_MIN_P"
    for arm in execution_arms:
        for seed_index, seed in enumerate(SEEDS):
            order_name, ordered_cases = execution._order_for_seed(validation.cases, seed_index)
            run_name = f"seed-{seed}-{order_name}"
            for case_position, case in enumerate(ordered_cases, start=1):
                case_id = str(case.get("case_id") or "")
                slot = execution._logical_slot(phase, arm, seed, case_id)
                if store.is_completed(slot):
                    continue
                execution._preflight_free_space(store.root)
                stage_completed = len(completed_stage_slots)
                progress.write(
                    state=phase_state,
                    phase=phase,
                    stage_completed=stage_completed,
                    stage_expected=stage_expected,
                    arm=arm,
                    seed=seed,
                    case_position=case_position,
                )
                engine = validation.engines[execution._source_language_name(case)]
                payload, request_identity = execution._case_request(engine, case, arm.sampler, seed)
                last_transient = ""
                for attempt in range(1, max_attempts + 1):
                    execution._record_attempt(
                        store,
                        phase=phase,
                        arm=arm,
                        seed=seed,
                        case_id=case_id,
                        attempt=attempt,
                        outcome="started",
                    )
                    try:
                        replay = runtime.request(payload, timeout_sec=timeout_sec)
                    except TransientReplayError as exc:
                        last_transient = str(exc)
                        execution._record_attempt(
                            store,
                            phase=phase,
                            arm=arm,
                            seed=seed,
                            case_id=case_id,
                            attempt=attempt,
                            outcome="indeterminate",
                            detail=last_transient,
                        )
                        progress.record("indeterminate")
                        if "timeout" in last_transient.casefold():
                            progress.record("timeout")
                        if attempt < max_attempts:
                            progress.record("retry")
                            pause = float(2 ** (attempt - 1))
                            progress.write(
                                state=phase_state,
                                phase=phase,
                                stage_completed=stage_completed,
                                stage_expected=stage_expected,
                                arm=arm,
                                seed=seed,
                                case_position=case_position,
                                detail="transient retry backoff",
                            )
                            time.sleep(pause)
                            progress.add_backoff(pause)
                            continue
                        raise execution.ResumeRequired(
                            "Transient sampler worker failure requires CUDA13 BAT resume."
                        ) from exc
                    except RuntimeErrorV2 as exc:
                        execution._record_attempt(
                            store,
                            phase=phase,
                            arm=arm,
                            seed=seed,
                            case_id=case_id,
                            attempt=attempt,
                            outcome="fatal",
                            detail=str(exc),
                        )
                        raise execution.ExecutionError(
                            "Sampler runtime contract or drain gate failed closed."
                        ) from exc
                    verdict: ResponseVerdict = validate_response_envelope(replay.envelope)
                    record = {
                        "schema_version": CAMPAIGN_EXECUTION_SCHEMA_VERSION,
                        "status": "complete",
                        "recorded_utc": utc_now(),
                        "phase": phase,
                        "arm_key": arm.key,
                        "sampler": arm.sampler.payload(),
                        "seed": seed,
                        "order": order_name,
                        "case_position": case_position,
                        "case_id": case_id,
                        "split": str(case.get("split") or ""),
                        "reference_sha256": reference_sha256,
                        "logical_slot": slot,
                        "request_identity": request_identity,
                        "request": payload,
                        "response": dict(replay.envelope),
                        "response_validation": verdict.payload(),
                        "translation": verdict.translation if verdict.status == "VALID" else "",
                        "latency_ms": replay.latency_ms,
                        "completion_tokens": replay.completion_tokens,
                        "runtime_fingerprint": runtime_contract["fingerprint"],
                    }
                    stored = store.record_case_if_first(
                        phase=phase,
                        arm=arm.key,
                        run=run_name,
                        case_id=case_id,
                        logical_slot=slot,
                        payload=record,
                    )
                    if stored:
                        execution._record_attempt(
                            store,
                            phase=phase,
                            arm=arm,
                            seed=seed,
                            case_id=case_id,
                            attempt=attempt,
                            outcome="complete",
                        )
                        progress.record("valid")
                        completed_stage_slots.add(slot)
                    break


def execute_campaign(
    *,
    store: RunStore,
    reference: Mapping[str, Any],
    r6_store: RunStore,
    timeout_sec: float = 180.0,
    max_attempts: int = 3,
    runtime: RouterGemmaReplayRuntime | None = None,
) -> dict[str, Any]:
    """Run the exact fixed campaign or safely resume its unfinished slots."""

    if max_attempts < 1:
        raise execution.ExecutionError("Sampler v2 requires at least one HTTP attempt.")
    execution._preflight_free_space(store.root)
    validation = validate_r6_reuse(
        reference=reference,
        r6_store=r6_store,
        campaign_store=store,
    )
    plan_payload = _campaign_plan_payload(
        validation=validation,
        reference=reference,
        r6_store=r6_store,
    )
    _assert_or_write_campaign_plan(store=store, payload=plan_payload)
    expected_joint_slots = _expected_stage_slots(
        phase="joint_top_p_top_k",
        arms=validation.plan.new_joint_arms,
        cases=validation.cases,
    )
    expected_min_p_slots = _expected_stage_slots(
        phase="min_p",
        arms=validation.plan.new_min_p_arms,
        cases=validation.cases,
    )
    expected_all_slots = expected_joint_slots | expected_min_p_slots
    if len(expected_all_slots) != campaign_response_counts()["total_new"]:
        raise execution.ExecutionError("Campaign matrix does not match its approved new-response total.")
    _assert_campaign_slots(store=store, expected_slots=expected_all_slots)
    progress = _CampaignProgress(store=store, validation=validation)
    progress.write(state="VALIDATING_REUSE")

    active_runtime = runtime or RouterGemmaReplayRuntime()
    started_here = False
    raised: BaseException | None = None
    cleanup_error: BaseException | None = None
    runtime_contract: dict[str, Any] | None = None
    try:
        active_runtime.start()
        started_here = True
        runtime_contract = execution._assert_runtime_contract(active_runtime)
        if str(runtime_contract.get("fingerprint") or "") != str(
            validation.r6_contract.get("fingerprint") or ""
        ):
            raise execution.ExecutionError(
                "Current Router runtime fingerprint differs from read-only r6 provenance."
            )
        execution._assert_reused_contracts(
            validation.r6_records,
            runtime_fingerprint=str(runtime_contract["fingerprint"]),
            request_identities=validation.request_identities,
        )
        _write_runtime_contract(store, runtime_contract)
        _run_stage(
            store=store,
            runtime=active_runtime,
            validation=validation,
            phase="joint_top_p_top_k",
            arms=validation.plan.joint_arms,
            timeout_sec=timeout_sec,
            max_attempts=max_attempts,
            runtime_contract=runtime_contract,
            reference_sha256=str(reference.get("reference_sha256") or ""),
            progress=progress,
        )
        _run_stage(
            store=store,
            runtime=active_runtime,
            validation=validation,
            phase="min_p",
            arms=validation.plan.min_p_arms,
            timeout_sec=timeout_sec,
            max_attempts=max_attempts,
            runtime_contract=runtime_contract,
            reference_sha256=str(reference.get("reference_sha256") or ""),
            progress=progress,
        )
        observed = _assert_campaign_slots(store=store, expected_slots=expected_all_slots)
        if observed != expected_all_slots:
            raise execution.ResumeRequired(
                "Campaign stopped before every new logical slot had one complete response."
            )
        progress.write(
            state="RELEASING",
            phase="min_p",
            stage_completed=len(expected_min_p_slots),
            stage_expected=len(expected_min_p_slots),
        )
    except BaseException as exc:  # Preserve a retry/fatal cause through cleanup.
        raised = exc
    finally:
        if started_here:
            try:
                active_runtime.close()
            except BaseException as exc:  # release proof failure is terminal
                cleanup_error = exc

    if cleanup_error is not None:
        progress.write(state="RELEASE_FAILED", detail=str(cleanup_error))
        if raised is not None:
            raise execution.ExecutionError(
                "Campaign execution failed and terminal Router/GPU cleanup also failed."
            ) from cleanup_error
        raise execution.ExecutionError("Campaign terminal Router/GPU cleanup failed.") from cleanup_error
    if raised is not None:
        if isinstance(raised, execution.ResumeRequired):
            progress.write(state="RESUME_REQUIRED", detail=str(raised))
            raise raised
        progress.write(state="FAILED_CLOSED", detail=str(raised))
        raise raised

    status = {
        "schema_version": CAMPAIGN_EXECUTION_SCHEMA_VERSION,
        "state": "WAITING_FOR_FINAL_JUDGMENT",
        "campaign_plan_sha256": validation.plan.sha256,
        "reference_sha256": str(reference.get("reference_sha256") or ""),
        "reference_summary": reference_summary(reference),
        "r6_run_id": r6_store.root.parent.name,
        "r6_runtime_fingerprint": str(validation.r6_contract.get("fingerprint") or ""),
        "runtime_fingerprint": str(runtime_contract.get("fingerprint") if runtime_contract else ""),
        "expected_new_logical_slots": len(expected_all_slots),
        "completed_new_logical_slots": len(expected_all_slots),
        "reused_logical_slots": campaign_response_counts()["r6_reused"],
        "expected_evidence_logical_slots": campaign_response_counts()["total_evidence"],
        "joint_arms": [arm.payload() for arm in validation.plan.joint_arms],
        "min_p_arms": [arm.payload() for arm in validation.plan.min_p_arms],
        "sampler_keys": list(validation.plan.sampler_keys),
        "seeds": list(SEEDS),
        "completed_utc": utc_now(),
    }
    atomic_write_json(store.root / "campaign-status.json", status)
    progress.write(
        state="WAITING_FOR_FINAL_JUDGMENT",
        phase="min_p",
        stage_completed=len(expected_min_p_slots),
        stage_expected=len(expected_min_p_slots),
    )
    return status
