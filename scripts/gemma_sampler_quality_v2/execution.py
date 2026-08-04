"""Fail-closed execution and resume logic for the sampler-quality v2 matrix."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, Mapping, Sequence

from scripts.benchmark_gemma_translation_only_matrix import (
    build_engine,
    build_single_block_translation_request,
)

from .corpus import CorpusError, REFERENCE_SCHEMA_VERSION, reference_summary
from .judgment import (
    RESPONSE_VALIDATION_SCHEMA_VERSION,
    ResponseVerdict,
    validate_response_envelope,
)
from .protocol import (
    CHUNK_SIZE,
    CONTEXT_SIZE,
    CORPUS_CASE_COUNT,
    GEMMA_MODEL_ALIAS,
    MAX_COMPLETION_TOKENS,
    REASONING_ENABLED,
    SEEDS,
    SamplerArm,
    SamplerTuple,
    canonical_sha256,
    fixed_request_contract_payload,
    joint_top_p_top_k_arms,
    min_p_arms,
    temperature_arms,
)
from .runtime import RouterGemmaReplayRuntime, RuntimeErrorV2, TransientReplayError
from .storage import RunStore, atomic_write_json, read_json, utc_now


RUN_SCHEMA_VERSION = "gemma-sampler-execution-v2"
MIN_FREE_BYTES = 5 * 1024 * 1024 * 1024
_SOURCE_LANGUAGE_NAMES = {
    "ja": "Japanese",
    "japanese": "Japanese",
    "ja-ko": "Japanese",
    "en": "English",
    "english": "English",
    "en-ko": "English",
}


class ExecutionError(RuntimeError):
    """A fixed sampler protocol or private execution artifact is invalid."""


class ResumeRequired(ExecutionError):
    """The BAT may restart this phase without discarding already completed slots."""


@dataclass(frozen=True)
class PhaseSelection:
    phase: str
    arms: tuple[SamplerArm, ...]


def parse_sampler_tuple(value: str) -> SamplerTuple:
    """Parse a tracked-safe sampler identifier such as t0.70-p0.95-k64-m0.00."""

    text = str(value or "").strip()
    # Avoid a permissive config format: exact values keep accidental matrix
    # expansion or temperature-zero reintroduction out of the long run.
    import re

    match = re.fullmatch(r"t(0\.\d{2}|1\.00)-p(0\.\d{2}|1\.00)-k(\d+)-m(0\.\d{2}|1\.00)", text)
    if not match:
        raise ExecutionError("Sampler tuple must use t0.70-p0.95-k64-m0.00 format.")
    try:
        return SamplerTuple(
            float(match.group(1)),
            float(match.group(2)),
            int(match.group(3)),
            float(match.group(4)),
        )
    except ValueError as exc:
        raise ExecutionError("Sampler tuple violates the fixed v2 value range.") from exc


def select_phase(
    phase: str,
    *,
    selected_temperatures: Sequence[float] = (),
    selected_tuples: Sequence[SamplerTuple] = (),
) -> PhaseSelection:
    if phase == "temperature":
        return PhaseSelection(phase, temperature_arms())
    if phase == "joint_top_p_top_k":
        return PhaseSelection(phase, joint_top_p_top_k_arms(selected_temperatures))
    if phase == "min_p":
        return PhaseSelection(phase, min_p_arms(selected_tuples))
    raise ExecutionError("Unknown sampler v2 phase.")


def _case_list(reference: Mapping[str, Any]) -> list[dict[str, Any]]:
    if reference.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        raise ExecutionError("Sampler execution requires the v2 frozen reference schema.")
    if reference.get("state") != "FROZEN":
        raise ExecutionError("Sampler execution requires the user-approved frozen reference.")
    if reference.get("case_identity") != "language+source_text+context_after_text":
        raise ExecutionError("Frozen reference case identity differs from the v2 protocol.")
    cases = reference.get("cases")
    if not isinstance(cases, list) or len(cases) != CORPUS_CASE_COUNT:
        raise ExecutionError("Sampler execution requires exactly 478 frozen cases.")
    normalized = [dict(case) for case in cases if isinstance(case, Mapping)]
    if len(normalized) != CORPUS_CASE_COUNT:
        raise ExecutionError("Frozen reference contains an invalid case record.")
    if any(case.get("review_status") != "APPROVED" for case in normalized):
        raise ExecutionError("Frozen reference contains unresolved judgments.")
    reference_sha256 = str(reference.get("reference_sha256") or "")
    if not reference_sha256:
        raise ExecutionError("Frozen reference lacks its identity hash.")
    expected_reference_sha256 = canonical_sha256(
        {
            "schema_version": reference.get("schema_version"),
            "case_identity": reference.get("case_identity"),
            "cases": normalized,
        }
    )
    if reference_sha256 != expected_reference_sha256:
        raise ExecutionError("Frozen reference hash does not match its private case data.")
    return sorted(normalized, key=lambda case: str(case.get("case_id") or ""))


def load_frozen_reference(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise ExecutionError("Frozen private reference is not an object.")
    _case_list(value)
    return dict(value)


def _case_request(engine: Any, case: Mapping[str, Any], sampler: SamplerTuple, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    source = str(case.get("source_text") or "")
    context_after = str(case.get("context_after_text") or "")
    if not source:
        raise ExecutionError("Frozen reference has an empty source text.")
    texts = [source] + ([context_after] if context_after else [])
    payload, expected_keys = build_single_block_translation_request(engine, texts, 0)
    if expected_keys != ["translation"]:
        raise ExecutionError("Gemma replay request schema changed from one translation key.")
    payload.update(sampler.payload())
    payload["seed"] = int(seed)
    if (
        payload.get("model") != GEMMA_MODEL_ALIAS
        or payload.get("max_completion_tokens") != MAX_COMPLETION_TOKENS
        or engine.chunk_size != CHUNK_SIZE
        or engine.think_briefly_prompt is not REASONING_ENABLED
    ):
        raise ExecutionError("Gemma replay request departed from the fixed product contract.")
    system_prompt = str(payload["messages"][0]["content"][0]["text"])
    if "If internal reasoning is needed" in system_prompt:
        raise ExecutionError("Sampler v2 request unexpectedly enabled reasoning output guidance.")
    base_payload = dict(payload)
    for key in ("temperature", "top_p", "top_k", "min_p", "seed"):
        base_payload.pop(key, None)
    identity = {
        "case_id": str(case.get("case_id") or ""),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "context_after_sha256": hashlib.sha256(context_after.encode("utf-8")).hexdigest(),
        "request_sha256_without_sampler_or_seed": canonical_sha256(base_payload),
        "schema_sha256": canonical_sha256(payload.get("response_format")),
        "stop_sha256": canonical_sha256(payload.get("stop")),
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "user_prompt_sha256": hashlib.sha256(
            str(payload["messages"][1]["content"][0]["text"]).encode("utf-8")
        ).hexdigest(),
    }
    return payload, identity


def _source_language_name(case: Mapping[str, Any]) -> str:
    source_language = str(case.get("language") or "").strip().casefold()
    try:
        return _SOURCE_LANGUAGE_NAMES[source_language]
    except KeyError as exc:
        raise ExecutionError("Frozen reference has an unsupported source-language identity.") from exc


def _build_engine(source_language: str) -> Any:
    engine = build_engine(
        model=GEMMA_MODEL_ALIAS,
        source_lang=source_language,
        target_lang="Korean",
        max_tokens=MAX_COMPLETION_TOKENS,
    )
    engine.chunk_size = CHUNK_SIZE
    engine.think_briefly_prompt = REASONING_ENABLED
    return engine


def _assert_runtime_contract(runtime: RouterGemmaReplayRuntime) -> dict[str, Any]:
    contract = runtime.contract
    if contract is None:
        raise ExecutionError("Router runtime contract is unavailable after start.")
    # Router v2 loads aliases from the pair's models.ini, rather than from a
    # container-level LLAMA_CTX_SIZE environment variable.  The prepared
    # Gemma material carries that model configuration and is fingerprinted
    # together with the Router preset, so use it as the context-size proof.
    gemma_runtime_options = dict(contract.gemma_model.runtime_options)
    effective_context_size = str(gemma_runtime_options.get("LLAMA_CTX_SIZE") or "")
    if effective_context_size != str(CONTEXT_SIZE):
        raise ExecutionError("Router Gemma model context size differs from the v2 contract.")
    fixed = fixed_request_contract_payload()
    return {
        "fingerprint": contract.fingerprint,
        "image_ref": contract.image_ref,
        "image_id": contract.image_id,
        "repo_digest": contract.repo_digest,
        "binary_version": contract.binary_version,
        "command_sha256": contract.command_sha256,
        "preset_sha256": contract.preset_sha256,
        "ocr_model_sha256": contract.ocr_model.model_sha256,
        "ocr_manifest_sha256": contract.ocr_model.ready_manifest_sha256,
        "gemma_model_sha256": contract.gemma_model.model_sha256,
        "gemma_manifest_sha256": contract.gemma_model.ready_manifest_sha256,
        "effective_context_size": effective_context_size,
        "fixed_request_contract_sha256": canonical_sha256(fixed),
    }


def _preflight_free_space(output_root: Path) -> None:
    free = shutil.disk_usage(output_root).free
    if free < MIN_FREE_BYTES:
        raise ExecutionError("Sampler v2 requires at least 5 GiB free private archive space.")


def _record_attempt(
    store: RunStore,
    *,
    phase: str,
    arm: SamplerArm,
    seed: int,
    case_id: str,
    attempt: int,
    outcome: str,
    detail: str = "",
) -> None:
    store.append_attempt(
        {
            "recorded_utc": utc_now(),
            "phase": phase,
            "arm_key": arm.key,
            "seed": seed,
            "case_id": case_id,
            "attempt": attempt,
            "outcome": outcome,
            "detail": detail[:512],
        }
    )


def _order_for_seed(cases: Sequence[dict[str, Any]], seed_index: int) -> tuple[str, list[dict[str, Any]]]:
    if seed_index == 0:
        return "forward", list(cases)
    if seed_index == 1:
        return "reverse", list(reversed(cases))
    raise ExecutionError("Sampler v2 supports exactly two fixed seeds.")


def _logical_slot(phase: str, arm: SamplerArm, seed: int, case_id: str) -> str:
    return f"{phase}|{arm.sampler.key}|seed-{seed}|{case_id}"


def _expected_phase_slots(
    selection: PhaseSelection,
    cases: Sequence[Mapping[str, Any]],
) -> set[str]:
    """Return only new-response slots; declared reuse remains in prior runs."""

    return {
        _logical_slot(selection.phase, arm, seed, str(case.get("case_id") or ""))
        for arm in selection.arms
        if not arm.reused
        for seed in SEEDS
        for case in cases
    }


def _assert_completed_phase_status(
    status: Mapping[str, Any],
    *,
    selection: PhaseSelection,
    reference_sha256: str,
    expected_slots: set[str],
    store: RunStore,
) -> dict[str, Any]:
    """Refuse a stale/corrupt completed marker instead of silently skipping work."""

    expected_arms = [arm.payload() for arm in selection.arms]
    if (
        status.get("state") != "WAITING_FOR_JUDGMENT"
        or str(status.get("phase") or "") != selection.phase
        or str(status.get("reference_sha256") or "") != reference_sha256
        or status.get("arms") != expected_arms
        or status.get("expected_logical_slots") != len(expected_slots)
        or status.get("completed_logical_slots") != len(expected_slots)
    ):
        raise ExecutionError("Completed sampler phase status does not match the frozen matrix.")
    observed_slots = {
        str(record.get("logical_slot") or "")
        for record in iter_completed_records(store)
        if str(record.get("phase") or "") == selection.phase
    }
    if observed_slots != expected_slots:
        raise ExecutionError("Completed sampler phase status does not match its stored logical slots.")
    return dict(status)


def _records_completed(store: RunStore) -> int:
    return store.completed_count()


def iter_completed_records(store: RunStore, *, snapshot: bool = False) -> Iterable[dict[str, Any]]:
    entries = (
        store.iter_snapshot_completion_entries()
        if snapshot
        else store.iter_completion_entries()
    )
    for entry in entries:
        indexed_slot = str(entry.pop("_completion_index_logical_slot", "") or "")
        relative = str(entry.get("path") or "")
        candidate = (store.root / relative).resolve()
        try:
            candidate.relative_to(store.root)
        except ValueError as exc:
            raise ExecutionError("Completion index path escaped the managed artifact root.") from exc
        record = read_json(candidate)
        if not isinstance(record, Mapping):
            raise ExecutionError("Completed response record is not an object.")
        if (
            not indexed_slot
            or str(record.get("logical_slot") or "") != indexed_slot
            or str(record.get("status") or "") != str(entry.get("status") or "")
            or str(record.get("phase") or "") != str(entry.get("phase") or "")
            or str(record.get("arm_key") or "") != str(entry.get("arm") or "")
            or str(record.get("case_id") or "") != str(entry.get("case_id") or "")
        ):
            raise ExecutionError("Completion index and response record identities disagree.")
        yield dict(record)


def iter_compatible_completed_records(
    stores: Sequence[RunStore],
    *,
    reference: Mapping[str, Any],
    snapshot: bool = False,
) -> Iterable[dict[str, Any]]:
    """Yield only mutually compatible private records from one or more phases."""

    cases = _case_list(reference)
    reference_sha256 = str(reference["reference_sha256"])
    case_by_id = {str(case["case_id"]): case for case in cases}
    logical_slots: set[str] = set()
    for store in stores:
        for record in iter_completed_records(store, snapshot=snapshot):
            if str(record.get("status") or "") != "complete":
                raise ExecutionError("Completion index points to a non-complete response record.")
            logical_slot = str(record.get("logical_slot") or "")
            if not logical_slot or logical_slot in logical_slots:
                raise ExecutionError("Private response runs contain duplicate or empty logical slots.")
            logical_slots.add(logical_slot)
            if str(record.get("reference_sha256") or "") != reference_sha256:
                raise ExecutionError("Private response run was produced from a different frozen reference.")
            case_id = str(record.get("case_id") or "")
            case = case_by_id.get(case_id)
            if case is None or str(record.get("split") or "") != str(case.get("split") or ""):
                raise ExecutionError("Private response record no longer matches the frozen reference case split.")
            if not str(record.get("runtime_fingerprint") or ""):
                raise ExecutionError("Private response record lacks its Router runtime fingerprint.")
            request_identity = record.get("request_identity")
            if not isinstance(request_identity, Mapping):
                raise ExecutionError("Private response record lacks its fixed request identity.")
            raw_response = record.get("response")
            if isinstance(raw_response, Mapping):
                # Raw response evidence is immutable.  Re-derive the current
                # quality view in memory so an evaluator change never forces
                # a costly GPU replay or mutates a completed private run.
                verdict = validate_response_envelope(raw_response)
                normalized = dict(record)
                normalized["response_validation"] = verdict.payload()
                normalized["translation"] = verdict.translation if verdict.status == "VALID" else ""
                yield normalized
                continue
            response_validation = record.get("response_validation")
            if (
                not isinstance(response_validation, Mapping)
                or response_validation.get("schema_version") != RESPONSE_VALIDATION_SCHEMA_VERSION
            ):
                raise ExecutionError(
                    "Private response record lacks raw Router output required for the current quality evaluation."
                )
            yield record


def collect_completed_records(
    stores: Sequence[RunStore],
    *,
    reference: Mapping[str, Any],
    snapshot: bool = False,
) -> list[dict[str, Any]]:
    """Materialize compatible records only for judgment/report aggregation."""

    return list(
        iter_compatible_completed_records(
            stores,
            reference=reference,
            snapshot=snapshot,
        )
    )


def sampler_keys_from_phase_status(
    phase_status: Mapping[str, Any],
    *,
    reference: Mapping[str, Any],
) -> tuple[str, ...]:
    """Read the exact candidate tuple set sealed by one completed phase."""

    phase = str(phase_status.get("phase") or "")
    if not phase or phase_status.get("state") != "WAITING_FOR_JUDGMENT":
        raise ExecutionError("Sampler judgment requires a completed WAITING_FOR_JUDGMENT phase status.")
    if str(phase_status.get("reference_sha256") or "") != str(reference.get("reference_sha256") or ""):
        raise ExecutionError("Phase status belongs to a different frozen reference.")
    arms = phase_status.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ExecutionError("Phase status lacks its sealed sampler arms.")
    keys: list[str] = []
    for arm in arms:
        if not isinstance(arm, Mapping) or str(arm.get("phase") or "") != phase:
            raise ExecutionError("Phase status contains an invalid sampler arm.")
        raw_sampler = arm.get("sampler")
        if not isinstance(raw_sampler, Mapping):
            raise ExecutionError("Phase status sampler arm lacks its tuple.")
        try:
            sampler = SamplerTuple(
                float(raw_sampler.get("temperature")),
                float(raw_sampler.get("top_p")),
                int(raw_sampler.get("top_k")),
                float(raw_sampler.get("min_p")),
            )
        except (TypeError, ValueError) as exc:
            raise ExecutionError("Phase status contains an invalid sampler tuple.") from exc
        if str(arm.get("arm_key") or "") != f"{phase}-{sampler.key}":
            raise ExecutionError("Phase status sampler arm identity changed.")
        keys.append(sampler.key)
    if len(set(keys)) != len(keys):
        raise ExecutionError("Phase status contains duplicate sampler tuples.")
    return tuple(keys)


def complete_scope_records(
    records: Iterable[Mapping[str, Any]],
    *,
    reference: Mapping[str, Any],
    scope: str,
    sampler_keys: Sequence[str],
) -> list[dict[str, Any]]:
    """Select a sealed candidate set only when every case and seed is present."""

    if scope not in {"tuning", "holdout"}:
        raise ExecutionError("Unknown sampler judgment scope.")
    allowed = tuple(str(key) for key in sampler_keys)
    if not allowed or len(set(allowed)) != len(allowed):
        raise ExecutionError("Sampler judgment requires a non-duplicate candidate tuple set.")
    all_cases = _case_list(reference)
    cases = [case for case in all_cases if case.get("split") == scope]
    if not cases:
        raise ExecutionError("Frozen reference has no cases for the requested judgment scope.")
    case_split_by_id = {str(case["case_id"]): str(case.get("split") or "") for case in all_cases}
    expected = {
        (sampler_key, seed, str(case["case_id"]))
        for sampler_key in allowed
        for seed in SEEDS
        for case in cases
    }
    observed: set[tuple[str, int, str]] = set()
    selected: list[dict[str, Any]] = []
    for record in records:
        raw_sampler = record.get("sampler")
        if not isinstance(raw_sampler, Mapping):
            raise ExecutionError("Private response record lacks its sampler tuple.")
        try:
            sampler_key = SamplerTuple(
                float(raw_sampler.get("temperature")),
                float(raw_sampler.get("top_p")),
                int(raw_sampler.get("top_k")),
                float(raw_sampler.get("min_p")),
            ).key
        except (TypeError, ValueError) as exc:
            raise ExecutionError("Private response record has an invalid sampler tuple.") from exc
        if sampler_key not in allowed:
            continue
        seed = record.get("seed")
        case_id = str(record.get("case_id") or "")
        if not isinstance(seed, int) or seed not in SEEDS:
            raise ExecutionError("Private response record has an invalid fixed seed.")
        if case_id not in case_split_by_id:
            raise ExecutionError("Private response record case is absent from the frozen reference.")
        if case_split_by_id[case_id] != scope:
            continue
        identity = (sampler_key, seed, case_id)
        if identity not in expected or identity in observed:
            raise ExecutionError("Private response records do not form one complete sampler matrix.")
        observed.add(identity)
        selected.append(dict(record))
    if observed != expected:
        raise ExecutionError("Sampler judgment cannot open before every selected case and seed is complete.")
    return selected


def _required_reused_records(
    records: Iterable[Mapping[str, Any]],
    *,
    reused_arms: Sequence[SamplerArm],
    cases: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    """Require prior complete results before a matrix row is declared reused."""

    required: set[tuple[str, int, str]] = {
        (arm.sampler.key, seed, str(case.get("case_id") or ""))
        for arm in reused_arms
        for seed in SEEDS
        for case in cases
    }
    observed: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for record in records:
        sampler = record.get("sampler")
        if not isinstance(sampler, Mapping):
            continue
        try:
            sampler_key = SamplerTuple(
                float(sampler.get("temperature")),
                float(sampler.get("top_p")),
                int(sampler.get("top_k")),
                float(sampler.get("min_p")),
            ).key
        except (TypeError, ValueError):
            continue
        seed = record.get("seed")
        case_id = str(record.get("case_id") or "")
        if isinstance(seed, int):
            key = (sampler_key, seed, case_id)
            if key in required:
                observed[key] = record
    if not required.issubset(observed):
        raise ExecutionError("A requested reused sampler arm has no complete prior 478x2 result set.")
    return {key: observed[key] for key in required}


def _assert_reused_contracts(
    records: Mapping[tuple[str, int, str], Mapping[str, Any]],
    *,
    runtime_fingerprint: str,
    request_identities: Mapping[str, Mapping[str, Any]],
) -> None:
    for _sampler_key, _seed, case_id in records:
        record = records[(_sampler_key, _seed, case_id)]
        if str(record.get("runtime_fingerprint") or "") != runtime_fingerprint:
            raise ExecutionError("A reused sampler response has a different Router runtime fingerprint.")
        if record.get("request_identity") != request_identities.get(case_id):
            raise ExecutionError("A reused sampler response has a different fixed request identity.")


def execute_phase(
    *,
    store: RunStore,
    reference: Mapping[str, Any],
    selection: PhaseSelection,
    timeout_sec: float = 180.0,
    max_attempts: int = 3,
    runtime: RouterGemmaReplayRuntime | None = None,
    prior_stores: Sequence[RunStore] = (),
) -> dict[str, Any]:
    """Run/resume a full phase, then stop at WAITING_FOR_JUDGMENT.

    A retryable worker failure exits with :class:`ResumeRequired`; all durable
    first complete responses remain indexed and the BAT can relaunch safely.
    """

    if max_attempts < 1:
        raise ExecutionError("Sampler v2 requires at least one HTTP attempt.")
    cases = _case_list(reference)
    reference_sha256 = str(reference["reference_sha256"])
    _preflight_free_space(store.root)
    execution_arms = tuple(arm for arm in selection.arms if not arm.reused)
    reused_arms = tuple(arm for arm in selection.arms if arm.reused)
    reused_records = _required_reused_records(
        iter_compatible_completed_records((store, *prior_stores), reference=reference),
        reused_arms=reused_arms,
        cases=cases,
    )
    expected_slots = _expected_phase_slots(selection, cases)
    expected_total = len(expected_slots)
    phase_status_path = store.phase_status_path(selection.phase)
    if phase_status_path.exists():
        previous = read_json(phase_status_path)
        if isinstance(previous, Mapping) and previous.get("state") == "WAITING_FOR_JUDGMENT":
            return _assert_completed_phase_status(
                previous,
                selection=selection,
                reference_sha256=reference_sha256,
                expected_slots=expected_slots,
                store=store,
            )
    active_runtime = runtime or RouterGemmaReplayRuntime()
    started_here = False
    try:
        # Freeze the non-sampler portion of every case request before the
        # Router is touched.  This remains valid across all phases and resume.
        identity_sampler = next((arm.sampler for arm in selection.arms), None)
        if identity_sampler is None:
            raise ExecutionError("Sampler phase has no arms.")
        engines: dict[str, Any] = {}
        request_identities: dict[str, Mapping[str, Any]] = {}
        for case in cases:
            source_language = _source_language_name(case)
            engine = engines.get(source_language)
            if engine is None:
                engine = _build_engine(source_language)
                engines[source_language] = engine
            _payload, request_identity = _case_request(
                engine,
                case,
                identity_sampler,
                SEEDS[0],
            )
            case_id = str(case.get("case_id") or "")
            store.bind_request_contract(
                case_id=case_id,
                identity=request_identity,
            )
            request_identities[case_id] = request_identity
        active_runtime.start()
        started_here = True
        runtime_contract = _assert_runtime_contract(active_runtime)
        _assert_reused_contracts(
            reused_records,
            runtime_fingerprint=str(runtime_contract["fingerprint"]),
            request_identities=request_identities,
        )
        runtime_contract_path = store.root / "runtime-contract.json"
        if runtime_contract_path.exists():
            previous_contract = read_json(runtime_contract_path)
            if not isinstance(previous_contract, Mapping) or previous_contract.get("fingerprint") != runtime_contract["fingerprint"]:
                raise ExecutionError("Resumed sampler phase has a different Router runtime fingerprint.")
        else:
            atomic_write_json(runtime_contract_path, runtime_contract)
        for arm in execution_arms:
            for seed_index, seed in enumerate(SEEDS):
                order_name, ordered_cases = _order_for_seed(cases, seed_index)
                run_name = f"seed-{seed}-{order_name}"
                for case_position, case in enumerate(ordered_cases, start=1):
                    case_id = str(case.get("case_id") or "")
                    slot = _logical_slot(selection.phase, arm, seed, case_id)
                    if store.is_completed(slot):
                        continue
                    _preflight_free_space(store.root)
                    engine = engines[_source_language_name(case)]
                    payload, request_identity = _case_request(engine, case, arm.sampler, seed)
                    last_transient = ""
                    for attempt in range(1, max_attempts + 1):
                        _record_attempt(
                            store,
                            phase=selection.phase,
                            arm=arm,
                            seed=seed,
                            case_id=case_id,
                            attempt=attempt,
                            outcome="started",
                        )
                        try:
                            replay = active_runtime.request(payload, timeout_sec=timeout_sec)
                        except TransientReplayError as exc:
                            last_transient = str(exc)
                            _record_attempt(
                                store,
                                phase=selection.phase,
                                arm=arm,
                                seed=seed,
                                case_id=case_id,
                                attempt=attempt,
                                outcome="indeterminate",
                                detail=last_transient,
                            )
                            if attempt < max_attempts:
                                time.sleep(float(2 ** (attempt - 1)))
                                continue
                            raise ResumeRequired("Transient sampler worker failure requires BAT resume.") from exc
                        except RuntimeErrorV2 as exc:
                            _record_attempt(
                                store,
                                phase=selection.phase,
                                arm=arm,
                                seed=seed,
                                case_id=case_id,
                                attempt=attempt,
                                outcome="fatal",
                                detail=str(exc),
                            )
                            raise ExecutionError("Sampler runtime contract or drain gate failed closed.") from exc
                        verdict: ResponseVerdict = validate_response_envelope(replay.envelope)
                        record = {
                            "schema_version": RUN_SCHEMA_VERSION,
                            "status": "complete",
                            "recorded_utc": utc_now(),
                            "phase": selection.phase,
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
                            phase=selection.phase,
                            arm=arm.key,
                            run=run_name,
                            case_id=case_id,
                            logical_slot=slot,
                            payload=record,
                        )
                        if stored:
                            _record_attempt(
                                store,
                                phase=selection.phase,
                                arm=arm,
                                seed=seed,
                                case_id=case_id,
                                attempt=attempt,
                                outcome="complete",
                            )
                        store.update_progress(
                            {
                                "schema_version": RUN_SCHEMA_VERSION,
                                "state": "RUNNING",
                                "phase": selection.phase,
                                "completed_logical_slots": _records_completed(store),
                                "phase_expected_logical_slots": expected_total,
                                "updated_utc": utc_now(),
                            }
                        )
                        break
    finally:
        if started_here:
            active_runtime.close()
    phase_records = [
        record for record in iter_completed_records(store) if record.get("phase") == selection.phase
    ]
    phase_slots = {str(record.get("logical_slot") or "") for record in phase_records}
    if len(phase_records) != expected_total or phase_slots != expected_slots:
        raise ResumeRequired("Sampler phase stopped before every logical slot had one complete response.")
    status = {
        "schema_version": RUN_SCHEMA_VERSION,
        "state": "WAITING_FOR_JUDGMENT",
        "phase": selection.phase,
        "reference_sha256": str(reference.get("reference_sha256") or ""),
        "reference_summary": reference_summary(reference),
        "expected_logical_slots": expected_total,
        "completed_logical_slots": len(phase_records),
        "arm_count": len(selection.arms),
        "new_arm_count": len(execution_arms),
        "reused_arm_count": len(reused_arms),
        "arms": [arm.payload() for arm in selection.arms],
        "seeds": list(SEEDS),
        "completed_utc": utc_now(),
    }
    store.write_phase_status(selection.phase, status)
    store.update_progress(
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "state": "WAITING_FOR_JUDGMENT",
            "phase": selection.phase,
            "completed_logical_slots": _records_completed(store),
            "updated_utc": utc_now(),
        }
    )
    return status
