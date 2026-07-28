#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import modules.translation.llm.custom_local_gemma as gemma_module  # noqa: E402
from modules.translation.llm.custom_local_gemma import (  # noqa: E402
    DEFAULT_GEMMA_LOCAL_ENDPOINT,
    DEFAULT_GEMMA_MAX_COMPLETION_TOKENS,
    DEFAULT_GEMMA_PROMPT_PROFILE,
    DEFAULT_GEMMA_RESPONSE_FORMAT_MODE,
    DEFAULT_GEMMA_RESPONSE_SCHEMA_MODE,
    DEFAULT_GEMMA_THINK_BRIEFLY_PROMPT,
    DEFAULT_GEMMA_TRANSLATION_MIN_P,
    DEFAULT_GEMMA_TRANSLATION_TEMPERATURE,
    DEFAULT_GEMMA_TRANSLATION_TOP_K,
    DEFAULT_GEMMA_TRANSLATION_TOP_P,
    GEMMA_REQUEST_MODE_CONTEXTUAL_SINGLE,
    CustomLocalGemmaTranslation,
)
from modules.translation.local_runtime import (  # noqa: E402
    LocalGemmaRuntimeManager,
)
from modules.translation.translation_memory import (  # noqa: E402
    ResultCacheRecord,
    TranslationMemoryStore,
    canonical_json,
)
from modules.utils.llama_cpp_runtime import (  # noqa: E402
    DEFAULT_MANAGED_RUNTIME_STOP_TIMEOUT_SEC,
    run_docker_command,
)
from modules.utils.textblock import TextBlock  # noqa: E402


DEFAULT_MODEL = "Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ4_XS.gguf"
DEFAULT_GROUP_SIZE = 7
CONTAINER_NAME = "gemma-local-server"
SEVERE_STATS = (
    "gemma_truncated_count",
    "gemma_empty_content_count",
    "gemma_missing_key_count",
    "gemma_partial_fallback_block_count",
    "gemma_split_count",
    "gemma_parser_error_count",
    "gemma_repetition_guard_count",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object: {path.name}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def _ensure_untracked_output(path: Path) -> None:
    try:
        relative = path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return
    completed = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative.as_posix()],
        cwd=ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            "Benchmark output inside the repository must be under a Git-ignored "
            f"path: {relative.as_posix()}"
        )


def _source_language(case_name: str) -> str:
    lowered = case_name.lower()
    for marker, language in (
        ("japanese", "Japanese"),
        ("chinese", "Chinese"),
        ("english", "English"),
    ):
        if marker in lowered:
            return language
    raise ValueError(f"Cannot infer a source language from case id: {case_name}")


def load_multilingual_corpora(
    source_summary: Path,
) -> OrderedDict[str, list[dict[str, str]]]:
    payload = _read_json(source_summary)
    records = [
        record
        for record in payload.get("translations", [])
        if isinstance(record, dict)
        and int(record.get("round", 0) or 0) == 1
        and str(record.get("mode", "")) == "contextual-single"
    ]
    corpora: OrderedDict[str, list[dict[str, str]]] = OrderedDict(
        (language, []) for language in ("Japanese", "Chinese", "English")
    )
    seen: set[str] = set()
    for record in records:
        case_id = str(record.get("case", "")).strip()
        if not case_id or case_id in seen:
            continue
        source = str(record.get("source", ""))
        if not source.strip():
            continue
        seen.add(case_id)
        corpora[_source_language(case_id)].append(
            {
                "case_id": case_id,
                "source": source,
                "reference": str(record.get("old_log_reference", "")),
            }
        )
    counts = {language: len(items) for language, items in corpora.items()}
    if counts != {"Japanese": 18, "Chinese": 18, "English": 18}:
        raise ValueError(f"Expected exactly 18 blocks per language, got {counts}")
    return corpora


def _new_blocks(items: Iterable[Mapping[str, str]]) -> list[TextBlock]:
    return [
        TextBlock(
            text_bbox=np.array([0, index * 10, 100, index * 10 + 8]),
            text=str(item["source"]),
        )
        for index, item in enumerate(items)
    ]


@dataclass(frozen=True)
class RuntimeSettings:
    model: str
    api_url: str = DEFAULT_GEMMA_LOCAL_ENDPOINT

    def get_credentials(self, _service_name: str) -> dict[str, str]:
        return {"api_url": self.api_url, "model": self.model}


class DictionarySettings:
    @staticmethod
    def get_translation_result_dictionary_rules() -> list[dict[str, str]]:
        return []

    @staticmethod
    def apply_translation_result_dictionary(value: str) -> str:
        return value


class RuntimeHarness:
    def __init__(self, model: str) -> None:
        self.settings = RuntimeSettings(model=model)
        self.manager = LocalGemmaRuntimeManager()
        self.ensure_calls = 0
        self.progress_events: list[dict[str, Any]] = []

    def identity(self) -> dict[str, Any] | None:
        return self.manager.get_translation_cache_identity(self.settings)

    def ensure(self) -> None:
        self.ensure_calls += 1
        self.manager.ensure_server(
            self.settings,
            progress_callback=lambda event: self.progress_events.append(dict(event)),
        )

    def shutdown(self) -> None:
        self.manager.shutdown()
        stop_managed_container()


def stop_managed_container() -> None:
    run_docker_command(
        [
            "docker",
            "stop",
            "--timeout",
            str(DEFAULT_MANAGED_RUNTIME_STOP_TIMEOUT_SEC),
            CONTAINER_NAME,
        ],
        check=False,
        timeout_sec=DEFAULT_MANAGED_RUNTIME_STOP_TIMEOUT_SEC + 15.0,
    )


def inspect_container() -> dict[str, Any]:
    completed = run_docker_command(
        ["docker", "inspect", CONTAINER_NAME],
        check=False,
    )
    if completed.returncode != 0:
        return {"exists": False, "running": False, "status": "missing"}
    rows = json.loads(completed.stdout or "[]")
    if not rows or not isinstance(rows[0], dict):
        return {"exists": False, "running": False, "status": "missing"}
    inspection = rows[0]
    state = inspection.get("State") if isinstance(inspection.get("State"), dict) else {}
    config = inspection.get("Config") if isinstance(inspection.get("Config"), dict) else {}
    labels = config.get("Labels") if isinstance(config.get("Labels"), dict) else {}
    command = config.get("Cmd") if isinstance(config.get("Cmd"), list) else []
    model = ""
    for index, value in enumerate(command[:-1]):
        if str(value) == "-m":
            model = Path(str(command[index + 1])).name
            break
    return {
        "exists": True,
        "running": bool(state.get("Running")),
        "status": str(state.get("Status", "")),
        "model": model,
        "image_id": str(inspection.get("Image", "")),
        "runtime_fingerprint": str(
            labels.get("comic-translate.runtime-fingerprint", "")
        ),
    }


def restore_stopped_container(model: str) -> None:
    if not model:
        stop_managed_container()
        return
    manager = LocalGemmaRuntimeManager()
    contract = manager._load_runtime_contract(model)
    stop_managed_container()
    manager._run_compose(
        "create",
        "--force-recreate",
        step_name="restore",
        runtime_contract=contract,
    )
    stop_managed_container()


def new_engine(
    *,
    language: str,
    store: TranslationMemoryStore | None,
    runtime: RuntimeHarness,
    persistent_cache_enabled: bool,
    exact_tm_enabled: bool,
    group_size: int,
    max_completion_tokens: int,
    top_p: float = DEFAULT_GEMMA_TRANSLATION_TOP_P,
) -> CustomLocalGemmaTranslation:
    engine = CustomLocalGemmaTranslation()
    engine.api_base_url = runtime.settings.api_url.rstrip("/")
    engine.model = runtime.settings.model
    engine.source_lang = language
    engine.target_lang = "Korean"
    engine.settings = DictionarySettings()
    engine.chunk_size = int(group_size)
    engine.max_tokens = int(max_completion_tokens)
    engine.timeout = 240
    engine.request_mode = GEMMA_REQUEST_MODE_CONTEXTUAL_SINGLE
    engine.raw_response_logging = False
    engine.prompt_profile = DEFAULT_GEMMA_PROMPT_PROFILE
    engine.response_format_mode = DEFAULT_GEMMA_RESPONSE_FORMAT_MODE
    engine.response_schema_mode = DEFAULT_GEMMA_RESPONSE_SCHEMA_MODE
    engine.think_briefly_prompt = DEFAULT_GEMMA_THINK_BRIEFLY_PROMPT
    engine.temperature = DEFAULT_GEMMA_TRANSLATION_TEMPERATURE
    engine.top_k = DEFAULT_GEMMA_TRANSLATION_TOP_K
    engine.top_p = float(top_p)
    engine.min_p = DEFAULT_GEMMA_TRANSLATION_MIN_P
    engine.contextual_merge_input = True
    engine.configure_runtime_hooks(
        ensure_runtime=runtime.ensure,
        runtime_identity_provider=runtime.identity,
    )
    engine.configure_translation_memory(
        store,
        {
            "persistent_cache_enabled": bool(persistent_cache_enabled),
            "exact_tm_enabled": bool(exact_tm_enabled),
            "result_cache_limit": 50_000,
            "candidate_limit": 5_000,
        },
    )
    return engine


def _sum_stats(
    destination: dict[str, int | float],
    source: Mapping[str, int | float],
) -> None:
    for key, value in source.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, float):
            destination[key] = float(destination.get(key, 0.0)) + value
        else:
            destination[key] = int(destination.get(key, 0)) + int(value or 0)


def run_corpus(
    *,
    corpora: OrderedDict[str, list[dict[str, str]]],
    store: TranslationMemoryStore | None,
    runtime: RuntimeHarness,
    persistent_cache_enabled: bool,
    exact_tm_enabled: bool,
    group_size: int,
    max_completion_tokens: int,
    requested_indices: Mapping[str, Iterable[int]] | None = None,
    top_p: float = DEFAULT_GEMMA_TRANSLATION_TOP_P,
) -> dict[str, Any]:
    started = time.perf_counter()
    ensure_before = runtime.ensure_calls
    stats: dict[str, int | float] = {}
    outputs: list[dict[str, Any]] = []
    runtime_required: dict[str, bool] = {}
    for language, items in corpora.items():
        indices = (
            tuple(int(index) for index in requested_indices[language])
            if requested_indices is not None
            else tuple(range(len(items)))
        )
        blocks = _new_blocks(items)
        engine = new_engine(
            language=language,
            store=store,
            runtime=runtime,
            persistent_cache_enabled=persistent_cache_enabled,
            exact_tm_enabled=exact_tm_enabled,
            group_size=group_size,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p,
        )
        runtime_required[language] = engine.prepare_translation(
            blocks,
            "",
            requested_indices=indices,
        )
        engine.translate(
            blocks,
            np.zeros((1, 1, 3), dtype=np.uint8),
            "",
            requested_indices=indices,
        )
        _sum_stats(stats, engine.last_benchmark_stats)
        for index in indices:
            outputs.append(
                {
                    "language": language,
                    "index": index,
                    "case_id": items[index]["case_id"],
                    "source": items[index]["source"],
                    "reference": items[index]["reference"],
                    "translation": str(blocks[index].translation or ""),
                }
            )
    elapsed = time.perf_counter() - started
    severe_count = sum(int(stats.get(key, 0) or 0) for key in SEVERE_STATS)
    return {
        "elapsed_sec": round(elapsed, 6),
        "output_count": len(outputs),
        "nonempty_count": sum(
            1 for output in outputs if str(output["translation"]).strip()
        ),
        "runtime_required": runtime_required,
        "runtime_ensure_calls": runtime.ensure_calls - ensure_before,
        "container_after": inspect_container(),
        "severe_telemetry_count": severe_count,
        "stats": stats,
        "outputs": outputs,
    }


def public_scenario(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key != "outputs"
    }


def checkpoint_report(output_dir: Path, report: Mapping[str, Any]) -> None:
    _write_json(output_dir / "summary.json", report)


def output_map(result: Mapping[str, Any]) -> dict[tuple[str, int], str]:
    return {
        (str(item["language"]), int(item["index"])): str(item["translation"])
        for item in result.get("outputs", [])
        if isinstance(item, Mapping)
    }


def compare_outputs(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    left_map = output_map(left)
    right_map = output_map(right)
    keys = sorted(set(left_map) & set(right_map))
    similarities = [
        SequenceMatcher(None, left_map[key].strip(), right_map[key].strip()).ratio()
        for key in keys
    ]
    return {
        "compared_count": len(keys),
        "exact_match_count": sum(
            1 for key in keys if left_map[key] == right_map[key]
        ),
        "mean_similarity": (
            round(statistics.fmean(similarities), 6) if similarities else None
        ),
        "minimum_similarity": (
            round(min(similarities), 6) if similarities else None
        ),
    }


def seed_result_cache(
    *,
    corpora: OrderedDict[str, list[dict[str, str]]],
    baseline: Mapping[str, Any],
    store: TranslationMemoryStore,
    runtime: RuntimeHarness,
    indices_by_language: Mapping[str, Iterable[int]],
    group_size: int,
    max_completion_tokens: int,
) -> int:
    baseline_map = output_map(baseline)
    records: list[ResultCacheRecord] = []
    for language, items in corpora.items():
        requested = tuple(int(index) for index in indices_by_language[language])
        blocks = _new_blocks(items)
        engine = new_engine(
            language=language,
            store=store,
            runtime=runtime,
            persistent_cache_enabled=True,
            exact_tm_enabled=False,
            group_size=group_size,
            max_completion_tokens=max_completion_tokens,
        )
        engine.prepare_translation(blocks, "", requested_indices=requested)
        plan = engine._pending_translation_plan
        if plan is None:
            raise RuntimeError("Gemma cache plan was not retained for benchmark seeding.")
        for target in plan.targets:
            translation = baseline_map[(language, target.global_index)]
            records.append(
                ResultCacheRecord(
                    cache_key=target.cache_key,
                    scope_key=target.scope_key,
                    identity_json=target.identity_json,
                    source_text=target.source_text,
                    translation=translation,
                    metadata_json=canonical_json({}),
                )
            )
    if not store.store_results(records):
        raise RuntimeError("Unable to seed the result cache.")
    return len(records)


@contextmanager
def patched_cache_prompt(enabled: bool) -> Iterator[None]:
    original_post = gemma_module.requests.post

    def post_with_cache_prompt(*args: Any, **kwargs: Any) -> Any:
        payload = dict(kwargs.get("json") or {})
        payload["cache_prompt"] = bool(enabled)
        kwargs["json"] = payload
        return original_post(*args, **kwargs)

    gemma_module.requests.post = post_with_cache_prompt
    try:
        yield
    finally:
        gemma_module.requests.post = original_post


@contextmanager
def temporary_cache_ram_mib(value: int) -> Iterator[None]:
    key = "LLAMA_CACHE_RAM_MIB"
    previous = os.environ.get(key)
    os.environ[key] = str(int(value))
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def run_prefix_candidate(
    *,
    corpora: OrderedDict[str, list[dict[str, str]]],
    model: str,
    cache_ram_mib: int,
    cache_prompt: bool,
    repeat_count: int,
    max_completion_tokens: int,
) -> dict[str, Any]:
    with temporary_cache_ram_mib(cache_ram_mib):
        stop_managed_container()
        runtime = RuntimeHarness(model)
        runtime.ensure()
        items = OrderedDict([("Japanese", corpora["Japanese"][:2])])
        timings: list[float] = []
        prompt_ms: list[float] = []
        completion_tokens: list[int] = []
        failures: list[str] = []
        try:
            with patched_cache_prompt(cache_prompt):
                run_corpus(
                    corpora=items,
                    store=None,
                    runtime=runtime,
                    persistent_cache_enabled=False,
                    exact_tm_enabled=False,
                    group_size=2,
                    max_completion_tokens=max_completion_tokens,
                )
                for _round_index in range(repeat_count):
                    result = run_corpus(
                        corpora=items,
                        store=None,
                        runtime=runtime,
                        persistent_cache_enabled=False,
                        exact_tm_enabled=False,
                        group_size=2,
                        max_completion_tokens=max_completion_tokens,
                    )
                    if (
                        result["nonempty_count"] != 2
                        or result["severe_telemetry_count"]
                    ):
                        failures.append(
                            "translation structure or safety telemetry failed"
                        )
                    timings.append(float(result["elapsed_sec"]))
                    prompt_ms.append(
                        float(result["stats"].get("gemma_prompt_eval_ms", 0.0) or 0.0)
                    )
                    completion_tokens.append(
                        int(
                            result["stats"].get(
                                "gemma_completion_tokens",
                                0,
                            )
                            or 0
                        )
                    )
            identity = runtime.identity() or {}
            return {
                "cache_ram_mib": int(cache_ram_mib),
                "cache_prompt": bool(cache_prompt),
                "repeat_count": int(repeat_count),
                "median_elapsed_sec": round(statistics.median(timings), 6),
                "median_prompt_eval_ms": round(statistics.median(prompt_ms), 6),
                "completion_tokens": completion_tokens,
                "failure_count": len(failures),
                "runtime_fingerprint": identity.get("runtime_fingerprint", ""),
                "runtime_options": identity.get("runtime_options", {}),
            }
        finally:
            runtime.shutdown()


def select_prefix_candidate(
    candidates: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    passed = [
        dict(candidate)
        for candidate in candidates
        if int(candidate.get("failure_count", 0) or 0) == 0
    ]
    if not passed:
        return None
    fastest = min(float(candidate["median_elapsed_sec"]) for candidate in passed)
    eligible = [
        candidate
        for candidate in passed
        if float(candidate["median_elapsed_sec"]) <= fastest * 1.03
    ]
    eligible.sort(
        key=lambda candidate: (
            int(candidate["cache_ram_mib"]),
            float(candidate["median_elapsed_sec"]),
            not bool(candidate["cache_prompt"]),
        )
    )
    return eligible[0] if eligible else None


def reduction_percent(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        return 0.0
    return round(((baseline - candidate) / baseline) * 100.0, 3)


def automated_gate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    scenarios = report.get("scenarios", {})
    comparisons = report.get("comparisons", {})
    checks: dict[str, bool] = {}

    def scenario(name: str) -> Mapping[str, Any]:
        value = scenarios.get(name, {})
        return value if isinstance(value, Mapping) else {}

    stopped_hit = scenario("stopped_all_result_cache_hit")
    approved_hit = scenario("stopped_all_approved_tm_hit")
    mixed = scenario("mixed_result_cache_hit")
    requested = scenario("requested_blocks_no_hit")
    stale = scenario("stale_sampler_key")
    corrupt = scenario("corrupt_db_fail_open_preflight")

    checks["stopped_result_cache_54_of_54"] = (
        int(stopped_hit.get("nonempty_count", 0) or 0) == 54
        and int(stopped_hit.get("runtime_ensure_calls", -1) or 0) == 0
        and int(
            (stopped_hit.get("stats") or {}).get(
                "gemma_http_attempt_count",
                -1,
            )
            or 0
        )
        == 0
        and not bool((stopped_hit.get("container_after") or {}).get("running"))
    )
    checks["approved_tm_54_of_54"] = (
        int(approved_hit.get("nonempty_count", 0) or 0) == 54
        and int(approved_hit.get("runtime_ensure_calls", -1) or 0) == 0
        and int(
            (approved_hit.get("stats") or {}).get(
                "gemma_http_attempt_count",
                -1,
            )
            or 0
        )
        == 0
        and int(
            (approved_hit.get("stats") or {}).get(
                "gemma_tm_exact_hit_count",
                0,
            )
            or 0
        )
        == 54
        and not bool((approved_hit.get("container_after") or {}).get("running"))
    )
    mixed_stats = mixed.get("stats") or {}
    checks["mixed_hit_contract"] = (
        int(mixed.get("nonempty_count", 0) or 0) == 54
        and int(
            mixed_stats.get("gemma_tm_result_cache_hit_count", 0) or 0
        )
        == 27
        and int(
            mixed_stats.get("gemma_tm_result_cache_miss_count", 0) or 0
        )
        == 27
        and int(mixed.get("severe_telemetry_count", -1) or 0) == 0
    )
    checks["requested_blocks_contract"] = (
        int(requested.get("nonempty_count", 0) or 0) == 27
        and int(requested.get("severe_telemetry_count", -1) or 0) == 0
    )
    checks["stale_key_rejected"] = (
        int(
            (stale.get("stats") or {}).get(
                "gemma_tm_stale_reject_count",
                0,
            )
            or 0
        )
        == 21
    )
    checks["corrupt_db_failed_open"] = (
        bool(corrupt.get("runtime_required"))
        and not bool(corrupt.get("store_enabled", True))
        and str(corrupt.get("disabled_reason_type", "")) == "DatabaseError"
        and str(corrupt.get("database_preserved_sha256", ""))
        == _sha256_bytes(b"not a sqlite database")
    )
    for comparison_name, check_name in (
        (
            "cold_vs_stopped_result_hit",
            "stopped_result_cache_exact_output_preservation",
        ),
        (
            "cold_vs_warm_result_hit",
            "warm_result_cache_exact_output_preservation",
        ),
    ):
        comparison = comparisons.get(comparison_name, {})
        checks[check_name] = (
            int(comparison.get("compared_count", 0) or 0) == 54
            and int(comparison.get("exact_match_count", 0) or 0) == 54
        )
    exact_comparison = comparisons.get("cold_vs_approved_tm", {})
    checks["approved_tm_exact_output_preservation"] = (
        int(exact_comparison.get("compared_count", 0) or 0) == 54
        and int(exact_comparison.get("exact_match_count", 0) or 0) == 54
    )
    prefix = report.get("prefix_cache_matrix")
    if isinstance(prefix, Mapping):
        checks["prefix_candidate_selected"] = isinstance(
            prefix.get("selected"),
            Mapping,
        )
    return {
        "checks": checks,
        "passed": bool(checks) and all(checks.values()),
        "semantic_review_required": (
            "Automated structural gates cannot decide speaker, relationship, "
            "negation, action, target, number, proper noun, or explicit meaning."
        ),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    source_summary = Path(args.source_summary).expanduser().resolve()
    if not source_summary.is_file():
        raise FileNotFoundError(f"Source summary was not found: {source_summary}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    _ensure_untracked_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpora = load_multilingual_corpora(source_summary)
    initial_container = inspect_container()
    initial_model = str(initial_container.get("model", ""))
    stop_managed_container()

    runtime = RuntimeHarness(args.model)
    cold_store = TranslationMemoryStore(output_dir / "cold-cache.sqlite3")
    warm_store = TranslationMemoryStore(output_dir / "warm-cache.sqlite3")
    mixed_store = TranslationMemoryStore(output_dir / "mixed-cache.sqlite3")
    requested_store = TranslationMemoryStore(output_dir / "requested-cache.sqlite3")
    report: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "branch": _git_output("branch", "--show-current"),
        "commit": _git_output("rev-parse", "HEAD"),
        "source_summary_sha256": _sha256_file(source_summary),
        "source_block_count": 54,
        "blocks_per_language": 18,
        "model": args.model,
        "group_size": int(args.group_size),
        "max_completion_tokens": int(args.max_completion_tokens),
        "initial_container": initial_container,
        "scenarios": {},
        "comparisons": {},
        "privacy_note": (
            "Summary omits local paths, source text, and translations. "
            "Private comparison artifacts remain in the ignored output directory."
        ),
    }
    _write_json(output_dir / "summary.json", report)

    even = {
        language: tuple(index for index in range(18) if index % 2 == 0)
        for language in corpora
    }
    odd = {
        language: tuple(index for index in range(18) if index % 2 == 1)
        for language in corpora
    }

    try:
        cold = run_corpus(
            corpora=corpora,
            store=cold_store,
            runtime=runtime,
            persistent_cache_enabled=True,
            exact_tm_enabled=True,
            group_size=args.group_size,
            max_completion_tokens=args.max_completion_tokens,
        )
        report["scenarios"]["stopped_empty_cache"] = public_scenario(cold)
        checkpoint_report(output_dir, report)
        runtime.shutdown()

        stopped_hit = run_corpus(
            corpora=corpora,
            store=cold_store,
            runtime=runtime,
            persistent_cache_enabled=True,
            exact_tm_enabled=True,
            group_size=args.group_size,
            max_completion_tokens=args.max_completion_tokens,
        )
        report["scenarios"]["stopped_all_result_cache_hit"] = public_scenario(
            stopped_hit
        )
        checkpoint_report(output_dir, report)

        runtime.ensure()
        warm_hit = run_corpus(
            corpora=corpora,
            store=cold_store,
            runtime=runtime,
            persistent_cache_enabled=True,
            exact_tm_enabled=True,
            group_size=args.group_size,
            max_completion_tokens=args.max_completion_tokens,
        )
        report["scenarios"]["warm_all_result_cache_hit"] = public_scenario(
            warm_hit
        )
        checkpoint_report(output_dir, report)

        warm_empty = run_corpus(
            corpora=corpora,
            store=warm_store,
            runtime=runtime,
            persistent_cache_enabled=True,
            exact_tm_enabled=True,
            group_size=args.group_size,
            max_completion_tokens=args.max_completion_tokens,
        )
        report["scenarios"]["warm_empty_cache"] = public_scenario(warm_empty)
        checkpoint_report(output_dir, report)

        requested_no_hit = run_corpus(
            corpora=corpora,
            store=requested_store,
            runtime=runtime,
            persistent_cache_enabled=True,
            exact_tm_enabled=False,
            group_size=args.group_size,
            max_completion_tokens=args.max_completion_tokens,
            requested_indices=odd,
        )
        report["scenarios"]["requested_blocks_no_hit"] = public_scenario(
            requested_no_hit
        )
        checkpoint_report(output_dir, report)

        seeded = seed_result_cache(
            corpora=corpora,
            baseline=cold,
            store=mixed_store,
            runtime=runtime,
            indices_by_language=even,
            group_size=args.group_size,
            max_completion_tokens=args.max_completion_tokens,
        )
        mixed = run_corpus(
            corpora=corpora,
            store=mixed_store,
            runtime=runtime,
            persistent_cache_enabled=True,
            exact_tm_enabled=False,
            group_size=args.group_size,
            max_completion_tokens=args.max_completion_tokens,
        )
        mixed_public = public_scenario(mixed)
        mixed_public["seeded_result_cache_entries"] = seeded
        report["scenarios"]["mixed_result_cache_hit"] = mixed_public
        report["comparisons"]["requested_no_hit_vs_mixed_misses"] = compare_outputs(
            requested_no_hit,
            {
                "outputs": [
                    output
                    for output in mixed["outputs"]
                    if int(output["index"]) % 2 == 1
                ]
            },
        )
        checkpoint_report(output_dir, report)

        stale_subset = OrderedDict(
            (language, items[: args.group_size])
            for language, items in corpora.items()
        )
        stale = run_corpus(
            corpora=stale_subset,
            store=warm_store,
            runtime=runtime,
            persistent_cache_enabled=True,
            exact_tm_enabled=False,
            group_size=args.group_size,
            max_completion_tokens=args.max_completion_tokens,
            top_p=0.90,
        )
        report["scenarios"]["stale_sampler_key"] = public_scenario(stale)
        checkpoint_report(output_dir, report)

        cache_disabled = run_corpus(
            corpora=stale_subset,
            store=None,
            runtime=runtime,
            persistent_cache_enabled=False,
            exact_tm_enabled=False,
            group_size=args.group_size,
            max_completion_tokens=args.max_completion_tokens,
        )
        report["scenarios"]["cache_disabled"] = public_scenario(cache_disabled)
        checkpoint_report(output_dir, report)
        runtime.shutdown()

        candidates = cold_store.list_tm_entries(limit=5_000)
        changed = cold_store.set_approved(
            [int(entry["id"]) for entry in candidates],
            True,
        )
        cleared = cold_store.clear_result_cache()
        exact_hit = run_corpus(
            corpora=corpora,
            store=cold_store,
            runtime=runtime,
            persistent_cache_enabled=True,
            exact_tm_enabled=True,
            group_size=args.group_size,
            max_completion_tokens=args.max_completion_tokens,
        )
        exact_public = public_scenario(exact_hit)
        exact_public["approved_entry_count"] = changed
        exact_public["cleared_result_cache_entries"] = cleared
        report["scenarios"]["stopped_all_approved_tm_hit"] = exact_public
        report["comparisons"]["cold_vs_approved_tm"] = compare_outputs(
            cold,
            exact_hit,
        )
        checkpoint_report(output_dir, report)

        corrupt_path = output_dir / "corrupt-cache.sqlite3"
        corrupt_path.write_bytes(b"not a sqlite database")
        corrupt_store = TranslationMemoryStore(corrupt_path)
        corrupt_engine = new_engine(
            language="Japanese",
            store=corrupt_store,
            runtime=runtime,
            persistent_cache_enabled=True,
            exact_tm_enabled=False,
            group_size=args.group_size,
            max_completion_tokens=args.max_completion_tokens,
        )
        corrupt_required = corrupt_engine.prepare_translation(
            _new_blocks(corpora["Japanese"][:1]),
            "",
        )
        report["scenarios"]["corrupt_db_fail_open_preflight"] = {
            "runtime_required": bool(corrupt_required),
            "store_enabled": corrupt_store.enabled,
            "disabled_reason_type": (
                corrupt_store.disabled_reason.split(":", 1)[0]
                if corrupt_store.disabled_reason
                else ""
            ),
            "database_preserved_sha256": _sha256_file(corrupt_path),
        }
        corrupt_store.close()
        checkpoint_report(output_dir, report)

        if not args.skip_prefix_matrix:
            prefix_results = []
            for cache_ram_mib, cache_prompt in (
                (0, True),
                (0, False),
                (256, True),
            ):
                prefix_results.append(
                    run_prefix_candidate(
                        corpora=corpora,
                        model=args.model,
                        cache_ram_mib=cache_ram_mib,
                        cache_prompt=cache_prompt,
                        repeat_count=args.prefix_repeat_count,
                        max_completion_tokens=args.max_completion_tokens,
                    )
                )
            report["prefix_cache_matrix"] = {
                "results": prefix_results,
                "selected": select_prefix_candidate(prefix_results),
                "selection_rule": (
                    "Choose the lowest cache-ram setting within 3% of the "
                    "fastest successful median; 8192 MiB is excluded."
                ),
            }
            checkpoint_report(output_dir, report)

        report["comparisons"]["cold_vs_stopped_result_hit"] = compare_outputs(
            cold,
            stopped_hit,
        )
        report["comparisons"]["cold_vs_warm_result_hit"] = compare_outputs(
            cold,
            warm_hit,
        )
        report["comparisons"]["cold_vs_warm_empty"] = compare_outputs(
            cold,
            warm_empty,
        )
        report["performance"] = {
            "stopped_result_hit_reduction_vs_stopped_empty_percent": (
                reduction_percent(
                    float(cold["elapsed_sec"]),
                    float(stopped_hit["elapsed_sec"]),
                )
            ),
            "warm_result_hit_reduction_vs_warm_empty_percent": reduction_percent(
                float(warm_empty["elapsed_sec"]),
                float(warm_hit["elapsed_sec"]),
            ),
            "approved_tm_reduction_vs_stopped_empty_percent": reduction_percent(
                float(cold["elapsed_sec"]),
                float(exact_hit["elapsed_sec"]),
            ),
        }
        report["automated_gate"] = automated_gate_report(report)
        report["runtime_identity"] = runtime.identity()
        report["status"] = "completed"
        report["completed_at"] = datetime.now().astimezone().isoformat()
        _write_json(
            output_dir / "private-comparisons.json",
            {
                "cold": cold["outputs"],
                "warm_empty": warm_empty["outputs"],
                "requested_blocks_no_hit": requested_no_hit["outputs"],
                "mixed": mixed["outputs"],
                "approved_tm": exact_hit["outputs"],
            },
        )
        _write_json(output_dir / "summary.json", report)
        return report
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["failed_at"] = datetime.now().astimezone().isoformat()
        _write_json(output_dir / "summary.json", report)
        raise
    finally:
        cold_store.close()
        warm_store.close()
        mixed_store.close()
        requested_store.close()
        try:
            runtime.shutdown()
        finally:
            restore_stopped_container(initial_model)
            report["final_container"] = inspect_container()
            _write_json(output_dir / "summary.json", report)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the persistent translation-result cache, approved exact "
            "translation memory, requested_blocks partial requests, and "
            "llama.cpp prefix-cache controls through the real Gemma product engine."
        )
    )
    parser.add_argument("--source-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_GEMMA_MAX_COMPLETION_TOKENS,
    )
    parser.add_argument("--prefix-repeat-count", type=int, default=3)
    parser.add_argument("--skip-prefix-matrix", action="store_true")
    args = parser.parse_args()
    if not 1 <= int(args.group_size) <= 12:
        parser.error("--group-size must be between 1 and 12")
    if int(args.max_completion_tokens) != 512:
        parser.error("--max-completion-tokens is fixed at 512")
    if not 1 <= int(args.prefix_repeat_count) <= 9:
        parser.error("--prefix-repeat-count must be between 1 and 9")

    report = run_benchmark(args)
    summary_path = Path(args.output_dir).expanduser().resolve() / "summary.json"
    print(f"status={report['status']}")
    print(f"summary={summary_path}")
    return (
        0
        if report.get("status") == "completed"
        and bool((report.get("automated_gate") or {}).get("passed"))
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
