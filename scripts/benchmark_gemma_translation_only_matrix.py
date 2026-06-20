#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import msgpack
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_common import (  # noqa: E402
    GEMMA_CONTAINER_NAMES,
    GEMMA_HEALTH_URLS,
    _stage_gemma_runtime,
    compose_up_detached,
    create_run_dir,
    load_preset,
    remove_containers,
    repo_relative_str,
    wait_for_health_urls,
    write_json,
)
from benchmark_gemma_runtime_matrix import (  # noqa: E402
    json_post,
    profile_concurrency,
    rank_successful_profiles,
)
from benchmark_stage_batched_archive_pipeline import (  # noqa: E402
    DEFAULT_OPTIMAL_PLUS_PRESET,
    SPEED_PROFILES,
    build_gemma_runtime_overrides,
    patch_preset_for_run,
)
from modules.translation.llm.custom_local_gemma import (  # noqa: E402
    DEFAULT_GEMMA_MAX_COMPLETION_TOKENS,
    DEFAULT_GEMMA_PROMPT_PROFILE,
    DEFAULT_GEMMA_RESPONSE_FORMAT_MODE,
    DEFAULT_GEMMA_RESPONSE_SCHEMA_MODE,
    DEFAULT_GEMMA_THINK_BRIEFLY_PROMPT,
    DEFAULT_GEMMA_TRANSLATION_MIN_P,
    DEFAULT_GEMMA_TRANSLATION_TEMPERATURE,
    DEFAULT_GEMMA_TRANSLATION_TOP_K,
    DEFAULT_GEMMA_TRANSLATION_TOP_P,
    STRICT_GEMMA_PROMPT_PROFILE,
    CustomLocalGemmaTranslation,
)
from modules.utils.gpu_metrics import collect_runtime_snapshot  # noqa: E402
from modules.utils.textblock import TextBlock  # noqa: E402
from modules.utils.translator_utils import extract_json_object  # noqa: E402


CTX_CANDIDATES = [4096, 3072, 2560, 2048, 1792, 1536, 1280, 1024, 768]
SYSTEM_TOKENS_EST = 430
DEFAULT_CHUNK_SIZE = 6
DEFAULT_MAX_CHUNKS = 96


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()


def unpack_msgpack(blob: bytes) -> Any:
    return msgpack.unpackb(blob, raw=False, strict_map_key=False)


def text_from_block(obj: Any) -> str:
    if isinstance(obj, dict) and obj.get("type") == "textblock":
        data = obj.get("data") or {}
    elif isinstance(obj, dict):
        data = obj
    else:
        data = getattr(obj, "__dict__", {})
    return str(data.get("text") or "")


def estimate_text_tokens(text: str) -> int:
    tokens = 0.0
    for ch in str(text or ""):
        if ch.isspace():
            tokens += 0.15
        elif "\u4e00" <= ch <= "\u9fff":
            tokens += 1.05
        elif "\u3040" <= ch <= "\u30ff" or "\uac00" <= ch <= "\ud7af":
            tokens += 0.95
        elif ch.isascii() and ch.isalnum():
            tokens += 0.35
        else:
            tokens += 0.55
    return max(1, int(math.ceil(tokens)))


def estimate_merged_prompt_tokens(texts: list[str]) -> int:
    return SYSTEM_TOKENS_EST + 28 + sum(estimate_text_tokens(text) for text in texts) + len(texts) * 10


def fits_ctx(texts: list[str], *, ctx_size: int, max_completion_tokens: int) -> bool:
    return estimate_merged_prompt_tokens(texts) + int(max_completion_tokens) <= int(ctx_size * 0.85)


def split_long_text_for_ctx(text: str, *, ctx_size: int, max_completion_tokens: int) -> list[str]:
    budget = max(120, int(ctx_size * 0.85) - SYSTEM_TOKENS_EST - int(max_completion_tokens) - 64)
    parts: list[str] = []
    current = ""
    separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
    queue = [str(text or "")]
    for separator in separators:
        next_queue: list[str] = []
        for item in queue:
            if estimate_text_tokens(item) <= budget:
                next_queue.append(item)
                continue
            chunks = item.split(separator)
            if len(chunks) <= 1:
                next_queue.append(item)
                continue
            for index, chunk in enumerate(chunks):
                suffix = separator if index + 1 < len(chunks) else ""
                next_queue.append(chunk + suffix)
        queue = next_queue
    for item in queue:
        for ch in item:
            candidate = current + ch
            if current and estimate_text_tokens(candidate) > budget:
                parts.append(current)
                current = ch
            else:
                current = candidate
    if current:
        parts.append(current)
    return [part for part in parts if part.strip()] or [str(text or "")]


def force_segment_text_for_retry(text: str, *, max_chars: int = 360) -> list[str]:
    normalized = str(text or "")
    candidates = [part for part in re.split(r"(\n+)", normalized) if part]
    if len([part for part in candidates if part.strip()]) <= 1:
        candidates = [part for part in re.split(r"(?<=[.!?。！？])(\s+)", normalized) if part]
    parts: list[str] = []
    current = ""
    for part in candidates:
        if not current:
            current = part
            continue
        if len(current) + len(part) <= max_chars:
            current += part
        else:
            parts.append(current)
            current = part
    if current:
        parts.append(current)
    if len(parts) <= 1 and len(normalized) > max_chars:
        parts = [normalized[index : index + max_chars] for index in range(0, len(normalized), max_chars)]
    return [part for part in parts if part.strip()] or [normalized]


def chunk_id(texts: list[str]) -> str:
    material = "\n".join(sha256_text(text) for text in texts)
    return sha256_text(material)[:16]


def iter_series_chunks(series_root: Path, *, chunk_size: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for series_path in sorted(series_root.rglob("*.seriesctpr")):
        con = sqlite3.connect(f"file:{series_path}?mode=ro&immutable=1", uri=True, timeout=30)
        try:
            rows = con.execute("select project_blob from embedded_projects")
            for row in rows:
                blob = bytes(row[0])
                child = sqlite3.connect(":memory:")
                try:
                    child.deserialize(blob)
                    for page_index, (row_blob,) in enumerate(
                        child.execute("select row_blob from page_state"),
                        start=1,
                    ):
                        data = unpack_msgpack(row_blob)
                        image_state = data.get("image_state") or {}
                        blocks_raw = image_state.get("blk_list") or []
                        texts = [text_from_block(block) for block in blocks_raw]
                        texts = [text for text in texts if text.strip()]
                        for start in range(0, len(texts), chunk_size):
                            current = texts[start : start + chunk_size]
                            if not current:
                                continue
                            chunks.append(
                                {
                                    "id": chunk_id(current),
                                    "texts": current,
                                    "block_count": len(current),
                                    "total_chars": sum(len(text) for text in current),
                                    "max_block_chars": max(len(text) for text in current),
                                    "page_index": page_index,
                                }
                            )
                finally:
                    child.close()
        finally:
            con.close()
    return chunks


def iter_page_snapshot_chunks(snapshot_path: Path, *, chunk_size: int) -> list[dict[str, Any]]:
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    pages = payload.get("pages", []) if isinstance(payload, dict) else []
    chunks: list[dict[str, Any]] = []
    for page_index, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            continue
        blocks_raw = page.get("blocks", [])
        if not isinstance(blocks_raw, list):
            continue
        texts = [text_from_block(block) for block in blocks_raw]
        texts = [text for text in texts if text.strip()]
        for start in range(0, len(texts), chunk_size):
            current = texts[start : start + chunk_size]
            if not current:
                continue
            chunks.append(
                {
                    "id": chunk_id(current),
                    "texts": current,
                    "block_count": len(current),
                    "total_chars": sum(len(text) for text in current),
                    "max_block_chars": max(len(text) for text in current),
                    "page_index": page_index,
                }
            )
    return chunks


def select_benchmark_chunks(chunks: list[dict[str, Any]], *, max_chunks: int) -> list[dict[str, Any]]:
    if max_chunks <= 0 or len(chunks) <= max_chunks:
        return list(chunks)
    longest = sorted(chunks, key=lambda item: (item["total_chars"], item["max_block_chars"]), reverse=True)
    selected: dict[str, dict[str, Any]] = {}
    for item in longest[: max(1, max_chunks // 4)]:
        selected[str(item["id"])] = item
    remaining_slots = max_chunks - len(selected)
    if remaining_slots <= 0:
        return list(selected.values())
    step = max(1, len(chunks) // remaining_slots)
    for index in range(0, len(chunks), step):
        selected[str(chunks[index]["id"])] = chunks[index]
        if len(selected) >= max_chunks:
            break
    return list(selected.values())


def coverage_report(chunks: list[dict[str, Any]], *, max_completion_tokens: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ctx in CTX_CANDIDATES:
        direct = sum(
            1
            for chunk in chunks
            if fits_ctx(chunk["texts"], ctx_size=ctx, max_completion_tokens=max_completion_tokens)
        )
        out[str(ctx)] = {
            "direct_chunk_count": direct,
            "overflow_chunk_count": len(chunks) - direct,
            "direct_percent": round((direct / len(chunks) * 100.0) if chunks else 0.0, 4),
        }
    return out


def build_engine(*, model: str, source_lang: str, target_lang: str, max_tokens: int) -> CustomLocalGemmaTranslation:
    engine = CustomLocalGemmaTranslation()
    engine.model = model
    engine.source_lang = source_lang
    engine.target_lang = target_lang
    engine.max_tokens = int(max_tokens)
    engine.temperature = DEFAULT_GEMMA_TRANSLATION_TEMPERATURE
    engine.top_k = DEFAULT_GEMMA_TRANSLATION_TOP_K
    engine.top_p = DEFAULT_GEMMA_TRANSLATION_TOP_P
    engine.min_p = DEFAULT_GEMMA_TRANSLATION_MIN_P
    engine.prompt_profile = DEFAULT_GEMMA_PROMPT_PROFILE
    engine.response_format_mode = DEFAULT_GEMMA_RESPONSE_FORMAT_MODE
    engine.response_schema_mode = DEFAULT_GEMMA_RESPONSE_SCHEMA_MODE
    engine.think_briefly_prompt = DEFAULT_GEMMA_THINK_BRIEFLY_PROMPT
    return engine


def build_translation_request(
    engine: CustomLocalGemmaTranslation,
    texts: list[str],
    *,
    prompt_profile: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    blocks = _blocks_from_texts(texts)
    expected_keys = engine._expected_block_keys(blocks)
    system_prompt = engine._build_system_prompt("", prompt_profile=prompt_profile or engine.prompt_profile)
    user_prompt = engine._build_contextual_merged_user_prompt(blocks, expected_keys)
    payload = {
        "model": engine.model,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ],
        "temperature": engine.temperature,
        "top_k": engine.top_k,
        "top_p": engine.top_p,
        "min_p": engine.min_p,
        "max_completion_tokens": engine.max_tokens,
        "response_format": engine._build_response_format(user_prompt, expected_keys=expected_keys),
    }
    return payload, expected_keys


def _blocks_from_texts(texts: list[str]) -> list[TextBlock]:
    return [
        TextBlock(text_bbox=np.array([0, index * 10, 100, index * 10 + 8]), text=text)
        for index, text in enumerate(texts)
    ]


def build_single_block_translation_request(
    engine: CustomLocalGemmaTranslation,
    texts: list[str],
    target_index: int,
    *,
    prompt_profile: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    blocks = _blocks_from_texts(texts)
    system_prompt = engine._build_system_prompt("", prompt_profile=prompt_profile or engine.prompt_profile)
    user_prompt = engine._build_contextual_single_block_user_prompt(blocks, target_index)
    expected_keys = ["translation"]
    payload = {
        "model": engine.model,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ],
        "temperature": engine.temperature,
        "top_k": engine.top_k,
        "top_p": engine.top_p,
        "min_p": engine.min_p,
        "max_completion_tokens": engine.max_tokens,
        "response_format": engine._build_response_format(user_prompt, expected_keys=expected_keys),
    }
    return payload, expected_keys


def _with_strict_payload(request: dict[str, Any], strict_payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(request)
    out["strict_payload"] = strict_payload
    return out


def build_payloads_for_chunk(
    engine: CustomLocalGemmaTranslation,
    texts: list[str],
    *,
    ctx_size: int,
    max_completion_tokens: int,
) -> tuple[list[dict[str, Any]], str]:
    if fits_ctx(texts, ctx_size=ctx_size, max_completion_tokens=max_completion_tokens):
        payload, expected = build_translation_request(engine, texts)
        return [{"payload": payload, "expected_keys": expected, "mode": "direct"}], "direct"

    payloads: list[dict[str, Any]] = []
    mode = "split_blocks"
    for text in texts:
        if fits_ctx([text], ctx_size=ctx_size, max_completion_tokens=max_completion_tokens):
            payload, expected = build_single_block_translation_request(engine, [text], 0)
            strict_payload, _ = build_single_block_translation_request(
                engine,
                [text],
                0,
                prompt_profile=STRICT_GEMMA_PROMPT_PROFILE,
            )
            payloads.append(
                _with_strict_payload(
                    {"payload": payload, "expected_keys": expected, "mode": "split_single_block", "source_text": text},
                    strict_payload,
                )
            )
            continue
        mode = "segment_block"
        for segment in split_long_text_for_ctx(text, ctx_size=ctx_size, max_completion_tokens=max_completion_tokens):
            payload, expected = build_single_block_translation_request(engine, [segment], 0)
            strict_payload, _ = build_single_block_translation_request(
                engine,
                [segment],
                0,
                prompt_profile=STRICT_GEMMA_PROMPT_PROFILE,
            )
            payloads.append(
                _with_strict_payload(
                    {"payload": payload, "expected_keys": expected, "mode": "segment_block", "source_text": segment},
                    strict_payload,
                )
            )
    return payloads, mode


def build_recovery_payloads_for_chunk(
    engine: CustomLocalGemmaTranslation,
    texts: list[str],
    *,
    ctx_size: int,
    max_completion_tokens: int,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    if fits_ctx(texts, ctx_size=ctx_size, max_completion_tokens=max_completion_tokens):
        for index, _text in enumerate(texts):
            payload, expected = build_single_block_translation_request(engine, texts, index)
            strict_payload, _ = build_single_block_translation_request(
                engine,
                texts,
                index,
                prompt_profile=STRICT_GEMMA_PROMPT_PROFILE,
            )
            payloads.append(
                _with_strict_payload(
                    {
                        "payload": payload,
                        "expected_keys": expected,
                        "mode": "recovered_single_block_context",
                        "source_text": _text,
                    },
                    strict_payload,
                )
            )
        return payloads
    for text in texts:
        if fits_ctx([text], ctx_size=ctx_size, max_completion_tokens=max_completion_tokens):
            payload, expected = build_single_block_translation_request(engine, [text], 0)
            strict_payload, _ = build_single_block_translation_request(
                engine,
                [text],
                0,
                prompt_profile=STRICT_GEMMA_PROMPT_PROFILE,
            )
            payloads.append(
                _with_strict_payload(
                    {
                        "payload": payload,
                        "expected_keys": expected,
                        "mode": "recovered_split_single_block",
                        "source_text": text,
                    },
                    strict_payload,
                )
            )
            continue
        for segment in split_long_text_for_ctx(text, ctx_size=ctx_size, max_completion_tokens=max_completion_tokens):
            payload, expected = build_single_block_translation_request(engine, [segment], 0)
            strict_payload, _ = build_single_block_translation_request(
                engine,
                [segment],
                0,
                prompt_profile=STRICT_GEMMA_PROMPT_PROFILE,
            )
            payloads.append(
                _with_strict_payload(
                    {
                        "payload": payload,
                        "expected_keys": expected,
                        "mode": "recovered_segment_block",
                        "source_text": segment,
                    },
                    strict_payload,
                )
            )
    return payloads


def build_forced_segment_payloads(engine: CustomLocalGemmaTranslation, text: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for segment in force_segment_text_for_retry(text):
        payload, expected = build_single_block_translation_request(engine, [segment], 0)
        strict_payload, _ = build_single_block_translation_request(
            engine,
            [segment],
            0,
            prompt_profile=STRICT_GEMMA_PROMPT_PROFILE,
        )
        payloads.append(
            _with_strict_payload(
                {
                    "payload": payload,
                    "expected_keys": expected,
                    "mode": "forced_segment_block",
                    "source_text": segment,
                },
                strict_payload,
            )
        )
    return payloads


def execute_payload(index: int, request: dict[str, Any], *, timeout_sec: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = json_post(
            "http://127.0.0.1:18080/v1/chat/completions",
            request["payload"],
            timeout_sec=timeout_sec,
        )
        elapsed_sec = time.perf_counter() - started
        choices = response.get("choices", []) if isinstance(response.get("choices"), list) else []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        finish_reason = choice.get("finish_reason")
        content = str(((choice.get("message") or {}) if isinstance(choice.get("message"), dict) else {}).get("content") or "")
        usage = response.get("usage", {}) if isinstance(response.get("usage"), dict) else {}
        parsed = extract_json_object(content)
        expected_keys = [str(key) for key in request["expected_keys"]]
        missing = [key for key in expected_keys if key not in parsed]
        extra = [str(key) for key in parsed if str(key) not in expected_keys]
        channel_token = any(
            "<|channel>" in str(key)
            or "<channel|>" in str(key)
            or "<|channel>" in str(value)
            or "<channel|>" in str(value)
            for key, value in parsed.items()
        )
        status = "passed"
        if finish_reason == "length" or missing or extra or channel_token:
            status = "failed"
        return {
            "index": index,
            "mode": request.get("mode", ""),
            "status": status,
            "elapsed_sec": round(elapsed_sec, 3),
            "finish_reason": finish_reason,
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
            "content_hash": sha256_text(content),
            "content_length": len(content),
            "missing_key_count": len(missing),
            "extra_key_count": len(extra),
            "channel_token": channel_token,
        }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, KeyError) as exc:
        return {
            "index": index,
            "mode": request.get("mode", ""),
            "status": "failed",
            "error": str(exc),
            "elapsed_sec": round(time.perf_counter() - started, 3),
        }


def execute_payload_with_retry(
    index: int,
    request: dict[str, Any],
    *,
    timeout_sec: int,
    max_attempts: int = 2,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for _ in range(max(1, int(max_attempts))):
        result = execute_payload(index, request, timeout_sec=timeout_sec)
        attempts.append(result)
        if result.get("status") == "passed":
            break
    if len(attempts) == 1:
        attempts[0]["retry_count"] = 0
        return attempts[0]
    total_elapsed = sum(float(item.get("elapsed_sec", 0.0) or 0.0) for item in attempts)
    final = dict(attempts[-1])
    final["elapsed_sec"] = round(total_elapsed, 3)
    final["retry_count"] = len(attempts) - 1
    final["previous_failure_count"] = sum(1 for item in attempts[:-1] if item.get("status") != "passed")
    return final


def profile_runtime_overrides(profile: str) -> dict[str, Any]:
    speed_profile = SPEED_PROFILES.get(profile, {})
    return build_gemma_runtime_overrides(
        context_size=speed_profile.get("gemma_ctx_size"),
        threads=speed_profile.get("gemma_threads"),
        n_gpu_layers=speed_profile.get("gemma_gpu_layers"),
        n_parallel=speed_profile.get("gemma_n_parallel"),
        predict=speed_profile.get("gemma_predict"),
        batch_size=speed_profile.get("gemma_batch_size"),
        ubatch_size=speed_profile.get("gemma_ubatch_size"),
        cache_type_k=speed_profile.get("gemma_cache_type_k"),
        cache_type_v=speed_profile.get("gemma_cache_type_v"),
        flash_attn=speed_profile.get("gemma_flash_attn"),
        no_warmup=speed_profile.get("gemma_no_warmup"),
    )


def run_profile(
    *,
    profile: str,
    chunks: list[dict[str, Any]],
    suite_dir: Path,
    source_lang: str,
    target_lang: str,
    health_timeout_sec: int,
    request_timeout_sec: int,
    max_completion_tokens: int,
    concurrency: int,
    warmup_requests: int,
) -> dict[str, Any]:
    profile_dir = suite_dir / profile
    profile_dir.mkdir(parents=True, exist_ok=True)
    preset, preset_path = load_preset(DEFAULT_OPTIMAL_PLUS_PRESET)
    overrides = profile_runtime_overrides(profile)
    preset = patch_preset_for_run(preset, ocr_mode="optimal-plus", gemma_runtime_overrides=overrides)
    gemma_cfg = preset.get("gemma", {}) if isinstance(preset.get("gemma"), dict) else {}
    model = str(gemma_cfg.get("model", "gemma-4-26B-IQ4_NL.gguf"))
    ctx_size = int(overrides.get("context_size", gemma_cfg.get("context_size", 4096)) or 4096)
    engine = build_engine(
        model=model,
        source_lang=source_lang,
        target_lang=target_lang,
        max_tokens=max_completion_tokens,
    )

    remove_containers(GEMMA_CONTAINER_NAMES)
    runtime_dir = profile_dir / "runtime"
    staged = _stage_gemma_runtime(preset, runtime_dir)
    compose_path = Path(staged["compose_path"])
    compose_started = time.perf_counter()
    compose_up_detached(compose_path, cwd=ROOT, project_directory=ROOT, force_recreate=True)
    compose_elapsed_sec = time.perf_counter() - compose_started
    health_started = time.perf_counter()
    health_failures = wait_for_health_urls(list(GEMMA_HEALTH_URLS), timeout_sec=health_timeout_sec, poll_interval_sec=2.0)
    health_elapsed_sec = time.perf_counter() - health_started

    chunk_jobs: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    direct_chunks = 0
    fallback_chunks = 0
    for chunk in chunks:
        payloads, mode = build_payloads_for_chunk(
            engine,
            chunk["texts"],
            ctx_size=ctx_size,
            max_completion_tokens=max_completion_tokens,
        )
        if mode == "direct":
            direct_chunks += 1
        else:
            fallback_chunks += 1
        for payload in payloads:
            payload["chunk_id"] = chunk["id"]
            requests.append(payload)
        chunk_jobs.append(
            {
                "chunk_id": chunk["id"],
                "texts": chunk["texts"],
                "primary_mode": mode,
                "primary_payloads": payloads,
            }
        )

    status = "passed"
    failure_reason = ""
    results: list[dict[str, Any]] = []
    recovered_failure_count = 0
    strict_recovery_count = 0
    forced_segment_recovery_count = 0
    before_snapshot = collect_runtime_snapshot(GEMMA_CONTAINER_NAMES)
    started = time.perf_counter()
    if health_failures:
        status = "failed"
        failure_reason = f"health timeout: {health_failures}"
    else:
        for warmup_index in range(max(0, int(warmup_requests))):
            if not requests:
                break
            execute_payload(-1 - warmup_index, requests[0], timeout_sec=request_timeout_sec)
        if max(1, int(concurrency)) <= 1:
            request_index = 0

            def execute_request_with_recoveries(request: dict[str, Any]) -> bool:
                nonlocal forced_segment_recovery_count, request_index, strict_recovery_count
                result = execute_payload_with_retry(request_index, request, timeout_sec=request_timeout_sec)
                request_index += 1
                results.append(result)
                if result.get("status") == "passed":
                    return True
                if request.get("strict_payload"):
                    result["status"] = "recovered"
                    strict_request = dict(request)
                    strict_request["payload"] = request["strict_payload"]
                    strict_request["mode"] = "strict_" + str(request.get("mode", ""))
                    strict_result = execute_payload_with_retry(
                        request_index,
                        strict_request,
                        timeout_sec=request_timeout_sec,
                    )
                    request_index += 1
                    results.append(strict_result)
                    if strict_result.get("status") == "passed":
                        strict_recovery_count += 1
                        return True
                    result = strict_result
                source_text = request.get("source_text")
                mode = str(request.get("mode", ""))
                if source_text and "forced_segment" not in mode:
                    result["status"] = "recovered"
                    segment_payloads = build_forced_segment_payloads(engine, str(source_text))
                    for segment_request in segment_payloads:
                        segment_request["chunk_id"] = request.get("chunk_id")
                        segment_result = execute_payload_with_retry(
                            request_index,
                            segment_request,
                            timeout_sec=request_timeout_sec,
                        )
                        request_index += 1
                        results.append(segment_result)
                        if segment_result.get("status") == "passed":
                            continue
                        if segment_request.get("strict_payload"):
                            segment_result["status"] = "recovered"
                            strict_segment_request = dict(segment_request)
                            strict_segment_request["payload"] = segment_request["strict_payload"]
                            strict_segment_request["mode"] = "strict_" + str(segment_request.get("mode", ""))
                            strict_segment_result = execute_payload_with_retry(
                                request_index,
                                strict_segment_request,
                                timeout_sec=request_timeout_sec,
                            )
                            request_index += 1
                            results.append(strict_segment_result)
                            if strict_segment_result.get("status") == "passed":
                                strict_recovery_count += 1
                                continue
                        return False
                    forced_segment_recovery_count += 1
                    return True
                return False

            for job in chunk_jobs:
                job_failed = False
                for request in job["primary_payloads"]:
                    if not execute_request_with_recoveries(request):
                        job_failed = True
                        break
                if not job_failed:
                    continue
                if job["primary_mode"] == "direct" and len(job["texts"]) > 1:
                    recovered_failure_count += 1
                    results[-1]["status"] = "recovered"
                    recovery_payloads = build_recovery_payloads_for_chunk(
                        engine,
                        job["texts"],
                        ctx_size=ctx_size,
                        max_completion_tokens=max_completion_tokens,
                    )
                    fallback_chunks += 1
                    recovered = True
                    for request in recovery_payloads:
                        request["chunk_id"] = job["chunk_id"]
                        if not execute_request_with_recoveries(request):
                            recovered = False
                            break
                    if recovered:
                        continue
                status = "failed"
                failure_reason = str(results[-1].get("error") or results[-1])
                break
        else:
            with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
                future_map = {
                    executor.submit(execute_payload, index, request, timeout_sec=request_timeout_sec): index
                    for index, request in enumerate(requests)
                }
                for future in as_completed(future_map):
                    result = future.result()
                    results.append(result)
                    if result.get("status") != "passed":
                        status = "failed"
                        failure_reason = str(result.get("error") or result)
            results.sort(key=lambda item: int(item.get("index", 0) or 0))
    total_elapsed_sec = time.perf_counter() - started
    after_snapshot = collect_runtime_snapshot(GEMMA_CONTAINER_NAMES)
    allow_failure = bool(SPEED_PROFILES.get(profile, {}).get("allow_failure", False))
    if status == "failed" and allow_failure:
        status = "shadow_failed"

    passed_results = [item for item in results if item.get("status") == "passed"]
    failed_results = [item for item in results if item.get("status") == "failed"]
    elapsed_values = sorted(float(item.get("elapsed_sec", 0.0) or 0.0) for item in passed_results)
    completion_tokens = sum(int(item.get("completion_tokens", 0) or 0) for item in passed_results)
    summary = {
        "profile": profile,
        "status": status,
        "failure_reason": failure_reason,
        "preset_path": repo_relative_str(preset_path),
        "gemma_runtime_overrides": overrides,
        "chunk_count": len(chunks),
        "planned_request_count": len(requests),
        "request_count": len(results),
        "direct_chunk_count": direct_chunks,
        "fallback_count": fallback_chunks,
        "recovered_failure_count": recovered_failure_count,
        "strict_recovery_count": strict_recovery_count,
        "forced_segment_recovery_count": forced_segment_recovery_count,
        "compose_elapsed_sec": round(compose_elapsed_sec, 3),
        "health_elapsed_sec": round(health_elapsed_sec, 3),
        "translation_total_elapsed_sec": round(total_elapsed_sec, 3),
        "warm_p50_sec": elapsed_values[len(elapsed_values) // 2] if elapsed_values else None,
        "warm_p95_sec": elapsed_values[min(len(elapsed_values) - 1, int(len(elapsed_values) * 0.95))] if elapsed_values else None,
        "best_tps": round(max((int(item.get("completion_tokens", 0) or 0) / max(float(item.get("elapsed_sec", 0.0) or 0.0), 0.001)) for item in passed_results), 3) if passed_results else None,
        "aggregate_tps": round(completion_tokens / max(total_elapsed_sec, 0.001), 3),
        "json_failure_count": len(failed_results),
        "missing_key_count": sum(int(item.get("missing_key_count", 0) or 0) for item in failed_results),
        "extra_key_count": sum(int(item.get("extra_key_count", 0) or 0) for item in failed_results),
        "channel_token_count": sum(1 for item in failed_results if item.get("channel_token")),
        "warmup_requests": max(0, int(warmup_requests)),
        "concurrency": max(1, int(concurrency)),
        "gpu_before": before_snapshot.get("gpu", {}),
        "gpu_after": after_snapshot.get("gpu", {}),
        "request_results": [
            {
                key: item.get(key)
                for key in (
                    "index",
                    "mode",
                    "status",
                    "elapsed_sec",
                    "finish_reason",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "retry_count",
                    "previous_failure_count",
                    "content_hash",
                    "content_length",
                    "missing_key_count",
                    "extra_key_count",
                    "channel_token",
                    "error",
                )
            }
            for item in results
        ],
    }
    write_json(profile_dir / "summary.json", summary)
    return summary


def rank_translation_profiles(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    passed = [item for item in results if item.get("status") == "passed"]
    return sorted(
        [
            {
                "profile": item.get("profile", ""),
                "translation_total_elapsed_sec": item.get("translation_total_elapsed_sec"),
                "warm_p50_sec": item.get("warm_p50_sec"),
                "first_request_latency_sec": None,
                "fallback_count": item.get("fallback_count", 0),
                "best_tps": item.get("best_tps"),
                "aggregate_tps": item.get("aggregate_tps"),
                "min_gpu_free_mb": ((item.get("gpu_after") or {}).get("primary") or {}).get("memory_free_mb")
                if isinstance(item.get("gpu_after"), dict)
                else None,
                "gemma_runtime_overrides": item.get("gemma_runtime_overrides", {}),
            }
            for item in passed
        ],
        key=lambda item: (
            float(item["translation_total_elapsed_sec"] if item["translation_total_elapsed_sec"] is not None else 10**9),
            float(item["warm_p50_sec"] if item["warm_p50_sec"] is not None else 10**9),
            int(item.get("fallback_count", 0) or 0),
        ),
    )


def render_report(results: list[dict[str, Any]], coverage: dict[str, Any]) -> str:
    lines = [
        "# Gemma Translation-Only Speed Matrix",
        "",
        "## Coverage",
        "",
        "| ctx | direct % | overflow chunks |",
        "|---:|---:|---:|",
    ]
    for ctx in CTX_CANDIDATES:
        row = coverage.get(str(ctx), {})
        lines.append(f"| {ctx} | {row.get('direct_percent', '')} | {row.get('overflow_chunk_count', '')} |")
    lines.extend(
        [
            "",
            "## Profiles",
            "",
            "| profile | status | elapsed_sec | chunks | requests | fallback | best_tps | agg_tps | gpu_free_after_mb |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in results:
        primary = ((item.get("gpu_after") or {}).get("primary") or {}) if isinstance(item.get("gpu_after"), dict) else {}
        lines.append(
            "| {profile} | {status} | {elapsed} | {chunks} | {requests} | {fallback} | {best_tps} | {agg_tps} | {free} |".format(
                profile=item.get("profile", ""),
                status=item.get("status", ""),
                elapsed=item.get("translation_total_elapsed_sec", ""),
                chunks=item.get("chunk_count", ""),
                requests=item.get("request_count", ""),
                fallback=item.get("fallback_count", ""),
                best_tps=item.get("best_tps", ""),
                agg_tps=item.get("aggregate_tps", ""),
                free=primary.get("memory_free_mb", ""),
            )
        )
    ranked = rank_translation_profiles(results)
    if ranked:
        lines.extend(["", f"Fastest successful profile: `{ranked[0]['profile']}`"])
    lines.append("")
    lines.append("Raw OCR text, translations, local source paths, images, and archives are intentionally omitted.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run actual Gemma translation-only speed profiles on an anonymized local corpus.")
    parser.add_argument("--series-root", default="", help="Folder containing .seriesctpr files.")
    parser.add_argument("--page-snapshots", default="", help="page_snapshots.json produced by an OCR/detect run.")
    parser.add_argument("--profiles", nargs="+", choices=tuple(SPEED_PROFILES.keys()), default=["ctx2560-fast-archive", "ctx2048-gpu23-fast", "ctx1792-gpu23-extreme", "ctx1536-gpu23-shadow"])
    parser.add_argument("--source-lang", default="English")
    parser.add_argument("--target-lang", default="Korean")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--max-chunks", type=int, default=DEFAULT_MAX_CHUNKS)
    parser.add_argument("--max-completion-tokens", type=int, default=DEFAULT_GEMMA_MAX_COMPLETION_TOKENS)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--request-timeout-sec", type=int, default=180)
    parser.add_argument("--concurrency", type=int, default=0, help="0 means use each profile's n_parallel.")
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--coverage-only", action="store_true")
    args = parser.parse_args()

    if bool(args.series_root) == bool(args.page_snapshots):
        raise ValueError("Provide exactly one of --series-root or --page-snapshots")
    series_root = Path(args.series_root).expanduser().resolve() if args.series_root else None
    snapshot_path = Path(args.page_snapshots).expanduser().resolve() if args.page_snapshots else None
    if series_root is not None and not series_root.exists():
        raise FileNotFoundError(f"Series root not found: {series_root}")
    if snapshot_path is not None and not snapshot_path.is_file():
        raise FileNotFoundError(f"Page snapshots not found: {snapshot_path}")

    suite_dir = Path(args.output_dir).resolve() if args.output_dir else create_run_dir(
        "gemma_translation_only_matrix",
        root=Path(args.output_root).resolve() if args.output_root else None,
    )
    suite_dir.mkdir(parents=True, exist_ok=True)
    chunks_all = (
        iter_series_chunks(series_root, chunk_size=max(1, int(args.chunk_size)))
        if series_root is not None
        else iter_page_snapshot_chunks(snapshot_path, chunk_size=max(1, int(args.chunk_size)))
    )
    selected_chunks = select_benchmark_chunks(chunks_all, max_chunks=int(args.max_chunks))
    coverage = coverage_report(chunks_all, max_completion_tokens=int(args.max_completion_tokens))
    corpus_summary = {
        "source_kind": "seriesctpr" if series_root is not None else "page_snapshots",
        "series_file_count": len(list(series_root.rglob("*.seriesctpr"))) if series_root is not None else 0,
        "chunk_count": len(chunks_all),
        "selected_chunk_count": len(selected_chunks),
        "chunk_size": int(args.chunk_size),
        "coverage": coverage,
        "privacy_note": "No raw OCR text, translations, source paths, or project names are stored.",
    }
    write_json(suite_dir / "corpus_summary.json", corpus_summary)
    if args.coverage_only:
        (suite_dir / "matrix_report.md").write_text(render_report([], coverage), encoding="utf-8")
        return 0

    results: list[dict[str, Any]] = []
    for profile in args.profiles:
        summary = run_profile(
            profile=profile,
            chunks=selected_chunks,
            suite_dir=suite_dir,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            health_timeout_sec=int(args.timeout_sec),
            request_timeout_sec=int(args.request_timeout_sec),
            max_completion_tokens=int(args.max_completion_tokens),
            concurrency=profile_concurrency(profile, args.concurrency),
            warmup_requests=int(args.warmup_requests),
        )
        results.append(summary)
        payload = {
            "profiles": list(args.profiles),
            "coverage": coverage,
            "ranking": rank_translation_profiles(results),
            "results": results,
        }
        write_json(suite_dir / "matrix_summary.json", payload)
        (suite_dir / "matrix_report.md").write_text(render_report(results, coverage), encoding="utf-8")
        if summary.get("status") == "failed":
            break
    remove_containers(GEMMA_CONTAINER_NAMES)
    return 1 if any(item.get("status") == "failed" for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
