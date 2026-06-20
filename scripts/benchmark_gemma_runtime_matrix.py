#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_common import (
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
from benchmark_stage_batched_archive_pipeline import (
    DEFAULT_OPTIMAL_PLUS_PRESET,
    SPEED_PROFILES,
    build_gemma_runtime_overrides,
    patch_preset_for_run,
)
from modules.utils.gpu_metrics import collect_runtime_snapshot


DEFAULT_PROFILES = [
    "baseline-safe",
    "ctx3072-fast-archive",
    "ctx2560-fast-archive",
    "ctx2048-gpu23-fast",
    "ctx1792-gpu23-extreme",
    "ctx1536-gpu23-shadow",
    "ctx1280-gpu23-shadow",
    "ctx1024-gpu23-shadow",
    "ctx768-gpu23-shadow",
    "ctx3072-gpu24-extreme",
    "ctx2560-gpu24-extreme",
    "ctx2560-gpu25-danger",
    "ctx2048-gpu24-extreme",
    "ctx1792-gpu24-shadow",
    "ctx2048-gpu25-danger",
    "ctx2048-gpu26-danger",
    "ctx2048-threads14",
    "ctx2048-batch1024",
    "ctx2048-batch2048",
    "ctx2048-flash-attn",
    "ctx2048-q8-kv",
    "ctx2048-no-warmup",
    "ctx2048-np2",
    "danger-shadow",
]
SYNTHETIC_SYSTEM_PROMPT = (
    "Translate the user's JSON object of comic OCR lines from English to Korean. "
    "Return exactly one JSON object with the same keys and no extra text."
)
SYNTHETIC_USER_PAYLOAD = {
    "block_0": "This is a short benchmark sentence.",
    "block_1": "Keep the JSON keys unchanged.",
    "block_2": "Return concise Korean comic dialogue.",
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()


def json_post(url: str, payload: dict[str, Any], *, timeout_sec: int) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        raw = response.read().decode("utf-8", errors="replace")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object response")
    return parsed


def build_chat_payload(model: str, *, max_tokens: int = 128) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYNTHETIC_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(SYNTHETIC_USER_PAYLOAD, ensure_ascii=False)},
        ],
        "temperature": 0.7,
        "top_k": 64,
        "top_p": 0.95,
        "min_p": 0.0,
        "max_tokens": int(max_tokens),
        "response_format": {"type": "json_object"},
    }


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


def profile_concurrency(profile: str, explicit_concurrency: int | None = None) -> int:
    if explicit_concurrency and explicit_concurrency > 0:
        return int(explicit_concurrency)
    overrides = profile_runtime_overrides(profile)
    return max(1, int(overrides.get("n_parallel", 1) or 1))


def summarize_request_timings(requests: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [item for item in requests if item.get("status") == "passed"]
    elapsed_values = sorted(float(item.get("elapsed_sec", 0.0) or 0.0) for item in passed)
    warm_values = sorted(float(item.get("elapsed_sec", 0.0) or 0.0) for item in passed[1:])
    tps_values = [float(item.get("completion_tps", 0.0) or 0.0) for item in passed]
    return {
        "passed_request_count": len(passed),
        "first_request_latency_sec": float(passed[0].get("elapsed_sec", 0.0) or 0.0) if passed else None,
        "warm_p50_sec": warm_values[len(warm_values) // 2] if warm_values else (elapsed_values[len(elapsed_values) // 2] if elapsed_values else None),
        "warm_p95_sec": warm_values[min(len(warm_values) - 1, int(len(warm_values) * 0.95))] if warm_values else None,
        "best_tps": max(tps_values) if tps_values else None,
        "total_request_elapsed_sec": round(sum(elapsed_values), 3),
    }


def rank_successful_profiles(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for item in results:
        if item.get("status") != "passed":
            continue
        timing = summarize_request_timings(item.get("requests", []) if isinstance(item.get("requests"), list) else [])
        ranked.append(
            {
                "profile": item.get("profile", ""),
                "status": item.get("status", ""),
                "translation_total_elapsed_sec": timing.get("total_request_elapsed_sec"),
                "warm_p50_sec": timing.get("warm_p50_sec"),
                "first_request_latency_sec": timing.get("first_request_latency_sec"),
                "best_tps": timing.get("best_tps"),
                "fallback_count": int(item.get("fallback_count", 0) or 0),
                "min_gpu_free_mb": ((item.get("gpu_after") or {}).get("primary") or {}).get("memory_free_mb")
                if isinstance(item.get("gpu_after"), dict)
                else None,
                "gemma_runtime_overrides": item.get("gemma_runtime_overrides", {}),
            }
        )
    return sorted(
        ranked,
        key=lambda item: (
            float(item["translation_total_elapsed_sec"] if item["translation_total_elapsed_sec"] is not None else 10**9),
            float(item["warm_p50_sec"] if item["warm_p50_sec"] is not None else 10**9),
            float(item["first_request_latency_sec"] if item["first_request_latency_sec"] is not None else 10**9),
            int(item.get("fallback_count", 0) or 0),
        ),
    )


def run_profile(
    *,
    profile: str,
    suite_dir: Path,
    health_timeout_sec: int,
    request_timeout_sec: int,
    request_count: int,
    max_tokens: int,
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

    remove_containers(GEMMA_CONTAINER_NAMES)
    runtime_dir = profile_dir / "runtime"
    staged = _stage_gemma_runtime(preset, runtime_dir)
    compose_path = Path(staged["compose_path"])
    started = time.perf_counter()
    compose_up_detached(compose_path, cwd=ROOT, project_directory=ROOT, force_recreate=True)
    compose_elapsed_sec = time.perf_counter() - started
    health_started = time.perf_counter()
    health_failures = wait_for_health_urls(list(GEMMA_HEALTH_URLS), timeout_sec=health_timeout_sec, poll_interval_sec=2.0)
    health_elapsed_sec = time.perf_counter() - health_started
    before_snapshot = collect_runtime_snapshot(GEMMA_CONTAINER_NAMES)
    requests: list[dict[str, Any]] = []
    status = "passed"
    failure_reason = ""
    if health_failures:
        status = "failed"
        failure_reason = f"health timeout: {health_failures}"
    else:
        payload = build_chat_payload(model, max_tokens=max_tokens)
        for _warmup_index in range(max(0, int(warmup_requests))):
            try:
                json_post(
                    "http://127.0.0.1:18080/v1/chat/completions",
                    payload,
                    timeout_sec=request_timeout_sec,
                )
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
                break

        def execute_request(index: int) -> dict[str, Any]:
            request_started = time.perf_counter()
            try:
                response = json_post(
                    "http://127.0.0.1:18080/v1/chat/completions",
                    payload,
                    timeout_sec=request_timeout_sec,
                )
                elapsed_sec = time.perf_counter() - request_started
                usage = response.get("usage", {}) if isinstance(response.get("usage"), dict) else {}
                choices = response.get("choices", []) if isinstance(response.get("choices"), list) else []
                content = ""
                if choices and isinstance(choices[0], dict):
                    message = choices[0].get("message", {})
                    if isinstance(message, dict):
                        content = str(message.get("content") or "")
                completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                return {
                    "index": index,
                    "status": "passed",
                    "elapsed_sec": round(elapsed_sec, 3),
                    "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    "completion_tokens": completion_tokens,
                    "total_tokens": int(usage.get("total_tokens", 0) or 0),
                    "completion_tps": round(completion_tokens / elapsed_sec, 3) if elapsed_sec > 0 else 0.0,
                    "content_hash": sha256_text(content),
                    "content_length": len(content),
                }
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
                return {"index": index, "status": "failed", "error": str(exc)}

        if int(concurrency) <= 1:
            for index in range(int(request_count)):
                request_result = execute_request(index)
                requests.append(request_result)
                if request_result.get("status") == "failed":
                    status = "failed"
                    failure_reason = str(request_result.get("error", ""))
                    break
        else:
            with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
                future_map = {
                    executor.submit(execute_request, index): index
                    for index in range(int(request_count))
                }
                for future in as_completed(future_map):
                    request_result = future.result()
                    requests.append(request_result)
                    if request_result.get("status") == "failed":
                        status = "failed"
                        failure_reason = str(request_result.get("error", ""))
            requests.sort(key=lambda item: int(item.get("index", 0) or 0))
    after_snapshot = collect_runtime_snapshot(GEMMA_CONTAINER_NAMES)
    allow_failure = bool(SPEED_PROFILES.get(profile, {}).get("allow_failure", False))
    if status == "failed" and allow_failure:
        status = "shadow_failed"
    summary = {
        "profile": profile,
        "status": status,
        "failure_reason": failure_reason,
        "preset_path": repo_relative_str(preset_path),
        "gemma_runtime_overrides": overrides,
        "compose_elapsed_sec": round(compose_elapsed_sec, 3),
        "health_elapsed_sec": round(health_elapsed_sec, 3),
        "request_count": len(requests),
        "requests": requests,
        "request_timing_summary": summarize_request_timings(requests),
        "warmup_requests": max(0, int(warmup_requests)),
        "concurrency": max(1, int(concurrency)),
        "gpu_before": before_snapshot.get("gpu", {}),
        "gpu_after": after_snapshot.get("gpu", {}),
        "prompt_hash": sha256_text(SYNTHETIC_SYSTEM_PROMPT),
        "user_payload_hash": sha256_text(json.dumps(SYNTHETIC_USER_PAYLOAD, ensure_ascii=False, sort_keys=True)),
    }
    write_json(profile_dir / "summary.json", summary)
    return summary


def render_report(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Gemma Runtime Speed Matrix",
        "",
        "| profile | status | ctx | gpu_layers | threads | health_sec | cold_req_sec | warm_p50_sec | best_tps | gpu_free_after_mb |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        requests = item.get("requests", []) if isinstance(item.get("requests"), list) else []
        timings = summarize_request_timings(requests)
        cold_req = timings.get("first_request_latency_sec") or ""
        warm_p50 = timings.get("warm_p50_sec") or ""
        best_tps = timings.get("best_tps") or ""
        overrides = item.get("gemma_runtime_overrides", {}) if isinstance(item.get("gemma_runtime_overrides"), dict) else {}
        primary = (item.get("gpu_after", {}) or {}).get("primary", {}) if isinstance(item.get("gpu_after"), dict) else {}
        lines.append(
            "| {profile} | {status} | {ctx} | {gpu_layers} | {threads} | {health} | {cold_req} | {warm_p50} | {best_tps} | {free} |".format(
                profile=item.get("profile", ""),
                status=item.get("status", ""),
                ctx=overrides.get("context_size", ""),
                gpu_layers=overrides.get("n_gpu_layers", ""),
                threads=overrides.get("threads", ""),
                health=item.get("health_elapsed_sec", ""),
                cold_req=cold_req,
                warm_p50=warm_p50,
                best_tps=best_tps,
                free=primary.get("memory_free_mb", ""),
            )
        )
    lines.append("")
    ranked = rank_successful_profiles(results)
    if ranked:
        lines.append(f"Fastest successful profile: `{ranked[0]['profile']}`")
        lines.append("")
    lines.append("Raw source OCR text, translations, local input paths, images, and archives are intentionally omitted.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Gemma llama.cpp runtime profiles without OCR/inpaint stages.")
    parser.add_argument("--profiles", nargs="+", choices=tuple(SPEED_PROFILES.keys()), default=DEFAULT_PROFILES)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--timeout-sec", type=int, default=240, help="Health wait timeout per profile.")
    parser.add_argument("--request-timeout-sec", type=int, default=60, help="Per-request timeout after health is ready.")
    parser.add_argument("--request-count", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=0, help="0 means use each profile's n_parallel value.")
    parser.add_argument("--warmup-requests", type=int, default=0)
    args = parser.parse_args()

    suite_dir = Path(args.output_dir).resolve() if args.output_dir else create_run_dir(
        "gemma_runtime_speed_matrix",
        root=Path(args.output_root).resolve() if args.output_root else None,
    )
    suite_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for profile in args.profiles:
        summary = run_profile(
            profile=profile,
            suite_dir=suite_dir,
            health_timeout_sec=args.timeout_sec,
            request_timeout_sec=args.request_timeout_sec,
            request_count=args.request_count,
            max_tokens=args.max_tokens,
            concurrency=profile_concurrency(profile, args.concurrency),
            warmup_requests=args.warmup_requests,
        )
        results.append(summary)
        write_json(
            suite_dir / "matrix_summary.json",
            {
                "profiles": list(args.profiles),
                "ranking": rank_successful_profiles(results),
                "results": results,
            },
        )
        (suite_dir / "matrix_report.md").write_text(render_report(results), encoding="utf-8")
        if summary.get("status") == "failed":
            break
    remove_containers(GEMMA_CONTAINER_NAMES)
    return 1 if any(item.get("status") == "failed" for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
