#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_common import (
    DEFAULT_CONTAINER_NAMES,
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SAMPLE_DIR,
    create_run_dir,
    load_metrics,
    load_preset,
    remove_containers,
    repo_relative_str,
    resolve_corpus,
    run_command,
    stage_runtime_files,
    write_json,
)
from paddleocr_vl15_compare_gold import build_stability_profile

FAMILY_NAME = "paddleocr_vl15"
FAMILY_OUTPUT_ROOT_NAME = "paddleocr_vl15"
FAMILY_PRESET_BASE = "paddleocr-vl15-baseline"
LAST_SUITE_RECORD = "last_paddleocr_vl15_suite.json"
REPORT_MANIFEST_NAME = "paddleocr_vl15_report_manifest.yaml"
FULL_RUNTIME_HEALTH_URLS = [
    "http://127.0.0.1:18080/health",
    "http://127.0.0.1:18080/v1/models",
    "http://127.0.0.1:28118/docs",
]
OCR_ONLY_HEALTH_URLS = [
    "http://127.0.0.1:28118/docs",
]
DEFAULT_EXECUTION_SCOPE = "detect-ocr"
LEGACY_EXECUTION_SCOPE = "full-pipeline"
DEFAULT_SCREEN_SUBSET_SIZE = 10
DEFAULT_CONFIRM_IMPROVEMENT_RATIO = 0.05
PHASE_SEQUENCE = [
    {
        "name": "phase-1-workers-and-hpip",
        "candidates": [
            {
                "suffix": "workers1",
                "description": "parallel_workers=1",
                "updates": {"ocr_client": {"parallel_workers": 1}},
            },
            {
                "suffix": "hpip-workers2",
                "description": "layout --use_hpip + parallel_workers=2",
                "updates": {
                    "ocr_client": {"parallel_workers": 2},
                    "ocr_runtime": {"use_hpip": True},
                },
            },
            {
                "suffix": "hpip-workers1",
                "description": "layout --use_hpip + parallel_workers=1",
                "updates": {
                    "ocr_client": {"parallel_workers": 1},
                    "ocr_runtime": {"use_hpip": True},
                },
            },
        ],
    },
    {
        "name": "phase-2-max-concurrency",
        "candidates": [
            {"suffix": "conc64", "description": "max_concurrency=64", "updates": {"ocr_runtime": {"max_concurrency": 64}}},
            {"suffix": "conc32", "description": "max_concurrency=32", "updates": {"ocr_runtime": {"max_concurrency": 32}}},
            {"suffix": "conc16", "description": "max_concurrency=16", "updates": {"ocr_runtime": {"max_concurrency": 16}}},
        ],
    },
    {
        "name": "phase-3a-gpu-memory-utilization",
        "candidates": [
            {"suffix": "vram080", "description": "gpu_memory_utilization=0.80", "updates": {"ocr_runtime": {"gpu_memory_utilization": 0.80}}},
            {"suffix": "vram076", "description": "gpu_memory_utilization=0.76", "updates": {"ocr_runtime": {"gpu_memory_utilization": 0.76}}},
            {"suffix": "vram072", "description": "gpu_memory_utilization=0.72", "updates": {"ocr_runtime": {"gpu_memory_utilization": 0.72}}},
        ],
    },
    {
        "name": "phase-3b-max-num-seqs",
        "candidates": [
            {"suffix": "seqs16", "description": "max_num_seqs=16", "updates": {"ocr_runtime": {"max_num_seqs": 16}}},
            {"suffix": "seqs12", "description": "max_num_seqs=12", "updates": {"ocr_runtime": {"max_num_seqs": 12}}},
            {"suffix": "seqs8", "description": "max_num_seqs=8", "updates": {"ocr_runtime": {"max_num_seqs": 8}}},
        ],
    },
    {
        "name": "phase-3c-max-num-batched-tokens",
        "candidates": [
            {"suffix": "tokens65536", "description": "max_num_batched_tokens=65536", "updates": {"ocr_runtime": {"max_num_batched_tokens": 65536}}},
            {"suffix": "tokens49152", "description": "max_num_batched_tokens=49152", "updates": {"ocr_runtime": {"max_num_batched_tokens": 49152}}},
            {"suffix": "tokens32768", "description": "max_num_batched_tokens=32768", "updates": {"ocr_runtime": {"max_num_batched_tokens": 32768}}},
        ],
    },
    {
        "name": "phase-4-layout-gpu",
        "min_gpu_floor_free_mb": 3072,
        "candidates": [
            {
                "suffix": "layout-gpu-hpip",
                "description": "layout --device gpu:0 + --use_hpip",
                "updates": {"ocr_runtime": {"front_device": "gpu:0", "use_hpip": True}},
            }
        ],
    },
]


def _log(message: str) -> None:
    print(f"[paddleocr-vl15] {message}", flush=True)


def family_output_root() -> Path:
    env_root = os.getenv("CT_BENCH_OUTPUT_ROOT", "").strip()
    if env_root:
        root = Path(env_root)
        if root.name != FAMILY_OUTPUT_ROOT_NAME:
            root = root / FAMILY_OUTPUT_ROOT_NAME
    else:
        root = ROOT / "banchmark_result_log" / FAMILY_OUTPUT_ROOT_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def _repo_root_results_root() -> str:
    try:
        return "./" + str(family_output_root().resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(family_output_root()).replace("\\", "/")


def _current_git_sha(ref: str = "HEAD") -> str:
    completed = run_command(["git", "rev-parse", ref], cwd=ROOT, check=False)
    return (completed.stdout or "").strip()


def _wait_for_url(url: str, timeout_sec: int = 180) -> None:
    started = time.time()
    while time.time() - started < timeout_sec:
        try:
            with urllib.request.urlopen(url, timeout=5):
                return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {url}")


def _enqueue_output(stream, queue: Queue[tuple[str, str]]) -> None:
    try:
        for line in iter(stream.readline, ""):
            queue.put(("line", line))
    finally:
        queue.put(("eof", ""))


def _run_command_streaming(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    output_dir: Path,
    step_name: str,
) -> subprocess.CompletedProcess[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "command_stdout.txt"
    stderr_path = output_dir / "command_stderr.txt"

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    if process.stdout is None:
        raise RuntimeError(f"{step_name} stdout pipe could not be created.")

    queue: Queue[tuple[str, str]] = Queue()
    reader = Thread(target=_enqueue_output, args=(process.stdout, queue), daemon=True)
    reader.start()

    combined: list[str] = []
    eof_seen = False
    while not eof_seen or process.poll() is None:
        try:
            kind, payload = queue.get(timeout=0.2)
        except Empty:
            continue
        if kind == "eof":
            eof_seen = True
            continue
        combined.append(payload)
        print(f"[paddleocr-vl15][{step_name}] {payload.rstrip()}", flush=True)

    return_code = process.wait()
    output_text = "".join(combined)
    stdout_path.write_text(output_text, encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    return subprocess.CompletedProcess(
        cmd,
        return_code,
        stdout=output_text,
        stderr="",
    )


def _deep_update(target: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _materialize_generated_preset(
    *,
    base_payload: dict[str, Any],
    output_path: Path,
    name: str,
    description: str,
    updates: dict[str, Any],
) -> Path:
    payload = json.loads(json.dumps(base_payload))
    payload["name"] = name
    payload["description"] = description
    _deep_update(payload, updates)
    write_json(output_path, payload)
    return output_path


def _prepare_runtime(preset_ref: str, runtime_dir: Path, *, runtime_services: str) -> dict[str, Any]:
    preset, _ = load_preset(preset_ref)
    staged = stage_runtime_files(preset, runtime_dir)
    remove_containers(DEFAULT_CONTAINER_NAMES)
    if runtime_services != "ocr-only":
        run_command(
            ["docker", "compose", "-f", staged["gemma"]["compose_path"], "up", "-d", "--force-recreate"],
            cwd=runtime_dir / "gemma",
        )
    run_command(
        ["docker", "compose", "-f", staged["ocr"]["compose_path"], "up", "-d", "--force-recreate"],
        cwd=runtime_dir / "ocr",
    )
    for url in OCR_ONLY_HEALTH_URLS if runtime_services == "ocr-only" else FULL_RUNTIME_HEALTH_URLS:
        _wait_for_url(url)
    return {
        "preset": preset,
        "runtime_dir": runtime_dir,
        "staged": staged,
        "runtime_services": runtime_services,
    }


def _run_pipeline(
    *,
    preset_ref: str,
    mode: str,
    runtime_mode: str,
    runtime_services: str,
    execution_scope: str,
    sample_dir: Path,
    sample_count: int,
    output_dir: Path,
    label: str,
) -> dict[str, Any]:
    stage_ceiling = "ocr" if execution_scope == DEFAULT_EXECUTION_SCOPE else "render"
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "benchmark_pipeline.py"),
        "--preset",
        str(preset_ref),
        "--mode",
        mode,
        "--repeat",
        "1",
        "--runtime-mode",
        runtime_mode,
        "--runtime-services",
        runtime_services,
        "--stage-ceiling",
        stage_ceiling,
        "--sample-dir",
        str(sample_dir),
        "--sample-count",
        str(sample_count),
        "--output-dir",
        str(output_dir),
        "--label",
        label,
        "--export-page-snapshots",
        "--clear-app-caches",
    ]
    env = os.environ.copy()
    env["CT_BENCH_OUTPUT_ROOT"] = str(family_output_root())
    completed = _run_command_streaming(
        cmd=cmd,
        cwd=ROOT,
        env=env,
        output_dir=output_dir,
        step_name=label,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed (code={completed.returncode})\n{(completed.stdout or '').strip()}"
        )
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "run_dir": output_dir,
        "summary": summary,
        "preset_input": preset_ref,
        "execution_scope": execution_scope,
        "runtime_services": runtime_services,
    }


def _generate_gold_from_run(run_dir: Path, output_path: Path, *, baseline_sha: str, baseline_ref_sha: str) -> Path:
    snapshots_path = run_dir / "page_snapshots.json"
    payload = json.loads(snapshots_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {snapshots_path}")
    payload["baseline_sha"] = baseline_sha
    payload["develop_ref_sha"] = baseline_ref_sha
    payload["generated_from_run_dir"] = repo_relative_str(run_dir)
    payload["generated_at"] = time.time()
    write_json(output_path, payload)
    return output_path


def _run_gold_compare(gold_path: Path, candidate_run_dir: Path, *, expected_pages: list[str] | None = None) -> dict[str, Any]:
    output_path = candidate_run_dir / "detect_ocr_compare.json"
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "paddleocr_vl15_compare_gold.py"),
        "--baseline-gold",
        str(gold_path),
        "--candidate-run-dir",
        str(candidate_run_dir),
        "--output",
        str(output_path),
    ]
    if expected_pages:
        cmd.extend(["--expected-pages", *expected_pages])
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if not output_path.is_file():
        raise RuntimeError(f"Missing compare output for {candidate_run_dir}")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    payload["exit_code"] = completed.returncode
    write_json(output_path, payload)
    return payload


def _median(values: list[float]) -> float:
    if not values:
        return float("inf")
    return float(statistics.median(values))


def _p95(values: list[float]) -> float:
    if not values:
        return float("inf")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95))))
    return float(ordered[index])


def _resolve_sample_paths(sample_dir: Path, sample_count: int) -> list[Path]:
    corpus = resolve_corpus(sample_dir, sample_count=sample_count)
    return list(corpus["representative"])


def _materialize_subset_corpus(
    *,
    suite_dir: Path,
    source_paths: list[Path],
    subset_page_names: list[str],
    label: str,
) -> Path:
    by_stem = {path.stem: path for path in source_paths}
    subset_dir = suite_dir / "_corpus" / label
    if subset_dir.exists():
        shutil.rmtree(subset_dir)
    subset_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for page_name in subset_page_names:
        source_path = by_stem.get(page_name)
        if source_path is None:
            continue
        shutil.copy2(source_path, subset_dir / source_path.name)
        copied += 1
    if copied < 1:
        raise RuntimeError(f"No subset pages materialized for {label}")
    return subset_dir


def _load_per_page_stage_metrics(run_dir: Path) -> dict[str, dict[str, float]]:
    stage_open: dict[tuple[str, str], list[float]] = {}
    per_page: dict[str, dict[str, float]] = {}
    for row in load_metrics(run_dir / "metrics.jsonl"):
        tag = str(row.get("tag", "") or "")
        image_path = str(row.get("image_path", "") or "")
        if not image_path:
            continue
        image_stem = Path(image_path).stem
        ts = float(row.get("ts", 0.0) or 0.0)
        if tag.endswith("_start"):
            stage_open.setdefault((tag[:-6], image_stem), []).append(ts)
        elif tag.endswith("_end"):
            stage_name = tag[:-4]
            key = (stage_name, image_stem)
            starts = stage_open.get(key, [])
            if not starts:
                continue
            started = starts.pop(0)
            per_page.setdefault(image_stem, {})
            per_page[image_stem][f"{stage_name}_sec"] = round(
                float(per_page[image_stem].get(f"{stage_name}_sec", 0.0)) + max(0.0, ts - started),
                6,
            )
    for page_name, payload in per_page.items():
        payload["detect_ocr_total_sec"] = round(
            float(payload.get("detect_sec", 0.0)) + float(payload.get("ocr_sec", 0.0)),
            6,
        )
        payload["image_stem"] = page_name
    return per_page


def _official_metrics_for_run(run_dir: Path, page_names: list[str]) -> dict[str, Any]:
    per_page = _load_per_page_stage_metrics(run_dir)
    detected_pages = [page for page in page_names if page in per_page]
    detect_ocr_total_sec = sum(float(per_page[page].get("detect_ocr_total_sec", 0.0)) for page in detected_pages)
    ocr_values = [float(per_page[page].get("ocr_sec", 0.0)) for page in detected_pages]
    return {
        "pages_used": detected_pages,
        "detect_ocr_total_sec": round(detect_ocr_total_sec, 3),
        "ocr_page_p95_sec": round(_p95(ocr_values), 3) if ocr_values else None,
    }


def _run_candidate(
    *,
    suite_dir: Path,
    name: str,
    preset_ref: str,
    sample_dir: Path,
    sample_count: int,
    execution_scope: str,
    gold_path: Path | None,
    expected_pages: list[str] | None,
    warm_count: int,
) -> dict[str, Any]:
    runtime_dir = suite_dir / "_runtime" / name
    runtime_services = "ocr-only" if execution_scope == DEFAULT_EXECUTION_SCOPE else "full"
    _log(f"runtime 준비: {name} preset={preset_ref} execution_scope={execution_scope}")
    runtime = _prepare_runtime(preset_ref, runtime_dir, runtime_services=runtime_services)
    preset_name = str(runtime["preset"].get("name", preset_ref))
    candidate_root = suite_dir / name
    candidate_root.mkdir(parents=True, exist_ok=True)

    cold_run = _run_pipeline(
        preset_ref=preset_ref,
        mode="batch",
        runtime_mode="attach-running",
        runtime_services=runtime_services,
        execution_scope=execution_scope,
        sample_dir=sample_dir,
        sample_count=sample_count,
        output_dir=candidate_root / "cold",
        label=f"{name}_cold",
    )
    if gold_path is not None:
        cold_run["compare"] = _run_gold_compare(gold_path, Path(cold_run["run_dir"]), expected_pages=expected_pages)
    cold_run["official_metrics"] = _official_metrics_for_run(Path(cold_run["run_dir"]), expected_pages or [])

    warm_runs: list[dict[str, Any]] = []
    for warm_index in range(1, warm_count + 1):
        warm_run = _run_pipeline(
            preset_ref=preset_ref,
            mode="batch",
            runtime_mode="attach-running",
            runtime_services=runtime_services,
            execution_scope=execution_scope,
            sample_dir=sample_dir,
            sample_count=sample_count,
            output_dir=candidate_root / f"warm{warm_index}",
            label=f"{name}_warm{warm_index}",
        )
        if gold_path is not None:
            warm_run["compare"] = _run_gold_compare(gold_path, Path(warm_run["run_dir"]), expected_pages=expected_pages)
        warm_run["official_metrics"] = _official_metrics_for_run(Path(warm_run["run_dir"]), expected_pages or [])
        warm_runs.append(warm_run)

    remove_containers(DEFAULT_CONTAINER_NAMES)
    return {
        "name": name,
        "preset": preset_name,
        "preset_input": preset_ref,
        "cold_run": cold_run,
        "warm_runs": warm_runs,
        "execution_scope": execution_scope,
        "runtime_services": runtime_services,
        "expected_pages": expected_pages or [],
    }


def _evaluate_candidate_result(
    *,
    name: str,
    preset_name: str,
    cold: dict[str, Any],
    warms: list[dict[str, Any]],
    current_best_official_score: float,
    compare_required: bool,
    require_improvement: bool,
) -> dict[str, Any]:
    warm_summaries = [item["summary"] for item in warms]
    warm_compare = [item.get("compare", {}) for item in warms]
    warm_scores = [float((item.get("official_metrics") or {}).get("detect_ocr_total_sec") or 1e12) for item in warms]
    warm_p95 = [float((item.get("official_metrics") or {}).get("ocr_page_p95_sec") or 1e12) for item in warms]
    rejection_reasons: list[str] = []

    detection_pass = all(bool(item.get("detection_pass", False)) for item in warm_compare) and bool(
        (cold.get("compare") or {}).get("detection_pass", True)
    )
    ocr_pass = all(bool(item.get("ocr_pass", False)) for item in warm_compare) and bool(
        (cold.get("compare") or {}).get("ocr_pass", True)
    )
    compare_pass = True
    if compare_required:
        compare_pass = bool((cold.get("compare") or {}).get("passed", False)) and all(
            bool(payload.get("passed", False)) for payload in warm_compare
        )
    if not detection_pass:
        rejection_reasons.append("detection gate failed")
    if not ocr_pass:
        rejection_reasons.append("ocr gate failed")
    if compare_required and not compare_pass:
        rejection_reasons.append("gold compare failed")

    for summary in [cold["summary"], *warm_summaries]:
        if int(summary.get("page_failed_count") or 0) > 0:
            rejection_reasons.append("page_failed_count > 0")
            break
    for summary in [cold["summary"], *warm_summaries]:
        if int(summary.get("ocr_cache_hit_count") or 0) > 0:
            rejection_reasons.append("ocr_cache_hit_count > 0")
            break

    official_score = _median(warm_scores)
    cold_score = float((cold.get("official_metrics") or {}).get("detect_ocr_total_sec") or 1e12)
    warm_p95_median = _median(warm_p95)
    improvement_ratio = (
        ((current_best_official_score - official_score) / current_best_official_score)
        if current_best_official_score and current_best_official_score < 1e12
        else 0.0
    )
    if require_improvement and improvement_ratio < DEFAULT_CONFIRM_IMPROVEMENT_RATIO:
        rejection_reasons.append("warm detect+ocr median improvement < 5%")

    promoted = not rejection_reasons
    return {
        "name": name,
        "preset": preset_name,
        "cold_run": cold,
        "warm_runs": warms,
        "compare_pass": compare_pass,
        "detection_pass": detection_pass,
        "ocr_pass": ocr_pass,
        "official_score_detect_ocr_median_sec": round(official_score, 3),
        "cold_detect_ocr_total_sec": round(cold_score, 3),
        "warm_ocr_page_p95_median_sec": round(warm_p95_median, 3),
        "improvement_vs_current_best_pct": round(improvement_ratio * 100.0, 2),
        "promoted": promoted,
        "rejection_reason": "; ".join(dict.fromkeys(rejection_reasons)),
    }


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(candidate.get("official_score_detect_ocr_median_sec") or 1e12),
        float(candidate.get("warm_ocr_page_p95_median_sec") or 1e12),
        float(candidate.get("cold_detect_ocr_total_sec") or 1e12),
    )


def _render_candidate_console(candidate: dict[str, Any], stage: str) -> str:
    return (
        f"{stage} {candidate['preset']} official={candidate['official_score_detect_ocr_median_sec']}s "
        f"p95={candidate['warm_ocr_page_p95_median_sec']}s promoted={candidate['promoted']} "
        f"reason={candidate['rejection_reason'] or '-'}"
    )


def _phase_candidates(
    *,
    suite_dir: Path,
    base_preset_payload: dict[str, Any],
    phase: dict[str, Any],
) -> list[tuple[str, str]]:
    generated_dir = suite_dir / "_generated_presets"
    generated_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[tuple[str, str]] = []
    base_name = str(base_preset_payload.get("name", FAMILY_PRESET_BASE))
    for candidate in phase.get("candidates", []):
        suffix = str(candidate["suffix"])
        preset_name = f"{base_name}-{suffix}"
        preset_path = generated_dir / f"{preset_name}.json"
        preset_ref = str(
            _materialize_generated_preset(
                base_payload=base_preset_payload,
                output_path=preset_path,
                name=preset_name,
                description=str(candidate["description"]),
                updates=dict(candidate.get("updates", {})),
            )
        )
        candidates.append((preset_name, preset_ref))
    return candidates


def _latest_gpu_floor_free_mb(candidate: dict[str, Any]) -> float:
    summaries = [candidate["cold_run"]["summary"], *[item["summary"] for item in candidate["warm_runs"]]]
    values = [float(summary.get("gpu_floor_free_mb") or 0.0) for summary in summaries]
    return min(values) if values else 0.0


def _encode_run(run: dict[str, Any]) -> dict[str, Any]:
    compare_path = Path(run["run_dir"]) / "detect_ocr_compare.json"
    compare = run.get("compare", {}) if isinstance(run.get("compare"), dict) else {}
    return {
        "run_dir": repo_relative_str(Path(run["run_dir"])),
        "summary": run["summary"],
        "compare_path": repo_relative_str(compare_path) if compare_path.is_file() else "",
        "compare": compare,
        "detection_pass": bool(compare.get("detection_pass", False)),
        "ocr_pass": bool(compare.get("ocr_pass", False)),
        "official_metrics": run.get("official_metrics", {}),
    }


def _suite_manifest(
    *,
    suite_dir: Path,
    baseline_sha: str,
    develop_ref_sha: str,
    gold_path: Path,
    profile: dict[str, Any],
    baseline_candidate: dict[str, Any],
    phase_results: list[dict[str, Any]],
    winner: dict[str, Any],
) -> Path:
    manifest = {
        "results_root": _repo_root_results_root(),
        "benchmark": {
            "name": "PaddleOCR-VL-1.5 Runtime Benchmark",
            "kind": "managed family suite",
            "scope": "official default suite uses detect+ocr-only offscreen execution with warm-stable quality gate",
            "execution_scope": "detect-ocr-only",
            "official_score_scope": "detect+ocr-only",
            "legacy_full_pipeline_available": True,
            "runtime_services": "ocr-only",
            "baseline_sha": baseline_sha,
            "develop_ref_sha": develop_ref_sha,
            "family": FAMILY_NAME,
        },
        "gold": {
            "path": repo_relative_str(gold_path),
            "profile_kind": str(profile.get("profile_kind", "") or ""),
            "stable_page_count": int(profile.get("stable_page_count", 0) or 0),
            "screen_subset": list(profile.get("screen_subset", [])),
            "excluded_unstable_pages": list(profile.get("excluded_unstable_pages", [])),
        },
        "baseline": {
            "preset": baseline_candidate["preset"],
            "scope": "full-confirm",
            "cold_run": _encode_run(baseline_candidate["cold_run"]),
            "warm_runs": [_encode_run(item) for item in baseline_candidate["warm_runs"]],
            "official_score_detect_ocr_median_sec": baseline_candidate["official_score_detect_ocr_median_sec"],
            "detection_pass": baseline_candidate["detection_pass"],
            "ocr_pass": baseline_candidate["ocr_pass"],
            "compare_pass": baseline_candidate["compare_pass"],
            "promoted": True,
            "rejection_reason": "",
        },
        "phases": phase_results,
        "winner": {
            "preset": winner["preset"],
            "official_score_detect_ocr_median_sec": winner["official_score_detect_ocr_median_sec"],
            "develop_promotion_ready": (
                winner["preset"] != baseline_candidate["preset"] and bool(winner.get("promoted", False))
            ),
            "improvement_vs_baseline_pct": round(
                (
                    (
                        float(baseline_candidate["official_score_detect_ocr_median_sec"])
                        - float(winner["official_score_detect_ocr_median_sec"])
                    )
                    / float(baseline_candidate["official_score_detect_ocr_median_sec"])
                )
                * 100.0,
                2,
            )
            if float(baseline_candidate["official_score_detect_ocr_median_sec"]) > 0
            else 0.0,
        },
        "report": {
            "markdown_output": "docs/banchmark_report/paddleocr-vl15-report-ko.md",
            "assets_dir": "docs/assets/benchmarking/paddleocr-vl15/latest",
        },
    }
    manifest_path = suite_dir / REPORT_MANIFEST_NAME
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return manifest_path


def _latest_manifest_path() -> Path | None:
    record_path = family_output_root() / LAST_SUITE_RECORD
    if not record_path.is_file():
        return None
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    manifest_rel = str(payload.get("manifest_path", "") or "")
    if not manifest_rel:
        return None
    if manifest_rel.startswith("./"):
        return ROOT / manifest_rel[2:]
    return Path(manifest_rel)


def _latest_gold_path() -> Path | None:
    gold_dir = family_output_root() / "gold"
    candidates = sorted(path for path in gold_dir.glob("*.json") if path.is_file())
    return candidates[-1] if candidates else None


def _baseline_variance_ok(candidate: dict[str, Any]) -> tuple[bool, float]:
    scores = [
        float((item.get("official_metrics") or {}).get("detect_ocr_total_sec") or 0.0)
        for item in candidate.get("warm_runs", [])
    ]
    scores = [item for item in scores if item > 0]
    if len(scores) < 2:
        return True, 0.0
    median_score = _median(scores)
    if median_score <= 0:
        return True, 0.0
    spread = (max(scores) - min(scores)) / median_score
    return spread <= 0.05, round(spread * 100.0, 2)


def _select_confirm_candidates(screen_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    passed = [candidate for candidate in screen_candidates if candidate.get("promoted", False)]
    passed.sort(key=_candidate_sort_key)
    if not passed:
        return []
    selected = [passed[0]]
    if len(passed) > 1:
        best = float(passed[0].get("official_score_detect_ocr_median_sec") or 1e12)
        second = float(passed[1].get("official_score_detect_ocr_median_sec") or 1e12)
        if best < 1e12 and abs(second - best) / best <= 0.01:
            selected.append(passed[1])
    return selected


def run_suite(*, sample_dir: Path = DEFAULT_SAMPLE_DIR, sample_count: int = DEFAULT_SAMPLE_COUNT) -> int:
    suite_dir = create_run_dir("paddleocr-vl15-runtime_suite", root=family_output_root())
    _log(f"suite output dir: {suite_dir}")
    gold_dir = suite_dir / "_gold"
    gold_dir.mkdir(parents=True, exist_ok=True)

    baseline_sha = _current_git_sha("HEAD")
    develop_ref_sha = _current_git_sha("develop") or baseline_sha
    source_paths = _resolve_sample_paths(sample_dir, sample_count)

    baseline_raw = _run_candidate(
        suite_dir=suite_dir,
        name="baseline-confirm",
        preset_ref=FAMILY_PRESET_BASE,
        sample_dir=sample_dir,
        sample_count=sample_count,
        execution_scope=DEFAULT_EXECUTION_SCOPE,
        gold_path=None,
        expected_pages=None,
        warm_count=3,
    )
    baseline_profile = build_stability_profile(
        [
            Path(baseline_raw["cold_run"]["run_dir"]),
            *[Path(item["run_dir"]) for item in baseline_raw["warm_runs"]],
        ],
        baseline_sha=baseline_sha,
        develop_ref_sha=develop_ref_sha,
        execution_scope="detect-ocr-only",
        official_score_scope="detect+ocr-only",
    )
    gold_path = gold_dir / "baseline_warm_stable_profile.json"
    write_json(gold_path, baseline_profile)

    stable_pages = list(baseline_profile.get("stable_pages", []))
    screen_subset = list(baseline_profile.get("screen_subset", []))
    if not stable_pages:
        raise RuntimeError("No stable baseline pages remained after warm-stable profiling.")

    baseline_raw["cold_run"]["compare"] = _run_gold_compare(
        gold_path,
        Path(baseline_raw["cold_run"]["run_dir"]),
        expected_pages=stable_pages,
    )
    for warm_run in baseline_raw["warm_runs"]:
        warm_run["compare"] = _run_gold_compare(
            gold_path,
            Path(warm_run["run_dir"]),
            expected_pages=stable_pages,
        )
        warm_run["official_metrics"] = _official_metrics_for_run(Path(warm_run["run_dir"]), stable_pages)
    baseline_raw["cold_run"]["official_metrics"] = _official_metrics_for_run(
        Path(baseline_raw["cold_run"]["run_dir"]),
        stable_pages,
    )
    baseline_candidate = _evaluate_candidate_result(
        name="baseline-confirm",
        preset_name=baseline_raw["preset"],
        cold=baseline_raw["cold_run"],
        warms=baseline_raw["warm_runs"],
        current_best_official_score=float("inf"),
        compare_required=True,
        require_improvement=False,
    )
    baseline_candidate["promoted"] = True
    baseline_candidate["rejection_reason"] = ""
    baseline_candidate["execution_scope"] = DEFAULT_EXECUTION_SCOPE
    baseline_candidate["runtime_services"] = "ocr-only"
    current_best = baseline_candidate
    current_best_payload, _ = load_preset(FAMILY_PRESET_BASE)

    baseline_variance_ok, baseline_spread_pct = _baseline_variance_ok(baseline_raw)
    if not baseline_variance_ok:
        raise RuntimeError(
            f"Baseline warm detect+ocr variance too high ({baseline_spread_pct}%). Stabilize runtime before comparing candidates."
        )

    screen_dir = _materialize_subset_corpus(
        suite_dir=suite_dir,
        source_paths=source_paths,
        subset_page_names=screen_subset,
        label="screen-subset",
    )

    phase_results: list[dict[str, Any]] = []
    for phase in PHASE_SEQUENCE:
        current_best_before = current_best["preset"]
        if phase.get("min_gpu_floor_free_mb") and _latest_gpu_floor_free_mb(current_best) < float(
            phase["min_gpu_floor_free_mb"]
        ):
            phase_results.append(
                {
                    "phase": phase["name"],
                    "current_best_before": current_best_before,
                    "best_preset_after": current_best["preset"],
                    "skipped": True,
                    "skip_reason": f"gpu_floor_free_mb < {phase['min_gpu_floor_free_mb']}",
                    "screen_candidates": [],
                    "confirm_candidates": [],
                }
            )
            continue

        generated = _phase_candidates(
            suite_dir=suite_dir,
            base_preset_payload=current_best_payload,
            phase=phase,
        )
        screen_results: list[dict[str, Any]] = []
        for candidate_name, preset_ref in generated:
            raw_candidate = _run_candidate(
                suite_dir=suite_dir,
                name=f"{candidate_name}-screen",
                preset_ref=preset_ref,
                sample_dir=screen_dir,
                sample_count=len(screen_subset),
                execution_scope=DEFAULT_EXECUTION_SCOPE,
                gold_path=gold_path,
                expected_pages=screen_subset,
                warm_count=1,
            )
            candidate_row = _evaluate_candidate_result(
                name=candidate_name,
                preset_name=raw_candidate["preset"],
                cold=raw_candidate["cold_run"],
                warms=raw_candidate["warm_runs"],
                current_best_official_score=float(current_best["official_score_detect_ocr_median_sec"]),
                compare_required=True,
                require_improvement=False,
            )
            candidate_row["stage"] = "screen"
            candidate_row["preset_input"] = preset_ref
            candidate_row["cold_run"] = raw_candidate["cold_run"]
            candidate_row["warm_runs"] = raw_candidate["warm_runs"]
            screen_results.append(candidate_row)
            _log(_render_candidate_console(candidate_row, "screen"))

        confirm_targets = _select_confirm_candidates(screen_results)
        confirm_results: list[dict[str, Any]] = []
        for candidate_row in confirm_targets:
            raw_candidate = _run_candidate(
                suite_dir=suite_dir,
                name=f"{candidate_row['name']}-confirm",
                preset_ref=str(candidate_row["preset_input"]),
                sample_dir=sample_dir,
                sample_count=sample_count,
                execution_scope=DEFAULT_EXECUTION_SCOPE,
                gold_path=gold_path,
                expected_pages=stable_pages,
                warm_count=3,
            )
            confirm_row = _evaluate_candidate_result(
                name=str(candidate_row["name"]),
                preset_name=raw_candidate["preset"],
                cold=raw_candidate["cold_run"],
                warms=raw_candidate["warm_runs"],
                current_best_official_score=float(current_best["official_score_detect_ocr_median_sec"]),
                compare_required=True,
                require_improvement=True,
            )
            confirm_row["stage"] = "confirm"
            confirm_row["preset_input"] = str(candidate_row["preset_input"])
            confirm_row["cold_run"] = raw_candidate["cold_run"]
            confirm_row["warm_runs"] = raw_candidate["warm_runs"]
            confirm_results.append(confirm_row)
            _log(_render_candidate_console(confirm_row, "confirm"))

        promoted = [candidate for candidate in confirm_results if candidate.get("promoted", False)]
        promoted.sort(key=_candidate_sort_key)
        if promoted:
            phase_best = promoted[0]
            current_best = phase_best
            current_best_payload, _ = load_preset(str(phase_best["preset_input"]))
            best_after = phase_best["preset"]
        else:
            best_after = current_best["preset"]

        phase_results.append(
            {
                "phase": phase["name"],
                "current_best_before": current_best_before,
                "best_preset_after": best_after,
                "skipped": False,
                "screen_candidates": screen_results,
                "confirm_candidates": confirm_results,
                "confirm_selected_presets": [candidate["preset"] for candidate in confirm_targets],
            }
        )

    phase_results.append(
        {
            "phase": "phase-5-code-candidate",
            "current_best_before": current_best["preset"],
            "best_preset_after": current_best["preset"],
            "skipped": True,
            "skip_reason": "Optional code candidate not implemented in this round.",
            "screen_candidates": [],
            "confirm_candidates": [],
        }
    )

    manifest_path = _suite_manifest(
        suite_dir=suite_dir,
        baseline_sha=baseline_sha,
        develop_ref_sha=develop_ref_sha,
        gold_path=gold_path,
        profile=baseline_profile,
        baseline_candidate=baseline_candidate,
        phase_results=phase_results,
        winner=current_best,
    )
    record_payload = {
        "suite_dir": repo_relative_str(suite_dir),
        "manifest_path": repo_relative_str(manifest_path),
        "winner_preset": current_best["preset"],
        "baseline_sha": baseline_sha,
    }
    write_json(family_output_root() / LAST_SUITE_RECORD, record_payload)

    subprocess.run(
        [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / "generate_paddleocr_vl15_report.py"),
            "--manifest",
            str(manifest_path),
        ],
        cwd=str(ROOT),
        check=True,
    )
    print(
        "winner={winner} official_detect_ocr_median={score}s manifest={manifest}".format(
            winner=current_best["preset"],
            score=current_best["official_score_detect_ocr_median_sec"],
            manifest=repo_relative_str(manifest_path),
        )
    )
    return 0


def _latest_family_run() -> Path | None:
    candidates = sorted(
        path for path in family_output_root().glob("*") if path.is_dir() and (path / "summary.json").is_file()
    )
    return candidates[-1] if candidates else None


def _open_dir(path: Path) -> int:
    if os.name == "nt" and hasattr(os, "startfile"):
        os.startfile(str(path))
        return 0
    print(str(path))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PaddleOCR-VL 1.5 detect+ocr-first benchmark family.")
    subparsers = parser.add_subparsers(dest="command", required=False)

    run_parser = subparsers.add_parser("run", help="Run a single PaddleOCR-VL 1.5 benchmark candidate.")
    run_parser.add_argument("preset", nargs="?", default=FAMILY_PRESET_BASE)
    run_parser.add_argument("mode", nargs="?", default="batch", choices=("one-page", "batch", "webtoon"))
    run_parser.add_argument("runtime_mode", nargs="?", default="managed", choices=("attach-running", "managed"))
    run_parser.add_argument("repeat", nargs="?", default="1")
    run_parser.add_argument("sample_dir", nargs="?", default=str(DEFAULT_SAMPLE_DIR))
    run_parser.add_argument("sample_count", nargs="?", default=str(DEFAULT_SAMPLE_COUNT))
    run_parser.add_argument(
        "--execution-scope",
        choices=(DEFAULT_EXECUTION_SCOPE, LEGACY_EXECUTION_SCOPE),
        default=DEFAULT_EXECUTION_SCOPE,
        help="Official default is detect-ocr. full-pipeline remains legacy/manual.",
    )

    gold_parser = subparsers.add_parser("gold", help="Generate a baseline gold JSON from an existing run.")
    gold_parser.add_argument("--run-dir", default="")
    gold_parser.add_argument("--output", default="")

    compare_parser = subparsers.add_parser("compare", help="Compare an existing run against baseline gold or warm-stable profile.")
    compare_parser.add_argument("--baseline-gold", default="")
    compare_parser.add_argument("--candidate-run-dir", default="")

    summary_parser = subparsers.add_parser("summary", help="Generate the PaddleOCR-VL 1.5 report from a suite manifest.")
    summary_parser.add_argument("--manifest", default="")

    subparsers.add_parser("suite", help="Run the official PaddleOCR-VL 1.5 suite.")
    subparsers.add_parser("open", help="Open the family result root.")
    subparsers.add_parser("help", help="Show help.")

    args = parser.parse_args()
    if args.command in (None, "help"):
        parser.print_help()
        return 0

    if args.command == "run":
        preset_ref = args.preset
        repeat = int(args.repeat)
        sample_count = int(args.sample_count)
        sample_dir = Path(args.sample_dir)
        runtime_services = "ocr-only" if args.execution_scope == DEFAULT_EXECUTION_SCOPE else "full"
        output_root = family_output_root()
        for repeat_index in range(1, max(1, repeat) + 1):
            run_dir = create_run_dir(
                f"{Path(str(preset_ref)).stem}_{args.mode}_r{repeat_index}",
                root=output_root,
            )
            if args.runtime_mode == "managed":
                _prepare_runtime(preset_ref, output_root / "_run_runtime" / run_dir.name, runtime_services=runtime_services)
            result = _run_pipeline(
                preset_ref=preset_ref,
                mode=args.mode,
                runtime_mode="attach-running" if args.runtime_mode == "managed" else args.runtime_mode,
                runtime_services=runtime_services,
                execution_scope=args.execution_scope,
                sample_dir=sample_dir,
                sample_count=sample_count,
                output_dir=run_dir,
                label=run_dir.name,
            )
            remove_containers(DEFAULT_CONTAINER_NAMES)
            print(repo_relative_str(result["run_dir"]))
        return 0

    if args.command == "gold":
        run_dir = Path(args.run_dir) if args.run_dir else _latest_family_run()
        if run_dir is None or not run_dir.is_dir():
            raise SystemExit("No run directory available for gold generation.")
        output_path = Path(args.output) if args.output else family_output_root() / "gold" / f"{run_dir.name}_baseline_gold.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gold_path = _generate_gold_from_run(
            run_dir,
            output_path,
            baseline_sha=_current_git_sha("HEAD"),
            baseline_ref_sha=_current_git_sha("develop") or _current_git_sha("HEAD"),
        )
        print(repo_relative_str(gold_path))
        return 0

    if args.command == "compare":
        gold_path = Path(args.baseline_gold) if args.baseline_gold else _latest_gold_path()
        run_dir = Path(args.candidate_run_dir) if args.candidate_run_dir else _latest_family_run()
        if gold_path is None or not gold_path.is_file():
            raise SystemExit("No baseline gold JSON available for compare.")
        if run_dir is None or not run_dir.is_dir():
            raise SystemExit("No candidate run directory available for compare.")
        payload = _run_gold_compare(gold_path, run_dir)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload.get("passed", False) else 1

    if args.command == "summary":
        manifest_path = Path(args.manifest) if args.manifest else _latest_manifest_path()
        if manifest_path is None or not manifest_path.is_file():
            raise SystemExit("No PaddleOCR-VL 1.5 suite manifest found.")
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(ROOT / "scripts" / "generate_paddleocr_vl15_report.py"),
                "--manifest",
                str(manifest_path),
            ],
            cwd=str(ROOT),
            check=True,
        )
        return 0

    if args.command == "suite":
        return run_suite()

    if args.command == "open":
        return _open_dir(family_output_root())

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
