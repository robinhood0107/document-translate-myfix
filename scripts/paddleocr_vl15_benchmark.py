#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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
    load_preset,
    remove_containers,
    repo_relative_str,
    run_command,
    stage_runtime_files,
    write_json,
)

FAMILY_NAME = "paddleocr_vl15"
FAMILY_OUTPUT_ROOT_NAME = "paddleocr_vl15"
FAMILY_PRESET_BASE = "paddleocr-vl15-baseline"
LAST_SUITE_RECORD = "last_paddleocr_vl15_suite.json"
REPORT_MANIFEST_NAME = "paddleocr_vl15_report_manifest.yaml"
HEALTH_URLS = [
    "http://127.0.0.1:18080/health",
    "http://127.0.0.1:18000/v1/models",
    "http://127.0.0.1:28118/docs",
]
PHASE_SEQUENCE = [
    {
        "name": "phase-1-workers-and-hpip",
        "baseline_dependent": True,
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
        "baseline_dependent": True,
        "candidates": [
            {"suffix": "conc64", "description": "max_concurrency=64", "updates": {"ocr_runtime": {"max_concurrency": 64}}},
            {"suffix": "conc32", "description": "max_concurrency=32", "updates": {"ocr_runtime": {"max_concurrency": 32}}},
            {"suffix": "conc16", "description": "max_concurrency=16", "updates": {"ocr_runtime": {"max_concurrency": 16}}},
        ],
    },
    {
        "name": "phase-3a-gpu-memory-utilization",
        "baseline_dependent": True,
        "candidates": [
            {"suffix": "vram080", "description": "gpu_memory_utilization=0.80", "updates": {"ocr_runtime": {"gpu_memory_utilization": 0.80}}},
            {"suffix": "vram076", "description": "gpu_memory_utilization=0.76", "updates": {"ocr_runtime": {"gpu_memory_utilization": 0.76}}},
            {"suffix": "vram072", "description": "gpu_memory_utilization=0.72", "updates": {"ocr_runtime": {"gpu_memory_utilization": 0.72}}},
        ],
    },
    {
        "name": "phase-3b-max-num-seqs",
        "baseline_dependent": True,
        "candidates": [
            {"suffix": "seqs16", "description": "max_num_seqs=16", "updates": {"ocr_runtime": {"max_num_seqs": 16}}},
            {"suffix": "seqs12", "description": "max_num_seqs=12", "updates": {"ocr_runtime": {"max_num_seqs": 12}}},
            {"suffix": "seqs8", "description": "max_num_seqs=8", "updates": {"ocr_runtime": {"max_num_seqs": 8}}},
        ],
    },
    {
        "name": "phase-3c-max-num-batched-tokens",
        "baseline_dependent": True,
        "candidates": [
            {
                "suffix": "tokens65536",
                "description": "max_num_batched_tokens=65536",
                "updates": {"ocr_runtime": {"max_num_batched_tokens": 65536}},
            },
            {
                "suffix": "tokens49152",
                "description": "max_num_batched_tokens=49152",
                "updates": {"ocr_runtime": {"max_num_batched_tokens": 49152}},
            },
            {
                "suffix": "tokens32768",
                "description": "max_num_batched_tokens=32768",
                "updates": {"ocr_runtime": {"max_num_batched_tokens": 32768}},
            },
        ],
    },
    {
        "name": "phase-4-layout-gpu",
        "baseline_dependent": True,
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
        except (urllib.error.URLError, TimeoutError):
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


def _prepare_runtime(preset_ref: str, runtime_dir: Path) -> dict[str, Any]:
    preset, _ = load_preset(preset_ref)
    staged = stage_runtime_files(preset, runtime_dir)
    remove_containers(DEFAULT_CONTAINER_NAMES)
    run_command(
        ["docker", "compose", "-f", staged["gemma"]["compose_path"], "up", "-d", "--force-recreate"],
        cwd=runtime_dir / "gemma",
    )
    run_command(
        ["docker", "compose", "-f", staged["ocr"]["compose_path"], "up", "-d", "--force-recreate"],
        cwd=runtime_dir / "ocr",
    )
    for url in HEALTH_URLS:
        _wait_for_url(url)
    return {
        "preset": preset,
        "runtime_dir": runtime_dir,
        "staged": staged,
    }


def _run_pipeline(
    *,
    preset_ref: str,
    mode: str,
    runtime_mode: str,
    sample_dir: Path,
    sample_count: int,
    output_dir: Path,
    label: str,
) -> dict[str, Any]:
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


def _run_gold_compare(gold_path: Path, candidate_run_dir: Path) -> dict[str, Any]:
    output_path = candidate_run_dir / "detect_ocr_compare.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / "paddleocr_vl15_compare_gold.py"),
            "--baseline-gold",
            str(gold_path),
            "--candidate-run-dir",
            str(candidate_run_dir),
            "--output",
            str(output_path),
        ],
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


def _compare_pass_flags(compare_payload: dict[str, Any]) -> tuple[bool, bool]:
    detection_pass = True
    ocr_pass = True
    for page in compare_payload.get("pages", []):
        for issue in page.get("issues", []):
            issue_text = str(issue)
            if any(keyword in issue_text for keyword in ("block_count", "unmatched_", "page_failed mismatch", "candidate page missing")):
                detection_pass = False
            if any(keyword in issue_text for keyword in ("text_mismatch", "non_empty regression", "empty regression", "single_char_like regression")):
                ocr_pass = False
    return detection_pass, ocr_pass


def _evaluate_candidate_result(
    *,
    name: str,
    preset_name: str,
    cold: dict[str, Any],
    warms: list[dict[str, Any]],
    current_best_official_score: float,
) -> dict[str, Any]:
    warm_summaries = [item["summary"] for item in warms]
    warm_compare = [item["compare"] for item in warms]
    warm_scores = [float(summary.get("detect_ocr_total_sec") or 1e12) for summary in warm_summaries]
    warm_p95 = [float(summary.get("ocr_page_p95_sec") or 1e12) for summary in warm_summaries]
    rejection_reasons: list[str] = []

    detection_pass = all(item.get("detection_pass", False) for item in warm_compare) and bool(cold["compare"].get("passed", False))
    ocr_pass = all(item.get("ocr_pass", False) for item in warm_compare) and bool(cold["compare"].get("passed", False))
    compare_pass = bool(cold["compare"].get("passed", False)) and all(
        payload.get("passed", False) for payload in warm_compare
    )
    if not detection_pass:
        rejection_reasons.append("detection gate failed")
    if not ocr_pass:
        rejection_reasons.append("ocr gate failed")
    if not compare_pass:
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
    cold_score = float(cold["summary"].get("detect_ocr_total_sec") or 1e12)
    warm_p95_median = _median(warm_p95)
    improvement_ratio = (
        ((current_best_official_score - official_score) / current_best_official_score)
        if current_best_official_score and current_best_official_score < 1e12
        else 0.0
    )
    if improvement_ratio < 0.05:
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


def _render_candidate_console(candidate: dict[str, Any]) -> str:
    return (
        f"{candidate['preset']} official={candidate['official_score_detect_ocr_median_sec']}s "
        f"p95={candidate['warm_ocr_page_p95_median_sec']}s promoted={candidate['promoted']} "
        f"reason={candidate['rejection_reason'] or '-'}"
    )


def _run_candidate(
    *,
    suite_dir: Path,
    name: str,
    preset_ref: str,
    sample_dir: Path,
    sample_count: int,
    gold_path: Path | None,
) -> dict[str, Any]:
    runtime_dir = suite_dir / "_runtime" / name
    _log(f"runtime 준비: {name} preset={preset_ref}")
    runtime = _prepare_runtime(preset_ref, runtime_dir)
    preset_name = str(runtime["preset"].get("name", preset_ref))
    candidate_root = suite_dir / name
    candidate_root.mkdir(parents=True, exist_ok=True)

    cold_run = _run_pipeline(
        preset_ref=preset_ref,
        mode="batch",
        runtime_mode="attach-running",
        sample_dir=sample_dir,
        sample_count=sample_count,
        output_dir=candidate_root / "cold",
        label=f"{name}_cold",
    )
    if gold_path is not None:
        cold_compare = _run_gold_compare(gold_path, Path(cold_run["run_dir"]))
        cold_detection_pass, cold_ocr_pass = _compare_pass_flags(cold_compare)
        cold_run["compare"] = cold_compare
        cold_run["detection_pass"] = cold_detection_pass
        cold_run["ocr_pass"] = cold_ocr_pass

    warm_runs: list[dict[str, Any]] = []
    for warm_index in range(1, 4):
        warm_run = _run_pipeline(
            preset_ref=preset_ref,
            mode="batch",
            runtime_mode="attach-running",
            sample_dir=sample_dir,
            sample_count=sample_count,
            output_dir=candidate_root / f"warm{warm_index}",
            label=f"{name}_warm{warm_index}",
        )
        if gold_path is not None:
            compare_payload = _run_gold_compare(gold_path, Path(warm_run["run_dir"]))
            detection_pass, ocr_pass = _compare_pass_flags(compare_payload)
            warm_run["compare"] = compare_payload
            warm_run["detection_pass"] = detection_pass
            warm_run["ocr_pass"] = ocr_pass
        warm_runs.append(warm_run)

    remove_containers(DEFAULT_CONTAINER_NAMES)
    return {
        "name": name,
        "preset": preset_name,
        "preset_input": preset_ref,
        "cold_run": cold_run,
        "warm_runs": warm_runs,
    }


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


def _suite_manifest(
    *,
    suite_dir: Path,
    baseline_sha: str,
    develop_ref_sha: str,
    gold_path: Path,
    baseline_candidate: dict[str, Any],
    phase_results: list[dict[str, Any]],
    winner: dict[str, Any],
) -> Path:
    def encode_run(run: dict[str, Any]) -> dict[str, Any]:
        compare_path = Path(run["run_dir"]) / "detect_ocr_compare.json"
        return {
            "run_dir": repo_relative_str(Path(run["run_dir"])),
            "summary": run["summary"],
            "compare_path": repo_relative_str(compare_path) if compare_path.is_file() else "",
            "detection_pass": bool(run.get("detection_pass", False)),
            "ocr_pass": bool(run.get("ocr_pass", False)),
        }

    manifest = {
        "results_root": _repo_root_results_root(),
        "benchmark": {
            "name": "PaddleOCR-VL-1.5 Runtime Benchmark",
            "kind": "managed family suite",
            "scope": "actual offscreen app pipeline; official score and quality gate use detect+ocr only",
            "baseline_sha": baseline_sha,
            "develop_ref_sha": develop_ref_sha,
            "family": FAMILY_NAME,
        },
        "gold": {
            "path": repo_relative_str(gold_path),
        },
        "baseline": {
            "preset": baseline_candidate["preset"],
            "cold_run": encode_run(baseline_candidate["cold_run"]),
            "warm_runs": [encode_run(item) for item in baseline_candidate["warm_runs"]],
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


def run_suite(*, sample_dir: Path = DEFAULT_SAMPLE_DIR, sample_count: int = DEFAULT_SAMPLE_COUNT) -> int:
    suite_dir = create_run_dir("paddleocr-vl15-runtime_suite", root=family_output_root())
    _log(f"suite output dir: {suite_dir}")
    gold_dir = suite_dir / "_gold"
    gold_dir.mkdir(parents=True, exist_ok=True)

    baseline_sha = _current_git_sha("HEAD")
    develop_ref_sha = _current_git_sha("develop") or baseline_sha

    baseline_raw = _run_candidate(
        suite_dir=suite_dir,
        name="baseline",
        preset_ref=FAMILY_PRESET_BASE,
        sample_dir=sample_dir,
        sample_count=sample_count,
        gold_path=None,
    )
    gold_path = _generate_gold_from_run(
        Path(baseline_raw["cold_run"]["run_dir"]),
        gold_dir / "baseline_gold.json",
        baseline_sha=baseline_sha,
        baseline_ref_sha=develop_ref_sha,
    )
    baseline_raw["cold_run"]["compare"] = _run_gold_compare(gold_path, Path(baseline_raw["cold_run"]["run_dir"]))
    baseline_raw["cold_run"]["detection_pass"], baseline_raw["cold_run"]["ocr_pass"] = _compare_pass_flags(
        baseline_raw["cold_run"]["compare"]
    )
    refreshed_warms: list[dict[str, Any]] = []
    for warm_run in baseline_raw["warm_runs"]:
        compare_payload = _run_gold_compare(gold_path, Path(warm_run["run_dir"]))
        warm_run["compare"] = compare_payload
        warm_run["detection_pass"], warm_run["ocr_pass"] = _compare_pass_flags(compare_payload)
        refreshed_warms.append(warm_run)
    baseline_raw["warm_runs"] = refreshed_warms
    baseline_candidate = _evaluate_candidate_result(
        name="baseline",
        preset_name=baseline_raw["preset"],
        cold=baseline_raw["cold_run"],
        warms=baseline_raw["warm_runs"],
        current_best_official_score=float("inf"),
    )
    baseline_candidate["promoted"] = True
    baseline_candidate["rejection_reason"] = ""
    current_best = baseline_candidate
    current_best_payload, _ = load_preset(FAMILY_PRESET_BASE)

    phase_results: list[dict[str, Any]] = []
    for phase in PHASE_SEQUENCE:
        if phase.get("min_gpu_floor_free_mb") and _latest_gpu_floor_free_mb(current_best) < float(
            phase["min_gpu_floor_free_mb"]
        ):
            phase_results.append(
                {
                    "phase": phase["name"],
                    "current_best_before": current_best["preset"],
                    "best_preset_after": current_best["preset"],
                    "skipped": True,
                    "skip_reason": f"gpu_floor_free_mb < {phase['min_gpu_floor_free_mb']}",
                    "candidates": [],
                }
            )
            continue

        generated = _phase_candidates(
            suite_dir=suite_dir,
            base_preset_payload=current_best_payload,
            phase=phase,
        )
        candidate_rows: list[dict[str, Any]] = []
        promoted: list[dict[str, Any]] = []
        for candidate_name, preset_ref in generated:
            raw_candidate = _run_candidate(
                suite_dir=suite_dir,
                name=candidate_name,
                preset_ref=preset_ref,
                sample_dir=sample_dir,
                sample_count=sample_count,
                gold_path=gold_path,
            )
            candidate_row = _evaluate_candidate_result(
                name=candidate_name,
                preset_name=raw_candidate["preset"],
                cold=raw_candidate["cold_run"],
                warms=raw_candidate["warm_runs"],
                current_best_official_score=float(current_best["official_score_detect_ocr_median_sec"]),
            )
            candidate_row["preset_input"] = preset_ref
            candidate_row["cold_run"] = {
                "run_dir": repo_relative_str(raw_candidate["cold_run"]["run_dir"]),
                "summary": raw_candidate["cold_run"]["summary"],
                "compare_path": repo_relative_str(Path(raw_candidate["cold_run"]["run_dir"]) / "detect_ocr_compare.json"),
            }
            candidate_row["warm_runs"] = [
                {
                    "run_dir": repo_relative_str(item["run_dir"]),
                    "summary": item["summary"],
                    "compare_path": repo_relative_str(Path(item["run_dir"]) / "detect_ocr_compare.json"),
                }
                for item in raw_candidate["warm_runs"]
            ]
            candidate_rows.append(candidate_row)
            if candidate_row["promoted"]:
                promoted.append(candidate_row)
            _log(_render_candidate_console(candidate_row))

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
                "current_best_before": baseline_candidate["preset"] if not phase_results else phase_results[-1]["best_preset_after"],
                "best_preset_after": best_after,
                "skipped": False,
                "candidates": candidate_rows,
            }
        )

    if current_best["official_score_detect_ocr_median_sec"] >= baseline_candidate["official_score_detect_ocr_median_sec"] * 0.95:
        phase_results.append(
            {
                "phase": "phase-5-code-candidate",
                "current_best_before": current_best["preset"],
                "best_preset_after": current_best["preset"],
                "skipped": True,
                "skip_reason": "No code candidate implemented; current config winner did not exceed 5% overall improvement threshold.",
                "candidates": [],
            }
        )

    manifest_path = _suite_manifest(
        suite_dir=suite_dir,
        baseline_sha=baseline_sha,
        develop_ref_sha=develop_ref_sha,
        gold_path=gold_path,
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

    report_cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "generate_paddleocr_vl15_report.py"),
        "--manifest",
        str(manifest_path),
    ]
    subprocess.run(report_cmd, cwd=str(ROOT), check=True)
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
    parser = argparse.ArgumentParser(description="PaddleOCR-VL 1.5 actual-pipeline benchmark family.")
    subparsers = parser.add_subparsers(dest="command", required=False)

    run_parser = subparsers.add_parser("run", help="Run a single PaddleOCR-VL 1.5 benchmark candidate.")
    run_parser.add_argument("preset", nargs="?", default=FAMILY_PRESET_BASE)
    run_parser.add_argument("mode", nargs="?", default="batch", choices=("one-page", "batch", "webtoon"))
    run_parser.add_argument("runtime_mode", nargs="?", default="managed", choices=("attach-running", "managed"))
    run_parser.add_argument("repeat", nargs="?", default="1")
    run_parser.add_argument("sample_dir", nargs="?", default=str(DEFAULT_SAMPLE_DIR))
    run_parser.add_argument("sample_count", nargs="?", default=str(DEFAULT_SAMPLE_COUNT))

    gold_parser = subparsers.add_parser("gold", help="Generate baseline gold JSON from an existing run.")
    gold_parser.add_argument("--run-dir", default="")
    gold_parser.add_argument("--output", default="")

    compare_parser = subparsers.add_parser("compare", help="Compare an existing run against baseline gold.")
    compare_parser.add_argument("--baseline-gold", default="")
    compare_parser.add_argument("--candidate-run-dir", default="")

    summary_parser = subparsers.add_parser("summary", help="Generate the PaddleOCR-VL 1.5 report from a suite manifest.")
    summary_parser.add_argument("--manifest", default="")

    subparsers.add_parser("suite", help="Run the full PaddleOCR-VL 1.5 suite.")
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
        output_root = family_output_root()
        for repeat_index in range(1, max(1, repeat) + 1):
            run_dir = create_run_dir(
                f"{Path(str(preset_ref)).stem}_{args.mode}_r{repeat_index}",
                root=output_root,
            )
            result = _run_pipeline(
                preset_ref=preset_ref,
                mode=args.mode,
                runtime_mode=args.runtime_mode,
                sample_dir=sample_dir,
                sample_count=sample_count,
                output_dir=run_dir,
                label=run_dir.name,
            )
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
