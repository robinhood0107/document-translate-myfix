#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_common import (
    collect_managed_llama_cpp_runtimes,
    create_run_dir,
    ensure_managed_runtime_health_first,
    load_preset,
    repo_relative_str,
    resolve_runtime_container_names,
    run_command,
    write_json,
)
from compare_ocr_combo_reference import compare_runs
from modules.utils.gpu_metrics import collect_runtime_snapshot, query_gpu_metrics, write_snapshot_json

FAMILY_NAME = "paddleocr_vl_parallel"
LAST_SUITE_RECORD = "last_paddleocr_vl_parallel_suite.json"
DEFAULT_SAMPLE_DIR = ROOT / "Sample" / "japan_vllm_parallel_subset"
DEFAULT_BASE_PRESET = ROOT / "benchmarks" / "paddleocr_vl_parallel" / "presets" / "paddleocr-vl-parallel-base.json"
DEFAULT_WARMUP_RUNS = 1
DEFAULT_MEASURED_RUNS = 3
DEFAULT_SOURCE_LANG = "Japanese"
DEFAULT_TARGET_LANG = "Korean"
DEFAULT_GPU_SAMPLE_INTERVAL_SEC = 0.2
DEFAULT_RUNTIME_SERVICES = "ocr-only"
DEFAULT_STAGE_CEILING = "ocr"
SMOKE_CANDIDATE_KEYS = ("fixed_w8", "fixed_area_desc_w8", "auto_v1_cap4")
BASELINE_CANDIDATE_KEY = "fixed_w8"


def _log(message: str) -> None:
    print(f"[paddleocr-vl-parallel] {message}", flush=True)


def family_output_root() -> Path:
    env_root = os.getenv("CT_BENCH_OUTPUT_ROOT", "").strip()
    if env_root:
        root = Path(env_root)
        if root.name != FAMILY_NAME:
            root = root / FAMILY_NAME
    else:
        root = ROOT / "banchmark_result_log" / FAMILY_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_logged_path(value: str | Path) -> Path:
    path = Path(str(value).strip())
    if path.is_absolute():
        return path
    text = str(path)
    if text.startswith("./"):
        return ROOT / text[2:]
    return ROOT / path


def _latest_suite_dir() -> Path | None:
    last_record = family_output_root() / LAST_SUITE_RECORD
    if last_record.is_file():
        try:
            payload = json.loads(last_record.read_text(encoding="utf-8"))
            suite_dir = _resolve_logged_path(payload.get("suite_dir", ""))
            if suite_dir.is_dir():
                return suite_dir
        except Exception:
            pass
    candidates = sorted(
        (path for path in family_output_root().iterdir() if path.is_dir()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _candidate_matrix() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for workers in range(1, 9):
        candidates.append(
            {
                "key": f"fixed_w{workers}",
                "label": f"fixed workers={workers}",
                "scheduler_mode": "fixed",
                "parallel_workers": workers,
            }
        )
    for workers in range(1, 9):
        candidates.append(
            {
                "key": f"fixed_area_desc_w{workers}",
                "label": f"fixed_area_desc workers={workers}",
                "scheduler_mode": "fixed_area_desc",
                "parallel_workers": workers,
            }
        )
    for cap in (4, 6, 8):
        candidates.append(
            {
                "key": f"auto_v1_cap{cap}",
                "label": f"auto_v1 cap={cap}",
                "scheduler_mode": "auto_v1",
                "parallel_workers": cap,
            }
        )
    return candidates


def _ordered_candidates(candidate_keys: set[str], *, smoke: bool) -> list[dict[str, Any]]:
    matrix = _candidate_matrix()
    order_index = {item["key"]: index for index, item in enumerate(matrix)}
    selected = [
        item
        for item in matrix
        if not candidate_keys or item["key"] in candidate_keys
    ]
    if smoke:
        selected = [item for item in selected if item["key"] in SMOKE_CANDIDATE_KEYS]
    if not selected:
        return []

    selected_keys = {item["key"] for item in selected}
    if BASELINE_CANDIDATE_KEY not in selected_keys:
        baseline = next((item for item in matrix if item["key"] == BASELINE_CANDIDATE_KEY), None)
        if baseline is not None:
            selected.append(baseline)

    return sorted(
        selected,
        key=lambda item: (
            0 if item["key"] == BASELINE_CANDIDATE_KEY else 1,
            order_index.get(item["key"], 10_000),
        ),
    )


def _resolve_sample_paths(sample_dir: str | Path) -> list[Path]:
    root = Path(sample_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Sample directory does not exist: {root}\n"
            "Expected the curated subset at ./Sample/japan_vllm_parallel_subset."
        )
    selected_list = root / "selected_files.txt"
    if selected_list.is_file():
        names = [
            line.strip()
            for line in selected_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        paths = [root / name for name in names]
        missing = [path.name for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing curated subset files: {', '.join(missing)}")
        return paths
    return sorted(path for path in root.iterdir() if path.is_file())


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
    step_name: str,
) -> subprocess.CompletedProcess[str]:
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
        print(f"[paddleocr-vl-parallel][{step_name}] {payload.rstrip()}", flush=True)

    return_code = process.wait()
    output_text = "".join(combined)
    return subprocess.CompletedProcess(cmd, return_code, stdout=output_text, stderr="")


def _write_container_logs(log_dir: Path, container_names: list[str]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    for container_name in container_names:
        completed = run_command(
            ["docker", "logs", "--tail", "200", container_name],
            check=False,
        )
        (log_dir / f"{container_name}.log").write_text(
            (completed.stdout or "") + (completed.stderr or ""),
            encoding="utf-8",
        )


def _sample_gpu(output_path: Path, stop_event: Event, interval_sec: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        while not stop_event.is_set():
            payload = {"ts": time.time(), "gpu": query_gpu_metrics()}
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            fh.flush()
            stop_event.wait(interval_sec)


def _run_pipeline_once(
    *,
    preset_path: Path,
    run_dir: Path,
    sample_dir: Path,
    source_lang: str,
    target_lang: str,
    runtime_mode: str,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "benchmark_pipeline.py"),
        "--preset",
        str(preset_path),
        "--mode",
        "batch",
        "--repeat",
        "1",
        "--runtime-mode",
        runtime_mode,
        "--runtime-services",
        DEFAULT_RUNTIME_SERVICES,
        "--sample-dir",
        str(sample_dir),
        "--sample-count",
        str(len(_resolve_sample_paths(sample_dir))),
        "--source-lang",
        source_lang,
        "--target-lang",
        target_lang,
        "--clear-app-caches",
        "--export-page-snapshots",
        "--stage-ceiling",
        DEFAULT_STAGE_CEILING,
        "--output-dir",
        str(run_dir),
    ]
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("CT_DISABLE_UPDATE_CHECK", "1")
    env.setdefault("CT_ENABLE_MEMLOG", "1")
    env.setdefault("CT_ENABLE_GPU_BENCH", "1")
    env.setdefault("CT_MEMLOG_INTERVAL_SEC", "5")

    stop_event = Event()
    sampler = Thread(
        target=_sample_gpu,
        args=(run_dir / "gpu_samples.jsonl", stop_event, DEFAULT_GPU_SAMPLE_INTERVAL_SEC),
        daemon=True,
    )
    sampler.start()
    try:
        completed = _run_command_streaming(
            cmd=cmd,
            cwd=ROOT,
            env=env,
            step_name=run_dir.name,
        )
    finally:
        stop_event.set()
        sampler.join(timeout=2.0)

    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark_pipeline failed for {run_dir.name} with exit code {completed.returncode}"
        )

    summary_path = run_dir / "summary.json"
    page_snapshots_path = run_dir / "page_snapshots.json"
    metrics_path = run_dir / "metrics.jsonl"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Expected summary.json at {summary_path}")
    if not page_snapshots_path.is_file():
        raise FileNotFoundError(f"Expected page_snapshots.json at {page_snapshots_path}")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Expected metrics.jsonl at {metrics_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"Expected summary object in {summary_path}")

    _extract_request_artifacts(metrics_path, run_dir)
    return summary


def _load_metrics_rows(metrics_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _extract_request_artifacts(metrics_path: Path, run_dir: Path) -> None:
    rows = _load_metrics_rows(metrics_path)
    page_profiles_path = run_dir / "page_profiles.jsonl"
    request_events_path = run_dir / "request_events.jsonl"
    with page_profiles_path.open("w", encoding="utf-8") as page_fh, request_events_path.open("w", encoding="utf-8") as req_fh:
        for row in rows:
            tag = str(row.get("tag", "") or "")
            if tag not in {"ocr_end", "page_failed"}:
                continue
            if tag == "page_failed" and str(row.get("failed_stage", "") or "") != "ocr":
                continue
            profile = row.get("ocr_page_profile")
            if not isinstance(profile, dict):
                continue
            page_record = {
                "tag": tag,
                "ts": row.get("ts"),
                "image_name": row.get("image_name"),
                "image_path": row.get("image_path"),
                "page_profile": profile,
            }
            page_fh.write(json.dumps(page_record, ensure_ascii=False) + "\n")
            for request_index, request in enumerate(profile.get("request_records", []) or []):
                if not isinstance(request, dict):
                    continue
                event = {
                    "tag": tag,
                    "ts": row.get("ts"),
                    "image_name": row.get("image_name"),
                    "image_path": row.get("image_path"),
                    "request_index": request_index,
                    "scheduler_mode": profile.get("scheduler_mode"),
                    "chosen_workers": profile.get("chosen_workers"),
                    **request,
                }
                req_fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def _build_detector_manifest(page_snapshots_path: Path, output_path: Path) -> None:
    payload = json.loads(page_snapshots_path.read_text(encoding="utf-8"))
    pages = payload.get("pages", []) if isinstance(payload, dict) else []
    manifest_pages: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        blocks = page.get("blocks", []) if isinstance(page.get("blocks"), list) else []
        manifest_pages.append(
            {
                "image_name": page.get("image_name"),
                "image_stem": page.get("image_stem"),
                "block_count": len(blocks),
                "blocks": [
                    {
                        "block_index": index,
                        "xyxy": block.get("xyxy"),
                        "bubble_xyxy": block.get("bubble_xyxy"),
                        "text_class": block.get("text_class"),
                    }
                    for index, block in enumerate(blocks)
                    if isinstance(block, dict)
                ],
            }
        )
    write_json(
        output_path,
        {
            "generated_at": time.time(),
            "source_page_snapshots": repo_relative_str(page_snapshots_path),
            "pages": manifest_pages,
        },
    )


def _build_baseline_gold(page_snapshots_path: Path, output_path: Path, sample_dir: Path, run_dir: Path) -> None:
    payload = json.loads(page_snapshots_path.read_text(encoding="utf-8"))
    pages = payload.get("pages", []) if isinstance(payload, dict) else []
    gold_pages: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        blocks = page.get("blocks", []) if isinstance(page.get("blocks"), list) else []
        gold_pages.append(
            {
                "image_name": page.get("image_name"),
                "image_stem": page.get("image_stem"),
                "status": "active",
                "blocks": [
                    {
                        "xyxy": block.get("xyxy"),
                        "bubble_xyxy": block.get("bubble_xyxy"),
                        "text_class": block.get("text_class"),
                        "seed_text": block.get("text", ""),
                        "gold_text": block.get("text", ""),
                        "seed_translation": "",
                        "seed_normalized_translation": "",
                    }
                    for block in blocks
                    if isinstance(block, dict)
                ],
            }
        )
    write_json(
        output_path,
        {
            "generated_at": time.time(),
            "corpus": repo_relative_str(sample_dir),
            "review_status": "baseline_seed",
            "generated_from_run_dir": repo_relative_str(run_dir),
            "pages": gold_pages,
        },
    )


def _median(values: list[Any]) -> float | int | None:
    ordered = [float(value) for value in values if isinstance(value, (int, float))]
    if not ordered:
        return None
    value = statistics.median(ordered)
    rounded = round(value, 4)
    return int(rounded) if float(rounded).is_integer() else rounded


def _mean(values: list[Any]) -> float | None:
    ordered = [float(value) for value in values if isinstance(value, (int, float))]
    if not ordered:
        return None
    return round(sum(ordered) / len(ordered), 4)


def _candidate_aggregate(candidate: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [run for run in runs if run["run_kind"] == "measured"]
    warmup = [run for run in runs if run["run_kind"] == "warmup"]
    quality_runs = [run["quality"] for run in measured if isinstance(run.get("quality"), dict)]
    return {
        "candidate_key": candidate["key"],
        "label": candidate["label"],
        "scheduler_mode": candidate["scheduler_mode"],
        "parallel_workers": candidate["parallel_workers"],
        "warmup_run_dirs": [repo_relative_str(run["run_dir"]) for run in warmup],
        "measured_run_dirs": [repo_relative_str(run["run_dir"]) for run in measured],
        "warmup_count": len(warmup),
        "measured_count": len(measured),
        "measured_elapsed_sec_median": _median([run["summary"].get("elapsed_sec") for run in measured]),
        "measured_ocr_total_sec_median": _median([run["summary"].get("ocr_total_sec") for run in measured]),
        "measured_detect_ocr_total_sec_median": _median([run["summary"].get("detect_ocr_total_sec") for run in measured]),
        "measured_ocr_page_p95_sec_median": _median([run["summary"].get("ocr_page_p95_sec") for run in measured]),
        "measured_gpu_peak_used_mb_median": _median([run["summary"].get("gpu_peak_used_mb") for run in measured]),
        "measured_gpu_floor_free_mb_median": _median([run["summary"].get("gpu_floor_free_mb") for run in measured]),
        "measured_page_failed_count_max": max(int(run["summary"].get("page_failed_count", 0) or 0) for run in measured) if measured else 0,
        "measured_ocr_empty_block_count_median": _median([run["summary"].get("ocr_empty_block_count") for run in measured]),
        "quality_mean_cer": _mean([(item.get("metrics") or {}).get("ocr_char_error_rate") for item in quality_runs]),
        "quality_mean_exact_match": _mean([(item.get("metrics") or {}).get("ocr_exact_text_match_ratio") for item in quality_runs]),
        "quality_hard_gate_failures": sorted(
            {
                failure
                for item in quality_runs
                for failure in (item.get("hard_gate_failures") or [])
            }
        ),
    }


def _evaluate_quality_gate(candidate_summary: dict[str, Any], baseline_summary: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    page_failed = int(candidate_summary.get("measured_page_failed_count_max") or 0)
    baseline_page_failed = int(baseline_summary.get("measured_page_failed_count_max") or 0)
    if page_failed > baseline_page_failed:
        failures.append(f"page_failed_count>{baseline_page_failed}: {page_failed}")

    empty_blocks = candidate_summary.get("measured_ocr_empty_block_count_median")
    baseline_empty_blocks = baseline_summary.get("measured_ocr_empty_block_count_median")
    if isinstance(empty_blocks, (int, float)) and isinstance(baseline_empty_blocks, (int, float)):
        if float(empty_blocks) > float(baseline_empty_blocks):
            failures.append(f"empty_block_count>{baseline_empty_blocks}: {empty_blocks}")

    baseline_cer = float(baseline_summary.get("quality_mean_cer") or 0.0)
    candidate_cer = candidate_summary.get("quality_mean_cer")
    if isinstance(candidate_cer, (int, float)) and float(candidate_cer) > baseline_cer + 0.001:
        failures.append(f"mean_CER>{baseline_cer + 0.001:.4f}: {candidate_cer}")

    baseline_exact = float(baseline_summary.get("quality_mean_exact_match") or 1.0)
    candidate_exact = candidate_summary.get("quality_mean_exact_match")
    if isinstance(candidate_exact, (int, float)) and float(candidate_exact) < baseline_exact - 0.002:
        failures.append(f"exact_match<{baseline_exact - 0.002:.4f}: {candidate_exact}")

    return not failures, failures


def _winner(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    passing = [item for item in candidates if item.get("quality_gate_pass")]
    if not passing:
        return {}
    return sorted(
        passing,
        key=lambda item: (
            float(item.get("measured_ocr_total_sec_median") or float("inf")),
            float(item.get("measured_ocr_page_p95_sec_median") or float("inf")),
            float(item.get("measured_elapsed_sec_median") or float("inf")),
            str(item.get("candidate_key", "")),
        ),
    )[0]


def _speed_rank(
    candidates: list[dict[str, Any]],
    *,
    include_baseline: bool,
    baseline_key: str = BASELINE_CANDIDATE_KEY,
) -> list[dict[str, Any]]:
    ranked = [
        item
        for item in candidates
        if include_baseline or str(item.get("candidate_key", "")) != baseline_key
    ]
    return sorted(
        ranked,
        key=lambda item: (
            float(item.get("measured_ocr_total_sec_median") or float("inf")),
            float(item.get("measured_ocr_page_p95_sec_median") or float("inf")),
            float(item.get("measured_elapsed_sec_median") or float("inf")),
            str(item.get("candidate_key", "")),
        ),
    )


def _review_candidate_snapshot(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_key": item.get("candidate_key"),
        "scheduler_mode": item.get("scheduler_mode"),
        "parallel_workers": item.get("parallel_workers"),
        "measured_ocr_total_sec_median": item.get("measured_ocr_total_sec_median"),
        "measured_ocr_page_p95_sec_median": item.get("measured_ocr_page_p95_sec_median"),
        "measured_elapsed_sec_median": item.get("measured_elapsed_sec_median"),
        "measured_page_failed_count_max": item.get("measured_page_failed_count_max"),
        "measured_ocr_empty_block_count_median": item.get("measured_ocr_empty_block_count_median"),
        "quality_mean_cer": item.get("quality_mean_cer"),
        "quality_mean_exact_match": item.get("quality_mean_exact_match"),
        "quality_gate_pass": item.get("quality_gate_pass"),
        "quality_gate_failures": item.get("quality_gate_failures"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_suite_state(suite_dir: Path, payload: dict[str, Any]) -> None:
    write_json(suite_dir / "suite_state.json", payload)


def _write_problem_solving_specs(
    suite_dir: Path,
    *,
    detector_manifest_path: Path,
    gold_path: Path,
    suite_summary: dict[str, Any],
) -> None:
    specs_dir = suite_dir / "problem_solving_specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    quality_gate_winner = suite_summary.get("quality_gate_winner", {}) or {}
    review_candidates = suite_summary.get("review_candidates", []) or []
    review_top1 = review_candidates[0] if len(review_candidates) >= 1 else {}
    review_top2 = review_candidates[1] if len(review_candidates) >= 2 else {}
    spec_payloads = {
        "01_detector_freeze.md": {
            "title": "문제 해결 명세서 01 - Detector Freeze",
            "condition": f"baseline manifest={repo_relative_str(detector_manifest_path)}",
            "measurement": f"pages={suite_summary.get('page_count')} baseline={suite_summary.get('baseline_candidate_key')}",
            "interpretation": "detector geometry를 baseline seed로 고정하고 이후 OCR만 비교한다.",
            "next_action": "baseline gold seed 생성",
        },
        "02_gold_seed.md": {
            "title": "문제 해결 명세서 02 - Gold Seed",
            "condition": f"baseline gold={repo_relative_str(gold_path)}",
            "measurement": "review_status=baseline_seed",
            "interpretation": "shipping baseline OCR 결과를 provisional gold로 고정했다.",
            "next_action": "candidate sweep 실행",
        },
        "03_candidate_sweep.md": {
            "title": "문제 해결 명세서 03 - Candidate Sweep",
            "condition": f"candidates={suite_summary.get('candidate_count')}",
            "measurement": f"pass_candidates={suite_summary.get('pass_candidate_count')}",
            "interpretation": "candidate matrix를 동일 corpus와 동일 runtime contract로 비교했다.",
            "next_action": "winner 결정",
        },
        "04_winner_decision.md": {
            "title": "문제 해결 명세서 04 - Review Candidate Selection",
            "condition": "rank_metric=ocr_total_sec median",
            "measurement": (
                f"quality_gate_winner={quality_gate_winner.get('candidate_key', 'n/a')} "
                f"review_top1={review_top1.get('candidate_key', 'n/a')} "
                f"review_top2={review_top2.get('candidate_key', 'n/a')}"
            ),
            "interpretation": (
                "속도 랭킹 상위 2개 비-baseline 후보를 사용자 OCR diff 검수 대상으로 고정하고, "
                "quality gate winner는 보조 해석 지표로만 유지한다."
            ),
            "next_action": "user OCR diff review",
        },
        "05_product_promotion.md": {
            "title": "문제 해결 명세서 05 - Product Promotion",
            "condition": "develop promotion requires explicit user approval after OCR diff review",
            "measurement": f"promotion_status={suite_summary.get('final_promotion_status', 'pending_user_review')}",
            "interpretation": "develop 기본값 승격은 사용자 검수 승인 전까지 잠겨 있으며, 이번 단계는 review pack 준비까지를 완료 상태로 본다.",
            "next_action": "prepare develop promotion branch after user approval",
        },
    }
    for filename, payload in spec_payloads.items():
        text = "\n".join(
            [
                f"# {payload['title']}",
                "",
                "핵심 문제 해결 방향은 사용자가 착안했다.",
                "",
                "## 가설",
                "",
                payload["interpretation"],
                "",
                "## 실험 조건",
                "",
                f"- {payload['condition']}",
                "",
                "## 측정값",
                "",
                f"- {payload['measurement']}",
                "",
                "## 해석",
                "",
                payload["interpretation"],
                "",
                "## 다음 행동",
                "",
                f"- {payload['next_action']}",
                "",
                "## 저자 및 기여",
                "",
                "- Idea Origin: User",
                "- Planning / Measurement Design / Implementation Detailing / Validation: Collaborative",
                "- Execution Support: Codex",
                "",
            ]
        )
        (specs_dir / filename).write_text(text, encoding="utf-8")


def _render_suite_markdown(summary: dict[str, Any]) -> str:
    quality_gate_winner = summary.get("quality_gate_winner", {}) or {}
    review_candidates = summary.get("review_candidates", []) or []
    lines = [
        "# PaddleOCR VL Parallel Suite Summary",
        "",
        f"- suite_dir: `{summary.get('suite_dir')}`",
        f"- runtime_contract: `{summary.get('runtime_contract')}`",
        f"- runtime_services: `{summary.get('runtime_services')}`",
        f"- stage_ceiling: `{summary.get('stage_ceiling')}`",
        f"- baseline_candidate_key: `{summary.get('baseline_candidate_key')}`",
        f"- page_count: `{summary.get('page_count')}`",
        f"- candidate_count: `{summary.get('candidate_count')}`",
        f"- pass_candidate_count: `{summary.get('pass_candidate_count')}`",
        f"- quality_gate_winner: `{quality_gate_winner.get('candidate_key', 'n/a')}`",
        f"- final_promotion_status: `{summary.get('final_promotion_status', 'pending_user_review')}`",
        "",
        "## Review Candidates",
        "",
    ]
    if review_candidates:
        for index, item in enumerate(review_candidates, start=1):
            lines.append(
                f"- top{index}: `{item.get('candidate_key', 'n/a')}` "
                f"(ocr_total_sec_median=`{item.get('measured_ocr_total_sec_median', 'n/a')}`, "
                f"gate_pass=`{item.get('quality_gate_pass', 'n/a')}`)"
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Candidate Table",
            "",
            "| candidate | scheduler | workers | ocr_total_sec_median | ocr_page_p95_sec_median | mean_CER | mean_exact_match | gate_pass |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in summary.get("candidates", []):
        lines.append(
            "| {candidate_key} | {scheduler_mode} | {parallel_workers} | {ocr_total} | {p95} | {cer} | {exact} | {passed} |".format(
                candidate_key=item.get("candidate_key"),
                scheduler_mode=item.get("scheduler_mode"),
                parallel_workers=item.get("parallel_workers"),
                ocr_total=item.get("measured_ocr_total_sec_median"),
                p95=item.get("measured_ocr_page_p95_sec_median"),
                cer=item.get("quality_mean_cer"),
                exact=item.get("quality_mean_exact_match"),
                passed=item.get("quality_gate_pass"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PaddleOCR VL OCR-only parallel tuning benchmark.")
    parser.add_argument("--base-preset", default=str(DEFAULT_BASE_PRESET))
    parser.add_argument("--sample-dir", default=str(DEFAULT_SAMPLE_DIR))
    parser.add_argument("--source-lang", default=DEFAULT_SOURCE_LANG)
    parser.add_argument("--target-lang", default=DEFAULT_TARGET_LANG)
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--measured-runs", type=int, default=DEFAULT_MEASURED_RUNS)
    parser.add_argument("--candidate-key", action="append", default=[])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-runtime-bootstrap", action="store_true")
    args = parser.parse_args()

    base_preset, base_preset_path = load_preset(args.base_preset)
    sample_dir = Path(args.sample_dir)
    sample_paths = _resolve_sample_paths(sample_dir)

    if args.smoke:
        selected_keys = set(SMOKE_CANDIDATE_KEYS)
        warmup_runs = 0
        measured_runs = 1
    else:
        selected_keys = set(args.candidate_key or [])
        warmup_runs = max(0, int(args.warmup_runs))
        measured_runs = max(1, int(args.measured_runs))

    candidates = _ordered_candidates(selected_keys, smoke=bool(args.smoke))
    if not candidates:
        raise SystemExit("No candidates selected.")

    suite_label = "paddleocr-vl-parallel-smoke" if args.smoke else "paddleocr-vl-parallel-suite"
    suite_dir = create_run_dir(suite_label, root=family_output_root())
    runtime_dir = suite_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    _log(f"suite start: {suite_dir}")
    suite_started_at = time.time()
    _write_suite_state(
        suite_dir,
        {
            "status": "running",
            "family": FAMILY_NAME,
            "suite_dir": repo_relative_str(suite_dir),
            "sample_dir": repo_relative_str(sample_dir),
            "page_count": len(sample_paths),
            "smoke": bool(args.smoke),
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "candidate_keys": [item["key"] for item in candidates],
            "completed_candidate_keys": [],
            "started_at": suite_started_at,
        },
    )

    if not args.skip_runtime_bootstrap:
        runtime_state = ensure_managed_runtime_health_first(
            base_preset,
            runtime_dir,
            runtime_services=DEFAULT_RUNTIME_SERVICES,
            log_fn=_log,
        )
        write_json(runtime_dir / "managed_runtime_policy.json", runtime_state["report"])
        container_names = list(runtime_state["container_names"])
    else:
        container_names = resolve_runtime_container_names(base_preset, runtime_services=DEFAULT_RUNTIME_SERVICES)

    runtime_snapshot = collect_runtime_snapshot(container_names)
    write_snapshot_json(runtime_dir / "runtime_snapshot.json", runtime_snapshot)
    write_json(runtime_dir / "llama_cpp_runtime.json", collect_managed_llama_cpp_runtimes(base_preset, DEFAULT_RUNTIME_SERVICES))

    candidate_results: list[dict[str, Any]] = []
    candidate_dir_by_key: dict[str, Path] = {}
    baseline_candidate_key = BASELINE_CANDIDATE_KEY
    baseline_gold_path = suite_dir / "baseline_gold.json"
    detector_manifest_path = suite_dir / "detector_manifest.json"

    for candidate in candidates:
        candidate_dir = suite_dir / candidate["key"]
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_dir_by_key[candidate["key"]] = candidate_dir
        candidate_preset = copy.deepcopy(base_preset)
        candidate_preset["name"] = candidate["key"]
        candidate_preset["description"] = candidate["label"]
        candidate_preset.setdefault("ocr_client", {})["parallel_workers"] = candidate["parallel_workers"]
        candidate_preset.setdefault("ocr_generic", {})["paddleocr_vl_scheduler_mode"] = candidate["scheduler_mode"]
        preset_path = candidate_dir / f"{candidate['key']}.json"
        preset_path.write_text(json.dumps(candidate_preset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        runs: list[dict[str, Any]] = []
        for run_kind, repeat_count in (("warmup", warmup_runs), ("measured", measured_runs)):
            for repeat_index in range(1, repeat_count + 1):
                run_dir = candidate_dir / f"{run_kind}_r{repeat_index}"
                run_dir.mkdir(parents=True, exist_ok=True)
                summary = _run_pipeline_once(
                    preset_path=preset_path,
                    run_dir=run_dir,
                    sample_dir=sample_dir,
                    source_lang=args.source_lang,
                    target_lang=args.target_lang,
                    runtime_mode="attach-running",
                )

                if candidate["key"] == baseline_candidate_key and run_kind == "measured" and repeat_index == 1:
                    _build_detector_manifest(run_dir / "page_snapshots.json", detector_manifest_path)
                    _build_baseline_gold(run_dir / "page_snapshots.json", baseline_gold_path, sample_dir, run_dir)

                quality = None
                if baseline_gold_path.is_file():
                    quality = compare_runs(baseline_gold_path, run_dir)
                    write_json(run_dir / "quality_summary.json", quality)

                runs.append(
                    {
                        "run_kind": run_kind,
                        "repeat_index": repeat_index,
                        "run_dir": run_dir,
                        "summary": summary,
                        "quality": quality,
                    }
                )

        candidate_summary = _candidate_aggregate(candidate, runs)
        candidate_results.append(candidate_summary)
        write_json(candidate_dir / "candidate_summary.json", candidate_summary)
        _write_suite_state(
            suite_dir,
            {
                "status": "running",
                "family": FAMILY_NAME,
                "suite_dir": repo_relative_str(suite_dir),
                "sample_dir": repo_relative_str(sample_dir),
                "page_count": len(sample_paths),
                "smoke": bool(args.smoke),
                "warmup_runs": warmup_runs,
                "measured_runs": measured_runs,
                "candidate_keys": [item["key"] for item in candidates],
                "completed_candidate_keys": [item["candidate_key"] for item in candidate_results],
                "started_at": suite_started_at,
                "last_completed_candidate": candidate_summary["candidate_key"],
            },
        )

    baseline_summary = next(
        (item for item in candidate_results if item["candidate_key"] == baseline_candidate_key),
        None,
    )
    if baseline_summary is None:
        raise RuntimeError("Baseline candidate fixed_w8 must be part of the suite.")

    for item in candidate_results:
        passed, failures = _evaluate_quality_gate(item, baseline_summary)
        item["quality_gate_pass"] = passed
        item["quality_gate_failures"] = failures
        candidate_dir = candidate_dir_by_key.get(str(item["candidate_key"]))
        if candidate_dir is not None:
            write_json(candidate_dir / "candidate_summary.json", item)

    quality_gate_winner = _winner(candidate_results)
    speed_ranked_candidates = _speed_rank(candidate_results, include_baseline=True)
    review_candidates = [
        _review_candidate_snapshot(item)
        for item in _speed_rank(candidate_results, include_baseline=False)[:2]
    ]
    candidate_results_sorted = sorted(
        candidate_results,
        key=lambda item: (
            0 if item.get("quality_gate_pass") else 1,
            float(item.get("measured_ocr_total_sec_median") or float("inf")),
            str(item.get("candidate_key", "")),
        ),
    )

    suite_summary = {
        "family": FAMILY_NAME,
        "suite_dir": repo_relative_str(suite_dir),
        "sample_dir": repo_relative_str(sample_dir),
        "page_count": len(sample_paths),
        "candidate_count": len(candidate_results_sorted),
        "pass_candidate_count": sum(1 for item in candidate_results_sorted if item.get("quality_gate_pass")),
        "baseline_candidate_key": baseline_candidate_key,
        "runtime_contract": "paddleocr-vl-single-tenant-ocr-only",
        "runtime_services": DEFAULT_RUNTIME_SERVICES,
        "stage_ceiling": DEFAULT_STAGE_CEILING,
        "baseline_gold_path": repo_relative_str(baseline_gold_path),
        "detector_manifest_path": repo_relative_str(detector_manifest_path),
        "quality_gate_winner": quality_gate_winner,
        "winner": quality_gate_winner,
        "speed_ranked_candidates": [_review_candidate_snapshot(item) for item in speed_ranked_candidates],
        "review_candidate_keys": [str(item.get("candidate_key", "")) for item in review_candidates],
        "review_candidates": review_candidates,
        "final_promotion_status": "pending_user_review",
        "candidates": candidate_results_sorted,
    }
    write_json(suite_dir / "suite_summary.json", suite_summary)
    (suite_dir / "suite_summary.md").write_text(_render_suite_markdown(suite_summary), encoding="utf-8")
    _write_csv(
        suite_dir / "candidate_summary.csv",
        candidate_results_sorted,
        [
            "candidate_key",
            "scheduler_mode",
            "parallel_workers",
            "measured_ocr_total_sec_median",
            "measured_detect_ocr_total_sec_median",
            "measured_ocr_page_p95_sec_median",
            "measured_gpu_peak_used_mb_median",
            "measured_gpu_floor_free_mb_median",
            "measured_page_failed_count_max",
            "measured_ocr_empty_block_count_median",
            "quality_mean_cer",
            "quality_mean_exact_match",
            "quality_gate_pass",
            "quality_gate_failures",
        ],
    )
    _write_problem_solving_specs(
        suite_dir,
        detector_manifest_path=detector_manifest_path,
        gold_path=baseline_gold_path,
        suite_summary=suite_summary,
    )
    _write_container_logs(suite_dir / "docker_logs", container_names)
    write_json(
        family_output_root() / LAST_SUITE_RECORD,
        {
            "suite_dir": repo_relative_str(suite_dir),
            "winner": quality_gate_winner,
            "quality_gate_winner": quality_gate_winner,
            "review_candidate_keys": [str(item.get("candidate_key", "")) for item in review_candidates],
        },
    )
    _write_suite_state(
        suite_dir,
        {
            "status": "completed",
            "family": FAMILY_NAME,
            "suite_dir": repo_relative_str(suite_dir),
            "sample_dir": repo_relative_str(sample_dir),
            "page_count": len(sample_paths),
            "smoke": bool(args.smoke),
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "candidate_keys": [item["key"] for item in candidates],
            "completed_candidate_keys": [item["candidate_key"] for item in candidate_results],
            "started_at": suite_started_at,
            "completed_at": time.time(),
            "winner": quality_gate_winner,
            "quality_gate_winner": quality_gate_winner,
            "review_candidate_keys": [str(item.get("candidate_key", "")) for item in review_candidates],
            "final_promotion_status": "pending_user_review",
        },
    )
    _log(
        "suite complete: quality_gate_winner={winner} review_top1={top1}".format(
            winner=quality_gate_winner.get("candidate_key", "n/a"),
            top1=review_candidates[0].get("candidate_key", "n/a") if review_candidates else "n/a",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
