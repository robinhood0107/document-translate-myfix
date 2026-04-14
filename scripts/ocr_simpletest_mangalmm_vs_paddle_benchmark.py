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
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_common import (
    collect_managed_llama_cpp_runtimes,
    create_run_dir,
    ensure_managed_runtime_health_first,
    load_preset,
    repo_relative_str,
    run_command,
    write_json,
)
from modules.utils.gpu_metrics import collect_runtime_snapshot, write_snapshot_json

FAMILY_NAME = "ocr_simpletest_mangalmm_vs_paddle"
LAST_SUITE_RECORD = "last_ocr_simpletest_mangalmm_vs_paddle_suite.json"
DEFAULT_SAMPLE_DIR = ROOT / "Sample" / "simpletest"
DEFAULT_SAMPLE_FILES = ("p_016.jpg", "p_017.jpg", "p_021.jpg")
CANDIDATES = [
    {
        "key": "paddleocr_vl",
        "label": "PaddleOCR VL + current promoted Gemma4",
        "preset": "ocr-simpletest-japan-paddleocr-vl-gemma4",
    },
    {
        "key": "mangalmm",
        "label": "MangaLMM + current promoted Gemma4",
        "preset": "ocr-simpletest-japan-mangalmm-gemma4",
    },
]


def _log(message: str) -> None:
    print(f"[ocr-simpletest] {message}", flush=True)


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


def _latest_suite_dir() -> Path | None:
    last_record = family_output_root() / LAST_SUITE_RECORD
    if last_record.is_file():
        try:
            payload = json.loads(last_record.read_text(encoding="utf-8"))
            suite_dir = Path(str(payload.get("suite_dir", "")).strip())
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


def _resolve_sample_paths(sample_dir: str | Path) -> list[Path]:
    root = Path(sample_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Sample directory does not exist: {root}\n"
            "Expected the simpletest corpus at ./Sample/simpletest or pass --sample-dir."
        )
    paths: list[Path] = []
    missing: list[str] = []
    for filename in DEFAULT_SAMPLE_FILES:
        candidate = root / filename
        if candidate.is_file():
            paths.append(candidate)
        else:
            missing.append(filename)
    if missing:
        raise FileNotFoundError(
            "Missing simpletest files: " + ", ".join(missing) + f"\nSample dir: {root}"
        )
    return paths


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
        print(f"[ocr-simpletest][{step_name}] {payload.rstrip()}", flush=True)

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


def _primary_gpu(snapshot: dict[str, Any]) -> dict[str, Any]:
    gpu = snapshot.get("gpu", {})
    primary = gpu.get("primary")
    return primary if isinstance(primary, dict) else {}


def _snapshot_memory_used(snapshot: dict[str, Any]) -> int | None:
    value = _primary_gpu(snapshot).get("memory_used_mb")
    return int(value) if isinstance(value, (int, float)) else None


def _snapshot_memory_free(snapshot: dict[str, Any]) -> int | None:
    value = _primary_gpu(snapshot).get("memory_free_mb")
    return int(value) if isinstance(value, (int, float)) else None


def _delta(current: int | None, baseline: int | None) -> int | None:
    if current is None or baseline is None:
        return None
    return int(current - baseline)


def _median(values: list[Any]) -> float | int | None:
    ordered = [float(value) for value in values if isinstance(value, (int, float))]
    if not ordered:
        return None
    median = statistics.median(ordered)
    rounded = round(median, 3)
    return int(rounded) if float(rounded).is_integer() else rounded


def _boot_runtime(
    preset_ref: str,
    runtime_dir: Path,
    *,
    runtime_services: str,
) -> tuple[dict[str, Any], list[str], dict[str, dict[str, str]]]:
    preset, _ = load_preset(preset_ref)
    runtime_state = ensure_managed_runtime_health_first(
        preset,
        runtime_dir,
        runtime_services=runtime_services,
        log_fn=_log,
    )
    write_json(runtime_dir / "managed_runtime_policy.json", runtime_state["report"])
    container_names = list(runtime_state["container_names"])
    runtime_meta = collect_managed_llama_cpp_runtimes(preset, runtime_services)
    return preset, container_names, runtime_meta


def _capture_resident_metrics(
    *,
    suite_dir: Path,
    candidate: dict[str, str],
) -> dict[str, Any]:
    candidate_dir = suite_dir / "resident_snapshots" / candidate["key"]
    candidate_dir.mkdir(parents=True, exist_ok=True)
    preset_ref = candidate["preset"]

    baseline_snapshot = collect_runtime_snapshot([])
    write_snapshot_json(candidate_dir / "empty_baseline_snapshot.json", baseline_snapshot)

    _log(f"{candidate['key']} OCR-only runtime 부팅")
    _, ocr_only_names, ocr_only_meta = _boot_runtime(
        preset_ref,
        candidate_dir / "ocr_only_runtime",
        runtime_services="ocr-only",
    )
    time.sleep(3)
    ocr_only_snapshot = collect_runtime_snapshot(ocr_only_names)
    write_snapshot_json(candidate_dir / "ocr_only_idle_snapshot.json", ocr_only_snapshot)
    write_json(candidate_dir / "ocr_only_llama_cpp_runtime.json", ocr_only_meta)
    _write_container_logs(candidate_dir / "ocr_only_docker_logs", ocr_only_names)
    _log(f"{candidate['key']} full runtime 부팅")
    _, full_names, full_meta = _boot_runtime(
        preset_ref,
        candidate_dir / "full_runtime",
        runtime_services="full",
    )
    time.sleep(3)
    full_snapshot = collect_runtime_snapshot(full_names)
    write_snapshot_json(candidate_dir / "full_idle_snapshot.json", full_snapshot)
    write_json(candidate_dir / "full_llama_cpp_runtime.json", full_meta)
    _write_container_logs(candidate_dir / "full_docker_logs", full_names)

    baseline_used = _snapshot_memory_used(baseline_snapshot)
    ocr_used = _snapshot_memory_used(ocr_only_snapshot)
    full_used = _snapshot_memory_used(full_snapshot)
    full_free = _snapshot_memory_free(full_snapshot)
    ocr_free = _snapshot_memory_free(ocr_only_snapshot)

    return {
        "baseline_snapshot_path": repo_relative_str(candidate_dir / "empty_baseline_snapshot.json"),
        "ocr_only_idle_snapshot_path": repo_relative_str(candidate_dir / "ocr_only_idle_snapshot.json"),
        "full_idle_snapshot_path": repo_relative_str(candidate_dir / "full_idle_snapshot.json"),
        "ocr_only_idle_gpu_used_delta_mb": _delta(ocr_used, baseline_used),
        "full_idle_gpu_used_delta_mb": _delta(full_used, baseline_used),
        "gemma_added_idle_gpu_used_delta_mb": _delta(full_used, ocr_used),
        "gpu_floor_free_mb_after_ocr_only": ocr_free,
        "gpu_floor_free_mb_after_full_runtime": full_free,
        "llama_cpp_runtime": full_meta,
        "container_names": full_names,
    }


def _run_pipeline_once(
    *,
    preset_ref: str,
    run_dir: Path,
    sample_dir: Path,
    source_lang: str,
    target_lang: str,
) -> dict[str, Any]:
    preset_payload, _resolved_path = load_preset(preset_ref)
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "benchmark_pipeline.py"),
        "--preset",
        preset_ref,
        "--mode",
        "batch",
        "--repeat",
        "1",
        "--runtime-mode",
        "attach-running",
        "--runtime-services",
        "full",
        "--sample-dir",
        str(sample_dir),
        "--sample-count",
        str(len(DEFAULT_SAMPLE_FILES)),
        "--source-lang",
        source_lang,
        "--target-lang",
        target_lang,
        "--clear-app-caches",
        "--output-dir",
        str(run_dir),
    ]
    env = dict(os.environ)
    mangalmm_cfg = preset_payload.get("mangalmm_ocr_client", {}) if isinstance(preset_payload, dict) else {}
    if isinstance(mangalmm_cfg, dict):
        env["CT_MANGALMM_DEBUG_ROOT"] = str(run_dir / "mangalmm_debug")
        env["CT_MANGALMM_DEBUG_EXPORT_LIMIT"] = str(int(mangalmm_cfg.get("debug_export_limit", 96)))
        for config_key, env_name in (
            ("temperature", "CT_MANGALMM_TEMPERATURE"),
            ("top_k", "CT_MANGALMM_TOP_K"),
            ("text_expansion_ratio_x", "CT_MANGALMM_TEXT_EXPANSION_RATIO_X"),
            ("text_expansion_ratio_y", "CT_MANGALMM_TEXT_EXPANSION_RATIO_Y"),
        ):
            if config_key in mangalmm_cfg:
                env[env_name] = str(mangalmm_cfg[config_key])
    completed = _run_command_streaming(
        cmd=cmd,
        cwd=ROOT,
        env=env,
        step_name=run_dir.name,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Pipeline benchmark failed for {run_dir.name} (exit={completed.returncode}).")
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Expected summary.json at {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _build_candidate_summary(
    *,
    candidate: dict[str, str],
    suite_dir: Path,
    runs: list[dict[str, Any]],
    resident_metrics: dict[str, Any],
) -> dict[str, Any]:
    cold_runs = [run for run in runs if str(run.get("phase", "")).startswith("cold")]
    warm_runs = [run for run in runs if str(run.get("phase", "")).startswith("warm")]
    warm_summaries = [run["summary"] for run in warm_runs if isinstance(run.get("summary"), dict)]
    cold_summaries = [run["summary"] for run in cold_runs if isinstance(run.get("summary"), dict)]

    def values(key: str, source: list[dict[str, Any]]) -> list[Any]:
        return [item.get(key) for item in source]

    summary = {
        "candidate_key": candidate["key"],
        "candidate_label": candidate["label"],
        "preset": candidate["preset"],
        "suite_dir": str(suite_dir),
        "run_count": len(runs),
        "cold_run_dirs": [repo_relative_str(run["run_dir"]) for run in cold_runs],
        "warm_run_dirs": [repo_relative_str(run["run_dir"]) for run in warm_runs],
        "cold_elapsed_sec": _median(values("elapsed_sec", cold_summaries)),
        "warm_median_elapsed_sec": _median(values("elapsed_sec", warm_summaries)),
        "warm_median_ocr_total_sec": _median(values("ocr_total_sec", warm_summaries)),
        "warm_median_ocr_median_sec": _median(values("ocr_median_sec", warm_summaries)),
        "warm_median_detect_ocr_total_sec": _median(values("detect_ocr_total_sec", warm_summaries)),
        "warm_median_translate_median_sec": _median(values("translate_median_sec", warm_summaries)),
        "warm_median_gpu_peak_used_mb": _median(values("gpu_peak_used_mb", warm_summaries)),
        "warm_median_gpu_floor_free_mb": _median(values("gpu_floor_free_mb", warm_summaries)),
        "warm_median_gpu_peak_util_percent": _median(values("gpu_peak_util_percent", warm_summaries)),
        "warm_median_gpu_peak_mem_util_percent": _median(values("gpu_peak_mem_util_percent", warm_summaries)),
        "warm_total_page_failed_count": int(
            sum(int(item.get("page_failed_count") or 0) for item in warm_summaries)
        ),
        "resident_metrics": resident_metrics,
        "warm_run_summaries": warm_summaries,
    }
    return summary


def _render_candidate_markdown(summary: dict[str, Any]) -> str:
    resident = summary.get("resident_metrics", {})
    lines = [
        f"# {summary.get('candidate_label')}",
        "",
        "## Warm Summary",
        "",
        f"- warm_median_elapsed_sec: `{summary.get('warm_median_elapsed_sec')}`",
        f"- warm_median_ocr_total_sec: `{summary.get('warm_median_ocr_total_sec')}`",
        f"- warm_median_ocr_median_sec: `{summary.get('warm_median_ocr_median_sec')}`",
        f"- warm_median_detect_ocr_total_sec: `{summary.get('warm_median_detect_ocr_total_sec')}`",
        f"- warm_median_translate_median_sec: `{summary.get('warm_median_translate_median_sec')}`",
        f"- warm_total_page_failed_count: `{summary.get('warm_total_page_failed_count')}`",
        f"- warm_median_gpu_peak_used_mb: `{summary.get('warm_median_gpu_peak_used_mb')}`",
        f"- warm_median_gpu_floor_free_mb: `{summary.get('warm_median_gpu_floor_free_mb')}`",
        "",
        "## Resident VRAM",
        "",
        f"- ocr_only_idle_gpu_used_delta_mb: `{resident.get('ocr_only_idle_gpu_used_delta_mb')}`",
        f"- full_idle_gpu_used_delta_mb: `{resident.get('full_idle_gpu_used_delta_mb')}`",
        f"- gemma_added_idle_gpu_used_delta_mb: `{resident.get('gemma_added_idle_gpu_used_delta_mb')}`",
        f"- gpu_floor_free_mb_after_ocr_only: `{resident.get('gpu_floor_free_mb_after_ocr_only')}`",
        f"- gpu_floor_free_mb_after_full_runtime: `{resident.get('gpu_floor_free_mb_after_full_runtime')}`",
    ]
    return "\n".join(lines) + "\n"


def _pick_winner(candidate_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    ranked = [
        item
        for item in candidate_summaries
        if isinstance(item.get("warm_median_elapsed_sec"), (int, float))
    ]
    if not ranked:
        return None
    ranked.sort(
        key=lambda item: (
            0 if int(item.get("warm_total_page_failed_count") or 0) == 0 else 1,
            int(item.get("warm_total_page_failed_count") or 0),
            float(item.get("warm_median_elapsed_sec") or 0.0),
        )
    )
    return ranked[0]


def _winner_reason(candidate_summaries: list[dict[str, Any]], winner: dict[str, Any] | None) -> str:
    if not winner:
        return "No valid candidate summary was available."
    successful = [
        item for item in candidate_summaries if int(item.get("warm_total_page_failed_count") or 0) == 0
    ]
    if successful:
        return (
            "Lowest warm_median_elapsed_sec among candidates with zero warm page failures. "
            "Candidates with warm page failures are ranked behind successful runs."
        )
    return (
        "All candidates had warm page failures, so the comparison fell back to the fewest "
        "warm page failures and then the lowest warm_median_elapsed_sec."
    )


def _render_comparison_markdown(payload: dict[str, Any]) -> str:
    winner = payload.get("winner") or {}
    candidates = payload.get("candidates") or []
    lines = [
        "# OCR Simpletest MangaLMM vs PaddleOCR VL",
        "",
        "## Decision",
        "",
        f"- winner: `{winner.get('candidate_key', 'n/a')}`",
        f"- reason: `{payload.get('winner_reason', 'n/a')}`",
        "",
        "## Warm Full-Pipeline Comparison",
        "",
        "| candidate | warm_median_elapsed_sec | warm_median_ocr_total_sec | warm_median_detect_ocr_total_sec | warm_median_translate_median_sec | warm_total_page_failed_count | warm_median_gpu_peak_used_mb | warm_median_gpu_floor_free_mb |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for candidate in candidates:
        lines.append(
            "| {candidate_key} | {warm_median_elapsed_sec} | {warm_median_ocr_total_sec} | {warm_median_detect_ocr_total_sec} | {warm_median_translate_median_sec} | {warm_total_page_failed_count} | {warm_median_gpu_peak_used_mb} | {warm_median_gpu_floor_free_mb} |".format(
                **candidate
            )
        )

    lines.extend(
        [
            "",
            "## Resident GPU Deltas",
            "",
            "| candidate | ocr_only_idle_gpu_used_delta_mb | full_idle_gpu_used_delta_mb | gemma_added_idle_gpu_used_delta_mb | gpu_floor_free_mb_after_ocr_only | gpu_floor_free_mb_after_full_runtime |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for candidate in candidates:
        resident = candidate.get("resident_metrics", {})
        lines.append(
            "| {candidate} | {ocr_only} | {full_idle} | {gemma_added} | {free_ocr} | {free_full} |".format(
                candidate=candidate.get("candidate_key"),
                ocr_only=resident.get("ocr_only_idle_gpu_used_delta_mb"),
                full_idle=resident.get("full_idle_gpu_used_delta_mb"),
                gemma_added=resident.get("gemma_added_idle_gpu_used_delta_mb"),
                free_ocr=resident.get("gpu_floor_free_mb_after_ocr_only"),
                free_full=resident.get("gpu_floor_free_mb_after_full_runtime"),
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- corpus: `Sample/simpletest` 3 pages (`p_016`, `p_017`, `p_021`)",
            "- run shape: `cold 1 + warm 2`",
            "- benchmark scope: `full-pipeline`",
            "- translator baseline: current promoted `Gemma4` preset kept fixed",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_comparison_artifacts(
    *,
    suite_dir: Path,
    candidate_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    winner = _pick_winner(candidate_summaries)
    payload = {
        "family": FAMILY_NAME,
        "suite_dir": str(suite_dir),
        "generated_at": time.time(),
        "winner": winner or {},
        "winner_reason": _winner_reason(candidate_summaries, winner),
        "candidates": candidate_summaries,
    }
    write_json(suite_dir / "comparison_summary.json", payload)
    (suite_dir / "comparison_summary.md").write_text(
        _render_comparison_markdown(payload),
        encoding="utf-8",
    )
    return payload


def _run_suite(
    *,
    sample_dir: Path,
    source_lang: str,
    target_lang: str,
) -> Path:
    _resolve_sample_paths(sample_dir)
    suite_dir = create_run_dir(f"{FAMILY_NAME}_suite", root=family_output_root())
    _log(f"suite 시작: {suite_dir}")
    write_json(
        suite_dir / "suite_request.json",
        {
            "family": FAMILY_NAME,
            "sample_dir": str(sample_dir),
            "sample_files": list(DEFAULT_SAMPLE_FILES),
            "source_lang": source_lang,
            "target_lang": target_lang,
            "run_shape": "cold 1 + warm 2",
            "candidates": CANDIDATES,
        },
    )

    candidate_summaries: list[dict[str, Any]] = []
    try:
        for candidate in CANDIDATES:
            _log(f"candidate 시작: {candidate['label']}")
            resident_metrics = _capture_resident_metrics(
                suite_dir=suite_dir,
                candidate=candidate,
            )
            runs: list[dict[str, Any]] = []
            candidate_dir = suite_dir / candidate["key"]
            candidate_dir.mkdir(parents=True, exist_ok=True)
            for phase in ("cold1", "warm1", "warm2"):
                run_dir = candidate_dir / phase
                summary = _run_pipeline_once(
                    preset_ref=candidate["preset"],
                    run_dir=run_dir,
                    sample_dir=sample_dir,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )
                runs.append({"phase": phase, "run_dir": run_dir, "summary": summary})
            candidate_summary = _build_candidate_summary(
                candidate=candidate,
                suite_dir=suite_dir,
                runs=runs,
                resident_metrics=resident_metrics,
            )
            write_json(candidate_dir / "candidate_summary.json", candidate_summary)
            (candidate_dir / "candidate_summary.md").write_text(
                _render_candidate_markdown(candidate_summary),
                encoding="utf-8",
            )
            candidate_summaries.append(candidate_summary)
        _write_comparison_artifacts(suite_dir=suite_dir, candidate_summaries=candidate_summaries)
        write_json(
            family_output_root() / LAST_SUITE_RECORD,
            {
                "family": FAMILY_NAME,
                "suite_dir": str(suite_dir),
                "comparison_summary": str(suite_dir / "comparison_summary.json"),
            },
        )
        return suite_dir
    finally:
        pass


def _summary_from_suite_dir(suite_dir: Path) -> dict[str, Any]:
    candidate_summaries: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        candidate_summary_path = suite_dir / candidate["key"] / "candidate_summary.json"
        if candidate_summary_path.is_file():
            candidate_summaries.append(json.loads(candidate_summary_path.read_text(encoding="utf-8")))
    if not candidate_summaries:
        raise FileNotFoundError(f"No candidate_summary.json files found in {suite_dir}")
    return _write_comparison_artifacts(suite_dir=suite_dir, candidate_summaries=candidate_summaries)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the simpletest 3-page MangaLMM vs PaddleOCR VL full-pipeline comparison."
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the comparison suite")
    run_parser.add_argument("--sample-dir", default=str(DEFAULT_SAMPLE_DIR))
    run_parser.add_argument("--source-lang", default="Japanese")
    run_parser.add_argument("--target-lang", default="Korean")

    summary_parser = subparsers.add_parser("summary", help="Regenerate comparison summary from an existing suite dir")
    summary_parser.add_argument("--suite-dir", default="")

    subparsers.add_parser("open", help="Open the benchmark output root on Windows")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = ["run"]
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "open":
        print(str(family_output_root()))
        return 0

    if args.command == "summary":
        suite_dir = Path(args.suite_dir) if args.suite_dir else _latest_suite_dir()
        if suite_dir is None:
            raise FileNotFoundError("No previous suite directory was found.")
        payload = _summary_from_suite_dir(suite_dir)
        _log(f"comparison summary 갱신 완료: {suite_dir / 'comparison_summary.json'}")
        print(json.dumps(payload.get("winner", {}), ensure_ascii=False, indent=2))
        return 0

    sample_dir = Path(args.sample_dir)
    suite_dir = _run_suite(
        sample_dir=sample_dir,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
    )
    _log(f"suite 완료: {repo_relative_str(suite_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
