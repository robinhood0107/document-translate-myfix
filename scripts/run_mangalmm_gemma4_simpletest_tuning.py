#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_common import create_run_dir, repo_relative_str, write_json

FAMILY_NAME = "mangalmm_gemma4_simpletest_tuning"
LAST_SUITE_RECORD = "last_mangalmm_gemma4_simpletest_tuning_suite.json"
STAGE1_NAME = "s1_ocr_probe"
STAGE2_NAME = "s2_ocr_confirm"
STAGE3_NAME = "s3_gemma_sweep"
STAGE4_NAME = "s4_thread_check"
STAGE5_NAME = "s5_confirm"
BASE_PRESET_PATH = (
    ROOT
    / "benchmarks"
    / "ocr_simpletest_mangalmm_vs_paddle"
    / "presets"
    / "ocr-simpletest-japan-mangalmm-gemma4.json"
)
SOURCE_LANG = "Japanese"
TARGET_LANG = "Korean"
DEFAULT_SAMPLE_FILES = ("p_016.jpg", "p_017.jpg", "p_021.jpg")
HARD_PAGE_FILES = ("p_016.jpg",)
STATE_FILE_NAME = "suite_state.json"
RETRY_DELAY_SEC = 15
MAX_INFRA_RETRIES = 3
RETRYABLE_INFRA_MARKERS = (
    "infra_retry",
    "connection refused",
    "remote end closed connection without response",
    "remotedisconnected",
    "failed to establish a new connection",
    "connection aborted",
    "service unavailable",
    "context deadline exceeded",
    "timed out waiting for",
    "temporary failure",
    "unexpected eof",
    "health/bootstrap/runtime connectivity issue detected",
)

OCR_STAGE_CANDIDATES = [
    {
        "slug": "ocr_diag_t00_k1_ctx06_mp1200_ls1280_tok256",
        "label": "Diag baseline t0.0 k1 ctx0.06 1200k/1280 tok256",
        "updates": {
            "mangalmm_ocr_client": {
                "raw_response_logging": True,
                "max_pixels": 1_200_000,
                "max_long_side": 1280,
                "max_completion_tokens": 256,
                "temperature": 0.0,
                "top_k": 1,
                "text_expansion_ratio_x": 0.06,
                "text_expansion_ratio_y": 0.06,
                "debug_export_limit": 160,
            }
        },
    },
    {
        "slug": "ocr_diag_t01_k32_ctx06_mp1200_ls1280_tok256",
        "label": "Diag t0.1 k32 ctx0.06 1200k/1280 tok256",
        "updates": {
            "mangalmm_ocr_client": {
                "raw_response_logging": True,
                "max_pixels": 1_200_000,
                "max_long_side": 1280,
                "max_completion_tokens": 256,
                "temperature": 0.1,
                "top_k": 32,
                "text_expansion_ratio_x": 0.06,
                "text_expansion_ratio_y": 0.06,
                "debug_export_limit": 160,
            }
        },
    },
    {
        "slug": "ocr_diag_t01_k32_ctx12_mp1200_ls1280_tok256",
        "label": "Diag t0.1 k32 ctx0.12 1200k/1280 tok256",
        "updates": {
            "mangalmm_ocr_client": {
                "raw_response_logging": True,
                "max_pixels": 1_200_000,
                "max_long_side": 1280,
                "max_completion_tokens": 256,
                "temperature": 0.1,
                "top_k": 32,
                "text_expansion_ratio_x": 0.12,
                "text_expansion_ratio_y": 0.12,
                "debug_export_limit": 160,
            }
        },
    },
    {
        "slug": "ocr_diag_t01_k48_ctx12_mp1100_ls1152_tok320",
        "label": "Diag t0.1 k48 ctx0.12 1100k/1152 tok320",
        "updates": {
            "mangalmm_ocr_client": {
                "raw_response_logging": True,
                "max_pixels": 1_100_000,
                "max_long_side": 1152,
                "max_completion_tokens": 320,
                "temperature": 0.1,
                "top_k": 48,
                "text_expansion_ratio_x": 0.12,
                "text_expansion_ratio_y": 0.12,
                "debug_export_limit": 160,
            }
        },
    },
    {
        "slug": "ocr_diag_t015_k64_ctx18_mp1000_ls1100_tok384",
        "label": "Diag t0.15 k64 ctx0.18 1000k/1100 tok384",
        "updates": {
            "mangalmm_ocr_client": {
                "raw_response_logging": True,
                "max_pixels": 1_000_000,
                "max_long_side": 1100,
                "max_completion_tokens": 384,
                "temperature": 0.15,
                "top_k": 64,
                "text_expansion_ratio_x": 0.18,
                "text_expansion_ratio_y": 0.16,
                "debug_export_limit": 160,
            }
        },
    },
    {
        "slug": "ocr_diag_t01_k32_ctx12_mp0900_ls1024_tok320",
        "label": "Diag t0.1 k32 ctx0.12 900k/1024 tok320",
        "updates": {
            "mangalmm_ocr_client": {
                "raw_response_logging": True,
                "max_pixels": 900_000,
                "max_long_side": 1024,
                "max_completion_tokens": 320,
                "temperature": 0.1,
                "top_k": 32,
                "text_expansion_ratio_x": 0.12,
                "text_expansion_ratio_y": 0.12,
                "debug_export_limit": 160,
            }
        },
    },
]

GEMMA_STAGE_CANDIDATES = [
    {
        "slug": "gemma_ngl23_ch6_tok512",
        "label": "Gemma ngl23 ch6 tok512 th10",
        "updates": {"gemma": {"n_gpu_layers": 23, "chunk_size": 6, "max_completion_tokens": 512, "threads": 10}},
    },
    {
        "slug": "gemma_ngl20_ch6_tok512",
        "label": "Gemma ngl20 ch6 tok512 th10",
        "updates": {"gemma": {"n_gpu_layers": 20, "chunk_size": 6, "max_completion_tokens": 512, "threads": 10}},
    },
    {
        "slug": "gemma_ngl23_ch5_tok512",
        "label": "Gemma ngl23 ch5 tok512 th10",
        "updates": {"gemma": {"n_gpu_layers": 23, "chunk_size": 5, "max_completion_tokens": 512, "threads": 10}},
    },
    {
        "slug": "gemma_ngl20_ch5_tok512",
        "label": "Gemma ngl20 ch5 tok512 th10",
        "updates": {"gemma": {"n_gpu_layers": 20, "chunk_size": 5, "max_completion_tokens": 512, "threads": 10}},
    },
    {
        "slug": "gemma_ngl23_ch6_tok384",
        "label": "Gemma ngl23 ch6 tok384 th10",
        "updates": {"gemma": {"n_gpu_layers": 23, "chunk_size": 6, "max_completion_tokens": 384, "threads": 10}},
    },
    {
        "slug": "gemma_ngl20_ch6_tok384",
        "label": "Gemma ngl20 ch6 tok384 th10",
        "updates": {"gemma": {"n_gpu_layers": 20, "chunk_size": 6, "max_completion_tokens": 384, "threads": 10}},
    },
    {
        "slug": "gemma_ngl23_ch5_tok384",
        "label": "Gemma ngl23 ch5 tok384 th10",
        "updates": {"gemma": {"n_gpu_layers": 23, "chunk_size": 5, "max_completion_tokens": 384, "threads": 10}},
    },
    {
        "slug": "gemma_ngl20_ch5_tok384",
        "label": "Gemma ngl20 ch5 tok384 th10",
        "updates": {"gemma": {"n_gpu_layers": 20, "chunk_size": 5, "max_completion_tokens": 384, "threads": 10}},
    },
]


def _log(message: str) -> None:
    print(f"[mangalmm-gemma4-tuning] {message}", flush=True)


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


def _state_path(suite_dir: Path) -> Path:
    return suite_dir / STATE_FILE_NAME


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _load_base_preset() -> dict[str, Any]:
    return _load_json(BASE_PRESET_PATH)


def _resolve_sample_root() -> Path:
    env_dir = os.getenv("CT_SIMPLETEST_DIR", "").strip()
    candidates: list[Path] = []
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(ROOT / "Sample" / "simpletest")
    candidates.append(ROOT.parent / "comic-translate" / "Sample" / "simpletest")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find Sample/simpletest. Set CT_SIMPLETEST_DIR or keep Sample/simpletest available."
    )


def _prepare_sample_subset(suite_dir: Path, subset_name: str, files: tuple[str, ...]) -> Path:
    source_root = _resolve_sample_root()
    sample_dir = suite_dir / "_samples" / subset_name
    if sample_dir.is_dir():
        return sample_dir
    sample_dir.mkdir(parents=True, exist_ok=True)
    for filename in files:
        src = source_root / filename
        if not src.is_file():
            raise FileNotFoundError(f"Missing sample file: {src}")
        shutil.copy2(src, sample_dir / filename)
    return sample_dir


def _materialize_preset(suite_dir: Path, slug: str, updates: dict[str, Any]) -> Path:
    preset_dir = suite_dir / "_materialized_presets"
    preset_dir.mkdir(parents=True, exist_ok=True)
    payload = copy.deepcopy(_load_base_preset())
    _deep_merge(payload, updates)
    payload["name"] = slug
    payload["description"] = f"Materialized tuning preset for {slug}"
    preset_path = preset_dir / f"{slug}.json"
    preset_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return preset_path


def _update_state(suite_dir: Path, **updates: Any) -> None:
    state = {}
    state_path = _state_path(suite_dir)
    if state_path.is_file():
        try:
            state = _load_json(state_path)
        except Exception:
            state = {}
    state.update(updates)
    write_json(state_path, state)


def _is_retryable(stdout: str, stderr: str) -> bool:
    combined = "\n".join([stdout or "", stderr or ""]).lower()
    return any(marker in combined for marker in RETRYABLE_INFRA_MARKERS)


def _run_pipeline_once(*, preset_path: Path, run_dir: Path, sample_dir: Path) -> tuple[subprocess.CompletedProcess[str], dict[str, Any] | None]:
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
        "managed",
        "--runtime-services",
        "full",
        "--sample-dir",
        str(sample_dir),
        "--sample-count",
        str(len([path for path in sample_dir.iterdir() if path.is_file()])),
        "--output-dir",
        str(run_dir),
        "--source-lang",
        SOURCE_LANG,
        "--target-lang",
        TARGET_LANG,
        "--export-page-snapshots",
        "--clear-app-caches",
    ]
    env = dict(os.environ)
    env["CT_BENCH_OUTPUT_ROOT"] = str(family_output_root())
    preset_payload = _load_json(preset_path)
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
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command_stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
    (run_dir / "command_stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
    summary_path = run_dir / "summary.json"
    summary = _load_json(summary_path) if summary_path.is_file() else None
    return completed, summary


def _collect_mangalmm_debug_summary(run_dir: Path) -> dict[str, Any]:
    debug_root = run_dir / "mangalmm_debug"
    if not debug_root.is_dir():
        return {}

    response_kind_counts: dict[str, int] = {}
    failure_reason_counts: dict[str, int] = {}
    crop_source_counts: dict[str, int] = {}
    sample_artifacts: list[str] = []
    artifact_count = 0

    for meta_path in sorted(debug_root.glob("*/meta.json")):
        payload = _load_json(meta_path)
        artifact_count += 1
        response_kind = str(payload.get("response_kind", "") or "unknown")
        failure_reason = str(payload.get("failure_reason", "") or "unknown")
        crop_source = str(payload.get("crop_source", "") or "unknown")
        response_kind_counts[response_kind] = int(response_kind_counts.get(response_kind, 0)) + 1
        failure_reason_counts[failure_reason] = int(failure_reason_counts.get(failure_reason, 0)) + 1
        crop_source_counts[crop_source] = int(crop_source_counts.get(crop_source, 0)) + 1
        if len(sample_artifacts) < 8:
            sample_artifacts.append(repo_relative_str(meta_path.parent))

    return {
        "debug_root": repo_relative_str(debug_root),
        "artifact_count": artifact_count,
        "response_kind_counts": dict(sorted(response_kind_counts.items())),
        "failure_reason_counts": dict(sorted(failure_reason_counts.items())),
        "crop_source_counts": dict(sorted(crop_source_counts.items())),
        "sample_artifacts": sample_artifacts,
    }


def _gate_reasons(summary: dict[str, Any] | None, expected_pages: int, returncode: int) -> list[str]:
    reasons: list[str] = []
    if returncode != 0:
        reasons.append(f"returncode={returncode}")
    if not isinstance(summary, dict):
        reasons.append("missing summary.json")
        return reasons
    page_failed = int(summary.get("page_failed_count") or 0)
    page_done = int(summary.get("page_done_count") or 0)
    if page_failed > 0:
        reasons.append(f"page_failed_count={page_failed}")
    if page_done < expected_pages:
        reasons.append(f"page_done_count={page_done}/{expected_pages}")
    total_blocks = int(summary.get("ocr_total_block_count") or 0)
    if total_blocks <= 0:
        reasons.append("ocr_total_block_count=0")
    if int(summary.get("gemma_empty_content_count") or 0) > 0:
        reasons.append(f"gemma_empty_content_count={int(summary.get('gemma_empty_content_count') or 0)}")
    if int(summary.get("gemma_schema_validation_fail_count") or 0) > 0:
        reasons.append(
            f"gemma_schema_validation_fail_count={int(summary.get('gemma_schema_validation_fail_count') or 0)}"
        )
    return reasons


def _number(summary: dict[str, Any] | None, key: str, default: float = 10**9) -> float:
    if not isinstance(summary, dict):
        return default
    value = summary.get(key)
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _candidate_sort_key(record: dict[str, Any]) -> tuple[int, int, float, float, float]:
    summary = record.get("summary") if isinstance(record.get("summary"), dict) else None
    return (
        0 if record.get("passed_gate") else 1,
        int(summary.get("page_failed_count") or 0) if isinstance(summary, dict) else 10**6,
        _number(summary, "elapsed_sec"),
        _number(summary, "translate_median_sec"),
        -_number(summary, "gpu_floor_free_mb", default=-10**9),
    )


def _pick_top(records: list[dict[str, Any]], count: int, stable_only: bool) -> list[dict[str, Any]]:
    ranked = sorted(records, key=_candidate_sort_key)
    if stable_only:
        ranked = [record for record in ranked if record.get("passed_gate")]
    return ranked[:count]


def _run_candidate(
    *,
    suite_dir: Path,
    stage_name: str,
    candidate_slug: str,
    label: str,
    updates: dict[str, Any],
    sample_files: tuple[str, ...],
) -> dict[str, Any]:
    sample_dir = _prepare_sample_subset(suite_dir, f"{stage_name}_{candidate_slug}", sample_files)
    preset_path = _materialize_preset(suite_dir, f"{stage_name}_{candidate_slug}", updates)
    final_completed: subprocess.CompletedProcess[str] | None = None
    final_summary: dict[str, Any] | None = None
    final_run_dir: Path | None = None
    infra_attempts = 0
    while infra_attempts < MAX_INFRA_RETRIES:
        infra_attempts += 1
        run_dir = suite_dir / stage_name / candidate_slug / f"infra_attempt{infra_attempts:02d}"
        _update_state(
            suite_dir,
            status="running",
            current_stage=stage_name,
            current_candidate=candidate_slug,
            infra_attempt=infra_attempts,
        )
        _log(f"run: stage={stage_name} candidate={candidate_slug} infra_attempt={infra_attempts}")
        completed, summary = _run_pipeline_once(preset_path=preset_path, run_dir=run_dir, sample_dir=sample_dir)
        final_completed = completed
        final_summary = summary
        final_run_dir = run_dir
        if completed.returncode == 0 or not _is_retryable(completed.stdout or "", completed.stderr or ""):
            break
        _log(f"retryable infra issue: stage={stage_name} candidate={candidate_slug}; retry after {RETRY_DELAY_SEC}s")
        time.sleep(RETRY_DELAY_SEC)

    assert final_completed is not None
    assert final_run_dir is not None
    reasons = _gate_reasons(final_summary, len(sample_files), final_completed.returncode)
    record = {
        "stage": stage_name,
        "candidate_slug": candidate_slug,
        "label": label,
        "sample_files": list(sample_files),
        "run_dir": repo_relative_str(final_run_dir),
        "preset_path": repo_relative_str(preset_path),
        "infra_attempts": infra_attempts,
        "returncode": final_completed.returncode,
        "passed_gate": not reasons,
        "gate_reasons": reasons,
        "updates": updates,
        "summary": final_summary or {},
        "mangalmm_debug": _collect_mangalmm_debug_summary(final_run_dir),
    }
    write_json(final_run_dir / "tuning_record.json", record)
    return record


def _write_stage_summary(suite_dir: Path, stage_name: str, records: list[dict[str, Any]], selected: list[dict[str, Any]]) -> None:
    payload = {
        "stage": stage_name,
        "generated_at": time.time(),
        "records": records,
        "selected": selected,
    }
    stage_dir = suite_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    write_json(stage_dir / "stage_summary.json", payload)


def _report_lines_for_record(record: dict[str, Any]) -> list[str]:
    summary = record.get("summary") if isinstance(record.get("summary"), dict) else {}
    return [
        f"- `{record.get('candidate_slug')}`",
        f"  - passed_gate: `{record.get('passed_gate')}`",
        f"  - gate_reasons: `{', '.join(record.get('gate_reasons', [])) or 'none'}`",
        f"  - elapsed_sec: `{summary.get('elapsed_sec')}`",
        f"  - detect_ocr_total_sec: `{summary.get('detect_ocr_total_sec')}`",
        f"  - ocr_total_sec: `{summary.get('ocr_total_sec')}`",
        f"  - translate_median_sec: `{summary.get('translate_median_sec')}`",
        f"  - gpu_peak_used_mb: `{summary.get('gpu_peak_used_mb')}`",
        f"  - gpu_floor_free_mb: `{summary.get('gpu_floor_free_mb')}`",
        f"  - mangalmm_response_kinds: `{(record.get('mangalmm_debug') or {}).get('response_kind_counts', {})}`",
        f"  - mangalmm_failure_reasons: `{(record.get('mangalmm_debug') or {}).get('failure_reason_counts', {})}`",
        f"  - mangalmm_crop_sources: `{(record.get('mangalmm_debug') or {}).get('crop_source_counts', {})}`",
        f"  - run_dir: `{record.get('run_dir')}`",
    ]


def _render_report(suite_dir: Path, stages: dict[str, list[dict[str, Any]]], final_winner: dict[str, Any] | None, estimate_hours: str) -> str:
    lines = [
        "# MangaLMM + Gemma4 Simpletest Tuning",
        "",
        f"- suite_dir: `{repo_relative_str(suite_dir)}`",
        f"- estimate_when_started: `{estimate_hours}`",
        f"- baseline_preset: `{repo_relative_str(BASE_PRESET_PATH)}`",
        "",
    ]
    if final_winner:
        summary = final_winner.get("summary") if isinstance(final_winner.get("summary"), dict) else {}
        lines.extend(
            [
                "## Current Winner",
                "",
                f"- candidate: `{final_winner.get('candidate_slug')}`",
                f"- passed_gate: `{final_winner.get('passed_gate')}`",
                f"- elapsed_sec: `{summary.get('elapsed_sec')}`",
                f"- translate_median_sec: `{summary.get('translate_median_sec')}`",
                f"- gpu_floor_free_mb: `{summary.get('gpu_floor_free_mb')}`",
                f"- run_dir: `{final_winner.get('run_dir')}`",
                "",
            ]
        )
    for stage_name, records in stages.items():
        lines.extend([f"## {stage_name}", ""])
        for record in records:
            lines.extend(_report_lines_for_record(record))
        lines.append("")
    return "\n".join(lines) + "\n"


def _write_suite_report(
    suite_dir: Path,
    *,
    stage_records: dict[str, list[dict[str, Any]]],
    final_winner: dict[str, Any] | None,
    estimate_hours: str,
) -> None:
    payload = {
        "family": FAMILY_NAME,
        "suite_dir": repo_relative_str(suite_dir),
        "generated_at": time.time(),
        "stage_records": stage_records,
        "winner": final_winner or {},
        "estimate_when_started": estimate_hours,
    }
    write_json(suite_dir / "final_summary.json", payload)
    (suite_dir / "final_summary.md").write_text(
        _render_report(suite_dir, stage_records, final_winner, estimate_hours),
        encoding="utf-8",
    )


def _stage1_ocr_probe(suite_dir: Path) -> list[dict[str, Any]]:
    records = [
        _run_candidate(
            suite_dir=suite_dir,
            stage_name=STAGE1_NAME,
            candidate_slug=candidate["slug"],
            label=candidate["label"],
            updates=candidate["updates"],
            sample_files=HARD_PAGE_FILES,
        )
        for candidate in OCR_STAGE_CANDIDATES
    ]
    selected = _pick_top(records, count=2, stable_only=True)
    if not selected:
        selected = _pick_top(records, count=2, stable_only=False)
    _write_stage_summary(suite_dir, STAGE1_NAME, records, selected)
    return records


def _stage2_ocr_confirm(suite_dir: Path, ocr_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for candidate in ocr_candidates:
        records.append(
            _run_candidate(
                suite_dir=suite_dir,
                stage_name=STAGE2_NAME,
                candidate_slug=str(candidate["candidate_slug"]),
                label=str(candidate["label"]),
                updates=copy.deepcopy(candidate["updates"]),
                sample_files=DEFAULT_SAMPLE_FILES,
            )
        )
    selected = _pick_top(records, count=1, stable_only=True)
    _write_stage_summary(suite_dir, STAGE2_NAME, records, selected)
    return records


def _stage3_gemma_sweep(suite_dir: Path, ocr_winner: dict[str, Any]) -> list[dict[str, Any]]:
    base_updates = copy.deepcopy(ocr_winner["updates"])
    records: list[dict[str, Any]] = []
    for candidate in GEMMA_STAGE_CANDIDATES:
        merged = copy.deepcopy(base_updates)
        _deep_merge(merged, candidate["updates"])
        records.append(
            _run_candidate(
                suite_dir=suite_dir,
                stage_name=STAGE3_NAME,
                candidate_slug=candidate["slug"],
                label=candidate["label"],
                updates=merged,
                sample_files=DEFAULT_SAMPLE_FILES,
            )
        )
    selected = _pick_top(records, count=2, stable_only=True)
    _write_stage_summary(suite_dir, STAGE3_NAME, records, selected)
    return records


def _stage4_thread_check(suite_dir: Path, finalists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for finalist in finalists[:2]:
        merged = copy.deepcopy(finalist["updates"])
        _deep_merge(merged, {"gemma": {"threads": 12}})
        records.append(
            _run_candidate(
                suite_dir=suite_dir,
                stage_name=STAGE4_NAME,
                candidate_slug=f"{finalist['candidate_slug']}_th12",
                label=f"{finalist['label']} th12",
                updates=merged,
                sample_files=DEFAULT_SAMPLE_FILES,
            )
        )
    selected = _pick_top(records, count=2, stable_only=True)
    _write_stage_summary(suite_dir, STAGE4_NAME, records, selected)
    return records


def _stage5_confirm(suite_dir: Path, winner: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for repeat_index in range(1, 3):
        records.append(
            _run_candidate(
                suite_dir=suite_dir,
                stage_name=STAGE5_NAME,
                candidate_slug=f"{winner['candidate_slug']}_confirm{repeat_index:02d}",
                label=f"{winner['label']} confirm {repeat_index}",
                updates=copy.deepcopy(winner["updates"]),
                sample_files=DEFAULT_SAMPLE_FILES,
            )
        )
    selected = _pick_top(records, count=1, stable_only=True)
    _write_stage_summary(suite_dir, STAGE5_NAME, records, selected)
    return records


def _write_latest_suite_record(suite_dir: Path) -> None:
    write_json(
        family_output_root() / LAST_SUITE_RECORD,
        {
            "family": FAMILY_NAME,
            "suite_dir": repo_relative_str(suite_dir),
            "summary_json": repo_relative_str(suite_dir / "final_summary.json"),
            "summary_md": repo_relative_str(suite_dir / "final_summary.md"),
        },
    )


def run_until_done(*, suite_dir: Path | None = None) -> Path:
    actual_suite_dir = suite_dir or create_run_dir("mg4_tune_suite", root=family_output_root())
    actual_suite_dir.mkdir(parents=True, exist_ok=True)
    estimate_hours = "about 1.5 to 3 hours on the current CUDA13 setup"
    _write_latest_suite_record(actual_suite_dir)
    _update_state(
        actual_suite_dir,
        family=FAMILY_NAME,
        status="starting",
        current_stage="",
        current_candidate="",
        estimate_when_started=estimate_hours,
        base_preset=repo_relative_str(BASE_PRESET_PATH),
    )

    stage_records: dict[str, list[dict[str, Any]]] = {}
    stage1_records = _stage1_ocr_probe(actual_suite_dir)
    stage_records[STAGE1_NAME] = stage1_records
    stage1_selected = _pick_top(stage1_records, count=2, stable_only=True)
    if not stage1_selected:
        stage1_selected = _pick_top(stage1_records, count=2, stable_only=False)

    stage2_records = _stage2_ocr_confirm(actual_suite_dir, stage1_selected)
    stage_records[STAGE2_NAME] = stage2_records
    stage2_winners = _pick_top(stage2_records, count=1, stable_only=True)
    if not stage2_winners:
        _update_state(actual_suite_dir, status="failed", failure_reason=f"No stable OCR candidate survived {STAGE2_NAME}.")
        _write_suite_report(actual_suite_dir, stage_records=stage_records, final_winner=None, estimate_hours=estimate_hours)
        raise RuntimeError(f"No stable OCR candidate survived {STAGE2_NAME}.")
    ocr_winner = stage2_winners[0]

    stage3_records = _stage3_gemma_sweep(actual_suite_dir, ocr_winner)
    stage_records[STAGE3_NAME] = stage3_records
    stage3_finalists = _pick_top(stage3_records, count=2, stable_only=True)
    if not stage3_finalists:
        _update_state(actual_suite_dir, status="failed", failure_reason=f"No stable Gemma candidate survived {STAGE3_NAME}.")
        _write_suite_report(actual_suite_dir, stage_records=stage_records, final_winner=None, estimate_hours=estimate_hours)
        raise RuntimeError(f"No stable Gemma candidate survived {STAGE3_NAME}.")

    stage4_records = _stage4_thread_check(actual_suite_dir, stage3_finalists)
    stage_records[STAGE4_NAME] = stage4_records
    combined_finalists = sorted(stage3_finalists + _pick_top(stage4_records, count=2, stable_only=True), key=_candidate_sort_key)
    winner = combined_finalists[0]

    stage5_records = _stage5_confirm(actual_suite_dir, winner)
    stage_records[STAGE5_NAME] = stage5_records
    confirm_winners = _pick_top(stage5_records, count=1, stable_only=True)
    final_winner = confirm_winners[0] if confirm_winners else winner

    confirm_elapsed_values = [
        _number(record.get("summary") if isinstance(record.get("summary"), dict) else None, "elapsed_sec")
        for record in stage5_records
        if record.get("passed_gate")
    ]
    if confirm_elapsed_values and isinstance(final_winner.get("summary"), dict):
        final_winner = copy.deepcopy(final_winner)
        final_winner["summary"]["confirm_median_elapsed_sec"] = round(statistics.median(confirm_elapsed_values), 3)

    _update_state(
        actual_suite_dir,
        status="completed",
        current_stage="",
        current_candidate="",
        winner=final_winner,
    )
    _write_suite_report(actual_suite_dir, stage_records=stage_records, final_winner=final_winner, estimate_hours=estimate_hours)
    _write_latest_suite_record(actual_suite_dir)
    _log(f"completed: {repo_relative_str(actual_suite_dir)}")
    return actual_suite_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a long CUDA13 supervisor that tunes MangaLMM + Gemma4 on Sample/simpletest."
    )
    parser.add_argument("--suite-dir", default="", help="Existing suite dir to resume writing into. Omit to create a fresh suite.")
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir) if args.suite_dir else None
    final_suite_dir = run_until_done(suite_dir=suite_dir)
    print(final_suite_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
