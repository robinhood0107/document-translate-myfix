#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_common import (  # noqa: E402
    create_run_dir,
    load_metrics,
    repo_relative_str,
    summarize_metrics,
    write_json,
)

FAMILY_NAME = "workflow-split-runtime"
LAST_SUITE_RECORD = "last_workflow_split_runtime_suite.json"
DEFAULT_SAMPLE_DIR = ROOT / "Sample" / "japan"
DEFAULT_SOURCE_LANG = "Japanese"
DEFAULT_TARGET_LANG = "Korean"
OFFICIAL_FILES = (
    "094.png",
    "097.png",
    "101.png",
    "i_099.jpg",
    "i_100.jpg",
    "i_102.jpg",
    "i_105.jpg",
    "p_016.jpg",
    "p_017.jpg",
    "p_018.jpg",
    "p_019.jpg",
    "p_020.jpg",
    "p_021.jpg",
)
SMOKE_FILES = ("094.png", "p_016.jpg")
SCENARIOS = {
    "baseline_legacy": {
        "label": "Baseline Legacy",
        "preset": "workflow-split-runtime-baseline-legacy",
        "workflow_mode": "legacy_page_pipeline",
        "runner_kind": "legacy",
        "runtime_mode": "managed",
        "runtime_services": "full",
        "stage_ceiling": "render",
        "runnable": True,
        "description": "Current promoted page-unit pipeline using PaddleOCR VL and Gemma.",
    },
    "candidate_stage_batched_single_ocr": {
        "label": "Candidate Stage-Batched Single OCR",
        "preset": "workflow-split-runtime-stage-batched-single-ocr",
        "workflow_mode": "stage_batched_pipeline",
        "runner_kind": "stage_batched",
        "resident_ocr_mode": "single",
        "runtime_mode": "managed",
        "runtime_services": "stage-batched",
        "stage_ceiling": "render",
        "runnable": True,
        "description": "Detect all -> OCR all -> translate all -> inpaint all -> render/export all with one OCR runtime.",
    },
    "candidate_stage_batched_dual_resident": {
        "label": "Candidate Stage-Batched Dual Resident",
        "preset": "workflow-split-runtime-stage-batched-dual-resident",
        "workflow_mode": "stage_batched_pipeline",
        "runner_kind": "stage_batched",
        "resident_ocr_mode": "dual",
        "runtime_mode": "managed",
        "runtime_services": "stage-batched",
        "stage_ceiling": "render",
        "runnable": True,
        "description": "Exploratory OCR-stage residency contract with MangaLMM and PaddleOCR VL both kept resident.",
    },
}


def _log(message: str) -> None:
    print(f"[workflow-split-runtime] {message}", flush=True)


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


def _latest_suite_record_path() -> Path:
    return family_output_root() / LAST_SUITE_RECORD


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return _load_json(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_scenarios(selected: list[str]) -> list[str]:
    if not selected or selected == ["all"]:
        return list(SCENARIOS.keys())
    unknown = [name for name in selected if name not in SCENARIOS]
    if unknown:
        raise SystemExit(f"Unknown scenario(s): {', '.join(unknown)}")
    return selected


def _resolve_sample_paths(sample_dir: str | Path, *, smoke: bool) -> list[Path]:
    root = Path(sample_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Sample directory does not exist: {root}\n"
            "Expected the curated Requirement 1 corpus at ./Sample/japan."
        )
    filenames = SMOKE_FILES if smoke else OFFICIAL_FILES
    paths = [root / name for name in filenames]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing curated Requirement 1 files: " + ", ".join(missing) + f"\nSample dir: {root}"
        )
    return paths


def _stage_curated_corpus(run_dir: Path, sample_paths: list[Path]) -> Path:
    corpus_dir = run_dir / "curated_corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for path in sample_paths:
        shutil.copy2(path, corpus_dir / path.name)
    return corpus_dir


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
        print(f"[workflow-split-runtime][{step_name}] {payload.rstrip()}", flush=True)

    return_code = process.wait()
    output_text = "".join(combined)
    return subprocess.CompletedProcess(cmd, return_code, stdout=output_text, stderr="")


def _base_event_payload(
    *,
    scenario_name: str,
    workflow_mode: str,
    event_name: str,
    status: str,
    detail: str = "",
    source: str = "workflow_split_runtime_benchmark",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "ts": time.time(),
        "scenario_name": scenario_name,
        "workflow_mode": workflow_mode,
        "phase": "",
        "service": "",
        "event_name": event_name,
        "status": status,
        "image_path": "",
        "image_name": "",
        "image_index": None,
        "total_images": 0,
        "elapsed_sec": None,
        "detail": detail,
        "attempt_count": None,
        "cache_status": "",
        "block_count": None,
        "timeout_sec": None,
        "retry_reason": "",
        "source": source,
    }
    if extra:
        payload.update(extra)
    return payload


def _normalize_runtime_progress_events(
    rows: list[dict[str, Any]],
    *,
    scenario_name: str,
    workflow_mode: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    health_wait_open: set[tuple[str, str]] = set()
    for row in rows:
        if str(row.get("tag", "") or "") != "runtime_progress":
            continue
        service = str(row.get("service", "") or "")
        step_key = str(row.get("step_key", "") or "")
        status = str(row.get("status", "") or "")
        message = str(row.get("message", "") or "")
        detail = str(row.get("detail", "") or "")
        is_ocr = service in {"paddleocr_vl", "mangalmm", "hunyuanocr"}
        if service == "gemma":
            prefix = "gemma_runtime"
        elif is_ocr:
            prefix = "ocr_runtime"
        else:
            continue

        event_name = ""
        wait_key = (prefix, step_key)
        message_lower = message.lower()
        detail_lower = detail.lower()
        if step_key == "compose_up":
            if status == "starting":
                event_name = f"{prefix}_compose_up_start"
            elif status == "completed":
                event_name = f"{prefix}_compose_up_end"
        elif step_key in {"health_wait", "health_probe"}:
            if status == "waiting_health":
                if wait_key in health_wait_open:
                    continue
                health_wait_open.add(wait_key)
                event_name = f"{prefix}_health_wait_start"
            elif status == "completed":
                if "reuse" in message_lower or "재사용" in message:
                    event_name = f"{prefix}_reuse_hit"
                elif wait_key in health_wait_open:
                    health_wait_open.remove(wait_key)
                    event_name = f"{prefix}_health_wait_end"
            elif status == "failed":
                event_name = f"{prefix}_timeout" if "timed out" in detail_lower else f"{prefix}_retry"

        if not event_name:
            continue

        normalized.append(
            _base_event_payload(
                scenario_name=scenario_name,
                workflow_mode=workflow_mode,
                event_name=event_name,
                status=status or "recorded",
                detail=detail or message,
                source="metrics.jsonl",
                extra={
                    "ts": row.get("ts"),
                    "phase": row.get("phase", ""),
                    "service": service,
                    "image_path": row.get("image_path", ""),
                    "image_name": row.get("image_name", ""),
                    "image_index": row.get("page_index"),
                    "total_images": row.get("total_images", row.get("page_total", 0)),
                    "elapsed_sec": row.get("elapsed_sec"),
                },
            )
        )
    return normalized


def _normalize_metrics_events(
    rows: list[dict[str, Any]],
    *,
    scenario_name: str,
    workflow_mode: str,
) -> list[dict[str, Any]]:
    tag_map = {
        "batch_run_start": "batch_run_start",
        "batch_run_done": "batch_run_end",
        "detect_start": "detect_stage_start",
        "detect_end": "detect_stage_end",
        "ocr_start": "ocr_stage_start",
        "ocr_end": "ocr_stage_end",
        "translate_start": "translate_stage_start",
        "translate_end": "translate_stage_end",
        "inpaint_start": "inpaint_stage_start",
        "inpaint_end": "inpaint_stage_end",
        "render_start": "render_stage_start",
        "render_end": "render_stage_end",
        "page_done": "page_done",
        "page_failed": "page_failed",
        "benchmark_run_start": "benchmark_run_start",
        "benchmark_run_finished": "benchmark_run_end",
    }
    normalized = _normalize_runtime_progress_events(
        rows,
        scenario_name=scenario_name,
        workflow_mode=workflow_mode,
    )
    for row in rows:
        tag = str(row.get("tag", "") or "")
        event_name = tag_map.get(tag)
        if not event_name:
            continue
        normalized.append(
            _base_event_payload(
                scenario_name=scenario_name,
                workflow_mode=workflow_mode,
                event_name=event_name,
                status="completed" if tag.endswith("_end") or tag in {"page_done", "batch_run_done", "benchmark_run_finished"} else "started",
                detail=str(row.get("detail", "") or ""),
                source="metrics.jsonl",
                extra={
                    "ts": row.get("ts"),
                    "phase": row.get("phase", ""),
                    "service": row.get("service", ""),
                    "image_path": row.get("image_path", ""),
                    "image_name": row.get("image_name", ""),
                    "image_index": row.get("image_index"),
                    "total_images": row.get("total_images"),
                    "elapsed_sec": row.get("elapsed_sec"),
                    "attempt_count": row.get("attempt_count"),
                    "cache_status": row.get("cache_status", ""),
                    "block_count": row.get("block_count"),
                    "timeout_sec": row.get("timeout_sec"),
                    "retry_reason": row.get("retry_reason", ""),
                    "original_tag": tag,
                },
            )
        )
    normalized.sort(key=lambda item: (float(item.get("ts") or 0.0), str(item.get("event_name", ""))))
    return normalized


def _build_docker_timeline(
    *,
    scenario_name: str,
    workflow_mode: str,
    managed_runtime_policy: dict[str, Any],
) -> dict[str, Any]:
    if isinstance(managed_runtime_policy.get("report"), dict) and not managed_runtime_policy.get("groups"):
        managed_runtime_policy = managed_runtime_policy["report"]
    groups = []
    total_compose = 0.0
    total_health = 0.0
    for group in managed_runtime_policy.get("groups", []):
        if not isinstance(group, dict):
            continue
        compose_sec = float(group.get("compose_up_elapsed_sec", 0.0) or 0.0)
        health_sec = float(group.get("health_wait_elapsed_sec", 0.0) or 0.0)
        total_compose += compose_sec
        total_health += health_sec
        groups.append(
            {
                "name": group.get("name", ""),
                "action": group.get("action", ""),
                "container_names": group.get("container_names", []),
                "health_urls": group.get("health_urls", []),
                "checked_at": group.get("checked_at"),
                "ready_at": group.get("ready_at"),
                "compose_up_started_at": group.get("compose_up_started_at"),
                "compose_up_finished_at": group.get("compose_up_finished_at"),
                "compose_up_elapsed_sec": round(compose_sec, 3),
                "health_wait_started_at": group.get("health_wait_started_at"),
                "health_wait_finished_at": group.get("health_wait_finished_at"),
                "health_wait_elapsed_sec": round(health_sec, 3),
                "quick_check_timeout_sec": group.get("quick_check_timeout_sec"),
                "quick_failed_url_count": group.get("quick_failed_url_count"),
                "failed_urls": group.get("failed_urls", []),
            }
        )
    return {
        "family": FAMILY_NAME,
        "scenario_name": scenario_name,
        "workflow_mode": workflow_mode,
        "mode": managed_runtime_policy.get("mode", ""),
        "runtime_services": managed_runtime_policy.get("runtime_services", ""),
        "container_names": managed_runtime_policy.get("container_names", []),
        "health_urls": managed_runtime_policy.get("health_urls", []),
        "singleton_ocr_runtime": managed_runtime_policy.get("singleton_ocr_runtime", {}),
        "groups": groups,
        "compose_up_total_sec": round(total_compose, 3),
        "health_wait_total_sec": round(total_health, 3),
    }


def _build_vram_rows(
    run_dir: Path,
    metrics_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, path in (
        ("run_start_pre_runtime", run_dir / "runtime_snapshot.json"),
        ("run_start_post_runtime_boot", run_dir / "docker_snapshot.json"),
    ):
        if path.is_file():
            rows.append(
                {
                    "point": label,
                    "source": path.name,
                    "ts": None,
                    "gpu": _load_json(path).get("gpu", {}),
                }
            )
    for row in metrics_rows:
        gpu = row.get("gpu")
        if isinstance(gpu, dict):
            rows.append(
                {
                    "point": str(row.get("tag", "") or ""),
                    "source": "metrics.jsonl",
                    "ts": row.get("ts"),
                    "service": row.get("service", ""),
                    "image_path": row.get("image_path", ""),
                    "image_name": row.get("image_name", ""),
                    "gpu": gpu,
                }
            )
    return rows


def _build_quality_summary(
    *,
    scenario_name: str,
    workflow_mode: str,
    summary: dict[str, Any],
    metrics_rows: list[dict[str, Any]],
    page_snapshots: dict[str, Any],
) -> dict[str, Any]:
    detect_counts: dict[str, int] = {}
    for row in metrics_rows:
        if str(row.get("tag", "") or "") == "detect_end":
            detect_counts[str(row.get("image_path", "") or "")] = int(row.get("block_count", 0) or 0)

    pages: list[dict[str, Any]] = []
    snapshot_pages = page_snapshots.get("pages", []) if isinstance(page_snapshots, dict) else []
    if isinstance(snapshot_pages, list):
        for item in snapshot_pages:
            if not isinstance(item, dict):
                continue
            image_path = str(item.get("image_path", "") or "")
            image_name = str(item.get("image_name", "") or Path(image_path).name)
            ocr_quality = item.get("ocr_quality", {}) if isinstance(item.get("ocr_quality"), dict) else {}
            pages.append(
                {
                    "image_path": repo_relative_str(image_path) if image_path else "",
                    "image_name": image_name,
                    "detect_box_count": detect_counts.get(image_path, int(ocr_quality.get("block_count", 0) or 0)),
                    "ocr_block_count": int(ocr_quality.get("block_count", 0) or 0),
                    "ocr_non_empty": int(ocr_quality.get("non_empty", 0) or 0),
                    "ocr_empty": int(ocr_quality.get("empty", 0) or 0),
                    "ocr_single_char_like": int(ocr_quality.get("single_char_like", 0) or 0),
                    "page_failed": bool(item.get("page_failed", False)),
                    "page_failed_reason": str(item.get("page_failed_reason", "") or ""),
                    "is_hard_page": image_name == "p_016.jpg",
                }
            )

    totals = {
        "detect_box_total": sum(int(page.get("detect_box_count", 0) or 0) for page in pages),
        "ocr_block_total": sum(int(page.get("ocr_block_count", 0) or 0) for page in pages),
        "ocr_non_empty_total": sum(int(page.get("ocr_non_empty", 0) or 0) for page in pages),
        "ocr_empty_total": sum(int(page.get("ocr_empty", 0) or 0) for page in pages),
        "ocr_single_char_like_total": sum(int(page.get("ocr_single_char_like", 0) or 0) for page in pages),
    }

    return {
        "family": FAMILY_NAME,
        "scenario_name": scenario_name,
        "workflow_mode": workflow_mode,
        "status": "completed",
        "page_count": len(pages),
        "page_done_count": int(summary.get("page_done_count", 0) or 0),
        "page_failed_count": int(summary.get("page_failed_count", 0) or 0),
        "totals": totals,
        "pages": pages,
        "hard_pages": [page for page in pages if page.get("is_hard_page")],
    }


def _build_timing_summary(
    *,
    scenario_name: str,
    workflow_mode: str,
    summary: dict[str, Any],
    docker_timeline: dict[str, Any],
    normalized_events: list[dict[str, Any]],
    total_wall_sec: float,
) -> dict[str, Any]:
    compose_up_sec = float(docker_timeline.get("compose_up_total_sec", 0.0) or 0.0)
    health_wait_sec = float(docker_timeline.get("health_wait_total_sec", 0.0) or 0.0)
    pure_processing_sec = float(summary.get("elapsed_sec", 0.0) or 0.0)
    timeout_event_count = sum(1 for item in normalized_events if str(item.get("event_name", "")).endswith("_timeout"))
    retry_event_count = sum(1 for item in normalized_events if str(item.get("event_name", "")).endswith("_retry"))
    reuse_hit_count = sum(1 for item in normalized_events if str(item.get("event_name", "")).endswith("_reuse_hit"))
    stage_transition_sec = max(total_wall_sec - compose_up_sec - health_wait_sec - pure_processing_sec, 0.0)
    return {
        "family": FAMILY_NAME,
        "scenario_name": scenario_name,
        "workflow_mode": workflow_mode,
        "status": "completed",
        "image_count": int(summary.get("image_count", 0) or 0),
        "total_elapsed_sec": round(total_wall_sec, 3),
        "pure_processing_sec": round(pure_processing_sec, 3),
        "compose_up_sec": round(compose_up_sec, 3),
        "health_wait_sec": round(health_wait_sec, 3),
        "reuse_hit_saved_sec": None,
        "reuse_hit_count": reuse_hit_count,
        "timeout_retry_penalty_sec": 0.0,
        "timeout_event_count": timeout_event_count,
        "retry_event_count": retry_event_count,
        "warm_up_sec": 0.0,
        "stage_transition_sec": round(stage_transition_sec, 3),
        "stage_stats": summary.get("stage_stats", {}),
    }


def _build_run_report_markdown(
    *,
    scenario_name: str,
    scenario_cfg: dict[str, Any],
    status: str,
    run_dir: Path,
    timing_summary: dict[str, Any],
    quality_summary: dict[str, Any],
    docker_timeline: dict[str, Any],
    blocked_reason: str = "",
) -> str:
    lines = [
        f"# Workflow Split Runtime Scenario Report - {scenario_name}",
        "",
        "핵심 문제 해결 방향은 사용자가 착안했다.",
        "",
        "## 시나리오",
        "",
        f"- scenario_name: `{scenario_name}`",
        f"- label: `{scenario_cfg.get('label', '')}`",
        f"- workflow_mode: `{scenario_cfg.get('workflow_mode', '')}`",
        f"- status: `{status}`",
        f"- run_dir: `{repo_relative_str(run_dir)}`",
        "",
    ]
    if blocked_reason:
        lines.extend(
            [
                "## 현재 상태",
                "",
                f"- blocked_reason: {blocked_reason}",
                "",
                "## 해석",
                "",
                "- 측정 패키지와 문서 계약은 고정되었지만, stage-batched 실험 러너가 아직 없어서 이 시나리오 실행은 보류 상태다.",
                "- Requirement 1 성공/실패 판정에는 아직 사용할 수 없다.",
                "",
            ]
        )
        return "\n".join(lines).rstrip() + "\n"

    lines.extend(
        [
            "## 시간 요약",
            "",
            f"- total_elapsed_sec: `{timing_summary.get('total_elapsed_sec')}`",
            f"- pure_processing_sec: `{timing_summary.get('pure_processing_sec')}`",
            f"- compose_up_sec: `{timing_summary.get('compose_up_sec')}`",
            f"- health_wait_sec: `{timing_summary.get('health_wait_sec')}`",
            f"- stage_transition_sec: `{timing_summary.get('stage_transition_sec')}`",
            f"- page_done_count: `{quality_summary.get('page_done_count')}`",
            f"- page_failed_count: `{quality_summary.get('page_failed_count')}`",
            "",
            "## Docker Breakdown",
            "",
        ]
    )
    groups = docker_timeline.get("groups", [])
    if groups:
        lines.extend(
            [
                "| group | action | compose_up_sec | health_wait_sec | failed_urls |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for group in groups:
            lines.append(
                "| {name} | {action} | {compose} | {health} | {failed} |".format(
                    name=group.get("name", ""),
                    action=group.get("action", ""),
                    compose=group.get("compose_up_elapsed_sec", ""),
                    health=group.get("health_wait_elapsed_sec", ""),
                    failed=", ".join(group.get("failed_urls", [])),
                )
            )
    else:
        lines.append("- managed_runtime_policy.json을 찾지 못해 Docker lifecycle 분해표를 만들지 못했다.")

    lines.extend(
        [
            "",
            "## 품질 요약",
            "",
            f"- detect_box_total: `{quality_summary.get('totals', {}).get('detect_box_total')}`",
            f"- ocr_non_empty_total: `{quality_summary.get('totals', {}).get('ocr_non_empty_total')}`",
            f"- ocr_empty_total: `{quality_summary.get('totals', {}).get('ocr_empty_total')}`",
            f"- ocr_single_char_like_total: `{quality_summary.get('totals', {}).get('ocr_single_char_like_total')}`",
            "",
            "## 비고",
            "",
            "- 이 보고서는 실측 근거를 포트폴리오용 narrative와 분리해 raw 계약 산출물 옆에 고정한다.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _normalize_completed_run(
    *,
    run_dir: Path,
    scenario_name: str,
    scenario_cfg: dict[str, Any],
    total_wall_sec: float,
) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.jsonl"
    summary_path = run_dir / "summary.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Expected metrics.jsonl at {metrics_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Expected summary.json at {summary_path}")

    summary = summarize_metrics(metrics_path)
    raw_summary = _load_json(summary_path)
    summary.update(
        {
            "image_count": raw_summary.get("image_count", summary.get("page_done_count", 0)),
            "image_paths": raw_summary.get("image_paths", []),
        }
    )
    metrics_rows = load_metrics(metrics_path)
    page_snapshots = _load_json_if_exists(run_dir / "page_snapshots.json")
    managed_runtime_policy = _load_json_if_exists(run_dir / "runtime" / "managed_runtime_policy.json")
    normalized_events = _normalize_metrics_events(
        metrics_rows,
        scenario_name=scenario_name,
        workflow_mode=scenario_cfg["workflow_mode"],
    )
    docker_timeline = _build_docker_timeline(
        scenario_name=scenario_name,
        workflow_mode=scenario_cfg["workflow_mode"],
        managed_runtime_policy=managed_runtime_policy,
    )
    timing_summary = _build_timing_summary(
        scenario_name=scenario_name,
        workflow_mode=scenario_cfg["workflow_mode"],
        summary=summary,
        docker_timeline=docker_timeline,
        normalized_events=normalized_events,
        total_wall_sec=total_wall_sec,
    )
    quality_summary = _build_quality_summary(
        scenario_name=scenario_name,
        workflow_mode=scenario_cfg["workflow_mode"],
        summary=summary,
        metrics_rows=metrics_rows,
        page_snapshots=page_snapshots,
    )
    vram_rows = _build_vram_rows(run_dir, metrics_rows)

    request_payload = _load_json_if_exists(run_dir / "benchmark_request.json")
    request_payload.update(
        {
            "family": FAMILY_NAME,
            "scenario_name": scenario_name,
            "workflow_mode": scenario_cfg["workflow_mode"],
            "status": "completed",
            "corpus_root": repo_relative_str(DEFAULT_SAMPLE_DIR),
        }
    )
    write_json(run_dir / "benchmark_request.json", request_payload)
    write_json(run_dir / "timing_summary.json", timing_summary)
    write_json(run_dir / "quality_summary.json", quality_summary)
    write_json(run_dir / "docker_timeline.json", docker_timeline)
    _write_jsonl(run_dir / "events.jsonl", normalized_events)
    _write_jsonl(run_dir / "vram_snapshots.jsonl", vram_rows)
    (run_dir / "report.md").write_text(
        _build_run_report_markdown(
            scenario_name=scenario_name,
            scenario_cfg=scenario_cfg,
            status="completed",
            run_dir=run_dir,
            timing_summary=timing_summary,
            quality_summary=quality_summary,
            docker_timeline=docker_timeline,
        ),
        encoding="utf-8",
    )
    return {
        "scenario_name": scenario_name,
        "status": "completed",
        "workflow_mode": scenario_cfg["workflow_mode"],
        "run_dir": repo_relative_str(run_dir),
        "events_path": repo_relative_str(run_dir / "events.jsonl"),
        "timing_summary_path": repo_relative_str(run_dir / "timing_summary.json"),
        "quality_summary_path": repo_relative_str(run_dir / "quality_summary.json"),
        "docker_timeline_path": repo_relative_str(run_dir / "docker_timeline.json"),
        "report_path": repo_relative_str(run_dir / "report.md"),
        "total_elapsed_sec": timing_summary.get("total_elapsed_sec"),
        "page_done_count": quality_summary.get("page_done_count"),
        "page_failed_count": quality_summary.get("page_failed_count"),
    }


def _write_blocked_run(
    *,
    scenario_name: str,
    scenario_cfg: dict[str, Any],
    sample_paths: list[Path],
) -> dict[str, Any]:
    run_dir = create_run_dir(scenario_name, root=family_output_root())
    corpus_dir = _stage_curated_corpus(run_dir, sample_paths)
    request_payload = {
        "family": FAMILY_NAME,
        "scenario_name": scenario_name,
        "workflow_mode": scenario_cfg["workflow_mode"],
        "status": "blocked",
        "preset": scenario_cfg["preset"],
        "runtime_mode": scenario_cfg["runtime_mode"],
        "runtime_services": scenario_cfg["runtime_services"],
        "source_lang": DEFAULT_SOURCE_LANG,
        "target_lang": DEFAULT_TARGET_LANG,
        "corpus_root": repo_relative_str(DEFAULT_SAMPLE_DIR),
        "selected_paths": [repo_relative_str(path) for path in sample_paths],
        "staged_paths": [repo_relative_str(corpus_dir / path.name) for path in sample_paths],
        "blocked_reason": scenario_cfg.get("blocked_reason", ""),
    }
    write_json(run_dir / "benchmark_request.json", request_payload)
    events = [
        _base_event_payload(
            scenario_name=scenario_name,
            workflow_mode=scenario_cfg["workflow_mode"],
            event_name="scenario_blocked",
            status="blocked",
            detail=scenario_cfg.get("blocked_reason", ""),
            extra={
                "total_images": len(sample_paths),
            },
        )
    ]
    timing_summary = {
        "family": FAMILY_NAME,
        "scenario_name": scenario_name,
        "workflow_mode": scenario_cfg["workflow_mode"],
        "status": "blocked",
        "blocked_reason": scenario_cfg.get("blocked_reason", ""),
        "image_count": len(sample_paths),
        "total_elapsed_sec": None,
        "pure_processing_sec": None,
        "compose_up_sec": None,
        "health_wait_sec": None,
        "reuse_hit_saved_sec": None,
        "timeout_retry_penalty_sec": None,
        "warm_up_sec": None,
        "stage_transition_sec": None,
    }
    quality_summary = {
        "family": FAMILY_NAME,
        "scenario_name": scenario_name,
        "workflow_mode": scenario_cfg["workflow_mode"],
        "status": "blocked",
        "blocked_reason": scenario_cfg.get("blocked_reason", ""),
        "page_count": len(sample_paths),
        "page_done_count": 0,
        "page_failed_count": 0,
        "pages": [
            {
                "image_path": repo_relative_str(path),
                "image_name": path.name,
                "is_hard_page": path.name == "p_016.jpg",
            }
            for path in sample_paths
        ],
    }
    docker_timeline = {
        "family": FAMILY_NAME,
        "scenario_name": scenario_name,
        "workflow_mode": scenario_cfg["workflow_mode"],
        "status": "blocked",
        "blocked_reason": scenario_cfg.get("blocked_reason", ""),
        "groups": [],
    }
    vram_rows = [
        {
            "point": "scenario_blocked",
            "source": "workflow_split_runtime_benchmark.py",
            "detail": scenario_cfg.get("blocked_reason", ""),
        }
    ]
    write_json(run_dir / "timing_summary.json", timing_summary)
    write_json(run_dir / "quality_summary.json", quality_summary)
    write_json(run_dir / "docker_timeline.json", docker_timeline)
    _write_jsonl(run_dir / "events.jsonl", events)
    _write_jsonl(run_dir / "vram_snapshots.jsonl", vram_rows)
    (run_dir / "report.md").write_text(
        _build_run_report_markdown(
            scenario_name=scenario_name,
            scenario_cfg=scenario_cfg,
            status="blocked",
            run_dir=run_dir,
            timing_summary=timing_summary,
            quality_summary=quality_summary,
            docker_timeline=docker_timeline,
            blocked_reason=scenario_cfg.get("blocked_reason", ""),
        ),
        encoding="utf-8",
    )
    return {
        "scenario_name": scenario_name,
        "status": "blocked",
        "workflow_mode": scenario_cfg["workflow_mode"],
        "run_dir": repo_relative_str(run_dir),
        "events_path": repo_relative_str(run_dir / "events.jsonl"),
        "timing_summary_path": repo_relative_str(run_dir / "timing_summary.json"),
        "quality_summary_path": repo_relative_str(run_dir / "quality_summary.json"),
        "docker_timeline_path": repo_relative_str(run_dir / "docker_timeline.json"),
        "report_path": repo_relative_str(run_dir / "report.md"),
        "blocked_reason": scenario_cfg.get("blocked_reason", ""),
    }


def _run_legacy_baseline(
    *,
    scenario_name: str,
    scenario_cfg: dict[str, Any],
    sample_paths: list[Path],
    source_lang: str,
    target_lang: str,
) -> dict[str, Any]:
    run_dir = create_run_dir(scenario_name, root=family_output_root())
    corpus_dir = _stage_curated_corpus(run_dir, sample_paths)
    env = dict(os.environ)
    env.setdefault("CT_BENCH_OUTPUT_ROOT", str(family_output_root()))
    cmd = [
        sys.executable,
        "-u",
        str(SCRIPT_DIR / "benchmark_pipeline.py"),
        "--preset",
        scenario_cfg["preset"],
        "--mode",
        "batch",
        "--repeat",
        "1",
        "--runtime-mode",
        scenario_cfg["runtime_mode"],
        "--runtime-services",
        scenario_cfg["runtime_services"],
        "--sample-dir",
        str(corpus_dir),
        "--sample-count",
        str(len(sample_paths)),
        "--source-lang",
        source_lang,
        "--target-lang",
        target_lang,
        "--stage-ceiling",
        scenario_cfg["stage_ceiling"],
        "--export-page-snapshots",
        "--output-dir",
        str(run_dir),
    ]
    _log(
        "scenario 실행: {scenario} preset={preset} images={count} run_dir={run_dir}".format(
            scenario=scenario_name,
            preset=scenario_cfg["preset"],
            count=len(sample_paths),
            run_dir=run_dir,
        )
    )
    started = time.perf_counter()
    completed = _run_command_streaming(
        cmd=cmd,
        cwd=ROOT,
        env=env,
        step_name=scenario_name,
    )
    total_wall_sec = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"{scenario_name} failed with exit code {completed.returncode}.\n"
            f"stdout:\n{(completed.stdout or '').strip()}"
        )
    return _normalize_completed_run(
        run_dir=run_dir,
        scenario_name=scenario_name,
        scenario_cfg=scenario_cfg,
        total_wall_sec=total_wall_sec,
    )


def _run_stage_batched_candidate(
    *,
    scenario_name: str,
    scenario_cfg: dict[str, Any],
    sample_paths: list[Path],
    source_lang: str,
    target_lang: str,
) -> dict[str, Any]:
    run_dir = create_run_dir(scenario_name, root=family_output_root())
    corpus_dir = _stage_curated_corpus(run_dir, sample_paths)
    env = dict(os.environ)
    env.setdefault("CT_BENCH_OUTPUT_ROOT", str(family_output_root()))
    cmd = [
        sys.executable,
        "-u",
        str(SCRIPT_DIR / "benchmark_stage_batched_pipeline.py"),
        "--preset",
        scenario_cfg["preset"],
        "--sample-dir",
        str(corpus_dir),
        "--sample-count",
        str(len(sample_paths)),
        "--source-lang",
        source_lang,
        "--target-lang",
        target_lang,
        "--output-dir",
        str(run_dir),
        "--resident-ocr-mode",
        str(scenario_cfg.get("resident_ocr_mode", "single")),
    ]
    _log(
        "scenario 실행: {scenario} preset={preset} resident_ocr_mode={resident} images={count} run_dir={run_dir}".format(
            scenario=scenario_name,
            preset=scenario_cfg["preset"],
            resident=scenario_cfg.get("resident_ocr_mode", "single"),
            count=len(sample_paths),
            run_dir=run_dir,
        )
    )
    started = time.perf_counter()
    completed = _run_command_streaming(
        cmd=cmd,
        cwd=ROOT,
        env=env,
        step_name=scenario_name,
    )
    total_wall_sec = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"{scenario_name} failed with exit code {completed.returncode}.\n"
            f"stdout:\n{(completed.stdout or '').strip()}"
        )
    return _normalize_completed_run(
        run_dir=run_dir,
        scenario_name=scenario_name,
        scenario_cfg=scenario_cfg,
        total_wall_sec=total_wall_sec,
    )


def _write_suite_record(
    *,
    records: list[dict[str, Any]],
    sample_paths: list[Path],
    smoke: bool,
    source_lang: str,
    target_lang: str,
) -> Path:
    suite_payload = {
        "family": FAMILY_NAME,
        "generated_at": time.time(),
        "sample_root": repo_relative_str(DEFAULT_SAMPLE_DIR),
        "selected_files": [path.name for path in sample_paths],
        "source_lang": source_lang,
        "target_lang": target_lang,
        "smoke": smoke,
        "scenario_records": records,
        "completed_scenarios": [item["scenario_name"] for item in records if item.get("status") == "completed"],
        "blocked_scenarios": [item["scenario_name"] for item in records if item.get("status") == "blocked"],
    }
    path = _latest_suite_record_path()
    write_json(path, suite_payload)
    return path


def _generate_family_report() -> None:
    subprocess.run(
        [sys.executable, "-u", str(SCRIPT_DIR / "generate_workflow_split_runtime_report.py")],
        cwd=str(ROOT),
        check=True,
    )


def _run_suite(args: argparse.Namespace) -> int:
    sample_paths = _resolve_sample_paths(args.sample_dir, smoke=args.smoke)
    scenario_names = _resolve_scenarios(args.scenario)
    records: list[dict[str, Any]] = []
    for scenario_name in scenario_names:
        scenario_cfg = SCENARIOS[scenario_name]
        if scenario_cfg.get("runnable", False):
            try:
                runner_kind = str(scenario_cfg.get("runner_kind", "legacy") or "legacy")
                if runner_kind == "stage_batched":
                    record = _run_stage_batched_candidate(
                        scenario_name=scenario_name,
                        scenario_cfg=scenario_cfg,
                        sample_paths=sample_paths,
                        source_lang=args.source_lang,
                        target_lang=args.target_lang,
                    )
                else:
                    record = _run_legacy_baseline(
                        scenario_name=scenario_name,
                        scenario_cfg=scenario_cfg,
                        sample_paths=sample_paths,
                        source_lang=args.source_lang,
                        target_lang=args.target_lang,
                    )
            except Exception as exc:
                _log(f"{scenario_name} 실행 실패: {exc}")
                raise
        else:
            record = _write_blocked_run(
                scenario_name=scenario_name,
                scenario_cfg=scenario_cfg,
                sample_paths=sample_paths,
            )
        records.append(record)
    suite_path = _write_suite_record(
        records=records,
        sample_paths=sample_paths,
        smoke=args.smoke,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
    )
    _log(f"latest suite record 갱신: {suite_path}")
    _generate_family_report()
    _log("family report 갱신 완료")
    return 0


def _summary() -> int:
    _generate_family_report()
    path = _latest_suite_record_path()
    if path.is_file():
        payload = _load_json(path)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("No workflow-split-runtime suite record was found.", file=sys.stderr)
        return 1
    return 0


def _open_dir() -> int:
    print(str(family_output_root()))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Requirement 1 workflow split runtime benchmark family.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the workflow split runtime suite.")
    run_parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Scenario name. Repeat for multiple scenarios, or pass all.",
    )
    run_parser.add_argument(
        "--sample-dir",
        default=str(DEFAULT_SAMPLE_DIR),
        help="Curated Requirement 1 corpus root. Defaults to Sample/japan.",
    )
    run_parser.add_argument("--source-lang", default=DEFAULT_SOURCE_LANG)
    run_parser.add_argument("--target-lang", default=DEFAULT_TARGET_LANG)
    run_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the 2-page smoke corpus (094.png + p_016.jpg).",
    )

    subparsers.add_parser("summary", help="Refresh docs/report from the latest suite record.")
    subparsers.add_parser("open", help="Print the family output directory.")

    args = parser.parse_args()
    command = args.command or "run"
    if command == "run":
        if not args.scenario:
            args.scenario = ["all"]
        return _run_suite(args)
    if command == "summary":
        return _summary()
    if command == "open":
        return _open_dir()
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
