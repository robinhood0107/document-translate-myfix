#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_common import create_run_dir, repo_relative_str, run_command, write_json

FAMILY_NAME = "gemma_iq4nl_japan"
FAMILY_PROFILE = "gemma-iq4nl-japan-fullgpu"
FAMILY_OUTPUT_ROOT_NAME = "gemma_iq4nl_japan"
DOCS_SLUG = "gemma-iq4nl-japan"
REPORT_MANIFEST_NAME = "gemma_iq4nl_japan_report_manifest.yaml"
LAST_SUITE_RECORD = "last_gemma_iq4nl_japan_fullgpu_suite.json"
BASE_PRESET_PATH = ROOT / "benchmarks" / "gemma_iq4nl_japan" / "presets" / "gemma-iq4nl-japan-fullgpu-base.json"
SMOKE_FILE = "094.png"
SOURCE_LANG = "Japanese"
TARGET_LANG = "Korean"
SAMPLE_SUBDIR = "japan"
SAMPLE_COUNT = 22
REQUESTED_TEMPERATURE = 0.7
TEMPERATURE_LADDER = [0.7, 0.6, 0.5, 0.4]
STATE_FILE_NAME = "suite_state.json"
HEALTH_WAIT_TIMEOUT_SEC = 900
HEARTBEAT_POLL_SEC = 60
STALL_TIMEOUT_SEC = 600
INFRA_RETRY_DELAY_SEC = 15
CONFIRM_REPEAT_COUNT = 1
ATTEMPT_DIR_PATTERN = re.compile(r"^attempt(?P<attempt>\d+)_t(?P<temp>\d+)_infra(?P<infra>\d+)$")

RETRYABLE_INFRA_MARKERS = (
    "timed out waiting for",
    "connection refused",
    "remote end closed connection without response",
    "remotedisconnected",
    "failed to establish a new connection",
    "connection aborted",
    "removal of container",
    "is already in progress",
    "service unavailable",
    "context deadline exceeded",
    "no route to host",
    "health-check failed",
    "health check failed",
    "health-check timeout",
    "health check timeout",
    "container is unhealthy",
    "temporary failure",
    "unexpected eof",
)

OOM_MARKERS = (
    "cuda out of memory",
    "out of memory",
    "cublas_status_alloc_failed",
    "failed to allocate memory",
    "memory allocation",
    "insufficient memory",
    "not enough memory",
    "model load failed",
)

FIXED_PIPELINE = {
    "translator": "Custom Local Server(Gemma)",
    "ocr": "PaddleOCR VL",
    "detector": "RT-DETR-v2",
    "inpainter": "lama_large_512px",
    "mask_refiner": "ctd",
    "use_gpu": True,
    "ocr_front_device": "cuda",
    "detector_device": "cuda",
    "ctd_device": "cuda",
    "inpainter_device": "cuda",
    "gemma_model": "gemma-4-26B-IQ4_NL.gguf",
    "requested_temperature": REQUESTED_TEMPERATURE,
}


def _log(message: str) -> None:
    print(f"[gemma-iq4nl] {message}", flush=True)


def family_output_root() -> Path:
    env_override = os.getenv("CT_BENCH_OUTPUT_ROOT", "").strip()
    if env_override:
        root = Path(env_override).expanduser()
        if root.name != FAMILY_OUTPUT_ROOT_NAME:
            root = root / FAMILY_OUTPUT_ROOT_NAME
    else:
        root = ROOT / "banchmark_result_log" / FAMILY_OUTPUT_ROOT_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_repo_relative_path(path_value: str | Path) -> Path:
    text = str(path_value or "").strip()
    if text.startswith("./"):
        return ROOT / text[2:]
    return Path(text)


def _docs_assets_latest_root() -> Path:
    return ROOT / "docs" / "assets" / "benchmarking" / DOCS_SLUG / "latest"


def _docs_assets_history_root() -> Path:
    return ROOT / "docs" / "assets" / "benchmarking" / DOCS_SLUG / "history"


def _docs_report_path() -> Path:
    return ROOT / "docs" / "banchmark_report" / f"{DOCS_SLUG}-report-ko.md"


def _state_path(suite_dir: Path) -> Path:
    return suite_dir / STATE_FILE_NAME


def _manifest_path(suite_dir: Path) -> Path:
    return suite_dir / REPORT_MANIFEST_NAME


def _now_str() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = value
    return dst


def _current_git_sha(ref: str = "HEAD") -> str:
    completed = run_command(["git", "rev-parse", ref], cwd=ROOT, check=False)
    return (completed.stdout or "").strip()


def _bootstrap_state(*, suite_dir: Path, sample_root: Path) -> dict[str, Any]:
    state_path = _state_path(suite_dir)
    if state_path.is_file():
        state = _load_json(state_path)
        state.setdefault("status", "idle")
        state.setdefault("current_stage", "")
        state.setdefault("current_candidate", "")
        state.setdefault("last_heartbeat_at", "")
        state.setdefault("infra_retry_count", 0)
        state.setdefault("last_failure_kind", "")
        state.setdefault("last_failure_reason", "")
        return state
    sample_dir = sample_root / SAMPLE_SUBDIR
    state = {
        "family_name": FAMILY_NAME,
        "suite_profile": FAMILY_PROFILE,
        "generated_at": _now_str(),
        "suite_dir": str(suite_dir),
        "results_root": repo_relative_str(family_output_root()),
        "git_sha": _current_git_sha(),
        "sample_root": str(sample_root),
        "sample_dir": str(sample_dir),
        "sample_count": SAMPLE_COUNT,
        "smoke_file": SMOKE_FILE,
        "fixed_pipeline": FIXED_PIPELINE,
        "baseline_run_dir": "",
        "stages": {},
        "winner": {},
        "runner_up": {},
        "status": "idle",
        "current_stage": "",
        "current_candidate": "",
        "last_heartbeat_at": _now_str(),
        "infra_retry_count": 0,
        "last_failure_kind": "",
        "last_failure_reason": "",
    }
    write_json(state_path, state)
    return state


def _save_state(state: dict[str, Any]) -> None:
    state["last_heartbeat_at"] = _now_str()
    write_json(_state_path(Path(state["suite_dir"])), state)


def _touch_state(state: dict[str, Any], **updates: Any) -> None:
    state.update(updates)
    _save_state(state)


def _load_base_preset() -> dict[str, Any]:
    return _load_json(BASE_PRESET_PATH)


def _materialize_preset(suite_dir: Path, slug: str, updates: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    materialized_dir = suite_dir / "_materialized_presets"
    materialized_dir.mkdir(parents=True, exist_ok=True)
    payload = copy.deepcopy(_load_base_preset())
    _deep_merge(payload, updates)
    payload["name"] = slug
    payload["description"] = f"Materialized preset for {slug}"
    preset_path = materialized_dir / f"{slug}.json"
    preset_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return preset_path, payload


def _candidate_output_dir(
    suite_dir: Path,
    stage_name: str,
    candidate_slug: str,
    attempt_index: int,
    temperature: float,
    infra_attempt_index: int = 1,
) -> Path:
    temp_slug = str(temperature).replace(".", "")
    return suite_dir / stage_name / candidate_slug / f"attempt{attempt_index:02d}_t{temp_slug}_infra{infra_attempt_index:02d}"


def _temperature_from_slug(temp_slug: str) -> float:
    for value in TEMPERATURE_LADDER:
        if _temperature_slug(value) == temp_slug:
            return float(value)
    digits = int(temp_slug)
    scale = 10 ** max(len(temp_slug) - 1, 0)
    return float(digits / scale)


def _parse_attempt_dir_name(name: str) -> tuple[int, float, int] | None:
    match = ATTEMPT_DIR_PATTERN.match(name)
    if not match:
        return None
    return (
        int(match.group("attempt")),
        _temperature_from_slug(match.group("temp")),
        int(match.group("infra")),
    )


def _existing_attempt_entries(
    suite_dir: Path,
    stage_name: str,
    candidate_slug: str,
    *,
    temperature: float,
    attempt_index: int,
) -> list[dict[str, Any]]:
    candidate_dir = suite_dir / stage_name / candidate_slug
    if not candidate_dir.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for run_dir in candidate_dir.iterdir():
        if not run_dir.is_dir():
            continue
        parsed = _parse_attempt_dir_name(run_dir.name)
        if parsed is None:
            continue
        parsed_attempt_index, parsed_temperature, infra_attempt_index = parsed
        if parsed_attempt_index != attempt_index:
            continue
        if abs(parsed_temperature - temperature) > 1e-9:
            continue
        entries.append(
            {
                "run_dir": run_dir,
                "attempt_index": parsed_attempt_index,
                "temperature": parsed_temperature,
                "infra_attempt_index": infra_attempt_index,
            }
        )
    entries.sort(key=lambda item: int(item["infra_attempt_index"]))
    return entries


def _run_pipeline_once(*, preset_path: Path, output_dir: Path, sample_dir: Path, sample_count: int, mode: str) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "benchmark_pipeline.py"),
        "--preset",
        str(preset_path),
        "--mode",
        mode,
        "--repeat",
        "1",
        "--runtime-mode",
        "managed",
        "--sample-dir",
        str(sample_dir),
        "--sample-count",
        str(sample_count),
        "--output-dir",
        str(output_dir),
        "--source-lang",
        SOURCE_LANG,
        "--target-lang",
        TARGET_LANG,
        "--export-page-snapshots",
        "--clear-app-caches",
    ]
    env = dict(__import__('os').environ)
    env["CT_BENCH_OUTPUT_ROOT"] = str(family_output_root())
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command_stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
    (output_dir / "command_stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
    return completed


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _collect_attempt_text(run_dir: Path) -> str:
    chunks = [
        _read_text(run_dir / "command_stdout.txt"),
        _read_text(run_dir / "command_stderr.txt"),
    ]
    docker_log_dir = run_dir / "docker_logs"
    if docker_log_dir.is_dir():
        for log_path in sorted(docker_log_dir.glob("*.log")):
            chunks.append(_read_text(log_path))
    return "\n".join(part for part in chunks if part)


def _normalized_text_blob(text: str) -> str:
    return text.lower().strip()


def _contains_marker(text: str, markers: tuple[str, ...]) -> bool:
    lowered = _normalized_text_blob(text)
    return any(marker in lowered for marker in markers)


def _candidate_status(record: dict[str, Any], *, recovered_by_oom: bool = False, original_candidate: bool = True) -> str:
    if record.get("passed_hard_gate"):
        return "recovered_by_oom_rescue" if recovered_by_oom else "passed"
    failure_kind = str(record.get("last_failure_kind", "") or "")
    if failure_kind == "oom_unfit":
        return "rejected_unfit_original" if original_candidate else "rejected_unfit"
    return "failed_quality"


def _classify_runtime_failure(record: dict[str, Any]) -> tuple[str, str]:
    summary = record.get("summary", {}) if isinstance(record.get("summary"), dict) else {}
    page_done_count = int(summary.get("page_done_count", 0) or 0)
    page_failed_count = int(summary.get("page_failed_count", 0) or 0)
    if page_done_count > 0 and page_failed_count == 0:
        return "completed", ""
    text_blob = _collect_attempt_text(Path(record["run_dir_abs"]))
    combined = "\n".join(
        [
            text_blob,
            str(summary.get("last_failure_reason", "") or ""),
            str(summary.get("error", "") or ""),
        ]
    )
    lowered = _normalized_text_blob(combined)
    if _contains_marker(lowered, OOM_MARKERS):
        return "oom_unfit", "OOM or VRAM allocation failure detected"
    if _contains_marker(lowered, RETRYABLE_INFRA_MARKERS):
        return "infra_retry", "health/bootstrap/runtime connectivity issue detected"
    if _returncode_requires_failure(record, summary) and page_done_count == 0:
        return "infra_retry", f"benchmark pipeline exited before completing any page: {record.get('returncode')}"
    return "completed", ""


def _update_record_outcome(
    record: dict[str, Any],
    *,
    failure_kind: str,
    failure_reason: str,
    recovered_by_oom: bool,
    original_candidate: bool,
) -> dict[str, Any]:
    record["last_failure_kind"] = failure_kind
    record["last_failure_reason"] = failure_reason
    record["hard_fail_issues"] = _hard_fail_issues(record)
    record["passed_hard_gate"] = not record["hard_fail_issues"]
    record["candidate_status"] = _candidate_status(
        record,
        recovered_by_oom=recovered_by_oom,
        original_candidate=original_candidate,
    )
    return record


def _next_lower_values(current: float | int, ladder: list[float | int]) -> list[float | int]:
    return [value for value in ladder if value < current]


def _oom_rescue_variants(base_updates: dict[str, Any]) -> list[dict[str, Any]]:
    updates = copy.deepcopy(base_updates)
    gemma = updates.setdefault("gemma", {})
    ocr_runtime = updates.setdefault("ocr_runtime", {})
    variants: list[dict[str, Any]] = []

    def _snapshot(level: int, changes: dict[str, Any]) -> None:
        variants.append(
            {
                "level": level,
                "changes": copy.deepcopy(changes),
                "updates": copy.deepcopy(updates),
            }
        )

    level = 1
    current_ocr_vram = float(ocr_runtime.get("gpu_memory_utilization", 0.72) or 0.72)
    for value in _next_lower_values(current_ocr_vram, [0.80, 0.76, 0.72, 0.68, 0.64, 0.60]):
        ocr_runtime["gpu_memory_utilization"] = float(value)
        _snapshot(level, {"ocr_runtime.gpu_memory_utilization": float(value)})
        level += 1

    current_ngl = int(gemma.get("n_gpu_layers", 18) or 18)
    for value in _next_lower_values(current_ngl, [23, 22, 20, 18, 16, 14, 12]):
        gemma["n_gpu_layers"] = int(value)
        _snapshot(level, {"gemma.n_gpu_layers": int(value)})
        level += 1

    current_ctx = int(gemma.get("context_size", 4096) or 4096)
    for value in _next_lower_values(current_ctx, [4096, 3072, 2560]):
        gemma["context_size"] = int(value)
        _snapshot(level, {"gemma.context_size": int(value)})
        level += 1

    current_max_tokens = int(gemma.get("max_completion_tokens", 512) or 512)
    for value in _next_lower_values(current_max_tokens, [512, 384, 320]):
        gemma["max_completion_tokens"] = int(value)
        _snapshot(level, {"gemma.max_completion_tokens": int(value)})
        level += 1

    current_chunk_size = int(gemma.get("chunk_size", 4) or 4)
    for value in _next_lower_values(current_chunk_size, [6, 5, 4, 3]):
        gemma["chunk_size"] = int(value)
        _snapshot(level, {"gemma.chunk_size": int(value)})
        level += 1

    return variants


def _normalize_device_label(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    if raw.startswith("gpu"):
        return "cuda"
    return raw


def _is_gpu_device(value: str | None) -> bool:
    normalized = _normalize_device_label(value)
    return normalized.startswith("cuda")


def _extract_layout_front_device(run_dir: Path) -> str:
    compose_path = run_dir / "runtime" / "ocr" / "docker-compose.yaml"
    if not compose_path.is_file():
        return ""
    try:
        payload = yaml.safe_load(compose_path.read_text(encoding="utf-8")) or {}
        command = str(((payload.get("services") or {}).get("paddleocr-layout") or {}).get("command") or "")
    except Exception:
        return ""
    marker = "--device "
    if marker not in command:
        return ""
    after = command.split(marker, 1)[1].strip()
    return _normalize_device_label(after.split()[0])


def _query_gemma_model_ids() -> list[str]:
    url = "http://127.0.0.1:18080/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []
    data = payload.get("data") if isinstance(payload, dict) else []
    if not isinstance(data, list):
        return []
    result: list[str] = []
    for item in data:
        if isinstance(item, dict):
            model_id = str(item.get("id", "") or "").strip()
            if model_id:
                result.append(model_id)
    return result


def _load_run_payload(run_dir: Path) -> dict[str, Any]:
    summary = _load_json(run_dir / "summary.json") if (run_dir / "summary.json").is_file() else {
        "page_failed_count": 1,
        "elapsed_sec": None,
        "translate_median_sec": None,
    }
    page_snapshots = _load_json(run_dir / "page_snapshots.json") if (run_dir / "page_snapshots.json").is_file() else {"pages": []}
    preset_resolved = _load_json(run_dir / "preset_resolved.json") if (run_dir / "preset_resolved.json").is_file() else {}
    first_page = {}
    pages = page_snapshots.get("pages") if isinstance(page_snapshots.get("pages"), list) else []
    if pages:
        first_page = pages[0] if isinstance(pages[0], dict) else {}
    runtime = first_page.get("runtime") if isinstance(first_page.get("runtime"), dict) else {}
    return {
        "summary": summary,
        "preset_resolved": preset_resolved,
        "page_snapshots_path": repo_relative_str(run_dir / "page_snapshots.json"),
        "runtime_observation": {
            "detector_backend": str(runtime.get("detector_engine", "") or ""),
            "detector_device": str(runtime.get("detector_device", "") or ""),
            "ocr_front_device": _extract_layout_front_device(run_dir),
            "mask_refiner": str(runtime.get("mask_refiner", "") or ""),
            "ctd_device": str(runtime.get("ctd_device", "") or ""),
            "inpainter_key": str(runtime.get("inpainter_key", runtime.get("inpainter", "")) or ""),
            "inpainter_device": str(runtime.get("inpainter_device", "") or ""),
            "inpaint_size": runtime.get("inpaint_size"),
            "precision": runtime.get("precision"),
            "gemma_loaded_model_ids": _query_gemma_model_ids(),
        },
    }


def _rescue_trigger_issues(summary: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if int(summary.get("page_failed_count", 0) or 0) > 0:
        issues.append("page_failed_count")
    for key in (
        "gemma_json_retry_count",
        "gemma_truncated_count",
        "gemma_empty_content_count",
        "gemma_missing_key_count",
        "gemma_reasoning_without_final_count",
        "gemma_schema_validation_fail_count",
    ):
        if int(summary.get(key, 0) or 0) > 0:
            issues.append(key)
    return issues


def _returncode_requires_failure(record: dict[str, Any], summary: dict[str, Any]) -> bool:
    returncode = int(record.get("returncode", 0) or 0)
    if returncode == 0:
        return False
    tag_counts = summary.get("tag_counts", {}) if isinstance(summary.get("tag_counts"), dict) else {}
    completed_pipeline = (
        int(summary.get("page_failed_count", 0) or 0) == 0
        and int(summary.get("page_done_count", 0) or 0) > 0
        and int(tag_counts.get("benchmark_run_finished", 0) or 0) > 0
    )
    if completed_pipeline and returncode in {-1, 4294967295}:
        return False
    return True


def _hard_fail_issues(record: dict[str, Any]) -> list[str]:
    summary = record.get("summary", {}) if isinstance(record.get("summary"), dict) else {}
    issues: list[str] = []
    if _returncode_requires_failure(record, summary):
        issues.append(f"benchmark_pipeline_exit={record.get('returncode')}")
    if int(summary.get("page_failed_count", 0) or 0) > 0:
        issues.append("page_failed_count")
    for key in (
        "gemma_truncated_count",
        "gemma_empty_content_count",
        "gemma_missing_key_count",
        "gemma_schema_validation_fail_count",
    ):
        if int(summary.get(key, 0) or 0) > 0:
            issues.append(key)
    translation_audit = record.get("translation_audit")
    if isinstance(translation_audit, dict) and translation_audit.get("passed") is False:
        issues.append("translation_audit_failed")
    page_snapshot_audit = record.get("page_snapshot_audit")
    if isinstance(page_snapshot_audit, dict) and page_snapshot_audit.get("passed") is False:
        issues.append("page_snapshot_audit_failed")
    return issues


def _run_audit(script_name: str, *, baseline_run_dir: Path, candidate_run_dir: Path, output_path: Path, extra_args: list[str] | None = None) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / script_name),
        "--baseline-run-dir",
        str(baseline_run_dir),
        "--candidate-run-dir",
        str(candidate_run_dir),
        "--output",
        str(output_path),
    ]
    if extra_args:
        cmd.extend(extra_args)
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (output_path.parent / f"{output_path.stem}_stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
    (output_path.parent / f"{output_path.stem}_stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
    payload = _load_json(output_path) if output_path.is_file() else {
        "passed": False,
        "issues": [f"audit runner failed: {script_name} exit={completed.returncode}"],
    }
    payload["runner_exit_code"] = completed.returncode
    return payload


def _attach_audits(record: dict[str, Any], *, baseline_run_dir: Path) -> dict[str, Any]:
    candidate_run_dir = Path(record["run_dir_abs"])
    if baseline_run_dir.resolve() == candidate_run_dir.resolve():
        record["translation_audit"] = {"passed": True, "status": "BASELINE"}
        record["page_snapshot_audit"] = {"passed": True, "status": "BASELINE"}
        return record
    audits_dir = candidate_run_dir / "audits"
    record["translation_audit"] = _run_audit(
        "compare_translation_exports.py",
        baseline_run_dir=baseline_run_dir,
        candidate_run_dir=candidate_run_dir,
        output_path=audits_dir / "translation_audit.json",
        extra_args=["--sample-dir", str(ROOT / "Sample" / SAMPLE_SUBDIR), "--sample-count", str(SAMPLE_COUNT)],
    )
    record["page_snapshot_audit"] = _run_audit(
        "compare_page_snapshots.py",
        baseline_run_dir=baseline_run_dir,
        candidate_run_dir=candidate_run_dir,
        output_path=audits_dir / "page_snapshot_audit.json",
    )
    return record


def _temperature_slug(value: float) -> str:
    return str(value).replace(".", "")


def _build_record(
    *,
    stage_name: str,
    candidate_slug: str,
    label: str,
    tuning: dict[str, Any],
    requested_temperature: float,
    effective_temperature: float,
    attempt_index: int,
    infra_attempt_index: int,
    returncode: int,
    run_dir: Path,
    derived_from: str = "",
    oom_rescue_level: int = 0,
    oom_rescue_changes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _load_run_payload(run_dir)
    record = {
        "stage": stage_name,
        "candidate_slug": candidate_slug,
        "label": label,
        "tuning": tuning,
        "requested_temperature": requested_temperature,
        "effective_temperature": effective_temperature,
        "rescue_attempt_count": attempt_index - 1,
        "temperature_attempt_count": attempt_index,
        "infra_attempt_count": infra_attempt_index,
        "health_wait_sec": HEALTH_WAIT_TIMEOUT_SEC,
        "run_dir": repo_relative_str(run_dir),
        "run_dir_abs": str(run_dir),
        "returncode": returncode,
        "summary": payload["summary"],
        "runtime_observation": payload["runtime_observation"],
        "preset_resolved": payload["preset_resolved"],
        "translation_audit": {},
        "page_snapshot_audit": {},
        "rescue_trigger_issues": _rescue_trigger_issues(payload["summary"]),
        "hard_fail_issues": [],
        "passed_hard_gate": False,
        "candidate_status": "pending",
        "derived_from": derived_from,
        "oom_rescue_level": oom_rescue_level,
        "oom_rescue_changes": copy.deepcopy(oom_rescue_changes or {}),
        "setup_retry_reasons": [],
        "last_failure_kind": "",
        "last_failure_reason": "",
    }
    return record


def _rehydrate_existing_temperature_attempt(
    *,
    suite_dir: Path,
    stage_name: str,
    candidate_slug: str,
    label: str,
    tuning: dict[str, Any],
    temperature: float,
    attempt_index: int,
    baseline_run_dir: Path | None,
    derived_from: str,
    oom_rescue_level: int,
    oom_rescue_changes: dict[str, Any],
) -> tuple[dict[str, Any] | None, int]:
    entries = _existing_attempt_entries(
        suite_dir,
        stage_name,
        candidate_slug,
        temperature=temperature,
        attempt_index=attempt_index,
    )
    if not entries:
        return None, 0
    completed_entry = next(
        (entry for entry in reversed(entries) if (Path(entry["run_dir"]) / "summary.json").is_file()),
        None,
    )
    if completed_entry is None:
        return None, int(entries[-1]["infra_attempt_index"])
    record = _build_record(
        stage_name=stage_name,
        candidate_slug=candidate_slug,
        label=label,
        tuning=tuning,
        requested_temperature=REQUESTED_TEMPERATURE,
        effective_temperature=temperature,
        attempt_index=attempt_index,
        infra_attempt_index=int(completed_entry["infra_attempt_index"]),
        returncode=0,
        run_dir=Path(completed_entry["run_dir"]),
        derived_from=derived_from,
        oom_rescue_level=oom_rescue_level,
        oom_rescue_changes=oom_rescue_changes,
    )
    existing_preset = record.get("preset_resolved") or {}
    record["tuning"] = {
        "ocr_gpu_memory_utilization": float((existing_preset.get("ocr_runtime") or {}).get("gpu_memory_utilization", tuning.get("ocr_gpu_memory_utilization", 0.0)) or 0.0),
        "gemma_n_gpu_layers": int((existing_preset.get("gemma") or {}).get("n_gpu_layers", tuning.get("gemma_n_gpu_layers", 0)) or 0),
        "context_size": int((existing_preset.get("gemma") or {}).get("context_size", tuning.get("context_size", 0)) or 0),
        "chunk_size": int((existing_preset.get("gemma") or {}).get("chunk_size", tuning.get("chunk_size", 0)) or 0),
        "threads": int((existing_preset.get("gemma") or {}).get("threads", tuning.get("threads", 0)) or 0),
        "max_completion_tokens": int((existing_preset.get("gemma") or {}).get("max_completion_tokens", tuning.get("max_completion_tokens", 0)) or 0),
    }
    if baseline_run_dir is not None:
        _attach_audits(record, baseline_run_dir=baseline_run_dir)
    else:
        record["translation_audit"] = {"passed": True, "status": "BASELINE"}
        record["page_snapshot_audit"] = {"passed": True, "status": "BASELINE"}
    failure_kind, failure_reason = _classify_runtime_failure(record)
    _update_record_outcome(
        record,
        failure_kind=failure_kind,
        failure_reason=failure_reason,
        recovered_by_oom=bool(derived_from),
        original_candidate=not bool(derived_from),
    )
    if int(completed_entry["infra_attempt_index"]) > 1:
        record["setup_retry_reasons"] = [
            f"reused completed infra attempt {int(completed_entry['infra_attempt_index'])}"
        ]
    _log(
        "reuse: stage=%s candidate=%s temp=%s infra_attempt=%s"
        % (stage_name, candidate_slug, temperature, completed_entry["infra_attempt_index"])
    )
    return record, int(entries[-1]["infra_attempt_index"])


def _run_single_temperature_attempt(
    *,
    state: dict[str, Any],
    suite_dir: Path,
    stage_name: str,
    candidate_slug: str,
    label: str,
    updates: dict[str, Any],
    sample_dir: Path,
    sample_count: int,
    mode: str,
    temperature: float,
    attempt_index: int,
    derived_from: str,
    oom_rescue_level: int,
    oom_rescue_changes: dict[str, Any],
    starting_infra_attempt_index: int = 0,
) -> dict[str, Any]:
    infra_attempt_index = starting_infra_attempt_index
    merged_updates = copy.deepcopy(updates)
    _deep_merge(merged_updates, {"gemma": {"temperature": temperature}})
    preset_path, preset_payload = _materialize_preset(
        suite_dir,
        f"{stage_name}-{candidate_slug}-t{_temperature_slug(temperature)}",
        merged_updates,
    )
    tuning = {
        "ocr_gpu_memory_utilization": float((preset_payload.get("ocr_runtime") or {}).get("gpu_memory_utilization", 0.0) or 0.0),
        "gemma_n_gpu_layers": int((preset_payload.get("gemma") or {}).get("n_gpu_layers", 0) or 0),
        "context_size": int((preset_payload.get("gemma") or {}).get("context_size", 0) or 0),
        "chunk_size": int((preset_payload.get("gemma") or {}).get("chunk_size", 0) or 0),
        "threads": int((preset_payload.get("gemma") or {}).get("threads", 0) or 0),
        "max_completion_tokens": int((preset_payload.get("gemma") or {}).get("max_completion_tokens", 0) or 0),
    }
    while True:
        infra_attempt_index += 1
        _touch_state(
            state,
            status="running",
            current_stage=stage_name,
            current_candidate=candidate_slug,
        )
        run_dir = _candidate_output_dir(
            suite_dir,
            stage_name,
            candidate_slug,
            attempt_index,
            temperature,
            infra_attempt_index,
        )
        _log(
            f"run: stage={stage_name} candidate={candidate_slug} temp={temperature} infra_attempt={infra_attempt_index} mode={mode}"
        )
        completed = _run_pipeline_once(
            preset_path=preset_path,
            output_dir=run_dir,
            sample_dir=sample_dir,
            sample_count=sample_count,
            mode=mode,
        )
        record = _build_record(
            stage_name=stage_name,
            candidate_slug=candidate_slug,
            label=label,
            tuning=tuning,
            requested_temperature=REQUESTED_TEMPERATURE,
            effective_temperature=temperature,
            attempt_index=attempt_index,
            infra_attempt_index=infra_attempt_index,
            returncode=completed.returncode,
            run_dir=run_dir,
            derived_from=derived_from,
            oom_rescue_level=oom_rescue_level,
            oom_rescue_changes=oom_rescue_changes,
        )
        failure_kind, failure_reason = _classify_runtime_failure(record)
        if failure_kind == "infra_retry":
            record["setup_retry_reasons"] = [failure_reason]
            state["infra_retry_count"] = int(state.get("infra_retry_count", 0) or 0) + 1
            _touch_state(
                state,
                status="retrying_infra",
                current_stage=stage_name,
                current_candidate=candidate_slug,
                last_failure_kind=failure_kind,
                last_failure_reason=failure_reason,
            )
            time.sleep(INFRA_RETRY_DELAY_SEC)
            continue
        record["setup_retry_reasons"] = []
        record["last_failure_kind"] = failure_kind
        record["last_failure_reason"] = failure_reason
        return record


def _run_candidate_with_temperature_rescue(
    *,
    state: dict[str, Any],
    suite_dir: Path,
    stage_name: str,
    candidate_slug: str,
    label: str,
    updates: dict[str, Any],
    sample_dir: Path,
    sample_count: int,
    mode: str,
    baseline_run_dir: Path | None,
    derived_from: str = "",
    oom_rescue_level: int = 0,
    oom_rescue_changes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    chosen: dict[str, Any] | None = None
    original_candidate = not derived_from
    recovered_by_oom = bool(derived_from)
    for attempt_index, temperature in enumerate(TEMPERATURE_LADDER, start=1):
        record, existing_infra_attempt_index = _rehydrate_existing_temperature_attempt(
            suite_dir=suite_dir,
            stage_name=stage_name,
            candidate_slug=candidate_slug,
            label=label,
            tuning={
                "ocr_gpu_memory_utilization": float((updates.get("ocr_runtime") or {}).get("gpu_memory_utilization", 0.0) or 0.0),
                "gemma_n_gpu_layers": int((updates.get("gemma") or {}).get("n_gpu_layers", 0) or 0),
                "context_size": int((updates.get("gemma") or {}).get("context_size", 0) or 0),
                "chunk_size": int((updates.get("gemma") or {}).get("chunk_size", 0) or 0),
                "threads": int((updates.get("gemma") or {}).get("threads", 0) or 0),
                "max_completion_tokens": int((updates.get("gemma") or {}).get("max_completion_tokens", 0) or 0),
            },
            temperature=temperature,
            attempt_index=attempt_index,
            baseline_run_dir=baseline_run_dir,
            derived_from=derived_from,
            oom_rescue_level=oom_rescue_level,
            oom_rescue_changes=oom_rescue_changes or {},
        )
        if record is None:
            record = _run_single_temperature_attempt(
                state=state,
                suite_dir=suite_dir,
                stage_name=stage_name,
                candidate_slug=candidate_slug,
                label=label,
                updates=updates,
                sample_dir=sample_dir,
                sample_count=sample_count,
                mode=mode,
                temperature=temperature,
                attempt_index=attempt_index,
                derived_from=derived_from,
                oom_rescue_level=oom_rescue_level,
                oom_rescue_changes=oom_rescue_changes or {},
                starting_infra_attempt_index=existing_infra_attempt_index,
            )
            if baseline_run_dir is not None:
                _attach_audits(record, baseline_run_dir=baseline_run_dir)
            else:
                record["translation_audit"] = {"passed": True, "status": "BASELINE"}
                record["page_snapshot_audit"] = {"passed": True, "status": "BASELINE"}
            _update_record_outcome(
                record,
                failure_kind=str(record.get("last_failure_kind", "") or "completed"),
                failure_reason=str(record.get("last_failure_reason", "") or ""),
                recovered_by_oom=recovered_by_oom,
                original_candidate=original_candidate,
            )
        attempts.append(record)
        if str(record.get("last_failure_kind", "")) == "oom_unfit":
            break
        if not record["rescue_trigger_issues"] and record.get("passed_hard_gate"):
            chosen = record
            break
    if chosen is None:
        chosen = attempts[-1]
    chosen["attempts"] = [
        {
            "run_dir": item["run_dir"],
            "effective_temperature": item["effective_temperature"],
            "rescue_trigger_issues": item["rescue_trigger_issues"],
            "returncode": item["returncode"],
            "infra_attempt_count": item.get("infra_attempt_count", 1),
            "last_failure_kind": item.get("last_failure_kind", ""),
        }
        for item in attempts
    ]
    return chosen


def _run_candidate_with_rescue(
    *,
    state: dict[str, Any],
    suite_dir: Path,
    stage_name: str,
    candidate_slug: str,
    label: str,
    updates: dict[str, Any],
    sample_dir: Path,
    sample_count: int,
    mode: str,
    baseline_run_dir: Path | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    primary = _run_candidate_with_temperature_rescue(
        state=state,
        suite_dir=suite_dir,
        stage_name=stage_name,
        candidate_slug=candidate_slug,
        label=label,
        updates=updates,
        sample_dir=sample_dir,
        sample_count=sample_count,
        mode=mode,
        baseline_run_dir=baseline_run_dir,
    )
    records.append(primary)
    if primary.get("passed_hard_gate") or str(primary.get("last_failure_kind", "")) != "oom_unfit":
        return records

    for variant in _oom_rescue_variants(updates):
        rescue_slug = f"{candidate_slug}__oomr{int(variant['level']):02d}"
        rescue_label = f"{label} [OOM rescue {int(variant['level']):02d}]"
        rescue_record = _run_candidate_with_temperature_rescue(
            state=state,
            suite_dir=suite_dir,
            stage_name=stage_name,
            candidate_slug=rescue_slug,
            label=rescue_label,
            updates=variant["updates"],
            sample_dir=sample_dir,
            sample_count=sample_count,
            mode=mode,
            baseline_run_dir=baseline_run_dir,
            derived_from=candidate_slug,
            oom_rescue_level=int(variant["level"]),
            oom_rescue_changes=variant["changes"],
        )
        records.append(rescue_record)
        if rescue_record.get("passed_hard_gate"):
            break
        if str(rescue_record.get("last_failure_kind", "")) != "oom_unfit":
            break
    return records


def _stage0_gpu_issues(record: dict[str, Any]) -> list[str]:
    runtime = record.get("runtime_observation", {}) if isinstance(record.get("runtime_observation"), dict) else {}
    issues: list[str] = []
    checks = {
        "detector_device": runtime.get("detector_device"),
        "ocr_front_device": runtime.get("ocr_front_device"),
        "ctd_device": runtime.get("ctd_device"),
        "inpainter_device": runtime.get("inpainter_device"),
    }
    for key, value in checks.items():
        if not _is_gpu_device(str(value or "")):
            issues.append(f"{key}={value!s}")
    detector_backend = str(runtime.get("detector_backend", "") or runtime.get("detector_engine", "") or "").lower()
    if "onnx" not in detector_backend:
        issues.append(f"detector_backend={runtime.get('detector_backend', runtime.get('detector_engine', ''))}")
    if str(runtime.get("mask_refiner", "") or "") != "ctd":
        issues.append(f"mask_refiner={runtime.get('mask_refiner')}")
    if str(runtime.get("inpainter_key", "") or "") != "lama_large_512px":
        issues.append(f"inpainter_key={runtime.get('inpainter_key')}")
    loaded_models = runtime.get("gemma_loaded_model_ids", [])
    if not isinstance(loaded_models, list) or FIXED_PIPELINE["gemma_model"] not in loaded_models:
        issues.append(f"gemma_loaded_model_ids={loaded_models}")
    return issues


def _numeric(record: dict[str, Any], key: str, *, default: float = 10**9) -> float:
    value = ((record.get("summary") or {}).get(key) if isinstance(record.get("summary"), dict) else None)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _floor_free(record: dict[str, Any]) -> float:
    return _numeric(record, "gpu_floor_free_mb")


def _candidate_sort_key(record: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        _numeric(record, "elapsed_sec"),
        _numeric(record, "translate_median_sec"),
        -_floor_free(record),
        1 if record.get("derived_from") else 0,
    )


def _pick_stage_winner(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]], str]:
    accepted = [record for record in records if record.get("passed_hard_gate")]
    if not accepted:
        raise RuntimeError("No hard-gate-passing candidates were found.")
    ranked = sorted(accepted, key=_candidate_sort_key)
    fastest = ranked[0]
    fastest_elapsed = _numeric(fastest, "elapsed_sec")
    safe_candidates = [
        record
        for record in accepted
        if _numeric(record, "elapsed_sec") <= fastest_elapsed * 1.03 and _floor_free(record) >= 1000
    ]
    if safe_candidates:
        winner = sorted(safe_candidates, key=_candidate_sort_key)[0]
        reason = "selected safer candidate within 3% elapsed because GPU floor free memory stayed >= 1000 MB"
    else:
        winner = fastest
        reason = "selected fastest hard-gate-passing candidate"
    runner_up = None
    for candidate in ranked:
        if candidate["candidate_slug"] != winner["candidate_slug"]:
            runner_up = candidate
            break
    return winner, runner_up, ranked, reason


def _choose_candidate_result(records: list[dict[str, Any]]) -> dict[str, Any]:
    for record in reversed(records):
        if record.get("passed_hard_gate"):
            return record
    return records[-1]


def _candidate_updates_from_record(record: dict[str, Any]) -> dict[str, Any]:
    tuning = record.get("tuning", {}) if isinstance(record.get("tuning"), dict) else {}
    return {
        "gemma": {
            "n_gpu_layers": int(tuning.get("gemma_n_gpu_layers", 18) or 18),
            "context_size": int(tuning.get("context_size", 4096) or 4096),
            "chunk_size": int(tuning.get("chunk_size", 4) or 4),
            "threads": int(tuning.get("threads", 12) or 12),
            "max_completion_tokens": int(tuning.get("max_completion_tokens", 512) or 512),
        },
        "ocr_runtime": {
            "gpu_memory_utilization": float(tuning.get("ocr_gpu_memory_utilization", 0.72) or 0.72),
        },
    }


def _record_compact(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_slug": record.get("candidate_slug", ""),
        "label": record.get("label", ""),
        "run_dir": record.get("run_dir", ""),
        "requested_temperature": record.get("requested_temperature", REQUESTED_TEMPERATURE),
        "effective_temperature": record.get("effective_temperature", REQUESTED_TEMPERATURE),
        "rescue_attempt_count": record.get("rescue_attempt_count", 0),
        "temperature_attempt_count": record.get("temperature_attempt_count", 0),
        "infra_attempt_count": record.get("infra_attempt_count", 1),
        "health_wait_sec": record.get("health_wait_sec", 0),
        "candidate_status": record.get("candidate_status", "pending"),
        "passed_hard_gate": record.get("passed_hard_gate", False),
        "hard_fail_issues": copy.deepcopy(record.get("hard_fail_issues", []) or []),
        "rescue_trigger_issues": copy.deepcopy(record.get("rescue_trigger_issues", []) or []),
        "derived_from": record.get("derived_from", ""),
        "oom_rescue_level": record.get("oom_rescue_level", 0),
        "oom_rescue_changes": copy.deepcopy(record.get("oom_rescue_changes", {}) or {}),
        "setup_retry_reasons": copy.deepcopy(record.get("setup_retry_reasons", []) or []),
        "last_failure_kind": record.get("last_failure_kind", ""),
        "last_failure_reason": record.get("last_failure_reason", ""),
        "summary": copy.deepcopy(record.get("summary", {}) or {}),
        "runtime_observation": copy.deepcopy(record.get("runtime_observation", {}) or {}),
        "translation_audit": copy.deepcopy(record.get("translation_audit", {}) or {}),
        "page_snapshot_audit": copy.deepcopy(record.get("page_snapshot_audit", {}) or {}),
        "tuning": copy.deepcopy(record.get("tuning", {}) or {}),
        "attempts": copy.deepcopy(record.get("attempts", []) or []),
        "winner_reason": record.get("winner_reason", ""),
    }


def _run_smoke_stage(state: dict[str, Any]) -> None:
    suite_dir = Path(state["suite_dir"])
    sample_root = Path(state["sample_root"])
    smoke_source = sample_root / SAMPLE_SUBDIR / SMOKE_FILE
    if not smoke_source.is_file():
        raise FileNotFoundError(f"Smoke image not found: {smoke_source}")
    _touch_state(state, status="running", current_stage="smoke", current_candidate="baseline-ov072-ngl18")
    smoke_dir = suite_dir / "_smoke_input"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(smoke_source, smoke_dir / SMOKE_FILE)
    updates = {
        "ocr_runtime": {"gpu_memory_utilization": 0.72},
        "gemma": {"n_gpu_layers": 18},
    }
    smoke_records = _run_candidate_with_rescue(
        state=state,
        suite_dir=suite_dir,
        stage_name="smoke",
        candidate_slug="baseline-ov072-ngl18",
        label="baseline smoke (ocr_vram=0.72, ngl=18)",
        updates=updates,
        sample_dir=smoke_dir,
        sample_count=1,
        mode="one-page",
        baseline_run_dir=None,
    )
    record = _choose_candidate_result(smoke_records)
    gpu_issues = _stage0_gpu_issues(record)
    if gpu_issues:
        record.setdefault("hard_fail_issues", []).extend(gpu_issues)
        record["passed_hard_gate"] = False
        record["candidate_status"] = "failed_quality"
        _touch_state(
            state,
            status="failed",
            current_stage="smoke",
            current_candidate=record.get("candidate_slug", "baseline-ov072-ngl18"),
            last_failure_kind="gpu_enforcement_failed",
            last_failure_reason="; ".join(gpu_issues),
        )
        raise RuntimeError("Stage 0 smoke failed GPU enforcement: " + "; ".join(gpu_issues))
    state["stages"]["smoke"] = {
        "completed_at": _now_str(),
        "record": _record_compact(record),
        "records": [_record_compact(item) for item in smoke_records],
    }
    _touch_state(
        state,
        status="completed",
        current_stage="smoke",
        current_candidate=record.get("candidate_slug", ""),
        last_failure_kind="",
        last_failure_reason="",
    )


def _stage1_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for ocr_vram in (0.68, 0.72, 0.76, 0.80):
        for ngl in (14, 16, 18, 20, 22, 23):
            specs.append(
                {
                    "slug": f"ov{str(ocr_vram).replace('.', '')}-ngl{ngl}",
                    "label": f"ocr_vram={ocr_vram:.2f}, n_gpu_layers={ngl}",
                    "updates": {"ocr_runtime": {"gpu_memory_utilization": ocr_vram}, "gemma": {"n_gpu_layers": ngl}},
                }
            )
    return specs


def _run_stage1(state: dict[str, Any]) -> None:
    suite_dir = Path(state["suite_dir"])
    sample_dir = Path(state["sample_dir"])
    records: list[dict[str, Any]] = []
    baseline_run_dir: Path | None = None
    _touch_state(state, status="running", current_stage="stage1", current_candidate="")

    for spec in _stage1_specs():
        candidate_records = _run_candidate_with_rescue(
            state=state,
            suite_dir=suite_dir,
            stage_name="stage1",
            candidate_slug=spec["slug"],
            label=spec["label"],
            updates=spec["updates"],
            sample_dir=sample_dir,
            sample_count=SAMPLE_COUNT,
            mode="batch",
            baseline_run_dir=baseline_run_dir,
        )
        records.extend(candidate_records)
        if baseline_run_dir is None:
            first_pass = next((record for record in candidate_records if record.get("passed_hard_gate")), None)
            if first_pass is not None:
                baseline_run_dir = Path(first_pass["run_dir_abs"])

    if baseline_run_dir is None:
        _touch_state(state, status="failed", current_stage="stage1", current_candidate="", last_failure_kind="no_baseline_anchor", last_failure_reason="Stage 1 failed: no baseline anchor passed the hard gate.")
        raise RuntimeError("Stage 1 failed: no baseline anchor passed the hard gate.")

    winner, runner_up, ranked, reason = _pick_stage_winner(records)
    baseline_record = next(record for record in records if record["run_dir"] == repo_relative_str(baseline_run_dir))
    state["baseline_run_dir"] = repo_relative_str(baseline_run_dir)
    state["stages"]["stage1"] = {
        "completed_at": _now_str(),
        "baseline_candidate": baseline_record["candidate_slug"],
        "winner_candidate": winner["candidate_slug"],
        "runner_up_candidate": runner_up["candidate_slug"] if runner_up else "",
        "winner_reason": reason,
        "records": [_record_compact(record) for record in records],
        "ranked_candidates": [record["candidate_slug"] for record in ranked],
    }
    _touch_state(state, status="completed", current_stage="stage1", current_candidate=winner["candidate_slug"], last_failure_kind="", last_failure_reason="")


def _run_linear_stage(state: dict[str, Any], *, stage_name: str, overrides: list[tuple[str, str, dict[str, Any]]]) -> None:
    suite_dir = Path(state["suite_dir"])
    sample_dir = Path(state["sample_dir"])
    baseline_run_dir = _resolve_repo_relative_path(state["baseline_run_dir"])
    previous_stage = {
        "stage2": "stage1",
        "stage3": "stage2",
        "stage4": "stage3",
        "stage5": "stage4",
    }[stage_name]
    previous_winner_slug = str((state.get("stages", {}).get(previous_stage, {}) or {}).get("winner_candidate", "") or "")
    previous_records = (state.get("stages", {}).get(previous_stage, {}) or {}).get("records", [])
    if not previous_winner_slug or not isinstance(previous_records, list):
        raise RuntimeError(f"{stage_name} requires completed {previous_stage} state.")
    previous_winner = next((record for record in previous_records if record.get("candidate_slug") == previous_winner_slug), None)
    if not isinstance(previous_winner, dict):
        raise RuntimeError(f"Could not resolve {previous_stage} winner in state.")
    base_updates = _candidate_updates_from_record(previous_winner)
    records: list[dict[str, Any]] = []
    _touch_state(state, status="running", current_stage=stage_name, current_candidate="")
    for slug_suffix, label, override in overrides:
        updates = copy.deepcopy(base_updates)
        _deep_merge(updates, override)
        candidate_records = _run_candidate_with_rescue(
            state=state,
            suite_dir=suite_dir,
            stage_name=stage_name,
            candidate_slug=slug_suffix,
            label=label,
            updates=updates,
            sample_dir=sample_dir,
            sample_count=SAMPLE_COUNT,
            mode="batch",
            baseline_run_dir=baseline_run_dir,
        )
        records.extend(candidate_records)
    winner, runner_up, ranked, reason = _pick_stage_winner(records)
    state["stages"][stage_name] = {
        "completed_at": _now_str(),
        "winner_candidate": winner["candidate_slug"],
        "runner_up_candidate": runner_up["candidate_slug"] if runner_up else "",
        "winner_reason": reason,
        "records": [_record_compact(record) for record in records],
        "ranked_candidates": [record["candidate_slug"] for record in ranked],
    }
    _touch_state(state, status="completed", current_stage=stage_name, current_candidate=winner["candidate_slug"], last_failure_kind="", last_failure_reason="")


def _run_stage2(state: dict[str, Any]) -> None:
    _run_linear_stage(
        state,
        stage_name="stage2",
        overrides=[
            ("ctx3072", "context_size=3072", {"gemma": {"context_size": 3072}}),
            ("ctx4096", "context_size=4096", {"gemma": {"context_size": 4096}}),
        ],
    )


def _run_stage3(state: dict[str, Any]) -> None:
    _run_linear_stage(
        state,
        stage_name="stage3",
        overrides=[
            ("chunk4", "chunk_size=4", {"gemma": {"chunk_size": 4}}),
            ("chunk5", "chunk_size=5", {"gemma": {"chunk_size": 5}}),
            ("chunk6", "chunk_size=6", {"gemma": {"chunk_size": 6}}),
        ],
    )


def _run_stage4(state: dict[str, Any]) -> None:
    _run_linear_stage(
        state,
        stage_name="stage4",
        overrides=[
            ("threads10", "threads=10", {"gemma": {"threads": 10}}),
            ("threads12", "threads=12", {"gemma": {"threads": 12}}),
            ("threads14", "threads=14", {"gemma": {"threads": 14}}),
        ],
    )


def _run_stage5(state: dict[str, Any]) -> None:
    _run_linear_stage(
        state,
        stage_name="stage5",
        overrides=[
            ("max384", "max_completion_tokens=384", {"gemma": {"max_completion_tokens": 384}}),
            ("max512", "max_completion_tokens=512", {"gemma": {"max_completion_tokens": 512}}),
            ("max640", "max_completion_tokens=640", {"gemma": {"max_completion_tokens": 640}}),
        ],
    )


def _run_confirm_stage(state: dict[str, Any]) -> None:
    suite_dir = Path(state["suite_dir"])
    sample_dir = Path(state["sample_dir"])
    baseline_run_dir = _resolve_repo_relative_path(state["baseline_run_dir"])
    stage5 = state.get("stages", {}).get("stage5", {})
    records = stage5.get("records", []) if isinstance(stage5, dict) else []
    ranked = stage5.get("ranked_candidates", []) if isinstance(stage5, dict) else []
    if not ranked or not records:
        raise RuntimeError("confirm requires completed stage5 state.")
    stage5_records = {record["candidate_slug"]: record for record in records if isinstance(record, dict)}
    confirm_candidates = [stage5_records[ranked[0]]]
    if len(ranked) > 1 and ranked[1] in stage5_records:
        confirm_candidates.append(stage5_records[ranked[1]])

    aggregated_records: list[dict[str, Any]] = []
    _touch_state(state, status="running", current_stage="confirm", current_candidate="")
    for candidate in confirm_candidates:
        updates = _candidate_updates_from_record(candidate)
        label = str(candidate.get("label", candidate["candidate_slug"]))
        runs: list[dict[str, Any]] = []
        for repeat_index in range(1, CONFIRM_REPEAT_COUNT + 1):
            confirm_records = _run_candidate_with_rescue(
                state=state,
                suite_dir=suite_dir,
                stage_name="confirm",
                candidate_slug=f"{candidate['candidate_slug']}-r{repeat_index}",
                label=f"{label} [repeat {repeat_index}]",
                updates=updates,
                sample_dir=sample_dir,
                sample_count=SAMPLE_COUNT,
                mode="batch",
                baseline_run_dir=baseline_run_dir,
            )
            runs.append(_choose_candidate_result(confirm_records))
        aggregated_records.append(
            _summarize_confirm_runs(
                candidate_slug=candidate["candidate_slug"],
                label=label,
                tuning=candidate.get("tuning", {}),
                runs=runs,
            )
        )

    winner, runner_up, ranked_records, reason = _pick_stage_winner(aggregated_records)
    winner_reason = reason
    if float(winner.get("effective_temperature", REQUESTED_TEMPERATURE)) != REQUESTED_TEMPERATURE:
        winner_reason += f"; required temperature rescue to {winner['effective_temperature']:.1f}"
    if winner.get("derived_from"):
        winner_reason += f"; selected OOM rescue variant derived from {winner.get('derived_from')}"
    winner["winner_reason"] = winner_reason
    state["stages"]["confirm"] = {
        "completed_at": _now_str(),
        "winner_candidate": winner["candidate_slug"],
        "runner_up_candidate": runner_up["candidate_slug"] if runner_up else "",
        "winner_reason": winner_reason,
        "records": [_record_compact(record) for record in aggregated_records],
        "ranked_candidates": [record["candidate_slug"] for record in ranked_records],
    }
    state["winner"] = _record_compact(winner)
    state["runner_up"] = _record_compact(runner_up) if runner_up else {}
    _touch_state(
        state,
        status="completed",
        current_stage="confirm",
        current_candidate=winner["candidate_slug"],
        last_failure_kind="",
        last_failure_reason="",
    )


def _summarize_confirm_runs(candidate_slug: str, label: str, tuning: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    def _median(key: str) -> float | None:
        values = [
            float(record.get("summary", {}).get(key))
            for record in runs
            if isinstance(record.get("summary"), dict) and record["summary"].get(key) is not None
        ]
        if not values:
            return None
        return round(float(statistics.median(values)), 3)

    floor_values = [
        float(record.get("summary", {}).get("gpu_floor_free_mb"))
        for record in runs
        if isinstance(record.get("summary"), dict) and record["summary"].get("gpu_floor_free_mb") is not None
    ]
    effective_temps = [float(record.get("effective_temperature", REQUESTED_TEMPERATURE)) for record in runs]
    aggregate = {
        "stage": "confirm",
        "candidate_slug": candidate_slug,
        "label": label,
        "tuning": tuning,
        "requested_temperature": REQUESTED_TEMPERATURE,
        "effective_temperature": round(float(statistics.median(effective_temps)), 1) if effective_temps else REQUESTED_TEMPERATURE,
        "rescue_attempt_count": sum(int(record.get("rescue_attempt_count", 0) or 0) for record in runs),
        "temperature_attempt_count": sum(int(record.get("temperature_attempt_count", 1) or 1) for record in runs),
        "infra_attempt_count": sum(int(record.get("infra_attempt_count", 1) or 1) for record in runs),
        "health_wait_sec": max(int(record.get("health_wait_sec", HEALTH_WAIT_TIMEOUT_SEC) or HEALTH_WAIT_TIMEOUT_SEC) for record in runs),
        "candidate_status": "pending",
        "derived_from": str(runs[0].get("derived_from", "") or "") if runs else "",
        "oom_rescue_level": int(runs[0].get("oom_rescue_level", 0) or 0) if runs else 0,
        "oom_rescue_changes": copy.deepcopy(runs[0].get("oom_rescue_changes", {}) or {}) if runs else {},
        "setup_retry_reasons": sorted({reason for record in runs for reason in (record.get("setup_retry_reasons", []) or [])}),
        "last_failure_kind": str(runs[-1].get("last_failure_kind", "") or "") if runs else "",
        "last_failure_reason": str(runs[-1].get("last_failure_reason", "") or "") if runs else "",
        "summary": {
            "elapsed_sec": _median("elapsed_sec"),
            "translate_median_sec": _median("translate_median_sec"),
            "detect_median_sec": _median("detect_median_sec"),
            "ocr_median_sec": _median("ocr_median_sec"),
            "inpaint_median_sec": _median("inpaint_median_sec"),
            "gpu_peak_used_mb": _median("gpu_peak_used_mb"),
            "gpu_floor_free_mb": round(min(floor_values), 3) if floor_values else None,
            "gpu_peak_util_percent": _median("gpu_peak_util_percent"),
            "gpu_peak_mem_util_percent": _median("gpu_peak_mem_util_percent"),
            "page_failed_count": max(int(record.get("summary", {}).get("page_failed_count", 0) or 0) for record in runs),
            "gemma_truncated_count": max(int(record.get("summary", {}).get("gemma_truncated_count", 0) or 0) for record in runs),
            "gemma_empty_content_count": max(int(record.get("summary", {}).get("gemma_empty_content_count", 0) or 0) for record in runs),
            "gemma_missing_key_count": max(int(record.get("summary", {}).get("gemma_missing_key_count", 0) or 0) for record in runs),
            "gemma_schema_validation_fail_count": max(int(record.get("summary", {}).get("gemma_schema_validation_fail_count", 0) or 0) for record in runs),
        },
        "runtime_observation": runs[0].get("runtime_observation", {}),
        "translation_audit": runs[0].get("translation_audit", {}),
        "page_snapshot_audit": runs[0].get("page_snapshot_audit", {}),
        "confirm_runs": [
            {
                "run_dir": record["run_dir"],
                "effective_temperature": record["effective_temperature"],
                "summary": record.get("summary", {}),
                "infra_attempt_count": record.get("infra_attempt_count", 1),
            }
            for record in runs
        ],
    }
    aggregate["hard_fail_issues"] = _hard_fail_issues(aggregate)
    aggregate["passed_hard_gate"] = not aggregate["hard_fail_issues"]
    aggregate["candidate_status"] = _candidate_status(
        aggregate,
        recovered_by_oom=bool(aggregate.get("derived_from")),
        original_candidate=not bool(aggregate.get("derived_from")),
    )
    return aggregate

def _format_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _render_stage_table(stage_payload: dict[str, Any]) -> list[str]:
    records = stage_payload.get("records", []) if isinstance(stage_payload, dict) else []
    lines = [
        "| candidate | status | temp | infra_attempts | elapsed | detect_p50 | ocr_p50 | translate_p50 | inpaint_p50 | gpu_floor_free_mb | oom_rescue | issues |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        if not isinstance(record, dict):
            continue
        issues = ", ".join(record.get("hard_fail_issues", []) or [])
        oom_rescue = "-"
        if record.get("derived_from"):
            oom_rescue = f"{record.get('derived_from')} / {record.get('oom_rescue_changes', {})}"
        lines.append(
            "| {candidate} | {status} | {temp} | {infra_attempts} | {elapsed} | {detect} | {ocr} | {translate} | {inpaint} | {free} | {oom_rescue} | {issues} |".format(
                candidate=record.get("candidate_slug", ""),
                status=record.get("candidate_status", "pending"),
                temp=_format_metric(record.get("effective_temperature")),
                infra_attempts=_format_metric(record.get("infra_attempt_count", 1)),
                elapsed=_format_metric((record.get("summary") or {}).get("elapsed_sec")),
                detect=_format_metric((record.get("summary") or {}).get("detect_median_sec")),
                ocr=_format_metric((record.get("summary") or {}).get("ocr_median_sec")),
                translate=_format_metric((record.get("summary") or {}).get("translate_median_sec")),
                inpaint=_format_metric((record.get("summary") or {}).get("inpaint_median_sec")),
                free=_format_metric((record.get("summary") or {}).get("gpu_floor_free_mb")),
                oom_rescue=oom_rescue,
                issues=issues or "-",
            )
        )
    lines.append("")
    return lines

def _render_report(state: dict[str, Any]) -> str:
    lines = [
        "# Gemma IQ4_NL Japan Full-Pipeline Full-GPU Benchmark",
        "",
        "## Metadata",
        "",
        f"- generated_at: `{_now_str()}`",
        f"- suite_dir: `{repo_relative_str(state['suite_dir'])}`",
        f"- git_sha: `{state.get('git_sha', '')}`",
        f"- sample_dir: `{repo_relative_str(state.get('sample_dir', ''))}`",
        f"- sample_count: `{state.get('sample_count', SAMPLE_COUNT)}`",
        f"- official_score_scope: `full-pipeline batch on Sample/japan (22 pages)`",
        f"- suite_status: `{state.get('status', '')}`",
        f"- last_failure_kind: `{state.get('last_failure_kind', '')}`",
        f"- last_failure_reason: `{state.get('last_failure_reason', '')}`",
        "",
        "## Fixed Pipeline",
        "",
    ]
    for key, value in FIXED_PIPELINE.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Stage 0 Smoke", ""])
    smoke = state.get("stages", {}).get("smoke", {})
    if smoke:
        record = smoke.get("record", {})
        lines.append(f"- run_dir: `{record.get('run_dir', '')}`")
        lines.append(f"- detector_backend: `{((record.get('runtime_observation') or {}).get('detector_backend', ''))}`")
        lines.append(f"- detector_device: `{((record.get('runtime_observation') or {}).get('detector_device', ''))}`")
        lines.append(f"- ocr_front_device: `{((record.get('runtime_observation') or {}).get('ocr_front_device', ''))}`")
        lines.append(f"- ctd_device: `{((record.get('runtime_observation') or {}).get('ctd_device', ''))}`")
        lines.append(f"- inpainter_device: `{((record.get('runtime_observation') or {}).get('inpainter_device', ''))}`")
        lines.append(f"- gemma_loaded_model_ids: `{((record.get('runtime_observation') or {}).get('gemma_loaded_model_ids', []))}`")
        lines.append(f"- passed_hard_gate: `{record.get('passed_hard_gate', False)}`")
        if record.get("hard_fail_issues"):
            lines.append(f"- hard_fail_issues: `{record.get('hard_fail_issues')}`")
    for stage_name, title in (
        ("stage1", "Stage 1 Shared-GPU Coarse Grid"),
        ("stage2", "Stage 2 Context Size Sweep"),
        ("stage3", "Stage 3 Chunk Size Sweep"),
        ("stage4", "Stage 4 Threads Sweep"),
        ("stage5", "Stage 5 Max Completion Tokens Sweep"),
        ("confirm", "Final Confirm"),
    ):
        stage_payload = state.get("stages", {}).get(stage_name, {})
        if not stage_payload:
            continue
        lines.extend(["", f"## {title}", ""])
        lines.append(f"- winner_candidate: `{stage_payload.get('winner_candidate', '')}`")
        lines.append(f"- runner_up_candidate: `{stage_payload.get('runner_up_candidate', '')}`")
        lines.append(f"- winner_reason: `{stage_payload.get('winner_reason', '')}`")
        lines.append("")
        lines.extend(_render_stage_table(stage_payload))
    winner = state.get("winner", {})
    runner_up = state.get("runner_up", {})
    lines.extend(["", "## Recommendation", ""])
    if winner:
        lines.append(f"- winner: `{winner.get('candidate_slug', '')}`")
        lines.append(f"- winner_reason: `{winner.get('winner_reason', '')}`")
        lines.append(f"- requested_temperature: `{winner.get('requested_temperature', REQUESTED_TEMPERATURE)}`")
        lines.append(f"- effective_temperature: `{winner.get('effective_temperature', REQUESTED_TEMPERATURE)}`")
        lines.append(f"- elapsed_sec: `{_format_metric((winner.get('summary') or {}).get('elapsed_sec'))}`")
        lines.append(f"- detect_median_sec: `{_format_metric((winner.get('summary') or {}).get('detect_median_sec'))}`")
        lines.append(f"- ocr_median_sec: `{_format_metric((winner.get('summary') or {}).get('ocr_median_sec'))}`")
        lines.append(f"- translate_median_sec: `{_format_metric((winner.get('summary') or {}).get('translate_median_sec'))}`")
        lines.append(f"- inpaint_median_sec: `{_format_metric((winner.get('summary') or {}).get('inpaint_median_sec'))}`")
        lines.append(f"- gpu_floor_free_mb: `{_format_metric((winner.get('summary') or {}).get('gpu_floor_free_mb'))}`")
        lines.append(f"- tuning: `{winner.get('tuning', {})}`")
        lines.append(f"- candidate_status: `{winner.get('candidate_status', '')}`")
        lines.append(f"- infra_attempt_count: `{winner.get('infra_attempt_count', 1)}`")
        lines.append(f"- derived_from: `{winner.get('derived_from', '')}`")
        lines.append(f"- oom_rescue_changes: `{winner.get('oom_rescue_changes', {})}`")
    if runner_up:
        lines.append(f"- runner_up: `{runner_up.get('candidate_slug', '')}`")
        lines.append(f"- runner_up_elapsed_sec: `{_format_metric((runner_up.get('summary') or {}).get('elapsed_sec'))}`")
        lines.append(f"- runner_up_gpu_floor_free_mb: `{_format_metric((runner_up.get('summary') or {}).get('gpu_floor_free_mb'))}`")
    lines.append("")
    return "\n".join(lines)


def _publish_report(state: dict[str, Any]) -> Path:
    suite_dir = Path(state["suite_dir"])
    manifest = {
        "generated_at": _now_str(),
        "family_name": FAMILY_NAME,
        "suite_profile": FAMILY_PROFILE,
        "suite_dir": repo_relative_str(suite_dir),
        "git_sha": state.get("git_sha", ""),
        "results_root": state.get("results_root", ""),
        "sample_dir": repo_relative_str(state.get("sample_dir", "")),
        "sample_count": state.get("sample_count", SAMPLE_COUNT),
        "fixed_pipeline": FIXED_PIPELINE,
        "winner": state.get("winner", {}),
        "runner_up": state.get("runner_up", {}),
        "stages": state.get("stages", {}),
        "markdown_output": repo_relative_str(_docs_report_path()),
    }
    manifest_path = _manifest_path(suite_dir)
    _write_yaml(manifest_path, manifest)
    report_markdown = _render_report(state)
    local_report_path = suite_dir / "report-ko.md"
    local_report_path.write_text(report_markdown, encoding="utf-8")
    report_summary = {
        "winner": state.get("winner", {}),
        "runner_up": state.get("runner_up", {}),
        "stages": {key: {"winner_candidate": value.get("winner_candidate", ""), "winner_reason": value.get("winner_reason", "")} for key, value in (state.get("stages", {}) or {}).items()},
        "official_score_scope": "full-pipeline batch on Sample/japan (22 pages)",
        "status": state.get("status", ""),
        "last_failure_kind": state.get("last_failure_kind", ""),
    }
    latest_root = _docs_assets_latest_root()
    history_root = _docs_assets_history_root() / suite_dir.name
    for target in (latest_root, history_root):
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "report_summary.json", report_summary)
        write_json(target / "suite_state.json", state)
        shutil.copy2(manifest_path, target / manifest_path.name)
        shutil.copy2(local_report_path, target / local_report_path.name)
    _docs_report_path().parent.mkdir(parents=True, exist_ok=True)
    _docs_report_path().write_text(report_markdown, encoding="utf-8")
    write_json(family_output_root() / LAST_SUITE_RECORD, {
        "suite_dir": repo_relative_str(suite_dir),
        "manifest_path": repo_relative_str(manifest_path),
    })
    return manifest_path


def run_suite(*, sample_root: Path = ROOT / "Sample") -> int:
    suite_dir = create_run_dir("gemma_iq4nl_japan_fullgpu_suite", root=family_output_root())
    state = _bootstrap_state(suite_dir=suite_dir, sample_root=sample_root)
    _touch_state(state, status="running", current_stage="", current_candidate="", last_failure_kind="", last_failure_reason="")
    try:
        _run_smoke_stage(state)
        _run_stage1(state)
        _run_stage2(state)
        _run_stage3(state)
        _run_stage4(state)
        _run_stage5(state)
        _run_confirm_stage(state)
        _touch_state(state, status="completed", current_stage="done", current_candidate=str((state.get("winner") or {}).get("candidate_slug", "")), last_failure_kind="", last_failure_reason="")
    except Exception as exc:
        _touch_state(state, status="failed", current_stage=str(state.get("current_stage", "") or ""), current_candidate=str(state.get("current_candidate", "") or ""), last_failure_kind=str(state.get("last_failure_kind", "") or exc.__class__.__name__), last_failure_reason=str(state.get("last_failure_reason", "") or str(exc)))
        _publish_report(state)
        raise
    manifest_path = _publish_report(state)
    _log(f"suite manifest written: {manifest_path}")
    print(suite_dir)
    return 0


def _resolve_latest_suite_dir() -> Path | None:
    record_path = family_output_root() / LAST_SUITE_RECORD
    if not record_path.is_file():
        return None
    payload = _load_json(record_path)
    suite_dir = str(payload.get("suite_dir", "") or "")
    if suite_dir.startswith("./"):
        return ROOT / suite_dir[2:]
    if suite_dir:
        return Path(suite_dir)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Gemma IQ4_NL Japan full-pipeline full-GPU benchmark family.")
    parser.add_argument("command", nargs="?", default="all", choices=("all", "smoke", "stage1", "stage2", "stage3", "stage4", "stage5", "confirm", "report"))
    parser.add_argument("--suite-dir", default="")
    parser.add_argument("--sample-root", default=str(ROOT / "Sample"))
    args = parser.parse_args()

    sample_root = Path(args.sample_root)
    if args.command == "all":
        return run_suite(sample_root=sample_root)

    if args.command == "smoke" and not args.suite_dir:
        suite_dir = create_run_dir("gemma_iq4nl_japan_fullgpu_suite", root=family_output_root())
    else:
        suite_dir = Path(args.suite_dir) if args.suite_dir else _resolve_latest_suite_dir()
    if suite_dir is None:
        raise SystemExit("No suite directory found. Run `all` first or pass --suite-dir.")
    state = _bootstrap_state(suite_dir=suite_dir, sample_root=sample_root)

    if args.command == "smoke":
        _run_smoke_stage(state)
    elif args.command == "stage1":
        _run_stage1(state)
    elif args.command == "stage2":
        _run_stage2(state)
    elif args.command == "stage3":
        _run_stage3(state)
    elif args.command == "stage4":
        _run_stage4(state)
    elif args.command == "stage5":
        _run_stage5(state)
    elif args.command == "confirm":
        _run_confirm_stage(state)
    elif args.command == "report":
        pass
    else:
        raise SystemExit(f"Unsupported command: {args.command}")

    manifest_path = _publish_report(state)
    _log(f"report refreshed: {manifest_path}")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
