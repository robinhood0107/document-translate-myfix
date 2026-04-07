#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
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
    benchmark_output_root,
    create_run_dir,
    load_preset,
    remove_containers,
    run_command,
    resolve_corpus,
    repo_relative_str,
    stage_runtime_files,
    write_json,
)


DEFAULT_SUITE_PROFILE = "default"
DEFAULT_SUITE_STEPS = [
    {
        "name": "01_translation_baseline_one_page",
        "preset": "translation-baseline",
        "mode": "one-page",
        "runtime_mode": "attach-running",
        "description": "현재 떠 있는 translation-baseline 서버 기준 one-page 측정",
    },
    {
        "name": "02_translation_baseline_batch",
        "preset": "translation-baseline",
        "mode": "batch",
        "runtime_mode": "attach-running",
        "description": "현재 떠 있는 translation-baseline 서버 기준 batch 측정",
    },
    {
        "name": "03_translation_ngl23_managed",
        "preset": "translation-ngl23",
        "mode": "batch",
        "runtime_mode": "managed",
        "description": "translation-ngl23 managed 측정",
    },
]
B8665_CONTROL_STEPS = [
    {
        "name": "01_old_image_one_page",
        "preset": "translation-old-image-baseline",
        "mode": "one-page",
        "runtime_mode": "managed",
        "description": "old-image baseline one-page control",
    },
    {
        "name": "02_old_image_batch",
        "preset": "translation-old-image-baseline",
        "mode": "batch",
        "runtime_mode": "managed",
        "description": "old-image baseline representative batch control",
    },
    {
        "name": "03_b8665_object_one_page",
        "preset": "b8665-object-control",
        "mode": "one-page",
        "runtime_mode": "managed",
        "description": "b8665 image-only control with response_format=json_object",
    },
    {
        "name": "04_b8665_object_batch",
        "preset": "b8665-object-control",
        "mode": "batch",
        "runtime_mode": "managed",
        "description": "b8665 image-only representative batch control with response_format=json_object",
    },
    {
        "name": "05_b8665_schema_one_page",
        "preset": "b8665-schema-control",
        "mode": "one-page",
        "runtime_mode": "managed",
        "description": "b8665 control with response_format=json_schema",
    },
    {
        "name": "06_b8665_schema_batch",
        "preset": "b8665-schema-control",
        "mode": "batch",
        "runtime_mode": "managed",
        "description": "b8665 representative batch control with response_format=json_schema",
    },
]
SUITE_PROFILES = {
    "default": {
        "benchmark_name": "Translation Benchmark Suite",
        "benchmark_kind": "default suite",
        "benchmark_scope": "translation-baseline attach-running + managed comparison",
        "baseline_batch_step": "02_translation_baseline_batch",
    },
    "b8665-gemma4": {
        "benchmark_name": "b8665 Gemma 4 Parser Translation Optimization",
        "benchmark_kind": "managed benchmark sweep",
        "benchmark_scope": (
            "old-image vs b8665, json_object vs json_schema, chunk_size sweep, "
            "temperature sweep, n_gpu_layers sweep"
        ),
        "build_id": "b8665",
        "baseline_batch_step": "02_old_image_batch",
        "active_image": "local/llama.cpp:server-cuda-b8665",
    },
    "paddleocr-vl15-runtime": {
        "benchmark_name": "PaddleOCR-VL-1.5 Runtime Benchmark",
        "benchmark_kind": "managed family suite",
        "benchmark_scope": "detect-ocr-only official suite with warm-stable gate; legacy full pipeline remains opt-in",
        "baseline_batch_step": "",
    },
    "ocr-combo-runtime": {
        "benchmark_name": "OCR Combo Runtime Benchmark",
        "benchmark_kind": "managed family suite",
        "benchmark_scope": (
            "full-pipeline OCR+Gemma comparison with language-aware winners for "
            "China and japan corpora"
        ),
        "baseline_batch_step": "",
    },
}

RUNTIME_SNAPSHOT_FILES = [
    Path("docker-compose.yaml"),
    Path("paddleocr_vl_docker_files/docker-compose.yaml"),
    Path("paddleocr_vl_docker_files/pipeline_conf.yaml"),
    Path("paddleocr_vl_docker_files/vllm_config.yml"),
]

ATTACH_RUNNING_HEALTH_URLS = [
    "http://127.0.0.1:18080/health",
    "http://127.0.0.1:18000/v1/models",
    "http://127.0.0.1:28118/docs",
]
GEMMA_VERIFICATION_DIR_NAME = "_server_verification"


def _log(message: str) -> None:
    print(f"[suite] {message}", flush=True)


def _check_imports() -> None:
    for module_name in ("PySide6", "cv2"):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"필수 Python 런타임이 없습니다: {module_name}. "
                "앱 실행 환경을 먼저 맞춘 뒤 다시 시도하세요."
            ) from exc


def _url_available(url: str, timeout_sec: int = 5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec):
            return True
    except (urllib.error.URLError, TimeoutError):
        return False


def _wait_for_url(url: str, timeout_sec: int = 180) -> None:
    started = time.time()
    while time.time() - started < timeout_sec:
        if _url_available(url, timeout_sec=5):
            return
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {url}")


def _json_post(url: str, payload: dict[str, Any], timeout_sec: int = 60) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object response from {url}")
    return parsed


def _response_has_valid_json_content(payload: dict[str, Any]) -> bool:
    try:
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = (message.get("content") or "").strip()
        if not content:
            return False
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start < 0 or end <= start:
                return False
            parsed = json.loads(content[start : end + 1])
        return isinstance(parsed, dict)
    except Exception:
        return False


def _verify_gemma4_runtime(suite_dir: Path) -> dict[str, Any]:
    verification_dir = suite_dir / GEMMA_VERIFICATION_DIR_NAME
    runtime_dir = verification_dir / "runtime"
    verification_dir.mkdir(parents=True, exist_ok=True)

    preset, _ = load_preset("b8665-schema-control")
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
    for url in ATTACH_RUNNING_HEALTH_URLS:
        _wait_for_url(url)

    inspect_completed = run_command(
        ["docker", "inspect", "--format", "{{.Config.Image}}", "gemma-local-server"],
        check=False,
    )
    container_image = (inspect_completed.stdout or "").strip()

    log_completed = run_command(
        ["docker", "logs", "--tail", "400", "gemma-local-server"],
        check=False,
    )
    gemma_log_tail = ((log_completed.stdout or "") + (log_completed.stderr or "")).strip()
    (verification_dir / "gemma_log_tail.txt").write_text(gemma_log_tail, encoding="utf-8")

    model_name = str((preset.get("gemma") or {}).get("model", ""))
    smoke_messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Translate the user's JSON object from Korean to English. Return only one valid JSON object with identical keys.",
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "{\"block_0\":\"안녕하세요\"}"}],
        },
    ]
    object_request = {
        "model": model_name,
        "messages": smoke_messages,
        "temperature": 0.2,
        "top_k": 64,
        "top_p": 0.95,
        "min_p": 0.0,
        "max_completion_tokens": 128,
        "response_format": {"type": "json_object"},
    }
    schema_request = {
        **object_request,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "verification_blocks_1",
                "schema": {
                    "type": "object",
                    "properties": {"block_0": {"type": ["string", "null"]}},
                    "required": ["block_0"],
                    "additionalProperties": False,
                },
            },
        },
    }
    object_response = _json_post("http://127.0.0.1:18080/v1/chat/completions", object_request, timeout_sec=60)
    schema_response = _json_post("http://127.0.0.1:18080/v1/chat/completions", schema_request, timeout_sec=60)
    write_json(verification_dir / "request_object.json", object_request)
    write_json(verification_dir / "response_object.json", object_response)
    write_json(verification_dir / "request_schema.json", schema_request)
    write_json(verification_dir / "response_schema.json", schema_response)

    expected_image = str((preset.get("gemma") or {}).get("image", ""))
    log_lower = gemma_log_tail.lower()
    checks = {
        "image_matches": container_image == expected_image,
        "build_marker_found": ("b8665" in log_lower) or ("b8665" in container_image.lower()),
        "arch_gemma4_found": ("arch" in log_lower) and ("gemma4" in log_lower),
        "tool_response_eog_found": "<|tool_response>" in log_lower,
        "object_smoke_ok": _response_has_valid_json_content(object_response),
        "schema_smoke_ok": _response_has_valid_json_content(schema_response),
    }
    issues = [f"{name}=false" for name, value in checks.items() if not value]
    verification = {
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "verification_dir": repo_relative_str(verification_dir),
        "container_image": container_image,
        "checks": checks,
    }
    write_json(verification_dir / "verification.json", verification)
    return verification


def _verification_result_entry(verification: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "00_gemma4_verification",
        "preset": "b8665-schema-control",
        "preset_input": "b8665-schema-control",
        "mode": "verification",
        "runtime_mode": "managed",
        "description": "Gemma 4 dedicated parser/template verification before sweep",
        "status": verification.get("status", "FAIL"),
        "issues": list(verification.get("issues", [])),
        "summary": {},
        "run_dir": str(ROOT / verification.get("verification_dir", ".")),
    }


def _preflight(sample_dir: Path, sample_count: int, *, check_attach_running: bool) -> None:
    _log(f"preflight: Sample 폴더 확인 중... dir={sample_dir} count>={sample_count}")
    resolve_corpus(sample_dir, sample_count=sample_count)
    _log("preflight: Python 런타임 import 확인 중... (PySide6, cv2)")
    _check_imports()
    if not check_attach_running:
        return
    _log("preflight: attach-running 서버 health-check 확인 중...")
    for url in ATTACH_RUNNING_HEALTH_URLS:
        if not _url_available(url, timeout_sec=5):
            raise RuntimeError(
                "attach-running 기준 서버가 준비되지 않았습니다. "
                f"응답이 없는 URL: {url}"
            )
        _log(f"preflight: OK {url}")


def _snapshot_runtime_files(snapshot_dir: Path) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for relative_path in RUNTIME_SNAPSHOT_FILES:
        source_path = ROOT / relative_path
        target_path = snapshot_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        _log(f"runtime snapshot 저장: {relative_path}")


def _restore_runtime(snapshot_dir: Path) -> None:
    gemma_snapshot = snapshot_dir / "docker-compose.yaml"
    ocr_snapshot = snapshot_dir / "paddleocr_vl_docker_files" / "docker-compose.yaml"

    _log("restore: 기존 컨테이너 정리 중...")
    remove_containers(["gemma-local-server", "paddleocr-server", "paddleocr-vllm"])
    _log("restore: Gemma docker-compose 복원 중...")
    subprocess.run(
        [
            "docker",
            "compose",
            "--project-directory",
            str(ROOT),
            "-f",
            str(gemma_snapshot),
            "up",
            "-d",
            "--force-recreate",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    _log("restore: OCR docker-compose 복원 중...")
    subprocess.run(
        [
            "docker",
            "compose",
            "--project-directory",
            str(snapshot_dir / "paddleocr_vl_docker_files"),
            "-f",
            str(ocr_snapshot),
            "up",
            "-d",
            "--force-recreate",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(ROOT / "paddleocr_vl_docker_files"),
    )

    for url in ATTACH_RUNNING_HEALTH_URLS:
        _log(f"restore: health-check 대기 중... {url}")
        _wait_for_url(url)
    _log("restore: 모든 서비스 복원 완료")


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
        print(f"[suite][{step_name}] {payload.rstrip()}", flush=True)

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


def _stage_median(summary: dict[str, Any], stage_name: str) -> float | None:
    top_level_key = f"{stage_name}_median_sec"
    top_level = summary.get(top_level_key)
    if isinstance(top_level, (int, float)):
        return float(top_level)
    stage_stats = summary.get("stage_stats", {})
    if not isinstance(stage_stats, dict):
        return None
    payload = stage_stats.get(stage_name, {})
    if isinstance(payload, dict):
        value = payload.get("median_sec")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _summary_issue_values(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "page_failed_count": float(summary.get("page_failed_count") or 0),
        "gemma_json_retry_count": float(summary.get("gemma_json_retry_count") or 0),
        "gemma_chunk_retry_events": float(summary.get("gemma_chunk_retry_events") or 0),
        "gemma_truncated_count": float(summary.get("gemma_truncated_count") or 0),
        "gemma_empty_content_count": float(summary.get("gemma_empty_content_count") or 0),
        "gemma_missing_key_count": float(summary.get("gemma_missing_key_count") or 0),
        "gemma_reasoning_without_final_count": float(
            summary.get("gemma_reasoning_without_final_count") or 0
        ),
        "gemma_schema_validation_fail_count": float(
            summary.get("gemma_schema_validation_fail_count") or 0
        ),
        "ocr_empty_rate": float(summary.get("ocr_empty_rate") or 0.0),
        "ocr_low_quality_rate": float(summary.get("ocr_low_quality_rate") or 0.0),
    }


def _step_status(summary: dict[str, Any]) -> tuple[str, list[str]]:
    issues: list[str] = []
    values = _summary_issue_values(summary)
    if values["page_failed_count"] > 0 or values["gemma_truncated_count"] > 0:
        issues.append("page_failed_count > 0")
        if values["gemma_truncated_count"] > 0:
            issues.append("gemma_truncated_count > 0")
        return "FAIL", issues
    for key in (
        "gemma_empty_content_count",
        "gemma_missing_key_count",
        "gemma_reasoning_without_final_count",
        "gemma_schema_validation_fail_count",
        "gemma_json_retry_count",
    ):
        if values[key] > 0:
            issues.append(f"{key} > 0")
    if values["ocr_empty_rate"] > 0:
        issues.append("ocr_empty_rate > 0")
    if values["ocr_low_quality_rate"] > 0:
        issues.append("ocr_low_quality_rate > 0")
    if issues:
        return "WARN", issues
    return "PASS", issues


def _hard_reject_issues(summary: dict[str, Any], baseline_summary: dict[str, Any] | None = None) -> list[str]:
    values = _summary_issue_values(summary)
    issues: list[str] = []
    for key in (
        "page_failed_count",
        "gemma_truncated_count",
        "gemma_empty_content_count",
        "gemma_missing_key_count",
        "gemma_schema_validation_fail_count",
    ):
        if values[key] > 0:
            issues.append(f"{key} > 0")
    if baseline_summary is not None:
        baseline = _summary_issue_values(baseline_summary)
        if values["gemma_json_retry_count"] > baseline["gemma_json_retry_count"]:
            issues.append("gemma_json_retry_count > baseline")
        if values["ocr_empty_rate"] > baseline["ocr_empty_rate"]:
            issues.append("ocr_empty_rate > baseline")
        if values["ocr_low_quality_rate"] > baseline["ocr_low_quality_rate"]:
            issues.append("ocr_low_quality_rate > baseline")
    return issues


def _candidate_sort_key(result: dict[str, Any]) -> tuple[float, float, float, float]:
    summary = result.get("summary", {})
    return (
        float(summary.get("gemma_json_retry_count") or 0),
        float(summary.get("translate_median_sec") or 1e12),
        float(summary.get("elapsed_sec") or 1e12),
        float(summary.get("page_failed_count") or 0),
    )


def _select_promoted_candidates(
    results: list[dict[str, Any]],
    *,
    baseline_summary: dict[str, Any],
    top_n: int,
) -> list[dict[str, Any]]:
    passing: list[dict[str, Any]] = []
    for result in results:
        issues = _hard_reject_issues(result.get("summary", {}), baseline_summary)
        result.setdefault("promotion_issues", issues)
        if not issues:
            passing.append(result)
    passing.sort(key=_candidate_sort_key)
    return passing[:top_n]


def _find_result(results: list[dict[str, Any]], *, preset: str, mode: str) -> dict[str, Any] | None:
    for result in results:
        if result.get("mode") != mode:
            continue
        if result.get("preset") == preset or str(result.get("preset_input", "")) == preset:
            return result
    return None


def _run_step(
    suite_dir: Path,
    step: dict[str, Any],
    sample_dir: Path,
    sample_count: int,
) -> dict[str, Any]:
    output_dir = suite_dir / step["name"]
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "benchmark_pipeline.py"),
        "--preset",
        str(step["preset"]),
        "--mode",
        str(step["mode"]),
        "--repeat",
        "1",
        "--runtime-mode",
        str(step["runtime_mode"]),
        "--sample-dir",
        str(sample_dir),
        "--sample-count",
        str(step.get("sample_count", sample_count)),
        "--output-dir",
        str(output_dir),
        "--label",
        step["name"],
    ]
    env = os.environ.copy()
    env["CT_BENCH_OUTPUT_ROOT"] = str(benchmark_output_root())
    _log(f"step command: {' '.join(cmd)}")
    _log(f"step output dir: {output_dir}")

    completed = _run_command_streaming(
        cmd=cmd,
        cwd=ROOT,
        env=env,
        output_dir=output_dir,
        step_name=step["name"],
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{step['name']} 실행 실패 (code={completed.returncode})\n"
            f"{(completed.stderr or completed.stdout).strip()}"
        )

    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    resolved_preset_path = output_dir / "preset_resolved.json"
    resolved_preset_name = str(step["preset"])
    if resolved_preset_path.is_file():
        try:
            resolved_payload = json.loads(resolved_preset_path.read_text(encoding="utf-8"))
            resolved_preset_name = str(resolved_payload.get("name", resolved_preset_name))
        except Exception:
            pass
    status, issues = _step_status(summary)
    return {
        "name": step["name"],
        "preset": resolved_preset_name,
        "preset_input": step["preset"],
        "mode": step["mode"],
        "runtime_mode": step["runtime_mode"],
        "description": step["description"],
        "status": status,
        "issues": issues,
        "summary": summary,
        "run_dir": str(output_dir),
    }


def _materialize_preset(
    output_path: Path,
    *,
    base_preset: str,
    name: str,
    description: str,
    gemma_updates: dict[str, Any],
) -> Path:
    preset, _ = load_preset(base_preset)
    preset["name"] = name
    preset["description"] = description
    gemma = dict(preset.get("gemma", {}))
    gemma.update(gemma_updates)
    preset["gemma"] = gemma
    write_json(output_path, preset)
    return output_path


def _temperature_slug(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "")


def _run_translation_audit(
    *,
    baseline_run_dir: Path,
    candidate_run_dir: Path,
    sample_dir: Path,
    sample_count: int,
) -> dict[str, Any] | None:
    output_path = candidate_run_dir / "translation_audit.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / "compare_translation_exports.py"),
            "--baseline-run-dir",
            str(baseline_run_dir),
            "--candidate-run-dir",
            str(candidate_run_dir),
            "--sample-dir",
            str(sample_dir),
            "--sample-count",
            str(sample_count),
            "--output",
            str(output_path),
        ],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if not output_path.is_file():
        return None
    report = json.loads(output_path.read_text(encoding="utf-8"))
    if completed.returncode not in (0, 1):
        report.setdefault("issues", []).append(
            f"compare_translation_exports.py exited with code {completed.returncode}"
        )
        report["passed"] = False
        write_json(output_path, report)
    return report


def _control_winner_format(object_batch: dict[str, Any], schema_batch: dict[str, Any]) -> str:
    object_issues = _hard_reject_issues(object_batch.get("summary", {}))
    schema_issues = _hard_reject_issues(schema_batch.get("summary", {}))
    if object_issues and not schema_issues:
        return "schema"
    if schema_issues and not object_issues:
        return "object"
    ordered = sorted(
        [("object", object_batch), ("schema", schema_batch)],
        key=lambda item: (
            float(item[1]["summary"].get("gemma_truncated_count") or 0),
            float(item[1]["summary"].get("gemma_missing_key_count") or 0),
            float(item[1]["summary"].get("gemma_json_retry_count") or 0),
            float(item[1]["summary"].get("elapsed_sec") or 1e12),
            float(item[1]["summary"].get("translate_median_sec") or 1e12),
        ),
    )
    return ordered[0][0]


def _build_recommendation(results: list[dict[str, Any]], baseline_step_name: str) -> list[str]:
    lines: list[str] = []
    baseline_batch = next((item for item in results if item["name"] == baseline_step_name), None)
    if baseline_batch is None:
        lines.append("기준 batch 결과가 없어 후보 비교를 진행하지 못했습니다.")
        return lines

    baseline_summary = baseline_batch.get("summary", {})
    candidates = [
        item
        for item in results
        if item.get("mode") == "batch" and item["name"] != baseline_step_name and item.get("summary")
    ]
    passing = [
        item for item in candidates if not _hard_reject_issues(item["summary"], baseline_summary)
    ]
    if not passing:
        lines.append("현재 기준선 유지 후보: 품질 gate를 통과하며 더 빠른 batch 후보를 찾지 못했습니다.")
        return lines

    ordered = sorted(
        passing,
        key=lambda item: (
            float(item["summary"].get("elapsed_sec") or 1e12),
            float(item["summary"].get("translate_median_sec") or 1e12),
            float(item["summary"].get("gemma_json_retry_count") or 1e12),
        ),
    )
    winner = ordered[0]
    lines.append(
        "속도/품질 균형 후보: {name} ({preset}) elapsed=`{elapsed}` translate_median=`{median}`".format(
            name=winner["name"],
            preset=winner["preset"],
            elapsed=winner["summary"].get("elapsed_sec"),
            median=winner["summary"].get("translate_median_sec"),
        )
    )
    return lines


def _render_suite_report(results: list[dict[str, Any]], restore_status: str, *, baseline_step_name: str) -> str:
    lines = [
        "# Benchmark Suite Report",
        "",
        "## Runs",
        "",
        "| run | preset | mode | runtime_mode | status | elapsed_sec | page_done_count | page_failed_count | gpu_floor_free_mb | gpu_peak_used_mb | ocr_median_sec | translate_median_sec | inpaint_median_sec |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        summary = result.get("summary", {})
        lines.append(
            "| {name} | {preset} | {mode} | {runtime_mode} | {status} | {elapsed} | {done} | {failed} | {free} | {used} | {ocr} | {translate} | {inpaint} |".format(
                name=result["name"],
                preset=result["preset"],
                mode=result["mode"],
                runtime_mode=result["runtime_mode"],
                status=result["status"],
                elapsed=summary.get("elapsed_sec", ""),
                done=summary.get("page_done_count", ""),
                failed=summary.get("page_failed_count", ""),
                free=summary.get("gpu_floor_free_mb", ""),
                used=summary.get("gpu_peak_used_mb", ""),
                ocr=_stage_median(summary, "ocr"),
                translate=_stage_median(summary, "translate"),
                inpaint=_stage_median(summary, "inpaint"),
            )
        )

    lines.extend(["", "## Recommendation", ""])
    for line in _build_recommendation(results, baseline_step_name):
        lines.append(f"- {line}")

    lines.extend(
        [
            "",
            "## Quality Metrics",
            "",
            "| run | json_retry | chunk_retry | truncated | empty_content | missing_key | reasoning_without_final | schema_validation_fail | ocr_empty_rate | ocr_low_quality_rate |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for result in results:
        summary = result.get("summary", {})
        lines.append(
            "| {name} | {json_retry} | {chunk_retry} | {truncated} | {empty_content} | {missing_key} | {reasoning_without_final} | {schema_validation_fail} | {ocr_empty_rate} | {ocr_low_quality_rate} |".format(
                name=result["name"],
                json_retry=summary.get("gemma_json_retry_count", ""),
                chunk_retry=summary.get("gemma_chunk_retry_events", ""),
                truncated=summary.get("gemma_truncated_count", ""),
                empty_content=summary.get("gemma_empty_content_count", ""),
                missing_key=summary.get("gemma_missing_key_count", ""),
                reasoning_without_final=summary.get("gemma_reasoning_without_final_count", ""),
                schema_validation_fail=summary.get("gemma_schema_validation_fail_count", ""),
                ocr_empty_rate=summary.get("ocr_empty_rate", ""),
                ocr_low_quality_rate=summary.get("ocr_low_quality_rate", ""),
            )
        )

    lines.extend(["", "## Runtime Restore", "", f"- restore_status: `{restore_status}`", ""])
    return "\n".join(lines)


def _render_console_summary(results: list[dict[str, Any]], restore_status: str, *, baseline_step_name: str) -> str:
    lines = ["==== 벤치마크 스위트 요약 ====", ""]
    for result in results:
        summary = result.get("summary", {})
        issues = ", ".join(result.get("issues", [])) or "-"
        lines.append(
            "[{status}] {name} | elapsed={elapsed}s | failed={failed} | free={free}MB | retry={retry} | missing={missing} | issues={issues}".format(
                status=result["status"],
                name=result["name"],
                elapsed=summary.get("elapsed_sec", "-"),
                failed=summary.get("page_failed_count", "-"),
                free=summary.get("gpu_floor_free_mb", "-"),
                retry=summary.get("gemma_json_retry_count", "-"),
                missing=summary.get("gemma_missing_key_count", "-"),
                issues=issues,
            )
        )
    lines.extend(["", "추천:"])
    for line in _build_recommendation(results, baseline_step_name):
        lines.append(f"- {line}")
    lines.extend(["", f"런타임 복원 상태: {restore_status}", ""])
    return "\n".join(lines)


def _open_results(suite_dir: Path) -> None:
    if os.name != "nt" or not hasattr(os, "startfile"):
        return
    try:
        os.startfile(str(suite_dir))
        os.startfile(str(suite_dir / "suite_report.md"))
        report_path = ROOT / "docs" / "banchmark_report" / "report-ko.md"
        if report_path.is_file():
            os.startfile(str(report_path))
    except OSError:
        pass


def _write_b8665_manifest(
    suite_dir: Path,
    *,
    verification: dict[str, Any],
    control_results: list[dict[str, Any]],
    chunk_results: list[dict[str, Any]],
    temperature_results: list[dict[str, Any]],
    ngl_results: list[dict[str, Any]],
    active_preset: str,
    winner_preset: str,
) -> Path:
    def run_name(result: dict[str, Any]) -> str:
        return Path(str(result["run_dir"])).name

    controls = []
    labels = {
        "translation-old-image-baseline": "old image baseline",
        "b8665-object-control": "b8665 json_object",
        "b8665-schema-control": "b8665 json_schema",
    }
    for preset in (
        "translation-old-image-baseline",
        "b8665-object-control",
        "b8665-schema-control",
    ):
        one_page = _find_result(control_results, preset=preset, mode="one-page")
        batch = _find_result(control_results, preset=preset, mode="batch")
        if one_page is None or batch is None:
            continue
        controls.append(
            {
                "label": labels.get(preset, preset),
                "preset": preset,
                "one_page": run_name(one_page),
                "batch": run_name(batch),
            }
        )

    def encode_rows(results: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for result in results:
            preset_name = str(result["preset"])
            request_path = Path(str(result["run_dir"])) / "preset_resolved.json"
            value = None
            if request_path.is_file():
                payload = json.loads(request_path.read_text(encoding="utf-8"))
                value = ((payload.get("gemma") or {}).get(key))
            row: dict[str, Any] = {
                "preset": preset_name,
                "one_page": run_name(result) if result["mode"] == "one-page" else "",
                "batch": run_name(result) if result["mode"] == "batch" else "",
            }
            if value is not None:
                row[key] = value
            rows.append(row)
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            grouped.setdefault(row["preset"], {"preset": row["preset"]})
            grouped[row["preset"]].update({k: v for k, v in row.items() if v not in ("", None)})
        return list(grouped.values())

    manifest = {
        "results_root": "banchmark_result_log",
        "active_preset": active_preset,
        "winning_candidate_preset": winner_preset,
        "benchmark": {
            "name": SUITE_PROFILES["b8665-gemma4"]["benchmark_name"],
            "kind": SUITE_PROFILES["b8665-gemma4"]["benchmark_kind"],
            "scope": SUITE_PROFILES["b8665-gemma4"]["benchmark_scope"],
            "build_id": SUITE_PROFILES["b8665-gemma4"]["build_id"],
            "active_image": SUITE_PROFILES["b8665-gemma4"]["active_image"],
        },
        "verification": verification,
        "controls": controls,
        "chunk_sweep": encode_rows(chunk_results, "chunk_size"),
        "temperature_sweep": encode_rows(temperature_results, "temperature"),
        "n_gpu_layers_sweep": encode_rows(ngl_results, "n_gpu_layers"),
        "report": {
            "markdown_output": "docs/banchmark_report/report-ko.md",
            "assets_dir": "docs/assets/benchmarking/latest",
        },
    }
    manifest_path = suite_dir / "report_manifest_b8665.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return manifest_path


def _run_report(manifest_path: Path | None = None) -> None:
    cmd = [sys.executable, "-u", str(ROOT / "scripts" / "generate_benchmark_report.py")]
    if manifest_path is not None:
        cmd.extend(["--manifest", str(manifest_path)])
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip():
        _log(completed.stdout.strip())


def _run_default_profile(
    *,
    suite_dir: Path,
    sample_dir: Path,
    sample_count: int,
) -> tuple[list[dict[str, Any]], Path | None]:
    results: list[dict[str, Any]] = []
    for step in DEFAULT_SUITE_STEPS:
        _log(f"시작: {step['name']} ({step['description']})")
        result = _run_step(suite_dir, step, sample_dir, sample_count)
        results.append(result)
        _log(f"완료: {step['name']} status={result['status']} elapsed={result['summary'].get('elapsed_sec')}")
    return results, None


def _run_b8665_profile(
    *,
    suite_dir: Path,
    sample_dir: Path,
    sample_count: int,
) -> tuple[list[dict[str, Any]], Path | None]:
    results: list[dict[str, Any]] = []
    control_results: list[dict[str, Any]] = []
    chunk_results: list[dict[str, Any]] = []
    temperature_results: list[dict[str, Any]] = []
    ngl_results: list[dict[str, Any]] = []
    generated_dir = suite_dir / "_generated_presets"
    generated_dir.mkdir(parents=True, exist_ok=True)

    verification = _verify_gemma4_runtime(suite_dir)
    results.append(_verification_result_entry(verification))
    if verification.get("status") != "PASS":
        manifest_path = _write_b8665_manifest(
            suite_dir,
            verification=verification,
            control_results=control_results,
            chunk_results=chunk_results,
            temperature_results=temperature_results,
            ngl_results=ngl_results,
            active_preset="b8665-schema-control",
            winner_preset="",
        )
        return results, manifest_path

    for step in B8665_CONTROL_STEPS:
        _log(f"시작: {step['name']} ({step['description']})")
        result = _run_step(suite_dir, step, sample_dir, sample_count)
        results.append(result)
        control_results.append(result)
        _log(f"완료: {step['name']} status={result['status']} elapsed={result['summary'].get('elapsed_sec')}")

    baseline_batch = _find_result(results, preset="translation-old-image-baseline", mode="batch")
    object_batch = _find_result(results, preset="b8665-object-control", mode="batch")
    schema_batch = _find_result(results, preset="b8665-schema-control", mode="batch")
    if baseline_batch is None or object_batch is None or schema_batch is None:
        raise RuntimeError("b8665 control steps did not produce all required batch results.")

    for candidate in (object_batch, schema_batch):
        audit = _run_translation_audit(
            baseline_run_dir=Path(str(baseline_batch["run_dir"])),
            candidate_run_dir=Path(str(candidate["run_dir"])),
            sample_dir=sample_dir,
            sample_count=5,
        )
        if audit and not audit.get("passed", False):
            candidate["status"] = "WARN"
            candidate.setdefault("issues", []).append("translation_audit failed")

    winning_format = _control_winner_format(object_batch, schema_batch)
    _log(f"format winner 결정: {winning_format}")
    winning_control_preset = "b8665-schema-control" if winning_format == "schema" else "b8665-object-control"

    chunk_candidates: list[tuple[str, str]] = []
    for chunk_size in (4, 5, 6):
        if winning_format == "schema":
            preset_ref = f"b8665-schema-ch{chunk_size}-t06-ngl23"
        else:
            preset_name = f"b8665-object-ch{chunk_size}-t06-ngl23"
            preset_path = generated_dir / f"{preset_name}.json"
            preset_ref = str(
                _materialize_preset(
                    preset_path,
                    base_preset=winning_control_preset,
                    name=preset_name,
                    description=f"Generated b8665 object chunk sweep candidate: chunk_size={chunk_size}",
                    gemma_updates={"chunk_size": chunk_size},
                )
            )
        step = {
            "name": f"07_chunk_{winning_format}_ch{chunk_size}_one_page",
            "preset": preset_ref,
            "mode": "one-page",
            "runtime_mode": "managed",
            "description": f"{winning_format} chunk sweep one-page ch={chunk_size}",
        }
        result = _run_step(suite_dir, step, sample_dir, sample_count)
        results.append(result)
        chunk_results.append(result)
        chunk_candidates.append((result["preset"], step["name"]))

    promoted_chunks = _select_promoted_candidates(
        [item for item in chunk_results if item["mode"] == "one-page"],
        baseline_summary=baseline_batch["summary"],
        top_n=2,
    )
    for promoted in promoted_chunks:
        step = {
            "name": f"10_{promoted['preset']}_batch",
            "preset": promoted.get("preset_input", promoted["preset"]),
            "mode": "batch",
            "runtime_mode": "managed",
            "description": f"chunk sweep promoted batch for {promoted['preset']}",
        }
        result = _run_step(suite_dir, step, sample_dir, sample_count)
        audit = _run_translation_audit(
            baseline_run_dir=Path(str(baseline_batch["run_dir"])),
            candidate_run_dir=Path(str(result["run_dir"])),
            sample_dir=sample_dir,
            sample_count=5,
        )
        if audit and not audit.get("passed", False):
            result["status"] = "WARN"
            result.setdefault("issues", []).append("translation_audit failed")
        results.append(result)
        chunk_results.append(result)

    chunk_batch_results = [item for item in chunk_results if item["mode"] == "batch"]
    chunk_winner_source = chunk_batch_results or [item for item in chunk_results if item["mode"] == "one-page"]
    if not chunk_winner_source:
        raise RuntimeError("chunk_size sweep did not produce any candidate results.")
    chunk_winner = sorted(chunk_winner_source, key=_candidate_sort_key)[0]
    chunk_winner_preset, chunk_winner_path = load_preset(
        str(chunk_winner.get("preset_input", chunk_winner["preset"]))
    )
    chunk_winner_size = int((chunk_winner_preset.get("gemma") or {}).get("chunk_size", 4))
    _log(f"chunk winner 결정: preset={chunk_winner_preset.get('name')} chunk_size={chunk_winner_size}")

    coarse_temps = (0.5, 0.6, 0.7, 0.8)
    temp_one_page_results: list[dict[str, Any]] = []
    for temp in coarse_temps:
        if winning_format == "schema" and chunk_winner_size == 5 and temp in (0.5, 0.6, 0.7, 0.8):
            preset_ref = f"b8665-schema-ch5-t{_temperature_slug(temp)}-ngl23"
        else:
            preset_name = f"b8665-{winning_format}-ch{chunk_winner_size}-t{_temperature_slug(temp)}-ngl23"
            preset_path = generated_dir / f"{preset_name}.json"
            preset_ref = str(
                _materialize_preset(
                    preset_path,
                    base_preset=winning_control_preset,
                    name=preset_name,
                    description=f"Generated {winning_format} temperature sweep candidate: temperature={temp}",
                    gemma_updates={"chunk_size": chunk_winner_size, "temperature": temp},
                )
            )
        step = {
            "name": f"20_temp_{winning_format}_t{_temperature_slug(temp)}_one_page",
            "preset": preset_ref,
            "mode": "one-page",
            "runtime_mode": "managed",
            "description": f"{winning_format} temperature sweep one-page t={temp}",
        }
        result = _run_step(suite_dir, step, sample_dir, sample_count)
        results.append(result)
        temperature_results.append(result)
        temp_one_page_results.append(result)

    promoted_temps = _select_promoted_candidates(
        temp_one_page_results,
        baseline_summary=baseline_batch["summary"],
        top_n=2,
    )
    for promoted in promoted_temps:
        step = {
            "name": f"23_{promoted['preset']}_batch",
            "preset": promoted.get("preset_input", promoted["preset"]),
            "mode": "batch",
            "runtime_mode": "managed",
            "description": f"temperature sweep promoted batch for {promoted['preset']}",
        }
        result = _run_step(suite_dir, step, sample_dir, sample_count)
        audit = _run_translation_audit(
            baseline_run_dir=Path(str(baseline_batch["run_dir"])),
            candidate_run_dir=Path(str(result["run_dir"])),
            sample_dir=sample_dir,
            sample_count=5,
        )
        if audit and not audit.get("passed", False):
            result["status"] = "WARN"
            result.setdefault("issues", []).append("translation_audit failed")
        results.append(result)
        temperature_results.append(result)

    temperature_batch_results = [item for item in temperature_results if item["mode"] == "batch"]
    temp_winner_source = temperature_batch_results or temp_one_page_results
    temp_winner = sorted(temp_winner_source, key=_candidate_sort_key)[0]
    temp_winner_preset, _ = load_preset(str(temp_winner.get("preset_input", temp_winner["preset"])))
    temperature_winner = float((temp_winner_preset.get("gemma") or {}).get("temperature", 0.6))

    fine_candidates: list[float] = []
    for candidate in (round(temperature_winner - 0.05, 2), round(temperature_winner + 0.05, 2)):
        if 0.0 < candidate < 1.0 and candidate not in coarse_temps:
            fine_candidates.append(candidate)
    fine_one_page_results: list[dict[str, Any]] = []
    for temp in fine_candidates:
        preset_name = f"b8665-{winning_format}-ch{chunk_winner_size}-t{_temperature_slug(temp)}-ngl23"
        preset_path = generated_dir / f"{preset_name}.json"
        preset_ref = str(
            _materialize_preset(
                preset_path,
                base_preset=winning_control_preset,
                name=preset_name,
                description=f"Generated fine temperature sweep candidate: temperature={temp}",
                gemma_updates={"chunk_size": chunk_winner_size, "temperature": temp},
            )
        )
        step = {
            "name": f"26_temp_fine_t{_temperature_slug(temp)}_one_page",
            "preset": preset_ref,
            "mode": "one-page",
            "runtime_mode": "managed",
            "description": f"fine temperature sweep one-page t={temp}",
        }
        result = _run_step(suite_dir, step, sample_dir, sample_count)
        results.append(result)
        temperature_results.append(result)
        fine_one_page_results.append(result)

    if fine_one_page_results:
        promoted_fine = _select_promoted_candidates(
            fine_one_page_results,
            baseline_summary=baseline_batch["summary"],
            top_n=2,
        )
        for promoted in promoted_fine:
            step = {
                "name": f"27_{promoted['preset']}_batch",
                "preset": promoted.get("preset_input", promoted["preset"]),
                "mode": "batch",
                "runtime_mode": "managed",
                "description": f"fine temperature sweep promoted batch for {promoted['preset']}",
            }
            result = _run_step(suite_dir, step, sample_dir, sample_count)
            audit = _run_translation_audit(
                baseline_run_dir=Path(str(baseline_batch["run_dir"])),
                candidate_run_dir=Path(str(result["run_dir"])),
                sample_dir=sample_dir,
                sample_count=5,
            )
            if audit and not audit.get("passed", False):
                result["status"] = "WARN"
                result.setdefault("issues", []).append("translation_audit failed")
            results.append(result)
            temperature_results.append(result)

    temperature_batch_results = [item for item in temperature_results if item["mode"] == "batch"]
    temp_winner_source = temperature_batch_results or [item for item in temperature_results if item["mode"] == "one-page"]
    temp_winner = sorted(temp_winner_source, key=_candidate_sort_key)[0]
    temp_winner_preset, _ = load_preset(str(temp_winner.get("preset_input", temp_winner["preset"])))
    temperature_winner = float((temp_winner_preset.get("gemma") or {}).get("temperature", 0.6))
    _log(f"temperature winner 결정: preset={temp_winner_preset.get('name')} temperature={temperature_winner}")

    ngl_one_page_results: list[dict[str, Any]] = []
    for ngl in (23, 24, 25):
        if (
            winning_format == "schema"
            and chunk_winner_size == 5
            and round(temperature_winner, 2) == 0.6
            and ngl in (23, 24, 25)
        ):
            preset_ref = f"b8665-schema-ch5-t06-ngl{ngl}"
        else:
            preset_name = f"b8665-{winning_format}-ch{chunk_winner_size}-t{_temperature_slug(temperature_winner)}-ngl{ngl}"
            preset_path = generated_dir / f"{preset_name}.json"
            preset_ref = str(
                _materialize_preset(
                    preset_path,
                    base_preset=winning_control_preset,
                    name=preset_name,
                    description=f"Generated n_gpu_layers sweep candidate: n_gpu_layers={ngl}",
                    gemma_updates={
                        "chunk_size": chunk_winner_size,
                        "temperature": temperature_winner,
                        "n_gpu_layers": ngl,
                    },
                )
            )
        existing = _find_result(results, preset=str(preset_ref), mode="one-page")
        if existing is not None:
            ngl_one_page_results.append(existing)
            ngl_results.append(existing)
            continue
        step = {
            "name": f"30_ngl_{ngl}_one_page",
            "preset": preset_ref,
            "mode": "one-page",
            "runtime_mode": "managed",
            "description": f"n_gpu_layers sweep one-page ngl={ngl}",
        }
        result = _run_step(suite_dir, step, sample_dir, sample_count)
        results.append(result)
        ngl_results.append(result)
        ngl_one_page_results.append(result)

    promoted_ngl = _select_promoted_candidates(
        ngl_one_page_results,
        baseline_summary=baseline_batch["summary"],
        top_n=3,
    )
    failed_ngl_values: set[int] = set()
    for candidate in ngl_one_page_results:
        if candidate not in promoted_ngl:
            preset_payload, _ = load_preset(str(candidate.get("preset_input", candidate["preset"])))
            failed_ngl_values.add(int((preset_payload.get("gemma") or {}).get("n_gpu_layers", 0)))
    for promoted in promoted_ngl:
        step = {
            "name": f"33_{promoted['preset']}_batch",
            "preset": promoted.get("preset_input", promoted["preset"]),
            "mode": "batch",
            "runtime_mode": "managed",
            "description": f"n_gpu_layers sweep promoted batch for {promoted['preset']}",
        }
        result = _run_step(suite_dir, step, sample_dir, sample_count)
        audit = _run_translation_audit(
            baseline_run_dir=Path(str(baseline_batch["run_dir"])),
            candidate_run_dir=Path(str(result["run_dir"])),
            sample_dir=sample_dir,
            sample_count=5,
        )
        if audit and not audit.get("passed", False):
            result["status"] = "WARN"
            result.setdefault("issues", []).append("translation_audit failed")
        results.append(result)
        ngl_results.append(result)
        if _hard_reject_issues(result["summary"], baseline_batch["summary"]):
            preset_payload, _ = load_preset(str(result.get("preset_input", result["preset"])))
            failed_ngl_values.add(int((preset_payload.get("gemma") or {}).get("n_gpu_layers", 0)))

    for ngl in sorted(value for value in failed_ngl_values if value in (24, 25)):
        if (
            winning_format == "schema"
            and chunk_winner_size == 5
            and round(temperature_winner, 2) == 0.6
        ):
            preset_ref = f"b8665-schema-ch5-t06-ngl{ngl}-ctx3072"
        else:
            preset_name = (
                f"b8665-{winning_format}-ch{chunk_winner_size}-t{_temperature_slug(temperature_winner)}"
                f"-ngl{ngl}-ctx3072"
            )
            preset_path = generated_dir / f"{preset_name}.json"
            preset_ref = str(
                _materialize_preset(
                    preset_path,
                    base_preset=winning_control_preset,
                    name=preset_name,
                    description=f"Generated rescue candidate for n_gpu_layers={ngl} with ctx=3072",
                    gemma_updates={
                        "chunk_size": chunk_winner_size,
                        "temperature": temperature_winner,
                        "n_gpu_layers": ngl,
                        "context_size": 3072,
                    },
                )
            )
        step = {
            "name": f"36_rescue_ngl_{ngl}_ctx3072_one_page",
            "preset": preset_ref,
            "mode": "one-page",
            "runtime_mode": "managed",
            "description": f"rescue one-page ngl={ngl} ctx=3072",
        }
        result = _run_step(suite_dir, step, sample_dir, sample_count)
        results.append(result)
        ngl_results.append(result)
        if not _hard_reject_issues(result["summary"], baseline_batch["summary"]):
            batch_step = {
                "name": f"37_{result['preset']}_batch",
                "preset": result.get("preset_input", result["preset"]),
                "mode": "batch",
                "runtime_mode": "managed",
                "description": f"rescue batch for {result['preset']}",
            }
            batch_result = _run_step(suite_dir, batch_step, sample_dir, sample_count)
            audit = _run_translation_audit(
                baseline_run_dir=Path(str(baseline_batch["run_dir"])),
                candidate_run_dir=Path(str(batch_result["run_dir"])),
                sample_dir=sample_dir,
                sample_count=5,
            )
            if audit and not audit.get("passed", False):
                batch_result["status"] = "WARN"
                batch_result.setdefault("issues", []).append("translation_audit failed")
            results.append(batch_result)
            ngl_results.append(batch_result)

    batch_candidates = [
        result
        for result in results
        if result["mode"] == "batch" and result["preset"] != "translation-old-image-baseline"
    ]
    winner_candidates = [
        item for item in batch_candidates if not _hard_reject_issues(item["summary"], baseline_batch["summary"])
    ]
    winner_result = sorted(
        winner_candidates or [baseline_batch],
        key=lambda item: (
            float(item["summary"].get("elapsed_sec") or 1e12),
            float(item["summary"].get("translate_median_sec") or 1e12),
            float(item["summary"].get("gemma_json_retry_count") or 1e12),
        ),
    )[0]

    if (
        float(winner_result["summary"].get("gemma_reasoning_without_final_count") or 0) > 0
        or float(winner_result["summary"].get("gemma_empty_content_count") or 0) > 0
    ):
        winner_payload, _ = load_preset(str(winner_result.get("preset_input", winner_result["preset"])))
        winner_gemma = winner_payload.get("gemma") or {}
        preset_name = f"{Path(str(winner_result['preset'])).stem}-think256"
        preset_path = generated_dir / f"{preset_name}.json"
        preset_ref = str(
            _materialize_preset(
                preset_path,
                base_preset=str(winner_result.get("preset_input", winner_result["preset"])),
                name=preset_name,
                description="Optional low-think fallback for reasoning_without_final or empty_content edge cases",
                gemma_updates={
                    "reasoning": "on",
                    "reasoning_budget": 256,
                    "reasoning_format": winner_gemma.get("reasoning_format", "deepseek") or "deepseek",
                    "think_briefly_prompt": True,
                },
            )
        )
        step = {
            "name": "40_low_think_subset_batch",
            "preset": preset_ref,
            "mode": "batch",
            "runtime_mode": "managed",
            "sample_count": 5,
            "description": "optional low-think fallback on hard 5-page subset",
        }
        result = _run_step(suite_dir, step, sample_dir, 5)
        results.append(result)
        if not _hard_reject_issues(result["summary"], baseline_batch["summary"]):
            batch_candidates.append(result)

    manifest_path = _write_b8665_manifest(
        suite_dir,
        verification=verification,
        control_results=control_results,
        chunk_results=chunk_results,
        temperature_results=temperature_results,
        ngl_results=ngl_results,
        active_preset=winning_control_preset,
        winner_preset=str(winner_result["preset"]),
    )
    return results, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the benchmark suite.")
    parser.add_argument(
        "--suite-profile",
        default=DEFAULT_SUITE_PROFILE,
        choices=tuple(SUITE_PROFILES.keys()),
        help="Suite profile to execute",
    )
    args = parser.parse_args()

    if args.suite_profile == "paddleocr-vl15-runtime":
        from paddleocr_vl15_benchmark import run_suite as run_paddleocr_vl15_suite

        return run_paddleocr_vl15_suite(
            sample_dir=DEFAULT_SAMPLE_DIR,
            sample_count=DEFAULT_SAMPLE_COUNT,
        )
    if args.suite_profile == "ocr-combo-runtime":
        from ocr_combo_benchmark import run_suite as run_ocr_combo_suite

        return run_ocr_combo_suite(
            sample_root=DEFAULT_SAMPLE_DIR,
        )

    sample_dir = DEFAULT_SAMPLE_DIR
    sample_count = DEFAULT_SAMPLE_COUNT
    profile_cfg = SUITE_PROFILES[args.suite_profile]
    check_attach_running = args.suite_profile == DEFAULT_SUITE_PROFILE

    try:
        _preflight(sample_dir, sample_count, check_attach_running=check_attach_running)
    except Exception as exc:
        print(f"[suite] 시작 전 검사 실패: {exc}", file=sys.stderr)
        return 2

    suite_root = benchmark_output_root()
    suite_label = "suite" if args.suite_profile == DEFAULT_SUITE_PROFILE else f"{args.suite_profile}_suite"
    suite_dir = create_run_dir(suite_label, root=suite_root)
    _log(f"suite output dir: {suite_dir}")
    snapshot_dir = suite_dir / "_runtime_snapshot"
    _snapshot_runtime_files(snapshot_dir)

    results: list[dict[str, Any]] = []
    restore_status = "NOT_STARTED"
    exit_code = 0
    manifest_path: Path | None = None

    try:
        if args.suite_profile == DEFAULT_SUITE_PROFILE:
            results, manifest_path = _run_default_profile(
                suite_dir=suite_dir,
                sample_dir=sample_dir,
                sample_count=sample_count,
            )
        else:
            results, manifest_path = _run_b8665_profile(
                suite_dir=suite_dir,
                sample_dir=sample_dir,
                sample_count=sample_count,
            )
    except Exception as exc:
        exit_code = 1
        results.append(
            {
                "name": "suite_error",
                "preset": "-",
                "mode": "-",
                "runtime_mode": "-",
                "description": "suite orchestration failure",
                "status": "FAIL",
                "issues": [str(exc)],
                "summary": {},
                "run_dir": str(suite_dir),
            }
        )
        _log(f"실패: {exc}")
    finally:
        try:
            _log("원래 Docker 런타임 상태 복원 중...")
            _restore_runtime(snapshot_dir)
            restore_status = "RESTORED"
        except Exception as exc:
            restore_status = f"RESTORE_FAILED: {exc}"
            exit_code = 1
            _log(f"복원 실패: {exc}")

    suite_report = _render_suite_report(
        results,
        restore_status,
        baseline_step_name=str(profile_cfg["baseline_batch_step"]),
    )
    console_summary = _render_console_summary(
        results,
        restore_status,
        baseline_step_name=str(profile_cfg["baseline_batch_step"]),
    )
    suite_payload = {
        "suite_dir": str(suite_dir),
        "suite_profile": args.suite_profile,
        "restore_status": restore_status,
        "results": results,
        "recommendation": _build_recommendation(results, str(profile_cfg["baseline_batch_step"])),
        "manifest_path": repo_relative_str(manifest_path) if manifest_path else "",
    }

    write_json(suite_dir / "suite_report.json", suite_payload)
    (suite_dir / "suite_report.md").write_text(suite_report, encoding="utf-8")
    (suite_dir / "suite_console_summary.txt").write_text(console_summary, encoding="utf-8")
    write_json(suite_root / "last_suite_run.json", suite_payload)

    try:
        _log("자동 benchmark report 생성 중...")
        _run_report(manifest_path)
        _log("자동 benchmark report 생성 완료")
    except Exception as exc:
        exit_code = 1
        _log(f"자동 benchmark report 생성 실패: {exc}")

    print(console_summary)
    _open_results(suite_dir)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
