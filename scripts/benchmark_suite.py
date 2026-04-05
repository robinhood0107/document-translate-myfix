#!/usr/bin/env python3
from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_common import (
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SAMPLE_DIR,
    benchmark_output_root,
    create_run_dir,
    remove_containers,
    resolve_corpus,
    write_json,
)


SUITE_STEPS = [
    {
        "name": "01_live_ops_batch",
        "preset": "live-ops-baseline",
        "mode": "batch",
        "runtime_mode": "attach-running",
        "description": "현재 떠 있는 서버 기준 batch 측정",
    },
    {
        "name": "02_live_ops_one_page",
        "preset": "live-ops-baseline",
        "mode": "one-page",
        "runtime_mode": "attach-running",
        "description": "현재 떠 있는 서버 기준 one-page 측정",
    },
    {
        "name": "03_gpu_shift_managed",
        "preset": "gpu-shift-ocr-front-cpu",
        "mode": "batch",
        "runtime_mode": "managed",
        "description": "OCR front CPU 전환 preset managed 측정",
    },
]

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


def _preflight(sample_dir: Path, sample_count: int) -> None:
    _log(f"preflight: Sample 폴더 확인 중... dir={sample_dir} count>={sample_count}")
    resolve_corpus(sample_dir, sample_count=sample_count)
    _log("preflight: Python 런타임 import 확인 중... (PySide6, cv2)")
    _check_imports()
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


def _step_status(summary: dict[str, Any]) -> tuple[str, list[str]]:
    issues: list[str] = []
    page_failed_count = int(summary.get("page_failed_count") or 0)
    floor_free = summary.get("gpu_floor_free_mb")
    gemma_truncated_count = int(summary.get("gemma_truncated_count") or 0)
    gemma_empty_content_count = int(summary.get("gemma_empty_content_count") or 0)
    gemma_json_retry_count = int(summary.get("gemma_json_retry_count") or 0)

    if page_failed_count > 0 or gemma_truncated_count > 0:
        issues.append("page_failed_count > 0")
        if gemma_truncated_count > 0:
            issues.append("gemma_truncated_count > 0")
        return "FAIL", issues

    if isinstance(floor_free, (int, float)) and float(floor_free) < 1536:
        issues.append("gpu_floor_free_mb < 1536")
    if gemma_empty_content_count > 0:
        issues.append("gemma_empty_content_count > 0")
    if gemma_json_retry_count > 0:
        issues.append("gemma_json_retry_count > 0")

    if issues:
        return "WARN", issues

    return "PASS", issues


def _build_recommendation(results: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    batch_result = next((item for item in results if item["name"] == "01_live_ops_batch"), None)
    managed_result = next((item for item in results if item["name"] == "03_gpu_shift_managed"), None)
    managed_baseline = next(
        (
            item
            for item in results
            if item.get("preset") == "gpu-shift-ocr-front-cpu" and item.get("mode") == "batch"
        ),
        None,
    )

    failing = [item for item in results if item["status"] == "FAIL"]
    vram_low = [item for item in results if "gpu_floor_free_mb < 1536" in item.get("issues", [])]

    if failing:
        joined = ", ".join(item["name"] for item in failing)
        lines.append(f"채택 금지: 실패가 발생한 시나리오가 있습니다. ({joined})")

    if vram_low:
        joined = ", ".join(item["name"] for item in vram_low)
        lines.append(f"VRAM 부족 후보: free VRAM 1.5GB 미만 구간이 있습니다. ({joined})")

    if batch_result and managed_result:
        batch_summary = batch_result.get("summary", {})
        managed_summary = managed_result.get("summary", {})
        batch_elapsed = batch_summary.get("elapsed_sec")
        managed_elapsed = managed_summary.get("elapsed_sec")
        managed_failed = int(managed_summary.get("page_failed_count") or 0)
        managed_floor = managed_summary.get("gpu_floor_free_mb")
        if (
            isinstance(batch_elapsed, (int, float))
            and isinstance(managed_elapsed, (int, float))
            and managed_elapsed < batch_elapsed
            and managed_failed == 0
            and isinstance(managed_floor, (int, float))
            and managed_floor >= 1536
        ):
            lines.append("OCR front CPU 전환 후보: gpu-shift-ocr-front-cpu가 더 빠르고 안전합니다.")

    if managed_baseline:
        baseline_summary = managed_baseline.get("summary", {})
        baseline_elapsed = baseline_summary.get("elapsed_sec")
        baseline_retry = float(baseline_summary.get("gemma_json_retry_count") or 0)
        baseline_empty_rate = float(baseline_summary.get("ocr_empty_rate") or 0.0)
        baseline_low_quality_rate = float(baseline_summary.get("ocr_low_quality_rate") or 0.0)
        for candidate in results:
            if candidate is managed_baseline or candidate.get("mode") != "batch":
                continue
            candidate_summary = candidate.get("summary", {})
            candidate_elapsed = candidate_summary.get("elapsed_sec")
            if not isinstance(baseline_elapsed, (int, float)) or not isinstance(candidate_elapsed, (int, float)):
                continue

            candidate_retry = float(candidate_summary.get("gemma_json_retry_count") or 0)
            candidate_empty_rate = float(candidate_summary.get("ocr_empty_rate") or 0.0)
            candidate_low_quality_rate = float(candidate_summary.get("ocr_low_quality_rate") or 0.0)

            if (
                candidate_elapsed < baseline_elapsed
                and candidate_retry <= baseline_retry
                and candidate_empty_rate <= baseline_empty_rate
                and candidate_low_quality_rate <= baseline_low_quality_rate
                and int(candidate_summary.get("page_failed_count") or 0) == 0
            ):
                lines.append(
                    f"속도/품질 균형 후보: {candidate['name']}가 managed baseline보다 빠르면서 품질 지표를 유지했습니다."
                )

    if not lines:
        lines.append("현재 live-ops-baseline 유지 후보: 실패 없이 기준선을 유지하는 것이 가장 안전합니다.")

    return lines


def _render_suite_report(results: list[dict[str, Any]], restore_status: str) -> str:
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
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
        ]
    )
    for line in _build_recommendation(results):
        lines.append(f"- {line}")

    lines.extend(
        [
            "",
            "## Quality Metrics",
            "",
            "| run | gemma_json_retry_count | gemma_chunk_retry_events | gemma_truncated_count | gemma_empty_content_count | ocr_empty_rate | ocr_low_quality_rate |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for result in results:
        summary = result.get("summary", {})
        lines.append(
            "| {name} | {json_retry} | {chunk_retry} | {truncated} | {empty_content} | {ocr_empty_rate} | {ocr_low_quality_rate} |".format(
                name=result["name"],
                json_retry=summary.get("gemma_json_retry_count", ""),
                chunk_retry=summary.get("gemma_chunk_retry_events", ""),
                truncated=summary.get("gemma_truncated_count", ""),
                empty_content=summary.get("gemma_empty_content_count", ""),
                ocr_empty_rate=summary.get("ocr_empty_rate", ""),
                ocr_low_quality_rate=summary.get("ocr_low_quality_rate", ""),
            )
        )

    lines.extend(
        [
            "",
            "## Runtime Restore",
            "",
            f"- restore_status: `{restore_status}`",
            "",
        ]
    )
    return "\n".join(lines)


def _render_console_summary(results: list[dict[str, Any]], restore_status: str) -> str:
    lines = [
        "==== 벤치마크 스위트 요약 ====",
        "",
    ]
    for result in results:
        summary = result.get("summary", {})
        issues = ", ".join(result.get("issues", [])) or "-"
        lines.append(
            "[{status}] {name} | elapsed={elapsed}s | failed={failed} | free={free}MB | retry={retry} | issues={issues}".format(
                status=result["status"],
                name=result["name"],
                elapsed=summary.get("elapsed_sec", "-"),
                failed=summary.get("page_failed_count", "-"),
                free=summary.get("gpu_floor_free_mb", "-"),
                retry=summary.get("gemma_json_retry_count", "-"),
                issues=issues,
            )
        )
    lines.extend(
        [
            "",
            "추천:",
        ]
    )
    for line in _build_recommendation(results):
        lines.append(f"- {line}")
    lines.extend(
        [
            "",
            f"런타임 복원 상태: {restore_status}",
            "",
        ]
    )
    return "\n".join(lines)


def _run_step(suite_dir: Path, step: dict[str, str], sample_dir: Path, sample_count: int) -> dict[str, Any]:
    output_dir = suite_dir / step["name"]
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "benchmark_pipeline.py"),
        "--preset",
        step["preset"],
        "--mode",
        step["mode"],
        "--repeat",
        "1",
        "--runtime-mode",
        step["runtime_mode"],
        "--sample-dir",
        str(sample_dir),
        "--sample-count",
        str(sample_count),
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
    status, issues = _step_status(summary)
    return {
        "name": step["name"],
        "preset": step["preset"],
        "mode": step["mode"],
        "runtime_mode": step["runtime_mode"],
        "description": step["description"],
        "status": status,
        "issues": issues,
        "summary": summary,
        "run_dir": str(output_dir),
    }


def _open_results(suite_dir: Path) -> None:
    if os.name != "nt" or not hasattr(os, "startfile"):
        return
    try:
        os.startfile(str(suite_dir))
        os.startfile(str(suite_dir / "suite_report.md"))
    except OSError:
        pass


def main() -> int:
    sample_dir = DEFAULT_SAMPLE_DIR
    sample_count = DEFAULT_SAMPLE_COUNT

    try:
        _preflight(sample_dir, sample_count)
    except Exception as exc:
        print(f"[suite] 시작 전 검사 실패: {exc}", file=sys.stderr)
        return 2

    suite_root = benchmark_output_root()
    suite_dir = create_run_dir("suite", root=suite_root)
    _log(f"suite output dir: {suite_dir}")
    snapshot_dir = suite_dir / "_runtime_snapshot"
    _snapshot_runtime_files(snapshot_dir)

    results: list[dict[str, Any]] = []
    restore_status = "NOT_STARTED"
    exit_code = 0

    try:
        for step in SUITE_STEPS:
            _log(f"시작: {step['name']} ({step['description']})")
            result = _run_step(suite_dir, step, sample_dir, sample_count)
            results.append(result)
            _log(
                f"완료: {step['name']} status={result['status']} elapsed={result['summary'].get('elapsed_sec')}"
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

    suite_report = _render_suite_report(results, restore_status)
    console_summary = _render_console_summary(results, restore_status)
    suite_payload = {
        "suite_dir": str(suite_dir),
        "restore_status": restore_status,
        "results": results,
        "recommendation": _build_recommendation(results),
    }

    write_json(suite_dir / "suite_report.json", suite_payload)
    (suite_dir / "suite_report.md").write_text(suite_report, encoding="utf-8")
    (suite_dir / "suite_console_summary.txt").write_text(console_summary, encoding="utf-8")
    write_json(suite_root / "last_suite_run.json", suite_payload)

    print(console_summary)
    _open_results(suite_dir)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
