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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_common import (
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SAMPLE_DIR,
    benchmark_output_root,
    create_run_dir,
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
    resolve_corpus(sample_dir, sample_count=sample_count)
    _check_imports()
    for url in ATTACH_RUNNING_HEALTH_URLS:
        if not _url_available(url, timeout_sec=5):
            raise RuntimeError(
                "attach-running 기준 서버가 준비되지 않았습니다. "
                f"응답이 없는 URL: {url}"
            )


def _snapshot_runtime_files(snapshot_dir: Path) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for relative_path in RUNTIME_SNAPSHOT_FILES:
        source_path = ROOT / relative_path
        target_path = snapshot_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def _restore_runtime(snapshot_dir: Path) -> None:
    gemma_snapshot = snapshot_dir / "docker-compose.yaml"
    ocr_snapshot = snapshot_dir / "paddleocr_vl_docker_files" / "docker-compose.yaml"

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
        _wait_for_url(url)


def _stage_median(summary: dict[str, Any], stage_name: str) -> float | None:
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

    if page_failed_count > 0:
        issues.append("page_failed_count > 0")
        return "FAIL", issues

    if isinstance(floor_free, (int, float)) and float(floor_free) < 1536:
        issues.append("gpu_floor_free_mb < 1536")
        return "WARN", issues

    return "PASS", issues


def _build_recommendation(results: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    batch_result = next((item for item in results if item["name"] == "01_live_ops_batch"), None)
    managed_result = next((item for item in results if item["name"] == "03_gpu_shift_managed"), None)

    failing = [item for item in results if item["status"] == "FAIL"]
    vram_low = [item for item in results if item["status"] == "WARN"]

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
            "[{status}] {name} | elapsed={elapsed}s | failed={failed} | free={free}MB | issues={issues}".format(
                status=result["status"],
                name=result["name"],
                elapsed=summary.get("elapsed_sec", "-"),
                failed=summary.get("page_failed_count", "-"),
                free=summary.get("gpu_floor_free_mb", "-"),
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

    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command_stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
    (output_dir / "command_stderr.txt").write_text(completed.stderr or "", encoding="utf-8")

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
