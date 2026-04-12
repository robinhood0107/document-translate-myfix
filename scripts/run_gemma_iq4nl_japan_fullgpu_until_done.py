#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_common import create_run_dir, repo_relative_str, write_json
from gemma_iq4nl_japan_fullgpu_benchmark import LAST_SUITE_RECORD, family_output_root

SCRIPT_PATH = ROOT / "scripts" / "gemma_iq4nl_japan_fullgpu_benchmark.py"
STATE_FILE_NAME = "suite_state.json"
RETRY_DELAY_SEC = 15
STAGE_SEQUENCE = [
    "smoke",
    "report",
    "stage1",
    "report",
    "stage2",
    "report",
    "stage3",
    "report",
    "stage4",
    "report",
    "stage5",
    "report",
    "confirm",
    "report",
]
RETRYABLE_INFRA_MARKERS = (
    "infra_retry",
    "connection refused",
    "remote end closed connection without response",
    "remotedisconnected",
    "failed to establish a new connection",
    "connection aborted",
    "removal of container",
    "is already in progress",
    "service unavailable",
    "context deadline exceeded",
    "timed out waiting for",
    "temporary failure",
    "unexpected eof",
    "health/bootstrap/runtime connectivity issue detected",
)


def _log(message: str) -> None:
    print(f"[gemma-iq4nl-supervisor] {message}", flush=True)


def _state_path(suite_dir: Path) -> Path:
    return suite_dir / STATE_FILE_NAME


def _load_state(suite_dir: Path) -> dict[str, Any]:
    path = _state_path(suite_dir)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_retryable_failure(stdout: str, stderr: str, state: dict[str, Any]) -> bool:
    combined = "\n".join(
        [
            stdout or "",
            stderr or "",
            str(state.get("last_failure_kind", "") or ""),
            str(state.get("last_failure_reason", "") or ""),
        ]
    ).lower()
    return any(marker in combined for marker in RETRYABLE_INFRA_MARKERS)


def _write_latest_suite_record(suite_dir: Path) -> None:
    payload = {
        "suite_dir": repo_relative_str(suite_dir),
        "manifest_path": repo_relative_str(suite_dir / "gemma_iq4nl_japan_report_manifest.yaml"),
        "managed_by": "run_gemma_iq4nl_japan_fullgpu_until_done.py",
    }
    write_json(family_output_root() / LAST_SUITE_RECORD, payload)


def _run_stage(stage: str, *, suite_dir: Path, sample_root: Path, supervisor_dir: Path) -> None:
    attempt = 0
    while True:
        attempt += 1
        _log(f"stage={stage} attempt={attempt} suite_dir={repo_relative_str(suite_dir)}")
        cmd = [
            sys.executable,
            "-u",
            str(SCRIPT_PATH),
            stage,
            "--suite-dir",
            str(suite_dir),
            "--sample-root",
            str(sample_root),
        ]
        completed = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        attempt_dir = supervisor_dir / f"{stage}_attempt{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
        (attempt_dir / "stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
        state = _load_state(suite_dir)
        write_json(attempt_dir / "state_snapshot.json", state or {})
        if completed.returncode == 0:
            _log(f"stage={stage} completed")
            return
        if _is_retryable_failure(completed.stdout or "", completed.stderr or "", state):
            _log(
                "stage=%s retryable infra failure detected; waiting %ss and retrying. kind=%s reason=%s"
                % (
                    stage,
                    RETRY_DELAY_SEC,
                    state.get("last_failure_kind", ""),
                    state.get("last_failure_reason", ""),
                )
            )
            time.sleep(RETRY_DELAY_SEC)
            continue
        raise RuntimeError(
            "Stage %s failed permanently (exit=%s). kind=%s reason=%s\nstdout:\n%s\nstderr:\n%s"
            % (
                stage,
                completed.returncode,
                state.get("last_failure_kind", ""),
                state.get("last_failure_reason", ""),
                (completed.stdout or "")[-4000:],
                (completed.stderr or "")[-4000:],
            )
        )


def _create_fresh_suite_dir() -> Path:
    suite_dir = create_run_dir("gemma_iq4nl_japan_fullgpu_suite", root=family_output_root())
    _write_latest_suite_record(suite_dir)
    return suite_dir


def run_until_done(*, suite_dir: Path | None, sample_root: Path) -> Path:
    actual_suite_dir = suite_dir or _create_fresh_suite_dir()
    actual_suite_dir.mkdir(parents=True, exist_ok=True)
    supervisor_dir = actual_suite_dir / "_supervisor"
    supervisor_dir.mkdir(parents=True, exist_ok=True)
    _write_latest_suite_record(actual_suite_dir)
    for stage in STAGE_SEQUENCE:
        _run_stage(stage, suite_dir=actual_suite_dir, sample_root=sample_root, supervisor_dir=supervisor_dir)
    _log(f"all stages completed: {repo_relative_str(actual_suite_dir)}")
    return actual_suite_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Gemma IQ4_NL full-GPU benchmark suite until all stages finish.")
    parser.add_argument("--suite-dir", default="", help="Existing suite dir to resume. Omit to create a fresh suite.")
    parser.add_argument("--sample-root", default=str(ROOT / "Sample"))
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir) if args.suite_dir else None
    final_suite_dir = run_until_done(suite_dir=suite_dir, sample_root=Path(args.sample_root))
    print(final_suite_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
