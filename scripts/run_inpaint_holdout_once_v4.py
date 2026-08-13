#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.holdout import (  # noqa: E402
    claim_holdout_once,
    execution_argv_sha256,
    validate_holdout_prerequisites,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Atomically consume and run one sealed inpaint holdout. The lock is "
            "retained on success and failure; reruns are forbidden."
        )
    )
    parser.add_argument("--prerequisites", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("the exactly-once holdout runner requires a command")
    prerequisites_path = args.prerequisites.resolve()
    prerequisites = json.loads(prerequisites_path.read_text(encoding="utf-8"))
    if not isinstance(prerequisites, dict):
        raise ValueError("holdout prerequisites root must be an object")
    validate_holdout_prerequisites(prerequisites)
    execution = prerequisites["execution_binding"]
    assert isinstance(execution, dict)
    if command != execution["argv"] or execution_argv_sha256(command) != execution[
        "argv_sha256"
    ]:
        raise ValueError("holdout command differs from sealed execution argv")
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if str(prerequisites.get("code_commit") or "") != current_commit:
        raise ValueError("holdout prerequisite code_commit does not match HEAD")
    claim_holdout_once(
        prerequisites_path=prerequisites_path,
        prerequisites=prerequisites,
    )
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
