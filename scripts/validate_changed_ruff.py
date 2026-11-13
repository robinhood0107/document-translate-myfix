#!/usr/bin/env python3
"""저장소 정책에 따라 변경된 Python 소스만 Ruff로 검사한다."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


EXCLUDED_PARTS = frozenset(
    {
        ".venv",
        ".venv-win",
        ".venv-win-cuda13",
        "build",
        "compiled",
        "dist",
        "generated",
        "third_party",
        "vendor",
        "vendors",
        "venv",
    }
)
EXCLUDED_PATH_PREFIXES = (
    "app/ui/dayu_widgets",
    "ctd_vendor",
    "modules/ocr/pororo/pororo",
)


def select_changed_python_paths(paths: list[str], *, root: Path) -> list[str]:
    selected: set[str] = set()
    resolved_root = root.resolve()
    for value in paths:
        relative = Path(value.replace("\\", "/"))
        normalized = relative.as_posix()
        if (
            relative.suffix.lower() != ".py"
            or any(part.lower() in EXCLUDED_PARTS for part in relative.parts)
            or any(
                normalized == prefix or normalized.startswith(prefix + "/")
                for prefix in EXCLUDED_PATH_PREFIXES
            )
        ):
            continue
        candidate = resolved_root / relative
        if candidate.is_symlink() or not candidate.is_file():
            continue
        try:
            candidate.resolve().relative_to(resolved_root)
        except ValueError:
            continue
        selected.add(normalized)
    return sorted(selected)


def _git_paths(root: Path, left: str, right: str, *, merge_base: bool) -> list[str]:
    if merge_base:
        revision = f"{left}...{right}"
        command = ["git", "diff", "--name-only", "-z", "--diff-filter=ACMRTUXB", revision, "--", "*.py"]
    elif set(left) == {"0"}:
        command = ["git", "diff-tree", "--root", "--no-commit-id", "--name-only", "-r", "-z", right, "--", "*.py"]
    else:
        command = ["git", "diff", "--name-only", "-z", "--diff-filter=ACMRTUXB", left, right, "--", "*.py"]
    result = subprocess.run(command, cwd=root, check=True, capture_output=True)
    return [os.fsdecode(path) for path in result.stdout.split(b"\0") if path]


def validate_changed_ruff(paths: list[str], *, root: Path) -> int:
    selected = select_changed_python_paths(paths, root=root)
    if not selected:
        print("No changed Python source files selected for Ruff.")
        return 0
    command = [sys.executable, "-m", "ruff", "check", "--no-cache", "--config", "ruff.toml", *selected]
    return subprocess.run(command, cwd=root, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Lint only changed Python source files.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-sha")
    source.add_argument("--before-sha")
    parser.add_argument("--head-sha", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        if args.base_sha:
            paths = _git_paths(root, args.base_sha, args.head_sha, merge_base=True)
        else:
            paths = _git_paths(root, args.before_sha, args.head_sha, merge_base=False)
    except subprocess.CalledProcessError:
        print("[RUFF-CHANGED] Unable to enumerate changed Python files.", file=sys.stderr)
        return 2
    return validate_changed_ruff(paths, root=root)


if __name__ == "__main__":
    raise SystemExit(main())
