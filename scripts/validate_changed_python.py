#!/usr/bin/env python3
from __future__ import annotations

import argparse
import py_compile
import subprocess
import sys
from pathlib import Path

SKIP_PREFIXES = (
    ".git/",
    ".venv/",
    ".venv-win/",
    ".venv-win-cuda13/",
    "__pycache__/",
)


def git_lines(args: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def is_python_path(path: str) -> bool:
    return path.endswith(".py") and not any(path.startswith(prefix) for prefix in SKIP_PREFIXES)


def resolve_files(cli_files: list[str], include_all: bool) -> list[str]:
    if cli_files:
        candidates = cli_files
    elif include_all:
        candidates = git_lines(["ls-files"])
    else:
        candidates = git_lines(["diff", "--cached", "--name-only", "--diff-filter=ACMRTUXB"])

    resolved: list[str] = []
    for raw in candidates:
        normalized = raw.replace("\\", "/")
        if not is_python_path(normalized):
            continue
        if Path(normalized).is_file():
            resolved.append(normalized)
    return sorted(dict.fromkeys(resolved))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile Python files to catch syntax errors.")
    parser.add_argument("--all", action="store_true", help="Validate all tracked Python files.")
    parser.add_argument("files", nargs="*", help="Explicit Python files to validate.")
    args = parser.parse_args()

    files = resolve_files(args.files, args.all)
    if not files:
        print("No Python files to validate.")
        return 0

    failures: list[tuple[str, BaseException]] = []
    for file_name in files:
        try:
            py_compile.compile(file_name, doraise=True)
        except BaseException as exc:  # pragma: no cover - direct CLI reporting
            failures.append((file_name, exc))

    if failures:
        for file_name, exc in failures:
            print(f"[FAIL] {file_name}: {exc}", file=sys.stderr)
        return 1

    print(f"Validated {len(files)} Python file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
