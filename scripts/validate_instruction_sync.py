#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys


INSTRUCTION_FILES = frozenset({"AGENTS.md", "CLAUDE.md", "rules.md"})


def normalize(path: str) -> str:
    return path.replace("\\", "/").strip()


def validate_changed_paths(paths: list[str]) -> list[str]:
    changed = {normalize(path) for path in paths if normalize(path)}
    changed_instruction_files = changed & INSTRUCTION_FILES
    if not changed_instruction_files or changed_instruction_files == INSTRUCTION_FILES:
        return []

    missing = sorted(INSTRUCTION_FILES - changed_instruction_files)
    changed_list = ", ".join(sorted(changed_instruction_files))
    missing_list = ", ".join(missing)
    return [
        "Instruction-harness updates must change AGENTS.md, CLAUDE.md, and rules.md together. "
        f"Changed: {changed_list}. Missing: {missing_list}."
    ]


def git_changed_paths(base_sha: str, head_sha: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--find-renames", f"{base_sha}..{head_sha}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Require AGENTS.md, CLAUDE.md, and rules.md to change together."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--paths",
        nargs="*",
        help="Changed paths, normally from the staged index.",
    )
    source.add_argument(
        "--base-sha",
        help="Base commit for a Git diff; requires --head-sha.",
    )
    parser.add_argument("--head-sha", help="Head commit for a Git diff.")
    args = parser.parse_args()

    if args.base_sha:
        if not args.head_sha:
            parser.error("--head-sha is required with --base-sha")
        paths = git_changed_paths(args.base_sha, args.head_sha)
    else:
        if args.head_sha:
            parser.error("--head-sha requires --base-sha")
        paths = args.paths or []

    errors = validate_changed_paths(paths)
    if errors:
        for error in errors:
            print(f"[INSTRUCTION-SYNC] {error}", file=sys.stderr)
        return 1

    print("Instruction-harness synchronization check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
