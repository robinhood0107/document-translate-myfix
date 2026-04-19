#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys

ALLOWED_DOC_PATTERNS = (
    re.compile(r"^README\.md$"),
    re.compile(r"^README_ko\.md$"),
    re.compile(r"^rules\.md$"),
    re.compile(r"^docs/setup/quickstart(?:-ko)?\.md$"),
    re.compile(r"^docs/gemma/[^/]+\.md$"),
    re.compile(r"^docs/hunyuan/[^/]+\.md$"),
    re.compile(r"^docs/repo/github-rulesets-public-free-ko\.md$"),
    re.compile(r"^hunyuanocr_docker_files/README\.md$"),
    re.compile(r"^paddleocr_vl_docker_files/README\.md$"),
)


def git_diff_name_status(base_sha: str, head_sha: str) -> list[list[str]]:
    result = subprocess.run(
        ["git", "diff", "--name-status", "--find-renames", f"{base_sha}..{head_sha}"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[list[str]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        rows.append(line.split("\t"))
    return rows


def is_doc_candidate(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return normalized.endswith(".md") or normalized.startswith("docs/")


def is_allowed(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return any(pattern.match(normalized) for pattern in ALLOWED_DOC_PATTERNS)


def validate_main_docs_policy(base_sha: str, head_sha: str) -> list[str]:
    errors: list[str] = []
    for row in git_diff_name_status(base_sha, head_sha):
        status = row[0]
        if status.startswith("D"):
            continue
        path = row[-1]
        if not is_doc_candidate(path):
            continue
        if is_allowed(path):
            continue
        errors.append(
            f"Disallowed markdown/doc path for main PR: {path}. Main only accepts minimal public operational docs."
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate markdown/doc allowlist for PRs targeting main.")
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    args = parser.parse_args()

    errors = validate_main_docs_policy(args.base_sha, args.head_sha)
    if errors:
        for error in errors:
            print(f"[MAIN-DOCS] {error}", file=sys.stderr)
        return 1

    print("Main docs allowlist check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
