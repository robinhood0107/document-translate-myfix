#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

PROTECTED_BRANCHES = {"main", "develop"}
WORK_BRANCH_RE = re.compile(r"^(feature|fix|chore|hotfix)/[a-z0-9][a-z0-9._-]*$")
BENCHMARK_BRANCH_RE = re.compile(r"^benchmarking/lab$")
BENCHMARK_WORK_BRANCH_RE = re.compile(r"^(feature|fix|chore)/benchmark[a-z0-9._/-]*$")
FORBIDDEN_TRACKED_PREFIXES = (
    ".venv/",
    ".venv-win/",
    ".venv-win-cuda13/",
    "Sample/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".idea/",
    "build/",
    "banchmark_result_log/",
    "benchmark_result_log/",
    ".gstack/",
    "docs/assets/benchmarking/",
    "fonts/",
    "testmodel/",
)
FORBIDDEN_TRACKED_NAMES = {
    ".DS_Store",
}
FORBIDDEN_TRACKED_PATTERNS = (
    re.compile(r"(?i)^.+\.(ttf|otf|woff|woff2|ttc|fon)$"),
    re.compile(r"(?i)(^|/)\.env(?:\.(?!example$).+)?$"),
    re.compile(r"(?i)^.+\.(key|pem|p12|pfx|crt|cer)$"),
    re.compile(r"(?i)(^|/)(result_|log_)[^/]+/"),
)
ALLOWED_MEDIA_PREFIXES = (
    "resources/icons/",
    "resources/static/",
)
PRIVATE_ARTIFACT_PATTERNS = (
    re.compile(r"(?i)^.+\.(cbz|zip|rar|7z|log)$"),
    re.compile(r"(?i)^.+\.(png|jpe?g|webp|gif|bmp|tiff?)$"),
)
BENCHMARK_ONLY_PREFIXES = (
    "benchmarks/",
    "docs/benchmark/",
    "docs/banchmark_report/",
)
BENCHMARK_ONLY_FILE_PATTERNS = (
    re.compile(r"^scripts/benchmark_[^/]+$"),
    re.compile(r"^scripts/generate_benchmark_report\.py$"),
    re.compile(r"^scripts/generate_paddleocr_vl15_report\.py$"),
    re.compile(r"^scripts/summarize_benchmarks\.py$"),
    re.compile(r"^scripts/compare_translation_exports\.py$"),
    re.compile(r"^scripts/apply_benchmark_preset\.py$"),
    re.compile(r"^scripts/paddleocr_vl15_[^/]+$"),
)
CONTENT_TEXT_EXTENSIONS = {
    ".bat",
    ".cfg",
    ".css",
    ".csv",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".qss",
    ".sh",
    ".toml",
    ".ts",
    ".txt",
    ".yaml",
    ".yml",
}
CONTENT_TEXT_NAMES = {
    ".gitignore",
    ".gitattributes",
    "AGENTS.md",
    "CLAUDE.md",
    "Dockerfile",
    "LICENSE",
    "README.md",
    "README_ko.md",
    "rules.md",
}
LOCAL_WINDOWS_USER_PATH_PATTERN = "C:" + r"\Users\pjjpj"
LOCAL_WSL_USER_PATH_PATTERN = "/mnt/c/Users/" + "pjjpj"
PRIVATE_SOURCE_TITLE_PATTERNS = (
    "False_" + "Honour",
    "我的" + "妈妈",
    "损" + "友",
    "警" + "花",
    "郑家" + "仪",
)
FORBIDDEN_CONTENT_PATTERNS = (
    (
        "local user path",
        re.compile(
            r"(?i)("
            + re.escape(LOCAL_WINDOWS_USER_PATH_PATTERN)
            + "|"
            + re.escape(LOCAL_WSL_USER_PATH_PATTERN)
            + ")"
        ),
    ),
    (
        "private source title",
        re.compile("|".join(re.escape(pattern) for pattern in PRIVATE_SOURCE_TITLE_PATTERNS)),
    ),
    (
        "concrete benchmark output path",
        re.compile(r"(?i)(banchmark_result_log|docs/assets/benchmarking)/[^\s`\"']*/20\d{6,}"),
    ),
)


def tracked_paths() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [
        os.fsdecode(raw_path)
        for raw_path in result.stdout.split(b"\0")
        if raw_path
    ]


def current_branch() -> str:
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_branch(branch: str, mode: str) -> list[str]:
    errors: list[str] = []
    if not branch:
        errors.append("Could not determine the current branch.")
        return errors

    if mode in {"commit", "push"} and branch in PROTECTED_BRANCHES:
        errors.append(f"Direct work on protected branch '{branch}' is not allowed.")
        return errors

    if mode in {"commit", "push"}:
        if WORK_BRANCH_RE.match(branch) or BENCHMARK_BRANCH_RE.match(branch):
            return errors
        errors.append(
            "Invalid work branch name. Use feature|fix|chore|hotfix/<slug> or benchmarking/lab."
        )
        return errors

    if mode == "ci":
        if branch in PROTECTED_BRANCHES or WORK_BRANCH_RE.match(branch) or BENCHMARK_BRANCH_RE.match(branch):
            return errors
        errors.append(
            "Invalid CI branch. Allowed: main, develop, benchmarking/lab, feature|fix|chore|hotfix/<slug>."
        )
    return errors


def validate_tracked_paths() -> list[str]:
    errors: list[str] = []
    for path in tracked_paths():
        normalized = path.replace("\\", "/")
        errors.extend(validate_tracked_path_name(normalized))
    return errors


def validate_tracked_path_name(normalized: str) -> list[str]:
    errors: list[str] = []
    if normalized in FORBIDDEN_TRACKED_NAMES:
        errors.append(f"Forbidden tracked file: {normalized}")
        return errors
    if any(normalized.startswith(prefix) for prefix in FORBIDDEN_TRACKED_PREFIXES):
        errors.append(f"Forbidden tracked path: {normalized}")
        return errors
    if any(pattern.match(normalized) for pattern in FORBIDDEN_TRACKED_PATTERNS):
        errors.append(f"Forbidden tracked generated/log path: {normalized}")
    if (
        any(pattern.match(normalized) for pattern in PRIVATE_ARTIFACT_PATTERNS)
        and not any(normalized.startswith(prefix) for prefix in ALLOWED_MEDIA_PREFIXES)
    ):
        errors.append(f"Forbidden tracked private artifact/media file: {normalized}")
    return errors


def is_text_candidate(path: str) -> bool:
    normalized = path.replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1]
    if name in CONTENT_TEXT_NAMES:
        return True
    return Path(name).suffix.lower() in CONTENT_TEXT_EXTENSIONS


def scan_sensitive_content(path: str, text: str) -> list[str]:
    errors: list[str] = []
    for label, pattern in FORBIDDEN_CONTENT_PATTERNS:
        for match in pattern.finditer(text):
            line_number = text.count("\n", 0, match.start()) + 1
            errors.append(f"Forbidden {label} in {path}:{line_number}")
            break
    return errors


def validate_sensitive_content() -> list[str]:
    errors: list[str] = []
    for path in tracked_paths():
        normalized = path.replace("\\", "/")
        if not is_text_candidate(normalized):
            continue
        try:
            raw = Path(path).read_bytes()
        except OSError as exc:
            errors.append(f"Could not read tracked file {normalized}: {exc}")
            continue
        if b"\0" in raw:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                continue
        errors.extend(scan_sensitive_content(normalized, text))
    return errors


def benchmark_assets_allowed(branch: str, base_branch: str = "") -> bool:
    normalized_base = str(base_branch or "").strip()
    if BENCHMARK_BRANCH_RE.match(branch):
        return True
    if BENCHMARK_WORK_BRANCH_RE.match(branch):
        return True
    if normalized_base == "benchmarking/lab":
        return True
    return False


def validate_benchmark_asset_placement(branch: str, base_branch: str = "") -> list[str]:
    if benchmark_assets_allowed(branch, base_branch):
        return []

    errors: list[str] = []
    for path in tracked_paths():
        normalized = path.replace("\\", "/")
        if any(normalized.startswith(prefix) for prefix in BENCHMARK_ONLY_PREFIXES):
            errors.append(
                f"Benchmark-only asset tracked outside benchmarking/lab: {normalized}"
            )
            continue
        if any(pattern.match(normalized) for pattern in BENCHMARK_ONLY_FILE_PATTERNS):
            errors.append(
                f"Benchmark-only script tracked outside benchmarking/lab: {normalized}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate repo branch and tracked-file policy.")
    parser.add_argument("--mode", choices=("commit", "push", "ci"), default="ci")
    parser.add_argument("--branch", help="Branch name to validate. Defaults to current branch.")
    parser.add_argument("--base-branch", default="", help="Optional PR base branch for CI policy checks.")
    args = parser.parse_args()

    branch = args.branch or current_branch()
    errors = []
    errors.extend(validate_branch(branch, args.mode))
    errors.extend(validate_tracked_paths())
    errors.extend(validate_sensitive_content())
    errors.extend(validate_benchmark_asset_placement(branch, args.base_branch))

    if errors:
        for error in errors:
            print(f"[POLICY] {error}", file=sys.stderr)
        return 1

    print(f"Repo policy checks passed for branch '{branch}' in mode '{args.mode}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
