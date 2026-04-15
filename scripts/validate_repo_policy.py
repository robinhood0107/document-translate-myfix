#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import PurePosixPath
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
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".idea/",
)
FORBIDDEN_TRACKED_NAMES = {
    ".DS_Store",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CHART_IMAGE_PATTERNS = (
    re.compile(r".*_comparison\.(png|jpg|jpeg|webp|bmp|svg)$", re.IGNORECASE),
    re.compile(r".*_median\.(png|jpg|jpeg|webp|bmp|svg)$", re.IGNORECASE),
    re.compile(r".*_score\.(png|jpg|jpeg|webp|bmp|svg)$", re.IGNORECASE),
    re.compile(r".*_p95\.(png|jpg|jpeg|webp|bmp|svg)$", re.IGNORECASE),
)
FORBIDDEN_BENCHMARK_MEDIA_DIR_PARTS = {
    "spotlight",
    "translated_images",
    "review_samples",
    "source_images",
    "cleaned_images",
    "raw_masks",
    "detector_overlays",
    "mask_overlays",
    "cleanup_mask_delta",
}
FORBIDDEN_BENCHMARK_MEDIA_NAME_PATTERNS = (
    re.compile(r".*_translated\.(png|jpg|jpeg|webp|bmp)$", re.IGNORECASE),
    re.compile(r".*_cleaned\.(png|jpg|jpeg|webp|bmp)$", re.IGNORECASE),
    re.compile(r".*_raw_mask\.(png|jpg|jpeg|webp|bmp)$", re.IGNORECASE),
    re.compile(r".*_detector_overlay\.(png|jpg|jpeg|webp|bmp)$", re.IGNORECASE),
    re.compile(r".*_mask_overlay\.(png|jpg|jpeg|webp|bmp)$", re.IGNORECASE),
    re.compile(r".*_cleanup_delta\.(png|jpg|jpeg|webp|bmp)$", re.IGNORECASE),
)
BENCHMARK_ONLY_PREFIXES = (
    "benchmarks/",
    "docs/benchmark/",
    "docs/banchmark_report/",
    "docs/assets/benchmarking/",
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


def git_lines(args: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


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
    for path in git_lines(["ls-files"]):
        normalized = path.replace("\\", "/")
        if normalized in FORBIDDEN_TRACKED_NAMES:
            errors.append(f"Forbidden tracked file: {normalized}")
            continue
        if any(normalized.startswith(prefix) for prefix in FORBIDDEN_TRACKED_PREFIXES):
            errors.append(f"Forbidden tracked path: {normalized}")
            continue
        if is_forbidden_experimental_media(normalized):
            errors.append(f"Forbidden tracked experimental/sample media: {normalized}")
    return errors


def is_benchmark_chart_media(normalized: str) -> bool:
    if not normalized.startswith("docs/assets/benchmarking/"):
        return False
    filename = PurePosixPath(normalized).name
    return any(pattern.match(filename) for pattern in CHART_IMAGE_PATTERNS)


def is_forbidden_experimental_media(normalized: str) -> bool:
    path_obj = PurePosixPath(normalized)
    suffix = path_obj.suffix.lower()

    if normalized.startswith("Sample/"):
        return True

    if normalized.startswith("banchmark_result_log/") and suffix in IMAGE_EXTENSIONS:
        return True

    if not normalized.startswith("docs/assets/benchmarking/"):
        return False

    if suffix == ".svg":
        return False
    if suffix not in IMAGE_EXTENSIONS:
        return False
    if is_benchmark_chart_media(normalized):
        return False

    if any(part in FORBIDDEN_BENCHMARK_MEDIA_DIR_PARTS for part in path_obj.parts):
        return True

    filename = path_obj.name
    if any(pattern.match(filename) for pattern in FORBIDDEN_BENCHMARK_MEDIA_NAME_PATTERNS):
        return True

    return True


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
    for path in git_lines(["ls-files"]):
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
    errors.extend(validate_benchmark_asset_placement(branch, args.base_branch))

    if errors:
        for error in errors:
            print(f"[POLICY] {error}", file=sys.stderr)
        return 1

    print(f"Repo policy checks passed for branch '{branch}' in mode '{args.mode}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
