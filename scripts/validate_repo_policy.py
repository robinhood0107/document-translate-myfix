#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys

PROTECTED_BRANCHES = {"main", "develop"}
WORK_BRANCH_RE = re.compile(r"^codex/(feature|fix|chore|hotfix)/[a-z0-9][a-z0-9._-]*$")
RELEASE_BRANCH_RE = re.compile(r"^release/[0-9A-Za-z][0-9A-Za-z._-]*$")
BENCHMARK_BRANCH_RE = re.compile(r"^benchmarking/lab$")
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
    re.compile(r"^scripts/generate_ocr_combo_report\.py$"),
    re.compile(r"^scripts/summarize_benchmarks\.py$"),
    re.compile(r"^scripts/compare_translation_exports\.py$"),
    re.compile(r"^scripts/apply_benchmark_preset\.py$"),
    re.compile(r"^scripts/paddleocr_vl15_[^/]+$"),
    re.compile(r"^scripts/ocr_combo_[^/]+$"),
    re.compile(r"^scripts/compare_ocr_combo_reference\.py$"),
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
        if WORK_BRANCH_RE.match(branch) or RELEASE_BRANCH_RE.match(branch) or BENCHMARK_BRANCH_RE.match(branch):
            return errors
        errors.append(
            "Invalid work branch name. Use codex/feature|fix|chore|hotfix/<slug>, benchmarking/lab, or release/<version>."
        )
        return errors

    if mode == "ci":
        if branch in PROTECTED_BRANCHES or WORK_BRANCH_RE.match(branch) or RELEASE_BRANCH_RE.match(branch) or BENCHMARK_BRANCH_RE.match(branch):
            return errors
        errors.append(
            "Invalid CI branch. Allowed: main, develop, release/<version>, benchmarking/lab, codex/feature|fix|chore|hotfix/<slug>."
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
    return errors


def validate_benchmark_asset_placement(branch: str) -> list[str]:
    if BENCHMARK_BRANCH_RE.match(branch):
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


def validate_benchmark_family_structure() -> list[str]:
    errors: list[str] = []
    root = Path.cwd()
    family_anchor = root / "docs" / "benchmark" / "paddleocr-vl15"
    if not family_anchor.exists():
        return errors
    required_paths = [
        root / "scripts" / "paddleocr_vl15_benchmark_pipeline.bat",
        root / "scripts" / "paddleocr_vl15_benchmark_pipeline_cuda13.bat",
        root / "scripts" / "paddleocr_vl15_benchmark_suite.bat",
        root / "scripts" / "paddleocr_vl15_benchmark_suite_cuda13.bat",
        root / "docs" / "benchmark" / "paddleocr-vl15" / "architecture-ko.md",
        root / "docs" / "benchmark" / "paddleocr-vl15" / "workflow-ko.md",
        root / "docs" / "benchmark" / "paddleocr-vl15" / "usage-ko.md",
        root / "docs" / "banchmark_report" / "paddleocr-vl15-report-ko.md",
    ]
    for path in required_paths:
        if not path.exists():
            errors.append(f"Missing required PaddleOCR-VL15 benchmark family asset: {path.relative_to(root)}")

    architecture_path = root / "docs" / "benchmark" / "paddleocr-vl15" / "architecture-ko.md"
    report_path = root / "docs" / "banchmark_report" / "paddleocr-vl15-report-ko.md"
    for path in (architecture_path, report_path):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if "execution_scope" not in text or "official_score_scope" not in text:
            errors.append(
                f"PaddleOCR-VL15 docs/report must mention both execution_scope and official_score_scope: {path.relative_to(root)}"
            )

    combo_anchor = root / "docs" / "benchmark" / "ocr-combo"
    if combo_anchor.exists():
        combo_required_paths = [
            root / "scripts" / "ocr_combo_benchmark.py",
            root / "scripts" / "generate_ocr_combo_report.py",
            root / "scripts" / "compare_ocr_combo_reference.py",
            root / "docs" / "benchmark" / "ocr-combo" / "architecture-ko.md",
            root / "docs" / "benchmark" / "ocr-combo" / "gold-review-ko.md",
            root / "docs" / "benchmark" / "ocr-combo" / "workflow-ko.md",
            root / "docs" / "benchmark" / "ocr-combo" / "usage-ko.md",
            root / "docs" / "benchmark" / "ocr-combo" / "results-history-ko.md",
            root / "docs" / "banchmark_report" / "ocr-combo-report-ko.md",
        ]
        for path in combo_required_paths:
            if not path.exists():
                errors.append(f"Missing required OCR combo benchmark family asset: {path.relative_to(root)}")
        for path in (
            root / "docs" / "benchmark" / "ocr-combo" / "architecture-ko.md",
            root / "docs" / "banchmark_report" / "ocr-combo-report-ko.md",
        ):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            if "execution_scope" not in text or "quality_gate_scope" not in text or "gold_source" not in text:
                errors.append(
                    "OCR combo docs/report must mention execution_scope, quality_gate_scope, "
                    f"and gold_source: {path.relative_to(root)}"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate repo branch and tracked-file policy.")
    parser.add_argument("--mode", choices=("commit", "push", "ci"), default="ci")
    parser.add_argument("--branch", help="Branch name to validate. Defaults to current branch.")
    args = parser.parse_args()

    branch = args.branch or current_branch()
    errors = []
    errors.extend(validate_branch(branch, args.mode))
    errors.extend(validate_tracked_paths())
    errors.extend(validate_benchmark_asset_placement(branch))
    errors.extend(validate_benchmark_family_structure())

    if errors:
        for error in errors:
            print(f"[POLICY] {error}", file=sys.stderr)
        return 1

    print(f"Repo policy checks passed for branch '{branch}' in mode '{args.mode}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
