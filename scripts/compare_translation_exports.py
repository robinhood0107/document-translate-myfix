#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_common import DEFAULT_SAMPLE_COUNT, DEFAULT_SMOKE_COUNT, select_sample_images, write_json


def _discover_translated_jsons(run_dir: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    for path in sorted(run_dir.rglob("*_translated.json")):
        base_name = path.name.removesuffix("_translated.json")
        discovered.setdefault(base_name, path)
    return discovered


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _looks_overgenerated(candidate_text: str, baseline_text: str) -> bool:
    normalized = candidate_text.strip()
    if not normalized:
        return False

    baseline_length = max(1, len((baseline_text or "").strip()))
    if len(normalized) > max(200, baseline_length * 4):
        return True

    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    # Allow short repeated interjections that are common in comic dialogue.
    if len(lines) >= 3 and len(set(lines)) == 1 and len(lines[0]) > 4:
        return True

    return False


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Translation Export Audit",
        "",
        "## Summary",
        "",
        f"- baseline_run_dir: `{report['baseline_run_dir']}`",
        f"- candidate_run_dir: `{report['candidate_run_dir']}`",
        f"- sample_count: `{report['sample_count']}`",
        f"- pages_checked: `{report['pages_checked']}`",
        f"- missing_page_count: `{report['missing_page_count']}`",
        f"- key_mismatch_count: `{report['key_mismatch_count']}`",
        f"- suspect_block_count: `{report['suspect_block_count']}`",
        f"- passed: `{report['passed']}`",
        "",
        "## Pages",
        "",
        "| page | baseline_file | candidate_file | missing | key_mismatch | suspect_blocks |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for page in report["pages"]:
        lines.append(
            "| {page} | {baseline} | {candidate} | {missing} | {key_mismatch} | {suspect_blocks} |".format(
                page=page["page"],
                baseline=page.get("baseline_file", ""),
                candidate=page.get("candidate_file", ""),
                missing=page.get("missing", False),
                key_mismatch=page.get("key_mismatch", False),
                suspect_blocks=page.get("suspect_blocks", 0),
            )
        )

    if report["issues"]:
        lines.extend(["", "## Issues", ""])
        for issue in report["issues"]:
            lines.append(f"- {issue}")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare translated text exports between two benchmark runs.")
    parser.add_argument("--baseline-run-dir", required=True)
    parser.add_argument("--candidate-run-dir", required=True)
    parser.add_argument("--sample-dir", default=str(ROOT / "Sample"))
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SMOKE_COUNT)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    baseline_run_dir = Path(args.baseline_run_dir)
    candidate_run_dir = Path(args.candidate_run_dir)
    if not baseline_run_dir.is_dir():
        raise SystemExit(f"Baseline run directory does not exist: {baseline_run_dir}")
    if not candidate_run_dir.is_dir():
        raise SystemExit(f"Candidate run directory does not exist: {candidate_run_dir}")

    images = select_sample_images(args.sample_dir, sample_count=max(1, min(args.sample_count, DEFAULT_SAMPLE_COUNT)))
    target_pages = [path.stem for path in images]
    baseline_exports = _discover_translated_jsons(baseline_run_dir)
    candidate_exports = _discover_translated_jsons(candidate_run_dir)

    report: dict[str, Any] = {
        "baseline_run_dir": str(baseline_run_dir),
        "candidate_run_dir": str(candidate_run_dir),
        "sample_count": len(target_pages),
        "pages_checked": 0,
        "missing_page_count": 0,
        "key_mismatch_count": 0,
        "suspect_block_count": 0,
        "passed": True,
        "pages": [],
        "issues": [],
    }

    for page_name in target_pages:
        baseline_path = baseline_exports.get(page_name)
        candidate_path = candidate_exports.get(page_name)
        page_report: dict[str, Any] = {
            "page": page_name,
            "baseline_file": str(baseline_path) if baseline_path else "",
            "candidate_file": str(candidate_path) if candidate_path else "",
            "missing": False,
            "key_mismatch": False,
            "suspect_blocks": 0,
        }

        report["pages_checked"] += 1
        if baseline_path is None or candidate_path is None:
            page_report["missing"] = True
            report["missing_page_count"] += 1
            report["passed"] = False
            report["issues"].append(f"{page_name}: translated export missing in baseline or candidate run.")
            report["pages"].append(page_report)
            continue

        baseline_payload = _load_json(baseline_path)
        candidate_payload = _load_json(candidate_path)
        baseline_keys = sorted(baseline_payload.keys())
        candidate_keys = sorted(candidate_payload.keys())
        if baseline_keys != candidate_keys:
            page_report["key_mismatch"] = True
            report["key_mismatch_count"] += 1
            report["passed"] = False
            report["issues"].append(
                f"{page_name}: block key mismatch. baseline={baseline_keys} candidate={candidate_keys}"
            )

        shared_keys = sorted(set(baseline_keys) & set(candidate_keys))
        for key in shared_keys:
            baseline_value = baseline_payload.get(key)
            candidate_value = candidate_payload.get(key)
            if not isinstance(candidate_value, str):
                continue
            if _looks_overgenerated(candidate_value, baseline_value if isinstance(baseline_value, str) else ""):
                page_report["suspect_blocks"] += 1
                report["suspect_block_count"] += 1

        if page_report["suspect_blocks"] > 0:
            report["passed"] = False
            report["issues"].append(
                f"{page_name}: suspect overgenerated blocks={page_report['suspect_blocks']}"
            )

        report["pages"].append(page_report)

    markdown = _render_markdown(report)
    if args.output:
        output_path = Path(args.output)
        write_json(output_path, report)
        output_path.with_suffix(".md").write_text(markdown, encoding="utf-8")
    else:
        print(markdown)

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
