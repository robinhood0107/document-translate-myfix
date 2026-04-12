#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _load_pages(run_dir: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(run_dir / "page_snapshots.json")
    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise ValueError(f"Expected pages list in {run_dir / 'page_snapshots.json'}")
    result: dict[str, dict[str, Any]] = {}
    for page in pages:
        if not isinstance(page, dict):
            continue
        result[str(page.get("image_name", ""))] = page
    return result


def _normalize_block_texts(page: dict[str, Any]) -> list[str]:
    blocks = page.get("blocks") if isinstance(page, dict) else []
    if not isinstance(blocks, list):
        return []
    values: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            values.append("")
            continue
        values.append(str(block.get("normalized_text", "") or ""))
    return values


def _ocr_quality(page: dict[str, Any]) -> dict[str, int]:
    quality = page.get("ocr_quality") if isinstance(page, dict) else {}
    if not isinstance(quality, dict):
        quality = {}
    return {
        "block_count": int(quality.get("block_count", 0) or 0),
        "non_empty": int(quality.get("non_empty", 0) or 0),
        "empty": int(quality.get("empty", 0) or 0),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Page Snapshot Audit",
        "",
        "## Summary",
        "",
        f"- baseline_run_dir: `{report['baseline_run_dir']}`",
        f"- candidate_run_dir: `{report['candidate_run_dir']}`",
        f"- pages_checked: `{report['pages_checked']}`",
        f"- mismatch_count: `{report['mismatch_count']}`",
        f"- passed: `{report['passed']}`",
        "",
        "## Pages",
        "",
        "| page | block_count_match | ocr_quality_match | normalized_text_match |",
        "| --- | --- | --- | --- |",
    ]
    for page in report["pages"]:
        lines.append(
            "| {page} | {block_match} | {quality_match} | {text_match} |".format(
                page=page["page"],
                block_match=page["block_count_match"],
                quality_match=page["ocr_quality_match"],
                text_match=page["normalized_text_match"],
            )
        )
    if report["issues"]:
        lines.extend(["", "## Issues", ""])
        for issue in report["issues"]:
            lines.append(f"- {issue}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare page snapshots between two benchmark runs.")
    parser.add_argument("--baseline-run-dir", required=True)
    parser.add_argument("--candidate-run-dir", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    baseline_run_dir = Path(args.baseline_run_dir)
    candidate_run_dir = Path(args.candidate_run_dir)
    if not baseline_run_dir.is_dir():
        raise SystemExit(f"Baseline run directory does not exist: {baseline_run_dir}")
    if not candidate_run_dir.is_dir():
        raise SystemExit(f"Candidate run directory does not exist: {candidate_run_dir}")

    baseline_pages = _load_pages(baseline_run_dir)
    candidate_pages = _load_pages(candidate_run_dir)
    all_pages = sorted(set(baseline_pages) | set(candidate_pages))

    report: dict[str, Any] = {
        "baseline_run_dir": str(baseline_run_dir),
        "candidate_run_dir": str(candidate_run_dir),
        "pages_checked": len(all_pages),
        "mismatch_count": 0,
        "passed": True,
        "pages": [],
        "issues": [],
    }

    for page_name in all_pages:
        baseline_page = baseline_pages.get(page_name)
        candidate_page = candidate_pages.get(page_name)
        page_report = {
            "page": page_name,
            "block_count_match": False,
            "ocr_quality_match": False,
            "normalized_text_match": False,
        }
        if baseline_page is None or candidate_page is None:
            report["mismatch_count"] += 1
            report["passed"] = False
            report["issues"].append(f"{page_name}: page missing in baseline or candidate snapshots")
            report["pages"].append(page_report)
            continue

        baseline_quality = _ocr_quality(baseline_page)
        candidate_quality = _ocr_quality(candidate_page)
        page_report["block_count_match"] = baseline_quality["block_count"] == candidate_quality["block_count"]
        page_report["ocr_quality_match"] = (
            baseline_quality["non_empty"] == candidate_quality["non_empty"]
            and baseline_quality["empty"] == candidate_quality["empty"]
        )
        page_report["normalized_text_match"] = _normalize_block_texts(baseline_page) == _normalize_block_texts(candidate_page)
        if not all(page_report.values()):
            report["mismatch_count"] += 1
            report["passed"] = False
            report["issues"].append(
                f"{page_name}: block_count_match={page_report['block_count_match']} "
                f"ocr_quality_match={page_report['ocr_quality_match']} "
                f"normalized_text_match={page_report['normalized_text_match']}"
            )
        report["pages"].append(page_report)

    markdown = _render_markdown(report)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        output_path.with_suffix(".md").write_text(markdown, encoding="utf-8")
    else:
        print(markdown)

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
