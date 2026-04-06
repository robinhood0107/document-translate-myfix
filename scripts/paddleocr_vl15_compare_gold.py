#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_common import write_json

XYXY_IOU_THRESHOLD = 0.90
BUBBLE_IOU_THRESHOLD = 0.85


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _load_pages(payload_or_path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(payload_or_path)
    if path.is_dir():
        path = path / "page_snapshots.json"
    payload = _load_json(path)
    pages = payload.get("pages", [])
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(pages, list):
        raise ValueError(f"Expected 'pages' list in {path}")
    for page in pages:
        if not isinstance(page, dict):
            continue
        image_stem = str(page.get("image_stem", "") or "")
        if image_stem:
            result[image_stem] = page
    return result


def _xyxy_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = a_area + b_area - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def _match_blocks(
    baseline_blocks: list[dict[str, Any]],
    candidate_blocks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    scores: list[tuple[float, int, int, str]] = []
    for baseline_index, baseline_block in enumerate(baseline_blocks):
        for candidate_index, candidate_block in enumerate(candidate_blocks):
            baseline_bubble = baseline_block.get("bubble_xyxy")
            candidate_bubble = candidate_block.get("bubble_xyxy")
            if baseline_bubble and candidate_bubble:
                score = _xyxy_iou(list(map(float, baseline_bubble)), list(map(float, candidate_bubble)))
                threshold = BUBBLE_IOU_THRESHOLD
                match_kind = "bubble"
            else:
                score = _xyxy_iou(
                    list(map(float, baseline_block.get("xyxy") or [])),
                    list(map(float, candidate_block.get("xyxy") or [])),
                )
                threshold = XYXY_IOU_THRESHOLD
                match_kind = "xyxy"
            if score >= threshold:
                scores.append((score, baseline_index, candidate_index, match_kind))

    matches: list[dict[str, Any]] = []
    used_baseline: set[int] = set()
    used_candidate: set[int] = set()
    for score, baseline_index, candidate_index, match_kind in sorted(scores, reverse=True):
        if baseline_index in used_baseline or candidate_index in used_candidate:
            continue
        used_baseline.add(baseline_index)
        used_candidate.add(candidate_index)
        matches.append(
            {
                "baseline_index": baseline_index,
                "candidate_index": candidate_index,
                "iou": round(score, 4),
                "match_kind": match_kind,
            }
        )

    issues: list[str] = []
    if len(matches) != len(baseline_blocks) or len(matches) != len(candidate_blocks):
        unmatched_baseline = sorted(set(range(len(baseline_blocks))) - used_baseline)
        unmatched_candidate = sorted(set(range(len(candidate_blocks))) - used_candidate)
        if unmatched_baseline:
            issues.append(f"unmatched_baseline_blocks={unmatched_baseline}")
        if unmatched_candidate:
            issues.append(f"unmatched_candidate_blocks={unmatched_candidate}")
    return sorted(matches, key=lambda item: item["baseline_index"]), issues


def _compare_page(
    baseline_page: dict[str, Any],
    candidate_page: dict[str, Any],
) -> dict[str, Any]:
    issues: list[str] = []
    baseline_failed = bool(baseline_page.get("page_failed"))
    candidate_failed = bool(candidate_page.get("page_failed"))
    if baseline_failed != candidate_failed:
        issues.append(
            "page_failed mismatch: baseline={baseline} candidate={candidate}".format(
                baseline=baseline_failed,
                candidate=candidate_failed,
            )
        )
    if candidate_failed:
        issues.append(f"candidate_page_failed_reason={candidate_page.get('page_failed_reason', '')}")

    baseline_quality = baseline_page.get("ocr_quality", {}) if isinstance(baseline_page.get("ocr_quality"), dict) else {}
    candidate_quality = candidate_page.get("ocr_quality", {}) if isinstance(candidate_page.get("ocr_quality"), dict) else {}
    baseline_blocks = baseline_page.get("blocks", []) if isinstance(baseline_page.get("blocks"), list) else []
    candidate_blocks = candidate_page.get("blocks", []) if isinstance(candidate_page.get("blocks"), list) else []

    if len(baseline_blocks) != len(candidate_blocks):
        issues.append(
            f"block_count mismatch: baseline={len(baseline_blocks)} candidate={len(candidate_blocks)}"
        )

    matches, match_issues = _match_blocks(baseline_blocks, candidate_blocks)
    issues.extend(match_issues)

    text_mismatches: list[dict[str, Any]] = []
    for match in matches:
        baseline_block = baseline_blocks[match["baseline_index"]]
        candidate_block = candidate_blocks[match["candidate_index"]]
        baseline_text = str(baseline_block.get("normalized_text", "") or "")
        candidate_text = str(candidate_block.get("normalized_text", "") or "")
        if baseline_text != candidate_text:
            text_mismatches.append(
                {
                    "baseline_index": match["baseline_index"],
                    "candidate_index": match["candidate_index"],
                    "baseline_text": baseline_text,
                    "candidate_text": candidate_text,
                }
            )
    if text_mismatches:
        issues.append(f"text_mismatch_count={len(text_mismatches)}")

    if int(candidate_quality.get("non_empty", 0) or 0) < int(baseline_quality.get("non_empty", 0) or 0):
        issues.append(
            "non_empty regression: baseline={baseline} candidate={candidate}".format(
                baseline=baseline_quality.get("non_empty", 0),
                candidate=candidate_quality.get("non_empty", 0),
            )
        )
    if int(candidate_quality.get("empty", 0) or 0) > int(baseline_quality.get("empty", 0) or 0):
        issues.append(
            "empty regression: baseline={baseline} candidate={candidate}".format(
                baseline=baseline_quality.get("empty", 0),
                candidate=candidate_quality.get("empty", 0),
            )
        )
    if int(candidate_quality.get("single_char_like", 0) or 0) > int(
        baseline_quality.get("single_char_like", 0) or 0
    ):
        issues.append(
            "single_char_like regression: baseline={baseline} candidate={candidate}".format(
                baseline=baseline_quality.get("single_char_like", 0),
                candidate=candidate_quality.get("single_char_like", 0),
            )
        )

    return {
        "page": str(baseline_page.get("image_stem", "") or candidate_page.get("image_stem", "")),
        "passed": not issues,
        "issues": issues,
        "match_count": len(matches),
        "matches": matches,
        "text_mismatches": text_mismatches,
        "baseline_quality": baseline_quality,
        "candidate_quality": candidate_quality,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PaddleOCR-VL 1.5 Gold Compare",
        "",
        "## Summary",
        "",
        f"- baseline_gold: `{report['baseline_gold']}`",
        f"- candidate_run_dir: `{report['candidate_run_dir']}`",
        f"- pages_checked: `{report['pages_checked']}`",
        f"- failed_pages: `{report['failed_pages']}`",
        f"- passed: `{report['passed']}`",
        "",
        "## Pages",
        "",
        "| page | passed | match_count | issues |",
        "| --- | --- | --- | --- |",
    ]
    for page in report["pages"]:
        issues = "<br>".join(page.get("issues", []))
        lines.append(
            f"| {page['page']} | {page['passed']} | {page['match_count']} | {issues} |"
        )
    if report["rejection_reasons"]:
        lines.extend(["", "## Rejection Reasons", ""])
        for reason in report["rejection_reasons"]:
            lines.append(f"- {reason}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare PaddleOCR-VL 1.5 benchmark run output against baseline gold.")
    parser.add_argument("--baseline-gold", required=True)
    parser.add_argument("--candidate-run-dir", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    baseline_pages = _load_pages(args.baseline_gold)
    candidate_pages = _load_pages(args.candidate_run_dir)

    pages: list[dict[str, Any]] = []
    rejection_reasons: list[str] = []
    for page_name, baseline_page in baseline_pages.items():
        candidate_page = candidate_pages.get(page_name)
        if candidate_page is None:
            page_report = {
                "page": page_name,
                "passed": False,
                "issues": ["candidate page missing"],
                "match_count": 0,
                "matches": [],
                "text_mismatches": [],
                "baseline_quality": baseline_page.get("ocr_quality", {}),
                "candidate_quality": {},
            }
        else:
            page_report = _compare_page(baseline_page, candidate_page)
        if not page_report["passed"]:
            rejection_reasons.extend(f"{page_name}: {issue}" for issue in page_report["issues"])
        pages.append(page_report)

    report = {
        "baseline_gold": str(args.baseline_gold),
        "candidate_run_dir": str(args.candidate_run_dir),
        "pages_checked": len(pages),
        "failed_pages": sum(1 for page in pages if not page["passed"]),
        "passed": all(page["passed"] for page in pages),
        "pages": pages,
        "rejection_reasons": rejection_reasons,
    }

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
