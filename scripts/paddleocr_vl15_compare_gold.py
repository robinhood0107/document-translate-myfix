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
PROFILE_KIND = "paddleocr_vl15_warm_stable_v1"


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


def _normalize_text(text: object) -> str:
    return "".join(str(text or "").split())


def _quality_counts(page: dict[str, Any]) -> dict[str, int]:
    quality = page.get("ocr_quality", {}) if isinstance(page.get("ocr_quality"), dict) else {}
    return {
        "block_count": int(quality.get("block_count", 0) or 0),
        "non_empty": int(quality.get("non_empty", 0) or 0),
        "empty": int(quality.get("empty", 0) or 0),
        "single_char_like": int(quality.get("single_char_like", 0) or 0),
    }


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


def _select_screen_subset(stable_page_profiles: dict[str, dict[str, Any]], subset_size: int = 10) -> list[str]:
    ordered = sorted(
        stable_page_profiles.values(),
        key=lambda item: (int(item.get("block_count", 0) or 0), str(item.get("page", "") or "")),
    )
    page_names = [str(item.get("page", "") or "") for item in ordered if str(item.get("page", "") or "")]
    if len(page_names) <= subset_size:
        return page_names
    indexes: list[int] = []
    last_index = len(page_names) - 1
    for idx in range(subset_size):
        candidate_index = round(idx * last_index / max(1, subset_size - 1))
        if not indexes or candidate_index != indexes[-1]:
            indexes.append(candidate_index)
    return [page_names[index] for index in indexes]


def build_stability_profile(
    run_dirs: list[str | Path],
    *,
    baseline_sha: str = "",
    develop_ref_sha: str = "",
    execution_scope: str = "detect-ocr-only",
    official_score_scope: str = "detect+ocr-only",
) -> dict[str, Any]:
    if len(run_dirs) < 2:
        raise ValueError("Warm-stable profile requires at least two baseline runs.")

    named_runs: list[dict[str, Any]] = []
    run_pages: list[dict[str, dict[str, Any]]] = []
    for index, run_dir in enumerate(run_dirs):
        path = Path(run_dir)
        named_runs.append({"index": index, "run_dir": str(path)})
        run_pages.append(_load_pages(path))

    reference_pages = run_pages[0]
    stable_page_profiles: dict[str, dict[str, Any]] = {}
    excluded_unstable_pages: list[dict[str, Any]] = []

    for page_name in sorted(reference_pages):
        per_run_pages = [pages.get(page_name) for pages in run_pages]
        if any(page is None for page in per_run_pages):
            excluded_unstable_pages.append(
                {"page": page_name, "reason": "page missing in one or more baseline runs"}
            )
            continue
        resolved_pages = [page for page in per_run_pages if isinstance(page, dict)]
        if any(bool(page.get("page_failed")) for page in resolved_pages):
            excluded_unstable_pages.append(
                {"page": page_name, "reason": "page_failed present in baseline confirm"}
            )
            continue

        reference_page = resolved_pages[0]
        reference_blocks = reference_page.get("blocks", []) if isinstance(reference_page.get("blocks"), list) else []
        block_count = len(reference_blocks)
        structural_issues: list[str] = []
        match_maps: list[dict[int, int]] = []
        for other_page in resolved_pages[1:]:
            other_blocks = other_page.get("blocks", []) if isinstance(other_page.get("blocks"), list) else []
            if len(other_blocks) != block_count:
                structural_issues.append(
                    f"block_count mismatch: reference={block_count} candidate={len(other_blocks)}"
                )
                continue
            matches, match_issues = _match_blocks(reference_blocks, other_blocks)
            if match_issues:
                structural_issues.extend(match_issues)
                continue
            match_maps.append({int(item["baseline_index"]): int(item["candidate_index"]) for item in matches})

        if structural_issues:
            excluded_unstable_pages.append(
                {
                    "page": page_name,
                    "reason": "; ".join(dict.fromkeys(structural_issues)),
                }
            )
            continue

        stable_text_indices: list[int] = []
        excluded_unstable_block_indices: list[int] = []
        for block_index in range(block_count):
            texts = [
                _normalize_text(reference_blocks[block_index].get("normalized_text", "")),
            ]
            for run_index, other_page in enumerate(resolved_pages[1:]):
                other_blocks = other_page.get("blocks", []) if isinstance(other_page.get("blocks"), list) else []
                mapped_index = match_maps[run_index].get(block_index)
                if mapped_index is None or mapped_index >= len(other_blocks):
                    texts.append("")
                else:
                    texts.append(_normalize_text(other_blocks[mapped_index].get("normalized_text", "")))
            if len(set(texts)) == 1:
                stable_text_indices.append(block_index)
            else:
                excluded_unstable_block_indices.append(block_index)

        qualities = [_quality_counts(page) for page in resolved_pages]
        stable_page_profiles[page_name] = {
            "page": page_name,
            "block_count": block_count,
            "min_non_empty": min(item["non_empty"] for item in qualities),
            "max_empty": max(item["empty"] for item in qualities),
            "max_single_char_like": max(item["single_char_like"] for item in qualities),
            "stable_text_indices": stable_text_indices,
            "excluded_unstable_block_indices": excluded_unstable_block_indices,
            "reference_page": reference_page,
        }

    stable_pages = sorted(stable_page_profiles)
    screen_subset = _select_screen_subset(stable_page_profiles)
    return {
        "profile_kind": PROFILE_KIND,
        "generated_from_run_dirs": [str(Path(run_dir)) for run_dir in run_dirs],
        "baseline_sha": baseline_sha,
        "develop_ref_sha": develop_ref_sha,
        "execution_scope": execution_scope,
        "official_score_scope": official_score_scope,
        "stable_pages": stable_pages,
        "stable_page_count": len(stable_pages),
        "excluded_unstable_pages": excluded_unstable_pages,
        "excluded_unstable_blocks": {
            page_name: profile["excluded_unstable_block_indices"]
            for page_name, profile in stable_page_profiles.items()
            if profile["excluded_unstable_block_indices"]
        },
        "screen_subset": screen_subset,
        "page_profiles": stable_page_profiles,
    }


def _compare_page_legacy(
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

    baseline_quality = _quality_counts(baseline_page)
    candidate_quality = _quality_counts(candidate_page)
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
        baseline_text = _normalize_text(baseline_block.get("normalized_text", ""))
        candidate_text = _normalize_text(candidate_block.get("normalized_text", ""))
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

    if candidate_quality["non_empty"] < baseline_quality["non_empty"]:
        issues.append(
            "non_empty regression: baseline={baseline} candidate={candidate}".format(
                baseline=baseline_quality["non_empty"],
                candidate=candidate_quality["non_empty"],
            )
        )
    if candidate_quality["empty"] > baseline_quality["empty"]:
        issues.append(
            "empty regression: baseline={baseline} candidate={candidate}".format(
                baseline=baseline_quality["empty"],
                candidate=candidate_quality["empty"],
            )
        )
    if candidate_quality["single_char_like"] > baseline_quality["single_char_like"]:
        issues.append(
            "single_char_like regression: baseline={baseline} candidate={candidate}".format(
                baseline=baseline_quality["single_char_like"],
                candidate=candidate_quality["single_char_like"],
            )
        )

    return {
        "page": str(baseline_page.get("image_stem", "") or candidate_page.get("image_stem", "")),
        "passed": not issues,
        "detection_pass": not any(
            keyword in issue
            for issue in issues
            for keyword in ("block_count", "unmatched_", "page_failed mismatch", "candidate page missing")
        ),
        "ocr_pass": not any(
            keyword in issue
            for issue in issues
            for keyword in ("text_mismatch", "non_empty regression", "empty regression", "single_char_like regression")
        ),
        "hard_issues": issues,
        "soft_issues": [],
        "issues": issues,
        "match_count": len(matches),
        "matches": matches,
        "text_mismatches": text_mismatches,
        "baseline_quality": baseline_quality,
        "candidate_quality": candidate_quality,
        "hard_text_mismatch_count": len(text_mismatches),
        "soft_text_mismatch_count": 0,
    }


def compare_candidate_with_legacy_gold(
    baseline_gold: dict[str, Any],
    candidate_run_dir: str | Path,
    *,
    baseline_gold_path: str,
) -> dict[str, Any]:
    baseline_pages = _load_pages(Path(baseline_gold_path))
    candidate_pages = _load_pages(candidate_run_dir)

    pages: list[dict[str, Any]] = []
    rejection_reasons: list[str] = []
    for page_name, baseline_page in baseline_pages.items():
        candidate_page = candidate_pages.get(page_name)
        if candidate_page is None:
            page_report = {
                "page": page_name,
                "passed": False,
                "detection_pass": False,
                "ocr_pass": False,
                "hard_issues": ["candidate page missing"],
                "soft_issues": [],
                "issues": ["candidate page missing"],
                "match_count": 0,
                "matches": [],
                "text_mismatches": [],
                "baseline_quality": _quality_counts(baseline_page),
                "candidate_quality": {},
                "hard_text_mismatch_count": 0,
                "soft_text_mismatch_count": 0,
            }
        else:
            page_report = _compare_page_legacy(baseline_page, candidate_page)
        if not page_report["passed"]:
            rejection_reasons.extend(f"{page_name}: {issue}" for issue in page_report["issues"])
        pages.append(page_report)

    detection_pass = all(bool(page.get("detection_pass", False)) for page in pages)
    ocr_pass = all(bool(page.get("ocr_pass", False)) for page in pages)
    return {
        "compare_mode": "legacy-exact",
        "baseline_gold": baseline_gold_path,
        "candidate_run_dir": str(candidate_run_dir),
        "pages_checked": len(pages),
        "failed_pages": sum(1 for page in pages if not page["passed"]),
        "passed": all(page["passed"] for page in pages),
        "detection_pass": detection_pass,
        "ocr_pass": ocr_pass,
        "hard_text_mismatch_count": sum(int(page.get("hard_text_mismatch_count", 0) or 0) for page in pages),
        "soft_text_mismatch_count": 0,
        "excluded_unstable_pages": [],
        "excluded_unstable_blocks": {},
        "pages": pages,
        "rejection_reasons": rejection_reasons,
    }


def compare_candidate_with_profile(
    profile: dict[str, Any],
    candidate_run_dir: str | Path,
    *,
    expected_pages: list[str] | None = None,
    baseline_gold_path: str = "",
) -> dict[str, Any]:
    candidate_pages = _load_pages(candidate_run_dir)
    page_profiles = profile.get("page_profiles", {}) if isinstance(profile.get("page_profiles"), dict) else {}
    compare_pages = expected_pages or list(profile.get("stable_pages", []))

    pages: list[dict[str, Any]] = []
    rejection_reasons: list[str] = []
    hard_text_mismatch_count = 0
    soft_text_mismatch_count = 0
    detection_pass = True
    ocr_pass = True

    for page_name in compare_pages:
        page_profile = page_profiles.get(page_name)
        if not isinstance(page_profile, dict):
            continue
        candidate_page = candidate_pages.get(page_name)
        if candidate_page is None:
            page_report = {
                "page": page_name,
                "passed": False,
                "detection_pass": False,
                "ocr_pass": False,
                "hard_issues": ["candidate page missing"],
                "soft_issues": [],
                "issues": ["candidate page missing"],
                "match_count": 0,
                "matches": [],
                "text_mismatches": [],
                "baseline_quality": {
                    "non_empty": page_profile.get("min_non_empty", 0),
                    "empty": page_profile.get("max_empty", 0),
                    "single_char_like": page_profile.get("max_single_char_like", 0),
                },
                "candidate_quality": {},
                "hard_text_mismatch_count": 0,
                "soft_text_mismatch_count": 0,
            }
        else:
            reference_page = page_profile.get("reference_page", {})
            reference_blocks = reference_page.get("blocks", []) if isinstance(reference_page.get("blocks"), list) else []
            candidate_blocks = candidate_page.get("blocks", []) if isinstance(candidate_page.get("blocks"), list) else []
            candidate_quality = _quality_counts(candidate_page)
            hard_issues: list[str] = []
            soft_issues: list[str] = []
            text_mismatches: list[dict[str, Any]] = []
            stable_text_indices = {int(value) for value in page_profile.get("stable_text_indices", [])}

            if bool(candidate_page.get("page_failed")):
                hard_issues.append(
                    f"candidate_page_failed_reason={candidate_page.get('page_failed_reason', '')}"
                )

            if len(candidate_blocks) != int(page_profile.get("block_count", 0) or 0):
                hard_issues.append(
                    "block_count mismatch: baseline={baseline} candidate={candidate}".format(
                        baseline=page_profile.get("block_count", 0),
                        candidate=len(candidate_blocks),
                    )
                )

            matches, match_issues = _match_blocks(reference_blocks, candidate_blocks)
            hard_issues.extend(match_issues)

            if candidate_quality["non_empty"] < int(page_profile.get("min_non_empty", 0) or 0):
                hard_issues.append(
                    "non_empty regression: baseline_min={baseline} candidate={candidate}".format(
                        baseline=page_profile.get("min_non_empty", 0),
                        candidate=candidate_quality["non_empty"],
                    )
                )
            if candidate_quality["empty"] > int(page_profile.get("max_empty", 0) or 0):
                hard_issues.append(
                    "empty regression: baseline_max={baseline} candidate={candidate}".format(
                        baseline=page_profile.get("max_empty", 0),
                        candidate=candidate_quality["empty"],
                    )
                )
            if candidate_quality["single_char_like"] > int(page_profile.get("max_single_char_like", 0) or 0):
                hard_issues.append(
                    "single_char_like regression: baseline_max={baseline} candidate={candidate}".format(
                        baseline=page_profile.get("max_single_char_like", 0),
                        candidate=candidate_quality["single_char_like"],
                    )
                )

            for match in matches:
                baseline_index = int(match["baseline_index"])
                candidate_index = int(match["candidate_index"])
                baseline_text = _normalize_text(reference_blocks[baseline_index].get("normalized_text", ""))
                candidate_text = _normalize_text(candidate_blocks[candidate_index].get("normalized_text", ""))
                if baseline_text == candidate_text:
                    continue
                mismatch = {
                    "baseline_index": baseline_index,
                    "candidate_index": candidate_index,
                    "baseline_text": baseline_text,
                    "candidate_text": candidate_text,
                    "stable_text_gate": baseline_index in stable_text_indices,
                }
                text_mismatches.append(mismatch)
                if baseline_index in stable_text_indices:
                    hard_issues.append(f"hard_text_mismatch index={baseline_index}")
                else:
                    soft_issues.append(f"soft_text_mismatch index={baseline_index}")

            page_report = {
                "page": page_name,
                "passed": not hard_issues,
                "detection_pass": not any(
                    keyword in issue
                    for issue in hard_issues
                    for keyword in ("block_count", "unmatched_", "candidate page missing", "page_failed")
                ),
                "ocr_pass": not any(
                    keyword in issue
                    for issue in hard_issues
                    for keyword in ("non_empty regression", "empty regression", "single_char_like regression", "hard_text_mismatch")
                ),
                "hard_issues": hard_issues,
                "soft_issues": soft_issues,
                "issues": hard_issues + soft_issues,
                "match_count": len(matches),
                "matches": matches,
                "text_mismatches": text_mismatches,
                "baseline_quality": {
                    "min_non_empty": int(page_profile.get("min_non_empty", 0) or 0),
                    "max_empty": int(page_profile.get("max_empty", 0) or 0),
                    "max_single_char_like": int(page_profile.get("max_single_char_like", 0) or 0),
                },
                "candidate_quality": candidate_quality,
                "hard_text_mismatch_count": sum(1 for item in text_mismatches if item["stable_text_gate"]),
                "soft_text_mismatch_count": sum(1 for item in text_mismatches if not item["stable_text_gate"]),
            }

        if not page_report["passed"]:
            rejection_reasons.extend(f"{page_name}: {issue}" for issue in page_report["hard_issues"])
        hard_text_mismatch_count += int(page_report.get("hard_text_mismatch_count", 0) or 0)
        soft_text_mismatch_count += int(page_report.get("soft_text_mismatch_count", 0) or 0)
        detection_pass = detection_pass and bool(page_report.get("detection_pass", False))
        ocr_pass = ocr_pass and bool(page_report.get("ocr_pass", False))
        pages.append(page_report)

    return {
        "compare_mode": "warm-stable-gate",
        "baseline_gold": baseline_gold_path,
        "candidate_run_dir": str(candidate_run_dir),
        "pages_checked": len(pages),
        "failed_pages": sum(1 for page in pages if not page["passed"]),
        "passed": all(page["passed"] for page in pages),
        "detection_pass": detection_pass,
        "ocr_pass": ocr_pass,
        "hard_text_mismatch_count": hard_text_mismatch_count,
        "soft_text_mismatch_count": soft_text_mismatch_count,
        "excluded_unstable_pages": profile.get("excluded_unstable_pages", []),
        "excluded_unstable_blocks": profile.get("excluded_unstable_blocks", {}),
        "screen_subset": profile.get("screen_subset", []),
        "stable_pages": profile.get("stable_pages", []),
        "pages": pages,
        "rejection_reasons": rejection_reasons,
    }


def compare_candidate(
    baseline_gold_path: str | Path,
    candidate_run_dir: str | Path,
    *,
    expected_pages: list[str] | None = None,
) -> dict[str, Any]:
    baseline_gold = _load_json(Path(baseline_gold_path))
    if str(baseline_gold.get("profile_kind", "") or "") == PROFILE_KIND:
        return compare_candidate_with_profile(
            baseline_gold,
            candidate_run_dir,
            expected_pages=expected_pages,
            baseline_gold_path=str(baseline_gold_path),
        )
    return compare_candidate_with_legacy_gold(
        baseline_gold,
        candidate_run_dir,
        baseline_gold_path=str(baseline_gold_path),
    )


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PaddleOCR-VL 1.5 Gold Compare",
        "",
        "## Summary",
        "",
        f"- compare_mode: `{report.get('compare_mode', '')}`",
        f"- baseline_gold: `{report['baseline_gold']}`",
        f"- candidate_run_dir: `{report['candidate_run_dir']}`",
        f"- pages_checked: `{report['pages_checked']}`",
        f"- failed_pages: `{report['failed_pages']}`",
        f"- passed: `{report['passed']}`",
        f"- detection_pass: `{report.get('detection_pass', False)}`",
        f"- ocr_pass: `{report.get('ocr_pass', False)}`",
        f"- hard_text_mismatch_count: `{report.get('hard_text_mismatch_count', 0)}`",
        f"- soft_text_mismatch_count: `{report.get('soft_text_mismatch_count', 0)}`",
    ]
    excluded_pages = report.get("excluded_unstable_pages", [])
    if excluded_pages:
        lines.extend(["", "## Excluded Unstable Pages", ""])
        for item in excluded_pages:
            if isinstance(item, dict):
                lines.append(f"- `{item.get('page', '')}`: {item.get('reason', '')}")
            else:
                lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            "## Pages",
            "",
            "| page | passed | detection_pass | ocr_pass | hard_text | soft_text | hard issues | soft issues |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for page in report["pages"]:
        hard_issues = "<br>".join(page.get("hard_issues", []))
        soft_issues = "<br>".join(page.get("soft_issues", []))
        lines.append(
            "| {page} | {passed} | {detection_pass} | {ocr_pass} | {hard_text} | {soft_text} | {hard_issues} | {soft_issues} |".format(
                page=page["page"],
                passed=page["passed"],
                detection_pass=page.get("detection_pass", False),
                ocr_pass=page.get("ocr_pass", False),
                hard_text=page.get("hard_text_mismatch_count", 0),
                soft_text=page.get("soft_text_mismatch_count", 0),
                hard_issues=hard_issues,
                soft_issues=soft_issues,
            )
        )
    if report["rejection_reasons"]:
        lines.extend(["", "## Rejection Reasons", ""])
        for reason in report["rejection_reasons"]:
            lines.append(f"- {reason}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare PaddleOCR-VL 1.5 benchmark run output against baseline gold or warm-stable profile.")
    parser.add_argument("--baseline-gold", required=True)
    parser.add_argument("--candidate-run-dir", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--expected-pages", nargs="*", default=[])
    args = parser.parse_args()

    report = compare_candidate(
        args.baseline_gold,
        args.candidate_run_dir,
        expected_pages=list(args.expected_pages or []) or None,
    )
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
