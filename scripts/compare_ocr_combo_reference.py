#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_common import write_json

XYXY_IOU_THRESHOLD = 0.90
BUBBLE_IOU_THRESHOLD = 0.85
GEOMETRY_RECALL_THRESHOLD = 0.98
GEOMETRY_PRECISION_THRESHOLD = 0.98
NON_EMPTY_RETENTION_THRESHOLD = 0.98
OCR_CHAR_ERROR_RATE_THRESHOLD = 0.02
PAGE_P95_OCR_CHAR_ERROR_RATE_THRESHOLD = 0.05
EMPTY_RATE_DELTA_THRESHOLD = 0.02
SINGLE_CHAR_RATE_DELTA_THRESHOLD = 0.02
IGNORABLE_OCR_CHARS = {"「", "」", "『", "』", ",", "，", "、", "♡", "♥"}
OCR_CANONICAL_REPLACEMENTS = {
    "あ゙": "あ",
    "ぁ": "あ",
    "い゙": "い",
    "ぃ": "い",
    "ゔ": "う",
    "ゔ": "う",
    "ぅ": "う",
    "え゙": "え",
    "ぇ": "え",
    "お゙": "お",
    "ぉ": "お",
    "ん゙": "ん",
    "ア゙": "ア",
    "ァ": "ア",
    "イ゙": "イ",
    "ィ": "イ",
    "ヴ": "ウ",
    "ヴ": "ウ",
    "ゥ": "ウ",
    "エ゙": "エ",
    "ェ": "エ",
    "オ゙": "オ",
    "ォ": "オ",
    "ン゙": "ン",
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _normalize_text(text: object) -> str:
    return "".join(str(text or "").split())


def _normalize_translation_text(text: object) -> str:
    return _normalize_text(text)


def _normalize_ocr_text(text: object) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        return ""
    normalized = unicodedata.normalize("NFD", normalized)
    for source, target in OCR_CANONICAL_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    normalized = "".join(ch for ch in normalized if ch not in IGNORABLE_OCR_CHARS)
    return unicodedata.normalize("NFC", normalized)


def _load_pages(run_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = _load_json(run_dir / "page_snapshots.json")
    pages = payload.get("pages", [])
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(pages, list):
        raise ValueError(f"Expected pages list in {run_dir / 'page_snapshots.json'}")
    for page in pages:
        if not isinstance(page, dict):
            continue
        image_stem = str(page.get("image_stem", "") or "")
        if image_stem:
            result[image_stem] = page
    return payload, result


def _load_summary(run_dir: Path) -> dict[str, Any]:
    return _load_json(run_dir / "summary.json")


def _page_quality(page: dict[str, Any]) -> dict[str, int]:
    blocks = page.get("blocks", []) if isinstance(page.get("blocks"), list) else []
    valid_blocks = [block for block in blocks if isinstance(block, dict)]
    total = len(valid_blocks)
    non_empty = 0
    empty = 0
    single_char_like = 0
    for block in valid_blocks:
        normalized = _normalize_ocr_text(block.get("normalized_text", block.get("text", "")))
        if normalized:
            non_empty += 1
        else:
            empty += 1
        if 0 < len(normalized) <= 1:
            single_char_like += 1
    return {
        "block_count": total,
        "non_empty": non_empty,
        "empty": empty,
        "single_char_like": single_char_like,
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


def _as_box(value: object) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, left_ch in enumerate(left, start=1):
        curr = [i]
        for j, right_ch in enumerate(right, start=1):
            insert_cost = curr[j - 1] + 1
            delete_cost = prev[j] + 1
            replace_cost = prev[j - 1] + (0 if left_ch == right_ch else 1)
            curr.append(min(insert_cost, delete_cost, replace_cost))
        prev = curr
    return prev[-1]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    rank = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * p)))
    return ordered[rank]


def _single_char_like(text: str) -> bool:
    return 0 < len(_normalize_ocr_text(text)) <= 1


def _load_gold_pages(gold_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = _load_json(gold_path)
    pages = payload.get("pages", [])
    if not isinstance(pages, list):
        raise ValueError(f"Expected pages list in {gold_path}")
    result: dict[str, dict[str, Any]] = {}
    for page in pages:
        if not isinstance(page, dict):
            continue
        image_stem = str(page.get("image_stem", "") or "")
        if image_stem:
            result[image_stem] = page
    return payload, result


def _gold_page_blocks(page: dict[str, Any]) -> list[dict[str, Any]]:
    if str(page.get("status", "active") or "active").lower() == "excluded":
        return []
    blocks = page.get("blocks", [])
    if not isinstance(blocks, list):
        return []
    return [block for block in blocks if isinstance(block, dict)]


def _gold_page_quality(page: dict[str, Any]) -> dict[str, int]:
    blocks = _gold_page_blocks(page)
    total = len(blocks)
    non_empty = 0
    empty = 0
    single_char_like = 0
    for block in blocks:
        gold_text = str(block.get("gold_text", block.get("seed_text", "")) or "")
        normalized = _normalize_ocr_text(gold_text)
        if normalized:
            non_empty += 1
        else:
            empty += 1
        if _single_char_like(gold_text):
            single_char_like += 1
    return {
        "block_count": total,
        "non_empty": non_empty,
        "empty": empty,
        "single_char_like": single_char_like,
    }


def _match_blocks(
    gold_blocks: list[dict[str, Any]],
    candidate_blocks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[int], list[int]]:
    scores: list[tuple[float, float, int, int, str]] = []
    for gold_index, gold_block in enumerate(gold_blocks):
        for cand_index, cand_block in enumerate(candidate_blocks):
            gold_xyxy = _as_box(gold_block.get("xyxy"))
            cand_xyxy = _as_box(cand_block.get("xyxy"))
            if gold_xyxy and cand_xyxy:
                xyxy_score = _xyxy_iou(gold_xyxy, cand_xyxy)
                if xyxy_score >= XYXY_IOU_THRESHOLD:
                    scores.append((2.0, xyxy_score, gold_index, cand_index, "xyxy"))
                continue

            gold_bubble = _as_box(gold_block.get("bubble_xyxy"))
            cand_bubble = _as_box(cand_block.get("bubble_xyxy"))
            if gold_bubble and cand_bubble:
                bubble_score = _xyxy_iou(gold_bubble, cand_bubble)
                if bubble_score >= BUBBLE_IOU_THRESHOLD:
                    scores.append((1.0, bubble_score, gold_index, cand_index, "bubble"))

    matches: list[dict[str, Any]] = []
    used_gold: set[int] = set()
    used_cand: set[int] = set()
    for _, score, gold_index, cand_index, match_kind in sorted(scores, reverse=True):
        if gold_index in used_gold or cand_index in used_cand:
            continue
        used_gold.add(gold_index)
        used_cand.add(cand_index)
        matches.append(
            {
                "gold_index": gold_index,
                "candidate_index": cand_index,
                "iou": round(score, 4),
                "match_kind": match_kind,
            }
        )

    unmatched_gold = sorted(set(range(len(gold_blocks))) - used_gold)
    unmatched_candidate = sorted(set(range(len(candidate_blocks))) - used_cand)
    issues: list[str] = []
    if unmatched_gold:
        issues.append(f"unmatched_gold_blocks={unmatched_gold}")
    if unmatched_candidate:
        issues.append(f"unmatched_candidate_blocks={unmatched_candidate}")
    return sorted(matches, key=lambda item: item["gold_index"]), issues, unmatched_gold, unmatched_candidate


def compare_runs(gold_path: Path, candidate_run_dir: Path) -> dict[str, Any]:
    gold_payload, gold_pages = _load_gold_pages(gold_path)
    candidate_summary = _load_summary(candidate_run_dir)
    _, candidate_pages = _load_pages(candidate_run_dir)

    total_gold_blocks = 0
    total_gold_non_empty = 0
    total_candidate_non_empty = 0
    matched_gold_blocks = 0
    matched_candidate_non_empty = 0
    matched_gold_non_empty = 0
    ocr_exact_matches = 0
    translation_exact_matches = 0
    translation_similarity_values: list[float] = []
    page_ocr_char_error_rates: list[float] = []
    total_char_errors = 0
    total_gold_chars = 0
    unmatched_candidate_block_count = 0
    overgenerated_block_count = 0
    excluded_unstable_pages: list[str] = []

    gold_quality = {"block_count": 0, "non_empty": 0, "empty": 0, "single_char_like": 0}
    candidate_quality = {"block_count": 0, "non_empty": 0, "empty": 0, "single_char_like": 0}
    page_results: list[dict[str, Any]] = []

    for page_name in sorted(gold_pages):
        gold_page = gold_pages[page_name]
        if str(gold_page.get("status", "active") or "active").lower() == "excluded":
            excluded_unstable_pages.append(page_name)
            continue

        candidate_page = candidate_pages.get(page_name)
        gold_page_quality = _gold_page_quality(gold_page)
        for key, value in gold_page_quality.items():
            gold_quality[key] += value

        gold_blocks = _gold_page_blocks(gold_page)
        total_gold_blocks += len(gold_blocks)
        gold_non_empty_indices = {
            idx
            for idx, block in enumerate(gold_blocks)
            if _normalize_ocr_text(block.get("gold_text", block.get("seed_text", "")))
        }
        total_gold_non_empty += len(gold_non_empty_indices)

        if not isinstance(candidate_page, dict):
            total_gold_chars += sum(
                len(_normalize_ocr_text(gold_blocks[idx].get("gold_text", gold_blocks[idx].get("seed_text", ""))))
                for idx in gold_non_empty_indices
            )
            page_results.append(
                {
                    "page": page_name,
                    "pass": False,
                    "reason": "candidate page missing",
                    "page_ocr_char_error_rate": 1.0,
                }
            )
            page_ocr_char_error_rates.append(1.0)
            continue

        candidate_page_quality = _page_quality(candidate_page)
        for key, value in candidate_page_quality.items():
            candidate_quality[key] += value

        candidate_blocks = (
            candidate_page.get("blocks", []) if isinstance(candidate_page.get("blocks"), list) else []
        )
        candidate_non_empty_indices = {
            idx
            for idx, block in enumerate(candidate_blocks)
            if _normalize_ocr_text(block.get("normalized_text", block.get("text", "")))
        }
        total_candidate_non_empty += len(candidate_non_empty_indices)

        matches, issues, unmatched_gold, unmatched_candidate = _match_blocks(gold_blocks, candidate_blocks)
        matched_gold_blocks += len(matches)
        unmatched_candidate_block_count += len(unmatched_candidate)

        match_map = {int(item["gold_index"]): int(item["candidate_index"]) for item in matches}

        page_char_errors = 0
        page_gold_chars = 0
        page_ocr_exact_matches = 0
        page_translation_similarity_values: list[float] = []
        page_translation_exact_matches = 0

        for gold_index in gold_non_empty_indices:
            gold_block = gold_blocks[gold_index]
            gold_text_raw = str(gold_block.get("gold_text", gold_block.get("seed_text", "")) or "")
            gold_text = _normalize_ocr_text(gold_text_raw)
            gold_char_count = max(1, len(gold_text))
            page_gold_chars += gold_char_count
            total_gold_chars += gold_char_count

            candidate_index = match_map.get(gold_index)
            if candidate_index is None:
                page_char_errors += gold_char_count
                total_char_errors += gold_char_count
                continue

            candidate_block = candidate_blocks[candidate_index]
            candidate_text = _normalize_ocr_text(candidate_block.get("normalized_text", candidate_block.get("text", "")))
            if candidate_text:
                matched_candidate_non_empty += 1
            matched_gold_non_empty += 1

            char_errors = _levenshtein_distance(gold_text, candidate_text)
            page_char_errors += char_errors
            total_char_errors += char_errors
            if gold_text == candidate_text:
                ocr_exact_matches += 1
                page_ocr_exact_matches += 1

            seed_translation = _normalize_translation_text(
                gold_block.get("seed_normalized_translation", gold_block.get("seed_translation", ""))
            )
            candidate_translation = _normalize_translation_text(candidate_block.get("normalized_translation", ""))
            if seed_translation or candidate_translation:
                if seed_translation == candidate_translation:
                    translation_exact_matches += 1
                    page_translation_exact_matches += 1
                page_translation_similarity_values.append(
                    SequenceMatcher(None, seed_translation, candidate_translation).ratio()
                )
                translation_similarity_values.append(
                    SequenceMatcher(None, seed_translation, candidate_translation).ratio()
                )

        for candidate_index in unmatched_candidate:
            candidate_block = candidate_blocks[candidate_index]
            if _normalize_ocr_text(candidate_block.get("normalized_text", candidate_block.get("text", ""))):
                overgenerated_block_count += 1

        page_cer = page_char_errors / page_gold_chars if page_gold_chars else 0.0
        page_ocr_char_error_rates.append(page_cer)
        page_results.append(
            {
                "page": page_name,
                "pass": not issues,
                "issues": issues,
                "gold_block_count": len(gold_blocks),
                "candidate_block_count": len(candidate_blocks),
                "matched_block_count": len(matches),
                "gold_non_empty": gold_page_quality["non_empty"],
                "candidate_non_empty": candidate_page_quality["non_empty"],
                "page_ocr_char_error_rate": round(page_cer, 4),
                "page_ocr_exact_match_ratio": round(
                    page_ocr_exact_matches / len(gold_non_empty_indices), 4
                )
                if gold_non_empty_indices
                else 1.0,
                "page_translation_exact_match_ratio": round(
                    page_translation_exact_matches / len(page_translation_similarity_values), 4
                )
                if page_translation_similarity_values
                else None,
                "page_translation_similarity_avg": round(
                    sum(page_translation_similarity_values) / len(page_translation_similarity_values), 4
                )
                if page_translation_similarity_values
                else None,
            }
        )

    geometry_match_recall = matched_gold_blocks / total_gold_blocks if total_gold_blocks else 1.0
    geometry_match_precision = (
        matched_gold_blocks / candidate_quality["block_count"] if candidate_quality["block_count"] else 1.0
    )
    non_empty_retention = (
        matched_candidate_non_empty / total_gold_non_empty if total_gold_non_empty else 1.0
    )
    gold_empty_rate = gold_quality["empty"] / gold_quality["block_count"] if gold_quality["block_count"] else 0.0
    candidate_empty_rate = (
        candidate_quality["empty"] / candidate_quality["block_count"] if candidate_quality["block_count"] else 0.0
    )
    gold_single_char_like_rate = (
        gold_quality["single_char_like"] / gold_quality["block_count"] if gold_quality["block_count"] else 0.0
    )
    candidate_single_char_like_rate = (
        candidate_quality["single_char_like"] / candidate_quality["block_count"]
        if candidate_quality["block_count"]
        else 0.0
    )
    ocr_char_error_rate = total_char_errors / total_gold_chars if total_gold_chars else 0.0
    page_p95_ocr_char_error_rate = _percentile(page_ocr_char_error_rates, 0.95)
    ocr_exact_text_match_ratio = (
        ocr_exact_matches / total_gold_non_empty if total_gold_non_empty else 1.0
    )
    translation_exact_match_ratio = (
        translation_exact_matches / len(translation_similarity_values) if translation_similarity_values else None
    )
    translation_similarity_ratio = (
        sum(translation_similarity_values) / len(translation_similarity_values)
        if translation_similarity_values
        else None
    )

    hard_gate_failures: list[str] = []
    for field in (
        "page_failed_count",
        "gemma_truncated_count",
        "gemma_empty_content_count",
        "gemma_missing_key_count",
        "gemma_schema_validation_fail_count",
        "ocr_cache_hit_count",
    ):
        if int(candidate_summary.get(field, 0) or 0) != 0:
            hard_gate_failures.append(f"{field}={candidate_summary.get(field)}")
    if geometry_match_recall < GEOMETRY_RECALL_THRESHOLD:
        hard_gate_failures.append(
            f"geometry_match_recall<{GEOMETRY_RECALL_THRESHOLD}: {geometry_match_recall:.4f}"
        )
    if geometry_match_precision < GEOMETRY_PRECISION_THRESHOLD:
        hard_gate_failures.append(
            f"geometry_match_precision<{GEOMETRY_PRECISION_THRESHOLD}: {geometry_match_precision:.4f}"
        )
    if non_empty_retention < NON_EMPTY_RETENTION_THRESHOLD:
        hard_gate_failures.append(
            f"non_empty_retention<{NON_EMPTY_RETENTION_THRESHOLD}: {non_empty_retention:.4f}"
        )
    if candidate_empty_rate > gold_empty_rate + EMPTY_RATE_DELTA_THRESHOLD:
        hard_gate_failures.append(
            f"candidate_empty_rate>{gold_empty_rate + EMPTY_RATE_DELTA_THRESHOLD:.4f}: {candidate_empty_rate:.4f}"
        )
    if candidate_single_char_like_rate > gold_single_char_like_rate + SINGLE_CHAR_RATE_DELTA_THRESHOLD:
        hard_gate_failures.append(
            "candidate_single_char_like_rate>"
            f"{gold_single_char_like_rate + SINGLE_CHAR_RATE_DELTA_THRESHOLD:.4f}: "
            f"{candidate_single_char_like_rate:.4f}"
        )
    if ocr_char_error_rate > OCR_CHAR_ERROR_RATE_THRESHOLD:
        hard_gate_failures.append(
            f"ocr_char_error_rate>{OCR_CHAR_ERROR_RATE_THRESHOLD}: {ocr_char_error_rate:.4f}"
        )
    if page_p95_ocr_char_error_rate > PAGE_P95_OCR_CHAR_ERROR_RATE_THRESHOLD:
        hard_gate_failures.append(
            "page_p95_ocr_char_error_rate>"
            f"{PAGE_P95_OCR_CHAR_ERROR_RATE_THRESHOLD}: {page_p95_ocr_char_error_rate:.4f}"
        )
    if overgenerated_block_count != 0:
        hard_gate_failures.append(f"overgenerated_block_count={overgenerated_block_count}")

    return {
        "gold_path": str(gold_path),
        "candidate_run_dir": str(candidate_run_dir),
        "quality_gate_pass": not hard_gate_failures,
        "hard_gate_failures": hard_gate_failures,
        "gold_metadata": {
            "corpus": gold_payload.get("corpus", ""),
            "review_status": gold_payload.get("review_status", ""),
            "generated_from_run_dir": gold_payload.get("generated_from_run_dir", ""),
        },
        "ocr_normalization": {
            "canonical_small_voiced_kana": True,
            "ignored_chars": "".join(sorted(IGNORABLE_OCR_CHARS)),
            "gold_empty_text_policy": "geometry-kept-text-skipped",
        },
        "candidate_summary": candidate_summary,
        "metrics": {
            "gold_block_count": gold_quality["block_count"],
            "candidate_block_count": candidate_quality["block_count"],
            "gold_non_empty_count": gold_quality["non_empty"],
            "candidate_non_empty_count": candidate_quality["non_empty"],
            "geometry_match_recall": round(geometry_match_recall, 4),
            "geometry_match_precision": round(geometry_match_precision, 4),
            "non_empty_retention": round(non_empty_retention, 4),
            "gold_empty_rate": round(gold_empty_rate, 4),
            "candidate_empty_rate": round(candidate_empty_rate, 4),
            "gold_single_char_like_rate": round(gold_single_char_like_rate, 4),
            "candidate_single_char_like_rate": round(candidate_single_char_like_rate, 4),
            "ocr_char_error_rate": round(ocr_char_error_rate, 4),
            "page_p95_ocr_char_error_rate": round(page_p95_ocr_char_error_rate, 4),
            "ocr_exact_text_match_ratio": round(ocr_exact_text_match_ratio, 4),
            "translation_exact_match_ratio": round(translation_exact_match_ratio, 4)
            if translation_exact_match_ratio is not None
            else None,
            "translation_similarity_ratio": round(translation_similarity_ratio, 4)
            if translation_similarity_ratio is not None
            else None,
            "unmatched_candidate_block_count": unmatched_candidate_block_count,
            "overgenerated_block_count": overgenerated_block_count,
            "excluded_unstable_page_count": len(excluded_unstable_pages),
            "excluded_unstable_pages": excluded_unstable_pages,
        },
        "page_results": page_results,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    metrics = payload.get("metrics", {})
    lines = [
        "# OCR Combo Compare",
        "",
        f"- quality_gate_pass: `{payload.get('quality_gate_pass')}`",
        f"- gold_path: `{payload.get('gold_path')}`",
        f"- candidate_run_dir: `{payload.get('candidate_run_dir')}`",
        f"- canonical_small_voiced_kana: `{(payload.get('ocr_normalization') or {}).get('canonical_small_voiced_kana')}`",
        f"- ignored_chars: `{(payload.get('ocr_normalization') or {}).get('ignored_chars')}`",
        f"- gold_empty_text_policy: `{(payload.get('ocr_normalization') or {}).get('gold_empty_text_policy')}`",
        f"- geometry_match_recall: `{metrics.get('geometry_match_recall')}`",
        f"- geometry_match_precision: `{metrics.get('geometry_match_precision')}`",
        f"- non_empty_retention: `{metrics.get('non_empty_retention')}`",
        f"- candidate_empty_rate: `{metrics.get('candidate_empty_rate')}`",
        f"- candidate_single_char_like_rate: `{metrics.get('candidate_single_char_like_rate')}`",
        f"- ocr_char_error_rate: `{metrics.get('ocr_char_error_rate')}`",
        f"- page_p95_ocr_char_error_rate: `{metrics.get('page_p95_ocr_char_error_rate')}`",
        f"- overgenerated_block_count: `{metrics.get('overgenerated_block_count')}`",
        "",
        "## Hard Gate Failures",
        "",
    ]
    failures = payload.get("hard_gate_failures", [])
    if failures:
        lines.extend(f"- {item}" for item in failures)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Page Results",
            "",
            "| page | pass | matched_block_count | page_ocr_char_error_rate | issues |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload.get("page_results", []):
        issues = row.get("issues", [])
        if isinstance(issues, list):
            issue_text = "; ".join(str(item) for item in issues)
        else:
            issue_text = str(row.get("reason", "") or "")
        lines.append(
            "| {page} | {passed} | {matched} | {cer} | {issues} |".format(
                page=row.get("page", ""),
                passed=row.get("pass", False),
                matched=row.get("matched_block_count", ""),
                cer=row.get("page_ocr_char_error_rate", ""),
                issues=issue_text,
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare OCR combo candidate run against locked OCR gold.")
    parser.add_argument("--gold-path", required=True)
    parser.add_argument("--candidate-run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gold_path = Path(args.gold_path)
    candidate_run_dir = Path(args.candidate_run_dir)
    output_path = Path(args.output)

    payload = compare_runs(gold_path, candidate_run_dir)
    write_json(output_path, payload)
    output_path.with_suffix(".md").write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({"quality_gate_pass": payload["quality_gate_pass"], "output": str(output_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
