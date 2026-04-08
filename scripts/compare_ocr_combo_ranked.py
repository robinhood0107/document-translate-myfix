#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_common import write_json
from compare_ocr_combo_reference import (
    _gold_page_blocks,
    _gold_page_quality,
    _levenshtein_distance,
    _load_gold_pages,
    _load_pages,
    _load_summary,
    _match_blocks,
    _normalize_ocr_text,
    _normalize_translation_text,
    _page_quality,
    _percentile,
)

READY_GEOMETRY_RECALL_THRESHOLD = 0.995
READY_GEOMETRY_PRECISION_THRESHOLD = 0.995
READY_NON_EMPTY_RETENTION_THRESHOLD = 0.98
READY_OCR_CHAR_ERROR_RATE_THRESHOLD = 0.08
READY_PAGE_P95_OCR_CHAR_ERROR_RATE_THRESHOLD = 0.18
READY_PAGE_FAILED_RATE_THRESHOLD = 0.05

CONDITIONAL_GEOMETRY_RECALL_THRESHOLD = 0.99
CONDITIONAL_GEOMETRY_PRECISION_THRESHOLD = 0.99
CONDITIONAL_NON_EMPTY_RETENTION_THRESHOLD = 0.98
CONDITIONAL_OCR_CHAR_ERROR_RATE_THRESHOLD = 0.20
CONDITIONAL_PAGE_P95_OCR_CHAR_ERROR_RATE_THRESHOLD = 0.35
CONDITIONAL_PAGE_FAILED_RATE_THRESHOLD = 0.10

CATASTROPHIC_GEOMETRY_RECALL_THRESHOLD = 0.98
CATASTROPHIC_GEOMETRY_PRECISION_THRESHOLD = 0.98
CATASTROPHIC_NON_EMPTY_RETENTION_THRESHOLD = 0.97
CATASTROPHIC_OCR_CHAR_ERROR_RATE_THRESHOLD = 0.35

QUALITY_BAND_RANK = {
    "catastrophic": 0,
    "hold": 1,
    "conditional": 2,
    "ready": 3,
}


def _normalize_for_words(text: object) -> str:
    return _normalize_ocr_text(text)


def _tokenize_words(text: object) -> list[str]:
    normalized = _normalize_for_words(text)
    if not normalized:
        return []
    return list(normalized)


def _sequence_levenshtein(left: list[str], right: list[str]) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        curr = [i]
        for j, right_item in enumerate(right, start=1):
            insert_cost = curr[j - 1] + 1
            delete_cost = prev[j] + 1
            replace_cost = prev[j - 1] + (0 if left_item == right_item else 1)
            curr.append(min(insert_cost, delete_cost, replace_cost))
        prev = curr
    return prev[-1]


def _page_denominator(candidate_summary: dict[str, Any], active_gold_pages: int) -> int:
    count = int(candidate_summary.get("image_count", 0) or 0)
    if count > 0:
        return count
    done = int(candidate_summary.get("page_done_count", 0) or 0)
    failed = int(candidate_summary.get("page_failed_count", 0) or 0)
    if done + failed > 0:
        return done + failed
    return max(1, active_gold_pages)


def _classify_band(metrics: dict[str, float], summary: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    page_failed_rate = float(metrics.get("page_failed_rate", 0.0) or 0.0)
    geometry_match_recall = float(metrics.get("geometry_match_recall", 0.0) or 0.0)
    geometry_match_precision = float(metrics.get("geometry_match_precision", 0.0) or 0.0)
    non_empty_retention = float(metrics.get("non_empty_retention", 0.0) or 0.0)
    ocr_char_error_rate = float(metrics.get("ocr_char_error_rate", 1.0) or 1.0)
    page_p95_ocr_char_error_rate = float(metrics.get("page_p95_ocr_char_error_rate", 1.0) or 1.0)

    catastrophic_reasons: list[str] = []
    if geometry_match_recall < CATASTROPHIC_GEOMETRY_RECALL_THRESHOLD:
        catastrophic_reasons.append(
            f"geometry_match_recall<{CATASTROPHIC_GEOMETRY_RECALL_THRESHOLD}: {geometry_match_recall:.4f}"
        )
    if geometry_match_precision < CATASTROPHIC_GEOMETRY_PRECISION_THRESHOLD:
        catastrophic_reasons.append(
            f"geometry_match_precision<{CATASTROPHIC_GEOMETRY_PRECISION_THRESHOLD}: {geometry_match_precision:.4f}"
        )
    if non_empty_retention < CATASTROPHIC_NON_EMPTY_RETENTION_THRESHOLD:
        catastrophic_reasons.append(
            f"non_empty_retention<{CATASTROPHIC_NON_EMPTY_RETENTION_THRESHOLD}: {non_empty_retention:.4f}"
        )
    if ocr_char_error_rate > CATASTROPHIC_OCR_CHAR_ERROR_RATE_THRESHOLD:
        catastrophic_reasons.append(
            f"ocr_char_error_rate>{CATASTROPHIC_OCR_CHAR_ERROR_RATE_THRESHOLD}: {ocr_char_error_rate:.4f}"
        )
    if catastrophic_reasons:
        return "catastrophic", catastrophic_reasons, []

    soft_penalties: list[str] = []
    page_failed_count = int(summary.get("page_failed_count", 0) or 0)
    gemma_truncated_count = int(summary.get("gemma_truncated_count", 0) or 0)
    overgenerated_block_count = int(metrics.get("overgenerated_block_count", 0) or 0)
    if page_failed_count:
        soft_penalties.append(f"page_failed_count={page_failed_count}")
    if gemma_truncated_count:
        soft_penalties.append(f"gemma_truncated_count={gemma_truncated_count}")
    if overgenerated_block_count:
        soft_penalties.append(f"overgenerated_block_count={overgenerated_block_count}")

    ready_checks = [
        geometry_match_recall >= READY_GEOMETRY_RECALL_THRESHOLD,
        geometry_match_precision >= READY_GEOMETRY_PRECISION_THRESHOLD,
        non_empty_retention >= READY_NON_EMPTY_RETENTION_THRESHOLD,
        ocr_char_error_rate <= READY_OCR_CHAR_ERROR_RATE_THRESHOLD,
        page_p95_ocr_char_error_rate <= READY_PAGE_P95_OCR_CHAR_ERROR_RATE_THRESHOLD,
        page_failed_rate <= READY_PAGE_FAILED_RATE_THRESHOLD,
    ]
    if all(ready_checks):
        return "ready", [], soft_penalties

    conditional_checks = [
        geometry_match_recall >= CONDITIONAL_GEOMETRY_RECALL_THRESHOLD,
        geometry_match_precision >= CONDITIONAL_GEOMETRY_PRECISION_THRESHOLD,
        non_empty_retention >= CONDITIONAL_NON_EMPTY_RETENTION_THRESHOLD,
        ocr_char_error_rate <= CONDITIONAL_OCR_CHAR_ERROR_RATE_THRESHOLD,
        page_p95_ocr_char_error_rate <= CONDITIONAL_PAGE_P95_OCR_CHAR_ERROR_RATE_THRESHOLD,
        page_failed_rate <= CONDITIONAL_PAGE_FAILED_RATE_THRESHOLD,
    ]
    if all(conditional_checks):
        return "conditional", [], soft_penalties

    return "hold", [], soft_penalties


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
    page_ocr_word_error_rates: list[float] = []
    total_char_errors = 0
    total_gold_chars = 0
    total_word_errors = 0
    total_gold_words = 0
    unmatched_candidate_block_count = 0
    overgenerated_block_count = 0
    excluded_unstable_pages: list[str] = []

    gold_quality = {"block_count": 0, "non_empty": 0, "empty": 0, "single_char_like": 0}
    candidate_quality = {"block_count": 0, "non_empty": 0, "empty": 0, "single_char_like": 0}
    page_results: list[dict[str, Any]] = []

    active_gold_page_count = 0
    for page_name in sorted(gold_pages):
        gold_page = gold_pages[page_name]
        if str(gold_page.get("status", "active") or "active").lower() == "excluded":
            excluded_unstable_pages.append(page_name)
            continue
        active_gold_page_count += 1

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
            missing_chars = sum(
                len(_normalize_ocr_text(gold_blocks[idx].get("gold_text", gold_blocks[idx].get("seed_text", ""))))
                or 1
                for idx in gold_non_empty_indices
            )
            missing_words = sum(
                max(
                    1,
                    len(
                        _tokenize_words(gold_blocks[idx].get("gold_text", gold_blocks[idx].get("seed_text", "")))
                    ),
                )
                for idx in gold_non_empty_indices
            )
            total_gold_chars += missing_chars
            total_char_errors += missing_chars
            total_gold_words += missing_words
            total_word_errors += missing_words
            page_results.append(
                {
                    "page": page_name,
                    "pass": False,
                    "reason": "candidate page missing",
                    "page_ocr_char_error_rate": 1.0,
                    "page_ocr_word_error_rate": 1.0,
                }
            )
            page_ocr_char_error_rates.append(1.0)
            page_ocr_word_error_rates.append(1.0)
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

        matches, issues, _, unmatched_candidate = _match_blocks(gold_blocks, candidate_blocks)
        matched_gold_blocks += len(matches)
        unmatched_candidate_block_count += len(unmatched_candidate)
        match_map = {int(item["gold_index"]): int(item["candidate_index"]) for item in matches}

        page_char_errors = 0
        page_gold_chars = 0
        page_word_errors = 0
        page_gold_words = 0
        page_ocr_exact_matches = 0
        page_translation_similarity_values: list[float] = []
        page_translation_exact_matches = 0

        for gold_index in gold_non_empty_indices:
            gold_block = gold_blocks[gold_index]
            gold_text_raw = str(gold_block.get("gold_text", gold_block.get("seed_text", "")) or "")
            gold_text = _normalize_ocr_text(gold_text_raw)
            gold_words = _tokenize_words(gold_text_raw)
            gold_char_count = max(1, len(gold_text))
            gold_word_count = max(1, len(gold_words))
            page_gold_chars += gold_char_count
            total_gold_chars += gold_char_count
            page_gold_words += gold_word_count
            total_gold_words += gold_word_count

            candidate_index = match_map.get(gold_index)
            if candidate_index is None:
                page_char_errors += gold_char_count
                total_char_errors += gold_char_count
                page_word_errors += gold_word_count
                total_word_errors += gold_word_count
                continue

            candidate_block = candidate_blocks[candidate_index]
            candidate_text_raw = candidate_block.get("normalized_text", candidate_block.get("text", ""))
            candidate_text = _normalize_ocr_text(candidate_text_raw)
            candidate_words = _tokenize_words(candidate_text_raw)
            if candidate_text:
                matched_candidate_non_empty += 1
            matched_gold_non_empty += 1

            char_errors = _levenshtein_distance(gold_text, candidate_text)
            word_errors = _sequence_levenshtein(gold_words, candidate_words)
            page_char_errors += char_errors
            total_char_errors += char_errors
            page_word_errors += word_errors
            total_word_errors += word_errors
            if gold_text == candidate_text:
                ocr_exact_matches += 1
                page_ocr_exact_matches += 1

            seed_translation = _normalize_translation_text(
                gold_block.get("seed_normalized_translation", gold_block.get("seed_translation", ""))
            )
            candidate_translation = _normalize_translation_text(candidate_block.get("normalized_translation", ""))
            if seed_translation or candidate_translation:
                similarity = SequenceMatcher(None, seed_translation, candidate_translation).ratio()
                if seed_translation == candidate_translation:
                    translation_exact_matches += 1
                    page_translation_exact_matches += 1
                page_translation_similarity_values.append(similarity)
                translation_similarity_values.append(similarity)

        for candidate_index in unmatched_candidate:
            candidate_block = candidate_blocks[candidate_index]
            if _normalize_ocr_text(candidate_block.get("normalized_text", candidate_block.get("text", ""))):
                overgenerated_block_count += 1

        page_cer = page_char_errors / page_gold_chars if page_gold_chars else 0.0
        page_wer = page_word_errors / page_gold_words if page_gold_words else 0.0
        page_ocr_char_error_rates.append(page_cer)
        page_ocr_word_error_rates.append(page_wer)
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
                "page_ocr_word_error_rate": round(page_wer, 4),
                "page_ocr_exact_match_ratio": round(page_ocr_exact_matches / len(gold_non_empty_indices), 4)
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
    non_empty_retention = matched_candidate_non_empty / total_gold_non_empty if total_gold_non_empty else 1.0
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
    ocr_word_error_rate = total_word_errors / total_gold_words if total_gold_words else 0.0
    page_p95_ocr_char_error_rate = _percentile(page_ocr_char_error_rates, 0.95)
    page_p95_ocr_word_error_rate = _percentile(page_ocr_word_error_rates, 0.95)
    ocr_exact_text_match_ratio = ocr_exact_matches / total_gold_non_empty if total_gold_non_empty else 1.0
    translation_exact_match_ratio = (
        translation_exact_matches / len(translation_similarity_values) if translation_similarity_values else None
    )
    translation_similarity_ratio = (
        sum(translation_similarity_values) / len(translation_similarity_values)
        if translation_similarity_values
        else None
    )
    page_failed_rate = (
        int(candidate_summary.get("page_failed_count", 0) or 0) / _page_denominator(candidate_summary, active_gold_page_count)
    )
    overgenerated_block_rate = overgenerated_block_count / max(1, total_gold_blocks)

    metrics = {
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
        "ocr_word_error_rate": round(ocr_word_error_rate, 4),
        "page_p95_ocr_char_error_rate": round(page_p95_ocr_char_error_rate, 4),
        "page_p95_ocr_word_error_rate": round(page_p95_ocr_word_error_rate, 4),
        "ocr_exact_text_match_ratio": round(ocr_exact_text_match_ratio, 4),
        "translation_exact_match_ratio": round(translation_exact_match_ratio, 4)
        if translation_exact_match_ratio is not None
        else None,
        "translation_similarity_ratio": round(translation_similarity_ratio, 4)
        if translation_similarity_ratio is not None
        else None,
        "unmatched_candidate_block_count": unmatched_candidate_block_count,
        "overgenerated_block_count": overgenerated_block_count,
        "overgenerated_block_rate": round(overgenerated_block_rate, 4),
        "page_failed_count": int(candidate_summary.get("page_failed_count", 0) or 0),
        "page_failed_rate": round(page_failed_rate, 4),
        "gemma_truncated_count": int(candidate_summary.get("gemma_truncated_count", 0) or 0),
        "excluded_unstable_page_count": len(excluded_unstable_pages),
        "excluded_unstable_pages": excluded_unstable_pages,
    }
    quality_band, catastrophic_reasons, soft_penalties = _classify_band(metrics, candidate_summary)

    return {
        "gold_path": str(gold_path),
        "candidate_run_dir": str(candidate_run_dir),
        "quality_gate_scope": "OCR-only",
        "quality_gate_pass": quality_band != "catastrophic",
        "quality_band": quality_band,
        "quality_band_rank": QUALITY_BAND_RANK[quality_band],
        "catastrophic_reasons": catastrophic_reasons,
        "soft_penalties": soft_penalties,
        "gold_metadata": {
            "corpus": gold_payload.get("corpus", ""),
            "review_status": gold_payload.get("review_status", ""),
            "generated_from_run_dir": gold_payload.get("generated_from_run_dir", ""),
        },
        "ocr_normalization": {
            "canonical_small_voiced_kana": True,
            "ignored_chars": "「」『』,，、♡♥",
            "gold_empty_text_policy": "geometry-kept-text-skipped",
        },
        "candidate_summary": candidate_summary,
        "metrics": metrics,
        "page_results": page_results,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    metrics = payload.get("metrics", {})
    lines = [
        "# OCR Combo Ranked Compare",
        "",
        f"- quality_gate_scope: `{payload.get('quality_gate_scope')}`",
        f"- quality_gate_pass: `{payload.get('quality_gate_pass')}`",
        f"- quality_band: `{payload.get('quality_band')}`",
        f"- gold_path: `{payload.get('gold_path')}`",
        f"- candidate_run_dir: `{payload.get('candidate_run_dir')}`",
        f"- canonical_small_voiced_kana: `{(payload.get('ocr_normalization') or {}).get('canonical_small_voiced_kana')}`",
        f"- ignored_chars: `{(payload.get('ocr_normalization') or {}).get('ignored_chars')}`",
        f"- gold_empty_text_policy: `{(payload.get('ocr_normalization') or {}).get('gold_empty_text_policy')}`",
        f"- geometry_match_recall: `{metrics.get('geometry_match_recall')}`",
        f"- geometry_match_precision: `{metrics.get('geometry_match_precision')}`",
        f"- non_empty_retention: `{metrics.get('non_empty_retention')}`",
        f"- ocr_char_error_rate: `{metrics.get('ocr_char_error_rate')}`",
        f"- ocr_word_error_rate: `{metrics.get('ocr_word_error_rate')}`",
        f"- page_p95_ocr_char_error_rate: `{metrics.get('page_p95_ocr_char_error_rate')}`",
        f"- page_failed_rate: `{metrics.get('page_failed_rate')}`",
        f"- overgenerated_block_rate: `{metrics.get('overgenerated_block_rate')}`",
        "",
        "## Catastrophic Reasons",
        "",
    ]
    catastrophic = payload.get("catastrophic_reasons", [])
    if catastrophic:
        lines.extend(f"- {item}" for item in catastrophic)
    else:
        lines.append("- none")
    lines.extend(["", "## Soft Penalties", ""])
    penalties = payload.get("soft_penalties", [])
    if penalties:
        lines.extend(f"- {item}" for item in penalties)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Page Results",
            "",
            "| page | pass | matched_block_count | page_ocr_char_error_rate | page_ocr_word_error_rate | issues |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload.get("page_results", []):
        issues = row.get("issues", [])
        if isinstance(issues, list):
            issue_text = "; ".join(str(item) for item in issues)
        else:
            issue_text = str(row.get("reason", "") or "")
        lines.append(
            "| {page} | {passed} | {matched} | {cer} | {wer} | {issues} |".format(
                page=row.get("page", ""),
                passed=row.get("pass", False),
                matched=row.get("matched_block_count", ""),
                cer=row.get("page_ocr_char_error_rate", ""),
                wer=row.get("page_ocr_word_error_rate", ""),
                issues=issue_text,
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare OCR combo ranked candidate run against locked OCR gold.")
    parser.add_argument("--gold-path", required=True)
    parser.add_argument("--candidate-run-dir", required=True)
    parser.add_argument("--output", help="Optional output JSON path.")
    args = parser.parse_args()

    gold_path = Path(args.gold_path).resolve()
    candidate_run_dir = Path(args.candidate_run_dir).resolve()
    output_path = Path(args.output).resolve() if args.output else candidate_run_dir / "ocr_combo_ranked_compare.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = compare_runs(gold_path, candidate_run_dir)
    write_json(output_path, payload)
    (output_path.with_suffix(".md")).write_text(_render_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "quality_gate_pass": payload["quality_gate_pass"],
                "quality_band": payload["quality_band"],
                "output": str(output_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
