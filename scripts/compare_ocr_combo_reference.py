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

XYXY_IOU_THRESHOLD = 0.90
BUBBLE_IOU_THRESHOLD = 0.85


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _normalize_text(text: object) -> str:
    return "".join(str(text or "").split())


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


def _load_translation_exports(run_dir: Path) -> dict[str, dict[str, str]]:
    exports: dict[str, dict[str, str]] = {}
    for path in sorted(run_dir.rglob("*_translated.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        page_name = path.name.replace("_translated.json", "")
        exports[page_name] = {str(key): str(value or "") for key, value in payload.items()}
    return exports


def _page_quality(page: dict[str, Any]) -> dict[str, int]:
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


def _as_box(value: object) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _match_blocks(
    reference_blocks: list[dict[str, Any]],
    candidate_blocks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[int]]:
    scores: list[tuple[float, int, int, str]] = []
    for ref_index, ref_block in enumerate(reference_blocks):
        for cand_index, cand_block in enumerate(candidate_blocks):
            ref_bubble = _as_box(ref_block.get("bubble_xyxy"))
            cand_bubble = _as_box(cand_block.get("bubble_xyxy"))
            if ref_bubble and cand_bubble:
                score = _xyxy_iou(ref_bubble, cand_bubble)
                threshold = BUBBLE_IOU_THRESHOLD
                match_kind = "bubble"
            else:
                ref_xyxy = _as_box(ref_block.get("xyxy"))
                cand_xyxy = _as_box(cand_block.get("xyxy"))
                if not ref_xyxy or not cand_xyxy:
                    continue
                score = _xyxy_iou(ref_xyxy, cand_xyxy)
                threshold = XYXY_IOU_THRESHOLD
                match_kind = "xyxy"
            if score >= threshold:
                scores.append((score, ref_index, cand_index, match_kind))

    matches: list[dict[str, Any]] = []
    used_ref: set[int] = set()
    used_cand: set[int] = set()
    for score, ref_index, cand_index, match_kind in sorted(scores, reverse=True):
        if ref_index in used_ref or cand_index in used_cand:
            continue
        used_ref.add(ref_index)
        used_cand.add(cand_index)
        matches.append(
            {
                "reference_index": ref_index,
                "candidate_index": cand_index,
                "iou": round(score, 4),
                "match_kind": match_kind,
            }
        )

    issues: list[str] = []
    unmatched_candidate = sorted(set(range(len(candidate_blocks))) - used_cand)
    if len(matches) < len(reference_blocks):
        unmatched_reference = sorted(set(range(len(reference_blocks))) - used_ref)
        if unmatched_reference:
            issues.append(f"unmatched_reference_blocks={unmatched_reference}")
    if unmatched_candidate:
        issues.append(f"unmatched_candidate_blocks={unmatched_candidate}")

    return sorted(matches, key=lambda item: item["reference_index"]), issues, unmatched_candidate


def _sorted_export_values(payload: dict[str, str]) -> list[str]:
    def sort_key(key: str) -> tuple[int, str]:
        prefix, _, suffix = key.partition("_")
        try:
            number = int(suffix) if prefix == "block" else 10**9
        except ValueError:
            number = 10**9
        return number, key

    return [str(payload[key] or "") for key in sorted(payload.keys(), key=sort_key)]


def _page_translation_text(page: dict[str, Any], exports: dict[str, dict[str, str]]) -> str:
    page_name = str(page.get("image_stem", "") or "")
    export_payload = exports.get(page_name)
    if export_payload:
        return _normalize_text("".join(_sorted_export_values(export_payload)))
    blocks = page.get("blocks", []) if isinstance(page.get("blocks"), list) else []
    return _normalize_text("".join(str(block.get("translation", "") or "") for block in blocks if isinstance(block, dict)))


def compare_runs(reference_run_dir: Path, candidate_run_dir: Path) -> dict[str, Any]:
    reference_summary = _load_summary(reference_run_dir)
    candidate_summary = _load_summary(candidate_run_dir)
    _, reference_pages = _load_pages(reference_run_dir)
    _, candidate_pages = _load_pages(candidate_run_dir)
    reference_exports = _load_translation_exports(reference_run_dir)
    candidate_exports = _load_translation_exports(candidate_run_dir)

    total_reference_blocks = 0
    total_reference_non_empty = 0
    matched_reference_non_empty = 0
    matched_candidate_non_empty = 0
    ocr_exact_matches = 0
    translation_exact_matches = 0
    translation_similarity_values: list[float] = []
    page_translation_similarity_values: list[float] = []
    unmatched_candidate_blocks = 0
    overgenerated_translation_blocks = 0

    reference_quality = {"block_count": 0, "non_empty": 0, "empty": 0, "single_char_like": 0}
    candidate_quality = {"block_count": 0, "non_empty": 0, "empty": 0, "single_char_like": 0}
    page_results: list[dict[str, Any]] = []

    for page_name in sorted(reference_pages):
        reference_page = reference_pages[page_name]
        candidate_page = candidate_pages.get(page_name)
        reference_quality_page = _page_quality(reference_page)
        for key, value in reference_quality_page.items():
            reference_quality[key] += value

        if not isinstance(candidate_page, dict):
            page_results.append(
                {
                    "page": page_name,
                    "pass": False,
                    "reason": "candidate page missing",
                }
            )
            total_reference_blocks += reference_quality_page["block_count"]
            total_reference_non_empty += reference_quality_page["non_empty"]
            continue

        candidate_quality_page = _page_quality(candidate_page)
        for key, value in candidate_quality_page.items():
            candidate_quality[key] += value

        reference_blocks = reference_page.get("blocks", []) if isinstance(reference_page.get("blocks"), list) else []
        candidate_blocks = candidate_page.get("blocks", []) if isinstance(candidate_page.get("blocks"), list) else []
        total_reference_blocks += len(reference_blocks)

        matches, issues, unmatched_candidate = _match_blocks(reference_blocks, candidate_blocks)
        unmatched_candidate_blocks += len(unmatched_candidate)

        reference_non_empty_indices = {
            idx
            for idx, block in enumerate(reference_blocks)
            if _normalize_text(block.get("normalized_text", ""))
        }
        total_reference_non_empty += len(reference_non_empty_indices)
        matched_index_map = {int(item["reference_index"]): int(item["candidate_index"]) for item in matches}
        matched_ref_non_empty_indices = reference_non_empty_indices & set(matched_index_map.keys())
        matched_reference_non_empty += len(matched_ref_non_empty_indices)

        for ref_index in matched_ref_non_empty_indices:
            candidate_index = matched_index_map[ref_index]
            ref_block = reference_blocks[ref_index]
            cand_block = candidate_blocks[candidate_index]
            ref_text = _normalize_text(ref_block.get("normalized_text", ""))
            cand_text = _normalize_text(cand_block.get("normalized_text", ""))
            if cand_text:
                matched_candidate_non_empty += 1
            if ref_text == cand_text:
                ocr_exact_matches += 1

            ref_translation = _normalize_text(ref_block.get("normalized_translation", ""))
            cand_translation = _normalize_text(cand_block.get("normalized_translation", ""))
            if ref_translation == cand_translation:
                translation_exact_matches += 1
            translation_similarity_values.append(
                SequenceMatcher(None, ref_translation, cand_translation).ratio()
            )

        for candidate_index in unmatched_candidate:
            candidate_block = candidate_blocks[candidate_index]
            if _normalize_text(candidate_block.get("normalized_translation", "")):
                overgenerated_translation_blocks += 1

        reference_page_translation = _page_translation_text(reference_page, reference_exports)
        candidate_page_translation = _page_translation_text(candidate_page, candidate_exports)
        page_similarity = SequenceMatcher(
            None,
            reference_page_translation,
            candidate_page_translation,
        ).ratio()
        page_translation_similarity_values.append(page_similarity)

        page_results.append(
            {
                "page": page_name,
                "pass": not issues,
                "issues": issues,
                "reference_block_count": len(reference_blocks),
                "candidate_block_count": len(candidate_blocks),
                "matched_block_count": len(matches),
                "page_translation_similarity": round(page_similarity, 4),
                "reference_non_empty": reference_quality_page["non_empty"],
                "candidate_non_empty": candidate_quality_page["non_empty"],
            }
        )

    geometry_match_recall = (
        matched_reference_non_empty / total_reference_non_empty if total_reference_non_empty else 1.0
    )
    non_empty_retention = (
        matched_candidate_non_empty / total_reference_non_empty if total_reference_non_empty else 1.0
    )
    reference_empty_rate = (
        reference_quality["empty"] / reference_quality["block_count"] if reference_quality["block_count"] else 0.0
    )
    candidate_empty_rate = (
        candidate_quality["empty"] / candidate_quality["block_count"] if candidate_quality["block_count"] else 0.0
    )
    reference_single_char_like_rate = (
        reference_quality["single_char_like"] / reference_quality["block_count"]
        if reference_quality["block_count"]
        else 0.0
    )
    candidate_single_char_like_rate = (
        candidate_quality["single_char_like"] / candidate_quality["block_count"]
        if candidate_quality["block_count"]
        else 0.0
    )
    translation_similarity_avg = (
        sum(page_translation_similarity_values) / len(page_translation_similarity_values)
        if page_translation_similarity_values
        else 1.0
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
    if geometry_match_recall < 0.95:
        hard_gate_failures.append(f"geometry_match_recall<{0.95}: {geometry_match_recall:.4f}")
    if non_empty_retention < 0.98:
        hard_gate_failures.append(f"non_empty_retention<{0.98}: {non_empty_retention:.4f}")
    if candidate_empty_rate > reference_empty_rate + 0.02:
        hard_gate_failures.append(
            f"candidate_empty_rate>{reference_empty_rate + 0.02:.4f}: {candidate_empty_rate:.4f}"
        )
    if candidate_single_char_like_rate > reference_single_char_like_rate + 0.02:
        hard_gate_failures.append(
            f"candidate_single_char_like_rate>{reference_single_char_like_rate + 0.02:.4f}: {candidate_single_char_like_rate:.4f}"
        )
    if translation_similarity_avg < 0.98:
        hard_gate_failures.append(
            f"translation_similarity_avg<{0.98}: {translation_similarity_avg:.4f}"
        )
    if overgenerated_translation_blocks != 0:
        hard_gate_failures.append(
            f"suspect_overgenerated_translation_block_count={overgenerated_translation_blocks}"
        )

    ocr_exact_text_match_ratio = (
        ocr_exact_matches / matched_reference_non_empty if matched_reference_non_empty else 1.0
    )
    translation_exact_match_ratio = (
        translation_exact_matches / matched_reference_non_empty if matched_reference_non_empty else 1.0
    )
    translation_similarity_ratio = (
        sum(translation_similarity_values) / len(translation_similarity_values)
        if translation_similarity_values
        else 1.0
    )

    return {
        "reference_run_dir": str(reference_run_dir),
        "candidate_run_dir": str(candidate_run_dir),
        "quality_gate_pass": not hard_gate_failures,
        "hard_gate_failures": hard_gate_failures,
        "reference_summary": reference_summary,
        "candidate_summary": candidate_summary,
        "metrics": {
            "reference_block_count": reference_quality["block_count"],
            "candidate_block_count": candidate_quality["block_count"],
            "reference_non_empty_count": reference_quality["non_empty"],
            "candidate_non_empty_count": candidate_quality["non_empty"],
            "geometry_match_recall": round(geometry_match_recall, 4),
            "non_empty_retention": round(non_empty_retention, 4),
            "reference_empty_rate": round(reference_empty_rate, 4),
            "candidate_empty_rate": round(candidate_empty_rate, 4),
            "reference_single_char_like_rate": round(reference_single_char_like_rate, 4),
            "candidate_single_char_like_rate": round(candidate_single_char_like_rate, 4),
            "ocr_exact_text_match_ratio": round(ocr_exact_text_match_ratio, 4),
            "translation_exact_match_ratio": round(translation_exact_match_ratio, 4),
            "translation_similarity_ratio": round(translation_similarity_ratio, 4),
            "page_translation_similarity_avg": round(translation_similarity_avg, 4),
            "unmatched_candidate_block_count": unmatched_candidate_blocks,
            "suspect_overgenerated_translation_block_count": overgenerated_translation_blocks,
        },
        "page_results": page_results,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    metrics = payload.get("metrics", {})
    lines = [
        "# OCR Combo Compare",
        "",
        f"- quality_gate_pass: `{payload.get('quality_gate_pass')}`",
        f"- reference_run_dir: `{payload.get('reference_run_dir')}`",
        f"- candidate_run_dir: `{payload.get('candidate_run_dir')}`",
        f"- geometry_match_recall: `{metrics.get('geometry_match_recall')}`",
        f"- non_empty_retention: `{metrics.get('non_empty_retention')}`",
        f"- candidate_empty_rate: `{metrics.get('candidate_empty_rate')}`",
        f"- candidate_single_char_like_rate: `{metrics.get('candidate_single_char_like_rate')}`",
        f"- page_translation_similarity_avg: `{metrics.get('page_translation_similarity_avg')}`",
        f"- suspect_overgenerated_translation_block_count: `{metrics.get('suspect_overgenerated_translation_block_count')}`",
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
            "| page | pass | matched_block_count | page_translation_similarity | issues |",
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
            "| {page} | {passed} | {matched} | {similarity} | {issues} |".format(
                page=row.get("page", ""),
                passed=row.get("pass", False),
                matched=row.get("matched_block_count", ""),
                similarity=row.get("page_translation_similarity", ""),
                issues=issue_text,
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare OCR combo candidate run against a reference run.")
    parser.add_argument("--reference-run-dir", required=True)
    parser.add_argument("--candidate-run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reference_run_dir = Path(args.reference_run_dir)
    candidate_run_dir = Path(args.candidate_run_dir)
    output_path = Path(args.output)

    payload = compare_runs(reference_run_dir, candidate_run_dir)
    write_json(output_path, payload)
    output_path.with_suffix(".md").write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({"quality_gate_pass": payload["quality_gate_pass"], "output": str(output_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
