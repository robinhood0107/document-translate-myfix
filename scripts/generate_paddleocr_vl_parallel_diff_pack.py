#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
FAMILY_NAME = "paddleocr_vl_parallel"
BENCH_ROOT = ROOT / "banchmark_result_log" / FAMILY_NAME

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_ocr_combo_reference import _match_blocks, _normalize_ocr_text  # type: ignore
from benchmark_common import write_json


def _repo_relative(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return "./" + path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def _resolve_logged_path(value: str | Path) -> Path:
    path = Path(str(value).strip())
    if path.is_absolute():
        return path
    text = str(path)
    if text.startswith("./"):
        return ROOT / text[2:]
    return ROOT / path


def _resolve_filesystem_path(value: str | Path) -> Path:
    text = str(value or "").strip()
    if not text:
        return Path()
    candidate = Path(text)
    if candidate.exists():
        return candidate
    if len(text) >= 3 and text[1] == ":" and text[2] in {"\\", "/"}:
        drive = text[0].lower()
        rest = text[2:].replace("\\", "/").lstrip("/")
        translated = Path(f"/mnt/{drive}/{rest}")
        if translated.exists():
            return translated
        return translated
    return candidate


def _markdown_local_path(path: Path) -> str:
    resolved = str(path.resolve())
    if len(resolved) >= 3 and resolved[1] == ":" and resolved[2] in {"\\", "/"}:
        drive = resolved[0].lower()
        rest = resolved[2:].replace("\\", "/").lstrip("/")
        return f"/mnt/{drive}/{rest}"
    return path.resolve().as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _latest_completed_suite() -> Path:
    candidates = sorted(
        [path for path in BENCH_ROOT.iterdir() if path.is_dir() and (path / "suite_summary.json").is_file()],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No completed paddleocr_vl_parallel suite was found.")
    return candidates[0]


def _load_pages(run_dir: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(run_dir / "page_snapshots.json")
    pages = payload.get("pages", [])
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(pages, list):
        return result
    for page in pages:
        if not isinstance(page, dict):
            continue
        image_stem = str(page.get("image_stem", "") or "")
        if image_stem:
            result[image_stem] = page
    return result


def _load_request_bboxes(run_dir: Path) -> dict[tuple[str, int], list[int]]:
    path = run_dir / "request_events.jsonl"
    mapping: dict[tuple[str, int], list[int]] = {}
    if not path.is_file():
        return mapping
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        image_name = str(payload.get("image_name", "") or "")
        job_index = payload.get("job_index")
        bbox = payload.get("bbox")
        if not image_name or not isinstance(job_index, int) or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            mapping[(image_name, int(job_index))] = [int(value) for value in bbox]
        except (TypeError, ValueError):
            continue
    return mapping


def _speed_rank_non_baseline(summary: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_key = str(summary.get("baseline_candidate_key", "") or "")
    rows = summary.get("candidates", [])
    candidates = [row for row in rows if isinstance(row, dict) and str(row.get("candidate_key", "")) != baseline_key]
    return sorted(
        candidates,
        key=lambda item: (
            float(item.get("measured_ocr_total_sec_median") or float("inf")),
            float(item.get("measured_ocr_page_p95_sec_median") or float("inf")),
            float(item.get("measured_elapsed_sec_median") or float("inf")),
            str(item.get("candidate_key", "")),
        ),
    )


def _choose_representative_run(candidate_summary: dict[str, Any]) -> Path:
    measured = candidate_summary.get("measured_run_dirs", []) or []
    median_value = float(candidate_summary.get("measured_ocr_total_sec_median") or 0.0)
    scored: list[tuple[float, int, Path]] = []
    for index, value in enumerate(measured):
        run_dir = _resolve_logged_path(value)
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            continue
        run_summary = _load_json(summary_path)
        elapsed = float(run_summary.get("ocr_total_sec") or 0.0)
        scored.append((abs(elapsed - median_value), index, run_dir))
    if not scored:
        raise FileNotFoundError(f"No measured runs found for {candidate_summary.get('candidate_key')}")
    return sorted(scored, key=lambda item: (item[0], item[1], str(item[2])))[0][2]


def _choose_baseline_reference_run(summary: dict[str, Any]) -> Path:
    gold_path = _resolve_logged_path(summary.get("baseline_gold_path", ""))
    gold_payload = _load_json(gold_path)
    generated = str(gold_payload.get("generated_from_run_dir", "") or "")
    if generated:
        return _resolve_logged_path(generated)
    baseline_key = str(summary.get("baseline_candidate_key", "fixed_w8") or "fixed_w8")
    fallback = _resolve_logged_path(summary.get("suite_dir", "")) / baseline_key / "measured_r1"
    if fallback.is_dir():
        return fallback
    raise FileNotFoundError("Unable to resolve baseline reference run directory.")


def _fallback_bbox(block: dict[str, Any]) -> list[int] | None:
    for key in ("bubble_xyxy", "xyxy"):
        value = block.get(key)
        if isinstance(value, list) and len(value) == 4:
            try:
                return [int(item) for item in value]
            except (TypeError, ValueError):
                continue
    return None


def _clamp_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(x1 + 1, min(width, int(x2)))
    y2 = max(y1 + 1, min(height, int(y2)))
    return [x1, y1, x2, y2]


def _pick_bbox(
    *,
    page_image_name: str,
    block_index: int | None,
    block: dict[str, Any] | None,
    request_bboxes: dict[tuple[str, int], list[int]],
    image_shape: tuple[int, int, int] | tuple[int, int],
) -> list[int] | None:
    width = int(image_shape[1])
    height = int(image_shape[0])
    if block_index is not None:
        bbox = request_bboxes.get((page_image_name, int(block_index)))
        if bbox:
            return _clamp_bbox(bbox, width, height)
    if isinstance(block, dict):
        fallback = _fallback_bbox(block)
        if fallback:
            return _clamp_bbox(fallback, width, height)
    return None


def _resize_overlay(image, max_width: int = 1280):
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    scale = max_width / float(width)
    return cv2.resize(image, (max_width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)


def _highlight_diff(left: str, right: str) -> tuple[str, str]:
    left_out: list[str] = []
    right_out: list[str] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, left, right).get_opcodes():
        left_seg = left[i1:i2]
        right_seg = right[j1:j2]
        if tag == "equal":
            left_out.append(left_seg)
            right_out.append(right_seg)
        elif tag == "delete":
            left_out.append(f"[-{left_seg}-]")
        elif tag == "insert":
            right_out.append(f"{{+{right_seg}+}}")
        else:
            left_out.append(f"[-{left_seg}-]")
            right_out.append(f"{{+{right_seg}+}}")
    return "".join(left_out), "".join(right_out)


def _write_review_assets(
    *,
    page_image_path: Path,
    bbox: list[int],
    label: str,
    output_dir: Path,
) -> dict[str, str]:
    image = cv2.imread(str(page_image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to load page image: {page_image_path}")
    bbox = _clamp_bbox(bbox, image.shape[1], image.shape[0])
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 220), 6)
    cv2.putText(
        overlay,
        label,
        (max(16, x1), max(36, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (30, 30, 220),
        2,
        cv2.LINE_AA,
    )
    overlay = _resize_overlay(overlay)
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_path = output_dir / "crop.png"
    page_path = output_dir / "page_location.png"
    cv2.imwrite(str(crop_path), crop)
    cv2.imwrite(str(page_path), overlay)
    return {
        "crop_path": _markdown_local_path(crop_path),
        "page_path": _markdown_local_path(page_path),
    }


def _markdown_escape(value: Any) -> str:
    return html.escape(str(value or ""), quote=False)


def _render_candidate_doc(
    *,
    suite_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    baseline_run_dir: Path,
    candidate_run_dir: Path,
    changed_entries: list[dict[str, Any]],
    page_counts: dict[str, int],
    rank_index: int,
) -> str:
    candidate_key = str(candidate_summary.get("candidate_key", "n/a") or "n/a")
    baseline_key = str(baseline_summary.get("candidate_key", "fixed_w8") or "fixed_w8")
    failures = candidate_summary.get("quality_gate_failures", []) or []
    lines = [
        f"# Top {rank_index} Diff Review - {candidate_key}",
        "",
        "핵심 문제 해결 방향은 사용자가 착안했다.",
        "",
        "## 계약",
        "",
        "- 이번 결과는 `PaddleOCR VL 단독 상주 상한선 benchmark` 기준이다.",
        "- `runtime_services=ocr-only`, `stage_ceiling=ocr` 계약에서 Gemma와 MangaLMM은 실행되지 않는다.",
        "- 최종 승격은 숫자 게이트가 아니라 사용자 OCR diff 검수 승인으로 확정한다.",
        "",
        "## 대표 run",
        "",
        f"- baseline candidate: `{baseline_key}`",
        f"- baseline reference run: `{_repo_relative(baseline_run_dir)}`",
        f"- candidate: `{candidate_key}`",
        f"- candidate representative run: `{_repo_relative(candidate_run_dir)}`",
        f"- final_promotion_status: `{suite_summary.get('final_promotion_status', 'pending_user_review')}`",
        "",
        "## Aggregate",
        "",
        f"- ocr_total_sec_median: `{candidate_summary.get('measured_ocr_total_sec_median')}`",
        f"- baseline_delta_ocr_total_sec: `{round(float(candidate_summary.get('measured_ocr_total_sec_median') or 0.0) - float(baseline_summary.get('measured_ocr_total_sec_median') or 0.0), 4)}`",
        f"- mean_CER: `{candidate_summary.get('quality_mean_cer')}`",
        f"- baseline_delta_mean_CER: `{round(float(candidate_summary.get('quality_mean_cer') or 0.0) - float(baseline_summary.get('quality_mean_cer') or 0.0), 4)}`",
        f"- exact_match: `{candidate_summary.get('quality_mean_exact_match')}`",
        f"- baseline_delta_exact_match: `{round(float(candidate_summary.get('quality_mean_exact_match') or 0.0) - float(baseline_summary.get('quality_mean_exact_match') or 0.0), 4)}`",
        f"- empty_block_median: `{candidate_summary.get('measured_ocr_empty_block_count_median')}`",
        f"- baseline_delta_empty_block_median: `{round(float(candidate_summary.get('measured_ocr_empty_block_count_median') or 0.0) - float(baseline_summary.get('measured_ocr_empty_block_count_median') or 0.0), 4)}`",
        f"- page_failed_count_max: `{candidate_summary.get('measured_page_failed_count_max')}`",
        f"- baseline_delta_page_failed_count_max: `{round(float(candidate_summary.get('measured_page_failed_count_max') or 0.0) - float(baseline_summary.get('measured_page_failed_count_max') or 0.0), 4)}`",
        f"- quality_gate_pass: `{candidate_summary.get('quality_gate_pass')}`",
        "",
        "## Quality Gate Notes",
        "",
    ]
    if failures:
        lines.extend(f"- {item}" for item in failures)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Page Changed Block Counts",
            "",
            "| page | changed_block_count |",
            "| --- | --- |",
        ]
    )
    for page_name, count in sorted(page_counts.items()):
        lines.append(f"| {page_name} | {count} |")
    lines.extend(["", "## Changed Blocks", ""])
    if not changed_entries:
        lines.append("- no OCR text difference against baseline")
        lines.append("")
        return "\n".join(lines)

    current_page = None
    for entry in changed_entries:
        page_name = str(entry.get("page_name", "") or "")
        if page_name != current_page:
            current_page = page_name
            lines.extend(
                [
                    f"### {page_name}",
                    "",
                    f"- changed_block_count: `{page_counts.get(page_name, 0)}`",
                    "",
                ]
            )
        block_label = str(entry.get("block_label", "") or "")
        lines.extend(
            [
                f"#### {block_label}",
                "",
                f"- bbox: `{entry.get('bbox')}`",
                f"- baseline OCR: `{_markdown_escape(entry.get('baseline_text', ''))}`",
                f"- candidate OCR: `{_markdown_escape(entry.get('candidate_text', ''))}`",
                f"- crop: ![crop]({entry.get('crop_path')})",
                f"- 위치: ![page]({entry.get('page_path')})",
                "",
                "Baseline diff",
                "",
                "```text",
                str(entry.get("baseline_diff", "") or ""),
                "```",
                "",
                "Candidate diff",
                "",
                "```text",
                str(entry.get("candidate_diff", "") or ""),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _build_changed_entries_for_candidate(
    *,
    suite_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    baseline_run_dir: Path,
    candidate_run_dir: Path,
    rank_index: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    baseline_pages = _load_pages(baseline_run_dir)
    candidate_pages = _load_pages(candidate_run_dir)
    baseline_requests = _load_request_bboxes(baseline_run_dir)
    candidate_requests = _load_request_bboxes(candidate_run_dir)
    suite_dir = _resolve_logged_path(suite_summary.get("suite_dir", ""))
    assets_root = suite_dir / "review" / "review_assets" / str(candidate_summary.get("candidate_key", "candidate"))

    changed_entries: list[dict[str, Any]] = []
    page_counts: dict[str, int] = {}

    for page_stem in sorted(baseline_pages):
        baseline_page = baseline_pages[page_stem]
        candidate_page = candidate_pages.get(page_stem)
        baseline_blocks = baseline_page.get("blocks", []) if isinstance(baseline_page.get("blocks"), list) else []
        candidate_blocks = candidate_page.get("blocks", []) if isinstance(candidate_page, dict) and isinstance(candidate_page.get("blocks"), list) else []
        matches, _, _, unmatched_candidate = _match_blocks(baseline_blocks, candidate_blocks)
        match_map = {int(item["gold_index"]): int(item["candidate_index"]) for item in matches}
        image_name = str(baseline_page.get("image_name", "") or "")
        image_path = _resolve_filesystem_path(baseline_page.get("image_path", ""))
        if not image_path.exists():
            image_path = baseline_run_dir / "corpus" / image_name
        page_label = image_name or page_stem

        page_changed = 0
        for baseline_index, baseline_block in enumerate(baseline_blocks):
            baseline_text = str(baseline_block.get("text", "") or "")
            baseline_norm = _normalize_ocr_text(baseline_block.get("normalized_text", baseline_text))
            candidate_index = match_map.get(baseline_index)
            candidate_block = candidate_blocks[candidate_index] if isinstance(candidate_index, int) and candidate_index < len(candidate_blocks) else {}
            candidate_text = str(candidate_block.get("text", "") or "")
            candidate_norm = _normalize_ocr_text(candidate_block.get("normalized_text", candidate_text))
            if candidate_index is not None and baseline_norm == candidate_norm:
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Unable to load page image: {image_path}")
            bbox = _pick_bbox(
                page_image_name=image_name,
                block_index=baseline_index,
                block=baseline_block,
                request_bboxes=baseline_requests,
                image_shape=image.shape,
            )
            if bbox is None and candidate_index is not None:
                bbox = _pick_bbox(
                    page_image_name=image_name,
                    block_index=candidate_index,
                    block=candidate_block if isinstance(candidate_block, dict) else None,
                    request_bboxes=candidate_requests,
                    image_shape=image.shape,
                )
            if bbox is None:
                continue

            asset_dir = assets_root / page_stem / f"block_{baseline_index:03d}"
            assets = _write_review_assets(
                page_image_path=image_path,
                bbox=bbox,
                label=f"{page_stem}:{baseline_index}",
                output_dir=asset_dir,
            )
            baseline_diff, candidate_diff = _highlight_diff(baseline_text, candidate_text)
            changed_entries.append(
                {
                    "page_name": page_label,
                    "page_stem": page_stem,
                    "block_label": f"block {baseline_index}",
                    "block_index": baseline_index,
                    "bbox": bbox,
                    "baseline_text": baseline_text,
                    "candidate_text": candidate_text,
                    "baseline_diff": baseline_diff,
                    "candidate_diff": candidate_diff,
                    **assets,
                }
            )
            page_changed += 1

        for candidate_index in unmatched_candidate:
            if candidate_index >= len(candidate_blocks):
                continue
            candidate_block = candidate_blocks[candidate_index]
            candidate_text = str(candidate_block.get("text", "") or "")
            if not _normalize_ocr_text(candidate_block.get("normalized_text", candidate_text)):
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Unable to load page image: {image_path}")
            bbox = _pick_bbox(
                page_image_name=image_name,
                block_index=candidate_index,
                block=candidate_block,
                request_bboxes=candidate_requests,
                image_shape=image.shape,
            )
            if bbox is None:
                continue

            asset_dir = assets_root / page_stem / f"candidate_extra_{candidate_index:03d}"
            assets = _write_review_assets(
                page_image_path=image_path,
                bbox=bbox,
                label=f"{page_stem}:extra{candidate_index}",
                output_dir=asset_dir,
            )
            baseline_diff, candidate_diff = _highlight_diff("", candidate_text)
            changed_entries.append(
                {
                    "page_name": page_label,
                    "page_stem": page_stem,
                    "block_label": f"candidate extra {candidate_index}",
                    "block_index": None,
                    "bbox": bbox,
                    "baseline_text": "",
                    "candidate_text": candidate_text,
                    "baseline_diff": baseline_diff,
                    "candidate_diff": candidate_diff,
                    **assets,
                }
            )
            page_changed += 1

        if page_changed > 0:
            page_counts[page_label] = page_changed

    changed_entries.sort(
        key=lambda item: (
            str(item.get("page_name", "")),
            10**9 if item.get("block_index") is None else int(item.get("block_index")),
            str(item.get("block_label", "")),
        )
    )
    return changed_entries, page_counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate top-2 OCR diff review pack for paddleocr_vl_parallel.")
    parser.add_argument("--suite-dir", default="")
    parser.add_argument("--final-promotion-status", default="")
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir).resolve() if args.suite_dir else _latest_completed_suite()
    suite_summary = _load_json(suite_dir / "suite_summary.json")
    if args.final_promotion_status:
        suite_summary["final_promotion_status"] = args.final_promotion_status
    baseline_key = str(suite_summary.get("baseline_candidate_key", "fixed_w8") or "fixed_w8")
    candidate_map = {
        str(item.get("candidate_key", "")): item
        for item in suite_summary.get("candidates", [])
        if isinstance(item, dict)
    }
    baseline_summary = candidate_map.get(baseline_key)
    if not isinstance(baseline_summary, dict):
        raise RuntimeError("Baseline candidate summary is missing.")

    review_candidates = _speed_rank_non_baseline(suite_summary)[:2]
    baseline_run_dir = _choose_baseline_reference_run(suite_summary)
    review_dir = suite_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    rendered_docs: list[dict[str, Any]] = []
    for rank_index, candidate_summary in enumerate(review_candidates, start=1):
        candidate_run_dir = _choose_representative_run(candidate_summary)
        changed_entries, page_counts = _build_changed_entries_for_candidate(
            suite_summary=suite_summary,
            baseline_summary=baseline_summary,
            candidate_summary=candidate_summary,
            baseline_run_dir=baseline_run_dir,
            candidate_run_dir=candidate_run_dir,
            rank_index=rank_index,
        )
        doc_text = _render_candidate_doc(
            suite_summary=suite_summary,
            baseline_summary=baseline_summary,
            candidate_summary=candidate_summary,
            baseline_run_dir=baseline_run_dir,
            candidate_run_dir=candidate_run_dir,
            changed_entries=changed_entries,
            page_counts=page_counts,
            rank_index=rank_index,
        )
        doc_path = review_dir / f"top{rank_index}_diff_review.md"
        doc_path.write_text(doc_text, encoding="utf-8")
        rendered_docs.append(
            {
                "rank": rank_index,
                "candidate_key": candidate_summary.get("candidate_key"),
                "doc_path": _repo_relative(doc_path),
                "representative_run_dir": _repo_relative(candidate_run_dir),
                "changed_block_count": len(changed_entries),
                "page_changed_counts": page_counts,
            }
        )

    write_json(
        review_dir / "review_summary.json",
        {
            "suite_dir": _repo_relative(suite_dir),
            "baseline_candidate_key": baseline_key,
            "review_candidates": rendered_docs,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
