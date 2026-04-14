#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FAMILY_NAME = "paddleocr_vl_parallel"
BENCH_ROOT = ROOT / "banchmark_result_log" / FAMILY_NAME
DOC_ROOT = ROOT / "docs" / "benchmark" / "paddleocr-vl-parallel"
REPORT_PATH = ROOT / "docs" / "banchmark_report" / "paddleocr-vl-parallel-report-ko.md"
ASSETS_ROOT = ROOT / "docs" / "assets" / "benchmarking" / "paddleocr-vl-parallel"
LATEST_ASSETS = ASSETS_ROOT / "latest"
HISTORY_ROOT = ASSETS_ROOT / "history"
RESULTS_HISTORY_PATH = DOC_ROOT / "results-history-ko.md"


def _repo_relative(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return "./" + path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return _load_json(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _promotion_status(summary: dict[str, Any]) -> str:
    return str(summary.get("final_promotion_status", "pending_user_review") or "pending_user_review")


def _approved_candidate_key(summary: dict[str, Any]) -> str:
    status = _promotion_status(summary)
    prefix = "approved_"
    if status.startswith(prefix):
        return status[len(prefix) :]
    return ""


def _promotion_title(summary: dict[str, Any]) -> str:
    return "## Promotion Status"


def _promotion_notes(summary: dict[str, Any]) -> list[str]:
    status = _promotion_status(summary)
    approved_candidate = _approved_candidate_key(summary)
    if approved_candidate:
        return [
            f"- final_promotion_status: `{status}`",
            f"- approved_promotion_candidate: `{approved_candidate}`",
            "- 사용자 OCR diff 검수 승인이 완료되었고, develop 기본값 승격 대상이 확정되었다.",
            "- 이번 승격은 `PaddleOCR VL 단독 상주 상한선 benchmark` 결과를 기준으로 한다.",
        ]
    return [
        f"- final_promotion_status: `{status}`",
        "- 숫자 품질 게이트는 검수 우선순위 보조 지표다.",
        "- 최종 기본값 승격은 OCR diff 검수 승인 후에만 진행한다.",
    ]


def _patched_summary(summary: dict[str, Any], final_promotion_status: str) -> dict[str, Any]:
    patched = copy.deepcopy(summary)
    if final_promotion_status:
        patched["final_promotion_status"] = final_promotion_status
    return patched


def _patch_markdown_status(text: str, *, final_promotion_status: str) -> str:
    if not final_promotion_status:
        return text
    return text.replace("pending_user_review", final_promotion_status)


def _write_patched_suite_markdown(source: Path, target: Path, *, final_promotion_status: str) -> None:
    if not source.is_file():
        return
    text = source.read_text(encoding="utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_patch_markdown_status(text, final_promotion_status=final_promotion_status), encoding="utf-8")


def _write_promotion_spec(path: Path, *, final_promotion_status: str) -> None:
    approved_candidate = ""
    prefix = "approved_"
    if final_promotion_status.startswith(prefix):
        approved_candidate = final_promotion_status[len(prefix) :]
    if approved_candidate:
        text = "\n".join(
            [
                "# 문제 해결 명세서 05 - Product Promotion",
                "",
                "핵심 문제 해결 방향은 사용자가 착안했다.",
                "",
                "## 가설",
                "",
                "사용자 OCR diff 검수 승인이 완료되면, 승인된 winner를 develop 기본값으로 승격할 수 있다.",
                "",
                "## 실험 조건",
                "",
                "- develop promotion was approved after OCR diff review",
                "",
                "## 측정값",
                "",
                f"- promotion_status={final_promotion_status}",
                f"- approved_promotion_candidate={approved_candidate}",
                "",
                "## 해석",
                "",
                f"`{approved_candidate}`가 최종 develop promotion winner로 잠겼고, 공통 런타임 인프라와 함께 기본값 승격을 진행한다.",
                "",
                "## 다음 행동",
                "",
                "- promote approved winner on develop",
                "",
                "## 저자 및 기여",
                "",
                "- Idea Origin: User",
                "- Planning / Measurement Design / Implementation Detailing / Validation: Collaborative",
                "- Execution Support: Codex",
                "",
            ]
        )
    else:
        text = "\n".join(
            [
                "# 문제 해결 명세서 05 - Product Promotion",
                "",
                "핵심 문제 해결 방향은 사용자가 착안했다.",
                "",
                "## 가설",
                "",
                "develop 기본값 승격은 사용자 검수 승인 전까지 잠겨 있으며, 이번 단계는 review pack 준비까지를 완료 상태로 본다.",
                "",
                "## 실험 조건",
                "",
                "- develop promotion requires explicit user approval after OCR diff review",
                "",
                "## 측정값",
                "",
                f"- promotion_status={final_promotion_status or 'pending_user_review'}",
                "",
                "## 해석",
                "",
                "develop 기본값 승격은 사용자 검수 승인 전까지 잠겨 있으며, 이번 단계는 review pack 준비까지를 완료 상태로 본다.",
                "",
                "## 다음 행동",
                "",
                "- prepare develop promotion branch after user approval",
                "",
                "## 저자 및 기여",
                "",
                "- Idea Origin: User",
                "- Planning / Measurement Design / Implementation Detailing / Validation: Collaborative",
                "- Execution Support: Codex",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _latest_completed_suite() -> Path:
    candidates = sorted(
        [path for path in BENCH_ROOT.iterdir() if path.is_dir() and (path / "suite_summary.json").is_file()],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No completed paddleocr_vl_parallel suite was found.")
    return candidates[0]


def _copy_file_if_exists(source: Path, target: Path) -> None:
    if not source.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_tree_if_exists(source: Path, target: Path) -> None:
    if not source.exists():
        return
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _svg_escape(text: Any) -> str:
    value = str(text or "")
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _bar_color(row: dict[str, Any]) -> str:
    if str(row.get("candidate_key", "")) == "fixed_w8":
        return "#1565c0"
    return "#2e7d32" if str(row.get("quality_gate_pass", "")).lower() == "true" else "#b71c1c"


def _render_bar_chart(
    rows: list[dict[str, Any]],
    *,
    title: str,
    value_key: str,
    value_suffix: str = "",
    width: int = 980,
) -> str:
    usable_rows = [row for row in rows if _safe_float(row.get(value_key)) is not None]
    if not usable_rows:
        return ""

    label_width = 240
    value_width = 90
    chart_left = label_width + 10
    chart_right = width - value_width - 20
    chart_width = max(200, chart_right - chart_left)
    row_height = 30
    header_height = 56
    footer_height = 24
    height = header_height + len(usable_rows) * row_height + footer_height
    max_value = max(_safe_float(row.get(value_key)) or 0.0 for row in usable_rows) or 1.0

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{_svg_escape(title)}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="20" y="32" font-size="20" font-family="Segoe UI, Arial, sans-serif" fill="#111111">{_svg_escape(title)}</text>',
        '<text x="20" y="50" font-size="12" font-family="Segoe UI, Arial, sans-serif" fill="#555555">Blue = baseline, Green = quality gate pass, Red = quality gate fail</text>',
    ]

    for index, row in enumerate(usable_rows):
        value = _safe_float(row.get(value_key)) or 0.0
        top = header_height + index * row_height
        bar_width = max(1.0, chart_width * (value / max_value))
        label = str(row.get("candidate_key", "") or "")
        value_text = f"{value:.4f}{value_suffix}"
        color = _bar_color(row)
        lines.extend(
            [
                f'<text x="20" y="{top + 20}" font-size="12" font-family="Consolas, Menlo, monospace" fill="#222222">{_svg_escape(label)}</text>',
                f'<rect x="{chart_left}" y="{top + 6}" width="{chart_width}" height="14" rx="4" fill="#eef2f7"/>',
                f'<rect x="{chart_left}" y="{top + 6}" width="{bar_width:.2f}" height="14" rx="4" fill="{color}"/>',
                f'<text x="{chart_right + 10}" y="{top + 18}" font-size="12" font-family="Segoe UI, Arial, sans-serif" fill="#222222">{_svg_escape(value_text)}</text>',
            ]
        )

    lines.append("</svg>")
    return "\n".join(lines)


def _candidate_table_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Candidate Table",
        "",
        "| candidate | scheduler | workers | ocr_total_sec_median | ocr_page_p95_sec_median | mean_CER | mean_exact_match | gate_pass |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {candidate} | {scheduler} | {workers} | {ocr_total} | {p95} | {cer} | {exact} | {passed} |".format(
                candidate=row.get("candidate_key"),
                scheduler=row.get("scheduler_mode"),
                workers=row.get("parallel_workers"),
                ocr_total=row.get("measured_ocr_total_sec_median"),
                p95=row.get("measured_ocr_page_p95_sec_median"),
                cer=row.get("quality_mean_cer"),
                exact=row.get("quality_mean_exact_match"),
                passed=row.get("quality_gate_pass"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _write_visual_assets(rows: list[dict[str, Any]], latest_dir: Path, history_dir: Path) -> None:
    assets = {
        "ocr_total_sec_median.svg": _render_bar_chart(
            rows,
            title="OCR Total Median by Candidate",
            value_key="measured_ocr_total_sec_median",
            value_suffix="s",
        ),
        "ocr_page_p95_sec_median.svg": _render_bar_chart(
            rows,
            title="OCR Page P95 Median by Candidate",
            value_key="measured_ocr_page_p95_sec_median",
            value_suffix="s",
        ),
        "quality_mean_cer.svg": _render_bar_chart(
            rows,
            title="Mean CER by Candidate",
            value_key="quality_mean_cer",
        ),
    }
    for filename, content in assets.items():
        if not content:
            continue
        (latest_dir / filename).write_text(content, encoding="utf-8")
        (history_dir / filename).write_text(content, encoding="utf-8")
    table_md = _candidate_table_markdown(rows)
    (latest_dir / "candidate_table.md").write_text(table_md, encoding="utf-8")
    (history_dir / "candidate_table.md").write_text(table_md, encoding="utf-8")


def _render_report(summary: dict[str, Any], rows: list[dict[str, Any]], suite_dir: Path) -> str:
    quality_gate_winner = summary.get("quality_gate_winner", {}) or {}
    review_candidates = summary.get("review_candidates", []) or []
    runtime_policy = _load_json_if_exists(suite_dir / "runtime" / "managed_runtime_policy.json")
    container_names = runtime_policy.get("container_names", []) if isinstance(runtime_policy, dict) else []
    review_dir = suite_dir / "review"
    lines = [
        "# PaddleOCR VL Parallel Report",
        "",
        "## Metadata",
        "",
        f"- latest suite dir: `{_repo_relative(suite_dir)}`",
        f"- latest assets dir: `{_repo_relative(LATEST_ASSETS)}`",
        f"- runtime_contract: `{summary.get('runtime_contract', 'paddleocr-vl-single-tenant-ocr-only')}`",
        f"- runtime_services: `{summary.get('runtime_services', 'ocr-only')}`",
        f"- stage_ceiling: `{summary.get('stage_ceiling', 'ocr')}`",
        f"- baseline gold: `{summary.get('baseline_gold_path')}`",
        f"- detector manifest: `{summary.get('detector_manifest_path')}`",
        f"- runtime container names: `{container_names}`",
        f"- gemma-local-server booted: `{'gemma-local-server' in container_names}`",
        "",
        "## Quality Gate Winner",
        "",
        f"- quality_gate_winner: `{quality_gate_winner.get('candidate_key', 'n/a')}`",
        f"- scheduler_mode: `{quality_gate_winner.get('scheduler_mode', 'n/a')}`",
        f"- parallel_workers: `{quality_gate_winner.get('parallel_workers', 'n/a')}`",
        f"- ocr_total_sec_median: `{quality_gate_winner.get('measured_ocr_total_sec_median', 'n/a')}`",
        f"- ocr_page_p95_sec_median: `{quality_gate_winner.get('measured_ocr_page_p95_sec_median', 'n/a')}`",
        "",
        _promotion_title(summary),
        "",
        *_promotion_notes(summary),
        "",
        "## Review Candidates",
        "",
    ]
    if review_candidates:
        for index, item in enumerate(review_candidates, start=1):
            lines.append(
                f"- top{index}: `{item.get('candidate_key', 'n/a')}` "
                f"(ocr_total_sec_median=`{item.get('measured_ocr_total_sec_median', 'n/a')}`, "
                f"gate_pass=`{item.get('quality_gate_pass', 'n/a')}`)"
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Candidate Table",
            "",
            "| candidate | scheduler | workers | ocr_total_sec_median | ocr_page_p95_sec_median | mean_CER | mean_exact_match | gate_pass |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {candidate} | {scheduler} | {workers} | {ocr_total} | {p95} | {cer} | {exact} | {passed} |".format(
                candidate=row.get("candidate_key"),
                scheduler=row.get("scheduler_mode"),
                workers=row.get("parallel_workers"),
                ocr_total=row.get("measured_ocr_total_sec_median"),
                p95=row.get("measured_ocr_page_p95_sec_median"),
                cer=row.get("quality_mean_cer"),
                exact=row.get("quality_mean_exact_match"),
                passed=row.get("quality_gate_pass"),
            )
        )
    lines.extend(
        [
            "",
            "## Visual Assets",
            "",
            f"- OCR total chart: `{_repo_relative(LATEST_ASSETS / 'ocr_total_sec_median.svg')}`",
            f"- OCR page p95 chart: `{_repo_relative(LATEST_ASSETS / 'ocr_page_p95_sec_median.svg')}`",
            f"- Mean CER chart: `{_repo_relative(LATEST_ASSETS / 'quality_mean_cer.svg')}`",
            f"- Candidate table: `{_repo_relative(LATEST_ASSETS / 'candidate_table.md')}`",
            f"- top1 diff review: `{_repo_relative(review_dir / 'top1_diff_review.md')}`",
            f"- top2 diff review: `{_repo_relative(review_dir / 'top2_diff_review.md')}`",
            "",
            "## Portfolio Notes",
            "",
            "- 핵심 문제 해결 방향은 사용자가 착안했다.",
            "- 실험 설계 구체화, 계측 설계, 구현, 검증은 공동 수행으로 정리한다.",
            "- 이번 결과는 Gemma/MangaLMM 혼재 benchmark가 아니라 PaddleOCR VL 단독 상주 상한선 benchmark다.",
            "- 향후 MangaLMM + PaddleOCR VL 동시 상주 환경에서 PaddleOCR VL이 더 큰 VRAM headroom을 얻을 수 있다는 가정 아래, 단독 상주 최대 병렬치를 먼저 확정하는 문맥으로 해석한다.",
            "- 세부 narrative는 `docs/benchmark/paddleocr-vl-parallel/`, latest problem-solving specs, review diff pack을 함께 본다.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_results_history(summary: dict[str, Any], suite_dir: Path) -> None:
    quality_gate_winner = summary.get("quality_gate_winner", {}) or {}
    review_candidates = summary.get("review_candidates", []) or []
    text = "\n".join(
        [
            "# PaddleOCR VL Parallel Results History",
            "",
            "## Current Policy",
            "",
            "- baseline은 항상 `fixed_w8`이다.",
            "- detector freeze와 gold seed는 baseline measured run에서 생성한다.",
            "- 이번 family는 `PaddleOCR VL 단독 상주 상한선 benchmark`다.",
            "- `runtime_services=ocr-only`, `stage_ceiling=ocr` 계약에서 Gemma는 실제 runtime/VRAM 점유에 참여하지 않는다.",
            "- subset winner만으로 자동 `default on` 승격을 하지 않는다. 사용자 OCR diff 검수 승인 상태를 기록한 뒤 develop promotion을 진행한다.",
            "",
            "## Latest Output",
            "",
            f"- latest suite root: `{_repo_relative(suite_dir)}`",
            f"- quality_gate_winner: `{quality_gate_winner.get('candidate_key', 'n/a')}`",
            f"- scheduler_mode: `{quality_gate_winner.get('scheduler_mode', 'n/a')}`",
            f"- ocr_total_sec_median: `{quality_gate_winner.get('measured_ocr_total_sec_median', 'n/a')}`",
            f"- final_promotion_status: `{_promotion_status(summary)}`",
            (
                f"- approved_promotion_candidate: `{_approved_candidate_key(summary)}`"
                if _approved_candidate_key(summary)
                else "- approved_promotion_candidate: `n/a`"
            ),
            f"- review_candidate_keys: `{[item.get('candidate_key', 'n/a') for item in review_candidates]}`",
            f"- baseline gold: `{summary.get('baseline_gold_path')}`",
            f"- detector manifest: `{summary.get('detector_manifest_path')}`",
            "",
        ]
    )
    RESULTS_HISTORY_PATH.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate latest report/assets for paddleocr_vl_parallel.")
    parser.add_argument("--suite-dir", default="")
    parser.add_argument("--final-promotion-status", default="")
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir).resolve() if args.suite_dir else _latest_completed_suite()
    raw_summary = _load_json(suite_dir / "suite_summary.json")
    summary = _patched_summary(raw_summary, args.final_promotion_status)
    rows = _load_csv_rows(suite_dir / "candidate_summary.csv")
    raw_state = _load_json_if_exists(suite_dir / "suite_state.json")
    suite_state = _patched_summary(raw_state, args.final_promotion_status) if raw_state else {}

    history_dir = HISTORY_ROOT / suite_dir.name
    latest_dir = LATEST_ASSETS
    latest_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    for filename in ("candidate_summary.csv", "baseline_gold.json", "detector_manifest.json"):
        _copy_file_if_exists(suite_dir / filename, latest_dir / filename)
        _copy_file_if_exists(suite_dir / filename, history_dir / filename)

    _write_json(latest_dir / "suite_summary.json", summary)
    _write_json(history_dir / "suite_summary.json", summary)
    if suite_state:
        _write_json(latest_dir / "suite_state.json", suite_state)
        _write_json(history_dir / "suite_state.json", suite_state)
    _write_patched_suite_markdown(
        suite_dir / "suite_summary.md",
        latest_dir / "suite_summary.md",
        final_promotion_status=args.final_promotion_status,
    )
    _write_patched_suite_markdown(
        suite_dir / "suite_summary.md",
        history_dir / "suite_summary.md",
        final_promotion_status=args.final_promotion_status,
    )

    for runtime_filename in (
        "managed_runtime_policy.json",
        "runtime_snapshot.json",
        "llama_cpp_runtime.json",
    ):
        _copy_file_if_exists(suite_dir / "runtime" / runtime_filename, latest_dir / "runtime" / runtime_filename)
        _copy_file_if_exists(suite_dir / "runtime" / runtime_filename, history_dir / "runtime" / runtime_filename)

    _copy_tree_if_exists(suite_dir / "problem_solving_specs", latest_dir / "problem_solving_specs")
    _copy_tree_if_exists(suite_dir / "problem_solving_specs", history_dir / "problem_solving_specs")
    _copy_tree_if_exists(suite_dir / "review", latest_dir / "review")
    _copy_tree_if_exists(suite_dir / "review", history_dir / "review")
    _copy_tree_if_exists(suite_dir / "docker_logs", latest_dir / "docker_logs")
    _copy_tree_if_exists(suite_dir / "docker_logs", history_dir / "docker_logs")
    if args.final_promotion_status:
        _write_promotion_spec(
            latest_dir / "problem_solving_specs" / "05_product_promotion.md",
            final_promotion_status=args.final_promotion_status,
        )
        _write_promotion_spec(
            history_dir / "problem_solving_specs" / "05_product_promotion.md",
            final_promotion_status=args.final_promotion_status,
        )

    _write_visual_assets(rows, latest_dir, history_dir)

    report_text = _render_report(summary, rows, suite_dir)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    (LATEST_ASSETS / "latest_summary.md").write_text(report_text, encoding="utf-8")
    (LATEST_ASSETS / "conclusion_card.md").write_text(
        "\n".join(
            [
                "# PaddleOCR VL Parallel Conclusion Card",
                "",
                f"- quality_gate_winner: `{(summary.get('quality_gate_winner') or {}).get('candidate_key', 'n/a')}`",
                f"- ocr_total_sec_median: `{(summary.get('quality_gate_winner') or {}).get('measured_ocr_total_sec_median', 'n/a')}`",
                f"- final_promotion_status: `{_promotion_status(summary)}`",
                (
                    f"- approved_promotion_candidate: `{_approved_candidate_key(summary)}`"
                    if _approved_candidate_key(summary)
                    else "- approved_promotion_candidate: `n/a`"
                ),
                f"- review_candidate_keys: `{summary.get('review_candidate_keys', [])}`",
                "- 핵심 문제 해결 방향은 사용자가 착안했다.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (LATEST_ASSETS / "report_summary.json").write_text(
        json.dumps(
            {
                "suite_dir": _repo_relative(suite_dir),
                "quality_gate_winner": summary.get("quality_gate_winner", {}),
                "review_candidates": summary.get("review_candidates", []),
                "final_promotion_status": _promotion_status(summary),
                "approved_promotion_candidate": _approved_candidate_key(summary) or None,
                "candidate_count": summary.get("candidate_count"),
                "pass_candidate_count": summary.get("pass_candidate_count"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_results_history(summary, suite_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
