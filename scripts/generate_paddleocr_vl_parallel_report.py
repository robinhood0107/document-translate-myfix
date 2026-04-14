#!/usr/bin/env python3
from __future__ import annotations

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
    winner = summary.get("winner", {}) or {}
    lines = [
        "# PaddleOCR VL Parallel Report",
        "",
        "## Metadata",
        "",
        f"- latest suite dir: `{_repo_relative(suite_dir)}`",
        f"- latest assets dir: `{_repo_relative(LATEST_ASSETS)}`",
        f"- baseline gold: `{summary.get('baseline_gold_path')}`",
        f"- detector manifest: `{summary.get('detector_manifest_path')}`",
        "",
        "## Winner",
        "",
        f"- winner: `{winner.get('candidate_key', 'n/a')}`",
        f"- scheduler_mode: `{winner.get('scheduler_mode', 'n/a')}`",
        f"- parallel_workers: `{winner.get('parallel_workers', 'n/a')}`",
        f"- ocr_total_sec_median: `{winner.get('measured_ocr_total_sec_median', 'n/a')}`",
        f"- ocr_page_p95_sec_median: `{winner.get('measured_ocr_page_p95_sec_median', 'n/a')}`",
        "",
        "## Candidate Table",
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
    lines.extend(
        [
            "",
            "## Visual Assets",
            "",
            f"- OCR total chart: `{_repo_relative(LATEST_ASSETS / 'ocr_total_sec_median.svg')}`",
            f"- OCR page p95 chart: `{_repo_relative(LATEST_ASSETS / 'ocr_page_p95_sec_median.svg')}`",
            f"- Mean CER chart: `{_repo_relative(LATEST_ASSETS / 'quality_mean_cer.svg')}`",
            f"- Candidate table: `{_repo_relative(LATEST_ASSETS / 'candidate_table.md')}`",
            "",
            "## Portfolio Notes",
            "",
            "- 핵심 문제 해결 방향은 사용자가 착안했다.",
            "- 실험 설계 구체화, 계측 설계, 구현, 검증은 공동 수행으로 정리한다.",
            "- 세부 narrative는 `docs/benchmark/paddleocr-vl-parallel/`와 latest problem-solving specs를 함께 본다.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_results_history(summary: dict[str, Any], suite_dir: Path) -> None:
    winner = summary.get("winner", {}) or {}
    text = "\n".join(
        [
            "# PaddleOCR VL Parallel Results History",
            "",
            "## Current Policy",
            "",
            "- baseline은 항상 `fixed_w8`이다.",
            "- detector freeze와 gold seed는 baseline measured run에서 생성한다.",
            "- subset winner만으로 `default on` 승격을 하지 않는다.",
            "",
            "## Latest Output",
            "",
            f"- latest suite root: `{_repo_relative(suite_dir)}`",
            f"- winner: `{winner.get('candidate_key', 'n/a')}`",
            f"- scheduler_mode: `{winner.get('scheduler_mode', 'n/a')}`",
            f"- ocr_total_sec_median: `{winner.get('measured_ocr_total_sec_median', 'n/a')}`",
            f"- baseline gold: `{summary.get('baseline_gold_path')}`",
            f"- detector manifest: `{summary.get('detector_manifest_path')}`",
            "",
        ]
    )
    RESULTS_HISTORY_PATH.write_text(text, encoding="utf-8")


def main() -> int:
    suite_dir = _latest_completed_suite()
    summary = _load_json(suite_dir / "suite_summary.json")
    rows = _load_csv_rows(suite_dir / "candidate_summary.csv")

    history_dir = HISTORY_ROOT / suite_dir.name
    latest_dir = LATEST_ASSETS
    latest_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    for filename in (
        "suite_state.json",
        "suite_summary.json",
        "suite_summary.md",
        "candidate_summary.csv",
        "baseline_gold.json",
        "detector_manifest.json",
    ):
        _copy_file_if_exists(suite_dir / filename, latest_dir / filename)
        _copy_file_if_exists(suite_dir / filename, history_dir / filename)

    _copy_tree_if_exists(suite_dir / "problem_solving_specs", latest_dir / "problem_solving_specs")
    _copy_tree_if_exists(suite_dir / "problem_solving_specs", history_dir / "problem_solving_specs")

    _write_visual_assets(rows, latest_dir, history_dir)

    report_text = _render_report(summary, rows, suite_dir)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    (LATEST_ASSETS / "latest_summary.md").write_text(report_text, encoding="utf-8")
    (LATEST_ASSETS / "conclusion_card.md").write_text(
        "\n".join(
            [
                "# PaddleOCR VL Parallel Conclusion Card",
                "",
                f"- winner: `{(summary.get('winner') or {}).get('candidate_key', 'n/a')}`",
                f"- ocr_total_sec_median: `{(summary.get('winner') or {}).get('measured_ocr_total_sec_median', 'n/a')}`",
                f"- quality_gate_pass: `{bool(summary.get('winner'))}`",
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
                "winner": summary.get("winner", {}),
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
