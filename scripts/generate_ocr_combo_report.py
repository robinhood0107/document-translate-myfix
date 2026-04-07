#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_common import repo_relative_str, write_json


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _slugify(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-") or "report"


def _archive_existing_latest(report_path: Path, assets_dir: Path, benchmark_name: str) -> dict[str, str] | None:
    if "history" in report_path.parts:
        return None
    if not report_path.exists() and not assets_dir.exists():
        return None
    snapshot_base = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_slugify(benchmark_name)}"
    snapshot_id = snapshot_base
    suffix = 2
    while (report_path.parent / "history" / snapshot_id).exists() or (assets_dir.parent / "history" / snapshot_id).exists():
        snapshot_id = f"{snapshot_base}-{suffix}"
        suffix += 1
    history_report_dir = report_path.parent / "history" / snapshot_id
    history_assets_dir = assets_dir.parent / "history" / snapshot_id
    history_report_dir.mkdir(parents=True, exist_ok=True)
    if assets_dir.exists():
        shutil.copytree(assets_dir, history_assets_dir)
    if report_path.exists():
        shutil.copy2(report_path, history_report_dir / report_path.name)
    return {
        "snapshot_id": snapshot_id,
        "report_path": repo_relative_str(history_report_dir / report_path.name),
        "assets_dir": repo_relative_str(history_assets_dir),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [str(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _candidate_row(corpus: str, item: dict[str, Any], *, stage: str) -> dict[str, Any]:
    summary = item.get("summary", {}) if isinstance(item.get("summary"), dict) else {}
    compare = item.get("compare", {}) if isinstance(item.get("compare"), dict) else {}
    metrics = compare.get("metrics", {}) if isinstance(compare.get("metrics"), dict) else {}
    return {
        "corpus": corpus,
        "stage": stage,
        "engine": item.get("engine", ""),
        "preset": item.get("preset", ""),
        "elapsed_sec": summary.get("elapsed_sec", ""),
        "ocr_total_sec": summary.get("ocr_total_sec", ""),
        "translate_total_sec": summary.get("stage_stats", {}).get("translate", {}).get("total_sec", "")
        if isinstance(summary.get("stage_stats"), dict)
        else "",
        "gpu_peak_used_mb": summary.get("gpu_peak_used_mb", ""),
        "gpu_floor_free_mb": summary.get("gpu_floor_free_mb", ""),
        "gpu_peak_util_percent": summary.get("gpu_peak_util_percent", ""),
        "quality_gate_pass": item.get("quality_gate_pass", ""),
        "geometry_match_recall": metrics.get("geometry_match_recall", ""),
        "geometry_match_precision": metrics.get("geometry_match_precision", ""),
        "non_empty_retention": metrics.get("non_empty_retention", ""),
        "ocr_char_error_rate": metrics.get("ocr_char_error_rate", ""),
        "page_p95_ocr_char_error_rate": metrics.get("page_p95_ocr_char_error_rate", ""),
        "ocr_exact_text_match_ratio": metrics.get("ocr_exact_text_match_ratio", ""),
        "translation_similarity_ratio": metrics.get("translation_similarity_ratio", ""),
        "hard_gate_failures": "; ".join(compare.get("hard_gate_failures", []))
        if isinstance(compare.get("hard_gate_failures"), list)
        else "",
    }


def _winner_row(corpus_payload: dict[str, Any]) -> dict[str, Any]:
    final_confirm = corpus_payload.get("final_confirm", {}) if isinstance(corpus_payload.get("final_confirm"), dict) else {}
    winner = corpus_payload.get("winner", {}) if isinstance(corpus_payload.get("winner"), dict) else {}
    return {
        "corpus": corpus_payload.get("corpus", ""),
        "engine": winner.get("engine", ""),
        "preset": winner.get("preset", ""),
        "official_score_elapsed_median_sec": final_confirm.get("official_score_elapsed_median_sec", ""),
        "ocr_median_sec": final_confirm.get("tie_breaker_ocr_median_sec", ""),
        "translate_median_sec": final_confirm.get("tie_breaker_translate_median_sec", ""),
        "gpu_peak_used_mb": final_confirm.get("tie_breaker_gpu_peak_used_mb", ""),
        "all_quality_gate_pass": final_confirm.get("all_quality_gate_pass", ""),
        "promotion_recommended": corpus_payload.get("promotion_recommended", False),
    }


def _review_row(corpus_payload: dict[str, Any]) -> dict[str, Any]:
    examples = corpus_payload.get("review_examples", []) if isinstance(corpus_payload.get("review_examples"), list) else []
    sample_example = examples[0] if examples else {}
    return {
        "corpus": corpus_payload.get("corpus", ""),
        "gold_review_status": corpus_payload.get("gold_review_status", ""),
        "gold_path": corpus_payload.get("gold_path", ""),
        "gold_review_packet_dir": corpus_payload.get("gold_review_packet_dir", ""),
        "gold_generated_from_run_dir": corpus_payload.get("gold_generated_from_run_dir", ""),
        "gold_page_count": corpus_payload.get("gold_page_count", ""),
        "example_page": sample_example.get("page", ""),
        "example_source_image": sample_example.get("source_image", ""),
        "example_overlay_image": sample_example.get("overlay_image", ""),
        "example_ocr_debug": sample_example.get("ocr_debug", ""),
    }


def _visual_lines(corpus_payload: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    examples = corpus_payload.get("visual_examples", []) if isinstance(corpus_payload.get("visual_examples"), list) else []
    if not examples:
        return ["- none"]
    for example in examples:
        if not isinstance(example, dict):
            continue
        lines.extend(
            [
                f"- page `{example.get('page', '')}`",
                f"  source: `{example.get('source_image', '')}`",
                f"  overlay: `{example.get('overlay_image', '')}`",
                f"  winner_translated_image: `{example.get('winner_translated_image', '')}`",
                f"  fastest_failed_translated_image: `{example.get('fastest_failed_translated_image', '')}`",
            ]
        )
    return lines or ["- none"]


def _smoke_row(item: dict[str, Any]) -> dict[str, Any]:
    summary = item.get("summary", {}) if isinstance(item.get("summary"), dict) else {}
    return {
        "corpus": item.get("corpus", ""),
        "engine": item.get("engine", ""),
        "elapsed_sec": summary.get("elapsed_sec", ""),
        "ocr_total_sec": summary.get("ocr_total_sec", ""),
        "translate_total_sec": summary.get("stage_stats", {}).get("translate", {}).get("total_sec", "")
        if isinstance(summary.get("stage_stats"), dict)
        else "",
        "run_dir": item.get("run_dir", ""),
    }


def _render_bootstrap_markdown(
    *,
    manifest: dict[str, Any],
    benchmark: dict[str, Any],
    review_rows: list[dict[str, Any]],
    assets_dir: Path,
) -> str:
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        "# OCR Combo Benchmark Report",
        "",
        "이 파일은 `scripts/generate_ocr_combo_report.py`가 suite manifest를 기준으로 갱신합니다.",
        "",
        "## Metadata",
        "",
        f"- generated_at: `{generated_at}`",
        f"- status: `{manifest.get('status', '')}`",
        f"- benchmark_name: `{benchmark.get('name', '')}`",
        f"- benchmark_kind: `{benchmark.get('kind', '')}`",
        f"- benchmark_scope: `{benchmark.get('scope', '')}`",
        f"- execution_scope: `{benchmark.get('execution_scope', '')}`",
        f"- speed_score_scope: `{benchmark.get('speed_score_scope', '')}`",
        f"- quality_gate_scope: `{benchmark.get('quality_gate_scope', '')}`",
        f"- gold_source: `{benchmark.get('gold_source', '')}`",
        f"- gold_empty_text_policy: `{benchmark.get('gold_empty_text_policy', '')}`",
        f"- crop_regression_focus: `{benchmark.get('crop_regression_focus', '')}`",
        f"- baseline_sha: `{benchmark.get('baseline_sha', '')}`",
        f"- develop_ref_sha: `{benchmark.get('develop_ref_sha', '')}`",
        f"- entrypoint: `{benchmark.get('entrypoint', '')}`",
        f"- results_root: `{manifest.get('results_root', '')}`",
        "",
        "## Fixed Gemma",
        "",
        f"- image: `{(benchmark.get('fixed_gemma') or {}).get('image', '')}`",
        f"- response_format_mode: `{(benchmark.get('fixed_gemma') or {}).get('response_format_mode', '')}`",
        f"- chunk_size: `{(benchmark.get('fixed_gemma') or {}).get('chunk_size', '')}`",
        f"- temperature: `{(benchmark.get('fixed_gemma') or {}).get('temperature', '')}`",
        f"- n_gpu_layers: `{(benchmark.get('fixed_gemma') or {}).get('n_gpu_layers', '')}`",
        "",
        "## OCR Gate Rules",
        "",
        f"- canonical_small_voiced_kana: `{((benchmark.get('ocr_normalization') or {}).get('canonical_small_voiced_kana', ''))}`",
        f"- ignored_chars: `{((benchmark.get('ocr_normalization') or {}).get('ignored_chars', ''))}`",
        "- `gold_text=\"\"` block은 geometry를 유지하되 non-empty OCR text hard gate에서는 제외합니다.",
        "",
        "## Awaiting Gold Review",
        "",
        "이번 latest run은 사람 검수 OCR gold가 아직 잠기지 않아 bootstrap 모드로 종료되었습니다.",
        "다음 단계는 `benchmarks/ocr_combo/gold/<corpus>/gold.json`을 검수해 `review_status=locked`로 저장한 뒤 같은 명령을 다시 실행하는 것입니다.",
        "",
        "## Gold Review Packets",
        "",
        _md_table(
            review_rows,
            [
                "corpus",
                "gold_review_status",
                "gold_path",
                "gold_review_packet_dir",
                "gold_generated_from_run_dir",
                "gold_page_count",
                "example_page",
            ],
        )
        if review_rows
        else "_none_",
        "",
        "검수는 [gold-review-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/benchmark/ocr-combo/gold-review-ko.md)의 절차를 따릅니다.",
        "",
        "## Artifacts",
        "",
        f"- gold review csv: `{repo_relative_str(assets_dir / 'gold_review_packets.csv')}`",
    ]
    return "\n".join(lines)


def _render_benchmark_markdown(
    *,
    manifest: dict[str, Any],
    benchmark: dict[str, Any],
    default_rows: list[dict[str, Any]],
    tuning_rows: list[dict[str, Any]],
    winner_rows: list[dict[str, Any]],
    smoke_rows: list[dict[str, Any]],
    assets_dir: Path,
) -> str:
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        "# OCR Combo Benchmark Report",
        "",
        "이 파일은 `scripts/generate_ocr_combo_report.py`가 suite manifest를 기준으로 갱신합니다.",
        "",
        "## Metadata",
        "",
        f"- generated_at: `{generated_at}`",
        f"- status: `{manifest.get('status', '')}`",
        f"- benchmark_name: `{benchmark.get('name', '')}`",
        f"- benchmark_kind: `{benchmark.get('kind', '')}`",
        f"- benchmark_scope: `{benchmark.get('scope', '')}`",
        f"- execution_scope: `{benchmark.get('execution_scope', '')}`",
        f"- speed_score_scope: `{benchmark.get('speed_score_scope', '')}`",
        f"- quality_gate_scope: `{benchmark.get('quality_gate_scope', '')}`",
        f"- gold_source: `{benchmark.get('gold_source', '')}`",
        f"- gold_empty_text_policy: `{benchmark.get('gold_empty_text_policy', '')}`",
        f"- crop_regression_focus: `{benchmark.get('crop_regression_focus', '')}`",
        f"- baseline_sha: `{benchmark.get('baseline_sha', '')}`",
        f"- develop_ref_sha: `{benchmark.get('develop_ref_sha', '')}`",
        f"- entrypoint: `{benchmark.get('entrypoint', '')}`",
        f"- results_root: `{manifest.get('results_root', '')}`",
        "",
        "## Fixed Gemma",
        "",
        f"- image: `{(benchmark.get('fixed_gemma') or {}).get('image', '')}`",
        f"- response_format_mode: `{(benchmark.get('fixed_gemma') or {}).get('response_format_mode', '')}`",
        f"- chunk_size: `{(benchmark.get('fixed_gemma') or {}).get('chunk_size', '')}`",
        f"- temperature: `{(benchmark.get('fixed_gemma') or {}).get('temperature', '')}`",
        f"- n_gpu_layers: `{(benchmark.get('fixed_gemma') or {}).get('n_gpu_layers', '')}`",
        "",
        "## OCR Gate Rules",
        "",
        f"- canonical_small_voiced_kana: `{((benchmark.get('ocr_normalization') or {}).get('canonical_small_voiced_kana', ''))}`",
        f"- ignored_chars: `{((benchmark.get('ocr_normalization') or {}).get('ignored_chars', ''))}`",
        "- `gold_text=\"\"` block은 geometry를 유지하되 non-empty OCR text hard gate에서는 제외합니다.",
        "- crop overreach 회귀는 `p_018` 사례를 기준으로 추적합니다.",
        "",
        "## Corpora",
        "",
        "| corpus | sample_dir | sample_count | source_lang | target_lang | gold_path | gold_review_status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in manifest.get("corpora", []):
        lines.append(
            f"| {item.get('corpus', '')} | {item.get('sample_dir', '')} | {item.get('sample_count', '')} | "
            f"{item.get('source_lang', '')} | {item.get('target_lang', '')} | {item.get('gold_path', '')} | "
            f"{item.get('gold_review_status', '')} |"
        )

    lines.extend(
        [
            "",
            "## Smoke",
            "",
            _md_table(
                smoke_rows,
                ["corpus", "engine", "elapsed_sec", "ocr_total_sec", "translate_total_sec", "run_dir"],
            )
            if smoke_rows
            else "_none_",
            "",
            "## Default Comparison",
            "",
            _md_table(
                default_rows,
                [
                    "corpus",
                    "engine",
                    "elapsed_sec",
                    "ocr_total_sec",
                    "translate_total_sec",
                    "gpu_peak_used_mb",
                    "gpu_floor_free_mb",
                    "quality_gate_pass",
                    "geometry_match_recall",
                    "geometry_match_precision",
                    "non_empty_retention",
                    "ocr_char_error_rate",
                    "page_p95_ocr_char_error_rate",
                    "ocr_exact_text_match_ratio",
                ],
            )
            if default_rows
            else "_none_",
            "",
            "## Tuning Results",
            "",
            _md_table(
                tuning_rows,
                [
                    "corpus",
                    "stage",
                    "engine",
                    "elapsed_sec",
                    "ocr_total_sec",
                    "translate_total_sec",
                    "gpu_peak_used_mb",
                    "quality_gate_pass",
                    "geometry_match_recall",
                    "geometry_match_precision",
                    "non_empty_retention",
                    "ocr_char_error_rate",
                    "page_p95_ocr_char_error_rate",
                ],
            )
            if tuning_rows
            else "_none_",
            "",
            "## Winners",
            "",
            _md_table(
                winner_rows,
                [
                    "corpus",
                    "engine",
                    "official_score_elapsed_median_sec",
                    "ocr_median_sec",
                    "translate_median_sec",
                    "gpu_peak_used_mb",
                    "all_quality_gate_pass",
                    "promotion_recommended",
                ],
            )
            if winner_rows
            else "_none_",
            "",
            "## Language Routing Policy",
            "",
            f"- China corpus 권장 OCR: `{(manifest.get('routing_policy') or {}).get('china', '')}`",
            f"- japan corpus 권장 OCR: `{(manifest.get('routing_policy') or {}).get('japan', '')}`",
            f"- mixed corpus 운영 권장 라우팅: {(manifest.get('routing_policy') or {}).get('mixed', '')}",
            "",
            "## Visual Appendix",
            "",
        ]
    )
    for corpus_payload in manifest.get("corpora", []):
        lines.append(f"### {corpus_payload.get('corpus', '')}")
        lines.extend(_visual_lines(corpus_payload))
        lines.append("")

    lines.extend(
        [
            "## Artifacts",
            "",
            f"- smoke csv: `{repo_relative_str(assets_dir / 'smoke_results.csv')}`",
            f"- default comparison csv: `{repo_relative_str(assets_dir / 'default_comparison.csv')}`",
            f"- tuning results csv: `{repo_relative_str(assets_dir / 'tuning_results.csv')}`",
            f"- winners csv: `{repo_relative_str(assets_dir / 'winners.csv')}`",
        ]
    )
    return "\n".join(lines)


def generate_report(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_yaml(manifest_path)
    benchmark = manifest.get("benchmark", {}) if isinstance(manifest.get("benchmark"), dict) else {}
    report_cfg = manifest.get("report", {}) if isinstance(manifest.get("report"), dict) else {}
    report_path = ROOT / str(report_cfg.get("markdown_output", "docs/banchmark_report/ocr-combo-report-ko.md"))
    assets_dir = ROOT / str(report_cfg.get("assets_dir", "docs/assets/benchmarking/ocr-combo/latest"))

    archived = _archive_existing_latest(report_path, assets_dir, str(benchmark.get("name", "OCR Combo Runtime Benchmark")))
    assets_dir.mkdir(parents=True, exist_ok=True)

    summary_payload: dict[str, Any] = {
        "manifest": repo_relative_str(manifest_path),
        "results_root": manifest.get("results_root", ""),
        "archived_previous_latest": archived,
        "routing_policy": manifest.get("routing_policy", {}),
        "status": manifest.get("status", ""),
        "winners": [],
    }

    if manifest.get("status") == "awaiting_gold_review":
        review_rows = [_review_row(item) for item in manifest.get("corpora", [])]
        _write_csv(assets_dir / "gold_review_packets.csv", review_rows)
        markdown = _render_bootstrap_markdown(
            manifest=manifest,
            benchmark=benchmark,
            review_rows=review_rows,
            assets_dir=assets_dir,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(markdown, encoding="utf-8")
        summary_payload["gold_review_packets"] = review_rows
        write_json(assets_dir / "report_summary.json", summary_payload)
        return summary_payload

    default_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    winner_rows: list[dict[str, Any]] = []
    smoke_rows: list[dict[str, Any]] = []

    for item in manifest.get("smoke_results", []):
        if isinstance(item, dict):
            smoke_rows.append(_smoke_row(item))

    for corpus_payload in manifest.get("corpora", []):
        corpus_name = str(corpus_payload.get("corpus", "") or "")
        for item in corpus_payload.get("default_candidates", []):
            if isinstance(item, dict):
                default_rows.append(_candidate_row(corpus_name, item, stage="default"))
        for tuning_payload in corpus_payload.get("tuning_results", []):
            if not isinstance(tuning_payload, dict):
                continue
            for step in tuning_payload.get("steps", []):
                if not isinstance(step, dict):
                    continue
                for item in step.get("candidates", []):
                    if isinstance(item, dict):
                        tuning_rows.append(_candidate_row(corpus_name, item, stage=str(step.get("step", ""))))
        winner_rows.append(_winner_row(corpus_payload))

    _write_csv(assets_dir / "smoke_results.csv", smoke_rows)
    _write_csv(assets_dir / "default_comparison.csv", default_rows)
    _write_csv(assets_dir / "tuning_results.csv", tuning_rows)
    _write_csv(assets_dir / "winners.csv", winner_rows)

    markdown = _render_benchmark_markdown(
        manifest=manifest,
        benchmark=benchmark,
        default_rows=default_rows,
        tuning_rows=tuning_rows,
        winner_rows=winner_rows,
        smoke_rows=smoke_rows,
        assets_dir=assets_dir,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown, encoding="utf-8")

    summary_payload["winners"] = winner_rows
    write_json(assets_dir / "report_summary.json", summary_payload)
    return summary_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate OCR combo benchmark report from a suite manifest.")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    summary = generate_report(Path(args.manifest))
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
