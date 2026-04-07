#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
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
    summary = item.get("summary", {})
    compare = item.get("compare", {})
    metrics = compare.get("metrics", {})
    return {
        "corpus": corpus,
        "stage": stage,
        "engine": item.get("engine", ""),
        "preset": item.get("preset", ""),
        "elapsed_sec": summary.get("elapsed_sec", ""),
        "ocr_total_sec": summary.get("ocr_total_sec", ""),
        "translate_total_sec": summary.get("stage_stats", {}).get("translate", {}).get("total_sec", ""),
        "gpu_peak_used_mb": summary.get("gpu_peak_used_mb", ""),
        "gpu_floor_free_mb": summary.get("gpu_floor_free_mb", ""),
        "gpu_peak_util_percent": summary.get("gpu_peak_util_percent", ""),
        "quality_gate_pass": item.get("quality_gate_pass", ""),
        "geometry_match_recall": metrics.get("geometry_match_recall", ""),
        "translation_similarity_avg": metrics.get("page_translation_similarity_avg", ""),
    }


def _winner_row(corpus_payload: dict[str, Any]) -> dict[str, Any]:
    final_confirm = corpus_payload.get("final_confirm", {}) or {}
    return {
        "corpus": corpus_payload.get("corpus", ""),
        "engine": (corpus_payload.get("winner") or {}).get("engine", ""),
        "preset": (corpus_payload.get("winner") or {}).get("preset", ""),
        "official_score_elapsed_median_sec": final_confirm.get("official_score_elapsed_median_sec", ""),
        "translate_median_sec": final_confirm.get("tie_breaker_translate_median_sec", ""),
        "ocr_median_sec": final_confirm.get("tie_breaker_ocr_median_sec", ""),
        "gpu_peak_used_mb": final_confirm.get("tie_breaker_gpu_peak_used_mb", ""),
        "promotion_recommended": corpus_payload.get("promotion_recommended", False),
    }


def generate_report(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_yaml(manifest_path)
    benchmark = manifest.get("benchmark", {})
    report_cfg = manifest.get("report", {})
    report_path = ROOT / str(report_cfg.get("markdown_output", "docs/banchmark_report/ocr-combo-report-ko.md"))
    assets_dir = ROOT / str(report_cfg.get("assets_dir", "docs/assets/benchmarking/ocr-combo/latest"))

    archived = _archive_existing_latest(report_path, assets_dir, str(benchmark.get("name", "OCR Combo Runtime Benchmark")))
    assets_dir.mkdir(parents=True, exist_ok=True)

    default_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    winner_rows: list[dict[str, Any]] = []
    for corpus_payload in manifest.get("corpora", []):
        corpus_name = str(corpus_payload.get("corpus", "") or "")
        for item in corpus_payload.get("default_candidates", []):
            default_rows.append(_candidate_row(corpus_name, item, stage="default"))
        for tuning_payload in corpus_payload.get("tuning_results", []):
            for step in tuning_payload.get("steps", []):
                for item in step.get("candidates", []):
                    tuning_rows.append(_candidate_row(corpus_name, item, stage=str(step.get("step", ""))))
        winner_rows.append(_winner_row(corpus_payload))

    _write_csv(assets_dir / "default_comparison.csv", default_rows)
    _write_csv(assets_dir / "tuning_results.csv", tuning_rows)
    _write_csv(assets_dir / "winners.csv", winner_rows)

    routing = manifest.get("routing_policy", {})
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    markdown = "\n".join(
        [
            "# OCR Combo Benchmark Report",
            "",
            "이 파일은 `scripts/generate_ocr_combo_report.py`가 suite manifest를 기준으로 갱신합니다.",
            "",
            "## Metadata",
            "",
            f"- generated_at: `{generated_at}`",
            f"- benchmark_name: `{benchmark.get('name', '')}`",
            f"- benchmark_kind: `{benchmark.get('kind', '')}`",
            f"- benchmark_scope: `{benchmark.get('scope', '')}`",
            f"- execution_scope: `{benchmark.get('execution_scope', '')}`",
            f"- official_score_scope: `{benchmark.get('official_score_scope', '')}`",
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
            "## Corpora",
            "",
            "| corpus | sample_dir | sample_count | source_lang | target_lang |",
            "| --- | --- | --- | --- | --- |",
        ]
        + [
            f"| {item.get('corpus', '')} | {item.get('sample_dir', '')} | {item.get('sample_count', '')} | {item.get('source_lang', '')} | {item.get('target_lang', '')} |"
            for item in manifest.get("corpora", [])
        ]
        + [
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
                    "gpu_peak_used_mb",
                    "gpu_floor_free_mb",
                    "quality_gate_pass",
                    "geometry_match_recall",
                    "translation_similarity_avg",
                ],
            ) if default_rows else "_none_",
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
                    "gpu_peak_used_mb",
                    "quality_gate_pass",
                    "geometry_match_recall",
                    "translation_similarity_avg",
                ],
            ) if tuning_rows else "_none_",
            "",
            "## Winners",
            "",
            _md_table(
                winner_rows,
                [
                    "corpus",
                    "engine",
                    "official_score_elapsed_median_sec",
                    "translate_median_sec",
                    "ocr_median_sec",
                    "gpu_peak_used_mb",
                    "promotion_recommended",
                ],
            ) if winner_rows else "_none_",
            "",
            "## Language Routing Policy",
            "",
            f"- China corpus 권장 OCR: `{routing.get('china', '')}`",
            f"- japan corpus 권장 OCR: `{routing.get('japan', '')}`",
            f"- 혼합 운영 권장 라우팅: {routing.get('mixed', '')}",
            "",
            "## Artifacts",
            "",
            f"- default comparison csv: `{repo_relative_str(assets_dir / 'default_comparison.csv')}`",
            f"- tuning results csv: `{repo_relative_str(assets_dir / 'tuning_results.csv')}`",
            f"- winners csv: `{repo_relative_str(assets_dir / 'winners.csv')}`",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown, encoding="utf-8")

    summary_payload = {
        "manifest": repo_relative_str(manifest_path),
        "results_root": manifest.get("results_root", ""),
        "archived_previous_latest": archived,
        "routing_policy": routing,
        "winners": winner_rows,
    }
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
