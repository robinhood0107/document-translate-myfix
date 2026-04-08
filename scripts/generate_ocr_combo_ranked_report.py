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
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
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
        "translate_median_sec": summary.get("translate_median_sec", ""),
        "gpu_peak_used_mb": summary.get("gpu_peak_used_mb", ""),
        "quality_band": compare.get("quality_band", ""),
        "quality_gate_pass": compare.get("quality_gate_pass", ""),
        "geometry_match_recall": metrics.get("geometry_match_recall", ""),
        "geometry_match_precision": metrics.get("geometry_match_precision", ""),
        "non_empty_retention": metrics.get("non_empty_retention", ""),
        "ocr_char_error_rate": metrics.get("ocr_char_error_rate", ""),
        "ocr_word_error_rate": metrics.get("ocr_word_error_rate", ""),
        "page_p95_ocr_char_error_rate": metrics.get("page_p95_ocr_char_error_rate", ""),
        "overgenerated_block_rate": metrics.get("overgenerated_block_rate", ""),
        "page_failed_rate": metrics.get("page_failed_rate", ""),
        "page_failed_count": summary.get("page_failed_count", ""),
        "gemma_truncated_count": summary.get("gemma_truncated_count", ""),
    }


def _tuning_rows(corpus_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tuning_result in corpus_payload.get("tuning_results", []):
        if not isinstance(tuning_result, dict):
            continue
        engine = tuning_result.get("engine", "")
        for step in tuning_result.get("steps", []):
            step_name = step.get("step", "")
            for candidate in step.get("candidates", []):
                row = _candidate_row(corpus_payload.get("corpus", ""), candidate, stage=f"tuning:{step_name}")
                row["engine"] = engine
                rows.append(row)
    return rows


def _engine_best_rows(corpus_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in corpus_payload.get("engine_best_candidates", []):
        if not isinstance(candidate, dict):
            continue
        rows.append(_candidate_row(corpus_payload.get("corpus", ""), candidate, stage="engine-best"))
    return rows


def _final_confirm_rows(corpus_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    final_confirm = corpus_payload.get("final_confirm", {})
    if not isinstance(final_confirm, dict):
        return rows
    for idx, run in enumerate(final_confirm.get("runs", []), start=1):
        if not isinstance(run, dict):
            continue
        row = _candidate_row(corpus_payload.get("corpus", ""), run, stage=f"final-confirm-r{idx}")
        row["winner_status"] = final_confirm.get("winner_status", "")
        rows.append(row)
    return rows


def _visual_lines(corpus_payload: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for example in corpus_payload.get("visual_examples", []):
        if not isinstance(example, dict):
            continue
        lines.extend(
            [
                f"- page `{example.get('page', '')}`",
                f"  source: `{example.get('source_image', '')}`",
                f"  overlay: `{example.get('overlay_image', '')}`",
                f"  winner: `{example.get('winner_translated_image', '')}`",
                f"  fastest: `{example.get('fastest_translated_image', '')}`",
                f"  lowest_cer: `{example.get('lowest_cer_translated_image', '')}`",
                f"  ppocr: `{example.get('ppocr_translated_image', '')}`",
            ]
        )
    return lines or ["- none"]


def _regression_lines(corpus_payload: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for example in corpus_payload.get("regression_examples", []):
        if not isinstance(example, dict):
            continue
        lines.extend(
            [
                f"- regression page `{example.get('page', '')}`",
                f"  winner_ocr_debug: `{example.get('winner_ocr_debug', '')}`",
                f"  ppocr_ocr_debug: `{example.get('ppocr_ocr_debug', '')}`",
                f"  paddle_ocr_debug: `{example.get('paddle_ocr_debug', '')}`",
            ]
        )
    return lines or ["- none"]


def _render_markdown(
    *,
    manifest: dict[str, Any],
    benchmark: dict[str, Any],
    china_frozen: dict[str, Any],
    japan_payload: dict[str, Any],
    smoke_rows: list[dict[str, Any]],
    default_rows: list[dict[str, Any]],
    tuning_rows: list[dict[str, Any]],
    engine_best_rows: list[dict[str, Any]],
    final_confirm_rows: list[dict[str, Any]],
    assets_dir: Path,
) -> str:
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    final_confirm = japan_payload.get("final_confirm", {}) if isinstance(japan_payload.get("final_confirm"), dict) else {}
    benchmark_winner = japan_payload.get("benchmark_winner", {}) if isinstance(japan_payload.get("benchmark_winner"), dict) else {}
    lines = [
        "# OCR Combo Ranked Report",
        "",
        "이 파일은 `scripts/generate_ocr_combo_ranked_report.py`가 ranked suite manifest를 기준으로 갱신합니다.",
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
        "## OCR Normalization",
        "",
        f"- canonical_small_voiced_kana: `{((benchmark.get('ocr_normalization') or {}).get('canonical_small_voiced_kana', ''))}`",
        f"- ignored_chars: `{((benchmark.get('ocr_normalization') or {}).get('ignored_chars', ''))}`",
        f"- gold_empty_text_policy: `{benchmark.get('gold_empty_text_policy', '')}`",
        "",
        "## China Frozen Winner",
        "",
        f"- source_run_dir: `{china_frozen.get('source_run_dir', '')}`",
        f"- winner_engine: `{china_frozen.get('winner_engine', '')}`",
        f"- winner_preset: `{china_frozen.get('winner_preset', '')}`",
        f"- official_score_elapsed_median_sec: `{china_frozen.get('official_score_elapsed_median_sec', '')}`",
        f"- winner_status: `{china_frozen.get('winner_status', '')}`",
        f"- promotion_recommended: `{china_frozen.get('promotion_recommended', '')}`",
        "",
        "## Japan Benchmark Winner",
        "",
        f"- benchmark_winner: `{benchmark_winner.get('engine', '')}`",
        f"- winner_preset: `{benchmark_winner.get('preset', '')}`",
        f"- winner_status: `{japan_payload.get('winner_status', '')}`",
        f"- promotion_recommended: `{japan_payload.get('promotion_recommended', False)}`",
        f"- official_score_elapsed_median_sec: `{final_confirm.get('official_score_elapsed_median_sec', '')}`",
        f"- ocr_median_sec: `{final_confirm.get('tie_breaker_ocr_median_sec', '')}`",
        f"- translate_median_sec: `{final_confirm.get('tie_breaker_translate_median_sec', '')}`",
        f"- gpu_peak_used_mb: `{final_confirm.get('tie_breaker_gpu_peak_used_mb', '')}`",
        "",
        "## Mixed Routing Policy",
        "",
        f"- china: `{(manifest.get('routing_policy') or {}).get('china', '')}`",
        f"- japan: `{(manifest.get('routing_policy') or {}).get('japan', '')}`",
        f"- mixed: `{(manifest.get('routing_policy') or {}).get('mixed', '')}`",
        "",
        "## Smoke Results",
        "",
        _md_table(
            smoke_rows,
            ["corpus", "engine", "elapsed_sec", "ocr_total_sec", "translate_median_sec", "gpu_peak_used_mb"],
        )
        if smoke_rows
        else "_none_",
        "",
        "## Japan Default Compare",
        "",
        _md_table(
            default_rows,
            [
                "engine",
                "preset",
                "elapsed_sec",
                "quality_band",
                "ocr_char_error_rate",
                "ocr_word_error_rate",
                "page_p95_ocr_char_error_rate",
                "overgenerated_block_rate",
                "page_failed_count",
                "gemma_truncated_count",
            ],
        )
        if default_rows
        else "_none_",
        "",
        "## Japan Tuning Ladder",
        "",
        _md_table(
            tuning_rows,
            [
                "engine",
                "stage",
                "preset",
                "elapsed_sec",
                "quality_band",
                "ocr_char_error_rate",
                "page_p95_ocr_char_error_rate",
                "overgenerated_block_rate",
            ],
        )
        if tuning_rows
        else "_none_",
        "",
        "## Japan Engine Best Presets",
        "",
        _md_table(
            engine_best_rows,
            [
                "engine",
                "preset",
                "elapsed_sec",
                "quality_band",
                "ocr_char_error_rate",
                "ocr_word_error_rate",
                "page_p95_ocr_char_error_rate",
                "overgenerated_block_rate",
                "page_failed_count",
                "gemma_truncated_count",
            ],
        )
        if engine_best_rows
        else "_none_",
        "",
        "## Japan Final Confirm",
        "",
        _md_table(
            final_confirm_rows,
            [
                "engine",
                "stage",
                "elapsed_sec",
                "quality_band",
                "ocr_char_error_rate",
                "page_p95_ocr_char_error_rate",
                "winner_status",
            ],
        )
        if final_confirm_rows
        else "_none_",
        "",
        "## Visual Appendix",
        "",
        *(_visual_lines(japan_payload)),
        "",
        "## Crop Overread Regression",
        "",
        *(_regression_lines(japan_payload)),
        "",
        "## Assets",
        "",
        f"- smoke csv: `{repo_relative_str(assets_dir / 'smoke_results.csv')}`",
        f"- default compare csv: `{repo_relative_str(assets_dir / 'default_compare.csv')}`",
        f"- tuning csv: `{repo_relative_str(assets_dir / 'tuning_results.csv')}`",
        f"- engine best csv: `{repo_relative_str(assets_dir / 'engine_best.csv')}`",
        f"- final confirm csv: `{repo_relative_str(assets_dir / 'final_confirm.csv')}`",
        f"- report summary json: `{repo_relative_str(assets_dir / 'report_summary.json')}`",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate OCR combo ranked benchmark report.")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = _load_yaml(manifest_path)
    benchmark = manifest.get("benchmark", {}) if isinstance(manifest.get("benchmark"), dict) else {}
    report_cfg = manifest.get("report", {}) if isinstance(manifest.get("report"), dict) else {}
    china_frozen = manifest.get("china_frozen", {}) if isinstance(manifest.get("china_frozen"), dict) else {}
    corpora = manifest.get("corpora", []) if isinstance(manifest.get("corpora"), list) else []
    japan_payload = next((item for item in corpora if isinstance(item, dict) and item.get("corpus") == "japan"), {})

    report_path = ROOT / str(report_cfg.get("markdown_output", "docs/banchmark_report/ocr-combo-ranked-report-ko.md"))
    assets_dir = ROOT / str(report_cfg.get("assets_dir", "docs/assets/benchmarking/ocr-combo-ranked/latest"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    archived = _archive_existing_latest(report_path, assets_dir, str(benchmark.get("name", "ocr-combo-ranked")))
    if assets_dir.exists():
        shutil.rmtree(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    smoke_rows = [
        _candidate_row(item.get("corpus", ""), item, stage="smoke")
        for item in manifest.get("smoke_results", [])
        if isinstance(item, dict)
    ]
    default_rows = [
        _candidate_row(japan_payload.get("corpus", ""), item, stage="default")
        for item in japan_payload.get("default_candidates", [])
        if isinstance(item, dict)
    ]
    tuning_rows = _tuning_rows(japan_payload)
    engine_best_rows = _engine_best_rows(japan_payload)
    final_confirm_rows = _final_confirm_rows(japan_payload)

    _write_csv(assets_dir / "smoke_results.csv", smoke_rows)
    _write_csv(assets_dir / "default_compare.csv", default_rows)
    _write_csv(assets_dir / "tuning_results.csv", tuning_rows)
    _write_csv(assets_dir / "engine_best.csv", engine_best_rows)
    _write_csv(assets_dir / "final_confirm.csv", final_confirm_rows)

    summary_payload = {
        "status": manifest.get("status", ""),
        "benchmark_name": benchmark.get("name", ""),
        "execution_scope": benchmark.get("execution_scope", ""),
        "speed_score_scope": benchmark.get("speed_score_scope", ""),
        "quality_gate_scope": benchmark.get("quality_gate_scope", ""),
        "gold_source": benchmark.get("gold_source", ""),
        "china_frozen": china_frozen,
        "japan_winner": {
            "benchmark_winner": ((japan_payload.get("benchmark_winner") or {}).get("engine", "")),
            "winner_status": japan_payload.get("winner_status", ""),
            "promotion_recommended": japan_payload.get("promotion_recommended", False),
            "official_score_elapsed_median_sec": (
                (japan_payload.get("final_confirm") or {}).get("official_score_elapsed_median_sec", "")
            ),
        },
        "routing_policy": manifest.get("routing_policy", {}),
        "archived_previous_latest": archived or {},
    }
    write_json(assets_dir / "report_summary.json", summary_payload)

    markdown = _render_markdown(
        manifest=manifest,
        benchmark=benchmark,
        china_frozen=china_frozen,
        japan_payload=japan_payload,
        smoke_rows=smoke_rows,
        default_rows=default_rows,
        tuning_rows=tuning_rows,
        engine_best_rows=engine_best_rows,
        final_confirm_rows=final_confirm_rows,
        assets_dir=assets_dir,
    )
    report_path.write_text(markdown, encoding="utf-8")
    print(json.dumps({"report": repo_relative_str(report_path), "assets": repo_relative_str(assets_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
