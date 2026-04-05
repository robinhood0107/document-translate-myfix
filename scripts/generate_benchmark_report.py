#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_common import load_preset, repo_relative_str, write_json


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _load_run(root: Path, run_name: str) -> dict[str, Any]:
    run_dir = root / run_name
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary.json for {run_name}")
    request_path = run_dir / "benchmark_request.json"
    audit_path = run_dir / "translation_audit.json"
    preset_path = run_dir / "preset_resolved.json"
    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "run_dir_rel": repo_relative_str(run_dir),
        "summary": _load_json(summary_path),
        "request": _load_json(request_path) if request_path.is_file() else {},
        "audit": _load_json(audit_path) if audit_path.is_file() else {},
        "preset": _load_json(preset_path) if preset_path.is_file() else {},
    }


def _run_row(
    label: str,
    preset: str,
    run: dict[str, Any],
    category: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = run["summary"]
    preset_payload = run.get("preset", {})
    gemma = preset_payload.get("gemma", {}) if isinstance(preset_payload, dict) else {}
    row = {
        "label": label,
        "preset": preset,
        "category": category,
        "run_name": run["run_name"],
        "run_dir_rel": run["run_dir_rel"],
        "elapsed_sec": float(summary.get("elapsed_sec") or np.nan),
        "translate_median_sec": float(summary.get("translate_median_sec") or np.nan),
        "ocr_median_sec": float(summary.get("ocr_median_sec") or np.nan),
        "inpaint_median_sec": float(summary.get("inpaint_median_sec") or np.nan),
        "page_failed_count": int(summary.get("page_failed_count") or 0),
        "gemma_json_retry_count": int(summary.get("gemma_json_retry_count") or 0),
        "gemma_chunk_retry_events": int(summary.get("gemma_chunk_retry_events") or 0),
        "gemma_truncated_count": int(summary.get("gemma_truncated_count") or 0),
        "gemma_empty_content_count": int(summary.get("gemma_empty_content_count") or 0),
        "gemma_missing_key_count": int(summary.get("gemma_missing_key_count") or 0),
        "gemma_reasoning_without_final_count": int(
            summary.get("gemma_reasoning_without_final_count") or 0
        ),
        "gemma_schema_validation_fail_count": int(
            summary.get("gemma_schema_validation_fail_count") or 0
        ),
        "ocr_empty_rate": float(summary.get("ocr_empty_rate") or 0.0),
        "ocr_low_quality_rate": float(summary.get("ocr_low_quality_rate") or 0.0),
        "gpu_peak_used_mb": float(summary.get("gpu_peak_used_mb") or np.nan),
        "gpu_floor_free_mb": float(summary.get("gpu_floor_free_mb") or np.nan),
        "audit_passed": bool(run["audit"].get("passed", False)) if run["audit"] else None,
        "image": str(gemma.get("image", "") or ""),
        "response_format_mode": str(gemma.get("response_format_mode", "") or ""),
        "chunk_size": gemma.get("chunk_size"),
        "temperature": gemma.get("temperature"),
        "n_gpu_layers": gemma.get("n_gpu_layers"),
    }
    if extra:
        row.update(extra)
    return row


def _write_plot(path: Path, fig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _bar_chart(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(df[x], df[y], color="#2c7fb8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    for idx, value in enumerate(df[y].tolist()):
        if pd.notna(value):
            ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    _write_plot(path, fig)


def _line_chart(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(df[x], df[y], marker="o", color="#d95f02")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    for x_val, y_val in zip(df[x].tolist(), df[y].tolist()):
        if pd.notna(y_val):
            ax.annotate(
                f"{y_val:.3f}",
                (x_val, y_val),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
            )
    _write_plot(path, fig)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df[columns].iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                if np.isnan(value):
                    values.append("")
                else:
                    values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _winner_summary(finalists: pd.DataFrame) -> tuple[str, str]:
    ordered = finalists.sort_values(
        by=[
            "page_failed_count",
            "gemma_truncated_count",
            "gemma_missing_key_count",
            "gemma_empty_content_count",
            "gemma_json_retry_count",
            "elapsed_sec",
            "translate_median_sec",
        ],
        ascending=[True, True, True, True, True, True, True],
    ).reset_index(drop=True)
    winner = ordered.iloc[0]
    return str(winner["preset"]), (
        f"{winner['preset']}가 batch elapsed `{winner['elapsed_sec']:.3f}s`, "
        f"translate median `{winner['translate_median_sec']:.3f}s`, "
        f"retry `{int(winner['gemma_json_retry_count'])}`, "
        f"missing key `{int(winner['gemma_missing_key_count'])}`, "
        f"truncated `{int(winner['gemma_truncated_count'])}`로 가장 균형이 좋았습니다."
    )


def _generated_metadata(manifest_path: Path, manifest: dict[str, Any], results_root: Path) -> dict[str, Any]:
    report_cfg = manifest.get("report", {})
    benchmark_cfg = manifest.get("benchmark", {})
    generated_at = datetime.now().astimezone()
    return {
        "manifest_path": manifest_path,
        "report_path": ROOT / str(report_cfg.get("markdown_output", "docs/banchmark_report/report-ko.md")),
        "assets_dir": ROOT / str(report_cfg.get("assets_dir", "docs/assets/benchmarking/latest")),
        "generated_at": generated_at,
        "generated_at_display": generated_at.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "benchmark_name": str(benchmark_cfg.get("name", "Translation Benchmark")),
        "benchmark_kind": str(benchmark_cfg.get("kind", "benchmark")),
        "benchmark_scope": str(benchmark_cfg.get("scope", "")),
        "build_id": str(benchmark_cfg.get("build_id", "") or ""),
        "active_image": str(benchmark_cfg.get("active_image", "") or ""),
        "results_root": results_root,
    }


def _legacy_report(manifest_path: Path, manifest: dict[str, Any], results_root: Path) -> int:
    meta = _generated_metadata(manifest_path, manifest, results_root)
    report_path = meta["report_path"]
    assets_dir = meta["assets_dir"]

    baseline_one = _load_run(results_root, manifest["baseline"]["one_page"])
    baseline_batch = _load_run(results_root, manifest["baseline"]["batch"])

    ngl_rows: list[dict[str, Any]] = []
    for item in manifest.get("n_gpu_layers_sweep", []):
        run = _load_run(results_root, item["one_page"])
        ngl_rows.append(
            _run_row(
                f"n_gpu_layers={item['n_gpu_layers']}",
                item["preset"],
                run,
                "ngl_one_page",
                {"n_gpu_layers": int(item["n_gpu_layers"])},
            )
        )

    temp_rows: list[dict[str, Any]] = []
    for item in manifest.get("temperature_sweep", []):
        run = _load_run(results_root, item["one_page"])
        temp_rows.append(
            _run_row(
                f"temperature={item['temperature']}",
                item["preset"],
                run,
                "temperature_one_page",
                {"temperature": float(item["temperature"])},
            )
        )

    finalist_rows: list[dict[str, Any]] = []
    finalist_lookup: dict[str, dict[str, Any]] = {"translation-baseline": baseline_batch}
    finalist_rows.append(
        _run_row("baseline batch", "translation-baseline", baseline_batch, "batch_finalist", {"source": "baseline"})
    )
    for item in manifest.get("n_gpu_layers_sweep", []):
        if item.get("batch"):
            finalist_lookup[item["preset"]] = _load_run(results_root, item["batch"])
    for item in manifest.get("temperature_sweep", []):
        if item.get("batch"):
            finalist_lookup[item["preset"]] = _load_run(results_root, item["batch"])
    for preset_name in manifest.get("batch_finalists", []):
        if preset_name == "translation-baseline":
            continue
        run = finalist_lookup[preset_name]
        finalist_rows.append(
            _run_row(f"{preset_name} batch", preset_name, run, "batch_finalist", {"source": "candidate"})
        )

    baseline_preset, _ = load_preset("translation-baseline")
    ngl_df = pd.DataFrame(ngl_rows).sort_values("n_gpu_layers").reset_index(drop=True)
    temp_df = pd.DataFrame(temp_rows).sort_values("temperature").reset_index(drop=True)
    finalists_df = pd.DataFrame(finalist_rows).sort_values("elapsed_sec").reset_index(drop=True)

    batch_chart_path = assets_dir / "batch_elapsed_comparison.png"
    ngl_chart_path = assets_dir / "n_gpu_layers_translate_median.png"
    temp_chart_path = assets_dir / "temperature_translate_median.png"
    quality_chart_path = assets_dir / "quality_metrics_comparison.png"

    _bar_chart(finalists_df, "preset", "elapsed_sec", "Representative Batch Elapsed Comparison", "elapsed_sec", batch_chart_path)
    _line_chart(ngl_df, "n_gpu_layers", "translate_median_sec", "n_gpu_layers vs translate median", "translate_median_sec", ngl_chart_path)
    _line_chart(temp_df, "temperature", "translate_median_sec", "temperature vs translate median", "translate_median_sec", temp_chart_path)

    quality_df = finalists_df[["preset", "gemma_json_retry_count", "gemma_truncated_count", "ocr_low_quality_rate"]].copy()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(quality_df))
    width = 0.25
    ax.bar(x - width, quality_df["gemma_json_retry_count"], width=width, label="json_retry")
    ax.bar(x, quality_df["gemma_truncated_count"], width=width, label="truncated")
    ax.bar(x + width, quality_df["ocr_low_quality_rate"], width=width, label="ocr_low_quality_rate")
    ax.set_xticks(x)
    ax.set_xticklabels(quality_df["preset"], rotation=20)
    ax.set_title("Quality Metrics Comparison")
    ax.legend()
    _write_plot(quality_chart_path, fig)

    winner_preset, winner_sentence = _winner_summary(finalists_df)
    winner_run = finalist_lookup.get(str(manifest.get("winning_candidate_preset")), baseline_batch)

    summary_payload = {
        "manifest": repo_relative_str(manifest_path),
        "results_root": repo_relative_str(results_root),
        "generated_at": meta["generated_at"].isoformat(),
        "generated_at_display": meta["generated_at_display"],
        "benchmark_name": meta["benchmark_name"],
        "benchmark_kind": meta["benchmark_kind"],
        "benchmark_scope": meta["benchmark_scope"],
        "active_preset": manifest.get("active_preset"),
        "winning_candidate_preset": manifest.get("winning_candidate_preset"),
        "resolved_winner_preset": winner_preset,
        "charts": {
            "batch_elapsed": repo_relative_str(batch_chart_path),
            "n_gpu_layers_translate_median": repo_relative_str(ngl_chart_path),
            "temperature_translate_median": repo_relative_str(temp_chart_path),
            "quality_metrics": repo_relative_str(quality_chart_path),
        },
    }
    write_json(assets_dir / "report_summary.json", summary_payload)

    report_lines = [
        f"# 자동번역 벤치마크 보고서 - {meta['benchmark_name']}",
        "",
        "이 문서는 `./banchmark_result_log`에 있는 실제 run 결과를 기준으로 자동 생성됩니다.",
        "",
        "## 보고서 메타데이터",
        "",
        f"- 생성 시각: `{meta['generated_at_display']}`",
        f"- 벤치마킹 이름: `{meta['benchmark_name']}`",
        f"- 벤치마킹 종류: `{meta['benchmark_kind']}`",
        f"- 벤치마킹 범위: `{meta['benchmark_scope']}`",
        "",
        "## 현재 기준 설정",
        "",
        f"- active preset: `{manifest.get('active_preset')}`",
        f"- 현재 preset 파일: `{repo_relative_str(ROOT / 'benchmarks' / 'presets' / 'translation-baseline.json')}`",
        f"- results root: `{repo_relative_str(results_root)}`",
        f"- Gemma sampler: `{baseline_preset['gemma']['temperature']} / {baseline_preset['gemma']['top_k']} / {baseline_preset['gemma']['top_p']} / {baseline_preset['gemma']['min_p']}`",
        f"- Gemma runtime: `n_gpu_layers={baseline_preset['gemma']['n_gpu_layers']}`, `threads={baseline_preset['gemma']['threads']}`, `ctx={baseline_preset['gemma']['context_size']}`",
        f"- OCR runtime: `front_device={baseline_preset['ocr_runtime']['front_device']}`, `parallel_workers={baseline_preset['ocr_client']['parallel_workers']}`, `max_new_tokens={baseline_preset['ocr_client']['max_new_tokens']}`",
        "",
        "## 판단 요약",
        "",
        f"- {winner_sentence}",
        f"- winning candidate run: `{repo_relative_str(winner_run['run_dir'])}`",
        f"- baseline one-page run: `{baseline_one['run_dir_rel']}`",
        f"- baseline batch run: `{baseline_batch['run_dir_rel']}`",
        "",
        "## Representative Batch 비교",
        "",
        _markdown_table(
            finalists_df[[
                "preset",
                "elapsed_sec",
                "translate_median_sec",
                "ocr_median_sec",
                "inpaint_median_sec",
                "gemma_json_retry_count",
                "gemma_truncated_count",
                "ocr_low_quality_rate",
                "run_dir_rel",
            ]],
            [
                "preset",
                "elapsed_sec",
                "translate_median_sec",
                "ocr_median_sec",
                "inpaint_median_sec",
                "gemma_json_retry_count",
                "gemma_truncated_count",
                "ocr_low_quality_rate",
                "run_dir_rel",
            ],
        ),
        "",
        f"![Batch Elapsed](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/assets/benchmarking/latest/{batch_chart_path.name})",
        "",
        "## `n_gpu_layers` Sweep",
        "",
        _markdown_table(
            ngl_df[["preset", "n_gpu_layers", "elapsed_sec", "translate_median_sec", "ocr_median_sec", "run_dir_rel"]],
            ["preset", "n_gpu_layers", "elapsed_sec", "translate_median_sec", "ocr_median_sec", "run_dir_rel"],
        ),
        "",
        f"![n_gpu_layers](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/assets/benchmarking/latest/{ngl_chart_path.name})",
        "",
        "## `temperature` Sweep",
        "",
        _markdown_table(
            temp_df[["preset", "temperature", "elapsed_sec", "translate_median_sec", "ocr_median_sec", "run_dir_rel"]],
            ["preset", "temperature", "elapsed_sec", "translate_median_sec", "ocr_median_sec", "run_dir_rel"],
        ),
        "",
        f"![temperature](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/assets/benchmarking/latest/{temp_chart_path.name})",
        "",
        "## 품질 지표 비교",
        "",
        _markdown_table(
            finalists_df[["preset", "gemma_json_retry_count", "gemma_truncated_count", "ocr_empty_rate", "ocr_low_quality_rate", "audit_passed"]],
            ["preset", "gemma_json_retry_count", "gemma_truncated_count", "ocr_empty_rate", "ocr_low_quality_rate", "audit_passed"],
        ),
        "",
        f"![quality](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/assets/benchmarking/latest/{quality_chart_path.name})",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[report] generated markdown: {report_path}")
    print(f"[report] generated assets dir: {assets_dir}")
    return 0


def _b8665_report(manifest_path: Path, manifest: dict[str, Any], results_root: Path) -> int:
    meta = _generated_metadata(manifest_path, manifest, results_root)
    report_path = meta["report_path"]
    assets_dir = meta["assets_dir"]

    controls_config = manifest.get("controls", [])
    if not controls_config:
        raise ValueError("b8665 manifest requires controls.")

    control_rows: list[dict[str, Any]] = []
    control_runs: dict[str, dict[str, Any]] = {}
    for item in controls_config:
        batch_run = _load_run(results_root, item["batch"])
        control_runs[item["preset"]] = batch_run
        control_rows.append(
            _run_row(
                item.get("label", item["preset"]),
                item["preset"],
                batch_run,
                "control_batch",
                {"control_label": item.get("label", item["preset"])},
            )
        )
    control_df = pd.DataFrame(control_rows).sort_values("elapsed_sec").reset_index(drop=True)

    baseline_run = control_runs.get("translation-old-image-baseline") or next(iter(control_runs.values()))
    baseline_row = control_df[control_df["preset"] == "translation-old-image-baseline"]
    baseline_elapsed = float(baseline_row.iloc[0]["elapsed_sec"]) if not baseline_row.empty else np.nan

    def sweep_rows(section_name: str, value_key: str) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
        rows: list[dict[str, Any]] = []
        batch_runs: dict[str, dict[str, Any]] = {}
        for item in manifest.get(section_name, []):
            one_page = item.get("one_page")
            if one_page:
                run = _load_run(results_root, one_page)
                rows.append(
                    _run_row(
                        f"{value_key}={item.get(value_key)}",
                        item["preset"],
                        run,
                        f"{section_name}_one_page",
                        {value_key: item.get(value_key)},
                    )
                )
            if item.get("batch"):
                batch_runs[item["preset"]] = _load_run(results_root, item["batch"])
        df = pd.DataFrame(rows)
        if not df.empty and value_key in df.columns:
            df = df.sort_values(value_key).reset_index(drop=True)
        return df, batch_runs

    chunk_df, chunk_batch_runs = sweep_rows("chunk_sweep", "chunk_size")
    temp_df, temp_batch_runs = sweep_rows("temperature_sweep", "temperature")
    ngl_df, ngl_batch_runs = sweep_rows("n_gpu_layers_sweep", "n_gpu_layers")

    finalists_lookup: dict[str, dict[str, Any]] = dict(control_runs)
    finalists_lookup.update(chunk_batch_runs)
    finalists_lookup.update(temp_batch_runs)
    finalists_lookup.update(ngl_batch_runs)
    finalist_rows = [
        _run_row(
            "baseline batch" if preset == "translation-old-image-baseline" else preset,
            preset,
            run,
            "batch_finalist",
        )
        for preset, run in finalists_lookup.items()
    ]
    finalists_df = pd.DataFrame(finalist_rows).sort_values("elapsed_sec").reset_index(drop=True)
    winner_preset, winner_sentence = _winner_summary(finalists_df)
    winner_run = finalists_lookup.get(str(manifest.get("winning_candidate_preset")), finalists_lookup[winner_preset])

    control_chart_path = assets_dir / "b8665_control_elapsed_comparison.png"
    chunk_chart_path = assets_dir / "b8665_chunk_translate_median.png"
    temp_chart_path = assets_dir / "b8665_temperature_translate_median.png"
    ngl_chart_path = assets_dir / "b8665_n_gpu_layers_translate_median.png"
    quality_chart_path = assets_dir / "b8665_quality_metrics_comparison.png"

    _bar_chart(
        control_df,
        "control_label",
        "elapsed_sec",
        "Old Image vs b8665 Control Batch Comparison",
        "elapsed_sec",
        control_chart_path,
    )
    if not chunk_df.empty:
        _line_chart(
            chunk_df,
            "chunk_size",
            "translate_median_sec",
            "chunk_size vs translate median",
            "translate_median_sec",
            chunk_chart_path,
        )
    if not temp_df.empty:
        _line_chart(
            temp_df,
            "temperature",
            "translate_median_sec",
            "temperature vs translate median",
            "translate_median_sec",
            temp_chart_path,
        )
    if not ngl_df.empty:
        _line_chart(
            ngl_df,
            "n_gpu_layers",
            "translate_median_sec",
            "n_gpu_layers vs translate median",
            "translate_median_sec",
            ngl_chart_path,
        )

    quality_df = finalists_df[
        [
            "preset",
            "gemma_json_retry_count",
            "gemma_missing_key_count",
            "gemma_truncated_count",
            "ocr_low_quality_rate",
        ]
    ].copy()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(quality_df))
    width = 0.2
    ax.bar(x - 1.5 * width, quality_df["gemma_json_retry_count"], width=width, label="json_retry")
    ax.bar(x - 0.5 * width, quality_df["gemma_missing_key_count"], width=width, label="missing_key")
    ax.bar(x + 0.5 * width, quality_df["gemma_truncated_count"], width=width, label="truncated")
    ax.bar(x + 1.5 * width, quality_df["ocr_low_quality_rate"], width=width, label="ocr_low_quality_rate")
    ax.set_xticks(x)
    ax.set_xticklabels(quality_df["preset"], rotation=20)
    ax.set_title("Quality Metrics Comparison")
    ax.legend()
    _write_plot(quality_chart_path, fig)

    summary_payload = {
        "manifest": repo_relative_str(manifest_path),
        "results_root": repo_relative_str(results_root),
        "generated_at": meta["generated_at"].isoformat(),
        "generated_at_display": meta["generated_at_display"],
        "benchmark_name": meta["benchmark_name"],
        "benchmark_kind": meta["benchmark_kind"],
        "benchmark_scope": meta["benchmark_scope"],
        "build_id": meta["build_id"],
        "active_image": meta["active_image"],
        "winning_candidate_preset": manifest.get("winning_candidate_preset"),
        "resolved_winner_preset": winner_preset,
        "charts": {
            "control_elapsed": repo_relative_str(control_chart_path),
            "chunk_translate_median": repo_relative_str(chunk_chart_path) if chunk_chart_path.is_file() else "",
            "temperature_translate_median": repo_relative_str(temp_chart_path) if temp_chart_path.is_file() else "",
            "n_gpu_layers_translate_median": repo_relative_str(ngl_chart_path) if ngl_chart_path.is_file() else "",
            "quality_metrics": repo_relative_str(quality_chart_path),
        },
    }
    write_json(assets_dir / "report_summary.json", summary_payload)

    report_lines = [
        f"# 자동번역 벤치마크 보고서 - {meta['benchmark_name']}",
        "",
        "이 문서는 `./banchmark_result_log`에 있는 실제 run 결과를 기준으로 자동 생성됩니다.",
        "",
        "## 보고서 메타데이터",
        "",
        f"- 생성 시각: `{meta['generated_at_display']}`",
        f"- 벤치마킹 이름: `{meta['benchmark_name']}`",
        f"- 벤치마킹 종류: `{meta['benchmark_kind']}`",
        f"- 벤치마킹 범위: `{meta['benchmark_scope']}`",
        f"- build id: `{meta['build_id']}`",
        f"- active image: `{meta['active_image']}`",
        "",
        "## 판단 요약",
        "",
        f"- {winner_sentence}",
        f"- old-image baseline batch run: `{baseline_run['run_dir_rel']}`",
        f"- winning candidate run: `{repo_relative_str(winner_run['run_dir'])}`",
    ]
    if pd.notna(baseline_elapsed):
        winner_elapsed = float(finalists_df[finalists_df["preset"] == winner_preset].iloc[0]["elapsed_sec"])
        delta = baseline_elapsed - winner_elapsed
        report_lines.append(f"- old-image baseline 대비 elapsed delta: `{delta:.3f}s`")

    report_lines.extend(
        [
            "",
            "## Old Image vs b8665 Control 비교",
            "",
            _markdown_table(
                control_df[
                    [
                        "control_label",
                        "preset",
                        "response_format_mode",
                        "image",
                        "elapsed_sec",
                        "translate_median_sec",
                        "gemma_json_retry_count",
                        "gemma_missing_key_count",
                        "gemma_truncated_count",
                        "run_dir_rel",
                    ]
                ],
                [
                    "control_label",
                    "preset",
                    "response_format_mode",
                    "image",
                    "elapsed_sec",
                    "translate_median_sec",
                    "gemma_json_retry_count",
                    "gemma_missing_key_count",
                    "gemma_truncated_count",
                    "run_dir_rel",
                ],
            ),
            "",
            f"![control](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/assets/benchmarking/latest/{control_chart_path.name})",
            "",
            "## Representative Batch Finalists",
            "",
            _markdown_table(
                finalists_df[
                    [
                        "preset",
                        "response_format_mode",
                        "chunk_size",
                        "temperature",
                        "n_gpu_layers",
                        "elapsed_sec",
                        "translate_median_sec",
                        "gemma_json_retry_count",
                        "gemma_missing_key_count",
                        "gemma_truncated_count",
                        "audit_passed",
                        "run_dir_rel",
                    ]
                ],
                [
                    "preset",
                    "response_format_mode",
                    "chunk_size",
                    "temperature",
                    "n_gpu_layers",
                    "elapsed_sec",
                    "translate_median_sec",
                    "gemma_json_retry_count",
                    "gemma_missing_key_count",
                    "gemma_truncated_count",
                    "audit_passed",
                    "run_dir_rel",
                ],
            ),
            "",
            "## Chunk Sweep",
            "",
        ]
    )
    if not chunk_df.empty:
        report_lines.extend(
            [
                _markdown_table(
                    chunk_df[
                        [
                            "preset",
                            "chunk_size",
                            "elapsed_sec",
                            "translate_median_sec",
                            "gemma_json_retry_count",
                            "gemma_missing_key_count",
                            "run_dir_rel",
                        ]
                    ],
                    [
                        "preset",
                        "chunk_size",
                        "elapsed_sec",
                        "translate_median_sec",
                        "gemma_json_retry_count",
                        "gemma_missing_key_count",
                        "run_dir_rel",
                    ],
                ),
                "",
                f"![chunk](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/assets/benchmarking/latest/{chunk_chart_path.name})",
                "",
            ]
        )
    else:
        report_lines.extend(["- chunk sweep 결과가 없습니다.", ""])

    report_lines.extend(["## Temperature Sweep", ""])
    if not temp_df.empty:
        report_lines.extend(
            [
                _markdown_table(
                    temp_df[
                        [
                            "preset",
                            "temperature",
                            "elapsed_sec",
                            "translate_median_sec",
                            "gemma_json_retry_count",
                            "gemma_missing_key_count",
                            "run_dir_rel",
                        ]
                    ],
                    [
                        "preset",
                        "temperature",
                        "elapsed_sec",
                        "translate_median_sec",
                        "gemma_json_retry_count",
                        "gemma_missing_key_count",
                        "run_dir_rel",
                    ],
                ),
                "",
                f"![temperature](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/assets/benchmarking/latest/{temp_chart_path.name})",
                "",
            ]
        )
    else:
        report_lines.extend(["- temperature sweep 결과가 없습니다.", ""])

    report_lines.extend(["## n_gpu_layers Sweep", ""])
    if not ngl_df.empty:
        report_lines.extend(
            [
                _markdown_table(
                    ngl_df[
                        [
                            "preset",
                            "n_gpu_layers",
                            "elapsed_sec",
                            "translate_median_sec",
                            "gemma_json_retry_count",
                            "gemma_missing_key_count",
                            "run_dir_rel",
                        ]
                    ],
                    [
                        "preset",
                        "n_gpu_layers",
                        "elapsed_sec",
                        "translate_median_sec",
                        "gemma_json_retry_count",
                        "gemma_missing_key_count",
                        "run_dir_rel",
                    ],
                ),
                "",
                f"![ngl](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/assets/benchmarking/latest/{ngl_chart_path.name})",
                "",
            ]
        )
    else:
        report_lines.extend(["- n_gpu_layers sweep 결과가 없습니다.", ""])

    report_lines.extend(
        [
            "## 품질 지표 비교",
            "",
            _markdown_table(
                finalists_df[
                    [
                        "preset",
                        "gemma_json_retry_count",
                        "gemma_missing_key_count",
                        "gemma_truncated_count",
                        "gemma_empty_content_count",
                        "gemma_reasoning_without_final_count",
                        "ocr_empty_rate",
                        "ocr_low_quality_rate",
                        "audit_passed",
                    ]
                ],
                [
                    "preset",
                    "gemma_json_retry_count",
                    "gemma_missing_key_count",
                    "gemma_truncated_count",
                    "gemma_empty_content_count",
                    "gemma_reasoning_without_final_count",
                    "ocr_empty_rate",
                    "ocr_low_quality_rate",
                    "audit_passed",
                ],
            ),
            "",
            f"![quality](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/assets/benchmarking/latest/{quality_chart_path.name})",
            "",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[report] generated markdown: {report_path}")
    print(f"[report] generated assets dir: {assets_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a benchmark markdown report with charts.")
    parser.add_argument("--manifest", default=str(ROOT / "benchmarks" / "report_manifest.yaml"))
    parser.add_argument("--results-root", default="")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = _load_yaml(manifest_path)
    results_root = Path(args.results_root) if args.results_root else (ROOT / str(manifest.get("results_root", "banchmark_result_log")))

    if manifest.get("controls"):
        return _b8665_report(manifest_path, manifest, results_root)
    return _legacy_report(manifest_path, manifest, results_root)


if __name__ == "__main__":
    raise SystemExit(main())
