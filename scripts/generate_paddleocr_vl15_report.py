#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
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

from benchmark_common import repo_relative_str, write_json


_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _markdown_relative_path(from_path: Path, target_path: Path) -> str:
    return Path(os.path.relpath(target_path, start=from_path.parent)).as_posix()


def _write_plot(path: Path, fig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _bar_chart(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(df[x], df[y], color="#2c7fb8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=24)
    for idx, value in enumerate(df[y].tolist()):
        if pd.notna(value):
            ax.text(idx, float(value), f"{float(value):.3f}", ha="center", va="bottom", fontsize=8)
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


def _generated_metadata(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    report_cfg = manifest.get("report", {})
    benchmark_cfg = manifest.get("benchmark", {})
    generated_at = datetime.now().astimezone()
    results_root_cfg = str(manifest.get("results_root", "banchmark_result_log/paddleocr_vl15"))
    results_root = ROOT / results_root_cfg[2:] if results_root_cfg.startswith("./") else ROOT / results_root_cfg
    return {
        "manifest_path": manifest_path,
        "report_path": ROOT / str(report_cfg.get("markdown_output", "docs/banchmark_report/paddleocr-vl15-report-ko.md")),
        "assets_dir": ROOT / str(report_cfg.get("assets_dir", "docs/assets/benchmarking/paddleocr-vl15/latest")),
        "generated_at": generated_at,
        "generated_at_display": generated_at.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "benchmark_name": str(benchmark_cfg.get("name", "PaddleOCR-VL 1.5 Runtime Benchmark")),
        "benchmark_kind": str(benchmark_cfg.get("kind", "benchmark")),
        "benchmark_scope": str(benchmark_cfg.get("scope", "")),
        "baseline_sha": str(benchmark_cfg.get("baseline_sha", "") or ""),
        "develop_ref_sha": str(benchmark_cfg.get("develop_ref_sha", "") or ""),
        "results_root": results_root,
    }


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "report"


def _rewrite_markdown_image_links(markdown: str, report_path: Path, assets_dir: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        alt_text, target = match.groups()
        if target.startswith(("http://", "https://")):
            return match.group(0)
        candidate = assets_dir / Path(target).name
        if not candidate.is_file():
            return match.group(0)
        return f"![{alt_text}]({_markdown_relative_path(report_path, candidate)})"

    return _MD_IMAGE_RE.sub(replace, markdown)


def _archive_existing_latest(meta: dict[str, Any]) -> dict[str, str] | None:
    report_path: Path = meta["report_path"]
    assets_dir: Path = meta["assets_dir"]

    if "history" in report_path.parts:
        return None
    if not report_path.is_file() and not assets_dir.exists():
        return None

    history_report_root = report_path.parent / "history"
    history_assets_root = assets_dir.parent / "history"
    snapshot_base = f"{meta['generated_at'].strftime('%Y%m%d_%H%M%S')}_{_slugify(meta['benchmark_name'])}"
    snapshot_id = snapshot_base
    suffix = 2
    while (history_report_root / snapshot_id).exists() or (history_assets_root / snapshot_id).exists():
        snapshot_id = f"{snapshot_base}-{suffix}"
        suffix += 1

    history_report_dir = history_report_root / snapshot_id
    history_report_path = history_report_dir / report_path.name
    history_assets_dir = history_assets_root / snapshot_id
    history_report_dir.mkdir(parents=True, exist_ok=True)
    history_assets_root.mkdir(parents=True, exist_ok=True)

    if assets_dir.is_dir():
        shutil.copytree(assets_dir, history_assets_dir)
    if report_path.is_file():
        archived_markdown = _rewrite_markdown_image_links(
            report_path.read_text(encoding="utf-8"),
            history_report_path,
            history_assets_dir,
        )
        history_report_path.write_text(archived_markdown, encoding="utf-8")

    return {
        "snapshot_id": snapshot_id,
        "report_path": repo_relative_str(history_report_path),
        "assets_dir": repo_relative_str(history_assets_dir),
    }


def _candidate_rows(manifest: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = manifest.get("baseline", {})
    baseline_row = {
        "phase": "phase-0-baseline",
        "preset": str(baseline.get("preset", "baseline")),
        "official_score_detect_ocr_median_sec": float(
            baseline.get("official_score_detect_ocr_median_sec") or np.nan
        ),
        "warm_ocr_page_p95_median_sec": float(
            np.median(
                [
                    float((item.get("summary") or {}).get("ocr_page_p95_sec") or np.nan)
                    for item in baseline.get("warm_runs", [])
                    if isinstance(item, dict)
                ]
            )
        )
        if baseline.get("warm_runs")
        else np.nan,
        "detection_pass": bool(baseline.get("detection_pass", False)),
        "ocr_pass": bool(baseline.get("ocr_pass", False)),
        "promoted": True,
        "rejection_reason": str(baseline.get("rejection_reason", "") or ""),
        "source": "baseline",
    }

    rows = [baseline_row]
    phase_best_rows = [baseline_row.copy()]
    for phase in manifest.get("phases", []):
        if not isinstance(phase, dict):
            continue
        phase_name = str(phase.get("phase", "phase"))
        if phase.get("skipped"):
            phase_best_rows.append(
                {
                    "phase": phase_name,
                    "preset": str(phase.get("best_preset_after", "") or ""),
                    "official_score_detect_ocr_median_sec": np.nan,
                    "warm_ocr_page_p95_median_sec": np.nan,
                    "detection_pass": "",
                    "ocr_pass": "",
                    "promoted": False,
                    "rejection_reason": str(phase.get("skip_reason", "") or ""),
                    "source": "skipped",
                }
            )
            continue

        candidates = phase.get("candidates", [])
        best_preset = str(phase.get("best_preset_after", "") or "")
        phase_best: dict[str, Any] | None = None
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            row = {
                "phase": phase_name,
                "preset": str(candidate.get("preset", "") or ""),
                "official_score_detect_ocr_median_sec": float(
                    candidate.get("official_score_detect_ocr_median_sec") or np.nan
                ),
                "warm_ocr_page_p95_median_sec": float(
                    candidate.get("warm_ocr_page_p95_median_sec") or np.nan
                ),
                "detection_pass": bool(candidate.get("detection_pass", False)),
                "ocr_pass": bool(candidate.get("ocr_pass", False)),
                "promoted": bool(candidate.get("promoted", False)),
                "rejection_reason": str(candidate.get("rejection_reason", "") or ""),
                "source": "candidate",
            }
            rows.append(row)
            if row["preset"] == best_preset:
                phase_best = row.copy()
        if phase_best is None:
            phase_best = {
                "phase": phase_name,
                "preset": best_preset,
                "official_score_detect_ocr_median_sec": np.nan,
                "warm_ocr_page_p95_median_sec": np.nan,
                "detection_pass": "",
                "ocr_pass": "",
                "promoted": False,
                "rejection_reason": "",
                "source": "phase-best-missing",
            }
        phase_best_rows.append(phase_best)

    return pd.DataFrame(rows), pd.DataFrame(phase_best_rows)


def _baseline_cold_warm_table(manifest: dict[str, Any]) -> pd.DataFrame:
    baseline = manifest.get("baseline", {})
    rows: list[dict[str, Any]] = []
    cold = baseline.get("cold_run", {})
    if isinstance(cold, dict):
        summary = cold.get("summary", {})
        rows.append(
            {
                "run": "cold",
                "run_dir": str(cold.get("run_dir", "") or ""),
                "detect_total_sec": float(summary.get("detect_total_sec") or np.nan),
                "ocr_total_sec": float(summary.get("ocr_total_sec") or np.nan),
                "detect_ocr_total_sec": float(summary.get("detect_ocr_total_sec") or np.nan),
                "ocr_page_p95_sec": float(summary.get("ocr_page_p95_sec") or np.nan),
                "detection_pass": bool(cold.get("detection_pass", False)),
                "ocr_pass": bool(cold.get("ocr_pass", False)),
            }
        )
    for index, warm in enumerate(baseline.get("warm_runs", []), start=1):
        if not isinstance(warm, dict):
            continue
        summary = warm.get("summary", {})
        rows.append(
            {
                "run": f"warm{index}",
                "run_dir": str(warm.get("run_dir", "") or ""),
                "detect_total_sec": float(summary.get("detect_total_sec") or np.nan),
                "ocr_total_sec": float(summary.get("ocr_total_sec") or np.nan),
                "detect_ocr_total_sec": float(summary.get("detect_ocr_total_sec") or np.nan),
                "ocr_page_p95_sec": float(summary.get("ocr_page_p95_sec") or np.nan),
                "detection_pass": bool(warm.get("detection_pass", False)),
                "ocr_pass": bool(warm.get("ocr_pass", False)),
            }
        )
    return pd.DataFrame(rows)


def generate_report(manifest_path: Path) -> int:
    manifest = _load_yaml(manifest_path)
    meta = _generated_metadata(manifest_path, manifest)
    report_path = meta["report_path"]
    assets_dir = meta["assets_dir"]
    archived_previous = _archive_existing_latest(meta)
    assets_dir.mkdir(parents=True, exist_ok=True)

    candidate_df, phase_best_df = _candidate_rows(manifest)
    baseline_runs_df = _baseline_cold_warm_table(manifest)
    winner = manifest.get("winner", {}) if isinstance(manifest.get("winner"), dict) else {}
    winner_preset = str(winner.get("preset", "") or "")

    candidate_plot_df = candidate_df.copy()
    candidate_plot_df = candidate_plot_df[candidate_plot_df["source"] != "baseline"].reset_index(drop=True)
    if candidate_plot_df.empty:
        candidate_plot_df = candidate_df.copy()

    phase_best_plot_df = phase_best_df.copy()
    phase_best_plot_df["phase_label"] = phase_best_plot_df["phase"] + "\n" + phase_best_plot_df["preset"].astype(str)

    official_chart_path = assets_dir / "paddleocr_vl15_candidate_official_score.png"
    p95_chart_path = assets_dir / "paddleocr_vl15_candidate_p95.png"
    phase_chart_path = assets_dir / "paddleocr_vl15_phase_best_official_score.png"

    if not candidate_plot_df.empty:
        _bar_chart(
            candidate_plot_df.sort_values("official_score_detect_ocr_median_sec"),
            "preset",
            "official_score_detect_ocr_median_sec",
            "Candidate Official Score (warm detect+ocr median)",
            "seconds",
            official_chart_path,
        )
        _bar_chart(
            candidate_plot_df.sort_values("warm_ocr_page_p95_median_sec"),
            "preset",
            "warm_ocr_page_p95_median_sec",
            "Candidate warm OCR p95",
            "seconds",
            p95_chart_path,
        )
    else:
        for path in (official_chart_path, p95_chart_path):
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.text(0.5, 0.5, "No candidate data", ha="center", va="center")
            ax.axis("off")
            _write_plot(path, fig)

    _bar_chart(
        phase_best_plot_df,
        "phase_label",
        "official_score_detect_ocr_median_sec",
        "Phase Best Official Score",
        "seconds",
        phase_chart_path,
    )

    candidate_csv = assets_dir / "candidates.csv"
    phase_best_csv = assets_dir / "phase_best.csv"
    baseline_csv = assets_dir / "baseline_runs.csv"
    candidate_df.to_csv(candidate_csv, index=False, encoding="utf-8")
    phase_best_df.to_csv(phase_best_csv, index=False, encoding="utf-8")
    baseline_runs_df.to_csv(baseline_csv, index=False, encoding="utf-8")

    summary_payload = {
        "manifest": repo_relative_str(manifest_path),
        "results_root": repo_relative_str(meta["results_root"]),
        "generated_at": meta["generated_at"].isoformat(),
        "generated_at_display": meta["generated_at_display"],
        "benchmark_name": meta["benchmark_name"],
        "benchmark_kind": meta["benchmark_kind"],
        "benchmark_scope": meta["benchmark_scope"],
        "winner_preset": winner_preset,
        "develop_promotion_ready": bool(winner.get("develop_promotion_ready", False)),
        "charts": {
            "candidate_official_score": repo_relative_str(official_chart_path),
            "candidate_p95": repo_relative_str(p95_chart_path),
            "phase_best": repo_relative_str(phase_chart_path),
        },
    }
    if archived_previous:
        summary_payload["archived_previous_report"] = archived_previous
    write_json(assets_dir / "report_summary.json", summary_payload)

    phase_order = [str(item.get("phase", "")) for item in manifest.get("phases", []) if isinstance(item, dict)]
    report_lines = [
        f"# 자동 벤치마크 보고서 - {meta['benchmark_name']}",
        "",
        "이 문서는 `PaddleOCR-VL 1.5` actual-pipeline family suite 결과에서 자동 생성됩니다.",
        "",
        "## 보고서 메타데이터",
        "",
        f"- 생성 시각: `{meta['generated_at_display']}`",
        f"- 벤치마킹 이름: `{meta['benchmark_name']}`",
        f"- 벤치마킹 종류: `{meta['benchmark_kind']}`",
        f"- 벤치마킹 범위: `{meta['benchmark_scope']}`",
        f"- baseline SHA: `{meta['baseline_sha']}`",
        f"- develop ref SHA: `{meta['develop_ref_sha']}`",
        f"- results root: `{repo_relative_str(meta['results_root'])}`",
        f"- gold path: `{manifest.get('gold', {}).get('path', '')}`",
        "",
        "## 라운드 결론",
        "",
        f"- 최종 winner: `{winner_preset}`",
        f"- official detect+ocr median: `{winner.get('official_score_detect_ocr_median_sec', '')}`",
        f"- develop 승격 가능: `{winner.get('develop_promotion_ready', False)}`",
        f"- baseline 대비 개선폭: `{winner.get('improvement_vs_baseline_pct', '')}%`",
        "",
        "## Candidate Phase 순서",
        "",
    ]
    for phase_name in phase_order:
        report_lines.append(f"- `{phase_name}`")
    report_lines.extend(
        [
            "",
            "## Baseline cold / warm",
            "",
            _markdown_table(
                baseline_runs_df,
                [
                    "run",
                    "detect_total_sec",
                    "ocr_total_sec",
                    "detect_ocr_total_sec",
                    "ocr_page_p95_sec",
                    "detection_pass",
                    "ocr_pass",
                    "run_dir",
                ],
            )
            if not baseline_runs_df.empty
            else "_baseline data not found_",
            "",
            "## Candidate 결과",
            "",
            _markdown_table(
                candidate_df.sort_values(["phase", "official_score_detect_ocr_median_sec"]),
                [
                    "phase",
                    "preset",
                    "official_score_detect_ocr_median_sec",
                    "warm_ocr_page_p95_median_sec",
                    "detection_pass",
                    "ocr_pass",
                    "promoted",
                    "rejection_reason",
                ],
            )
            if not candidate_df.empty
            else "_candidate data not found_",
            "",
            f"![candidate-official-score]({_markdown_relative_path(report_path, official_chart_path)})",
            "",
            f"![candidate-p95]({_markdown_relative_path(report_path, p95_chart_path)})",
            "",
            "## Phase Best",
            "",
            _markdown_table(
                phase_best_df,
                [
                    "phase",
                    "preset",
                    "official_score_detect_ocr_median_sec",
                    "warm_ocr_page_p95_median_sec",
                    "promoted",
                    "rejection_reason",
                ],
            )
            if not phase_best_df.empty
            else "_phase best data not found_",
            "",
            f"![phase-best]({_markdown_relative_path(report_path, phase_chart_path)})",
            "",
            "## 산출물",
            "",
            f"- candidate CSV: `{repo_relative_str(candidate_csv)}`",
            f"- phase-best CSV: `{repo_relative_str(phase_best_csv)}`",
            f"- baseline CSV: `{repo_relative_str(baseline_csv)}`",
            "",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[paddleocr-vl15-report] generated markdown: {report_path}")
    print(f"[paddleocr-vl15-report] generated assets dir: {assets_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate PaddleOCR-VL 1.5 benchmark report from suite manifest.")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    return generate_report(Path(args.manifest))


if __name__ == "__main__":
    raise SystemExit(main())
