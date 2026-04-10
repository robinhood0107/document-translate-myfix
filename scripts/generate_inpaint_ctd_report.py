#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

REPORT_PATH = ROOT / "docs" / "banchmark_report" / "inpaint-ctd-report-ko.md"
ASSET_ROOT = ROOT / "docs" / "assets" / "benchmarking" / "inpaint-ctd"
LATEST_ASSETS = ASSET_ROOT / "latest"
HISTORY_ASSETS = ASSET_ROOT / "history"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _archive_latest() -> dict[str, str] | None:
    if not REPORT_PATH.exists() and not LATEST_ASSETS.exists():
        return None
    snapshot_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_inpaint-ctd"
    report_hist_dir = REPORT_PATH.parent / "history" / snapshot_id
    asset_hist_dir = HISTORY_ASSETS / snapshot_id
    report_hist_dir.mkdir(parents=True, exist_ok=True)
    if REPORT_PATH.exists():
        shutil.copy2(REPORT_PATH, report_hist_dir / REPORT_PATH.name)
    if LATEST_ASSETS.exists():
        shutil.copytree(LATEST_ASSETS, asset_hist_dir)
    return {
        "snapshot_id": snapshot_id,
        "report_path": repo_relative_str(report_hist_dir / REPORT_PATH.name),
        "assets_dir": repo_relative_str(asset_hist_dir),
    }


def _copy_file(src: str, dst: Path) -> str:
    if not src:
        return ""
    source_path = ROOT / src.replace("./", "", 1) if src.startswith("./") else Path(src)
    if not source_path.is_file():
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, dst)
    return repo_relative_str(dst)


def _metric(summary: dict[str, Any], key: str, stage: str | None = None) -> str:
    if stage is None:
        value = summary.get(key, "")
    else:
        value = ((summary.get("stage_stats") or {}).get(stage) or {}).get(key, "")
    return str(value)


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def _spotlight_rows(corpus_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for run in corpus_payload.get("spotlight_runs", []):
        summary = run.get("summary", {}) if isinstance(run.get("summary"), dict) else {}
        rows.append({
            "case": run.get("case_label", ""),
            "elapsed_sec": summary.get("elapsed_sec", ""),
            "ocr_median_sec": _metric(summary, "median_sec", "ocr"),
            "translate_median_sec": _metric(summary, "median_sec", "translate"),
            "inpaint_median_sec": _metric(summary, "median_sec", "inpaint"),
            "gpu_peak_used_mb": summary.get("gpu_peak_used_mb", ""),
            "gpu_floor_free_mb": summary.get("gpu_floor_free_mb", ""),
            "cleanup_applied_count": run.get("cleanup_applied_count", 0),
        })
    return rows


def _full_rows(corpus_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for run in corpus_payload.get("full_runs", []):
        summary = run.get("summary", {}) if isinstance(run.get("summary"), dict) else {}
        rows.append({
            "case": run.get("case_label", ""),
            "elapsed_sec": summary.get("elapsed_sec", ""),
            "detect_median_sec": _metric(summary, "median_sec", "detect"),
            "ocr_median_sec": _metric(summary, "median_sec", "ocr"),
            "translate_median_sec": _metric(summary, "median_sec", "translate"),
            "inpaint_median_sec": _metric(summary, "median_sec", "inpaint"),
            "gpu_peak_used_mb": summary.get("gpu_peak_used_mb", ""),
            "gpu_floor_free_mb": summary.get("gpu_floor_free_mb", ""),
            "cleanup_applied_count": run.get("cleanup_applied_count", 0),
            "page_failed_count": summary.get("page_failed_count", ""),
        })
    return rows


def _runtime_lines(corpus_payload: dict[str, Any]) -> list[str]:
    runtime = corpus_payload.get("llama_cpp_runtime", {}) if isinstance(corpus_payload.get("llama_cpp_runtime"), dict) else {}
    lines: list[str] = []
    for key in ("gemma", "hunyuanocr"):
        item = runtime.get(key, {}) if isinstance(runtime.get(key), dict) else {}
        if not item:
            continue
        lines.append(f"- {key} image: `{item.get('llama_cpp_image', '')}`")
        lines.append(f"- {key} digest: `{item.get('llama_cpp_digest', '')}`")
        lines.append(f"- {key} version: `{item.get('llama_cpp_version', '')}`")
    if lines:
        lines.append("")
    return lines


def _copy_spotlight_assets(manifest: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for corpus_name, corpus_payload in (manifest.get("corpora") or {}).items():
        corpus_copy = {"spotlight_runs": []}
        for run in corpus_payload.get("spotlight_runs", []):
            case_slug = str(run.get("case_slug", ""))
            asset_dir = LATEST_ASSETS / "spotlight" / corpus_name / case_slug
            artifacts = run.get("artifacts", {}) if isinstance(run.get("artifacts"), dict) else {}
            copied_artifacts = {
                key: _copy_file(value, asset_dir / Path(value).name) if value else ""
                for key, value in artifacts.items()
            }
            corpus_copy["spotlight_runs"].append(
                {
                    "case_slug": case_slug,
                    "case_label": run.get("case_label", ""),
                    "artifacts": copied_artifacts,
                }
            )
        copied[corpus_name] = corpus_copy
    return copied


def _render(manifest: dict[str, Any], copied_assets: dict[str, Any], archive_info: dict[str, str] | None) -> str:
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    fixed = manifest.get("fixed_runtime", {}) if isinstance(manifest.get("fixed_runtime"), dict) else {}
    lines = [
        "# Inpaint CTD Benchmark Report",
        "",
        "## Metadata",
        "",
        f"- generated_at: `{generated_at}`",
        f"- git_sha: `{manifest.get('git_sha', '')}`",
        f"- suite_dir: `{manifest.get('suite_dir', '')}`",
        f"- results_root: `{manifest.get('results_root', '')}`",
        f"- execution_scope: `{manifest.get('execution_scope', '')}`",
        f"- speed_score_scope: `{manifest.get('speed_score_scope', '')}`",
        f"- quality_gate_scope: `{manifest.get('quality_gate_scope', '')}`",
        "",
        "## Fixed Runtime",
        "",
        f"- detector: `{fixed.get('detector', '')}`",
        f"- China OCR: `{fixed.get('china', '')}`",
        f"- japan OCR: `{fixed.get('japan', '')}`",
        f"- default recommendation: `{fixed.get('default_recommendation', '')}`",
        f"- quality mode: `{fixed.get('quality_mode', '')}`",
        f"- offline review mode: `{fixed.get('offline_review_mode', '')}`",
        "- note: CUDA13 benchmark family pins RT-DETR-v2 to CPU because the current ONNX CUDA path regresses with CuDNN internal errors in this environment.",
        "",
    ]
    if archive_info:
        lines.extend(
            [
                "## Previous Latest Archive",
                "",
                f"- snapshot_id: `{archive_info['snapshot_id']}`",
                f"- report: `{archive_info['report_path']}`",
                f"- assets: `{archive_info['assets_dir']}`",
                "",
            ]
        )
    for corpus_name, corpus_payload in (manifest.get("corpora") or {}).items():
        lines.extend(
            [
                f"## {corpus_name.capitalize()} Spotlight 5-way",
                "",
                _table(
                    _spotlight_rows(corpus_payload),
                    [
                        "case",
                        "elapsed_sec",
                        "ocr_median_sec",
                        "translate_median_sec",
                        "inpaint_median_sec",
                        "gpu_peak_used_mb",
                        "gpu_floor_free_mb",
                        "cleanup_applied_count",
                    ],
                ),
                "",
                f"- OCR invariance (spotlight): `{((corpus_payload.get('spotlight_ocr_invariance') or {}).get('status', ''))}`",
                "",
                *(_runtime_lines(corpus_payload)),
                f"## {corpus_name.capitalize()} Full Summary",
                "",
                _table(
                    _full_rows(corpus_payload),
                    [
                        "case",
                        "elapsed_sec",
                        "detect_median_sec",
                        "ocr_median_sec",
                        "translate_median_sec",
                        "inpaint_median_sec",
                        "gpu_peak_used_mb",
                        "gpu_floor_free_mb",
                        "cleanup_applied_count",
                        "page_failed_count",
                    ],
                ),
                "",
                f"- OCR invariance (full): `{((corpus_payload.get('full_ocr_invariance') or {}).get('status', ''))}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Operations Recommendation",
            "",
            f"- 기본값: `{fixed.get('default_recommendation', '')}`",
            f"- 수동 품질 모드: `{fixed.get('quality_mode', '')}`",
            f"- 오프라인 검수 전용: `{fixed.get('offline_review_mode', '')}`",
            "- VRAM headroom은 측정 peak보다 10~15% 이상 남기는 것을 권장합니다.",
            "",
            "## Visual Appendix",
            "",
        ]
    )
    for corpus_name, corpus_payload in copied_assets.items():
        lines.append(f"### {corpus_name.capitalize()}")
        lines.append("")
        for run in corpus_payload.get("spotlight_runs", []):
            lines.append(f"- `{run.get('case_label', '')}`")
            for key in (
                "source",
                "detector_overlay",
                "raw_mask",
                "mask_overlay",
                "cleanup_delta",
                "cleaned",
                "translated",
                "debug_metadata",
            ):
                value = (run.get("artifacts") or {}).get(key, "")
                if value:
                    lines.append(f"  {key}: `{value}`")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the inpaint CTD benchmark report.")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = _load_yaml(manifest_path)
    archive_info = _archive_latest()
    if LATEST_ASSETS.exists():
        shutil.rmtree(LATEST_ASSETS)
    LATEST_ASSETS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, LATEST_ASSETS / manifest_path.name)
    copied_assets = _copy_spotlight_assets(manifest)
    write_json(LATEST_ASSETS / "copied_assets.json", copied_assets)
    report_text = _render(manifest, copied_assets, archive_info)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
