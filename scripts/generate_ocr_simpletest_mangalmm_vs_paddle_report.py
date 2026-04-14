#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

FAMILY_NAME = "ocr_simpletest_mangalmm_vs_paddle"
BENCH_ROOT = ROOT / "banchmark_result_log" / FAMILY_NAME
DOC_ROOT = ROOT / "docs" / "benchmark" / "ocr-simpletest-mangalmm-vs-paddle"
REPORT_PATH = ROOT / "docs" / "banchmark_report" / "ocr-simpletest-mangalmm-vs-paddle-report-ko.md"
ASSETS_ROOT = ROOT / "docs" / "assets" / "benchmarking" / "ocr-simpletest-mangalmm-vs-paddle"
LATEST_ASSETS = ASSETS_ROOT / "latest"
HISTORY_ROOT = ASSETS_ROOT / "history"
RESULTS_HISTORY_PATH = DOC_ROOT / "results-history-ko.md"
GEMMA_BASELINE_REPORT = ROOT / "docs" / "banchmark_report" / "gemma-iq4nl-japan-report-ko.md"
GEMMA_BASELINE = {
    "context_size": 4096,
    "threads": 10,
    "n_gpu_layers": 23,
    "chunk_size": 6,
    "temperature": 0.7,
    "max_completion_tokens": 512,
}


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    text = str(path)
    if text.startswith("./"):
        return ROOT / text[2:]
    return ROOT / path


def _latest_completed_suite() -> Path:
    candidates = sorted(
        [
            path
            for path in BENCH_ROOT.iterdir()
            if path.is_dir() and (path / "comparison_summary.json").is_file()
        ],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No completed simpletest suite was found. Run the CUDA13 benchmark first."
        )
    return candidates[0]


def _find_export_root(run_dir: Path) -> Path | None:
    corpus_dir = run_dir / "corpus"
    if not corpus_dir.is_dir():
        return None
    candidates = sorted(
        [path for path in corpus_dir.glob("comic_translate_*") if path.is_dir()],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _copy_candidate_artifacts(
    candidate: dict[str, Any],
    *,
    latest_dir: Path,
    history_dir: Path,
) -> dict[str, Any]:
    candidate_key = str(candidate.get("candidate_key", "") or "candidate")
    warm_run_dirs = [Path(_resolve_path(item)) for item in candidate.get("warm_run_dirs", [])]
    selected_run_dir = next((path for path in reversed(warm_run_dirs) if path.is_dir()), None)
    export_root = _find_export_root(selected_run_dir) if selected_run_dir else None
    translated_dir = export_root / "translated_images" if export_root else None

    copied_images: list[dict[str, str]] = []
    if translated_dir and translated_dir.is_dir():
        image_paths = sorted([path for path in translated_dir.iterdir() if path.is_file()])
        for image_path in image_paths:
            latest_target = latest_dir / "translated_images" / candidate_key / image_path.name
            history_target = history_dir / "translated_images" / candidate_key / image_path.name
            latest_target.parent.mkdir(parents=True, exist_ok=True)
            history_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, latest_target)
            shutil.copy2(image_path, history_target)
            copied_images.append(
                {
                    "original": _repo_relative(image_path),
                    "latest_copy": _repo_relative(latest_target),
                    "history_copy": _repo_relative(history_target),
                }
            )

    candidate_dir = Path(_resolve_path(candidate.get("warm_run_dirs", [""])[-1])).parents[1]
    candidate_summary_json = candidate_dir / "candidate_summary.json"
    candidate_summary_md = candidate_dir / "candidate_summary.md"
    if candidate_summary_json.is_file():
        shutil.copy2(candidate_summary_json, latest_dir / f"{candidate_key}_candidate_summary.json")
        shutil.copy2(candidate_summary_json, history_dir / f"{candidate_key}_candidate_summary.json")
    if candidate_summary_md.is_file():
        shutil.copy2(candidate_summary_md, latest_dir / f"{candidate_key}_candidate_summary.md")
        shutil.copy2(candidate_summary_md, history_dir / f"{candidate_key}_candidate_summary.md")

    return {
        "candidate_key": candidate_key,
        "selected_warm_run_dir": _repo_relative(selected_run_dir),
        "selected_export_root": _repo_relative(export_root),
        "translated_images": copied_images,
    }


def _render_comparison_table(candidates: list[dict[str, Any]]) -> str:
    lines = [
        "| candidate | warm_median_elapsed_sec | warm_median_ocr_total_sec | warm_median_detect_ocr_total_sec | warm_median_translate_median_sec | warm_total_page_failed_count | warm_median_gpu_peak_used_mb | warm_median_gpu_floor_free_mb |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in candidates:
        lines.append(
            "| {candidate_key} | {warm_median_elapsed_sec} | {warm_median_ocr_total_sec} | {warm_median_detect_ocr_total_sec} | {warm_median_translate_median_sec} | {warm_total_page_failed_count} | {warm_median_gpu_peak_used_mb} | {warm_median_gpu_floor_free_mb} |".format(
                **item
            )
        )
    return "\n".join(lines)


def _render_resident_table(candidates: list[dict[str, Any]]) -> str:
    lines = [
        "| candidate | ocr_only_idle_gpu_used_delta_mb | full_idle_gpu_used_delta_mb | gemma_added_idle_gpu_used_delta_mb | gpu_floor_free_mb_after_ocr_only | gpu_floor_free_mb_after_full_runtime |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in candidates:
        resident = item.get("resident_metrics", {}) or {}
        lines.append(
            "| {candidate} | {ocr_only} | {full_idle} | {gemma_added} | {free_ocr} | {free_full} |".format(
                candidate=item.get("candidate_key"),
                ocr_only=resident.get("ocr_only_idle_gpu_used_delta_mb"),
                full_idle=resident.get("full_idle_gpu_used_delta_mb"),
                gemma_added=resident.get("gemma_added_idle_gpu_used_delta_mb"),
                free_ocr=resident.get("gpu_floor_free_mb_after_ocr_only"),
                free_full=resident.get("gpu_floor_free_mb_after_full_runtime"),
            )
        )
    return "\n".join(lines)


def _recommendation(candidate_map: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    paddle = candidate_map.get("paddleocr_vl") or {}
    mangalmm = candidate_map.get("mangalmm") or {}
    paddle_elapsed = paddle.get("warm_median_elapsed_sec")
    mangalmm_elapsed = mangalmm.get("warm_median_elapsed_sec")
    mangalmm_failures = int(mangalmm.get("warm_total_page_failed_count") or 0)
    if mangalmm_failures > 0:
        reasons.append(
            "MangaLMM warm run에서 page_failed_count가 0이 아니므로, 현재 이슈는 Gemma 수치보다 OCR 안정성에 먼저 가깝다"
        )
        reasons.append(
            "simpletest 결과 기준으로는 Gemma4 현행 promoted 값을 유지하고, MangaLMM OCR 성공률과 empty 응답 원인을 먼저 해결하는 편이 안전하다"
        )
        return "keep current Gemma settings", reasons
    if isinstance(paddle_elapsed, (int, float)) and isinstance(mangalmm_elapsed, (int, float)):
        if paddle_elapsed > 0 and mangalmm_elapsed > paddle_elapsed * 1.05:
            reasons.append("MangaLMM warm median elapsed가 PaddleOCR VL보다 5% 이상 느림")

    recommendation = "run focused retune" if reasons else "keep current Gemma settings"
    if not reasons:
        reasons.append(
            "simpletest 결과 기준으로 MangaLMM 안정성 이슈가 없고, Gemma4 현행 promoted 값이 최신 공식 winner를 유지함"
        )
    return recommendation, reasons


def _render_report(
    *,
    suite_dir: Path,
    comparison: dict[str, Any],
    copied_assets: dict[str, Any],
    report_summary: dict[str, Any],
) -> str:
    winner = comparison.get("winner", {}) or {}
    candidates = comparison.get("candidates", []) or []
    lines = [
        "# OCR Simpletest MangaLMM vs PaddleOCR VL Report",
        "",
        "## Metadata",
        "",
        f"- latest suite dir: `{_repo_relative(suite_dir)}`",
        f"- latest assets dir: `{_repo_relative(LATEST_ASSETS)}`",
        f"- baseline gemma report: `{_repo_relative(GEMMA_BASELINE_REPORT)}`",
        "",
        "## Winner",
        "",
        f"- winner: `{winner.get('candidate_key', 'n/a')}`",
        f"- winner_reason: `{comparison.get('winner_reason', 'n/a')}`",
        f"- gemma recommendation: `{report_summary.get('gemma_recommendation')}`",
        "",
        "## Warm Full-Pipeline Comparison",
        "",
        _render_comparison_table(candidates),
        "",
        "## Resident VRAM Delta",
        "",
        _render_resident_table(candidates),
        "",
        "## Candidate Assets",
        "",
    ]
    for candidate_key, payload in copied_assets.items():
        lines.extend(
            [
                f"### {candidate_key}",
                "",
                f"- selected warm run dir: `{payload.get('selected_warm_run_dir')}`",
                f"- selected export root: `{payload.get('selected_export_root')}`",
            ]
        )
        images = payload.get("translated_images", [])
        if images:
            lines.append("- copied translated images:")
            for image in images:
                lines.append(f"  - `{image.get('latest_copy')}`")
        else:
            lines.append("- copied translated images: `none`")
        lines.append("")

    lines.extend(
        [
            "## Gemma Recommendation",
            "",
            "- fixed baseline:",
            f"  - `context_size={GEMMA_BASELINE['context_size']}`",
            f"  - `threads={GEMMA_BASELINE['threads']}`",
            f"  - `n_gpu_layers={GEMMA_BASELINE['n_gpu_layers']}`",
            f"  - `chunk_size={GEMMA_BASELINE['chunk_size']}`",
            f"  - `temperature={GEMMA_BASELINE['temperature']}`",
            f"  - `max_completion_tokens={GEMMA_BASELINE['max_completion_tokens']}`",
            "- decision reasons:",
        ]
    )
    for reason in report_summary.get("gemma_recommendation_reasons", []):
        lines.append(f"  - {reason}")
    lines.append("")
    return "\n".join(lines)


def _write_results_history(
    *,
    suite_dir: Path,
    comparison: dict[str, Any],
    report_summary: dict[str, Any],
) -> None:
    winner = comparison.get("winner", {}) or {}
    text = "\n".join(
        [
            "# OCR Simpletest MangaLMM vs PaddleOCR VL Results History",
            "",
            "## Current Policy",
            "",
            "- corpus는 항상 `Sample/simpletest` 3장만 사용한다.",
            "- winner는 `warm_total_page_failed_count=0` 후보를 우선하고, 그 안에서 `warm_median_elapsed_sec`가 가장 낮은 후보로 정한다.",
            "- 품질 승격은 사용자의 수동 검수 후에만 한다.",
            "",
            "## Latest Output",
            "",
            f"- latest suite root: `{_repo_relative(suite_dir)}`",
            f"- winner: `{winner.get('candidate_key', 'n/a')}`",
            f"- warm_median_elapsed_sec: `{winner.get('warm_median_elapsed_sec')}`",
            f"- warm_total_page_failed_count: `{winner.get('warm_total_page_failed_count')}`",
            f"- gemma recommendation: `{report_summary.get('gemma_recommendation')}`",
            "- latest copied assets:",
            f"  - `{_repo_relative(LATEST_ASSETS / 'comparison_summary.json')}`",
            f"  - `{_repo_relative(LATEST_ASSETS / 'comparison_summary.md')}`",
            f"  - `{_repo_relative(LATEST_ASSETS / 'report_summary.json')}`",
            f"  - `{_repo_relative(LATEST_ASSETS / 'copied_assets.json')}`",
            "",
            "## History",
            "",
            "- 공식 latest/history는 완료된 CUDA13 suite만 기준으로 갱신한다.",
            f"- current suite archive: `{_repo_relative(HISTORY_ROOT / suite_dir.name)}`",
            "",
        ]
    )
    RESULTS_HISTORY_PATH.write_text(text + "\n", encoding="utf-8")


def generate(suite_dir: Path) -> dict[str, Any]:
    comparison_path = suite_dir / "comparison_summary.json"
    if not comparison_path.is_file():
        raise FileNotFoundError(f"Missing comparison_summary.json in {suite_dir}")

    comparison = _load_json(comparison_path)
    suite_id = suite_dir.name
    latest_dir = LATEST_ASSETS
    history_dir = HISTORY_ROOT / suite_id
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    if history_dir.exists():
        shutil.rmtree(history_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(comparison_path, latest_dir / "comparison_summary.json")
    shutil.copy2(comparison_path, history_dir / "comparison_summary.json")
    shutil.copy2(suite_dir / "comparison_summary.md", latest_dir / "comparison_summary.md")
    shutil.copy2(suite_dir / "comparison_summary.md", history_dir / "comparison_summary.md")

    copied_assets: dict[str, Any] = {}
    candidate_map = {item.get("candidate_key"): item for item in comparison.get("candidates", []) or []}
    for candidate in comparison.get("candidates", []) or []:
        copied_assets[str(candidate.get("candidate_key"))] = _copy_candidate_artifacts(
            candidate,
            latest_dir=latest_dir,
            history_dir=history_dir,
        )

    recommendation, reasons = _recommendation(candidate_map)
    report_summary = {
        "family": FAMILY_NAME,
        "suite_id": suite_id,
        "suite_dir": _repo_relative(suite_dir),
        "winner": comparison.get("winner", {}) or {},
        "gemma_recommendation": recommendation,
        "gemma_recommendation_reasons": reasons,
        "baseline_gemma_report": _repo_relative(GEMMA_BASELINE_REPORT),
        "baseline_gemma_tuning": GEMMA_BASELINE,
        "candidate_count": len(comparison.get("candidates", []) or []),
    }

    _write_json(latest_dir / "copied_assets.json", copied_assets)
    _write_json(history_dir / "copied_assets.json", copied_assets)
    _write_json(latest_dir / "report_summary.json", report_summary)
    _write_json(history_dir / "report_summary.json", report_summary)

    report_text = _render_report(
        suite_dir=suite_dir,
        comparison=comparison,
        copied_assets=copied_assets,
        report_summary=report_summary,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text + "\n", encoding="utf-8")
    _write_results_history(
        suite_dir=suite_dir,
        comparison=comparison,
        report_summary=report_summary,
    )
    return report_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the OCR simpletest MangaLMM vs PaddleOCR VL report and latest/history assets."
    )
    parser.add_argument("--suite-dir", default="", help="Exact suite directory. Defaults to latest completed suite.")
    args = parser.parse_args(argv)
    suite_dir = _resolve_path(args.suite_dir) if args.suite_dir else _latest_completed_suite()
    summary = generate(suite_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
