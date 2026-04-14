#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FAMILY_NAME = "workflow-split-runtime"
BENCH_ROOT = ROOT / "banchmark_result_log" / FAMILY_NAME
LAST_SUITE_RECORD = BENCH_ROOT / "last_workflow_split_runtime_suite.json"
DOC_ROOT = ROOT / "docs" / "benchmark" / "workflow-split-runtime"
REPORT_PATH = ROOT / "docs" / "banchmark_report" / "workflow-split-runtime-report-ko.md"
RESULTS_HISTORY_PATH = DOC_ROOT / "results-history-ko.md"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _latest_suite() -> dict[str, Any]:
    if LAST_SUITE_RECORD.is_file():
        return _load_json(LAST_SUITE_RECORD)
    return {}


def _suite_status(suite: dict[str, Any]) -> str:
    if not suite:
        return "measurement_package_locked"
    completed = suite.get("completed_scenarios", [])
    smoke = bool(suite.get("smoke", False))
    if completed:
        if smoke:
            return "baseline_smoke_completed_with_blocked_stage_batched_candidates"
        return "baseline_measured_with_blocked_stage_batched_candidates"
    return "measurement_package_locked"


def _requirement_1_status(suite: dict[str, Any]) -> str:
    if not suite:
        return "not_started"
    completed = set(suite.get("completed_scenarios", []))
    if "baseline_legacy" in completed and "candidate_stage_batched_single_ocr" in completed:
        return "ready_for_gate_review"
    if "baseline_legacy" in completed:
        return "baseline_only_measured"
    return "not_started"


def _render_results_history(suite: dict[str, Any]) -> str:
    status = _suite_status(suite)
    requirement_1_status = _requirement_1_status(suite)
    completed = suite.get("completed_scenarios", []) if suite else []
    blocked = suite.get("blocked_scenarios", []) if suite else []
    selected_files = suite.get("selected_files", []) if suite else []
    source_lang = suite.get("source_lang", "Japanese") if suite else "Japanese"
    target_lang = suite.get("target_lang", "Korean") if suite else "Korean"
    lines = [
        "# Workflow Split Runtime Results History",
        "",
        "## Current Policy",
        "",
        "- Requirement 1 성공 전까지 Requirement 2 제품 승격은 하지 않는다.",
        "- 시간 이득은 실측으로만 판정한다.",
        "- Docker compose up / health wait / timeout / retry는 총 시간에서 분리해 기록한다.",
        "- 품질이 같거나 더 좋아야만 승격 후보가 된다.",
        "- `develop`에는 raw benchmark 결과를 옮기지 않는다.",
        "",
        "## Latest Output",
        "",
        f"- current_status: `{status}`",
        "- benchmark_family_created: `true`",
        f"- measured_runs: `{len(completed)}`",
        f"- requirement_1_status: `{requirement_1_status}`",
        "- requirement_2_status: `blocked_by_requirement_1`",
        f"- source_lang: `{source_lang}`",
        f"- target_lang: `{target_lang}`",
        f"- corpus_root: `Sample/japan`",
        f"- selected_files: `{', '.join(selected_files) if selected_files else '094.png, 097.png, 101.png, i_099.jpg, i_100.jpg, i_102.jpg, i_105.jpg, p_016.jpg, p_017.jpg, p_018.jpg, p_019.jpg, p_020.jpg, p_021.jpg'}`",
        "",
        "## Latest Suite",
        "",
    ]
    if suite:
        lines.extend(
            [
                f"- latest_suite_record: `{LAST_SUITE_RECORD.relative_to(ROOT).as_posix()}`",
                f"- smoke: `{bool(suite.get('smoke', False))}`",
                f"- completed_scenarios: `{', '.join(completed) if completed else 'none'}`",
                f"- blocked_scenarios: `{', '.join(blocked) if blocked else 'none'}`",
                "",
                "| scenario | status | report | timing | quality |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for record in suite.get("scenario_records", []):
            if not isinstance(record, dict):
                continue
            lines.append(
                "| {scenario} | {status} | `{report}` | `{timing}` | `{quality}` |".format(
                    scenario=record.get("scenario_name", ""),
                    status=record.get("status", ""),
                    report=record.get("report_path", ""),
                    timing=record.get("timing_summary_path", ""),
                    quality=record.get("quality_summary_path", ""),
                )
            )
    else:
        lines.extend(
            [
                "- latest_suite_record: `not_created_yet`",
                "- package status: runner/preset/BAT/report generator are locked and ready for smoke execution.",
            ]
        )

    lines.extend(
        [
            "",
            "## Required Tables",
            "",
            "1. 레거시 vs stage-batched 총 시간 비교표",
            "2. OCR compose / health wait / actual OCR 비교표",
            "3. Gemma compose / health wait / actual translation 비교표",
            "4. VRAM / free memory / `ngl` 비교표",
            "5. 품질 동등성 비교표",
            "6. 사용자 검수 결과 표",
            "",
            "## Promotion Policy",
            "",
            "- Requirement 1:",
            "  - benchmark/lab에서 full evidence 고정",
            "  - develop에는 runtime/code/doc summary만 승격",
            "- Requirement 2:",
            "  - 사용자 승인 전에는 selector default-on 승격 금지",
            "  - 사용자가 승인한 임계값만 selector rule 후보로 승격 가능",
            "",
        ]
    )
    return "\n".join(lines)


def _render_report(suite: dict[str, Any]) -> str:
    status = _suite_status(suite)
    requirement_1_status = _requirement_1_status(suite)
    lines = [
        "# Workflow Split Runtime Report",
        "",
        "핵심 문제 해결 방향은 사용자가 착안했다.",
        "",
        "## 현재 상태",
        "",
        f"- family: `{FAMILY_NAME}`",
        f"- status: `{status}`",
        "- corpus: `Sample/japan`",
        "- pages: `13`",
        f"- requirement_1_status: `{requirement_1_status}`",
        "- requirement_2_status: `blocked`",
        "",
        "## 보고서 목적",
        "",
        "이 문서는 Requirement 1과 Requirement 2의 최신 measured run을 요약하는 generated report다. 지금 단계의 목표는 먼저 실측 패키지를 고정하고, baseline smoke부터 근거를 쌓아 stage-batched 후보의 공식 판정 준비 상태를 만드는 것이다.",
        "",
        "## 최신 요약",
        "",
    ]
    if suite:
        lines.extend(
            [
                f"- latest_suite_record: `{LAST_SUITE_RECORD.relative_to(ROOT).as_posix()}`",
                f"- smoke: `{bool(suite.get('smoke', False))}`",
                f"- completed_scenarios: `{', '.join(suite.get('completed_scenarios', [])) or 'none'}`",
                f"- blocked_scenarios: `{', '.join(suite.get('blocked_scenarios', [])) or 'none'}`",
                "",
                "| scenario | status | total_elapsed_sec | page_done | page_failed |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for record in suite.get("scenario_records", []):
            if not isinstance(record, dict):
                continue
            lines.append(
                "| {scenario} | {status} | {elapsed} | {done} | {failed} |".format(
                    scenario=record.get("scenario_name", ""),
                    status=record.get("status", ""),
                    elapsed=record.get("total_elapsed_sec", ""),
                    done=record.get("page_done_count", ""),
                    failed=record.get("page_failed_count", ""),
                )
            )
    else:
        lines.extend(
            [
                "- latest_suite_record: `not_created_yet`",
                "- baseline smoke: `pending`",
                "- stage-batched 후보: `contract_locked_but_not_executable_yet`",
            ]
        )

    lines.extend(
        [
            "",
            "## 해석",
            "",
            "- 현재 벤치마크 패키지는 `Sample/japan` curated 13장, 공식 시나리오 3개, 필수 산출물 7종, CUDA12/CUDA13 BAT 쌍 기준으로 잠겨 있다.",
            "- `baseline_legacy`는 즉시 실행 가능하고, `candidate_stage_batched_single_ocr` 및 `candidate_stage_batched_dual_resident`는 하네스 계약대로 결과 파일 구조만 먼저 고정한 상태다.",
            "- 따라서 지금 보고서는 “Requirement 1 측정 인프라가 잠겼는가”에 대한 상태 보고이며, 최종 성공 판정 보고서는 아니다.",
            "",
            "## 다음 액션",
            "",
            "1. `run_workflow_split_runtime_cuda13.bat smoke`로 2페이지 smoke를 실행한다.",
            "2. `baseline_legacy` full 13장 measured run을 누적한다.",
            "3. stage-batched experimental runner를 benchmarking/lab에서 추가한 뒤 candidate 두 시나리오를 실제 실행한다.",
            "4. 세 시나리오의 시간/품질/VRAM 근거가 모이면 Requirement 1 성공 게이트를 판정한다.",
            "",
            "## 저자 및 기여",
            "",
            "- Idea Origin: User",
            "- Planning / Measurement Design / Implementation Detailing / Validation: Collaborative",
            "- Execution Support: Codex",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    suite = _latest_suite()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_render_report(suite), encoding="utf-8")
    RESULTS_HISTORY_PATH.write_text(_render_results_history(suite), encoding="utf-8")
    print(str(REPORT_PATH))
    print(str(RESULTS_HISTORY_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
