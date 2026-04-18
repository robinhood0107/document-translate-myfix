# Workflow Split Runtime Results History

## Current Policy

- Requirement 1 성공 전까지 Requirement 2 제품 승격은 하지 않는다.
- 시간 이득은 실측으로만 판정한다.
- Docker compose up / health wait / timeout / retry는 총 시간에서 분리해 기록한다.
- 품질이 같거나 더 좋아야만 승격 후보가 된다.
- `candidate_stage_batched_dual_resident`는 단일 OCR 후보보다 불리해도 Requirement 1 자체를 무효화하지 않는다.
- 정식 신규 전체 플로우는 `stage_batched_pipeline`이며, OCR stage resident set은 `OCR mode + source_lang`으로 결정한다.
- `develop`에는 raw benchmark 결과를 옮기지 않는다.

## Latest Output

- current_status: `requirement_1_flow_gain_confirmed_requirement_2_closed_failed`
- benchmark_family_created: `true`
- measured_runs: `3`
- requirement_1_status: `flow_gain_confirmed_with_quality_parity`
- requirement_2_status: `closed_failed_by_user_decision_after_benchmark`
- source_lang: `Japanese`
- target_lang: `Korean`
- corpus_root: `Sample/japan`
- selected_files: `094.png, 097.png, 101.png, i_099.jpg, i_100.jpg, i_102.jpg, i_105.jpg, p_016.jpg, p_017.jpg, p_018.jpg, p_019.jpg, p_020.jpg, p_021.jpg`
- supplementary_routing_smoke: `see 01_project-spec-and-decision-log-ko.md`

## Latest Suite

- latest_suite_record: `banchmark_result_log/workflow-split-runtime/last_workflow_split_runtime_suite.json`
- smoke: `False`
- completed_scenarios: `baseline_legacy, candidate_stage_batched_single_ocr, candidate_stage_batched_dual_resident`
- blocked_scenarios: `none`
- runner_state: `stage_batched_candidates_runnable`

| scenario | status | report | timing | quality |
| --- | --- | --- | --- | --- |
| baseline_legacy | completed | `./banchmark_result_log/workflow-split-runtime/20260415_055838_baseline_legacy/report.md` | `./banchmark_result_log/workflow-split-runtime/20260415_055838_baseline_legacy/timing_summary.json` | `./banchmark_result_log/workflow-split-runtime/20260415_055838_baseline_legacy/quality_summary.json` |
| candidate_stage_batched_single_ocr | completed | `./banchmark_result_log/workflow-split-runtime/20260415_090653_candidate_stage_batched_single_ocr/report.md` | `./banchmark_result_log/workflow-split-runtime/20260415_090653_candidate_stage_batched_single_ocr/timing_summary.json` | `./banchmark_result_log/workflow-split-runtime/20260415_090653_candidate_stage_batched_single_ocr/quality_summary.json` |
| candidate_stage_batched_dual_resident | completed | `./banchmark_result_log/workflow-split-runtime/20260415_091848_candidate_stage_batched_dual_resident/report.md` | `./banchmark_result_log/workflow-split-runtime/20260415_091848_candidate_stage_batched_dual_resident/timing_summary.json` | `./banchmark_result_log/workflow-split-runtime/20260415_091848_candidate_stage_batched_dual_resident/quality_summary.json` |

## Required Tables

1. 레거시 vs stage-batched 총 시간 비교표
2. OCR compose / health wait / actual OCR 비교표
3. Gemma compose / health wait / actual translation 비교표
4. VRAM / free memory / `ngl` 비교표
5. 품질 동등성 비교표
6. 사용자 검수 결과 표

## Final Flow Comparison

| scenario | total_elapsed_sec | delta_vs_baseline_sec | delta_vs_baseline_pct | quality_note |
| --- | ---: | ---: | ---: | --- |
| `baseline_legacy` | `995.846` | `0.000` | `0.0%` | `detect_box_total=212`, `ocr_non_empty_total=212`, `page_failed_count=0` |
| `candidate_stage_batched_single_ocr` | `714.725` | `-281.121` | `-28.2%` | `detect_box_total=212`, `ocr_non_empty_total=212`, `page_failed_count=0` |
| `candidate_stage_batched_dual_resident` | `1664.021` | `+668.175` | `+67.1%` | `analysis mode only, not a promotion candidate` |

해석:

- Requirement 1의 핵심 질문이었던 flow 비교에서는 `stage_batched_pipeline`가 실질적인 시간 이득을 보였다.
- 이 이득은 `total_elapsed_sec` 기준이므로 Docker compose up / health wait를 포함한 값이다.
- 반면 `candidate_stage_batched_dual_resident`는 Requirement 2 hybrid selector 트랙의 실패 사례로 기록한다.

## Promotion Policy

- Requirement 1:
  - benchmark/lab에서 full evidence 고정
  - develop에는 runtime/code/doc summary만 승격
- Requirement 2:
  - 2026-04-18 기준 `failed_closed`
  - 새로운 근거가 생기기 전까지 selector default-on 승격을 진행하지 않음
