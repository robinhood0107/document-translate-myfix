# Workflow Split Runtime Results History

## Current Policy

- Requirement 1 성공 전까지 Requirement 2 제품 승격은 하지 않는다.
- 시간 이득은 실측으로만 판정한다.
- Docker compose up / health wait / timeout / retry는 총 시간에서 분리해 기록한다.
- 품질이 같거나 더 좋아야만 승격 후보가 된다.
- `develop`에는 raw benchmark 결과를 옮기지 않는다.

## Latest Output

- current_status: `baseline_smoke_completed_with_blocked_stage_batched_candidates`
- benchmark_family_created: `true`
- measured_runs: `1`
- requirement_1_status: `baseline_only_measured`
- requirement_2_status: `blocked_by_requirement_1`
- source_lang: `Japanese`
- target_lang: `Korean`
- corpus_root: `Sample/japan`
- selected_files: `094.png, p_016.jpg`

## Latest Suite

- latest_suite_record: `banchmark_result_log/workflow-split-runtime/last_workflow_split_runtime_suite.json`
- smoke: `True`
- completed_scenarios: `baseline_legacy`
- blocked_scenarios: `candidate_stage_batched_single_ocr, candidate_stage_batched_dual_resident`

| scenario | status | report | timing | quality |
| --- | --- | --- | --- | --- |
| baseline_legacy | completed | `./banchmark_result_log/workflow-split-runtime/20260415_050150_baseline_legacy/report.md` | `./banchmark_result_log/workflow-split-runtime/20260415_050150_baseline_legacy/timing_summary.json` | `./banchmark_result_log/workflow-split-runtime/20260415_050150_baseline_legacy/quality_summary.json` |
| candidate_stage_batched_single_ocr | blocked | `./banchmark_result_log/workflow-split-runtime/20260415_050749_candidate_stage_batched_single_ocr/report.md` | `./banchmark_result_log/workflow-split-runtime/20260415_050749_candidate_stage_batched_single_ocr/timing_summary.json` | `./banchmark_result_log/workflow-split-runtime/20260415_050749_candidate_stage_batched_single_ocr/quality_summary.json` |
| candidate_stage_batched_dual_resident | blocked | `./banchmark_result_log/workflow-split-runtime/20260415_050749_candidate_stage_batched_dual_resident/report.md` | `./banchmark_result_log/workflow-split-runtime/20260415_050749_candidate_stage_batched_dual_resident/timing_summary.json` | `./banchmark_result_log/workflow-split-runtime/20260415_050749_candidate_stage_batched_dual_resident/quality_summary.json` |

## Required Tables

1. 레거시 vs stage-batched 총 시간 비교표
2. OCR compose / health wait / actual OCR 비교표
3. Gemma compose / health wait / actual translation 비교표
4. VRAM / free memory / `ngl` 비교표
5. 품질 동등성 비교표
6. 사용자 검수 결과 표

## Promotion Policy

- Requirement 1:
  - benchmark/lab에서 full evidence 고정
  - develop에는 runtime/code/doc summary만 승격
- Requirement 2:
  - 사용자 승인 전에는 selector default-on 승격 금지
  - 사용자가 승인한 임계값만 selector rule 후보로 승격 가능
