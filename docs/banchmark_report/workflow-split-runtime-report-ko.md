# Workflow Split Runtime Report

핵심 문제 해결 방향은 사용자가 착안했다.

## 현재 상태

- family: `workflow-split-runtime`
- status: `baseline_smoke_completed_with_blocked_stage_batched_candidates`
- corpus: `Sample/japan`
- pages: `13`
- requirement_1_status: `baseline_only_measured`
- requirement_2_status: `blocked`

## 보고서 목적

이 문서는 Requirement 1과 Requirement 2의 최신 measured run을 요약하는 generated report다. 지금 단계의 목표는 먼저 실측 패키지를 고정하고, baseline smoke부터 근거를 쌓아 stage-batched 후보의 공식 판정 준비 상태를 만드는 것이다.

## 최신 요약

- latest_suite_record: `banchmark_result_log/workflow-split-runtime/last_workflow_split_runtime_suite.json`
- smoke: `True`
- completed_scenarios: `baseline_legacy`
- blocked_scenarios: `candidate_stage_batched_single_ocr, candidate_stage_batched_dual_resident`

| scenario | status | total_elapsed_sec | page_done | page_failed |
| --- | --- | --- | --- | --- |
| baseline_legacy | completed | 359.756 | 2 | 0 |
| candidate_stage_batched_single_ocr | blocked |  |  |  |
| candidate_stage_batched_dual_resident | blocked |  |  |  |

## 해석

- 현재 벤치마크 패키지는 `Sample/japan` curated 13장, 공식 시나리오 3개, 필수 산출물 7종, CUDA12/CUDA13 BAT 쌍 기준으로 잠겨 있다.
- `baseline_legacy`는 즉시 실행 가능하고, `candidate_stage_batched_single_ocr` 및 `candidate_stage_batched_dual_resident`는 하네스 계약대로 결과 파일 구조만 먼저 고정한 상태다.
- 따라서 지금 보고서는 “Requirement 1 측정 인프라가 잠겼는가”에 대한 상태 보고이며, 최종 성공 판정 보고서는 아니다.

## 다음 액션

1. `run_workflow_split_runtime_cuda13.bat smoke`로 2페이지 smoke를 실행한다.
2. `baseline_legacy` full 13장 measured run을 누적한다.
3. stage-batched experimental runner를 benchmarking/lab에서 추가한 뒤 candidate 두 시나리오를 실제 실행한다.
4. 세 시나리오의 시간/품질/VRAM 근거가 모이면 Requirement 1 성공 게이트를 판정한다.

## 저자 및 기여

- Idea Origin: User
- Planning / Measurement Design / Implementation Detailing / Validation: Collaborative
- Execution Support: Codex
