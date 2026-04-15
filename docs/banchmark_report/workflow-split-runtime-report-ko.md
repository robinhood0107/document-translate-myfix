# Workflow Split Runtime Report

핵심 문제 해결 방향은 사용자가 착안했다.

## 현재 상태

- family: `workflow-split-runtime`
- status: `requirement_1_full_measurement_completed`
- corpus: `Sample/japan`
- pages: `13`
- requirement_1_status: `ready_for_gate_review_with_dual_resident_recorded`
- requirement_2_status: `blocked`

## 보고서 목적

이 문서는 Requirement 1과 Requirement 2의 최신 measured run을 요약하는 generated report다. 지금 단계의 목표는 실측 패키지를 고정한 뒤 baseline과 stage-batched 후보를 모두 같은 계약 산출물로 측정해 Requirement 1 공식 판정 준비 상태를 만드는 것이다.

## 최신 요약

- latest_suite_record: `banchmark_result_log/workflow-split-runtime/last_workflow_split_runtime_suite.json`
- smoke: `False`
- completed_scenarios: `baseline_legacy, candidate_stage_batched_single_ocr, candidate_stage_batched_dual_resident`
- blocked_scenarios: `none`
- stage_batched_runner: `implemented_on_benchmarking_lab`

| scenario | status | total_elapsed_sec | page_done | page_failed |
| --- | --- | --- | --- | --- |
| baseline_legacy | completed | 995.846 | 13 | 0 |
| candidate_stage_batched_single_ocr | completed | 714.725 | 13 | 0 |
| candidate_stage_batched_dual_resident | completed | 1664.021 | 13 | 0 |

## 해석

- 현재 벤치마크 패키지는 `Sample/japan` curated 13장, 공식 시나리오 3개, 필수 산출물 7종, CUDA12/CUDA13 BAT 쌍 기준으로 잠겨 있다.
- `baseline_legacy`와 두 stage-batched candidate는 모두 같은 family runner에서 실행 가능하도록 연결되었고, 최신 suite record가 무엇을 실제로 측정했는지가 현재 상태를 결정한다.
- 따라서 지금 보고서는 “Requirement 1 측정 인프라와 최신 실측 근거가 어디까지 왔는가”에 대한 상태 보고이며, 최종 성공 판정 보고서는 아니다.

## 다음 액션

1. 세 시나리오의 시간/품질/VRAM 근거를 비교해 Requirement 1 성공 게이트를 잠근다.
2. `candidate_stage_batched_dual_resident` sidecar review pack을 사용자 검수 입력으로 정리한다.
3. Chinese routing smoke와 stage lifecycle 보조 근거를 결정 로그에 반영한다.
4. Requirement 1이 유효하면 `legacy` + `stage_batched_pipeline` 제품 승격 구현으로 넘어간다.

## 저자 및 기여

- Idea Origin: User
- Planning / Measurement Design / Implementation Detailing / Validation: Collaborative
- Execution Support: Codex
