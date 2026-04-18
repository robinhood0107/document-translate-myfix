# Workflow Split Runtime Report

핵심 문제 해결 방향은 사용자가 착안했다.

## 현재 상태

- family: `workflow-split-runtime`
- status: `requirement_1_flow_gain_confirmed_requirement_2_closed_failed`
- corpus: `Sample/japan`
- pages: `13`
- requirement_1_status: `flow_gain_confirmed_with_quality_parity`
- requirement_2_status: `failed_closed`

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
- Requirement 1 flow 비교는 공식 suite 기준으로 판정이 끝났다.
  - `baseline_legacy`: `995.846s`
  - `candidate_stage_batched_single_ocr`: `714.725s`
  - 순이득: `281.121s`, 약 `28.2%`
- official quality summary 기준으로 두 시나리오는 모두 `detect_box_total=212`, `ocr_non_empty_total=212`, `page_failed_count=0`이므로 품질 동등성이 유지된다.
- 따라서 Requirement 1의 결론은 `stage_batched_pipeline`에 실질적 시간 이득이 있으며, 현재 정식 승격 후보는 `Japanese Optimal(PaddleOCR VL 중심)`이라는 것이다.
- 반면 `candidate_stage_batched_dual_resident`는 `1664.021s`로 공식 suite에서 가장 느렸고, Requirement 2 MangaLMM hybrid selector 트랙은 benchmark 실패로 종료한다.

## 다음 액션

1. `feature/workflow-split-runtime`에서 `legacy` + `stage_batched_pipeline` 제품 승격 구현으로 넘어간다.
2. 다음 검증 우선순위는 hybrid selector가 아니라, 현재 레거시로 강제되는 마스킹 경로를 사용자가 의도한 방식으로 교체하는 작업이다.
3. Requirement 2는 `failed_closed` 상태로 문서화해 두고, 새 근거가 생기기 전까지 재개하지 않는다.

## 저자 및 기여

- Idea Origin: User
- Planning / Measurement Design / Implementation Detailing / Validation: Collaborative
- Execution Support: Codex
