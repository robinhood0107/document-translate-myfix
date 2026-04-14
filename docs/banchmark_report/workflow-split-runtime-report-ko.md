# Workflow Split Runtime Report

## 현재 상태

- family: `workflow-split-runtime`
- status: `planning_scaffold_only`
- corpus: `Sample/japan_vllm_parallel_subset`
- pages: `13`
- requirement_1_status: `not_measured`
- requirement_2_status: `blocked`

## 보고서 목적

이 문서는 Requirement 1과 Requirement 2의 최신 measured run을 요약하는 generated report의 자리다. 현재는 하네스 기반 계획과 문서 구조를 먼저 고정한 상태이며, 실측 결과가 생기면 아래 항목을 채운다.

## 예정 섹션

1. 실행 계약
2. baseline / candidate 비교
3. Docker lifecycle 분해
4. VRAM / runtime snapshot
5. 품질 동등성
6. 사용자 검수 결과
7. 승격 여부

## 현재 메모

- 하네스 기준으로 Requirement 1과 Requirement 2를 분리했다.
- benchmark full docs/assets는 `benchmarking/lab`에만 유지한다.
- develop에는 raw 결과가 없는 `docs/portfolio/...` 요약 문서만 반영한다.
