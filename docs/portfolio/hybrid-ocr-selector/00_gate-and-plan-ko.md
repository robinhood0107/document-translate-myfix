# Hybrid OCR Selector Gate And Plan

## 목적

이 문서는 Requirement 2를 `develop` 관점에서 언제 시작하고 무엇을 승격할 수 있는지 정리한다.

## 시작 게이트

아래를 모두 만족해야 Requirement 2 제품 구현을 시작한다.

1. Requirement 1이 실측 기준으로 성공했다.
2. `candidate_stage_batched_dual_resident` workflow가 Requirement 1 정식 신규 전체 플로우로 승격됐다.
3. benchmark/lab에서 MangaLMM vs PaddleOCR VL 비교 리포트가 준비됐다.
4. 사용자 검수 패키지가 만들어졌다.

## 제품 목표

1. 이미 승격된 dual-resident workflow 위에서 페이지별 품질/위험도 평가 추가
2. 사용자 승인 기반 selector rule 설계
3. selector 판단 근거 로깅
4. 보수적 fallback 중심 자동 전환 전략 정리

## `develop`에 남길 것

- selector runtime
- selector logging
- 사용자에게 보이는 설정 옵션
- 요약형 포트폴리오 문서

## `develop`에 남기지 않을 것

- 페이지별 raw diff pack
- benchmark review asset
- full comparison history

## 현재 상태

- status: `blocked_by_requirement_1`
- next_action: `Requirement 1 실측 완료와 dual-resident workflow 승격 방향 확정 후 benchmark/lab에서 검수 패키지 생성`
