# Hybrid OCR Selector Gate And Plan

## 목적

이 문서는 Requirement 2를 `develop` 관점에서 언제 시작하고 무엇을 승격할 수 있는지 정리한다.

## 시작 게이트

아래를 모두 만족해야 Requirement 2 제품 구현을 시작한다.

1. Requirement 1이 실측 기준으로 성공했다.
2. stage-batched workflow가 품질 동등성을 유지한다.
3. benchmark/lab에서 MangaLMM vs PaddleOCR VL 비교 리포트가 준비됐다.
4. 사용자 검수 패키지가 만들어졌다.

## 제품 목표

1. MangaLMM과 PaddleOCR VL 동시 상주 가능
2. 페이지별 품질/위험도 평가
3. 사용자 승인 기반 selector rule
4. selector 판단 근거 로깅

## `develop`에 남길 것

- selector runtime
- dual-resident OCR 정책
- selector logging
- 사용자에게 보이는 설정 옵션
- 요약형 포트폴리오 문서

## `develop`에 남기지 않을 것

- 페이지별 raw diff pack
- benchmark review asset
- full comparison history

## 현재 상태

- status: `blocked_by_requirement_1`
- next_action: `Requirement 1 실측 완료 후 benchmark/lab에서 검수 패키지 생성`
