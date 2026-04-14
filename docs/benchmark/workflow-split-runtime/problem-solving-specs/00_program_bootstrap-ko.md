# 문제 해결 명세서 00 - Program Bootstrap

핵심 문제 해결 방향은 사용자가 착안했다.

## 문제

Requirement 1과 Requirement 2가 하나의 긴 작업으로 얽혀 있어, 측정 전에 구현이 먼저 진행되면 결론의 근거가 약해질 위험이 있었다.

## 사용자 착안 요약

사용자는 전체 워크플로우를 `detect -> OCR -> translation -> inpaint`의 단계형 구조로 재조정할 때 시간 이득이 있는지 먼저 증명하고, 그 다음에만 MangaLMM / PaddleOCR VL 하이브리드 선택기로 넘어가자고 제안했다.

## 현재 구조와 병목 가설

- 현재 제품은 페이지 단위 파이프라인이라 runtime이 단계 전체 기준으로 관리되지 않는다.
- Docker compose up과 health wait가 전체 시간에 큰 고정비일 수 있다.
- 근거 없이 dual-resident 정책이나 selector rule부터 구현하면 위험하다.

## 측정 설계

- Requirement 1과 Requirement 2를 분리한다.
- 먼저 문서/체크리스트/결정 로그를 고정한다.
- 이후 실측 runner와 checkpoint 수집을 추가한다.

## 현재까지 수집한 근거

- `batch_processor.py`는 페이지 단위 파이프라인이다.
- `LocalOCRRuntimeManager`는 단일 OCR 엔진 활성 모델이다.
- `LocalGemmaRuntimeManager`는 별도 runtime이다.
- `develop`에는 benchmark raw docs를 넣지 않는 정책이 있다.

## 해석

이 작업은 구현보다 먼저 "무엇을 측정하고 어떤 문서를 남길지"를 잠그는 것이 맞다. 이번 bootstrap 단계는 그 기준선을 만든다.

## 구현/문서화 내용

- master checklist 생성
- project spec / decision log 생성
- workflow / architecture / usage / results history / report placeholder 생성

## 기대 효과

- 이후 구현이 하네스와 어긋나지 않는다.
- 진행 상황을 체크리스트로 계속 추적할 수 있다.
- 사용자 제안과 설계 변화를 포트폴리오용 narrative로 재사용할 수 있다.

## 실제 효과

- Requirement 1과 Requirement 2의 경계가 문서상으로 잠겼다.
- 브랜치 전략과 문서 저장 전략이 명확해졌다.

## 남은 리스크

- benchmark/lab 원격 기준선 정렬 필요
- 아직 실측 runner와 checkpoint capture가 없다

## 다음 액션

- runtime entrypoint / telemetry map 문서화
- measurement contract 명세화
- family runner scaffold 설계

## 저자 및 기여

- Idea Origin: User
- Planning / Measurement Design / Implementation Detailing / Validation: Collaborative
- Execution Support: Codex
