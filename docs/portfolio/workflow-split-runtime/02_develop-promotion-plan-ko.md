# Workflow Split Runtime Develop Promotion Plan

## 목적

`benchmarking/lab`에서 검증한 결과 중 실제 제품에 필요한 변경만 `develop`로 가져오는 계획을 정리한다.

## develop에 남길 것

1. `workflow_mode` 설정
2. legacy / stage-batched workflow 분기
3. OCR runtime lifecycle generic 이벤트
4. Gemma runtime lifecycle generic 이벤트
5. 상태 패널/배치 리포트와 호환되는 stage telemetry
6. 포트폴리오형 요약 문서

## develop에 남기지 않을 것

1. raw benchmark outputs
2. generated charts
3. family runner / suite scripts
4. preset
5. latest/history asset trees

## 구현 단계

1. settings 저장 구조에 `workflow_mode` 추가
2. settings UI에 legacy / stage-batched 선택지 추가
3. 배치 오케스트레이터 분기 추가
4. runtime lifecycle 이벤트 일반화
5. regression 확인
6. 번역 자산 갱신

## 성공 조건

- 사용자가 설정창에서 두 workflow를 선택할 수 있다.
- legacy workflow는 회귀가 없다.
- stage-batched workflow는 Requirement 1 evidence와 일치하는 동작을 한다.
- benchmark-specific 자산은 `develop`에 남지 않는다.
