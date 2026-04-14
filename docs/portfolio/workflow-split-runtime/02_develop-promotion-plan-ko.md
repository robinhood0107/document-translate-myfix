# Workflow Split Runtime Develop Promotion Plan

## 목적

`benchmarking/lab`에서 검증한 결과 중 실제 제품에 필요한 변경만 `develop`로 가져오는 계획을 정리한다.

## develop에 남길 것

1. `workflow_mode` 설정
2. legacy / `candidate_stage_batched_dual_resident` workflow 분기
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
2. settings UI에 `legacy` / `candidate_stage_batched_dual_resident` 선택지 추가
3. 배치 오케스트레이터에 dual-resident stage-batched workflow 분기 추가
4. OCR stage에서 dual-resident runtime 수용 가능한 lifecycle 이벤트 일반화
5. Gemma runtime lifecycle 이벤트 일반화
6. regression 확인
7. 번역 자산 갱신

## 브랜치 계획

1. 시작점
   - benchmark family와 full evidence는 `benchmarking/lab`에서 먼저 정리한다.
2. Requirement 1 제품 승격
   - 분기: `develop -> feature/workflow-split-runtime`
   - 범위: 포트폴리오형 요약 문서 + `legacy` / `candidate_stage_batched_dual_resident` 제품 코드 + UI/i18n
   - 목표 머지: `feature/workflow-split-runtime -> develop`
3. Requirement 2 제품 승격
   - 전제: Requirement 1 성공
   - 분기: `develop -> feature/hybrid-ocr-selector`
   - 범위: selector runtime + selector logging + 사용자 승인 기반 전환 규칙 + UI/i18n
   - 목표 머지: `feature/hybrid-ocr-selector -> develop`

## 현재 진행 상태

- `benchmarking/lab` 반영: Requirement 1 family + baseline smoke + dual-resident 승격 정책 반영 완료
- `feature/workflow-split-runtime` 문서 기준선: 원격 push 완료
- `feature/workflow-split-runtime` 원격 publish: 완료
- 제품 코드 구현: 아직 시작 전

## 현재 블로커와 대응

1. 첫 push 훅 이슈는 원격 동일 이름 브랜치 선생성 후 upstream 연결로 해소했다.
2. 이제 문서 기준선은 원격에 publish된 상태이므로 제품 코드 구현 전에 dual-resident 승격 방향을 설계 문서에 고정한다.
3. Requirement 1 13장 full 실측 결과가 없으면 제품 승격 범위는 문서/설계 준비 단계에서 멈춘다.
4. `candidate_stage_batched_dual_resident`가 `single_ocr`보다 덜 유리해도 정식 신규 전체 플로우 승격 방향은 유지한다.

## 성공 조건

- 사용자가 설정창에서 `legacy`와 `candidate_stage_batched_dual_resident` 두 전체 workflow를 선택할 수 있다.
- legacy workflow는 회귀가 없다.
- dual-resident stage-batched workflow는 Requirement 1 evidence와 일치하는 동작을 한다.
- benchmark-specific 자산은 `develop`에 남지 않는다.
