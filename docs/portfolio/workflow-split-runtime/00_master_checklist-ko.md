# Workflow Split Runtime Portfolio Checklist

## 문서 목적

- `develop` 관점에서 이번 기능의 진행 상태와 승격 범위를 추적한다.
- benchmark/lab에서 쌓인 full evidence를 제품 승격 체크리스트로 변환한다.
- 아이디어 착안자: 사용자

## 기준 문서

- `harness_collection/01_requirement_workflow_split_harness.md`
- `harness_collection/02_requirement_hybrid_ocr_selector_harness.md`
- full benchmark evidence branch: `benchmarking/lab`

## 현재 진행 순서

1. `완료` 하네스 2종 분석 및 전체 프로그램 계획 수립
2. `완료` `benchmarking/lab`에 Requirement 1 family 문서/체크리스트 기준선 반영
3. `진행 중` `develop`용 포트폴리오 문서 기준선 반영 및 하네스 원문 동기화
4. `진행 중` `feature/workflow-split-runtime` 첫 push 블로커 원인 확인
5. `대기` `workflow_mode` 제품 설계 및 설정 UI 반영
6. `대기` generic stage telemetry / runtime lifecycle event 설계
7. `대기` Requirement 1 승격 코드 구현
8. `대기` Requirement 2 게이트 문서화 및 사용자 검수 체계 반영

## 현재 브랜치 진행 기록

- `benchmarking/lab`
  - status: `pushed`
  - commits:
    - `5ce885b` `docs(benchmark): scaffold workflow split runtime family`
    - `5794def` `docs(benchmark): record benchmark branch publish policy`
- `feature/workflow-split-runtime`
  - status: `in_progress`
  - local_commit: `45232ef` `docs(portfolio): scaffold workflow split promotion docs`
  - current_focus:
    - 하네스 원문 동기화
    - 포트폴리오 체크리스트 갱신
    - 첫 push 훅 블로커 분석

## 시작 브랜치부터 최종 머지까지

1. `benchmarking/lab`에서 full evidence와 benchmark family를 누적한다.
2. `develop` 기준으로 `feature/workflow-split-runtime`에서 제품 승격용 문서와 제품 코드를 정리한다.
3. Requirement 1 실측 성공 시 `feature/workflow-split-runtime`을 `develop`로 머지한다.
4. Requirement 2는 `benchmarking/lab`에서 비교/검수 자산을 먼저 확정한다.
5. 이후 `develop` 기준 `feature/hybrid-ocr-selector`에서 selector 제품 코드를 정리한다.
6. Requirement 2 승인 후 `feature/hybrid-ocr-selector`를 `develop`로 머지한다.

## develop 승격 체크리스트

### A. 문서

- [x] 포트폴리오 인덱스 생성
- [x] Workflow Split Runtime 체크리스트 생성
- [x] Workflow Split Runtime 여정 문서 생성
- [x] Workflow Split Runtime 제품 승격 계획 문서 생성
- [x] Hybrid OCR Selector 게이트 문서 생성

### B. 제품 코드

- [ ] `workflow_mode` 설정 추가
- [ ] legacy / stage-batched workflow 분기 추가
- [ ] OCR runtime lifecycle 이벤트 일반화
- [ ] Gemma runtime lifecycle 이벤트 일반화
- [ ] 상태 패널/배치 리포트 회귀 확인

### C. UI / i18n

- [ ] 설정창에 workflow mode 추가
- [ ] 새 UI 문구 번역 `.ts` 반영
- [ ] `.qm` 재생성

### D. 승격 게이트

- [ ] Requirement 1 실측 결과가 시간 이득을 증명
- [ ] 품질 동등성 확인
- [ ] benchmark 전용 자산 없이 제품 코드만 선별
- [ ] `feature/workflow-split-runtime` commit / push / PR

## 현재 판단

`develop`에는 "무엇을 왜 바꾸는지"가 잘 읽히는 문서와 실제 제품 코드만 올라가야 한다. 벤치마크 풀 리포트는 `benchmarking/lab`에서 유지하고, 여기서는 승격 판단의 핵심만 남긴다.

## 현재 리스크

1. `feature/workflow-split-runtime` 첫 push에서 로컬 pre-push 훅이 upstream 부재 상태를 엄격하게 검사해 publish를 막고 있다.
2. Requirement 1 실측 로그가 아직 없으므로 제품 코드 승격은 문서 기준선 단계에 머물러 있다.
3. Hybrid OCR Selector는 Requirement 1 성공 판정 전까지 시작하지 않는다.
