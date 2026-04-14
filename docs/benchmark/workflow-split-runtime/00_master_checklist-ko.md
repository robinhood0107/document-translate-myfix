# Workflow Split Runtime Master Checklist

## 기준 문서

- Source of Truth 1: `harness_collection/01_requirement_workflow_split_harness.md`
- Source of Truth 2: `harness_collection/02_requirement_hybrid_ocr_selector_harness.md`
- 이 문서는 두 하네스를 실제 작업 체크리스트로 변환한 실행 문서다.
- 아이디어 착안자: 사용자

## 브랜치 계획

1. `benchmarking/lab` 직접 반영
   - Requirement 1 full benchmark family
   - raw 결과, full docs, problem solving specs, generated assets
   - 서버 push 정책상 benchmark 자산 포함 브랜치는 `benchmarking/lab` 외 이름으로 publish 불가
2. `develop` -> `feature/workflow-split-runtime`
   - 제품 runtime, 설정 UI, generic telemetry, develop-safe portfolio docs
   - 최종 머지 대상: `develop`
3. `benchmarking/lab` 직접 반영
   - Requirement 2 full benchmark family, 검수 리포트, selector 근거
   - 서버 push 정책상 benchmark 자산 포함 브랜치는 `benchmarking/lab` 외 이름으로 publish 불가
4. `develop` -> `feature/hybrid-ocr-selector`
   - 하이브리드 OCR 선택기, dual-resident 정책, develop-safe portfolio docs
   - 최종 머지 대상: `develop`

## 현재 진행 순서

1. `완료` 하네스 분석 및 전체 계획 수립
2. `완료` Requirement 1 벤치마크 family 문서/체크리스트 scaffold 작성
3. `완료` 현재 제품 파이프라인 진입점 및 런타임 계측 지점 문서화
4. `완료` Requirement 1 family runner / preset / BAT / report generator 명세 고정
5. `완료` Requirement 1 2페이지 smoke baseline 측정 및 근거 파일 고정
6. `진행 중` Requirement 1 13장 full measured run 누적
7. `완료` stage-batched experimental runner 구현 및 candidate suite 연결
8. `대기` Requirement 1 candidate 2종 full measured run 누적
9. `대기` Requirement 1 제품 승격 브랜치에서 `legacy` + `candidate_stage_batched_dual_resident` 정식 전체 플로우 반영
10. `대기` Requirement 2 family 설계 및 사용자 검수 패키지 설계
11. `대기` Requirement 2 실측, selector rule 후보 도출, 문제 해결 명세서 누적
12. `대기` Requirement 2 제품 승격 브랜치 생성 및 develop-safe portfolio 문서 반영

## 프로그램 체크리스트

### A. 하네스 추적

- [x] Requirement 1 하네스를 기준 문서로 잠금
- [x] Requirement 2 하네스를 기준 문서로 잠금
- [x] 기준 데이터셋을 `Sample/japan` curated 13장으로 잠금
- [x] 문서 형식을 "결정 로그 + 프로젝트 명세서"로 잠금
- [x] 벤치마크 full docs는 `benchmarking/lab`, 요약형 포트폴리오 문서는 `develop`으로 분리

### B. Requirement 1 문서와 설계

- [x] family 이름을 `workflow-split-runtime`으로 잠금
- [x] 마스터 체크리스트 문서 생성
- [x] 프로젝트 명세/결정 로그 문서 생성
- [x] 워크플로우 문서 생성
- [x] 아키텍처 문서 생성
- [x] 결과 이력 문서 생성
- [x] 보고서 placeholder 생성
- [x] problem solving specs 초기 세트 생성
- [x] 런타임 계측 체크포인트 표 문서화
- [x] 기존 페이지 단위 파이프라인과 단계형 파이프라인 비교 기준 문서화

### C. Requirement 1 구현 준비

- [x] `pipeline/batch_processor.py`의 현재 단계 이벤트를 맵핑
- [x] `modules/ocr/local_runtime.py`의 OCR runtime 정책 맵핑
- [x] `modules/translation/local_runtime.py`의 Gemma runtime 정책 맵핑
- [ ] 설정 UI에 `workflow_mode` 추가 설계 고정
- [ ] 설정 UI의 정식 신규 전체 플로우 대상을 `candidate_stage_batched_dual_resident`로 잠금
- [x] benchmark family runner 명세 고정
- [x] Windows BAT 쌍 명세 고정
- [x] `Sample/japan` curated 13장 staging runner 구현
- [x] stage-batched experimental runner 구현 및 suite 연결
- [x] `events.jsonl` / `timing_summary.json` / `quality_summary.json` / `vram_snapshots.jsonl` / `docker_timeline.json` 변환 규약 구현

### D. Requirement 1 측정

- [x] 2페이지 smoke 입력 세트 고정
- [x] 공식 시나리오 3개 잠금
- [x] 기존 워크플로우 baseline smoke 측정
- [ ] 기존 워크플로우 baseline 13장 측정
- [ ] 단계형 워크플로우 단일 OCR runtime 측정
- [ ] 단계형 워크플로우 dual-resident OCR 후보 측정
- [x] smoke 기준 Docker compose up / health wait / reuse hit 분해표 작성
- [ ] full 13장 기준 Docker compose up / health wait / reuse hit / timeout / retry 분해표 작성
- [ ] VRAM / ngl / idle runtime snapshot 비교
- [ ] 첫 결과 시간과 전체 완료 시간 비교
- [ ] 페이지 수 증가 시 고정비/변동비 모델 정리

### E. Requirement 1 성공 게이트

- [ ] 총 시간 순이득이 실측으로 확인됨
- [ ] Docker 재기동 패널티를 포함해도 순이득이 유지됨
- [ ] 품질이 동일 이상임
- [ ] 설정창에서 `legacy` / `candidate_stage_batched_dual_resident` 선택안이 설계 완료됨
- [ ] 제품 코드와 benchmark 코드의 경계가 유지됨
- [ ] `candidate_stage_batched_dual_resident`가 단일 OCR 후보보다 불리해도 Requirement 1 자체를 무효화하지 않는다는 정책이 문서/승격 계획에 반영됨

### F. Requirement 2 사전 게이트

- [ ] Requirement 1 성공 판정 문서가 잠김
- [ ] Requirement 2 family 이름과 검수 프로토콜 문서화
- [ ] MangaLMM vs detect box count vs PaddleOCR VL 비교표 형식 확정
- [ ] 사용자 승인/비승인 저장 포맷 확정

### G. Requirement 2 측정과 구현

- [ ] 13장 페이지별 detect 박스 수 정리
- [ ] 13장 페이지별 MangaLMM 결과/실패/`bbox_2d` 상태 정리
- [ ] 13장 페이지별 PaddleOCR VL 보완 결과 정리
- [ ] `p_016.jpg` 포함 난페이지 사례 문서화
- [ ] 사용자 검수 패키지 생성
- [ ] selector rule 후보 도출
- [ ] dual-resident runtime 정책과 selector logging 설계
- [ ] 제품 옵션 추가 설계

### H. develop 반영

- [ ] `feature/workflow-split-runtime` 브랜치 생성
- [ ] develop-safe portfolio 문서 생성
- [ ] workflow mode 제품 코드 반영
- [ ] `candidate_stage_batched_dual_resident`를 정식 신규 전체 플로우로 제품 UI에 반영
- [ ] UI 문구 번역 및 `.qm` 갱신
- [ ] commit / push / PR
- [ ] `feature/hybrid-ocr-selector` 브랜치 생성
- [ ] selector 제품 코드 반영
- [ ] develop-safe hybrid portfolio 문서 생성
- [ ] commit / push / PR

## 진행 기록 규칙

1. 이 문서의 `현재 진행 순서`와 체크박스는 마일스톤마다 업데이트한다.
2. 큰 설계 변경이 생기면 `01_project-spec-and-decision-log-ko.md`에 이유를 남긴다.
3. 실측이 시작되면 `results-history-ko.md`와 report를 함께 갱신한다.
4. 문제 해결 명세서는 raw asset history 안에도 남기고, 이 문서에서는 링크만 유지한다.
5. candidate runner 구현 완료 후에도 latest suite record가 옛 blocked 결과일 수 있으므로, 문서에는 "runner 구현 상태"와 "최신 실측 상태"를 분리해 기록한다.
