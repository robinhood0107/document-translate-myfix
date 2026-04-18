# Workflow Split Runtime Portfolio Checklist

## 목적

- `develop` 승격용 제품 변경과 요약 문서만 추적한다.
- full benchmark evidence와 하네스 원문은 `benchmarking/lab`의 canonical 문서를 기준으로 본다.
- 아이디어 착안자: 사용자

## 기준 문서

- benchmark canonical requirements
  - `docs/benchmark/workflow-split-runtime/requirements/01_requirement_workflow_split_harness.md`
  - `docs/benchmark/workflow-split-runtime/requirements/02_requirement_hybrid_ocr_selector_harness.md`
- benchmark canonical checklist/report
  - `docs/benchmark/workflow-split-runtime/00_master_checklist-ko.md`
  - `docs/benchmark/workflow-split-runtime/results-history-ko.md`
  - `docs/banchmark_report/workflow-split-runtime-report-ko.md`

## 현재 결론

1. Requirement 1 시간 이득은 benchmark로 확인됐다.
   - `baseline_legacy`: `995.846s`
   - `candidate_stage_batched_single_ocr`: `714.725s`
   - 개선률: 약 `28.2%`
2. Requirement 1 승격 후보는 `stage_batched_pipeline + Japanese Optimal(PaddleOCR VL 중심)`으로 잠겼다.
3. Requirement 2 `MangaLMM` hybrid selector는 benchmark 실패로 종료됐다.
4. 제품 승격 전 마지막 blocker는 `CTD + keep_existing_lines` 마스킹 경로를 실제 배치 루트에 연결하는 일이었다.

## 현재 진행 순서

1. `완료` benchmark/harness 기준선과 full evidence를 `benchmarking/lab`에 고정
2. `완료` Requirement 1 시간 이득과 OCR parity를 benchmark로 확인
3. `완료` Requirement 2 `MangaLMM` hybrid는 실패 종료 상태로 문서화
4. `완료` CTD 마스킹 경로를 benchmark branch에서 복구하고 smoke로 검증
5. `진행 중` `feature/workflow-split-runtime`에 CTD 마스킹 연결 + `workflow_mode` 제품 승격 코드 반영
6. `진행 중` `develop` 승격용 포트폴리오 문서 재정리
7. `대기` `.ts` / `.qm` 갱신과 최종 smoke
8. `대기` `feature/workflow-split-runtime -> develop` PR 갱신

## 제품 승격 범위

### 남길 것

- `workflow_mode`
  - `stage_batched_pipeline`
  - `legacy_page_pipeline`
- `Stage-Batched Pipeline (Recommended)` UI
- `Legacy Page Pipeline (Legacy)` UI
- CTD 마스킹 + residue cleanup 실제 연결
- `detect -> OCR -> inpaint -> translate -> render/export` 순서의 제품 stage-batched processor
- develop-safe 요약 문서

### 남기지 않을 것

- benchmark runner / preset / suite
- raw benchmark outputs
- chart/history asset tree
- `MangaLMM` hybrid selector 제품 코드

## 체크리스트

### A. 제품 코드

- [x] CTD mask refiner 설정 round-trip 복구
- [x] `generate_mask()`가 `ctd` / `legacy_bbox`를 실제 분기
- [x] residue cleanup 실제 호출
- [x] `workflow_mode` 저장/로드 추가
- [x] `main_pipeline`에 `legacy` / `stage_batched` 분기 추가
- [x] 제품용 `stage_batched_processor` 추가
- [ ] product stage-batched smoke 재확인

### B. UI / i18n

- [x] Settings > Tools에 `Workflow Mode` UI 추가
- [x] `Stage-Batched Pipeline (Recommended)` / `Legacy Page Pipeline (Legacy)` 라벨 추가
- [ ] `.ts` 갱신
- [ ] `.qm` 재생성

### C. 문서

- [x] benchmark canonical requirement 경로를 `benchmarking/lab`로 고정
- [x] Requirement 2 `failed_closed` 상태를 `develop` 문서에도 반영
- [ ] 승격 요약 문서 최종 검토

### D. 검증 게이트

- [x] Requirement 1 시간 이득 확인
- [x] OCR parity benchmark 결과 재사용
- [x] CTD 마스킹 unit test 통과
- [x] headless smoke 통과
- [ ] 대표 2페이지 CTD smoke 결과 재확인
- [ ] 변경 커밋 / push

## 현재 판단

이번 승격은 “순서 변경만 있는 승격”이 아니다. `CTD 마스킹 연결`이 함께 들어가므로, 최종 검증 범위는 OCR이 아니라 `mask_details`, `cleanup_stats`, cleaned output artifact 여부다. 이 검증만 통과하면 `stage_batched_pipeline`를 기본값으로 올리는 준비가 끝난다.
