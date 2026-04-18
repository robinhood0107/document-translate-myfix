# Workflow Split Runtime Develop Promotion Plan

## 목적

`benchmarking/lab`에서 검증한 결과 중 실제 제품에 필요한 변경만 `develop`로 가져오는 범위를 고정한다.

## develop에 남길 것

1. `workflow_mode`
   - `stage_batched_pipeline`
   - `legacy_page_pipeline`
2. `main_pipeline`의 workflow 분기
3. 제품용 `stage_batched_processor`
4. CTD 마스킹 경로와 residue cleanup 실제 연결
5. Settings > Tools UI와 번역 자산
6. 요약형 포트폴리오 문서

## develop에 남기지 않을 것

1. benchmark runner / suite / preset
2. raw benchmark outputs
3. generated charts/history assets
4. `MangaLMM` hybrid selector
5. benchmark용 raw requirement copies

## 제품 동작

### `Stage-Batched Pipeline (Recommended)`

- `detect all`
- `OCR all`
- `inpaint all`
- `translate all`
- `render/export all`

이번 승격에서는 benchmark winner 기준으로 Japanese `Optimal -> PaddleOCR VL` 경로를 기본 운영안으로 본다.

### `Legacy Page Pipeline (Legacy)`

- 기존 페이지 단위 루프 유지
- UI에서 명시적으로 `Legacy`로 표시

## 기본값 정책

- `stage_batched_pipeline`는 CTD 마스킹 smoke와 i18n 갱신까지 확인되면 기본값으로 전환한다.
- `legacy_page_pipeline`는 숨기지 않고 남겨두되, UI에서 레거시임을 분명히 표시한다.

## 브랜치 계획

1. benchmark canonical docs/evidence
   - 유지 브랜치: `benchmarking/lab`
2. 제품 승격 브랜치
   - 분기: `develop -> feature/workflow-split-runtime`
   - 목표 머지: `feature/workflow-split-runtime -> develop`

## 성공 조건

1. `workflow_mode`가 저장/로드된다.
2. `main_pipeline`이 `legacy` / `stage_batched`를 실제 분기한다.
3. 두 루트 모두 `mask_details.mask_refiner == "ctd"`와 `keep_existing_lines == True`를 반영한다.
4. 대표 smoke에서 신규 심각 artifact가 없다.
5. `.ts` / `.qm`이 최신이다.
