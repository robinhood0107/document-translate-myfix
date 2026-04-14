# Workflow Split Runtime Architecture

## 계층

### 1. 제품 runtime 계층

- OCR runtime manager
  - 현재: `LocalOCRRuntimeManager`
  - 목표: stage-aware orchestration + optional dual-resident policy
- translation runtime manager
  - 현재: `LocalGemmaRuntimeManager`
  - 목표: stage-batched lifecycle orchestration
- batch pipeline
  - 현재: page-unit pipeline
  - 목표: legacy pipeline 유지 + stage-batched orchestrator 추가

### 2. benchmark orchestration 계층

- family runner: [workflow_split_runtime_benchmark.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/scripts/workflow_split_runtime_benchmark.py)
- preset / runtime mode 조합기
- curated 13-page corpus stager
- contract artifact normalizer
- generated report updater

### 3. docs / report 계층

- benchmark family docs
- problem solving specs
- generated report
- develop-safe portfolio summary docs

## 현재 구조에서 확인된 핵심 경계

1. [batch_processor.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/pipeline/batch_processor.py:730)는 페이지 단위로 단계를 한 번에 처리한다.
2. [modules/ocr/processor.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/modules/ocr/processor.py:36)는 OCR 시작 직전에 runtime manager를 호출한다.
3. [modules/translation/processor.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/modules/translation/processor.py:38)는 translator 생성 시 Gemma runtime manager를 호출한다.
4. [controller.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/controller.py:1098)는 runtime progress를 UI에 반영하고, 이번 단계부터 `metrics.jsonl`에도 남긴다.
5. [settings_page.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/ui/settings/settings_page.py:580)는 도구 설정 저장의 중심 지점이다.

## Requirement 1 목표 아키텍처

1. legacy page pipeline 유지
2. stage-batched pipeline 추가
3. runtime lifecycle policy를 pipeline 외부 서비스로 분리
4. stage timing 이벤트를 generic telemetry로 노출
5. benchmark harness는 stage telemetry만 읽고 비교/판정/보고서를 담당

## 현재 family runner 구조

1. `Sample/japan` curated 13장을 suite 기준으로 고정한다.
2. `baseline_legacy`는 `benchmark_pipeline.py`를 정확한 output dir로 감싼다.
3. raw `metrics.jsonl` / `summary.json` / `page_snapshots.json` / `managed_runtime_policy.json`를 읽어 contract 산출물 7종으로 정규화한다.
4. 아직 실행할 수 없는 stage-batched candidate 두 개는 같은 산출물 구조를 가진 blocked run으로 남긴다.
5. latest suite record를 읽어 `results-history-ko.md`와 generated report를 갱신한다.

## Requirement 2 목표 아키텍처

1. OCR 평가기
   - detect box count
   - OCR quality summary
   - `bbox_2d` 생성 상태
2. OCR selector
   - MangaLMM 유지
   - PaddleOCR VL fallback
   - 사용자 승인 기반 임계값
3. dual-resident OCR runtime policy
   - 동시 상주 가능
   - 동시 작업 금지
   - 전환 근거 로깅

## 제품과 benchmark의 경계

- `benchmarking/lab`에 남길 것
  - family runner
  - preset
  - raw 결과
  - generated report
  - full narrative docs
- `develop`에 남길 것
  - workflow mode
  - runtime lifecycle policy
  - generic stage telemetry
  - selector rule runtime
  - 요약형 portfolio docs

## 현재 판단

이번 작업의 핵심은 benchmark 코드를 제품 pipeline에 끼워 넣는 것이 아니라, 제품 pipeline에서 필요한 관측 surface와 workflow mode만 추가하는 것이다. 실험 정책과 winner 판정, 검수 패키지 생성, report narrative는 끝까지 benchmark 계층에 남겨야 한다.
