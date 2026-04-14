# Workflow Mode And Runtime Design

## 목적

하네스 요구를 계속 참조하면서, Requirement 1 제품 승격 전에 어떤 코드 경계와 책임 분리를 가져가야 하는지 설계 수준에서 먼저 고정한다.

## 아이디어 착안자

- 사용자

## 현재 코드에서 확인한 진입점

1. 설정 저장/로드
   - `app/ui/settings/settings_page.py`
   - `get_all_settings()`가 전체 설정 딕셔너리를 구성한다.
   - `save_settings()`와 `load_settings()`가 `QSettings` round-trip을 담당한다.
2. 설정 UI
   - `app/ui/settings/tools_page.py`
   - 현재는 translator / OCR / detector / inpainter / HD strategy만 노출되고 workflow mode는 없다.
3. 배치 실행 오케스트레이터
   - `pipeline/batch_processor.py`
   - 현재 실 구현은 페이지 단위 `detect -> ocr -> inpaint -> translate -> render/export` 순서다.
4. OCR 런타임 수명주기
   - `modules/ocr/local_runtime.py`
   - 현재는 `_active_engine` 기반의 단일 활성 OCR 엔진 모델이다.
5. Gemma 런타임 수명주기
   - `modules/translation/local_runtime.py`
   - 현재는 Gemma 전용 compose/health/models 확인 로직이 별도 구현돼 있다.

## 현재 구조에서 보이는 제약

1. 배치 파이프라인이 페이지 단위로 고정되어 있어 stage-batched workflow를 끼워 넣을 자리가 명확하지 않다.
2. OCR runtime과 Gemma runtime이 모두 "자기 전용" 이벤트를 내보내므로, stage 비교용 공통 timing surface가 부족하다.
3. 설정 모델에 workflow mode가 없어서 사용자가 기존/신규 전체 워크플로우를 선택할 수 없다.
4. 정식 신규 전체 플로우를 `candidate_stage_batched_dual_resident`로 승격하려면 현재 단일 `_active_engine` 구조를 그대로 둘 수 없다.

## 제품 설계 원칙

1. Requirement 1 실측 근거가 확정되기 전까지 legacy 동작은 절대 바꾸지 않는다.
2. benchmark 전용 preset, runner, raw evidence는 계속 `benchmarking/lab`에 둔다.
3. `develop` 제품 코드에는 generic orchestration, generic telemetry, workflow selection surface만 둔다.
4. Requirement 1이 제품 승격으로 이어질 경우 정식 신규 전체 플로우는 `candidate_stage_batched_dual_resident`로 구현하고, `candidate_stage_batched_single_ocr`는 benchmark 비교 경로로만 유지한다.
5. Requirement 2는 dual-resident 상주 자체가 아니라, 그 위에서 동작하는 자동 선택기와 사용자 승인 기반 전환 규칙을 추가하는 단계로 취급한다.

## 제안 설정 모델

`tools` 그룹 아래에 아래 키를 추가하는 안을 기본안으로 둔다.

```yaml
tools:
  workflow_mode: legacy_page_pipeline | stage_batched_dual_resident
```

추가 이유는 아래와 같다.

1. 기존 OCR/translator/inpainter 선택과 같은 사용 맥락에 있다.
2. `QSettings` round-trip에 자연스럽게 들어간다.
3. 설정창에서 사용자가 기존 방식과 새 방식을 직접 고를 수 있다는 하네스 요구를 만족한다.

Requirement 2까지 고려한 예약 슬롯은 아래처럼 문서만 선반영한다.

```yaml
tools:
  workflow_mode: legacy_page_pipeline | stage_batched_dual_resident
  ocr_stage_policy: dual_resident_fixed | selector_auto
```

단, `ocr_stage_policy`는 Requirement 1 성공 후에도 Requirement 2 자동 선택기 설계가 끝나기 전까지 제품에 노출하지 않는다.

## 제안 객체 책임 분리

### 1. `BatchWorkflowStrategy`

- 역할: 배치 전체 실행 순서를 추상화한다.
- 최소 구현:
  - `LegacyPageWorkflowStrategy`
  - `DualResidentStageBatchedWorkflowStrategy`

### 2. `BatchWorkflowSelector`

- 역할: `workflow_mode` 설정을 읽고 어떤 strategy를 사용할지 결정한다.
- 위치 후보: `pipeline/batch_processor.py` 인접 모듈 또는 `pipeline/workflows/`

### 3. `StageBatchState`

- 역할: detect/OCR/translate/inpaint 사이를 넘겨야 하는 페이지별 산출물을 보관한다.
- 예상 필드:
  - `image_path`
  - `image`
  - `blk_list`
  - `detect_metadata`
  - `ocr_quality`
  - `translation_metadata`
  - `inpaint_metadata`

### 4. `RuntimeLifecycleCoordinator`

- 역할: OCR/Gemma runtime의 시작, 재사용, health 대기, 종료 이벤트를 공통 포맷으로 정리한다.
- 핵심 목적:
  - Docker 기동 시간
  - healthcheck 대기 시간
  - timeout / retry
  - reuse hit
  - shutdown 비용
  를 같은 축으로 비교 가능하게 만든다.

### 5. `RuntimeTimingSink`

- 역할: benchmark/lab 전용 리포터가 아니라 제품 코드에 남겨도 되는 generic timing event surface를 제공한다.
- 출력 예시:
  - `runtime_compose_up_start`
  - `runtime_compose_up_end`
  - `runtime_health_wait_start`
  - `runtime_health_wait_end`
  - `runtime_reuse_hit`
  - `stage_start`
  - `stage_end`

## 단계형 워크플로우의 목표 구조

```text
detect all pages
-> ensure MangaLMM + PaddleOCR VL runtime(s)
-> OCR all pages under dual-resident policy
-> shutdown OCR runtime(s)
-> ensure Gemma runtime
-> translate all pages
-> shutdown Gemma runtime
-> inpaint all pages
-> render/export all pages
```

여기서 핵심은 "전체 OCR 단계"와 "전체 번역 단계"의 경계를 명시적으로 만드는 것이다.

## 현재 코드에 적용할 때의 안전한 절차

1. legacy page pipeline을 `LegacyPageWorkflowStrategy`로 감싼다.
2. 현재 `BatchProcessor` 내부 stage 코드를 재사용 가능한 함수 단위로 분리한다.
3. dual-resident stage-batched workflow는 같은 내부 stage 함수들을 다른 순서로 호출하는 방식으로 조합하되, OCR stage에서 두 엔진 상주를 수용할 lifecycle 경계를 먼저 만든다.
4. 이때 benchmark 전용 비교/리포트 코드는 넣지 않고, generic telemetry만 남긴다.

## Requirement 1 성공 전까지 구현하지 않을 것

1. 기본 workflow를 stage-batched로 바꾸는 승격
2. benchmark 비교용 `candidate_stage_batched_single_ocr`를 제품 UI에 노출하는 일
3. selector rule
4. benchmark preset / raw report / suite runner의 제품 브랜치 반영

## 다음 설계 단계 체크리스트

1. `workflow_mode` UI 위치와 `legacy` / `candidate_stage_batched_dual_resident` 라벨 확정
2. settings round-trip 키 이름 확정
3. `BatchWorkflowStrategy`와 dual-resident OCR stage 책임 경계 초안 작성
4. OCR/Gemma 공통 lifecycle event payload 초안 작성
5. Requirement 1 실측 체크포인트와 제품 telemetry 이벤트 이름 매핑
