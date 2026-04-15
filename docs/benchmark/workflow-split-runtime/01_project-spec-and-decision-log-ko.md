# Workflow Split Runtime Project Spec And Decision Log

## 문서 목적

- 사용자의 요구를 구현 가능한 프로젝트 명세로 고정한다.
- 대화에서 정리된 판단 근거를 유지보수 가능한 결정 로그로 남긴다.
- 이후 Requirement 1, Requirement 2 구현자와 리뷰어가 같은 기준을 공유하도록 만든다.
- 아이디어 착안자: 사용자

## 사용자 문제 정의 요약

사용자는 현재 페이지 단위로 묶여 있는 파이프라인이 Docker 기동과 healthcheck 대기 때문에 전체 성능을 충분히 끌어내지 못한다고 보고 있다. 특히 OCR 컨테이너는 약 1분, Gemma4는 약 3~4분의 기동 대기가 있을 수 있으므로, OCR 단계와 번역 단계를 서로 상주시키지 않고 전체 단계별로 분리하면 VRAM을 더 공격적으로 재배치하고 `ngl` 최대치를 높여 더 빠르게 돌릴 수 있는지 검증하고자 한다.

또한 MangaLMM 단독은 `p_016.jpg` 같은 어려운 페이지에서 텍스트 감지와 `bbox_2d` 생성에 취약하므로, Requirement 1이 성공하면 MangaLMM과 PaddleOCR VL을 동시에 상주시켜 페이지별로 더 적합한 OCR 경로를 선택하는 기준을 만들고 싶어 한다.

## 이번 문서에서 잠근 결정

1. Requirement 1과 Requirement 2는 분리된 단계로 진행한다.
2. Requirement 2는 Requirement 1 성공 후에만 진행한다.
3. full benchmark docs/assets/raw logs는 `benchmarking/lab`에 둔다.
4. `develop`에는 raw 결과 없는 요약형 포트폴리오 문서만 둔다.
5. 문서의 사고과정은 내부 추론 원문이 아니라, "문제를 어떻게 정의했고 어떤 근거로 어떤 결정을 내렸는지"를 시간순 결정 로그로 남긴다.
6. family 이름은 Requirement 1 기준 `workflow-split-runtime`으로 잠근다.
7. 하네스의 `Sample/japan_vllm_parallel_subset` 문구는 legacy 표현으로 보고, 실제 코퍼스 루트는 `Sample/japan` curated 13장으로 잠근다.
8. 원격 push 정책상 benchmark 자산이 포함된 publish 브랜치는 사실상 `benchmarking/lab`이어야 하므로, benchmark family는 `benchmarking/lab`에 직접 반영한다.
9. Requirement 1 family는 먼저 “실행 계약이 잠긴 패키지”로 완성하고, baseline legacy부터 실측을 시작한다.
10. stage-batched candidate 두 개는 experimental runner 구현 전까지 blocked 계약 run으로 남기고, 구현 후에는 같은 family suite에서 실제 measured run으로 승격한다.
11. `candidate_stage_batched_dual_resident`는 단일 OCR 후보와의 benchmark 비교 결과가 더 나쁘더라도 Requirement 1 자체를 무효화하는 실패 조건으로 취급하지 않는다.
12. 최종 제품 승격 대상은 `candidate_stage_batched_dual_resident` 자체가 아니라 `stage_batched_pipeline`이며, `candidate_stage_batched_dual_resident`는 그 안의 `Optimal+ Japanese analysis mode` benchmark 기준 시나리오로 잠근다.
13. `stage_batched_pipeline`의 OCR stage Docker routing은 `OCR mode + source language` 기준으로 결정한다.
14. routing matrix는 다음으로 잠근다.
    - `Optimal`: Chinese -> `HunyuanOCR` only, Japanese/Other -> `PaddleOCR VL` only
    - `Optimal+`: Chinese -> `HunyuanOCR` only, Japanese -> `PaddleOCR VL + MangaLMM`, Other -> `PaddleOCR VL` only
15. OCR stage 종료 후 OCR 관련 Docker는 전부 내려가고, translation stage는 `Gemma4`만 올리는 구조를 유지한다.
16. `Optimal+ Japanese`는 selector 승인 전까지 `PaddleOCR VL`을 downstream 기준으로 사용하고 `MangaLMM`은 sidecar 비교 데이터로만 수집한다.

## 운영 기준으로 잠근 입력 세트

- corpus root: `Sample/japan`
- source language: `Japanese`
- target language: `Korean`
- official 13 pages:
  - `094.png`
  - `097.png`
  - `101.png`
  - `i_099.jpg`
  - `i_100.jpg`
  - `i_102.jpg`
  - `i_105.jpg`
  - `p_016.jpg`
  - `p_017.jpg`
  - `p_018.jpg`
  - `p_019.jpg`
  - `p_020.jpg`
  - `p_021.jpg`
- smoke pages:
  - `094.png`
  - `p_016.jpg`

## 레포에서 확인한 현재 상태

### 1. 현재 제품 배치 파이프라인

- [batch_processor.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/pipeline/batch_processor.py:730) 기준 실제 순서는 페이지 단위 `detect -> ocr -> inpaint -> translate -> render`다.
- 즉 사용자가 원하는 `detect all -> ocr all -> translate all -> inpaint all` 구조는 아직 제품에 없다.

### 2. OCR runtime 정책

- [modules/ocr/local_runtime.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/modules/ocr/local_runtime.py:61)의 `LocalOCRRuntimeManager`는 한 번에 하나의 OCR 엔진만 활성화하는 모델이다.
- 활성 엔진이 바뀌면 이전 OCR 엔진을 `docker compose down`으로 내리는 구조다.
- Requirement 2는 이 정책을 "동시 상주 가능하지만 동시 작업은 하지 않는" 모델로 확장해야 한다.

### 3. Gemma runtime 정책

- [modules/translation/local_runtime.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/modules/translation/local_runtime.py:84)의 `LocalGemmaRuntimeManager`는 Gemma 런타임을 별도 관리한다.
- 현재 OCR 런타임과 Gemma 런타임의 수명주기는 제품 단계 기준으로 분리된 오케스트레이션이 없다.

### 4. 설정 UI 상태

- [settings_ui.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/ui/settings/settings_ui.py:52)와 [settings_page.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/ui/settings/settings_page.py:580)에는 OCR/translator 선택은 있지만 workflow mode는 없다.
- Requirement 1 제품 승격 시 설정창에 legacy / stage-batched workflow 선택지를 추가해야 한다.

### 5. benchmark 정책

- [docs/repo/benchmark-branch-policy-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/repo/benchmark-branch-policy-ko.md:1)에 따라 `develop`에는 benchmark raw docs와 generated assets를 넣지 않는다.
- 따라서 사용자 요청인 "develop에도 저장"은 `docs/portfolio/...`의 요약형 문서로 해석해 잠갔다.

## 이번 단계에서 실제 구현한 것

1. `workflow_split_runtime` preset 3종 고정
2. family runner 추가
3. CUDA12/CUDA13 BAT launcher 추가
4. generated report 갱신 스크립트 추가
5. runtime progress를 `metrics.jsonl`에 남길 memlog bridge 추가
6. runner contract / execution protocol 문제 해결 명세서 추가
7. baseline smoke 실측 완료 및 Docker lifecycle 분해표 고정
8. stage-batched experimental runner 구현 및 family suite 연결

## 정책 변경으로 새로 잠근 사항

사용자 지시에 따라 `candidate_stage_batched_dual_resident`는 더 이상 “불리하면 버릴 수 있는 탐색 후보”가 아니다. 다만 최종 제품 개념은 `dual-resident 전체 플로우`가 아니라 `stage_batched_pipeline`이며, 이 시나리오는 그 안의 `Optimal+ Japanese analysis mode` benchmark 기준으로 다룬다.

1. benchmark에서는 계속 baseline 및 single OCR 후보와 비교한다.
2. 성능이 single OCR 후보보다 불리하더라도 Requirement 1 전체 성공/실패 판정 자체를 무효화하지 않는다.
3. 최종 제품 승격에서는 `legacy`와 `stage_batched_pipeline`를 사용자가 선택할 수 있는 두 개의 전체 플로우를 기본안으로 둔다.
4. `stage_batched_pipeline` 내부 OCR stage residency는 `OCR mode + source_lang`로 결정하며, `Optimal+ Japanese`에서만 `PaddleOCR VL + MangaLMM` dual-resident가 일어난다.
5. `Optimal+ Japanese` selector 승인 전에는 `PaddleOCR VL`을 downstream 기준으로 고정한다.
6. 따라서 이후 문서, 체크리스트, develop 승격 설계는 모두 이 정책을 기준으로 맞춘다.

## 최신 실측 결과 요약

### 1. Requirement 1 공식 일본어 13장 suite

- latest suite record: `banchmark_result_log/workflow-split-runtime/last_workflow_split_runtime_suite.json`
- baseline full run: `20260415_055838_baseline_legacy`
  - `total_elapsed_sec=995.846`
  - `page_done_count=13`
  - `page_failed_count=0`
- Japanese `Optimal` stage-batched run: `20260415_090653_candidate_stage_batched_single_ocr`
  - `total_elapsed_sec=714.725`
  - `page_done_count=13`
  - `page_failed_count=0`
- Japanese `Optimal+ analysis mode` stage-batched run: `20260415_091848_candidate_stage_batched_dual_resident`
  - `total_elapsed_sec=1664.021`
  - `page_done_count=13`
  - `page_failed_count=0`
  - sidecar review pack:
    - `banchmark_result_log/workflow-split-runtime/20260415_091848_candidate_stage_batched_dual_resident/sidecar_review_pack.json`
    - `banchmark_result_log/workflow-split-runtime/20260415_091848_candidate_stage_batched_dual_resident/sidecar_review_pack.md`

### 2. Chinese routing smoke

최신 검증 출력은 checkpoint 집계 보정 후 다시 실행한 아래 두 run으로 잠근다.

- `20260415_095534_chinese_optimal_smoke`
- `20260415_095534_chinese_optimal_plus_smoke`

두 run 모두 아래가 확인되었다.

1. `ocr_stage_policy.json`에서 `Optimal` / `Optimal+` 모두 Chinese 기준 `HunyuanOCR` only로 해석됨
2. `ocr_stage_shutdown` checkpoint 이후 active container 집합이 빈 배열로 기록됨
3. `translate_stage_runtime_ready` checkpoint에서는 `gemma-local-server`만 기록됨
4. 즉 사용자가 의도한 “OCR stage 종료 후 OCR Docker 전부 down -> Gemma만 up” 수명주기가 smoke 수준에서 확인됨

### 3. 계측 의미 보정

초기 Chinese smoke에서는 `translate_stage_runtime_ready`와 이후 checkpoint에 OCR container 이름이 함께 남았다. 원인은 실제 활성 컨테이너가 아니라 `runtime_policy` 누적 이력을 checkpoint에 재사용했기 때문이다.

이 문제는 `scripts/benchmark_stage_batched_pipeline.py`에서 active container 집합을 stage별로 별도 추적하도록 수정해 해결했다. 그 뒤 Chinese smoke를 다시 실행해 아래 semantics를 잠갔다.

- OCR stage ready: OCR container만 기록
- OCR stage shutdown: 빈 배열
- translate stage ready / end: Gemma만 기록
- translate stage shutdown: 빈 배열
- inpaint / render end: 빈 배열

## 단계별 실행 전략

### Phase 1. Requirement 1 benchmarking/lab

- 목적: 시간 이득, Docker 대기 병목, VRAM/ngl 이득, 품질 동등성을 실측으로 판정
- 산출물:
  - family docs
  - suite/report
  - problem solving specs
  - raw logs
  - runner/bat/preset
  - 최종 승격 조건 문서

### Phase 2. Requirement 1 develop promotion

- 목적: workflow mode, generic telemetry, runtime orchestration, 포트폴리오 요약 문서를 제품 브랜치에 반영
- 산출물:
  - 제품 코드
  - UI 번역
  - develop-safe portfolio docs

### Phase 3. Requirement 2 benchmarking/lab

- 목적: MangaLMM vs PaddleOCR VL vs detect-box 기준을 비교하고 사용자 검수 패키지 생성
- 산출물:
  - 페이지별 diff pack
  - 승인/비승인 기록
  - selector rule 후보
  - 문제 해결 명세서

### Phase 4. Requirement 2 develop promotion

- 목적: dual-resident OCR runtime 정책과 selector rule을 제품 옵션으로 반영
- 산출물:
  - selector 구현
  - 설정 UI
  - develop-safe portfolio docs

## 아직 열어 둔 기술 리스크

1. benchmark 자산 포함 브랜치는 `benchmarking/lab` 외 이름으로 push가 거부되는 운영 제약이 확인되었다.
2. 단계형 워크플로우가 인페이팅/렌더 순서에 어떤 부수효과를 주는지 아직 실측이 없다.
3. Requirement 2의 selector rule은 사용자 검수 결과가 쌓이기 전까지 자동 승격하면 안 된다.
4. 최신 suite record가 옛 blocked 상태를 가리킬 수 있으므로, "runner 구현 완료"와 "latest measured suite"를 구분해서 해석해야 한다.

## 다음 액션

1. Requirement 1 공식 게이트용 비교표를 고정
   - 레거시 vs Japanese `Optimal`
   - Japanese `Optimal` vs Japanese `Optimal+ analysis mode`
   - Docker startup / health wait / VRAM / 품질 비교
2. 아래 문서를 기준으로 사용자 O/X 채점을 수집
   - `docs/benchmark/workflow-split-runtime/02_review-pack-scoring-guide-ko.md`
   - `docs/benchmark/workflow-split-runtime/03_requirement-1-gate-and-threshold-proposal-ko.md`
   - `banchmark_result_log/workflow-split-runtime/20260415_091848_candidate_stage_batched_dual_resident/review_decision_sheet-ko.md`
3. 사용자 O/X 결과를 기준으로 selector threshold 후보를 잠근다.
4. Requirement 1 유효성 판정이 끝나면 `feature/workflow-split-runtime`에서 `stage_batched_pipeline` 제품 승격 준비를 시작

## 저자 및 기여

- Idea Origin: User
- Planning / Measurement Design / Implementation Detailing / Validation: Collaborative
- Execution Support: Codex
