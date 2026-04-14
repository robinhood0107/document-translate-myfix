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
10. stage-batched candidate 두 개는 experimental runner가 추가되기 전까지 blocked 계약 run으로 정직하게 남긴다.

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
4. Requirement 1 candidate 두 시나리오는 experimental runner가 없어서 아직 실제 실행이 막혀 있다.

## 다음 액션

1. `run_workflow_split_runtime_cuda13.bat smoke`로 baseline smoke 실행
2. baseline 13장 measured run 누적
3. stage-batched experimental runner 추가
4. candidate 두 시나리오 실제 측정

## 저자 및 기여

- Idea Origin: User
- Planning / Measurement Design / Implementation Detailing / Validation: Collaborative
- Execution Support: Codex
