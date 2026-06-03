# 자동번역 성능개선 진실명세서

이 문서는 자동번역 성능개선 프로젝트의 단일 기준 문서다. 아래 내용과 다른 문서가 충돌하면 이 문서를 먼저 고치고, 그 다음 세부 문서를 갱신한다.

## 현재 확정 사실

- 기본 권장 자동번역 흐름은 `StageBatchedProcessor.batch_process()`가 `detect -> ocr -> inpaint -> translation -> render`를 stage 단위로 순차 실행한다.
- 실제 364페이지 Part 3 run에서는 page state timestamp span 기준으로 `inpaint 1777s`, `OCR 821s`, `translation 634s`, `detect 258s`, `render/export 205s`, stage gap/prewarm 74s가 관측됐다.
- Gemma 반복 확인은 실제로 존재한다. `StageBatchedProcessor._await_gemma_runtime()`이 stage 시작 전에 `ensure_server()`를 한 번 호출하지만, page loop 안에서 `Translator(...)`를 다시 만들고 `Translator.__init__()`이 `LocalGemmaRuntimeManager.ensure_server()`를 다시 호출한다.
- OCR 반복 확인도 존재한다. stage-batched는 prewarm/await 경로가 있지만, `OCRProcessor.initialize()`는 local OCR이면 항상 `LocalOCRRuntimeManager.ensure_engine()`을 호출한다. legacy, manual, webtoon 경로에서는 이 initialize가 페이지 또는 visible-area 작업마다 반복될 수 있다.
- PaddleOCR VL과 Hunyuan OCR은 이미 block 단위 내부 `ThreadPoolExecutor`를 사용한다. 따라서 page 단위 OCR concurrency는 기존 block worker 수와 곱해져 runtime 부하가 커질 수 있다.
- inpaint/debug/final output write는 현재 대부분 동기 경로다. 특히 cleaned image, raw mask, overlay, cleanup delta, debug metadata, final output write가 page loop 안에서 바로 실행된다.
- Gemma HTTP translation request는 `requests.post()` 기반 동기 호출이다. 앱 쪽 page loop도 순차이므로 `LLAMA_N_PARALLEL`을 2 이상으로 올려도 앱이 동시에 요청하지 않으면 효과가 제한된다.
- 사용자 환경에서는 Gemma 실행 중 GPU util 85% 이상 또는 VRAM 여유 2~3GB 미만 조건에 거의 항상 도달한다. 따라서 GPU-bound page concurrency는 현재 활성 제품 계획에서 제외한다.
- 단일 GPU에서 OCR, inpaint, Gemma를 동시에 강하게 밀어붙이는 방식은 현재 develop 제품 계획에 넣지 않는다.

## 불변 조건

- 결과 이미지, OCR 텍스트, 번역 텍스트, inpaint patch, project save/load 상태가 바뀌는 성능개선은 반드시 회귀 테스트와 샘플 산출물을 남긴다.
- 사용자에게 보이는 이미지/결과물 품질이 바뀌면 병합 전 사용자 검토를 요청한다.
- 기본값은 보수적으로 둔다. 새 병렬화 옵션은 먼저 feature flag 또는 설정값 default-off로 실험한다.
- cancellation은 실패나 skip으로 기록하지 않는다.
- external runtime startup/probe cache는 연결 오류, HTTP 오류, 모델 불일치, 설정 변경 시 invalidate할 수 있어야 한다.
- Qt render/layout 객체를 임의 worker thread로 옮기지 않는다.
- `LLAMA_N_PARALLEL > 1`, Gemma page concurrency, OCR page concurrency, inpaint page concurrency는 기본값 또는 활성 제품 PR로 도입하지 않는다.
- 비동기 writer도 결과물 보장과 cancel/drain 정책이 필요한 동시성 변경이므로, I/O 병목이 별도로 입증되기 전에는 활성 계획에서 제외한다.
- `main`, `develop`, `benchmarking/lab` 장기 브랜치 정책과 `rules.md`를 우선한다.

## 성능개선 우선순위

1. Runtime readiness cache
   - Gemma/OCR의 반복 health/model check 제거
   - 실패 시 invalidate
   - 작은 범위, 낮은 기능 회귀 위험
2. Translator stage reuse
   - stage-batched translation loop에서 `Translator`를 page마다 만들지 않는다.
   - source/target/translator/settings invariant가 같을 때만 재사용한다.
3. GPU-safe prewarm scheduling
   - Gemma prewarm이 inpaint/OCR의 GPU 작업과 싸우지 않도록 스케줄링한다.
   - GPU util 85% 이상 또는 VRAM 여유 2~3GB 미만이면 prewarm overlap을 피한다.
4. 성능 계측 보강
   - runtime probe 시간, model check 시간, stage gap, prewarm wait, per-stage duration을 남긴다.
   - 동시성 없이도 개선 전후를 비교할 수 있게 한다.

## 활성 계획에서 제외한 항목

- Gemma page concurrency 2 이상
- `LLAMA_N_PARALLEL > 1` 제품 기본값
- OCR page concurrency
- inpaint page concurrency
- async output/debug writer

위 항목은 현재 GPU 포화 환경에서는 빠를 가능성보다 자원 경합, latency 증가, OOM, timeout, 결과물 누락 리스크가 더 크다. 필요하면 `benchmarking/lab`에서만 별도 실험한다.

## 증거 로그

- `C:\path\to\comic-translate_validation_logs\2026-05-29\chore-auto-translation-performance-audit\performance_audit_extract.log`
- `C:\path\to\comic-translate_validation_logs\2026-05-29\chore-auto-translation-performance-audit\performance_audit_extract.json`
- `C:\path\to\comic-translate_validation_logs\2026-05-29\chore-auto-translation-performance-audit\ast_surface_auto_translation_performance.log`
- `C:\path\to\comic-translate_validation_logs\2026-05-29\chore-auto-translation-performance-audit\ast_callsite_auto_translation_performance_utf8sig.log`

## 관련 문서

- `01-parallelism-audit-ko.md`: 현재 병목과 speedup 계산
- `02-implementation-spec-ko.md`: AST 기반 코드 검토와 구현 명세
- `03-final-execution-plan-ko.md`: 동시성 제외 후 최종 실행 순서와 PR별 준비 명세
