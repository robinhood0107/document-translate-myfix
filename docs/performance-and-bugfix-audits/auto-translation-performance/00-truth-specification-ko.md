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
- 단일 GPU에서 OCR, inpaint, Gemma를 동시에 강하게 밀어붙이는 방식은 benchmark 없이 제품 기본값으로 넣지 않는다.

## 불변 조건

- 결과 이미지, OCR 텍스트, 번역 텍스트, inpaint patch, project save/load 상태가 바뀌는 성능개선은 반드시 회귀 테스트와 샘플 산출물을 남긴다.
- 사용자에게 보이는 이미지/결과물 품질이 바뀌면 병합 전 사용자 검토를 요청한다.
- 기본값은 보수적으로 둔다. 새 병렬화 옵션은 먼저 feature flag 또는 설정값 default-off로 실험한다.
- cancellation은 실패나 skip으로 기록하지 않는다.
- external runtime startup/probe cache는 연결 오류, HTTP 오류, 모델 불일치, 설정 변경 시 invalidate할 수 있어야 한다.
- Qt render/layout 객체를 임의 worker thread로 옮기지 않는다.
- `main`, `develop`, `benchmarking/lab` 장기 브랜치 정책과 `rules.md`를 우선한다.

## 성능개선 우선순위

1. Runtime readiness cache
   - Gemma/OCR의 반복 health/model check 제거
   - 실패 시 invalidate
   - 작은 범위, 낮은 기능 회귀 위험
2. Translator stage reuse
   - stage-batched translation loop에서 `Translator`를 page마다 만들지 않는다.
   - source/target/translator/settings invariant가 같을 때만 재사용한다.
3. Bounded Gemma concurrency 실험
   - 앱 동시 요청 수와 `LLAMA_N_PARALLEL`을 함께 기록한다.
   - 품질/JSON 안정성/VRAM/tok-per-sec를 함께 본다.
4. Async debug/output writer
   - 모델 inference는 먼저 유지하고 이미지/JSON write만 bounded queue로 분리한다.
   - queue drain, cancel, failure report를 먼저 설계한다.
5. OCR page concurrency 실험
   - 기존 block worker와 곱해지는 총 in-flight 요청 수를 제한한다.
6. Inpaint page concurrency
   - 제품 변경 전 benchmark-only로 검증한다.

## 증거 로그

- `C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate_validation_logs\2026-05-29\chore-auto-translation-performance-audit\performance_audit_extract.log`
- `C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate_validation_logs\2026-05-29\chore-auto-translation-performance-audit\performance_audit_extract.json`
- `C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate_validation_logs\2026-05-29\chore-auto-translation-performance-audit\ast_surface_auto_translation_performance.log`
- `C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate_validation_logs\2026-05-29\chore-auto-translation-performance-audit\ast_callsite_auto_translation_performance_utf8sig.log`

## 관련 문서

- `01-parallelism-audit-ko.md`: 현재 병목과 speedup 계산
- `02-implementation-spec-ko.md`: AST 기반 코드 검토와 구현 명세
