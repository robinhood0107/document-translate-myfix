# 자동번역 성능개선 구현 명세

작성일: 2026-05-29

이 문서는 `01-parallelism-audit-ko.md`를 읽은 뒤, 자동번역 성능과 직접 관련된 코드 경로를 AST 기준으로 추적해 만든 구현 명세다. 바로 한 번에 구현하지 않고, 작은 PR로 쪼개서 검증한다.

## AST 검토 범위

AST 표면 추출 로그:

- `C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate_validation_logs\2026-05-29\chore-auto-translation-performance-audit\ast_surface_auto_translation_performance.log`
- `C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate_validation_logs\2026-05-29\chore-auto-translation-performance-audit\ast_callsite_auto_translation_performance_utf8sig.log`

UTF-8 BOM 허용 AST pass 기준 parse failure는 0개다.

직접 검토한 주요 표면:

| 영역 | 파일/함수 | 확인한 사실 |
| --- | --- | --- |
| stage-batched orchestration | `pipeline/stage_batched_processor.py` `StageBatchedProcessor` | prewarm executor, OCR/Gemma await, inpaint/translate/render loop, per-page failure isolation |
| legacy batch | `pipeline/batch_processor.py` `BatchProcessor.batch_process()` | OCR/translation/inpaint/export가 page loop 내부에서 순차 실행 |
| manual workflow | `app/controllers/manual_workflow.py` | 다중 페이지 manual OCR/translation도 loop 내부에서 `OCRProcessor.initialize()` 또는 `Translator(...)` 사용 |
| webtoon batch | `pipeline/webtoon_batch/chunk.py`, `flow.py`, `render.py` | virtual page streaming, OCR/translation/inpaint/render가 순차 처리되며 translation마다 `Translator(...)` 생성 |
| translation runtime | `modules/translation/processor.py`, `modules/translation/local_runtime.py`, `modules/translation/llm/custom_local_gemma.py` | `Translator.__init__()`이 Gemma `ensure_server()`, 실제 요청은 `requests.post()` |
| OCR runtime | `modules/ocr/processor.py`, `modules/ocr/local_runtime.py`, `modules/ocr/ocr_paddle_VL.py`, `modules/ocr/hunyuan_ocr.py` | `OCRProcessor.initialize()`가 `ensure_engine()`, OCR engine은 block-level thread pool 보유 |
| inpaint | `pipeline/inpainting.py`, `modules/inpainting/source_lama_blockwise.py` | source LaMa blockwise는 bbox clip 이후 block별 inpaint, 모델 자체 page concurrency는 고위험 |
| debug/output write | `modules/utils/inpaint_debug.py`, `modules/utils/automatic_output.py`, `imkit/io.py` | 이미지/JSON write가 동기 실행 |
| render/export | `pipeline/stage_batched_processor.py`, `pipeline/webtoon_batch/render.py`, `app/controllers/projects.py` | final render/write 경로는 Qt render 후 동기 이미지 write |

AST callsite 기준 성능 관련 주요 호출 수:

- `Translator(...)`: 6개 제품 callsite와 DeepL 내부 callsite 1개
- `OCRProcessor(...)`: 2개 제품 callsite
- `ensure_server(...)`: 3개
- `ensure_engine(...)`: 3개
- `export_inpaint_debug_artifacts(...)`: 2개
- `write_output_image(...)`: 5개
- `write_archive_image(...)`: 3개
- `imk.write_image(...)`: 8개
- `ThreadPoolExecutor(...)`: 4개
- `requests.post(...)`: 15개
- `urlopen(...)`: 6개

## 문제 정의

### P1. Runtime readiness probe 반복

현재 경로:

1. `StageBatchedProcessor._start_gemma_prewarm()`이 inpaint stage 시작 때 Gemma prewarm을 시작한다.
2. `StageBatchedProcessor._translate_all()`이 translation stage 시작 때 `_await_gemma_runtime()`으로 prewarm 결과를 기다린다.
3. 같은 함수의 page loop 안에서 매번 `Translator(self.main_page, ...)`를 생성한다.
4. `Translator.__init__()`은 Gemma translator일 때 다시 `runtime_manager.ensure_server(...)`를 호출한다.
5. `LocalGemmaRuntimeManager.ensure_server()`는 health probe와 models check를 다시 수행한다.

OCR도 같은 패턴이다. `StageBatchedProcessor._start_ocr_prewarm()`과 `_await_ocr_runtime()`이 있어도 `OCRProcessor.initialize()`는 local OCR이면 다시 `ensure_engine()`을 호출한다.

개선 원칙:

- runtime manager에 readiness cache를 둔다.
- cache key는 사용자 설정과 runtime 실행 조건을 포함한다.
- cache hit에서는 progress noise와 network probe를 생략한다.
- 실제 요청 실패, 설정 변경, model mismatch, compose restart, shutdown 시 invalidate한다.

### P2. Translator/OCR object lifetime이 page loop와 결합됨

stage-batched translation은 모든 페이지가 같은 source/target/tool 설정으로 도는 일반 케이스가 많다. 그런데 현재는 page마다 `Translator`를 새로 만들고 engine initialize도 반복한다.

개선 원칙:

- stage-batched에서 먼저 `Translator`를 stage 단위로 재사용한다.
- source_lang, target_lang, translator_key, Gemma settings, LLM extra context가 바뀌면 재사용하지 않는다.
- legacy/manual/webtoon은 별도 PR에서 다룬다. 이 경로들은 사용자 선택 페이지와 visible area, per-page state가 섞이므로 stage-batched보다 보수적으로 접근한다.

### P3. Translation concurrency는 앱과 llama.cpp 양쪽이 맞아야 함

`docker-compose.yaml`은 `LLAMA_N_PARALLEL`을 받지만 기본값은 1이고, 앱은 순차 요청이다. 서버 parallel slot만 올려도 앱이 한 번에 요청을 하나만 보내면 효과가 작다.

개선 원칙:

- `LLAMA_N_PARALLEL=2` 실험은 앱 bounded concurrency 2와 함께 한다.
- page 결과는 원래 page index 순서대로 commit한다.
- JSON parse 실패, retry, timeout, truncate, empty content metrics를 page별로 남긴다.
- 긴 페이지 또는 많은 block chunk는 concurrency 1로 자동 degrade할 수 있어야 한다.

### P4. OCR page concurrency는 기존 block concurrency와 곱해짐

PaddleOCR VL 기본 `parallel_workers`는 최대 8이고, Hunyuan OCR도 block-level thread pool을 사용한다. page concurrency 2를 단순히 얹으면 crop request가 최대 16개 이상 동시에 날아갈 수 있다.

개선 원칙:

- 총 in-flight OCR request budget을 둔다.
- `page_concurrency * block_workers <= global_budget`을 만족하도록 조정한다.
- GPU memory, GPU util, crop area p90, timeout을 benchmark에 기록한다.

### P5. Debug/output writer는 분리 가능하지만 결과물 보장이 먼저임

현재 cleaned image, detector overlay, raw mask, mask overlay, cleanup delta, debug metadata, final output image는 page loop에서 동기 저장된다.

개선 원칙:

- 모델 inference와 Qt render thread-safety는 건드리지 않고 write만 bounded queue로 분리한다.
- queue size는 1 또는 2로 시작한다. 이미지 array copy 비용과 메모리 사용량 때문이다.
- cancellation 시 queue drain 정책을 명확히 한다.
- write 실패는 page summary/report에 남긴다.
- 이미지 결과물 변경이므로 synthetic/edge/sample 이미지 산출물을 저장하고 사용자 검토를 요청한다.

## 구현 PR 계획

### PR 1: Gemma/OCR readiness cache

Branch: `perf/runtime-readiness-cache`

수정 대상:

- `modules/translation/local_runtime.py`
- `modules/ocr/local_runtime.py`
- 필요 시 `modules/translation/processor.py`, `modules/ocr/processor.py`

명세:

- `LocalGemmaRuntimeManager`에 `_ready_key`, `_ready_at`, `_ready_generation` 또는 동등한 cache state를 추가한다.
- Gemma ready key에는 normalized endpoint, model name, managed/custom mode, compose file path, container URL, relevant env를 포함한다.
- `ensure_server(..., force_probe=False)` 또는 내부 helper로 cache hit를 처리한다.
- cache hit는 health/model probe와 progress emit을 생략한다.
- `_validate_loaded_model()` 실패, connection error, compose up, shutdown, credential change는 cache를 invalidate한다.
- `LocalOCRRuntimeManager`도 engine key, normalized server URL, managed/custom mode, compose file path를 key로 ready cache를 둔다.
- cancellation 중에는 cache hit라도 cancellation check를 먼저 수행한다.

테스트:

- `tests/test_local_gemma_runtime.py`
  - 같은 config에서 첫 `ensure_server()`만 `_wait_for_any_probe`와 `_validate_loaded_model`을 호출한다.
  - model name 변경 시 cache miss가 된다.
  - invalidate 후 재확인한다.
  - custom URL도 cache key에 포함된다.
- `tests/test_local_ocr_runtime.py`
  - 같은 managed engine에서 repeated `ensure_engine()`이 `_probe_health_state`를 반복하지 않는다.
  - engine key 또는 URL 변경 시 cache miss가 된다.
  - loading 상태는 ready cache에 넣지 않는다.

검증:

- `.venv-win/Scripts/python.exe -m pytest tests/test_local_gemma_runtime.py tests/test_local_ocr_runtime.py`
- `.venv-win/Scripts/python.exe scripts/validate_changed_python.py --all`
- `.venv-win/Scripts/python.exe scripts/compile_translations.py --check`

### PR 2: stage-batched Translator reuse

Branch: `perf/stage-batched-translator-reuse`

수정 대상:

- `pipeline/stage_batched_processor.py`
- 필요 시 `modules/translation/processor.py`

명세:

- `_translate_all()` 시작 시 reusable translator를 한 번 만든다.
- reuse key는 source language, target language, translator key, selected LLM/Gemma settings signature를 포함한다.
- 모든 ctx가 같은 key일 때만 stage-wide reuse한다.
- page별 cache hit에서도 translator metrics 접근을 유지한다.
- page별 translation failure는 기존처럼 해당 page만 failed로 표시한다.
- `runtime_manager.shutdown()` 위치는 현행 유지하되, reuse object가 shutdown 이후 요청하지 않도록 한다.

테스트:

- `tests/test_stage_batched_translation_reuse.py`
  - 동일 language/settings의 N pages에서 `Translator` 생성이 1회인지 확인한다.
  - source/target이 다른 page가 섞이면 page별 생성으로 fallback한다.
  - translation cache hit page도 stage metrics/report가 깨지지 않는다.
  - 한 page 실패가 다음 page 처리를 막지 않는다.

검증:

- 위 신규 테스트
- `tests/test_stage_batched_cancel.py`
- `tests/test_batch_report_runtime.py`
- validation/translation check

### PR 3: Gemma bounded concurrency 실험

Branch: `perf/gemma-bounded-concurrency`

기본값: off

수정 대상:

- `pipeline/stage_batched_processor.py`
- `modules/translation/llm/custom_local_gemma.py`
- settings/pipeline config는 최소 범위로 추가

명세:

- 설정값 예: `translation_page_concurrency`, default 1.
- concurrency 1이면 현행 동작과 동일해야 한다.
- concurrency > 1이면 page 작업을 bounded executor로 실행하되, commit은 page index 순서로 한다.
- 각 page result에는 translator stats, cache status, exception detail을 포함한다.
- retry/chunk split 로직은 engine 내부 기존 로직을 그대로 사용한다.
- `LLAMA_N_PARALLEL`, request timeout, token stats를 benchmark event에 남긴다.

테스트:

- mock translator로 concurrency 2일 때 completion order가 뒤섞여도 page state commit 순서가 안정적인지 확인한다.
- 한 page exception이 batch abort가 아니라 해당 page failed인지 확인한다.
- cancellation 시 pending future가 취소되는지 확인한다.

사용자 검토:

- 실제 테스트 이미지/페이지 subset으로 translation JSON과 최종 이미지 샘플을 validation log에 저장한다.
- 병합 전 사용자에게 결과물 검토를 요청한다.

### PR 4: async debug/output writer

Branch: `perf/async-output-writer`

기본값: off 또는 debug-output 전용 opt-in

수정 대상:

- 새 모듈 후보: `modules/utils/async_output_writer.py`
- `pipeline/stage_batched_processor.py`
- `pipeline/batch_processor.py`
- `pipeline/webtoon_batch/render.py`
- `modules/utils/inpaint_debug.py`
- `modules/utils/automatic_output.py`

명세:

- writer는 `ThreadPoolExecutor(max_workers=1)` 또는 dedicated worker thread 하나로 시작한다.
- task는 path, image copy, writer callable, metadata를 가진다.
- queue size는 1 또는 2로 제한한다.
- submit이 막히면 backpressure로 page loop가 기다린다. 무제한 memory growth는 금지한다.
- batch 완료/취소/오류 시 `drain()` 또는 `cancel_pending()` 정책을 명시한다.
- write failure는 page summary와 batch report에 남긴다.

테스트:

- synthetic RGB image write, raw mask write, metadata JSON write가 모두 완료되는지 확인한다.
- writer failure가 조용히 묻히지 않는지 확인한다.
- cancellation path에서 pending task가 처리/취소 정책대로 남는지 확인한다.

사용자 검토:

- synthetic, boundary, 실제 샘플 이미지를 각각 저장한다.
- 기존 동기 write 결과와 async writer 결과의 파일 수, 크기, pixel equality 또는 허용 차이를 비교한다.
- 결과물 경로를 PR에 남기고 사용자 검토를 요청한다.

### PR 5: OCR page concurrency 실험

Branch: `perf/ocr-page-concurrency`

기본값: off

수정 대상:

- `pipeline/stage_batched_processor.py`
- `modules/ocr/processor.py` 또는 새 orchestration helper
- 설정/benchmark event 최소 추가

명세:

- `ocr_page_concurrency` default 1.
- `global_ocr_request_budget`을 둔다.
- PaddleOCR VL/Hunyuan `parallel_workers`와 page concurrency의 곱이 budget을 넘지 않게 한다.
- OCR cache hit page는 executor에 넣지 않거나 즉시 완료 처리한다.
- page 결과 commit은 원래 page 순서를 보장한다.

테스트:

- mock OCR engine으로 concurrency budget이 지켜지는지 확인한다.
- OCR cache hit/miss 혼합에서 state와 benchmark event가 유지되는지 확인한다.
- 한 page OCR 실패가 다른 page를 막지 않는지 확인한다.

사용자 검토:

- OCR JSON/텍스트 결과가 concurrency 1과 2에서 동일한지 샘플 비교한다.
- OCR crop/debug output이 있으면 결과물 경로를 사용자에게 전달한다.

### PR 6: inpaint concurrency benchmark-only

Branch: `benchmarking/lab` 전용 실험 또는 별도 benchmark branch 후 `benchmarking/lab` PR

명세:

- 제품 기본값 변경 금지.
- page concurrency 2가 단일 GPU에서 실제로 빠른지 확인한다.
- VRAM peak, GPU util, OOM/retry, image quality, output equality를 기록한다.
- benchmark 결과 없이 `develop` 제품 PR로 승격하지 않는다.

## 금지선

- readiness cache와 object reuse PR에서 output image, OCR text, translation text를 바꾸지 않는다.
- async writer 전에는 Qt render 작업을 worker로 옮기지 않는다.
- concurrency PR에서 retry/report/cancel semantics를 단순화하지 않는다.
- benchmark-only asset을 `develop` 제품 PR에 섞지 않는다.
- 검증 산출물을 repo에 커밋하지 않는다.

## 성공 기준

- Micro PR 후 반복 Gemma/OCR runtime progress 로그가 같은 batch/config 안에서 사라진다.
- repeated ensure 테스트가 실제 probe 호출 횟수를 검증한다.
- stage-batched Translator reuse 후 기존 batch report와 page summaries가 동일하게 남는다.
- concurrency default 1에서 기존 테스트와 산출물이 동일하다.
- default-off 실험에서 before/after benchmark와 사용자 검토 산출물이 남는다.
