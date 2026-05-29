# 자동번역 성능개선 최종 실행 명세서

이 문서는 `auto-translation-performance` 작업을 실제 PR로 실행하기 직전의 최종 명세서다. `00-truth-specification-ko.md`의 결정 사항을 따른다.

## 최종 결정

- 이번 제품 성능개선에서 page concurrency는 도입하지 않는다.
- 사용자 환경에서는 Gemma 실행 중 GPU util 85% 이상 또는 VRAM 여유 2~3GB 미만 조건에 거의 항상 도달한다.
- 이 조건에서 Gemma/OCR/inpaint page concurrency는 속도 개선보다 latency 증가, OOM, timeout, retry 증가, 결과물 누락 위험이 더 크다.
- 따라서 성능개선은 반복 readiness probe 제거, 객체 재사용, GPU-safe prewarm, 계측 보강만 먼저 진행한다.

## 목표

- 같은 batch/config 안에서 Gemma/OCR runtime readiness 확인을 반복하지 않는다.
- stage-batched translation에서 page마다 `Translator`를 새로 만들지 않는다.
- GPU가 바쁠 때 Gemma prewarm이 OCR/inpaint와 불필요하게 겹치지 않도록 한다.
- 다음 성능개선 판단을 위해 runtime probe, model check, prewarm wait, stage gap, page duration을 일관되게 기록한다.

## 비목표와 금지선

- `LLAMA_N_PARALLEL > 1`을 develop 제품 기본값으로 올리지 않는다.
- Gemma page concurrency를 도입하지 않는다.
- OCR page concurrency를 도입하지 않는다.
- inpaint page concurrency를 develop 제품 PR에 넣지 않는다.
- async output/debug writer는 I/O 병목이 별도로 입증되기 전까지 도입하지 않는다.
- Qt render/layout 객체를 worker thread로 옮기지 않는다.
- 결과 이미지, OCR 텍스트, 번역 텍스트가 바뀌는 변경은 이 계획에 포함하지 않는다.

## 실행 준비 순서

1. 현재 문서 PR을 `develop`에 병합한다.
2. `develop`을 `origin/develop`으로 fast-forward 갱신한다.
3. 병합 완료된 단기 브랜치를 로컬/원격에서 정리한다.
4. `develop`에서 첫 구현 브랜치 `chore/runtime-readiness-cache`를 새로 만든다.
5. 이후 각 PR은 아래 순서대로 하나씩 만들고, 병합 후 다음 브랜치를 만든다.

장기 브랜치 `main`, `develop`, `benchmarking/lab`은 절대 삭제하지 않는다.

## PR 1. Runtime Readiness Cache

- Branch: `chore/runtime-readiness-cache`
- Base: `develop`
- Issue title: `Avoid repeated Gemma and OCR runtime readiness probes within same batch config`
- 목적: 같은 batch/config에서 반복되는 Gemma/OCR health/model check를 제거한다.

수정 후보:

- `modules/translation/local_runtime.py`
- `modules/ocr/local_runtime.py`

구현 원칙:

- `Translator`와 `OCRProcessor.initialize()` 호출 구조는 이번 PR에서 바꾸지 않는다.
- Gemma cache key는 normalized API base URL, configured model name, managed/unmanaged mode로 만든다.
- OCR cache key는 OCR engine key, normalized server URL, managed mode로 만든다.
- cache hit이면 실제 health/model HTTP probe를 생략하고 `readiness_cache_hit=True` progress event를 남긴다.
- connection error, HTTP error, model mismatch, timeout, cancel, 설정 변경 시 cache를 seed하지 않거나 invalidate한다.
- `shutdown()`과 active OCR engine stop은 readiness cache를 clear한다.
- 결과 이미지, OCR 텍스트, 번역 텍스트는 바뀌면 안 된다.

테스트:

- Gemma first ensure는 health/model check를 수행하고, second ensure는 같은 key에서 둘 다 생략한다.
- Gemma model mismatch 또는 connection failure 후 다음 ensure가 cache hit 하지 않는다.
- OCR first ensure는 health check를 수행하고, second ensure는 같은 engine/url에서 health check를 생략한다.
- OCR engine 또는 URL 변경 시 cache miss가 발생한다.
- cancel 중에는 successful readiness cache가 기록되지 않는다.
- `Sample/japan/094.png`, `095.png`, `096.png` 실제 path를 쓰는 mocked pytest에서 3개 page 반복 초기화의 readiness probe 호출이 1회로 제한되는지 검증한다.

검증:

- `.venv-win/Scripts/python.exe -m pytest tests/test_local_gemma_runtime.py tests/test_local_ocr_runtime.py tests/test_runtime_readiness_cache_sample_japan.py tests/test_batch_report_runtime.py`
- `.venv-win/Scripts/python.exe scripts/validate_changed_python.py --all`
- `.venv-win/Scripts/python.exe scripts/compile_translations.py --check`

실제 GPU 전과정 측정:

- PR 1은 `Sample/japan` 22장 전체를 before/after로 모두 실행한다.
- before log: `C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate_validation_logs\2026-05-29\auto-translation-performance\chore-runtime-readiness-cache\before\`
- after log: `C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate_validation_logs\2026-05-29\auto-translation-performance\chore-runtime-readiness-cache\after\`
- 각 run은 시작/종료 시각, wall-clock elapsed time, stage별 event, output root, output file count, skip/fail reason, Gemma/OCR readiness event count, readiness cache hit count를 남긴다.
- 실제 GPU runtime이 준비되지 않거나 22장 전과정이 실패하면 PR 생성 전에 멈추고 `blocked-runtime-unavailable` 또는 실패 reason을 보고한다.
- after가 before보다 5% 이상 느려지면 원인 분석 후 PR 전 사용자에게 보고한다.

사용자 검토:

- 결과물 이미지가 바뀌는 PR이 아니므로 사용자 이미지 검토는 필요 없다.
- 단, 실제 로그에서 페이지마다 Gemma/OCR 확인 문구가 사라졌는지는 확인 결과를 보고한다.

## PR 2. Stage-Batched Translator Reuse

- Branch: `chore/stage-batched-translator-reuse`
- Base: `develop`
- Issue title: `Reuse Translator during stage-batched translation when config is invariant`
- 목적: stage-batched translation loop에서 page마다 `Translator`를 새로 만들지 않는다.

수정 후보:

- `pipeline/stage_batched_processor.py`
- `modules/translation/processor.py`

구현 원칙:

- source language, target language, translator key, Gemma settings, extra context가 동일한 stage 안에서만 재사용한다.
- page별 block input과 page summary/report는 현행과 동일하게 유지한다.
- 한 page 실패가 다른 page의 report/cancellation semantics를 바꾸면 안 된다.
- cache/reuse 상태는 benchmark event 또는 debug log에만 남긴다.

테스트:

- stage-batched translation에서 `Translator` 생성 횟수가 page 수가 아니라 stage당 1회인지 검증
- page별 translation 결과 commit 순서 유지 테스트
- 한 page exception/report/cancel semantics 유지 테스트

검증:

- `.venv-win/Scripts/python.exe -m pytest <new-translator-reuse-tests> tests/test_batch_report_runtime.py tests/test_stage_batched_cancel.py`
- `.venv-win/Scripts/python.exe scripts/validate_changed_python.py --all`
- `.venv-win/Scripts/python.exe scripts/compile_translations.py --check`

사용자 검토:

- 결과 이미지가 바뀌는 PR이 아니므로 사용자 이미지 검토는 필요 없다.
- 실제 run 로그에서 페이지별 Gemma readiness 반복 문구가 사라졌는지 보고한다.

## PR 3. GPU-Safe Prewarm Scheduling

- Branch: `chore/gpu-safe-prewarm-scheduling`
- Base: `develop`
- Issue title: `Avoid Gemma prewarm overlap when GPU is already saturated`
- 목적: GPU가 포화 상태일 때 Gemma prewarm이 OCR/inpaint 작업과 싸우지 않도록 한다.

수정 후보:

- `pipeline/stage_batched_processor.py`
- `modules/utils/gpu_metrics.py`
- 필요 시 runtime event payload

구현 원칙:

- Gemma prewarm 시작 전에 GPU snapshot을 읽는다.
- GPU util 85% 이상 또는 VRAM free 3072MB 미만이면 OCR/inpaint 중 prewarm overlap을 피한다.
- prewarm을 생략하거나 미룬 경우 reason, GPU snapshot, wait duration을 event에 남긴다.
- GPU metrics를 얻을 수 없으면 현행 prewarm 정책을 유지하고 `gpu_metrics_unavailable` reason을 남긴다.
- cancellation 중에는 새 prewarm 작업을 시작하지 않는다.
- 결과 이미지, OCR 텍스트, 번역 텍스트는 바뀌면 안 된다.

테스트:

- saturated snapshot이면 prewarm executor submit이 발생하지 않는 테스트
- 충분한 GPU headroom이면 기존 prewarm submit이 유지되는 테스트
- GPU metrics unavailable이면 기존 동작 유지 테스트
- cancellation 상태에서 prewarm scheduling이 시작되지 않는 테스트

검증:

- `.venv-win/Scripts/python.exe -m pytest <new-prewarm-tests> tests/test_stage_batched_cancel.py`
- `.venv-win/Scripts/python.exe scripts/validate_changed_python.py --all`
- `.venv-win/Scripts/python.exe scripts/compile_translations.py --check`

사용자 검토:

- 결과 이미지가 바뀌는 PR이 아니므로 사용자 이미지 검토는 필요 없다.
- 실제 GPU 포화 로그에서 prewarm skip/defer reason이 기록되는지 보고한다.

## PR 4. Performance Telemetry Cleanup

- Branch: `chore/auto-translation-telemetry`
- Base: `develop`
- Issue title: `Record auto-translation runtime probe and stage timing metrics consistently`
- 목적: 다음 성능개선 판단을 위해 stage별 시간과 runtime probe 시간을 일관되게 남긴다.

수정 후보:

- `pipeline/stage_batched_processor.py`
- `pipeline/batch_processor.py`
- `modules/translation/local_runtime.py`
- `modules/ocr/local_runtime.py`
- batch report 또는 benchmark event helper

구현 원칙:

- runtime probe duration, model check duration, readiness cache hit/miss, prewarm wait, skipped prewarm reason, stage gap, page duration을 optional field로 남긴다.
- 기존 report schema와 호환되게 한다.
- telemetry 누락은 기능 실패로 처리하지 않는다.
- 결과물과 모델 설정은 바꾸지 않는다.

테스트:

- mock clock으로 duration field가 남는지 검증
- optional field가 없는 기존 report를 읽는 경로가 유지되는지 검증
- stage-batched/batch report regression 테스트

검증:

- `.venv-win/Scripts/python.exe -m pytest <new-telemetry-tests> tests/test_batch_report_runtime.py`
- `.venv-win/Scripts/python.exe scripts/validate_changed_python.py --all`
- `.venv-win/Scripts/python.exe scripts/compile_translations.py --check`

사용자 검토:

- 결과 이미지가 바뀌는 PR이 아니므로 사용자 이미지 검토는 필요 없다.
- before/after 로그 요약을 보고한다.

## 보류 항목

아래 항목은 현재 제품 계획에서 제외한다.

- Gemma concurrency 2 이상
- `LLAMA_N_PARALLEL > 1`
- OCR page concurrency
- inpaint page concurrency
- async output/debug writer

이 항목들은 GPU headroom이 충분하거나 telemetry로 별도 병목이 입증된 뒤에만 다시 검토한다. inpaint concurrency는 제품 브랜치가 아니라 `benchmarking/lab`에서만 실험한다.

## 로그 규칙

각 PR의 검증 로그는 repo 밖에 저장한다.

```text
C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate_validation_logs\2026-05-29\auto-translation-performance\<branch-slug>\
```

각 PR 본문에는 실행 명령, pass/fail 요약, 로그 파일 경로를 남긴다. 로그와 테스트 산출물은 커밋하지 않는다.

## 실행 시작 조건

- 이 문서 PR이 `develop`에 병합되어야 한다.
- `develop`이 `origin/develop`과 일치해야 한다.
- 단기 브랜치는 병합 여부를 확인한 뒤 정리해야 한다.
- 첫 구현 브랜치는 반드시 `develop`에서 `chore/runtime-readiness-cache`로 분기한다.
