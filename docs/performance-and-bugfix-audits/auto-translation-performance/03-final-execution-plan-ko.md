# 자동번역 성능개선 최종 실행 명세서

이 문서는 `auto-translation-performance` 작업을 실제 PR로 실행하기 직전의 최종 명세서다. `00-truth-specification-ko.md`의 결정 사항을 따른다.

## 최종 결정

- 이번 제품 성능개선에서 page concurrency는 도입하지 않는다.
- 사용자 환경에서는 Gemma 실행 중 GPU util 85% 이상 또는 VRAM 여유 2~3GB 미만 조건에 거의 항상 도달한다.
- 이 조건에서 Gemma/OCR/inpaint page concurrency는 속도 개선보다 latency 증가, OOM, timeout, retry 증가, 결과물 누락 위험이 더 크다.
- 따라서 성능개선은 반복 readiness probe 제거, 객체 재사용, 인페인터 해제 후 GPU handoff, 계측 보강만 먼저 진행한다.
- Gemma prewarm은 모든 인페인트 결과를 확정하고 인페인터 VRAM 감소를 확인한 뒤 시작한다.

## 목표

- 같은 batch/config 안에서 Gemma/OCR runtime readiness 확인을 반복하지 않는다.
- stage-batched translation에서 page마다 `Translator`를 새로 만들지 않는다.
- 인페인터 모델을 결과물과 분리해 해제하고 실제 VRAM 반환을 확인한 뒤 Gemma를 시작한다.
- 취소 또는 종료 뒤 queued/running prewarm이 늦게 runtime을 시작하거나 ready를 보고하지 못하게 한다.
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
- `<sample-input-root>/japan/094.png`, `095.png`, `096.png` 실제 path를 쓰는 mocked pytest에서 3개 page 반복 초기화의 readiness probe 호출이 1회로 제한되는지 검증한다.

검증:

- `.venv-win/Scripts/python.exe -m pytest tests/test_local_gemma_runtime.py tests/test_local_ocr_runtime.py tests/test_runtime_readiness_cache_sample_japan.py tests/test_batch_report_runtime.py`
- `.venv-win/Scripts/python.exe scripts/validate_changed_python.py --all`
- `.venv-win/Scripts/python.exe scripts/compile_translations.py --check`

실제 GPU 전과정 측정:

- PR 1은 `<sample-input-root>/japan` 22장 전체를 before/after로 모두 실행한다.
- before log: `C:\path\to\comic-translate_validation_logs\2026-05-29\auto-translation-performance\chore-runtime-readiness-cache\before\`
- after log: `C:\path\to\comic-translate_validation_logs\2026-05-29\auto-translation-performance\chore-runtime-readiness-cache\after\`
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

## PR 3. GPU Runtime Handoff

- Branch: `chore/gpu-runtime-handoff`
- Base: `develop`
- Issue title: `Release inpainter GPU resources before starting Gemma`
- 목적: 인페인트 결과를 모두 보존한 상태에서 인페인터 VRAM을 실제 반환한 뒤 Gemma를 시작한다.

수정 후보:

- `pipeline/stage_batched_processor.py`
- `pipeline/inpainting.py`
- `modules/inpainting/source_lama_blockwise.py`
- `modules/utils/gpu_handoff.py`
- `modules/utils/gpu_metrics.py`

구현 원칙:

- 모든 페이지의 inpaint image, mask, patch, debug 산출물을 먼저 저장한다.
- 전체 model cache가 아니라 인페인터 handler와 Source LaMa의 model/session 참조만 해제한다.
- 해제 대상 CUDA tensor 저장공간을 계산하고 그 90% 이상에 해당하는 process allocated 감소를 확인한다.
- PyTorch가 추적하지 않는 GPU backend는 현재 PID와 정확한 GPU UUID의 사용량이 인페인터 로드 전 기준선으로 돌아왔는지 확인한다.
- 해제가 예상되는데 측정 불가 또는 timeout이면 Gemma 시작을 차단한다.
- inpaint 중 취소·예외에서도 targeted release를 실행하되 Gemma는 시작하지 않는다.
- cancellation/shutdown은 내부 cancel event를 세우고 queued future를 취소한 뒤 running future 종료를 기다린다.
- Docker 시작과 HTTP 예열은 cancel-aware bounded I/O로 실행한다. 취소·시간초과 시 Windows Docker CLI process tree 또는 POSIX process group을 종료하고, stop 성공은 exact container inspect로 확인한다. stop 실패 시 활성 상태를 보존해 stage transition에서 즉시 한 번, final batch cleanup에서도 한 번 재시도하며, 다음 batch의 OCR prewarm 전에 잔존 OCR/Gemma cleanup preflight를 강제한다.
- `inpainter_release` telemetry에 측정 전후, 감소량, evidence source, gate status, elapsed를 남긴다.
- 결과 이미지, OCR 텍스트, 번역 텍스트는 바뀌면 안 된다.

테스트:

- model-sized process allocator release 관측, 작은 우연한 감소, timeout 테스트
- 다른 프로세스와 다른 GPU의 driver 감소가 gate를 통과하지 않는 테스트
- non-PyTorch GPU allocation의 PID·GPU UUID 기준선 복귀 테스트
- measurement unavailable이면 Gemma 시작을 차단하는 테스트
- materialized image, mask, patch, edit mask, debug PNG, 최종 고정 번역 PNG의 SHA-256 보존 테스트
- inpaint 중 취소의 targeted release 테스트
- blocked Docker/HTTP 취소, Docker process tree 종료, queued future cancel, running future 종료 동기화 테스트
- stop 실패 상태 보존, exact inspect 확인, stage transition 및 final cleanup 재시도, 다음 batch startup cleanup gate 테스트

검증:

- `.venv-win/Scripts/python.exe -m pytest tests/test_gpu_handoff.py tests/test_gpu_metrics.py tests/test_inpainter_release.py tests/test_source_lama_blockwise.py tests/test_llama_cpp_runtime_policy.py tests/test_local_gemma_runtime.py tests/test_local_ocr_runtime.py tests/test_stage_batched_cancel.py`
- `.venv-win/Scripts/python.exe scripts/validate_changed_python.py --all`
- `.venv-win/Scripts/python.exe scripts/compile_translations.py --check`

사용자 검토:

- 결과 이미지가 바뀌는 PR이 아니므로 사용자 이미지 검토는 필요 없다.
- repo 밖 실제 4페이지 후보 로그에서 `after-release`, 75%, 0% 후보의 시간, VRAM, shared GPU memory, WSL swap, 인페인트 p95를 보고한다.

실제 후보 결과:

| 후보 | 결과 |
| --- | --- |
| `after-release` | 제품 정책 유지. 최종 CUDA13 smoke에서 대상 storage 389.537MiB, gate 350.583MiB, process allocated 392.196MiB 감소를 0.062초에 관측 |
| 75% 시작 | 약 5% 속도 개선은 재현됐지만 GPU 여유가 절반 이하로 줄고 shared GPU memory와 WSL swap 압력이 증가해 탈락 |
| 0% 시작 | 기준선보다 느리고 인페인트 p95가 크게 악화되어 탈락 |

검증 로그는 `<validation-log-root>\gpu-runtime-handoff\decision-20260728.md`에 남긴다.

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
C:\path\to\comic-translate_validation_logs\2026-05-29\auto-translation-performance\<branch-slug>\
```

각 PR 본문에는 실행 명령, pass/fail 요약, 로그 파일 경로를 남긴다. 로그와 테스트 산출물은 커밋하지 않는다.

## 실행 시작 조건

- 이 문서 PR이 `develop`에 병합되어야 한다.
- `develop`이 `origin/develop`과 일치해야 한다.
- 단기 브랜치는 병합 여부를 확인한 뒤 정리해야 한다.
- 첫 구현 브랜치는 반드시 `develop`에서 `chore/runtime-readiness-cache`로 분기한다.
