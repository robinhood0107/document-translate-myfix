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
- 2026-07-28 실제 4페이지 offscreen 후보 비교에서 Gemma를 인페인트 시작과 동시에 올린 후보는 안전 기준선보다 느렸고 인페인트 p95가 크게 악화됐다. 75% 시점 후보는 약 5% 빨랐지만 GPU 여유가 절반 이하로 줄고 shared GPU memory와 WSL swap 압력이 반복 증가했다.
- 제품 정책은 모든 페이지의 인페인트 이미지, mask, patch, debug 산출물을 먼저 확정한 뒤 인페인터 모델만 해제하고 Gemma를 시작하는 `after-release`다.
- GPU 해제가 예상되면 해제 대상 CUDA tensor 저장공간의 90% 이상에 해당하는 현재 프로세스 allocator 감소를 요구한다. PyTorch가 추적하지 않는 GPU 할당은 같은 PID와 정확한 GPU UUID가 인페인터 로드 전 기준선으로 돌아왔을 때만 통과한다.
- 측정 불가, 해제 대상 크기 미상, 다른 프로세스 또는 다른 GPU의 감소, 제한 시간 내 감소 미관측 상태에서는 Gemma 시작을 차단한다.
- prewarm 종료 시 내부 cancel event를 먼저 세우고 queued future를 취소한 뒤 running future 종료를 기다린다. Docker 명령과 HTTP 예열도 취소 가능하고 시간 제한이 있으며, stop 실패 시 활성 상태를 보존해 재시도한다.
- `IQ4_XS + contextual-grouped + chunk 7 + no-spec + F16` 후보는 번역 속도 게이트를 통과했지만 292행 blind 의미 품질 게이트에서 탈락했다. 제품 기본 프로필은 `IQ4_NL + contextual-single + chunk 6 + no-spec + F16`으로 유지한다.
- `contextual-grouped` 제품 경로는 퇴역했다. strict JSON decoder, 불변 request context, HTTP 오류 분류, contextual-single 재시도, 논리 요청/HTTP telemetry와 번역 캐시는 유지한다.
- grouped 전체 22페이지 파이프라인 비교는 미완료가 아니라 선행 품질 게이트에 따른 정상 실행 취소다.
- PaddleOCR-VL 관리형 폴더 처리에는 exact 영구 결과 캐시를 둔다. 캐시는 crop 이미지를 저장하지 않고 사전 적용 전 raw OCR 결과와 진단만 저장한다.
- 영구 OCR 캐시는 공식 digest로 고정된 관리형 PaddleOCR-VL 런타임에서만 사용한다. 사용자 지정 endpoint는 신뢰 가능한 runtime identity가 없으므로 캐시 없이 정상 처리한다.
- exact OCR 캐시의 all-hit 경로는 Paddle runtime 시작과 OCR HTTP 요청을 모두 생략한다. sampled-image 또는 fuzzy-coordinate 캐시는 PaddleOCR-VL 자동 폴더 처리에 사용하지 않는다.

## 불변 조건

- 결과 이미지, OCR 텍스트, 번역 텍스트, inpaint patch, project save/load 상태가 바뀌는 성능개선은 반드시 회귀 테스트와 샘플 산출물을 남긴다.
- 사용자에게 보이는 이미지/결과물 품질이 바뀌면 병합 전 사용자 검토를 요청한다.
- 기본값은 보수적으로 둔다. 새 병렬화 또는 출력 변경 옵션은 먼저 feature flag 또는 설정값 default-off로 실험한다.
- exact content hash와 완전한 runtime identity로 출력 동일성이 증명된 결과 캐시는 fail-open, 명시적 clear/export, 보존 한도를 갖춘 경우에만 기본 활성화할 수 있다.
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
   - 인페인트 결과를 모두 materialize한 뒤 인페인터 전용 release를 실행한다.
   - 실제 VRAM 감소가 확인된 뒤에만 Gemma prewarm을 시작한다.
   - 0%와 75% overlap 후보는 자원 경합 게이트에서 탈락했으므로 제품 기본값으로 두지 않는다.
4. 성능 계측 보강
   - runtime probe 시간, model check 시간, stage gap, prewarm wait, per-stage duration을 남긴다.
   - 동시성 없이도 개선 전후를 비교할 수 있게 한다.
5. Exact 반복 실행 캐시
   - 번역은 SQLite result cache와 승인형 Exact TM을 유지한다.
   - OCR은 관리형 PaddleOCR-VL exact crop result cache를 사용한다.
   - 후속 프로젝트 checkpoint는 detection, OCR, inpaint, render를 stage fingerprint와 content-addressed object로 복원하되 손상·누락 시 정상 계산으로 fail-open한다.

## 활성 계획에서 제외한 항목

- Gemma page concurrency 2 이상
- `LLAMA_N_PARALLEL > 1` 제품 기본값
- OCR page concurrency
- inpaint page concurrency
- 인페인트 완료 전 Gemma prewarm overlap
- async output/debug writer

위 항목은 현재 GPU 포화 환경에서는 빠를 가능성보다 자원 경합, latency 증가, OOM, timeout, 결과물 누락 리스크가 더 크다. 필요하면 `benchmarking/lab`에서만 별도 실험한다.

## 증거 로그

- `C:\path\to\comic-translate_validation_logs\2026-05-29\chore-auto-translation-performance-audit\performance_audit_extract.log`
- `C:\path\to\comic-translate_validation_logs\2026-05-29\chore-auto-translation-performance-audit\performance_audit_extract.json`
- `C:\path\to\comic-translate_validation_logs\2026-05-29\chore-auto-translation-performance-audit\ast_surface_auto_translation_performance.log`
- `C:\path\to\comic-translate_validation_logs\2026-05-29\chore-auto-translation-performance-audit\ast_callsite_auto_translation_performance_utf8sig.log`
- `<validation-log-root>\gpu-runtime-handoff\decision-20260728.md`

## 관련 문서

- `01-parallelism-audit-ko.md`: 현재 병목과 speedup 계산
- `02-implementation-spec-ko.md`: AST 기반 코드 검토와 구현 명세
- `03-final-execution-plan-ko.md`: 동시성 제외 후 최종 실행 순서와 PR별 준비 명세
- `04-paddleocr-persistent-result-cache-ko.md`: 관리형 PaddleOCR-VL exact 영구 결과 캐시 계약
