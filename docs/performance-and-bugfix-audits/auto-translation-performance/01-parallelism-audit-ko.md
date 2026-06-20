# 자동번역 성능 병렬화 감사

작성일: 2026-05-29
브랜치: `chore/auto-translation-performance-audit`

## 목적

자동번역 전체 흐름에서 동기 대기, 반복 runtime 확인, 단일 페이지 순차 처리 때문에 손실되는 시간을 특정하고, 비동기/멀티스레드/멀티프로세스/외부 서버 병렬화를 어디까지 안전하게 적용할 수 있는지 계산한다.

이 문서는 바로 구현 명세가 아니라, 다음 성능 PR을 보수적으로 쪼개기 위한 근거 장부다. 이미지 결과물이나 번역 품질에 닿는 변경은 반드시 테스트 이미지/실제 산출물을 남기고 사용자 검토를 받은 뒤 병합한다.

## 근거 자료

- 실제 프로젝트 상태:
  - `C:\path\to\comic-translate-project\project_20260529_034843.ctpr`
  - `C:\path\to\comic-translate-project\example_source_chapter.ctpr`
- 기존 benchmark 산출물:
  - `<benchmark-log-root>/workflow-split-runtime/<run-id>/timing_summary.json`
  - `<benchmark-log-root>/workflow-split-runtime/<run-id>/timing_summary.json`
  - `<benchmark-log-root>/workflow-split-runtime/<run-id>/timing_summary.json`
- 추출 로그:
  - `C:\path\to\comic-translate_validation_logs\2026-05-29\chore-auto-translation-performance-audit\performance_audit_extract.log`
  - `C:\path\to\comic-translate_validation_logs\2026-05-29\chore-auto-translation-performance-audit\performance_audit_extract.json`
  - `C:\path\to\comic-translate_validation_logs\2026-05-29\chore-auto-translation-performance-audit\gemma_endpoint_timing.log`
- 계산 원칙:
  - Amdahl, G. M. “Validity of the Single Processor Approach to Achieving Large Scale Computing Capabilities”, 1967. https://www.cs.cmu.edu/~18742/papers/Amdahl1967.pdf
  - Gustafson, J. L. “Reevaluating Amdahl’s Law”, Communications of the ACM, 1988. https://cacm.acm.org/research/reevaluating-amdahls-law/
  - Little, J. D. C. “A Proof for the Queuing Formula: L = λW”, Operations Research, 1961. https://pubsonline.informs.org/doi/abs/10.1287/opre.9.3.383
  - Python `threading`: CPython GIL 때문에 CPU-bound Python bytecode는 한 번에 한 스레드만 실행되며, I/O-bound 작업에는 thread가 적합하다. https://docs.python.org/3/library/threading.html
  - Python `asyncio`: high-level structured network I/O에 적합하다. https://docs.python.org/3.12/library/asyncio.html
  - Python `ProcessPoolExecutor`: multiprocessing 기반이라 GIL을 우회하지만 picklable 작업/결과여야 한다. https://docs.python.org/3/library/concurrent.futures.html
  - llama.cpp server multi-GPU/parallel slots: `--parallel`은 concurrent sequence 수와 KV cache 예산에 영향을 준다. https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
  - llama.cpp server 문서: OpenAI-compatible HTTP API server와 continuous batching/parallel 환경 변수를 제공한다. https://www.mintlify.com/ggml-org/llama.cpp/inference/server

## 현재 자동번역 구조

`stage_batched_processor.py` 기준 기본 흐름은 아래처럼 stage 단위로 순차 실행된다.

1. `_detect_all`: 모든 페이지 detect
2. `_ocr_all`: 모든 페이지 OCR
3. `_inpaint_all`: 모든 페이지 inpaint
4. `_translate_all`: 모든 페이지 translation
5. `_render_all`: 모든 페이지 render/export

이 구조는 모델 runtime 재사용에는 좋지만, 페이지 단위 pipeline overlap은 하지 않는다. 즉, 1페이지가 OCR을 끝내도 전체 364페이지 OCR이 끝날 때까지 inpaint로 넘어가지 않는다.

## 실제 Part 3 전체 run 계측

`example_source_chapter.ctpr`의 page state timestamp 기준이다. stage의 첫 완료 시각과 마지막 완료 시각 사이 span이므로 정확한 wall-clock stage runtime과 완전히 같지는 않지만, 364페이지 run의 병목 분포를 보기에 충분하다.

| stage | 완료 페이지 | span | 비중 |
| --- | ---: | ---: | ---: |
| detect | 364 | 258s | 6.8% |
| ocr | 364 | 821s | 21.8% |
| inpaint | 364 | 1777s | 47.1% |
| translation | 364 | 634s | 16.8% |
| render/export | 364 | 205s | 5.4% |
| stage gap/prewarm | - | 74s | 2.0% |
| 합계 | 364 | 3769s | 100% |

결론: 이번 실제 run에서는 inpaint가 가장 크고, OCR, translation이 그 다음이다. Gemma runtime 확인 반복은 고쳐야 하지만 전체 병목 1순위는 아니다.

## 기존 benchmark와 비교

13페이지 benchmark 기준, `stage_batched_pipeline`은 legacy 대비 translation stage가 크게 줄었다.

| run | 전체 | detect | OCR | inpaint | translate | render |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy page pipeline | 995.846s | 10.897s | 331.850s | 37.703s | 567.353s | 4.394s |
| stage-batched single OCR | 714.725s | 5.602s | 298.000s | 30.544s | 229.769s | 2.897s |
| stage-batched dual resident | 1664.021s | 4.739s | 308.602s | 31.219s | 229.742s | 2.903s |

관찰:

- stage-batched single OCR은 legacy 대비 약 1.39x 빠르다.
- translation은 legacy 567.353s에서 stage-batched 229.769s로 약 2.47x 개선됐다.
- dual resident run은 전체가 느려졌고, health/prewarm 대기가 231.691s였다. 따라서 GPU resident runtime을 무조건 동시에 올리는 방식은 위험하다.

## 반복 runtime 확인 손실 후보

### P-001 Gemma runtime/model 확인이 페이지마다 반복됨

증상:

```text
124/364 페이지 Gemma 번역 중...
Gemma 상태를 확인하는 중...
이미 실행 중인 Gemma 런타임을 재사용합니다.
Gemma 모델 목록을 확인하는 중...
Gemma 모델 확인 완료: gemma-4-26B-IQ4_NL.gguf
```

원인:

- `_translate_all()`은 translation stage 시작 때 `_await_gemma_runtime()`을 한 번 호출한다.
- 하지만 페이지 loop 안에서 매번 `Translator(...)`를 새로 만들고, `Translator.__init__()`이 `runtime_manager.ensure_server(...)`를 다시 호출한다.
- `ensure_server()`는 health probe와 `/v1/models` 확인을 다시 수행한다.

관련 파일:

- `pipeline/stage_batched_processor.py`
- `modules/translation/processor.py`
- `modules/translation/local_runtime.py`
- `pipeline/batch_processor.py`
- `pipeline/webtoon_batch/chunk.py`
- `pipeline/translation_handler.py`

손실 계산:

반복 횟수는 364페이지 기준 최대 364회다.

| 페이지당 추가 대기 | 364페이지 손실 | 전체 3769s 대비 |
| ---: | ---: | ---: |
| 0.01s | 3.6s | 0.1% |
| 0.05s | 18.2s | 0.5% |
| 0.10s | 36.4s | 1.0% |
| 0.30s | 109.2s | 2.9% |
| 0.50s | 182.0s | 4.8% |
| 1.00s | 364.0s | 9.7% |

현재 재측정 시점에는 Gemma 서버가 꺼져 있어 endpoint timing log는 connection refused다. 이전 실행 중 관측처럼 localhost endpoint가 ms 단위로 응답하면 실제 손실은 몇 초대다. 다만 번역 중 서버가 바쁠 때 `/v1/models`가 느려지면 분 단위 손실로 커질 수 있고, timeout 경로는 health 2초, models 5초까지 갈 수 있다.

권장 수정:

- `LocalGemmaRuntimeManager`에 batch/config-scoped readiness cache를 둔다.
- key는 endpoint URL, model name, compose file, container name, managed URL, relevant env로 구성한다.
- batch 시작 때 한 번 확인이 끝나면 같은 key에서는 `ensure_server()`가 health/model list를 생략한다.
- 번역 HTTP 오류가 발생하면 cache를 invalidate하고 재확인한다.
- `Translator` engine은 가능하면 translation stage에서 한 번 만들고 페이지 loop에서 재사용한다.

예상 효과:

- 일반 상태: 전체 0.1~1% 개선.
- endpoint가 바쁜 상태: 전체 3~10%까지 개선 가능.
- 안정성 개선 효과가 더 크다. 반복 progress noise와 불필요한 I/O 대기를 제거한다.

### P-002 OCR runtime reuse 확인 반복

legacy page pipeline benchmark에는 `ocr_runtime_reuse_hit`가 13회 기록되어 있다. stage-batched는 `_start_ocr_prewarm()`과 `_await_ocr_runtime()` 경로가 있어 더 낫지만, legacy/manual/webtoon 경로에서는 OCR processor 초기화가 페이지마다 runtime 확인을 다시 할 수 있다.

권장 수정:

- Gemma와 같은 readiness cache 패턴을 `LocalOCRRuntimeManager`에도 적용한다.
- stage-batched, legacy, manual, webtoon 모두 같은 manager cache를 공유하게 한다.

예상 효과:

- 정상 localhost health 응답이면 작다.
- 모델 로딩/health busy 상태에서는 수십 초 이상 줄일 수 있다.

## 병렬화 후보와 현재 판단

사용자 환경에서는 Gemma 실행 중 GPU util 85% 이상 또는 VRAM 여유 2~3GB 미만 조건에 거의 항상 도달한다. 따라서 아래 병렬화 후보는 “활성 제품 계획”이 아니라 왜 보류하는지 남기는 판단 근거다.

### P-003 Translation HTTP concurrency + llama.cpp parallel slots

현재 `docker-compose.yaml`은 `LLAMA_N_PARALLEL:-1`이다. 앱은 translation page loop를 순차 실행하므로 서버에 동시 요청이 들어가지 않는다.

현재 판단:

- 활성 계획에서 제외한다.
- `LLAMA_N_PARALLEL=2` 또는 app concurrency 2는 GPU/KV cache 경합을 늘릴 가능성이 높다.
- 지금 환경에서는 동시 요청 2개가 token/sec를 2배로 만들기보다 각 요청 latency를 늘릴 가능성이 크다.
- 이 실험은 필요하면 `benchmarking/lab`에서만 수행한다.

보류 조건:

- GPU util이 번역 중 85% 미만으로 안정적이거나 VRAM 여유가 3GB 이상일 때
- concurrency 2에서 total token/sec가 1.3x 이상 증가할 때
- JSON retry, truncation, timeout, empty content가 증가하지 않을 때

기존 도입 아이디어는 아래와 같지만 현재 활성 계획은 아니다.

- 앱 쪽: translation page 작업을 bounded queue로 실행한다. 시작값은 concurrency 2.
- 서버 쪽: `LLAMA_N_PARALLEL=2`, 필요 시 `--cont-batching` 지원 여부를 런타임 로그로 확인한다.
- 응답 순서는 page index로 재정렬한다.
- JSON schema/재시도/캐시 기록은 페이지별로 독립 유지한다.
- context budget은 `LLAMA_CTX_SIZE / LLAMA_N_PARALLEL`로 줄어들 수 있으므로, 긴 페이지는 자동으로 concurrency 1로 내려야 한다.

계산:

Part 3 translation span은 634s다.

| translation speedup | translation 새 시간 | 전체 새 시간 | 전체 개선 |
| ---: | ---: | ---: | ---: |
| 1.3x | 488s | 3623s | 1.04x |
| 1.5x | 423s | 3558s | 1.06x |
| 2.0x | 317s | 3452s | 1.09x |

위험:

- 한 GPU에서 inpaint/OCR와 동시에 돌리면 더 느려질 수 있다.
- llama.cpp parallel slot은 KV cache를 늘리거나 per-slot context를 줄일 수 있다.
- 번역 품질/JSON 안정성은 동시 요청에서도 별도 회귀 테스트가 필요하다.

### P-004 OCR page concurrency

`PaddleOCRVLEngine`은 이미 block 단위 내부 `ThreadPoolExecutor`를 사용한다. 즉 페이지 안에서 여러 crop OCR 요청은 병렬화되어 있다. 아직 남은 후보는 페이지 단위 OCR concurrency다.

현재 판단:

- 활성 계획에서 제외한다.
- page concurrency는 기존 block worker 수와 곱해져 GPU/HTTP server 부하를 급격히 키운다.
- Gemma가 이미 GPU를 거의 점유하는 환경에서는 OCR page concurrency가 전체 pipeline을 더 불안정하게 만들 수 있다.
- 필요하면 `benchmarking/lab`에서만 실험한다.

기존 도입 아이디어는 아래와 같지만 현재 활성 계획은 아니다.

- OCR stage에서 page concurrency 2를 실험한다.
- OCR 서버가 queue/parallel worker를 감당하는지 GPU utilization, VRAM, timeout을 같이 기록한다.
- 현재 `parallel_workers`와 page concurrency의 곱이 너무 커지지 않게 resource gate를 둔다.

계산:

Part 3 OCR span은 821s다.

| OCR speedup | OCR 새 시간 | 전체 새 시간 | 전체 개선 |
| ---: | ---: | ---: | ---: |
| 1.25x | 657s | 3605s | 1.05x |
| 1.5x | 547s | 3495s | 1.08x |
| 2.0x | 411s | 3359s | 1.12x |

위험:

- OCR 서버가 이미 block-level parallel worker 8을 쓰면 page concurrency 2는 실제로 16 in-flight crop 요청이 될 수 있다.
- GPU memory가 낮으면 timeout/품질 저하가 생길 수 있다.

### P-005 Inpaint/debug output 비동기화

Part 3 inpaint span은 1777s로 가장 크다. 이 run에서는 patch 합계가 9038개, median 20개, max 348개다. 또한 `log_example_source_chapter.../inpainted_images`의 cleaned image는 페이지당 약 24.9MB 수준이라, cleaned PNG만으로도 약 9GB가 만들어진다.

관련 동기 I/O:

- cleaned image export
- detector overlay
- raw mask
- mask overlay
- cleanup delta
- debug metadata JSON
- preview emit

현재 판단:

- 활성 계획에서 제외한다.
- GPU-bound 동시성은 아니지만 writer queue도 결과물 누락, cancel/drain, 실패 보고 정책을 바꾸는 동시성 변경이다.
- 먼저 동기 I/O가 실제 병목인지 계측한다. I/O가 inpaint/render 시간의 10% 이상임이 확인될 때만 별도 계획으로 되살린다.

기존 도입 아이디어는 아래와 같지만 현재 활성 계획은 아니다.

- 모델 inference 자체는 우선 순차 유지한다.
- debug/output write만 bounded writer queue로 넘긴다.
- 큐 크기는 1~2로 제한한다. 이미지 array copy 때문에 무제한 큐는 메모리 위험이 크다.
- write 완료 경로는 page summary에 나중에 반영하거나, path는 선계산하고 실패만 report한다.

계산:

inpaint 1777s 중 10~20%가 이미지 인코딩/파일쓰기라면:

| 겹칠 수 있는 I/O 비율 | 예상 절감 | 전체 개선 |
| ---: | ---: | ---: |
| 10% | 178s | 1.05x |
| 15% | 267s | 1.08x |
| 20% | 355s | 1.10x |

위험:

- 디버그 산출물 누락이 생기면 사용자 검토가 어려워진다.
- 앱 종료/취소 시 writer queue drain 정책이 필요하다.

### P-006 Inpaint model 자체 병렬화

inpaint model을 페이지 단위로 병렬 실행하면 이론상 효과가 가장 크다.

계산:

| inpaint speedup | inpaint 새 시간 | 전체 새 시간 | 전체 개선 |
| ---: | ---: | ---: | ---: |
| 1.25x | 1422s | 3414s | 1.10x |
| 1.5x | 1185s | 3177s | 1.19x |
| 2.0x | 889s | 2881s | 1.31x |

하지만 단일 GPU에서는 위험도가 높다. 같은 모델을 여러 process에 로드하면 VRAM을 중복 사용하고, 같은 GPU queue에서 context switching만 늘 수 있다. 먼저 mask generation/debug writing을 분리하고, 그 다음 GPU utilization/VRAM이 여유 있을 때만 page concurrency 2를 실험한다.

현재 판단:

- 활성 계획에서 제외한다.
- 이 항목은 `benchmarking/lab` 전용으로만 남긴다.

### P-007 Render/export 병렬화

Part 3 render/export span은 205s다. 전체 5.4%라 우선순위는 낮다.

주의:

- Qt text layout과 `QFontMetrics`/`QPainter`는 GUI thread와 thread-safety를 매우 조심해야 한다.
- 단순 final image composite/write는 worker 후보지만, 텍스트 layout 자체를 병렬화하는 것은 위험하다.

현재 판단:

- 활성 계획에서 제외한다.
- render/export는 전체 5.4%라 우선순위가 낮고, writer queue도 결과물 보장과 cancel/drain 정책을 바꾸는 동시성 변경이다.
- 먼저 telemetry로 실제 I/O 병목이 있는지 확인한다.
- text item state 생성과 Qt render thread는 현행 유지한다.

## 전체 speedup 계산

고정 작업량 기준 Amdahl 식:

```text
S(N) = 1 / ((1 - P) + P / N + O(N))
```

여기서 `P`는 병렬화 가능한 시간 비율, `O(N)`은 큐/복사/동기화/VRAM 경합 오버헤드다.

Part 3에서 OCR + inpaint + translation은 3232s, 전체 3769s의 85.8%다. 이 세 stage만 완벽하게 2배 빨라진다면:

```text
new_time = 3769 - 3232 + 3232 / 2 = 2153s
speedup = 1.75x
```

하지만 이 값은 상한이다. 세 stage가 GPU/외부 runtime을 공유하므로 실제로는 오버헤드와 자원 경합이 생긴다.

현실적인 목표:

| 시나리오 | 구성 | 예상 전체 시간 | 예상 speedup |
| --- | --- | ---: | ---: |
| Micro | Gemma/OCR readiness cache, Translator reuse | 3730~3765s | 1.00~1.01x |
| Safe Scheduling | Micro + GPU 포화 시 prewarm overlap 회피 + stage gap 계측 | 3700~3765s | 1.00~1.02x |
| I/O Evidence Only | 동기 I/O 병목이 10% 이상으로 확인된 뒤 별도 검토 | 미정 | 미정 |

현재 목표는 동시성으로 큰 speedup을 노리는 것이 아니라, GPU 포화 환경에서 자원 경합을 줄이고 반복 probe를 제거해 안정성을 높이는 것이다.

## 구현 로드맵

### PR 1: runtime readiness cache

- Gemma `ensure_server()` batch/config-scoped cache
- OCR `ensure_engine()` batch/config-scoped cache
- 번역/ OCR 요청 실패 시 cache invalidate
- 테스트: 같은 config에서 repeated ensure가 health/model endpoint를 한 번만 호출하는지 확인
- 기대: 반복 로그 제거, timeout-risk 감소

### PR 2: Translator stage reuse

- `_translate_all()`에서 `Translator`를 페이지마다 만들지 않고 stage 단위로 재사용
- source/target language가 stage-batched에서 동일하다는 기존 invariant 활용
- legacy/manual/webtoon은 별도 검토
- 테스트: 캐시 hit/miss, translation metrics, failure reporting 유지

### PR 3: GPU-safe prewarm scheduling

- GPU util/VRAM headroom을 기준으로 Gemma prewarm overlap을 제한한다.
- Gemma가 이미 GPU를 거의 점유하는 환경에서는 inpaint/OCR 중 추가 runtime prewarm을 피한다.
- prewarm wait, skipped prewarm reason, GPU snapshot을 benchmark event에 남긴다.
- 결과물은 바꾸지 않는다.

### PR 4: performance telemetry cleanup

- runtime probe 시간, model check 시간, prewarm wait, stage gap, page duration을 일관된 event로 남긴다.
- 추후 최적화 전후 비교가 가능하게 한다.
- 동시성, 결과물, 모델 설정은 바꾸지 않는다.

### 보류: inpaint concurrency benchmark only

- 제품 변경 전 `benchmarking/lab`에서 먼저 실험
- single GPU에서 page concurrency 2가 실제로 빠른지 확인
- VRAM peak, failure, 이미지 품질 비교 없이는 develop 승격 금지

## 결론

- 지금 보이는 Gemma 반복 확인은 확실한 낭비다. 다만 정상 localhost 응답 기준으로는 전체 병목 1순위가 아니다.
- 실제 Part 3 병목은 inpaint 47.1%, OCR 21.8%, translation 16.8%다.
- 가장 안전한 첫 개선은 readiness cache + Translator reuse다.
- 현재 GPU 포화 환경에서는 translation concurrency, OCR page concurrency, inpaint concurrency, async writer를 활성 계획에서 제외한다.
- 앞으로 해야 할 일은 더 세게 밀어붙이는 것이 아니라 GPU 자원 경합을 줄이고, prewarm/probe/stage gap을 계측 가능하게 만드는 것이다.
