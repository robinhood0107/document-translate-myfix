# Gemma IQ4_NL Japan Full-GPU 벤치 사용법

## 목적
- `Sample/japan` 22장 전체를 대상으로 `gemma-4-26B-IQ4_NL.gguf`의 full-pipeline 운영값을 찾습니다.
- 파이프라인은 `Custom Local Server(Gemma) + PaddleOCR VL + RT-DETR-v2 + ctd + lama_large_512px`로 고정합니다.
- hidden CPU fallback은 허용하지 않습니다.
- 공식 결과는 fresh suite에서 `Sample/japan` 22장 full pipeline batch를 끝까지 돌린 결과만 채택합니다.

## 권장 진입점
- 끝까지 자동 실행 supervisor:
  - `scripts\run_gemma_iq4nl_japan_fullgpu_until_done_cuda13.bat`
- stage executor 수동 실행:
  - `scripts\gemma_iq4nl_japan_fullgpu_benchmark_suite_cuda13.bat`
- 개별 pipeline 확인:
  - `scripts\gemma_iq4nl_japan_fullgpu_benchmark_pipeline_cuda13.bat`
- 기존 suite profile 경유:
  - `scripts\benchmark_suite_cuda13.bat --suite-profile gemma-iq4nl-japan-fullgpu`

## supervisor 동작
- 기본값은 fresh suite 생성입니다.
- 실행 순서:
  - `smoke -> report -> stage1 -> report -> stage2 -> report -> stage3 -> report -> stage4 -> report -> stage5 -> report -> confirm -> report`
- health 지연, 컨테이너 recreate 충돌, 연결 거부 같은 retryable infra failure는 같은 stage를 다시 실행합니다.
- OOM/VRAM 부족은 후보를 바로 버리지 않고, benchmark runner 안에서 rescue ladder를 적용합니다.

## stage 설명
- `smoke`: `094.png` 1장으로 full GPU 강제 조건과 Gemma IQ4_NL 실제 로드를 확인
- `stage1`: full matrix
  - `ocr_runtime.gpu_memory_utilization = 0.68, 0.72, 0.76, 0.80`
  - `gemma.n_gpu_layers = 14, 16, 18, 20, 22, 23`
  - 원본 후보 24개 전부 실행
- `stage2`: `context_size = 3072, 4096`
- `stage3`: `chunk_size = 4, 5, 6`
- `stage4`: `threads = 10, 12, 14`
- `stage5`: `max_completion_tokens = 384, 512, 640`
- `confirm`: winner / runner-up를 각각 3회 재검증

## rescue 정책
- temperature rescue
  - 각 후보는 먼저 `0.7`
  - 출력 오류가 있으면 `0.6 -> 0.5 -> 0.4`
- OOM rescue
  - `ocr_runtime.gpu_memory_utilization` 축소
  - `gemma.n_gpu_layers` 축소
  - `gemma.context_size` 축소
  - `gemma.max_completion_tokens` 축소
  - `gemma.chunk_size` 축소
- rescue 성공 후보도 ranking에 포함되지만, 같은 성능권이면 원본 후보를 우선합니다.

## 산출물
- raw results: `banchmark_result_log/gemma_iq4nl_japan/`
- suite state: `suite_state.json`
- manifest: `gemma_iq4nl_japan_report_manifest.yaml`
- latest summary assets: `docs/assets/benchmarking/gemma-iq4nl-japan/latest/`
- latest markdown report: `docs/banchmark_report/gemma-iq4nl-japan-report-ko.md`
- live progress markdown: `docs/benchmark/gemma-iq4nl-japan/live-progress-ko.md`
