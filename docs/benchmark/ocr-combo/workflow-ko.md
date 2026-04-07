# OCR Combo 벤치 워크플로

## 1. launcher

공식 launcher는 [benchmark_suite_cuda13.bat](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/scripts/benchmark_suite_cuda13.bat)이며, `ocr-combo-runtime` profile로 실행합니다.

## 2. smoke

- China 1장: `PPOCRv5 / PaddleOCR VL / HunyuanOCR`
- japan 1장: `MangaOCR / PaddleOCR VL / HunyuanOCR`

smoke는 service 기동, page snapshot, translated export, runtime snapshot이 모두 나오는지 확인하는 단계입니다.

## 3. reference

- China: `PaddleOCR VL + Gemma`
- japan: `PaddleOCR VL + Gemma`

각 corpus별 reference를 fresh 생성한 뒤 self-compare로 기준점을 고정합니다.

## 4. default compare

corpus별 3후보를 full pipeline cold 1회씩 비교하고 hard gate 통과 후보만 남깁니다.

## 5. bounded tuning

- `PaddleOCR VL`
  - `parallel_workers`
  - `max_new_tokens`
  - `max_concurrency`
  - `gpu_memory_utilization`
- `HunyuanOCR`
  - `parallel_workers`
  - `max_completion_tokens`
  - `n_gpu_layers`

각 축은 stepwise winner만 다음 축으로 보냅니다.

## 6. final confirm

corpus별 최종 후보 1개만 `cold 3회` 재실행하고 median elapsed로 winner를 확정합니다.

## 7. 산출물

- suite manifest
- corpus별 reference/default/tuning/final confirm raw 결과
- latest report + history snapshot
- language-aware routing policy
