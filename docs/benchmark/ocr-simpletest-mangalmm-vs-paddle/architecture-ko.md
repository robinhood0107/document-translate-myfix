# OCR Simpletest MangaLMM vs PaddleOCR VL Architecture

## 목적

- 일본어 OCR 후보 두 개를 짧은 full-pipeline 기준으로 직접 비교한다.
- `Gemma4`는 현재 promoted preset을 그대로 유지하고 OCR만 바꾼다.

## 구성

- baseline:
  - `PaddleOCR VL + current promoted Gemma4`
- candidate:
  - `MangaLMM + current promoted Gemma4`
- shared pipeline:
  - detector: `RT-DETR-v2`
  - mask refiner: `CTD`
  - inpainter: `lama_large_512px`
  - translator: `Custom Local Server(Gemma)`

## 수집 항목

- speed:
  - `elapsed_sec`
  - `ocr_total_sec`
  - `ocr_median_sec`
  - `detect_ocr_total_sec`
  - `translate_median_sec`
  - `page_failed_count`
- resident load:
  - `ocr_only_idle_gpu_used_delta_mb`
  - `full_idle_gpu_used_delta_mb`
  - `gemma_added_idle_gpu_used_delta_mb`
  - `gpu_floor_free_mb_after_ocr_only`
  - `gpu_floor_free_mb_after_full_runtime`
- runtime metadata:
  - `llama.cpp image/digest/version`
  - `docker logs`

## 결과 해석

- 속도 우열은 warm median 기준으로 본다.
- VRAM은 peak와 별개로 idle resident delta를 같이 본다.
- 품질은 자동 점수 대신 사용자의 3장 수동 검수로 확정한다.
