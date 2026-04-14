# OCR Simpletest MangaLMM vs PaddleOCR VL Results History

## Current Policy

- corpus는 항상 `Sample/simpletest` 3장만 사용한다.
- winner는 `warm_total_page_failed_count=0` 후보를 우선하고, 그 안에서 `warm_median_elapsed_sec`가 가장 낮은 후보로 정한다.
- 품질 승격은 사용자의 수동 검수 후에만 한다.

## Latest Output

- latest suite root: `./banchmark_result_log/ocr_simpletest_mangalmm_vs_paddle/20260414_152650_ocr_simpletest_mangalmm_vs_paddle_suite`
- winner: `paddleocr_vl`
- warm_median_elapsed_sec: `153.018`
- warm_total_page_failed_count: `0`
- gemma recommendation: `keep current Gemma settings`
- latest copied assets:
  - `./docs/assets/benchmarking/ocr-simpletest-mangalmm-vs-paddle/latest/comparison_summary.json`
  - `./docs/assets/benchmarking/ocr-simpletest-mangalmm-vs-paddle/latest/comparison_summary.md`
  - `./docs/assets/benchmarking/ocr-simpletest-mangalmm-vs-paddle/latest/report_summary.json`
  - `./docs/assets/benchmarking/ocr-simpletest-mangalmm-vs-paddle/latest/copied_assets.json`

## History

- 공식 latest/history는 완료된 CUDA13 suite만 기준으로 갱신한다.
- current suite archive: `./docs/assets/benchmarking/ocr-simpletest-mangalmm-vs-paddle/history/20260414_152650_ocr_simpletest_mangalmm_vs_paddle_suite`

