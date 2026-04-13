# OCR Simpletest MangaLMM vs PaddleOCR VL Results History

## Current Policy

- corpus는 항상 `Sample/simpletest` 3장만 사용한다.
- speed winner는 `warm_median_elapsed_sec` 기준으로 정한다.
- 품질 승격은 사용자의 수동 검수 후에만 한다.

## Latest Output

- latest suite root:
  - `./banchmark_result_log/ocr_simpletest_mangalmm_vs_paddle/`
- latest comparison summary:
  - `comparison_summary.json`
  - `comparison_summary.md`

## History

- suite run은 timestamped directory로 누적 저장한다.
- per-run raw outputs와 translated images는 각 suite 디렉터리 안에 그대로 유지한다.
