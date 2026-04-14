# PaddleOCR VL Parallel Results History

## Current Policy

- baseline은 항상 `fixed_w8`이다.
- detector freeze와 gold seed는 baseline measured run에서 생성한다.
- subset winner만으로 `default on` 승격을 하지 않는다.

## Latest Output

- latest suite root: `./banchmark_result_log/paddleocr_vl_parallel/20260415_020217_paddleocr-vl-parallel-smoke`
- winner: `fixed_w8`
- scheduler_mode: `fixed`
- ocr_total_sec_median: `305.231`
- baseline gold: `./banchmark_result_log/paddleocr_vl_parallel/20260415_020217_paddleocr-vl-parallel-smoke/baseline_gold.json`
- detector manifest: `./banchmark_result_log/paddleocr_vl_parallel/20260415_020217_paddleocr-vl-parallel-smoke/detector_manifest.json`
