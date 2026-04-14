# PaddleOCR VL Parallel Report

## Metadata

- latest suite dir: `./banchmark_result_log/paddleocr_vl_parallel/20260415_020217_paddleocr-vl-parallel-smoke`
- latest assets dir: `./docs/assets/benchmarking/paddleocr-vl-parallel/latest`
- baseline gold: `./banchmark_result_log/paddleocr_vl_parallel/20260415_020217_paddleocr-vl-parallel-smoke/baseline_gold.json`
- detector manifest: `./banchmark_result_log/paddleocr_vl_parallel/20260415_020217_paddleocr-vl-parallel-smoke/detector_manifest.json`

## Winner

- winner: `fixed_w8`
- scheduler_mode: `fixed`
- parallel_workers: `8`
- ocr_total_sec_median: `305.231`
- ocr_page_p95_sec_median: `43.893`

## Candidate Table

| candidate | scheduler | workers | ocr_total_sec_median | ocr_page_p95_sec_median | mean_CER | mean_exact_match | gate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_w8 | fixed | 8 | 305.231 | 43.893 | 0.0 | 1.0 | True |
| fixed_area_desc_w8 | fixed_area_desc | 8 | 290.731 | 39.859 | 0.0011 | 0.9953 | False |
| auto_v1_cap4 | auto_v1 | 4 | 292.363 | 41.442 | 0.0011 | 0.9953 | False |

## Visual Assets

- OCR total chart: `./docs/assets/benchmarking/paddleocr-vl-parallel/latest/ocr_total_sec_median.svg`
- OCR page p95 chart: `./docs/assets/benchmarking/paddleocr-vl-parallel/latest/ocr_page_p95_sec_median.svg`
- Mean CER chart: `./docs/assets/benchmarking/paddleocr-vl-parallel/latest/quality_mean_cer.svg`
- Candidate table: `./docs/assets/benchmarking/paddleocr-vl-parallel/latest/candidate_table.md`

## Portfolio Notes

- 핵심 문제 해결 방향은 사용자가 착안했다.
- 실험 설계 구체화, 계측 설계, 구현, 검증은 공동 수행으로 정리한다.
- 세부 narrative는 `docs/benchmark/paddleocr-vl-parallel/`와 latest problem-solving specs를 함께 본다.
