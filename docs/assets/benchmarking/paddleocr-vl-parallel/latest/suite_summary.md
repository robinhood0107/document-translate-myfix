# PaddleOCR VL Parallel Suite Summary

- suite_dir: `./banchmark_result_log/paddleocr_vl_parallel/20260415_020217_paddleocr-vl-parallel-smoke`
- baseline_candidate_key: `fixed_w8`
- page_count: `13`
- candidate_count: `3`
- pass_candidate_count: `1`
- winner: `fixed_w8`

| candidate | scheduler | workers | ocr_total_sec_median | ocr_page_p95_sec_median | mean_CER | mean_exact_match | gate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_w8 | fixed | 8 | 305.231 | 43.893 | 0.0 | 1.0 | True |
| fixed_area_desc_w8 | fixed_area_desc | 8 | 290.731 | 39.859 | 0.0011 | 0.9953 | False |
| auto_v1_cap4 | auto_v1 | 4 | 292.363 | 41.442 | 0.0011 | 0.9953 | False |
