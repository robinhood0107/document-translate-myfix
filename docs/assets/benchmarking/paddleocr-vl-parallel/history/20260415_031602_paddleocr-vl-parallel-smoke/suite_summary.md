# PaddleOCR VL Parallel Suite Summary

- suite_dir: `./banchmark_result_log/paddleocr_vl_parallel/20260415_031602_paddleocr-vl-parallel-smoke`
- runtime_contract: `paddleocr-vl-single-tenant-ocr-only`
- runtime_services: `ocr-only`
- stage_ceiling: `ocr`
- baseline_candidate_key: `fixed_w8`
- page_count: `13`
- candidate_count: `3`
- pass_candidate_count: `3`
- quality_gate_winner: `fixed_area_desc_w8`
- final_promotion_status: `approved_fixed_area_desc_w8`

## Review Candidates

- top1: `fixed_area_desc_w8` (ocr_total_sec_median=`292.526`, gate_pass=`True`)
- top2: `auto_v1_cap4` (ocr_total_sec_median=`295.765`, gate_pass=`True`)

## Candidate Table

| candidate | scheduler | workers | ocr_total_sec_median | ocr_page_p95_sec_median | mean_CER | mean_exact_match | gate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_area_desc_w8 | fixed_area_desc | 8 | 292.526 | 43.66 | 0.0 | 1.0 | True |
| fixed_w8 | fixed | 8 | 293.048 | 42.082 | 0.0 | 1.0 | True |
| auto_v1_cap4 | auto_v1 | 4 | 295.765 | 40.871 | 0.0 | 1.0 | True |
