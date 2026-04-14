# Top 2 Diff Review - auto_v1_cap4

핵심 문제 해결 방향은 사용자가 착안했다.

## 계약

- 이번 결과는 `PaddleOCR VL 단독 상주 상한선 benchmark` 기준이다.
- `runtime_services=ocr-only`, `stage_ceiling=ocr` 계약에서 Gemma와 MangaLMM은 실행되지 않는다.
- 최종 승격은 숫자 게이트가 아니라 사용자 OCR diff 검수 승인으로 확정한다.

## 대표 run

- baseline candidate: `fixed_w8`
- baseline reference run: `./banchmark_result_log/paddleocr_vl_parallel/20260415_031602_paddleocr-vl-parallel-smoke/fixed_w8/measured_r1`
- candidate: `auto_v1_cap4`
- candidate representative run: `./banchmark_result_log/paddleocr_vl_parallel/20260415_031602_paddleocr-vl-parallel-smoke/auto_v1_cap4/measured_r1`
- final_promotion_status: `approved_fixed_area_desc_w8`

## Aggregate

- ocr_total_sec_median: `295.765`
- baseline_delta_ocr_total_sec: `2.717`
- mean_CER: `0.0`
- baseline_delta_mean_CER: `0.0`
- exact_match: `1.0`
- baseline_delta_exact_match: `0.0`
- empty_block_median: `0`
- baseline_delta_empty_block_median: `0.0`
- page_failed_count_max: `0`
- baseline_delta_page_failed_count_max: `0.0`
- quality_gate_pass: `True`

## Quality Gate Notes

- none

## Page Changed Block Counts

| page | changed_block_count |
| --- | --- |

## Changed Blocks

- no OCR text difference against baseline
