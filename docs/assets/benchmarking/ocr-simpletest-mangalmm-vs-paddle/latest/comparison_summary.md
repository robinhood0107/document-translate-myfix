# OCR Simpletest MangaLMM vs PaddleOCR VL

## Decision

- winner: `paddleocr_vl`
- reason: `Lowest warm_median_elapsed_sec among candidates with zero warm page failures. Candidates with warm page failures are ranked behind successful runs.`

## Warm Full-Pipeline Comparison

| candidate | warm_median_elapsed_sec | warm_median_ocr_total_sec | warm_median_detect_ocr_total_sec | warm_median_translate_median_sec | warm_total_page_failed_count | warm_median_gpu_peak_used_mb | warm_median_gpu_floor_free_mb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paddleocr_vl | 137.48 | 72.837 | 77.308 | 10.431 | 0 | 11985 | 15 |
| mangalmm | 14.58 | None | 2.692 | None | 6 | 11345.5 | 654.5 |

## Resident GPU Deltas

| candidate | ocr_only_idle_gpu_used_delta_mb | full_idle_gpu_used_delta_mb | gemma_added_idle_gpu_used_delta_mb | gpu_floor_free_mb_after_ocr_only | gpu_floor_free_mb_after_full_runtime |
| --- | --- | --- | --- | --- | --- |
| paddleocr_vl | 1433 | 11103 | 9670 | 10155 | 485 |
| mangalmm | 9865 | 11261 | 1396 | 1967 | 571 |

## Notes

- corpus: `Sample/simpletest` 3 pages (`p_016`, `p_017`, `p_021`)
- run shape: `cold 1 + warm 2`
- benchmark scope: `full-pipeline`
- translator baseline: current promoted `Gemma4` preset kept fixed
