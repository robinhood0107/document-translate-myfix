# OCR Simpletest MangaLMM vs PaddleOCR VL

## Decision

- winner: `paddleocr_vl`
- reason: `Lowest warm_median_elapsed_sec among candidates with zero warm page failures. Candidates with warm page failures are ranked behind successful runs.`

## Warm Full-Pipeline Comparison

| candidate | warm_median_elapsed_sec | warm_median_ocr_total_sec | warm_median_detect_ocr_total_sec | warm_median_translate_median_sec | warm_total_page_failed_count | warm_median_gpu_peak_used_mb | warm_median_gpu_floor_free_mb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paddleocr_vl | 153.018 | 76.01 | 79.574 | 15.735 | 0 | 11963.5 | 36.5 |
| mangalmm | 37.167 | None | 2.522 | None | 6 | 11673.5 | 326.5 |

## Resident GPU Deltas

| candidate | ocr_only_idle_gpu_used_delta_mb | full_idle_gpu_used_delta_mb | gemma_added_idle_gpu_used_delta_mb | gpu_floor_free_mb_after_ocr_only | gpu_floor_free_mb_after_full_runtime |
| --- | --- | --- | --- | --- | --- |
| paddleocr_vl | 227 | 188 | -39 | 1530 | 1569 |
| mangalmm | -451 | -514 | -63 | 727 | 790 |

## Notes

- corpus: `Sample/simpletest` 3 pages (`p_016`, `p_017`, `p_021`)
- run shape: `cold 1 + warm 2`
- benchmark scope: `full-pipeline`
- Japanese `Optimal+` maps to the `mangalmm` candidate in this family
- translator baseline: current promoted `Gemma4` preset kept fixed
