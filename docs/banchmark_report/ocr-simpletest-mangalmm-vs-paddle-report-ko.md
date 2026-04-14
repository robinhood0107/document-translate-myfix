# OCR Simpletest MangaLMM vs PaddleOCR VL Report

## Metadata

- latest suite dir: `./banchmark_result_log/ocr_simpletest_mangalmm_vs_paddle/20260414_090047_ocr_simpletest_mangalmm_vs_paddle_suite`
- latest assets dir: `./docs/assets/benchmarking/ocr-simpletest-mangalmm-vs-paddle/latest`
- baseline gemma report: `./docs/banchmark_report/gemma-iq4nl-japan-report-ko.md`

## Winner

- winner: `paddleocr_vl`
- winner_reason: `Lowest warm_median_elapsed_sec among candidates with zero warm page failures. Candidates with warm page failures are ranked behind successful runs.`
- gemma recommendation: `keep current Gemma settings`

## Warm Full-Pipeline Comparison

| candidate | warm_median_elapsed_sec | warm_median_ocr_total_sec | warm_median_detect_ocr_total_sec | warm_median_translate_median_sec | warm_total_page_failed_count | warm_median_gpu_peak_used_mb | warm_median_gpu_floor_free_mb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paddleocr_vl | 137.48 | 72.837 | 77.308 | 10.431 | 0 | 11985 | 15 |
| mangalmm | 14.58 | None | 2.692 | None | 6 | 11345.5 | 654.5 |

## Resident VRAM Delta

| candidate | ocr_only_idle_gpu_used_delta_mb | full_idle_gpu_used_delta_mb | gemma_added_idle_gpu_used_delta_mb | gpu_floor_free_mb_after_ocr_only | gpu_floor_free_mb_after_full_runtime |
| --- | --- | --- | --- | --- | --- |
| paddleocr_vl | 1433 | 11103 | 9670 | 10155 | 485 |
| mangalmm | 9865 | 11261 | 1396 | 1967 | 571 |

## Candidate Assets

### paddleocr_vl

- selected warm run dir: `./banchmark_result_log/ocr_simpletest_mangalmm_vs_paddle/20260414_090047_ocr_simpletest_mangalmm_vs_paddle_suite/paddleocr_vl/warm2`
- selected export root: `./banchmark_result_log/ocr_simpletest_mangalmm_vs_paddle/20260414_090047_ocr_simpletest_mangalmm_vs_paddle_suite/paddleocr_vl/warm2/corpus/comic_translate_Apr-14-2026_09-09-44AM`
- copied translated images:
  - `[local-only]`
  - `[local-only]`
  - `[local-only]`

### mangalmm

- selected warm run dir: `./banchmark_result_log/ocr_simpletest_mangalmm_vs_paddle/20260414_090047_ocr_simpletest_mangalmm_vs_paddle_suite/mangalmm/warm2`
- selected export root: `./banchmark_result_log/ocr_simpletest_mangalmm_vs_paddle/20260414_090047_ocr_simpletest_mangalmm_vs_paddle_suite/mangalmm/warm2/corpus/comic_translate_Apr-14-2026_09-17-30AM`
- copied translated images: `none`

## Gemma Recommendation

- fixed baseline:
  - `context_size=4096`
  - `threads=10`
  - `n_gpu_layers=23`
  - `chunk_size=6`
  - `temperature=0.7`
  - `max_completion_tokens=512`
- decision reasons:
  - MangaLMM warm run에서 page_failed_count가 0이 아니므로, 현재 이슈는 Gemma 수치보다 OCR 안정성에 먼저 가깝다
  - simpletest 결과 기준으로는 Gemma4 현행 promoted 값을 유지하고, MangaLMM OCR 성공률과 empty 응답 원인을 먼저 해결하는 편이 안전하다

