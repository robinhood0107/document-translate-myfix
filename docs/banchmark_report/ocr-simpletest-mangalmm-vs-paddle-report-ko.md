# OCR Simpletest MangaLMM vs PaddleOCR VL Report

## Metadata

- latest suite dir: `./banchmark_result_log/ocr_simpletest_mangalmm_vs_paddle/20260414_152650_ocr_simpletest_mangalmm_vs_paddle_suite`
- latest assets dir: `./docs/assets/benchmarking/ocr-simpletest-mangalmm-vs-paddle/latest`
- baseline gemma report: `./docs/banchmark_report/gemma-iq4nl-japan-report-ko.md`

## Winner

- winner: `paddleocr_vl`
- winner_reason: `Lowest warm_median_elapsed_sec among candidates with zero warm page failures. Candidates with warm page failures are ranked behind successful runs.`
- gemma recommendation: `keep current Gemma settings`

## Warm Full-Pipeline Comparison

| candidate | warm_median_elapsed_sec | warm_median_ocr_total_sec | warm_median_detect_ocr_total_sec | warm_median_translate_median_sec | warm_total_page_failed_count | warm_median_gpu_peak_used_mb | warm_median_gpu_floor_free_mb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paddleocr_vl | 153.018 | 76.01 | 79.574 | 15.735 | 0 | 11963.5 | 36.5 |
| mangalmm | 37.167 | None | 2.522 | None | 6 | 11673.5 | 326.5 |

## Resident VRAM Delta

| candidate | ocr_only_idle_gpu_used_delta_mb | full_idle_gpu_used_delta_mb | gemma_added_idle_gpu_used_delta_mb | gpu_floor_free_mb_after_ocr_only | gpu_floor_free_mb_after_full_runtime |
| --- | --- | --- | --- | --- | --- |
| paddleocr_vl | 227 | 188 | -39 | 1530 | 1569 |
| mangalmm | -451 | -514 | -63 | 727 | 790 |

## Candidate Assets

### paddleocr_vl

- selected warm run dir: `./banchmark_result_log/ocr_simpletest_mangalmm_vs_paddle/20260414_152650_ocr_simpletest_mangalmm_vs_paddle_suite/paddleocr_vl/warm2`
- selected export root: `./banchmark_result_log/ocr_simpletest_mangalmm_vs_paddle/20260414_152650_ocr_simpletest_mangalmm_vs_paddle_suite/paddleocr_vl/warm2/corpus/comic_translate_Apr-14-2026_03-32-38PM`
- copied translated images:
  - `./docs/assets/benchmarking/ocr-simpletest-mangalmm-vs-paddle/latest/translated_images/paddleocr_vl/p_016_translated.jpg`
  - `./docs/assets/benchmarking/ocr-simpletest-mangalmm-vs-paddle/latest/translated_images/paddleocr_vl/p_017_translated.jpg`
  - `./docs/assets/benchmarking/ocr-simpletest-mangalmm-vs-paddle/latest/translated_images/paddleocr_vl/p_021_translated.jpg`

### mangalmm

- selected warm run dir: `./banchmark_result_log/ocr_simpletest_mangalmm_vs_paddle/20260414_152650_ocr_simpletest_mangalmm_vs_paddle_suite/mangalmm/warm2`
- selected export root: `./banchmark_result_log/ocr_simpletest_mangalmm_vs_paddle/20260414_152650_ocr_simpletest_mangalmm_vs_paddle_suite/mangalmm/warm2/corpus/comic_translate_Apr-14-2026_03-39-17PM`
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

