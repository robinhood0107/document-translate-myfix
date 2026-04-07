# OCR Combo Benchmark Report

이 파일은 `scripts/generate_ocr_combo_report.py`가 suite manifest를 기준으로 갱신합니다.

## Metadata

- generated_at: `2026-04-08 03:56:02 대한민국 표준시`
- status: `benchmark_complete`
- benchmark_name: `OCR Combo Runtime Benchmark`
- benchmark_kind: `managed family suite`
- benchmark_scope: `language-aware OCR+Gemma comparison using benchmark_suite_cuda13.bat --suite-profile ocr-combo-runtime`
- execution_scope: `full-pipeline`
- speed_score_scope: `full-pipeline elapsed_sec`
- quality_gate_scope: `OCR-only`
- gold_source: `human-reviewed`
- gold_empty_text_policy: `geometry-kept-text-skipped`
- crop_regression_focus: `xyxy-first OCR crop with bubble clamp; p_018 overread regression`
- baseline_sha: `6b3a01ade0e9e4918124a64b053487d7a7f10d7a`
- develop_ref_sha: `c1cd90d4da7419df213893e8ddfd1451d16ed0eb`
- entrypoint: `scripts\benchmark_suite_cuda13.bat --suite-profile ocr-combo-runtime`
- results_root: `./banchmark_result_log/ocr_combo`

## Fixed Gemma

- image: `local/llama.cpp:server-cuda-b8665`
- response_format_mode: `json_schema`
- chunk_size: `6`
- temperature: `0.6`
- n_gpu_layers: `23`

## OCR Gate Rules

- canonical_small_voiced_kana: `True`
- ignored_chars: `「」『』,，、♡♥`
- `gold_text=""` block은 geometry를 유지하되 non-empty OCR text hard gate에서는 제외합니다.
- crop overreach 회귀는 `p_018` 사례를 기준으로 추적합니다.

## Corpora

| corpus | sample_dir | sample_count | source_lang | target_lang | gold_path | gold_review_status |
| --- | --- | --- | --- | --- | --- | --- |
| china | ./Sample/China | 8 | Chinese | Korean | ./benchmarks/ocr_combo/gold/china/gold.json | locked |
| japan | ./Sample/japan | 22 | Japanese | Korean | ./benchmarks/ocr_combo/gold/japan/gold.json | locked |

## Smoke

| corpus | engine | elapsed_sec | ocr_total_sec | translate_total_sec | run_dir |
| --- | --- | --- | --- | --- | --- |
| china | PPOCRv5 + Gemma | 35.666 | 2.794 | 25.549 | ./banchmark_result_log/ocr_combo/20260408_005922_ocr-combo-runtime_suite/smoke/china/china-ppocrv5-smoke |
| china | PaddleOCR VL + Gemma | 70.27 | 21.802 | 37.188 | ./banchmark_result_log/ocr_combo/20260408_005922_ocr-combo-runtime_suite/smoke/china/china-paddleocr-smoke |
| china | HunyuanOCR + Gemma | 44.041 | 2.02 | 33.383 | ./banchmark_result_log/ocr_combo/20260408_005922_ocr-combo-runtime_suite/smoke/china/china-hunyuanocr-smoke |
| japan | MangaOCR + Gemma | 30.072 | 2.734 | 19.129 | ./banchmark_result_log/ocr_combo/20260408_005922_ocr-combo-runtime_suite/smoke/japan/japan-mangaocr-smoke |
| japan | PaddleOCR VL + Gemma | 65.877 | 32.182 | 24.321 | ./banchmark_result_log/ocr_combo/20260408_005922_ocr-combo-runtime_suite/smoke/japan/japan-paddleocr-smoke |
| japan | HunyuanOCR + Gemma | 31.275 | 3.787 | 18.388 | ./banchmark_result_log/ocr_combo/20260408_005922_ocr-combo-runtime_suite/smoke/japan/japan-hunyuanocr-smoke |

## Default Comparison

| corpus | engine | elapsed_sec | ocr_total_sec | translate_total_sec | gpu_peak_used_mb | gpu_floor_free_mb | quality_gate_pass | geometry_match_recall | geometry_match_precision | non_empty_retention | ocr_char_error_rate | page_p95_ocr_char_error_rate | ocr_exact_text_match_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| china | PPOCRv5 + Gemma | 201.344 | 16.737 | 146.91 | 11959 | 41 | False | 1.0 | 1.0 | 0.9556 | 0.3333 | 0.3896 | 0.0667 |
| china | PaddleOCR VL + Gemma | 324.733 | 89.516 | 199.222 | 11975 | 25 | True | 1.0 | 1.0 | 1.0 | 0.0066 | 0.0138 | 0.8444 |
| china | HunyuanOCR + Gemma | 251.549 | 19.917 | 194.618 | 11973 | 27 | True | 1.0 | 1.0 | 1.0 | 0.0159 | 0.0345 | 0.8 |
| japan | MangaOCR + Gemma | 548.851 | 9.829 | 373.515 | 11952 | 48 | False | 1.0 | 0.9967 | 1.0 | 0.2879 | 0.4016 | 0.5663 |
| japan | PaddleOCR VL + Gemma | 984.152 | 451.397 | 453.344 | 11983 | 17 | False | 1.0 | 0.9967 | 1.0 | 0.0603 | 0.1452 | 0.7912 |
| japan | HunyuanOCR + Gemma | 561.275 | 53.832 | 429.325 | 11969 | 31 | False | 1.0 | 0.9967 | 1.0 | 0.177 | 0.3231 | 0.502 |

## Tuning Results

| corpus | stage | engine | elapsed_sec | ocr_total_sec | translate_total_sec | gpu_peak_used_mb | quality_gate_pass | geometry_match_recall | geometry_match_precision | non_empty_retention | ocr_char_error_rate | page_p95_ocr_char_error_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| china | parallel_workers | HunyuanOCR + Gemma | 270.771 | 25.227 | 209.724 | 11963 | True | 1.0 | 1.0 | 1.0 | 0.0159 | 0.0345 |
| china | parallel_workers | HunyuanOCR + Gemma | 245.219 | 20.364 | 188.928 | 11961 | True | 1.0 | 1.0 | 1.0 | 0.0159 | 0.0345 |
| china | parallel_workers | HunyuanOCR + Gemma | 262.407 | 23.115 | 204.037 | 11920 | True | 1.0 | 1.0 | 1.0 | 0.0159 | 0.0345 |
| china | max_completion_tokens | HunyuanOCR + Gemma | 228.678 | 18.404 | 177.077 | 11957 | True | 1.0 | 1.0 | 1.0 | 0.0159 | 0.0345 |
| china | max_completion_tokens | HunyuanOCR + Gemma | 207.884 | 17.394 | 158.19 | 11987 | True | 1.0 | 1.0 | 1.0 | 0.0159 | 0.0345 |
| china | n_gpu_layers | HunyuanOCR + Gemma | 133.841 | 11.988 | 88.854 | 11983 | True | 1.0 | 1.0 | 1.0 | 0.0159 | 0.0345 |
| china | n_gpu_layers | HunyuanOCR + Gemma | 134.536 | 13.26 | 87.429 | 11978 | True | 1.0 | 1.0 | 1.0 | 0.0159 | 0.0345 |
| china | parallel_workers | PaddleOCR VL + Gemma | 229.373 | 85.658 | 111.149 | 11977 | True | 1.0 | 1.0 | 1.0 | 0.0066 | 0.0138 |
| china | parallel_workers | PaddleOCR VL + Gemma | 225.392 | 88.027 | 105.219 | 11977 | True | 1.0 | 1.0 | 1.0 | 0.0066 | 0.0138 |
| china | max_new_tokens | PaddleOCR VL + Gemma | 228.48 | 91.081 | 105.242 | 11985 | True | 1.0 | 1.0 | 1.0 | 0.0066 | 0.0138 |
| china | max_new_tokens | PaddleOCR VL + Gemma | 229.704 | 90.477 | 106.457 | 11979 | True | 1.0 | 1.0 | 1.0 | 0.0066 | 0.0138 |
| china | max_concurrency | PaddleOCR VL + Gemma | 219.279 | 87.092 | 99.368 | 11990 | True | 1.0 | 1.0 | 1.0 | 0.0066 | 0.0138 |
| china | max_concurrency | PaddleOCR VL + Gemma | 229.457 | 89.867 | 105.282 | 11986 | True | 1.0 | 1.0 | 1.0 | 0.0066 | 0.0138 |
| china | gpu_memory_utilization | PaddleOCR VL + Gemma | 230.541 | 86.979 | 110.896 | 11977 | True | 1.0 | 1.0 | 1.0 | 0.0066 | 0.0138 |
| china | gpu_memory_utilization | PaddleOCR VL + Gemma | 227.671 | 90.846 | 103.705 | 11974 | True | 1.0 | 1.0 | 1.0 | 0.0066 | 0.0138 |

## Winners

| corpus | engine | official_score_elapsed_median_sec | ocr_median_sec | translate_median_sec | gpu_peak_used_mb | all_quality_gate_pass | promotion_recommended |
| --- | --- | --- | --- | --- | --- | --- | --- |
| china | HunyuanOCR + Gemma | 136.107 | 1.743 | 11.118 | 11977.0 | True | True |
| japan |  |  |  |  |  |  | False |

## Language Routing Policy

- China corpus 권장 OCR: `HunyuanOCR + Gemma`
- japan corpus 권장 OCR: `no winner`
- mixed corpus 운영 권장 라우팅: 중국어 페이지는 `HunyuanOCR + Gemma`, 일본어 페이지는 `no winner`로 라우팅하고 mixed corpus는 source language 판별 뒤 분기 권장

## Visual Appendix

### china
- page `0006_0005`
  source: `[local-only]`
  overlay: `[local-only]`
  winner_translated_image: `[local-only]`
  fastest_failed_translated_image: `[local-only]`
- page `0008_0007`
  source: `[local-only]`
  overlay: `[local-only]`
  winner_translated_image: `[local-only]`
  fastest_failed_translated_image: `[local-only]`

### japan
- page `094`
  source: `[local-only]`
  overlay: `[local-only]`
  winner_translated_image: ``
  fastest_failed_translated_image: `[local-only]`
- page `095`
  source: `[local-only]`
  overlay: `[local-only]`
  winner_translated_image: ``
  fastest_failed_translated_image: `[local-only]`

## Artifacts

- smoke csv: `./docs/assets/benchmarking/ocr-combo/latest/smoke_results.csv`
- default comparison csv: `./docs/assets/benchmarking/ocr-combo/latest/default_comparison.csv`
- tuning results csv: `./docs/assets/benchmarking/ocr-combo/latest/tuning_results.csv`
- winners csv: `./docs/assets/benchmarking/ocr-combo/latest/winners.csv`