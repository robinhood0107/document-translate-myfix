# OCR Combo Ranked Report

이 파일은 `scripts/generate_ocr_combo_ranked_report.py`가 ranked suite manifest를 기준으로 갱신합니다.

## Metadata

- generated_at: `2026-04-08 11:41:25 대한민국 표준시`
- status: `benchmark_complete`
- benchmark_name: `OCR Combo Ranked Runtime Benchmark`
- benchmark_kind: `managed ranked family suite`
- benchmark_scope: `Japan full-pipeline OCR+Gemma timing with OCR-only quality bands and always-winner ranking`
- execution_scope: `full-pipeline`
- speed_score_scope: `full-pipeline elapsed_sec`
- quality_gate_scope: `OCR-only`
- gold_source: `human-reviewed`
- baseline_sha: `d4949bd985fdd10d6276d917134833eb12404edd`
- develop_ref_sha: `a0b06692a0229b30ad9bfde074e211f508ee68c6`
- entrypoint: `scripts\ocr_combo_ranked_benchmark_suite_cuda13.bat`
- results_root: `./banchmark_result_log/ocr_combo_ranked`

## Fixed Gemma

- image: `local/llama.cpp:server-cuda-b8665`
- response_format_mode: `json_schema`
- chunk_size: `6`
- temperature: `0.6`
- n_gpu_layers: `23`

## OCR Normalization

- canonical_small_voiced_kana: `True`
- ignored_chars: `「」『』,，、♡♥`
- gold_empty_text_policy: `geometry-kept-text-skipped`

## China Frozen Winner

- source_run_dir: `./banchmark_result_log/ocr_combo/20260408_005922_ocr-combo-runtime_suite`
- winner_engine: `HunyuanOCR + Gemma`
- winner_preset: `china-hunyuanocr--gemma-ngl80`
- official_score_elapsed_median_sec: `136.107`
- winner_status: `ready`
- promotion_recommended: `True`

## Japan Benchmark Winner

- benchmark_winner: `PaddleOCR VL + Gemma`
- winner_preset: `C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-paddleocr-vl--gemma-conc16.json`
- winner_status: `ready`
- promotion_recommended: `True`
- official_score_elapsed_median_sec: `779.051`
- ocr_median_sec: `17.153`
- translate_median_sec: `10.921`
- gpu_peak_used_mb: `11985.0`

## Mixed Routing Policy

- china: `HunyuanOCR + Gemma`
- japan: `PaddleOCR VL + Gemma`
- mixed: `중국어 페이지는 `HunyuanOCR + Gemma`, 일본어 페이지는 `PaddleOCR VL + Gemma`로 라우팅하고 mixed corpus는 source language 판별 뒤 분기 권장`

## Smoke Results

| corpus | engine | elapsed_sec | ocr_total_sec | translate_median_sec | gpu_peak_used_mb |
| --- | --- | --- | --- | --- | --- |
| japan | MangaOCR + Gemma | 28.214 | 2.839 | 16.644 | 11914 |
| japan | PaddleOCR VL + Gemma | 65.599 | 28.073 | 27.439 | 11983 |
| japan | HunyuanOCR + Gemma | 39.902 | 3.984 | 27.363 | 11938 |
| japan | PPOCRv5 + Gemma | 30.043 | 6.106 | 15.437 | 11952 |

## Japan Default Compare

| engine | preset | elapsed_sec | quality_band | ocr_char_error_rate | ocr_word_error_rate | page_p95_ocr_char_error_rate | overgenerated_block_rate | page_failed_count | gemma_truncated_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MangaOCR + Gemma | ocr-combo-ranked-japan-mangaocr-gemma | 306.516 | conditional | 0.1457 | 0.1457 | 0.3361 | 0.0033 | 0 | 0 |
| PaddleOCR VL + Gemma | ocr-combo-ranked-japan-paddleocr-vl-gemma | 819.797 | ready | 0.0603 | 0.0603 | 0.1452 | 0.0033 | 0 | 0 |
| HunyuanOCR + Gemma | ocr-combo-ranked-japan-hunyuanocr-gemma | 317.94 | conditional | 0.177 | 0.177 | 0.3231 | 0.0033 | 0 | 0 |
| PPOCRv5 + Gemma | ocr-combo-ranked-japan-ppocrv5-gemma | 320.417 | catastrophic | 0.629 | 0.629 | 0.9195 | 0.0033 | 0 | 0 |

## Japan Tuning Ladder

| engine | stage | preset | elapsed_sec | quality_band | ocr_char_error_rate | page_p95_ocr_char_error_rate | overgenerated_block_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MangaOCR + Gemma | tuning:expansion_percentage | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-mangaocr--gemma-exp3.json | 315.309 | hold | 0.2731 | 0.3361 | 0.0033 |
| MangaOCR + Gemma | tuning:expansion_percentage | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-mangaocr--gemma-exp5.json | 242.054 | conditional | 0.1457 | 0.3361 | 0.0033 |
| MangaOCR + Gemma | tuning:expansion_percentage | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-mangaocr--gemma-exp7.json | 240.715 | conditional | 0.1493 | 0.2903 | 0.0033 |
| PaddleOCR VL + Gemma | tuning:parallel_workers | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-paddleocr-vl--gemma-pw4.json | 765.686 | ready | 0.0603 | 0.1452 | 0.0033 |
| PaddleOCR VL + Gemma | tuning:parallel_workers | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-paddleocr-vl--gemma-pw8.json | 873.605 | ready | 0.0603 | 0.1452 | 0.0033 |
| PaddleOCR VL + Gemma | tuning:max_new_tokens | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-paddleocr-vl--gemma-mnt512.json | 821.225 | ready | 0.0608 | 0.1452 | 0.0033 |
| PaddleOCR VL + Gemma | tuning:max_new_tokens | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-paddleocr-vl--gemma-mnt1024.json | 848.605 | ready | 0.0603 | 0.1452 | 0.0033 |
| PaddleOCR VL + Gemma | tuning:max_concurrency | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-paddleocr-vl--gemma-conc16.json | 763.153 | ready | 0.0603 | 0.1452 | 0.0033 |
| PaddleOCR VL + Gemma | tuning:max_concurrency | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-paddleocr-vl--gemma-conc32.json | 774.207 | ready | 0.0603 | 0.1452 | 0.0033 |
| PaddleOCR VL + Gemma | tuning:gpu_memory_utilization | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-paddleocr-vl--gemma-vram080.json | 866.385 | ready | 0.0603 | 0.1452 | 0.0033 |
| PaddleOCR VL + Gemma | tuning:gpu_memory_utilization | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-paddleocr-vl--gemma-vram084.json | 874.572 | ready | 0.0603 | 0.1452 | 0.0033 |
| PaddleOCR VL + Gemma | tuning:crop_padding_ratio | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-paddleocr-vl--gemma-crop002.json | 857.292 | ready | 0.0617 | 0.1379 | 0.0033 |
| PaddleOCR VL + Gemma | tuning:crop_padding_ratio | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-paddleocr-vl--gemma-crop005.json | 840.012 | ready | 0.0603 | 0.1452 | 0.0033 |
| HunyuanOCR + Gemma | tuning:parallel_workers | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-hunyuanocr--gemma-pw1.json | 392.906 | conditional | 0.177 | 0.3231 | 0.0033 |
| HunyuanOCR + Gemma | tuning:parallel_workers | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-hunyuanocr--gemma-pw2.json | 317.922 | conditional | 0.177 | 0.3231 | 0.0033 |
| HunyuanOCR + Gemma | tuning:parallel_workers | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-hunyuanocr--gemma-pw4.json | 330.576 | conditional | 0.177 | 0.3231 | 0.0033 |
| HunyuanOCR + Gemma | tuning:max_completion_tokens | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-hunyuanocr--gemma-mct128.json | 360.58 | conditional | 0.177 | 0.3231 | 0.0033 |
| HunyuanOCR + Gemma | tuning:max_completion_tokens | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-hunyuanocr--gemma-mct256.json | 412.328 | conditional | 0.177 | 0.3231 | 0.0033 |
| HunyuanOCR + Gemma | tuning:n_gpu_layers | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-hunyuanocr--gemma-ngl80.json | 365.806 | conditional | 0.177 | 0.3231 | 0.0033 |
| HunyuanOCR + Gemma | tuning:n_gpu_layers | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-hunyuanocr--gemma-ngl99.json | 334.053 | conditional | 0.177 | 0.3231 | 0.0033 |
| HunyuanOCR + Gemma | tuning:crop_padding_ratio | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-hunyuanocr--gemma-crop002.json | 358.915 | hold | 0.3147 | 0.4648 | 0.0 |
| HunyuanOCR + Gemma | tuning:crop_padding_ratio | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-hunyuanocr--gemma-crop005.json | 348.938 | conditional | 0.177 | 0.3231 | 0.0033 |
| PPOCRv5 + Gemma | tuning:retry_crop_ratios | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-ppocrv5--gemma-retry0306.json | 302.009 | catastrophic | 0.6281 | 0.8889 | 0.0033 |
| PPOCRv5 + Gemma | tuning:retry_crop_ratios | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-ppocrv5--gemma-retry0610.json | 319.387 | catastrophic | 0.629 | 0.9195 | 0.0033 |
| PPOCRv5 + Gemma | tuning:crop_padding_ratio | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-ppocrv5--gemma-crop002.json | 322.735 | catastrophic | 0.6598 | 0.9394 | 0.0033 |
| PPOCRv5 + Gemma | tuning:crop_padding_ratio | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-ppocrv5--gemma-crop005.json | 317.134 | catastrophic | 0.6281 | 0.8889 | 0.0033 |

## Japan Engine Best Presets

| engine | preset | elapsed_sec | quality_band | ocr_char_error_rate | ocr_word_error_rate | page_p95_ocr_char_error_rate | overgenerated_block_rate | page_failed_count | gemma_truncated_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PaddleOCR VL + Gemma | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-paddleocr-vl--gemma-conc16.json | 763.153 | ready | 0.0603 | 0.0603 | 0.1452 | 0.0033 | 0 | 0 |
| MangaOCR + Gemma | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-mangaocr--gemma-exp7.json | 240.715 | conditional | 0.1493 | 0.1493 | 0.2903 | 0.0033 | 0 | 0 |
| HunyuanOCR + Gemma | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-hunyuanocr--gemma-pw2.json | 317.922 | conditional | 0.177 | 0.177 | 0.3231 | 0.0033 | 0 | 0 |
| PPOCRv5 + Gemma | C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\banchmark_result_log\ocr_combo_ranked\20260408_052525_ocr-combo-ranked-runtime_suite\_generated_presets\japan-ppocrv5--gemma-retry0306.json | 302.009 | catastrophic | 0.6281 | 0.6281 | 0.8889 | 0.0033 | 0 | 0 |

## Japan Final Confirm

| engine | stage | elapsed_sec | quality_band | ocr_char_error_rate | page_p95_ocr_char_error_rate | winner_status |
| --- | --- | --- | --- | --- | --- | --- |
| PaddleOCR VL + Gemma | final-confirm-r1 | 779.051 | ready | 0.0603 | 0.1452 | ready |
| PaddleOCR VL + Gemma | final-confirm-r2 | 775.553 | ready | 0.0603 | 0.1452 | ready |
| PaddleOCR VL + Gemma | final-confirm-r3 | 800.288 | ready | 0.0603 | 0.1452 | ready |

## Visual Appendix

- page `094`
  source: `[local-only]`
  overlay: `[local-only]`
  winner: `[local-only]`
  fastest: `[local-only]`
  lowest_cer: `[local-only]`
  ppocr: `[local-only]`
- page `095`
  source: `[local-only]`
  overlay: `[local-only]`
  winner: `[local-only]`
  fastest: `[local-only]`
  lowest_cer: `[local-only]`
  ppocr: `[local-only]`

## Crop Overread Regression

- regression page `p_018`
  winner_ocr_debug: `./banchmark_result_log/ocr_combo_ranked/20260408_052525_ocr-combo-ranked-runtime_suite/tuning/japan/PaddleOCR_VL_+_Gemma/max_concurrency/japan-conc16/corpus/comic_translate_Apr-08-2026_07-40-25AM/ocr_debugs/p_018_ocr_debug.json`
  ppocr_ocr_debug: `./banchmark_result_log/ocr_combo_ranked/20260408_052525_ocr-combo-ranked-runtime_suite/tuning/japan/PPOCRv5_+_Gemma/retry_crop_ratios/japan-retry0306/corpus/comic_translate_Apr-08-2026_10-29-15AM/ocr_debugs/p_018_ocr_debug.json`
  paddle_ocr_debug: `./banchmark_result_log/ocr_combo_ranked/20260408_052525_ocr-combo-ranked-runtime_suite/tuning/japan/PaddleOCR_VL_+_Gemma/max_concurrency/japan-conc16/corpus/comic_translate_Apr-08-2026_07-40-25AM/ocr_debugs/p_018_ocr_debug.json`

## Assets

- smoke csv: `./docs/assets/benchmarking/ocr-combo-ranked/latest/smoke_results.csv`
- default compare csv: `./docs/assets/benchmarking/ocr-combo-ranked/latest/default_compare.csv`
- tuning csv: `./docs/assets/benchmarking/ocr-combo-ranked/latest/tuning_results.csv`
- engine best csv: `./docs/assets/benchmarking/ocr-combo-ranked/latest/engine_best.csv`
- final confirm csv: `./docs/assets/benchmarking/ocr-combo-ranked/latest/final_confirm.csv`
- report summary json: `./docs/assets/benchmarking/ocr-combo-ranked/latest/report_summary.json`