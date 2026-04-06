# 자동번역 벤치마크 보고서 - b8665 Gemma 4 Parser Translation Optimization

이 문서는 `./banchmark_result_log`에 있는 실제 run 결과를 기준으로 자동 생성됩니다.

## 보고서 메타데이터

- 생성 시각: `2026-04-06 10:14:05 대한민국 표준시`
- 벤치마킹 이름: `b8665 Gemma 4 Parser Translation Optimization`
- 벤치마킹 종류: `managed benchmark sweep`
- 벤치마킹 범위: `old-image vs b8665, json_object vs json_schema, chunk_size sweep, temperature sweep, n_gpu_layers sweep`
- build id: `b8665`
- active image: `local/llama.cpp:server-cuda-b8665`
- winning format: `json_schema`
- winning chunk: `6`
- winning temperature: `0.6`
- winning n_gpu_layers: `23`

## Gemma 4 Verification

- verification status: `PASS`
- verification dir: `./banchmark_result_log/20260406_054332_b8665-gemma4_suite/_server_verification`
- container image: `local/llama.cpp:server-cuda-b8665`
- checks: `image_matches=True`, `build_marker_found=True`, `arch_gemma4_found=True`, `tool_response_eog_found=True`, `object_smoke_ok=True`, `schema_smoke_ok=True`

## 판단 요약

- b8665-schema-ch6-t06-ngl23가 batch elapsed `934.053s`, translate median `9.705s`, retry `0`, missing key `0`, truncated `0`로 가장 균형이 좋았습니다.
- old-image baseline batch run: `./banchmark_result_log/20260406_054332_b8665-gemma4_suite/02_old_image_batch`
- winning candidate run: `./banchmark_result_log/20260406_054332_b8665-gemma4_suite/10_b8665-schema-ch6-t06-ngl23_batch`
- old-image baseline 대비 elapsed delta: `75.524s`

## Old Image vs b8665 Control 비교

| control_label | preset | response_format_mode | image | elapsed_sec | translate_median_sec | gemma_json_retry_count | gemma_missing_key_count | gemma_truncated_count | run_dir_rel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b8665 json_schema | b8665-schema-control | json_schema | local/llama.cpp:server-cuda-b8665 | 973.183 | 10.225 | 0 | 0 | 0 | ./banchmark_result_log/20260406_054332_b8665-gemma4_suite/06_b8665_schema_batch |
| old image baseline | translation-old-image-baseline | json_object | local/llama.cpp:server-cuda-pre-b8665 | 1009.577 | 10.959 | 1 | 0 | 1 | ./banchmark_result_log/20260406_054332_b8665-gemma4_suite/02_old_image_batch |
| b8665 json_object | b8665-object-control | json_object | local/llama.cpp:server-cuda-b8665 | 1054.138 | 12.663 | 1 | 0 | 1 | ./banchmark_result_log/20260406_054332_b8665-gemma4_suite/04_b8665_object_batch |

![control](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/assets/benchmarking/latest/b8665_control_elapsed_comparison.png)

## Representative Batch Finalists

| preset | response_format_mode | chunk_size | temperature | n_gpu_layers | elapsed_sec | translate_median_sec | gemma_json_retry_count | gemma_missing_key_count | gemma_truncated_count | audit_passed | run_dir_rel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b8665-schema-ch6-t06-ngl23 | json_schema | 6 | 0.600 | 23 | 934.053 | 9.705 | 0 | 0 | 0 | True | ./banchmark_result_log/20260406_054332_b8665-gemma4_suite/10_b8665-schema-ch6-t06-ngl23_batch |
| b8665-schema-ch4-t06-ngl23 | json_schema | 4 | 0.600 | 23 | 950.813 | 10.290 | 0 | 0 | 0 | True | ./banchmark_result_log/20260406_054332_b8665-gemma4_suite/10_b8665-schema-ch4-t06-ngl23_batch |
| b8665-schema-ch6-t08-ngl23 | json_schema | 6 | 0.800 | 23 | 964.892 | 11.278 | 0 | 0 | 0 | True | ./banchmark_result_log/20260406_054332_b8665-gemma4_suite/23_b8665-schema-ch6-t08-ngl23_batch |
| b8665-schema-control | json_schema | 4 | 0.600 | 23 | 973.183 | 10.225 | 0 | 0 | 0 | True | ./banchmark_result_log/20260406_054332_b8665-gemma4_suite/06_b8665_schema_batch |
| b8665-schema-ch6-t065-ngl23 | json_schema | 6 | 0.650 | 23 | 1016.760 | 11.872 | 0 | 0 | 0 | True | ./banchmark_result_log/20260406_054332_b8665-gemma4_suite/27_b8665-schema-ch6-t065-ngl23_batch |
| b8665-schema-ch6-t06-ngl24 | json_schema | 6 | 0.600 | 24 | 1030.479 | 11.809 | 0 | 0 | 0 | True | ./banchmark_result_log/20260406_054332_b8665-gemma4_suite/33_b8665-schema-ch6-t06-ngl24_batch |
| b8665-schema-ch6-t06-ngl25 | json_schema | 6 | 0.600 | 25 | 1101.646 | 14.981 | 0 | 0 | 0 | True | ./banchmark_result_log/20260406_054332_b8665-gemma4_suite/33_b8665-schema-ch6-t06-ngl25_batch |
| translation-old-image-baseline | json_object | 4 | 0.600 | 23 | 1009.577 | 10.959 | 1 | 0 | 1 | None | ./banchmark_result_log/20260406_054332_b8665-gemma4_suite/02_old_image_batch |
| b8665-object-control | json_object | 4 | 0.600 | 23 | 1054.138 | 12.663 | 1 | 0 | 1 | True | ./banchmark_result_log/20260406_054332_b8665-gemma4_suite/04_b8665_object_batch |
| b8665-schema-ch6-t055-ngl23 | json_schema | 6 | 0.550 | 23 | 1070.784 | 11.318 | 0 | 0 | 3 | True | ./banchmark_result_log/20260406_054332_b8665-gemma4_suite/27_b8665-schema-ch6-t055-ngl23_batch |

## Chunk Sweep

| preset | chunk_size | elapsed_sec | translate_median_sec | gemma_json_retry_count | gemma_missing_key_count | run_dir_rel |
| --- | --- | --- | --- | --- | --- | --- |
| b8665-schema-ch4-t06-ngl23 | 4 | 39.554 | 11.207 | 0 | 0 | ./banchmark_result_log/20260406_054332_b8665-gemma4_suite/07_chunk_schema_ch4_one_page |
| b8665-schema-ch5-t06-ngl23 | 5 | 40.350 | 12.104 | 0 | 0 | ./banchmark_result_log/20260406_054332_b8665-gemma4_suite/07_chunk_schema_ch5_one_page |
| b8665-schema-ch6-t06-ngl23 | 6 | 40.179 | 11.446 | 0 | 0 | ./banchmark_result_log/20260406_054332_b8665-gemma4_suite/07_chunk_schema_ch6_one_page |

![chunk](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/assets/benchmarking/latest/b8665_chunk_translate_median.png)

## Temperature Sweep

| preset | temperature | elapsed_sec | translate_median_sec | gemma_json_retry_count | gemma_missing_key_count | run_dir_rel |
| --- | --- | --- | --- | --- | --- | --- |
| b8665-schema-ch6-t05-ngl23 | 0.500 | 40.967 | 11.845 | 0 | 0 | ./banchmark_result_log/20260406_054332_b8665-gemma4_suite/20_temp_schema_t05_one_page |
| b8665-schema-ch6-t055-ngl23 | 0.550 | 44.252 | 11.318 | 0 | 0 | ./banchmark_result_log/20260406_054332_b8665-gemma4_suite/26_temp_fine_t055_one_page |
| b8665-schema-ch6-t06-ngl23 | 0.600 | 38.196 | 11.254 | 0 | 0 | ./banchmark_result_log/20260406_054332_b8665-gemma4_suite/20_temp_schema_t06_one_page |
| b8665-schema-ch6-t065-ngl23 | 0.650 | 40.396 | 12.156 | 0 | 0 | ./banchmark_result_log/20260406_054332_b8665-gemma4_suite/26_temp_fine_t065_one_page |
| b8665-schema-ch6-t07-ngl23 | 0.700 | 38.341 | 11.362 | 0 | 0 | ./banchmark_result_log/20260406_054332_b8665-gemma4_suite/20_temp_schema_t07_one_page |
| b8665-schema-ch6-t08-ngl23 | 0.800 | 37.659 | 11.015 | 0 | 0 | ./banchmark_result_log/20260406_054332_b8665-gemma4_suite/20_temp_schema_t08_one_page |

![temperature](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/assets/benchmarking/latest/b8665_temperature_translate_median.png)

## n_gpu_layers Sweep

| preset | n_gpu_layers | elapsed_sec | translate_median_sec | gemma_json_retry_count | gemma_missing_key_count | run_dir_rel |
| --- | --- | --- | --- | --- | --- | --- |
| b8665-schema-ch6-t06-ngl23 | 23 | 38.196 | 11.254 | 0 | 0 | ./banchmark_result_log/20260406_054332_b8665-gemma4_suite/20_temp_schema_t06_one_page |
| b8665-schema-ch6-t06-ngl24 | 24 | 47.673 | 15.179 | 0 | 0 | ./banchmark_result_log/20260406_054332_b8665-gemma4_suite/30_ngl_24_one_page |
| b8665-schema-ch6-t06-ngl25 | 25 | 44.013 | 14.743 | 0 | 0 | ./banchmark_result_log/20260406_054332_b8665-gemma4_suite/30_ngl_25_one_page |

![ngl](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/assets/benchmarking/latest/b8665_n_gpu_layers_translate_median.png)

## 품질 지표 비교

| preset | gemma_json_retry_count | gemma_missing_key_count | gemma_truncated_count | gemma_empty_content_count | gemma_reasoning_without_final_count | ocr_empty_rate | ocr_low_quality_rate | audit_passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b8665-schema-ch6-t06-ngl23 | 0 | 0 | 0 | 0 | 0 | 0.000 | 0.040 | True |
| b8665-schema-ch4-t06-ngl23 | 0 | 0 | 0 | 0 | 0 | 0.000 | 0.040 | True |
| b8665-schema-ch6-t08-ngl23 | 0 | 0 | 0 | 0 | 0 | 0.000 | 0.040 | True |
| b8665-schema-control | 0 | 0 | 0 | 0 | 0 | 0.000 | 0.040 | True |
| b8665-schema-ch6-t065-ngl23 | 0 | 0 | 0 | 0 | 0 | 0.000 | 0.040 | True |
| b8665-schema-ch6-t06-ngl24 | 0 | 0 | 0 | 0 | 0 | 0.000 | 0.040 | True |
| b8665-schema-ch6-t06-ngl25 | 0 | 0 | 0 | 0 | 0 | 0.000 | 0.040 | True |
| translation-old-image-baseline | 1 | 0 | 1 | 0 | 0 | 0.000 | 0.040 | None |
| b8665-object-control | 1 | 0 | 1 | 0 | 0 | 0.000 | 0.040 | True |
| b8665-schema-ch6-t055-ngl23 | 0 | 0 | 3 | 0 | 0 | 0.000 | 0.040 | True |

![quality](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/assets/benchmarking/latest/b8665_quality_metrics_comparison.png)
