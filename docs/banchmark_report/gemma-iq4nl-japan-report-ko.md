# Gemma IQ4_NL Japan Full-Pipeline Full-GPU Benchmark

## Metadata

- generated_at: `2026-04-12 19:58:42 대한민국 표준시`
- suite_dir: `./banchmark_result_log/gemma_iq4nl_japan/20260411_171639_gemma_iq4nl_japan_fullgpu_suite`
- git_sha: `43c165a07cd0441e22163db484fc31b3da5d9af6`
- sample_dir: `./Sample/japan`
- sample_count: `22`
- official_score_scope: `full-pipeline batch on Sample/japan (22 pages)`
- suite_status: `completed`
- last_failure_kind: ``
- last_failure_reason: ``

## Fixed Pipeline

- translator: `Custom Local Server(Gemma)`
- ocr: `PaddleOCR VL`
- detector: `RT-DETR-v2`
- inpainter: `lama_large_512px`
- mask_refiner: `ctd`
- use_gpu: `True`
- ocr_front_device: `cuda`
- detector_device: `cuda`
- ctd_device: `cuda`
- inpainter_device: `cuda`
- gemma_model: `gemma-4-26B-IQ4_NL.gguf`
- requested_temperature: `0.7`

## Stage 0 Smoke

- run_dir: `./banchmark_result_log/gemma_iq4nl_japan/20260411_171639_gemma_iq4nl_japan_fullgpu_suite/smoke/baseline-ov072-ngl18/attempt01_t07_infra01`
- detector_backend: `RTDetrV2ONNXDetection`
- detector_device: `cuda`
- ocr_front_device: `cuda`
- ctd_device: `cuda`
- inpainter_device: `cuda`
- gemma_loaded_model_ids: `['gemma-4-26B-IQ4_NL.gguf']`
- passed_hard_gate: `True`

## Stage 1 Shared-GPU Coarse Grid

- winner_candidate: `ov08-ngl23`
- runner_up_candidate: `ov08-ngl22`
- winner_reason: `selected fastest hard-gate-passing candidate`

| candidate | status | temp | infra_attempts | elapsed | detect_p50 | ocr_p50 | translate_p50 | inpaint_p50 | gpu_floor_free_mb | oom_rescue | issues |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ov068-ngl14 | passed | 0.600 | 1 | 2632.192 | 0.780 | 20.432 | 88.301 | 3.146 | 131 | - | - |
| ov068-ngl16 | passed | 0.700 | 1 | 1871.466 | 0.655 | 16.825 | 53.742 | 3.105 | 79 | - | - |
| ov068-ngl18 | passed | 0.700 | 1 | 1653.387 | 0.833 | 19.174 | 44.136 | 3.156 | 27 | - | - |
| ov068-ngl20 | passed | 0.700 | 1 | 1553.597 | 1.281 | 18.512 | 41.645 | 3.177 | 12 | - | - |
| ov068-ngl22 | passed | 0.700 | 1 | 1417.897 | 1.145 | 18.409 | 35.335 | 3.196 | 8 | - | - |
| ov068-ngl23 | passed | 0.700 | 1 | 1403.564 | 1.635 | 17.654 | 34.372 | 2.928 | 9 | - | - |
| ov072-ngl14 | passed | 0.700 | 2 | 1942.822 | 0.648 | 17.256 | 58.052 | 3.240 | 39 | - | - |
| ov072-ngl16 | passed | 0.700 | 2 | 1922.773 | 0.681 | 18.732 | 58.626 | 3.072 | 68 | - | - |
| ov072-ngl18 | passed | 0.700 | 1 | 1798.925 | 1.061 | 18.305 | 51.488 | 3.229 | 36 | - | - |
| ov072-ngl20 | passed | 0.700 | 1 | 1520.610 | 1.123 | 17.842 | 39.253 | 3.126 | 13 | - | - |
| ov072-ngl22 | passed | 0.700 | 1 | 1404.367 | 1.646 | 17.656 | 34.442 | 3.301 | 15 | - | - |
| ov072-ngl23 | passed | 0.700 | 1 | 1373.196 | 1.517 | 18.422 | 30.907 | 3.303 | 10 | - | - |
| ov076-ngl14 | passed | 0.700 | 2 | 2282.597 | 0.578 | 18.871 | 72.185 | 2.903 | 283 | - | - |
| ov076-ngl16 | passed | 0.600 | 1 | 2065.875 | 0.734 | 20.569 | 60.917 | 3.401 | 39 | - | - |
| ov076-ngl18 | passed | 0.700 | 1 | 1968.816 | 1.271 | 20.176 | 51.157 | 3.374 | 31 | - | - |
| ov076-ngl20 | passed | 0.700 | 1 | 1741.225 | 1.544 | 18.862 | 44.711 | 3.721 | 50 | - | - |
| ov076-ngl22 | passed | 0.700 | 1 | 1634.221 | 1.243 | 19.035 | 42.254 | 3.469 | 31 | - | - |
| ov076-ngl23 | passed | 0.700 | 1 | 1693.197 | 1.525 | 19.833 | 42.176 | 3.270 | 9 | - | - |
| ov08-ngl14 | passed | 0.500 | 1 | 1242.492 | 0.567 | 17.642 | 28.301 | 3.124 | 132 | - | - |
| ov08-ngl16 | passed | 0.700 | 1 | 1179.316 | 0.504 | 18.802 | 26.230 | 3.090 | 82 | - | - |
| ov08-ngl18 | passed | 0.700 | 1 | 1189.276 | 0.801 | 18.872 | 26.494 | 3.060 | 22 | - | - |
| ov08-ngl20 | passed | 0.700 | 1 | 1135.166 | 1.296 | 17.627 | 22.796 | 3.204 | 12 | - | - |
| ov08-ngl22 | passed | 0.700 | 1 | 1109.715 | 1.138 | 19.291 | 21.122 | 3.256 | 15 | - | - |
| ov08-ngl23 | passed | 0.700 | 1 | 1109.261 | 0.942 | 19.167 | 20.710 | 3.495 | 12 | - | - |


## Stage 2 Context Size Sweep

- winner_candidate: `ctx4096`
- runner_up_candidate: `ctx3072`
- winner_reason: `selected fastest hard-gate-passing candidate`

| candidate | status | temp | infra_attempts | elapsed | detect_p50 | ocr_p50 | translate_p50 | inpaint_p50 | gpu_floor_free_mb | oom_rescue | issues |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ctx3072 | passed | 0.700 | 2 | 1151.252 | 0.706 | 19.742 | 23.836 | 3.272 | 9 | - | - |
| ctx4096 | passed | 0.700 | 1 | 1094.258 | 1.123 | 18.853 | 21.620 | 3.261 | 8 | - | - |


## Stage 3 Chunk Size Sweep

- winner_candidate: `chunk6`
- runner_up_candidate: `chunk5`
- winner_reason: `selected fastest hard-gate-passing candidate`

| candidate | status | temp | infra_attempts | elapsed | detect_p50 | ocr_p50 | translate_p50 | inpaint_p50 | gpu_floor_free_mb | oom_rescue | issues |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chunk4 | passed | 0.700 | 1 | 1114.308 | 0.961 | 18.406 | 21.942 | 2.843 | 13 | - | - |
| chunk5 | passed | 0.700 | 1 | 1093.186 | 0.972 | 18.591 | 20.643 | 3.224 | 12 | - | - |
| chunk6 | passed | 0.700 | 1 | 1076.805 | 1.126 | 19.370 | 19.711 | 3.122 | 12 | - | - |


## Stage 4 Threads Sweep

- winner_candidate: `threads10`
- runner_up_candidate: `threads12`
- winner_reason: `selected fastest hard-gate-passing candidate`

| candidate | status | temp | infra_attempts | elapsed | detect_p50 | ocr_p50 | translate_p50 | inpaint_p50 | gpu_floor_free_mb | oom_rescue | issues |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| threads10 | passed | 0.700 | 1 | 950.625 | 0.958 | 18.746 | 13.563 | 3.757 | 8 | - | - |
| threads12 | passed | 0.700 | 1 | 1109.207 | 1.076 | 18.731 | 21.454 | 3.180 | 9 | - | - |
| threads14 | passed | 0.700 | 1 | 1294.394 | 1.227 | 19.590 | 32.142 | 3.045 | 12 | - | - |


## Stage 5 Max Completion Tokens Sweep

- winner_candidate: `max384`
- runner_up_candidate: `max512`
- winner_reason: `selected fastest hard-gate-passing candidate`

| candidate | status | temp | infra_attempts | elapsed | detect_p50 | ocr_p50 | translate_p50 | inpaint_p50 | gpu_floor_free_mb | oom_rescue | issues |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| max384 | passed | 0.700 | 1 | 930.731 | 1.095 | 18.342 | 12.910 | 3.344 | 9 | - | - |
| max512 | passed | 0.400 | 1 | 963.814 | 1.251 | 19.663 | 15.286 | 3.160 | 10 | - | - |
| max640 | passed | 0.700 | 1 | 1018.525 | 1.286 | 18.942 | 16.917 | 3.678 | 18 | - | - |


## Final Confirm

- winner_candidate: `max512`
- runner_up_candidate: `max384`
- winner_reason: `selected fastest hard-gate-passing candidate`

| candidate | status | temp | infra_attempts | elapsed | detect_p50 | ocr_p50 | translate_p50 | inpaint_p50 | gpu_floor_free_mb | oom_rescue | issues |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| max384 | passed | 0.700 | 1 | 1002.261 | 1.429 | 18.257 | 15.249 | 3.713 | 12.000 | - | - |
| max512 | passed | 0.700 | 1 | 954.214 | 1.195 | 19.613 | 14.275 | 3.195 | 14.000 | - | - |


## Recommendation

- winner: `max512`
- winner_reason: `selected fastest hard-gate-passing candidate`
- requested_temperature: `0.7`
- effective_temperature: `0.7`
- elapsed_sec: `954.214`
- detect_median_sec: `1.195`
- ocr_median_sec: `19.613`
- translate_median_sec: `14.275`
- inpaint_median_sec: `3.195`
- gpu_floor_free_mb: `14.000`
- tuning: `{'ocr_gpu_memory_utilization': 0.8, 'gemma_n_gpu_layers': 23, 'context_size': 4096, 'chunk_size': 6, 'threads': 10, 'max_completion_tokens': 512}`
- candidate_status: `passed`
- infra_attempt_count: `1`
- derived_from: ``
- oom_rescue_changes: `{}`
- runner_up: `max384`
- runner_up_elapsed_sec: `1002.261`
- runner_up_gpu_floor_free_mb: `12.000`
