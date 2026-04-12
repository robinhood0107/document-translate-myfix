# Gemma IQ4_NL Japan Full-Pipeline Full-GPU Benchmark

## Metadata

- generated_at: `2026-04-11 16:19:59 대한민국 표준시`
- suite_dir: `./banchmark_result_log/gemma_iq4nl_japan/20260411_160014_gemma_iq4nl_japan_fullgpu_suite`
- git_sha: `43c165a07cd0441e22163db484fc31b3da5d9af6`
- sample_dir: `./Sample/japan`
- sample_count: `22`
- official_score_scope: `full-pipeline batch on Sample/japan (22 pages)`

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

- run_dir: `./banchmark_result_log/gemma_iq4nl_japan/20260411_160014_gemma_iq4nl_japan_fullgpu_suite/smoke/baseline-ov072-ngl18/attempt03_t05`
- detector_backend: `RTDetrV2ONNXDetection`
- detector_device: `cuda`
- ocr_front_device: `cuda`
- ctd_device: `cuda`
- inpainter_device: `cuda`
- gemma_loaded_model_ids: `['gemma-4-26B-IQ4_NL.gguf']`
- passed_hard_gate: `True`

## Recommendation

