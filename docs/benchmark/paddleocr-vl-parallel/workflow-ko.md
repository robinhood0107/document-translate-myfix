# PaddleOCR VL Parallel Workflow

- execution_scope: `detect-ocr-only`
- corpus: `Sample/japan_vllm_parallel_subset`
- pages: `13`
- baseline_origin: `fixed + workers=8 + original order`
- run_shape: `warmup 1 + measured 3`

## 실행 순서

1. OCR-only Paddle runtime health를 먼저 확인한다.
2. baseline 후보 `fixed_w8` measured run으로 detector freeze manifest와 baseline gold를 만든다.
3. 후보 행렬을 `fixed`, `fixed_area_desc`, `auto_v1` 순서로 돈다.
4. 각 run마다 `summary.json`, `metrics.jsonl`, `page_snapshots.json`, `gpu_samples.jsonl`, `page_profiles.jsonl`, `request_events.jsonl`을 저장한다.
5. baseline gold 기준으로 CER, exact match, empty block delta를 계산한다.
6. 품질 게이트를 통과한 후보 중 `ocr_total_sec` median이 가장 빠른 후보를 1차 winner로 정한다.
7. suite 종료 후 latest report, 문제 해결 명세서, 포트폴리오 문서를 갱신한다.
