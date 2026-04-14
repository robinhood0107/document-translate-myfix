# PaddleOCR VL Parallel Architecture

## 계층

- 제품 runtime:
  - `modules/ocr/ocr_paddle_VL.py`
  - `modules/utils/gpu_metrics.py`
  - `pipeline/batch_processor.py`
- benchmark harness:
  - `scripts/paddleocr_vl_parallel_benchmark.py`
  - `scripts/generate_paddleocr_vl_parallel_report.py`
- 문서/결과:
  - `docs/benchmark/paddleocr-vl-parallel/`
  - `docs/assets/benchmarking/paddleocr-vl-parallel/`
  - `banchmark_result_log/paddleocr_vl_parallel/`

## 데이터 흐름

1. base preset에서 candidate별 hidden scheduler mode와 worker cap을 합성한다.
2. `benchmark_pipeline.py`를 `stage_ceiling=ocr`, `runtime_services=ocr-only`로 실행한다.
3. pipeline이 `metrics.jsonl`과 `page_snapshots.json`를 남긴다.
4. family harness가 `ocr_page_profile`을 분리해 `page_profiles.jsonl`, `request_events.jsonl`을 만든다.
5. baseline gold와 compare를 수행해 quality summary를 만든다.
6. suite summary와 report generator가 latest assets 및 portfolio docs를 갱신한다.
