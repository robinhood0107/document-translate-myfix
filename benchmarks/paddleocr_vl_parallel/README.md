# PaddleOCR VL Parallel Benchmark

OCR-only benchmark family for tuning `PaddleOCR VL` block-crop scheduling against the curated `Sample/japan_vllm_parallel_subset` corpus.

## Family Goals

- Keep detector behavior fixed and compare only the OCR stage.
- Seed a baseline answer sheet from current shipping behavior.
- Compare `fixed`, `fixed_area_desc`, and `auto_v1` scheduler modes.
- Preserve raw suite output under `banchmark_result_log/paddleocr_vl_parallel/`.

## Key Files

- `presets/paddleocr-vl-parallel-base.json`
- `../../scripts/paddleocr_vl_parallel_benchmark.py`
- `../../scripts/generate_paddleocr_vl_parallel_report.py`

## Output Model

- Raw suite output: `banchmark_result_log/paddleocr_vl_parallel/<suite>/`
- Latest copied assets: `docs/assets/benchmarking/paddleocr-vl-parallel/latest/`
- Family docs: `docs/benchmark/paddleocr-vl-parallel/`
