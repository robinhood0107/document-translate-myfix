# PaddleOCR VL Parallel Report

## Metadata

- latest suite dir: `./banchmark_result_log/paddleocr_vl_parallel/20260415_031602_paddleocr-vl-parallel-smoke`
- latest assets dir: `./docs/assets/benchmarking/paddleocr-vl-parallel/latest`
- runtime_contract: `paddleocr-vl-single-tenant-ocr-only`
- runtime_services: `ocr-only`
- stage_ceiling: `ocr`
- baseline gold: `./banchmark_result_log/paddleocr_vl_parallel/20260415_031602_paddleocr-vl-parallel-smoke/baseline_gold.json`
- detector manifest: `./banchmark_result_log/paddleocr_vl_parallel/20260415_031602_paddleocr-vl-parallel-smoke/detector_manifest.json`
- runtime container names: `['paddleocr-server', 'paddleocr-vllm']`
- gemma-local-server booted: `False`

## Quality Gate Winner

- quality_gate_winner: `fixed_area_desc_w8`
- scheduler_mode: `fixed_area_desc`
- parallel_workers: `8`
- ocr_total_sec_median: `292.526`
- ocr_page_p95_sec_median: `43.66`

## Promotion Status

- final_promotion_status: `approved_fixed_area_desc_w8`
- approved_promotion_candidate: `fixed_area_desc_w8`
- 사용자 OCR diff 검수 승인이 완료되었고, develop 기본값 승격 대상이 확정되었다.
- 이번 승격은 `PaddleOCR VL 단독 상주 상한선 benchmark` 결과를 기준으로 한다.

## Review Candidates

- top1: `fixed_area_desc_w8` (ocr_total_sec_median=`292.526`, gate_pass=`True`)
- top2: `auto_v1_cap4` (ocr_total_sec_median=`295.765`, gate_pass=`True`)

## Candidate Table

| candidate | scheduler | workers | ocr_total_sec_median | ocr_page_p95_sec_median | mean_CER | mean_exact_match | gate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_area_desc_w8 | fixed_area_desc | 8 | 292.526 | 43.66 | 0.0 | 1.0 | True |
| fixed_w8 | fixed | 8 | 293.048 | 42.082 | 0.0 | 1.0 | True |
| auto_v1_cap4 | auto_v1 | 4 | 295.765 | 40.871 | 0.0 | 1.0 | True |

## Visual Assets

- OCR total chart: `./docs/assets/benchmarking/paddleocr-vl-parallel/latest/ocr_total_sec_median.svg`
- OCR page p95 chart: `./docs/assets/benchmarking/paddleocr-vl-parallel/latest/ocr_page_p95_sec_median.svg`
- Mean CER chart: `./docs/assets/benchmarking/paddleocr-vl-parallel/latest/quality_mean_cer.svg`
- Candidate table: `./docs/assets/benchmarking/paddleocr-vl-parallel/latest/candidate_table.md`
- top1 diff review: `./banchmark_result_log/paddleocr_vl_parallel/20260415_031602_paddleocr-vl-parallel-smoke/review/top1_diff_review.md`
- top2 diff review: `./banchmark_result_log/paddleocr_vl_parallel/20260415_031602_paddleocr-vl-parallel-smoke/review/top2_diff_review.md`

## Portfolio Notes

- 핵심 문제 해결 방향은 사용자가 착안했다.
- 실험 설계 구체화, 계측 설계, 구현, 검증은 공동 수행으로 정리한다.
- 이번 결과는 Gemma/MangaLMM 혼재 benchmark가 아니라 PaddleOCR VL 단독 상주 상한선 benchmark다.
- 향후 MangaLMM + PaddleOCR VL 동시 상주 환경에서 PaddleOCR VL이 더 큰 VRAM headroom을 얻을 수 있다는 가정 아래, 단독 상주 최대 병렬치를 먼저 확정하는 문맥으로 해석한다.
- 세부 narrative는 `docs/benchmark/paddleocr-vl-parallel/`, latest problem-solving specs, review diff pack을 함께 본다.
