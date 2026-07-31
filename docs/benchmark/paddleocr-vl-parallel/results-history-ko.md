# PaddleOCR VL Parallel Results History

## Current Policy

- baseline은 항상 `fixed_w8`이다.
- detector freeze와 gold seed는 baseline measured run에서 생성한다.
- 이번 family는 `PaddleOCR VL 단독 상주 상한선 benchmark`다.
- `runtime_services=ocr-only`, `stage_ceiling=ocr` 계약에서 Gemma는 실제 runtime/VRAM 점유에 참여하지 않는다.
- subset winner만으로 자동 `default on` 승격을 하지 않는다. 사용자 OCR diff 검수 승인 상태를 기록한 뒤 develop promotion을 진행한다.

## Latest Output

- latest suite root: `./banchmark_result_log/paddleocr_vl_parallel/<run-id>_031602_paddleocr-vl-parallel-smoke`
- quality_gate_winner: `fixed_area_desc_w8`
- scheduler_mode: `fixed_area_desc`
- ocr_total_sec_median: `292.526`
- final_promotion_status: `approved_fixed_area_desc_w8`
- approved_promotion_candidate: `fixed_area_desc_w8`
- review_candidate_keys: `['fixed_area_desc_w8', 'auto_v1_cap4']`
- baseline gold: `./banchmark_result_log/paddleocr_vl_parallel/<run-id>_031602_paddleocr-vl-parallel-smoke/baseline_gold.json`
- detector manifest: `./banchmark_result_log/paddleocr_vl_parallel/<run-id>_031602_paddleocr-vl-parallel-smoke/detector_manifest.json`

## 2026-08-01 Folder-global Queue Final Decision

- candidate: `folder-global queue + worker 4`
- baseline: page barrier + worker 8
- actual runtime: `.venv-win-cuda13`, Stage-Batched OCR ceiling
- workload: 일본어 3페이지, 실행당 실제 OCR HTTP 60회
- measured rounds: 7 paired rounds
- snapshot equality: 7/7
- baseline median: 66.613초
- candidate median: 66.442초
- nominal median improvement: 0.257%
- one-sided 95% bootstrap lower bound: -0.891%
- warm rounds mean improvement: -0.521%
- decision: `reject_no_proven_speed_gain`
- product action: 기존 페이지 단위 장벽과 worker 8 유지

초기 2.8904% 관측치는 반복 CUDA 실행에서 재현되지 않았다. 원시 결과는 Git 밖에 보존하며, 상세 판정은 [folder-global queue CUDA 최종 판정](./folder-global-queue-final-decision-ko.md)에 기록한다.
