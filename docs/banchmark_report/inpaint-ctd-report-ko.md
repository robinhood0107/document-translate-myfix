# Inpaint CTD Benchmark Report

## Metadata

- generated_at: `2026-04-10 10:39:51 대한민국 표준시`
- git_sha: `3529961881b3055d9c7a37f2b3fe3fbfdfc9587a`
- suite_dir: `./banchmark_result_log/inpaint_ctd/20260410_091232_inpaint_ctd_suite`
- results_root: `./banchmark_result_log/inpaint_ctd`
- execution_scope: `full-pipeline`
- speed_score_scope: `full-pipeline elapsed`
- quality_gate_scope: `OCR invariance + manual inpaint review`

## Fixed Runtime

- detector: `RT-DETR-v2 (CPU pinned in CUDA13 benchmark family)`
- China OCR: `HunyuanOCR + Gemma`
- japan OCR: `PaddleOCR VL + Gemma`
- default recommendation: `ctd + protect + AOT`
- quality mode: `ctd + protect + lama_large_512px`
- offline review mode: `ctd + protect + lama_mpe`
- note: CUDA13 benchmark family pins RT-DETR-v2 to CPU because the current ONNX CUDA path regresses with CuDNN internal errors in this environment.

## Previous Latest Archive

- snapshot_id: `20260410_103951_inpaint-ctd`
- report: `./docs/banchmark_report/history/20260410_103951_inpaint-ctd/inpaint-ctd-report-ko.md`
- assets: `./docs/assets/benchmarking/inpaint-ctd/history/20260410_103951_inpaint-ctd`

## China Spotlight 5-way

| case | elapsed_sec | ocr_median_sec | translate_median_sec | inpaint_median_sec | gpu_peak_used_mb | gpu_floor_free_mb | cleanup_applied_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| legacy_bbox + AOT | 11.341 | 1.448 | 4.537 | 2.267 | 11928 | 72 | 0 |
| ctd + AOT | 14.811 | 1.05 | 6.526 | 4.236 | 11885 | 115 | 1 |
| ctd + protect + AOT | 17.529 | 1.457 | 7.367 | 5.391 | 11907 | 93 | 1 |
| ctd + protect + lama_large_512px | 17.215 | 1.26 | 5.539 | 6.968 | 11921 | 79 | 1 |
| ctd + protect + lama_mpe | 20.615 | 1.133 | 9.879 | 6.111 | 11890 | 110 | 1 |

- OCR invariance (spotlight): `PASS`

## China Full Summary

| case | elapsed_sec | detect_median_sec | ocr_median_sec | translate_median_sec | inpaint_median_sec | gpu_peak_used_mb | gpu_floor_free_mb | cleanup_applied_count | page_failed_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| legacy_bbox + AOT | 103.269 | 0.583 | 1.449 | 7.612 | 3.94 | 11916 | 84 | 3 | 0 |
| ctd + AOT | 104.546 | 0.589 | 1.477 | 7.479 | 4.299 | 11941 | 59 | 4 | 0 |
| ctd + protect + AOT | 99.562 | 0.568 | 1.4 | 6.343 | 4.687 | 11935 | 65 | 8 | 0 |
| ctd + protect + lama_large_512px | 106.789 | 0.566 | 1.393 | 6.777 | 5.038 | 11981 | 19 | 8 | 0 |
| ctd + protect + lama_mpe | 115.813 | 0.566 | 1.435 | 6.458 | 5.705 | 11964 | 36 | 8 | 0 |

- OCR invariance (full): `PASS`

## Japan Spotlight 5-way

| case | elapsed_sec | ocr_median_sec | translate_median_sec | inpaint_median_sec | gpu_peak_used_mb | gpu_floor_free_mb | cleanup_applied_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| legacy_bbox + AOT | 58.582 | 35.196 | 14.156 | 5.096 | 11853 | 147 | 1 |
| ctd + AOT | 55.34 | 27.049 | 15.906 | 7.246 | 11900 | 100 | 1 |
| ctd + protect + AOT | 55.606 | 25.698 | 16.677 | 8.771 | 11919 | 81 | 1 |
| ctd + protect + lama_large_512px | 53.758 | 26.185 | 14.618 | 8.576 | 11926 | 74 | 1 |
| ctd + protect + lama_mpe | 59.811 | 24.607 | 17.298 | 13.648 | 11737 | 263 | 1 |

- OCR invariance (spotlight): `FAIL`

## Japan Full Summary

| case | elapsed_sec | detect_median_sec | ocr_median_sec | translate_median_sec | inpaint_median_sec | gpu_peak_used_mb | gpu_floor_free_mb | cleanup_applied_count | page_failed_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| legacy_bbox + AOT | 790.208 | 0.707 | 16.742 | 11.396 | 3.359 | 11979 | 21 | 22 | 0 |
| ctd + AOT | 752.216 | 0.688 | 16.863 | 9.596 | 3.693 | 11973 | 27 | 22 | 0 |
| ctd + protect + AOT | 796.062 | 0.717 | 16.148 | 10.136 | 4.399 | 11971 | 29 | 22 | 0 |
| ctd + protect + lama_large_512px | 809.5 | 0.694 | 16.514 | 10.903 | 4.411 | 11978 | 22 | 22 | 0 |
| ctd + protect + lama_mpe | 794.335 | 0.69 | 16.849 | 9.955 | 4.953 | 11966 | 34 | 22 | 0 |

- OCR invariance (full): `FAIL`
- japan FAIL 메모: 블록 수 붕괴나 페이지 실패가 아니라, 일부 페이지에서 마스킹/인페인팅 변화에 따라 OCR 텍스트가 소폭 달라지는 사례가 확인되었습니다. 대표 예시는 `094.png`의 `この本は、私の本を読んでみた。 -> この本は、私の本を読んでみたのだ。`, `p_018.jpg`의 `お -> ぶ`, `p_021.jpg`의 `ぎ -> おい`입니다.

## Operations Recommendation

- 기본값: `ctd + protect + AOT`
- 수동 품질 모드: `ctd + protect + lama_large_512px`
- 오프라인 검수 전용: `ctd + protect + lama_mpe`
- VRAM headroom은 측정 peak보다 10~15% 이상 남기는 것을 권장합니다.

## Visual Samples Policy

- 이 benchmark의 시각 샘플, 원본 컷, detector/mask overlay, cleaned/translated 결과 이미지는 Git에 포함하지 않는다.
- spotlight 계열 자산은 로컬 전용으로만 유지하며, 문서/차트/요약 통계만 브랜치에 남긴다.
- 품질 검수는 로컬 benchmark export 또는 별도 외부 공유 채널에서 수행한다.
