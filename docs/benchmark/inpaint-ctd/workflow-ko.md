# Inpaint CTD Workflow

- 제품 파이프라인은 `RT-DETR-v2 -> CTD refined mask -> protect mask -> Torch CUDA inpainter`를 기준으로 한다.
- benchmark family는 위 제품 구조를 유지하되, detector만 CPU로 고정하고 CTD/AOT/LaMa는 Torch CUDA로 측정한다.
- OCR/번역은 corpus별 고정값을 사용한다.
  - China: `HunyuanOCR + Gemma`
  - japan: `PaddleOCR VL + Gemma`
- spotlight 2장과 `/Sample` 전체 30장 full suite를 각각 5-way로 돌린다.
- 결과는 `banchmark_result_log/inpaint_ctd/` 아래 raw run으로 남기고, 검수용 요약은 `docs/banchmark_report/inpaint-ctd-report-ko.md`와 `docs/assets/benchmarking/inpaint-ctd/`로 정리한다.
- 최신 accepted suite는 `20260410_091232_inpaint_ctd_suite`이고, japan corpus는 OCR invariance `FAIL`이므로 수동 검수 우선순위를 높게 둔다.
