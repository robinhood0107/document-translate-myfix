# OCR Combo Ranked Results History

- execution_scope: `full-pipeline`
- quality_gate_scope: `OCR-only`
- gold_source: `human-reviewed`

## Current Policy

- China는 strict family frozen winner를 재사용한다.
- Japan는 ranked family에서 항상 `benchmark_winner`를 만든다.
- `promotion_recommended`는 `winner_status == ready`일 때만 `true`다.

## History

- latest report:
  - `docs/banchmark_report/ocr-combo-ranked-report-ko.md`
- latest assets:
  - `docs/assets/benchmarking/ocr-combo-ranked/latest`
- history snapshots:
  - `docs/banchmark_report/history/<snapshot-id>/ocr-combo-ranked-report-ko.md`
  - `docs/assets/benchmarking/ocr-combo-ranked/history/<snapshot-id>/...`
