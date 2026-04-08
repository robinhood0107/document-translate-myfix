# OCR Combo Ranked Architecture

- family: `ocr-combo-ranked`
- execution_scope: `full-pipeline`
- speed_score_scope: `full-pipeline elapsed_sec`
- quality_gate_scope: `OCR-only`
- gold_source: `human-reviewed`

## 목적

`ocr-combo-ranked`는 strict `ocr-combo-runtime`의 historical evidence를 보존한 채, Japan OCR 후보를 **전체 실행 + 품질 등급 + 조건부 winner** 구조로 다시 평가하는 family다.

## 핵심 구조

- China는 재실행하지 않는다.
- China winner는 strict run의 frozen manifest를 재사용한다.
- Japan는 아래 4후보를 모두 실행한다.
  - `MangaOCR + Gemma`
  - `PaddleOCR VL + Gemma`
  - `HunyuanOCR + Gemma`
  - `PPOCRv5 + Gemma`
- 속도는 Gemma까지 포함한 full-pipeline elapsed로 계산한다.
- 품질은 사람 검수 OCR gold 기준 OCR-only 비교로 판정한다.

## 품질 평가

- canonical normalization:
  - small/voiced kana -> base glyph
  - `「」『』,，、♡♥` 무시
  - `gold_text=""`는 geometry 유지, non-empty OCR text 비교 제외
- 품질 결과는 `ready`, `conditional`, `hold`, `catastrophic` 밴드로 분류한다.
- `benchmark_winner`는 항상 1개 만든다.
- `promotion_recommended`는 `winner_status == ready`일 때만 `true`다.

## 결과 산출물

- raw results: `./banchmark_result_log/ocr_combo_ranked/...`
- report: `docs/banchmark_report/ocr-combo-ranked-report-ko.md`
- history: `docs/benchmark/ocr-combo-ranked/results-history-ko.md`
