# OCR Combo 결과 이력

이 문서는 `ocr-combo` family의 latest 상태와 운영 해석을 기록합니다.

## latest

- report: [ocr-combo-report-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/banchmark_report/ocr-combo-report-ko.md)
- raw run: `./banchmark_result_log/ocr_combo/20260408_005922_ocr-combo-runtime_suite`
- official entrypoint: `scripts\benchmark_suite_cuda13.bat --suite-profile ocr-combo-runtime`
- status: `benchmark_complete`
- execution_scope: `full-pipeline`
- speed_score_scope: `full-pipeline elapsed`
- quality_gate_scope: `OCR-only`
- gold_source: `human-reviewed`
- crop_regression_focus: `xyxy-first OCR crop with bubble clamp; p_018 overread regression`

latest 상태는 generated report를 기준으로 읽습니다. 현재 latest는 locked gold 기준 정식 suite가 끝난 상태이며, bootstrap mode는 더 이상 active latest가 아닙니다.

## latest 결론

- China corpus 권장 OCR: `HunyuanOCR + Gemma`
- China 공식 winner preset: `china-hunyuanocr--gemma-ngl80`
- China final confirm elapsed median: `136.107s`
- japan corpus 권장 OCR: `no winner`
- mixed corpus 운영 권장 라우팅:
  - 중국어 페이지는 `HunyuanOCR + Gemma`
  - 일본어 페이지는 현재 `no winner`
  - mixed corpus는 source language 판별 뒤 분기 권장

## latest 해석

- China는 `PPOCRv5 + Gemma`가 가장 빨랐지만 OCR-only hard gate를 통과하지 못했습니다.
  - `non_empty_retention<0.98`
  - `ocr_char_error_rate>0.02`
  - `page_p95_ocr_char_error_rate>0.05`
- China에서는 `PaddleOCR VL + Gemma`와 `HunyuanOCR + Gemma`가 둘 다 gate를 통과했고, stepwise tuning과 final confirm 결과 `HunyuanOCR + Gemma (n_gpu_layers=80)`가 가장 빨랐습니다.
- japan은 세 후보 모두 hard gate를 통과하지 못했습니다.
  - `MangaOCR + Gemma`: `page_failed_count=1`, `gemma_truncated_count=3`, OCR CER 초과, overgenerated block 발생
  - `PaddleOCR VL + Gemma`: `ocr_char_error_rate=0.0603`, `page_p95_ocr_char_error_rate=0.1452`, overgenerated block 발생
  - `HunyuanOCR + Gemma`: `candidate_single_char_like_rate` 초과, OCR CER 초과, overgenerated block 발생
- 따라서 이번 latest에서는 China만 winner가 있고, japan은 promotion-ready OCR이 없습니다.

## 해석 원칙

- 결과는 China/japan corpus를 분리해서 읽습니다.
- 공식 속도 점수는 Gemma까지 포함한 전체 elapsed입니다.
- 공식 품질 게이트는 OCR geometry와 OCR text 품질만 사용합니다.
- 번역 결과는 디버그 참고용이지 hard gate가 아닙니다.
- benchmark 자산은 계속 `benchmarking/lab`에만 보존합니다.

## 이전 라운드 메모

- `2026-04-07`의 초기 convergence run은 translation similarity를 hard gate에 넣어 과도하게 판정한 사례였습니다.
- 현재 family는 그 문제를 피하기 위해 OCR-only hard gate와 human-reviewed gold 기반으로 재설계됐습니다.
- `2026-04-08` latest run은 crop overreach 회귀와 canonical OCR normalization을 반영한 첫 정식 운영 실행입니다.
