# OCR Combo 벤치 아키텍처

`ocr-combo` family는 기존 [benchmark_suite_cuda13.bat](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/scripts/benchmark_suite_cuda13.bat)을 공식 진입점으로 재사용하는 full-pipeline OCR 비교 harness입니다.

## 목적

- `Sample/China`와 `Sample/japan`을 분리해 언어별 OCR 권장 정책을 뽑습니다.
- Gemma 설정은 전 후보에서 고정합니다.
- OCR 엔진 차이만 비교하고, 최종 산출물은 단일 글로벌 winner가 아니라 언어별 winner입니다.

## execution_scope / official_score_scope

- `execution_scope`: `full-pipeline`
- `official_score_scope`: `full-pipeline cold median elapsed_sec + geometry/quality/translation hard gate`

즉 실제 실행은 detect, OCR, translate, inpaint, render, save까지 모두 돌리고, 공식 비교 점수도 전체 elapsed를 사용합니다.

## 비교군

### China

- `PPOCRv5 + Gemma`
- `PaddleOCR VL + Gemma`
- `HunyuanOCR + Gemma`

### japan

- `MangaOCR + Gemma`
- `PaddleOCR VL + Gemma`
- `HunyuanOCR + Gemma`

## 기준 reference

- China reference: `PaddleOCR VL + fixed Gemma`
- japan reference: `PaddleOCR VL + fixed Gemma`

reference는 매 라운드 fresh 생성합니다.

## 품질 게이트

hard gate는 아래를 동시에 만족해야 합니다.

- `page_failed_count = 0`
- Gemma failure counter 전부 0
- `ocr_cache_hit_count = 0`
- geometry match recall `>= 0.95`
- `non_empty_retention >= 0.98`
- empty/single-char-like rate 무회귀
- page-level `normalized_translation` similarity 평균 `>= 0.98`
- overgenerated translation block `0`

OCR exact text 전수 일치는 hard gate가 아니라 soft metric으로 남깁니다.

## 튜닝 원칙

- `PPOCRv5`와 `MangaOCR`는 internal GPU ONNX baseline 고정
- `PaddleOCR VL`, `HunyuanOCR`만 bounded stepwise tuning
- 각 축 winner만 다음 축으로 진입
- 최종 confirm은 corpus winner만 `cold 3회`

## 결과 루트

- raw 결과: `./banchmark_result_log/ocr_combo/`
- latest report: [ocr-combo-report-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/banchmark_report/ocr-combo-report-ko.md)
