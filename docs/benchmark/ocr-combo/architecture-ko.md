# OCR Combo 벤치 아키텍처

`ocr-combo` family는 기존 [benchmark_suite_cuda13.bat](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/scripts/benchmark_suite_cuda13.bat)을 공식 진입점으로 재사용하는 full-pipeline OCR 비교 harness입니다.

## 목적

- `Sample/China`와 `Sample/japan`을 분리해 언어별 OCR 권장 정책을 뽑습니다.
- Gemma 설정은 전 후보에서 고정합니다.
- 속도는 Gemma까지 포함한 실제 앱 end-to-end 시간으로 측정합니다.
- 품질 판정은 번역이 아니라 OCR만 사용합니다.

## execution_scope / speed_score_scope / quality_gate_scope

- `execution_scope`: `full-pipeline`
- `speed_score_scope`: `full-pipeline elapsed_sec`
- `quality_gate_scope`: `OCR-only`
- `gold_source`: `human-reviewed`

즉 실제 실행은 detect, OCR, translate, inpaint, render, save까지 모두 돌리고, 공식 점수는 전체 elapsed를 사용합니다. 다만 합격/탈락 게이트는 OCR geometry와 OCR text 품질만 봅니다.

## canonical OCR normalization

OCR text 비교는 사람 검수 gold를 그대로 byte-for-byte 비교하지 않습니다.

- 공백/줄바꿈은 제거합니다.
- 작은 가나, 탁점이 붙은 특수 가나, `ヴ` 계열은 같은 base 글자로 접어 비교합니다.
- `「」『』,，、♡♥`는 있어도 되고 없어도 되는 문자로 취급합니다.
- `gold_text=""` block은 geometry를 유지하되 non-empty OCR text hard gate에서는 제외합니다.

이 규칙은 OCR exact match와 CER 계산에 동일하게 적용됩니다.

## 비교군

### China

- `PPOCRv5 + Gemma`
- `PaddleOCR VL + Gemma`
- `HunyuanOCR + Gemma`

### japan

- `MangaOCR + Gemma`
- `PaddleOCR VL + Gemma`
- `HunyuanOCR + Gemma`

## 고정 Gemma

- image: `local/llama.cpp:server-cuda-b8665`
- response format: `json_schema`
- chunk size: `6`
- temperature: `0.6`
- `n_gpu_layers`: `23`

## OCR gold

공식 품질 기준은 machine-generated reference가 아니라 사람이 잠근 OCR gold입니다.

- 최초 실행 시 suite는 `PaddleOCR VL + Gemma` seed를 corpus별 1회 생성합니다.
- seed run에서 block overlay, `ocr_debug.json`, editable gold JSON을 포함한 review packet을 만듭니다.
- 사용자는 `gold_text`만 수정하고, geometry는 기본적으로 수정하지 않습니다.
- geometry 자체가 unusable한 페이지는 `status=excluded`로 표시합니다.
- 검수가 끝나면 `benchmarks/ocr_combo/gold/<corpus>/gold.json`의 `review_status`를 `locked`로 바꾸고 같은 명령을 다시 실행합니다.

## OCR-only hard gate

hard gate는 아래만 사용합니다.

- `page_failed_count = 0`
- `gemma_truncated_count = 0`
- `gemma_empty_content_count = 0`
- `gemma_missing_key_count = 0`
- `gemma_schema_validation_fail_count = 0`
- `ocr_cache_hit_count = 0`
- geometry match recall `>= 0.98`
- geometry match precision `>= 0.98`
- `non_empty_retention >= 0.98`
- `candidate_empty_rate <= gold_empty_rate + 0.02`
- `candidate_single_char_like_rate <= gold_single_char_like_rate + 0.02`
- `ocr_char_error_rate <= 0.02`
- `page_p95_ocr_char_error_rate <= 0.05`
- `overgenerated_block_count = 0`

translation similarity는 hard gate가 아니라 soft metric으로만 남깁니다.

## crop overreach 회귀

- OCR crop은 `xyxy` 우선입니다.
- `bubble_xyxy`는 OCR 입력을 넓히는 기본 crop이 아니라 clamp boundary로만 사용합니다.
- 대표 회귀 케이스는 `p_018`처럼 같은 말풍선 안 다른 block까지 같이 읽어버리는 경우입니다.

## 튜닝 원칙

- `PPOCRv5`와 `MangaOCR`는 internal GPU ONNX baseline 고정
- `PaddleOCR VL`, `HunyuanOCR`만 bounded stepwise tuning
- 각 축 winner만 다음 축으로 진입
- 최종 confirm은 corpus winner만 `cold 3회`

## 결과 루트

- raw 결과: `./banchmark_result_log/ocr_combo/`
- locked gold: `benchmarks/ocr_combo/gold/<corpus>/gold.json`
- latest report: [ocr-combo-report-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/banchmark_report/ocr-combo-report-ko.md)
