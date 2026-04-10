# OCR Combo Gold Review Guide

이 문서는 `ocr-combo-runtime`의 사람 검수 OCR gold를 어떻게 잠그는지 설명합니다.

## 어디를 열어야 하나

각 corpus별 gold 파일은 아래에 있습니다.

- `benchmarks/ocr_combo/gold/china/gold.json`
- `benchmarks/ocr_combo/gold/japan/gold.json`

review packet은 최신 run 아래에 생성됩니다.

- `banchmark_result_log/ocr_combo/<timestamp>_ocr-combo-runtime_suite/gold-review/china/`
- `banchmark_result_log/ocr_combo/<timestamp>_ocr-combo-runtime_suite/gold-review/japan/`

대표적으로 보는 파일은 아래 4개입니다.

- `source_images/<page>.png`
- `overlay_images/<page>_overlay.png`
- `ocr_debugs/<page>_ocr_debug.json`
- `editable_gold.json`

## 무엇을 판단하나

판단 기준은 번역이 아니라 OCR입니다.

- 줄바꿈/공백은 크게 중요하지 않습니다.
- 이미지에 **실제로 보이는 문자 자체가 맞는지**만 봅니다.
- 번역 자연스러움은 보지 않습니다.

## 무엇을 수정하나

block마다 `gold_text`만 수정합니다.

- seed OCR이 맞으면 그대로 둡니다.
- seed OCR이 틀렸거나 빠졌으면 `gold_text`를 고칩니다.
- geometry는 기본적으로 수정하지 않습니다.
- block을 통째로 지우지 않습니다.
- 너무 인식이 어렵거나, 품질 비교에서 텍스트를 강제하고 싶지 않은 block은 `gold_text=""`로 둡니다.

## 어떤 경우를 허용하나

OCR 비교는 exact glyph 복제가 아니라 canonical OCR normalization을 사용합니다.

- `あ゙`, `ぁ`, `あ`는 모두 `あ`로 비교합니다.
- `お゙`, `ぉ`, `お`도 같은 방식으로 base 글자로 접습니다.
- `ヴ`는 `ウ` 계열로 비교합니다.
- `「」『』,，、♡♥`는 있어도 되고 없어도 됩니다.

즉 정확히 같은 글리프가 아니어도, 이번 benchmark 규칙에서 허용한 변형이면 정답으로 봅니다.

## false positive / 배경 오검출은 어떻게 하나

- 배경을 글자로 잘못 잡았더라도 block을 삭제하지 않습니다.
- 그 block을 benchmark 구조에 남겨야 geometry 비교가 꼬이지 않습니다.
- 이런 경우는 보통 `gold_text=""`로 두고 텍스트 hard gate에서 제외합니다.

## 언제 exclude를 쓰나

페이지 자체가 아래 상태면 `status=excluded`를 사용합니다.

- block 위치가 전반적으로 완전히 틀림
- 말풍선 구조가 무너져 text comparison 자체가 무의미함
- seed geometry를 유지한 채 `gold_text` 수정만으로는 복구가 안 됨

이 경우 `exclude_reason`도 같이 적습니다.

## 검수 완료 처리

검수가 끝나면 corpus gold 파일의 아래 값을 바꿉니다.

```json
"review_status": "locked"
```

그 다음 같은 명령을 다시 실행합니다.

```bat
scripts\benchmark_suite_cuda13.bat --suite-profile ocr-combo-runtime
```

그러면 suite가 bootstrap 모드가 아니라 정식 benchmark mode로 끝까지 수행됩니다.
