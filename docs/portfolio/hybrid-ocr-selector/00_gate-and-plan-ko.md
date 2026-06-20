# Hybrid OCR Selector Gate And Status

## 상태

- status: `failed_closed`
- branch_of_record: `benchmarking/lab`

## 결론

`MangaLMM` hybrid selector는 이번 승격 범위에서 제외한다.

이유:

1. 시간
   - `PaddleOCR VL` OCR-only가 `MangaLMM` OCR-only보다 빨랐다.
2. 품질
   - detector block 대비 `text_bubble` 누락과 merge/split 오류가 반복됐다.
3. 제품 리스크
   - sidecar/selector route를 기본 workflow에 얹을 근거가 부족하다.

## develop에 남길 것

- 없음

## develop에 남기지 않을 것

- selector runtime
- selector logging
- review pack
- threshold UI

이번 단계의 제품 승격은 `PaddleOCR VL 중심 stage_batched_pipeline`까지만 진행한다.
