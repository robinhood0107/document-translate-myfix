# OCR Combo 벤치 워크플로

## 1. launcher

공식 launcher는 [benchmark_suite_cuda13.bat](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/scripts/benchmark_suite_cuda13.bat)이며, `ocr-combo-runtime` profile로 실행합니다.

## 2. bootstrap mode

locked gold가 없으면 suite는 benchmark를 끝까지 돌리지 않고 아래까지만 수행합니다.

- China seed 생성: `PaddleOCR VL + Gemma`
- japan seed 생성: `PaddleOCR VL + Gemma`
- gold review packet 생성
- latest report를 `awaiting_gold_review` 상태로 갱신

이 단계의 목적은 사람 검수 OCR gold를 잠그는 것입니다.

## 3. gold review

- `benchmarks/ocr_combo/gold/<corpus>/gold.json`을 엽니다.
- source image와 overlay, `ocr_debug`를 보고 block별 `gold_text`를 수정합니다.
- geometry가 unusable한 페이지는 `status=excluded`로 표시합니다.
- 검수가 끝나면 `review_status=locked`로 저장합니다.

세부 절차는 [gold-review-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/benchmark/ocr-combo/gold-review-ko.md)를 따릅니다.

## 4. benchmark mode

locked gold가 있으면 같은 명령으로 아래를 끝까지 수행합니다.

1. China/japan smoke
2. corpus별 default compare
3. hard gate 통과 후보만 유지
4. `PaddleOCR VL` / `HunyuanOCR` stepwise tuning
5. corpus winner `cold 3회` final confirm
6. latest report/history snapshot 갱신

## 5. 산출물

- suite manifest
- corpus별 smoke/default/tuning/final confirm raw 결과
- locked gold review packet
- latest report + history snapshot
- language-aware routing policy
