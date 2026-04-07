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
2. crop debug (`effective_crop_xyxy`, `retry_crop_xyxy`, `crop_source`) 확인
3. corpus별 default compare
4. hard gate 통과 후보만 유지
5. `PaddleOCR VL` / `HunyuanOCR` stepwise tuning
6. corpus winner `cold 3회` final confirm
7. latest report/history snapshot 갱신

## 5. 현재 판정 원칙

- 속도는 Gemma까지 포함한 full pipeline elapsed입니다.
- 품질 게이트는 OCR-only입니다.
- translation similarity는 참고 지표일 뿐 hard gate가 아닙니다.
- `gold_text=""` block은 geometry를 유지하고 텍스트 hard gate만 제외합니다.
- crop overreach는 `xyxy` 우선 crop과 bubble clamp 회귀로 확인합니다.

## 6. 산출물

- suite manifest
- corpus별 smoke/default/tuning/final confirm raw 결과
- locked gold review packet
- latest report + history snapshot
- language-aware routing policy
