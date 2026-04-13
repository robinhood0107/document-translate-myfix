# OCR Simpletest MangaLMM vs PaddleOCR VL Workflow

- execution_scope: `full-pipeline`
- corpus: `Sample/simpletest`
- pages: `p_016.jpg`, `p_017.jpg`, `p_021.jpg`
- run_shape: `cold 1 + warm 2`

## 실행 순서

1. empty baseline GPU snapshot을 기록한다.
2. 후보별로 OCR-only idle snapshot을 기록한다.
3. 후보별로 Gemma+OCR full idle snapshot을 기록한다.
4. full runtime을 유지한 채 `cold1 -> warm1 -> warm2` 순서로 batch full-pipeline을 돌린다.
5. warm 2회의 `elapsed_sec` median으로 속도 우열을 정한다.
6. translated images와 raw summary를 보고 사용자가 품질을 최종 검수한다.

## 판단 기준

- 1차 승부는 `warm_median_elapsed_sec`
- 동률이면 `warm_total_page_failed_count`가 낮은 후보 우선
- Gemma 설정은 고정하고 OCR만 바꾼다.
- resident VRAM 수치는 idle snapshot으로만 비교한다.
