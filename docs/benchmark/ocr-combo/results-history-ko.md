# OCR Combo 결과 이력

이 문서는 `ocr-combo` family의 latest 상태와 운영 해석을 기록합니다.

## latest

- report: [ocr-combo-report-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/banchmark_report/ocr-combo-report-ko.md)
- raw run: `./banchmark_result_log/ocr_combo/20260407_211341_ocr-combo-runtime_suite`
- official entrypoint: `scripts\benchmark_suite_cuda13.bat --suite-profile ocr-combo-runtime`
- status: `awaiting_gold_review`
- execution_scope: `full-pipeline`
- speed_score_scope: `full-pipeline elapsed`
- quality_gate_scope: `OCR-only`
- gold_source: `human-reviewed`

latest 상태는 generated report를 기준으로 읽습니다. 현재 latest는 bootstrap mode가 정상 종료된 상태이며, China/japan seed run과 review packet이 생성되었습니다. 다음 단계는 `benchmarks/ocr_combo/gold/<corpus>/gold.json`을 검수해 `review_status=locked`로 바꾸는 것입니다.

## 해석 원칙

- 결과는 China/japan corpus를 분리해서 읽습니다.
- 공식 속도 점수는 Gemma까지 포함한 전체 elapsed입니다.
- 공식 품질 게이트는 OCR geometry와 OCR text 품질만 사용합니다.
- 번역 결과는 디버그 참고용이지 hard gate가 아닙니다.
- benchmark 자산은 계속 `benchmarking/lab`에만 보존합니다.

## 이전 라운드 메모

- `2026-04-07`의 초기 convergence run은 translation similarity를 hard gate에 넣어 과도하게 판정한 사례였습니다.
- 현재 family는 그 문제를 피하기 위해 OCR-only hard gate와 human-reviewed gold 기반으로 재설계됐습니다.
