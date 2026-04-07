# OCR Combo 벤치 사용법

## 공식 진입점

새 BAT를 만들지 않고 기존 launcher를 그대로 씁니다.

```bat
scripts\benchmark_suite_cuda13.bat --suite-profile ocr-combo-runtime
```

CUDA12 환경에서도 동일하게 profile만 바꿔 실행할 수 있습니다.

```bat
scripts\benchmark_suite.bat --suite-profile ocr-combo-runtime
```

## 실행 전제

- CUDA13 launcher는 `.venv-win-cuda13`
- raw 결과는 `./banchmark_result_log/ocr_combo/`
- benchmark 자산은 `benchmarking/lab`에만 둡니다.
- 공식 메타데이터는 아래로 고정합니다.
  - `execution_scope=full-pipeline`
  - `speed_score_scope=full-pipeline elapsed`
  - `quality_gate_scope=OCR-only`
  - `gold_source=human-reviewed`

## 첫 실행

처음 실행하면 locked gold가 없으므로 benchmark는 끝까지 돌지 않습니다.

1. China seed 생성
2. japan seed 생성
3. review packet 생성
4. latest report를 `awaiting_gold_review` 상태로 갱신하고 종료

그 다음 사용자가 gold를 검수해 `review_status=locked`로 저장해야 합니다.

## 이후 실행

locked gold가 있으면 같은 명령이 정식 benchmark mode로 동작합니다.

1. China/japan smoke
2. corpus별 default 비교
3. `PaddleOCR VL` / `HunyuanOCR` stepwise tuning
4. corpus winner `cold 3회` final confirm
5. latest report/history snapshot 갱신

## 보고서

- latest report: [ocr-combo-report-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/banchmark_report/ocr-combo-report-ko.md)
- latest assets: `docs/assets/benchmarking/ocr-combo/latest`
- history assets: `docs/assets/benchmarking/ocr-combo/history/<snapshot-id>`
- gold review guide: [gold-review-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/benchmark/ocr-combo/gold-review-ko.md)
