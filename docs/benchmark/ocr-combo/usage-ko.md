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

## 수행 흐름

1. China/japan smoke
2. China/japan fresh reference 생성
3. corpus별 default 비교
4. `PaddleOCR VL` / `HunyuanOCR` stepwise tuning
5. corpus winner `cold 3회` final confirm
6. latest report/history snapshot 갱신

## 보고서

- latest report: [ocr-combo-report-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/banchmark_report/ocr-combo-report-ko.md)
- latest assets: `docs/assets/benchmarking/ocr-combo/latest`
- history assets: `docs/assets/benchmarking/ocr-combo/history/<snapshot-id>`
