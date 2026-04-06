# PaddleOCR-VL 1.5 벤치 사용법

## 런처

- CUDA12 pipeline: [paddleocr_vl15_benchmark_pipeline.bat](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/scripts/paddleocr_vl15_benchmark_pipeline.bat)
- CUDA13 pipeline: [paddleocr_vl15_benchmark_pipeline_cuda13.bat](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/scripts/paddleocr_vl15_benchmark_pipeline_cuda13.bat)
- CUDA12 suite: [paddleocr_vl15_benchmark_suite.bat](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/scripts/paddleocr_vl15_benchmark_suite.bat)
- CUDA13 suite: [paddleocr_vl15_benchmark_suite_cuda13.bat](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/scripts/paddleocr_vl15_benchmark_suite_cuda13.bat)

## 기본 전제

- CUDA12 런처는 `.venv-win`
- CUDA13 런처는 `.venv-win-cuda13`
- raw 결과는 `./banchmark_result_log/paddleocr_vl15/`

## 단일 실행

```bat
scripts\paddleocr_vl15_benchmark_pipeline_cuda13.bat
scripts\paddleocr_vl15_benchmark_pipeline_cuda13.bat run paddleocr-vl15-baseline one-page managed 1
```

## gold 생성 / compare

```bat
scripts\paddleocr_vl15_benchmark_pipeline_cuda13.bat gold --run-dir <run-dir>
scripts\paddleocr_vl15_benchmark_pipeline_cuda13.bat compare --baseline-gold <gold.json> --candidate-run-dir <run-dir>
```

## full suite

```bat
scripts\paddleocr_vl15_benchmark_suite_cuda13.bat
```

이 명령은 내부적으로 `benchmark_suite.py --suite-profile paddleocr-vl15-runtime`을 호출합니다.

## 보고서 생성

```bat
scripts\paddleocr_vl15_benchmark_pipeline_cuda13.bat summary --manifest <suite-manifest>
```

최신 보고서는 [paddleocr-vl15-report-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/banchmark_report/paddleocr-vl15-report-ko.md)에, 이전 보고서는 `history/` 아래에 보존됩니다.
