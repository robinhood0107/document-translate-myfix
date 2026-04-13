# OCR Simpletest MangaLMM vs PaddleOCR VL Usage

## 기본 실행

Windows:

```bat
scripts\ocr_simpletest_mangalmm_vs_paddle_benchmark_pipeline.bat
```

CUDA 13:

```bat
scripts\ocr_simpletest_mangalmm_vs_paddle_benchmark_pipeline_cuda13.bat
```

직접 Python 실행:

```bash
python scripts/ocr_simpletest_mangalmm_vs_paddle_benchmark.py run --sample-dir ./Sample/simpletest
```

## 요약 다시 만들기

```bash
python scripts/ocr_simpletest_mangalmm_vs_paddle_benchmark.py summary --suite-dir <suite-dir>
```

## 출력 위치

- root: `./banchmark_result_log/ocr_simpletest_mangalmm_vs_paddle/`
- per-suite summary:
  - `comparison_summary.json`
  - `comparison_summary.md`
- per-candidate:
  - `candidate_summary.json`
  - `candidate_summary.md`
- per-run:
  - `summary.json`
  - `summary.md`
  - `docker_logs/`

## 주의

- 기본 sample dir는 `./Sample/simpletest`다.
- 샘플 3장이 로컬에 없으면 `--sample-dir`로 직접 지정해야 한다.
- 이 family는 Gemma 설정을 고정한 상태에서 OCR 엔진 두 개만 비교한다.
