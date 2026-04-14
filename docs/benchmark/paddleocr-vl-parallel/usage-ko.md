# PaddleOCR VL Parallel Usage

## 기본 실행

```bash
./.venv-win/Scripts/python.exe scripts/paddleocr_vl_parallel_benchmark.py
```

## smoke 실행

```bash
./.venv-win/Scripts/python.exe scripts/paddleocr_vl_parallel_benchmark.py --smoke
```

## 리포트 재생성

```bash
./.venv-win/Scripts/python.exe scripts/generate_paddleocr_vl_parallel_report.py
```

## 주요 출력

- raw suite: `banchmark_result_log/paddleocr_vl_parallel/<suite>/`
- latest assets: `docs/assets/benchmarking/paddleocr-vl-parallel/latest/`
- report: `docs/banchmark_report/paddleocr-vl-parallel-report-ko.md`
