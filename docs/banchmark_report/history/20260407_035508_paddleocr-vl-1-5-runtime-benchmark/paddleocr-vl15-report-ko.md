# PaddleOCR-VL 1.5 벤치마크 보고서

이 파일은 `scripts/generate_paddleocr_vl15_report.py`가 최신 suite manifest를 기준으로 덮어씁니다.

- 실제 실행 범위: offscreen 앱 파이프라인 전체
- 공식 점수 범위: detect + ocr
- raw 결과 루트: `./banchmark_result_log/paddleocr_vl15/`

아직 최신 suite 보고서가 생성되지 않았다면, 먼저 아래를 실행하세요.

```bat
scripts\paddleocr_vl15_benchmark_suite_cuda13.bat
scripts\paddleocr_vl15_benchmark_pipeline_cuda13.bat summary --manifest <suite-manifest>
```
