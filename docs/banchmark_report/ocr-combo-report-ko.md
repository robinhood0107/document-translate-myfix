# OCR Combo Benchmark Report

이 파일은 `scripts/generate_ocr_combo_report.py`가 suite manifest를 기준으로 덮어씁니다.

- execution_scope: `full-pipeline`
- official_score_scope: `full-pipeline cold median elapsed_sec + geometry/quality/translation hard gate`
- 공식 실행 진입점: `scripts\benchmark_suite_cuda13.bat --suite-profile ocr-combo-runtime`
- raw 결과 루트: `./banchmark_result_log/ocr_combo/`
