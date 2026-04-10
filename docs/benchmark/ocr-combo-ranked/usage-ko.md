# OCR Combo Ranked Usage

- execution_scope: `full-pipeline`
- speed_score_scope: `full-pipeline elapsed_sec`
- quality_gate_scope: `OCR-only`
- gold_source: `human-reviewed`

## Launchers

- CUDA12:
  - `scripts\ocr_combo_ranked_benchmark_suite.bat`
- CUDA13:
  - `scripts\ocr_combo_ranked_benchmark_suite_cuda13.bat`

## Generic entrypoint

아래 generic suite launcher로도 실행할 수 있다.

```bat
scripts\benchmark_suite_cuda13.bat --suite-profile ocr-combo-ranked-runtime
```

## 전제

- Japan gold는 `benchmarks/ocr_combo/gold/japan/gold.json`의 locked OCR gold를 사용한다.
- China는 `benchmarks/ocr_combo_ranked/frozen/china_winner.json`을 재사용한다.
- strict `ocr-combo-runtime` 자산은 historical baseline으로 그대로 둔다.
