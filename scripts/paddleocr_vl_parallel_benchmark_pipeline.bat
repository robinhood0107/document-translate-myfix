@echo off
setlocal
set SCRIPT_DIR=%~dp0
"%SCRIPT_DIR%..\.venv-win\Scripts\python.exe" "%SCRIPT_DIR%benchmark_pipeline.py" ^
  --preset "%SCRIPT_DIR%..\benchmarks\paddleocr_vl_parallel\presets\paddleocr-vl-parallel-base.json" ^
  --mode batch ^
  --repeat 1 ^
  --runtime-mode managed ^
  --runtime-services ocr-only ^
  --sample-dir "%SCRIPT_DIR%..\Sample\japan_vllm_parallel_subset" ^
  --sample-count 13 ^
  --source-lang Japanese ^
  --target-lang Korean ^
  --clear-app-caches ^
  --export-page-snapshots ^
  --stage-ceiling ocr
endlocal
