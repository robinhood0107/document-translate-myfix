@echo off
setlocal
set SCRIPT_DIR=%~dp0
"%SCRIPT_DIR%..\.venv-win-cuda13\Scripts\python.exe" "%SCRIPT_DIR%paddleocr_vl_parallel_benchmark.py"
endlocal
