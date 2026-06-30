@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"

if defined CT_BENCH_OUTPUT_ROOT (
    set "BENCH_ROOT=%CT_BENCH_OUTPUT_ROOT%"
) else (
    set "BENCH_ROOT=%REPO_ROOT%\banchmark_result_log"
    set "CT_BENCH_OUTPUT_ROOT=%BENCH_ROOT%"
)

set "PYTHON_EXE=%REPO_ROOT%\.venv-win\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Windows benchmark environment not found: "%PYTHON_EXE%"
    exit /b 1
)

if /I "%~1"=="help" goto :help

echo [benchmark] launcher=scripts\benchmark_no_gemma_replay_pipeline.bat
echo [benchmark] started-at=%DATE% %TIME%
echo [benchmark] repo-root=%REPO_ROOT%
echo [benchmark] python=%PYTHON_EXE%
echo [benchmark] output-root=%BENCH_ROOT%
echo [benchmark] mode=no-gemma-replay

call "%PYTHON_EXE%" -u "%SCRIPT_DIR%benchmark_no_gemma_replay_pipeline.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
echo [benchmark] finished exit-code=%EXIT_CODE% at=%DATE% %TIME%
exit /b %EXIT_CODE%

:help
echo.
echo Usage:
echo   scripts\benchmark_no_gemma_replay_pipeline.bat --source-root "C:\path\to\Leather root"
echo   scripts\benchmark_no_gemma_replay_pipeline.bat --source-root "C:\path\to\Leather root" --only sample_japan
echo.
echo Runs the product stage-batch entrypoint while skipping Gemma translation.
echo Existing page_snapshots.json translations are matched back to current OCR blocks by page and bbox IoU.
goto :eof
