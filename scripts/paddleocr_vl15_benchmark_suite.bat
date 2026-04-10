@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"

if defined CT_BENCH_OUTPUT_ROOT (
    set "BENCH_ROOT=%CT_BENCH_OUTPUT_ROOT%"
) else (
    set "BENCH_ROOT=%REPO_ROOT%\banchmark_result_log\paddleocr_vl15"
    set "CT_BENCH_OUTPUT_ROOT=%BENCH_ROOT%"
)

set "PYTHON_EXE=%REPO_ROOT%\.venv-win\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Windows PaddleOCR-VL15 benchmark environment not found: "%PYTHON_EXE%"
    echo Create or repair .venv-win before running this launcher.
    exit /b 1
)

if /I "%~1"=="help" goto :help

echo [suite] launcher=scripts\paddleocr_vl15_benchmark_suite.bat
echo [suite] started-at=%DATE% %TIME%
echo [suite] repo-root=%REPO_ROOT%
echo [suite] python=%PYTHON_EXE%
echo [suite] output-root=%BENCH_ROOT%
echo [suite] sample-dir=%REPO_ROOT%\Sample
echo [suite] sample-count=30
echo [suite] runtime=.venv-win
echo [suite] suite-profile=paddleocr-vl15-runtime
echo [suite] official-execution-scope=detect-ocr-only
echo [suite] suite-args=%*
call "%PYTHON_EXE%" -u "%SCRIPT_DIR%benchmark_suite.py" --suite-profile paddleocr-vl15-runtime %*
set "EXIT_CODE=%ERRORLEVEL%"
echo [suite] finished exit-code=%EXIT_CODE% at=%DATE% %TIME%
exit /b %EXIT_CODE%

:help
echo.
echo Usage:
echo   scripts\paddleocr_vl15_benchmark_suite.bat
echo   scripts\paddleocr_vl15_benchmark_suite.bat help
echo.
echo Runtime:
echo   paddleocr_vl15_benchmark_suite.bat uses .venv-win
echo.
echo Results are saved to:
echo   %%REPO_ROOT%%\banchmark_result_log\paddleocr_vl15
echo.
echo Official default:
echo   detect+ocr-only suite with warm-stable gate and screen-to-confirm protocol.
echo.
goto :eof
