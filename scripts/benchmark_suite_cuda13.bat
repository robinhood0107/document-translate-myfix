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

set "PYTHON_EXE=%REPO_ROOT%\.venv-win-cuda13\Scripts\python.exe"
set "CUDA13_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
set "TENSORRT_LIBS=%REPO_ROOT%\.venv-win-cuda13\Lib\site-packages\tensorrt_libs"
set "CUDNN_BIN=%REPO_ROOT%\.venv-win-cuda13\Lib\site-packages\nvidia\cudnn\bin"

if not exist "%PYTHON_EXE%" (
    echo CUDA 13 benchmark environment not found: "%PYTHON_EXE%"
    exit /b 1
)

if exist "%CUDA13_BIN%" set "PATH=%CUDA13_BIN%;%PATH%"
if exist "%TENSORRT_LIBS%" set "PATH=%TENSORRT_LIBS%;%PATH%"
if exist "%CUDNN_BIN%" set "PATH=%CUDNN_BIN%;%PATH%"

if /I "%~1"=="help" goto :help

echo [suite] launcher=scripts\benchmark_suite_cuda13.bat
echo [suite] started-at=%DATE% %TIME%
echo [suite] repo-root=%REPO_ROOT%
echo [suite] python=%PYTHON_EXE%
echo [suite] output-root=%BENCH_ROOT%
echo [suite] sample-dir=%REPO_ROOT%\Sample
echo [suite] sample-count=30
echo [suite] runtime=.venv-win-cuda13
echo [suite] cuda13-bin=%CUDA13_BIN%
echo [suite] tensorrt-libs=%TENSORRT_LIBS%
echo [suite] cudnn-bin=%CUDNN_BIN%
echo [suite] suite-args=%*
echo [suite] one-click suite starting...
call "%PYTHON_EXE%" -u "%SCRIPT_DIR%benchmark_suite.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
echo [suite] finished exit-code=%EXIT_CODE% at=%DATE% %TIME%
exit /b %EXIT_CODE%

:help
echo.
echo Usage:
echo   scripts\benchmark_suite_cuda13.bat
echo   scripts\benchmark_suite_cuda13.bat --suite-profile b8665-gemma4
echo   scripts\benchmark_suite_cuda13.bat --suite-profile ocr-combo-runtime
echo   scripts\benchmark_suite_cuda13.bat --suite-profile ocr-combo-ranked-runtime
echo.
echo Default suite runs:
echo   1. translation-baseline one-page attach-running
echo   2. translation-baseline batch attach-running
echo   3. translation-ngl23 batch managed
echo.
echo b8665 experiment suite:
echo   scripts\benchmark_suite_cuda13.bat --suite-profile b8665-gemma4
echo.
echo OCR combo language-aware suite:
echo   scripts\benchmark_suite_cuda13.bat --suite-profile ocr-combo-runtime
echo.
echo OCR combo ranked Japan suite:
echo   scripts\benchmark_suite_cuda13.bat --suite-profile ocr-combo-ranked-runtime
echo.
echo Runtime:
echo   benchmark_suite_cuda13.bat uses .venv-win-cuda13
echo.
echo Results are saved to:
echo   %%REPO_ROOT%%\banchmark_result_log
echo.
goto :eof
