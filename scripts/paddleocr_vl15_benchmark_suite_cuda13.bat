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

set "PYTHON_EXE=%REPO_ROOT%\.venv-win-cuda13\Scripts\python.exe"
set "CUDA13_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
set "TENSORRT_LIBS=%REPO_ROOT%\.venv-win-cuda13\Lib\site-packages\tensorrt_libs"
set "CUDNN_BIN=%REPO_ROOT%\.venv-win-cuda13\Lib\site-packages\nvidia\cudnn\bin"

if not exist "%PYTHON_EXE%" (
    echo CUDA 13 PaddleOCR-VL15 benchmark environment not found: "%PYTHON_EXE%"
    exit /b 1
)

if exist "%CUDA13_BIN%" set "PATH=%CUDA13_BIN%;%PATH%"
if exist "%TENSORRT_LIBS%" set "PATH=%TENSORRT_LIBS%;%PATH%"
if exist "%CUDNN_BIN%" set "PATH=%CUDNN_BIN%;%PATH%"

if /I "%~1"=="help" goto :help

echo [suite] launcher=scripts\paddleocr_vl15_benchmark_suite_cuda13.bat
echo [suite] started-at=%DATE% %TIME%
echo [suite] repo-root=%REPO_ROOT%
echo [suite] python=%PYTHON_EXE%
echo [suite] output-root=%BENCH_ROOT%
echo [suite] sample-dir=%REPO_ROOT%\Sample
echo [suite] sample-count=30
echo [suite] runtime=.venv-win-cuda13
echo [suite] suite-profile=paddleocr-vl15-runtime
echo [suite] suite-args=%*
call "%PYTHON_EXE%" -u "%SCRIPT_DIR%benchmark_suite.py" --suite-profile paddleocr-vl15-runtime %*
set "EXIT_CODE=%ERRORLEVEL%"
echo [suite] finished exit-code=%EXIT_CODE% at=%DATE% %TIME%
exit /b %EXIT_CODE%

:help
echo.
echo Usage:
echo   scripts\paddleocr_vl15_benchmark_suite_cuda13.bat
echo   scripts\paddleocr_vl15_benchmark_suite_cuda13.bat help
echo.
echo Runtime:
echo   paddleocr_vl15_benchmark_suite_cuda13.bat uses .venv-win-cuda13
echo.
echo Results are saved to:
echo   %%REPO_ROOT%%\banchmark_result_log\paddleocr_vl15
echo.
goto :eof
