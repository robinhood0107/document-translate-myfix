@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"

if defined CT_BENCH_OUTPUT_ROOT (
    set "BENCH_ROOT=%CT_BENCH_OUTPUT_ROOT%"
) else (
    set "BENCH_ROOT=%USERPROFILE%\Documents\Comic Translate"
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

echo [suite] output-root=%BENCH_ROOT%
echo [suite] runtime=.venv-win-cuda13
echo [suite] one-click suite starting...
call "%PYTHON_EXE%" "%SCRIPT_DIR%benchmark_suite.py"
exit /b %ERRORLEVEL%

:help
echo.
echo Usage:
echo   scripts\benchmark_suite_cuda13.bat
echo.
echo This one-click launcher runs:
echo   1. live-ops-baseline batch attach-running
echo   2. live-ops-baseline one-page attach-running
echo   3. gpu-shift-ocr-front-cpu batch managed
echo.
echo Runtime:
echo   benchmark_suite_cuda13.bat uses .venv-win-cuda13
echo.
echo Results are saved to:
echo   %%USERPROFILE%%\Documents\Comic Translate
echo.
goto :eof
