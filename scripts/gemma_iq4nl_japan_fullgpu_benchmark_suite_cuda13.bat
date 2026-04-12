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

set "MODE=%~1"
if "%MODE%"=="" goto :run_default
if /I "%MODE%"=="help" goto :help

goto :run_forward

:run_default
call "%PYTHON_EXE%" -u "%SCRIPT_DIR%gemma_iq4nl_japan_fullgpu_benchmark.py" all
exit /b %ERRORLEVEL%

:run_forward
call "%PYTHON_EXE%" -u "%SCRIPT_DIR%gemma_iq4nl_japan_fullgpu_benchmark.py" %*
exit /b %ERRORLEVEL%

:help
echo.
echo Usage:
echo   scripts\gemma_iq4nl_japan_fullgpu_benchmark_suite_cuda13.bat
echo   scripts\gemma_iq4nl_japan_fullgpu_benchmark_suite_cuda13.bat all
echo   scripts\gemma_iq4nl_japan_fullgpu_benchmark_suite_cuda13.bat smoke --suite-dir ^<dir^>
echo   scripts\gemma_iq4nl_japan_fullgpu_benchmark_suite_cuda13.bat stage1 --suite-dir ^<dir^>
echo   scripts\gemma_iq4nl_japan_fullgpu_benchmark_suite_cuda13.bat report --suite-dir ^<dir^>
echo.
