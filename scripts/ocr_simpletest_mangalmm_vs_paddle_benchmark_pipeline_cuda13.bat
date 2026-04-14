@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"

if defined CT_BENCH_OUTPUT_ROOT (
    set "BENCH_ROOT=%CT_BENCH_OUTPUT_ROOT%"
) else (
    set "BENCH_ROOT=%REPO_ROOT%\banchmark_result_log\ocr_simpletest_mangalmm_vs_paddle"
    set "CT_BENCH_OUTPUT_ROOT=%BENCH_ROOT%"
)

set "PYTHON_EXE=%REPO_ROOT%\.venv-win-cuda13\Scripts\python.exe"
set "CUDA13_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
set "TENSORRT_LIBS=%REPO_ROOT%\.venv-win-cuda13\Lib\site-packages\tensorrt_libs"
set "CUDNN_BIN=%REPO_ROOT%\.venv-win-cuda13\Lib\site-packages\nvidia\cudnn\bin"
set "DOCKER_BIN=C:\Program Files\Docker\Docker\resources\bin"
set "DOCKER_CONFIG=%REPO_ROOT%\.docker-bench"

if not exist "%PYTHON_EXE%" (
    echo CUDA 13 benchmark environment not found: "%PYTHON_EXE%"
    exit /b 1
)

if exist "%CUDA13_BIN%" set "PATH=%CUDA13_BIN%;%PATH%"
if exist "%TENSORRT_LIBS%" set "PATH=%TENSORRT_LIBS%;%PATH%"
if exist "%CUDNN_BIN%" set "PATH=%CUDNN_BIN%;%PATH%"
if exist "%DOCKER_BIN%" set "PATH=%DOCKER_BIN%;%PATH%"
if not exist "%DOCKER_CONFIG%" mkdir "%DOCKER_CONFIG%" >nul 2>&1
if not exist "%DOCKER_CONFIG%\config.json" (
    >"%DOCKER_CONFIG%\config.json" echo {}
)

if "%~1"=="" goto :run_default
if /I "%~1"=="run" goto :run_command
if /I "%~1"=="summary" goto :summary
if /I "%~1"=="open" goto :open_dir
if /I "%~1"=="help" goto :help

goto :run_default

:run_default
call "%PYTHON_EXE%" -u "%SCRIPT_DIR%ocr_simpletest_mangalmm_vs_paddle_benchmark.py" run --sample-dir "%REPO_ROOT%\Sample\simpletest"
exit /b %ERRORLEVEL%

:run_command
shift
call "%PYTHON_EXE%" -u "%SCRIPT_DIR%ocr_simpletest_mangalmm_vs_paddle_benchmark.py" run %*
exit /b %ERRORLEVEL%

:summary
shift
call "%PYTHON_EXE%" -u "%SCRIPT_DIR%ocr_simpletest_mangalmm_vs_paddle_benchmark.py" summary %*
exit /b %ERRORLEVEL%

:open_dir
if not exist "%BENCH_ROOT%" (
    mkdir "%BENCH_ROOT%" >nul 2>&1
)
start "" "%BENCH_ROOT%"
goto :eof

:help
echo.
echo Usage:
echo   scripts\ocr_simpletest_mangalmm_vs_paddle_benchmark_pipeline_cuda13.bat
echo   scripts\ocr_simpletest_mangalmm_vs_paddle_benchmark_pipeline_cuda13.bat run [--sample-dir DIR]
echo   scripts\ocr_simpletest_mangalmm_vs_paddle_benchmark_pipeline_cuda13.bat summary [--suite-dir DIR]
echo   scripts\ocr_simpletest_mangalmm_vs_paddle_benchmark_pipeline_cuda13.bat open
echo.
goto :eof
