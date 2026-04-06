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

if "%~1"=="" goto :run_default
if /I "%~1"=="run" goto :run_command
if /I "%~1"=="gold" goto :gold
if /I "%~1"=="compare" goto :compare
if /I "%~1"=="summary" goto :summary
if /I "%~1"=="open" goto :open_dir
if /I "%~1"=="help" goto :help

goto :run_default

:run_default
set "PRESET=paddleocr-vl15-baseline"
set "MODE=batch"
set "RUNTIME_MODE=managed"
set "REPEAT=1"
set "SAMPLE_DIR=%REPO_ROOT%\Sample"
set "SAMPLE_COUNT=30"
goto :execute_run

:run_command
:execute_with_args
echo [paddleocr-vl15] launcher=scripts\paddleocr_vl15_benchmark_pipeline_cuda13.bat
echo [paddleocr-vl15] started-at=%DATE% %TIME%
echo [paddleocr-vl15] repo-root=%REPO_ROOT%
echo [paddleocr-vl15] python=%PYTHON_EXE%
echo [paddleocr-vl15] output-root=%BENCH_ROOT%
echo [paddleocr-vl15] runtime=.venv-win-cuda13
echo [paddleocr-vl15] default-execution-scope=detect-ocr-only
call "%PYTHON_EXE%" -u "%SCRIPT_DIR%paddleocr_vl15_benchmark.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
echo [paddleocr-vl15] finished exit-code=%EXIT_CODE% at=%DATE% %TIME%
exit /b %EXIT_CODE%

:execute_run
echo [paddleocr-vl15] launcher=scripts\paddleocr_vl15_benchmark_pipeline_cuda13.bat
echo [paddleocr-vl15] started-at=%DATE% %TIME%
echo [paddleocr-vl15] repo-root=%REPO_ROOT%
echo [paddleocr-vl15] python=%PYTHON_EXE%
echo [paddleocr-vl15] preset=%PRESET% mode=%MODE% runtime-mode=%RUNTIME_MODE% repeat=%REPEAT%
echo [paddleocr-vl15] sample-dir=%SAMPLE_DIR% sample-count=%SAMPLE_COUNT%
echo [paddleocr-vl15] output-root=%BENCH_ROOT%
echo [paddleocr-vl15] runtime=.venv-win-cuda13
echo [paddleocr-vl15] default-execution-scope=detect-ocr-only
call "%PYTHON_EXE%" -u "%SCRIPT_DIR%paddleocr_vl15_benchmark.py" run "%PRESET%" "%MODE%" "%RUNTIME_MODE%" "%REPEAT%" "%SAMPLE_DIR%" "%SAMPLE_COUNT%"
set "EXIT_CODE=%ERRORLEVEL%"
echo [paddleocr-vl15] finished exit-code=%EXIT_CODE% at=%DATE% %TIME%
exit /b %EXIT_CODE%

:gold
call "%PYTHON_EXE%" -u "%SCRIPT_DIR%paddleocr_vl15_benchmark.py" gold %~2 %~3 %~4 %~5 %~6 %~7 %~8 %~9
exit /b %ERRORLEVEL%

:compare
call "%PYTHON_EXE%" -u "%SCRIPT_DIR%paddleocr_vl15_benchmark.py" compare %~2 %~3 %~4 %~5 %~6 %~7 %~8 %~9
exit /b %ERRORLEVEL%

:summary
call "%PYTHON_EXE%" -u "%SCRIPT_DIR%paddleocr_vl15_benchmark.py" summary %~2 %~3 %~4 %~5 %~6 %~7 %~8 %~9
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
echo   scripts\paddleocr_vl15_benchmark_pipeline_cuda13.bat
echo   scripts\paddleocr_vl15_benchmark_pipeline_cuda13.bat run [preset] [mode] [runtime-mode] [repeat] [sample-dir] [sample-count]
echo   scripts\paddleocr_vl15_benchmark_pipeline_cuda13.bat gold [--run-dir DIR] [--output FILE]
echo   scripts\paddleocr_vl15_benchmark_pipeline_cuda13.bat compare [--baseline-gold FILE] [--candidate-run-dir DIR]
echo   scripts\paddleocr_vl15_benchmark_pipeline_cuda13.bat summary [--manifest FILE]
echo   scripts\paddleocr_vl15_benchmark_pipeline_cuda13.bat open
echo.
echo Runtime:
echo   paddleocr_vl15_benchmark_pipeline_cuda13.bat uses .venv-win-cuda13
echo.
echo Output root:
echo   %%REPO_ROOT%%\banchmark_result_log\paddleocr_vl15
echo.
echo Official default:
echo   detect+ocr-only suite. Use --execution-scope full-pipeline for legacy/manual full pipeline runs.
echo.
goto :eof
