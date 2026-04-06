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
set "PRESET=%~2"
if "%PRESET%"=="" set "PRESET=paddleocr-vl15-baseline"
set "MODE=%~3"
if "%MODE%"=="" set "MODE=batch"
set "RUNTIME_MODE=%~4"
if "%RUNTIME_MODE%"=="" set "RUNTIME_MODE=managed"
set "REPEAT=%~5"
if "%REPEAT%"=="" set "REPEAT=1"
set "SAMPLE_DIR=%~6"
if "%SAMPLE_DIR%"=="" set "SAMPLE_DIR=%REPO_ROOT%\Sample"
set "SAMPLE_COUNT=%~7"
if "%SAMPLE_COUNT%"=="" set "SAMPLE_COUNT=30"

:execute_run
echo [paddleocr-vl15] launcher=scripts\paddleocr_vl15_benchmark_pipeline.bat
echo [paddleocr-vl15] started-at=%DATE% %TIME%
echo [paddleocr-vl15] repo-root=%REPO_ROOT%
echo [paddleocr-vl15] python=%PYTHON_EXE%
echo [paddleocr-vl15] preset=%PRESET% mode=%MODE% runtime-mode=%RUNTIME_MODE% repeat=%REPEAT%
echo [paddleocr-vl15] sample-dir=%SAMPLE_DIR% sample-count=%SAMPLE_COUNT%
echo [paddleocr-vl15] output-root=%BENCH_ROOT%
echo [paddleocr-vl15] runtime=.venv-win
call "%PYTHON_EXE%" -u "%SCRIPT_DIR%paddleocr_vl15_benchmark.py" run "%PRESET%" "%MODE%" "%RUNTIME_MODE%" "%REPEAT%" "%SAMPLE_DIR%" "%SAMPLE_COUNT%"
set "EXIT_CODE=%ERRORLEVEL%"
echo [paddleocr-vl15] finished exit-code=%EXIT_CODE% at=%DATE% %TIME%
exit /b %EXIT_CODE%

:gold
echo [paddleocr-vl15] generating gold
call "%PYTHON_EXE%" -u "%SCRIPT_DIR%paddleocr_vl15_benchmark.py" gold %~2 %~3 %~4 %~5 %~6 %~7 %~8 %~9
exit /b %ERRORLEVEL%

:compare
echo [paddleocr-vl15] running gold compare
call "%PYTHON_EXE%" -u "%SCRIPT_DIR%paddleocr_vl15_benchmark.py" compare %~2 %~3 %~4 %~5 %~6 %~7 %~8 %~9
exit /b %ERRORLEVEL%

:summary
echo [paddleocr-vl15] generating report
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
echo   scripts\paddleocr_vl15_benchmark_pipeline.bat
echo   scripts\paddleocr_vl15_benchmark_pipeline.bat run [preset] [mode] [runtime-mode] [repeat] [sample-dir] [sample-count]
echo   scripts\paddleocr_vl15_benchmark_pipeline.bat gold [--run-dir DIR] [--output FILE]
echo   scripts\paddleocr_vl15_benchmark_pipeline.bat compare [--baseline-gold FILE] [--candidate-run-dir DIR]
echo   scripts\paddleocr_vl15_benchmark_pipeline.bat summary [--manifest FILE]
echo   scripts\paddleocr_vl15_benchmark_pipeline.bat open
echo.
echo Runtime:
echo   paddleocr_vl15_benchmark_pipeline.bat uses .venv-win
echo.
echo Output root:
echo   %%REPO_ROOT%%\banchmark_result_log\paddleocr_vl15
echo.
goto :eof
