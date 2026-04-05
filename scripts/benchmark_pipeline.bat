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

set "PYTHON_EXE=%REPO_ROOT%\.venv-win\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Windows benchmark environment not found: "%PYTHON_EXE%"
    echo Create or repair .venv-win before running this launcher.
    exit /b 1
)

if "%~1"=="" goto :run_default
if /I "%~1"=="run" goto :run_command
if /I "%~1"=="summary" goto :summary
if /I "%~1"=="open" goto :open_dir
if /I "%~1"=="help" goto :help

goto :run_default

:run_default
set "PRESET=live-ops-baseline"
set "MODE=batch"
set "RUNTIME_MODE=attach-running"
set "REPEAT=1"
set "SAMPLE_DIR=%REPO_ROOT%\Sample"
set "SAMPLE_COUNT=30"
goto :execute_run

:run_command
set "PRESET=%~2"
if "%PRESET%"=="" set "PRESET=live-ops-baseline"
set "MODE=%~3"
if "%MODE%"=="" set "MODE=batch"
set "RUNTIME_MODE=%~4"
if "%RUNTIME_MODE%"=="" set "RUNTIME_MODE=attach-running"
set "REPEAT=%~5"
if "%REPEAT%"=="" set "REPEAT=1"
set "SAMPLE_DIR=%~6"
if "%SAMPLE_DIR%"=="" set "SAMPLE_DIR=%REPO_ROOT%\Sample"
set "SAMPLE_COUNT=%~7"
if "%SAMPLE_COUNT%"=="" set "SAMPLE_COUNT=30"

:execute_run
echo [benchmark] launcher=scripts\benchmark_pipeline.bat
echo [benchmark] started-at=%DATE% %TIME%
echo [benchmark] repo-root=%REPO_ROOT%
echo [benchmark] python=%PYTHON_EXE%
echo [benchmark] preset=%PRESET% mode=%MODE% runtime-mode=%RUNTIME_MODE% repeat=%REPEAT%
echo [benchmark] sample-dir=%SAMPLE_DIR% sample-count=%SAMPLE_COUNT%
echo [benchmark] output-root=%BENCH_ROOT%
echo [benchmark] runtime=.venv-win
call "%PYTHON_EXE%" -u "%SCRIPT_DIR%benchmark_pipeline.py" ^
  --preset "%PRESET%" ^
  --mode "%MODE%" ^
  --repeat "%REPEAT%" ^
  --runtime-mode "%RUNTIME_MODE%" ^
  --sample-dir "%SAMPLE_DIR%" ^
  --sample-count "%SAMPLE_COUNT%"
set "EXIT_CODE=%ERRORLEVEL%"
echo [benchmark] finished exit-code=%EXIT_CODE% at=%DATE% %TIME%
goto :eof

:summary
if not exist "%BENCH_ROOT%" (
    echo [benchmark] benchmark root not found: %BENCH_ROOT%
    exit /b 1
)
echo [benchmark] writing summary to %BENCH_ROOT%\summary.md
echo [benchmark] python=%PYTHON_EXE%
call "%PYTHON_EXE%" -u "%SCRIPT_DIR%summarize_benchmarks.py" ^
  --input "%BENCH_ROOT%" ^
  --output "%BENCH_ROOT%\summary.md"
if exist "%BENCH_ROOT%\summary.md" (
    echo [benchmark] summary ready: %BENCH_ROOT%\summary.md
)
goto :eof

:open_dir
if not exist "%BENCH_ROOT%" (
    mkdir "%BENCH_ROOT%" >nul 2>&1
)
echo [benchmark] opening result directory: %BENCH_ROOT%
start "" "%BENCH_ROOT%"
goto :eof

:help
echo.
echo Usage:
echo   scripts\benchmark_pipeline.bat
echo   scripts\benchmark_pipeline_cuda13.bat
echo   scripts\benchmark_pipeline.bat run [preset] [mode] [runtime-mode] [repeat] [sample-dir] [sample-count]
echo   scripts\benchmark_pipeline.bat summary
echo   scripts\benchmark_pipeline.bat open
echo.
echo Runtime:
echo   benchmark_pipeline.bat uses .venv-win
echo   benchmark_pipeline_cuda13.bat uses .venv-win-cuda13
echo.
echo Output root:
echo   %%USERPROFILE%%\Documents\Comic Translate
echo.
echo Examples:
echo   scripts\benchmark_pipeline.bat
echo   scripts\benchmark_pipeline.bat run live-ops-baseline one-page attach-running 1
echo   scripts\benchmark_pipeline.bat run gpu-shift-ocr-front-cpu batch managed 1
echo   scripts\benchmark_pipeline.bat summary
echo   scripts\benchmark_pipeline.bat open
echo.
goto :eof
