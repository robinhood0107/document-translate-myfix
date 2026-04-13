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
echo [ocr-simpletest] launcher=scripts\ocr_simpletest_mangalmm_vs_paddle_benchmark_pipeline.bat
echo [ocr-simpletest] output-root=%BENCH_ROOT%
call "%PYTHON_EXE%" -u "%SCRIPT_DIR%ocr_simpletest_mangalmm_vs_paddle_benchmark.py" run --sample-dir "%REPO_ROOT%\Sample\simpletest"
exit /b %ERRORLEVEL%

:run_command
shift
echo [ocr-simpletest] launcher=scripts\ocr_simpletest_mangalmm_vs_paddle_benchmark_pipeline.bat
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
echo   scripts\ocr_simpletest_mangalmm_vs_paddle_benchmark_pipeline.bat
echo   scripts\ocr_simpletest_mangalmm_vs_paddle_benchmark_pipeline.bat run [--sample-dir DIR]
echo   scripts\ocr_simpletest_mangalmm_vs_paddle_benchmark_pipeline.bat summary [--suite-dir DIR]
echo   scripts\ocr_simpletest_mangalmm_vs_paddle_benchmark_pipeline.bat open
echo.
goto :eof
