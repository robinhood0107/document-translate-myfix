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

set "PYTHON_EXE=%REPO_ROOT%\.venv-win\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Windows benchmark environment not found: "%PYTHON_EXE%"
    echo Create or repair .venv-win before running this launcher.
    exit /b 1
)

if /I "%~1"=="help" goto :help

echo [suite] launcher=scripts\benchmark_suite.bat
echo [suite] started-at=%DATE% %TIME%
echo [suite] repo-root=%REPO_ROOT%
echo [suite] python=%PYTHON_EXE%
echo [suite] output-root=%BENCH_ROOT%
echo [suite] sample-dir=%REPO_ROOT%\Sample
echo [suite] sample-count=30
echo [suite] runtime=.venv-win
echo [suite] one-click suite starting...
call "%PYTHON_EXE%" -u "%SCRIPT_DIR%benchmark_suite.py"
set "EXIT_CODE=%ERRORLEVEL%"
echo [suite] finished exit-code=%EXIT_CODE% at=%DATE% %TIME%
exit /b %EXIT_CODE%

:help
echo.
echo Usage:
echo   scripts\benchmark_suite.bat
echo   scripts\benchmark_suite_cuda13.bat
echo.
echo This one-click launcher runs:
echo   1. translation-baseline one-page attach-running
echo   2. translation-baseline batch attach-running
echo   3. translation-ngl23 batch managed
echo.
echo Runtime:
echo   benchmark_suite.bat uses .venv-win
echo   benchmark_suite_cuda13.bat uses .venv-win-cuda13
echo.
echo Results are saved to:
echo   %%REPO_ROOT%%\banchmark_result_log
echo.
goto :eof
