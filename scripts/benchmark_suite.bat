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

set "PYTHON_EXE=%REPO_ROOT%\.venv\Scripts\python.exe"
set "PYTHON_ARGS="
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=py"
    set "PYTHON_ARGS=-3"
)

if /I "%~1"=="help" goto :help

echo [suite] output-root=%BENCH_ROOT%
echo [suite] one-click suite starting...
call "%PYTHON_EXE%" %PYTHON_ARGS% "%SCRIPT_DIR%benchmark_suite.py"
exit /b %ERRORLEVEL%

:help
echo.
echo Usage:
echo   scripts\benchmark_suite.bat
echo.
echo This one-click launcher runs:
echo   1. live-ops-baseline batch attach-running
echo   2. live-ops-baseline one-page attach-running
echo   3. gpu-shift-ocr-front-cpu batch managed
echo.
echo Results are saved to:
echo   %%USERPROFILE%%\Documents\Comic Translate
echo.
goto :eof
