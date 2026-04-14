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
    echo CUDA 12 benchmark environment not found: "%PYTHON_EXE%"
    exit /b 1
)

if "%~1"=="" goto :run_default
if /I "%~1"=="run" goto :run_full
if /I "%~1"=="smoke" goto :run_smoke
if /I "%~1"=="summary" goto :summary
if /I "%~1"=="open" goto :open_dir
if /I "%~1"=="help" goto :help

goto :run_default

:run_default
echo [workflow-split-runtime] launcher=scripts\run_workflow_split_runtime_cuda12.bat
echo [workflow-split-runtime] runtime=.venv-win
echo [workflow-split-runtime] output-root=%BENCH_ROOT%
call "%PYTHON_EXE%" -u "%SCRIPT_DIR%workflow_split_runtime_benchmark.py" run --scenario all
goto :eof

:run_full
echo [workflow-split-runtime] launcher=scripts\run_workflow_split_runtime_cuda12.bat
echo [workflow-split-runtime] runtime=.venv-win
echo [workflow-split-runtime] output-root=%BENCH_ROOT%
call "%PYTHON_EXE%" -u "%SCRIPT_DIR%workflow_split_runtime_benchmark.py" run %2 %3 %4 %5 %6 %7 %8 %9
goto :eof

:run_smoke
echo [workflow-split-runtime] launcher=scripts\run_workflow_split_runtime_cuda12.bat
echo [workflow-split-runtime] runtime=.venv-win
echo [workflow-split-runtime] output-root=%BENCH_ROOT%
call "%PYTHON_EXE%" -u "%SCRIPT_DIR%workflow_split_runtime_benchmark.py" run --scenario all --smoke %2 %3 %4 %5 %6 %7 %8 %9
goto :eof

:summary
call "%PYTHON_EXE%" -u "%SCRIPT_DIR%workflow_split_runtime_benchmark.py" summary
goto :eof

:open_dir
if not exist "%BENCH_ROOT%\workflow-split-runtime" (
    mkdir "%BENCH_ROOT%\workflow-split-runtime" >nul 2>&1
)
start "" "%BENCH_ROOT%\workflow-split-runtime"
goto :eof

:help
echo.
echo Usage:
echo   scripts\run_workflow_split_runtime_cuda12.bat
echo   scripts\run_workflow_split_runtime_cuda12.bat run [workflow_split_runtime_benchmark.py args]
echo   scripts\run_workflow_split_runtime_cuda12.bat smoke [workflow_split_runtime_benchmark.py args]
echo   scripts\run_workflow_split_runtime_cuda12.bat summary
echo   scripts\run_workflow_split_runtime_cuda12.bat open
echo.
echo Default action:
echo   Runs the Requirement 1 family suite on .venv-win.
echo.
echo Smoke action:
echo   Runs the curated 2-page smoke corpus (094.png, p_016.jpg).
echo.
goto :eof
