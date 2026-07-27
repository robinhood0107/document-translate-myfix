@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"
set "PYTHON_EXE=%REPO_ROOT%\.venv-win-cuda13\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo Supported CUDA 13 Windows environment not found: "%PYTHON_EXE%"
    exit /b 1
)

if "%~1"=="" goto :help
if /I "%~1"=="help" goto :help

call "%PYTHON_EXE%" -u "%SCRIPT_DIR%benchmark_translation_memory_fast_path.py" %*
exit /b %ERRORLEVEL%

:help
echo.
echo Usage:
echo   scripts\translation_memory_fast_path_benchmark_suite_cuda13.bat ^
  --source-summary ^<summary.json^> ^
  --output-dir ^<ignored-validation-dir^>
echo.
echo Gemma runs in the pinned Docker image; this launcher validates the CUDA 13 app environment.
exit /b 0
