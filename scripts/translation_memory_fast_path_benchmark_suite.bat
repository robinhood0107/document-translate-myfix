@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"
set "PYTHON_EXE=%REPO_ROOT%\.venv-win\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo Supported Windows environment not found: "%PYTHON_EXE%"
    exit /b 1
)

if "%~1"=="" goto :help
if /I "%~1"=="help" goto :help

call "%PYTHON_EXE%" -u "%SCRIPT_DIR%benchmark_translation_memory_fast_path.py" %*
exit /b %ERRORLEVEL%

:help
echo.
echo Usage:
echo   scripts\translation_memory_fast_path_benchmark_suite.bat ^
  --source-summary ^<summary.json^> ^
  --output-dir ^<ignored-validation-dir^>
echo.
echo The output directory must be outside Git tracking.
exit /b 0
