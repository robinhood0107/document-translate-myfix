@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"
set "PYTHON_EXE=%REPO_ROOT%\.venv-win\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Windows benchmark environment not found: "%PYTHON_EXE%"
    exit /b 1
)
if /I "%~1"=="help" goto :help
if "%~1"=="" (
  call "%PYTHON_EXE%" -u "%SCRIPT_DIR%inpaint_ctd_benchmark.py" --scope spotlight
) else (
  call "%PYTHON_EXE%" -u "%SCRIPT_DIR%inpaint_ctd_benchmark.py" %*
)
exit /b %ERRORLEVEL%
:help
echo Usage:
echo   scripts\inpaint_ctd_benchmark_pipeline.bat
echo   scripts\inpaint_ctd_benchmark_pipeline.bat --scope spotlight --corpus china --case ctd-protect-aot
goto :eof
