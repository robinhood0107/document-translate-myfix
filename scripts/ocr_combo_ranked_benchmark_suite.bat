@echo off
setlocal

if /I "%~1"=="help" goto :help

set "SCRIPT_DIR=%~dp0"
call "%SCRIPT_DIR%benchmark_suite.bat" --suite-profile ocr-combo-ranked-runtime %*
exit /b %ERRORLEVEL%

:help
echo.
echo Usage:
echo   scripts\ocr_combo_ranked_benchmark_suite.bat
echo.
echo Runs the OCR combo ranked Japan suite using .venv-win.
echo China is reused from the frozen strict-family winner.
echo.
goto :eof
