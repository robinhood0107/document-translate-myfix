@echo off
setlocal

if /I "%~1"=="help" goto :help

set "SCRIPT_DIR=%~dp0"
call "%SCRIPT_DIR%ocr_simpletest_mangalmm_vs_paddle_benchmark_pipeline.bat" %*
exit /b %ERRORLEVEL%

:help
echo.
echo Usage:
echo   scripts\ocr_simpletest_mangalmm_vs_paddle_benchmark_suite.bat
echo.
echo Runs the simpletest 3-page warm full-pipeline comparison suite using .venv-win.
echo.
goto :eof
