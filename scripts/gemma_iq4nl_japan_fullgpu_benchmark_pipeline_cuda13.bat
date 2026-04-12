@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"
set "BASE_PRESET=%REPO_ROOT%\benchmarks\gemma_iq4nl_japan\presets\gemma-iq4nl-japan-fullgpu-base.json"
set "SAMPLE_DIR=%REPO_ROOT%\Sample\japan"
set "SAMPLE_COUNT=22"

if "%~1"=="" goto :run_default
if /I "%~1"=="run" goto :run_custom
if /I "%~1"=="help" goto :help
goto :run_default

:run_default
call "%SCRIPT_DIR%benchmark_pipeline_cuda13.bat" run "%BASE_PRESET%" batch managed 1 "%SAMPLE_DIR%" "%SAMPLE_COUNT%"
exit /b %ERRORLEVEL%

:run_custom
set "PRESET=%~2"
if "%PRESET%"=="" set "PRESET=%BASE_PRESET%"
set "MODE=%~3"
if "%MODE%"=="" set "MODE=batch"
set "RUNTIME_MODE=%~4"
if "%RUNTIME_MODE%"=="" set "RUNTIME_MODE=managed"
set "REPEAT=%~5"
if "%REPEAT%"=="" set "REPEAT=1"
set "CUSTOM_SAMPLE_DIR=%~6"
if "%CUSTOM_SAMPLE_DIR%"=="" set "CUSTOM_SAMPLE_DIR=%SAMPLE_DIR%"
set "CUSTOM_SAMPLE_COUNT=%~7"
if "%CUSTOM_SAMPLE_COUNT%"=="" set "CUSTOM_SAMPLE_COUNT=%SAMPLE_COUNT%"
call "%SCRIPT_DIR%benchmark_pipeline_cuda13.bat" run "%PRESET%" "%MODE%" "%RUNTIME_MODE%" "%REPEAT%" "%CUSTOM_SAMPLE_DIR%" "%CUSTOM_SAMPLE_COUNT%"
exit /b %ERRORLEVEL%

:help
echo.
echo Usage:
echo   scripts\gemma_iq4nl_japan_fullgpu_benchmark_pipeline_cuda13.bat
echo   scripts\gemma_iq4nl_japan_fullgpu_benchmark_pipeline_cuda13.bat run [preset] [mode] [runtime-mode] [repeat] [sample-dir] [sample-count]
echo.
