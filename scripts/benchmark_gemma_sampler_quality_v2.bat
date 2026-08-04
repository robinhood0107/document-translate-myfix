@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0.."

rem This non-CUDA13 counterpart is deliberately read-only.  The only supported
rem long inference entrypoint is gemma-sampler-launcher.exe, which starts the
rem CUDA13 BAT and the single Bubble Tea monitor together.
set "SAMPLER_RUN_ID=gemma-sampler-v2-single-campaign"
set "SAMPLER_LOG_DIR=%CD%\banchmark_result_log\managed-runs\10-gemma-translation\gemma-sampler-quality-v2\supervisor-logs"
set "SAMPLER_LOG=%SAMPLER_LOG_DIR%\%SAMPLER_RUN_ID%.log"
set "SAMPLER_PYTHON=%CD%\.venv-win\Scripts\python.exe"
if not exist "%SAMPLER_PYTHON%" (
  echo [GEMMA-SAMPLER-V2] .venv-win Python was not found.
  exit /b 2
)
if not exist "%SAMPLER_LOG_DIR%" mkdir "%SAMPLER_LOG_DIR%"

echo [GEMMA-SAMPLER-V2] Read-only campaign preflight only. Use gemma-sampler-launcher.exe to run CUDA13 inference.
"%SAMPLER_PYTHON%" scripts\benchmark_gemma_sampler_quality_v2.py verify-campaign >> "%SAMPLER_LOG%" 2>&1
exit /b %ERRORLEVEL%
