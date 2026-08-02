@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

if "%SAMPLER_REFERENCE%"=="" (
  echo [GEMMA-SAMPLER-V2] Set SAMPLER_REFERENCE to an ignored frozen reference JSON path.
  exit /b 2
)

if "%SAMPLER_PHASE%"=="" set "SAMPLER_PHASE=temperature"
if "%SAMPLER_RUN_ID%"=="" set "SAMPLER_RUN_ID=gemma-sampler-v2-%SAMPLER_PHASE%"
if "%SAMPLER_MAX_ATTEMPTS%"=="" set "SAMPLER_MAX_ATTEMPTS=3"
if "%SAMPLER_TIMEOUT_SEC%"=="" set "SAMPLER_TIMEOUT_SEC=180"

set "SAMPLER_RUN_ROOT=%CD%\banchmark_result_log\managed-runs\10-gemma-translation\gemma-sampler-quality-v2\%SAMPLER_RUN_ID%"
set "SAMPLER_PYTHON=%CD%\.venv-win-cuda13\Scripts\python.exe"
if not exist "%SAMPLER_PYTHON%" (
  echo [GEMMA-SAMPLER-V2] .venv-win-cuda13 Python was not found.
  exit /b 2
)

:retry
if "%SAMPLER_PRIOR_RESPONSE_RUN%"=="" (
  call :run_phase
) else (
  call :run_phase --prior-response-run "%SAMPLER_PRIOR_RESPONSE_RUN%"
)
set "SAMPLER_EXIT=%ERRORLEVEL%"
if "%SAMPLER_EXIT%"=="75" (
  timeout /t 5 /nobreak >nul
  goto retry
)
if not "%SAMPLER_EXIT%"=="0" if not "%SAMPLER_EXIT%"=="2" (
  timeout /t 5 /nobreak >nul
  goto retry
)
exit /b %SAMPLER_EXIT%

:run_phase
if exist "%SAMPLER_RUN_ROOT%" (
  "%SAMPLER_PYTHON%" scripts\benchmark_gemma_sampler_quality_v2.py run-phase --resume-run "%SAMPLER_RUN_ROOT%" --reference "%SAMPLER_REFERENCE%" --phase "%SAMPLER_PHASE%" --timeout-sec %SAMPLER_TIMEOUT_SEC% --max-attempts %SAMPLER_MAX_ATTEMPTS% %SAMPLER_SELECTION% %*
) else (
  "%SAMPLER_PYTHON%" scripts\benchmark_gemma_sampler_quality_v2.py run-phase --run-id "%SAMPLER_RUN_ID%" --reference "%SAMPLER_REFERENCE%" --phase "%SAMPLER_PHASE%" --timeout-sec %SAMPLER_TIMEOUT_SEC% --max-attempts %SAMPLER_MAX_ATTEMPTS% %SAMPLER_SELECTION% %*
)
exit /b %ERRORLEVEL%
