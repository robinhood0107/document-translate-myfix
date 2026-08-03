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
set "SAMPLER_LOG_DIR=%CD%\banchmark_result_log\managed-runs\10-gemma-translation\gemma-sampler-quality-v2\supervisor-logs"
set "SAMPLER_LOG=%SAMPLER_LOG_DIR%\%SAMPLER_RUN_ID%.log"
set "SAMPLER_PYTHON=%CD%\.venv-win\Scripts\python.exe"
if not exist "%SAMPLER_PYTHON%" (
  echo [GEMMA-SAMPLER-V2] .venv-win Python was not found.
  exit /b 2
)
if not exist "%SAMPLER_LOG_DIR%" mkdir "%SAMPLER_LOG_DIR%"

if /I "%SAMPLER_NO_MONITOR%"=="1" goto monitor_ready
if "%GEMMA_MONITOR_OUTPUT%"=="" set "GEMMA_MONITOR_OUTPUT=%CD%\banchmark_result_log\tools\gemma-monitor.exe"
set "SAMPLER_MONITOR_EXE=%GEMMA_MONITOR_OUTPUT%"
call scripts\build_gemma_sampler_monitor.bat --if-stale
if errorlevel 1 (
  echo [GEMMA-SAMPLER-V2] gemma-monitor build failed. Set SAMPLER_NO_MONITOR=1 only for a deliberate headless run.
  exit /b 2
)
start "Gemma Sampler Monitor" "%SAMPLER_MONITOR_EXE%" --run-root "%SAMPLER_RUN_ROOT%" --poll-interval 1s --exit-on-completion

:monitor_ready
echo [GEMMA-SAMPLER-V2] Runner logs: %SAMPLER_LOG%

:retry
if "%SAMPLER_PRIOR_RESPONSE_RUN%"=="" (
  call :run_phase >> "%SAMPLER_LOG%" 2>&1
) else (
  call :run_phase --prior-response-run "%SAMPLER_PRIOR_RESPONSE_RUN%" >> "%SAMPLER_LOG%" 2>&1
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
