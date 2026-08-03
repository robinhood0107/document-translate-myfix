@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0.."

if /I "%~1"=="--verify" set "SAMPLER_VERIFY_ONLY=1"

rem One immutable campaign.  The runner auto-discovers the approved frozen
rem reference and the one complete r6 provenance run; callers never paste raw
rem private-artifact paths into this BAT.
if "%SAMPLER_RUN_ID%"=="" set "SAMPLER_RUN_ID=gemma-sampler-v2-single-campaign"
if "%SAMPLER_MAX_ATTEMPTS%"=="" set "SAMPLER_MAX_ATTEMPTS=3"
if "%SAMPLER_TIMEOUT_SEC%"=="" set "SAMPLER_TIMEOUT_SEC=180"

set "SAMPLER_RUN_ROOT=%CD%\banchmark_result_log\managed-runs\10-gemma-translation\gemma-sampler-quality-v2\%SAMPLER_RUN_ID%"
set "SAMPLER_LOG_DIR=%CD%\banchmark_result_log\managed-runs\10-gemma-translation\gemma-sampler-quality-v2\supervisor-logs"
set "SAMPLER_LOG=%SAMPLER_LOG_DIR%\%SAMPLER_RUN_ID%.log"
set "SAMPLER_PYTHON=%CD%\.venv-win-cuda13\Scripts\python.exe"
if not exist "%SAMPLER_PYTHON%" (
  echo [GEMMA-SAMPLER-V2] .venv-win-cuda13 Python was not found.
  exit /b 2
)
if not exist "%SAMPLER_LOG_DIR%" mkdir "%SAMPLER_LOG_DIR%"

if /I "%SAMPLER_VERIFY_ONLY%"=="1" goto verify_only

rem The double-click launcher owns the one visible Bubble Tea window.  A user
rem may still start this BAT directly; it then opens the same monitor itself.
if /I "%SAMPLER_LAUNCHED_BY_EXE%"=="1" goto monitor_ready
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
echo [GEMMA-SAMPLER-V2] One campaign runner log: %SAMPLER_LOG%

:retry
call :run_campaign >> "%SAMPLER_LOG%" 2>&1
set "SAMPLER_EXIT=%ERRORLEVEL%"
if "%SAMPLER_EXIT%"=="75" (
  timeout /t 5 /nobreak >nul
  goto retry
)
exit /b %SAMPLER_EXIT%

:verify_only
echo [GEMMA-SAMPLER-V2] Read-only campaign preflight: %SAMPLER_LOG%
"%SAMPLER_PYTHON%" scripts\benchmark_gemma_sampler_quality_v2.py verify-campaign >> "%SAMPLER_LOG%" 2>&1
exit /b %ERRORLEVEL%

:run_campaign
if exist "%SAMPLER_RUN_ROOT%\artifact-manifest.json" (
  "%SAMPLER_PYTHON%" scripts\benchmark_gemma_sampler_quality_v2.py run-campaign --resume-run "%SAMPLER_RUN_ROOT%" --timeout-sec %SAMPLER_TIMEOUT_SEC% --max-attempts %SAMPLER_MAX_ATTEMPTS%
) else (
  "%SAMPLER_PYTHON%" scripts\benchmark_gemma_sampler_quality_v2.py run-campaign --run-id "%SAMPLER_RUN_ID%" --timeout-sec %SAMPLER_TIMEOUT_SEC% --max-attempts %SAMPLER_MAX_ATTEMPTS%
)
exit /b %ERRORLEVEL%
