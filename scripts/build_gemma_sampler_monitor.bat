@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0.."

if "%GEMMA_MONITOR_OUTPUT%"=="" set "GEMMA_MONITOR_OUTPUT=%CD%\banchmark_result_log\tools\gemma-monitor.exe"
if "%GEMMA_MONITOR_GO%"=="" if exist "%USERPROFILE%\scoop\shims\go.exe" set "GEMMA_MONITOR_GO=%USERPROFILE%\scoop\shims\go.exe"
if "%GEMMA_MONITOR_GO%"=="" if exist "%USERPROFILE%\scoop\apps\go\current\bin\go.exe" set "GEMMA_MONITOR_GO=%USERPROFILE%\scoop\apps\go\current\bin\go.exe"
if "%GEMMA_MONITOR_GO%"=="" (
  for /f "delims=" %%G in ('where go 2^>nul') do if "%GEMMA_MONITOR_GO%"=="" set "GEMMA_MONITOR_GO=%%G"
)
if "%GEMMA_MONITOR_GO%"=="" (
  echo [GEMMA-MONITOR] Go SDK was not found. Install Scoop Go or set GEMMA_MONITOR_GO.
  exit /b 2
)
if not exist "%GEMMA_MONITOR_GO%" (
  echo [GEMMA-MONITOR] Configured Go executable was not found: %GEMMA_MONITOR_GO%
  exit /b 2
)

set "GEMMA_MONITOR_NEEDS_BUILD=1"
if /I "%~1"=="--if-stale" if exist "%GEMMA_MONITOR_OUTPUT%" (
  set "GEMMA_MONITOR_NEEDS_BUILD="
  powershell.exe -NoProfile -Command "$output = Get-Item -LiteralPath $env:GEMMA_MONITOR_OUTPUT; $sources = Get-ChildItem -LiteralPath '%CD%\scripts\gemma_sampler_monitor' -File | Where-Object { $_.Extension -eq '.go' -or $_.Name -eq 'go.mod' -or $_.Name -eq 'go.sum' }; if ($sources | Where-Object { $_.LastWriteTimeUtc -gt $output.LastWriteTimeUtc }) { exit 1 }; exit 0"
  if errorlevel 1 set "GEMMA_MONITOR_NEEDS_BUILD=1"
)
if not defined GEMMA_MONITOR_NEEDS_BUILD exit /b 0

for %%D in ("%GEMMA_MONITOR_OUTPUT%") do if not exist "%%~dpD" mkdir "%%~dpD"
echo [GEMMA-MONITOR] Building with "%GEMMA_MONITOR_GO%"...
"%GEMMA_MONITOR_GO%" version
if errorlevel 1 exit /b %ERRORLEVEL%
pushd scripts\gemma_sampler_monitor
"%GEMMA_MONITOR_GO%" build -trimpath -ldflags "-s -w" -o "%GEMMA_MONITOR_OUTPUT%" .
set "GEMMA_MONITOR_BUILD_EXIT=%ERRORLEVEL%"
popd
if not "%GEMMA_MONITOR_BUILD_EXIT%"=="0" exit /b %GEMMA_MONITOR_BUILD_EXIT%
echo [GEMMA-MONITOR] Ready: %GEMMA_MONITOR_OUTPUT%
exit /b 0
