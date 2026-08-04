@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0.."

if "%GEMMA_MONITOR_OUTPUT%"=="" set "GEMMA_MONITOR_OUTPUT=%CD%\banchmark_result_log\tools\gemma-monitor.exe"
if "%GEMMA_LAUNCHER_OUTPUT%"=="" set "GEMMA_LAUNCHER_OUTPUT=%CD%\banchmark_result_log\tools\gemma-sampler-launcher.exe"
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

set "GEMMA_MONITOR_ONLY="
if /I "%~1"=="--monitor-only-if-stale" set "GEMMA_MONITOR_ONLY=1"
set "GEMMA_MONITOR_NEEDS_BUILD=1"
if /I "%~1"=="--if-stale" goto stale_pair
if /I "%~1"=="--monitor-only-if-stale" goto stale_monitor
goto build

:stale_pair
if exist "%GEMMA_MONITOR_OUTPUT%" if exist "%GEMMA_LAUNCHER_OUTPUT%" (
  set "GEMMA_MONITOR_NEEDS_BUILD="
  powershell.exe -NoProfile -Command "$outputs = @((Get-Item -LiteralPath $env:GEMMA_MONITOR_OUTPUT), (Get-Item -LiteralPath $env:GEMMA_LAUNCHER_OUTPUT)); $sources = Get-ChildItem -LiteralPath '%CD%\scripts\gemma_sampler_monitor' -File | Where-Object { $_.Extension -eq '.go' -or $_.Name -eq 'go.mod' -or $_.Name -eq 'go.sum' }; if ($sources | Where-Object { $_.LastWriteTimeUtc -gt ($outputs | Measure-Object -Property LastWriteTimeUtc -Minimum).Minimum }) { exit 1 }; exit 0"
  if errorlevel 1 set "GEMMA_MONITOR_NEEDS_BUILD=1"
)
goto build

:stale_monitor
if exist "%GEMMA_MONITOR_OUTPUT%" (
  set "GEMMA_MONITOR_NEEDS_BUILD="
  powershell.exe -NoProfile -Command "$output = Get-Item -LiteralPath $env:GEMMA_MONITOR_OUTPUT; $sources = Get-ChildItem -LiteralPath '%CD%\scripts\gemma_sampler_monitor' -File | Where-Object { $_.Extension -eq '.go' -or $_.Name -eq 'go.mod' -or $_.Name -eq 'go.sum' }; if ($sources | Where-Object { $_.LastWriteTimeUtc -gt $output.LastWriteTimeUtc }) { exit 1 }; exit 0"
  if errorlevel 1 set "GEMMA_MONITOR_NEEDS_BUILD=1"
)

:build
if not defined GEMMA_MONITOR_NEEDS_BUILD exit /b 0

for %%D in ("%GEMMA_MONITOR_OUTPUT%") do if not exist "%%~dpD" mkdir "%%~dpD"
if not defined GEMMA_MONITOR_ONLY for %%D in ("%GEMMA_LAUNCHER_OUTPUT%") do if not exist "%%~dpD" mkdir "%%~dpD"
echo [GEMMA-MONITOR] Building monitor with "%GEMMA_MONITOR_GO%"...
"%GEMMA_MONITOR_GO%" version
if errorlevel 1 exit /b %ERRORLEVEL%
pushd scripts\gemma_sampler_monitor
"%GEMMA_MONITOR_GO%" build -trimpath -ldflags "-s -w" -o "%GEMMA_MONITOR_OUTPUT%" .
set "GEMMA_MONITOR_BUILD_EXIT=%ERRORLEVEL%"
if not defined GEMMA_MONITOR_ONLY if "%GEMMA_MONITOR_BUILD_EXIT%"=="0" "%GEMMA_MONITOR_GO%" build -trimpath -ldflags "-s -w" -o "%GEMMA_LAUNCHER_OUTPUT%" .
if not defined GEMMA_MONITOR_ONLY if "%GEMMA_MONITOR_BUILD_EXIT%"=="0" set "GEMMA_MONITOR_BUILD_EXIT=%ERRORLEVEL%"
popd
if not "%GEMMA_MONITOR_BUILD_EXIT%"=="0" exit /b %GEMMA_MONITOR_BUILD_EXIT%
echo [GEMMA-MONITOR] Ready: %GEMMA_MONITOR_OUTPUT%
if not defined GEMMA_MONITOR_ONLY echo [GEMMA-MONITOR] Ready: %GEMMA_LAUNCHER_OUTPUT%
exit /b 0
