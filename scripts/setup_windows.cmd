@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT=%%~fI"
set "RUNTIME=%~1"
set "TIER=%~2"
shift
shift

chcp 65001 >nul
mode con cols=120 lines=40 >nul 2>&1
title Comic Translate Setup - %RUNTIME%
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%SCRIPT_DIR%configure_console.ps1" >nul 2>&1

set "BOOTSTRAP_ARGS=-Runtime %RUNTIME%"
if /I "%TIER%"=="full" set "BOOTSTRAP_ARGS=%BOOTSTRAP_ARGS% -Full"
if /I "%COMIC_VERIFY_ONLY%"=="1" set "BOOTSTRAP_ARGS=%BOOTSTRAP_ARGS% -SourceVerify"

pushd "%ROOT%" >nul
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%bootstrap_windows.ps1" %BOOTSTRAP_ARGS% %1 %2 %3 %4 %5 %6 %7 %8 %9
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" if not defined COMIC_NO_PAUSE pause
popd >nul
exit /b %EXITCODE%
