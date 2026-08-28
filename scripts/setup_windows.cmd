@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT=%%~fI"
set "RUNTIME=%~1"
set "TIER=%~2"
set "RUN_BAT=run_comic.bat"
if /I "%RUNTIME%"=="cuda13" set "RUN_BAT=run_comic_cuda13.bat"
shift
shift

chcp 65001 >nul
color 07
mode con cols=120 lines=40 >nul 2>&1
title Comic Translate Setup - %RUNTIME%
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%SCRIPT_DIR%configure_console.ps1" >nul 2>&1

set "BOOTSTRAP_ARGS=-Runtime %RUNTIME%"
if /I "%TIER%"=="full" set "BOOTSTRAP_ARGS=%BOOTSTRAP_ARGS% -Full"
if /I "%COMIC_VERIFY_ONLY%"=="1" set "BOOTSTRAP_ARGS=%BOOTSTRAP_ARGS% -SourceVerify"

pushd "%ROOT%" >nul
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%bootstrap_windows.ps1" %BOOTSTRAP_ARGS% %1 %2 %3 %4 %5 %6 %7 %8 %9
set "EXITCODE=%ERRORLEVEL%"
popd >nul

echo.
if "%EXITCODE%"=="0" (
    title Comic Translate Setup - DONE
    echo +----------------------------------------------------------------------------+
    echo ^| DONE! Comic Translate setup completed successfully.                       ^|
    echo +----------------------------------------------------------------------------+
    echo   Next step: %RUN_BAT%
) else (
    title Comic Translate Setup - FAILED
    echo +----------------------------------------------------------------------------+
    echo ^| FAILED! Comic Translate setup did not complete.                            ^|
    echo +----------------------------------------------------------------------------+
    echo   Review the error and log path shown above, then run this setup again.
)

set "SHOULD_PAUSE="
if not defined COMIC_NO_PAUSE if /I not "%COMIC_VERIFY_ONLY%"=="1" set "SHOULD_PAUSE=1"
if defined SHOULD_PAUSE (
    echo.
    echo   Press any key to close this window.
    pause >nul
)
exit /b %EXITCODE%
