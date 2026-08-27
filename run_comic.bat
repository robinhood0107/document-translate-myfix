@echo off
setlocal
pushd "%~dp0" >nul
set "BOOTSTRAP_ARGS=-Runtime cuda12"
if /I "%COMIC_VERIFY_ONLY%"=="1" set "BOOTSTRAP_ARGS=%BOOTSTRAP_ARGS% -SourceVerify"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap_windows.ps1" %BOOTSTRAP_ARGS% %*
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" if not defined COMIC_NO_PAUSE pause
popd >nul
exit /b %EXITCODE%
