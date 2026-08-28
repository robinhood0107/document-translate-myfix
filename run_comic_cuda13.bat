@echo off
call "%~dp0scripts\run_windows.cmd" cuda13 %*
exit /b %ERRORLEVEL%
