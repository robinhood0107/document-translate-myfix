@echo off
call "%~dp0scripts\run_windows.cmd" cuda12 %*
exit /b %ERRORLEVEL%
