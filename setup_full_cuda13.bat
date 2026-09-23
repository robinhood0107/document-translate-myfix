@echo off
call "%~dp0scripts\setup_windows.cmd" cuda13 full %*
exit /b %ERRORLEVEL%
