@echo off
call "%~dp0scripts\setup_windows.cmd" cuda13 core %*
exit /b %ERRORLEVEL%
