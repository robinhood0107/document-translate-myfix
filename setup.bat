@echo off
call "%~dp0scripts\setup_windows.cmd" cuda12 core %*
exit /b %ERRORLEVEL%
