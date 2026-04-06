@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul

set "PYTHON_EXE=%SCRIPT_DIR%.venv-win\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Windows runtime environment not found: "%PYTHON_EXE%"
    echo Create or repair .venv-win before running this launcher.
    popd >nul
    exit /b 1
)

"%PYTHON_EXE%" comic.py %*
set "EXITCODE=%ERRORLEVEL%"

popd >nul
exit /b %EXITCODE%
