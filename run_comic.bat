@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul

set "UV_CMD=uv.exe"
where /q %UV_CMD%
if errorlevel 1 (
    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "UV_CMD=%USERPROFILE%\.local\bin\uv.exe"
    ) else (
        echo uv.exe was not found. Install uv or add it to PATH.
        popd >nul
        exit /b 1
    )
)

if exist ".venv\bin\python" if not exist ".venv\Scripts\python.exe" (
    set "UV_PROJECT_ENVIRONMENT=.venv-win"
)

"%UV_CMD%" run comic.py %*
set "EXITCODE=%ERRORLEVEL%"

popd >nul
exit /b %EXITCODE%
