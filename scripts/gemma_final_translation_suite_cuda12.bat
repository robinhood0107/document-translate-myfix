@echo off
setlocal

set "REPO_ROOT=%~dp0.."
set "PYTHON=%REPO_ROOT%\.venv-win\Scripts\python.exe"
set "RUNNER=%REPO_ROOT%\scripts\benchmark_gemma_final_translation.py"

if not exist "%PYTHON%" (
  echo Missing supported Python environment: %PYTHON%
  exit /b 2
)

"%PYTHON%" -B "%RUNNER%" %*
exit /b %ERRORLEVEL%
