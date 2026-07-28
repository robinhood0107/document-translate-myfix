@echo off
setlocal

set "REPO_ROOT=%~dp0.."
set "PYTHON=%REPO_ROOT%\.venv-win-cuda13\Scripts\python.exe"
set "RUNNER=%REPO_ROOT%\scripts\benchmark_cold_cache_finalization.py"

if not exist "%PYTHON%" (
  echo Missing supported Python environment: %PYTHON%
  exit /b 2
)

if "%~1"=="" goto usage

"%PYTHON%" -B "%RUNNER%" %*
exit /b %ERRORLEVEL%

:usage
echo Usage:
echo   %~nx0 describe --output-dir C:\validation\cold-cache\protocol
echo   %~nx0 run-pipeline --family paddle-workers --input-dir C:\samples --output-dir C:\validation\cold-cache\workers
echo   %~nx0 run-translation --family gemma-model --source-summary C:\validation\source-summary.json --output-dir C:\validation\cold-cache\gemma-model
echo   %~nx0 run-cache --scenario global-ocr --input-dir C:\samples --output-dir C:\validation\cold-cache\global-ocr
echo   %~nx0 run-cache --scenario project --input-dir C:\samples --output-dir C:\validation\cold-cache\project
echo.
echo The output directory must be new and outside the Git repository.
exit /b 2
