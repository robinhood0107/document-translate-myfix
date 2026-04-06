@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul

set "MODE=%~1"
if "%MODE%"=="" set "MODE=all"

if /I "%MODE%"=="help" goto :help
if /I "%MODE%"=="--help" goto :help
if /I "%MODE%"=="-h" goto :help

call :pyfind || goto :fail

echo [setup] repo-root=%SCRIPT_DIR%
echo [setup] mode=%MODE%
echo [setup] python=%PYTHON_EXE%

if /I "%MODE%"=="all" (
    call :setupwin || goto :fail
    call :setupc13 || goto :fail
    goto :done
)

if /I "%MODE%"=="win" (
    call :setupwin || goto :fail
    goto :done
)

if /I "%MODE%"=="cuda13" (
    call :setupc13 || goto :fail
    goto :done
)

echo [setup] unknown mode: %MODE%
goto :help

:pyfind
where /q py
if not errorlevel 1 (
    py -3.12 -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_CMD=py"
        set "PYTHON_ARGS=-3.12"
        goto :pyok
    )
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_CMD=py"
        set "PYTHON_ARGS=-3"
        goto :pyok
    )
)

where /q python
if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_CMD=python"
        set "PYTHON_ARGS="
        goto :pyok
    )
)

echo [setup] Python 3.12+ was not found.
echo [setup] Install Python 3.12 and ensure py.exe or python.exe is available on PATH.
exit /b 1

:pyok
for /f "usebackq delims=" %%I in (`%PYTHON_CMD% %PYTHON_ARGS% -c "import sys; print(sys.executable)"`) do set "PYTHON_EXE=%%I"
exit /b 0

:setupwin
echo [setup] === building .venv-win ===
    call :venv ".venv-win" || exit /b 1
    call :boot ".venv-win" || exit /b 1
    call :base ".venv-win" || exit /b 1
    call :chkwin ".venv-win" || exit /b 1
exit /b 0

:setupc13
echo [setup] === building .venv-win-cuda13 ===
    call :venv ".venv-win-cuda13" || exit /b 1
    call :boot ".venv-win-cuda13" || exit /b 1
    call :base ".venv-win-cuda13" || exit /b 1
    call :c13pkgs ".venv-win-cuda13" || exit /b 1
    call :chkc13 ".venv-win-cuda13" || exit /b 1
exit /b 0

:venv
set "ENV_DIR=%~1"
set "VENV_PY=%SCRIPT_DIR%%ENV_DIR%\Scripts\python.exe"
if exist "%VENV_PY%" (
    echo [setup] reusing %ENV_DIR%
    exit /b 0
)
echo [setup] creating %ENV_DIR%
%PYTHON_CMD% %PYTHON_ARGS% -m venv "%ENV_DIR%"
if errorlevel 1 (
    echo [setup] failed to create %ENV_DIR%
    exit /b 1
)
exit /b 0

:boot
set "ENV_DIR=%~1"
set "VENV_PY=%SCRIPT_DIR%%ENV_DIR%\Scripts\python.exe"
echo [setup] upgrading pip tooling in %ENV_DIR%
"%VENV_PY%" -m pip install --upgrade pip wheel setuptools
if errorlevel 1 (
    echo [setup] failed to upgrade pip tooling in %ENV_DIR%
    exit /b 1
)
exit /b 0

:base
set "ENV_DIR=%~1"
set "VENV_PY=%SCRIPT_DIR%%ENV_DIR%\Scripts\python.exe"
echo [setup] installing base requirements into %ENV_DIR%
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [setup] failed to install requirements.txt into %ENV_DIR%
    exit /b 1
)
exit /b 0

:c13pkgs
set "ENV_DIR=%~1"
set "VENV_PY=%SCRIPT_DIR%%ENV_DIR%\Scripts\python.exe"
echo [setup] installing CUDA 13 packages into %ENV_DIR%
"%VENV_PY%" -m pip install --upgrade --pre "onnx==1.21.0" "onnxruntime-gpu==1.25.0.dev20260403004"
if errorlevel 1 (
    echo [setup] failed to install ONNX Runtime nightly build for CUDA 13
    exit /b 1
)
"%VENV_PY%" -m pip install --upgrade --extra-index-url https://pypi.nvidia.com ^
    "nvidia-cublas==13.3.0.5" ^
    "nvidia-cudnn-cu13==9.20.0.48" ^
    "tensorrt-cu13==10.16.0.72"
if errorlevel 1 (
    echo [setup] failed to install TensorRT/cuDNN CUDA 13 packages
    exit /b 1
)
exit /b 0

:chkwin
set "ENV_DIR=%~1"
set "VENV_PY=%SCRIPT_DIR%%ENV_DIR%\Scripts\python.exe"
echo [setup] verifying %ENV_DIR%
"%VENV_PY%" -c "import PySide6, requests, numpy, yaml, onnxruntime; print('OK base runtime')"
if errorlevel 1 (
    echo [setup] base runtime verification failed for %ENV_DIR%
    exit /b 1
)
exit /b 0

:chkc13
set "ENV_DIR=%~1"
set "VENV_PY=%SCRIPT_DIR%%ENV_DIR%\Scripts\python.exe"
echo [setup] verifying %ENV_DIR% with scripts\verify_cuda13_runtime.py
"%VENV_PY%" scripts\verify_cuda13_runtime.py
if errorlevel 1 (
    echo [setup] CUDA 13 runtime verification failed for %ENV_DIR%
    echo [setup] Confirm CUDA v13.1 is installed at C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
    exit /b 1
)
exit /b 0

:done
echo [setup] done
echo [setup] run_comic.bat uses .venv-win
echo [setup] run_comic_cuda13.bat uses .venv-win-cuda13
popd >nul
exit /b 0

:help
echo.
echo Usage:
echo   setup.bat
echo   setup.bat all
echo   setup.bat win
echo   setup.bat cuda13
echo.
echo Modes:
echo   all     Create or update both .venv-win and .venv-win-cuda13
echo   win     Create or update only .venv-win
echo   cuda13  Create or update only .venv-win-cuda13
echo.
echo Notes:
echo   - Python 3.12+ is required.
echo   - CUDA 13 mode expects NVIDIA CUDA Toolkit v13.1 to already be installed.
echo   - run_comic.bat uses .venv-win.
echo   - run_comic_cuda13.bat uses .venv-win-cuda13.
echo.
popd >nul
exit /b 0

:fail
echo [setup] failed
popd >nul
exit /b 1
