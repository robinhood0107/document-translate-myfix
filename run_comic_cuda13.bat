@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul

set "VENV_DIR=%SCRIPT_DIR%.venv-win-cuda13"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "TORCH_LIB=%VENV_DIR%\Lib\site-packages\torch\lib"
set "TENSORRT_LIBS=%VENV_DIR%\Lib\site-packages\tensorrt_libs"
set "CUDNN_BIN=%VENV_DIR%\Lib\site-packages\nvidia\cudnn\bin"
set "NVIDIA_CU13_BIN=%VENV_DIR%\Lib\site-packages\nvidia\cu13\bin\x86_64"
set "CUBLAS_BIN=%VENV_DIR%\Lib\site-packages\nvidia\cublas\bin"
set "CUDA_RUNTIME_BIN=%VENV_DIR%\Lib\site-packages\nvidia\cuda_runtime\bin"
set "CUDA_NVRTC_BIN=%VENV_DIR%\Lib\site-packages\nvidia\cuda_nvrtc\bin"
set "NVJITLINK_BIN=%VENV_DIR%\Lib\site-packages\nvidia\nvjitlink\bin"
set "BOOTSTRAP_CMD="

if /I "%COMIC_VERIFY_ONLY%"=="1" (
    for %%F in (
        "comic.py"
        "controller.py"
        "app\version.py"
        "requirements-base.txt"
        "requirements-cuda13.txt"
        "docker-compose.yaml"
        "paddleocr_vl_docker_files\docker-compose.yaml"
        "resources\translations\compiled\ct_ko.qm"
        "scripts\prepare_gemma_runtime.ps1"
        "scripts\verify_windows_runtime.py"
    ) do (
        if not exist "%SCRIPT_DIR%%%~F" (
            echo [verify] Missing launcher-source file: %%~F
            popd >nul
            exit /b 1
        )
    )
    echo [verify] CUDA 13 launcher-source contract is valid.
    popd >nul
    exit /b 0
)

py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1 && set "BOOTSTRAP_CMD=py -3.12"
if not defined BOOTSTRAP_CMD python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1 && set "BOOTSTRAP_CMD=python"
if not defined BOOTSTRAP_CMD python3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1 && set "BOOTSTRAP_CMD=python3"
if not defined BOOTSTRAP_CMD py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1 && set "BOOTSTRAP_CMD=py -3"

if not defined BOOTSTRAP_CMD (
    echo Python 3.12 or newer is required to create %VENV_DIR%.
    popd >nul
    exit /b 1
)

if not exist "%PYTHON_EXE%" (
    echo [bootstrap] Creating virtual environment: %VENV_DIR%
    call %BOOTSTRAP_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        popd >nul
        exit /b 1
    )
)

echo [bootstrap] Checking .venv-win-cuda13 (CUDA 13.x)...
"%PYTHON_EXE%" -B -s scripts\verify_windows_runtime.py --requirements requirements-cuda13.txt --expected-cuda 13.0 >nul 2>&1
if errorlevel 1 (
    echo [bootstrap] Installing pinned runtime for .venv-win-cuda13...
    "%PYTHON_EXE%" -m pip install --upgrade pip==26.0.1 wheel==0.46.3 setuptools==80.9.0
    if errorlevel 1 (
        popd >nul
        exit /b 1
    )
    "%PYTHON_EXE%" -m pip install -r requirements-cuda13.txt
    if errorlevel 1 (
        popd >nul
        exit /b 1
    )
)

"%PYTHON_EXE%" -B -s scripts\verify_windows_runtime.py --requirements requirements-cuda13.txt --expected-cuda 13.0
if errorlevel 1 (
    echo [bootstrap] .venv-win-cuda13 verification failed.
    popd >nul
    exit /b 1
)

if exist "%TORCH_LIB%" set "PATH=%TORCH_LIB%;%PATH%"
if exist "%TENSORRT_LIBS%" set "PATH=%TENSORRT_LIBS%;%PATH%"
if exist "%CUDNN_BIN%" set "PATH=%CUDNN_BIN%;%PATH%"
if exist "%NVIDIA_CU13_BIN%" set "PATH=%NVIDIA_CU13_BIN%;%PATH%"
if exist "%CUBLAS_BIN%" set "PATH=%CUBLAS_BIN%;%PATH%"
if exist "%CUDA_RUNTIME_BIN%" set "PATH=%CUDA_RUNTIME_BIN%;%PATH%"
if exist "%CUDA_NVRTC_BIN%" set "PATH=%CUDA_NVRTC_BIN%;%PATH%"
if exist "%NVJITLINK_BIN%" set "PATH=%NVJITLINK_BIN%;%PATH%"
set "CUDA_PATH="
set "CUDA_PATH_V13_1="
set "CUDA_HOME="
set "CUDA_ROOT="
set "CUDNN_PATH="
set "QT_QPA_PLATFORM=windows"
set "PYTHONNOUSERSITE=1"
set "PYTHONWARNINGS=ignore:pkg_resources is deprecated as an API:UserWarning"

if defined COMIC_BOOTSTRAP_ONLY (
    echo [bootstrap] .venv-win-cuda13 is ready.
    popd >nul
    exit /b 0
)

if not defined COMIC_SKIP_STARTUP_MODELS if not defined COMIC_SMOKE_EXIT_MS (
    echo [bootstrap] Preparing required local runtime models...
    "%PYTHON_EXE%" -c "from modules.utils.download import ensure_startup_runtime_models; ensure_startup_runtime_models(prefer_cuda=True)"
    if errorlevel 1 (
        echo [bootstrap] Required local model preparation failed.
        popd >nul
        exit /b 1
    )
)

"%PYTHON_EXE%" comic.py %*
set "EXITCODE=%ERRORLEVEL%"

popd >nul
exit /b %EXITCODE%
