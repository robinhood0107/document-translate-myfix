@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul

set "VENV_DIR=%SCRIPT_DIR%.venv-win-cuda13"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "TORCH_LIB=%VENV_DIR%\Lib\site-packages\torch\lib"
set "TENSORRT_LIBS=%VENV_DIR%\Lib\site-packages\tensorrt_libs"
set "CUDNN_BIN=%VENV_DIR%\Lib\site-packages\nvidia\cudnn\bin"
set "CUBLAS_BIN=%VENV_DIR%\Lib\site-packages\nvidia\cublas\bin"
set "CUDA_RUNTIME_BIN=%VENV_DIR%\Lib\site-packages\nvidia\cuda_runtime\bin"
set "CUDA_NVRTC_BIN=%VENV_DIR%\Lib\site-packages\nvidia\cuda_nvrtc\bin"
set "NVJITLINK_BIN=%VENV_DIR%\Lib\site-packages\nvidia\nvjitlink\bin"
set "BOOTSTRAP_CMD="

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
"%PYTHON_EXE%" -c "import importlib.metadata as md, sys, torch; required={'torch':'2.11.0+cu130','torchvision':'0.26.0+cu130','setuptools':'80.9.0','einops':'0.8.2'}; required_any=('onnxruntime-gpu','PySide6'); ok=all(md.version(k)==v for k,v in required.items()) and all(md.version(k) for k in required_any) and getattr(torch.version,'cuda',None)=='13.0'; raise SystemExit(0 if ok else 1)" >nul 2>&1
if errorlevel 1 (
    echo [bootstrap] Installing pinned runtime for .venv-win-cuda13...
    "%PYTHON_EXE%" -m pip install --upgrade pip wheel setuptools==80.9.0
    if errorlevel 1 (
        popd >nul
        exit /b 1
    )
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    if errorlevel 1 (
        popd >nul
        exit /b 1
    )
    "%PYTHON_EXE%" -m pip install --upgrade einops==0.8.2
    if errorlevel 1 (
        popd >nul
        exit /b 1
    )
    "%PYTHON_EXE%" -m pip install --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0+cu130 torchvision==0.26.0+cu130
    if errorlevel 1 (
        popd >nul
        exit /b 1
    )
    "%PYTHON_EXE%" -m pip install --upgrade --force-reinstall setuptools==80.9.0
    if errorlevel 1 (
        popd >nul
        exit /b 1
    )
)

"%PYTHON_EXE%" -c "import importlib.metadata as md, sys, torch; required={'torch':'2.11.0+cu130','torchvision':'0.26.0+cu130','setuptools':'80.9.0','einops':'0.8.2'}; required_any=('onnxruntime-gpu','PySide6'); ok=all(md.version(k)==v for k,v in required.items()) and all(md.version(k) for k in required_any) and getattr(torch.version,'cuda',None)=='13.0'; raise SystemExit(0 if ok else 1)"
if errorlevel 1 (
    echo [bootstrap] .venv-win-cuda13 verification failed.
    popd >nul
    exit /b 1
)

if exist "%TORCH_LIB%" set "PATH=%TORCH_LIB%;%PATH%"
if exist "%TENSORRT_LIBS%" set "PATH=%TENSORRT_LIBS%;%PATH%"
if exist "%CUDNN_BIN%" set "PATH=%CUDNN_BIN%;%PATH%"
if exist "%CUBLAS_BIN%" set "PATH=%CUBLAS_BIN%;%PATH%"
if exist "%CUDA_RUNTIME_BIN%" set "PATH=%CUDA_RUNTIME_BIN%;%PATH%"
if exist "%CUDA_NVRTC_BIN%" set "PATH=%CUDA_NVRTC_BIN%;%PATH%"
if exist "%NVJITLINK_BIN%" set "PATH=%NVJITLINK_BIN%;%PATH%"
set "CUDA_PATH="
set "CUDA_PATH_V13_1="
set "CUDA_HOME="
set "CUDA_ROOT="
set "CUDNN_PATH="
set "PYTHONNOUSERSITE=1"
set "PYTHONWARNINGS=ignore:pkg_resources is deprecated as an API:UserWarning"

if defined COMIC_BOOTSTRAP_ONLY (
    echo [bootstrap] .venv-win-cuda13 is ready.
    popd >nul
    exit /b 0
)

echo [bootstrap] Preparing required local runtime models...
"%PYTHON_EXE%" -c "from modules.utils.download import ensure_startup_runtime_models; ensure_startup_runtime_models(prefer_cuda=True)"
if errorlevel 1 (
    echo [bootstrap] Required local model preparation failed.
    popd >nul
    exit /b 1
)

"%PYTHON_EXE%" comic.py %*
set "EXITCODE=%ERRORLEVEL%"

popd >nul
exit /b %EXITCODE%
