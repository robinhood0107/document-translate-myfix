@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT=%%~fI"
set "RUNTIME=%~1"
shift

if /I "%RUNTIME%"=="cuda12" (
    set "VENV_NAME=.venv-win"
    set "REQUIREMENTS=requirements-cuda12.txt"
    set "EXPECTED_CUDA=12.8"
    set "CUDA_BIN_NAME=cu12"
    set "SETUP_BAT=setup.bat"
) else if /I "%RUNTIME%"=="cuda13" (
    set "VENV_NAME=.venv-win-cuda13"
    set "REQUIREMENTS=requirements-cuda13.txt"
    set "EXPECTED_CUDA=13.0"
    set "CUDA_BIN_NAME=cu13"
    set "SETUP_BAT=setup_cuda13.bat"
) else (
    echo [ERROR] Unsupported Windows runtime: %RUNTIME%
    exit /b 2
)

chcp 65001 >nul
color 07
title Comic Translate - %RUNTIME%

if /I "%COMIC_VERIFY_ONLY%"=="1" (
    for %%F in (
        "comic.py"
        "controller.py"
        "app\version.py"
        "%REQUIREMENTS%"
        "scripts\run_windows.cmd"
        "scripts\windows_install_state.py"
        "scripts\verify_windows_runtime.py"
        "resources\translations\compiled\ct_ko.qm"
    ) do if not exist "%ROOT%\%%~F" (
        echo [verify] Missing launcher-source file: %%~F
        exit /b 1
    )
    echo [verify] %RUNTIME% run launcher-source contract is valid.
    exit /b 0
)

set "VENV_DIR=%ROOT%\%VENV_NAME%"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo [ERROR] %VENV_NAME% is not installed.
    echo         Run %SETUP_BAT% before starting Comic Translate.
    exit /b 1
)

pushd "%ROOT%" >nul
echo [preflight] Checking %VENV_NAME%...
"%PYTHON_EXE%" -B -s "%SCRIPT_DIR%verify_windows_runtime.py" --requirements "%ROOT%\%REQUIREMENTS%" --expected-cuda %EXPECTED_CUDA% --metadata-only
if errorlevel 1 (
    echo [ERROR] The pinned Python runtime is not ready. Run the matching setup BAT.
    popd >nul
    exit /b 1
)

set "STATE_ENV=%TEMP%\comic-translate-%RUNTIME%-%RANDOM%-%RANDOM%.cmd"
"%PYTHON_EXE%" -B -s "%SCRIPT_DIR%windows_install_state.py" preflight --runtime %RUNTIME% --requirements %REQUIREMENTS% --emit-cmd > "%STATE_ENV%"
if errorlevel 1 (
    if exist "%STATE_ENV%" del /q "%STATE_ENV%" >nul 2>&1
    echo [ERROR] Setup state validation failed. Run the matching setup BAT.
    popd >nul
    exit /b 1
)
for /f "usebackq tokens=1,* delims==" %%A in ("%STATE_ENV%") do set "%%A=%%B"
del /q "%STATE_ENV%" >nul 2>&1

set "TORCH_LIB=%VENV_DIR%\Lib\site-packages\torch\lib"
set "TENSORRT_LIBS=%VENV_DIR%\Lib\site-packages\tensorrt_libs"
set "CUDNN_BIN=%VENV_DIR%\Lib\site-packages\nvidia\cudnn\bin"
set "NVIDIA_CUDA_BIN=%VENV_DIR%\Lib\site-packages\nvidia\%CUDA_BIN_NAME%\bin\x86_64"
set "CUBLAS_BIN=%VENV_DIR%\Lib\site-packages\nvidia\cublas\bin"
set "CUDA_RUNTIME_BIN=%VENV_DIR%\Lib\site-packages\nvidia\cuda_runtime\bin"
set "CUDA_NVRTC_BIN=%VENV_DIR%\Lib\site-packages\nvidia\cuda_nvrtc\bin"
set "NVJITLINK_BIN=%VENV_DIR%\Lib\site-packages\nvidia\nvjitlink\bin"
if exist "%TORCH_LIB%" set "PATH=%TORCH_LIB%;%PATH%"
if exist "%TENSORRT_LIBS%" set "PATH=%TENSORRT_LIBS%;%PATH%"
if exist "%CUDNN_BIN%" set "PATH=%CUDNN_BIN%;%PATH%"
if exist "%NVIDIA_CUDA_BIN%" set "PATH=%NVIDIA_CUDA_BIN%;%PATH%"
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
    echo [preflight] %VENV_NAME% and sealed runtime state are ready.
    popd >nul
    exit /b 0
)

echo [launch] Starting Comic Translate with sealed image %LLAMA_CPP_IMAGE%...
"%PYTHON_EXE%" -B -s comic.py %1 %2 %3 %4 %5 %6 %7 %8 %9
set "EXITCODE=%ERRORLEVEL%"
popd >nul
exit /b %EXITCODE%
