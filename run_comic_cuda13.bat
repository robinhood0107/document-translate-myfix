@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul

set "PYTHON_EXE=%SCRIPT_DIR%.venv-win-cuda13\Scripts\python.exe"
set "CUDA13_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
set "TENSORRT_LIBS=%SCRIPT_DIR%.venv-win-cuda13\Lib\site-packages\tensorrt_libs"
set "CUDNN_BIN=%SCRIPT_DIR%.venv-win-cuda13\Lib\site-packages\nvidia\cudnn\bin"

if not exist "%PYTHON_EXE%" (
    echo CUDA 13 test environment not found: "%PYTHON_EXE%"
    popd >nul
    exit /b 1
)

if exist "%CUDA13_BIN%" set "PATH=%CUDA13_BIN%;%PATH%"
if exist "%TENSORRT_LIBS%" set "PATH=%TENSORRT_LIBS%;%PATH%"
if exist "%CUDNN_BIN%" set "PATH=%CUDNN_BIN%;%PATH%"

"%PYTHON_EXE%" comic.py %*
set "EXITCODE=%ERRORLEVEL%"

popd >nul
exit /b %EXITCODE%
