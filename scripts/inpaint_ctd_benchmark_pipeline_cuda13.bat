@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"
set "PYTHON_EXE=%REPO_ROOT%\.venv-win-cuda13\Scripts\python.exe"
set "CUDA13_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
set "TENSORRT_LIBS=%REPO_ROOT%\.venv-win-cuda13\Lib\site-packages\tensorrt_libs"
set "CUDNN_BIN=%REPO_ROOT%\.venv-win-cuda13\Lib\site-packages\nvidia\cudnn\bin"
if not exist "%PYTHON_EXE%" (
    echo CUDA 13 benchmark environment not found: "%PYTHON_EXE%"
    exit /b 1
)
if exist "%CUDA13_BIN%" set "PATH=%CUDA13_BIN%;%PATH%"
if exist "%TENSORRT_LIBS%" set "PATH=%TENSORRT_LIBS%;%PATH%"
if exist "%CUDNN_BIN%" set "PATH=%CUDNN_BIN%;%PATH%"
if /I "%~1"=="help" goto :help
if "%~1"=="" (
  call "%PYTHON_EXE%" -u "%SCRIPT_DIR%inpaint_ctd_benchmark.py" --scope spotlight
) else (
  call "%PYTHON_EXE%" -u "%SCRIPT_DIR%inpaint_ctd_benchmark.py" %*
)
exit /b %ERRORLEVEL%
:help
echo Usage:
echo   scripts\inpaint_ctd_benchmark_pipeline_cuda13.bat
echo   scripts\inpaint_ctd_benchmark_pipeline_cuda13.bat --scope spotlight --corpus japan --case ctd-protect-lama-large-512px
goto :eof
