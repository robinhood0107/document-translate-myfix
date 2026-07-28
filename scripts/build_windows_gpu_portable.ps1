param(
    [Parameter(Mandatory = $true)]
    [string]$AppVersion,

    [string]$PythonPath = "python",

    [switch]$Fast,

    [int]$Jobs = 0,

    [string]$OutputDir = "build/nuitka-gpu",

    [string]$OutputName = "comic-translate-gpu"
)

$ErrorActionPreference = "Stop"
Write-Warning "Unofficial manual Nuitka tool: this output is not an official Comic Translate release asset."

$extraNuitkaArgs = @()
if (-not $Fast) {
  $extraNuitkaArgs += "--low-memory"
}
if ($Jobs -gt 0) {
  $extraNuitkaArgs += "--jobs=$Jobs"
}

& $PythonPath -m nuitka `
  --standalone `
  $extraNuitkaArgs `
  --assume-yes-for-downloads `
  --enable-plugin=pyside6 `
  --module-parameter=torch-disable-jit=no `
  --include-package=torch `
  --include-package=torchvision `
  --nofollow-import-to=pytest `
  --nofollow-import-to=onnxruntime.tools `
  --nofollow-import-to=onnxruntime.transformers `
  --noinclude-pytest-mode=nofollow `
  --noinclude-unittest-mode=nofollow `
  --noinclude-dlls=onnxruntime/capi/onnxruntime_providers_tensorrt.dll `
  --windows-console-mode=disable `
  --windows-icon-from-ico=resources/icons/icon.ico `
  --company-name="ComicLabs" `
  --product-name="Comic Translate GPU" `
  --file-version="$AppVersion.0" `
  --product-version="$AppVersion" `
  --output-dir="$OutputDir" `
  --output-filename="$OutputName" `
  --include-windows-runtime-dlls=yes `
  --include-data-dir=resources=resources `
  --include-data-dir=music=music `
  comic.py
