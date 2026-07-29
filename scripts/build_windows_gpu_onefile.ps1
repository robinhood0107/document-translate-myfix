param(
    [Parameter(Mandatory = $true)]
    [string]$AppVersion,

    [string]$PythonPath = "python",

    [switch]$Fast,

    [int]$Jobs = 0,

    [string]$OutputName = "comic-translate-gpu",

    [string]$OutputDir = "build/nuitka-gpu-onefile",

    [switch]$NoCompression
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
if ($NoCompression) {
  $extraNuitkaArgs += "--onefile-no-compression"
}

& $PythonPath -m nuitka `
  --onefile `
  $extraNuitkaArgs `
  --onefile-tempdir-spec="{CACHE_DIR}/{COMPANY}/$OutputName/{VERSION}" `
  --onefile-child-grace-time=30000 `
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
