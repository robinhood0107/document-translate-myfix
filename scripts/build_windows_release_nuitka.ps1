param(
    [Parameter(Mandatory = $true)]
    [string]$AppVersion
)

$ErrorActionPreference = "Stop"

python -m nuitka `
  --standalone `
  --low-memory `
  --assume-yes-for-downloads `
  --enable-plugin=pyside6 `
  --module-parameter=torch-disable-jit=no `
  --nofollow-import-to=sympy `
  --nofollow-import-to=mpmath `
  --nofollow-import-to=isympy `
  --nofollow-import-to=onnxruntime.tools `
  --nofollow-import-to=onnxruntime.transformers `
  --windows-console-mode=disable `
  --windows-icon-from-ico=resources/icons/icon.ico `
  --company-name="ComicLabs" `
  --product-name="Comic Translate" `
  --file-version="$AppVersion.0" `
  --product-version="$AppVersion" `
  --output-dir=build/nuitka `
  --output-filename=comic-translate `
  --include-windows-runtime-dlls=yes `
  --include-data-dir=resources=resources `
  --include-data-dir=music=music `
  comic.py
