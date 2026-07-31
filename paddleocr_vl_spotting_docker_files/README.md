# PaddleOCR-VL full-page Spotting runtime

This bundle is intentionally separate from the detector + crop OCR runtime.
It runs the official PaddleOCR-VL 1.6 `Spotting:` contract directly through
the pinned llama.cpp server.

## Fixed contract

- prompt: `Spotting:`
- native location-token output: enabled with `--special`
- coordinate space: normalized `0..1000` quadrilaterals
- projector metadata: `clip.vision.image_max_pixels=1605632`
- model volume:
  `comic-translate-paddleocr-vl-spotting-llamacpp-models-v2`

The regular crop OCR route keeps its original projector and
`clip.vision.image_max_pixels=1003520`. The two projectors, volumes,
containers, ports, settings, cache identities, and parsers are not shared.

## Prepare once

Keep the official crop model and projector in one source directory:

- `PaddleOCR-VL-1.6-GGUF.gguf`
- `PaddleOCR-VL-1.6-GGUF-mmproj.gguf`

Run from Windows PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\prepare_paddleocr_spotting_llamacpp_runtime.ps1 `
  -Mode Prepare `
  -ModelDirectory 'C:\ExampleWorkspace\models\PaddleOCR-VL-1.6-GGUF'
```

The preparation tool verifies the original crop projector, copies it to a
temporary file, changes only the official Spotting pixel-budget metadata,
checks the exact derived SHA-256, and atomically stores it under a distinct
name. The source crop projector is never modified.

Preparation records the ready manifest only after an actual CUDA request
returns native location tokens. Full verification is available with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\prepare_paddleocr_spotting_llamacpp_runtime.ps1 `
  -Mode Verify
```

## Product behavior

The app starts this runtime only when `PaddleOCR VL Spotting (Full Page)` is
selected. It sends one full-page PNG, parses the complete native response,
maps normalized coordinates back to the original image, and retains detector
geometry as the destructive edit authority. Ambiguous and unmatched regions
fail closed for review; there is no hidden crop OCR fallback.

The managed container is normally stopped and preserved. Automatic code never
uses `docker compose down`.
