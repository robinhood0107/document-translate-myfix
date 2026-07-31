# Quickstart Guide

This guide is the shortest path from a fresh checkout to a working local setup.

## 1. Prerequisites

- Windows 10/11
- Python 3.12 or newer
- Git
- Docker Desktop with GPU support enabled
- NVIDIA driver / CUDA-compatible GPU if you want local Gemma, HunyuanOCR, or PaddleOCR VL acceleration
- At least 60 GiB free on `C:` for the initial Gemma volume preparation check

## 2. Clone and launch

From the repository root, use one of the supported Windows launchers:

```bat
run_comic.bat
```

CUDA13 path:

```bat
run_comic_cuda13.bat
```

The launchers bootstrap `.venv-win` or `.venv-win-cuda13` automatically.
The CUDA13 path uses the official ONNX Runtime CUDA13 nightly feed because CUDA13 GPU wheels are not the default PyPI `onnxruntime-gpu` package yet.

If you prefer to create a venv manually and install everything with one pip command, use one of these runtime-specific requirement files:

```bat
py -3.12 -m venv .venv-win
.venv-win\Scripts\python.exe -m pip install -r requirements-cuda12.txt
```

```bat
py -3.12 -m venv .venv-win-cuda13
.venv-win-cuda13\Scripts\python.exe -m pip install -r requirements-cuda13.txt
```

`requirements.txt` is the default Windows CUDA12 alias. Shared dependencies live in `requirements-base.txt`.

## 3. Optional local runtimes

### Gemma local translation runtime

- Compose file: `/docker-compose.yaml`
- Docker image: `ghcr.io/ggml-org/llama.cpp@sha256:22e0e3bfe967af4fd1df6a918022abbfd4e72e4d40a4769e616a4176790acbcb`
- Runtime reference: [llama.cpp](https://github.com/ggml-org/llama.cpp)
- Model reference: [Gemma](https://ai.google.dev/gemma)

Prepare the versioned external model volume once from Windows PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\prepare_gemma_runtime.ps1 -Mode Prepare `
  -CandidateModelPath 'C:\ExampleWorkspace\models\Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ4_XS.gguf' `
  -LegacyModelPath 'C:\ExampleWorkspace\models\gemma-4-26B-IQ4_NL.gguf'
```

Then choose `Custom Local Server(Gemma)` in the app. The managed runtime mounts the prepared volume read-only and starts the exact prepared container automatically.

### HunyuanOCR local runtime

- Compose file: `/hunyuanocr_docker_files/docker-compose.yaml`
- Docker image: `ghcr.io/ggml-org/llama.cpp:server-cuda`
- Runtime/model references:
  - [HunyuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR)
  - [llama.cpp](https://github.com/ggml-org/llama.cpp)

Required local model files:

- `HunyuanOCR-BF16.gguf`
- `mmproj-HunyuanOCR-BF16.gguf`

Start it:

```bash
docker compose -f hunyuanocr_docker_files/docker-compose.yaml pull --policy always
docker compose -f hunyuanocr_docker_files/docker-compose.yaml up -d --force-recreate
```

### PaddleOCR VL local runtime

- Compose file: `/paddleocr_vl_docker_files/docker-compose.yaml`
- Inference backend: pinned llama.cpp with the official PaddleOCR-VL 1.6
  GGUF and multimodal projector
- Layout frontend: pinned PaddleX/PaddleOCR image
- Runtime/model references:
  - [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
  - [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)

Put these two files in one model directory:

- `PaddleOCR-VL-1.6-GGUF.gguf`
- `PaddleOCR-VL-1.6-GGUF-mmproj.gguf`

Prepare and verify the versioned external model volume once from Windows
PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\prepare_paddleocr_llamacpp_runtime.ps1 `
  -Mode Prepare `
  -ModelDirectory 'C:\ExampleWorkspace\models\PaddleOCR-VL-1.6-GGUF'
```

The app then starts the managed runtime automatically when PaddleOCR VL is
needed. It mounts the prepared model volume read-only, reuses only exact
stopped containers, and force-recreates stale containers. After an OCR stage,
llama.cpp unloads the model after five idle seconds while the lightweight
containers remain available; failure to confirm unload falls back to a normal
`stop`. The automatic path never uses `down`.

The stage-batched folder workflow can also reuse exact crop results from the
persistent OCR cache configured under `Settings > PaddleOCR VL Settings`.

For bundle details, see [/paddleocr_vl_docker_files/README.md](/paddleocr_vl_docker_files/README.md).

### Optional PaddleOCR-VL full-page Spotting route

Full-page Spotting is a separate OCR choice. It does not replace or modify the
default detector + crop OCR projector. Prepare its dedicated named volume:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\prepare_paddleocr_spotting_llamacpp_runtime.ps1 `
  -Mode Prepare `
  -ModelDirectory 'C:\ExampleWorkspace\models\PaddleOCR-VL-1.6-GGUF'
```

This route fixes the official `Spotting:` prompt, `--special` location-token
mode, and `1,605,632` projector pixel budget. The crop route keeps its original
`1,003,520` projector. See
[/paddleocr_vl_spotting_docker_files/README.md](/paddleocr_vl_spotting_docker_files/README.md).

## 4. Recommended app settings

- Workflow mode: `Stage-Batched Pipeline (Recommended)`
- OCR: `Optimal (HunyuanOCR / PaddleOCR VL)`
- Translator: `Custom Local Server(Gemma)` after the Gemma volume is prepared

Project stage checkpoints are available as a default-off preview under
`Settings > Project`. Save the `.ctpr` file before using its cache management
actions. The adjacent `.ctpr.cache` folder is disposable and is never required
to open the project. When valid, it restores detection, raw OCR, inpaint masks
and cleaned artifacts, and render outputs. Translation content stays in the
project file and is accepted only when its sidecar signature matches.

Routing summary:

- Chinese -> `HunyuanOCR`
- Japanese / other languages -> `PaddleOCR VL`

## 5. Optional ntfy notifications

Open:

- `Settings > Notifications`

Configure:

- enable ntfy notifications
- server URL
- topic
- optional access token
- success / failure / cancellation delivery toggles

This app sends **plain-text only** ntfy notifications and keeps the message body below the default ntfy text limit documented by ntfy. It does not send attachments.

Official ntfy references:

- [ntfy publish docs](https://docs.ntfy.sh/publish/)
- [ntfy config docs](https://docs.ntfy.sh/config/)

## 6. Upstream model/runtime references used by this product

Detection / masking:

- [RT-DETR v2](https://huggingface.co/ogkalu/comic-text-and-bubble-detector)
- [ComicTextDetector (CTD)](https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.3)
- [Font Detector](https://huggingface.co/gyrojeff/YuzuMarker.FontDetection)

OCR:

- [MangaOCR](https://huggingface.co/kha-white/manga-ocr-base)
- [MangaOCR ONNX](https://huggingface.co/mayocream/manga-ocr-onnx)
- [Pororo OCR](https://huggingface.co/ogkalu/pororo)
- [RapidOCR / PPOCRv5](https://www.modelscope.cn/models/RapidAI/RapidOCR)
- [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)
- [HunyuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR)

Inpainting:

- [AOT](https://huggingface.co/ogkalu/aot-inpainting)
- [AnimeMangaInpainting / LaMa legacy](https://github.com/Sanster/models/releases/tag/AnimeMangaInpainting)
- [lama_large_512px](https://huggingface.co/dreMaz/AnimeMangaInpainting)
- [lama_mpe](https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.3)
- [MI-GAN](https://github.com/Sanster/models/releases/tag/migan)

## 7. Related docs

- [/README.md](/README.md)
- [/README_ko.md](/README_ko.md)
- [/docs/gemma/local-server-ko.md](/docs/gemma/local-server-ko.md)
- [/docs/hunyuan/local-server-ko.md](/docs/hunyuan/local-server-ko.md)

## 8. Official Windows release packages

Official Windows release packages are published only from `vX.Y.Z` tags that point to commits already contained in `main`.

- release trigger: Git tag push
- accepted tag shape: `vX.Y.Z`
- official assets:
  `comic-translate-vX.Y.Z-windows-launcher-source.zip` and
  `SHA256SUMS.txt`
- release order: local deterministic bundle and extracted-launcher
  verification, `main` promotion, Windows CI preflight, then tag release CI
- bundled scope: allowlisted product source, launchers, CUDA12/CUDA13
  requirements, Docker Compose/config, Gemma/PaddleOCR preparation tooling,
  translations/resources, README, and LICENSE
- not bundled: venvs, models, checkpoints, caches, benchmark tools/results,
  Python/CUDA runtimes, NVIDIA driver, local paths, or secrets

Before promoting a release candidate to `main`, build the ZIP from `HEAD`,
verify its manifest and SHA-256, extract it into a new folder, and run both
launchers with `COMIC_VERIFY_ONLY=1`. Record the Windows commands, ZIP hash,
and both launcher results in the PR.

The launchers bootstrap their pinned environment on first normal run. The
retained Nuitka scripts are unofficial manual tools and are not release gates.
