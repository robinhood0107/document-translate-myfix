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
- Docker image: `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu-offline`
- Runtime/model references:
  - [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
  - [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)

Start it:

```bash
docker compose -f paddleocr_vl_docker_files/docker-compose.yaml pull --policy always
docker compose -f paddleocr_vl_docker_files/docker-compose.yaml up -d --force-recreate
```

For bundle details, see [/paddleocr_vl_docker_files/README.md](/paddleocr_vl_docker_files/README.md).

## 4. Recommended app settings

- Workflow mode: `Stage-Batched Pipeline (Recommended)`
- OCR: `Optimal (HunyuanOCR / PaddleOCR VL)`
- Translator: `Custom Local Server(Gemma)` after the Gemma volume is prepared

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
- build target: Windows executable/portable packages built with `Nuitka`
- release order: local Windows Nuitka build verification, then `main` promotion, then Windows CI preflight, then tag release CI
- bundled scope: app, Python runtime, PySide6, torch/onnxruntime runtime, translations/resources
- not bundled: models, checkpoints, Docker runtimes, NVIDIA driver

Before promoting a release candidate to `main`, run the relevant Nuitka build scripts from Windows PowerShell and record the successful commands and `build/nuitka-*` outputs in the PR. WSL-only checks are not a substitute for this local Windows build verification.

If you need the full local runtime stack, follow the runtime setup steps in this quickstart after downloading the release package.
