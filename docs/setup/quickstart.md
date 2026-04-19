# Quickstart Guide

This guide is the shortest path from a fresh checkout to a working local setup.

## 1. Prerequisites

- Windows 10/11
- Python 3.11
- Git
- Docker Desktop with GPU support enabled
- NVIDIA driver / CUDA-compatible GPU if you want local Gemma, HunyuanOCR, or PaddleOCR VL acceleration

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

## 3. Optional local runtimes

### Gemma local translation runtime

- Compose file: `/docker-compose.yaml`
- Docker image: `ghcr.io/ggml-org/llama.cpp:server-cuda`
- Runtime reference: [llama.cpp](https://github.com/ggml-org/llama.cpp)
- Model reference: [Gemma](https://ai.google.dev/gemma)

Start it:

```bash
docker compose pull --policy always
docker compose up -d --force-recreate
```

Then choose `Custom Local Server(Gemma)` in the app.

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

For bundle details, see [/paddleocr_vl_docker_files/README.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/paddleocr_vl_docker_files/README.md).

## 4. Recommended app settings

- Workflow mode: `Stage-Batched Pipeline (Recommended)`
- OCR: `Optimal (HunyuanOCR / PaddleOCR VL)`
- Translator: `Custom Local Server(Gemma)` if your local Gemma runtime is running

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

- [/README.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/README.md)
- [/README_ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/README_ko.md)
- [/docs/gemma/local-server-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/gemma/local-server-ko.md)
- [/docs/hunyuan/local-server-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/hunyuan/local-server-ko.md)
