# Quickstart Guide

This guide is the shortest path from a fresh checkout to a working local setup.

## 1. Prerequisites

- Windows 10/11 x64
- [Official Python 3.12.10 Windows installer (64-bit)](https://www.python.org/downloads/release/python-31210/)
- Docker Desktop with its WSL2 backend and GPU support enabled
- NVIDIA driver and a CUDA-compatible GPU
- About 6–8 GiB for the selected venv plus the exact size of model files that are still missing
- Git only when using a source checkout

The pinned `mahotas` dependency currently ships a Windows wheel only for
CPython 3.12. A machine with only Python 3.13/3.14 is therefore not supported;
installing 3.12 x64 side by side is sufficient. The launcher prefers
`py -3.12` and never imports global packages into its venv.
Keep the `py` launcher enabled in the installer. Adding Python to PATH is
optional because `py -3.12` is checked first.

## 2. Provision once

Setup and launch are separate entry points. Setup does all the slow work; the
run launcher does none of it.

For the CUDA12 path (Python cu128 plus llama.cpp `server-cuda`):

```bat
setup.bat
```

For the CUDA13 Python path (cu130; Docker deliberately uses the same broadly
compatible llama.cpp `server-cuda` image as the CUDA12 setup):

```bat
setup_cuda13.bat
```

Both are backed by one PowerShell bootstrap. System Python is used only to
create the selected venv; global packages and Python environment variables are
ignored. Setup starts Docker Desktop when necessary and pulls the selected
llama.cpp image only when it is missing.

The non-interactive setup prepares:

- the selected `.venv-win` or `.venv-win-cuda13` and pinned packages
- RT-DETR v2 ONNX, font-detector ONNX, CTD Torch/ONNX and positive-claim ONNX
- LaMa large and LaMa MPE application models
- HunyuanOCR Q8 model/mmproj
- PaddleOCR VL 1.6 model/mmproj
- `gemma-4-26B-IQ4_NL.gguf` (about 13.58 GiB)

`setup_full.bat` and `setup_full_cuda13.bat` add MangaLMM and PaddleOCR VL
Spotting to that list. The core tier is a subset of the full tier, so running
`setup.bat` after `setup_full.bat` does not discard the extra volumes.

Docker model sources are reused from `models\managed-runtime-sources` under
the installation folder. Interrupted downloads resume on the
next run. Completion requires exact size/SHA-256 validation and real model
load smokes. Use `setup.bat --doctor` or
`setup_cuda13.bat --doctor` for a read-only report.

The BAT keeps the classic Command Prompt host, uses UTF-8 with Consolas 16px
for the current window only, and does not change the registry. The console shows
package-metadata substeps (without loading CUDA DLLs), each model boundary,
compact 10% download updates, runtime
preparation boundaries, and the final result. Full child-command output remains
in the timestamped `logs\bootstrap\*-detail.log` file. A setup opened by
double-click stays on its final `DONE!` or `FAILED!` screen until a key is
pressed; automated checks can set `COMIC_NO_PAUSE=1`.

## 3. Launch

```bat
run_comic.bat
```

```bat
run_comic_cuda13.bat
```

The run launchers verify the venv and atomic install state, export the exact
setup-selected llama.cpp image, and start the application. They never install,
download, pull, create, or reseal anything. Missing core state requires the
matching setup BAT; missing MangaLMM/Spotting state requires setup_full before
page processing starts.

Before creating a container, the CUDA13 launcher still compares the image's
`NVIDIA_REQUIRE_CUDA` value with the driver compatibility version. After setup,
it reuses volumes whose ready manifest, image ID, and model sizes still match,
so later launches do not repeat full hashes or GPU smokes.

The CUDA13 path uses the official ONNX Runtime CUDA13 nightly feed.

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

### Installation locations

- CUDA12 venv: `.venv-win` under the installation folder (about 5.9 GB currently)
- CUDA13 venv: `.venv-win-cuda13` (about 4.3 GB)
- GGUF source cache: `models\managed-runtime-sources`
- bootstrap logs/state: `logs\bootstrap` and `.comic-bootstrap`
- prepared GGUF files: Docker external named volumes

When the repository is on `D:`, its venvs, source cache, logs, and state stay on
`D:`. Docker named volumes follow Docker Desktop's configured disk-image
location, which may still be on `C:`. The three default source models total
about 16.50 GiB and Docker stores a similarly sized prepared copy. Bootstrap
does not require a fixed 60 GiB; it checks each missing file's exact size plus
512 MiB immediately before downloading it.

A separate CUDA Toolkit install is not required. PyTorch and ONNX Runtime CUDA
user-space DLLs live in the selected venv and are preferred by the launcher;
llama.cpp user-space CUDA libraries live inside the selected Docker image. The
NVIDIA display driver and Docker Desktop/WSL2 GPU passthrough remain system
prerequisites and cannot be isolated inside a Python venv.

## 4. Manual local runtime management

Normal first runs do not require the commands below. Use them only for explicit
full hash verification, custom volumes, or optional runtime maintenance.

### Gemma local translation runtime

- Compose file: `/docker-compose.yaml`
- Docker image: `ghcr.io/ggml-org/llama.cpp:server-cuda`
- Runtime reference: [llama.cpp](https://github.com/ggml-org/llama.cpp)
- Model reference: [Gemma](https://ai.google.dev/gemma)

The launcher prepares this exact model automatically. For manual preparation:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\prepare_gemma_runtime.ps1 -Mode Prepare `
  -ModelPath 'C:\ExampleWorkspace\models\gemma-4-26B-IQ4_NL.gguf'
```

Then choose `Custom Local Server(Gemma)` in the app. The managed runtime mounts the prepared volume read-only and starts the exact prepared container automatically.

### PaddleOCR VL local runtime

- Compose file: `/paddleocr_vl_docker_files/docker-compose.yaml`
- Inference backend: pinned llama.cpp with the official PaddleOCR-VL 1.6
  GGUF and multimodal projector
- Request path: detector crop -> direct llama.cpp `OCR:` request. The managed
  crop route does not start a PaddleX layout pipeline or a vLLM process.
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
stopped containers, and force-recreates stale containers. The application
ends the OCR stage explicitly before releasing the model; failure to confirm
release falls back to a normal `stop`. The automatic path never uses `down`.

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

### HunyuanOCR local runtime

This is the engine the Optimal choice uses for Chinese. Prepare its versioned
external model volume once, under the same contract as the other managed
engines. Put these two files in one model directory:

- `HunyuanOCR.Q8_0.gguf`
- `HunyuanOCR.mmproj-Q8_0.gguf`

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\prepare_hunyuanocr_llamacpp_runtime.ps1 `
  -Mode Prepare `
  -ModelDirectory 'C:\ExampleWorkspace\models\HunyuanOCR-GGUF'
```

The tool reuses a file already present in the volume when its SHA-256 matches
instead of copying it again. It then runs a CUDA model-load smoke test and
records the result in the ready manifest. Use `-Mode Verify` to re-check only.

### When a prepared volume is suddenly rejected

Every preparation script accepts `-Mode Auto`: it immediately reuses a valid
seal and prepares an empty volume. When upstream refreshes the llama.cpp tag the
image digest moves, so the models stay correct while the ready manifest no
longer matches; only then does setup's `Auto` choose `Reseal` and recover without
the original source files. The running app reports the stale seal and requires
the matching setup BAT; it never repairs the volume. See the Korean guide at
[docs/runtime/managed-volume-repair-ko.md](../runtime/managed-volume-repair-ko.md).

Omitting `-ModelDirectory` searches the repository's gitignored `testmodel/` and
its immediate subdirectories first. The launcher instead supplies its shared
ignored `models\managed-runtime-sources` cache under the installation folder.

HunyuanOCR previously bind-mounted the `testmodel` folder and read generic
environment names such as `LLAMA_CTX_SIZE`, which it shared with Gemma. It now
uses the prepared volume and dedicated `HUNYUAN_OCR_LLAMA_*` names, so tuning
one engine no longer changes another.

## 5. Recommended app settings

- Workflow mode: `Stage-Batched Pipeline (Recommended)`
- OCR: `Optimal (HunyuanOCR / PaddleOCR VL)`
- Translator: `Custom Local Server(Gemma)` after the Gemma volume is prepared

Project stage checkpoints are enabled once by the validated migration under
`Settings > Project`, while later user choices are preserved. Save the `.ctpr`
file before using its cache management actions. The adjacent `.ctpr.cache`
folder is disposable and is never required to open the project. When valid, it
restores detection, raw OCR, inpaint masks and cleaned artifacts, and render
outputs. Translation content stays in the project file and is accepted only
when its sidecar signature matches.

Routing summary:

- Chinese -> `HunyuanOCR`
- Japanese / other languages -> `PaddleOCR VL`

## 6. Optional ntfy notifications

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

## 7. Upstream model/runtime references used by this product

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

## 8. Related docs

- [/README.md](/README.md)
- [/README_ko.md](/README_ko.md)
- [/docs/gemma/local-server-ko.md](/docs/gemma/local-server-ko.md)
- [/docs/hunyuan/local-server-ko.md](/docs/hunyuan/local-server-ko.md)

## 9. Official Windows release packages

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
