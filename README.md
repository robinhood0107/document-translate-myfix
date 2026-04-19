[English](README.md) | [한국어](README_ko.md)

# Comic Translate Fork

This repository is a local-first fork of upstream `comic-translate` that started from the upstream `v2.6.7` codebase and then diverged with product-specific runtime, OCR, workflow, and Windows setup changes.

The fork is maintained around a practical desktop workflow:

- local Gemma translation runtime support
- local OCR runtimes such as `PaddleOCR VL` and `HunyuanOCR`
- Windows-oriented setup and launch tooling
- selective manual backports from upstream `v2.7.0` and `v2.7.1`
- benchmark work isolated from product branches

## Important Features

- Local Gemma translation runtime for desktop-first translation workflows.
- Local OCR runtimes with optimal routing between `HunyuanOCR` and `PaddleOCR VL`.
- Inpainting add, exclude, and restore tools with saved mask and patch state.
- TXT/MD source export and translation import with OCR and translation correction dictionaries.
- CBZ/CBR comic archive import with lazy page materialization.
- Bottom-left automatic pipeline status panel with overlay locking and latest-result preview.

## Supporting Features

- Reuse-only OCR preflight checks avoid restarting already running local OCR containers.
- Automatic runs update the latest completed translated image preview page by page.
- Completion sounds support the system alert or repo-provided `music/*.wav` files.
- Windows launchers bootstrap `.venv-win` and `.venv-win-cuda13` automatically.
- Localized tooltips, help text, and compiled Qt translation assets stay aligned with UI changes.

## Origin and Upstream Attribution

This repository started from [ogkalu2/comic-translate](https://github.com/ogkalu2/comic-translate) and should be understood as a downstream, product-focused fork/derivative of that upstream work. It began from the upstream `v2.6.7` codebase and then diverged with local runtime, OCR, Windows, and workflow changes.

## License and Redistribution

The upstream project is distributed under the Apache License 2.0, and this fork keeps that license basis for the upstream-derived code in this repository.

If you publicly redistribute this fork or a modified build of it, the practical minimum checklist is:

- include the Apache 2.0 license text with the redistributed work
- keep upstream copyright, patent, attribution, and origin notices that still apply
- make it clear that this repository is a modified downstream fork/derivative, not the original upstream project
- add prominent notices for files you modified when redistributing the source
- review third-party asset licenses separately from the code license

## Third-Party Models and Runtime Notice

This project uses, downloads, or interoperates with third-party models, checkpoints, and runtime images. The copyright, license, and usage terms for those assets belong to their original authors and distributors, and this repository does not claim ownership of them. You are responsible for reviewing and complying with each upstream model/runtime license before using them.

### Models and runtimes used by the product code

Detection / masking:
- [RT-DETR v2](https://huggingface.co/ogkalu/comic-text-and-bubble-detector)
- [ComicTextDetector (CTD)](https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.3) (`comictextdetector.pt`, `comictextdetector.pt.onnx`)
- [Font Detector](https://huggingface.co/gyrojeff/YuzuMarker.FontDetection)

OCR:
- [MangaOCR](https://huggingface.co/kha-white/manga-ocr-base)
- [MangaOCR ONNX](https://huggingface.co/mayocream/manga-ocr-onnx)
- [Pororo OCR](https://huggingface.co/ogkalu/pororo)
- [PPOCRv5 / RapidOCR](https://www.modelscope.cn/models/RapidAI/RapidOCR)
- [PaddleOCR VL](https://github.com/PaddlePaddle/PaddleOCR)
- [HunyuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR)

Inpainting:
- [AOT](https://huggingface.co/ogkalu/aot-inpainting)
- [LaMa legacy runtime](https://github.com/Sanster/models/releases/tag/AnimeMangaInpainting)
- [lama_large_512px](https://huggingface.co/dreMaz/AnimeMangaInpainting)
- [lama_mpe / manga-image-translator inpainting checkpoint](https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.3)
- [MI-GAN](https://github.com/Sanster/models/releases/tag/migan)

Local translation/runtime:
- [Gemma](https://ai.google.dev/gemma) local GGUF runtime
- [llama.cpp](https://github.com/ggml-org/llama.cpp) Docker runtime image

### Auto-downloaded vs user-supplied assets

Automatically downloaded by the app when missing:
- CTD model files (`comictextdetector.pt`, `comictextdetector.pt.onnx`)
- Inpainting checkpoints such as `AOT`, `lama_large_512px`, and `lama_mpe`
- OCR checkpoints such as `MangaOCR`, `Pororo OCR`, and `PPOCRv5`

Provided separately by the user or local runtime bundle:
- Gemma GGUF files mounted into the local translation runtime
- HunyuanOCR GGUF and mmproj files
- PaddleOCR VL Docker/runtime bundle files

## Release Policy

This repository now uses a strict `main + develop + tag` model.

- `develop` is the integration branch for upcoming product work.
- `main` is the shipping baseline.
- Official releases are created only from `vX.Y.Z` version tags that point to commits already contained in `main`.
- Before creating a release tag, run the `Release Preflight` workflow on `main` and wait for a green Windows Nuitka build.
- The Windows release asset is built with `Nuitka` and published as a GitHub Release artifact from that tag.
- Models, checkpoints, and Docker runtimes are not bundled into the release executable and remain separately provisioned.
- `release/*` branches are not used.

The authoritative repository policy lives in [rules.md](rules.md).

## Fork Improvements Since Upstream `v2.6.7`

Local product work since the `v2.6.7` base has focused on a few technical areas.

### Rendering and manual editing

- documented and then centralized shared render policy behavior
- expanded render state with forced color, block anchoring, source rect tracking, and vertical alignment metadata
- refined the render panel layout, wording, and selection affordances
- kept manual rendering and batch/webtoon rendering on the same shared policy path

### Windows runtime and repo workflow

- added dedicated Windows launchers and a CUDA13 environment path
- made `run_comic.bat` and `run_comic_cuda13.bat` self-bootstrapping for local venv/runtime setup
- hardened local Git hook setup and CI validation flow
- cleaned branch policy and standardized on `feature/*`, `fix/*`, `chore/*`, `hotfix/*`, `benchmarking/lab`

### OCR quality and diagnostics

- added block-local OCR fallback and suspicious-result retry behavior
- added bubble residue cleanup and mask widening for residual glyph removal
- improved one-page auto / batch OCR parity and diagnostics
- added local PaddleOCR VL support and tuned its defaults
- added local HunyuanOCR support
- added `Optimal (HunyuanOCR / PaddleOCR VL)` OCR routing with run-start language confirmation and on-demand local runtime management

### Local translation runtime

- specialized the local Gemma translation server flow
- split custom translator modes and improved keyless local endpoint support
- normalized Gemma input and sanitized problematic glyphs
- aligned local sampler/runtime defaults with measured benchmark presets

### Benchmarking and branch separation

- added a dedicated benchmark toolkit and one-click runners
- separated benchmark harness/report assets from product branches
- codified the `benchmarking/lab` promotion boundary

## Selective Backports

This fork does not merge upstream releases wholesale. Instead, it performs selective compare-based backports and adapts only the changes that fit the local product structure.

### `v2.6.7 -> v2.7.0`

The `v2.7.0` backport brought in selected user-facing features such as:

- configurable keyboard shortcuts
- PSD export and PSD import
- chapter-aware export flow
- project rename/move actions
- startup recent-project actions such as copy path and delete file
- multi-select text block formatting
- undo text render as a single undo step
- unlimited extra context for the custom translator
- new target languages and improved RTL handling
- selected webtoon/list-view behavior fixes

Audit details stay on `develop` and are not promoted to the public `main` documentation set.

### `v2.7.0 -> v2.7.1`

The `v2.7.1` round selectively applies the upstream fixes that matter to this fork:

- PSD import stabilization with explicit font-catalog preparation and safer decode fallback logging
- main-thread-safe `QTimer.singleShot(...)` dispatch for async UI callbacks
- list thumbnail loading reworked around `QImage` in the worker thread and `QPixmap` conversion on the main thread
- import menu cleanup so `PSD` appears next to `Project File`
- app version bump to `2.7.1`

Audit details stay on `develop` and are not promoted to the public `main` documentation set.

## Quick Start

For a more explicit setup path, see:

- [docs/setup/quickstart.md](docs/setup/quickstart.md)
- [docs/setup/quickstart-ko.md](docs/setup/quickstart-ko.md)

### 1. Launch the application

The launchers create or update their own local runtime environment on first run.


Default Windows runtime:

```bat
run_comic.bat
```

CUDA13 runtime:

```bat
run_comic_cuda13.bat
```

### 2. Optional local translation runtime

Start the local Gemma server from the repository root:

```bash
docker compose up -d
```

Then use `Custom Local Server(Gemma)` in the app.

### 3. Optional local OCR runtimes

HunyuanOCR:

```bash
docker compose -f hunyuanocr_docker_files/docker-compose.yaml up -d
```

PaddleOCR VL uses the tracked bundle under [paddleocr_vl_docker_files/README.md](paddleocr_vl_docker_files/README.md).

### 4. Recommended OCR setting

In Settings, choose:

- `Default (existing auto: MangaOCR / PPOCR / Pororo...)` to keep legacy OCR routing
- `Optimal (HunyuanOCR / PaddleOCR VL)` to route Chinese to `HunyuanOCR` and Japanese/other languages to `PaddleOCR VL`

### 5. Optional ntfy notifications

Open `Settings > Notifications` to configure:

- completion sound
- ntfy server URL / topic / optional token
- whether to send on completion / failure / cancellation

The app sends plain-text-only ntfy notifications and keeps the message body below the default ntfy text limit documented by ntfy.

Official ntfy docs:

- [Publish notifications](https://docs.ntfy.sh/publish/)
- [Server configuration](https://docs.ntfy.sh/config/)

## Docker Images Used by This Repository

Tracked compose/runtime images used by the repo:

- Gemma local server: `ghcr.io/ggml-org/llama.cpp:server-cuda`
- HunyuanOCR local server: `ghcr.io/ggml-org/llama.cpp:server-cuda`
- PaddleOCR VL runtime: `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu-offline`

## Reference Setup Docs

- [docs/setup/quickstart.md](docs/setup/quickstart.md)
- [docs/setup/quickstart-ko.md](docs/setup/quickstart-ko.md)
- [docs/gemma/local-server-ko.md](docs/gemma/local-server-ko.md)
- [docs/hunyuan/local-server-ko.md](docs/hunyuan/local-server-ko.md)
- [paddleocr_vl_docker_files/README.md](paddleocr_vl_docker_files/README.md)

## Repository Documents

- [rules.md](rules.md)
Release-facing documentation for `main` is intentionally kept minimal; deeper history and audit notes remain on `develop`.
- [docs/gemma/local-server-ko.md](docs/gemma/local-server-ko.md)
- [docs/hunyuan/local-server-ko.md](docs/hunyuan/local-server-ko.md)
- [docs/repo/github-rulesets-public-free-ko.md](docs/repo/github-rulesets-public-free-ko.md)

## Legacy Localized READMEs

The root `README.md` and `README_ko.md` are the source of truth for the public branch documentation set.

Use:

- [README.md](README.md) for the English source of truth
- [README_ko.md](README_ko.md) for the Korean source of truth
