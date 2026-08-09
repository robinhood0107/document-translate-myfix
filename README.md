[English](README.md) | [한국어](README_ko.md)

# Comic Translate Fork

This repository is a local-first fork of upstream `comic-translate` that started from the upstream `v2.6.7` codebase and then diverged with product-specific runtime, OCR, workflow, and Windows setup changes.

The fork's product release version is `1.5.1`. Upstream `2.7.1` is recorded
separately as the latest selective-backport lineage and is not this fork's
product version.

The fork is maintained around a practical desktop workflow:

- local Gemma translation runtime support
- local OCR runtimes such as `PaddleOCR VL` and `HunyuanOCR`
- Windows-oriented setup and launch tooling
- selective manual backports from upstream `v2.7.0` and `v2.7.1`
- benchmark work isolated from product branches

## Important Features

- Local Gemma translation runtime for desktop-first translation workflows.
- Persistent full-identity Gemma result caching plus user-approved exact translation memory.
- Local OCR runtimes with optimal routing between `HunyuanOCR` and `PaddleOCR VL`.
- Inpainting add, exclude, and restore tools with saved mask and patch state.
- TXT/MD source export and translation import with OCR and translation correction dictionaries.
- CBZ/CBR comic archive import with lazy page materialization.
- Bottom-left automatic pipeline status panel with overlay locking and latest-result preview.

## Supporting Features

- Reuse-only OCR preflight checks avoid restarting already running local OCR containers.
- Managed local Docker runtimes stop without deleting their containers so later runs can reuse them.
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
- Gemma GGUF files imported once into the versioned external model volume
- HunyuanOCR GGUF and mmproj files
- PaddleOCR VL Docker/runtime bundle files

## Release Policy

This repository now uses a strict `main + develop + tag` model.

- `develop` is the integration branch for upcoming product work.
- `main` is the shipping baseline.
- Official releases are created only from `vX.Y.Z` version tags that point to commits already contained in `main`.
- The official Windows asset is a deterministic
  `comic-translate-vX.Y.Z-windows-launcher-source.zip` plus
  `SHA256SUMS.txt`.
- The ZIP contains allowlisted product source, both first-run Windows
  launchers, pinned CUDA12/CUDA13 requirements, runtime Compose/config files,
  the Gemma/PaddleOCR preparation scripts, translations/resources, README files, and the
  license.
- Virtual environments, models, caches, benchmark runners/results, local
  paths, and secrets are not bundled. The launchers install their supported
  environment on first run.
- Release candidates must reproduce the ZIP and pass both extracted
  launchers with `COMIC_VERIFY_ONLY=1` before `main` promotion.
- The retained Nuitka PowerShell scripts are unofficial manual tools and do
  not produce official release assets.
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
- added a SQLite result-cache fast path that can skip Gemma startup on complete hits
- added separately approved exact translation memory with conservative matching and explicit import/export controls

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

### `v2.7.0 -> v2.7.1`

The `v2.7.1` round selectively applies the upstream fixes that matter to this fork:

- PSD import stabilization with explicit font-catalog preparation and safer decode fallback logging
- main-thread-safe `QTimer.singleShot(...)` dispatch for async UI callbacks
- list thumbnail loading reworked around `QImage` in the worker thread and `QPixmap` conversion on the main thread
- import menu cleanup so `PSD` appears next to `Project File`
- upstream selective-backport lineage recorded as `2.7.1`

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

Prepare the versioned Gemma model volume once from Windows PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\prepare_gemma_runtime.ps1 -Mode Prepare `
  -ModelPath 'C:\ExampleWorkspace\models\gemma-4-26B-IQ4_NL.gguf'
```

Then use `Custom Local Server(Gemma)` in the app. The managed runtime validates the ready manifest and model size, mounts the prepared volume read-only, and starts or recreates the container only when needed. To explicitly recompute the model hash, run the same script with `-Mode Verify`.

The **User Dictionaries** settings page also controls the persistent block-result cache and exact translation memory. Result-cache entries use the complete translation/runtime identity. Exact source-to-translation pairs bypass Gemma only after explicit approval; imported approved entries require confirmation. These databases contain sensitive local text, remain in the app user-data directory, and are never silently deleted after a lock or corruption error. See [the translation-memory guide](docs/gemma/translation-memory.md).

### 3. Optional local OCR runtimes

HunyuanOCR:

```bash
docker compose -f hunyuanocr_docker_files/docker-compose.yaml up -d
```

PaddleOCR VL uses the tracked bundle under [paddleocr_vl_docker_files/README.md](paddleocr_vl_docker_files/README.md).

The optional full-page `Spotting:` route has an independent projector,
container, named volume, and cache identity. Its setup is documented in
[paddleocr_vl_spotting_docker_files/README.md](paddleocr_vl_spotting_docker_files/README.md).

Managed containers are stopped and preserved after stage completion, cancellation, or app shutdown. Use `docker compose down` only when you explicitly want to reset or remove a runtime.

### 4. Recommended OCR setting

In Settings, choose:

- `Default (existing auto: MangaOCR / PPOCR / Pororo...)` to keep legacy OCR routing
- `Optimal (HunyuanOCR / PaddleOCR VL)` to route Chinese to `HunyuanOCR` and Japanese/other languages to `PaddleOCR VL`

For the stage-batched folder workflow, `Settings > PaddleOCR VL Settings`
also provides a managed exact persistent OCR cache. It stores raw OCR results
and diagnostics, never crop images, and is disabled automatically for custom
endpoints.

`Settings > Project` also contains a validated project checkpoint store. A
one-time migration enables it once while preserving any later user choice.
Detection geometry, raw PaddleOCR-VL results, lossless cleaned images, final
masks, and encoded render outputs can be restored before their runtimes start.
Translation text remains owned by the `.ctpr`; the sidecar stores only its
validated stage signature, so a full project hit skips detector, Paddle,
Gemma, inpainter, and renderer inference. Current OCR and translation
dictionary rules are still applied exactly once, and changing a dictionary
invalidates only its consumed result and downstream stages. Reusable manifests
and content-addressed artifacts live in a
`<project>.ctpr.cache` folder beside the `.ctpr` file. Missing, locked, or
damaged checkpoint data never prevents the project from opening or processing;
the affected stages are recalculated.

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

Every managed llama.cpp runtime (Gemma, PaddleOCR VL, PaddleOCR VL Spotting,
HunyuanOCR, MangaLMM, and the Router) uses one CUDA server image:

- Default: `ghcr.io/ggml-org/llama.cpp:server-cuda13`
- Also supported: `ghcr.io/ggml-org/llama.cpp:server-cuda`

## Reference Setup Docs

- [docs/setup/quickstart.md](docs/setup/quickstart.md)
- [docs/setup/quickstart-ko.md](docs/setup/quickstart-ko.md)
- [docs/gemma/local-server-ko.md](docs/gemma/local-server-ko.md)
- [docs/hunyuan/local-server-ko.md](docs/hunyuan/local-server-ko.md)
- [paddleocr_vl_docker_files/README.md](paddleocr_vl_docker_files/README.md)

## Repository Documents

- [rules.md](rules.md)
- [Codebase map and OCR strategy boundaries (Korean)](docs/architecture/codebase-map-ko.md)
- [Managed llama.cpp-only runtime policy (Korean)](docs/runtime/managed-llamacpp-only-ko.md)
- [docs/gemma/local-server-ko.md](docs/gemma/local-server-ko.md)
- [docs/hunyuan/local-server-ko.md](docs/hunyuan/local-server-ko.md)
- [docs/repo/github-rulesets-public-free-ko.md](docs/repo/github-rulesets-public-free-ko.md)

## Legacy Localized READMEs

The old localized README files under `docs/i18n/` are no longer the source of truth for this fork.

Use:

- [README.md](README.md) for the English source of truth
- [README_ko.md](README_ko.md) for the Korean source of truth
