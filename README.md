[English](README.md) | [한국어](README_ko.md)

# Comic Translate Fork

This repository is a local-first fork of upstream `comic-translate` that started from the upstream `v2.6.7` codebase and then diverged with product-specific runtime, OCR, workflow, and Windows setup changes.

The fork is maintained around a practical desktop workflow:

- local Gemma translation runtime support
- local OCR runtimes such as `PaddleOCR VL` and `HunyuanOCR`
- Windows-oriented setup and launch tooling
- selective manual backports from upstream `v2.7.0` and `v2.7.1`
- benchmark work isolated from product branches

## Release Policy

This repository now uses a strict `main + develop + tag` model.

- `develop` is the integration branch for upcoming product work.
- `main` is the shipping baseline.
- Releases are created from version tags on `main`.
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
- added `setup.bat` to create and verify `.venv-win` and `.venv-win-cuda13`
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

Audit document:

- [docs/history/v267-to-v270-backport-audit.md](docs/history/v267-to-v270-backport-audit.md)
- [docs/history/v267-to-v270-backport-audit-ko.md](docs/history/v267-to-v270-backport-audit-ko.md)

### `v2.7.0 -> v2.7.1`

The `v2.7.1` round selectively applies the upstream fixes that matter to this fork:

- PSD import stabilization with explicit font-catalog preparation and safer decode fallback logging
- main-thread-safe `QTimer.singleShot(...)` dispatch for async UI callbacks
- list thumbnail loading reworked around `QImage` in the worker thread and `QPixmap` conversion on the main thread
- import menu cleanup so `PSD` appears next to `Project File`
- app version bump to `2.7.1`

Audit document:

- [docs/history/v270-to-v271-backport-audit.md](docs/history/v270-to-v271-backport-audit.md)
- [docs/history/v270-to-v271-backport-audit-ko.md](docs/history/v270-to-v271-backport-audit-ko.md)

## Quick Start

### 1. Prepare the Windows environments

```bat
setup.bat
```

This creates or updates:

- `.venv-win`
- `.venv-win-cuda13`

### 2. Launch the application

Default Windows runtime:

```bat
run_comic.bat
```

CUDA13 runtime:

```bat
run_comic_cuda13.bat
```

### 3. Optional local translation runtime

Start the local Gemma server from the repository root:

```bash
docker compose up -d
```

Then use `Custom Local Server(Gemma)` in the app.

### 4. Optional local OCR runtimes

HunyuanOCR:

```bash
docker compose -f hunyuanocr_docker_files/docker-compose.yaml up -d
```

PaddleOCR VL uses the tracked bundle under [paddleocr_vl_docker_files/README.md](paddleocr_vl_docker_files/README.md).

### 5. Recommended OCR setting

In Settings, choose:

- `Default (existing auto: MangaOCR / PPOCR / Pororo...)` to keep legacy OCR routing
- `Optimal (HunyuanOCR / PaddleOCR VL)` to route Chinese to `HunyuanOCR` and Japanese/other languages to `PaddleOCR VL`

## Repository Documents

- [rules.md](rules.md)
- [docs/history/change-log.md](docs/history/change-log.md)
- [docs/history/change-log-ko.md](docs/history/change-log-ko.md)
- [docs/gemma/local-server-ko.md](docs/gemma/local-server-ko.md)
- [docs/hunyuan/local-server-ko.md](docs/hunyuan/local-server-ko.md)
- [docs/repo/github-rulesets-public-free-ko.md](docs/repo/github-rulesets-public-free-ko.md)

## Legacy Localized READMEs

The old localized README files under `docs/i18n/` are no longer the source of truth for this fork.

Use:

- [README.md](README.md) for the English source of truth
- [README_ko.md](README_ko.md) for the Korean source of truth
