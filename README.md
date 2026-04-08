# Comic Translate

This repository is an unofficial clone project that started from the upstream `comic-translate` `v2.6.7` codebase and then diverged with local product changes.

It is maintained as a practical local-first fork for comic translation workflows, with an emphasis on:

- local Gemma translation runtime support
- local OCR runtime integrations such as PaddleOCR VL and HunyuanOCR
- Windows-first setup and launch tooling
- selective manual backports of user-facing features from upstream `v2.7.0`

## Project Base And History

This repository did **not** start from current upstream `main`.

It started from an upstream `v2.6.7` snapshot and then accumulated local changes in stages:

1. local runtime cleanup and fork-specific workflow simplification
2. Gemma local server/runtime integration and default tuning
3. PaddleOCR VL and HunyuanOCR product integration
4. Windows environment bootstrap and CUDA13 launcher support
5. selective manual backport work based on the upstream `v2.6.7...v2.7.0` compare

The `v2.7.0` upgrade work in this repository was performed by **manual compare-based selection and adaptation**, not by merging or copying the upstream release wholesale.

## What Is Backported From Upstream `v2.7.0`

The current branch selectively backports the user-facing parts that fit this fork safely:

- configurable keyboard shortcuts
- PSD export and PSD import
- chapter-aware export flow
- startup recent-project actions such as copy path and delete file
- current project rename/move from the title bar
- multi-select text formatting
- undo text render as a single undo step
- unlimited extra context for the custom translator
- Hebrew and Croatian target languages, plus RTL handling for Persian, Hebrew, and Arabic
- Windows snap support for the frameless window path
- selected webtoon and list-view performance/behavior fixes
- Claude 4.6 Sonnet label refresh

A detailed audit of the upstream `v2.6.7 -> v2.7.0` delta and exactly what was brought into this fork lives in:

- [docs/history/v267-to-v270-backport-audit-ko.md](docs/history/v267-to-v270-backport-audit-ko.md)

## Current Repo Notes

- benchmark runners, generated benchmark results, and benchmark reports are kept off the product branches
- `/banchmark_result_log/` and `/Sample/` are local-only and ignored by Git
- CUDA13 runtime setup uses a dedicated launcher: `run_comic_cuda13.bat`
- general Windows runtime setup is handled by `setup.bat`

Useful docs in this repository:

- [docs/gemma/local-server-ko.md](docs/gemma/local-server-ko.md)
- [docs/hunyuan/local-server-ko.md](docs/hunyuan/local-server-ko.md)
- [docs/repo/benchmark-branch-policy-ko.md](docs/repo/benchmark-branch-policy-ko.md)

## Copyright And Licensing

This repository is a derivative project of the upstream `comic-translate` codebase and keeps the repository license in [LICENSE](LICENSE).

Important points:

- the upstream base code and any preserved notices remain subject to their original license terms
- this repository's fork-specific code and documentation are distributed inside the same repository, but they do **not** erase upstream attribution requirements
- upstream release names, version numbers, and compare references are used here only to document provenance and compatibility history
- third-party models, OCR runtimes, translation services, Docker images, and other external dependencies keep their own licenses and usage terms

If you redistribute or publish derivative work from this repository, preserve the upstream license and attribution trail, and review any third-party component licenses separately.

## Status Of Legacy Localized READMEs

The old localized README files under `docs/i18n/` are no longer the source of truth for this fork.

Use this root `README.md` for the current repository overview, and use the backport audit document for the `v2.6.7 -> v2.7.0` migration history.
