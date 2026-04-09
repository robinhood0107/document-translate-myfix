# Change Log

## Base: upstream `v2.6.7`

This fork started from upstream `comic-translate` `v2.6.7` and then diverged through a series of local product changes.

## Local Improvement Track

### Rendering and manual workflow

- documented the header/render flow before changing behavior
- introduced shared render-policy helpers for force color, alignment, and outline state
- extended serialized text state with `vertical_alignment`, `source_rect`, and `block_anchor`
- reorganized the render panel into clearer user-facing groups
- kept manual render, batch render, and webtoon render on the same shared policy path

### Windows runtime and repo operations

- added Windows launchers for the default runtime and CUDA13 runtime
- added `setup.bat` to provision `.venv-win` and `.venv-win-cuda13`
- hardened local hooks and CI checks
- standardized branch policy and later removed the old `codex/` branch prefix
- switched the repo policy to `main + develop + tag`

### OCR reliability and diagnostics

- improved OCR parity between one-page auto and batch flows
- added block-local detection fallback for OCR
- added suspicious short-result retry behavior
- widened text masks and added bubble residue cleanup for inpainting/OCR cleanup interactions
- added OCR diagnostics and runtime selection tests

### Local model/runtime integration

- added a local Gemma server flow and runtime tuning
- added PaddleOCR VL integration and tuned defaults
- added HunyuanOCR integration for local OCR serving
- added `Optimal (HunyuanOCR / PaddleOCR VL)` routing with language-aware runtime selection

### Benchmark tooling and branch separation

- introduced benchmark toolkit scripts and one-click runners
- split benchmark assets away from product branches
- codified `benchmarking/lab` as the benchmark-only branch

## Selective Backport Track

### `v2.6.7 -> v2.7.0`

Selected upstream `v2.7.0` features were adapted into this fork rather than merged wholesale.

See:

- [v267-to-v270-backport-audit.md](v267-to-v270-backport-audit.md)
- [v267-to-v270-backport-audit-ko.md](v267-to-v270-backport-audit-ko.md)

### `v2.7.0 -> v2.7.1`

Selected upstream `v2.7.1` fixes were adapted for this fork, focusing on:

- PSD import stability
- main-thread callback safety
- list thumbnail loader stability
- PSD menu cleanup
- version bump to `2.7.1`

See:

- [v270-to-v271-backport-audit.md](v270-to-v271-backport-audit.md)
- [v270-to-v271-backport-audit-ko.md](v270-to-v271-backport-audit-ko.md)
