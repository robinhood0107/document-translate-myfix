# Comic Translate
English | [한국어](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/i18n/README_ko.md) | [Français](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/i18n/README_fr.md) | [简体中文](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/i18n/README_zh-CN.md)

<img src="https://i.imgur.com/QUVK6mK.png">

## Overview
This fork focuses on a practical local pipeline for comic translation:

- `Custom Local Server(Gemma)` for local LLM translation
- `PaddleOCR VL` local Docker services for OCR
- Windows benchmark launchers for `.venv-win` and `.venv-win-cuda13`
- repo-local benchmark result logging in `./banchmark_result_log`
- auto-generated benchmark report docs with charts

The goal of this fork is not just to run the app, but to make the local Gemma + PaddleOCR VL path measurable, reproducible, and tunable.

## What This Fork Updates
- Removed legacy account/login dependencies from the local workflow
- Split local Gemma translation into a dedicated runtime/config path
- Added tracked OCR Docker runtime bundle in [paddleocr_vl_docker_files/README.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/paddleocr_vl_docker_files/README.md)
- Added benchmark presets, stage metrics, suite launchers, and generated reports
- Tuned the current translation-only baseline around:
  - Gemma `temperature=0.6`, `top_k=64`, `top_p=0.95`, `min_p=0.0`
  - Gemma `n_gpu_layers=23`, `threads=12`, `ctx=4096`
  - `paddleocr-server=cpu`, `paddleocr-vllm=gpu`

## Quick Start
### App
Run the app from your prepared environment.

```bash
uv run comic.py
```

### Benchmark
Windows launchers are split by environment:

- CUDA12 / standard: [scripts/benchmark_suite.bat](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/scripts/benchmark_suite.bat)
- CUDA13: [scripts/benchmark_suite_cuda13.bat](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/scripts/benchmark_suite_cuda13.bat)

Current benchmark outputs are stored under:

```text
./banchmark_result_log
```

Generated charts and the latest report are written to:

- [report-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/banchmark_report/report-ko.md)
- [latest](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/assets/benchmarking/latest)

## Docs Map
### Benchmark
- Workflow: [workflow-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/benchmark/workflow-ko.md)
- Usage and result reading: [usage-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/benchmark/usage-ko.md)
- Checklist: [checklist-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/benchmark/checklist-ko.md)
- Resource strategy: [resource-strategy-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/benchmark/resource-strategy-ko.md)
- Architecture and code separation: [architecture-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/benchmark/architecture-ko.md)
- Optimization journey: [optimization-journey-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/benchmark/optimization-journey-ko.md)
- Results history: [results-history-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/benchmark/results-history-ko.md)
- Generated report: [report-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/banchmark_report/report-ko.md)

### Gemma
- Local server setup: [local-server-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/gemma/local-server-ko.md)
- Profile history: [profiles-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/gemma/profiles-ko.md)
- Translation tuning summary: [translation-optimization-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/gemma/translation-optimization-ko.md)

## Notes
- `/Sample/` is local-only and ignored by Git.
- `/banchmark_result_log/` is local-only and ignored by Git.
- Benchmark-specific experiment logic lives in scripts/docs; the core pipeline only exposes lightweight measurement hooks.
