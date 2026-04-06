# Comic Translate
English | [한국어](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/i18n/README_ko.md) | [Français](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/i18n/README_fr.md) | [简体中文](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/i18n/README_zh-CN.md)

<img src="https://i.imgur.com/QUVK6mK.png">

## Overview
This fork focuses on a practical local pipeline for comic translation:

- `Custom Local Server(Gemma)` for local LLM translation
- `PaddleOCR VL` local Docker services for OCR
- `HunyuanOCR` local `llama.cpp` server for OCR

The goal of this fork is to make the local Gemma + OCR runtime path practical and reproducible for day-to-day comic translation.

## What This Fork Updates
- Removed legacy account/login dependencies from the local workflow
- Split local Gemma translation into a dedicated runtime/config path
- Added tracked OCR Docker runtime bundle in [paddleocr_vl_docker_files/README.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/paddleocr_vl_docker_files/README.md)
- Added tracked HunyuanOCR runtime bundle in [hunyuanocr_docker_files/README.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/hunyuanocr_docker_files/README.md)
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

## Docs Map
### Gemma
- Local server setup: [local-server-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/gemma/local-server-ko.md)

### OCR / Runtime
- OCR Docker bundle: [paddleocr_vl_docker_files/README.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/paddleocr_vl_docker_files/README.md)
- PaddleOCR VL 1.5 speed plan: [paddleocr-vl-15-speed-plan-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/ocr/paddleocr-vl-15-speed-plan-ko.md)
- HunyuanOCR runtime bundle: [hunyuanocr_docker_files/README.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/hunyuanocr_docker_files/README.md)
- HunyuanOCR setup guide: [local-server-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/hunyuan/local-server-ko.md)

### Repo Policy
- Benchmark branch / merge policy: [benchmark-branch-policy-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/repo/benchmark-branch-policy-ko.md)

### Rendering / History
- Rendering notes: [rendering-notes-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/rendering/rendering-notes-ko.md)
- Change log: [change-log-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/history/change-log-ko.md)

## Notes
- `/Sample/` is local-only and ignored by Git.
- Benchmark-specific tooling and reports are maintained on the `benchmarking/lab` branch.
