# Comic Translate
English | [한국어](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/i18n/README_ko.md) | [Français](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/i18n/README_fr.md) | [简体中文](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/i18n/README_zh-CN.md)

This fork focuses on a practical local comic translation workflow on Windows.

## What Changed In This Fork
- Added a dedicated local Gemma translation path for `Custom Local Server(Gemma)`.
- Added a tracked local OCR runtime bundle in [paddleocr_vl_docker_files/README.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/paddleocr_vl_docker_files/README.md).
- Tuned the local translation path around Gemma + PaddleOCR VL.
- Split benchmark tooling and reports into the dedicated `benchmarking/lab` branch.

## Usage
### Run the app
Use your prepared environment and start the desktop app:

```bash
uv run comic.py
```

### Local Gemma + OCR setup
- Gemma local server guide: [local-server-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/gemma/local-server-ko.md)
- OCR Docker bundle: [paddleocr_vl_docker_files/README.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/paddleocr_vl_docker_files/README.md)

### Benchmarking
Benchmark tooling, presets, and reports are maintained on the `benchmarking/lab` branch.

## Notes
- `main` and `develop` are product branches.
- Benchmark-specific docs, presets, reports, and chart assets stay on `benchmarking/lab`.
- `/Sample/` and `/banchmark_result_log/` are local-only and ignored by Git.
