#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_common import load_preset, stage_runtime_files, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage runtime files for a benchmark preset.")
    parser.add_argument("--preset", required=True, help="Preset name or preset file path")
    parser.add_argument("--runtime-dir", required=True, help="Output directory for staged runtime files")
    args = parser.parse_args()

    preset, preset_path = load_preset(args.preset)
    staged = stage_runtime_files(preset, args.runtime_dir)
    write_json(Path(args.runtime_dir) / "preset_resolved.json", preset)

    print(f"Preset: {preset.get('name', args.preset)}")
    print(f"Preset source: {preset_path}")
    print(f"Runtime dir: {staged['runtime_dir']}")
    print(f"Gemma compose: {staged['gemma']['compose_path']}")
    print(f"OCR compose: {staged['ocr']['compose_path']}")
    print(f"App settings: {staged['app_settings_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
