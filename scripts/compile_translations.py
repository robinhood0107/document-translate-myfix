#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TS_DIR = ROOT / "resources" / "translations"
QM_DIR = TS_DIR / "compiled"
SOURCE_TARGETS = [
    str(ROOT / "app"),
    str(ROOT / "modules"),
    str(ROOT / "pipeline"),
    str(ROOT / "controller.py"),
    str(ROOT / "comic.py"),
]


def resolve_tool(name: str) -> str:
    candidates = [
        Path(sys.executable).with_name(f"pyside6-{name}"),
        Path(sys.executable).resolve().with_name(f"pyside6-{name}"),
        ROOT / ".venv" / "bin" / f"pyside6-{name}",
        ROOT / ".venv" / "Scripts" / f"pyside6-{name}.exe",
        ROOT / ".venv-win" / "Scripts" / f"pyside6-{name}.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(f"pyside6-{name}")
    if found:
        return found
    raise FileNotFoundError(f"Could not find pyside6-{name}. Install PySide6 first.")


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Update and compile Qt translation files.")
    parser.add_argument("--check", action="store_true", help="Fail if translation files would change.")
    args = parser.parse_args()

    ts_files = sorted(str(path) for path in TS_DIR.glob("ct_*.ts"))
    if not ts_files:
        print("No TS files found.", file=sys.stderr)
        return 1

    lupdate = resolve_tool("lupdate")
    lrelease = resolve_tool("lrelease")

    run([lupdate, "-extensions", "py", *SOURCE_TARGETS, "-ts", *ts_files])

    QM_DIR.mkdir(parents=True, exist_ok=True)
    for ts_path in ts_files:
        qm_path = QM_DIR / Path(ts_path).with_suffix(".qm").name
        run([lrelease, ts_path, "-qm", str(qm_path)])

    if args.check:
        diff = subprocess.run(
            ["git", "diff", "--exit-code", "--", "resources/translations"],
            cwd=ROOT,
        )
        if diff.returncode != 0:
            print("Translation assets are out of date. Run python scripts/compile_translations.py", file=sys.stderr)
            return diff.returncode

    print("Translation files are up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
