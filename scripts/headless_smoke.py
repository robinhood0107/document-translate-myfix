#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTranslator
from PySide6.QtWidgets import QApplication

from app.controllers.batch_report import BatchReportController  # noqa: F401
from app.ui.main_window.builders.workspace import WorkspaceMixin  # noqa: F401

QM_DIR = ROOT / "resources" / "translations" / "compiled"
LANG_CODES = ["de", "es", "fr", "it", "ja", "ko", "ru", "tr", "zh-CN"]


def main() -> int:
    app = QApplication.instance() or QApplication([])
    missing: list[str] = []
    for lang_code in LANG_CODES:
        translator = QTranslator(app)
        if not translator.load(f"ct_{lang_code}", str(QM_DIR)):
            missing.append(lang_code)

    if missing:
        print("Missing or unloadable translation assets: " + ", ".join(missing), file=sys.stderr)
        return 1

    print("Headless smoke check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
