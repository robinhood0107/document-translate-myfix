from __future__ import annotations

import json
import os
from pathlib import Path

from .paths import get_repo_root


def active_windows_runtime() -> str:
    value = str(os.environ.get("COMIC_WINDOWS_RUNTIME", "") or "").strip().lower()
    return value if value in {"cuda12", "cuda13"} else ""


def active_windows_install_state() -> dict:
    runtime = active_windows_runtime()
    if not runtime:
        return {}
    path = Path(get_repo_root()) / ".comic-bootstrap" / f"install-state-{runtime}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def active_windows_install_tier() -> str:
    value = str(active_windows_install_state().get("provisioned_tier") or "").lower()
    return value if value in {"core", "full"} else ""
