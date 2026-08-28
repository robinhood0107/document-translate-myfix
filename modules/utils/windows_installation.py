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


def assert_selected_windows_models_installed(settings_page, source_lang_english: str) -> None:
    """Fail before page decoding when a selected optional model needs full setup."""

    if not active_windows_runtime() or active_windows_install_tier() == "full":
        return
    from modules.ocr.selection import resolve_ocr_engine
    from modules.utils.inpainting_runtime import normalize_inpainter_key

    ocr_engine = resolve_ocr_engine(
        settings_page.get_tool_selection("ocr"),
        source_lang_english,
    )
    if ocr_engine in {"Default", "PaddleOCR VL Spotting", "MangaLMM"}:
        raise RuntimeError(
            "The selected OCR runtime is not installed. "
            "Run the matching setup_full BAT before starting this job."
        )
    inpainter = normalize_inpainter_key(settings_page.get_tool_selection("inpainter"))
    if inpainter == "AOT":
        raise RuntimeError(
            "The selected optional inpainter is not installed. "
            "Run the matching setup_full BAT before starting this job."
        )
