from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6 import QtWidgets

SYSTEM_SOUND_MODE = "system"
FILE_SOUND_MODE = "file"

_REPO_ROOT = Path(__file__).resolve().parents[2]


def get_music_dir() -> Path:
    music_dir = _REPO_ROOT / "music"
    music_dir.mkdir(parents=True, exist_ok=True)
    return music_dir


def list_music_wav_files() -> list[str]:
    music_dir = get_music_dir()
    return sorted(
        file.name
        for file in music_dir.iterdir()
        if file.is_file() and file.suffix.lower() == ".wav"
    )


def resolve_music_wav_path(file_name: str | None) -> Path | None:
    safe_name = os.path.basename(str(file_name or "").strip())
    if not safe_name:
        return None
    path = get_music_dir() / safe_name
    if path.is_file() and path.suffix.lower() == ".wav":
        return path
    return None


def _play_system_sound() -> bool:
    if sys.platform == "win32":
        try:
            import winsound

            winsound.MessageBeep(winsound.MB_ICONASTERISK)
            return True
        except Exception:
            pass
    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.beep()
        return True
    return False


def _play_wav_file(path: Path) -> bool:
    if sys.platform == "win32":
        try:
            import winsound

            winsound.PlaySound(
                str(path),
                winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
            )
            return True
        except Exception:
            pass
    return _play_system_sound()


def play_completion_sound(mode: str | None, file_name: str | None = None) -> bool:
    normalized_mode = str(mode or SYSTEM_SOUND_MODE).strip().lower() or SYSTEM_SOUND_MODE
    if normalized_mode == FILE_SOUND_MODE:
        path = resolve_music_wav_path(file_name)
        if path is not None:
            return _play_wav_file(path)
    return _play_system_sound()


def notify_pipeline_event(event: dict) -> None:
    """Reserved for future ntfy integration.

    Keep this a no-op in the current iteration so notification delivery never
    blocks or destabilizes the main pipeline.
    """
    _ = event

