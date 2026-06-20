from __future__ import annotations

import logging
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Mapping

import requests
from PySide6 import QtCore, QtWidgets

SYSTEM_SOUND_MODE = "system"
FILE_SOUND_MODE = "file"

DEFAULT_NTFY_SERVER_URL = "https://ntfy.sh"
DEFAULT_NTFY_TIMEOUT_SEC = 10
NTFY_DEFAULT_MESSAGE_LIMIT_BYTES = 4096
# Keep well below the default message limit so the app never falls back into
# attachment-oriented behavior and always stays in plain-text mode.
NTFY_SAFE_MESSAGE_LIMIT_BYTES = 3500

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOGGER = logging.getLogger(__name__)

TR_AUTOMATIC_TRANSLATION = QtCore.QT_TRANSLATE_NOOP("NotificationSound", "Automatic translation")
TR_CURRENT_PAGE_AUTOMATIC_TRANSLATION = QtCore.QT_TRANSLATE_NOOP(
    "NotificationSound", "Current page automatic translation"
)
TR_RETRY_FAILED_PAGES = QtCore.QT_TRANSLATE_NOOP("NotificationSound", "Retry failed pages")
TR_SERIES_QUEUE_AUTOMATIC_TRANSLATION = QtCore.QT_TRANSLATE_NOOP(
    "NotificationSound", "Series queue automatic translation"
)
TR_MANUAL_TASK = QtCore.QT_TRANSLATE_NOOP("NotificationSound", "Manual task")
TR_NOTIFICATION_TEST = QtCore.QT_TRANSLATE_NOOP("NotificationSound", "Notification test")
TR_AUTOMATIC_TRANSLATION_COMPLETED = QtCore.QT_TRANSLATE_NOOP(
    "NotificationSound", "Automatic translation completed"
)
TR_AUTOMATIC_TRANSLATION_FAILED = QtCore.QT_TRANSLATE_NOOP(
    "NotificationSound", "Automatic translation failed"
)
TR_AUTOMATIC_TRANSLATION_CANCELLED = QtCore.QT_TRANSLATE_NOOP(
    "NotificationSound", "Automatic translation cancelled"
)
TR_AUTOMATIC_TRANSLATION_UPDATE = QtCore.QT_TRANSLATE_NOOP(
    "NotificationSound", "Automatic translation update"
)
TR_COMPLETED = QtCore.QT_TRANSLATE_NOOP("NotificationSound", "Completed")
TR_FAILED = QtCore.QT_TRANSLATE_NOOP("NotificationSound", "Failed")
TR_CANCELLED = QtCore.QT_TRANSLATE_NOOP("NotificationSound", "Cancelled")
TR_TEST = QtCore.QT_TRANSLATE_NOOP("NotificationSound", "Test")
TR_UPDATED = QtCore.QT_TRANSLATE_NOOP("NotificationSound", "Updated")
TR_STAGE_BATCHED_WORKFLOW = QtCore.QT_TRANSLATE_NOOP(
    "NotificationSound", "Stage-Batched Pipeline (Recommended)"
)
TR_LEGACY_WORKFLOW = QtCore.QT_TRANSLATE_NOOP(
    "NotificationSound", "Legacy Page Pipeline (Legacy)"
)
TR_OPTIMAL_OCR = QtCore.QT_TRANSLATE_NOOP(
    "NotificationSound", "Optimal (HunyuanOCR / PaddleOCR VL)"
)
TR_COMIC_TRANSLATE = QtCore.QT_TRANSLATE_NOOP("NotificationSound", "Comic Translate")
TR_STATUS = QtCore.QT_TRANSLATE_NOOP("NotificationSound", "Status")
TR_RUN = QtCore.QT_TRANSLATE_NOOP("NotificationSound", "Run")
TR_IMAGES = QtCore.QT_TRANSLATE_NOOP("NotificationSound", "Images")
TR_WORKFLOW = QtCore.QT_TRANSLATE_NOOP("NotificationSound", "Workflow")
TR_OCR = QtCore.QT_TRANSLATE_NOOP("NotificationSound", "OCR")
TR_TRANSLATOR = QtCore.QT_TRANSLATE_NOOP("NotificationSound", "Translator")
TR_SOURCE_LANGUAGE = QtCore.QT_TRANSLATE_NOOP("NotificationSound", "Source language")
TR_TARGET_LANGUAGE = QtCore.QT_TRANSLATE_NOOP("NotificationSound", "Target language")
TR_OUTPUT = QtCore.QT_TRANSLATE_NOOP("NotificationSound", "Output")
TR_SUMMARY = QtCore.QT_TRANSLATE_NOOP("NotificationSound", "Summary")
TR_DETAIL = QtCore.QT_TRANSLATE_NOOP("NotificationSound", "Detail")
TR_TIME = QtCore.QT_TRANSLATE_NOOP("NotificationSound", "Time")
TR_TEST_MESSAGE = QtCore.QT_TRANSLATE_NOOP(
    "NotificationSound", "This is a test notification from Comic Translate."
)


def _tr(text: str) -> str:
    return QtCore.QCoreApplication.translate("NotificationSound", text)


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


def normalize_ntfy_settings(data: Mapping[str, object] | None) -> dict[str, object]:
    payload = dict(data or {})
    server_url = str(payload.get("ntfy_server_url") or DEFAULT_NTFY_SERVER_URL).strip()
    if not server_url:
        server_url = DEFAULT_NTFY_SERVER_URL
    server_url = server_url.rstrip("/")
    timeout = int(payload.get("ntfy_timeout_sec") or DEFAULT_NTFY_TIMEOUT_SEC)
    timeout = max(3, min(timeout, 60))
    return {
        "enable_ntfy_notifications": bool(payload.get("enable_ntfy_notifications", False)),
        "ntfy_server_url": server_url,
        "ntfy_topic": str(payload.get("ntfy_topic") or "").strip(),
        "ntfy_access_token": str(payload.get("ntfy_access_token") or "").strip(),
        "ntfy_send_success": bool(payload.get("ntfy_send_success", True)),
        "ntfy_send_failure": bool(payload.get("ntfy_send_failure", True)),
        "ntfy_send_cancelled": bool(payload.get("ntfy_send_cancelled", True)),
        "ntfy_timeout_sec": timeout,
    }


def _load_ntfy_settings() -> dict[str, object]:
    settings = QtCore.QSettings("ComicLabs", "ComicTranslate")
    settings.beginGroup("notifications")
    try:
        return normalize_ntfy_settings(
            {
                "enable_ntfy_notifications": settings.value(
                    "enable_ntfy_notifications", False, type=bool
                ),
                "ntfy_server_url": settings.value(
                    "ntfy_server_url", DEFAULT_NTFY_SERVER_URL, type=str
                ),
                "ntfy_topic": settings.value("ntfy_topic", "", type=str),
                "ntfy_access_token": settings.value("ntfy_access_token", "", type=str),
                "ntfy_send_success": settings.value("ntfy_send_success", True, type=bool),
                "ntfy_send_failure": settings.value("ntfy_send_failure", True, type=bool),
                "ntfy_send_cancelled": settings.value("ntfy_send_cancelled", True, type=bool),
                "ntfy_timeout_sec": settings.value(
                    "ntfy_timeout_sec", DEFAULT_NTFY_TIMEOUT_SEC, type=int
                ),
            }
        )
    finally:
        settings.endGroup()


def _utf8_truncate(text: str, max_bytes: int) -> str:
    encoded = (text or "").encode("utf-8")
    if len(encoded) <= max_bytes:
        return text or ""

    suffix = "\n…"
    suffix_bytes = suffix.encode("utf-8")
    budget = max(0, max_bytes - len(suffix_bytes))
    trimmed = text or ""
    while trimmed and len(trimmed.encode("utf-8")) > budget:
        trimmed = trimmed[:-1]
    return f"{trimmed.rstrip()}{suffix}"


def _format_run_type(run_type: str) -> str:
    mapping = {
        "batch": _tr(TR_AUTOMATIC_TRANSLATION),
        "one_page_auto": _tr(TR_CURRENT_PAGE_AUTOMATIC_TRANSLATION),
        "retry_failed": _tr(TR_RETRY_FAILED_PAGES),
        "series_queue": _tr(TR_SERIES_QUEUE_AUTOMATIC_TRANSLATION),
        "manual": _tr(TR_MANUAL_TASK),
        "test": _tr(TR_NOTIFICATION_TEST),
    }
    normalized = str(run_type or "").strip()
    return mapping.get(normalized, normalized or _tr(TR_AUTOMATIC_TRANSLATION))


def _format_event_title(event_type: str) -> str:
    mapping = {
        "pipeline_completed": _tr(TR_AUTOMATIC_TRANSLATION_COMPLETED),
        "pipeline_failed": _tr(TR_AUTOMATIC_TRANSLATION_FAILED),
        "pipeline_cancelled": _tr(TR_AUTOMATIC_TRANSLATION_CANCELLED),
        "test": _tr(TR_NOTIFICATION_TEST),
    }
    return mapping.get(str(event_type or ""), _tr(TR_AUTOMATIC_TRANSLATION_UPDATE))


def _format_event_state(event_type: str) -> str:
    mapping = {
        "pipeline_completed": _tr(TR_COMPLETED),
        "pipeline_failed": _tr(TR_FAILED),
        "pipeline_cancelled": _tr(TR_CANCELLED),
        "test": _tr(TR_TEST),
    }
    return mapping.get(str(event_type or ""), _tr(TR_UPDATED))


def _format_tool_name(raw_value: str, fallback: str) -> str:
    return str(raw_value or fallback or "").strip() or fallback


def _current_tool_summary() -> dict[str, str]:
    settings = QtCore.QSettings("ComicLabs", "ComicTranslate")
    workflow = str(settings.value("tools/workflow_mode", "stage_batched_pipeline", type=str) or "")
    ocr = str(settings.value("tools/ocr", "best_local", type=str) or "")
    translator = str(settings.value("tools/translator", "Custom Local Server(Gemma)", type=str) or "")

    workflow_label = {
        "stage_batched_pipeline": _tr(TR_STAGE_BATCHED_WORKFLOW),
        "legacy_page_pipeline": _tr(TR_LEGACY_WORKFLOW),
    }.get(workflow, workflow or _tr(TR_STAGE_BATCHED_WORKFLOW))
    ocr_label = {
        "best_local": _tr(TR_OPTIMAL_OCR),
        "default": _tr("Default (existing auto: MangaOCR / PPOCR / Pororo...)"),
        "paddleocr_vl": "PaddleOCR VL",
        "hunyuanocr": "HunyuanOCR",
        "mangalmm": "MangaLMM",
    }.get(ocr, ocr or _tr(TR_OPTIMAL_OCR))
    return {
        "workflow": _format_tool_name(workflow_label, _tr(TR_STAGE_BATCHED_WORKFLOW)),
        "ocr": _format_tool_name(ocr_label, _tr(TR_OPTIMAL_OCR)),
        "translator": _format_tool_name(translator, "Custom Local Server(Gemma)"),
    }


def build_ntfy_message(event: Mapping[str, object] | None) -> dict[str, str]:
    payload = dict(event or {})
    event_type = str(payload.get("event_type") or "pipeline_completed")
    run_type = _format_run_type(str(payload.get("run_type") or "batch"))
    state_text = _format_event_state(event_type)
    title = _format_event_title(event_type)
    tool_summary = _current_tool_summary()

    lines = [
        _tr(TR_COMIC_TRANSLATE),
        f"{_tr(TR_STATUS)}: {state_text}",
        f"{_tr(TR_RUN)}: {run_type}",
    ]

    image_count = int(payload.get("image_count") or 0)
    if image_count > 0:
        lines.append(f"{_tr(TR_IMAGES)}: {image_count}")

    lines.append(f"{_tr(TR_WORKFLOW)}: {tool_summary['workflow']}")
    lines.append(f"{_tr(TR_OCR)}: {tool_summary['ocr']}")
    lines.append(f"{_tr(TR_TRANSLATOR)}: {tool_summary['translator']}")

    source_language = str(payload.get("source_language") or "").strip()
    target_language = str(payload.get("target_language") or "").strip()
    if source_language:
        lines.append(f"{_tr(TR_SOURCE_LANGUAGE)}: {source_language}")
    if target_language:
        lines.append(f"{_tr(TR_TARGET_LANGUAGE)}: {target_language}")

    output_root = str(payload.get("output_root") or "").strip()
    if output_root:
        lines.append(f"{_tr(TR_OUTPUT)}: {output_root}")

    message = str(payload.get("message") or "").strip()
    if message:
        lines.append(f"{_tr(TR_SUMMARY)}: {message}")

    detail = str(payload.get("detail") or "").strip()
    if detail:
        lines.append(f"{_tr(TR_DETAIL)}: {detail}")

    lines.append(f"{_tr(TR_TIME)}: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    body = _utf8_truncate("\n".join(lines), NTFY_SAFE_MESSAGE_LIMIT_BYTES)

    tags = {
        "pipeline_completed": "white_check_mark",
        "pipeline_failed": "warning",
        "pipeline_cancelled": "pause_button",
        "test": "test_tube",
    }.get(event_type, "speech_balloon")
    priority = {
        "pipeline_completed": "default",
        "pipeline_failed": "high",
        "pipeline_cancelled": "default",
        "test": "default",
    }.get(event_type, "default")
    return {"title": title, "body": body, "priority": priority, "tags": tags}


def _should_send_ntfy_event(settings: Mapping[str, object], event_type: str, *, force: bool) -> bool:
    if force:
        return True
    if not bool(settings.get("enable_ntfy_notifications")):
        return False
    if event_type == "pipeline_completed":
        return bool(settings.get("ntfy_send_success", True))
    if event_type == "pipeline_failed":
        return bool(settings.get("ntfy_send_failure", True))
    if event_type == "pipeline_cancelled":
        return bool(settings.get("ntfy_send_cancelled", True))
    return False


def _deliver_ntfy_message(settings: Mapping[str, object], event: Mapping[str, object]) -> None:
    normalized = normalize_ntfy_settings(settings)
    topic = str(normalized.get("ntfy_topic") or "").strip()
    if not topic:
        return

    message = build_ntfy_message(event)
    url = f"{str(normalized.get('ntfy_server_url') or DEFAULT_NTFY_SERVER_URL).rstrip('/')}/{topic}"
    headers = {
        "Title": message["title"],
        "Tags": message["tags"],
        "Priority": message["priority"],
        "Content-Type": "text/plain; charset=utf-8",
    }
    access_token = str(normalized.get("ntfy_access_token") or "").strip()
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    try:
        response = requests.post(
            url,
            data=message["body"].encode("utf-8"),
            headers=headers,
            timeout=int(normalized.get("ntfy_timeout_sec") or DEFAULT_NTFY_TIMEOUT_SEC),
        )
        response.raise_for_status()
    except Exception:
        _LOGGER.debug("Failed to deliver ntfy notification.", exc_info=True)


def _queue_ntfy_notification(event: Mapping[str, object], settings: Mapping[str, object], *, force: bool) -> bool:
    normalized = normalize_ntfy_settings(settings)
    topic = str(normalized.get("ntfy_topic") or "").strip()
    if not topic:
        return False
    event_type = str(event.get("event_type") or "")
    if not _should_send_ntfy_event(normalized, event_type, force=force):
        return False
    thread = threading.Thread(
        target=_deliver_ntfy_message,
        args=(normalized, dict(event)),
        daemon=True,
        name="ntfy-notification",
    )
    thread.start()
    return True


def notify_pipeline_event(event: dict) -> None:
    settings = _load_ntfy_settings()
    _queue_ntfy_notification(dict(event or {}), settings, force=False)


def send_test_ntfy_notification(settings: Mapping[str, object] | None = None) -> bool:
    merged_settings = normalize_ntfy_settings(settings if settings is not None else _load_ntfy_settings())
    return _queue_ntfy_notification(
        {
            "event_type": "test",
            "run_type": "test",
            "message": _tr(TR_TEST_MESSAGE),
        },
        merged_settings,
        force=True,
    )
