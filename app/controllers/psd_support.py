from __future__ import annotations

from typing import Any

from PySide6.QtCore import QCoreApplication

from app.ui.messages import Messages

_PSAPI_MODULE: Any | None = None
_PSAPI_ERROR: Exception | None = None


def get_missing_dependency_text() -> str:
    return QCoreApplication.translate(
        "Messages",
        "PSD import/export requires the optional Python package 'PhotoshopAPI'.\n"
        "Install it to use PSD features.",
    )


def import_photoshopapi() -> Any:
    global _PSAPI_MODULE, _PSAPI_ERROR
    if _PSAPI_MODULE is not None:
        return _PSAPI_MODULE
    if _PSAPI_ERROR is not None:
        raise RuntimeError(get_missing_dependency_text()) from _PSAPI_ERROR

    try:
        import photoshopapi as psapi
    except Exception as exc:  # pragma: no cover - environment dependent
        _PSAPI_ERROR = exc
        raise RuntimeError(get_missing_dependency_text()) from exc

    _PSAPI_MODULE = psapi
    return psapi


def is_photoshopapi_available() -> bool:
    try:
        import_photoshopapi()
    except RuntimeError:
        return False
    return True


def ensure_photoshopapi_available(parent=None) -> bool:
    try:
        import_photoshopapi()
    except RuntimeError as exc:
        Messages.show_error_with_copy(
            parent,
            QCoreApplication.translate("Messages", "PSD Feature Unavailable"),
            str(exc),
            detailed_text=str(exc),
        )
        return False
    return True
