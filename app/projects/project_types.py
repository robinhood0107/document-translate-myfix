from __future__ import annotations

import os

PROJECT_FILE_EXT = ".ctpr"
SERIES_PROJECT_FILE_EXT = ".seriesctpr"
PROJECT_FILE_EXTENSIONS = (PROJECT_FILE_EXT, SERIES_PROJECT_FILE_EXT)

PROJECT_KIND_SINGLE = "project"
PROJECT_KIND_SERIES = "series_project"

PROJECT_FILE_FILTER = "Project Files (*.ctpr);;All Files (*)"
SERIES_PROJECT_FILE_FILTER = "Series Project Files (*.seriesctpr);;All Files (*)"


def normalize_project_path(path: str) -> str:
    return os.path.normpath(os.path.abspath(os.path.expanduser(path or "")))


def has_project_file_extension(path: str) -> bool:
    lower = str(path or "").lower()
    return lower.endswith(PROJECT_FILE_EXTENSIONS)


def is_single_project_file(path: str) -> bool:
    return str(path or "").lower().endswith(PROJECT_FILE_EXT)


def is_series_project_file(path: str) -> bool:
    return str(path or "").lower().endswith(SERIES_PROJECT_FILE_EXT)


def project_extension_for_path(path: str, default: str = PROJECT_FILE_EXT) -> str:
    lower = str(path or "").lower()
    if lower.endswith(SERIES_PROJECT_FILE_EXT):
        return SERIES_PROJECT_FILE_EXT
    if lower.endswith(PROJECT_FILE_EXT):
        return PROJECT_FILE_EXT
    return default


def ensure_project_extension(path: str, extension: str = PROJECT_FILE_EXT) -> str:
    normalized = normalize_project_path(path)
    if normalized.lower().endswith(PROJECT_FILE_EXTENSIONS):
        return normalized
    return f"{normalized}{extension}"


def project_kind_for_path(path: str) -> str:
    return PROJECT_KIND_SERIES if is_series_project_file(path) else PROJECT_KIND_SINGLE


def strip_project_extension(value: str) -> str:
    cleaned = str(value or "")
    lower = cleaned.lower()
    if lower.endswith(SERIES_PROJECT_FILE_EXT):
        return cleaned[: -len(SERIES_PROJECT_FILE_EXT)]
    if lower.endswith(PROJECT_FILE_EXT):
        return cleaned[: -len(PROJECT_FILE_EXT)]
    return cleaned


def project_extension_for_kind(kind: str) -> str:
    return SERIES_PROJECT_FILE_EXT if str(kind) == PROJECT_KIND_SERIES else PROJECT_FILE_EXT


def project_file_filter_for_kind(kind: str) -> str:
    return SERIES_PROJECT_FILE_FILTER if str(kind) == PROJECT_KIND_SERIES else PROJECT_FILE_FILTER


def recovered_project_name_for_kind(kind: str) -> str:
    return (
        f"RecoveredProject{SERIES_PROJECT_FILE_EXT}"
        if str(kind) == PROJECT_KIND_SERIES
        else f"RecoveredProject{PROJECT_FILE_EXT}"
    )
