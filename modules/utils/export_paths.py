from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import PureWindowsPath
from typing import Mapping

_ILLEGAL_FILE_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE_RE = re.compile(r"\s+")
_SEPARATOR_RE = re.compile(r"[_-]{2,}")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def build_export_timestamp(now: datetime | None = None) -> str:
    current = now or datetime.now()
    return current.strftime("%b-%d-%Y_%I-%M-%S%p")


def sanitize_export_path_component(stem: str, fallback: str = "comic_translate_output") -> str:
    candidate = str(stem or "").strip()
    candidate = _ILLEGAL_FILE_CHARS_RE.sub("", candidate)
    candidate = candidate.replace("\n", " ").replace("\r", " ").strip(" .")
    candidate = _WHITESPACE_RE.sub(" ", candidate)
    candidate = _SEPARATOR_RE.sub("_", candidate)
    return candidate or fallback


def _run_folder_name(token: str, source_name: str | None = None) -> str:
    source_component = sanitize_export_path_component(source_name or "", fallback="")
    if source_component:
        return f"log_{source_component}_{token}"
    return f"comic_translate_{token}"


def reserve_export_run_token(
    base_dir: str,
    base_timestamp: str,
    cache: dict[str, str] | None = None,
    source_name: str | None = None,
) -> str:
    abs_dir = os.path.abspath(base_dir)
    cache_key = f"{abs_dir}\0{sanitize_export_path_component(source_name or '', fallback='')}"
    if cache is not None:
        cached = cache.get(cache_key)
        if cached:
            return cached

    suffix = 0
    while True:
        token = base_timestamp if suffix == 0 else f"{base_timestamp}_{suffix:03d}"
        run_root = os.path.join(base_dir, _run_folder_name(token, source_name))
        try:
            os.makedirs(run_root, exist_ok=False)
            if cache is not None:
                cache[cache_key] = token
            return token
        except FileExistsError:
            suffix += 1


def export_run_root(base_dir: str, token: str, source_name: str | None = None) -> str:
    return os.path.join(base_dir, _run_folder_name(token, source_name))


def _looks_like_windows_path(path: str) -> bool:
    return bool(_WINDOWS_DRIVE_RE.match(path)) or "\\" in path


def _preserve_abspath(path: str) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    if _looks_like_windows_path(text):
        return str(PureWindowsPath(text))
    return os.path.abspath(text)


def _path_dirname(path: str) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    if _looks_like_windows_path(text):
        parent = PureWindowsPath(text).parent
        return "" if str(parent) == "." else str(parent)
    return os.path.dirname(os.path.abspath(text))


def _path_stem(path: str) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    if _looks_like_windows_path(text):
        return PureWindowsPath(text).stem
    return os.path.splitext(os.path.basename(text))[0].strip()


def _canonical_path_key(path: str | None) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    text = text.replace("\\", "/")
    if text.startswith("/mnt/") and len(text) > 7 and text[6] == "/":
        drive = text[5].lower()
        text = f"{drive}:{text[6:]}"
    elif _WINDOWS_DRIVE_RE.match(text):
        text = f"{text[0].lower()}:{text[2:]}"
    else:
        text = os.path.abspath(text).replace("\\", "/")
    text = re.sub(r"/+", "/", text).rstrip("/")
    return text.lower()


def _lookup_source_record(
    source_records: Mapping[str, Mapping[str, object]] | None,
    image_path: str,
) -> dict[str, str] | None:
    if not source_records:
        return None
    direct = normalize_export_source_record(source_records.get(image_path))
    if direct is not None:
        return direct
    abs_image_path = _preserve_abspath(image_path)
    direct = normalize_export_source_record(source_records.get(abs_image_path))
    if direct is not None:
        return direct

    wanted_keys = {
        _canonical_path_key(image_path),
        _canonical_path_key(abs_image_path),
    }
    for record_path, record in source_records.items():
        if _canonical_path_key(str(record_path)) in wanted_keys:
            normalized = normalize_export_source_record(record)
            if normalized is not None:
                return normalized
    return None


def normalize_export_source_record(record: Mapping[str, object] | None) -> dict[str, str] | None:
    if not isinstance(record, Mapping):
        return None
    source_path = str(record.get("source_path", "")).strip()
    if not source_path:
        return None
    kind = str(record.get("kind", "file")).strip().lower() or "file"
    if kind != "archive":
        kind = "file"
    return {
        "kind": kind,
        "source_path": _preserve_abspath(source_path),
    }


def _is_path_within(path: str | None, base_dir: str | None) -> bool:
    if not path or not base_dir:
        return False
    abs_path = _canonical_path_key(path)
    abs_base = _canonical_path_key(base_dir)
    prefix = abs_base if abs_base.endswith("/") else f"{abs_base}/"
    return abs_path == abs_base or abs_path.startswith(prefix)


def resolve_export_source_identity(
    image_path: str,
    *,
    archive_info: list[dict] | None = None,
    source_records: Mapping[str, Mapping[str, object]] | None = None,
    project_file: str | None = None,
    temp_dir: str | None = None,
) -> dict[str, str]:
    abs_image_path = _preserve_abspath(image_path)
    source_record = _lookup_source_record(source_records, image_path)
    if source_record is not None:
        source_path = source_record["source_path"]
        return {
            "kind": source_record["kind"],
            "directory": _path_dirname(source_path),
            "source_name": _path_stem(source_path),
        }

    image_key = _canonical_path_key(abs_image_path)
    for archive in archive_info or []:
        archive_path = str(archive.get("archive_path", "")).strip()
        if not archive_path:
            continue
        for extracted_image in archive.get("extracted_images", []) or []:
            if _canonical_path_key(str(extracted_image)) == image_key:
                return {
                    "kind": "archive",
                    "directory": _path_dirname(archive_path),
                    "source_name": _path_stem(archive_path),
                }

    if project_file and _is_path_within(abs_image_path, temp_dir):
        return {
            "kind": "project",
            "directory": _path_dirname(project_file),
            "source_name": _path_stem(project_file),
        }

    return {
        "kind": "file",
        "directory": _path_dirname(abs_image_path),
        "source_name": _path_stem(abs_image_path),
    }


def resolve_export_directory(
    image_path: str,
    *,
    archive_info: list[dict] | None = None,
    source_records: Mapping[str, Mapping[str, object]] | None = None,
    project_file: str | None = None,
    temp_dir: str | None = None,
) -> tuple[str, str]:
    identity = resolve_export_source_identity(
        image_path,
        archive_info=archive_info,
        source_records=source_records,
        project_file=project_file,
        temp_dir=temp_dir,
    )
    return identity["directory"], sanitize_export_path_component(identity["source_name"])
