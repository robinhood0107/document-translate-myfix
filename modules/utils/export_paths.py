from __future__ import annotations

import os
from datetime import datetime
from typing import Mapping


def build_export_timestamp(now: datetime | None = None) -> str:
    current = now or datetime.now()
    return current.strftime("%b-%d-%Y_%I-%M-%S%p")


def reserve_export_run_token(
    base_dir: str,
    base_timestamp: str,
    cache: dict[str, str] | None = None,
) -> str:
    abs_dir = os.path.abspath(base_dir)
    if cache is not None:
        cached = cache.get(abs_dir)
        if cached:
            return cached

    suffix = 0
    while True:
        token = base_timestamp if suffix == 0 else f"{base_timestamp}_{suffix:03d}"
        run_root = os.path.join(base_dir, f"comic_translate_{token}")
        try:
            os.makedirs(run_root, exist_ok=False)
            if cache is not None:
                cache[abs_dir] = token
            return token
        except FileExistsError:
            suffix += 1


def export_run_root(base_dir: str, token: str) -> str:
    return os.path.join(base_dir, f"comic_translate_{token}")


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
        "source_path": os.path.abspath(source_path),
    }


def _is_path_within(path: str | None, base_dir: str | None) -> bool:
    if not path or not base_dir:
        return False
    abs_path = os.path.abspath(path)
    abs_base = os.path.abspath(base_dir)
    prefix = abs_base if abs_base.endswith(os.sep) else f"{abs_base}{os.sep}"
    return abs_path == abs_base or abs_path.startswith(prefix)


def resolve_export_directory(
    image_path: str,
    *,
    archive_info: list[dict] | None = None,
    source_records: Mapping[str, Mapping[str, object]] | None = None,
    project_file: str | None = None,
    temp_dir: str | None = None,
) -> tuple[str, str]:
    abs_image_path = os.path.abspath(image_path)
    source_record = None
    if source_records:
        source_record = normalize_export_source_record(source_records.get(image_path))
        if source_record is None:
            source_record = normalize_export_source_record(source_records.get(abs_image_path))
    if source_record is not None:
        source_path = source_record["source_path"]
        if source_record["kind"] == "archive":
            return (
                os.path.dirname(source_path),
                os.path.splitext(os.path.basename(source_path))[0].strip(),
            )
        return os.path.dirname(source_path), ""

    for archive in archive_info or []:
        archive_path = str(archive.get("archive_path", "")).strip()
        if not archive_path:
            continue
        for extracted_image in archive.get("extracted_images", []) or []:
            if os.path.abspath(str(extracted_image)) == abs_image_path:
                return (
                    os.path.dirname(archive_path),
                    os.path.splitext(os.path.basename(archive_path))[0].strip(),
                )

    if project_file and _is_path_within(abs_image_path, temp_dir):
        return os.path.dirname(os.path.abspath(project_file)), ""

    return os.path.dirname(abs_image_path), ""
