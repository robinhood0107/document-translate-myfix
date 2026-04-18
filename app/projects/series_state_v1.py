from __future__ import annotations

import hashlib
import msgpack
import os
import shutil
import sqlite3
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from modules.utils.automatic_output import default_project_output_preferences
from modules.utils.file_handler import FileHandler, get_prepared_path_source

from .project_state import close_state_store, save_state_to_proj_file
from .project_types import (
    PROJECT_FILE_EXT,
    PROJECT_KIND_SERIES,
    SERIES_PROJECT_FILE_EXT,
    is_single_project_file,
)


SERIES_PROJECT_FORMAT_VERSION = 1

SERIES_QUEUE_STATUS_PENDING = "pending"
SERIES_QUEUE_STATUS_RUNNING = "running"
SERIES_QUEUE_STATUS_DONE = "done"
SERIES_QUEUE_STATUS_FAILED = "failed"
SERIES_QUEUE_STATUS_SKIPPED = "skipped"

SERIES_SOURCE_KIND_FILE = "source_file"
SERIES_SOURCE_KIND_CTPR = "ctpr_import"

SUPPORTED_SERIES_SOURCE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".psd",
    ".pdf",
    ".epub",
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".cbz",
    ".cbr",
    ".cb7",
    ".cbt",
    PROJECT_FILE_EXT,
}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def _configure_connection(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA temp_store=MEMORY")


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS series_manifest (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            manifest_blob BLOB NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS series_items (
            series_item_id TEXT PRIMARY KEY,
            item_blob BLOB NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embedded_projects (
            project_hash TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            project_size INTEGER NOT NULL,
            project_blob BLOB NOT NULL
        )
        """
    )


def default_series_settings() -> dict[str, object]:
    return {
        "queue_failure_policy": "stop",
        "retry_count": 0,
        "retry_delay_sec": 0,
        "auto_open_failed_child": True,
        "resume_from_first_incomplete": True,
        "return_to_series_after_completion": True,
    }


def normalize_series_settings(data: dict[str, object] | None) -> dict[str, object]:
    merged = dict(default_series_settings())
    if isinstance(data, dict):
        merged.update(data)
    merged["queue_failure_policy"] = str(
        merged.get("queue_failure_policy") or "stop"
    ).strip().lower() or "stop"
    if merged["queue_failure_policy"] not in {"stop", "skip", "retry"}:
        merged["queue_failure_policy"] = "stop"
    merged["retry_count"] = max(0, int(merged.get("retry_count", 0) or 0))
    merged["retry_delay_sec"] = max(0, int(merged.get("retry_delay_sec", 0) or 0))
    merged["auto_open_failed_child"] = bool(merged.get("auto_open_failed_child", True))
    merged["resume_from_first_incomplete"] = bool(
        merged.get("resume_from_first_incomplete", True)
    )
    merged["return_to_series_after_completion"] = bool(
        merged.get("return_to_series_after_completion", True)
    )
    return merged


def default_series_global_settings() -> dict[str, object]:
    return {
        "source_language": "",
        "target_language": "",
        "ocr": "",
        "translator": "",
        "workflow_mode": "",
        "use_gpu": True,
    }


def normalize_series_global_settings(data: dict[str, object] | None) -> dict[str, object]:
    merged = dict(default_series_global_settings())
    if isinstance(data, dict):
        merged.update(data)
    merged["source_language"] = str(merged.get("source_language") or "").strip()
    merged["target_language"] = str(merged.get("target_language") or "").strip()
    merged["ocr"] = str(merged.get("ocr") or "").strip()
    merged["translator"] = str(merged.get("translator") or "").strip()
    merged["workflow_mode"] = str(merged.get("workflow_mode") or "").strip()
    merged["use_gpu"] = bool(merged.get("use_gpu", True))
    return merged


def scan_series_source_files(root_dir: str) -> list[str]:
    root_dir = os.path.normpath(os.path.abspath(root_dir or ""))
    if not root_dir or not os.path.isdir(root_dir):
        return []

    results: list[str] = []
    pending = [root_dir]
    while pending:
        current_dir = pending.pop()
        try:
            with os.scandir(current_dir) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(entry.path)
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                    except OSError:
                        continue

                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext == SERIES_PROJECT_FILE_EXT:
                        continue
                    if ext in SUPPORTED_SERIES_SOURCE_EXTS:
                        results.append(os.path.normpath(os.path.abspath(entry.path)))
        except OSError:
            continue

    results.sort(key=lambda path: path.lower())
    return results


def relative_series_source_path(root_dir: str, path: str) -> str:
    root = os.path.normpath(os.path.abspath(root_dir or ""))
    target = os.path.normpath(os.path.abspath(path or ""))
    if not root or not target:
        return os.path.basename(target or "")
    try:
        return os.path.relpath(target, root)
    except ValueError:
        return os.path.basename(target)


@dataclass
class _StubSettingsPage:
    source_lang: str
    target_lang: str
    extra_context: str = ""

    def get_llm_settings(self) -> dict[str, str]:
        return {"extra_context": self.extra_context}


@dataclass
class _StubImageViewer:
    webtoon_view_state: dict


@dataclass
class _StubBatchReportController:
    def export_latest_report_for_project(self) -> None:
        return None


class _StubSeriesChildContext:
    def __init__(
        self,
        image_files: list[str],
        source_lang: str,
        target_lang: str,
        temp_dir: str,
    ) -> None:
        self.image_files = list(image_files)
        self.curr_img_idx = 0 if self.image_files else -1
        self.image_states = {}
        self.image_data = {}
        self.image_history = {}
        self.in_memory_history = {}
        self.current_history_index = {}
        self.displayed_images = set()
        self.loaded_images = []
        self.image_patches = {}
        self.export_source_by_path = {}
        self.project_file = None
        self.temp_dir = temp_dir
        self.webtoon_mode = False
        self.project_output_preferences = default_project_output_preferences()
        self.settings_page = _StubSettingsPage(source_lang=source_lang, target_lang=target_lang)
        self.image_viewer = _StubImageViewer(webtoon_view_state={})
        self.batch_report_ctrl = _StubBatchReportController()

        for file_path in self.image_files:
            self.image_history[file_path] = [file_path]
            self.current_history_index[file_path] = 0
            self.image_states[file_path] = {
                "viewer_state": {},
                "source_lang": source_lang,
                "target_lang": target_lang,
                "brush_strokes": [],
                "blk_list": [],
                "skip": False,
                "export_group_name": os.path.splitext(os.path.basename(file_path))[0],
            }


def _load_embedded_projects(file_name: str) -> dict[str, dict[str, object]]:
    conn = sqlite3.connect(file_name, timeout=30.0)
    try:
        projects = {}
        for project_hash, display_name, project_size, project_blob in conn.execute(
            """
            SELECT project_hash, display_name, project_size, project_blob
            FROM embedded_projects
            """
        ):
            projects[str(project_hash)] = {
                "project_hash": str(project_hash),
                "display_name": str(display_name),
                "project_size": int(project_size),
                "project_blob": bytes(project_blob),
            }
        return projects
    finally:
        conn.close()


def _write_series_snapshot(
    file_name: str,
    *,
    manifest: dict[str, object],
    items: list[dict[str, object]],
    embedded_projects: dict[str, dict[str, object]],
) -> None:
    target_dir = os.path.dirname(os.path.abspath(file_name))
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)

    referenced_hashes = {
        str(item.get("embedded_project_blob_hash") or "").strip()
        for item in items
        if str(item.get("embedded_project_blob_hash") or "").strip()
    }
    filtered_projects = {
        project_hash: project
        for project_hash, project in embedded_projects.items()
        if project_hash in referenced_hashes
    }

    fd, temp_db_path = tempfile.mkstemp(
        prefix=".seriesctpr_tmp_",
        suffix=SERIES_PROJECT_FILE_EXT,
        dir=target_dir or None,
    )
    os.close(fd)
    conn = sqlite3.connect(temp_db_path, check_same_thread=False, timeout=30.0)
    _configure_connection(conn)
    _init_schema(conn)
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                ("series_project_format_version", str(SERIES_PROJECT_FORMAT_VERSION)),
            )
            conn.execute(
                "INSERT OR REPLACE INTO series_manifest(id, manifest_blob) VALUES(1, ?)",
                (sqlite3.Binary(msgpack.packb(manifest, use_bin_type=True)),),
            )
            for item in items:
                conn.execute(
                    "INSERT OR REPLACE INTO series_items(series_item_id, item_blob) VALUES(?, ?)",
                    (
                        str(item["series_item_id"]),
                        sqlite3.Binary(msgpack.packb(item, use_bin_type=True)),
                    ),
                )
            for project in filtered_projects.values():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO embedded_projects(project_hash, display_name, project_size, project_blob)
                    VALUES(?, ?, ?, ?)
                    """,
                    (
                        str(project["project_hash"]),
                        str(project["display_name"]),
                        int(project["project_size"]),
                        sqlite3.Binary(project["project_blob"]),
                    ),
                )
        conn.close()
        os.replace(temp_db_path, file_name)
    finally:
        try:
            conn.close()
        except Exception:
            pass
        if os.path.exists(temp_db_path):
            try:
                os.remove(temp_db_path)
            except OSError:
                pass


def _create_child_project_blob_from_source(
    source_path: str,
    *,
    source_lang: str,
    target_lang: str,
) -> tuple[bytes, dict[str, object]]:
    source_path = os.path.normpath(os.path.abspath(source_path))
    temp_root = tempfile.mkdtemp(prefix="series_child_")
    child_project_path = os.path.join(
        temp_root,
        f"{os.path.splitext(os.path.basename(source_path))[0]}{PROJECT_FILE_EXT}",
    )
    file_handler = FileHandler()
    prepared = file_handler.prepare_files([source_path], extend=False)
    try:
        stub = _StubSeriesChildContext(
            image_files=prepared,
            source_lang=source_lang,
            target_lang=target_lang,
            temp_dir=tempfile.mkdtemp(prefix="series_child_project_", dir=temp_root),
        )
        save_state_to_proj_file(stub, child_project_path)
        payload = _read_file_bytes(child_project_path)
        source_records = {}
        for prepared_path in prepared:
            lazy_source = get_prepared_path_source(prepared_path)
            archive_path = str((lazy_source or {}).get("archive_path", "")).strip()
            if archive_path:
                source_records[prepared_path] = {
                    "kind": "archive",
                    "source_path": os.path.abspath(archive_path),
                }
            else:
                source_records[prepared_path] = {
                    "kind": "file",
                    "source_path": os.path.abspath(prepared_path),
                }
        return payload, {
            "page_count": len(prepared),
            "source_records": source_records,
        }
    finally:
        close_state_store(child_project_path)
        for archive in list(file_handler.archive_info):
            archive_temp = archive.get("temp_dir")
            if archive_temp and os.path.isdir(archive_temp):
                shutil.rmtree(archive_temp, ignore_errors=True)
        shutil.rmtree(temp_root, ignore_errors=True)


def build_series_item_from_path(
    source_path: str,
    *,
    root_dir: str,
    queue_index: int,
    source_lang: str,
    target_lang: str,
) -> tuple[dict[str, object], dict[str, object]]:
    source_path = os.path.normpath(os.path.abspath(source_path))
    display_name = os.path.basename(source_path)
    source_kind = SERIES_SOURCE_KIND_CTPR if is_single_project_file(source_path) else SERIES_SOURCE_KIND_FILE

    if source_kind == SERIES_SOURCE_KIND_CTPR:
        payload = _read_file_bytes(source_path)
        child_meta = {"page_count": None}
    else:
        payload, child_meta = _create_child_project_blob_from_source(
            source_path,
            source_lang=source_lang,
            target_lang=target_lang,
        )

    blob_hash = _sha256_bytes(payload)
    item = {
        "series_item_id": str(uuid.uuid4()),
        "queue_index": int(queue_index),
        "display_name": display_name,
        "source_kind": source_kind,
        "source_origin_path": source_path,
        "source_origin_relpath": relative_series_source_path(root_dir, source_path),
        "imported_at": _now_iso(),
        "updated_at": _now_iso(),
        "status": SERIES_QUEUE_STATUS_PENDING,
        "embedded_project_blob_hash": blob_hash,
        "child_page_count": child_meta.get("page_count"),
    }
    project_entry = {
        "project_hash": blob_hash,
        "display_name": display_name,
        "project_size": len(payload),
        "project_blob": payload,
    }
    return item, project_entry


def create_series_project(
    file_name: str,
    *,
    root_dir: str,
    items: list[dict[str, object]],
    embedded_projects: list[dict[str, object]],
    series_settings: dict[str, object] | None = None,
    global_settings: dict[str, object] | None = None,
) -> None:
    root_dir = os.path.normpath(os.path.abspath(root_dir or ""))
    manifest = {
        "series_project_type": PROJECT_KIND_SERIES,
        "series_project_format_version": SERIES_PROJECT_FORMAT_VERSION,
        "root_dir": root_dir,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "series_settings": normalize_series_settings(series_settings),
        "global_settings": normalize_series_global_settings(global_settings),
        "series_navigation_history": {"back": [], "forward": []},
        "series_queue_runtime": {
            "active_item_id": None,
            "completed_item_ids": [],
            "failed_item_id": None,
            "last_run_at": None,
        },
    }
    project_map = {
        str(project["project_hash"]): dict(project)
        for project in embedded_projects
    }
    _write_series_snapshot(
        file_name,
        manifest=manifest,
        items=list(items),
        embedded_projects=project_map,
    )


def load_series_project(file_name: str) -> dict[str, object]:
    conn = sqlite3.connect(file_name, timeout=30.0)
    try:
        manifest_row = conn.execute("SELECT manifest_blob FROM series_manifest WHERE id = 1").fetchone()
        if manifest_row is None or manifest_row[0] is None:
            raise ValueError("Invalid .seriesctpr: missing manifest")
        manifest = msgpack.unpackb(manifest_row[0], raw=False)
        manifest["series_settings"] = normalize_series_settings(
            manifest.get("series_settings")
            if isinstance(manifest.get("series_settings"), dict)
            else None
        )
        manifest["global_settings"] = normalize_series_global_settings(
            manifest.get("global_settings")
            if isinstance(manifest.get("global_settings"), dict)
            else None
        )
        items = [
            msgpack.unpackb(row[0], raw=False)
            for row in conn.execute("SELECT item_blob FROM series_items")
        ]
        items.sort(key=lambda item: int(item.get("queue_index", 0)))
        return {
            "manifest": manifest,
            "items": items,
        }
    finally:
        conn.close()


def load_series_project_blob(file_name: str, project_hash: str) -> bytes:
    conn = sqlite3.connect(file_name, timeout=30.0)
    try:
        row = conn.execute(
            "SELECT project_blob FROM embedded_projects WHERE project_hash = ?",
            (project_hash,),
        ).fetchone()
        if row is None or row[0] is None:
            raise KeyError(project_hash)
        return bytes(row[0])
    finally:
        conn.close()


def materialize_series_child_project(
    file_name: str,
    item: dict[str, object],
    *,
    temp_dir: str | None = None,
) -> str:
    blob_hash = str(item.get("embedded_project_blob_hash") or "")
    if not blob_hash:
        raise ValueError("Series item is missing embedded project hash")
    payload = load_series_project_blob(file_name, blob_hash)
    work_dir = temp_dir or tempfile.mkdtemp(prefix="series_child_materialized_")
    os.makedirs(work_dir, exist_ok=True)
    target_path = os.path.join(
        work_dir,
        f"{os.path.splitext(str(item.get('display_name') or item.get('series_item_id') or 'child'))[0]}{PROJECT_FILE_EXT}",
    )
    with open(target_path, "wb") as fh:
        fh.write(payload)
    return target_path


def save_series_manifest(
    file_name: str,
    *,
    manifest: dict[str, object] | None = None,
    items: list[dict[str, object]] | None = None,
    embedded_projects: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    state = load_series_project(file_name)
    next_manifest = dict(state["manifest"])
    if manifest is not None:
        next_manifest.update(dict(manifest))
    next_manifest["updated_at"] = _now_iso()
    next_manifest["series_settings"] = normalize_series_settings(
        next_manifest.get("series_settings")
        if isinstance(next_manifest.get("series_settings"), dict)
        else None
    )
    next_manifest["global_settings"] = normalize_series_global_settings(
        next_manifest.get("global_settings")
        if isinstance(next_manifest.get("global_settings"), dict)
        else None
    )
    next_items = list(items) if items is not None else list(state["items"])
    next_projects = embedded_projects or _load_embedded_projects(file_name)
    _write_series_snapshot(
        file_name,
        manifest=next_manifest,
        items=next_items,
        embedded_projects=next_projects,
    )
    return next_manifest


def add_series_paths(
    file_name: str,
    *,
    root_dir: str,
    paths: list[str],
    source_lang: str,
    target_lang: str,
) -> list[dict[str, object]]:
    state = load_series_project(file_name)
    items = list(state["items"])
    project_map = _load_embedded_projects(file_name)
    start_index = len(items) + 1
    appended_items: list[dict[str, object]] = []
    for offset, path in enumerate(paths):
        item, project_entry = build_series_item_from_path(
            path,
            root_dir=root_dir,
            queue_index=start_index + offset,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        items.append(item)
        project_map[str(project_entry["project_hash"])] = project_entry
        appended_items.append(item)
    manifest = dict(state["manifest"])
    manifest["updated_at"] = _now_iso()
    _write_series_snapshot(
        file_name,
        manifest=manifest,
        items=items,
        embedded_projects=project_map,
    )
    return items


def update_series_child_from_file(
    file_name: str,
    *,
    series_item_id: str,
    child_project_path: str,
) -> dict[str, object]:
    payload = _read_file_bytes(child_project_path)
    new_hash = _sha256_bytes(payload)
    state = load_series_project(file_name)
    items = list(state["items"])
    target_item = None
    for item in items:
        if str(item.get("series_item_id")) == str(series_item_id):
            target_item = item
            break
    if target_item is None:
        raise KeyError(series_item_id)

    target_item["embedded_project_blob_hash"] = new_hash
    target_item["updated_at"] = _now_iso()

    manifest = dict(state["manifest"])
    manifest["updated_at"] = _now_iso()
    project_map = _load_embedded_projects(file_name)
    project_map[new_hash] = {
        "project_hash": new_hash,
        "display_name": str(target_item.get("display_name") or os.path.basename(child_project_path)),
        "project_size": len(payload),
        "project_blob": payload,
    }
    _write_series_snapshot(
        file_name,
        manifest=manifest,
        items=items,
        embedded_projects=project_map,
    )
    return target_item


def update_series_queue_runtime(
    file_name: str,
    *,
    active_item_id: str | None = None,
    failed_item_id: str | None = None,
    completed_item_ids: Iterable[str] | None = None,
) -> dict[str, object]:
    state = load_series_project(file_name)
    manifest = dict(state["manifest"])
    queue_runtime = dict(manifest.get("series_queue_runtime") or {})
    queue_runtime["active_item_id"] = active_item_id
    queue_runtime["failed_item_id"] = failed_item_id
    if completed_item_ids is not None:
        queue_runtime["completed_item_ids"] = list(completed_item_ids)
    queue_runtime["last_run_at"] = _now_iso()
    manifest["series_queue_runtime"] = queue_runtime
    save_series_manifest(file_name, manifest=manifest, items=state["items"])
    return manifest


def update_series_navigation_history(
    file_name: str,
    *,
    back: list[dict[str, object]],
    forward: list[dict[str, object]],
) -> dict[str, object]:
    state = load_series_project(file_name)
    manifest = dict(state["manifest"])
    manifest["series_navigation_history"] = {
        "back": list(back),
        "forward": list(forward),
    }
    save_series_manifest(file_name, manifest=manifest, items=state["items"])
    return manifest


def update_series_global_settings(
    file_name: str,
    global_settings: dict[str, object],
) -> dict[str, object]:
    state = load_series_project(file_name)
    manifest = dict(state["manifest"])
    manifest["global_settings"] = normalize_series_global_settings(global_settings)
    save_series_manifest(file_name, manifest=manifest, items=state["items"])
    return manifest


def update_series_settings(
    file_name: str,
    series_settings: dict[str, object],
) -> dict[str, object]:
    state = load_series_project(file_name)
    manifest = dict(state["manifest"])
    manifest["series_settings"] = normalize_series_settings(series_settings)
    save_series_manifest(file_name, manifest=manifest, items=state["items"])
    return manifest


def update_series_item_status(
    file_name: str,
    *,
    series_item_id: str,
    status: str,
) -> list[dict[str, object]]:
    state = load_series_project(file_name)
    items = list(state["items"])
    for item in items:
        if str(item.get("series_item_id")) == str(series_item_id):
            item["status"] = str(status)
            item["updated_at"] = _now_iso()
            break
    else:
        raise KeyError(series_item_id)
    save_series_manifest(file_name, manifest=state["manifest"], items=items)
    return items


def update_series_items_order(
    file_name: str,
    ordered_item_ids: list[str],
) -> list[dict[str, object]]:
    state = load_series_project(file_name)
    items_by_id = {
        str(item["series_item_id"]): dict(item)
        for item in state["items"]
    }
    ordered_items: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    for index, item_id in enumerate(ordered_item_ids, start=1):
        item = items_by_id.get(str(item_id))
        if item is None:
            continue
        seen_ids.add(str(item_id))
        item["queue_index"] = index
        item["updated_at"] = _now_iso()
        ordered_items.append(item)

    for item_id, item in items_by_id.items():
        if item_id in seen_ids:
            continue
        item["queue_index"] = len(ordered_items) + 1
        item["updated_at"] = _now_iso()
        ordered_items.append(item)

    save_series_manifest(file_name, manifest=state["manifest"], items=ordered_items)
    return ordered_items


def remove_series_item(file_name: str, series_item_id: str) -> list[dict[str, object]]:
    state = load_series_project(file_name)
    items = [
        dict(item)
        for item in state["items"]
        if str(item.get("series_item_id")) != str(series_item_id)
    ]
    for index, item in enumerate(items, start=1):
        item["queue_index"] = index
        item["updated_at"] = _now_iso()
    save_series_manifest(file_name, manifest=state["manifest"], items=items)
    return items
