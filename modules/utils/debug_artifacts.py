from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping, Sequence
import hashlib
import json
import logging
import os
import re
import uuid


logger = logging.getLogger(__name__)


DEBUG_ARTIFACT_SCHEMA_VERSION = 1
DEBUG_ARTIFACT_FOLDER_NAME = "debug"
DEBUG_MANIFEST_NAME = "manifest.json"

_DEBUG_EXPORT_KEYS = (
    "export_ocr_debug",
    "export_detector_overlay",
    "export_raw_mask",
    "export_mask_overlay",
    "export_cleanup_mask_delta",
    "export_debug_metadata",
)
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)
_UNSAFE_COMPONENT_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MULTISPACE_RE = re.compile(r"\s+")
_ACTIVE_LOCK = RLock()
_ACTIVE_RUN: DebugArtifactRun | None = None


class DebugArtifactError(RuntimeError):
    """Raised when a diagnostic sidecar cannot be used safely."""


def _is_link_like(path: Path) -> bool:
    try:
        return path.is_symlink() or bool(
            getattr(path, "is_junction", lambda: False)()
        )
    except OSError:
        return True


def _env_enabled(name: str) -> bool:
    value = str(os.environ.get(name, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def sanitize_debug_component(value: Any, *, fallback: str = "page") -> str:
    candidate = str(value or "").strip()
    candidate = _UNSAFE_COMPONENT_RE.sub("_", candidate)
    candidate = candidate.replace("\n", " ").replace("\r", " ")
    candidate = _MULTISPACE_RE.sub(" ", candidate).strip(" ._")
    if not candidate:
        candidate = fallback
    stem = candidate.split(".", 1)[0].upper()
    if stem in _WINDOWS_RESERVED_NAMES:
        candidate = f"_{candidate}"
    return candidate[:96].rstrip(" .") or fallback


def _canonical_path(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(
    main_page: Any,
    image_path: str,
) -> dict[str, str] | None:
    records = getattr(main_page, "export_source_by_path", {}) or {}
    wanted = _canonical_path(image_path)
    for record_path, raw_record in records.items():
        if _canonical_path(str(record_path)) != wanted:
            continue
        if not isinstance(raw_record, Mapping):
            return None
        source_path = str(raw_record.get("source_path", "") or "").strip()
        if not source_path:
            return None
        kind = str(raw_record.get("kind", "file") or "file").strip().lower()
        return {
            "kind": "archive" if kind == "archive" else "file",
            "source_path": os.path.abspath(source_path),
        }
    return None


def _archive_source(main_page: Any, image_path: str) -> str:
    image_key = _canonical_path(image_path)
    file_handler = getattr(main_page, "file_handler", None)
    for record in getattr(file_handler, "archive_info", []) or []:
        if not isinstance(record, Mapping):
            continue
        archive_path = str(record.get("archive_path", "") or "").strip()
        if not archive_path:
            continue
        for extracted in record.get("extracted_images", []) or []:
            if _canonical_path(str(extracted)) == image_key:
                return os.path.abspath(archive_path)
    return ""


def _debug_sidecar_path(
    main_page: Any,
    selected_paths: Sequence[str],
) -> tuple[Path, str]:
    project_file = str(getattr(main_page, "project_file", "") or "").strip()
    if project_file:
        return Path(os.path.abspath(project_file + ".cache")), "project"

    source_paths: list[str] = []
    archive_paths: set[str] = set()
    for image_path in selected_paths:
        record = _source_record(main_page, image_path)
        if record is not None:
            if record["kind"] == "archive":
                archive_paths.add(_canonical_path(record["source_path"]))
            else:
                source_paths.append(record["source_path"])
            continue
        archive_path = _archive_source(main_page, image_path)
        if archive_path:
            archive_paths.add(_canonical_path(archive_path))
        else:
            source_paths.append(os.path.abspath(image_path))

    if len(archive_paths) == 1 and not source_paths:
        archive_path = next(iter(archive_paths))
        return Path(f"{archive_path}.ctcache"), "archive"

    parents = {
        _canonical_path(os.path.dirname(source_path))
        for source_path in source_paths
        if source_path
    }
    if len(parents) == 1:
        source_folder = Path(next(iter(parents)))
        folder_name = sanitize_debug_component(
            source_folder.name,
            fallback="source-folder",
        )
        return source_folder.parent / f"{folder_name}.ctcache", "folder"

    seed = "\n".join(
        sorted([*archive_paths, *(_canonical_path(path) for path in source_paths)])
    )
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
    first = (
        Path(source_paths[0]).parent
        if source_paths
        else Path(next(iter(archive_paths))).parent
        if archive_paths
        else Path.cwd()
    )
    return first / f"comic-translate-mixed-{digest}.ctcache", "mixed"


def _ensure_safe_directory(path: Path) -> None:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and _is_link_like(path):
        raise DebugArtifactError(
            "Symbolic-link diagnostic sidecars are not supported."
        )
    path.mkdir(parents=False, exist_ok=True)
    if _is_link_like(path):
        raise DebugArtifactError(
            "Symbolic-link diagnostic sidecars are not supported."
        )
    if Path(os.path.realpath(path)).parent != Path(os.path.realpath(parent)):
        raise DebugArtifactError("Diagnostic sidecar escaped its parent folder.")


def _ensure_direct_child_directory(parent: Path, name: str) -> Path:
    if (
        Path(name).name != name
        or "/" in name
        or "\\" in name
        or name in {"", ".", ".."}
    ):
        raise DebugArtifactError("Invalid diagnostic subdirectory name.")
    if not parent.is_dir() or _is_link_like(parent):
        raise DebugArtifactError("Diagnostic parent folder is unavailable.")
    child = parent / name
    if child.exists() and _is_link_like(child):
        raise DebugArtifactError(
            "Symbolic-link diagnostic folders are not supported."
        )
    child.mkdir(parents=False, exist_ok=True)
    if _is_link_like(child):
        raise DebugArtifactError(
            "Symbolic-link diagnostic folders are not supported."
        )
    if Path(os.path.realpath(child)).parent != Path(os.path.realpath(parent)):
        raise DebugArtifactError("Diagnostic folder escaped its parent.")
    return child


def _safe_debug_file_path(
    directory: str | os.PathLike[str],
    file_name: str,
) -> Path:
    if (
        Path(file_name).name != file_name
        or "/" in file_name
        or "\\" in file_name
        or file_name in {"", ".", ".."}
    ):
        raise DebugArtifactError("Invalid diagnostic file name.")
    root = Path(directory)
    if not root.is_dir() or _is_link_like(root):
        raise DebugArtifactError("Diagnostic output folder is unavailable.")
    target = root / file_name
    if target.exists() and _is_link_like(target):
        raise DebugArtifactError(
            "Symbolic-link diagnostic files are not supported."
        )
    if Path(os.path.realpath(target.parent)) != Path(os.path.realpath(root)):
        raise DebugArtifactError("Diagnostic file escaped its output folder.")
    return target


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(
        f".{path.name}.partial-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    try:
        with open(temporary, "xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_debug_json(
    directory: str | os.PathLike[str],
    file_name: str,
    payload: Mapping[str, Any],
) -> str:
    target = _safe_debug_file_path(directory, file_name)
    _atomic_json(target, _json_safe(payload))
    return str(target)


def atomic_debug_file(
    directory: str | os.PathLike[str],
    file_name: str,
    writer: Callable[[str], Any],
) -> str:
    target = _safe_debug_file_path(directory, file_name)
    suffix = target.suffix or ".png"
    temporary = target.with_name(
        f".{target.stem}.partial-{os.getpid()}-"
        f"{uuid.uuid4().hex[:8]}{suffix}"
    )
    try:
        result = writer(str(temporary))
        if result is False:
            raise DebugArtifactError("Diagnostic file writer reported failure.")
        if not temporary.is_file() or _is_link_like(temporary):
            raise DebugArtifactError("Diagnostic file write did not complete.")
        if target.exists() and _is_link_like(target):
            raise DebugArtifactError(
                "Symbolic-link diagnostic files are not supported."
            )
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return str(target)


def atomic_debug_image(
    directory: str | os.PathLike[str],
    file_name: str,
    image: Any,
) -> str:
    def write_image(path: str) -> Any:
        import imkit as imk

        return imk.write_image(path, image)

    return atomic_debug_file(directory, file_name, write_image)


def append_debug_jsonl(
    directory: str | os.PathLike[str],
    file_name: str,
    payload: Mapping[str, Any],
    *,
    ensure_ascii: bool = False,
) -> str:
    target = _safe_debug_file_path(directory, file_name)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOINHERIT", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags, 0o600)
    try:
        with os.fdopen(
            descriptor,
            "a",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            descriptor = -1
            handle.write(
                json.dumps(
                    _json_safe(payload),
                    ensure_ascii=ensure_ascii,
                    separators=(",", ":"),
                )
                + "\n"
            )
            handle.flush()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return str(target)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


@dataclass
class DebugArtifactRun:
    sidecar_root: Path
    run_root: Path
    run_id: str
    run_type: str
    source_kind: str
    selected_paths: tuple[str, ...]
    toggles: dict[str, bool]
    started_at: str
    _lock: RLock = field(default_factory=RLock, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def runtime_root(self) -> Path:
        return self.run_root / "runtime"

    @property
    def manifest_path(self) -> Path:
        return self.run_root / DEBUG_MANIFEST_NAME

    def page_directory(self, image_path: str) -> Path:
        wanted = _canonical_path(image_path)
        index: int | None = None
        for candidate_index, candidate in enumerate(self.selected_paths):
            if _canonical_path(candidate) == wanted:
                index = candidate_index
                break
        if index is None:
            raise DebugArtifactError(
                "Diagnostic page is not part of the active run."
            )
        safe_name = sanitize_debug_component(
            Path(image_path).stem,
            fallback=f"page-{index + 1:04d}",
        )
        page_name = f"page-{index + 1:04d}_{safe_name}"
        with self._lock:
            if self._closed:
                raise DebugArtifactError("Diagnostic run is already closed.")
            return _ensure_direct_child_directory(self.run_root, page_name)

    def runtime_directory(self, component: str = "") -> Path:
        with self._lock:
            if self._closed:
                raise DebugArtifactError("Diagnostic run is already closed.")
            runtime_root = _ensure_direct_child_directory(
                self.run_root,
                "runtime",
            )
            if not component:
                return runtime_root
            safe_component = sanitize_debug_component(
                component,
                fallback="component",
            )
            return _ensure_direct_child_directory(
                runtime_root,
                safe_component,
            )

    def append_runtime(
        self,
        service: str,
        payload: Any,
        *,
        kind: str,
    ) -> bool:
        file_names = {
            "gemma": "gemma-raw-responses.jsonl",
            "hunyuanocr": "hunyuanocr-raw-responses.jsonl",
            "mangalmm": "mangalmm-raw-responses.jsonl",
        }
        file_name = file_names.get(str(service or "").strip().lower())
        if not file_name:
            return False
        toggle_name = {
            "gemma": "gemma_raw_response",
            "hunyuanocr": "hunyuanocr_raw_response",
            "mangalmm": "mangalmm_raw_response",
        }.get(str(service or "").strip().lower(), "")
        if not toggle_name or not self.toggles.get(toggle_name, False):
            return False
        with self._lock:
            if self._closed:
                return False
            record = {
                "schema_version": DEBUG_ARTIFACT_SCHEMA_VERSION,
                "recorded_at": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "kind": str(kind or "response"),
                "payload": _json_safe(payload),
            }
            runtime_root = self.runtime_directory()
            append_debug_jsonl(
                runtime_root,
                file_name,
                record,
            )
        return True

    def write_manifest(self, status: str) -> None:
        with self._lock:
            self._write_manifest_locked(status)

    def finalize(self, status: str) -> None:
        with self._lock:
            self._closed = True
            self._write_manifest_locked(status)

    def _write_manifest_locked(self, status: str) -> None:
        files: list[dict[str, Any]] = []
        if status != "running":
            for root, directory_names, file_names in os.walk(
                self.run_root,
                topdown=True,
                followlinks=False,
            ):
                root_path = Path(root)
                directory_names[:] = [
                    name
                    for name in sorted(directory_names)
                    if not _is_link_like(root_path / name)
                ]
                for file_name in sorted(file_names):
                    path = root_path / file_name
                    if (
                        not path.is_file()
                        or _is_link_like(path)
                        or path.name == DEBUG_MANIFEST_NAME
                        or ".partial-" in path.name
                    ):
                        continue
                    relative = path.relative_to(self.run_root).as_posix()
                    files.append(
                        {
                            "path": relative,
                            "size": int(path.stat().st_size),
                            "sha256": _sha256_file(path),
                        }
                    )
            files.sort(key=lambda item: str(item["path"]))
        payload = {
            "schema_version": DEBUG_ARTIFACT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "run_type": self.run_type,
            "source_kind": self.source_kind,
            "status": str(status or "unknown"),
            "started_at": self.started_at,
            "finished_at": (
                datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
                if status != "running"
                else ""
            ),
            "page_count": len(self.selected_paths),
            "toggles": dict(sorted(self.toggles.items())),
            "files": files,
        }
        _atomic_json(self.manifest_path, payload)


def _runtime_toggles(main_page: Any) -> dict[str, bool]:
    settings_page = getattr(main_page, "settings_page", None)

    def enabled(getter_name: str) -> bool:
        getter = getattr(settings_page, getter_name, None)
        if not callable(getter):
            return False
        try:
            return bool((getter() or {}).get("raw_response_logging", False))
        except Exception:
            return False

    return {
        "gemma_raw_response": enabled("get_gemma_local_server_settings"),
        "hunyuanocr_raw_response": enabled("get_hunyuan_ocr_settings"),
        "mangalmm_raw_response": enabled("get_mangalmm_ocr_settings"),
        "memlog": _env_enabled("CT_ENABLE_MEMLOG"),
        "gpu_bench": _env_enabled("CT_ENABLE_GPU_BENCH"),
        "mangalmm_engine_debug": bool(
            str(os.environ.get("CT_MANGALMM_DEBUG_ROOT", "") or "").strip()
        ),
    }


def prepare_debug_artifact_run(
    main_page: Any,
    selected_paths: Sequence[str],
    *,
    run_type: str,
) -> DebugArtifactRun | None:
    if isinstance(
        getattr(main_page, "_debug_artifact_run", None),
        DebugArtifactRun,
    ):
        finish_debug_artifact_run(main_page, status="interrupted")
    paths = tuple(str(path) for path in selected_paths if str(path))
    if not paths:
        return None
    try:
        export_settings = dict(main_page.get_resolved_export_settings() or {})
    except Exception:
        export_settings = {}
    toggles = {
        key: bool(export_settings.get(key, False))
        for key in _DEBUG_EXPORT_KEYS
    }
    toggles.update(_runtime_toggles(main_page))
    if not any(toggles.values()):
        setattr(main_page, "_debug_artifact_run", None)
        return None

    try:
        sidecar_root, source_kind = _debug_sidecar_path(main_page, paths)
        _ensure_safe_directory(sidecar_root)
        debug_root = sidecar_root / DEBUG_ARTIFACT_FOLDER_NAME
        if debug_root.exists() and _is_link_like(debug_root):
            raise DebugArtifactError(
                "Symbolic-link diagnostic roots are not supported."
            )
        debug_root.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc)
        while True:
            run_id = (
                f"run-{started.strftime('%Y%m%dT%H%M%SZ')}-"
                f"{uuid.uuid4().hex[:8]}"
            )
            run_root = debug_root / run_id
            try:
                run_root.mkdir(parents=False, exist_ok=False)
                break
            except FileExistsError:
                continue
        run = DebugArtifactRun(
            sidecar_root=sidecar_root,
            run_root=run_root,
            run_id=run_id,
            run_type=str(run_type or "batch"),
            source_kind=source_kind,
            selected_paths=paths,
            toggles=toggles,
            started_at=started.isoformat().replace("+00:00", "Z"),
        )
        run.write_manifest("running")
    except Exception:
        logger.warning(
            "Diagnostic artifact sidecar is unavailable; diagnostics will "
            "continue without persistent artifacts.",
            exc_info=True,
        )
        setattr(main_page, "_debug_artifact_run", None)
        return None

    setattr(main_page, "_debug_artifact_run", run)
    setattr(main_page, "_last_debug_artifact_root", str(run.run_root))
    set_active_debug_run(run)
    memlogger = getattr(main_page, "_memlogger", None)
    binder = getattr(memlogger, "bind_debug_run", None)
    if callable(binder):
        try:
            binder(str(run.runtime_root))
        except Exception:
            logger.debug("Unable to bind memory diagnostics to the run.", exc_info=True)
    return run


def finish_debug_artifact_run(main_page: Any, *, status: str) -> str:
    run = getattr(main_page, "_debug_artifact_run", None)
    if not isinstance(run, DebugArtifactRun):
        return ""
    memlogger = getattr(main_page, "_memlogger", None)
    try:
        if memlogger is not None:
            memlogger.emit("debug_run_end", extra={"status": str(status or "")})
    except Exception:
        pass
    unbinder = getattr(memlogger, "unbind_debug_run", None)
    if callable(unbinder):
        try:
            unbinder()
        except Exception:
            pass
    _clear_active_debug_run(run)
    try:
        run.finalize(str(status or "unknown"))
    except Exception:
        logger.warning(
            "Unable to finalize the diagnostic run manifest.",
            exc_info=True,
        )
    setattr(main_page, "_debug_artifact_run", None)
    return str(run.run_root)


def active_debug_page_directory(main_page: Any, image_path: str) -> str:
    run = getattr(main_page, "_debug_artifact_run", None)
    if not isinstance(run, DebugArtifactRun):
        return ""
    try:
        return str(run.page_directory(image_path))
    except Exception:
        logger.warning(
            "Unable to prepare a diagnostic page folder.",
            exc_info=True,
        )
        return ""


def active_debug_runtime_directory(component: str = "") -> str:
    with _ACTIVE_LOCK:
        run = _ACTIVE_RUN
    if run is None:
        return ""
    try:
        return str(run.runtime_directory(component))
    except Exception:
        logger.warning(
            "Unable to prepare a diagnostic runtime folder.",
            exc_info=True,
        )
        return ""


def set_active_debug_run(run: DebugArtifactRun | None) -> None:
    global _ACTIVE_RUN
    with _ACTIVE_LOCK:
        _ACTIVE_RUN = run


def _clear_active_debug_run(run: DebugArtifactRun) -> None:
    global _ACTIVE_RUN
    with _ACTIVE_LOCK:
        if _ACTIVE_RUN is run:
            _ACTIVE_RUN = None


def append_active_raw_response(
    service: str,
    payload: Any,
    *,
    kind: str = "response_json",
) -> bool:
    with _ACTIVE_LOCK:
        run = _ACTIVE_RUN
    if run is None:
        return False
    try:
        return run.append_runtime(service, payload, kind=kind)
    except Exception:
        logger.debug(
            "Unable to append a raw diagnostic response.",
            exc_info=True,
        )
        return False
