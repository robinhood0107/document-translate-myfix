from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import tempfile
import time
import uuid


logger = logging.getLogger(__name__)


PROJECT_CHECKPOINT_SCHEMA_VERSION = 1
PROJECT_CHECKPOINT_REFERENCE_KEY = "project_checkpoint"
PROJECT_CHECKPOINT_DB_NAME = "checkpoint.sqlite3"
PROJECT_CHECKPOINT_README_NAME = "README.txt"
PROJECT_CHECKPOINT_OBJECT_ROOT = ("objects", "sha256")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ROLE_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_.-]{0,127}$")

# Translation and inpainting are parallel after OCR. Inpainting consumes the
# OCR-filtered/protected ordered block list in addition to detection geometry,
# while render consumes both branches.
PROJECT_STAGE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "detection": (),
    "ocr": ("detection",),
    "translation": ("ocr",),
    "inpaint": ("detection", "ocr"),
    "render": ("translation", "inpaint"),
}

_README_TEXT = """Comic Translate project checkpoint cache

This folder belongs to the adjacent .ctpr project file.
Large stage artifacts are stored as immutable SHA-256 objects. The SQLite
database contains only manifests and stage fingerprints.

The project remains usable if this folder is missing or damaged. Comic
Translate will recompute the affected stages. Do not edit files in this folder
while the project is open.
"""


class ProjectCheckpointError(RuntimeError):
    """Raised for an unavailable or invalid project checkpoint store."""


@dataclass(frozen=True)
class ProjectCheckpointReference:
    schema_version: int
    project_uuid: str
    cache_id: str
    sidecar_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "project_uuid": self.project_uuid,
            "cache_id": self.cache_id,
            "sidecar_name": self.sidecar_name,
        }


@dataclass(frozen=True)
class ProjectCheckpointHit:
    page_key: str
    stage: str
    fingerprint: str
    payload: Mapping[str, Any]
    objects: Mapping[str, str]


@dataclass(frozen=True)
class ProjectCheckpointStats:
    stage_records: int
    object_records: int
    object_files: int
    object_bytes: int


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_uuid(value: Any) -> str:
    return str(uuid.UUID(str(value)))


def _new_uuid() -> str:
    return str(uuid.uuid4())


def expected_checkpoint_sidecar_name(project_file: str | os.PathLike[str]) -> str:
    file_name = os.path.basename(os.fspath(project_file))
    if not file_name:
        raise ProjectCheckpointError("Project file name is required.")
    return f"{file_name}.cache"


def _validate_sidecar_name(value: Any) -> str:
    sidecar_name = str(value or "")
    if (
        not sidecar_name
        or sidecar_name in {".", ".."}
        or os.path.isabs(sidecar_name)
        or "/" in sidecar_name
        or "\\" in sidecar_name
        or os.path.basename(sidecar_name) != sidecar_name
        or not sidecar_name.lower().endswith(".ctpr.cache")
    ):
        raise ProjectCheckpointError(
            "Project checkpoint sidecar must be a relative .ctpr.cache folder name."
        )
    return sidecar_name


def normalize_checkpoint_reference(
    value: Mapping[str, Any] | ProjectCheckpointReference | None,
    project_file: str | os.PathLike[str],
    *,
    create_if_missing: bool = True,
    refresh_sidecar_name: bool = False,
) -> ProjectCheckpointReference | None:
    if isinstance(value, ProjectCheckpointReference):
        if value.schema_version != PROJECT_CHECKPOINT_SCHEMA_VERSION:
            raise ProjectCheckpointError(
                f"Unsupported project checkpoint schema version: "
                f"{value.schema_version}"
            )
        try:
            project_uuid = _canonical_uuid(value.project_uuid)
            cache_id = _canonical_uuid(value.cache_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ProjectCheckpointError(
                "Invalid project checkpoint identity."
            ) from exc
        reference = ProjectCheckpointReference(
            schema_version=PROJECT_CHECKPOINT_SCHEMA_VERSION,
            project_uuid=project_uuid,
            cache_id=cache_id,
            sidecar_name=_validate_sidecar_name(value.sidecar_name),
        )
    elif isinstance(value, Mapping):
        try:
            schema_version = int(value.get("schema_version", 0))
        except (TypeError, ValueError) as exc:
            raise ProjectCheckpointError(
                "Invalid project checkpoint schema version."
            ) from exc
        if schema_version != PROJECT_CHECKPOINT_SCHEMA_VERSION:
            raise ProjectCheckpointError(
                f"Unsupported project checkpoint schema version: {schema_version}"
            )
        try:
            project_uuid = _canonical_uuid(value.get("project_uuid"))
            cache_id = _canonical_uuid(value.get("cache_id"))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ProjectCheckpointError(
                "Invalid project checkpoint identity."
            ) from exc
        reference = ProjectCheckpointReference(
            schema_version=schema_version,
            project_uuid=project_uuid,
            cache_id=cache_id,
            sidecar_name=_validate_sidecar_name(value.get("sidecar_name")),
        )
    elif create_if_missing:
        reference = ProjectCheckpointReference(
            schema_version=PROJECT_CHECKPOINT_SCHEMA_VERSION,
            project_uuid=_new_uuid(),
            cache_id=_new_uuid(),
            sidecar_name=expected_checkpoint_sidecar_name(project_file),
        )
    else:
        return None

    if refresh_sidecar_name:
        reference = ProjectCheckpointReference(
            schema_version=reference.schema_version,
            project_uuid=reference.project_uuid,
            cache_id=reference.cache_id,
            sidecar_name=expected_checkpoint_sidecar_name(project_file),
        )
    return reference


def checkpoint_reference_for_save(
    value: Mapping[str, Any] | ProjectCheckpointReference | None,
    project_file: str | os.PathLike[str],
    *,
    clone_identity: bool = False,
) -> ProjectCheckpointReference:
    try:
        current = normalize_checkpoint_reference(
            value,
            project_file,
            create_if_missing=True,
            refresh_sidecar_name=True,
        )
    except ProjectCheckpointError:
        current = normalize_checkpoint_reference(
            None,
            project_file,
            create_if_missing=True,
            refresh_sidecar_name=True,
        )
    assert current is not None
    if not clone_identity:
        return current
    return ProjectCheckpointReference(
        schema_version=PROJECT_CHECKPOINT_SCHEMA_VERSION,
        project_uuid=_new_uuid(),
        cache_id=_new_uuid(),
        sidecar_name=expected_checkpoint_sidecar_name(project_file),
    )


def load_checkpoint_reference_into_project(
    project: Any,
    project_file: str | os.PathLike[str],
    value: Mapping[str, Any] | None,
) -> ProjectCheckpointReference:
    reference_was_persisted = isinstance(value, Mapping)
    try:
        reference = normalize_checkpoint_reference(
            value,
            project_file,
            create_if_missing=True,
        )
        assert reference is not None
        warning = ""
    except ProjectCheckpointError as exc:
        # A malformed reference must not prevent an old or hand-edited project
        # from opening. Generate a fresh in-memory identity; it is persisted only
        # when the user later saves the project.
        reference = normalize_checkpoint_reference(
            None,
            project_file,
            create_if_missing=True,
        )
        assert reference is not None
        warning = str(exc)
        reference_was_persisted = False
        logger.warning(
            "Project checkpoint reference ignored; stages will be recomputed. "
            "project=%s reason=%s",
            os.path.basename(os.fspath(project_file)),
            warning,
        )
    project.project_checkpoint_reference = reference.to_dict()
    project.project_checkpoint_reference_persisted = bool(
        reference_was_persisted
    )
    project.project_checkpoint_warning = warning
    return reference


def checkpoint_sidecar_path(
    project_file: str | os.PathLike[str],
    reference: Mapping[str, Any] | ProjectCheckpointReference,
) -> Path:
    normalized = normalize_checkpoint_reference(
        reference,
        project_file,
        create_if_missing=False,
    )
    if normalized is None:
        raise ProjectCheckpointError("Project checkpoint reference is missing.")
    project_path = Path(project_file).expanduser().absolute()
    project_parent = project_path.parent
    sidecar = project_parent / normalized.sidecar_name
    parent_real = Path(os.path.realpath(project_parent))
    sidecar_real = Path(os.path.realpath(sidecar))
    try:
        sidecar_real.relative_to(parent_real)
    except ValueError as exc:
        raise ProjectCheckpointError(
            "Project checkpoint sidecar escapes the project folder."
        ) from exc
    if sidecar.exists() and sidecar.is_symlink():
        raise ProjectCheckpointError(
            "Symbolic-link project checkpoint sidecars are not supported."
        )
    return sidecar


def _validate_page_key(page_key: Any) -> str:
    normalized = str(page_key or "").strip()
    if not normalized or "\x00" in normalized or len(normalized) > 1024:
        raise ValueError("Invalid project checkpoint page key.")
    return normalized


def _validate_stage(stage: Any) -> str:
    normalized = str(stage or "").strip().lower()
    if normalized not in PROJECT_STAGE_DEPENDENCIES:
        raise ValueError(f"Unsupported project checkpoint stage: {normalized}")
    return normalized


def _validate_fingerprint(fingerprint: Any) -> str:
    normalized = str(fingerprint or "").strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError("Project checkpoint fingerprint must be a SHA-256 digest.")
    return normalized


def _validate_object_hash(object_hash: Any) -> str:
    normalized = str(object_hash or "").strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError("Project checkpoint object hash must be a SHA-256 digest.")
    return normalized


def stage_downstream(stage: str, *, include_self: bool = True) -> tuple[str, ...]:
    normalized = _validate_stage(stage)
    selected = {normalized} if include_self else set()
    changed = True
    while changed:
        changed = False
        for candidate, dependencies in PROJECT_STAGE_DEPENDENCIES.items():
            if candidate in selected:
                continue
            if any(dependency in selected or dependency == normalized for dependency in dependencies):
                selected.add(candidate)
                changed = True
    return tuple(
        stage_name
        for stage_name in PROJECT_STAGE_DEPENDENCIES
        if stage_name in selected
    )


class ProjectCheckpointStore:
    """Fail-open project-local stage manifest and content-addressed object store."""

    def __init__(
        self,
        project_file: str | os.PathLike[str],
        reference: Mapping[str, Any] | ProjectCheckpointReference,
        *,
        enabled: bool,
        timeout_sec: float = 0.5,
    ) -> None:
        self.project_file = Path(project_file).expanduser().absolute()
        normalized = normalize_checkpoint_reference(
            reference,
            self.project_file,
            create_if_missing=False,
        )
        if normalized is None:
            raise ProjectCheckpointError("Project checkpoint reference is missing.")
        self.reference = normalized
        self.sidecar_path = checkpoint_sidecar_path(self.project_file, normalized)
        self.db_path = self.sidecar_path / PROJECT_CHECKPOINT_DB_NAME
        self.object_root = self.sidecar_path.joinpath(*PROJECT_CHECKPOINT_OBJECT_ROOT)
        self.enabled = bool(enabled)
        self.timeout_sec = max(0.0, float(timeout_sec))
        self.disabled_reason = ""

    @property
    def available(self) -> bool:
        return self.enabled and not self.disabled_reason

    def _disable(self, exc: BaseException) -> None:
        if self.disabled_reason:
            return
        self.disabled_reason = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "Project checkpoint cache disabled for this run; project processing "
            "will continue with normal stage computation. Existing cache data "
            "was not deleted or rewritten. reason=%s",
            self.disabled_reason,
        )

    def _connect(self, *, create: bool) -> sqlite3.Connection:
        if not self.enabled:
            raise ProjectCheckpointError("Project checkpoint cache is disabled.")
        if self.disabled_reason:
            raise ProjectCheckpointError(self.disabled_reason)
        if self.db_path.is_symlink():
            raise ProjectCheckpointError(
                "Symbolic-link project checkpoint databases are not supported."
            )
        db_preexisted = self.db_path.exists()
        if not create and not self.db_path.is_file():
            raise FileNotFoundError(self.db_path)
        if create and not db_preexisted:
            self._ensure_layout()

        connection = sqlite3.connect(
            self.db_path,
            timeout=self.timeout_sec,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute(
                f"PRAGMA busy_timeout = {int(self.timeout_sec * 1000)}"
            )
            connection.execute("PRAGMA foreign_keys = ON")
            self._initialize_or_validate_schema(
                connection,
                create=create and not db_preexisted,
            )
            if create and db_preexisted:
                # Do not add or rewrite sidecar files until an existing DB has
                # passed schema and identity validation.
                self._ensure_layout()
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
        except BaseException:
            connection.close()
            raise
        return connection

    def _ensure_layout(self) -> None:
        sidecar_parent = self.sidecar_path.parent
        sidecar_parent.mkdir(parents=True, exist_ok=True)
        if self.sidecar_path.exists() and self.sidecar_path.is_symlink():
            raise ProjectCheckpointError(
                "Symbolic-link project checkpoint sidecars are not supported."
            )
        if self.object_root.exists() and self.object_root.is_symlink():
            raise ProjectCheckpointError(
                "Symbolic-link checkpoint object roots are not supported."
            )
        self.object_root.mkdir(parents=True, exist_ok=True)
        readme = self.sidecar_path / PROJECT_CHECKPOINT_README_NAME
        if readme.exists() and readme.is_symlink():
            raise ProjectCheckpointError(
                "Symbolic-link checkpoint README files are not supported."
            )
        if not readme.exists():
            _atomic_write_bytes(readme, _README_TEXT.encode("utf-8"))

    def _initialize_or_validate_schema(
        self,
        connection: sqlite3.Connection,
        *,
        create: bool,
    ) -> None:
        tables = {
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if tables:
            required = {"metadata", "stage_records", "stage_objects"}
            if not required.issubset(tables):
                raise sqlite3.DatabaseError(
                    "Existing project checkpoint is missing required schema tables."
                )
            metadata = {
                str(row["key"]): str(row["value"])
                for row in connection.execute(
                    "SELECT key, value FROM metadata"
                ).fetchall()
            }
            try:
                schema_version = int(metadata.get("schema_version", "0"))
            except ValueError as exc:
                raise sqlite3.DatabaseError(
                    "Invalid project checkpoint schema version."
                ) from exc
            if schema_version != PROJECT_CHECKPOINT_SCHEMA_VERSION:
                raise sqlite3.DatabaseError(
                    "Unsupported project checkpoint schema version."
                )
            if metadata.get("project_uuid") != self.reference.project_uuid:
                raise sqlite3.DatabaseError(
                    "Project checkpoint project identity does not match."
                )
            if metadata.get("cache_id") != self.reference.cache_id:
                raise sqlite3.DatabaseError(
                    "Project checkpoint cache identity does not match."
                )
            return

        if not create:
            raise sqlite3.DatabaseError("Project checkpoint schema is missing.")

        with connection:
            connection.executescript(
                """
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE stage_records (
                    page_key TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(page_key, stage)
                );

                CREATE TABLE stage_objects (
                    page_key TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    role TEXT NOT NULL,
                    object_hash TEXT NOT NULL,
                    PRIMARY KEY(page_key, stage, role),
                    FOREIGN KEY(page_key, stage)
                        REFERENCES stage_records(page_key, stage)
                        ON DELETE CASCADE
                );

                CREATE INDEX idx_stage_objects_hash
                    ON stage_objects(object_hash);
                """
            )
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES(?, ?)",
                (
                    ("schema_version", str(PROJECT_CHECKPOINT_SCHEMA_VERSION)),
                    ("project_uuid", self.reference.project_uuid),
                    ("cache_id", self.reference.cache_id),
                    ("created_at", str(time.time())),
                ),
            )

    def ensure_initialized(self) -> bool:
        if not self.enabled or self.disabled_reason:
            return False
        try:
            connection = self._connect(create=True)
            connection.close()
            return True
        except (OSError, RuntimeError, sqlite3.Error, ProjectCheckpointError) as exc:
            self._disable(exc)
            return False

    def _object_path(self, object_hash: str) -> Path:
        normalized = _validate_object_hash(object_hash)
        path = self.object_root / normalized[:2] / normalized
        root_real = Path(os.path.realpath(self.object_root))
        path_real = Path(os.path.realpath(path))
        try:
            path_real.relative_to(root_real)
        except ValueError as exc:
            raise ProjectCheckpointError(
                "Project checkpoint object path escapes the object root."
            ) from exc
        if path.exists() and path.is_symlink():
            raise ProjectCheckpointError(
                "Symbolic-link project checkpoint objects are not supported."
            )
        return path

    def put_object(self, payload: bytes | bytearray | memoryview) -> str | None:
        if not self.available:
            return None
        try:
            raw = bytes(payload)
            object_hash = hashlib.sha256(raw).hexdigest()
            if not self.ensure_initialized():
                return None
            path = self._object_path(object_hash)
            if path.is_file():
                if path.stat().st_size != len(raw) or _sha256_file(path) != object_hash:
                    raise ProjectCheckpointError(
                        "Existing checkpoint object failed integrity verification."
                    )
                return object_hash
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.parent.is_symlink():
                raise ProjectCheckpointError(
                    "Symbolic-link object directories are not supported."
                )
            _atomic_write_bytes(path, raw)
            return object_hash
        except (OSError, ValueError, ProjectCheckpointError) as exc:
            self._disable(exc)
            return None

    def read_object(self, object_hash: str) -> bytes | None:
        if not self.available:
            return None
        try:
            path = self._object_path(object_hash)
            if not path.is_file():
                return None
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != object_hash:
                raise ProjectCheckpointError(
                    "Project checkpoint object failed integrity verification."
                )
            return payload
        except (OSError, ValueError, ProjectCheckpointError) as exc:
            self._disable(exc)
            return None

    @staticmethod
    def _delete_stage_rows(
        connection: sqlite3.Connection,
        *,
        page_key: str | None,
        stages: tuple[str, ...] | None,
    ) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if page_key is not None:
            clauses.append("page_key = ?")
            params.append(page_key)
        if stages is not None:
            placeholders = ",".join("?" for _ in stages)
            clauses.append(f"stage IN ({placeholders})")
            params.extend(stages)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        cursor = connection.execute(f"DELETE FROM stage_records{where}", params)
        return max(0, int(cursor.rowcount))

    def record_stage(
        self,
        page_key: str,
        stage: str,
        fingerprint: str,
        *,
        payload: Mapping[str, Any] | None = None,
        objects: Mapping[str, str] | None = None,
    ) -> bool:
        if not self.available:
            return False
        try:
            normalized_page = _validate_page_key(page_key)
            normalized_stage = _validate_stage(stage)
            normalized_fingerprint = _validate_fingerprint(fingerprint)
            normalized_payload = dict(payload or {})
            payload_json = _canonical_json(normalized_payload)
            if not isinstance(json.loads(payload_json), dict):
                raise ValueError("Project checkpoint payload must be an object.")
            normalized_objects: dict[str, str] = {}
            for role, object_hash in (objects or {}).items():
                normalized_role = str(role or "").strip()
                if not _SAFE_ROLE_RE.fullmatch(normalized_role):
                    raise ValueError("Invalid project checkpoint object role.")
                normalized_hash = _validate_object_hash(object_hash)
                normalized_objects[normalized_role] = normalized_hash

            connection = self._connect(create=True)
            try:
                # Serialize manifest publication with cache cleanup. Object
                # verification happens after the write lock is acquired so a
                # concurrent cleanup cannot create a committed dangling
                # reference.
                connection.execute("BEGIN IMMEDIATE")
                for normalized_hash in normalized_objects.values():
                    object_path = self._object_path(normalized_hash)
                    if (
                        not object_path.is_file()
                        or _sha256_file(object_path) != normalized_hash
                    ):
                        raise ProjectCheckpointError(
                            "Stage manifest references a missing or corrupt object."
                        )
                now = time.time()
                existing = connection.execute(
                    """
                    SELECT fingerprint, payload_json
                    FROM stage_records
                    WHERE page_key = ? AND stage = ?
                    """,
                    (normalized_page, normalized_stage),
                ).fetchone()
                existing_objects = (
                    {
                        str(row["role"]): str(row["object_hash"])
                        for row in connection.execute(
                            """
                            SELECT role, object_hash
                            FROM stage_objects
                            WHERE page_key = ? AND stage = ?
                            """,
                            (normalized_page, normalized_stage),
                        ).fetchall()
                    }
                    if existing is not None
                    else {}
                )
                if (
                    existing is not None
                    and (
                        str(existing["fingerprint"]) != normalized_fingerprint
                        or str(existing["payload_json"]) != payload_json
                        or existing_objects != normalized_objects
                    )
                ):
                    downstream = stage_downstream(
                        normalized_stage,
                        include_self=False,
                    )
                    if downstream:
                        self._delete_stage_rows(
                            connection,
                            page_key=normalized_page,
                            stages=downstream,
                        )
                connection.execute(
                    """
                    INSERT INTO stage_records(
                        page_key, stage, fingerprint, payload_json,
                        created_at, updated_at
                    )
                    VALUES(?, ?, ?, ?, ?, ?)
                    ON CONFLICT(page_key, stage) DO UPDATE SET
                        fingerprint = excluded.fingerprint,
                        payload_json = excluded.payload_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        normalized_page,
                        normalized_stage,
                        normalized_fingerprint,
                        payload_json,
                        now,
                        now,
                    ),
                )
                connection.execute(
                    "DELETE FROM stage_objects WHERE page_key = ? AND stage = ?",
                    (normalized_page, normalized_stage),
                )
                connection.executemany(
                    """
                    INSERT INTO stage_objects(
                        page_key, stage, role, object_hash
                    )
                    VALUES(?, ?, ?, ?)
                    """,
                    [
                        (
                            normalized_page,
                            normalized_stage,
                            role,
                            object_hash,
                        )
                        for role, object_hash in sorted(normalized_objects.items())
                    ],
                )
                connection.commit()
                return True
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise
            finally:
                connection.close()
        except (
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            sqlite3.Error,
            ProjectCheckpointError,
        ) as exc:
            self._disable(exc)
            return False

    def lookup_stage(
        self,
        page_key: str,
        stage: str,
        fingerprint: str,
    ) -> ProjectCheckpointHit | None:
        if not self.available:
            return None
        try:
            normalized_page = _validate_page_key(page_key)
            normalized_stage = _validate_stage(stage)
            normalized_fingerprint = _validate_fingerprint(fingerprint)
            connection = self._connect(create=False)
            try:
                row = connection.execute(
                    """
                    SELECT payload_json
                    FROM stage_records
                    WHERE page_key = ? AND stage = ? AND fingerprint = ?
                    """,
                    (
                        normalized_page,
                        normalized_stage,
                        normalized_fingerprint,
                    ),
                ).fetchone()
                if row is None:
                    return None
                payload = json.loads(str(row["payload_json"]))
                if not isinstance(payload, dict):
                    raise sqlite3.DatabaseError(
                        "Project checkpoint payload must be an object."
                    )
                objects = {
                    str(object_row["role"]): _validate_object_hash(
                        object_row["object_hash"]
                    )
                    for object_row in connection.execute(
                        """
                        SELECT role, object_hash
                        FROM stage_objects
                        WHERE page_key = ? AND stage = ?
                        ORDER BY role
                        """,
                        (normalized_page, normalized_stage),
                    ).fetchall()
                }
                for object_hash in objects.values():
                    object_path = self._object_path(object_hash)
                    if (
                        not object_path.is_file()
                        or _sha256_file(object_path) != object_hash
                    ):
                        # Missing stage objects are ordinary cache misses. Keep
                        # the manifest untouched for manual inspection.
                        return None
                return ProjectCheckpointHit(
                    page_key=normalized_page,
                    stage=normalized_stage,
                    fingerprint=normalized_fingerprint,
                    payload=payload,
                    objects=objects,
                )
            finally:
                connection.close()
        except FileNotFoundError:
            return None
        except (
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            sqlite3.Error,
            ProjectCheckpointError,
        ) as exc:
            self._disable(exc)
            return None

    def invalidate(
        self,
        *,
        page_key: str | None = None,
        stage: str | None = None,
    ) -> int:
        if not self.available:
            return 0
        try:
            normalized_page = (
                _validate_page_key(page_key) if page_key is not None else None
            )
            stages = (
                stage_downstream(_validate_stage(stage), include_self=True)
                if stage is not None
                else None
            )
            connection = self._connect(create=False)
            try:
                connection.execute("PRAGMA foreign_keys = ON")
                with connection:
                    return self._delete_stage_rows(
                        connection,
                        page_key=normalized_page,
                        stages=stages,
                    )
            finally:
                connection.close()
        except FileNotFoundError:
            return 0
        except (OSError, ValueError, sqlite3.Error, ProjectCheckpointError) as exc:
            self._disable(exc)
            return 0

    def clean_unused_objects(self) -> dict[str, int]:
        if not self.available:
            return {"removed_files": 0, "removed_bytes": 0}
        try:
            connection = self._connect(create=False)
            try:
                connection.execute("BEGIN IMMEDIATE")
                referenced = {
                    _validate_object_hash(row["object_hash"])
                    for row in connection.execute(
                        "SELECT DISTINCT object_hash FROM stage_objects"
                    ).fetchall()
                }
                removed_files = 0
                removed_bytes = 0
                if not self.object_root.exists():
                    connection.commit()
                    return {
                        "removed_files": removed_files,
                        "removed_bytes": removed_bytes,
                    }
                for prefix_dir in self.object_root.iterdir():
                    if prefix_dir.is_symlink():
                        raise ProjectCheckpointError(
                            "Symbolic-link object directories are not supported."
                        )
                    if not prefix_dir.is_dir():
                        continue
                    for path in prefix_dir.iterdir():
                        if path.is_symlink():
                            raise ProjectCheckpointError(
                                "Symbolic-link checkpoint objects are not supported."
                            )
                        if (
                            not path.is_file()
                            or not _SHA256_RE.fullmatch(path.name)
                        ):
                            continue
                        if path.name in referenced:
                            continue
                        size = int(path.stat().st_size)
                        path.unlink()
                        removed_files += 1
                        removed_bytes += size
                    try:
                        prefix_dir.rmdir()
                    except OSError:
                        pass
                connection.commit()
                return {
                    "removed_files": removed_files,
                    "removed_bytes": removed_bytes,
                }
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise
            finally:
                connection.close()
        except FileNotFoundError:
            return {"removed_files": 0, "removed_bytes": 0}
        except (OSError, ValueError, sqlite3.Error, ProjectCheckpointError) as exc:
            self._disable(exc)
            return {"removed_files": 0, "removed_bytes": 0}

    def stats(self) -> ProjectCheckpointStats:
        if not self.available:
            return ProjectCheckpointStats(0, 0, 0, 0)
        try:
            connection = self._connect(create=False)
            try:
                stage_records = int(
                    connection.execute(
                        "SELECT COUNT(*) AS count FROM stage_records"
                    ).fetchone()["count"]
                )
                object_records = int(
                    connection.execute(
                        "SELECT COUNT(DISTINCT object_hash) AS count FROM stage_objects"
                    ).fetchone()["count"]
                )
            finally:
                connection.close()
            object_files = 0
            object_bytes = 0
            if self.object_root.exists():
                for path in self.object_root.glob("*/*"):
                    if path.is_file() and not path.is_symlink() and _SHA256_RE.fullmatch(path.name):
                        object_files += 1
                        object_bytes += int(path.stat().st_size)
            return ProjectCheckpointStats(
                stage_records=stage_records,
                object_records=object_records,
                object_files=object_files,
                object_bytes=object_bytes,
            )
        except FileNotFoundError:
            return ProjectCheckpointStats(0, 0, 0, 0)
        except (OSError, ValueError, sqlite3.Error, ProjectCheckpointError) as exc:
            self._disable(exc)
            return ProjectCheckpointStats(0, 0, 0, 0)

    def has_stage_records(self, stage: str) -> bool:
        if not self.available:
            return False
        try:
            normalized_stage = _validate_stage(stage)
            connection = self._connect(create=False)
            try:
                row = connection.execute(
                    "SELECT 1 FROM stage_records WHERE stage = ? LIMIT 1",
                    (normalized_stage,),
                ).fetchone()
                return row is not None
            finally:
                connection.close()
        except FileNotFoundError:
            return False
        except (OSError, ValueError, sqlite3.Error, ProjectCheckpointError) as exc:
            self._disable(exc)
            return False

    def has_stage_record(self, page_key: str, stage: str) -> bool:
        if not self.available:
            return False
        try:
            normalized_page = _validate_page_key(page_key)
            normalized_stage = _validate_stage(stage)
            connection = self._connect(create=False)
            try:
                row = connection.execute(
                    """
                    SELECT 1
                    FROM stage_records
                    WHERE page_key = ? AND stage = ?
                    LIMIT 1
                    """,
                    (normalized_page, normalized_stage),
                ).fetchone()
                return row is not None
            finally:
                connection.close()
        except FileNotFoundError:
            return False
        except (OSError, ValueError, sqlite3.Error, ProjectCheckpointError) as exc:
            self._disable(exc)
            return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".partial",
        dir=path.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _remove_tree_exact(path: Path, *, expected_parent: Path) -> None:
    if not path.exists():
        return
    parent_real = Path(os.path.realpath(expected_parent))
    path_real = Path(os.path.realpath(path))
    try:
        relative = path_real.relative_to(parent_real)
    except ValueError as exc:
        raise ProjectCheckpointError(
            "Refusing to remove a checkpoint folder outside its project folder."
        ) from exc
    lower_name = path.name.lower()
    is_sidecar = lower_name.endswith(".ctpr.cache")
    is_transient = (
        ".ctpr.cache." in lower_name
        or ".ctpr.cache.backup-" in lower_name
    )
    if len(relative.parts) != 1 or not (is_sidecar or is_transient):
        raise ProjectCheckpointError(
            "Refusing to remove an unexpected checkpoint folder."
        )
    if path.is_symlink():
        raise ProjectCheckpointError(
            "Refusing to remove a symbolic-link checkpoint folder."
        )
    shutil.rmtree(path)


def _copy_checkpoint_db(
    source_db: Path,
    target_db: Path,
    source_reference: ProjectCheckpointReference,
    target_reference: ProjectCheckpointReference,
) -> None:
    if source_db.is_symlink():
        raise ProjectCheckpointError(
            "Symbolic-link checkpoint databases are not supported."
        )
    source_connection = sqlite3.connect(
        source_db,
        timeout=5.0,
        check_same_thread=False,
    )
    target_connection = sqlite3.connect(
        target_db,
        timeout=5.0,
        check_same_thread=False,
    )
    try:
        source_metadata = {
            str(row[0]): str(row[1])
            for row in source_connection.execute(
                """
                SELECT key, value
                FROM metadata
                WHERE key IN ('schema_version', 'project_uuid', 'cache_id')
                """
            ).fetchall()
        }
        if (
            source_metadata.get("schema_version")
            != str(PROJECT_CHECKPOINT_SCHEMA_VERSION)
            or source_metadata.get("project_uuid")
            != source_reference.project_uuid
            or source_metadata.get("cache_id") != source_reference.cache_id
        ):
            raise ProjectCheckpointError(
                "Source checkpoint identity does not match the project reference."
            )
        source_connection.backup(target_connection)
        with target_connection:
            target_connection.execute(
                "UPDATE metadata SET value = ? WHERE key = 'project_uuid'",
                (target_reference.project_uuid,),
            )
            target_connection.execute(
                "UPDATE metadata SET value = ? WHERE key = 'cache_id'",
                (target_reference.cache_id,),
            )
        target_connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        target_connection.close()
        source_connection.close()


def _clone_object_tree(
    source_root: Path,
    target_root: Path,
) -> tuple[int, int]:
    linked_files = 0
    copied_files = 0
    if not source_root.exists():
        return linked_files, copied_files
    if source_root.is_symlink():
        raise ProjectCheckpointError(
            "Symbolic-link checkpoint object roots are not supported."
        )
    for source_path in source_root.glob("*/*"):
        if source_path.is_symlink():
            raise ProjectCheckpointError(
                "Symbolic-link checkpoint objects are not supported."
            )
        if not source_path.is_file() or not _SHA256_RE.fullmatch(source_path.name):
            continue
        if _sha256_file(source_path) != source_path.name:
            raise ProjectCheckpointError(
                "Source checkpoint object failed integrity verification."
            )
        target_path = target_root / source_path.name[:2] / source_path.name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source_path, target_path)
            linked_files += 1
        except OSError:
            shutil.copy2(source_path, target_path)
            copied_files += 1
    return linked_files, copied_files


class PreparedCheckpointSidecar:
    """Installed target sidecar that can be committed or rolled back."""

    def __init__(
        self,
        *,
        target_path: Path,
        backup_path: Path | None,
        installed: bool,
        linked_files: int = 0,
        copied_files: int = 0,
    ) -> None:
        self.target_path = target_path
        self.backup_path = backup_path
        self.installed = installed
        self.linked_files = int(linked_files)
        self.copied_files = int(copied_files)
        self._finished = False

    def commit(self) -> None:
        if self._finished:
            return
        if self.backup_path is not None and self.backup_path.exists():
            _remove_tree_exact(
                self.backup_path,
                expected_parent=self.backup_path.parent,
            )
        self._finished = True

    def rollback(self) -> None:
        if self._finished:
            return
        if self.installed and self.target_path.exists():
            _remove_tree_exact(
                self.target_path,
                expected_parent=self.target_path.parent,
            )
        if self.backup_path is not None and self.backup_path.exists():
            os.replace(self.backup_path, self.target_path)
        self._finished = True


def prepare_checkpoint_sidecar(
    source_project_file: str | os.PathLike[str] | None,
    source_reference: Mapping[str, Any] | ProjectCheckpointReference | None,
    target_project_file: str | os.PathLike[str],
    target_reference: Mapping[str, Any] | ProjectCheckpointReference,
) -> PreparedCheckpointSidecar:
    target_ref = normalize_checkpoint_reference(
        target_reference,
        target_project_file,
        create_if_missing=False,
    )
    if target_ref is None:
        raise ProjectCheckpointError("Target checkpoint reference is missing.")
    target_path = checkpoint_sidecar_path(target_project_file, target_ref)
    source_path: Path | None = None
    source_ref: ProjectCheckpointReference | None = None
    if source_project_file and source_reference is not None:
        source_ref = normalize_checkpoint_reference(
            source_reference,
            source_project_file,
            create_if_missing=False,
        )
        if source_ref is None:
            raise ProjectCheckpointError("Source checkpoint reference is missing.")
        source_path = checkpoint_sidecar_path(
            source_project_file,
            source_ref,
        )
        if os.path.normcase(os.path.abspath(source_path)) == os.path.normcase(
            os.path.abspath(target_path)
        ):
            return PreparedCheckpointSidecar(
                target_path=target_path,
                backup_path=None,
                installed=False,
            )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = target_path.parent / (
        f".{target_path.name}.{uuid.uuid4().hex}.partial"
    )
    backup_path: Path | None = None
    linked_files = 0
    copied_files = 0
    installed = False
    try:
        source_exists = bool(source_path and source_path.is_dir())
        if source_exists:
            assert source_path is not None
            assert source_ref is not None
            if source_path.is_symlink():
                raise ProjectCheckpointError(
                    "Symbolic-link checkpoint sidecars are not supported."
                )
            source_db = source_path / PROJECT_CHECKPOINT_DB_NAME
            if not source_db.is_file():
                raise ProjectCheckpointError(
                    "Source checkpoint database is missing."
                )
            partial_path.mkdir(parents=False, exist_ok=False)
            (partial_path / PROJECT_CHECKPOINT_README_NAME).write_text(
                _README_TEXT,
                encoding="utf-8",
                newline="\n",
            )
            partial_object_root = partial_path.joinpath(
                *PROJECT_CHECKPOINT_OBJECT_ROOT
            )
            partial_object_root.mkdir(parents=True, exist_ok=True)
            _copy_checkpoint_db(
                source_db,
                partial_path / PROJECT_CHECKPOINT_DB_NAME,
                source_ref,
                target_ref,
            )
            linked_files, copied_files = _clone_object_tree(
                source_path.joinpath(*PROJECT_CHECKPOINT_OBJECT_ROOT),
                partial_object_root,
            )

        if target_path.exists():
            if target_path.is_symlink() or not target_path.is_dir():
                raise ProjectCheckpointError(
                    "Target checkpoint path is not a regular folder."
                )
            backup_path = target_path.parent / (
                f"{target_path.name}.backup-{uuid.uuid4().hex}"
            )
            os.replace(target_path, backup_path)

        if source_exists:
            os.replace(partial_path, target_path)
            installed = True

        return PreparedCheckpointSidecar(
            target_path=target_path,
            backup_path=backup_path,
            installed=installed,
            linked_files=linked_files,
            copied_files=copied_files,
        )
    except BaseException:
        if installed and target_path.exists():
            _remove_tree_exact(target_path, expected_parent=target_path.parent)
        if backup_path is not None and backup_path.exists():
            os.replace(backup_path, target_path)
        if partial_path.exists():
            _remove_tree_exact(partial_path, expected_parent=partial_path.parent)
        raise


def remove_checkpoint_sidecar(
    project_file: str | os.PathLike[str],
    reference: Mapping[str, Any] | ProjectCheckpointReference,
) -> bool:
    """Remove exactly one matching sidecar after a confirmed project move."""

    sidecar = checkpoint_sidecar_path(project_file, reference)
    if not sidecar.exists():
        return True
    if not checkpoint_sidecar_matches_identity(project_file, reference):
        return False
    _remove_tree_exact(sidecar, expected_parent=sidecar.parent)
    return True


def checkpoint_sidecar_matches_identity(
    project_file: str | os.PathLike[str],
    reference: Mapping[str, Any] | ProjectCheckpointReference,
) -> bool:
    """Return whether an existing sidecar belongs to exactly this project."""

    normalized = normalize_checkpoint_reference(
        reference,
        project_file,
        create_if_missing=False,
    )
    if normalized is None:
        return False
    sidecar = checkpoint_sidecar_path(project_file, normalized)
    if not sidecar.exists():
        return False
    db_path = sidecar / PROJECT_CHECKPOINT_DB_NAME
    if not db_path.is_file() or db_path.is_symlink():
        return False
    connection = sqlite3.connect(db_path, timeout=0.5)
    try:
        metadata = {
            str(row[0]): str(row[1])
            for row in connection.execute(
                """
                SELECT key, value
                FROM metadata
                WHERE key IN ('schema_version', 'project_uuid', 'cache_id')
                """
            ).fetchall()
        }
    except sqlite3.Error:
        return False
    finally:
        connection.close()
    if (
        metadata.get("schema_version")
        != str(PROJECT_CHECKPOINT_SCHEMA_VERSION)
        or metadata.get("project_uuid") != normalized.project_uuid
        or metadata.get("cache_id") != normalized.cache_id
    ):
        return False
    return True


def finalize_checkpoint_sidecar_move(
    source_project_file: str | os.PathLike[str],
    source_reference: Mapping[str, Any] | ProjectCheckpointReference,
    target_project_file: str | os.PathLike[str],
    target_reference: Mapping[str, Any] | ProjectCheckpointReference,
) -> bool:
    """Remove the source sidecar only after the target copy is verified."""

    source_sidecar = checkpoint_sidecar_path(
        source_project_file,
        source_reference,
    )
    if not source_sidecar.exists():
        return True
    if not checkpoint_sidecar_matches_identity(
        target_project_file,
        target_reference,
    ):
        return False
    return remove_checkpoint_sidecar(
        source_project_file,
        source_reference,
    )
