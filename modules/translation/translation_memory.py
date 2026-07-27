from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import unicodedata

from modules.utils.paths import get_user_data_dir


logger = logging.getLogger(__name__)

TRANSLATION_MEMORY_SCHEMA_VERSION = 1
TRANSLATION_RESULT_CACHE_VERSION = 1
EXACT_TM_NORMALIZATION_VERSION = 1
DEFAULT_RESULT_CACHE_LIMIT = 50_000
DEFAULT_TM_CANDIDATE_LIMIT = 5_000
DEFAULT_TM_DISPLAY_LIMIT = 500
DEFAULT_TRANSLATION_MEMORY_DB_NAME = "translation-memory-v1.sqlite3"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_exact_tm_text(value: str | None) -> str:
    """Apply only the conservative normalization allowed for exact TM."""

    normalized = unicodedata.normalize(
        "NFC",
        str(value or "").replace("\r\n", "\n").replace("\r", "\n"),
    )
    return normalized.strip()


def default_translation_memory_path() -> Path:
    return (
        Path(get_user_data_dir())
        / "translation-memory"
        / DEFAULT_TRANSLATION_MEMORY_DB_NAME
    )


@dataclass(frozen=True)
class ResultCacheLookup:
    translation: str | None = None
    metadata: Mapping[str, Any] | None = None
    stale_reject: bool = False
    disabled: bool = False

    @property
    def hit(self) -> bool:
        return self.translation is not None


@dataclass(frozen=True)
class ExactTMLookup:
    translation: str | None = None
    entry_ids: tuple[int, ...] = ()
    ambiguous: bool = False
    disabled: bool = False

    @property
    def hit(self) -> bool:
        return self.translation is not None and not self.ambiguous


@dataclass(frozen=True)
class ResultCacheRecord:
    cache_key: str
    scope_key: str
    identity_json: str
    source_text: str
    translation: str
    metadata_json: str = "{}"


@dataclass(frozen=True)
class ExactTMCandidate:
    source_text: str
    translation: str
    source_lang: str
    target_lang: str


class TranslationMemoryStore:
    """Thread-safe persistent result cache plus separately approved exact TM."""

    def __init__(
        self,
        db_path: str | os.PathLike[str] | None = None,
        *,
        result_cache_limit: int = DEFAULT_RESULT_CACHE_LIMIT,
        candidate_limit: int = DEFAULT_TM_CANDIDATE_LIMIT,
        timeout_sec: float = 0.5,
    ) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_translation_memory_path()
        self.result_cache_limit = max(1, int(result_cache_limit))
        self.candidate_limit = max(1, int(candidate_limit))
        self.timeout_sec = max(0.0, float(timeout_sec))
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None
        self._disabled_reason = ""

    @property
    def disabled_reason(self) -> str:
        return self._disabled_reason

    @property
    def enabled(self) -> bool:
        return not self._disabled_reason

    def __enter__(self) -> TranslationMemoryStore:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def configure_limits(
        self,
        *,
        result_cache_limit: int | None = None,
        candidate_limit: int | None = None,
    ) -> None:
        with self._lock:
            if result_cache_limit is not None:
                self.result_cache_limit = max(1, int(result_cache_limit))
            if candidate_limit is not None:
                self.candidate_limit = max(1, int(candidate_limit))

    def close(self) -> None:
        with self._lock:
            self._close_connection()

    def _close_connection(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is None:
            return
        try:
            connection.close()
        except sqlite3.Error:
            logger.debug("Failed to close translation memory database.", exc_info=True)

    def _disable(self, exc: BaseException) -> None:
        if self._disabled_reason:
            return
        self._disabled_reason = f"{type(exc).__name__}: {exc}"
        self._close_connection()
        logger.warning(
            "Persistent translation cache disabled for this run; translation will continue "
            "without it. The database was not deleted or rewritten. reason=%s",
            self._disabled_reason,
        )

    def _connect(self) -> sqlite3.Connection:
        if self._disabled_reason:
            raise RuntimeError(self._disabled_reason)
        if self._connection is not None:
            return self._connection

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.db_path,
            timeout=self.timeout_sec,
            check_same_thread=False,
        )
        try:
            connection.row_factory = sqlite3.Row
            connection.execute(f"PRAGMA busy_timeout = {int(self.timeout_sec * 1000)}")
            connection.execute("PRAGMA foreign_keys = ON")
            self._initialize_schema(connection)
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
        except BaseException:
            connection.close()
            raise
        self._connection = connection
        return connection

    @staticmethod
    def _initialize_schema(connection: sqlite3.Connection) -> None:
        existing_tables = {
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if existing_tables:
            if "metadata" not in existing_tables:
                raise sqlite3.DatabaseError(
                    "Existing translation memory database has no schema metadata."
                )
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            try:
                schema_version = int(row["value"]) if row is not None else 0
            except (TypeError, ValueError) as exc:
                raise sqlite3.DatabaseError(
                    "Invalid translation memory schema version."
                ) from exc
            if schema_version != TRANSLATION_MEMORY_SCHEMA_VERSION:
                raise sqlite3.DatabaseError(
                    "Unsupported translation memory schema version."
                )

        with connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS result_cache (
                    cache_key TEXT PRIMARY KEY,
                    scope_key TEXT NOT NULL,
                    identity_json TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    translation TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    last_used_at REAL NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_result_cache_scope
                    ON result_cache(scope_key);
                CREATE INDEX IF NOT EXISTS idx_result_cache_last_used
                    ON result_cache(last_used_at);

                CREATE TABLE IF NOT EXISTS exact_tm (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_normalized TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    translation TEXT NOT NULL,
                    source_lang TEXT NOT NULL,
                    target_lang TEXT NOT NULL,
                    approved INTEGER NOT NULL DEFAULT 0 CHECK (approved IN (0, 1)),
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_used_at REAL NOT NULL,
                    use_count INTEGER NOT NULL DEFAULT 0,
                    UNIQUE (
                        source_normalized,
                        translation,
                        source_lang,
                        target_lang
                    )
                );

                CREATE INDEX IF NOT EXISTS idx_exact_tm_lookup
                    ON exact_tm(source_normalized, source_lang, target_lang, approved);
                CREATE INDEX IF NOT EXISTS idx_exact_tm_candidate_retention
                    ON exact_tm(approved, last_used_at);
                """
            )
            connection.execute(
                """
                INSERT INTO metadata(key, value)
                VALUES('schema_version', ?)
                ON CONFLICT(key) DO NOTHING
                """,
                (str(TRANSLATION_MEMORY_SCHEMA_VERSION),),
            )
            connection.execute(
                """
                INSERT INTO metadata(key, value)
                VALUES('tm_revision', '0')
                ON CONFLICT(key) DO NOTHING
                """
            )


    def _connection_or_disable(self) -> sqlite3.Connection | None:
        if self._disabled_reason:
            return None
        try:
            return self._connect()
        except (OSError, RuntimeError, sqlite3.Error) as exc:
            self._disable(exc)
            return None

    def _management_unavailable_error(self) -> RuntimeError:
        reason = self._disabled_reason or "translation memory database is unavailable"
        return RuntimeError(
            "Translation memory operation could not be completed. "
            f"The database was left unchanged. Reason: {reason}"
        )

    def get_tm_revision(self) -> int:
        with self._lock:
            connection = self._connection_or_disable()
            if connection is None:
                return 0
            try:
                row = connection.execute(
                    "SELECT value FROM metadata WHERE key = 'tm_revision'"
                ).fetchone()
                return int(row["value"]) if row is not None else 0
            except (TypeError, ValueError, sqlite3.Error) as exc:
                self._disable(exc)
                return 0

    @staticmethod
    def _increment_tm_revision(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            UPDATE metadata
            SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)
            WHERE key = 'tm_revision'
            """
        )

    def lookup_result(self, cache_key: str, scope_key: str) -> ResultCacheLookup:
        with self._lock:
            connection = self._connection_or_disable()
            if connection is None:
                return ResultCacheLookup(disabled=True)
            try:
                row = connection.execute(
                    """
                    SELECT translation, metadata_json
                    FROM result_cache
                    WHERE cache_key = ?
                    """,
                    (cache_key,),
                ).fetchone()
                if row is not None:
                    try:
                        metadata = json.loads(row["metadata_json"] or "{}")
                    except (TypeError, ValueError, json.JSONDecodeError):
                        metadata = {}
                    if not isinstance(metadata, dict):
                        metadata = {}
                    return ResultCacheLookup(
                        translation=str(row["translation"]),
                        metadata=metadata,
                    )

                stale_row = connection.execute(
                    "SELECT 1 FROM result_cache WHERE scope_key = ? LIMIT 1",
                    (scope_key,),
                ).fetchone()
                return ResultCacheLookup(stale_reject=stale_row is not None)
            except sqlite3.Error as exc:
                self._disable(exc)
                return ResultCacheLookup(disabled=True)

    def store_results(
        self,
        records: Iterable[ResultCacheRecord],
        *,
        touched_cache_keys: Iterable[str] = (),
        touched_tm_entry_ids: Iterable[int] = (),
    ) -> bool:
        records = tuple(records)
        touched_keys = tuple(
            sorted(
                {
                    str(cache_key)
                    for cache_key in touched_cache_keys
                    if str(cache_key)
                }
            )
        )
        touched_tm_ids = tuple(
            sorted({int(entry_id) for entry_id in touched_tm_entry_ids})
        )
        if not records and not touched_keys and not touched_tm_ids:
            return True
        with self._lock:
            connection = self._connection_or_disable()
            if connection is None:
                return False
            now = time.time()
            try:
                with connection:
                    connection.executemany(
                        """
                        INSERT INTO result_cache(
                            cache_key,
                            scope_key,
                            identity_json,
                            source_text,
                            translation,
                            metadata_json,
                            created_at,
                            last_used_at,
                            hit_count
                        )
                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, 0)
                        ON CONFLICT(cache_key) DO UPDATE SET
                            scope_key = excluded.scope_key,
                            identity_json = excluded.identity_json,
                            source_text = excluded.source_text,
                            translation = excluded.translation,
                            metadata_json = excluded.metadata_json,
                            last_used_at = excluded.last_used_at
                        """,
                        [
                            (
                                record.cache_key,
                                record.scope_key,
                                record.identity_json,
                                record.source_text,
                                record.translation,
                                record.metadata_json,
                                now,
                                now,
                            )
                            for record in records
                        ],
                    )
                    if touched_keys:
                        connection.executemany(
                            """
                            UPDATE result_cache
                            SET last_used_at = ?, hit_count = hit_count + 1
                            WHERE cache_key = ?
                            """,
                            [(now, cache_key) for cache_key in touched_keys],
                        )
                    if touched_tm_ids:
                        connection.executemany(
                            """
                            UPDATE exact_tm
                            SET last_used_at = ?, use_count = use_count + 1
                            WHERE id = ?
                            """,
                            [(now, entry_id) for entry_id in touched_tm_ids],
                        )
                    if records:
                        self._prune_result_cache(connection)
                return True
            except sqlite3.Error as exc:
                self._disable(exc)
                return False

    def _prune_result_cache(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            DELETE FROM result_cache
            WHERE cache_key IN (
                SELECT cache_key
                FROM result_cache
                ORDER BY last_used_at DESC, created_at DESC
                LIMIT -1 OFFSET ?
            )
            """,
            (self.result_cache_limit,),
        )

    def lookup_exact_tm(
        self,
        source_text: str,
        source_lang: str,
        target_lang: str,
    ) -> ExactTMLookup:
        normalized = normalize_exact_tm_text(source_text)
        if not normalized:
            return ExactTMLookup()
        with self._lock:
            connection = self._connection_or_disable()
            if connection is None:
                return ExactTMLookup(disabled=True)
            try:
                rows = connection.execute(
                    """
                    SELECT id, translation
                    FROM exact_tm
                    WHERE source_normalized = ?
                      AND source_lang = ?
                      AND target_lang = ?
                      AND approved = 1
                    ORDER BY updated_at DESC, id DESC
                    """,
                    (normalized, source_lang, target_lang),
                ).fetchall()
                translations = {str(row["translation"]) for row in rows}
                if len(translations) != 1:
                    return ExactTMLookup(ambiguous=len(translations) > 1)
                translation = next(iter(translations))
                return ExactTMLookup(
                    translation=translation,
                    entry_ids=tuple(int(row["id"]) for row in rows),
                )
            except sqlite3.Error as exc:
                self._disable(exc)
                return ExactTMLookup(disabled=True)

    def record_tm_candidate(
        self,
        source_text: str,
        translation: str,
        source_lang: str,
        target_lang: str,
    ) -> bool:
        return bool(
            self.record_tm_candidates(
                [
                    ExactTMCandidate(
                        source_text=source_text,
                        translation=translation,
                        source_lang=source_lang,
                        target_lang=target_lang,
                    )
                ]
            )
        )

    def record_tm_candidates(
        self,
        candidates: Iterable[ExactTMCandidate],
    ) -> int:
        normalized_candidates: dict[
            tuple[str, str, str, str],
            tuple[str, str, str, str, str],
        ] = {}
        for candidate in candidates:
            source_text = str(candidate.source_text or "")
            translation = str(candidate.translation or "")
            source_lang = str(candidate.source_lang or "").strip()
            target_lang = str(candidate.target_lang or "").strip()
            normalized = normalize_exact_tm_text(source_text)
            if (
                not normalized
                or not translation.strip()
                or not source_lang
                or not target_lang
            ):
                continue
            unique_key = (normalized, translation, source_lang, target_lang)
            normalized_candidates[unique_key] = (
                normalized,
                source_text,
                translation,
                source_lang,
                target_lang,
            )
        if not normalized_candidates:
            return 0

        with self._lock:
            connection = self._connection_or_disable()
            if connection is None:
                return 0
            now = time.time()
            try:
                with connection:
                    connection.executemany(
                        """
                        INSERT INTO exact_tm(
                            source_normalized,
                            source_text,
                            translation,
                            source_lang,
                            target_lang,
                            approved,
                            created_at,
                            updated_at,
                            last_used_at,
                            use_count
                        )
                        VALUES(?, ?, ?, ?, ?, 0, ?, ?, ?, 0)
                        ON CONFLICT(
                            source_normalized,
                            translation,
                            source_lang,
                            target_lang
                        ) DO UPDATE SET
                            source_text = excluded.source_text,
                            updated_at = excluded.updated_at,
                            last_used_at = excluded.last_used_at
                        """,
                        [
                            (
                                normalized,
                                source_text,
                                translation,
                                source_lang,
                                target_lang,
                                now,
                                now,
                                now,
                            )
                            for (
                                normalized,
                                source_text,
                                translation,
                                source_lang,
                                target_lang,
                            ) in normalized_candidates.values()
                        ],
                    )
                    self._prune_candidates(connection)
                return len(normalized_candidates)
            except sqlite3.Error as exc:
                self._disable(exc)
                return 0

    def _prune_candidates(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            DELETE FROM exact_tm
            WHERE id IN (
                SELECT id
                FROM exact_tm
                WHERE approved = 0
                ORDER BY last_used_at DESC, updated_at DESC, id DESC
                LIMIT -1 OFFSET ?
            )
            """,
            (self.candidate_limit,),
        )

    def list_tm_entries(
        self,
        *,
        limit: int = DEFAULT_TM_DISPLAY_LIMIT,
    ) -> list[dict[str, Any]]:
        with self._lock:
            connection = self._connection_or_disable()
            if connection is None:
                return []
            try:
                rows = connection.execute(
                    """
                    SELECT
                        id,
                        source_text,
                        translation,
                        source_lang,
                        target_lang,
                        approved,
                        created_at,
                        updated_at,
                        last_used_at,
                        use_count
                    FROM exact_tm
                    ORDER BY updated_at DESC, id DESC
                    LIMIT ?
                    """,
                    (max(1, int(limit)),),
                ).fetchall()
                return [
                    {
                        "id": int(row["id"]),
                        "source_text": str(row["source_text"]),
                        "translation": str(row["translation"]),
                        "source_lang": str(row["source_lang"]),
                        "target_lang": str(row["target_lang"]),
                        "approved": bool(row["approved"]),
                        "created_at": float(row["created_at"]),
                        "updated_at": float(row["updated_at"]),
                        "last_used_at": float(row["last_used_at"]),
                        "use_count": int(row["use_count"]),
                    }
                    for row in rows
                ]
            except sqlite3.Error as exc:
                self._disable(exc)
                return []

    def set_approved(self, entry_ids: Iterable[int], approved: bool) -> int:
        ids = sorted({int(entry_id) for entry_id in entry_ids})
        if not ids:
            return 0
        with self._lock:
            connection = self._connection_or_disable()
            if connection is None:
                raise self._management_unavailable_error()
            now = time.time()
            placeholders = ",".join("?" for _ in ids)
            try:
                with connection:
                    cursor = connection.execute(
                        f"""
                        UPDATE exact_tm
                        SET approved = ?, updated_at = ?
                        WHERE id IN ({placeholders})
                          AND approved <> ?
                        """,
                        (
                            int(bool(approved)),
                            now,
                            *ids,
                            int(bool(approved)),
                        ),
                    )
                    changed = max(0, int(cursor.rowcount))
                    if changed:
                        self._increment_tm_revision(connection)
                return changed
            except sqlite3.Error as exc:
                self._disable(exc)
                raise self._management_unavailable_error() from exc

    def delete_tm_entries(self, entry_ids: Iterable[int]) -> int:
        ids = sorted({int(entry_id) for entry_id in entry_ids})
        if not ids:
            return 0
        with self._lock:
            connection = self._connection_or_disable()
            if connection is None:
                raise self._management_unavailable_error()
            placeholders = ",".join("?" for _ in ids)
            try:
                with connection:
                    cursor = connection.execute(
                        f"DELETE FROM exact_tm WHERE id IN ({placeholders})",
                        ids,
                    )
                    changed = max(0, int(cursor.rowcount))
                    if changed:
                        self._increment_tm_revision(connection)
                return changed
            except sqlite3.Error as exc:
                self._disable(exc)
                raise self._management_unavailable_error() from exc

    def clear_result_cache(self) -> int:
        with self._lock:
            connection = self._connection_or_disable()
            if connection is None:
                raise self._management_unavailable_error()
            try:
                with connection:
                    row = connection.execute(
                        "SELECT COUNT(*) AS count FROM result_cache"
                    ).fetchone()
                    count = int(row["count"]) if row is not None else 0
                    connection.execute("DELETE FROM result_cache")
                return count
            except sqlite3.Error as exc:
                self._disable(exc)
                raise self._management_unavailable_error() from exc

    def stats(self) -> dict[str, Any]:
        with self._lock:
            connection = self._connection_or_disable()
            if connection is None:
                return {
                    "disabled": True,
                    "disabled_reason": self._disabled_reason,
                    "result_cache_entries": 0,
                    "approved_tm_entries": 0,
                    "candidate_tm_entries": 0,
                    "tm_revision": 0,
                }
            try:
                cache_row = connection.execute(
                    "SELECT COUNT(*) AS count FROM result_cache"
                ).fetchone()
                tm_rows = connection.execute(
                    """
                    SELECT approved, COUNT(*) AS count
                    FROM exact_tm
                    GROUP BY approved
                    """
                ).fetchall()
                counts = {bool(row["approved"]): int(row["count"]) for row in tm_rows}
                tm_revision = self.get_tm_revision()
                if not self.enabled:
                    return {
                        "disabled": True,
                        "disabled_reason": self._disabled_reason,
                        "result_cache_entries": 0,
                        "approved_tm_entries": 0,
                        "candidate_tm_entries": 0,
                        "tm_revision": 0,
                    }
                return {
                    "disabled": False,
                    "disabled_reason": "",
                    "result_cache_entries": int(cache_row["count"]) if cache_row else 0,
                    "approved_tm_entries": counts.get(True, 0),
                    "candidate_tm_entries": counts.get(False, 0),
                    "tm_revision": tm_revision,
                }
            except sqlite3.Error as exc:
                self._disable(exc)
                return {
                    "disabled": True,
                    "disabled_reason": self._disabled_reason,
                    "result_cache_entries": 0,
                    "approved_tm_entries": 0,
                    "candidate_tm_entries": 0,
                    "tm_revision": 0,
                }

    def export_tm(self, destination: str | os.PathLike[str]) -> int:
        entries = self.list_tm_entries(limit=2_000_000_000)
        if not self.enabled:
            raise self._management_unavailable_error()
        payload = {
            "format": "comic-translate-exact-tm",
            "schema_version": TRANSLATION_MEMORY_SCHEMA_VERSION,
            "normalization_version": EXACT_TM_NORMALIZATION_VERSION,
            "entries": [
                {
                    "source_text": entry["source_text"],
                    "translation": entry["translation"],
                    "source_lang": entry["source_lang"],
                    "target_lang": entry["target_lang"],
                    "approved": entry["approved"],
                }
                for entry in entries
            ],
        }
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination_path.with_name(f".{destination_path.name}.partial")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, destination_path)
        return len(entries)

    def import_tm(self, source: str | os.PathLike[str]) -> int:
        payload = json.loads(Path(source).read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict) or payload.get("format") != "comic-translate-exact-tm":
            raise ValueError("Unsupported exact translation memory file.")
        schema_version = payload.get("schema_version")
        if (
            not isinstance(schema_version, int)
            or isinstance(schema_version, bool)
            or schema_version != TRANSLATION_MEMORY_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported exact translation memory schema version.")
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise ValueError("Exact translation memory entries must be a list.")

        normalized_entries: list[tuple[str, str, str, str, str, int]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Exact translation memory entries must be objects.")
            raw_fields = {
                "source_text": entry.get("source_text"),
                "translation": entry.get("translation"),
                "source_lang": entry.get("source_lang"),
                "target_lang": entry.get("target_lang"),
            }
            if any(not isinstance(value, str) for value in raw_fields.values()):
                raise ValueError(
                    "Exact translation memory text and language values must be strings."
                )
            source_text = raw_fields["source_text"]
            translation = raw_fields["translation"]
            source_lang = raw_fields["source_lang"].strip()
            target_lang = raw_fields["target_lang"].strip()
            source_normalized = normalize_exact_tm_text(source_text)
            if not source_normalized or not translation.strip() or not source_lang or not target_lang:
                raise ValueError("Exact translation memory entries contain empty required fields.")
            approved_value = entry.get("approved", False)
            if not isinstance(approved_value, bool):
                raise ValueError(
                    "Exact translation memory approved values must be JSON booleans."
                )
            normalized_entries.append(
                (
                    source_normalized,
                    source_text,
                    translation,
                    source_lang,
                    target_lang,
                    int(approved_value),
                )
            )

        with self._lock:
            connection = self._connection_or_disable()
            if connection is None:
                raise self._management_unavailable_error()
            now = time.time()
            try:
                with connection:
                    connection.executemany(
                        """
                        INSERT INTO exact_tm(
                            source_normalized,
                            source_text,
                            translation,
                            source_lang,
                            target_lang,
                            approved,
                            created_at,
                            updated_at,
                            last_used_at,
                            use_count
                        )
                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                        ON CONFLICT(
                            source_normalized,
                            translation,
                            source_lang,
                            target_lang
                        ) DO UPDATE SET
                            source_text = excluded.source_text,
                            approved = excluded.approved,
                            updated_at = excluded.updated_at,
                            last_used_at = excluded.last_used_at
                        """,
                        [
                            (
                                source_normalized,
                                source_text,
                                translation,
                                source_lang,
                                target_lang,
                                approved,
                                now,
                                now,
                                now,
                            )
                            for (
                                source_normalized,
                                source_text,
                                translation,
                                source_lang,
                                target_lang,
                                approved,
                            ) in normalized_entries
                        ],
                    )
                    if normalized_entries:
                        self._increment_tm_revision(connection)
                    self._prune_candidates(connection)
                return len(normalized_entries)
            except sqlite3.Error as exc:
                self._disable(exc)
                raise self._management_unavailable_error() from exc
