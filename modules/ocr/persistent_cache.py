from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import copy
import hashlib
import json
import logging
import math
import os
import sqlite3
import threading
import time

from modules.utils.paths import get_user_data_dir


logger = logging.getLogger(__name__)

OCR_RESULT_CACHE_SCHEMA_VERSION = 1
OCR_RESULT_CACHE_VERSION = 1
DEFAULT_OCR_RESULT_CACHE_LIMIT = 50_000
DEFAULT_OCR_RESULT_CACHE_DB_NAME = "paddleocr-vl-results-v1.sqlite3"


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


def default_ocr_result_cache_path() -> Path:
    return (
        Path(get_user_data_dir())
        / "paddleocr-vl-cache"
        / DEFAULT_OCR_RESULT_CACHE_DB_NAME
    )


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
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except Exception:
            pass
    return str(value)


def _validate_optional_bbox(value: Any, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{field_name} must be null or a four-item list.")
    for coordinate in value:
        try:
            numeric = float(coordinate)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} contains a non-numeric value.") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"{field_name} contains a non-finite value.")


def validate_raw_ocr_result(payload: Mapping[str, Any]) -> None:
    string_fields = (
        "text",
        "status",
        "empty_reason",
        "raw_text",
        "sanitized_text",
        "reject_reason",
        "ocr_crop_source",
    )
    for field_name in string_fields:
        if field_name in payload and not isinstance(payload[field_name], str):
            raise ValueError(f"{field_name} must be a string.")
    for field_name in ("texts", "ocr_regions"):
        if field_name in payload and not isinstance(payload[field_name], list):
            raise ValueError(f"{field_name} must be a list.")
    for field_name in ("confidence", "ocr_resize_scale"):
        if field_name not in payload:
            continue
        try:
            numeric = float(payload[field_name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be numeric.") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"{field_name} must be finite.")
    if "attempt_count" in payload:
        try:
            attempt_count = int(payload["attempt_count"])
        except (TypeError, ValueError) as exc:
            raise ValueError("attempt_count must be an integer.") from exc
        if attempt_count < 0:
            raise ValueError("attempt_count must not be negative.")
    for field_name in (
        "ocr_crop_bbox",
        "ocr_effective_crop_xyxy",
        "ocr_retry_crop_xyxy",
    ):
        if field_name in payload:
            _validate_optional_bbox(payload[field_name], field_name)


def snapshot_raw_ocr_result(block: Any) -> dict[str, Any]:
    """Capture the OCR result before the user result dictionary is applied."""

    return {
        "text": str(getattr(block, "text", "") or ""),
        "texts": _json_safe(copy.deepcopy(getattr(block, "texts", []) or [])),
        "confidence": float(getattr(block, "ocr_confidence", 0.0) or 0.0),
        "status": str(getattr(block, "ocr_status", "") or ""),
        "empty_reason": str(getattr(block, "ocr_empty_reason", "") or ""),
        "attempt_count": int(getattr(block, "ocr_attempt_count", 0) or 0),
        "raw_text": str(
            getattr(block, "ocr_raw_text", getattr(block, "text", "")) or ""
        ),
        "sanitized_text": str(
            getattr(block, "ocr_sanitized_text", getattr(block, "text", "")) or ""
        ),
        "reject_reason": str(getattr(block, "ocr_reject_reason", "") or ""),
        "ocr_regions": _json_safe(
            copy.deepcopy(getattr(block, "ocr_regions", []) or [])
        ),
        "ocr_crop_bbox": _json_safe(
            copy.deepcopy(getattr(block, "ocr_crop_bbox", None))
        ),
        "ocr_resize_scale": float(
            getattr(block, "ocr_resize_scale", 1.0) or 1.0
        ),
        "ocr_effective_crop_xyxy": _json_safe(
            copy.deepcopy(getattr(block, "ocr_effective_crop_xyxy", None))
        ),
        "ocr_retry_crop_xyxy": _json_safe(
            copy.deepcopy(getattr(block, "ocr_retry_crop_xyxy", None))
        ),
        "ocr_crop_source": str(getattr(block, "ocr_crop_source", "") or ""),
    }


def apply_raw_ocr_result(block: Any, payload: Mapping[str, Any]) -> None:
    """Restore a cached raw result without applying any user dictionary."""

    block.text = str(payload.get("text", "") or "")
    texts = payload.get("texts", [])
    block.texts = copy.deepcopy(texts) if isinstance(texts, list) else []
    block.ocr_confidence = float(payload.get("confidence", 0.0) or 0.0)
    block.ocr_status = str(payload.get("status", "") or "")
    block.ocr_empty_reason = str(payload.get("empty_reason", "") or "")
    block.ocr_attempt_count = int(payload.get("attempt_count", 0) or 0)
    block.ocr_raw_text = str(payload.get("raw_text", "") or "")
    block.ocr_sanitized_text = str(payload.get("sanitized_text", "") or "")
    block.ocr_reject_reason = str(payload.get("reject_reason", "") or "")
    regions = payload.get("ocr_regions", [])
    block.ocr_regions = copy.deepcopy(regions) if isinstance(regions, list) else []
    block.ocr_crop_bbox = copy.deepcopy(payload.get("ocr_crop_bbox"))
    block.ocr_resize_scale = float(payload.get("ocr_resize_scale", 1.0) or 1.0)
    block.ocr_effective_crop_xyxy = copy.deepcopy(
        payload.get("ocr_effective_crop_xyxy")
    )
    block.ocr_retry_crop_xyxy = copy.deepcopy(payload.get("ocr_retry_crop_xyxy"))
    block.ocr_crop_source = str(payload.get("ocr_crop_source", "") or "")


@dataclass(frozen=True)
class OCRResultCacheLookup:
    result: Mapping[str, Any] | None = None
    disabled: bool = False

    @property
    def hit(self) -> bool:
        return self.result is not None


@dataclass(frozen=True)
class OCRResultCacheRecord:
    cache_key: str
    identity_json: str
    result_json: str


class OCRPersistentResultCache:
    """Thread-safe exact PaddleOCR-VL result cache with fail-open behavior."""

    def __init__(
        self,
        db_path: str | os.PathLike[str] | None = None,
        *,
        result_cache_limit: int = DEFAULT_OCR_RESULT_CACHE_LIMIT,
        timeout_sec: float = 0.5,
    ) -> None:
        self.db_path = (
            Path(db_path) if db_path is not None else default_ocr_result_cache_path()
        )
        self.result_cache_limit = max(1, int(result_cache_limit))
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

    def __enter__(self) -> OCRPersistentResultCache:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def configure_limit(self, result_cache_limit: int) -> None:
        with self._lock:
            self.result_cache_limit = max(1, int(result_cache_limit))
            connection = self._connection_or_disable()
            if connection is None:
                return
            try:
                with connection:
                    self._prune(connection)
            except sqlite3.Error as exc:
                self._disable(exc)

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
            logger.debug("Failed to close PaddleOCR-VL cache database.", exc_info=True)

    def _disable(self, exc: BaseException) -> None:
        if self._disabled_reason:
            return
        self._disabled_reason = f"{type(exc).__name__}: {exc}"
        self._close_connection()
        logger.warning(
            "Persistent PaddleOCR-VL result cache disabled for this run; OCR will "
            "continue without it. The database was not deleted or rewritten. reason=%s",
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
            self._initialize_schema(connection)
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            with connection:
                self._prune(connection)
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
            required_tables = {"metadata", "ocr_results"}
            if not required_tables.issubset(existing_tables):
                raise sqlite3.DatabaseError(
                    "Existing PaddleOCR-VL cache is missing required schema tables."
                )
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            try:
                schema_version = int(row["value"]) if row is not None else 0
            except (TypeError, ValueError) as exc:
                raise sqlite3.DatabaseError(
                    "Invalid PaddleOCR-VL cache schema version."
                ) from exc
            if schema_version != OCR_RESULT_CACHE_SCHEMA_VERSION:
                raise sqlite3.DatabaseError(
                    "Unsupported PaddleOCR-VL cache schema version."
                )

        with connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ocr_results (
                    cache_key TEXT PRIMARY KEY,
                    identity_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_used_at REAL NOT NULL,
                    access_order INTEGER NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_ocr_results_last_used
                    ON ocr_results(access_order, last_used_at);
                """
            )
            for key, value in (
                ("schema_version", str(OCR_RESULT_CACHE_SCHEMA_VERSION)),
                ("lookup_hits", "0"),
                ("lookup_misses", "0"),
                ("access_sequence", "0"),
            ):
                connection.execute(
                    """
                    INSERT INTO metadata(key, value)
                    VALUES(?, ?)
                    ON CONFLICT(key) DO NOTHING
                    """,
                    (key, value),
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
        reason = self._disabled_reason or "PaddleOCR-VL cache database is unavailable"
        return RuntimeError(
            "PaddleOCR-VL cache operation could not be completed. "
            f"The database was left unchanged. Reason: {reason}"
        )

    @staticmethod
    def _increment_metadata(
        connection: sqlite3.Connection,
        key: str,
        amount: int,
    ) -> None:
        connection.execute(
            """
            UPDATE metadata
            SET value = CAST(CAST(value AS INTEGER) + ? AS TEXT)
            WHERE key = ?
            """,
            (int(amount), key),
        )

    @staticmethod
    def _next_access_order(connection: sqlite3.Connection) -> int:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'access_sequence'"
        ).fetchone()
        try:
            access_order = int(row["value"]) + 1 if row is not None else 1
        except (TypeError, ValueError) as exc:
            raise sqlite3.DatabaseError(
                "Invalid PaddleOCR-VL cache access sequence."
            ) from exc
        connection.execute(
            """
            INSERT INTO metadata(key, value)
            VALUES('access_sequence', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (str(access_order),),
        )
        return access_order

    def lookup_many(
        self,
        cache_keys: Iterable[str],
    ) -> dict[str, OCRResultCacheLookup]:
        ordered_keys = tuple(
            dict.fromkeys(str(key) for key in cache_keys if str(key))
        )
        if not ordered_keys:
            return {}
        with self._lock:
            connection = self._connection_or_disable()
            if connection is None:
                return {
                    key: OCRResultCacheLookup(disabled=True)
                    for key in ordered_keys
                }
            try:
                rows_by_key: dict[str, sqlite3.Row] = {}
                for start in range(0, len(ordered_keys), 500):
                    chunk = ordered_keys[start : start + 500]
                    placeholders = ",".join("?" for _ in chunk)
                    rows = connection.execute(
                        f"""
                        SELECT cache_key, result_json
                        FROM ocr_results
                        WHERE cache_key IN ({placeholders})
                        """,
                        chunk,
                    ).fetchall()
                    rows_by_key.update(
                        {str(row["cache_key"]): row for row in rows}
                    )

                now = time.time()
                hit_keys = tuple(
                    key for key in ordered_keys if key in rows_by_key
                )
                miss_count = len(ordered_keys) - len(hit_keys)
                parsed_results: dict[str, dict[str, Any]] = {}
                for key in hit_keys:
                    row = rows_by_key[key]
                    try:
                        result = json.loads(str(row["result_json"] or "{}"))
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise sqlite3.DatabaseError(
                            "Invalid cached PaddleOCR-VL result JSON."
                        ) from exc
                    if not isinstance(result, dict):
                        raise sqlite3.DatabaseError(
                            "Cached PaddleOCR-VL result must be an object."
                        )
                    try:
                        validate_raw_ocr_result(result)
                    except ValueError as exc:
                        raise sqlite3.DatabaseError(
                            "Cached PaddleOCR-VL result has an invalid field shape."
                        ) from exc
                    parsed_results[key] = result

                with connection:
                    access_order = self._next_access_order(connection)
                    if hit_keys:
                        connection.executemany(
                            """
                            UPDATE ocr_results
                            SET last_used_at = ?,
                                access_order = ?,
                                hit_count = hit_count + 1
                            WHERE cache_key = ?
                            """,
                            [(now, access_order, key) for key in hit_keys],
                        )
                    self._increment_metadata(
                        connection,
                        "lookup_hits",
                        len(hit_keys),
                    )
                    self._increment_metadata(
                        connection,
                        "lookup_misses",
                        miss_count,
                    )

                lookups: dict[str, OCRResultCacheLookup] = {}
                for key in ordered_keys:
                    result = parsed_results.get(key)
                    if result is None:
                        lookups[key] = OCRResultCacheLookup()
                        continue
                    lookups[key] = OCRResultCacheLookup(result=result)
                return lookups
            except sqlite3.Error as exc:
                self._disable(exc)
                return {
                    key: OCRResultCacheLookup(disabled=True)
                    for key in ordered_keys
                }

    def store_records(self, records: Iterable[OCRResultCacheRecord]) -> bool:
        records_by_key: dict[str, OCRResultCacheRecord] = {}
        try:
            for record in records:
                cache_key = str(record.cache_key)
                if not cache_key:
                    continue
                identity = json.loads(record.identity_json)
                result = json.loads(record.result_json)
                if not isinstance(identity, dict):
                    raise ValueError("PaddleOCR-VL cache identity must be an object.")
                if not isinstance(result, dict):
                    raise ValueError("PaddleOCR-VL cache result must be an object.")
                validate_raw_ocr_result(result)
                records_by_key[cache_key] = record
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "Refusing to store an invalid PaddleOCR-VL cache record: %s",
                exc,
            )
            return False
        if not records_by_key:
            return True
        with self._lock:
            connection = self._connection_or_disable()
            if connection is None:
                return False
            now = time.time()
            try:
                with connection:
                    access_order = self._next_access_order(connection)
                    connection.executemany(
                        """
                        INSERT INTO ocr_results(
                            cache_key,
                            identity_json,
                            result_json,
                            created_at,
                            last_used_at,
                            access_order,
                            hit_count
                        )
                        VALUES(?, ?, ?, ?, ?, ?, 0)
                        ON CONFLICT(cache_key) DO UPDATE SET
                            identity_json = excluded.identity_json,
                            result_json = excluded.result_json,
                            last_used_at = excluded.last_used_at,
                            access_order = excluded.access_order
                        """,
                        [
                            (
                                record.cache_key,
                                record.identity_json,
                                record.result_json,
                                now,
                                now,
                                access_order,
                            )
                            for record in records_by_key.values()
                        ],
                    )
                    self._prune(connection)
                return True
            except sqlite3.Error as exc:
                self._disable(exc)
                return False

    def _prune(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            DELETE FROM ocr_results
            WHERE cache_key IN (
                SELECT cache_key
                FROM ocr_results
                ORDER BY access_order DESC, last_used_at DESC, created_at DESC
                LIMIT -1 OFFSET ?
            )
            """,
            (self.result_cache_limit,),
        )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            connection = self._connection_or_disable()
            if connection is None:
                return {
                    "enabled": False,
                    "disabled_reason": self._disabled_reason,
                    "item_count": 0,
                    "lookup_hits": 0,
                    "lookup_misses": 0,
                }
            try:
                item_count = int(
                    connection.execute(
                        "SELECT COUNT(*) AS count FROM ocr_results"
                    ).fetchone()["count"]
                )
                metadata = {
                    str(row["key"]): str(row["value"])
                    for row in connection.execute(
                        """
                        SELECT key, value
                        FROM metadata
                        WHERE key IN ('lookup_hits', 'lookup_misses')
                        """
                    ).fetchall()
                }
                return {
                    "enabled": True,
                    "disabled_reason": "",
                    "item_count": item_count,
                    "lookup_hits": int(metadata.get("lookup_hits", "0")),
                    "lookup_misses": int(metadata.get("lookup_misses", "0")),
                }
            except (TypeError, ValueError, sqlite3.Error) as exc:
                self._disable(exc)
                return {
                    "enabled": False,
                    "disabled_reason": self._disabled_reason,
                    "item_count": 0,
                    "lookup_hits": 0,
                    "lookup_misses": 0,
                }

    def clear(self) -> None:
        with self._lock:
            connection = self._connection_or_disable()
            if connection is None:
                raise self._management_unavailable_error()
            try:
                with connection:
                    connection.execute("DELETE FROM ocr_results")
                    connection.execute(
                        """
                        UPDATE metadata
                        SET value = '0'
                        WHERE key IN (
                            'lookup_hits',
                            'lookup_misses',
                            'access_sequence'
                        )
                        """
                    )
            except sqlite3.Error as exc:
                self._disable(exc)
                raise self._management_unavailable_error() from exc

    def export_jsonl(self, output_path: str | os.PathLike[str]) -> int:
        output = Path(output_path)
        partial = output.with_name(output.name + ".partial")
        with self._lock:
            connection = self._connection_or_disable()
            if connection is None:
                raise self._management_unavailable_error()
            try:
                rows = connection.execute(
                    """
                    SELECT cache_key, identity_json, result_json,
                           created_at, last_used_at, access_order, hit_count
                    FROM ocr_results
                    ORDER BY created_at, cache_key
                    """
                ).fetchall()
                output.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with partial.open("w", encoding="utf-8", newline="\n") as handle:
                        for row in rows:
                            identity = json.loads(
                                str(row["identity_json"] or "{}")
                            )
                            raw_ocr_result = json.loads(
                                str(row["result_json"] or "{}")
                            )
                            if not isinstance(identity, dict):
                                raise ValueError(
                                    "Cached PaddleOCR-VL identity must be an object."
                                )
                            if not isinstance(raw_ocr_result, dict):
                                raise ValueError(
                                    "Cached PaddleOCR-VL result must be an object."
                                )
                            validate_raw_ocr_result(raw_ocr_result)
                            payload = {
                                "cache_key": str(row["cache_key"]),
                                "identity": identity,
                                "raw_ocr_result": raw_ocr_result,
                                "created_at": float(row["created_at"]),
                                "last_used_at": float(row["last_used_at"]),
                                "access_order": int(row["access_order"]),
                                "hit_count": int(row["hit_count"]),
                            }
                            handle.write(canonical_json(payload))
                            handle.write("\n")
                    os.replace(partial, output)
                except BaseException:
                    try:
                        partial.unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise
                return len(rows)
            except (
                OSError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                sqlite3.Error,
            ) as exc:
                if isinstance(
                    exc,
                    (TypeError, ValueError, json.JSONDecodeError, sqlite3.Error),
                ):
                    self._disable(exc)
                raise self._management_unavailable_error() from exc
