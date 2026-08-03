"""Atomic private-artifact persistence for resumable sampler runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping
import uuid


class StorageError(RuntimeError):
    """Raised when a private sampler artifact would escape its managed run."""


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_COMPLETION_INDEX_FILE_NAME = "completion-index.jsonl"
_LEGACY_COMPLETION_INDEX_FILE_NAME = "completion-index.json"
_ATOMIC_REPLACE_ATTEMPTS = 8
_ATOMIC_REPLACE_INITIAL_BACKOFF_SECONDS = 0.025


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"


def atomic_write_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    """Write one complete JSON document or leave the prior document intact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(_ATOMIC_REPLACE_ATTEMPTS):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt + 1 >= _ATOMIC_REPLACE_ATTEMPTS:
                    raise
                # Windows readers that do not request FILE_SHARE_DELETE can
                # briefly reject the replace despite the old JSON remaining
                # valid. Retrying the already-fsynced temporary file keeps
                # the write atomic without treating a short read lock as a
                # sampler failure.
                time.sleep(_ATOMIC_REPLACE_INITIAL_BACKOFF_SECONDS * (2**attempt))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StorageError(f"Unreadable private sampler artifact: {path.name}") from exc


def _safe_component(value: str, *, field: str) -> str:
    candidate = str(value or "").strip()
    if not _SAFE_COMPONENT.fullmatch(candidate):
        raise StorageError(f"Unsafe {field} component.")
    return candidate


@dataclass
class RunStore:
    """Per-run layout with a first-valid-response completion index."""

    root: Path
    _completion_cache: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _completion_loaded: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        resolved = self.root.resolve()
        if not resolved.is_dir():
            raise StorageError("Private sampler artifact root does not exist.")
        self.root = resolved

    def _path(self, *parts: str) -> Path:
        validated = tuple(_safe_component(part, field="artifact") for part in parts)
        candidate = (self.root.joinpath(*validated)).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise StorageError("Private sampler artifact escaped its managed root.") from exc
        return candidate

    @property
    def progress_path(self) -> Path:
        return self._path("progress.json")

    @property
    def completion_index_path(self) -> Path:
        return self._path(_COMPLETION_INDEX_FILE_NAME)

    @property
    def _legacy_completion_index_path(self) -> Path:
        return self._path(_LEGACY_COMPLETION_INDEX_FILE_NAME)

    @property
    def attempt_ledger_path(self) -> Path:
        return self._path("attempt-ledger.jsonl")

    @property
    def request_contract_index_path(self) -> Path:
        return self._path("request-contract-index.json")

    def case_path(self, *, phase: str, arm: str, run: str, case_id: str) -> Path:
        return self._path("phase", phase, "arm", arm, "run", run, "case", f"{case_id}.json")

    def record_case_if_first(
        self,
        *,
        phase: str,
        arm: str,
        run: str,
        case_id: str,
        logical_slot: str,
        payload: Mapping[str, Any],
    ) -> bool:
        """Persist a logical result exactly once after a valid complete reply.

        Invalid output is itself a completed observable result and is retained;
        retryable transport records are deliberately not entered in the index.
        """

        existing = self._completion_index()
        if logical_slot in existing:
            return False
        if str(payload.get("status") or "") != "complete":
            raise StorageError("Only complete response records may enter the completion index.")
        if str(payload.get("logical_slot") or "") != str(logical_slot):
            raise StorageError("Response record logical slot does not match its completion index slot.")
        if (
            str(payload.get("phase") or "") != str(phase)
            or str(payload.get("arm_key") or "") != str(arm)
            or str(payload.get("case_id") or "") != str(case_id)
        ):
            raise StorageError("Response record identity does not match its artifact path.")
        destination = self.case_path(phase=phase, arm=arm, run=run, case_id=case_id)
        if destination.exists():
            raise StorageError("Case artifact exists without a completion-index entry.")
        atomic_write_json(destination, dict(payload))
        entry = {
            "phase": phase,
            "arm": arm,
            "run": run,
            "case_id": case_id,
            "path": destination.relative_to(self.root).as_posix(),
            "status": str(payload.get("status", "")),
            "recorded_utc": str(payload.get("recorded_utc") or utc_now()),
        }
        # A full JSON object rewritten after every one of 76,480 responses is
        # quadratic I/O.  The immutable case artifact is authoritative; this
        # fsynced append-only index is rebuilt from those artifacts if a crash
        # happens between the two writes.
        self._append_completion_index(logical_slot, entry)
        existing[logical_slot] = entry
        return True

    def completed_index(self) -> dict[str, dict[str, Any]]:
        return {key: dict(value) for key, value in self._completion_index().items()}

    def iter_completion_entries(self) -> Iterable[dict[str, Any]]:
        for logical_slot, entry in self._completion_index().items():
            result = dict(entry)
            result["_completion_index_logical_slot"] = logical_slot
            yield result

    def iter_snapshot_completion_entries(self) -> Iterable[dict[str, Any]]:
        """Read only fully appended index entries without recovery or writes.

        A live campaign can have one case JSON between its atomic write and the
        following index append.  Incremental review deliberately waits for the
        index entry instead of invoking normal crash recovery against a writer
        that is still healthy.
        """

        index: dict[str, dict[str, Any]] = {}
        if self.completion_index_path.exists():
            self._read_completion_jsonl(self.completion_index_path, index)
        elif self._legacy_completion_index_path.exists():
            legacy = read_json(self._legacy_completion_index_path)
            if not isinstance(legacy, Mapping):
                raise StorageError("Legacy completion index is not an object.")
            for logical_slot, entry in legacy.items():
                if not isinstance(entry, Mapping) or not str(logical_slot or ""):
                    raise StorageError("Legacy completion index has an invalid entry.")
                index[str(logical_slot)] = dict(entry)
        for logical_slot, entry in index.items():
            result = dict(entry)
            result["_completion_index_logical_slot"] = logical_slot
            yield result

    def is_completed(self, logical_slot: str) -> bool:
        return str(logical_slot) in self._completion_index()

    def completed_count(self) -> int:
        return len(self._completion_index())

    def _completion_index(self) -> dict[str, dict[str, Any]]:
        if self._completion_loaded:
            return self._completion_cache
        index: dict[str, dict[str, Any]] = {}
        index_path = self.completion_index_path
        if index_path.exists():
            self._read_completion_jsonl(index_path, index)
        elif self._legacy_completion_index_path.exists():
            legacy = read_json(self._legacy_completion_index_path)
            if not isinstance(legacy, Mapping):
                raise StorageError("Legacy completion index is not an object.")
            for logical_slot, entry in legacy.items():
                if not isinstance(entry, Mapping):
                    raise StorageError("Legacy completion index has an invalid entry.")
                safe_slot = str(logical_slot or "")
                if not safe_slot:
                    raise StorageError("Legacy completion index has an empty logical slot.")
                index[safe_slot] = dict(entry)
        recovered = self._recover_unindexed_case_artifacts(index)
        if recovered:
            self._rewrite_completion_index(index)
        self._completion_cache = index
        self._completion_loaded = True
        return self._completion_cache

    @staticmethod
    def _read_completion_jsonl(path: Path, index: dict[str, dict[str, Any]]) -> None:
        try:
            lines = path.read_bytes().splitlines(keepends=True)
        except OSError as exc:
            raise StorageError("Completion index is unreadable.") from exc
        for position, raw_line in enumerate(lines):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if position == len(lines) - 1 and not raw_line.endswith(b"\n"):
                    # A process can be interrupted mid-append.  Its atomic
                    # case JSON is recovered below; earlier malformed lines
                    # are evidence corruption and remain fail-closed.
                    break
                raise StorageError("Completion index contains an invalid JSONL entry.") from exc
            if not isinstance(payload, Mapping):
                raise StorageError("Completion index entry is not an object.")
            logical_slot = str(payload.get("logical_slot") or "")
            entry = payload.get("entry")
            if not logical_slot or not isinstance(entry, Mapping):
                raise StorageError("Completion index entry is incomplete.")
            normalized = dict(entry)
            previous = index.get(logical_slot)
            if previous is not None and previous != normalized:
                raise StorageError("Completion index has conflicting logical slots.")
            index[logical_slot] = normalized

    def _recover_unindexed_case_artifacts(self, index: dict[str, dict[str, Any]]) -> bool:
        recovered = False
        for candidate in sorted(self.root.glob("phase/*/arm/*/run/*/case/*.json")):
            parts = candidate.relative_to(self.root).parts
            if (
                len(parts) != 8
                or parts[0] != "phase"
                or parts[2] != "arm"
                or parts[4] != "run"
                or parts[6] != "case"
            ):
                raise StorageError("Case artifact path has an invalid managed layout.")
            record = read_json(candidate)
            if not isinstance(record, Mapping):
                raise StorageError("Case artifact is not an object.")
            logical_slot = str(record.get("logical_slot") or "")
            if not logical_slot:
                raise StorageError("Case artifact has no logical slot.")
            if str(record.get("status") or "") != "complete":
                raise StorageError("Case artifact is not a completed response.")
            if (
                str(record.get("phase") or "") != parts[1]
                or str(record.get("arm_key") or "") != parts[3]
                or str(record.get("case_id") or "") != Path(parts[7]).stem
            ):
                raise StorageError("Case artifact identity disagrees with its managed path.")
            relative = candidate.relative_to(self.root).as_posix()
            entry = {
                "phase": str(record.get("phase") or ""),
                "arm": str(record.get("arm_key") or ""),
                "run": candidate.parents[1].name,
                "case_id": str(record.get("case_id") or ""),
                "path": relative,
                "status": "complete",
                "recorded_utc": str(record.get("recorded_utc") or ""),
            }
            previous = index.get(logical_slot)
            if previous is None:
                index[logical_slot] = entry
                recovered = True
            elif previous != entry:
                raise StorageError("Completion index and case artifact disagree.")
        return recovered

    def _append_completion_index(self, logical_slot: str, entry: Mapping[str, Any]) -> None:
        destination = self.completion_index_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            {"logical_slot": str(logical_slot), "entry": dict(entry)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        with destination.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    def _rewrite_completion_index(self, index: Mapping[str, Mapping[str, Any]]) -> None:
        destination = self.completion_index_path
        temporary = destination.with_name(
            f".{destination.name}.partial-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                for logical_slot in sorted(index):
                    handle.write(
                        json.dumps(
                            {"logical_slot": logical_slot, "entry": dict(index[logical_slot])},
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def append_attempt(self, payload: Mapping[str, Any]) -> None:
        """Durably append a non-secret attempt event; raw data stays per case."""

        destination = self.attempt_ledger_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = (
            json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        with destination.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    def bind_request_contract(self, *, case_id: str, identity: Mapping[str, Any]) -> None:
        """Bind immutable prompt/schema/context hashes before any sampler run.

        Sampler and seed fields are intentionally absent from ``identity``. A
        later arm may therefore vary only those declared values, never prompt
        serialization, context order, schema, model, or token limit.
        """

        safe_case_id = _safe_component(case_id, field="case id")
        if self.request_contract_index_path.exists():
            raw = read_json(self.request_contract_index_path)
            if not isinstance(raw, dict):
                raise StorageError("Request contract index is not an object.")
            index = {str(key): value for key, value in raw.items()}
        else:
            index = {}
        normalized = dict(identity)
        existing = index.get(safe_case_id)
        if existing is not None:
            if existing != normalized:
                raise StorageError("Sampler request contract changed for a frozen case.")
            return
        index[safe_case_id] = normalized
        atomic_write_json(self.request_contract_index_path, index)

    def update_progress(self, payload: Mapping[str, Any]) -> None:
        atomic_write_json(self.progress_path, dict(payload))

    def phase_status_path(self, phase: str) -> Path:
        return self._path("phase-status", f"{phase}.json")

    def write_phase_status(self, phase: str, payload: Mapping[str, Any]) -> None:
        atomic_write_json(self.phase_status_path(phase), dict(payload))
