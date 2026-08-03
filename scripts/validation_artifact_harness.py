#!/usr/bin/env python3
"""Create and verify private, classified validation-artifact runs.

This harness deliberately writes only below the ignored
``banchmark_result_log/`` root.  It is for benchmark, debug, migration, and
implementation evidence, never for product output selected by an end user.

There are two normal entry points:

* Python scripts call :func:`select_managed_output_directory`; when no output
  directory is supplied, it creates a classified run and finalizes a manifest.
* Shell/PowerShell experiments call ``run``.  The child receives
  ``CT_VALIDATION_OUTPUT_DIR`` and its stdout/stderr are retained with the run.

The manifest contains relative paths and filesystem metadata.  Large media and
model files are recorded but not re-hashed by default, so image/model evidence
is kept without turning finalization into a second benchmark run.  Small
documents are always hashed even when a caller deliberately sets a low size
limit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
import uuid


ARTIFACT_SCHEMA_VERSION = 1
ARCHIVE_DIRECTORY_NAME = "banchmark_result_log"
MANAGED_RUNS_DIRECTORY_NAME = "managed-runs"
ARTIFACT_DIRECTORY_NAME = "artifacts"
LOG_DIRECTORY_NAME = "logs"
MANIFEST_FILE_NAME = "artifact-manifest.json"
DEFAULT_HASH_LIMIT_BYTES = 8 * 1024 * 1024
ALWAYS_HASH_EXTENSIONS = frozenset(
    {
        ".csv",
        ".ini",
        ".json",
        ".jsonl",
        ".md",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)

CATEGORY_NAMES = frozenset(
    {
        "10-gemma-translation",
        "20-paddle-ocr",
        "30-mangalmm-coo",
        "40-inpaint-mask-render",
        "50-cache-checkpoint",
        "60-runtime-release",
        "70-project-output",
        "80-pr-governance",
        "90-cross-cutting",
    }
)

_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
class ArtifactHarnessError(RuntimeError):
    """Raised when an artifact run cannot be created or verified safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_archive_root() -> Path:
    return repository_root() / ARCHIVE_DIRECTORY_NAME


def _relative_to(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError as exc:
        raise ArtifactHarnessError(f"Path escaped validation archive: {path}") from exc


def _assert_archive_root(archive_root: Path) -> Path:
    configured_root = default_archive_root()
    if archive_root.absolute() != configured_root.absolute():
        raise ArtifactHarnessError(
            "Validation archive root must be the repository's ignored "
            f"'{ARCHIVE_DIRECTORY_NAME}' directory."
        )
    if configured_root.is_symlink():
        raise ArtifactHarnessError("Validation archive root cannot be a symbolic link.")
    if configured_root.exists() and not configured_root.is_dir():
        raise ArtifactHarnessError("Validation archive root must be a directory.")
    if not _is_ignored_by_git(configured_root):
        raise ArtifactHarnessError(
            f"Validation archive root is not ignored by Git: {ARCHIVE_DIRECTORY_NAME}"
        )
    try:
        configured_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ArtifactHarnessError("Unable to create the private validation archive root.") from exc
    return configured_root.resolve()


def _is_ignored_by_git(path: Path) -> bool:
    """Return whether the canonical private archive is ignored in this checkout."""

    root = repository_root().resolve()
    try:
        relative_path = path.resolve().relative_to(root).as_posix()
    except ValueError:
        return False
    try:
        completed = subprocess.run(
            ["git", "check-ignore", "-q", "--", relative_path],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return completed.returncode == 0


def _validate_component(value: str, *, field_name: str) -> str:
    candidate = str(value or "").strip()
    if not _SAFE_COMPONENT_RE.fullmatch(candidate):
        raise ArtifactHarnessError(
            f"{field_name} must be 1-96 ASCII letters, digits, '.', '_' or '-', "
            "and cannot contain a path separator."
        )
    return candidate


def _git_value(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repository_root(),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_record(path: Path, *, root: Path, hash_limit_bytes: int) -> dict[str, Any]:
    stat = path.lstat()
    relative_path = _relative_to(path, root).as_posix()
    if path.is_symlink():
        try:
            target = os.readlink(path)
        except OSError:
            target = ""
        return {
            "path": relative_path,
            "kind": "symlink",
            "target": target,
            "size": 0,
            "creation_time_ns": getattr(stat, "st_birthtime_ns", stat.st_ctime_ns),
            "modified_time_ns": stat.st_mtime_ns,
            "access_time_ns": stat.st_atime_ns,
            "mode": stat.st_mode,
            "file_attributes": getattr(stat, "st_file_attributes", None),
            "sha256": None,
            "hash_state": "not-applicable",
        }

    if path.is_dir():
        kind = "directory"
        size = 0
        digest: str | None = None
        hash_state = "not-applicable"
    elif path.is_file():
        kind = "file"
        size = stat.st_size
        if path.suffix.lower() in ALWAYS_HASH_EXTENSIONS:
            digest = _sha256_file(path)
            hash_state = "recorded-document"
        elif size <= hash_limit_bytes:
            digest = _sha256_file(path)
            hash_state = "recorded"
        else:
            digest = None
            hash_state = "size-limit"
    else:
        kind = "other"
        size = stat.st_size
        digest = None
        hash_state = "not-applicable"

    return {
        "path": relative_path,
        "kind": kind,
        "size": size,
        "creation_time_ns": getattr(stat, "st_birthtime_ns", stat.st_ctime_ns),
        "modified_time_ns": stat.st_mtime_ns,
        "access_time_ns": stat.st_atime_ns,
        "mode": stat.st_mode,
        "file_attributes": getattr(stat, "st_file_attributes", None),
        "sha256": digest,
        "hash_state": hash_state,
    }


def _iter_tree(root: Path) -> Iterable[Path]:
    """Yield all descendants without following links or escaping the run root."""

    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        entries = [*directories, *files]
        for name in sorted(entries):
            child = current_path / name
            yield child
        directories[:] = [
            name for name in directories if not (current_path / name).is_symlink()
        ]


def _collect_entries(run_root: Path, *, hash_limit_bytes: int) -> list[dict[str, Any]]:
    manifest_path = run_root / MANIFEST_FILE_NAME
    entries = [
        _metadata_record(path, root=run_root, hash_limit_bytes=hash_limit_bytes)
        for path in _iter_tree(run_root)
        if path != manifest_path and not path.name.startswith(f".{MANIFEST_FILE_NAME}.partial-")
    ]
    return sorted(entries, key=lambda entry: str(entry["path"]))


def _create_managed_run_root(
    *,
    archive_root: Path,
    category: str,
    family: str,
    run_id: str,
) -> Path:
    """Create a run without allowing an ignored archive symlink to escape it."""

    current = archive_root
    for component in (MANAGED_RUNS_DIRECTORY_NAME, category, family):
        candidate = current / component
        if candidate.is_symlink():
            raise ArtifactHarnessError(
                "Managed validation archive directories cannot be symbolic links."
            )
        try:
            candidate.mkdir(exist_ok=True)
        except OSError as exc:
            raise ArtifactHarnessError(
                f"Unable to create managed validation archive directory: {component}"
            ) from exc
        current = candidate.resolve()
        _relative_to(current, archive_root)

    run_root = current / run_id
    if run_root.exists():
        raise ArtifactHarnessError(f"Validation run already exists: {run_root.name}")
    try:
        run_root.mkdir(exist_ok=False)
    except OSError as exc:
        raise ArtifactHarnessError(
            f"Unable to create managed validation run: {run_root.name}"
        ) from exc
    resolved_run_root = run_root.resolve()
    _relative_to(resolved_run_root, archive_root)
    return resolved_run_root


@dataclass
class ManagedArtifactRun:
    """A single private run with a durable, atomic manifest."""

    archive_root: Path
    category: str
    family: str
    run_id: str
    run_root: Path
    artifact_root: Path
    log_root: Path
    manifest_path: Path
    created_utc: str
    hash_limit_bytes: int = DEFAULT_HASH_LIMIT_BYTES
    _closed: bool = field(default=False, init=False, repr=False)

    @classmethod
    def create(
        cls,
        *,
        family: str,
        category: str,
        run_id: str | None = None,
        archive_root: Path | None = None,
        hash_limit_bytes: int = DEFAULT_HASH_LIMIT_BYTES,
    ) -> "ManagedArtifactRun":
        safe_family = _validate_component(family, field_name="family")
        selected_category = str(category or "").strip()
        if selected_category not in CATEGORY_NAMES:
            raise ArtifactHarnessError(
                "A known validation category is required; automatic category inference is not allowed."
            )
        selected_run_id = run_id or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        safe_run_id = _validate_component(selected_run_id, field_name="run_id")
        if hash_limit_bytes < 0:
            raise ArtifactHarnessError("hash_limit_bytes must be non-negative.")

        root = _assert_archive_root(archive_root or default_archive_root())
        run_root = _create_managed_run_root(
            archive_root=root,
            category=selected_category,
            family=safe_family,
            run_id=safe_run_id,
        )
        artifact_root = run_root / ARTIFACT_DIRECTORY_NAME
        log_root = run_root / LOG_DIRECTORY_NAME
        artifact_root.mkdir()
        log_root.mkdir()
        run = cls(
            archive_root=root,
            category=selected_category,
            family=safe_family,
            run_id=safe_run_id,
            run_root=run_root,
            artifact_root=artifact_root,
            log_root=log_root,
            manifest_path=run_root / MANIFEST_FILE_NAME,
            created_utc=_utc_now(),
            hash_limit_bytes=hash_limit_bytes,
        )
        run._write_manifest(status="running", metadata={})
        return run

    @classmethod
    def _open_existing(cls, run_root: Path) -> tuple["ManagedArtifactRun", dict[str, Any]]:
        archive_root = _assert_archive_root(default_archive_root())
        resolved_run_root = run_root.resolve()
        category, family, run_id = _assert_managed_run_root(resolved_run_root, archive_root)
        manifest_path = resolved_run_root / MANIFEST_FILE_NAME
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactHarnessError("Managed validation run manifest is unreadable.") from exc
        archive = manifest.get("archive")
        if not isinstance(archive, Mapping):
            raise ArtifactHarnessError("Managed validation run manifest has no archive identity.")
        if (
            str(archive.get("category") or "") != category
            or str(archive.get("family") or "") != family
            or str(archive.get("run_id") or "") != run_id
        ):
            raise ArtifactHarnessError("Managed validation run manifest identity does not match its path.")
        artifact_root = resolved_run_root / ARTIFACT_DIRECTORY_NAME
        log_root = resolved_run_root / LOG_DIRECTORY_NAME
        if not artifact_root.is_dir() or not log_root.is_dir():
            raise ArtifactHarnessError("Managed validation run is missing its artifact or log directory.")
        hash_limit = manifest.get("hash_limit_bytes", DEFAULT_HASH_LIMIT_BYTES)
        if isinstance(hash_limit, bool) or not isinstance(hash_limit, int) or hash_limit < 0:
            raise ArtifactHarnessError("Managed validation run has an invalid hash limit.")
        created_utc = str(manifest.get("created_utc") or "").strip()
        if not created_utc:
            raise ArtifactHarnessError("Managed validation run is missing its creation timestamp.")
        return (
            cls(
            archive_root=archive_root,
            category=category,
            family=family,
            run_id=run_id,
            run_root=resolved_run_root,
            artifact_root=artifact_root,
            log_root=log_root,
            manifest_path=manifest_path,
            created_utc=created_utc,
            hash_limit_bytes=hash_limit,
            ),
            dict(manifest),
        )

    @classmethod
    def resume(cls, run_root: Path) -> "ManagedArtifactRun":
        """Reopen an interrupted managed run whose manifest is still running.

        Long private benchmarks deliberately leave a ``running`` manifest on a
        transient worker exit. Reopening only that state preserves the original
        provenance while refusing to append artifacts to completed or failed
        evidence sets.
        """

        run, manifest = cls._open_existing(run_root)
        if str(manifest.get("status") or "") != "running":
            raise ArtifactHarnessError(
                "Only an interrupted managed validation run with status 'running' can resume."
            )
        return run

    @classmethod
    def recover_failed_atomic_replace(
        cls,
        run_root: Path,
        *,
        command: str,
        target_file_name: str,
    ) -> "ManagedArtifactRun":
        """Reopen one audited, known-safe atomic-replace interruption.

        This is intentionally narrower than ``resume``: generic failed runs
        remain terminal. The caller must name the original command and target
        file, and the failure must be the exact Windows-style temporary-file
        replacement signature.
        """

        safe_target = str(target_file_name or "").strip()
        if not safe_target or Path(safe_target).name != safe_target:
            raise ArtifactHarnessError("Recovery target file name is unsafe.")
        run, manifest = cls._open_existing(run_root)
        metadata = manifest.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ArtifactHarnessError("Failed validation run has no recovery metadata.")
        error_message = str(metadata.get("error_message") or "")
        temporary_marker = f".{safe_target}.partial-"
        if (
            str(manifest.get("status") or "") != "failed"
            or str(metadata.get("command") or "") != str(command)
            or str(metadata.get("error_type") or "") != "PermissionError"
            or temporary_marker not in error_message
            or safe_target not in error_message
        ):
            raise ArtifactHarnessError("Failed validation run is not a recoverable atomic-replace interruption.")

        original_manifest_sha256 = _sha256_file(run.manifest_path)
        recovery_path = run.log_root / f"recovery-{original_manifest_sha256[:24]}.json"
        recovery_record = {
            "schema_version": "managed-artifact-recovery-v1",
            "recovered_utc": _utc_now(),
            "reason": "atomic-replace-permission-error",
            "original_manifest_sha256": original_manifest_sha256,
            "original_manifest": manifest,
        }
        if recovery_path.exists():
            try:
                existing = json.loads(recovery_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ArtifactHarnessError("Existing recovery evidence is unreadable.") from exc
            if not isinstance(existing, Mapping) or existing.get("original_manifest_sha256") != original_manifest_sha256:
                raise ArtifactHarnessError("Existing recovery evidence does not match the failed manifest.")
        else:
            _atomic_write_json(recovery_path, recovery_record)

        run.checkpoint(
            metadata={
                "command": str(command),
                "state": "RECOVERED_ATOMIC_REPLACE",
                "recovery_record": recovery_path.relative_to(run.run_root).as_posix(),
                "original_manifest_sha256": original_manifest_sha256,
                "original_error_type": str(metadata.get("error_type") or ""),
                "original_error_message": error_message[:4096],
            }
        )
        return run

    def _base_manifest(self, *, status: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": status,
            "created_utc": self.created_utc,
            "finished_utc": _utc_now() if status != "running" else None,
            "archive": {
                "category": self.category,
                "family": self.family,
                "run_id": self.run_id,
                "run_path": _relative_to(self.run_root, self.archive_root).as_posix(),
                "artifact_path": _relative_to(self.artifact_root, self.run_root).as_posix(),
            },
            "provenance": {
                "git_head": _git_value("rev-parse", "HEAD"),
                "git_branch": _git_value("branch", "--show-current"),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            "hash_limit_bytes": self.hash_limit_bytes,
            "metadata": dict(metadata),
        }

    def _write_manifest(self, *, status: str, metadata: Mapping[str, Any]) -> None:
        payload = self._base_manifest(status=status, metadata=metadata)
        payload["entries"] = _collect_entries(
            self.run_root,
            hash_limit_bytes=self.hash_limit_bytes,
        )
        _atomic_write_json(self.manifest_path, payload)

    def complete(self, *, metadata: Mapping[str, Any] | None = None) -> None:
        if self._closed:
            return
        self._write_manifest(status="completed", metadata=metadata or {})
        self._closed = True

    def checkpoint(self, *, metadata: Mapping[str, Any] | None = None) -> None:
        """Refresh an interrupted run's running manifest without closing it."""

        if self._closed:
            raise ArtifactHarnessError("Cannot checkpoint a closed validation artifact run.")
        self._write_manifest(status="running", metadata=metadata or {})

    def fail(
        self,
        error: BaseException | str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if self._closed:
            return
        details = dict(metadata or {})
        details["error_type"] = type(error).__name__ if isinstance(error, BaseException) else "Error"
        details["error_message"] = str(error)[:4096]
        self._write_manifest(status="failed", metadata=details)
        self._closed = True

    def verify(self) -> list[str]:
        return verify_run(self.run_root)


def select_managed_output_directory(
    *,
    family: str,
    category: str,
    explicit_output_directory: Path | None = None,
) -> tuple[Path, ManagedArtifactRun | None]:
    """Choose an output directory for a script without changing explicit overrides.

    A parent ``run`` command owns the manifest when it provides
    ``CT_VALIDATION_OUTPUT_DIR``.  Direct script execution creates a complete
    managed run.  An explicit ``--output-dir`` remains a deliberate local
    override and therefore does not create a second manifest beside user data.
    """

    if explicit_output_directory is not None:
        return explicit_output_directory.resolve(), None

    inherited = str(os.environ.get("CT_VALIDATION_OUTPUT_DIR", "") or "").strip()
    if inherited:
        output_root = Path(inherited).resolve()
        archive_root = _assert_archive_root(default_archive_root())
        inherited_run_root = str(os.environ.get("CT_VALIDATION_RUN_ROOT", "") or "").strip()
        if not inherited_run_root:
            raise ArtifactHarnessError(
                "CT_VALIDATION_OUTPUT_DIR requires CT_VALIDATION_RUN_ROOT from the managed runner."
            )
        run_root = Path(inherited_run_root).resolve()
        run_category, run_family, _ = _assert_managed_run_root(run_root, archive_root)
        safe_family = _validate_component(family, field_name="family")
        expected_family = str(os.environ.get("CT_VALIDATION_FAMILY", "") or "").strip()
        expected_category = str(os.environ.get("CT_VALIDATION_CATEGORY", "") or "").strip()
        if (
            expected_family != safe_family
            or expected_category != category
            or run_family != safe_family
            or run_category != category
        ):
            raise ArtifactHarnessError(
                "Managed child category and family must match the parent validation run."
            )
        if output_root != run_root / ARTIFACT_DIRECTORY_NAME or not output_root.is_dir():
            raise ArtifactHarnessError(
                "CT_VALIDATION_OUTPUT_DIR must be the managed run's artifacts directory."
            )
        return output_root, None

    run = ManagedArtifactRun.create(family=family, category=category)
    return run.artifact_root, run


def _entry_comparable(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        entry.get("kind"),
        entry.get("size"),
        entry.get("creation_time_ns"),
        entry.get("modified_time_ns"),
        entry.get("mode"),
        entry.get("file_attributes"),
        entry.get("sha256"),
        entry.get("hash_state"),
        entry.get("target"),
    )


def _assert_managed_run_root(run_root: Path, archive_root: Path) -> tuple[str, str, str]:
    """Reject arbitrary archive descendants passed to verification or child scripts."""

    relative = _relative_to(run_root, archive_root)
    parts = relative.parts
    if (
        len(parts) != 4
        or parts[0] != MANAGED_RUNS_DIRECTORY_NAME
        or parts[1] not in CATEGORY_NAMES
        or not _SAFE_COMPONENT_RE.fullmatch(parts[2])
        or not _SAFE_COMPONENT_RE.fullmatch(parts[3])
    ):
        raise ArtifactHarnessError(
            "Managed run root must be managed-runs/<category>/<family>/<run-id>."
        )
    return parts[1], parts[2], parts[3]


def verify_run(run_root: Path) -> list[str]:
    """Return manifest mismatches; an empty list means the archived run is intact."""

    root = run_root.resolve()
    try:
        _assert_managed_run_root(root, _assert_archive_root(default_archive_root()))
    except ArtifactHarnessError as exc:
        return [str(exc)]
    manifest_path = root / MANIFEST_FILE_NAME
    if not manifest_path.is_file():
        return [f"Missing manifest: {MANIFEST_FILE_NAME}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Unreadable manifest: {exc}"]

    limit = int(manifest.get("hash_limit_bytes", DEFAULT_HASH_LIMIT_BYTES))
    expected = {
        str(entry.get("path")): entry
        for entry in manifest.get("entries", [])
        if isinstance(entry, Mapping) and str(entry.get("path", ""))
    }
    actual = {
        str(entry["path"]): entry
        for entry in _collect_entries(root, hash_limit_bytes=limit)
    }
    errors: list[str] = []
    for path in sorted(expected.keys() - actual.keys()):
        errors.append(f"Missing archived entry: {path}")
    for path in sorted(actual.keys() - expected.keys()):
        errors.append(f"Unexpected archived entry: {path}")
    for path in sorted(expected.keys() & actual.keys()):
        if _entry_comparable(expected[path]) != _entry_comparable(actual[path]):
            errors.append(f"Metadata mismatch: {path}")
    return errors


def _run_command(args: argparse.Namespace) -> int:
    command = list(args.command or [])
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise ArtifactHarnessError("run requires a command after '--'.")
    run = ManagedArtifactRun.create(
        family=args.family,
        category=args.category,
        run_id=args.run_id,
    )
    environment = os.environ.copy()
    environment.update(
        {
            "CT_VALIDATION_OUTPUT_DIR": str(run.artifact_root),
            "CT_VALIDATION_RUN_ROOT": str(run.run_root),
            "CT_VALIDATION_RUN_ID": run.run_id,
            "CT_VALIDATION_CATEGORY": run.category,
            "CT_VALIDATION_FAMILY": run.family,
        }
    )
    cwd = Path(args.cwd).resolve() if args.cwd else repository_root()
    stdout_path = run.log_root / "stdout.log"
    stderr_path = run.log_root / "stderr.log"
    try:
        with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout, stderr_path.open(
            "w", encoding="utf-8", errors="replace"
        ) as stderr:
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=environment,
                check=False,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
    except BaseException as exc:
        run.fail(exc, metadata={"command_name": Path(command[0]).name})
        raise

    metadata = {
        "command_name": Path(command[0]).name,
        "command": command,
        "exit_code": completed.returncode,
    }
    if completed.returncode == 0:
        run.complete(metadata=metadata)
        print(run.run_root)
        return 0
    run.fail(f"command exited with {completed.returncode}", metadata=metadata)
    print(run.run_root, file=sys.stderr)
    return completed.returncode


def _verify_command(args: argparse.Namespace) -> int:
    errors = verify_run(Path(args.run_root))
    if errors:
        for error in errors:
            print(f"[VALIDATION-ARCHIVE] {error}", file=sys.stderr)
        return 1
    print("Validation artifact manifest verification passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create classified private validation-artifact runs."
    )
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    run = subparsers.add_parser(
        "run",
        help="Run a command with a classified private artifact directory.",
    )
    run.add_argument("--family", required=True, help="Stable safe run family name.")
    run.add_argument("--category", choices=sorted(CATEGORY_NAMES), required=True)
    run.add_argument("--run-id", help="Optional unique safe run identifier.")
    run.add_argument("--cwd", help="Optional working directory for the child command.")
    run.add_argument("command", nargs=argparse.REMAINDER)
    run.set_defaults(handler=_run_command)

    verify = subparsers.add_parser("verify", help="Verify a finalized private run.")
    verify.add_argument("--run-root", required=True, type=Path)
    verify.set_defaults(handler=_verify_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except ArtifactHarnessError as exc:
        print(f"[VALIDATION-ARCHIVE] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
