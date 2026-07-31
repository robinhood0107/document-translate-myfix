#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
PRODUCT_NAME = "comic-translate"
RELEASE_KIND = "windows-launcher-source"

ALLOWLIST_FILES = frozenset(
    {
        "LICENSE",
        "README.md",
        "README_ko.md",
        "comic.py",
        "controller.py",
        "docker-compose.gemma-host-rollback.yaml",
        "docker-compose.yaml",
        "requirements-base.txt",
        "requirements-cuda12.txt",
        "requirements-cuda13.txt",
        "requirements.txt",
        "run_comic.bat",
        "run_comic_cuda13.bat",
        "scripts/prepare_gemma_runtime.ps1",
        "scripts/prepare_mangalmm_llamacpp_runtime.ps1",
        "scripts/prepare_paddleocr_llamacpp_runtime.ps1",
        "scripts/derive_paddleocr_spotting_mmproj.py",
        "scripts/prepare_paddleocr_spotting_llamacpp_runtime.ps1",
        "scripts/verify_windows_runtime.py",
    }
)
ALLOWLIST_PREFIXES = (
    "app/",
    "hunyuanocr_docker_files/",
    "imkit/",
    "mangalmm_docker_files/",
    "modules/",
    "music/",
    "paddleocr_vl_docker_files/",
    "paddleocr_vl_spotting_docker_files/",
    "pipeline/",
    "resources/",
)
REQUIRED_BUNDLE_FILES = frozenset(
    {
        "LICENSE",
        "README.md",
        "README_ko.md",
        "app/version.py",
        "comic.py",
        "controller.py",
        "docker-compose.yaml",
        "mangalmm_docker_files/docker-compose.yaml",
        "paddleocr_vl_docker_files/docker-compose.yaml",
        "paddleocr_vl_docker_files/pipeline_conf.yaml",
        "paddleocr_vl_spotting_docker_files/README.md",
        "paddleocr_vl_spotting_docker_files/docker-compose.yaml",
        "requirements-base.txt",
        "requirements-cuda12.txt",
        "requirements-cuda13.txt",
        "run_comic.bat",
        "run_comic_cuda13.bat",
        "scripts/prepare_gemma_runtime.ps1",
        "scripts/prepare_mangalmm_llamacpp_runtime.ps1",
        "scripts/prepare_paddleocr_llamacpp_runtime.ps1",
        "scripts/derive_paddleocr_spotting_mmproj.py",
        "scripts/prepare_paddleocr_spotting_llamacpp_runtime.ps1",
        "scripts/verify_windows_runtime.py",
    }
)
DENIED_SUFFIXES = frozenset(
    {
        ".7z",
        ".cbz",
        ".db",
        ".gguf",
        ".key",
        ".log",
        ".onnx",
        ".pem",
        ".pt",
        ".pth",
        ".rar",
        ".sqlite",
        ".sqlite3",
        ".zip",
    }
)
DENIED_BASENAMES = frozenset(
    {
        ".env",
        "credentials.json",
        "id_dsa",
        "id_ed25519",
        "id_rsa",
        "secrets.json",
    }
)
WINDOWS_RESERVED_NAMES = frozenset(
    {
        "aux",
        "clock$",
        "con",
        "nul",
        "prn",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)
LOCAL_PATH_PATTERNS = (
    re.compile(rb"/mnt/c/users/", re.IGNORECASE),
    re.compile(rb"/mnt/[a-z]/users/[^/\s]+/", re.IGNORECASE),
    re.compile(rb"/home/(?!paddleocr/|user/|example/)[A-Za-z0-9._-]+/"),
    re.compile(rb"[a-z]:\\users\\[^\\\r\n]+", re.IGNORECASE),
)
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(rb"\bgh[oprsu]_[A-Za-z0-9]{30,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b"),
    re.compile(rb"\bsk-ant-[A-Za-z0-9_-]{32,}\b"),
    re.compile(rb"\bsk_live_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(rb"\bAIza[0-9A-Za-z_-]{30,}\b"),
)
TEXT_SCAN_SUFFIXES = frozenset(
    {
        ".bat",
        ".cfg",
        ".ini",
        ".json",
        ".md",
        ".ps1",
        ".py",
        ".qss",
        ".svg",
        ".toml",
        ".ts",
        ".txt",
        ".yaml",
        ".yml",
    }
)
CRLF_SUFFIXES = frozenset({".bat", ".ps1"})
SEMVER_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
APP_VERSION_RE = re.compile(rb'__version__\s*=\s*["\']([^"\']+)["\']')


@dataclass(frozen=True)
class GitTreeEntry:
    mode: str
    object_id: str
    path: str


def _run_git(
    repo_root: Path,
    args: list[str],
    *,
    input_bytes: bytes | None = None,
) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def resolve_commit(repo_root: Path, commit: str) -> str:
    resolved = _run_git(repo_root, ["rev-parse", "--verify", f"{commit}^{{commit}}"])
    return resolved.decode("ascii").strip()


def commit_epoch(repo_root: Path, commit: str) -> int:
    raw = _run_git(repo_root, ["show", "-s", "--format=%ct", commit])
    return int(raw.decode("ascii").strip())


def _is_allowlisted(path: str) -> bool:
    if any(part.startswith(".") for part in PurePosixPath(path).parts):
        return False
    return path in ALLOWLIST_FILES or path.startswith(ALLOWLIST_PREFIXES)


def _validate_relative_path(path: str) -> None:
    pure = PurePosixPath(path)
    if not path or pure.is_absolute() or "\\" in path:
        raise RuntimeError(f"Unsafe bundle path: {path!r}")
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise RuntimeError(f"Unsafe bundle path component: {path!r}")
    for part in pure.parts:
        if part != part.rstrip(". ") or any(
            ord(character) < 32 or character in '<>:"|?*'
            for character in part
        ):
            raise RuntimeError(f"Windows-unsafe bundle path: {path!r}")
        stem = part.rstrip(". ").split(".", 1)[0].lower()
        if stem in WINDOWS_RESERVED_NAMES:
            raise RuntimeError(f"Windows-reserved bundle path: {path!r}")


def _validate_allowed_file(path: str) -> None:
    _validate_relative_path(path)
    if not _is_allowlisted(path):
        raise RuntimeError(f"File is outside the release allowlist: {path}")
    pure = PurePosixPath(path)
    basename = pure.name.lower()
    suffix = pure.suffix.lower()
    if basename in DENIED_BASENAMES or suffix in DENIED_SUFFIXES:
        raise RuntimeError(f"Denied release file selected by allowlist: {path}")


def list_release_entries(repo_root: Path, commit: str) -> list[GitTreeEntry]:
    raw = _run_git(repo_root, ["ls-tree", "-r", "-z", "--full-tree", commit])
    selected: list[GitTreeEntry] = []
    windows_paths: dict[str, str] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split(" ", 2)
        path = raw_path.decode("utf-8")
        if not _is_allowlisted(path):
            continue
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise RuntimeError(f"Only regular files may enter the release: {path}")
        _validate_allowed_file(path)
        folded = path.casefold()
        if folded in windows_paths:
            raise RuntimeError(
                "Case-insensitive Windows path collision: "
                f"{windows_paths[folded]!r} and {path!r}"
            )
        windows_paths[folded] = path
        selected.append(GitTreeEntry(mode=mode, object_id=object_id, path=path))

    selected.sort(key=lambda entry: entry.path)
    selected_paths = {entry.path for entry in selected}
    missing = sorted(REQUIRED_BUNDLE_FILES - selected_paths)
    if missing:
        raise RuntimeError(f"Required release files are missing: {', '.join(missing)}")
    return selected


def read_git_blobs(
    repo_root: Path,
    entries: Iterable[GitTreeEntry],
) -> dict[str, bytes]:
    ordered = list(entries)
    request = b"".join(f"{entry.object_id}\n".encode("ascii") for entry in ordered)
    raw = _run_git(repo_root, ["cat-file", "--batch"], input_bytes=request)
    offset = 0
    blobs: dict[str, bytes] = {}
    for entry in ordered:
        header_end = raw.find(b"\n", offset)
        if header_end < 0:
            raise RuntimeError(f"Truncated git cat-file header for {entry.path}")
        header = raw[offset:header_end].decode("ascii").split()
        if len(header) != 3 or header[1] != "blob":
            raise RuntimeError(f"Unexpected git object for {entry.path}: {' '.join(header)}")
        object_id, _, size_raw = header
        if object_id != entry.object_id:
            raise RuntimeError(f"Git object order mismatch for {entry.path}")
        size = int(size_raw)
        data_start = header_end + 1
        data_end = data_start + size
        if data_end >= len(raw) or raw[data_end : data_end + 1] != b"\n":
            raise RuntimeError(f"Truncated git blob for {entry.path}")
        blobs[entry.path] = raw[data_start:data_end]
        offset = data_end + 1
    if offset != len(raw):
        raise RuntimeError("Unexpected trailing data from git cat-file")
    return blobs


def normalize_release_bytes(path: str, data: bytes) -> bytes:
    if PurePosixPath(path).suffix.lower() not in CRLF_SUFFIXES:
        return data
    normalized = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return normalized.replace(b"\n", b"\r\n")


def _scan_release_bytes(path: str, data: bytes) -> None:
    if PurePosixPath(path).suffix.lower() not in TEXT_SCAN_SUFFIXES:
        return
    for pattern in LOCAL_PATH_PATTERNS:
        if pattern.search(data):
            raise RuntimeError(f"User-local absolute path found in release file: {path}")
    for pattern in SECRET_PATTERNS:
        if pattern.search(data):
            raise RuntimeError(f"Possible secret material found in release file: {path}")


def _zip_datetime(epoch: int) -> tuple[int, int, int, int, int, int]:
    minimum = int(dt.datetime(1980, 1, 1, tzinfo=dt.timezone.utc).timestamp())
    maximum = int(dt.datetime(2107, 12, 31, 23, 59, 58, tzinfo=dt.timezone.utc).timestamp())
    safe_epoch = min(max(epoch, minimum), maximum)
    value = dt.datetime.fromtimestamp(safe_epoch, tz=dt.timezone.utc)
    zip_second = value.second - (value.second % 2)
    return (value.year, value.month, value.day, value.hour, value.minute, zip_second)


def _zip_info(path: str, epoch: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, date_time=_zip_datetime(epoch))
    # Source bundles are deliberately stored without DEFLATE so their bytes
    # do not depend on the host zlib version.
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_app_version(files: dict[str, bytes]) -> str:
    match = APP_VERSION_RE.search(files["app/version.py"])
    if not match:
        raise RuntimeError("app/version.py does not contain __version__")
    return match.group(1).decode("ascii")


def build_release_bundle(
    *,
    repo_root: Path,
    commit: str,
    version: str,
    output_dir: Path,
) -> dict[str, object]:
    if not SEMVER_RE.fullmatch(version):
        raise RuntimeError(f"Version must use X.Y.Z form: {version!r}")
    repo_root = repo_root.resolve()
    resolved_commit = resolve_commit(repo_root, commit)
    epoch = commit_epoch(repo_root, resolved_commit)
    entries = list_release_entries(repo_root, resolved_commit)
    source_blobs = read_git_blobs(repo_root, entries)

    files: dict[str, bytes] = {}
    for entry in entries:
        data = normalize_release_bytes(entry.path, source_blobs[entry.path])
        _scan_release_bytes(entry.path, data)
        files[entry.path] = data

    app_version = _read_app_version(files)
    if app_version != version:
        raise RuntimeError(
            f"Release version {version} does not match app/version.py {app_version}"
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "product": PRODUCT_NAME,
        "release_kind": RELEASE_KIND,
        "version": version,
        "commit": resolved_commit,
        "source_date_epoch": epoch,
        "files": [
            {
                "path": path,
                "sha256": _sha256(data),
                "size": len(data),
            }
            for path, data in sorted(files.items())
        ],
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    archive_stem = f"{PRODUCT_NAME}-v{version}-windows-launcher-source"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{archive_stem}.zip"
    checksums_path = output_dir / "SHA256SUMS.txt"
    root_prefix = f"{archive_stem}/"

    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_STORED,
        strict_timestamps=True,
    ) as archive:
        archive.writestr(
            _zip_info(f"{root_prefix}RELEASE-MANIFEST.json", epoch),
            manifest_bytes,
        )
        for path, data in sorted(files.items()):
            archive.writestr(
                _zip_info(f"{root_prefix}{path}", epoch),
                data,
            )

    archive_sha256 = _sha256_file(archive_path)
    checksums_path.write_text(
        f"{archive_sha256}  {archive_path.name}\n",
        encoding="ascii",
        newline="\n",
    )
    verification = verify_release_bundle(
        archive_path=archive_path,
        checksums_path=checksums_path,
        expected_version=version,
        expected_commit=resolved_commit,
    )
    return {
        "archive": str(archive_path),
        "checksums": str(checksums_path),
        "archive_sha256": archive_sha256,
        "commit": resolved_commit,
        "file_count": len(files),
        "verification": verification,
    }


def _validate_archive_member(name: str, root_prefix: str) -> str:
    _validate_relative_path(name)
    if not name.startswith(root_prefix):
        raise RuntimeError(f"Archive member escaped release root: {name}")
    relative = name[len(root_prefix) :]
    _validate_relative_path(relative)
    return relative


def verify_release_bundle(
    *,
    archive_path: Path,
    checksums_path: Path,
    expected_version: str | None = None,
    expected_commit: str | None = None,
) -> dict[str, object]:
    archive_path = archive_path.resolve()
    checksums_path = checksums_path.resolve()
    expected_checksum_line = f"{_sha256_file(archive_path)}  {archive_path.name}"
    checksum_lines = checksums_path.read_text(encoding="ascii").splitlines()
    if checksum_lines != [expected_checksum_line]:
        raise RuntimeError("SHA256SUMS.txt does not match the release archive")

    with zipfile.ZipFile(archive_path, mode="r") as archive:
        infos = archive.infolist()
        if len(infos) > 10_000:
            raise RuntimeError("Release archive contains too many files")
        if sum(info.file_size for info in infos) > 1_000_000_000:
            raise RuntimeError("Release archive exceeds the verification size limit")
        for info in infos:
            unix_mode = (info.external_attr >> 16) & 0o170000
            if unix_mode not in {0, 0o100000}:
                raise RuntimeError(f"Release archive contains a non-file entry: {info.filename}")
            if info.flag_bits & 0x1:
                raise RuntimeError(f"Encrypted release entries are not supported: {info.filename}")
        if archive.testzip() is not None:
            raise RuntimeError("Release archive failed its CRC check")
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise RuntimeError("Release archive contains duplicate paths")
        if not names:
            raise RuntimeError("Release archive is empty")
        roots = {name.split("/", 1)[0] for name in names}
        if len(roots) != 1:
            raise RuntimeError("Release archive must contain one top-level directory")
        root = roots.pop()
        root_prefix = f"{root}/"
        relative_names = {
            _validate_archive_member(name, root_prefix)
            for name in names
        }
        manifest_path = f"{root_prefix}RELEASE-MANIFEST.json"
        if manifest_path not in names:
            raise RuntimeError("Release archive is missing RELEASE-MANIFEST.json")
        manifest = json.loads(archive.read(manifest_path).decode("utf-8"))

        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("Unsupported release manifest schema")
        if manifest.get("product") != PRODUCT_NAME:
            raise RuntimeError("Unexpected release product")
        if manifest.get("release_kind") != RELEASE_KIND:
            raise RuntimeError("Unexpected release kind")
        if expected_version is not None and manifest.get("version") != expected_version:
            raise RuntimeError("Release manifest version mismatch")
        if expected_commit is not None and manifest.get("commit") != expected_commit:
            raise RuntimeError("Release manifest commit mismatch")

        expected_root = (
            f"{PRODUCT_NAME}-v{manifest['version']}-windows-launcher-source"
        )
        if root != expected_root:
            raise RuntimeError(f"Unexpected archive root: {root}")

        manifest_files = manifest.get("files")
        if not isinstance(manifest_files, list):
            raise RuntimeError("Release manifest files must be a list")
        manifest_paths: list[str] = []
        for item in manifest_files:
            if not isinstance(item, dict):
                raise RuntimeError("Malformed release manifest file entry")
            path = item.get("path")
            if not isinstance(path, str):
                raise RuntimeError("Release manifest file path must be a string")
            _validate_allowed_file(path)
            data = archive.read(f"{root_prefix}{path}")
            if item.get("size") != len(data) or item.get("sha256") != _sha256(data):
                raise RuntimeError(f"Release manifest hash mismatch: {path}")
            _scan_release_bytes(path, data)
            if PurePosixPath(path).suffix.lower() in CRLF_SUFFIXES:
                if b"\n" in data.replace(b"\r\n", b""):
                    raise RuntimeError(f"Release script is not CRLF-normalized: {path}")
            manifest_paths.append(path)

        if manifest_paths != sorted(manifest_paths) or len(manifest_paths) != len(
            set(manifest_paths)
        ):
            raise RuntimeError("Release manifest paths are not unique and sorted")
        if set(manifest_paths) | {"RELEASE-MANIFEST.json"} != relative_names:
            raise RuntimeError("Archive contents do not match RELEASE-MANIFEST.json")
        missing = sorted(REQUIRED_BUNDLE_FILES - set(manifest_paths))
        if missing:
            raise RuntimeError(f"Release archive is missing: {', '.join(missing)}")

        epoch = int(manifest["source_date_epoch"])
        expected_time = _zip_datetime(epoch)
        if any(info.date_time != expected_time for info in infos):
            raise RuntimeError("Release archive timestamps are not commit-derived")

        app_version_data = archive.read(f"{root_prefix}app/version.py")
        app_match = APP_VERSION_RE.search(app_version_data)
        if not app_match or app_match.group(1).decode("ascii") != manifest["version"]:
            raise RuntimeError("Bundled app/version.py does not match the manifest")

    return {
        "archive_sha256": expected_checksum_line.split(" ", 1)[0],
        "commit": manifest["commit"],
        "file_count": len(manifest_files),
        "root": root,
        "version": manifest["version"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify the deterministic Windows launcher-source release."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build and verify the release bundle.")
    build.add_argument("--version", required=True)
    build.add_argument("--commit", default="HEAD")
    build.add_argument("--repo-root", type=Path, default=ROOT)
    build.add_argument("--output-dir", type=Path, default=ROOT / "build" / "release")

    verify = subparsers.add_parser("verify", help="Verify an existing release bundle.")
    verify.add_argument("--archive", type=Path, required=True)
    verify.add_argument("--checksums", type=Path, required=True)
    verify.add_argument("--expected-version")
    verify.add_argument("--expected-commit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "build":
            result = build_release_bundle(
                repo_root=args.repo_root,
                commit=args.commit,
                version=args.version,
                output_dir=args.output_dir,
            )
        else:
            result = verify_release_bundle(
                archive_path=args.archive,
                checksums_path=args.checksums,
                expected_version=args.expected_version,
                expected_commit=args.expected_commit,
            )
    except (KeyError, OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        print(f"release bundle error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
