from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.utils.download import (  # noqa: E402
    ModelDownloader,
    application_model_profile,
    provision_profile,
)

SCHEMA_VERSION = 1
STATE_DIR = ROOT / ".comic-bootstrap"


class InstallStateError(RuntimeError):
    pass


def _state_path(runtime: str) -> Path:
    return STATE_DIR / f"install-state-{runtime}.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - compatibility with pinned upstream artifacts
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _requirements_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(ROOT).as_posix(),
        "sha256": _sha256_file(resolved),
    }


def _model_records(profile: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for model_id in application_model_profile(profile):
        spec = ModelDownloader.registry[model_id]
        files: list[dict[str, Any]] = []
        for index, remote_name in enumerate(spec.files):
            local_name = (
                spec.save_as.get(remote_name, remote_name)
                if spec.save_as
                else remote_name
            )
            path = Path(spec.save_dir, local_name).resolve()
            if not path.is_file():
                raise InstallStateError(f"Application model is missing: {model_id.value}/{local_name}")
            expected = str(spec.sha256[index] or "").strip().lower()
            algorithm = "sha256" if len(expected) == 64 else "md5" if len(expected) == 32 else ""
            stat = path.stat()
            files.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "digest_algorithm": algorithm,
                    "digest": expected,
                }
            )
        records.append({"id": model_id.value, "files": files})
    return records


def _read_state(runtime: str) -> dict[str, Any]:
    path = _state_path(runtime)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise InstallStateError(
            f"Windows setup is incomplete for {runtime}. Run the matching setup BAT."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise InstallStateError(
            f"Windows setup state is unreadable for {runtime}. Run the matching setup BAT."
        ) from exc
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
        raise InstallStateError(
            f"Windows setup state is obsolete for {runtime}. Run the matching setup BAT."
        )
    return payload


def _validate_application_models(
    payload: dict[str, Any],
    *,
    profile: str,
    allow_digest_fallback: bool,
) -> None:
    expected_ids = {model_id.value for model_id in application_model_profile(profile)}
    records = payload.get("application_models")
    if not isinstance(records, list):
        raise InstallStateError("Application model seal is missing.")
    by_id = {
        str(record.get("id") or ""): record
        for record in records
        if isinstance(record, dict)
    }
    if not expected_ids.issubset(by_id):
        missing = ", ".join(sorted(expected_ids.difference(by_id)))
        raise InstallStateError(f"Application model seal is incomplete: {missing}")

    for model_id in sorted(expected_ids):
        files = by_id[model_id].get("files")
        if not isinstance(files, list) or not files:
            raise InstallStateError(f"Application model seal has no files: {model_id}")
        for record in files:
            if not isinstance(record, dict):
                raise InstallStateError(f"Invalid application model record: {model_id}")
            relative = str(record.get("path") or "")
            path = (ROOT / relative).resolve()
            try:
                path.relative_to(ROOT)
            except ValueError as exc:
                raise InstallStateError(f"Unsafe application model path: {relative}") from exc
            if not path.is_file():
                raise InstallStateError(f"Application model is missing: {relative}")
            stat = path.stat()
            if int(record.get("bytes", -1)) != int(stat.st_size):
                raise InstallStateError(f"Application model size changed: {relative}")
            if int(record.get("mtime_ns", -1)) == int(stat.st_mtime_ns):
                continue
            if not allow_digest_fallback:
                raise InstallStateError(f"Application model metadata changed: {relative}")
            expected = str(record.get("digest") or "").lower()
            algorithm = str(record.get("digest_algorithm") or "").lower()
            if algorithm == "sha256":
                actual = _sha256_file(path)
            elif algorithm == "md5":
                actual = _md5_file(path)
            else:
                raise InstallStateError(f"Application model has no usable digest: {relative}")
            if actual.lower() != expected:
                raise InstallStateError(f"Application model digest changed: {relative}")


def _run_docker(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    return subprocess.run(
        ["docker", *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        creationflags=creationflags,
    )


def _validate_docker_state(payload: dict[str, Any]) -> None:
    image = payload.get("llama_image")
    if not isinstance(image, dict):
        raise InstallStateError("llama.cpp image seal is missing.")
    image_ref = str(image.get("ref") or "")
    image_id = str(image.get("id") or "")
    inspected = _run_docker(["image", "inspect", "--format", "{{.Id}}", image_ref])
    if inspected.returncode != 0:
        raise InstallStateError(
            "Docker Desktop is unavailable or the sealed llama.cpp image is missing. "
            "Run the matching setup BAT."
        )
    if inspected.stdout.strip() != image_id:
        raise InstallStateError(
            "The local llama.cpp image changed after setup. Run the matching setup BAT."
        )

    runtimes = payload.get("managed_runtimes")
    if not isinstance(runtimes, list) or not runtimes:
        raise InstallStateError("Managed runtime seal is missing.")
    for runtime in runtimes:
        if not isinstance(runtime, dict):
            raise InstallStateError("Invalid managed runtime seal.")
        name = str(runtime.get("name") or "")
        expected_runtime = str(runtime.get("runtime_name") or "")
        expected_version = int(runtime.get("preparation_version", 0))
        inspected = _run_docker(
            ["volume", "inspect", "--format", "{{json .Labels}}", name]
        )
        if inspected.returncode != 0:
            raise InstallStateError(f"Managed runtime volume is missing: {name}")
        try:
            labels = json.loads(inspected.stdout)
        except json.JSONDecodeError as exc:
            raise InstallStateError(f"Managed runtime labels are unreadable: {name}") from exc
        if str(labels.get("comic-translate.runtime") or "") != expected_runtime:
            raise InstallStateError(f"Managed runtime ownership changed: {name}")
        if int(labels.get("comic-translate.preparation-version") or 0) != expected_version:
            raise InstallStateError(f"Managed runtime version changed: {name}")


def command_provision(args: argparse.Namespace) -> int:
    requirements = _requirements_record((ROOT / args.requirements).resolve())
    effective_profile = args.profile
    try:
        state = _read_state(args.runtime)
        if str(state.get("provisioned_tier") or "") == "full":
            effective_profile = "full"
        if state.get("requirements") != requirements:
            raise InstallStateError("Requirements fingerprint changed.")
        _validate_application_models(
            state,
            profile=effective_profile,
            allow_digest_fallback=False,
        )
    except InstallStateError:
        provision_profile(
            effective_profile,
            progress_callback=lambda message: print(message, flush=True),
        )
        print("Application model profile prepared and verified.")
    else:
        print("Application model seal is unchanged; model hashing and downloads skipped.")
    return 0


def command_write(args: argparse.Namespace) -> int:
    managed_path = Path(args.managed_state).resolve()
    try:
        managed = json.loads(managed_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InstallStateError("Managed runtime setup state is missing or invalid.") from exc
    if str(managed.get("image_ref") or "") != args.image_ref:
        raise InstallStateError("Managed runtime image reference does not match setup.")
    if str(managed.get("image_id") or "") != args.image_id:
        raise InstallStateError("Managed runtime image identity does not match setup.")

    managed_runtimes = list(managed.get("volumes") or [])
    has_full_runtime = any(
        str(item.get("runtime_name") or "")
        in {"MangaLMM-llama.cpp", "PaddleOCR-VL-Spotting-llama.cpp"}
        for item in managed_runtimes
        if isinstance(item, dict)
    )
    effective_tier = "full" if args.tier == "full" or has_full_runtime else "core"
    effective_profile = "full" if effective_tier == "full" else args.profile
    payload = {
        "schema_version": SCHEMA_VERSION,
        "runtime": args.runtime,
        "provisioned_tier": effective_tier,
        "requirements": _requirements_record((ROOT / args.requirements).resolve()),
        "application_models": _model_records(effective_profile),
        "llama_image": {
            "ref": args.image_ref,
            "id": args.image_id,
            "required_cuda": args.required_cuda,
        },
        "managed_runtimes": managed_runtimes,
    }
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    destination = _state_path(args.runtime)
    handle, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=STATE_DIR,
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, destination)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    print(f"Windows install state sealed: {destination.name}")
    return 0


def command_preflight(args: argparse.Namespace) -> int:
    payload = _read_state(args.runtime)
    if str(payload.get("runtime") or "") != args.runtime:
        raise InstallStateError("Windows setup runtime does not match this launcher.")
    tier = str(payload.get("provisioned_tier") or "")
    if tier not in {"core", "full"}:
        raise InstallStateError("Windows setup tier is invalid.")
    expected_requirements = _requirements_record((ROOT / args.requirements).resolve())
    if payload.get("requirements") != expected_requirements:
        raise InstallStateError("Pinned requirements changed. Run the matching setup BAT.")
    _validate_application_models(payload, profile="core", allow_digest_fallback=True)
    _validate_docker_state(payload)
    image_ref = str(payload["llama_image"]["ref"])
    if args.emit_cmd:
        print(f"LLAMA_CPP_IMAGE={image_ref}")
        print(f"COMIC_WINDOWS_RUNTIME={args.runtime}")
        print("COMIC_MODEL_DOWNLOAD_POLICY=forbid")
        print("HF_HUB_OFFLINE=1")
        print("TRANSFORMERS_OFFLINE=1")
    else:
        print(f"Windows install state is ready: {args.runtime}/{tier}/{image_ref}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    provision = subparsers.add_parser("provision")
    provision.add_argument("--runtime", choices=("cuda12", "cuda13"), required=True)
    provision.add_argument("--profile", choices=("core", "full"), default="core")
    provision.add_argument("--requirements", required=True)
    provision.set_defaults(handler=command_provision)

    write = subparsers.add_parser("write")
    write.add_argument("--runtime", choices=("cuda12", "cuda13"), required=True)
    write.add_argument("--tier", choices=("core", "full"), required=True)
    write.add_argument("--profile", choices=("core", "full"), default="core")
    write.add_argument("--requirements", required=True)
    write.add_argument("--image-ref", required=True)
    write.add_argument("--image-id", required=True)
    write.add_argument("--required-cuda", default="")
    write.add_argument("--managed-state", required=True)
    write.set_defaults(handler=command_write)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--runtime", choices=("cuda12", "cuda13"), required=True)
    preflight.add_argument("--requirements", required=True)
    preflight.add_argument("--emit-cmd", action="store_true")
    preflight.set_defaults(handler=command_preflight)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.handler(args))
    except InstallStateError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
