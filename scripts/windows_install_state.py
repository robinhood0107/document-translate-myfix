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
    FULL_ONLY_APPLICATION_MODELS,
    ModelDownloader,
    ModelID,
    application_model_profile,
    provision_model_ids,
)

SCHEMA_VERSION = 1
STATE_DIR = ROOT / ".comic-bootstrap"
CORE_MANAGED_RUNTIMES = frozenset(
    {"HunyuanOCR-llama.cpp", "PaddleOCR-VL-llama.cpp", "Gemma"}
)
FULL_MANAGED_RUNTIMES = CORE_MANAGED_RUNTIMES | frozenset(
    {"MangaLMM-llama.cpp", "PaddleOCR-VL-Spotting-llama.cpp"}
)
EXPECTED_IMAGE_BY_RUNTIME = {
    "cuda12": "ghcr.io/ggml-org/llama.cpp:server-cuda",
    "cuda13": "ghcr.io/ggml-org/llama.cpp:server-cuda13",
}


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


def _ordered_model_ids(model_ids: set[ModelID]) -> tuple[ModelID, ...]:
    return tuple(model_id for model_id in ModelID if model_id in model_ids)


def _model_file_pins(model_id: ModelID) -> list[dict[str, str]]:
    spec = ModelDownloader.registry[model_id]
    if len(spec.files) != len(spec.sha256):
        raise InstallStateError(f"Application model pin count is invalid: {model_id.value}")
    pins: list[dict[str, str]] = []
    for index, remote_name in enumerate(spec.files):
        local_name = (
            spec.save_as.get(remote_name, remote_name)
            if spec.save_as
            else remote_name
        )
        path = Path(spec.save_dir, local_name).resolve()
        try:
            relative = path.relative_to(ROOT).as_posix()
        except ValueError as exc:
            raise InstallStateError(
                f"Unsafe registered application model path: {path}"
            ) from exc
        expected = str(spec.sha256[index] or "").strip().lower()
        pins.append({
            "path": relative,
            "digest_algorithm": (
                "sha256" if len(expected) == 64 else "md5" if len(expected) == 32 else ""
            ),
            "digest": expected,
        })
    return pins


def _recorded_application_model_ids(
    payload: dict[str, Any],
    *,
    required_ids: set[ModelID] | None = None,
) -> set[ModelID]:
    records = payload.get("application_models")
    if not isinstance(records, list):
        raise InstallStateError("Application model seal is missing.")
    known_models = {model_id.value: model_id for model_id in ModelID}
    model_ids: set[ModelID] = set()
    seen_ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise InstallStateError("Invalid application model record.")
        model_id = str(record.get("id") or "")
        if model_id not in known_models:
            raise InstallStateError(f"Unknown application model ID: {model_id or '<missing>'}")
        if model_id in seen_ids:
            raise InstallStateError(f"Duplicate application model ID: {model_id}")
        seen_ids.add(model_id)
        model_ids.add(known_models[model_id])

    if required_ids is not None:
        missing_required = {
            model_id.value
            for model_id in required_ids
            if model_id not in model_ids
        }
        if missing_required:
            raise InstallStateError(
                "Application model seal is incomplete: "
                + ", ".join(sorted(missing_required))
            )
    return model_ids


def _model_records(
    profile: str,
    *,
    model_ids: set[ModelID] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    selected_models = (
        application_model_profile(profile)
        if model_ids is None
        else _ordered_model_ids(model_ids)
    )
    for model_id in selected_models:
        files: list[dict[str, Any]] = []
        for pin in _model_file_pins(model_id):
            path = ROOT / pin["path"]
            if not path.is_file():
                raise InstallStateError(
                    f"Application model is missing: {model_id.value}/{path.name}"
                )
            stat = path.stat()
            files.append(
                {
                    "path": pin["path"],
                    "bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "digest_algorithm": pin["digest_algorithm"],
                    "digest": pin["digest"],
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
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InstallStateError(
            f"Windows setup state is unreadable for {runtime}. Run the matching setup BAT."
        ) from exc
    if not isinstance(payload, dict):
        raise InstallStateError(
            f"Windows setup state is obsolete for {runtime}. Run the matching setup BAT."
        )
    schema_version = payload.get("schema_version", 0)
    if isinstance(schema_version, bool) or not isinstance(schema_version, (int, str)):
        raise InstallStateError(
            f"Windows setup state is obsolete for {runtime}. Run the matching setup BAT."
        )
    try:
        schema_version = int(schema_version)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InstallStateError(
            f"Windows setup state is obsolete for {runtime}. Run the matching setup BAT."
        ) from exc
    if schema_version != SCHEMA_VERSION:
        raise InstallStateError(
            f"Windows setup state is obsolete for {runtime}. Run the matching setup BAT."
        )
    return payload


def _validate_application_models(
    payload: dict[str, Any],
    *,
    profile: str | None = None,
    model_ids: set[str] | None = None,
    allow_digest_fallback: bool,
) -> None:
    expected_ids = (
        {model_id.value for model_id in application_model_profile(profile or "core")}
        if model_ids is None
        else set(model_ids)
    )
    records = payload.get("application_models")
    if not isinstance(records, list):
        raise InstallStateError("Application model seal is missing.")
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise InstallStateError("Invalid application model record.")
        model_id = str(record.get("id") or "")
        if not model_id:
            raise InstallStateError("Invalid application model ID.")
        if model_id in by_id:
            raise InstallStateError(f"Duplicate application model ID: {model_id}")
        by_id[model_id] = record
    if not expected_ids.issubset(by_id):
        missing = ", ".join(sorted(expected_ids.difference(by_id)))
        raise InstallStateError(f"Application model seal is incomplete: {missing}")

    for model_id in sorted(expected_ids):
        files = by_id[model_id].get("files")
        if not isinstance(files, list) or not files:
            raise InstallStateError(f"Application model seal has no files: {model_id}")
        registered_pins: list[dict[str, str]] | None = None
        if model_id in {item.value for item in FULL_ONLY_APPLICATION_MODELS}:
            registered_pins = _model_file_pins(ModelID(model_id))
            if len(files) != len(registered_pins):
                raise InstallStateError(
                    f"Application model seal shape changed: {model_id}"
                )
        for index, record in enumerate(files):
            if not isinstance(record, dict):
                raise InstallStateError(f"Invalid application model record: {model_id}")
            relative = str(record.get("path") or "")
            if registered_pins is not None:
                pin = registered_pins[index]
                if any(record.get(key) != pin[key] for key in pin):
                    raise InstallStateError(
                        f"Application model registry pin changed: {relative or model_id}"
                    )
            path = (ROOT / relative).resolve()
            try:
                path.relative_to(ROOT)
            except ValueError as exc:
                raise InstallStateError(f"Unsafe application model path: {relative}") from exc
            if not path.is_file():
                raise InstallStateError(f"Application model is missing: {relative}")
            stat = path.stat()
            recorded_bytes = record.get("bytes")
            recorded_mtime = record.get("mtime_ns")
            if (
                isinstance(recorded_bytes, bool)
                or not isinstance(recorded_bytes, int)
                or isinstance(recorded_mtime, bool)
                or not isinstance(recorded_mtime, int)
            ):
                raise InstallStateError(
                    f"Application model metadata is invalid: {relative}"
                )
            if recorded_bytes != stat.st_size:
                raise InstallStateError(f"Application model size changed: {relative}")
            if recorded_mtime == stat.st_mtime_ns:
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


def _validate_managed_runtime_records(
    payload: dict[str, Any],
    *,
    tier: str,
) -> list[dict[str, Any]]:
    runtimes = payload.get("managed_runtimes")
    if not isinstance(runtimes, list) or not runtimes:
        raise InstallStateError("Managed runtime seal is missing.")
    records = [runtime for runtime in runtimes if isinstance(runtime, dict)]
    if len(records) != len(runtimes):
        raise InstallStateError("Invalid managed runtime seal.")
    expected = FULL_MANAGED_RUNTIMES if tier == "full" else CORE_MANAGED_RUNTIMES
    sealed = {str(runtime.get("runtime_name") or "") for runtime in records}
    missing = sorted(expected.difference(sealed))
    if missing:
        raise InstallStateError(
            "Managed runtime seal is incomplete: " + ", ".join(missing)
        )
    return records


def _validate_docker_state(payload: dict[str, Any], *, tier: str) -> None:
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

    for runtime in _validate_managed_runtime_records(payload, tier=tier):
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
    requested_ids = set(application_model_profile(args.profile))
    sealed_ids: set[str] = set()
    previous_state: dict[str, Any] | None = None
    try:
        previous_state = _read_state(args.runtime)
    except InstallStateError:
        pass

    if previous_state is not None:
        try:
            previous_ids = _recorded_application_model_ids(previous_state)
        except InstallStateError:
            previous_ids = set()
        for model_id in previous_ids:
            try:
                _validate_application_models(
                    previous_state,
                    model_ids={model_id.value},
                    allow_digest_fallback=False,
                )
            except InstallStateError:
                continue
            sealed_ids.add(model_id.value)

    missing_ids = _ordered_model_ids(
        {model_id for model_id in requested_ids if model_id.value not in sealed_ids}
    )
    if missing_ids:
        provision_model_ids(
            missing_ids,
            progress_callback=lambda message: print(message, flush=True),
        )
        print("Application model additions prepared and verified.")
    else:
        print("Application model seal is unchanged; model hashing and downloads skipped.")
    return 0


def command_write(args: argparse.Namespace) -> int:
    _assert_runtime_image(args.runtime, args.image_ref)
    previous_state: dict[str, Any] | None = None
    previous_state_path = _state_path(args.runtime)
    if previous_state_path.is_file():
        try:
            previous_state = _read_state(args.runtime)
        except InstallStateError:
            previous_state = None
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
    seal_model_ids = set(application_model_profile(args.profile))
    if args.profile == "core" and previous_state is not None:
        try:
            previous_ids = _recorded_application_model_ids(previous_state)
        except InstallStateError:
            previous_ids = set()
        for model_id in previous_ids:
            try:
                _validate_application_models(
                    previous_state,
                    model_ids={model_id.value},
                    allow_digest_fallback=False,
                )
            except InstallStateError:
                continue
            seal_model_ids.add(model_id)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "runtime": args.runtime,
        "provisioned_tier": effective_tier,
        "requirements": _requirements_record((ROOT / args.requirements).resolve()),
        "application_models": _model_records(
            args.profile,
            model_ids=seal_model_ids,
        ),
        "llama_image": {
            "ref": args.image_ref,
            "id": args.image_id,
            "required_cuda": args.required_cuda,
        },
        "managed_runtimes": managed_runtimes,
    }
    _validate_managed_runtime_records(payload, tier=effective_tier)
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
    sealed_image = payload.get("llama_image")
    image_ref = (
        str(sealed_image.get("ref") or "")
        if isinstance(sealed_image, dict)
        else ""
    )
    _assert_runtime_image(args.runtime, image_ref)
    required_model_ids = set(application_model_profile("core"))
    if tier == "full":
        required_model_ids = set(application_model_profile("full")) - set(
            FULL_ONLY_APPLICATION_MODELS
        )
    sealed_model_ids = _recorded_application_model_ids(
        payload,
        required_ids=required_model_ids,
    )
    for model_id in _ordered_model_ids(sealed_model_ids):
        try:
            _validate_application_models(
                payload,
                model_ids={model_id.value},
                allow_digest_fallback=True,
            )
        except InstallStateError as exc:
            if model_id in FULL_ONLY_APPLICATION_MODELS:
                raise InstallStateError(
                    f"{exc} Run the matching setup_full BAT."
                ) from exc
            raise
    _validate_docker_state(payload, tier=tier)
    if args.emit_cmd:
        print(f"LLAMA_CPP_IMAGE={image_ref}")
        print(f"COMIC_WINDOWS_RUNTIME={args.runtime}")
        print(f"COMIC_WINDOWS_INSTALL_TIER={tier}")
        print("COMIC_MODEL_DOWNLOAD_POLICY=forbid")
        print("HF_HUB_OFFLINE=1")
        print("TRANSFORMERS_OFFLINE=1")
    else:
        print(f"Windows install state is ready: {args.runtime}/{tier}/{image_ref}")
    return 0


def _assert_runtime_image(runtime: str, image_ref: str) -> None:
    expected = EXPECTED_IMAGE_BY_RUNTIME[runtime]
    if image_ref != expected:
        raise InstallStateError(
            f"{runtime.upper()} setup requires llama.cpp image {expected}; "
            f"found {image_ref or '<missing>'}. Run the matching setup BAT."
        )


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
