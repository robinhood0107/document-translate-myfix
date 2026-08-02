#!/usr/bin/env python3
"""Create and verify the canonical HunyuanOCR router lab volume without deletion.

The legacy bind mount and legacy named volume remain untouched.  This tool
copies only the two pinned Q8 artifacts to a new named volume and records the
identity in the private validation archive.  It never removes or overwrites a
model volume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import validation_artifact_harness as artifact_harness


FAMILY = "llamacpp-router-handoff"
CATEGORY = "60-runtime-release"
VOLUME = "comic-translate-hunyuanocr-models-v1"
LABEL_READY = "com.comictranslate.hunyuanocr.router-ready"
LABEL_PROTOCOL = "com.comictranslate.hunyuanocr.router-protocol"
PROTOCOL = "llamacpp-router-handoff-v1"
HELPER_IMAGE = "alpine:3.22"
FILES = {
    "HunyuanOCR.Q8_0.gguf": {
        "bytes": 577_949_408,
        "sha256": "cdafc794cafeae377868d7a40a70e282a737e39abe77c0d8b73614447b364a21",
    },
    "HunyuanOCR.mmproj-Q8_0.gguf": {
        "bytes": 732_938_240,
        "sha256": "b77913164ff73d4c0dc4d994e236ed72bacbbe5c5db1ec9b2828627b46c32804",
    },
}


class HunyuanVolumeError(RuntimeError):
    pass


def _run(command: Sequence[str], *, timeout: float = 180.0, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(command),
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        raise HunyuanVolumeError(
            f"Command failed ({completed.returncode}): {Path(command[0]).name}\n"
            f"{(completed.stderr or completed.stdout)[-4096:]}"
        )
    return completed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_manifest(source: Path) -> dict[str, Any]:
    source = source.resolve()
    if not source.is_dir():
        raise HunyuanVolumeError("Hunyuan source directory is unavailable.")
    records: dict[str, Any] = {}
    for name, expected in FILES.items():
        path = source / name
        if not path.is_file():
            raise HunyuanVolumeError(f"Hunyuan source file is missing: {name}")
        observed = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        if observed != expected:
            raise HunyuanVolumeError(f"Hunyuan source identity mismatch: {name}")
        records[name] = observed
    return {"protocol": PROTOCOL, "files": records}


def _volume_exists() -> bool:
    return _run(["docker", "volume", "inspect", VOLUME], check=False, timeout=20.0).returncode == 0


def _volume_helper(script: str, *, source: Path | None = None) -> subprocess.CompletedProcess[str]:
    command = ["docker", "run", "--rm", "--pull", "never"]
    if source is not None:
        command.extend(
            [
                "--mount",
                f"type=bind,source={source.resolve()},target=/source,readonly",
            ]
        )
    command.extend(
        [
            "--mount",
            f"type=volume,source={VOLUME},target=/models,readonly=false",
            HELPER_IMAGE,
            "sh",
            "-ec",
            script,
        ]
    )
    return _run(command, timeout=900.0, check=True)


def _verify_volume() -> dict[str, Any]:
    if not _volume_exists():
        raise HunyuanVolumeError("Canonical Hunyuan router volume is missing.")
    checks = " && ".join(
        [
            f"test $(wc -c < /models/{name}) -eq {item['bytes']}"
            f" && test $(sha256sum /models/{name} | awk '{{print $1}}') = {item['sha256']}"
            for name, item in FILES.items()
        ]
        + ["test -f /models/.comic-translate-hunyuanocr-router-ready-v1.json"]
    )
    _volume_helper(checks)
    labels_raw = _run(["docker", "volume", "inspect", VOLUME, "--format", "{{json .Labels}}"], timeout=20.0).stdout
    try:
        labels = json.loads(labels_raw)
    except json.JSONDecodeError as exc:
        raise HunyuanVolumeError("Canonical Hunyuan volume labels are invalid.") from exc
    if not isinstance(labels, Mapping) or labels.get(LABEL_READY) != "v1" or labels.get(LABEL_PROTOCOL) != PROTOCOL:
        raise HunyuanVolumeError("Canonical Hunyuan volume labels do not match the router contract.")
    return {"volume": VOLUME, "verified": True, "files": FILES, "labels": dict(labels)}


def prepare(source: Path) -> dict[str, Any]:
    manifest = _source_manifest(source)
    created = False
    if not _volume_exists():
        _run(
            [
                "docker",
                "volume",
                "create",
                "--label",
                f"{LABEL_READY}=v1",
                "--label",
                f"{LABEL_PROTOCOL}={PROTOCOL}",
                VOLUME,
            ],
            timeout=30.0,
        )
        created = True
    if created:
        copies = " && ".join(
            [
                f"test -f /source/{name} && test ! -e /models/{name} && cp -p /source/{name} /models/{name}"
                for name in FILES
            ]
        )
        ready = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
        _volume_helper(
            copies
            + " && printf '%s\\n' '"
            + ready.replace("'", "'\\''")
            + "' > /models/.comic-translate-hunyuanocr-router-ready-v1.json",
            source=source,
        )
    verified = _verify_volume()
    return {"prepared": created, **verified}


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare only the named HunyuanOCR router lab volume.")
    parser.add_argument("--mode", choices=("prepare", "verify"), required=True)
    parser.add_argument("--source-dir", type=Path, help="Existing private Q8 source directory; required for prepare.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Private explicit output override.")
    args = parser.parse_args(argv)
    if args.mode == "prepare" and args.source_dir is None:
        parser.error("--source-dir is required for --mode prepare")
    output_root, managed = artifact_harness.select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    try:
        result = prepare(args.source_dir.expanduser().resolve()) if args.mode == "prepare" else _verify_volume()
        _write(output_root / "hunyuanocr-router-volume.json", result)
        if managed is not None:
            managed.complete(metadata={"mode": args.mode, "prepared": bool(result.get("prepared", False))})
    except BaseException as exc:
        if managed is not None:
            managed.fail(exc, metadata={"mode": args.mode})
        raise
    print(json.dumps({"volume": VOLUME, "verified": True, "prepared": bool(result.get("prepared", False))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
