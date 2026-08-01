#!/usr/bin/env python3
"""Remove only manifest-approved retired Paddle relay/vLLM Docker assets.

The command is dry-run by default and never performs a broad prune.  Image
removal is allowed only after immutable-ID resolution and only when no Docker
container still references that exact image.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT_DIR / "docs" / "runtime" / "obsolete-vllm-runtime-manifest.json"
)


class LegacyRuntimeRetirementError(RuntimeError):
    pass


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise LegacyRuntimeRetirementError("Unsupported retirement manifest.")
    return payload


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inspect_container(name: str) -> dict[str, Any] | None:
    completed = _docker("inspect", name)
    detail = ((completed.stdout or "") + (completed.stderr or "")).lower()
    if completed.returncode != 0 and (
        "no such object" in detail or "no such container" in detail
    ):
        return None
    if completed.returncode != 0:
        raise LegacyRuntimeRetirementError(
            f"Unable to inspect {name}: {detail.strip()}"
        )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list) or len(payload) != 1:
        raise LegacyRuntimeRetirementError(
            f"Unexpected Docker inspect payload for {name}."
        )
    return payload[0]


def _inspect_image(reference: str) -> dict[str, Any] | None:
    completed = _docker("image", "inspect", reference)
    detail = ((completed.stdout or "") + (completed.stderr or "")).lower()
    if completed.returncode != 0 and (
        "no such object" in detail
        or "no such image" in detail
        or "not found" in detail
    ):
        return None
    if completed.returncode != 0:
        raise LegacyRuntimeRetirementError(
            f"Unable to inspect image {reference}: {detail.strip()}"
        )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list) or len(payload) != 1:
        raise LegacyRuntimeRetirementError(
            f"Unexpected Docker image inspect payload for {reference}."
        )
    return payload[0]


def _containers_referencing_image(image_id: str) -> list[str]:
    completed = _docker(
        "ps",
        "-a",
        "--no-trunc",
        "--filter",
        f"ancestor={image_id}",
        "--format",
        "{{.ID}}|{{.Names}}",
    )
    if completed.returncode != 0:
        raise LegacyRuntimeRetirementError(
            "Unable to inspect containers that reference retired image "
            f"{image_id}: {(completed.stderr or '').strip()}"
        )
    return [
        line.strip()
        for line in (completed.stdout or "").splitlines()
        if line.strip()
    ]


def _validate_owned_container(
    inspected: dict[str, Any],
    specification: dict[str, Any],
) -> tuple[str, str]:
    container_id = str(inspected.get("Id") or "").strip()
    if not container_id:
        raise LegacyRuntimeRetirementError(
            f"Docker inspect did not return an ID for {specification.get('name')}."
        )
    configured_image = str(
        (inspected.get("Config") or {}).get("Image") or ""
    )
    if configured_image != str(specification.get("expected_image") or ""):
        raise LegacyRuntimeRetirementError(
            "Refusing to remove a container whose image does not match the "
            f"manifest: {specification.get('name')}"
        )
    labels = (inspected.get("Config") or {}).get("Labels") or {}
    required_label = str(specification.get("required_label") or "")
    label_value = str(labels.get(required_label) or "").strip()
    if not label_value:
        raise LegacyRuntimeRetirementError(
            "Refusing to remove a container without a non-empty historical product "
            f"label: {specification.get('name')}"
        )
    expected_container_id = str(
        specification.get("container_id") or ""
    ).strip()
    if expected_container_id and expected_container_id != container_id:
        raise LegacyRuntimeRetirementError(
            "Refusing to remove a container whose immutable ID changed after "
            f"snapshot: {specification.get('name')}"
        )
    expected_label_value = str(
        specification.get("expected_label_value") or ""
    ).strip()
    if expected_label_value and expected_label_value != label_value:
        raise LegacyRuntimeRetirementError(
            "Refusing to remove a container whose product label changed after "
            f"snapshot: {specification.get('name')}"
        )
    return container_id, label_value


def resolve_manifest(path: Path) -> dict[str, Any]:
    """Capture immutable IDs and label values before any deletion is allowed."""

    manifest = copy.deepcopy(_load_manifest(path))
    containers = manifest.get("containers", [])
    if not isinstance(containers, list):
        raise LegacyRuntimeRetirementError(
            "Retirement manifest containers must be a list."
        )
    resolved_containers: list[dict[str, Any]] = []
    for raw_specification in containers:
        if not isinstance(raw_specification, dict):
            raise LegacyRuntimeRetirementError(
                "Retirement manifest container entries must be objects."
            )
        specification = dict(raw_specification)
        name = str(specification.get("name") or "").strip()
        if not name:
            raise LegacyRuntimeRetirementError(
                "Retirement manifest contains an empty container name."
            )
        inspected = _inspect_container(name)
        if inspected is None:
            continue
        container_id, label_value = _validate_owned_container(
            inspected,
            specification,
        )
        specification["container_id"] = container_id
        specification["expected_label_value"] = label_value
        resolved_containers.append(specification)
    manifest["containers"] = resolved_containers
    images = manifest.get("images", [])
    if not isinstance(images, list):
        raise LegacyRuntimeRetirementError(
            "Retirement manifest images must be a list."
        )
    resolved_images: list[dict[str, Any]] = []
    for raw_specification in images:
        if not isinstance(raw_specification, dict):
            raise LegacyRuntimeRetirementError(
                "Retirement manifest image entries must be objects."
            )
        specification = dict(raw_specification)
        reference = str(specification.get("reference") or "").strip()
        if not reference:
            raise LegacyRuntimeRetirementError(
                "Retirement manifest contains an empty image reference."
            )
        inspected = _inspect_image(reference)
        if inspected is None:
            continue
        image_id = str(inspected.get("Id") or "").strip()
        if not image_id:
            raise LegacyRuntimeRetirementError(
                f"Docker inspect returned no immutable ID for {reference}."
            )
        specification["image_id"] = image_id
        resolved_images.append(specification)
    manifest["images"] = resolved_images
    manifest["resolved_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["source_manifest_sha256"] = _manifest_sha256(path)
    return manifest


def retire_manifest(path: Path, *, execute: bool) -> list[dict[str, str]]:
    manifest = _load_manifest(path)
    containers = manifest.get("containers", [])
    if not isinstance(containers, list):
        raise LegacyRuntimeRetirementError(
            "Retirement manifest containers must be a list."
        )
    actions: list[dict[str, str]] = []
    for raw_specification in containers:
        if not isinstance(raw_specification, dict):
            raise LegacyRuntimeRetirementError(
                "Retirement manifest container entries must be objects."
            )
        specification = dict(raw_specification)
        name = str(specification.get("name") or "").strip()
        if not name:
            raise LegacyRuntimeRetirementError(
                "Retirement manifest contains an empty container name."
            )
        inspected = _inspect_container(name)
        if inspected is None:
            actions.append({"container": name, "status": "absent"})
            continue
        container_id, _label_value = _validate_owned_container(
            inspected,
            specification,
        )
        if not execute:
            actions.append(
                {
                    "container": name,
                    "container_id": container_id,
                    "status": "would-remove",
                }
            )
            continue
        if not specification.get("container_id") or not specification.get(
            "expected_label_value"
        ):
            raise LegacyRuntimeRetirementError(
                "Execution requires a resolved manifest with immutable container "
                "ID and label value. Create it with --snapshot-output first."
            )
        running = bool((inspected.get("State") or {}).get("Running"))
        if running:
            stopped = _docker("stop", "--time", "30", container_id)
            if stopped.returncode != 0:
                raise LegacyRuntimeRetirementError(
                    f"Unable to stop {name}: {(stopped.stderr or '').strip()}"
                )
        removed = _docker("rm", container_id)
        if removed.returncode != 0:
            raise LegacyRuntimeRetirementError(
                f"Unable to remove {name}: {(removed.stderr or '').strip()}"
            )
        actions.append(
            {
                "container": name,
                "container_id": container_id,
                "status": "removed",
            }
        )
    images = manifest.get("images", [])
    if not isinstance(images, list):
        raise LegacyRuntimeRetirementError(
            "Retirement manifest images must be a list."
        )
    for raw_specification in images:
        if not isinstance(raw_specification, dict):
            raise LegacyRuntimeRetirementError(
                "Retirement manifest image entries must be objects."
            )
        specification = dict(raw_specification)
        reference = str(specification.get("reference") or "").strip()
        action = str(specification.get("action") or "preserve").strip()
        if not reference:
            raise LegacyRuntimeRetirementError(
                "Retirement manifest contains an empty image reference."
            )
        if action == "preserve":
            actions.append({"image": reference, "status": "preserved"})
            continue
        if action != "remove-if-unreferenced":
            raise LegacyRuntimeRetirementError(
                f"Unsupported image retirement action: {action}"
            )
        inspected = _inspect_image(reference)
        if inspected is None:
            actions.append({"image": reference, "status": "absent"})
            continue
        image_id = str(inspected.get("Id") or "").strip()
        expected_image_id = str(specification.get("image_id") or "").strip()
        if execute and not expected_image_id:
            raise LegacyRuntimeRetirementError(
                "Execution requires a resolved manifest with an immutable "
                f"image ID: {reference}"
            )
        if expected_image_id and image_id != expected_image_id:
            raise LegacyRuntimeRetirementError(
                "Refusing to remove a Docker image whose immutable ID changed "
                f"after snapshot: {reference}"
            )
        references = _containers_referencing_image(image_id)
        if references:
            actions.append(
                {
                    "image": reference,
                    "image_id": image_id,
                    "status": "preserved-referenced",
                    "references": ",".join(references),
                }
            )
            continue
        if not execute:
            actions.append(
                {
                    "image": reference,
                    "image_id": image_id,
                    "status": "would-remove",
                }
            )
            continue
        removed = _docker("image", "rm", image_id)
        if removed.returncode != 0:
            raise LegacyRuntimeRetirementError(
                f"Unable to remove image {reference}: "
                f"{(removed.stderr or '').strip()}"
            )
        actions.append(
            {
                "image": reference,
                "image_id": image_id,
                "status": "removed",
            }
        )
    return actions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--snapshot-output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        if args.snapshot_output and args.execute:
            raise LegacyRuntimeRetirementError(
                "--snapshot-output and --execute cannot be used together."
            )
        if args.snapshot_output:
            resolved = resolve_manifest(args.manifest)
            args.snapshot_output.write_text(
                json.dumps(resolved, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            actions: Any = {
                "resolved_manifest": str(args.snapshot_output),
                "containers": len(resolved.get("containers", [])),
            }
        else:
            actions = retire_manifest(args.manifest, execute=bool(args.execute))
    except (LegacyRuntimeRetirementError, OSError, json.JSONDecodeError) as exc:
        print(f"Legacy vLLM retirement failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(actions, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
