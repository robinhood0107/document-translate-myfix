#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    validate_source_only_manifest_v4,
)


SCHEMA_VERSION = "inpaint-product-policy-overlay-v33"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    body = {key: value for key, value in payload.items() if key != "overlay_sha256"}
    return hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def build_policy_overlay(manifest_path: Path) -> dict[str, object]:
    source_binding = validate_source_only_manifest_v4(manifest_path)
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    pages = source.get("pages")
    if not isinstance(pages, list) or any(
        not isinstance(page, Mapping) for page in pages
    ):
        raise ValueError("v3.3 policy overlay requires source-only pages")
    records: list[dict[str, str]] = []
    for page in pages:
        page_id = str(page.get("page_id") or "").strip()
        instances = page.get("target_instances")
        if not page_id or not isinstance(instances, list):
            raise ValueError("v3.3 policy overlay page inventory is invalid")
        for instance in instances:
            if not isinstance(instance, Mapping):
                raise ValueError("v3.3 policy overlay instance must be an object")
            instance_id = str(instance.get("instance_id") or "").strip()
            priority = str(instance.get("priority") or "").strip().lower()
            action = str(instance.get("processing_action") or "").strip().lower()
            role = str(instance.get("semantic_role") or "").strip().lower()
            if not instance_id:
                raise ValueError("v3.3 policy overlay instance id is empty")
            if priority == "required" and action == "translate_inpaint":
                evaluation_class = "required_translate"
            elif priority == "optional" and action == "preserve":
                # Source-reviewed UI/decorative/SFX content is neutral for the
                # relative product score. Runtime explicit-SFX actions remain
                # fail-closed and are validated independently.
                evaluation_class = "optional_neutral"
            elif priority == "ambiguous" and action == "review":
                evaluation_class = "hard_ambiguous"
            else:
                raise ValueError(
                    f"v3.3 policy overlay has inconsistent source action: "
                    f"{page_id}/{instance_id}"
                )
            records.append(
                {
                    "page_id": page_id,
                    "instance_id": instance_id,
                    "semantic_role": role,
                    "source_processing_action": action,
                    "evaluation_class": evaluation_class,
                }
            )
    identities = [(row["page_id"], row["instance_id"]) for row in records]
    if len(identities) != len(set(identities)):
        raise ValueError("v3.3 policy overlay instance inventory is not unique")
    records.sort(key=lambda row: (row["page_id"], row["instance_id"]))
    counts = {
        value: sum(row["evaluation_class"] == value for row in records)
        for value in (
            "required_translate",
            "optional_neutral",
            "hard_ambiguous",
        )
    }
    overlay: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "policy_id": "recognized-general-text-explicit-sfx-preserve-v1",
        "source_manifest_sha256": str(source_binding["manifest_sha256"]),
        "source_manifest_file_sha256": _sha256(manifest_path),
        "source_page_inventory_sha256": str(
            source_binding["page_inventory_sha256"]
        ),
        "page_count": int(source_binding["page_count"]),
        "candidate_seen": False,
        "annotation_frozen_before_candidate": True,
        "runtime_policy": {
            "explicit_sfx_onomatopoeia_decorative": "preserve",
            "recognized_bubble_free_narration_caption": "translate_inpaint",
            "missing_conflicting_review": "abstain",
            "ocr_or_text_bbox_creates_edit_pixels": False,
        },
        "evaluation_policy": {
            "optional_neutral_affects_pass_fail": False,
            "optional_neutral_is_reported": True,
            "exact_structure_and_ambiguous_remain_hard": True,
        },
        "instance_counts": counts,
        "instances": records,
    }
    overlay["overlay_sha256"] = _canonical_sha256(overlay)
    return overlay


def validate_policy_overlay(
    overlay_path: Path,
    *,
    manifest_path: Path,
) -> dict[str, object]:
    expected = build_policy_overlay(manifest_path)
    actual = json.loads(overlay_path.read_text(encoding="utf-8"))
    if not isinstance(actual, dict) or actual != expected:
        raise ValueError("v3.3 policy overlay differs from sealed source policy")
    if actual.get("overlay_sha256") != _canonical_sha256(actual):
        raise ValueError("v3.3 policy overlay SHA differs")
    return actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seal the source-only v3.3 product scoring policy overlay."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("v3.3 policy overlay output must be fresh")
    payload = build_policy_overlay(args.manifest.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
