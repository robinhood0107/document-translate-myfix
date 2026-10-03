from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import scripts.build_inpaint_source_routing_overlay_v34 as routing_builder
from scripts.build_inpaint_glyph_visual_manifest_v34 import (
    VisualManifestError,
    _require_same_source_pixels,
    _semantic_action,
)
from benchmarking.inpaint_detector_bakeoff.contracts import mask_sha256
from scripts.benchmark_inpaint_glyph_refinement_v34 import (
    load_source_region_evidence,
    validate_source_routing_overlay,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.mark.parametrize(
    "block",
    [
        {"processing_action": "translate_inpaint", "semantic_role": "sfx", "text_class": "sfx"},
        {"processing_action": "preserve", "semantic_role": "dialogue_free", "text_class": "text_free"},
        {"semantic_role": "dialogue_free", "text_class": "sfx"},
    ],
)
def test_visual_source_semantic_conflicts_abstain(block: dict[str, object]) -> None:
    assert _semantic_action(block)[0] == "abstain"


def test_visual_manifest_rejects_different_debug_source_pixels() -> None:
    source = np.full((6, 8, 3), 170, dtype=np.uint8)
    debug = source.copy()
    _require_same_source_pixels(source, debug, page_id="neutral-page")
    debug[2, 3] = 171
    with pytest.raises(VisualManifestError, match="product source pixels differ"):
        _require_same_source_pixels(source, debug, page_id="neutral-page")


def _write_mask(path: Path, mask: np.ndarray) -> None:
    encoded, buffer = cv2.imencode(".png", mask)
    assert encoded
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.tobytes())


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_sha = "1" * 64
    inventory_sha = "2" * 64
    ownership = np.zeros((8, 8), dtype=np.uint8)
    ownership[1:7, 1:7] = 255
    zero = np.zeros((8, 8), dtype=np.uint8)
    masks = {
        "ownership": ownership,
        "protected": zero,
        "ambiguous": zero,
        "corner": zero,
    }
    paths: dict[str, Path] = {}
    for role, mask in masks.items():
        path = tmp_path / "explicit-source-routing" / f"{role}.png"
        _write_mask(path, mask)
        paths[role] = path
    raw_region = {
        "region_id": "region-0",
        "ownership_mask": str(paths["ownership"]),
        "proposal": {"text_class": "text_free"},
    }
    source = {
        "pages": [
            {
                "page_id": "neutral-page",
                "source_sha256": source_sha,
                "regions": [{"region_id": "region-0"}],
            }
        ]
    }
    relative = {
        "pages": [
            {
                "page_id": "neutral-page",
                "source_sha256": source_sha,
                "regions": [raw_region],
            }
        ]
    }
    source_path = tmp_path / "source.json"
    relative_path = tmp_path / "relative.json"
    source_path.write_text(json.dumps(source, sort_keys=True), encoding="utf-8")
    relative_path.write_text(json.dumps(relative, sort_keys=True), encoding="utf-8")
    source_seal_path = source_path.with_suffix(".json.seal.json")
    relative_seal_path = relative_path.with_suffix(".json.seal.json")
    source_seal_path.write_text(json.dumps({"source": True}), encoding="utf-8")
    relative_seal_path.write_text(
        json.dumps({"source_manifest_sha256": _sha(source_path)}),
        encoding="utf-8",
    )
    bindings = {
        source_path.resolve(): {
            "manifest_sha256": _sha(source_path),
            "seal_sha256": _sha(source_seal_path),
            "page_inventory_sha256": inventory_sha,
        },
        relative_path.resolve(): {
            "manifest_sha256": _sha(relative_path),
            "seal_sha256": _sha(relative_seal_path),
            "page_inventory_sha256": inventory_sha,
        },
    }
    monkeypatch.setattr(
        routing_builder,
        "validate_source_only_manifest_v4",
        lambda path: bindings[Path(path).resolve()],
    )
    evidence: dict[str, object] = {
        "schema_version": routing_builder.SOURCE_EVIDENCE_SCHEMA_VERSION,
        "source_manifest_sha256": _sha(source_path),
        "relative_manifest_sha256": _sha(relative_path),
        "source_page_inventory_sha256": inventory_sha,
        "candidate_generated": False,
        "candidate_seen": False,
        "annotation_frozen_before_candidate": True,
        "producer": "neutral-source-routing-export",
        "pages": [
            {
                "page_id": "neutral-page",
                "source_sha256": source_sha,
                "candidate_generated": False,
                "candidate_seen": False,
                "annotation_frozen_before_candidate": True,
                "regions": [
                    {
                        "region_id": "region-0",
                        "source_region_record_sha256": _canonical(raw_region),
                        "candidate_generated": False,
                        "candidate_seen": False,
                        "annotation_frozen_before_candidate": True,
                        "artifacts": {
                            role: {
                                "path": str(path),
                                "file_sha256": _sha(path),
                                "pixel_sha256": mask_sha256(masks[role]),
                            }
                            for role, path in paths.items()
                        },
                    }
                ],
            }
        ],
    }
    evidence["evidence_sha256"] = routing_builder._unsigned_sha256(
        evidence, "evidence_sha256"
    )
    evidence_path = tmp_path / "source-routing-evidence.json"
    evidence_path.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")
    return {
        "source": source,
        "relative": relative,
        "source_path": source_path,
        "relative_path": relative_path,
        "source_binding": bindings[source_path.resolve()],
        "relative_binding": bindings[relative_path.resolve()],
        "evidence": evidence,
        "evidence_path": evidence_path,
        "paths": paths,
        "raw_region": raw_region,
    }


def test_explicit_routing_overlay_round_trip_and_runner_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "sealed-routing.json"
    routing_builder.write_source_routing_overlay(
        fixture["source_path"],
        fixture["relative_path"],
        fixture["evidence_path"],
        output,
    )

    binding = validate_source_routing_overlay(
        output,
        source_manifest_path=fixture["source_path"],
        relative_manifest_path=fixture["relative_path"],
        source_binding=fixture["source_binding"],
        relative_binding=fixture["relative_binding"],
        source_manifest_payload=fixture["source"],
        relative_manifest_payload=fixture["relative"],
    )
    record = binding["records"][("neutral-page", "region-0")]
    loaded = load_source_region_evidence(
        fixture["raw_region"], record, shape=(8, 8)
    )
    assert loaded.available is True
    assert np.count_nonzero(loaded.ownership) == 36
    assert record.semantic_action == "translate_inpaint"
    assert record.semantic_provenance == "proposal_text_class_fallback"


@pytest.mark.parametrize("failure", ["missing_role", "duplicate_page", "sha_tamper"])
def test_source_routing_evidence_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    evidence = deepcopy(fixture["evidence"])
    if failure == "missing_role":
        evidence["pages"][0]["regions"][0]["artifacts"].pop("corner")
    elif failure == "duplicate_page":
        evidence["pages"].append(deepcopy(evidence["pages"][0]))
    else:
        evidence["pages"][0]["regions"][0]["artifacts"]["ownership"][
            "pixel_sha256"
        ] = "9" * 64
    evidence["evidence_sha256"] = routing_builder._unsigned_sha256(
        evidence, "evidence_sha256"
    )
    fixture["evidence_path"].write_text(
        json.dumps(evidence, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(ValueError):
        routing_builder.build_source_routing_overlay(
            fixture["source_path"],
            fixture["relative_path"],
            fixture["evidence_path"],
        )


def test_semantic_role_action_conflict_is_sealed_as_abstain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["relative"]["pages"][0]["regions"][0]["proposal"] = {
        "text_class": "sfx",
        "processing_action": "translate_inpaint",
    }
    fixture["relative_path"].write_text(
        json.dumps(fixture["relative"], sort_keys=True), encoding="utf-8"
    )
    fixture["relative_binding"]["manifest_sha256"] = _sha(
        fixture["relative_path"]
    )
    fixture["evidence"]["relative_manifest_sha256"] = _sha(
        fixture["relative_path"]
    )
    raw = fixture["relative"]["pages"][0]["regions"][0]
    fixture["evidence"]["pages"][0]["regions"][0][
        "source_region_record_sha256"
    ] = _canonical(raw)
    fixture["evidence"]["evidence_sha256"] = routing_builder._unsigned_sha256(
        fixture["evidence"], "evidence_sha256"
    )
    fixture["evidence_path"].write_text(
        json.dumps(fixture["evidence"], sort_keys=True), encoding="utf-8"
    )
    relative_seal = fixture["relative_path"].with_suffix(".json.seal.json")
    relative_seal.write_text(
        json.dumps({"source_manifest_sha256": _sha(fixture["source_path"])}),
        encoding="utf-8",
    )
    fixture["relative_binding"]["seal_sha256"] = _sha(relative_seal)

    payload = routing_builder.build_source_routing_overlay(
        fixture["source_path"],
        fixture["relative_path"],
        fixture["evidence_path"],
    )
    region = payload["pages"][0]["regions"][0]
    assert region["semantic_action"] == "review"
    assert region["semantic_available"] is False
    assert region["semantic_reason"] == "semantic_role_action_conflict"
