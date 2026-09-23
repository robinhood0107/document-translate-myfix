from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest
import scripts.build_inpaint_seedless_ocr_overlay_v34 as seedless_contract

from scripts.benchmark_inpaint_glyph_refinement_v34 import (
    validate_seedless_ocr_overlay,
)
from scripts.build_inpaint_seedless_ocr_overlay_v34 import (
    SOURCE_EVIDENCE_SCHEMA_VERSION,
    build_seedless_ocr_overlay,
    main,
    write_seedless_ocr_overlay,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@pytest.fixture(autouse=True)
def _neutral_tracked_dependency_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    # The unit fixture binds one unchanged tracked file. Production requires
    # the full shared runtime dependency inventory.
    monkeypatch.setattr(seedless_contract, "TRACKED_RUNTIME_DEPENDENCIES", ("rules.md",))


def _write_source_manifest(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    source_sha = "1" * 64
    inventory_sha = "2" * 64
    manifest = {
        "schema_version": "inpaint-factorized-source-manifest-v4",
        "candidate_seen": False,
        "annotation_frozen_before_candidate": True,
        "page_count": 1,
        "page_inventory_sha256": inventory_sha,
        "pages": [
            {
                "page_id": "neutral-page",
                "source_sha256": source_sha,
                "candidate_seen": False,
                "annotation_frozen_before_candidate": True,
                # Deliberately nonexistent evaluation paths prove that the
                # producer never resolves or opens them.
                "target_text_mask": str(tmp_path / "must-not-open-target.png"),
                "protected_structure_mask": str(
                    tmp_path / "must-not-open-protected.png"
                ),
                "regions": [
                    {
                        "region_id": "region-0",
                        "source_reviewed": True,
                        "ambiguous_structure_mask": str(
                            tmp_path / "must-not-open-ambiguous.png"
                        ),
                    }
                ],
            }
        ],
    }
    path = tmp_path / "source-manifest-v4.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    seal = {
        "schema_version": "inpaint-factorized-manifest-seal-v4-independent",
        "manifest_sha256": _sha256(path),
        "candidate_generated": False,
        "candidate_seen": False,
        "annotation_frozen_before_candidate": True,
    }
    path.with_suffix(path.suffix + ".seal.json").write_text(
        json.dumps(seal, sort_keys=True),
        encoding="utf-8",
    )
    return path, manifest


def _valid_region() -> dict[str, object]:
    source_artifact_sha = "3" * 64
    return {
        "region_id": "region-0",
        "owner_region_id": "region-0",
        "owner_count": 1,
        "owner_binding_kind": "canonical_block_region_id",
        "canonical_block_id": "block-0",
        "ocr_text": "白い文字",
        "ocr_script": "Japanese",
        "provider": "product-ocr-provider",
        "ocr_confidence": 0.91,
        "confidence_kind": "detector_block_confidence",
        "processing_action": "translate_inpaint",
        "route_class": "clean_translucent",
        "authoritative_ocr": True,
        "candidate_seen": False,
        "annotation_frozen_before_candidate": True,
        "provenance": {
            "ocr_artifact_sha256": source_artifact_sha,
            "confidence_artifact_sha256": source_artifact_sha,
            "owner_binding_sha256": "4" * 64,
            "action_artifact_sha256": "5" * 64,
            "route_artifact_sha256": "6" * 64,
        },
    }


def _write_evidence(
    tmp_path: Path,
    manifest_path: Path,
    manifest: dict[str, object],
    *,
    region: dict[str, object] | None = None,
    extra_top: dict[str, object] | None = None,
) -> Path:
    page = manifest["pages"][0]
    assert isinstance(page, dict)
    root = Path(__file__).resolve().parents[1]
    relative = "rules.md"
    dependencies = [
        {
            "path": relative,
            "tracked": True,
            "unchanged_from_head": True,
            "head_blob_id": subprocess.check_output(
                ["git", "rev-parse", f"HEAD:{relative}"], cwd=root, text=True
            ).strip(),
            "working_file_sha256": _sha256(root / relative),
        }
    ]
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    tracked = {
        "git_head": git_head,
        "git_branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=root, text=True
        ).strip(),
        "tracked_worktree_clean": True,
        "dependency_count": len(dependencies),
        "dependencies": dependencies,
        "dependency_inventory_sha256": _canonical_sha256(dependencies),
    }
    detector = {"model_sha256": "a" * 64}
    ocr = {"model_sha256": "b" * 64}
    provider_identity = _canonical_sha256(
        {"provider": "neutral-product-source", "detector": detector, "ocr": ocr, "code": tracked}
    )
    provider = f"neutral-product-source:{provider_identity}"
    evidence: dict[str, object] = {
        "schema_version": SOURCE_EVIDENCE_SCHEMA_VERSION,
        "source_manifest_sha256": _sha256(manifest_path),
        "source_page_inventory_sha256": manifest["page_inventory_sha256"],
        "candidate_seen": False,
        "annotation_frozen_before_candidate": True,
        "producer": provider,
        "pages": [
            {
                "page_id": page["page_id"],
                "source_sha256": page["source_sha256"],
                "candidate_seen": False,
                "annotation_frozen_before_candidate": True,
                "regions": [] if region is None else [region],
            }
        ],
    }
    if extra_top:
        evidence.update(extra_top)
    path = tmp_path / "source-product-evidence.json"
    path.write_text(
        json.dumps(evidence, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    source_inventory = [
        {
            "page_id": page["page_id"],
            "source_sha256": page["source_sha256"],
            "regions": [{"region_id": "region-0"}],
        }
    ]
    record_inventory = []
    if region is not None:
        provenance = region.get("provenance")
        record_inventory.append(
            {
                "page_id": page["page_id"],
                "source_sha256": page["source_sha256"],
                "region_id": region.get("region_id"),
                "source_record_sha256": _canonical_sha256(region),
                "provider": region.get("provider"),
                "owner_count": region.get("owner_count"),
                "owner_binding_sha256": (
                    provenance.get("owner_binding_sha256") if isinstance(provenance, dict) else None
                ),
                "confidence_artifact_sha256": (
                    provenance.get("confidence_artifact_sha256") if isinstance(provenance, dict) else None
                ),
            }
        )
    receipt = {
        "schema_version": "inpaint-source-ocr-runtime-receipt-v34",
        "source_manifest_sha256": _sha256(manifest_path),
        "source_manifest_seal_sha256": _sha256(manifest_path.with_suffix(manifest_path.suffix + ".seal.json")),
        "source_page_inventory_sha256": manifest["page_inventory_sha256"],
        "candidate_seen": False,
        "annotation_frozen_before_candidate": True,
        "git_head": git_head,
        "provider": provider,
        "provider_identity_sha256": provider_identity,
        "evidence_payload_sha256": _canonical_sha256(evidence),
        "evidence_file_sha256": _sha256(path),
        "source_region_inventory_sha256": _canonical_sha256(source_inventory),
        "record_inventory_sha256": _canonical_sha256(record_inventory),
        "source_region_inventory": source_inventory,
        "record_inventory": record_inventory,
        "detector_identity": detector,
        "detector_identity_sha256": _canonical_sha256(detector),
        "ocr_identity": ocr,
        "ocr_identity_sha256": _canonical_sha256(ocr),
        "tracked_dependency_identity": tracked,
        "tracked_dependency_identity_sha256": _canonical_sha256(tracked),
        "reuse_evidence_sha256": None,
        "summary": {},
        "pages": [],
    }
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    seal = {
        "schema_version": "inpaint-source-ocr-block-evidence-seal-v34",
        "evidence_file_sha256": _sha256(path),
        "receipt_file_sha256": _sha256(receipt_path),
        "evidence_payload_sha256": _canonical_sha256(evidence),
        "receipt_payload_sha256": _canonical_sha256(receipt),
        "source_region_inventory_sha256": _canonical_sha256(source_inventory),
        "record_inventory_sha256": _canonical_sha256(record_inventory),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_page_inventory_sha256": manifest["page_inventory_sha256"],
        "provider_identity_sha256": provider_identity,
        "tracked_dependency_identity_sha256": _canonical_sha256(tracked),
        "page_count": 1,
        "complete_page_set": True,
        "candidate_generated": False,
        "candidate_seen": False,
        "annotation_frozen_before_candidate": True,
    }
    path.with_suffix(path.suffix + ".seal.json").write_text(
        json.dumps(seal, sort_keys=True), encoding="utf-8"
    )
    return path


def test_valid_source_evidence_writes_runner_compatible_sealed_overlay(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _write_source_manifest(tmp_path)
    evidence_path = _write_evidence(
        tmp_path,
        manifest_path,
        manifest,
        region=_valid_region(),
    )
    output = tmp_path / "seedless-overlay.json"

    written, seal_path, payload = write_seedless_ocr_overlay(
        manifest_path,
        evidence_path,
        output,
    )

    assert written == output
    assert seal_path == output.with_suffix(".json.seal.json")
    assert payload["summary"] == {
        "page_count": 1,
        "source_region_count": 1,
        "input_evidence_record_count": 1,
        "admitted_record_count": 1,
        "missing_record_count": 0,
        "information_limited_record_count": 0,
        "reason_counts": {},
    }
    source_binding = {
        "manifest_sha256": _sha256(manifest_path),
        "page_inventory_sha256": manifest["page_inventory_sha256"],
    }
    validated = validate_seedless_ocr_overlay(
        output,
        source_manifest_path=manifest_path,
        source_binding=source_binding,
        source_manifest_payload=manifest,
    )
    record = validated["records"][("neutral-page", "region-0")]
    assert record.ocr_text == "白い文字"
    assert record.provider == "product-ocr-provider"
    assert record.owner_count == 1
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    assert seal["overlay_file_sha256"] == _sha256(output)
    assert seal["candidate_generated"] is False


def test_missing_required_runtime_dependency_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, manifest = _write_source_manifest(tmp_path)
    evidence_path = _write_evidence(
        tmp_path, manifest_path, manifest, region=_valid_region()
    )
    monkeypatch.setattr(
        seedless_contract,
        "TRACKED_RUNTIME_DEPENDENCIES",
        ("rules.md", "AGENTS.md"),
    )
    with pytest.raises(ValueError, match="tracked dependency proof differs"):
        build_seedless_ocr_overlay(manifest_path, evidence_path)


def test_missing_source_record_is_explicitly_accounted(tmp_path: Path) -> None:
    manifest_path, manifest = _write_source_manifest(tmp_path)
    evidence_path = _write_evidence(tmp_path, manifest_path, manifest)

    payload = build_seedless_ocr_overlay(manifest_path, evidence_path)

    assert payload["pages"][0]["regions"] == []
    assert payload["pages"][0]["information_limited_regions"] == [
        {
            "region_id": "region-0",
            "status": "missing",
            "reasons": ["source_ocr_evidence_missing"],
        }
    ]
    assert payload["summary"]["missing_record_count"] == 1


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        (lambda row: row.pop("provenance"), "source_provenance_missing"),
        (
            lambda row: row.update(confidence_kind="generative_unscored"),
            "ocr_confidence_kind_untrusted",
        ),
        (lambda row: row.update(ocr_confidence=0.0), "ocr_confidence_unavailable"),
        (
            lambda row: row.update(processing_action="review"),
            "processing_action_not_translate",
        ),
        (lambda row: row.update(owner_count=2), "exact_owner_count_unclear"),
        (lambda row: row.update(owner_count=True), "exact_owner_count_unclear"),
        (
            lambda row: row.update(owner_region_id="region-other"),
            "exact_owner_region_mismatch",
        ),
        (lambda row: row.update(provider=""), "ocr_provider_missing"),
        (
            lambda row: row.update(route_class="ambiguous"),
            "route_not_clean_translucent",
        ),
        (
            lambda row: row.update(canonical_block_id=""),
            "canonical_block_id_missing",
        ),
    ],
)
def test_incomplete_or_unclear_evidence_remains_information_limited(
    tmp_path: Path,
    mutation,
    expected_reason: str,
) -> None:
    manifest_path, manifest = _write_source_manifest(tmp_path)
    region = deepcopy(_valid_region())
    mutation(region)
    evidence_path = _write_evidence(
        tmp_path,
        manifest_path,
        manifest,
        region=region,
    )

    payload = build_seedless_ocr_overlay(manifest_path, evidence_path)

    assert payload["pages"][0]["regions"] == []
    limited = payload["pages"][0]["information_limited_regions"][0]
    assert limited["status"] == "information_limited"
    assert expected_reason in limited["reasons"]
    assert payload["summary"]["information_limited_record_count"] == 1


@pytest.mark.parametrize("forbidden_field", ["bbox", "target_text_mask", "candidate_image"])
def test_geometry_or_evaluation_fields_are_rejected_without_being_read(
    tmp_path: Path,
    forbidden_field: str,
) -> None:
    manifest_path, manifest = _write_source_manifest(tmp_path)
    region = _valid_region()
    region[forbidden_field] = str(tmp_path / "must-not-open.png")
    evidence_path = _write_evidence(
        tmp_path,
        manifest_path,
        manifest,
        region=region,
    )

    with pytest.raises(ValueError, match="unsupported fields"):
        build_seedless_ocr_overlay(manifest_path, evidence_path)


@pytest.mark.parametrize(
    "top_mutation",
    [
        {"candidate_seen": True},
        {"annotation_frozen_before_candidate": False},
        {"source_manifest_sha256": "9" * 64},
        {"source_page_inventory_sha256": "8" * 64},
    ],
)
def test_source_binding_or_candidate_tamper_fails_closed(
    tmp_path: Path,
    top_mutation: dict[str, object],
) -> None:
    manifest_path, manifest = _write_source_manifest(tmp_path)
    evidence_path = _write_evidence(
        tmp_path,
        manifest_path,
        manifest,
        region=_valid_region(),
        extra_top=top_mutation,
    )

    with pytest.raises(ValueError):
        build_seedless_ocr_overlay(manifest_path, evidence_path)


def test_cli_requires_fresh_output_and_never_embeds_eval_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path, manifest = _write_source_manifest(tmp_path)
    evidence_path = _write_evidence(
        tmp_path,
        manifest_path,
        manifest,
        region=_valid_region(),
    )
    output = tmp_path / "sealed" / "overlay.json"

    assert (
        main(
            [
                "--manifest",
                str(manifest_path),
                "--product-evidence",
                str(evidence_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["summary"]["admitted_record_count"] == 1
    output_text = output.read_text(encoding="utf-8")
    assert "must-not-open" not in output_text
    assert "target_text_mask" not in output_text
    assert "protected_structure_mask" not in output_text
    with pytest.raises(FileExistsError):
        main(
            [
                "--manifest",
                str(manifest_path),
                "--product-evidence",
                str(evidence_path),
                "--output",
                str(output),
            ]
        )
