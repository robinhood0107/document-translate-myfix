from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import pytest
import scripts.build_inpaint_seedless_ocr_overlay_v34 as seedless_contract

from scripts.build_inpaint_seedless_ocr_overlay_v34 import (
    build_seedless_ocr_overlay,
)
from scripts.export_inpaint_source_ocr_evidence_v34 import (
    RuntimeBlock,
    RuntimeIdentity,
    build_source_product_evidence,
    write_source_product_evidence,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def _neutral_tracked_dependency_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seedless_contract, "TRACKED_RUNTIME_DEPENDENCIES", ("rules.md",))


def _write_png(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)
    assert success
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def _write_manifest(
    tmp_path: Path,
    *,
    source: np.ndarray | None = None,
    ownerships: tuple[np.ndarray, ...] | None = None,
    route_class: str = "clean_translucent",
) -> Path:
    if source is None:
        source = np.full((32, 48, 3), 180, dtype=np.uint8)
    if ownerships is None:
        ownership = np.zeros(source.shape[:2], dtype=np.uint8)
        ownership[4:28, 4:44] = 255
        ownerships = (ownership,)
    source_path = tmp_path / "neutral-source.png"
    _write_png(source_path, cv2.cvtColor(source, cv2.COLOR_RGB2BGR))
    regions: list[dict[str, object]] = []
    region_hashes: dict[str, dict[str, str]] = {}
    for index, ownership in enumerate(ownerships):
        region_id = f"region-{index}"
        ownership_path = tmp_path / f"ownership-{index}.png"
        _write_png(ownership_path, ownership)
        regions.append(
            {
                "region_id": region_id,
                "ownership_mask": str(ownership_path),
                "bubble_route_class": route_class,
                "source_reviewed": True,
                # None of these candidate/evaluation paths exists.  The
                # exporter must not resolve or open them.
                "target_text_mask": str(tmp_path / "must-not-open-target.png"),
                "protected_structure_mask": str(
                    tmp_path / "must-not-open-protected.png"
                ),
                "candidate_image": str(tmp_path / "must-not-open-candidate.png"),
            }
        )
        region_hashes[region_id] = {"ownership_mask": _sha256(ownership_path)}
    manifest = {
        "schema_version": "inpaint-factorized-source-manifest-v4",
        "candidate_seen": False,
        "annotation_frozen_before_candidate": True,
        "page_count": 1,
        "page_inventory_sha256": "2" * 64,
        "pages": [
            {
                "page_id": "neutral-page",
                "path": str(source_path),
                "source_sha256": _sha256(source_path),
                "candidate_seen": False,
                "annotation_frozen_before_candidate": True,
                "target_text_mask": str(tmp_path / "must-not-open-page-target.png"),
                "protected_structure_mask": str(
                    tmp_path / "must-not-open-page-protected.png"
                ),
                "regions": regions,
                "artifact_sha256": {"regions": region_hashes},
            }
        ],
    }
    path = tmp_path / "source-manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    path.with_suffix(path.suffix + ".seal.json").write_text(
        json.dumps(
            {
                "schema_version": "inpaint-factorized-manifest-seal-v4-independent",
                "manifest_sha256": _sha256(path),
                "candidate_generated": False,
                "candidate_seen": False,
                "annotation_frozen_before_candidate": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _identity(*, provider: str = "neutral-product-runtime") -> RuntimeIdentity:
    root = Path(__file__).resolve().parents[1]
    relative = "rules.md"
    dependency = {
        "path": relative,
        "tracked": True,
        "unchanged_from_head": True,
        "head_blob_id": subprocess.check_output(
            ["git", "rev-parse", f"HEAD:{relative}"], cwd=root, text=True
        ).strip(),
        "working_file_sha256": _sha256(root / relative),
    }
    dependencies = [dependency]
    code = {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "git_branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=root, text=True
        ).strip(),
        "tracked_worktree_clean": True,
        "dependency_count": len(dependencies),
        "dependencies": dependencies,
        "dependency_inventory_sha256": hashlib.sha256(
            json.dumps(dependencies, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    return RuntimeIdentity(
        provider=provider,
        detector={
            "model_sha256": "3" * 64,
            "runtime_sha256": "4" * 64,
            "provider": "neutral-detector",
        },
        ocr={
            "model_sha256": "5" * 64,
            "runtime_sha256": "6" * 64,
            "provider": "neutral-ocr",
        },
        code=code,
    )


def _block(*, confidence: float = 0.91) -> RuntimeBlock:
    return RuntimeBlock(
        xyxy=(8, 7, 36, 25),
        text_class="text_free",
        detector_confidence=confidence,
        confidence_receipt={
            "model_sha256": "3" * 64,
            "runtime_sha256": "4" * 64,
            "proposal": {"xyxy": [8, 7, 36, 25], "score": confidence},
        },
    )


class _FakeRuntime:
    def __init__(
        self,
        blocks: list[RuntimeBlock],
        *,
        identity: RuntimeIdentity | None = None,
        text: str = "白色文字",
        fail_recognize: bool = False,
    ) -> None:
        self.identity = identity or _identity()
        self.blocks = blocks
        self.text = text
        self.fail_recognize = fail_recognize
        self.detect_calls = 0
        self.recognize_calls = 0
        self.recognized_images: list[np.ndarray] = []

    def detect(self, image_rgb: np.ndarray) -> list[RuntimeBlock]:
        self.detect_calls += 1
        return self.blocks

    def recognize(
        self,
        image_rgb: np.ndarray,
        blocks: list[RuntimeBlock],
    ) -> None:
        self.recognize_calls += 1
        self.recognized_images.append(image_rgb.copy())
        if self.fail_recognize:
            raise AssertionError("OCR should have been reused")
        for block in blocks:
            block.text = self.text
            block.ocr_status = "ok" if self.text else "empty_initial"
            block.ocr_strategy = "paddle_crop"
            block.ocr_model_identity = "neutral-model"
            block.ocr_runtime_identity = "neutral-runtime"
            block.semantic_role = "dialogue_free"
            block.processing_action = "translate_inpaint"
            block.processing_decision_source = "ocr_processing_default"
            block.processing_decision_reasons = ("detector_text_class_default",)


def test_export_reads_only_source_and_ownership_and_builds_valid_overlay(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    runtime = _FakeRuntime([_block()])

    evidence_path = tmp_path / "product-evidence.json"
    _output, receipt_path, seal_path, receipt = write_source_product_evidence(
        manifest,
        evidence_path,
        runtime,
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

    assert runtime.detect_calls == 1
    assert runtime.recognize_calls == 1
    record = evidence["pages"][0]["regions"][0]
    assert record["ocr_text"] == "白色文字"
    assert record["confidence_kind"] == "detector_block_confidence"
    assert record["ocr_confidence"] == 0.91
    assert record["owner_count"] == 1
    assert receipt["detector_identity"]["model_sha256"] == "3" * 64
    assert receipt["ocr_identity"]["runtime_sha256"] == "6" * 64
    assert receipt["evidence_payload_sha256"]
    assert receipt["source_region_inventory_sha256"]
    assert receipt["record_inventory_sha256"]
    assert _sha256(receipt_path)
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    assert seal["complete_page_set"] is True
    assert seal["page_count"] == 1
    assert seal["receipt_file_sha256"] == _sha256(receipt_path)
    assert seal["record_inventory_sha256"] == receipt["record_inventory_sha256"]
    serialized = json.dumps(evidence, ensure_ascii=False)
    assert "must-not-open" not in serialized
    assert "target_text_mask" not in serialized
    assert "candidate_image" not in serialized

    overlay = build_seedless_ocr_overlay(manifest, evidence_path)
    assert overlay["summary"]["admitted_record_count"] == 1


def test_owner_overlap_conflict_is_exported_fail_closed_without_ocr(
    tmp_path: Path,
) -> None:
    shape = (32, 48)
    first = np.zeros(shape, dtype=np.uint8)
    second = np.zeros(shape, dtype=np.uint8)
    first[4:28, 4:30] = 255
    second[4:28, 24:44] = 255
    manifest = _write_manifest(tmp_path, ownerships=(first, second))
    runtime = _FakeRuntime([_block()])

    evidence, receipt = build_source_product_evidence(manifest, runtime)

    assert runtime.recognize_calls == 0
    records = evidence["pages"][0]["regions"]
    assert len(records) == 2
    assert {row["owner_count"] for row in records} == {2}
    assert {row["owner_binding_kind"] for row in records} == {"conflict"}
    assert {row["processing_action"] for row in records} == {"review"}
    assert receipt["summary"]["ownership_conflict_region_count"] == 2


def test_empty_ocr_and_zero_confidence_remain_information_limited(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    runtime = _FakeRuntime([_block(confidence=0.0)], text="")

    evidence_path = tmp_path / "empty-product-evidence.json"
    write_source_product_evidence(manifest, evidence_path, runtime)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    record = evidence["pages"][0]["regions"][0]

    assert record["ocr_text"] == ""
    assert record["authoritative_ocr"] is False
    assert record["ocr_confidence"] == 0.0
    assert record["confidence_kind"] == "unavailable"

    overlay = build_seedless_ocr_overlay(manifest, evidence_path)
    assert overlay["summary"]["admitted_record_count"] == 0
    reasons = overlay["pages"][0]["information_limited_regions"][0]["reasons"]
    assert "ocr_text_missing" in reasons
    assert "ocr_confidence_unavailable" in reasons


def test_exact_cache_reuse_skips_ocr_but_provider_or_seal_tamper_reruns(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    first_runtime = _FakeRuntime([_block()])
    first_path = tmp_path / "first.json"
    write_source_product_evidence(manifest, first_path, first_runtime)
    assert first_runtime.recognize_calls == 1

    reused_runtime = _FakeRuntime([_block()], fail_recognize=True)
    evidence, receipt = build_source_product_evidence(
        manifest,
        reused_runtime,
        reuse_evidence_path=first_path,
    )
    assert reused_runtime.recognize_calls == 0
    assert evidence["pages"][0]["regions"][0]["ocr_text"] == "白色文字"
    assert receipt["summary"]["cache_reuse_region_count"] == 1

    changed_provider = _FakeRuntime(
        [_block()],
        identity=_identity(provider="different-product-runtime"),
    )
    build_source_product_evidence(
        manifest,
        changed_provider,
        reuse_evidence_path=first_path,
    )
    assert changed_provider.recognize_calls == 1

    tampered = json.loads(first_path.read_text(encoding="utf-8"))
    tampered["pages"][0]["regions"][0]["ocr_text"] = "tampered"
    first_path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    tamper_runtime = _FakeRuntime([_block()])
    build_source_product_evidence(
        manifest,
        tamper_runtime,
        reuse_evidence_path=first_path,
    )
    assert tamper_runtime.recognize_calls == 1


def test_cache_from_different_source_manifest_is_not_reused(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_manifest = _write_manifest(first_root)
    cache_path = first_root / "evidence.json"
    write_source_product_evidence(
        first_manifest,
        cache_path,
        _FakeRuntime([_block()]),
    )

    changed_source = np.full((32, 48, 3), 181, dtype=np.uint8)
    second_manifest = _write_manifest(second_root, source=changed_source)
    runtime = _FakeRuntime([_block()])
    build_source_product_evidence(
        second_manifest,
        runtime,
        reuse_evidence_path=cache_path,
    )
    assert runtime.recognize_calls == 1


def test_low_contrast_white_glyph_source_is_forwarded_unchanged_to_ocr(
    tmp_path: Path,
) -> None:
    source = np.full((32, 48, 3), 158, dtype=np.uint8)
    # A low-contrast white glyph-like cross on a translucent-gray carrier.
    source[9:24, 20:24] = 190
    source[14:18, 14:30] = 190
    manifest = _write_manifest(tmp_path, source=source)
    runtime = _FakeRuntime([_block()], text="白字")

    evidence, _receipt = build_source_product_evidence(manifest, runtime)

    assert runtime.recognize_calls == 1
    assert np.array_equal(runtime.recognized_images[0], source)
    assert evidence["pages"][0]["regions"][0]["ocr_text"] == "白字"


def test_runtime_failure_is_information_limited_not_authoritative(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    runtime = _FakeRuntime([deepcopy(_block())], fail_recognize=True)

    evidence, receipt = build_source_product_evidence(manifest, runtime)

    record = evidence["pages"][0]["regions"][0]
    assert record["authoritative_ocr"] is False
    assert record["ocr_text"] == ""
    assert record["processing_action"] == "review"
    assert receipt["summary"]["ocr_failed_page_count"] == 1
