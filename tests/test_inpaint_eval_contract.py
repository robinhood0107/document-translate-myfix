from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import imkit as imk

from scripts.inpaint_eval_contract import (
    EvalImageReference,
    InpaintEvalManifestError,
    build_quality_metrics,
    derive_blind_review_seed,
    finalize_optional_eval_manifest,
    load_binary_mask,
    load_eval_manifest,
    load_eval_manifests,
    load_eval_source_array,
    load_rgb_reference_array,
    pixel_sha256,
    seal_manifest_payload,
    seal_source_only_evidence_manifest,
    sha256_file,
    write_blind_review_jsonl,
    write_comparison_and_blind_panels,
    verify_eval_page_spec,
)


def _write_image(path: Path, array: np.ndarray | None = None) -> Path:
    if array is None:
        array = np.full((12, 16, 3), 240, dtype=np.uint8)
    Image.fromarray(array).save(path)
    return path


def _manifest_payload(
    source: Path,
    *,
    corpus_id: str = "corpus-a1",
    page_id: str = "a1-001",
    expected_count: int = 1,
    width: int = 16,
    height: int = 12,
    expected_edit: str = "required",
    split_role: str = "tuning",
) -> dict:
    return {
        "schema_version": 1,
        "corpus_id": corpus_id,
        "split_role": split_role,
        "source_lock_git_sha": "1" * 40,
        "expected_count": expected_count,
        "pages": [
            {
                "page_id": page_id,
                "path": str(source),
                "sha256": sha256_file(source),
                "size_bytes": source.stat().st_size,
                "width": width,
                "height": height,
                "expected_edit": expected_edit,
            }
        ],
    }


def _write_manifest(path: Path, payload: dict) -> Path:
    path.write_text(
        json.dumps(seal_manifest_payload(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _load_export_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "export_inpaint_debug.py"
    spec = importlib.util.spec_from_file_location(
        "export_inpaint_debug_eval_contract_test",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_finalize_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "finalize_inpaint_eval_manifest.py"
    )
    spec = importlib.util.spec_from_file_location(
        "finalize_inpaint_eval_manifest_contract_test",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_fake_export_runtime(
    module,
    monkeypatch: pytest.MonkeyPatch,
    output: Path,
    calls: list[dict],
    events: list[tuple[str, dict]],
) -> None:
    class _FakeInpainter:
        def __init__(self, *_args, **_kwargs) -> None:
            self.runtime_device = "cpu"
            self.device = "cpu"
            self.precision = "fp32"

    artifact_run = SimpleNamespace(
        complete=lambda **kwargs: events.append(("complete", kwargs)),
        fail=lambda *_args, **kwargs: events.append(("fail", kwargs)),
    )

    def fake_process(
        image_path: Path,
        corpus_output: Path,
        *_args,
        page_spec=None,
        public_corpus_id: str | None = None,
        public_page_id: str | None = None,
        **_kwargs,
    ) -> dict:
        page_id = page_spec.page_id if page_spec is not None else public_page_id
        assert page_id is not None
        expected_edit = (
            page_spec.expected_edit if page_spec is not None else "required"
        )
        calls.append(
            {
                "image_path": image_path,
                "public_corpus_id": public_corpus_id,
                "public_page_id": public_page_id,
            }
        )
        rgb = np.full((12, 16, 3), 240, dtype=np.uint8)
        mask = np.zeros((12, 16), dtype=np.uint8)
        mask[2:10, 3:13] = 255

        def save(relative: str, array: np.ndarray) -> Path:
            path = corpus_output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_image(path, array)
            return path

        source_output = save(f"source_images/{page_id}_source.png", rgb)
        cleaned = save(f"cleaned_images/{page_id}_cleaned.png", rgb)
        final_mask = save(f"final_masks/{page_id}_final_mask.png", mask)
        protected = save(
            f"protected_corner_masks/{page_id}_protected_corners.png",
            np.zeros_like(mask),
        )
        detector_overlay = save(
            f"detector_overlays/{page_id}_detector_overlay.png",
            rgb,
        )
        raw_mask = save(f"raw_masks/{page_id}_raw_mask.png", mask)
        mask_overlay = save(f"mask_overlays/{page_id}_mask_overlay.png", mask)
        cleanup_delta = save(
            f"cleanup_mask_delta/{page_id}_cleanup_delta.png",
            np.zeros_like(mask),
        )
        metadata = corpus_output / "debug_metadata" / f"{page_id}_debug.json"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(
            json.dumps(
                {
                    "corpus_id": public_corpus_id,
                    "page_id": page_id,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return {
            "image": page_id,
            "page_id": page_id,
            "expected_edit": expected_edit,
            "source": source_output,
            "cleaned": cleaned,
            "final_mask": final_mask,
            "protected_corner_mask": protected,
            "detector_overlay": detector_overlay,
            "raw_mask": raw_mask,
            "mask_overlay": mask_overlay,
            "cleanup_delta": cleanup_delta,
            "metadata": metadata,
            "source_sha256": sha256_file(source_output),
            "source_pixel_sha256": pixel_sha256(rgb),
            "source_size_bytes": source_output.stat().st_size,
            "cleaned_sha256": sha256_file(cleaned),
            "final_mask_sha256": sha256_file(final_mask),
            "cleaned_pixel_sha256": pixel_sha256(rgb),
            "final_mask_pixel_sha256": pixel_sha256(mask),
            "baseline_cleaned_sha256": None,
            "baseline_final_mask_sha256": None,
            "baseline_cleaned_pixel_sha256": None,
            "baseline_final_mask_pixel_sha256": None,
            "cleaned_matches_baseline_sha256": None,
            "final_mask_matches_baseline_sha256": None,
            "cleaned_matches_baseline_pixel_sha256": None,
            "final_mask_matches_baseline_pixel_sha256": None,
            "block_count": 1,
            "final_mask_pixel_count": int(np.count_nonzero(mask)),
            "refiner_device": "cpu",
            "inpaint_runtime_inference_call_count": 0,
            "inpaint_runtime_cpu_fallback_count": 0,
            "bubble_silhouette_fallback_count": 0,
            "protected_corner_final_mask_pixel_count": 0,
            "protected_corner_changed_pixel_count": 0,
            "changed_outside_final_mask_pixel_count_exact": 0,
            "protected_structure_changed_pixel_count_exact": 0,
            "protected_structure_annotation_available": True,
            "protected_structure_annotation_changed_pixel_count_exact": 0,
            "residue_pixel_count": 0,
            "residue_source_contrast_pixel_count": 0,
            "residue_pass_truncated_block_count": 0,
            "erase_mode_distribution": {"unassigned": 1},
            "erase_skipped_reason_distribution": {},
            "block_runtime_seconds": [],
            "pipeline_elapsed_seconds": 0.01,
            "peak_vram_allocated_mb": 0.0,
            "peak_vram_reserved_mb": 0.0,
            "inpaint_runtime_diagnostics": [],
            "text_anchor_final_mask_pixel_count": int(np.count_nonzero(mask)),
            "text_anchor_changed_pixel_count": int(np.count_nonzero(mask)),
            "cleanup_applied": False,
            "cleanup_component_count": 0,
            "cleanup_block_count": 0,
        }

    monkeypatch.setattr(
        module,
        "select_managed_output_directory",
        lambda **_kwargs: (output, artifact_run),
    )
    monkeypatch.setattr(module, "TextBlockDetector", lambda *_args: SimpleNamespace())
    monkeypatch.setattr(
        module,
        "get_inpainter_runtime",
        lambda *_args: {
            "key": "fake",
            "backend": "test",
            "device": "cpu",
            "precision": "fp32",
            "inpaint_size": 64,
        },
    )
    monkeypatch.setattr(module, "inpaint_map", {"fake": _FakeInpainter})
    monkeypatch.setattr(module, "resolve_device", lambda *_args, **_kwargs: "cpu")
    monkeypatch.setattr(module, "_process_image", fake_process)


def test_manifest_loads_valid_sealed_neutral_page(tmp_path: Path) -> None:
    source = _write_image(tmp_path / "private-source.png")
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        _manifest_payload(source, expected_edit="optional"),
    )

    manifest = load_eval_manifest(manifest_path)

    assert manifest.corpus_id == "corpus-a1"
    assert manifest.split_role == "tuning"
    assert manifest.source_lock_git_sha == "1" * 40
    assert manifest.expected_count == 1
    assert manifest.pages[0].page_id == "a1-001"
    assert manifest.pages[0].source.path == source.resolve()
    assert manifest.pages[0].expected_edit == "optional"


def test_optional_manifest_finalization_is_parent_linked_and_fail_closed(
    tmp_path: Path,
) -> None:
    source = _write_image(tmp_path / "private-source.png")
    parent_path = _write_manifest(
        tmp_path / "parent.json",
        _manifest_payload(source, expected_edit="optional"),
    )
    parent_bytes = parent_path.read_bytes()
    parent = load_eval_manifest(parent_path)
    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "parent_manifest_sha256": parent.manifest_sha256,
                "decision_basis": "source-only-review",
                "pages": [
                    {"page_id": "a1-001", "expected_edit": "required"}
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "final.json"

    finalized = finalize_optional_eval_manifest(
        parent_path,
        decisions_path,
        output_path,
    )

    assert parent_path.read_bytes() == parent_bytes
    assert finalized.parent_manifest_sha256 == parent.manifest_sha256
    assert finalized.expected_edit_basis == "source-only-review"
    assert finalized.expected_edit_decisions_sha256 == sha256_file(decisions_path)
    assert [page.expected_edit for page in finalized.pages] == ["required"]

    bad_decisions = tmp_path / "bad-decisions.json"
    bad_decisions.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "parent_manifest_sha256": parent.manifest_sha256,
                "decision_basis": "source-only-review",
                "pages": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(InpaintEvalManifestError) as raised:
        finalize_optional_eval_manifest(
            parent_path,
            bad_decisions,
            tmp_path / "bad-final.json",
        )
    assert raised.value.code == "manifest_finalization_page_set_mismatch"
    assert not (tmp_path / "bad-final.json").exists()


def test_source_only_evidence_manifest_seals_new_references_without_overwrite(
    tmp_path: Path,
) -> None:
    source = _write_image(tmp_path / "private-source.png")
    baseline = _write_image(
        tmp_path / "pr3-cleaned.png",
        np.full((12, 16, 3), 220, dtype=np.uint8),
    )
    baseline_mask = _write_image(
        tmp_path / "pr3-final-mask.png",
        np.full((12, 16), 255, dtype=np.uint8),
    )
    protected = np.zeros((12, 16), dtype=np.uint8)
    protected[3:5, 2:14] = 255
    protected_path = _write_image(tmp_path / "protected.png", protected)
    parent_path = _write_manifest(
        tmp_path / "parent.json",
        _manifest_payload(source),
    )
    parent_bytes = parent_path.read_bytes()
    parent = load_eval_manifest(parent_path)
    review_path = tmp_path / "source-only-review.json"
    review_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "parent_manifest_sha256": parent.manifest_sha256,
                "decision_basis": "source-only-inpaint-evidence-v1",
                "pages": [
                    {
                        "page_id": "a1-001",
                        "baseline": {
                            "path": str(baseline),
                            "sha256": sha256_file(baseline),
                        },
                        "baseline_mask": {
                            "path": str(baseline_mask),
                            "sha256": sha256_file(baseline_mask),
                        },
                        "protected_structure_mask": {
                            "path": str(protected_path),
                            "sha256": sha256_file(protected_path),
                        },
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    sealed = seal_source_only_evidence_manifest(
        parent_path,
        review_path,
        tmp_path / "evidence-v2.json",
    )

    assert parent_path.read_bytes() == parent_bytes
    assert sealed.evidence_parent_manifest_sha256 == parent.manifest_sha256
    assert sealed.evidence_basis == "source-only-inpaint-evidence-v1"
    assert sealed.evidence_review_sha256 == sha256_file(review_path)
    assert sealed.pages[0].baseline is not None
    assert sealed.pages[0].baseline_mask is not None
    assert sealed.pages[0].protected_structure_mask is not None


def test_manifest_v2_requires_three_disjoint_annotation_masks(
    tmp_path: Path,
) -> None:
    source = _write_image(tmp_path / "source.png")
    target = np.zeros((12, 16), dtype=np.uint8)
    target[1:3, 1:4] = 255
    protected = np.zeros_like(target)
    protected[5:7, 2:10] = 255
    ambiguous = np.zeros_like(target)
    ambiguous[9:11, 12:15] = 255
    references = {
        "target_text_mask": _write_image(tmp_path / "target.png", target),
        "protected_structure_mask": _write_image(
            tmp_path / "protected.png", protected
        ),
        "ambiguous_structure_mask": _write_image(
            tmp_path / "ambiguous.png", ambiguous
        ),
    }
    payload = _manifest_payload(source)
    payload["schema_version"] = 2
    payload["pages"][0].update(
        {
            field: {"path": str(path), "sha256": sha256_file(path)}
            for field, path in references.items()
        }
    )

    manifest = load_eval_manifest(
        _write_manifest(tmp_path / "manifest-v2.json", payload)
    )

    assert manifest.schema_version == 2
    page = manifest.pages[0]
    assert page.target_text_mask is not None
    assert page.target_glyph_mask is page.target_text_mask
    assert page.protected_structure_mask is not None
    assert page.ambiguous_structure_mask is not None

    overlapping = ambiguous.copy()
    overlapping[1, 1] = 255
    overlap_path = _write_image(tmp_path / "ambiguous-overlap.png", overlapping)
    payload["pages"][0]["ambiguous_structure_mask"] = {
        "path": str(overlap_path),
        "sha256": sha256_file(overlap_path),
    }
    with pytest.raises(InpaintEvalManifestError) as raised:
        load_eval_manifest(
            _write_manifest(tmp_path / "manifest-v2-overlap.json", payload)
        )
    assert raised.value.code == "manifest_annotation_masks_overlap"


def test_manifest_v2_accepts_source_only_target_adjudication(
    tmp_path: Path,
) -> None:
    source = _write_image(tmp_path / "source.png")
    payload = _manifest_payload(source)
    payload.update(
        {
            "schema_version": 2,
            "evidence_parent_manifest_sha256": "2" * 64,
            "evidence_basis": (
                "source-only-inpaint-evidence-v2-target-adjudicated"
            ),
            "evidence_review_sha256": "3" * 64,
        }
    )
    for field in (
        "target_text_mask",
        "protected_structure_mask",
        "ambiguous_structure_mask",
    ):
        mask_path = _write_image(
            tmp_path / f"{field}.png",
            np.zeros((12, 16), dtype=np.uint8),
        )
        payload["pages"][0][field] = {
            "path": str(mask_path),
            "sha256": sha256_file(mask_path),
        }

    manifest = load_eval_manifest(
        _write_manifest(tmp_path / "adjudicated-v2.json", payload)
    )

    assert (
        manifest.evidence_basis
        == "source-only-inpaint-evidence-v2-target-adjudicated"
    )


def test_source_only_evidence_review_v2_migrates_v1_target_name(
    tmp_path: Path,
) -> None:
    source = _write_image(tmp_path / "source.png")
    baseline = _write_image(tmp_path / "baseline.png")
    baseline_mask = _write_image(
        tmp_path / "baseline-mask.png",
        np.zeros((12, 16), dtype=np.uint8),
    )
    annotation_paths = {}
    for field, row in (
        ("target_text_mask", 1),
        ("protected_structure_mask", 5),
        ("ambiguous_structure_mask", 9),
    ):
        mask = np.zeros((12, 16), dtype=np.uint8)
        mask[row : row + 1, 2:6] = 255
        annotation_paths[field] = _write_image(
            tmp_path / f"{field}.png",
            mask,
        )
    parent_path = _write_manifest(
        tmp_path / "parent.json",
        _manifest_payload(source),
    )
    parent = load_eval_manifest(parent_path)
    review_path = tmp_path / "review-v2.json"
    review_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "parent_manifest_sha256": parent.manifest_sha256,
                "decision_basis": "source-only-inpaint-evidence-v2",
                "pages": [
                    {
                        "page_id": "a1-001",
                        "baseline": {
                            "path": str(baseline),
                            "sha256": sha256_file(baseline),
                        },
                        "baseline_mask": {
                            "path": str(baseline_mask),
                            "sha256": sha256_file(baseline_mask),
                        },
                        **{
                            field: {
                                "path": str(path),
                                "sha256": sha256_file(path),
                            }
                            for field, path in annotation_paths.items()
                        },
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    sealed = seal_source_only_evidence_manifest(
        parent_path,
        review_path,
        tmp_path / "evidence-v2.json",
    )

    assert sealed.schema_version == 2
    assert sealed.evidence_basis == "source-only-inpaint-evidence-v2"
    assert sealed.pages[0].target_text_mask is not None
    assert sealed.pages[0].ambiguous_structure_mask is not None


def test_optional_manifest_finalization_never_clobbers_racing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_image(tmp_path / "private-source.png")
    parent_path = _write_manifest(
        tmp_path / "parent.json",
        _manifest_payload(source, expected_edit="optional"),
    )
    parent = load_eval_manifest(parent_path)
    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "parent_manifest_sha256": parent.manifest_sha256,
                "decision_basis": "source-only-review",
                "pages": [
                    {"page_id": "a1-001", "expected_edit": "required"}
                ],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "racing-final.json"
    original_link = os.link

    def create_racing_output(source_path: Path, destination_path: Path) -> None:
        Path(destination_path).write_text("unrelated", encoding="utf-8")
        original_link(source_path, destination_path)

    monkeypatch.setattr(
        "scripts.inpaint_eval_contract.os.link",
        create_racing_output,
    )

    with pytest.raises(InpaintEvalManifestError) as raised:
        finalize_optional_eval_manifest(
            parent_path,
            decisions_path,
            output_path,
        )

    assert raised.value.code == "manifest_finalization_output_exists"
    assert output_path.read_text(encoding="utf-8") == "unrelated"
    assert not list(tmp_path.glob(".racing-final.json.*.tmp"))


def test_optional_manifest_finalization_rejects_parent_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_source = _write_image(
        tmp_path / "first-source.png",
        np.full((12, 16, 3), 220, dtype=np.uint8),
    )
    second_source = _write_image(
        tmp_path / "second-source.png",
        np.full((12, 16, 3), 180, dtype=np.uint8),
    )
    parent_path = _write_manifest(
        tmp_path / "parent.json",
        _manifest_payload(first_source, expected_edit="optional"),
    )
    original_parent = load_eval_manifest(parent_path)
    replacement_path = _write_manifest(
        tmp_path / "replacement.json",
        _manifest_payload(second_source, expected_edit="optional"),
    )
    replacement_bytes = replacement_path.read_bytes()
    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "parent_manifest_sha256": original_parent.manifest_sha256,
                "decision_basis": "source-only-review",
                "pages": [
                    {"page_id": "a1-001", "expected_edit": "required"}
                ],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "final.json"
    original_read_text = Path.read_text
    parent_replaced = False

    def replace_parent_before_validation(
        path: Path,
        *args,
        **kwargs,
    ) -> str:
        nonlocal parent_replaced
        if path.resolve() == parent_path.resolve() and not parent_replaced:
            parent_path.write_bytes(replacement_bytes)
            parent_replaced = True
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", replace_parent_before_validation)

    with pytest.raises(InpaintEvalManifestError) as raised:
        finalize_optional_eval_manifest(
            parent_path,
            decisions_path,
            output_path,
        )

    assert raised.value.code == "manifest_finalization_parent_changed"
    assert parent_replaced is True
    assert not output_path.exists()


def test_finalize_manifest_cli_returns_sanitized_json_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_finalize_module()
    source = _write_image(tmp_path / "private-source-title.png")
    parent_path = _write_manifest(
        tmp_path / "parent.json",
        _manifest_payload(source, expected_edit="optional"),
    )
    parent = load_eval_manifest(parent_path)
    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "parent_manifest_sha256": parent.manifest_sha256,
                "decision_basis": "source-only-review",
                "pages": [
                    {"page_id": "a1-001", "expected_edit": "none"}
                ],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "final.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_inpaint_eval_manifest.py",
            "--manifest",
            str(parent_path),
            "--decisions",
            str(decisions_path),
            "--output",
            str(output_path),
        ],
    )

    assert module.main() == 0
    success = json.loads(capsys.readouterr().out)
    assert success["corpus_id"] == "corpus-a1"
    assert success["parent_manifest_sha256"] == parent.manifest_sha256
    assert output_path.is_file()

    assert module.main() == 1
    failure = json.loads(capsys.readouterr().err)
    assert failure["error_code"] == "manifest_finalization_output_exists"
    assert source.name not in json.dumps(failure, ensure_ascii=False)


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        (lambda payload: payload.update(expected_count=2), "manifest_count_mismatch"),
        (lambda payload: payload["pages"][0].update(width=17), "manifest_dimension_mismatch"),
        (lambda payload: payload["pages"][0].update(sha256="0" * 64), "manifest_hash_mismatch"),
        (lambda payload: payload["pages"][0].update(size_bytes=1), "manifest_size_mismatch"),
        (lambda payload: payload.update(split_role="tunlng"), "manifest_split_role_invalid"),
        (
            lambda payload: payload.update(parent_manifest_sha256="a" * 64),
            "manifest_finalization_incomplete",
        ),
        (lambda payload: payload["pages"][0].update(page_id="원본 제목"), "manifest_page_id_invalid"),
        (lambda payload: payload["pages"][0].update(expected_edit="erase"), "manifest_expected_edit_invalid"),
    ],
)
def test_manifest_rejects_count_dimension_hash_and_identity_mutation(
    tmp_path: Path,
    mutation,
    error_code: str,
) -> None:
    source = _write_image(tmp_path / "private-source.png")
    payload = _manifest_payload(source)
    mutation(payload)
    manifest_path = _write_manifest(tmp_path / "manifest.json", payload)

    with pytest.raises(InpaintEvalManifestError) as raised:
        load_eval_manifest(manifest_path)

    assert raised.value.code == error_code
    assert str(source) not in str(raised.value)


def test_manifest_rejects_tampered_seal_and_invalid_image(tmp_path: Path) -> None:
    source = _write_image(tmp_path / "private-source.png")
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        _manifest_payload(source),
    )
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["pages"][0]["expected_edit"] = "none"
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(InpaintEvalManifestError) as raised:
        load_eval_manifest(manifest_path)
    assert raised.value.code == "manifest_seal_mismatch"

    invalid = tmp_path / "invalid.png"
    invalid.write_bytes(b"not-an-image")
    invalid_manifest = _write_manifest(
        tmp_path / "invalid-manifest.json",
        _manifest_payload(invalid),
    )
    with pytest.raises(InpaintEvalManifestError) as raised:
        load_eval_manifest(invalid_manifest)
    assert raised.value.code == "manifest_image_invalid"


def test_manifest_rejects_missing_and_duplicate_ids(tmp_path: Path) -> None:
    source = _write_image(tmp_path / "private-source.png")
    missing_payload = _manifest_payload(source)
    missing_payload["pages"][0]["path"] = str(tmp_path / "missing-sensitive-name.png")
    missing_payload["pages"][0]["sha256"] = "0" * 64
    missing_manifest = _write_manifest(tmp_path / "missing.json", missing_payload)
    with pytest.raises(InpaintEvalManifestError) as raised:
        load_eval_manifest(missing_manifest)
    assert raised.value.code == "manifest_file_missing"
    assert "missing-sensitive-name" not in str(raised.value)

    duplicate_payload = _manifest_payload(source, expected_count=2)
    duplicate_payload["pages"].append(dict(duplicate_payload["pages"][0]))
    duplicate_manifest = _write_manifest(tmp_path / "duplicate.json", duplicate_payload)
    with pytest.raises(InpaintEvalManifestError) as raised:
        load_eval_manifest(duplicate_manifest)
    assert raised.value.code == "manifest_duplicate_page_id"


def test_manifest_rejects_empty_and_unknown_schema_keys(tmp_path: Path) -> None:
    source = _write_image(tmp_path / "private-source.png")
    empty_payload = {
        "schema_version": 1,
        "corpus_id": "corpus-a1",
        "split_role": "tuning",
        "source_lock_git_sha": "1" * 40,
        "expected_count": 0,
        "pages": [],
    }
    empty_manifest = _write_manifest(tmp_path / "empty.json", empty_payload)
    with pytest.raises(InpaintEvalManifestError) as raised:
        load_eval_manifest(empty_manifest)
    assert raised.value.code == "manifest_expected_count_invalid"

    unknown_payload = _manifest_payload(source)
    unknown_payload["pages"][0]["target_glyph_masks"] = {}
    unknown_manifest = _write_manifest(tmp_path / "unknown.json", unknown_payload)
    with pytest.raises(InpaintEvalManifestError) as raised:
        load_eval_manifest(unknown_manifest)
    assert raised.value.code == "manifest_page_unknown_key"

    reference_payload = _manifest_payload(source)
    reference_payload["pages"][0]["baseline"] = {
        "path": str(source),
        "sha256": sha256_file(source),
        "unexpected": True,
    }
    reference_manifest = _write_manifest(
        tmp_path / "unknown-reference.json",
        reference_payload,
    )
    with pytest.raises(InpaintEvalManifestError) as raised:
        load_eval_manifest(reference_manifest)
    assert raised.value.code == "manifest_reference_unknown_key"


@pytest.mark.parametrize("page_id", ["con", "aux.png", "page..001", "page."])
def test_manifest_rejects_windows_unsafe_neutral_ids(
    tmp_path: Path,
    page_id: str,
) -> None:
    source = _write_image(tmp_path / "private-source.png")
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        _manifest_payload(source, page_id=page_id),
    )

    with pytest.raises(InpaintEvalManifestError) as raised:
        load_eval_manifest(manifest_path)

    assert raised.value.code == "manifest_page_id_invalid"


def test_page_spec_reverification_detects_post_load_source_replacement(
    tmp_path: Path,
) -> None:
    source = _write_image(tmp_path / "private-source.png")
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        _manifest_payload(source),
    )
    page = load_eval_manifest(manifest_path).pages[0]
    replacement = np.full((12, 16, 3), 20, dtype=np.uint8)
    _write_image(source, replacement)

    with pytest.raises(InpaintEvalManifestError) as raised:
        verify_eval_page_spec(page)

    assert raised.value.code == "manifest_hash_mismatch"
    assert str(source) not in str(raised.value)


def test_source_decode_rechecks_exact_bytes_after_page_verification(
    tmp_path: Path,
) -> None:
    source = _write_image(tmp_path / "private-source.bmp")
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        _manifest_payload(source),
    )
    page = load_eval_manifest(manifest_path).pages[0]
    verify_eval_page_spec(page)
    replacement = np.full((12, 16, 3), 20, dtype=np.uint8)
    _write_image(source, replacement)
    assert source.stat().st_size == page.size_bytes

    with pytest.raises(InpaintEvalManifestError) as raised:
        load_eval_source_array(page)

    assert raised.value.code == "manifest_hash_mismatch"
    assert str(source) not in str(raised.value)


def test_optional_references_are_verified_loaded_and_rechecked(
    tmp_path: Path,
) -> None:
    source = _write_image(tmp_path / "private-source.png")
    baseline = _write_image(
        tmp_path / "baseline.png",
        np.full((12, 16, 3), 220, dtype=np.uint8),
    )
    binary = np.zeros((12, 16), dtype=np.uint8)
    binary[2:10, 3:13] = 255
    baseline_mask = _write_image(tmp_path / "baseline-mask.png", binary)
    target_mask = _write_image(tmp_path / "target-mask.png", binary)
    protected_mask = _write_image(tmp_path / "protected-mask.png", binary)
    payload = _manifest_payload(source)
    payload["pages"][0].update(
        {
            "baseline": {
                "path": str(baseline),
                "sha256": sha256_file(baseline),
            },
            "baseline_mask": {
                "path": str(baseline_mask),
                "sha256": sha256_file(baseline_mask),
            },
            "target_glyph_mask": {
                "path": str(target_mask),
                "sha256": sha256_file(target_mask),
            },
            "protected_structure_mask": {
                "path": str(protected_mask),
                "sha256": sha256_file(protected_mask),
            },
        }
    )
    page = load_eval_manifest(
        _write_manifest(tmp_path / "manifest.json", payload)
    ).pages[0]

    verify_eval_page_spec(page)
    assert page.baseline is not None
    assert page.baseline_mask is not None
    assert page.target_glyph_mask is not None
    assert page.protected_structure_mask is not None
    assert load_rgb_reference_array(page.baseline, (12, 16, 3)).shape == (
        12,
        16,
        3,
    )
    target = load_binary_mask(page.target_glyph_mask, (12, 16, 3))
    assert target.shape == (12, 16)
    assert int(np.count_nonzero(target)) == 80

    _write_image(target_mask, np.full((12, 16), 255, dtype=np.uint8))
    with pytest.raises(InpaintEvalManifestError) as raised:
        verify_eval_page_spec(page)
    assert raised.value.code == "manifest_hash_mismatch"


def test_binary_mask_loader_uses_product_nonzero_mask_semantics(
    tmp_path: Path,
) -> None:
    raw = np.asarray([[0, 1, 127, 128, 255]], dtype=np.uint8)
    path = _write_image(tmp_path / "mask.png", raw)
    reference = EvalImageReference(path=path, sha256=sha256_file(path))

    loaded = load_binary_mask(reference, raw.shape)

    assert loaded.tolist() == [[0, 255, 255, 255, 255]]
    with pytest.raises(ValueError, match="^evaluation_mask_shape_mismatch$"):
        load_binary_mask(reference, (2, 5))


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P", "I;16"])
def test_manifest_and_direct_decoders_produce_the_same_rgb_pixels(
    tmp_path: Path,
    mode: str,
) -> None:
    source = tmp_path / f"source-{mode.replace(';', '-')}.png"
    if mode == "RGB":
        image = Image.new(mode, (16, 12), (10, 40, 220))
    elif mode == "RGBA":
        image = Image.new(mode, (16, 12), (10, 40, 220, 100))
    elif mode == "I;16":
        image = Image.fromarray(
            np.arange(16 * 12, dtype=np.uint16).reshape(12, 16),
        )
    else:
        image = Image.new(mode, (16, 12), 7)
    image.save(source)
    page = load_eval_manifest(
        _write_manifest(tmp_path / "manifest.json", _manifest_payload(source))
    ).pages[0]

    assert np.array_equal(load_eval_source_array(page), imk.read_image(str(source)))


def test_pixel_sha256_covers_shape_dtype_and_pixels() -> None:
    pixels = np.zeros((3, 4, 3), dtype=np.uint8)
    assert pixel_sha256(pixels) == pixel_sha256(pixels.copy())
    assert pixel_sha256(pixels) != pixel_sha256(pixels.reshape(2, 6, 3))
    changed = pixels.copy()
    changed[0, 0, 0] = 1
    assert pixel_sha256(pixels) != pixel_sha256(changed)


def test_multiple_manifests_require_globally_unique_neutral_ids(tmp_path: Path) -> None:
    source = _write_image(tmp_path / "private-source.png")
    first = _write_manifest(
        tmp_path / "first.json",
        _manifest_payload(source, corpus_id="corpus-a1", page_id="shared-001"),
    )
    second = _write_manifest(
        tmp_path / "second.json",
        _manifest_payload(source, corpus_id="corpus-a2", page_id="shared-001"),
    )

    with pytest.raises(InpaintEvalManifestError) as raised:
        load_eval_manifests([first, second])

    assert raised.value.code == "manifest_duplicate_page_id_global"


def test_multiple_manifests_reject_duplicate_sources_and_mixed_locks(
    tmp_path: Path,
) -> None:
    first_source = _write_image(tmp_path / "first.png")
    second_source = _write_image(
        tmp_path / "second.png",
        np.full((12, 16, 3), 220, dtype=np.uint8),
    )
    first = _write_manifest(
        tmp_path / "first.json",
        _manifest_payload(first_source, corpus_id="corpus-a1", page_id="a1-001"),
    )
    duplicate_source = _write_manifest(
        tmp_path / "duplicate-source.json",
        _manifest_payload(first_source, corpus_id="corpus-a2", page_id="a2-001"),
    )
    with pytest.raises(InpaintEvalManifestError) as raised:
        load_eval_manifests([first, duplicate_source])
    assert raised.value.code == "manifest_duplicate_source_hash"

    mixed_payload = _manifest_payload(
        second_source,
        corpus_id="corpus-a2",
        page_id="a2-001",
    )
    mixed_payload["source_lock_git_sha"] = "2" * 40
    mixed_lock = _write_manifest(tmp_path / "mixed-lock.json", mixed_payload)
    with pytest.raises(InpaintEvalManifestError) as raised:
        load_eval_manifests([first, mixed_lock])
    assert raised.value.code == "manifest_source_lock_mismatch"


def test_quality_metrics_measure_residue_damage_color_and_outside_change() -> None:
    source = np.full((32, 32, 3), 240, dtype=np.uint8)
    source[10:18, 11:19] = 20
    target = np.zeros((32, 32), dtype=np.uint8)
    target[10:18, 11:19] = 255
    mask = target.copy()
    cleaned = source.copy()
    cleaned[target > 0] = 240

    clean_metrics = build_quality_metrics(
        source,
        cleaned,
        mask,
        residue_target_mask=target,
        residue_target_is_annotation=True,
    )
    residue_metrics = build_quality_metrics(
        source,
        source.copy(),
        mask,
        residue_target_mask=target,
        residue_target_is_annotation=True,
    )

    assert clean_metrics["outside_changed_pixel_count_exact"] == 0
    assert clean_metrics["residue_target_coverage"] == 1.0
    assert clean_metrics["residue_target_component_coverages"] == [1.0]
    assert clean_metrics["residue_target_minimum_component_coverage"] == 1.0
    assert clean_metrics["residue_pixel_count"] < residue_metrics["residue_pixel_count"]
    assert clean_metrics["residue_score"] < residue_metrics["residue_score"]
    assert clean_metrics["color_delta_mean"] > 0.0

    damaged = cleaned.copy()
    protected = np.zeros((32, 32), dtype=np.uint8)
    protected[2:4, 3:5] = 255
    damaged[protected > 0] = 0
    damaged_metrics = build_quality_metrics(
        source,
        damaged,
        mask,
        residue_target_mask=target,
        residue_target_is_annotation=True,
        protected_structure_mask=protected,
    )
    assert damaged_metrics["outside_changed_pixel_count_exact"] == 4
    assert damaged_metrics["protected_structure_changed_pixel_count_exact"] == 4
    assert damaged_metrics["outline_damage_ratio"] == 1.0

    repaired_metrics = build_quality_metrics(
        source,
        cleaned,
        mask,
        residue_target_mask=target,
        residue_target_is_annotation=True,
        protected_structure_mask=protected,
        pre_composite_candidate_image=damaged,
    )
    assert repaired_metrics["protected_structure_changed_pixel_count_exact"] == 0
    assert (
        repaired_metrics[
            "pre_composite_protected_structure_changed_pixel_count_exact"
        ]
        == 4
    )

    unannotated_metrics = build_quality_metrics(source, cleaned, mask)
    assert unannotated_metrics["residue_target_coverage"] is None
    assert unannotated_metrics["residue_score"] is None

    empty_annotation_metrics = build_quality_metrics(
        source,
        cleaned,
        mask,
        residue_target_mask=np.zeros(mask.shape, dtype=np.uint8),
        residue_target_is_annotation=True,
    )
    assert empty_annotation_metrics["residue_target_is_annotation"] is True
    assert empty_annotation_metrics["residue_target_coverage"] is None
    assert empty_annotation_metrics["residue_target_component_coverages"] == []
    assert (
        empty_annotation_metrics["residue_target_minimum_component_coverage"]
        is None
    )


def test_quality_metrics_measure_each_connected_target_component() -> None:
    source = np.full((24, 32, 3), 240, dtype=np.uint8)
    target = np.zeros((24, 32), dtype=np.uint8)
    target[4:8, 4:8] = 255
    target[14:18, 22:26] = 255
    source[target > 0] = 20
    final_mask = np.zeros_like(target)
    final_mask[4:8, 4:8] = 255
    final_mask[14:18, 22:24] = 255

    metrics = build_quality_metrics(
        source,
        source.copy(),
        final_mask,
        residue_target_mask=target,
        residue_target_is_annotation=True,
    )

    assert metrics["residue_target_coverage"] == 0.75
    assert metrics["residue_target_component_coverages"] == [1.0, 0.5]
    assert metrics["residue_target_minimum_component_coverage"] == 0.5


def test_blind_panels_and_review_rows_are_deterministic_and_hide_key(
    tmp_path: Path,
) -> None:
    source = _write_image(tmp_path / "source.png")
    baseline_array = np.full((12, 16, 3), 220, dtype=np.uint8)
    candidate_array = np.full((12, 16, 3), 250, dtype=np.uint8)
    baseline = _write_image(tmp_path / "baseline.png", baseline_array)
    candidate = _write_image(tmp_path / "candidate.png", candidate_array)
    final_mask = _write_image(
        tmp_path / "mask.png",
        np.full((12, 16), 255, dtype=np.uint8),
    )
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = write_comparison_and_blind_panels(
        root_output=first_root,
        corpus_id="corpus-a1",
        page_id="a1-001",
        source_path=source,
        baseline_path=baseline,
        candidate_path=candidate,
        final_mask_path=final_mask,
    )
    second = write_comparison_and_blind_panels(
        root_output=second_root,
        corpus_id="corpus-a1",
        page_id="a1-001",
        source_path=source,
        baseline_path=baseline,
        candidate_path=candidate,
        final_mask_path=final_mask,
    )

    assert sha256_file(first["comparison_panel"]) == sha256_file(second["comparison_panel"])
    assert first["blind_eligible"] is True
    assert not (first_root / "blind_panels").exists()

    review_path, key_path = write_blind_review_jsonl(
        first_root,
        [first],
        duplicate_count=1,
        assignment_seed="a" * 64,
    )
    second_review_path, second_key_path = write_blind_review_jsonl(
        second_root,
        [second],
        duplicate_count=1,
        assignment_seed="a" * 64,
    )
    review_text = review_path.read_text(encoding="utf-8")
    key_text = key_path.read_text(encoding="utf-8")
    assert review_text == second_review_path.read_text(encoding="utf-8")
    assert key_text == second_key_path.read_text(encoding="utf-8")
    assert "candidate_label" not in review_text
    assert "baseline_label" not in review_text
    assert "candidate_label" in key_text
    assert len(review_text.splitlines()) == 2
    assert "duplicate_of" in key_text
    review_rows = [json.loads(line) for line in review_text.splitlines()]
    assert all("undetected_text" in row for row in review_rows)
    assert review_rows[0]["panel"] != review_rows[1]["panel"]
    assert "corpus-a1" not in review_text
    assert "a1-001" not in review_text
    assert key_path.parent.name == "blind_keys"
    key_rows = [json.loads(line) for line in key_text.splitlines()]
    original_key = next(row for row in key_rows if row["duplicate_of"] is None)
    duplicate_key = next(row for row in key_rows if row["duplicate_of"] is not None)
    assert original_key["candidate_label"] != duplicate_key["candidate_label"]
    review_by_id = {row["review_id"]: row for row in review_rows}
    original_panel = (
        first_root
        / "review"
        / review_by_id[original_key["review_id"]]["panel"]
    )
    duplicate_panel = (
        first_root
        / "review"
        / review_by_id[duplicate_key["review_id"]]["panel"]
    )
    assert sha256_file(original_panel) != sha256_file(duplicate_panel)
    second_review_by_id = {
        row["review_id"]: row
        for row in (
            json.loads(line)
            for line in second_review_path.read_text(encoding="utf-8").splitlines()
        )
    }
    second_original_panel = (
        second_root
        / "review"
        / second_review_by_id[original_key["review_id"]]["panel"]
    )
    assert sha256_file(original_panel) == sha256_file(second_original_panel)
    assert sha256_file(original_panel) != sha256_file(first["comparison_panel"])
    contract_text = (first_root / "review" / "review-contract.json").read_text(
        encoding="utf-8"
    )
    assert "corpus-a1" not in contract_text
    assert "a1-001" not in contract_text
    assert "blind_keys" in contract_text
    assert "comparison_panels" in contract_text
    with Image.open(original_panel) as opened:
        assert opened.info == {"review_id": original_key["review_id"]}


def test_blind_review_refuses_to_reuse_existing_package(
    tmp_path: Path,
) -> None:
    baseline = _write_image(
        tmp_path / "baseline.png",
        np.full((12, 16, 3), 220, dtype=np.uint8),
    )
    candidate = _write_image(
        tmp_path / "candidate.png",
        np.full((12, 16, 3), 250, dtype=np.uint8),
    )
    record = {
        "corpus_id": "corpus-a1",
        "page_id": "a1-001",
        "baseline_path": baseline,
        "baseline_sha256": sha256_file(baseline),
        "candidate_path": candidate,
        "candidate_sha256": sha256_file(candidate),
        "panel_size": (16, 12),
    }
    output = tmp_path / "output"
    review_path, _key_path = write_blind_review_jsonl(
        output,
        [record],
        duplicate_count=1,
        assignment_seed="a" * 64,
    )
    original_review = review_path.read_bytes()
    original_panels = sorted((output / "review" / "panels").glob("*.png"))
    assert len(original_panels) == 2

    with pytest.raises(FileExistsError, match="^blind_review_output_exists$"):
        write_blind_review_jsonl(
            output,
            [record],
            duplicate_count=0,
            assignment_seed="a" * 64,
        )

    assert review_path.read_bytes() == original_review
    assert sorted((output / "review" / "panels").glob("*.png")) == original_panels


def test_blind_review_binds_candidate_to_recorded_artifact_hash(
    tmp_path: Path,
) -> None:
    source = _write_image(tmp_path / "source.png")
    baseline = _write_image(
        tmp_path / "baseline.png",
        np.full((12, 16, 3), 220, dtype=np.uint8),
    )
    candidate = _write_image(
        tmp_path / "candidate.png",
        np.full((12, 16, 3), 250, dtype=np.uint8),
    )
    final_mask = _write_image(
        tmp_path / "mask.png",
        np.full((12, 16), 255, dtype=np.uint8),
    )
    output = tmp_path / "output"
    record = write_comparison_and_blind_panels(
        root_output=output,
        corpus_id="corpus-a1",
        page_id="a1-001",
        source_path=source,
        baseline_path=baseline,
        candidate_path=candidate,
        final_mask_path=final_mask,
    )
    _write_image(
        candidate,
        np.full((12, 16, 3), 30, dtype=np.uint8),
    )

    with pytest.raises(
        ValueError,
        match="^evaluation_reference_hash_mismatch$",
    ):
        write_blind_review_jsonl(
            output,
            [record],
            assignment_seed="a" * 64,
        )

    assert not (output / "review").exists()
    assert not (output / "blind_keys").exists()


def test_blind_review_seed_is_stable_and_requires_sealed_manifests(
    tmp_path: Path,
) -> None:
    source = _write_image(tmp_path / "private-source.png")
    first = load_eval_manifest(
        _write_manifest(tmp_path / "first.json", _manifest_payload(source))
    )
    second_payload = _manifest_payload(source, expected_edit="none")
    second = load_eval_manifest(
        _write_manifest(tmp_path / "second.json", second_payload)
    )

    assert derive_blind_review_seed([first]) == derive_blind_review_seed([first])
    assert derive_blind_review_seed([first]) != derive_blind_review_seed([second])
    assert derive_blind_review_seed([first], ["a" * 64]) != (
        derive_blind_review_seed([first], ["b" * 64])
    )
    with pytest.raises(
        ValueError,
        match="^blind_review_candidate_hash_invalid$",
    ):
        derive_blind_review_seed([first], ["private-title"])
    with pytest.raises(ValueError, match="^blind_review_seed_unavailable$"):
        derive_blind_review_seed([])


def test_missing_baseline_creates_comparison_but_not_fake_blind_panel(
    tmp_path: Path,
) -> None:
    source = _write_image(tmp_path / "source.png")
    candidate = _write_image(tmp_path / "candidate.png")
    final_mask = _write_image(
        tmp_path / "mask.png",
        np.full((12, 16), 255, dtype=np.uint8),
    )

    record = write_comparison_and_blind_panels(
        root_output=tmp_path / "output",
        corpus_id="corpus-a1",
        page_id="a1-001",
        source_path=source,
        baseline_path=None,
        candidate_path=candidate,
        final_mask_path=final_mask,
    )

    assert Path(record["comparison_panel"]).is_file()
    assert record["blind_eligible"] is False


def test_manifest_failure_summary_is_path_free_and_precedes_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_export_module()
    sensitive_path = tmp_path / "private-title-sensitive.png"
    payload = {
        "schema_version": 1,
        "corpus_id": "corpus-a1",
        "split_role": "tuning",
        "source_lock_git_sha": "1" * 40,
        "expected_count": 1,
        "pages": [
            {
                "page_id": "a1-001",
                "path": str(sensitive_path),
                "sha256": "0" * 64,
                "size_bytes": 1,
                "width": 16,
                "height": 12,
                "expected_edit": "required",
            }
        ],
    }
    manifest_path = _write_manifest(tmp_path / "manifest.json", payload)
    output = tmp_path / "output"
    monkeypatch.setattr(
        module,
        "select_managed_output_directory",
        lambda **_kwargs: (output, None),
    )
    monkeypatch.setattr(
        module,
        "TextBlockDetector",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("model must not load")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["export_inpaint_debug.py", "--manifest", str(manifest_path)],
    )

    assert module.main() == 1
    summary_text = (output / "metrics" / "summary.json").read_text(encoding="utf-8")
    assert "manifest_file_missing" in summary_text
    assert str(sensitive_path) not in summary_text
    assert sensitive_path.name not in summary_text


def test_main_refuses_nonempty_output_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_export_module()
    output = tmp_path / "existing-output"
    output.mkdir()
    marker = output / "existing.txt"
    marker.write_text("unchanged", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "select_managed_output_directory",
        lambda **_kwargs: (output, None),
    )
    monkeypatch.setattr(
        module,
        "TextBlockDetector",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("model must not load")
        ),
    )
    monkeypatch.setattr(sys, "argv", ["export_inpaint_debug.py"])

    assert module.main() == 1
    assert capsys.readouterr().err.strip() == "inpaint_output_directory_not_empty"
    assert marker.read_text(encoding="utf-8") == "unchanged"
    assert sorted(path.name for path in output.iterdir()) == ["existing.txt"]


@pytest.mark.parametrize("expected_edit", ["optional", "required"])
def test_holdout_manifest_preflight_blocks_candidate_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expected_edit: str,
) -> None:
    module = _load_export_module()
    source = _write_image(tmp_path / "private-holdout-source.png")
    manifest_path = _write_manifest(
        tmp_path / "holdout.json",
        _manifest_payload(
            source,
            corpus_id="corpus-b-primary",
            page_id="b-001",
            expected_edit=expected_edit,
            split_role="final-holdout-primary",
        ),
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        module,
        "select_managed_output_directory",
        lambda **_kwargs: (output, None),
    )
    monkeypatch.setattr(
        module,
        "TextBlockDetector",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("model must not load")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["export_inpaint_debug.py", "--manifest", str(manifest_path)],
    )

    assert module.main() == 1
    summary_text = (output / "metrics" / "summary.json").read_text(
        encoding="utf-8"
    )
    assert "manifest_holdout_not_source_review_finalized" in summary_text
    assert not (output / "corpus-b-primary").exists()
    assert not (output / "cleaned_contact_sheet.png").exists()


def test_holdout_manifest_preflight_accepts_source_review_derivation(
    tmp_path: Path,
) -> None:
    module = _load_export_module()
    source = _write_image(tmp_path / "private-holdout-source.png")
    payload = _manifest_payload(
        source,
        corpus_id="corpus-b-primary",
        page_id="b-001",
        expected_edit="required",
        split_role="final-holdout-primary",
    )
    payload.update(
        {
            "parent_manifest_sha256": "a" * 64,
            "expected_edit_basis": "source-only-review",
            "expected_edit_decisions_sha256": "b" * 64,
        }
    )
    manifest = load_eval_manifest(
        _write_manifest(tmp_path / "finalized.json", payload)
    )

    assert module._holdout_manifest_preflight_error((manifest,)) is None


def test_valid_manifest_main_orchestration_completes_with_neutral_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_export_module()
    source = _write_image(tmp_path / "private-title-sensitive.png")
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        _manifest_payload(source, expected_edit="optional"),
    )
    output = tmp_path / "output"
    calls: list[dict] = []
    events: list[tuple[str, dict]] = []
    _install_fake_export_runtime(module, monkeypatch, output, calls, events)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_inpaint_debug.py",
            "--manifest",
            str(manifest_path),
            "--require-image-count",
            "1",
        ],
    )

    assert module.main() == 0
    summary = json.loads(
        (output / "metrics" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["input_mode"] == "manifest"
    assert summary["image_count"] == 1
    assert summary["success_count"] == 1
    assert summary["required_gate_failure_count"] == 0
    assert (
        summary["manifest_corpora"]["corpus-a1"]["parent_manifest_sha256"]
        is None
    )
    assert calls[0]["public_page_id"] == "a1-001"
    assert [event[0] for event in events] == ["complete"]
    assert (output / "cleaned_contact_sheet.png").is_file()
    retained_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in output.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl", ".md"}
    )
    assert source.name not in retained_text
    assert str(source) not in retained_text


def test_main_projects_required_skip_to_page_summary_and_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_export_module()
    source = _write_image(tmp_path / "private-title-sensitive.png")
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        _manifest_payload(source, expected_edit="required"),
    )
    output = tmp_path / "output"
    calls: list[dict] = []
    events: list[tuple[str, dict]] = []
    _install_fake_export_runtime(module, monkeypatch, output, calls, events)
    base_process = module._process_image

    def skipped_process(*args, **kwargs):
        record = base_process(*args, **kwargs)
        record.update(
            {
                "outside_changed_pixel_count_exact": 0,
                "residue_target_is_annotation": False,
                "erase_mode_distribution": {"bubble_skipped": 1},
                "erase_skipped_reason_distribution": {
                    "microtexture_source_seed_unavailable": 1,
                },
            }
        )
        return record

    monkeypatch.setattr(module, "_process_image", skipped_process)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_inpaint_debug.py",
            "--manifest",
            str(manifest_path),
            "--require-quality-gates",
        ],
    )

    assert module.main() == 1
    page = json.loads(
        (output / "metrics" / "pages.jsonl").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (output / "metrics" / "summary.json").read_text(encoding="utf-8")
    )
    assert page["erase_mode_distribution"] == {"bubble_skipped": 1}
    assert page["erase_skipped_reason_distribution"] == {
        "microtexture_source_seed_unavailable": 1,
    }
    assert summary["erase_mode_distribution"] == {"bubble_skipped": 1}
    assert summary["erase_skipped_reason_distribution"] == {
        "microtexture_source_seed_unavailable": 1,
    }
    assert summary["required_skipped_block_count"] == 1
    assert summary["required_gate_failures"] == [
        "corpus-a1/a1-001:required_bubble_erase_skipped"
    ]
    assert [event[0] for event in events] == ["fail"]


def test_manifest_main_builds_standalone_seeded_blind_review_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_export_module()
    source = _write_image(tmp_path / "private-source-title.png")
    baseline = _write_image(
        tmp_path / "private-baseline-title.png",
        np.full((12, 16, 3), 220, dtype=np.uint8),
    )
    payload = _manifest_payload(source, expected_edit="optional")
    payload["pages"][0]["baseline"] = {
        "path": str(baseline),
        "sha256": sha256_file(baseline),
    }
    manifest_path = _write_manifest(tmp_path / "manifest.json", payload)
    output = tmp_path / "output"
    calls: list[dict] = []
    events: list[tuple[str, dict]] = []
    _install_fake_export_runtime(module, monkeypatch, output, calls, events)
    captured_seed_inputs: list[list[str]] = []
    real_derive_blind_review_seed = module.derive_blind_review_seed

    def capture_seed_inputs(manifests, candidate_sha256s=()):
        manifest_list = list(manifests)
        candidate_list = list(candidate_sha256s)
        captured_seed_inputs.append(candidate_list)
        return real_derive_blind_review_seed(
            manifest_list,
            candidate_list,
        )

    monkeypatch.setattr(
        module,
        "derive_blind_review_seed",
        capture_seed_inputs,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_inpaint_debug.py",
            "--manifest",
            str(manifest_path),
            "--blind-review-duplicate-count",
            "1",
        ],
    )

    assert module.main() == 0

    review_dir = output / "review"
    review_rows = [
        json.loads(line)
        for line in (review_dir / "blind-review.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(review_rows) == 2
    assert all((review_dir / row["panel"]).is_file() for row in review_rows)
    assert not (output / "blind_panels").exists()
    reviewer_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in review_dir.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl"}
    )
    assert "corpus-a1" not in reviewer_text
    assert "a1-001" not in reviewer_text
    assert source.name not in reviewer_text
    assert baseline.name not in reviewer_text
    candidate_path = output / "corpus-a1" / "cleaned_images" / "a1-001_cleaned.png"
    assert captured_seed_inputs == [[sha256_file(candidate_path)]]
    assert [event[0] for event in events] == ["complete"]


def test_manifest_main_counts_missing_cuda_memory_availability_as_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_export_module()
    source = _write_image(tmp_path / "private-source-title.png")
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        _manifest_payload(source, expected_edit="optional"),
    )
    output = tmp_path / "output"
    calls: list[dict] = []
    events: list[tuple[str, dict]] = []
    _install_fake_export_runtime(module, monkeypatch, output, calls, events)
    fake_process = module._process_image

    def process_with_missing_availability(*args, **kwargs):
        record = fake_process(*args, **kwargs)
        record["inpaint_runtime_inference_call_count"] = 1
        record["inpaint_runtime_diagnostics"] = [
            {
                "is_inference": True,
                "phase": "block",
                "status": "completed",
            }
        ]
        return record

    monkeypatch.setattr(module, "_process_image", process_with_missing_availability)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_inpaint_debug.py",
            "--manifest",
            str(manifest_path),
        ],
    )

    assert module.main() == 0
    summary = json.loads(
        (output / "metrics" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["runtime_inference_call_count"] == 1
    assert summary["cuda_memory_diagnostics_unavailable_count"] == 1


def test_direct_main_replaces_filename_path_and_contact_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_export_module()
    source = _write_image(tmp_path / "private-title-日本語-sensitive.png")
    output = tmp_path / "output"
    calls: list[dict] = []
    events: list[tuple[str, dict]] = []
    _install_fake_export_runtime(module, monkeypatch, output, calls, events)
    contact_records: list[dict] = []
    real_write_contact_sheet = module._write_contact_sheet

    def capture_contact_sheet(*args, **kwargs) -> None:
        records_by_corpus = args[1]
        contact_records.extend(
            record
            for records in records_by_corpus.values()
            for record in records
        )
        real_write_contact_sheet(*args, **kwargs)

    monkeypatch.setattr(module, "_write_contact_sheet", capture_contact_sheet)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_inpaint_debug.py",
            "--input",
            str(source),
            "--glob",
            source.name,
            "--require-image-count",
            "1",
        ],
    )

    assert module.main() == 0
    summary = json.loads(
        (output / "metrics" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["input_mode"] == "direct"
    assert summary["glob"] is None
    assert calls[0]["public_page_id"] == "direct-001"
    assert {record["image"] for record in contact_records} == {"direct-001"}
    assert [event[0] for event in events] == ["complete"]
    retained_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in output.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl", ".md"}
    )
    retained_names = "\n".join(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
    )
    assert source.name not in retained_text
    assert str(source) not in retained_text
    assert source.name not in retained_names
    assert "direct-001" in retained_text


def test_direct_processing_failure_summary_uses_neutral_id_and_safe_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_export_module()
    source = _write_image(tmp_path / "private-title-日本語-sensitive.png")
    output = tmp_path / "output"
    calls: list[dict] = []
    events: list[tuple[str, dict]] = []
    _install_fake_export_runtime(module, monkeypatch, output, calls, events)

    def fail_process(*_args, **_kwargs):
        raise RuntimeError(str(source))

    monkeypatch.setattr(module, "_process_image", fail_process)
    monkeypatch.setattr(
        sys,
        "argv",
        ["export_inpaint_debug.py", "--input", str(source)],
    )

    assert module.main() == 1
    summary_text = (output / "metrics" / "summary.json").read_text(
        encoding="utf-8"
    )
    assert source.name not in summary_text
    assert str(source) not in summary_text
    assert "direct-001" in summary_text
    assert "processing_failed" in summary_text
    summary = json.loads(summary_text)
    assert summary["failures"][0]["cause_code"] is None
    assert [event[0] for event in events] == ["fail"]


def test_processing_failure_preserves_only_safe_cause_codes() -> None:
    module = _load_export_module()

    safe = module._processing_failure_record(
        corpus_id="corpus-a1",
        page_id="a1-001",
        exc=ValueError("evaluation_reference_hash_mismatch"),
    )
    manifest_safe = module._processing_failure_record(
        corpus_id="corpus-a1",
        page_id="a1-001",
        exc=InpaintEvalManifestError("manifest_hash_mismatch"),
    )
    missing_input = module._processing_failure_record(
        corpus_id="private",
        page_id="direct-001",
        exc=FileNotFoundError("inpaint_input_image_missing"),
    )
    unsafe = module._processing_failure_record(
        corpus_id="corpus-a1",
        page_id="a1-001",
        exc=RuntimeError("C:/private/source-title.png"),
    )
    shaped_private = module._processing_failure_record(
        corpus_id="corpus-a1",
        page_id="a1-001",
        exc=RuntimeError("evaluation_private_title"),
    )

    assert safe["error_code"] == "processing_failed"
    assert safe["cause_code"] == "evaluation_reference_hash_mismatch"
    assert manifest_safe["cause_code"] == "manifest_hash_mismatch"
    assert missing_input["cause_code"] == "inpaint_input_image_missing"
    assert unsafe["cause_code"] is None
    assert shaped_private["cause_code"] is None
    assert "source-title" not in json.dumps(unsafe, ensure_ascii=False)


def test_page_metrics_projects_runtime_diagnostics_to_safe_fields(
    tmp_path: Path,
) -> None:
    module = _load_export_module()
    sensitive_path = "C:/private/models/lama.ckpt"

    output = module._write_page_metrics_jsonl(
        tmp_path,
        {
            "corpus-a1": [
                {
                    "page_id": "a1-001",
                    "routing_source_raw_owned_pixel_count": 11,
                    "routing_ownership_protect_pixel_count": 12,
                    "routing_positive_claim_pixel_count": 13,
                    "routing_positive_edit_pixel_count": 14,
                    "routing_claim_providers": ["ctd_full_page_raw"],
                    "residue_target_component_coverages": [1.0, 0.99],
                    "residue_target_minimum_component_coverage": 0.99,
                    "inpaint_runtime_diagnostics": [
                        {
                            "phase": "block",
                            "actual_device": "cuda:0",
                            "cuda_memory_diagnostics_available": True,
                            "model_path": sensitive_path,
                            "unexpected_private_value": "source-title.png",
                        },
                        {
                            "phase": "generic",
                            "actual_device": "cuda:1",
                            "status": "completed",
                            "is_inference": True,
                            "mask_pixel_count": 20,
                        },
                        {
                            "phase": "positive_evidence",
                            "actual_device": "cuda:0",
                            "status": "completed",
                            "is_inference": True,
                            "mask_pixel_count": 14,
                        },
                        {
                            "phase": sensitive_path,
                            "actual_device": "secret_model_name",
                            "status": "private-source-title",
                            "elapsed_seconds": float("nan"),
                            "mask_bbox": [0, 0, 10, sensitive_path],
                            "session_providers": [sensitive_path],
                            "cpu_fallback_used": "false",
                            "cuda_memory_diagnostics_unavailable": sensitive_path,
                        },
                    ],
                }
            ]
        },
    )
    row = json.loads(output.read_text(encoding="utf-8"))

    assert row["routing_source_raw_owned_pixel_count"] == 11
    assert row["routing_ownership_protect_pixel_count"] == 12
    assert row["routing_positive_claim_pixel_count"] == 13
    assert row["routing_positive_edit_pixel_count"] == 14
    assert row["routing_claim_providers"] == ["ctd_full_page_raw"]
    assert row["residue_target_component_coverages"] == [1.0, 0.99]
    assert row["residue_target_minimum_component_coverage"] == 0.99

    assert row["inpaint_runtime_diagnostics"] == [
        {
            "actual_device": "cuda:0",
            "cuda_memory_diagnostics_available": True,
            "phase": "block",
        },
        {
            "actual_device": "cuda:1",
            "is_inference": True,
            "mask_pixel_count": 20,
            "phase": "generic",
            "status": "completed",
        },
        {
            "actual_device": "cuda:0",
            "is_inference": True,
            "mask_pixel_count": 14,
            "phase": "positive_evidence",
            "status": "completed",
        },
    ]
    retained = output.read_text(encoding="utf-8")
    assert sensitive_path not in retained
    assert "source-title" not in retained


@pytest.mark.parametrize(
    ("case", "error_code"),
    [
        ("conflict", "manifest_and_direct_input_conflict"),
        ("negative-duplicates", "blind_duplicate_count_invalid"),
        ("duplicate-out-of-range", "blind_duplicate_count_out_of_range"),
    ],
)
def test_cli_contract_errors_fail_before_model_load_without_path_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    error_code: str,
) -> None:
    module = _load_export_module()
    source = _write_image(tmp_path / "private-title-sensitive.png")
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        _manifest_payload(source, expected_edit="optional"),
    )
    output = tmp_path / case
    monkeypatch.setattr(
        module,
        "select_managed_output_directory",
        lambda **_kwargs: (output, None),
    )
    monkeypatch.setattr(
        module,
        "TextBlockDetector",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("model must not load")
        ),
    )
    if case == "conflict":
        arguments = [
            "--manifest",
            str(manifest_path),
            "--input",
            str(source),
        ]
    elif case == "negative-duplicates":
        arguments = [
            "--input",
            str(source),
            "--blind-review-duplicate-count",
            "-1",
        ]
    else:
        arguments = [
            "--manifest",
            str(manifest_path),
            "--blind-review-duplicate-count",
            "1",
        ]
    monkeypatch.setattr(sys, "argv", ["export_inpaint_debug.py", *arguments])

    assert module.main() == 1
    summary_text = (output / "metrics" / "summary.json").read_text(
        encoding="utf-8"
    )
    assert error_code in summary_text
    assert source.name not in summary_text
    assert str(source) not in summary_text


def test_manifest_expected_none_allows_empty_cuda_page_gate() -> None:
    module = _load_export_module()
    summary = {
        "input_mode": "manifest",
        "inpainter": "lama_large_512px",
        "use_gpu": True,
        "hd_strategy": "Original",
        "inpainter_runtime": {
            "actual_device": "cuda",
            "device_verified_from_model": True,
            "cpu_fallback_used": False,
        },
        "cpu_fallback_count": 0,
        "non_cuda_refiner_count": 0,
        "peak_vram_unavailable_count": 0,
        "peak_vram_reset_failure_count": 0,
        "cuda_memory_diagnostics_unavailable_count": 0,
        "required_zero_block_count": 0,
        "required_empty_final_mask_count": 0,
        "expected_edit_required_count": 0,
        "expected_edit_active_count": 0,
        "runtime_inference_call_count": 0,
        "image_count": 1,
        "success_count": 1,
    }
    record = {
        "page_id": "a1-001",
        "expected_edit": "none",
        "block_count": 0,
        "final_mask_pixel_count": 0,
    }

    assert module._required_gate_failures(
        summary,
        {"corpus-a1": [record]},
        require_cuda_lama=True,
        require_rounded_bubble_gate=False,
    ) == []

    record["final_mask_pixel_count"] = 1
    failures = module._required_gate_failures(
        summary,
        {"corpus-a1": [record]},
        require_cuda_lama=True,
        require_rounded_bubble_gate=False,
    )
    assert failures == ["corpus-a1/a1-001:unexpected_edit_mask"]


def test_quality_gate_fails_closed_for_damage_truncation_and_empty_annotation() -> None:
    module = _load_export_module()
    summary = {
        "input_mode": "direct",
        "image_count": 1,
        "success_count": 1,
    }
    record = {
        "page_id": "a1-001",
        "outside_changed_pixel_count_exact": 1,
        "protected_structure_changed_pixel_count_exact": 2,
        "protected_structure_annotation_available": True,
        "protected_structure_annotation_changed_pixel_count_exact": 2,
        "residue_pass_truncated_block_count": 3,
        "residue_target_is_annotation": True,
        "residue_target_coverage": None,
        "erase_mode_distribution": {},
        "erase_skipped_reason_distribution": {},
    }

    failures = module._required_gate_failures(
        summary,
        {"corpus-a1": [record]},
        require_cuda_lama=False,
        require_rounded_bubble_gate=False,
        require_quality_gates=True,
    )

    assert failures == [
        "corpus-a1/a1-001:changed_outside_final_mask",
        "corpus-a1/a1-001:protected_structure_changed",
        "corpus-a1/a1-001:cleanup_truncated",
        "corpus-a1/a1-001:target_coverage_below_98pct",
        "corpus-a1/a1-001:target_component_coverage_below_98pct",
    ]


def test_quality_gate_rejects_unclassified_optional_pages() -> None:
    module = _load_export_module()
    failures = module._required_gate_failures(
        {"input_mode": "direct", "image_count": 1, "success_count": 1},
        {
            "corpus-b-primary": [
                {
                    "page_id": "b-001",
                    "expected_edit": "optional",
                    "outside_changed_pixel_count_exact": 0,
                    "protected_structure_changed_pixel_count_exact": 0,
                    "protected_structure_annotation_available": True,
                    "protected_structure_annotation_changed_pixel_count_exact": 0,
                    "residue_pass_truncated_block_count": 0,
                    "residue_target_is_annotation": False,
                    "erase_mode_distribution": {},
                    "erase_skipped_reason_distribution": {},
                }
            ]
        },
        require_cuda_lama=False,
        require_rounded_bubble_gate=False,
        require_quality_gates=True,
    )

    assert failures == [
        "corpus-b-primary/b-001:expected_edit_optional_not_final"
    ]


def test_quality_gate_rejects_required_bubble_erase_skip() -> None:
    module = _load_export_module()
    failures = module._required_gate_failures(
        {"input_mode": "direct", "image_count": 1, "success_count": 1},
        {
            "corpus-a1": [
                {
                    "page_id": "a1-001",
                    "expected_edit": "required",
                    "outside_changed_pixel_count_exact": 0,
                    "protected_structure_changed_pixel_count_exact": 0,
                    "protected_structure_annotation_available": True,
                    "protected_structure_annotation_changed_pixel_count_exact": 0,
                    "residue_pass_truncated_block_count": 0,
                    "residue_target_is_annotation": False,
                    "erase_mode_distribution": {"bubble_skipped": 1},
                    "erase_skipped_reason_distribution": {
                        "microtexture_source_seed_unavailable": 1,
                    },
                }
            ]
        },
        require_cuda_lama=False,
        require_rounded_bubble_gate=False,
        require_quality_gates=True,
    )

    assert failures == [
        "corpus-a1/a1-001:required_bubble_erase_skipped"
    ]

    benign_empty_seed = {
        "page_id": "a1-001",
        "expected_edit": "required",
        "outside_changed_pixel_count_exact": 0,
        "protected_structure_changed_pixel_count_exact": 0,
        "protected_structure_annotation_available": True,
        "protected_structure_annotation_changed_pixel_count_exact": 0,
        "residue_pass_truncated_block_count": 0,
        "residue_target_is_annotation": False,
        "erase_mode_distribution": {"bubble_skipped": 1},
        "erase_skipped_reason_distribution": {"empty_seed": 1},
    }
    assert module._required_gate_failures(
        {"input_mode": "direct", "image_count": 1, "success_count": 1},
        {"corpus-a1": [benign_empty_seed]},
        require_cuda_lama=False,
        require_rounded_bubble_gate=False,
        require_quality_gates=True,
    ) == []


@pytest.mark.parametrize(
    "reason",
    (
        "bubble_interior_cap_source_seed_unavailable",
        "bubble_interior_cap_source_seed_partially_suppressed",
        "bubble_protected_source_seed_unavailable",
        "bubble_residual_source_seed_unavailable",
        "line_art_source_seed_unavailable",
        "microtexture_source_seed_unavailable",
        "microtexture_source_seed_partially_suppressed",
        "text_prior_unavailable_source_seed_unavailable",
    ),
)
def test_quality_gate_rejects_each_required_source_seed_unavailable_route(
    reason: str,
) -> None:
    module = _load_export_module()
    record = {
        "page_id": "a1-001",
        "expected_edit": "required",
        "outside_changed_pixel_count_exact": 0,
        "protected_structure_changed_pixel_count_exact": 0,
        "protected_structure_annotation_available": True,
        "protected_structure_annotation_changed_pixel_count_exact": 0,
        "residue_pass_truncated_block_count": 0,
        "residue_target_is_annotation": False,
        "erase_mode_distribution": {"bubble_skipped": 1},
        "erase_skipped_reason_distribution": {reason: 1},
    }

    failures = module._required_gate_failures(
        {"input_mode": "direct", "image_count": 1, "success_count": 1},
        {"corpus-a1": [record]},
        require_cuda_lama=False,
        require_rounded_bubble_gate=False,
        require_quality_gates=True,
    )

    assert failures == [
        "corpus-a1/a1-001:required_bubble_erase_skipped"
    ]


def test_quality_gate_accepts_bubble_delegated_to_lama_priority() -> None:
    module = _load_export_module()
    record = {
        "page_id": "a1-001",
        "expected_edit": "required",
        "block_count": 2,
        "final_mask_pixel_count": 16,
        "outside_changed_pixel_count_exact": 0,
        "protected_structure_changed_pixel_count_exact": 0,
        "protected_structure_annotation_available": True,
        "protected_structure_annotation_changed_pixel_count_exact": 0,
        "residue_pass_truncated_block_count": 0,
        "residue_target_is_annotation": False,
        "erase_mode_distribution": {"bubble_lama_fallback": 1},
        "erase_skipped_reason_distribution": {"lama_priority_owned": 1},
    }

    failures = module._required_gate_failures(
        {"input_mode": "direct", "image_count": 1, "success_count": 1},
        {"corpus-a1": [record]},
        require_cuda_lama=False,
        require_rounded_bubble_gate=False,
        require_quality_gates=True,
    )

    assert failures == []


def test_quality_gate_requires_source_review_finalization_for_holdout() -> None:
    module = _load_export_module()
    summary = {
        "input_mode": "manifest",
        "image_count": 1,
        "success_count": 1,
        "aggregate_residue_score": 0.5,
        "baseline_aggregate_residue_score": 0.6,
        "manifest_corpora": {
            "corpus-b-primary": {
                "expected_count": 1,
                "split_role": "final-holdout-primary",
                "parent_manifest_sha256": None,
                "expected_edit_basis": None,
                "expected_edit_decisions_sha256": None,
            }
        },
    }
    record = {
        "page_id": "b-001",
        "expected_edit": "required",
        "block_count": 1,
        "final_mask_pixel_count": 10,
        "outside_changed_pixel_count_exact": 0,
        "protected_structure_changed_pixel_count_exact": 0,
        "protected_structure_annotation_available": True,
        "protected_structure_annotation_changed_pixel_count_exact": 0,
        "residue_pass_truncated_block_count": 0,
        "residue_target_is_annotation": False,
        "erase_mode_distribution": {},
        "erase_skipped_reason_distribution": {},
    }

    failures = module._required_gate_failures(
        summary,
        {"corpus-b-primary": [record]},
        require_cuda_lama=False,
        require_rounded_bubble_gate=False,
        require_quality_gates=True,
    )

    assert failures == [
        "corpus-b-primary:holdout_not_source_review_finalized"
    ]

    summary["manifest_corpora"]["corpus-b-primary"].update(
        {
            "parent_manifest_sha256": "a" * 64,
            "expected_edit_basis": "source-only-review",
            "expected_edit_decisions_sha256": "b" * 64,
        }
    )
    assert module._required_gate_failures(
        summary,
        {"corpus-b-primary": [record]},
        require_cuda_lama=False,
        require_rounded_bubble_gate=False,
        require_quality_gates=True,
    ) == []


def test_quality_gate_fails_closed_when_a_required_metric_is_missing() -> None:
    module = _load_export_module()
    failures = module._required_gate_failures(
        {"input_mode": "direct", "image_count": 1, "success_count": 1},
        {
            "corpus-a1": [
                {
                    "page_id": "a1-001",
                    "expected_edit": "required",
                    "protected_structure_changed_pixel_count_exact": 0,
                    "protected_structure_annotation_available": True,
                    "protected_structure_annotation_changed_pixel_count_exact": 0,
                    "residue_pass_truncated_block_count": 0,
                    "residue_target_is_annotation": False,
                    "erase_mode_distribution": {},
                    "erase_skipped_reason_distribution": {},
                }
            ]
        },
        require_cuda_lama=False,
        require_rounded_bubble_gate=False,
        require_quality_gates=True,
    )

    assert failures == [
        (
            "corpus-a1/a1-001:quality_metric_missing:"
            "outside_changed_pixel_count_exact"
        )
    ]


def test_quality_gate_treats_derived_structure_proxy_as_advisory_only() -> None:
    module = _load_export_module()
    record = {
        "page_id": "a1-001",
        "expected_edit": "required",
        "outside_changed_pixel_count_exact": 0,
        "protected_structure_changed_pixel_count_exact": 6971,
        "protected_structure_annotation_available": False,
        "protected_structure_annotation_changed_pixel_count_exact": None,
        "residue_pass_truncated_block_count": 0,
        "residue_target_is_annotation": False,
        "erase_mode_distribution": {},
        "erase_skipped_reason_distribution": {},
    }

    failures = module._required_gate_failures(
        {"input_mode": "direct", "image_count": 1, "success_count": 1},
        {"corpus-a1": [record]},
        require_cuda_lama=False,
        require_rounded_bubble_gate=False,
        require_quality_gates=True,
    )

    assert failures == [
        "corpus-a1/a1-001:protected_structure_annotation_missing"
    ]


def test_quality_gate_requires_ambiguous_annotation_for_manifest_v2() -> None:
    module = _load_export_module()
    summary = {
        "input_mode": "manifest",
        "image_count": 1,
        "success_count": 1,
        "aggregate_residue_score": 0.5,
        "baseline_aggregate_residue_score": 0.6,
        "manifest_corpora": {
            "corpus-a1": {
                "schema_version": 2,
                "expected_count": 1,
                "split_role": "tuning",
            }
        },
    }
    record = {
        "page_id": "a1-001",
        "expected_edit": "required",
        "block_count": 1,
        "final_mask_pixel_count": 10,
        "outside_changed_pixel_count_exact": 0,
        "protected_structure_changed_pixel_count_exact": 0,
        "protected_structure_annotation_available": True,
        "protected_structure_annotation_changed_pixel_count_exact": 0,
        "ambiguous_structure_annotation_available": False,
        "residue_pass_truncated_block_count": 0,
        "residue_target_is_annotation": False,
        "erase_mode_distribution": {},
        "erase_skipped_reason_distribution": {},
    }

    failures = module._required_gate_failures(
        summary,
        {"corpus-a1": [record]},
        require_cuda_lama=False,
        require_rounded_bubble_gate=False,
        require_quality_gates=True,
    )

    assert failures == [
        "corpus-a1/a1-001:ambiguous_structure_annotation_missing"
    ]


def test_quality_gate_requires_component_coverage_and_residue_improvement() -> None:
    module = _load_export_module()
    summary = {
        "input_mode": "manifest",
        "image_count": 1,
        "success_count": 1,
        "aggregate_residue_score": 0.7,
        "baseline_aggregate_residue_score": 0.6,
        "manifest_corpora": {
            "corpus-a1": {
                "schema_version": 2,
                "expected_count": 1,
                "split_role": "tuning",
            }
        },
    }
    record = {
        "page_id": "a1-001",
        "expected_edit": "required",
        "block_count": 1,
        "final_mask_pixel_count": 10,
        "outside_changed_pixel_count_exact": 0,
        "protected_structure_changed_pixel_count_exact": 0,
        "protected_structure_annotation_available": True,
        "protected_structure_annotation_changed_pixel_count_exact": 0,
        "ambiguous_structure_annotation_available": True,
        "residue_pass_truncated_block_count": 0,
        "residue_target_is_annotation": True,
        "residue_target_pixel_count": 10,
        "residue_target_coverage": 0.99,
        "residue_target_minimum_component_coverage": 0.75,
        "residue_score": 0.7,
        "baseline_residue_score": 0.6,
        "erase_mode_distribution": {},
        "erase_skipped_reason_distribution": {},
    }

    failures = module._required_gate_failures(
        summary,
        {"corpus-a1": [record]},
        require_cuda_lama=False,
        require_rounded_bubble_gate=False,
        require_quality_gates=True,
    )

    assert failures == [
        "corpus-a1/a1-001:target_component_coverage_below_98pct",
        "corpus-a1/a1-001:residue_worse_than_baseline",
        "aggregate:residue_not_reduced_from_baseline",
    ]


def test_quality_gate_uses_private_structure_annotation_for_damage() -> None:
    module = _load_export_module()
    record = {
        "page_id": "a1-001",
        "expected_edit": "required",
        "outside_changed_pixel_count_exact": 0,
        "protected_structure_changed_pixel_count_exact": 0,
        "protected_structure_annotation_available": True,
        "protected_structure_annotation_changed_pixel_count_exact": 1,
        "residue_pass_truncated_block_count": 0,
        "residue_target_is_annotation": False,
        "erase_mode_distribution": {},
        "erase_skipped_reason_distribution": {},
    }

    failures = module._required_gate_failures(
        {"input_mode": "direct", "image_count": 1, "success_count": 1},
        {"corpus-a1": [record]},
        require_cuda_lama=False,
        require_rounded_bubble_gate=False,
        require_quality_gates=True,
    )

    assert failures == ["corpus-a1/a1-001:protected_structure_changed"]


def test_baseline_parity_gate_requires_both_locked_artifacts_and_exact_sha() -> None:
    module = _load_export_module()
    summary = {
        "input_mode": "manifest",
        "image_count": 2,
        "success_count": 2,
        "manifest_corpora": {"corpus-a1": {"expected_count": 2}},
    }
    records = [
        {
            "page_id": "a1-001",
            "expected_edit": "optional",
            "block_count": 1,
            "final_mask_pixel_count": 10,
            "cleaned_matches_baseline_sha256": True,
            "final_mask_matches_baseline_sha256": False,
            "cleaned_matches_baseline_pixel_sha256": True,
            "final_mask_matches_baseline_pixel_sha256": True,
        },
        {
            "page_id": "a1-002",
            "expected_edit": "optional",
            "block_count": 1,
            "final_mask_pixel_count": 10,
            "cleaned_matches_baseline_sha256": None,
            "final_mask_matches_baseline_sha256": True,
            "cleaned_matches_baseline_pixel_sha256": True,
            "final_mask_matches_baseline_pixel_sha256": True,
        },
    ]

    failures = module._required_gate_failures(
        summary,
        {"corpus-a1": records},
        require_cuda_lama=False,
        require_rounded_bubble_gate=False,
        require_baseline_parity=True,
    )

    assert failures == [
        "corpus-a1/a1-001:baseline_final_mask_sha_mismatch",
        "corpus-a1/a1-002:baseline_cleaned_unavailable",
    ]
