from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark_inpaint_quality_gates as gates  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _fake_result_root(root: Path) -> tuple[Path, list[str]]:
    result_root = root / "results"
    result_root.mkdir()
    profiles = [
        gates.BASELINE,
        gates.CandidateProfile(
            slug="candidate-fp32",
            label="Candidate FP32",
            phase="model",
            inpainter_key="lama_large_512px",
            precision="fp32",
            inpaint_size=2048,
            mask_mode="glyph",
            dilation=2,
            structure_protect=True,
        ),
    ]
    case_ids = ["case-one", "case-two"]
    candidate_results = []
    for profile_order, profile in enumerate(profiles):
        cases = []
        for case_order, case_id in enumerate(case_ids):
            artifacts = {}
            for name in (
                "original",
                "raw_mask",
                "final_mask",
                "mask_overlay",
                "cleaned",
                "diff",
            ):
                path = (
                    result_root
                    / "candidates"
                    / f"{profile_order:03d}"
                    / f"{case_order:03d}"
                    / f"{name}.png"
                )
                gates._write_image(
                    path,
                    np.full(
                        (8, 8, 3),
                        profile_order * 40 + case_order * 5,
                        dtype=np.uint8,
                    ),
                )
                artifacts[name] = gates._artifact_record(path, result_root)
            case_result = {
                "case_order": case_order,
                "case_id": case_id,
                "case_contract_sha256": str(case_order) * 64,
                "status": "completed",
                "failure": "",
                "review_roi": [0, 0, 8, 8],
                "model_roi": [0, 0, 8, 8],
                "mask_pixel_count": 8,
                "inference_seconds": 1.25 + profile_order,
                "run_diagnostics": {
                    "status": "completed",
                    "oom_retry_count": 0,
                    "oom_retry_roi": None,
                },
                "changed_pixels": {
                    "changed_pixel_count": 8,
                    "changed_inside_mask_pixel_count": 8,
                    "changed_outside_mask_pixel_count": 0,
                },
                "artifacts": artifacts,
            }
            case_result["result_sha256"] = gates.canonical_sha256(
                case_result
            )
            cases.append(case_result)
        profile_result = {
            "profile_order": profile_order,
            "profile": gates.asdict(profile),
            "runtime": {
                "actual_device": "cuda:0",
                "actual_precision": profile.precision,
                "fp32_promotion_eligible": profile.precision == "fp32",
            },
            "model_load_seconds": 2.5 + profile_order,
            "total_seconds": 10.0 + profile_order,
            "case_results": cases,
            "case_order": case_ids,
            "hard_gate_passed": True,
        }
        profile_result["profile_result_sha256"] = gates.canonical_sha256(
            profile_result
        )
        candidate_results.append(profile_result)
    payload_without_digest = {
        "protocol_version": gates.PROTOCOL_VERSION,
        "kind": "inpaint-quality-screen-results",
        "phase": "model",
        "selected_dilation": 2,
        "frozen_contract_sha256": "f" * 64,
        "case_order": case_ids,
        "candidate_order": [profile.slug for profile in profiles],
        "results": candidate_results,
    }
    payload = {
        **payload_without_digest,
        "result_contract_sha256": gates.canonical_sha256(
            payload_without_digest
        ),
    }
    _write_json(result_root / gates.RESULT_FILENAME, payload)
    return result_root, [profile.slug for profile in profiles]


def _complete_review(path: Path, *, fail_candidate: str | None = None) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
        fields = list(rows[0])
    by_case: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_case.setdefault(row["case_key"], []).append(row)
    for case_rows in by_case.values():
        for rank, row in enumerate(case_rows, start=1):
            for field in gates.PROMOTION_REVIEW_FIELDS:
                row[field] = "pass"
            row["rank"] = str(rank)
            if fail_candidate and row["candidate"] == fail_candidate:
                row["residue"] = "fail"
                row["notes"] = "원문 잔상"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_exact_duplicate_canonicalization_keeps_distinct_fragment() -> None:
    page = {
        "blocks": [
            {
                "xyxy": [10, 10, 30, 40],
                "bubble_xyxy": [5, 5, 60, 70],
                "text_class": "text_bubble",
                "text": "same",
                "translation": "같음",
            },
            {
                "xyxy": [10, 10, 30, 40],
                "bubble_xyxy": [5, 5, 60, 70],
                "text_class": "text_bubble",
                "text": "same",
                "translation": "같음",
            },
            {
                "xyxy": [31, 10, 50, 40],
                "bubble_xyxy": [5, 5, 60, 70],
                "text_class": "text_bubble",
                "text": "different",
                "translation": "다름",
            },
        ]
    }

    records, summary = gates._snapshot_block_records(
        case_id="neutral-case",
        page=page,
        case={"annotations": []},
        image_shape=(100, 100, 3),
    )

    assert len(records) == 2
    assert summary["input_block_count"] == 3
    assert summary["canonical_block_count"] == 2
    assert summary["duplicate_alias_count"] == 1
    assert records[0]["duplicate_alias_count"] == 1
    assert records[1]["text"] == "different"


def test_text_free_requires_explicit_annotation_before_inpainting() -> None:
    assert gates._default_action(
        {
            "text_class": "text_bubble",
            "translation": "translated",
        }
    ) == "translate_inpaint"
    assert gates._default_action(
        {
            "text_class": "text_free",
            "translation": "translated",
        }
    ) == "review"


def test_exact_duplicate_annotations_must_agree() -> None:
    page = {
        "blocks": [
            {
                "xyxy": [10, 10, 30, 40],
                "bubble_xyxy": [5, 5, 60, 70],
                "text_class": "text_bubble",
                "text": "same",
                "translation": "same",
            },
            {
                "xyxy": [10, 10, 30, 40],
                "bubble_xyxy": [5, 5, 60, 70],
                "text_class": "text_bubble",
                "text": "same",
                "translation": "same",
            },
        ]
    }
    case = {
        "annotations": [
            {
                "block_index": 0,
                "semantic_role": "dialogue_bubble",
                "processing_action": "translate_inpaint",
            },
            {
                "block_index": 1,
                "semantic_role": "ui_or_sign",
                "processing_action": "preserve",
            },
        ]
    }

    with pytest.raises(gates.ProtocolError, match="annotations disagree"):
        gates._snapshot_block_records(
            case_id="neutral-case",
            page=page,
            case=case,
            image_shape=(100, 100, 3),
        )


def test_capture_freezes_source_snapshot_masks_and_rejects_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source.png"
        gates._write_image(
            source,
            np.full((32, 40, 3), 180, dtype=np.uint8),
        )
        snapshot = root / "page_snapshots.json"
        _write_json(
            snapshot,
            {
                "pages": [
                    {
                        "image_name": "source.png",
                        "blocks": [
                            {
                                "xyxy": [8, 8, 20, 24],
                                "bubble_xyxy": [5, 5, 25, 28],
                                "text_class": "text_bubble",
                                "text": "source",
                                "translation": "translation",
                            }
                        ],
                    }
                ]
            },
        )
        manifest = root / "cases.json"
        _write_json(
            manifest,
            {
                "protocol_version": gates.PROTOCOL_VERSION,
                "cases": [
                    {
                        "case_id": "neutral-case",
                        "source_image": str(source),
                        "source_sha256": gates.sha256_file(source),
                        "page_snapshot": str(snapshot),
                        "review_roi": [0, 0, 32, 32],
                        "annotations": [
                            {
                                "block_index": 0,
                                "semantic_role": "dialogue_bubble",
                                "processing_action": "translate_inpaint",
                            }
                        ],
                    }
                ],
            },
        )

        def fake_masks(image, blocks):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[10:14, 10:14] = 255
            empty = np.zeros_like(mask)
            full = np.full_like(mask, 255)
            return {
                "raw_mask": mask,
                "refined_mask": mask,
                "glyph_base_mask": mask,
                "bubble_protect_mask": empty,
                "structure_protect_mask": empty,
                "allowed_window_mask": full,
                "product_final_mask": mask,
                "refiner_backend": "test",
                "refiner_device": "cuda",
                "refiner_fallback_used": False,
                "product_mask_pixel_count": 16,
                "glyph_base_mask_pixel_count": 16,
            }

        monkeypatch.setattr(gates, "_capture_masks", fake_masks)
        frozen_root = root / "frozen"
        contract = gates.capture_cases(manifest, frozen_root)

        assert contract["case_count"] == 1
        assert gates.validate_frozen_contract(frozen_root)[
            "contract_sha256"
        ] == contract["contract_sha256"]

        raw_record = contract["cases"][0]["artifacts"]["raw_mask"]
        (frozen_root / raw_record["path"]).write_bytes(b"tampered")
        with pytest.raises(gates.ProtocolError, match="SHA-256 differs"):
            gates.validate_frozen_contract(frozen_root)


def test_mask_screen_uses_structure_protection_and_exact_window() -> None:
    glyph = np.zeros((16, 16), dtype=np.uint8)
    glyph[7:9, 7:9] = 255
    bubble = np.zeros_like(glyph)
    bubble[6, 6:10] = 255
    structure = np.zeros_like(glyph)
    structure[:, 8] = 255
    allowed = np.zeros_like(glyph)
    allowed[4:12, 4:12] = 255
    profile = gates.MASK_SCREEN_PROFILES[1]

    candidate = gates.build_candidate_mask(
        profile=profile,
        product_mask=np.full_like(glyph, 255),
        glyph_base_mask=glyph,
        bubble_protect_mask=bubble,
        structure_protect_mask=structure,
        allowed_window_mask=allowed,
    )

    assert np.count_nonzero(candidate[:, 8]) == 0
    assert np.count_nonzero(candidate[:4]) == 0
    assert np.count_nonzero(candidate[:, :4]) == 0
    assert np.count_nonzero(candidate) > 0


def test_model_roi_is_bounded_by_mask_not_full_page_review() -> None:
    mask = np.zeros((1000, 1000), dtype=np.uint8)
    mask[480:520, 480:520] = 255

    roi = gates._context_roi(
        mask,
        (1000, 1000, 3),
        requested_roi=[0, 0, 1000, 1000],
    )

    assert roi != [0, 0, 1000, 1000]
    assert roi[0] < 480 < roi[2]
    assert roi[1] < 480 < roi[3]


def test_model_screen_requires_locked_dilation_and_fp32() -> None:
    with pytest.raises(gates.ProtocolError, match="selected-dilation"):
        gates._profiles_for_phase(
            "model",
            selected_dilation=None,
            include_feasibility=False,
        )

    profiles = gates._profiles_for_phase(
        "model",
        selected_dilation=4,
        include_feasibility=True,
    )

    assert profiles[0].baseline
    assert not profiles[0].promotable
    assert all(
        profile.precision == "fp32"
        for profile in profiles
        if profile.promotable
    )
    assert any(profile.feasibility_only for profile in profiles)


def test_model_load_failure_is_isolated_as_profile_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_load(_profile):
        raise RuntimeError("synthetic load failure")

    monkeypatch.setattr(gates, "_instantiate_inpainter", fail_load)

    inpainter, runtime, failure = gates._load_profile_runtime(
        gates.MASK_SCREEN_PROFILES[0]
    )

    assert inpainter is None
    assert runtime["status"] == "load_failed"
    assert runtime["fp32_promotion_eligible"] is False
    assert "synthetic load failure" in failure


def test_cuda_oom_retries_once_with_smaller_roi() -> None:
    class OOMOnceInpainter:
        def __init__(self) -> None:
            self.shapes: list[tuple[int, ...]] = []

        def __call__(self, image, _mask, _config):
            self.shapes.append(tuple(image.shape))
            if len(self.shapes) == 1:
                raise RuntimeError("CUDA out of memory")
            return image.copy()

    image = np.full((100, 100, 3), 120, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[45:55, 45:55] = 255
    inpainter = OOMOnceInpainter()

    cleaned, _elapsed, diagnostics = gates._run_direct_roi(
        inpainter=inpainter,
        image=image,
        mask=mask,
        roi=[0, 0, 100, 100],
    )

    assert diagnostics["status"] == "completed_after_roi_retry"
    assert diagnostics["oom_retry_count"] == 1
    assert len(inpainter.shapes) == 2
    assert inpainter.shapes[1][0] < inpainter.shapes[0][0]
    assert np.array_equal(cleaned, image)


def test_result_artifact_tamper_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        result_root, _candidates = _fake_result_root(Path(temporary))
        payload = gates.validate_results(result_root)
        artifact = Path(
            payload["results"][0]["case_results"][0]["artifacts"][
                "cleaned"
            ]["path"]
        )
        (result_root / artifact).write_bytes(b"tampered")

        with pytest.raises(gates.ProtocolError, match="SHA-256 differs"):
            gates.validate_results(result_root)


def test_completed_result_cannot_hide_outside_mask_changes() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        result_root, _candidates = _fake_result_root(Path(temporary))
        payload = gates.read_json(result_root / gates.RESULT_FILENAME)
        case = payload["results"][1]["case_results"][0]
        case["changed_pixels"]["changed_outside_mask_pixel_count"] = 1
        case["result_sha256"] = gates.canonical_sha256(
            {key: value for key, value in case.items() if key != "result_sha256"}
        )
        profile = payload["results"][1]
        profile["profile_result_sha256"] = gates.canonical_sha256(
            {
                key: value
                for key, value in profile.items()
                if key != "profile_result_sha256"
            }
        )
        payload["result_contract_sha256"] = gates.canonical_sha256(
            {
                key: value
                for key, value in payload.items()
                if key != "result_contract_sha256"
            }
        )
        _write_json(result_root / gates.RESULT_FILENAME, payload)

        with pytest.raises(
            gates.ProtocolError,
            match="changed pixels outside",
        ):
            gates.validate_results(result_root)


def test_blind_bundle_hides_names_and_timings_until_complete_review() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        result_root, candidates = _fake_result_root(root)
        review_root = root / "review"
        state = gates.build_blind_review(
            result_root,
            review_root,
            mapping={"A": candidates[0], "B": candidates[1]},
        )
        public_text = "\n".join(
            (review_root / name).read_text(encoding="utf-8-sig")
            for name in (
                gates.REVIEW_FILENAME,
                gates.REVIEW_HTML_FILENAME,
                gates.STATE_FILENAME,
            )
        )

        assert candidates[0] not in public_text
        assert candidates[1] not in public_text
        assert "10.0" not in public_text
        assert state["review_row_count"] == 4
        with pytest.raises(gates.ReviewIncompleteError):
            gates.unblind_review(
                review_root,
                review_root / gates.REVIEW_FILENAME,
                confirmation="4-ROWS-REVIEWED",
            )


def test_preliminary_unblind_keeps_candidate_screen_only() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        result_root, candidates = _fake_result_root(root)
        review_root = root / "review"
        gates.build_blind_review(
            result_root,
            review_root,
            mapping={"A": candidates[0], "B": candidates[1]},
        )
        _complete_review(review_root / gates.REVIEW_FILENAME)

        summary = gates.unblind_review(
            review_root,
            review_root / gates.REVIEW_FILENAME,
            confirmation="4-ROWS-REVIEWED",
        )

        assert summary["screen_eligible_candidates"] == [candidates[1]]
        assert summary["promotion_eligible_candidates"] == []
        assert (review_root / gates.UNBLIND_FILENAME).is_file()


def test_duplicate_or_incomplete_ranks_block_unblind() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        result_root, candidates = _fake_result_root(root)
        review_root = root / "review"
        gates.build_blind_review(
            result_root,
            review_root,
            mapping={"A": candidates[0], "B": candidates[1]},
        )
        review_path = review_root / gates.REVIEW_FILENAME
        _complete_review(review_path)
        with review_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            rows = list(csv.DictReader(stream))
            fields = list(rows[0])
        rows[1]["rank"] = rows[0]["rank"]
        with review_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        with pytest.raises(
            gates.ReviewIncompleteError,
            match="ranks must be unique",
        ):
            gates.validate_review(review_root, review_path)


def test_case_number_is_part_of_blind_review_identity() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        result_root, candidates = _fake_result_root(root)
        review_root = root / "review"
        gates.build_blind_review(
            result_root,
            review_root,
            mapping={"A": candidates[0], "B": candidates[1]},
        )
        review_path = review_root / gates.REVIEW_FILENAME
        _complete_review(review_path)
        with review_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            rows = list(csv.DictReader(stream))
            fields = list(rows[0])
        rows[0]["case_number"] = "999"
        with review_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        with pytest.raises(gates.ProtocolError, match="identity differs"):
            gates.validate_review(review_root, review_path)


def test_blind_asset_or_private_key_tamper_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        result_root, candidates = _fake_result_root(root)
        review_root = root / "review"
        gates.build_blind_review(
            result_root,
            review_root,
            mapping={"A": candidates[0], "B": candidates[1]},
        )
        _complete_review(review_root / gates.REVIEW_FILENAME)
        payload = gates.read_json(
            review_root
            / gates.PRIVATE_DIRNAME
            / gates.PRIVATE_PAYLOAD_FILENAME
        )
        artifact = payload["cases"][0]["candidates"][0]["artifacts"][
            "cleaned"
        ]
        (review_root / artifact["path"]).write_bytes(b"tampered")
        with pytest.raises(gates.ProtocolError, match="SHA-256 differs"):
            gates.validate_review(
                review_root,
                review_root / gates.REVIEW_FILENAME,
            )

        review_root_2 = root / "review-key"
        gates.build_blind_review(
            result_root,
            review_root_2,
            mapping={"A": candidates[0], "B": candidates[1]},
        )
        _complete_review(review_root_2 / gates.REVIEW_FILENAME)
        key_path = (
            review_root_2
            / gates.PRIVATE_DIRNAME
            / gates.PRIVATE_KEY_FILENAME
        )
        key = gates.read_json(key_path)
        key["label_to_candidate"]["A"] = candidates[1]
        _write_json(key_path, key)
        with pytest.raises(gates.ProtocolError, match="key digest differs"):
            gates.unblind_review(
                review_root_2,
                review_root_2 / gates.REVIEW_FILENAME,
                confirmation="4-ROWS-REVIEWED",
            )


def test_failed_hard_gate_cannot_enter_blind_review() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        result_root, candidates = _fake_result_root(root)
        payload = gates.read_json(result_root / gates.RESULT_FILENAME)
        payload["results"][1]["hard_gate_passed"] = False
        gates._refresh_result_digests(payload)
        _write_json(result_root / gates.RESULT_FILENAME, payload)

        with pytest.raises(gates.ProtocolError, match="hard gates"):
            gates.build_blind_review(
                result_root,
                root / "review",
                mapping={"A": candidates[0], "B": candidates[1]},
            )


def test_final_review_can_require_render_artifacts() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        result_root, candidates = _fake_result_root(root)

        with pytest.raises(gates.ProtocolError, match="requires render"):
            gates.build_blind_review(
                result_root,
                root / "review",
                candidate_slugs=candidates,
                require_render=True,
                mapping={"A": candidates[0], "B": candidates[1]},
            )


def test_attach_renders_requires_complete_ordered_matrix() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        result_root, candidates = _fake_result_root(root)
        results = gates.validate_results(result_root)
        attachments = []
        for candidate_index, candidate in enumerate(candidates):
            for case_index, case_id in enumerate(results["case_order"]):
                render = (
                    root
                    / "external-renders"
                    / candidate
                    / f"{case_id}.png"
                )
                gates._write_image(
                    render,
                    np.full(
                        (8, 8, 3),
                        candidate_index * 50 + case_index,
                        dtype=np.uint8,
                    ),
                )
                attachments.append(
                    {
                        "candidate_slug": candidate,
                        "case_id": case_id,
                        "render_image": str(render),
                        "render_sha256": gates.sha256_file(render),
                    }
                )
        manifest = root / "render-manifest.json"
        _write_json(
            manifest,
            {
                "protocol_version": gates.PROTOCOL_VERSION,
                "kind": gates.RENDER_MANIFEST_KIND,
                "result_contract_sha256": results[
                    "result_contract_sha256"
                ],
                "renders": attachments,
            },
        )

        attached_root = root / "results-with-renders"
        attached = gates.attach_renders(
            result_root,
            manifest,
            attached_root,
        )

        assert all(
            "render" in case["artifacts"]
            for result in attached["results"]
            for case in result["case_results"]
        )
        gates.build_blind_review(
            attached_root,
            root / "review-with-renders",
            candidate_slugs=candidates,
            require_render=True,
            mapping={"A": candidates[0], "B": candidates[1]},
        )
        final_review_root = root / "review-with-renders"
        final_review_path = final_review_root / gates.REVIEW_FILENAME
        _complete_review(final_review_path)
        summary = gates.unblind_review(
            final_review_root,
            final_review_path,
            confirmation="4-ROWS-REVIEWED",
        )
        assert summary["promotion_eligible_candidates"] == [candidates[1]]


def test_required_render_review_rejects_na() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        result_root, candidates = _fake_result_root(root)
        results = gates.validate_results(result_root)
        attachments = []
        for candidate in candidates:
            for case_id in results["case_order"]:
                render = root / "renders" / candidate / f"{case_id}.png"
                gates._write_image(
                    render,
                    np.zeros((8, 8, 3), dtype=np.uint8),
                )
                attachments.append(
                    {
                        "candidate_slug": candidate,
                        "case_id": case_id,
                        "render_image": str(render),
                        "render_sha256": gates.sha256_file(render),
                    }
                )
        manifest = root / "render-manifest.json"
        _write_json(
            manifest,
            {
                "protocol_version": gates.PROTOCOL_VERSION,
                "kind": gates.RENDER_MANIFEST_KIND,
                "result_contract_sha256": results[
                    "result_contract_sha256"
                ],
                "renders": attachments,
            },
        )
        attached_root = root / "attached"
        gates.attach_renders(result_root, manifest, attached_root)
        review_root = root / "review"
        gates.build_blind_review(
            attached_root,
            review_root,
            require_render=True,
            mapping={"A": candidates[0], "B": candidates[1]},
        )
        review_path = review_root / gates.REVIEW_FILENAME
        _complete_review(review_path)
        with review_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            rows = list(csv.DictReader(stream))
            fields = list(rows[0])
        rows[0]["render"] = "na"
        with review_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        with pytest.raises(gates.ReviewIncompleteError, match="render"):
            gates.validate_review(review_root, review_path)
