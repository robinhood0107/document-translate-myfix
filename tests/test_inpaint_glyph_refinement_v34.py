from __future__ import annotations

from dataclasses import fields, replace
import inspect

import cv2
import numpy as np
import pytest

from benchmarking.inpaint_detector_bakeoff.contracts import mask_sha256
from benchmarking.inpaint_detector_bakeoff.glyph_refinement import (
    SeedlessRoiEvidence,
    extract_roi_local_glyph_masks,
    extract_seedless_roi_glyph_masks,
)
from benchmarking.inpaint_detector_bakeoff.semantic import PRESERVE, REVIEW, TRANSLATE


SHAPE = (72, 96)


def _mask(*boxes: tuple[int, int, int, int]) -> np.ndarray:
    result = np.zeros(SHAPE, dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        result[y1:y2, x1:x2] = 255
    return result


def _owner() -> np.ndarray:
    return _mask((8, 8, 88, 64))


def _run(
    image: np.ndarray,
    seed: np.ndarray,
    *,
    ownership: np.ndarray | None = None,
    protect: np.ndarray | None = None,
    effect_support: np.ndarray | None = None,
):
    zero = np.zeros(SHAPE, dtype=np.uint8)
    return extract_roi_local_glyph_masks(
        image,
        detector_seed=seed,
        ocr_ownership=ownership if ownership is not None else _owner(),
        hard_protect=protect if protect is not None else zero,
        detector_provider="synthetic-detector",
        ownership_provider="synthetic-ocr",
        effect_support=effect_support,
        effect_support_provider=(
            "synthetic-source-effect" if effect_support is not None else ""
        ),
    )


def _cjk_like_mask(*, offset_x: int = 0) -> np.ndarray:
    target = np.zeros(SHAPE, dtype=np.uint8)
    cv2.rectangle(target, (20 + offset_x, 18), (42 + offset_x, 42), 255, 5)
    cv2.line(target, (22 + offset_x, 30), (40 + offset_x, 30), 255, 5)
    cv2.line(target, (52 + offset_x, 18), (52 + offset_x, 43), 255, 5)
    cv2.line(target, (46 + offset_x, 25), (59 + offset_x, 25), 255, 5)
    cv2.line(target, (46 + offset_x, 38), (59 + offset_x, 38), 255, 5)
    return target


def _eroded_seed(target: np.ndarray) -> np.ndarray:
    return cv2.erode(
        target,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )


def _coverage(mask: np.ndarray, target: np.ndarray) -> float:
    denominator = int(np.count_nonzero(target))
    return float(np.count_nonzero((mask > 0) & (target > 0))) / denominator


def test_bright_cjk_like_glyphs_are_recovered_from_owned_detector_seeds() -> None:
    target = _cjk_like_mask()
    image = np.full(SHAPE, 68, dtype=np.uint8)
    image[_owner() > 0] = 96
    image[target > 0] = 224

    result = _run(image, _eroded_seed(target))

    assert _coverage(result.refined_mask, target) >= 0.90
    assert np.count_nonzero(result.refined_mask[_owner() == 0]) == 0
    assert any(row.polarity in {"bright", "mixed"} for row in result.component_records)


def test_dark_japanese_like_glyphs_use_the_opposite_local_polarity() -> None:
    target = _cjk_like_mask(offset_x=2)
    image = np.full(SHAPE, 238, dtype=np.uint8)
    image[_owner() > 0] = 216
    image[target > 0] = 28

    result = _run(image, _eroded_seed(target))

    assert _coverage(result.refined_mask, target) >= 0.90
    assert any(row.polarity in {"dark", "mixed"} for row in result.component_records)


def test_opposite_polarity_outline_is_recorded_as_connected_effect() -> None:
    image = np.full(SHAPE, 108, dtype=np.uint8)
    outline = np.zeros(SHAPE, dtype=np.uint8)
    core = np.zeros(SHAPE, dtype=np.uint8)
    cv2.line(outline, (20, 35), (72, 35), 255, 9, cv2.LINE_8)
    cv2.line(core, (20, 35), (72, 35), 255, 3, cv2.LINE_8)
    image[outline > 0] = 232
    image[core > 0] = 24
    outline_only = (outline > 0) & (core == 0)

    result = _run(image, core, effect_support=outline)

    assert np.count_nonzero(result.glyph_core & core) > 0
    assert np.count_nonzero((result.glyph_effect > 0) & outline_only) > 0
    assert np.count_nonzero(result.glyph_core & result.glyph_effect) == 0


def test_alpha_like_low_contrast_white_glyph_survives_translucent_carrier() -> None:
    image = np.tile(np.linspace(55, 105, SHAPE[1], dtype=np.uint8), (SHAPE[0], 1))
    carrier = _mask((12, 12, 84, 60))
    target = _cjk_like_mask()
    image[carrier > 0] = np.clip(
        (image[carrier > 0].astype(np.uint16) + 128) // 2,
        0,
        255,
    ).astype(np.uint8)
    image[target > 0] = np.minimum(
        image[target > 0].astype(np.uint16) + 34,
        255,
    ).astype(np.uint8)

    result = _run(image, _eroded_seed(target), ownership=carrier)

    assert _coverage(result.refined_mask, target) >= 0.85
    assert result.provenance["status"] == "completed"


def test_remote_halftone_is_not_pulled_into_detector_seeded_glyph() -> None:
    image = np.full(SHAPE, 192, dtype=np.uint8)
    texture = np.zeros(SHAPE, dtype=np.uint8)
    for y in range(12, 64, 7):
        for x in range(10, 88, 7):
            cv2.circle(texture, (x, y), 1, 255, -1)
    image[texture > 0] = 78
    target = _mask((38, 25, 58, 31), (45, 18, 51, 43))
    image[target > 0] = 246
    seed = _eroded_seed(target)
    remote_texture = (texture > 0) & (_dilate_for_test(target, 7) == 0)

    result = _run(image, seed)

    assert np.count_nonzero((result.refined_mask > 0) & remote_texture) == 0


def test_remote_hatching_is_not_selected_by_same_polarity_text_seed() -> None:
    image = np.full(SHAPE, 226, dtype=np.uint8)
    hatching = np.zeros(SHAPE, dtype=np.uint8)
    for shift in range(-30, 80, 8):
        cv2.line(hatching, (0, shift), (36, shift + 36), 255, 1, cv2.LINE_8)
    image[hatching > 0] = 54
    target = _mask((57, 22, 63, 48), (50, 30, 72, 36))
    image[target > 0] = 28
    remote_hatching = (hatching > 0) & (_dilate_for_test(target, 7) == 0)

    result = _run(image, _eroded_seed(target))

    assert np.count_nonzero((result.refined_mask > 0) & remote_hatching) == 0


def _dilate_for_test(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        ((radius * 2) + 1, (radius * 2) + 1),
    )
    return cv2.dilate(mask, kernel, iterations=1)


def test_one_exact_protect_pixel_rejects_the_whole_connected_component() -> None:
    image = np.full(SHAPE, 210, dtype=np.uint8)
    target = _mask((24, 26, 68, 38))
    image[target > 0] = 34
    protect = _mask((46, 31, 47, 32))

    result = _run(image, _eroded_seed(target), protect=protect)

    assert np.count_nonzero(result.refined_mask) == 0
    assert np.count_nonzero(result.rejected_mask) > 0
    assert any(
        row.reason == "exact_protect_overlap"
        and row.protect_overlap_pixel_count == 1
        and row.protect_adjacency_pixel_count > 0
        for row in result.component_records
    )


def test_disconnected_halo_near_a_seeded_core_is_not_admitted() -> None:
    image = np.full(SHAPE, 100, dtype=np.uint8)
    core = _mask((24, 25, 34, 37))
    disconnected = _mask((36, 29, 38, 33))
    image[core > 0] = 222
    image[disconnected > 0] = 218

    result = _run(image, _eroded_seed(core))

    assert np.count_nonzero(result.refined_mask & core) > 0
    assert np.count_nonzero(result.refined_mask & disconnected) == 0


def test_no_detector_seed_means_no_expansion() -> None:
    image = np.full(SHAPE, 120, dtype=np.uint8)
    image[_cjk_like_mask() > 0] = 230

    result = _run(image, np.zeros(SHAPE, dtype=np.uint8))

    assert np.count_nonzero(result.refined_mask) == 0
    assert result.component_records == ()
    assert result.provenance["status"] == "no_owned_detector_seed"
    assert result.provenance["no_seed_no_expansion"] is True


def test_ocr_ownership_is_a_gate_and_never_creates_edit_pixels() -> None:
    image = np.full(SHAPE, 104, dtype=np.uint8)
    left = _mask((18, 24, 32, 38))
    right = _mask((62, 24, 76, 38))
    image[(left > 0) | (right > 0)] = 232
    ownership = _mask((10, 12, 46, 58))
    detector = cv2.bitwise_or(_eroded_seed(left), _eroded_seed(right))

    result = _run(image, detector, ownership=ownership)

    assert np.count_nonzero(result.refined_mask & left) > 0
    assert np.count_nonzero(result.refined_mask & right) == 0
    assert result.provenance["discarded_detector_seed_pixel_count"] > 0
    assert np.count_nonzero(result.refined_mask[detector == 0]) > 0
    assert np.count_nonzero(result.refined_mask[ownership == 0]) == 0


def test_provenance_records_providers_hashes_and_subset_invariants() -> None:
    image = np.full((SHAPE[0], SHAPE[1], 4), 118, dtype=np.uint8)
    image[..., 3] = 160
    target = _mask((30, 25, 58, 39))
    image[target > 0, :3] = 225

    result = _run(image, _eroded_seed(target))

    assert result.provenance["detector_provider"] == "synthetic-detector"
    assert result.provenance["ownership_provider"] == "synthetic-ocr"
    assert result.provenance["refined_mask_sha256"] == mask_sha256(
        result.refined_mask
    )
    assert result.provenance["output_subset_of_ownership"] is True
    assert result.provenance["effect_disjoint_from_core"] is True
    assert result.provenance["no_seed_no_expansion"] is True


def test_invalid_contracts_fail_closed() -> None:
    image = np.full(SHAPE, 120, dtype=np.float32)
    zero = np.zeros(SHAPE, dtype=np.uint8)
    with pytest.raises(ValueError, match="uint8"):
        extract_roi_local_glyph_masks(
            image,
            detector_seed=zero,
            ocr_ownership=zero,
            hard_protect=zero,
            detector_provider="detector",
            ownership_provider="ocr",
        )
    with pytest.raises(ValueError, match="provider"):
        extract_roi_local_glyph_masks(
            image.astype(np.uint8),
            detector_seed=zero,
            ocr_ownership=zero,
            hard_protect=zero,
            detector_provider="",
            ownership_provider="ocr",
        )


def test_one_pixel_protect_adjacency_rejects_the_whole_component() -> None:
    image = np.full(SHAPE, 210, dtype=np.uint8)
    target = _mask((24, 26, 68, 38))
    image[target > 0] = 34
    protect = _mask((46, 38, 47, 39))

    result = _run(image, _eroded_seed(target), protect=protect)

    assert np.count_nonzero(result.refined_mask) == 0
    assert any(
        row.reason == "exact_protect_adjacency"
        and row.protect_overlap_pixel_count == 0
        and row.protect_adjacency_pixel_count > 0
        for row in result.component_records
    )


def test_effect_requires_explicit_source_support_and_stays_inside_it() -> None:
    image = np.full(SHAPE, 108, dtype=np.uint8)
    core = _mask((24, 28, 60, 36))
    nearby_effect = _mask((22, 26, 62, 38))
    image[nearby_effect > 0] = 122
    image[core > 0] = 230
    seed = _eroded_seed(core)

    unsupported = _run(image, seed)
    supported = _run(image, seed, effect_support=nearby_effect)

    assert np.count_nonzero(unsupported.glyph_effect) == 0
    assert np.count_nonzero(supported.glyph_effect) > 0
    assert np.count_nonzero(
        (supported.glyph_effect > 0) & (nearby_effect == 0)
    ) == 0
    assert supported.provenance["effect_subset_of_support"] is True


@pytest.mark.parametrize("texture_kind", ["halftone", "hatching"])
def test_touching_low_contrast_texture_does_not_leak_without_effect_support(
    texture_kind: str,
) -> None:
    image = np.full(SHAPE, 104, dtype=np.uint8)
    target = _mask((35, 26, 61, 38))
    image[target > 0] = 230
    texture = np.zeros(SHAPE, dtype=np.uint8)
    if texture_kind == "halftone":
        for y in range(22, 44, 4):
            cv2.circle(texture, (63, y), 1, 255, -1)
    else:
        for shift in range(18, 45, 4):
            cv2.line(texture, (61, shift), (72, shift + 5), 255, 1)
    image[texture > 0] = 114

    result = _run(image, _eroded_seed(target))

    assert np.count_nonzero(result.glyph_effect) == 0
    assert np.count_nonzero((result.refined_mask > 0) & (texture > 0)) == 0


def test_owner_diagnostics_account_for_silent_and_unaccounted_seeds() -> None:
    image = np.full(SHAPE, 120, dtype=np.uint8)
    ownership = _mask((8, 8, 42, 64), (54, 8, 88, 64))
    seed = _mask((18, 24, 24, 32))

    result = _run(image, seed, ownership=ownership)

    assert len(result.owner_records) == 2
    assert result.owner_records[0].reason == "seed_contrast_unavailable"
    assert result.owner_records[1].reason == "no_detector_seed"
    assert result.provenance["silent_owner_count"] == 2
    assert result.provenance["unaccounted_seed_pixel_count"] == np.count_nonzero(
        seed
    )


def test_all_returned_masks_are_readonly_and_hashed() -> None:
    image = np.full(SHAPE, 220, dtype=np.uint8)
    target = _mask((28, 24, 60, 40))
    image[target > 0] = 30
    result = _run(image, _eroded_seed(target))
    mask_fields = (
        "owned_detector_seed",
        "effect_support",
        "candidate_mask",
        "glyph_core",
        "glyph_effect",
        "refined_mask",
        "rejected_mask",
        "hard_protect",
    )
    hash_keys = (
        "owned_detector_seed_sha256",
        "effect_support_sha256",
        "candidate_mask_sha256",
        "glyph_core_sha256",
        "glyph_effect_sha256",
        "refined_mask_sha256",
        "rejected_mask_sha256",
        "hard_protect_sha256",
    )

    for field_name, hash_key in zip(mask_fields, hash_keys, strict=True):
        mask = getattr(result, field_name)
        assert mask.flags.writeable is False
        assert result.provenance[hash_key] == mask_sha256(mask)
        with pytest.raises(ValueError):
            mask[0, 0] = 255


def test_component_bbox_scanning_remains_linear_on_a_4k_page() -> None:
    shape = (2160, 3840)
    image = np.full(shape, 210, dtype=np.uint8)
    owner = np.zeros(shape, dtype=np.uint8)
    owner[400:1760, 600:3240] = 255
    target = np.zeros(shape, dtype=np.uint8)
    for y in range(600, 1500, 160):
        for x in range(900, 3000, 220):
            target[y : y + 12, x : x + 42] = 255
    image[target > 0] = 30
    seed = cv2.erode(target, np.ones((3, 3), dtype=np.uint8))

    result = extract_roi_local_glyph_masks(
        image,
        detector_seed=seed,
        ocr_ownership=owner,
        hard_protect=np.zeros(shape, dtype=np.uint8),
        detector_provider="synthetic-detector",
        ownership_provider="synthetic-ocr",
    )

    assert result.provenance["component_bbox_scan_ratio"] < 4.0
    assert np.count_nonzero(result.refined_mask) > 0


def _seedless_evidence(
    image: np.ndarray,
    ownership: np.ndarray,
    protect: np.ndarray,
    **overrides: object,
) -> SeedlessRoiEvidence:
    values: dict[str, object] = {
        "owner_region_id": "neutral-owner",
        "owner_count": 1,
        "authoritative_ocr": True,
        "ocr_text": "白字",
        "ocr_script": "Han",
        "ocr_confidence": 0.94,
        "ocr_provider": "neutral-ocr-provider",
        "processing_action": TRANSLATE,
        "route_class": "clean_translucent",
    }
    values.update(overrides)
    return SeedlessRoiEvidence.seal(
        image,
        ocr_ownership=ownership,
        hard_protect=protect,
        **values,
    )


def _seedless_run(
    image: np.ndarray,
    ownership: np.ndarray,
    *,
    protect: np.ndarray | None = None,
    **evidence_overrides: object,
):
    exact_protect = (
        np.zeros(SHAPE, dtype=np.uint8) if protect is None else protect
    )
    evidence = _seedless_evidence(
        image,
        ownership,
        exact_protect,
        **evidence_overrides,
    )
    return extract_seedless_roi_glyph_masks(
        image,
        ocr_ownership=ownership,
        hard_protect=exact_protect,
        evidence=evidence,
    )


def _alpha_white_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.tile(
        np.linspace(54, 104, SHAPE[1], dtype=np.uint8),
        (SHAPE[0], 1),
    )
    owner = _mask((12, 12, 84, 60))
    target = _cjk_like_mask()
    image[owner > 0] = (
        (image[owner > 0].astype(np.uint16) + 130) // 2
    ).astype(np.uint8)
    image[target > 0] = np.minimum(
        image[target > 0].astype(np.uint16) + 34,
        255,
    ).astype(np.uint8)
    return image, owner, target


def _seedless_texture_fixture(texture_kind: str) -> tuple[np.ndarray, np.ndarray]:
    owner = _mask((12, 12, 84, 60))
    if texture_kind == "translucent_halftone":
        image = np.tile(
            np.linspace(76, 126, SHAPE[1], dtype=np.uint8),
            (SHAPE[0], 1),
        )
        image[owner > 0] = (
            (image[owner > 0].astype(np.uint16) + 174) // 2
        ).astype(np.uint8)
        for y in range(16, 58, 7):
            for x in range(16, 82, 7):
                cv2.circle(image, (x, y), 1, 166, -1)
    elif texture_kind == "regular_dots":
        image = np.full(SHAPE, 112, dtype=np.uint8)
        image[owner > 0] = 148
        for y in range(16, 58, 6):
            for x in range(16, 82, 6):
                cv2.circle(image, (x, y), 1, 214, -1)
    elif texture_kind == "hatching":
        image = np.full(SHAPE, 118, dtype=np.uint8)
        image[owner > 0] = 176
        for shift in range(-54, 94, 6):
            cv2.line(
                image,
                (12, shift),
                (84, shift + 72),
                92,
                1,
                cv2.LINE_8,
            )
        image[owner == 0] = 118
    elif texture_kind == "textured_carrier":
        image = np.full(SHAPE, 104, dtype=np.uint8)
        image[owner > 0] = 150
        for row, y in enumerate(range(15, 59, 4)):
            for column, x in enumerate(range(15, 83, 4)):
                value = 104 if (row * 3 + column * 5) % 4 else 202
                image[y : y + 2, x : x + 2] = value
        image[owner == 0] = 104
    else:  # pragma: no cover - fixture contract
        raise AssertionError(texture_kind)
    return image, owner


def test_seedless_authoritative_ocr_recovers_bright_alpha_cjk_pixels() -> None:
    image, owner, target = _alpha_white_fixture()

    result = _seedless_run(image, owner)

    assert result.provenance["status"] == "completed"
    assert np.count_nonzero(result.refined_mask) > 0
    assert _coverage(result.refined_mask, target) >= 0.25
    assert np.count_nonzero(result.refined_mask[owner == 0]) == 0
    assert result.provenance["lab_multiscale_cue_pixel_count"] > 0
    assert result.provenance["stroke_outline_cue_pixel_count"] > 0
    assert result.provenance["source_roi_microtexture_vetoed"] is False
    assert result.provenance["source_roi_microtexture_veto_reason"] == "none"


@pytest.mark.parametrize(
    "texture_kind",
    [
        "translucent_halftone",
        "regular_dots",
        "hatching",
        "textured_carrier",
    ],
)
def test_seedless_roi_microtexture_and_periodicity_fail_closed(
    texture_kind: str,
) -> None:
    image, owner = _seedless_texture_fixture(texture_kind)

    result = _seedless_run(image, owner)

    assert np.count_nonzero(result.candidate_mask) > 0
    assert np.count_nonzero(result.refined_mask) == 0
    assert result.provenance["status"] == "rejected"
    assert result.provenance["reason"] == "source_roi_microtexture_veto"
    assert result.provenance["source_roi_microtexture_vetoed"] is True
    assert result.provenance["source_roi_microtexture_veto_reason"] in {
        "periodic_source_cue_saturation",
        "oriented_source_cue_saturation",
        "textured_carrier_source_cue_saturation",
        "regular_micro_components",
        "periodic_hatching",
        "dense_microtexture_components",
        "textured_carrier_high_frequency",
    }
    assert result.provenance["source_roi_candidate_component_count"] > 0
    assert result.provenance["source_roi_periodicity_peak_correlation"] >= 0.0


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({"authoritative_ocr": False}, "authoritative_ocr_missing"),
        ({"ocr_text": ""}, "ocr_text_missing"),
        ({"ocr_script": ""}, "ocr_script_missing"),
        ({"ocr_provider": ""}, "ocr_provider_missing"),
        ({"ocr_confidence": None}, "ocr_confidence_missing_or_invalid"),
    ],
)
def test_seedless_missing_ocr_evidence_remains_information_limited(
    overrides: dict[str, object],
    expected_reason: str,
) -> None:
    image, owner, _target = _alpha_white_fixture()

    result = _seedless_run(image, owner, **overrides)

    assert np.count_nonzero(result.refined_mask) == 0
    assert result.provenance["status"] == "information_limited"
    assert result.provenance["reason"] == expected_reason


@pytest.mark.parametrize(
    "route",
    ["flat", "texture", "line_art", "ambiguous"],
)
def test_seedless_unsafe_routes_fail_closed(route: str) -> None:
    image, owner, _target = _alpha_white_fixture()

    result = _seedless_run(image, owner, route_class=route)

    assert np.count_nonzero(result.refined_mask) == 0
    assert result.provenance["reason"] == "route_not_seedless_safe"


@pytest.mark.parametrize("action", [PRESERVE, REVIEW])
def test_seedless_non_translate_actions_fail_closed(action: str) -> None:
    image, owner, _target = _alpha_white_fixture()

    result = _seedless_run(image, owner, processing_action=action)

    assert np.count_nonzero(result.refined_mask) == 0
    assert result.provenance["reason"] == "processing_action_not_translate"


def test_seedless_owner_conflict_fails_closed() -> None:
    image, _owner, _target = _alpha_white_fixture()
    conflict = _mask((8, 8, 42, 64), (54, 8, 88, 64))

    result = _seedless_run(image, conflict, owner_count=2)

    assert np.count_nonzero(result.refined_mask) == 0
    assert result.provenance["reason"] == "ownership_conflict"


def test_seedless_runtime_owner_conflict_fails_closed() -> None:
    image, _owner, _target = _alpha_white_fixture()
    conflict = _mask((8, 8, 42, 64), (54, 8, 88, 64))
    evidence = _seedless_evidence(
        image,
        conflict,
        np.zeros(SHAPE, dtype=np.uint8),
        owner_count=1,
    )

    result = extract_seedless_roi_glyph_masks(
        image,
        ocr_ownership=conflict,
        hard_protect=np.zeros(SHAPE, dtype=np.uint8),
        evidence=evidence,
    )

    assert np.count_nonzero(result.refined_mask) == 0
    assert result.provenance["reason"] == "runtime_ownership_conflict"


def test_seedless_protect_adjacency_empties_the_entire_roi() -> None:
    image, owner, _target = _alpha_white_fixture()
    control = _seedless_run(image, owner)
    y, x = np.argwhere(control.refined_mask > 0)[0]
    protect = np.zeros(SHAPE, dtype=np.uint8)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        ny, nx = int(y + dy), int(x + dx)
        if (
            0 <= ny < SHAPE[0]
            and 0 <= nx < SHAPE[1]
            and control.refined_mask[ny, nx] == 0
        ):
            protect[ny, nx] = 255
            break
    assert np.count_nonzero(protect) == 1

    result = _seedless_run(image, owner, protect=protect)

    assert np.count_nonzero(result.refined_mask) == 0
    assert result.provenance["reason"] == "exact_protect_contact"
    assert any(
        row.protect_adjacency_pixel_count > 0
        for row in result.component_records
    )


def test_seedless_flat_source_has_no_bbox_fill() -> None:
    image = np.full(SHAPE, 128, dtype=np.uint8)
    owner = _owner()

    result = _seedless_run(image, owner)

    assert np.count_nonzero(result.candidate_mask) == 0
    assert np.count_nonzero(result.refined_mask) == 0
    assert result.provenance["status"] == "information_limited"
    assert result.provenance["reason"] == "source_local_cue_absent"


def test_seedless_source_or_seal_tamper_fails_closed() -> None:
    image, owner, _target = _alpha_white_fixture()
    protect = np.zeros(SHAPE, dtype=np.uint8)
    evidence = _seedless_evidence(image, owner, protect)
    changed_source = image.copy()
    changed_source[0, 0] ^= 1

    source_mismatch = extract_seedless_roi_glyph_masks(
        changed_source,
        ocr_ownership=owner,
        hard_protect=protect,
        evidence=evidence,
    )
    seal_mismatch = extract_seedless_roi_glyph_masks(
        image,
        ocr_ownership=owner,
        hard_protect=protect,
        evidence=replace(evidence, ocr_text="tampered"),
    )

    assert source_mismatch.provenance["reason"] == "source_sha256_mismatch"
    assert seal_mismatch.provenance["reason"] == "evidence_seal_mismatch"
    assert np.count_nonzero(source_mismatch.refined_mask) == 0
    assert np.count_nonzero(seal_mismatch.refined_mask) == 0


def test_seedless_api_cannot_receive_evaluation_masks() -> None:
    parameter_names = set(
        inspect.signature(extract_seedless_roi_glyph_masks).parameters
    )
    evidence_names = {field.name for field in fields(SeedlessRoiEvidence)}
    forbidden = {"target", "annotation", "evaluation", "expected_edit"}

    assert parameter_names.isdisjoint(forbidden)
    assert evidence_names.isdisjoint(forbidden)
