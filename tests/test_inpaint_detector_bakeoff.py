from __future__ import annotations

import hashlib
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import benchmarking.inpaint_detector_bakeoff.ballons_e2e as ballons_e2e
from benchmarking.inpaint_detector_bakeoff.ballons_e2e import (
    BallonsEndToEndReference,
)

from benchmarking.inpaint_detector_bakeoff.ballons_ctbd import (
    CTBDSettings,
    _build_content_mask,
    _boxes_from_outputs,
    preprocess_ctbd,
)
from benchmarking.inpaint_detector_bakeoff.ballons_ysg import (
    YSGSettings,
    mask_and_boxes_from_result,
)
from benchmarking.inpaint_detector_bakeoff.contracts import (
    CandidateMaskResult,
    DetectorBox,
    Stage1Page,
    assert_disjoint_masks,
)
from benchmarking.inpaint_detector_bakeoff.stage1 import (
    PageMasks,
    positive_edit_from_claim,
    score_page,
)
from benchmarking.inpaint_detector_bakeoff.stage2 import (
    changed_mask,
    composite_positive_result,
    residue_score,
    score_stage2_page,
)
from benchmarking.inpaint_detector_bakeoff.sickzil import (
    class_map_to_mask,
    modulo_padded,
    preprocess_sickzil,
)
from benchmarking.inpaint_detector_bakeoff.ownership import build_existing_ownership_mask
from benchmarking.inpaint_detector_bakeoff.provenance_fusion import (
    build_provenance_fusion,
)
from benchmarking.inpaint_detector_bakeoff.fixed_ctd_onnx import (
    _letterboxed_rgb_to_nchw,
    _rgb_batch_to_nchw,
    _require_primary_provider,
)
from scripts.check_inpaint_positive_mask_parity import _compare_kind


def _tensor_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def test_ballons_e2e_routing_retains_native_whole_bubble_flat_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = BallonsEndToEndReference.__new__(BallonsEndToEndReference)
    reference.inpainter = SimpleNamespace()
    image = np.full((16, 16, 3), 150, dtype=np.uint8)
    image[6, 6] = 220
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[7:9, 7:9] = 255

    def fake_extract(_image: np.ndarray, local_mask: np.ndarray):
        balloon = np.full(local_mask.shape, 255, dtype=np.uint8)
        non_text = np.zeros(local_mask.shape, dtype=np.uint8)
        non_text[0, 0] = 255
        return balloon, non_text

    monkeypatch.setattr(ballons_e2e, "extract_ballon_mask", fake_extract)
    result = reference._inpaint_with_ballons_routing(
        image,
        mask,
        (SimpleNamespace(xyxy=[5, 5, 11, 11]),),
    )

    assert np.array_equal(result[7:9, 7:9], np.full((2, 2, 3), 150, np.uint8))
    assert result[6, 6].tolist() == [150, 150, 150]


def test_ballons_ctbd_preprocess_golden_is_bgr_to_rgb_resize() -> None:
    y, x = np.mgrid[0:37, 0:53]
    image = np.stack(
        (
            (x * 3 + y * 5) % 256,
            (x * 7 + 11) % 256,
            (y * 13 + 19) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)

    tensor, original_size = preprocess_ctbd(image)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert original_size.tolist() == [[53, 37]]
    assert _tensor_sha256(tensor, original_size) == (
        "74ffd39889da3109108eab1d290965f9fb5365bc0586aa98f83946d8098c1926"
    )


def test_ballons_ctbd_output_filter_keeps_only_supported_confident_classes() -> None:
    labels = np.array([[0, 1, 2, 7]], dtype=np.int64)
    boxes = np.array(
        [[[1, 2, 20, 30], [3, 4, 18, 22], [7, 8, 28, 33], [0, 0, 5, 5]]],
        dtype=np.float32,
    )
    scores = np.array([[0.9, 0.8, 0.29, 0.99]], dtype=np.float32)

    bubbles, texts = _boxes_from_outputs(
        (labels, boxes, scores),
        threshold=0.3,
        provider="reference",
    )

    assert [record.label for record in bubbles] == ["bubble"]
    assert [record.label for record in texts] == ["text_bubble"]
    assert bubbles[0].xyxy == (1, 2, 20, 30)


def test_ctbd_content_mask_never_fills_the_detector_bbox_wholesale() -> None:
    image = np.full((80, 96, 3), 220, dtype=np.uint8)
    cv2.putText(image, "A", (35, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    text = DetectorBox((22, 20, 66, 60), "text", 1.0, "reference")
    bubble = DetectorBox((10, 10, 82, 70), "bubble", 1.0, "reference")

    mask, accepted = _build_content_mask(
        image,
        (text,),
        (bubble,),
        CTBDSettings(inpaint_mask_dilate=0),
    )

    assert accepted[0].label == "text_bubble"
    bbox_area = (66 - 22) * (60 - 20)
    assert 0 < np.count_nonzero(mask) < bbox_area


def test_positive_edit_is_detector_claim_minus_all_exact_protection() -> None:
    shape = (16, 20)
    claim = np.zeros(shape, dtype=np.uint8)
    claim[3:13, 3:17] = 255
    target = np.zeros(shape, dtype=np.uint8)
    target[5:9, 5:9] = 255
    protected = np.zeros(shape, dtype=np.uint8)
    protected[3:13, 10:12] = 255
    ambiguous = np.zeros(shape, dtype=np.uint8)
    ambiguous[10:12, 3:17] = 255
    ownership = np.zeros(shape, dtype=np.uint8)
    ownership[2:14, 2:15] = 255
    existing = np.zeros(shape, dtype=np.uint8)
    existing[5:7, 5:7] = 255
    masks = PageMasks(
        target,
        protected,
        ambiguous,
        ownership,
        np.full(shape, 255, np.uint8),
        existing,
    )

    edit = positive_edit_from_claim(claim, masks)

    assert np.count_nonzero(edit & protected) == 0
    assert np.count_nonzero(edit & ambiguous) == 0
    assert np.count_nonzero(edit & existing) == 0
    assert np.count_nonzero(edit[ownership == 0]) == 0


def test_provenance_fusion_uses_raw_text_box_only_as_claim_ownership() -> None:
    shape = (28, 40)
    raw_claim = np.zeros(shape, dtype=np.uint8)
    raw_claim[7:12, 6:13] = 255
    raw_claim[16:21, 25:32] = 255
    prior = np.zeros(shape, dtype=np.uint8)
    prior[3:24, 3:36] = 255
    seed = np.zeros(shape, dtype=np.uint8)
    seed[8:10, 8:10] = 255
    content = np.zeros(shape, dtype=np.uint8)
    content[17:20, 27:30] = 255
    zeros = np.zeros(shape, dtype=np.uint8)

    result = build_provenance_fusion(
        raw_claim,
        required_skip_prior=prior,
        required_skip_seed=seed,
        content_component_ownership=content,
        raw_detector_boxes=(
            DetectorBox((4, 5, 16, 14), "text", 1.0, "rtdetr"),
            DetectorBox((22, 14, 35, 23), "text", 1.0, "rtdetr"),
        ),
        existing_edit=zeros,
        structure_protect=zeros,
        ambiguous_protect=zeros,
    )

    assert [box.xyxy for box in result.selected_raw_text_boxes] == [(4, 5, 16, 14)]
    assert np.count_nonzero(result.positive_edit[7:12, 6:13]) == 35
    assert np.count_nonzero(result.positive_edit[17:20, 27:30]) == 9
    assert np.count_nonzero(result.positive_edit & cv2.bitwise_not(raw_claim)) == 0
    assert np.count_nonzero(result.positive_edit) < np.count_nonzero(result.ownership)


def test_provenance_fusion_rescue_without_raw_text_box_stays_on_content_pixels() -> None:
    shape = (20, 28)
    raw_claim = np.zeros(shape, dtype=np.uint8)
    raw_claim[5:15, 5:23] = 255
    prior = np.zeros(shape, dtype=np.uint8)
    prior[3:17, 3:25] = 255
    seed = np.zeros(shape, dtype=np.uint8)
    seed[8:12, 10:14] = 255
    content = np.zeros(shape, dtype=np.uint8)
    content[7:13, 9:15] = 255
    protected = np.zeros(shape, dtype=np.uint8)
    protected[7:9, 9:15] = 255
    existing = np.zeros(shape, dtype=np.uint8)
    existing[11:13, 9:11] = 255

    result = build_provenance_fusion(
        raw_claim,
        required_skip_prior=prior,
        required_skip_seed=seed,
        content_component_ownership=content,
        raw_detector_boxes=(
            DetectorBox((16, 4, 25, 16), "text", 1.0, "rtdetr"),
            DetectorBox((2, 2, 26, 18), "bubble", 1.0, "rtdetr"),
        ),
        existing_edit=existing,
        structure_protect=protected,
        ambiguous_protect=np.zeros(shape, dtype=np.uint8),
    )

    assert result.selected_raw_text_boxes == ()
    assert np.count_nonzero(result.positive_claim[content == 0]) == 0
    assert np.count_nonzero(result.positive_edit & protected) == 0
    assert np.count_nonzero(result.positive_edit & existing) == 0


def test_fixed_ctd_onnx_preprocess_preserves_rgb_channel_order() -> None:
    batch = np.array([[[[1, 2, 3]]]], dtype=np.uint8)
    tensor = _rgb_batch_to_nchw(batch)
    assert tensor.shape == (1, 3, 1, 1)
    assert tensor[0, :, 0, 0].tolist() == pytest.approx(
        [1 / 255, 2 / 255, 3 / 255]
    )


def test_fixed_ctd_onnx_letterbox_matches_reference_tensor_shape() -> None:
    image = np.zeros((6, 4, 3), dtype=np.uint8)
    image[..., 0] = 11
    tensor, _ratio, dw, dh = _letterboxed_rgb_to_nchw(image, 8)
    assert tensor.shape == (1, 3, 8, 8)
    assert (dw, dh) == (3, 0)
    assert tensor[0, 0, 0, 0] == pytest.approx(11 / 255)


def test_fixed_ctd_onnx_provider_selection_is_fail_closed() -> None:
    _require_primary_provider(
        "CUDAExecutionProvider",
        ("CUDAExecutionProvider", "CPUExecutionProvider"),
    )
    with pytest.raises(RuntimeError, match="was not honored"):
        _require_primary_provider(
            "CUDAExecutionProvider",
            ("CPUExecutionProvider",),
        )


def test_positive_mask_parity_rejects_empty_or_different_masks(tmp_path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    with pytest.raises(ValueError, match="contains no parity masks"):
        _compare_kind(reference, candidate, "positive_edit_masks")

    for root in (reference, candidate):
        (root / "positive_edit_masks").mkdir(parents=True)
    left = np.zeros((8, 8), dtype=np.uint8)
    right = left.copy()
    right[3, 4] = 255
    assert cv2.imwrite(
        str(reference / "positive_edit_masks" / "page_positive_edit.png"),
        left,
    )
    assert cv2.imwrite(
        str(candidate / "positive_edit_masks" / "page_positive_edit.png"),
        right,
    )
    rows = _compare_kind(reference, candidate, "positive_edit_masks")
    assert rows[0]["page_id"] == "page"
    assert rows[0]["xor_pixel_count"] == 1


def test_stage1_scores_every_connected_target_component() -> None:
    shape = (24, 32)
    target = np.zeros(shape, dtype=np.uint8)
    target[2:6, 3:7] = 255
    target[14:20, 22:28] = 255
    claim = target.copy()
    claim[14:16, 22:28] = 0
    zeros = np.zeros(shape, dtype=np.uint8)
    masks = PageMasks(
        target,
        zeros,
        zeros,
        np.full(shape, 255, np.uint8),
        np.full(shape, 255, np.uint8),
        zeros,
    )
    result = CandidateMaskResult("candidate", claim, claim, claim)
    page = Stage1Page("page", "unused", None, None, None)

    record, _edit = score_page(page, result, masks, variant="raw")

    assert record["target_edit_coverage"] == pytest.approx(40 / 52)
    assert sorted(record["component_coverages"]) == pytest.approx([2 / 3, 1.0])
    assert record["minimum_component_coverage"] == pytest.approx(2 / 3)


def test_stage1_reports_raw_claim_conflicts_before_exact_protection() -> None:
    shape = (12, 16)
    claim = np.zeros(shape, dtype=np.uint8)
    claim[2:10, 2:14] = 255
    target = np.zeros(shape, dtype=np.uint8)
    protected = np.zeros(shape, dtype=np.uint8)
    protected[3:5, 4:7] = 255
    ambiguous = np.zeros(shape, dtype=np.uint8)
    ambiguous[7:9, 8:12] = 255
    ownership = np.zeros(shape, dtype=np.uint8)
    ownership[1:11, 1:10] = 255
    existing = np.zeros(shape, dtype=np.uint8)
    masks = PageMasks(
        target,
        protected,
        ambiguous,
        ownership,
        np.full(shape, 255, np.uint8),
        existing,
    )
    result = CandidateMaskResult("candidate", claim, claim, claim)
    page = Stage1Page("page", "unused", None, None, None, no_edit=True)

    record, edit = score_page(page, result, masks, variant="raw")

    assert record["raw_claim_protected_overlap"] == 6
    assert record["raw_claim_ambiguous_overlap"] == 8
    assert record["raw_claim_outside_ownership_pixel_count"] == 32
    assert record["protected_edit_overlap"] == 0
    assert record["ambiguous_edit_overlap"] == 0
    assert np.count_nonzero(edit[ownership == 0]) == 0


def test_positive_edit_keeps_only_detector_components_touching_content_seed() -> None:
    shape = (20, 28)
    claim = np.zeros(shape, dtype=np.uint8)
    claim[4:9, 3:10] = 255
    claim[12:17, 18:25] = 255
    seed = np.zeros(shape, dtype=np.uint8)
    seed[6:8, 5:7] = 255
    full = np.full(shape, 255, dtype=np.uint8)
    zeros = np.zeros(shape, dtype=np.uint8)
    masks = PageMasks(zeros, zeros, zeros, full, seed, zeros)

    edit = positive_edit_from_claim(claim, masks)

    assert np.count_nonzero(edit[4:9, 3:10]) == 35
    assert np.count_nonzero(edit[12:17, 18:25]) == 0


def test_claim_is_split_at_ownership_boundary_before_seed_grouping() -> None:
    shape = (16, 30)
    claim = np.zeros(shape, dtype=np.uint8)
    claim[4:8, 2:9] = 255
    claim[4:8, 21:28] = 255
    claim[5:7, 8:22] = 255
    ownership = np.zeros(shape, dtype=np.uint8)
    ownership[2:12, 1:10] = 255
    ownership[2:12, 20:29] = 255
    seed = np.zeros(shape, dtype=np.uint8)
    seed[5:7, 3:5] = 255
    zeros = np.zeros(shape, dtype=np.uint8)
    masks = PageMasks(zeros, zeros, zeros, ownership, seed, zeros)

    edit = positive_edit_from_claim(claim, masks)

    assert np.count_nonzero(edit[:, 1:10]) > 0
    assert np.count_nonzero(edit[:, 20:29]) == 0


def test_manifest_v2_masks_must_be_pairwise_disjoint() -> None:
    left = np.zeros((8, 8), dtype=np.uint8)
    right = np.zeros_like(left)
    left[2:4, 2:4] = 255
    right[3:5, 3:5] = 255

    with pytest.raises(ValueError, match="evaluation masks overlap"):
        assert_disjoint_masks({"target": left, "protected": right})


def test_stage2_scores_actual_structure_damage_and_mask_outside_change() -> None:
    shape = (24, 32)
    source = np.full((*shape, 3), 180, dtype=np.uint8)
    candidate = source.copy()
    candidate[4:8, 5:9] = 120
    candidate[15:18, 22:26] = 60
    detector = np.zeros(shape, dtype=np.uint8)
    detector[4:8, 5:9] = 255
    target = detector.copy()
    protected = np.zeros(shape, dtype=np.uint8)
    protected[15:18, 22:26] = 255
    zeros = np.zeros(shape, dtype=np.uint8)
    masks = PageMasks(
        target,
        protected,
        zeros,
        np.full(shape, 255, np.uint8),
        np.full(shape, 255, np.uint8),
        zeros,
    )

    record, changed = score_stage2_page(
        source,
        candidate,
        detector,
        masks,
    )

    assert np.count_nonzero(changed_mask(source, candidate)) == 28
    assert np.count_nonzero(changed) == 28
    assert record["target_detector_coverage"] == 1.0
    assert record["minimum_target_component_coverage"] == 1.0
    assert record["protected_changed_pixel_count"] == 12
    assert record["changed_outside_detector_mask_pixel_count"] == 12


def test_positive_stage2_composite_changes_only_exact_edit_pixels() -> None:
    baseline = np.full((12, 16, 3), 150, dtype=np.uint8)
    baseline[2:4, 2:5] = 90
    generated = np.full_like(baseline, 230)
    positive = np.zeros(baseline.shape[:2], dtype=np.uint8)
    positive[7:10, 9:13] = 255
    baseline_mask = np.zeros_like(positive)
    baseline_mask[2:4, 2:5] = 255

    candidate, final_mask = composite_positive_result(
        baseline,
        generated,
        positive,
        baseline_mask,
    )

    assert np.all(candidate[positive > 0] == 230)
    assert np.array_equal(candidate[positive == 0], baseline[positive == 0])
    assert np.all(final_mask[2:4, 2:5] == 255)
    assert np.all(final_mask[7:10, 9:13] == 255)
    assert np.count_nonzero(final_mask) == 18


def test_stage2_residue_score_decreases_when_target_contrast_is_removed() -> None:
    source = np.full((48, 64, 3), 230, dtype=np.uint8)
    cv2.putText(source, "A", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    target = np.zeros(source.shape[:2], dtype=np.uint8)
    target[12:38, 16:40] = 255
    removed = source.copy()
    removed[target > 0] = 230

    baseline, _baseline_sum, count = residue_score(source, source, target)
    candidate, _candidate_sum, candidate_count = residue_score(source, removed, target)

    assert count == candidate_count
    assert baseline == pytest.approx(1.0)
    assert candidate is not None and candidate < baseline


def test_ownership_reconstruction_uses_bubble_or_text_free_envelope_without_claiming_page() -> None:
    mask = build_existing_ownership_mask(
        [
            {
                "text_class": "text_bubble",
                "bubble_xyxy": [4, 5, 20, 22],
                "mask_anchor_xyxy": [8, 9, 12, 13],
            },
            {
                "text_class": "text_free",
                "text_free_erase_envelope_xyxy": [24, 2, 31, 11],
            },
            {
                "text_class": "text_bubble",
                "bubble_xyxy": [-8, 25, 10, 99],
            },
        ],
        (32, 40),
    )

    assert np.all(mask[5:22, 4:20] == 255)
    assert np.all(mask[2:11, 24:31] == 255)
    assert np.all(mask[25:32, 0:10] == 255)
    assert mask[0, 39] == 0


def test_text_prior_ownership_prefers_semantic_anchor_over_whole_bubble() -> None:
    mask = build_existing_ownership_mask(
        [
            {
                "text_class": "text_bubble",
                "bubble_xyxy": [2, 2, 30, 30],
                "xyxy": [9, 10, 21, 23],
                "mask_anchor_xyxy": [11, 12, 19, 21],
            }
        ],
        (32, 32),
        scope="text_prior",
    )

    assert np.all(mask[12:21, 11:19] == 255)
    assert mask[3, 3] == 0
    assert mask[11, 10] == 0


def test_content_component_ownership_uses_inpaint_boxes_only() -> None:
    mask = build_existing_ownership_mask(
        [
            {
                "text_class": "text_bubble",
                "bubble_xyxy": [2, 2, 30, 30],
                "mask_anchor_xyxy": [6, 7, 26, 25],
                "inpaint_bboxes": [[8, 9, 12, 14], [20, 16, 24, 22]],
            }
        ],
        (32, 32),
        scope="content_components",
    )

    assert np.all(mask[9:14, 8:12] == 255)
    assert np.all(mask[16:22, 20:24] == 255)
    assert mask[8, 8] == 0
    assert mask[10, 15] == 0


def test_content_prior_matches_existing_one_pixel_text_prior_dilation() -> None:
    mask = build_existing_ownership_mask(
        [{"inpaint_bboxes": [[8, 9, 12, 14]]}],
        (24, 24),
        scope="content_prior",
    )

    assert np.all(mask[8:15, 7:13] == 255)
    assert mask[7, 7] == 0
    assert mask[15, 13] == 0


def test_required_skip_scope_excludes_nonrequired_blocks_without_pixel_guessing() -> None:
    blocks = [
        {
            "erase_skipped_reason": "microtexture_source_seed_unavailable",
            "mask_anchor_xyxy": [3, 4, 13, 15],
            "inpaint_bboxes": [[6, 7, 10, 12]],
        },
        {
            "erase_skipped_reason": "microtexture_intrusion",
            "mask_anchor_xyxy": [17, 4, 27, 15],
            "inpaint_bboxes": [[20, 7, 24, 12]],
        },
    ]

    prior = build_existing_ownership_mask(
        blocks,
        (24, 32),
        scope="required_skip_text_prior",
    )
    seeds = build_existing_ownership_mask(
        blocks,
        (24, 32),
        scope="required_skip_components",
    )

    assert np.all(prior[4:15, 3:13] == 255)
    assert np.count_nonzero(prior[:, 17:27]) == 0
    assert np.all(seeds[7:12, 6:10] == 255)
    assert np.count_nonzero(seeds[:, 20:24]) == 0


def test_ballons_ysg_postprocess_keeps_labels_boxes_and_exact_dilation() -> None:
    class _Tensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def cpu(self):
            return self

        def numpy(self):
            return self.value

        def __int__(self):
            return int(self.value)

    class _Boxes:
        cls = [_Tensor(0), _Tensor(1)]
        xyxy = [_Tensor([4, 5, 10, 12]), _Tensor([14, 3, 18, 8])]

    class _Result:
        names = {0: "balloon", 1: "other"}
        boxes = _Boxes()
        obb = None

    mask, boxes = mask_and_boxes_from_result(
        _Result(),
        (20, 24),
        YSGSettings(mask_dilate_size=0),
    )

    assert [box.label for box in boxes] == ["balloon"]
    assert np.all(mask[5:13, 4:11] == 255)
    assert np.count_nonzero(mask[3:9, 14:19]) == 0


def test_sickzil_preprocess_padding_and_class_mapping_match_reference() -> None:
    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    preprocessed = preprocess_sickzil(image)
    padded = modulo_padded(preprocessed, 16)
    logits = np.zeros((5, 7, 2), dtype=np.float32)
    logits[..., 0] = 0.7
    logits[1:4, 2:6, 1] = 0.9

    mask = class_map_to_mask(logits)

    assert preprocessed.dtype == np.float32
    assert preprocessed.min() == pytest.approx(0.0)
    assert preprocessed.max() == pytest.approx(float(image.max()) / 255.0)
    assert padded.shape == (16, 16, 3)
    assert np.all(mask[1:4, 2:6] == 255)
    assert np.count_nonzero(mask) == 12
