from __future__ import annotations

from unittest import mock

import numpy as np

from modules.masking.ctd_refiner import _block_text_search_mask
from modules.masking.legacy_bbox_mask import _build_legacy_base_block_mask
from modules.masking.legacy_bbox_rescue import build_block_rescue_mask
from modules.utils.block_geometry import legacy_adjust_xyxy, normalize_block_xyxy
from modules.utils.image_utils import annotate_block_mask_attribution
from modules.utils.mask_roi import build_text_prior_mask, resolve_inpaint_text_xyxy
from modules.utils.textblock import TextBlock, adjust_text_line_coordinates


def _block(
    *,
    xyxy=(30, 20, 170, 180),
    text_class="text_bubble",
    bubble_xyxy=(10, 10, 190, 190),
) -> TextBlock:
    block = TextBlock(
        text_bbox=np.asarray(xyxy, dtype=np.int32),
        bubble_bbox=(
            np.asarray(bubble_xyxy, dtype=np.int32)
            if bubble_xyxy is not None
            else None
        ),
        text_class=text_class,
        text="demo",
    )
    block._render_original_xyxy = [80, 80, 120, 110]
    block._render_bubble_xyxy = list(bubble_xyxy) if bubble_xyxy is not None else None
    block._render_area_source = "detected_bubble"
    block._render_area_xyxy = list(xyxy)
    return block


def test_resolver_uses_original_anchor_for_coherent_render_area() -> None:
    block = _block()

    resolved = resolve_inpaint_text_xyxy(block, (200, 200, 3))

    assert resolved == (80, 80, 120, 110)
    assert block._mask_anchor_xyxy == [80, 80, 120, 110]
    assert block._mask_anchor_source == "render_original"
    assert block._mask_anchor_relation == "render_area"


def test_resolver_accepts_legacy_minus_five_percent_checkpoint() -> None:
    block = _block(xyxy=(24, 24, 176, 176))
    block._render_area_xyxy = [20, 20, 180, 180]

    resolved = resolve_inpaint_text_xyxy(block, (200, 200, 3))

    assert resolved == (80, 80, 120, 110)
    assert block._mask_anchor_relation == "legacy_minus_five_percent"


def test_resolver_falls_back_for_manual_or_corrupt_render_metadata() -> None:
    manual = _block()
    manual.xyxy[:] = [40, 45, 90, 95]
    corrupt = _block()
    corrupt._render_original_xyxy = ["bad", 80, 120, 110]
    reversed_metadata = _block()
    reversed_metadata._render_original_xyxy = [120, 110, 80, 80]
    changed_bubble = _block()
    changed_bubble.bubble_xyxy[:] = [0, 0, 195, 195]

    assert resolve_inpaint_text_xyxy(manual, (200, 200, 3)) == (40, 45, 90, 95)
    assert resolve_inpaint_text_xyxy(corrupt, (200, 200, 3)) == (30, 20, 170, 180)
    assert resolve_inpaint_text_xyxy(reversed_metadata, (200, 200, 3)) == (30, 20, 170, 180)
    assert resolve_inpaint_text_xyxy(changed_bubble, (200, 200, 3)) == (30, 20, 170, 180)
    assert manual._mask_anchor_source == "current_xyxy"


def test_resolver_keeps_text_free_and_bubble_without_metadata_on_current_bbox() -> None:
    text_free = _block(
        xyxy=(40, 50, 90, 100),
        text_class="text_free",
        bubble_xyxy=None,
    )
    no_bubble = _block(xyxy=(45, 55, 95, 105), bubble_xyxy=None)

    assert resolve_inpaint_text_xyxy(text_free, (200, 200, 3)) == (40, 50, 90, 100)
    assert resolve_inpaint_text_xyxy(no_bubble, (200, 200, 3)) == (45, 55, 95, 105)


def test_ctd_prior_and_search_use_same_original_anchor() -> None:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    block = _block()
    roi = (0, 0, 200, 200)

    with mock.patch(
        "modules.utils.mask_roi.get_inpaint_bboxes",
        return_value=[[80, 80, 120, 110]],
    ) as get_boxes:
        prior = build_text_prior_mask(image, block, roi, dilate_iterations=0)
    search = _block_text_search_mask(block, roi, image.shape)

    assert tuple(get_boxes.call_args.args[0]) == (80, 80, 120, 110)
    assert prior[90, 90] == 255
    assert prior[25, 25] == 0
    assert search[90, 90] == 255
    assert search[25, 25] == 0


def test_legacy_bbox_and_hard_box_rescue_use_original_anchor() -> None:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    block = _block()

    with mock.patch(
        "modules.masking.legacy_bbox_mask.get_inpaint_bboxes",
        return_value=[[80, 80, 120, 110]],
    ) as get_boxes:
        legacy_mask, _ = _build_legacy_base_block_mask(image, block)
    rescue = build_block_rescue_mask(
        image,
        block,
        np.zeros((200, 200), dtype=np.uint8),
    )

    assert tuple(get_boxes.call_args.args[0]) == (80, 80, 120, 110)
    assert legacy_mask[90, 90] == 255
    assert legacy_mask[25, 25] == 0
    assert rescue["reason_codes"] != ["bbox_too_large"]
    assert rescue["metrics"]["bbox_area"] == 1200


def test_mask_attribution_span_uses_original_anchor() -> None:
    block = _block()
    final_mask = np.zeros((200, 200), dtype=np.uint8)
    final_mask[80:110, 80:120] = 255

    annotate_block_mask_attribution([block], final_mask, (200, 200, 3))

    assert block.block_mask_span_coverage == 1.0
    assert block._mask_anchor_source == "render_original"


def test_legacy_minus_five_relation_matches_existing_coordinate_adjustment() -> None:
    image = np.zeros((97, 131, 3), dtype=np.uint8)
    for x1 in (0, 1, 13, 64):
        for y1 in (0, 2, 17, 48):
            for width in (1, 2, 19, 20, 63):
                for height in (1, 3, 18, 21, 45):
                    x2 = min(image.shape[1], x1 + width)
                    y2 = min(image.shape[0], y1 + height)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    bbox = (x1, y1, x2, y2)

                    expected = adjust_text_line_coordinates(bbox, -5, -5, image)
                    actual = legacy_adjust_xyxy(bbox, image.shape, -5, -5)

                    assert actual == normalize_block_xyxy(expected, image.shape)

    for bbox in (
        (0.25, 0.75, 19.6, 20.4),
        (11.49, 7.51, 64.5, 41.49),
        (63.6, 48.4, 130.75, 96.6),
    ):
        expected = adjust_text_line_coordinates(bbox, -5, -5, image)
        actual = legacy_adjust_xyxy(bbox, image.shape, -5, -5)

        assert actual == normalize_block_xyxy(expected, image.shape)
