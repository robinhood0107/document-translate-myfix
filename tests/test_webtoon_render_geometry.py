from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QPointF

from modules.utils.block_geometry import (
    map_block_bbox_fields,
    render_geometry_relation,
    restore_block_bbox_fields,
    snapshot_block_bbox_fields,
)
from modules.utils.textblock import TextBlock
from pipeline.block_detection import BlockDetectionHandler
from pipeline.virtual_page import VirtualPage
from pipeline.webtoon_batch.chunk import ChunkMixin
from pipeline.webtoon_batch.flow import FlowMixin
from pipeline.webtoon_utils import (
    filter_and_convert_visible_blocks,
    restore_original_block_coordinates,
)


def _block(xyxy=(20, 30, 80, 70), bubble_xyxy=(10, 20, 90, 90)) -> TextBlock:
    block = TextBlock(
        text_bbox=np.asarray(xyxy, dtype=np.int32),
        bubble_bbox=np.asarray(bubble_xyxy, dtype=np.int32),
        text_class="text_bubble",
        text="demo",
    )
    block._render_original_xyxy = [30, 40, 70, 60]
    block._render_area_xyxy = list(xyxy)
    block._render_bubble_xyxy = list(bubble_xyxy)
    block._render_area_source = "detected_bubble"
    block.bubble_panel_group_xyxy = [5, 15, 95, 95]
    block.bubble_panel_render_xyxy = [12, 22, 88, 88]
    block.bubble_panel_extra_xyxy = [14, 24, 86, 86]
    return block


def _as_ints(value) -> list[int]:
    return [int(round(float(item))) for item in value]


def test_common_mapper_updates_render_and_dynamic_panel_geometry() -> None:
    block = _block()

    map_block_bbox_fields(
        block,
        lambda box: [box[0] + 7, box[1] - 3, box[2] + 7, box[3] - 3],
    )

    assert _as_ints(block.xyxy) == [27, 27, 87, 67]
    assert _as_ints(block.bubble_xyxy) == [17, 17, 97, 87]
    assert _as_ints(block._render_original_xyxy) == [37, 37, 77, 57]
    assert _as_ints(block._render_area_xyxy) == [27, 27, 87, 67]
    assert _as_ints(block._render_bubble_xyxy) == [17, 17, 97, 87]
    assert _as_ints(block.bubble_panel_render_xyxy) == [19, 19, 95, 85]
    assert _as_ints(block.bubble_panel_extra_xyxy) == [21, 21, 93, 83]
    assert isinstance(block.xyxy, np.ndarray)
    assert block.xyxy.dtype == np.int32


def test_common_mapper_promotes_integer_array_for_fractional_coordinates() -> None:
    block = _block()

    map_block_bbox_fields(
        block,
        lambda box: [value + 0.5 for value in box],
    )

    assert isinstance(block.xyxy, np.ndarray)
    assert block.xyxy.dtype == np.float64
    np.testing.assert_array_equal(
        block.xyxy,
        np.asarray([20.5, 30.5, 80.5, 70.5], dtype=np.float64),
    )


def test_common_mapper_preserves_legacy_minus_five_relation_when_clipped() -> None:
    block = _block(
        xyxy=(-7, 2, 107, 98),
        bubble_xyxy=(-10, 0, 110, 100),
    )
    block._render_original_xyxy = [-2, 30, 102, 70]
    block._render_area_xyxy = [-10, 0, 110, 100]
    block._render_bubble_xyxy = [-10, 0, 110, 100]

    def clamp(box: list[float]) -> list[int]:
        return [
            max(0, min(int(np.floor(box[0])), 100)),
            max(0, min(int(np.floor(box[1])), 100)),
            max(0, min(int(np.ceil(box[2])), 100)),
            max(0, min(int(np.ceil(box[3])), 100)),
        ]

    map_block_bbox_fields(block, clamp)

    assert _as_ints(block._render_area_xyxy) == [0, 0, 100, 100]
    assert _as_ints(block.xyxy) == [2, 2, 98, 98]
    assert _as_ints(block._render_original_xyxy) == [0, 30, 100, 70]
    assert render_geometry_relation(block, (100, 100, 3)) == (
        "legacy_minus_five_percent"
    )


def test_temporary_geometry_snapshot_restores_value_types_and_float_precision() -> None:
    block = _block()
    block.xyxy = np.asarray([20.25, 30.5, 80.75, 70.125], dtype=np.float64)
    block._render_original_xyxy = (30.125, 40.25, 70.5, 60.875)
    original_xyxy = block.xyxy.copy()
    original_render = block._render_original_xyxy
    snapshot = snapshot_block_bbox_fields(block)

    map_block_bbox_fields(
        block,
        lambda box: [value + 0.375 for value in box],
    )
    restore_block_bbox_fields(block, snapshot)

    assert isinstance(block.xyxy, np.ndarray)
    assert block.xyxy.dtype == np.float64
    np.testing.assert_array_equal(block.xyxy, original_xyxy)
    assert isinstance(block._render_original_xyxy, tuple)
    assert block._render_original_xyxy == original_render


def test_visible_coordinate_round_trip_restores_render_metadata() -> None:
    block = _block(
        xyxy=(300, 1150, 400, 1200),
        bubble_xyxy=(280, 1120, 430, 1240),
    )
    block._render_original_xyxy = [320, 1160, 380, 1190]
    block._render_area_xyxy = [300, 1150, 400, 1200]
    block._render_bubble_xyxy = [280, 1120, 430, 1240]
    block.bubble_panel_group_xyxy = [275, 1115, 435, 1245]
    block.bubble_panel_render_xyxy = [290, 1130, 420, 1230]
    original = {
        field: _as_ints(getattr(block, field))
        for field in (
            "xyxy",
            "bubble_xyxy",
            "_render_original_xyxy",
            "_render_area_xyxy",
            "_render_bubble_xyxy",
            "bubble_panel_group_xyxy",
            "bubble_panel_render_xyxy",
        )
    }
    manager = SimpleNamespace(
        image_positions=[1000],
        image_heights=[500],
        image_data={0: np.zeros((500, 800, 3), dtype=np.uint8)},
        webtoon_width=1000,
    )
    main_page = SimpleNamespace(
        blk_list=[block],
        image_viewer=SimpleNamespace(webtoon_manager=manager),
    )
    mapping = {
        "page_index": 0,
        "page_crop_top": 100,
        "page_crop_bottom": 400,
        "combined_y_start": 0,
    }

    visible = filter_and_convert_visible_blocks(
        main_page,
        SimpleNamespace(),
        [mapping],
    )

    assert visible == [block]
    assert _as_ints(block.xyxy) == [200, 50, 300, 100]
    assert _as_ints(block._render_original_xyxy) == [220, 60, 280, 90]
    restore_original_block_coordinates(visible)
    for field, expected in original.items():
        assert _as_ints(getattr(block, field)) == expected
    assert not hasattr(block, "_visible_geometry_snapshot")


def test_visible_detection_to_scene_maps_render_metadata() -> None:
    block = _block()
    viewer = SimpleNamespace(
        page_to_scene_coordinates=lambda _page, point: QPointF(
            point.x() + 100,
            point.y() + 1000,
        )
    )
    handler = BlockDetectionHandler(SimpleNamespace(image_viewer=viewer))

    handler._convert_visible_block_to_scene(
        block,
        {
            "page_index": 2,
            "page_crop_top": 50,
            "combined_y_start": 10,
        },
    )

    assert _as_ints(block.xyxy) == [120, 1070, 180, 1110]
    assert _as_ints(block.bubble_xyxy) == [110, 1060, 190, 1130]
    assert _as_ints(block._render_original_xyxy) == [130, 1080, 170, 1100]
    assert _as_ints(block._render_area_xyxy) == [120, 1070, 180, 1110]
    assert _as_ints(block.bubble_panel_render_xyxy) == [112, 1062, 188, 1128]


def test_virtual_to_physical_and_crop_local_map_every_render_bbox() -> None:
    block = _block()
    original_xyxy = _as_ints(block.xyxy)
    original_render = list(block._render_original_xyxy)
    vpage = VirtualPage(
        physical_page_index=0,
        physical_page_path="example.png",
        virtual_index=1,
        crop_top=500,
        crop_bottom=1000,
        crop_height=500,
        physical_width=200,
        physical_height=1000,
        virtual_id="p0_v1",
    )

    physical = FlowMixin._convert_blocks_to_physical([block], vpage)[0]

    assert _as_ints(block.xyxy) == original_xyxy
    assert block._render_original_xyxy == original_render
    assert _as_ints(physical.xyxy) == [20, 530, 80, 570]
    assert _as_ints(physical._render_original_xyxy) == [30, 540, 70, 560]
    assert _as_ints(physical.bubble_panel_render_xyxy) == [12, 522, 88, 588]

    harness = SimpleNamespace(_shift_xyxy=ChunkMixin._shift_xyxy)
    localized = ChunkMixin._localize_blocks_to_crop(
        harness,
        [physical],
        [5, 510, 100, 600],
    )[0]
    assert _as_ints(localized.xyxy) == [15, 20, 75, 60]
    assert _as_ints(localized._render_original_xyxy) == [25, 30, 65, 50]
    assert _as_ints(localized.bubble_panel_render_xyxy) == [7, 12, 83, 78]


def test_seam_semantic_merge_invalidates_inherited_render_geometry() -> None:
    top = _block(xyxy=(20, 80, 80, 100), bubble_xyxy=(10, 70, 90, 100))
    bottom = _block(xyxy=(20, 0, 80, 20), bubble_xyxy=(10, 0, 90, 30))
    harness = SimpleNamespace(
        _match_split_blocks=lambda *_args: [(0, 0)],
        _shift_xyxy=ChunkMixin._shift_xyxy,
        _union_xyxy=ChunkMixin._union_xyxy,
    )

    matches, consumed = FlowMixin._build_pair_split_matches(
        harness,
        {"detected_blocks": [top], "image": np.zeros((100, 100, 3), dtype=np.uint8)},
        {"detected_blocks": [bottom]},
        set(),
    )

    assert consumed == {0}
    assert len(matches) == 1
    owner = matches[0].owner_block
    assert owner._render_area_source == "text_bbox"
    assert owner._render_original_xyxy is None
    assert owner._render_area_xyxy is None
    assert owner._render_bubble_xyxy is None
    assert owner.bubble_panel_render_xyxy is None
    assert owner.bubble_panel_extra_xyxy is None
