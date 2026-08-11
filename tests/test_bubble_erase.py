from __future__ import annotations

import math
import unittest
from unittest import mock

import cv2
import numpy as np

from modules.utils.bubble_erase import (
    BubbleEraseBlockStats,
    ERASE_MODE_BUBBLE_FLAT_FILL,
    ERASE_MODE_BUBBLE_GRADIENT_FILL,
    ERASE_MODE_BUBBLE_LAMA_FALLBACK,
    ERASE_MODE_BUBBLE_SKIPPED,
    ERASE_MODE_BUBBLE_TELEA,
    build_bubble_residual_edit_mask,
    erase_text_bubble_regions,
    mask_pixel_count,
    set_block_erase_metadata,
    _bubble_interior_cap_mask,
    _bubble_roi_median_fill,
    _build_bubble_line_art_context,
    _capless_safe_source_seed_mask,
    _fill_bubble_mask,
    _line_art_protect_mask,
    _line_candidate_has_outside_text_support,
    _validated_bubble_interior_cap_mask,
)
from modules.utils.bubble_silhouette import extract_bubble_interior_cap_crop
from modules.utils.textblock import TextBlock


def _block(*, xyxy, bubble_xyxy=None, text_class="text_bubble") -> TextBlock:
    return TextBlock(
        text_bbox=np.asarray(xyxy, dtype=np.int32),
        bubble_bbox=np.asarray(bubble_xyxy, dtype=np.int32) if bubble_xyxy is not None else None,
        text_class=text_class,
    )


def _round_texture_fixture(
    *,
    sizes: tuple[int, int],
    pitch: int,
    row_offset: float,
    angle_degrees: float,
    phase: tuple[float, float] = (0.0, 0.0),
    texture_value: int = 85,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, TextBlock]:
    original = np.full((100, 120, 3), 180, dtype=np.uint8)
    outer = np.zeros((100, 120), dtype=np.uint8)
    inner = np.zeros((100, 120), dtype=np.uint8)
    cv2.ellipse(
        outer,
        (60, 50),
        (48, 36),
        0,
        0,
        360,
        255,
        -1,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        inner,
        (60, 50),
        (44, 32),
        0,
        0,
        360,
        255,
        -1,
        cv2.LINE_AA,
    )
    original[outer > 0] = 20
    original[inner > 0] = 245
    texture_mask = np.zeros((100, 120), dtype=np.uint8)
    angle = math.radians(float(angle_degrees))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    for grid_y in range(-8, 9):
        for grid_x in range(-8, 9):
            shifted_x = (
                grid_x * pitch
                + ((grid_y & 1) * float(row_offset))
            )
            x = round(
                60
                + float(phase[0])
                + (shifted_x * cosine)
                - (grid_y * pitch * sine)
            )
            y = round(
                50
                + float(phase[1])
                + (shifted_x * sine)
                + (grid_y * pitch * cosine)
            )
            dot_size = int(sizes[(grid_x + grid_y) & 1])
            if not (0 <= x < 120 and 0 <= y < 100):
                continue
            patch = inner[y:y + dot_size, x:x + dot_size] > 0
            texture_mask[y:y + dot_size, x:x + dot_size][patch] = 255
    original[texture_mask > 0] = int(texture_value)
    source_mask = np.zeros((100, 120), dtype=np.uint8)
    source_mask[47:53, 57:63] = 255
    original[source_mask > 0] = 248
    texture_mask[source_mask > 0] = 0
    block = _block(
        xyxy=[40, 30, 78, 68],
        bubble_xyxy=[0, 0, 120, 100],
    )
    return original, texture_mask, source_mask, block


class BubbleEraseMetadataTests(unittest.TestCase):
    def test_set_block_erase_metadata_persists_debug_fields(self) -> None:
        class Block:
            pass

        block = Block()

        set_block_erase_metadata(
            block,
            BubbleEraseBlockStats(
                mode="bubble_flat_fill",
                edit_pixel_count=42,
                protect_pixel_count=7,
                skipped_reason="",
            ),
        )

        self.assertEqual(block._erase_mode, "bubble_flat_fill")
        self.assertEqual(block._erase_edit_pixel_count, 42)
        self.assertEqual(block._erase_protect_pixel_count, 7)
        self.assertEqual(block._erase_skipped_reason, "")

    def test_mask_pixel_count_counts_binary_pixels(self) -> None:
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 1:3] = 255

        self.assertEqual(mask_pixel_count(mask), 4)

    def test_interior_cap_keeps_the_existing_erase_policy(self) -> None:
        crop = np.zeros((8, 8, 3), dtype=np.uint8)
        seed = np.zeros((8, 8), dtype=np.uint8)
        seed[3:5, 3:5] = 255
        detected = np.full((8, 8), 255, dtype=np.uint8)

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=detected,
        ) as extract:
            result = _bubble_interior_cap_mask(crop, seed)

        np.testing.assert_array_equal(result, detected)
        self.assertEqual(extract.call_args.kwargs["min_area_ratio"], 0.20)
        self.assertEqual(extract.call_args.kwargs["max_area_ratio"], 1.0)
        self.assertEqual(extract.call_args.kwargs["min_seed_coverage"], 0.0)
        self.assertFalse(extract.call_args.kwargs["preserve_seed_after_erode"])

    def test_silhouette_cap_accepts_only_area_and_seed_coverage_bounds(self) -> None:
        crop = np.zeros((10, 10, 3), dtype=np.uint8)
        seed = np.zeros((10, 10), dtype=np.uint8)
        seed[4:6, 4:6] = 255
        accepted = np.zeros((10, 10), dtype=np.uint8)
        accepted[1:9, 1:9] = 255
        too_small = np.zeros((10, 10), dtype=np.uint8)
        too_small[4:6, 4:6] = 255
        misses_seed = np.zeros((10, 10), dtype=np.uint8)
        misses_seed[0:5, 0:5] = 255
        misses_seed[4:6, 4:6] = 0

        vendor_path = (
            "modules.source_parity_vendor.utils.textblock_mask.extract_ballon_mask"
        )
        with mock.patch(vendor_path, return_value=(accepted, None)):
            result = extract_bubble_interior_cap_crop(crop, seed)
        self.assertIsNotNone(result)
        self.assertTrue(np.all(result[seed > 0] == 255))

        for rejected in (too_small, np.full((10, 10), 255, dtype=np.uint8), misses_seed):
            with self.subTest(nonzero=int(np.count_nonzero(rejected))):
                with mock.patch(vendor_path, return_value=(rejected, None)):
                    self.assertIsNone(
                        extract_bubble_interior_cap_crop(crop, seed)
                    )

    def test_silhouette_cap_fails_closed_when_vendor_detection_raises(self) -> None:
        crop = np.zeros((10, 10, 3), dtype=np.uint8)
        seed = np.zeros((10, 10), dtype=np.uint8)
        seed[4:6, 4:6] = 255

        with mock.patch(
            "modules.source_parity_vendor.utils.textblock_mask.extract_ballon_mask",
            side_effect=RuntimeError("detector failed"),
        ):
            result = extract_bubble_interior_cap_crop(crop, seed)

        self.assertIsNone(result)

    def test_validated_cap_uses_real_extractor_for_large_and_small_bubbles(self) -> None:
        cases = (
            ((80, 120), (60, 40), (48, 28), (30, 50, 50, 70)),
            ((32, 48), (24, 16), (19, 11), (12, 19, 20, 29)),
        )

        for shape, center, axes, seed_roi in cases:
            with self.subTest(shape=shape):
                height, width = shape
                crop = np.full((height, width, 3), 255, dtype=np.uint8)
                cv2.ellipse(crop, center, axes, 0, 0, 360, (0, 0, 0), 2)
                seed = np.zeros((height, width), dtype=np.uint8)
                y1, x1, y2, x2 = seed_roi
                seed[y1:y2, x1:x2] = 255

                result = _validated_bubble_interior_cap_mask(crop, seed)

                self.assertIsNotNone(result)
                self.assertTrue(np.all(result[seed > 0] == 255))

    def test_validated_cap_real_extractor_rejects_boundary_seed_after_erosion(self) -> None:
        crop = np.full((80, 120, 3), 255, dtype=np.uint8)
        cv2.ellipse(crop, (60, 40), (48, 28), 0, 0, 360, (0, 0, 0), 2)
        seed = np.zeros((80, 120), dtype=np.uint8)
        seed[10:18, 50:70] = 255

        result = _validated_bubble_interior_cap_mask(crop, seed)

        self.assertIsNone(result)


class BubbleResidualMaskTests(unittest.TestCase):
    def test_residual_mask_expands_text_like_pixels_near_seed(self) -> None:
        image = np.full((48, 48, 3), 120, dtype=np.uint8)
        image[20:24, 24:27] = 245
        source_mask = np.zeros((48, 48), dtype=np.uint8)
        source_mask[20:24, 20:24] = 255
        block = _block(xyxy=[18, 18, 30, 28], bubble_xyxy=[8, 8, 40, 40])

        edit_mask, stats = build_bubble_residual_edit_mask(image, source_mask, block)

        self.assertEqual(stats.mode, ERASE_MODE_BUBBLE_TELEA)
        self.assertGreater(stats.edit_pixel_count, mask_pixel_count(source_mask))
        self.assertGreater(np.count_nonzero(edit_mask[20:24, 24:27]), 0)

    def test_residual_mask_captures_orphan_dark_glyphs_inside_text_prior(self) -> None:
        image = np.full((64, 64, 3), 135, dtype=np.uint8)
        image[28:38, 36:42] = 15
        source_mask = np.zeros((64, 64), dtype=np.uint8)
        source_mask[20:24, 20:24] = 255
        block = _block(xyxy=[18, 18, 46, 44], bubble_xyxy=[8, 8, 56, 56])

        edit_mask, stats = build_bubble_residual_edit_mask(
            image,
            source_mask,
            block,
            seed_dilate_px=1,
        )

        self.assertEqual(stats.mode, ERASE_MODE_BUBBLE_TELEA)
        self.assertGreater(np.count_nonzero(edit_mask[28:38, 36:42]), 0)

    def test_residual_mask_captures_orphan_bright_glyphs_inside_text_prior(self) -> None:
        image = np.full((64, 64, 3), 135, dtype=np.uint8)
        image[28:38, 36:42] = 245
        source_mask = np.zeros((64, 64), dtype=np.uint8)
        source_mask[20:24, 20:24] = 255
        block = _block(xyxy=[18, 18, 46, 44], bubble_xyxy=[8, 8, 56, 56])

        edit_mask, stats = build_bubble_residual_edit_mask(
            image,
            source_mask,
            block,
            seed_dilate_px=1,
        )

        self.assertEqual(stats.mode, ERASE_MODE_BUBBLE_TELEA)
        self.assertGreater(np.count_nonzero(edit_mask[28:38, 36:42]), 0)

    def test_residual_mask_does_not_copy_boxy_source_seed_wholesale(self) -> None:
        image = np.full((72, 72, 3), 142, dtype=np.uint8)
        image[28:46, 30:35] = 20
        image[28:46, 42:47] = 245
        source_mask = np.zeros((72, 72), dtype=np.uint8)
        source_mask[24:50, 24:54] = 255
        block = _block(xyxy=[22, 20, 56, 54], bubble_xyxy=[12, 12, 62, 62])

        edit_mask, stats = build_bubble_residual_edit_mask(image, source_mask, block)

        self.assertEqual(stats.mode, ERASE_MODE_BUBBLE_TELEA)
        self.assertLess(mask_pixel_count(edit_mask), mask_pixel_count(source_mask))
        self.assertGreater(np.count_nonzero(edit_mask[28:46, 30:47]), 0)

    def test_residual_mask_rejects_long_rule_like_components(self) -> None:
        image = np.full((48, 48, 3), 120, dtype=np.uint8)
        image[22:24, 10:38] = 245
        source_mask = np.zeros((48, 48), dtype=np.uint8)
        source_mask[20:24, 20:24] = 255
        block = _block(xyxy=[8, 18, 40, 28], bubble_xyxy=[4, 4, 44, 44])

        edit_mask, stats = build_bubble_residual_edit_mask(image, source_mask, block)

        self.assertEqual(stats.mode, ERASE_MODE_BUBBLE_TELEA)
        self.assertEqual(np.count_nonzero(edit_mask[22:24, 32:38]), 0)

    def test_non_bubble_blocks_are_skipped(self) -> None:
        image = np.full((32, 32, 3), 120, dtype=np.uint8)
        source_mask = np.zeros((32, 32), dtype=np.uint8)
        source_mask[10:14, 10:14] = 255
        block = _block(xyxy=[8, 8, 16, 16], text_class="text_free")

        edit_mask, stats = build_bubble_residual_edit_mask(image, source_mask, block)

        self.assertEqual(stats.mode, ERASE_MODE_BUBBLE_SKIPPED)
        self.assertEqual(stats.skipped_reason, "not_text_bubble")
        self.assertEqual(mask_pixel_count(edit_mask), 0)

    def test_missing_bubble_roi_is_skipped_without_mask_expansion(self) -> None:
        original = np.full((64, 64, 3), 150, dtype=np.uint8)
        original[30:34, 30:34] = 245
        source_mask = np.zeros((64, 64), dtype=np.uint8)
        source_mask[30:34, 30:34] = 255
        block = _block(
            xyxy=[26, 26, 38, 38],
            bubble_xyxy=None,
        )

        result = erase_text_bubble_regions(
            original,
            original.copy(),
            source_mask,
            [block],
        )

        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_SKIPPED)
        self.assertEqual(block._erase_skipped_reason, "missing_bubble_roi")
        self.assertEqual(mask_pixel_count(result.edit_mask), 0)
        self.assertEqual(mask_pixel_count(result.fallback_mask), 0)
        np.testing.assert_array_equal(result.image, original)


class BubbleFillBackendTests(unittest.TestCase):
    def test_real_extractor_cap_failure_keeps_dense_texture_out_of_fallback(self) -> None:
        original = np.full((100, 120, 3), 180, dtype=np.uint8)
        outer = np.zeros((100, 120), dtype=np.uint8)
        inner = np.zeros((100, 120), dtype=np.uint8)
        cv2.ellipse(outer, (60, 50), (48, 36), 0, 0, 360, 255, -1, cv2.LINE_AA)
        cv2.ellipse(inner, (60, 50), (44, 32), 0, 0, 360, 255, -1, cv2.LINE_AA)
        original[outer > 0] = 20
        original[inner > 0] = 245
        texture_mask = np.zeros((100, 120), dtype=np.uint8)
        for y in range(20, 82, 5):
            for x in range(18, 104, 5):
                patch = inner[y:y + 3, x:x + 3] > 0
                texture_mask[y:y + 3, x:x + 3][patch] = 255
        original[texture_mask > 0] = 85
        source_mask = np.zeros((100, 120), dtype=np.uint8)
        source_mask[44:56, 54:60] = 255
        original[source_mask > 0] = 248
        texture_mask[source_mask > 0] = 0
        block = _block(
            xyxy=[45, 38, 75, 64],
            bubble_xyxy=[0, 0, 120, 100],
        )

        self.assertIsNone(
            _validated_bubble_interior_cap_mask(original, source_mask)
        )

        result = erase_text_bubble_regions(
            original,
            original.copy(),
            source_mask,
            [block],
        )

        union_mask = np.where(
            (result.edit_mask > 0) | (result.fallback_mask > 0),
            255,
            0,
        ).astype(np.uint8)
        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        self.assertEqual(
            np.count_nonzero(union_mask[texture_mask > 0]),
            0,
        )
        self.assertTrue(np.all(union_mask[source_mask > 0] > 0))

    def test_real_extractor_sparse_texture_stays_out_of_valid_cap_edit(self) -> None:
        original = np.full((100, 120, 3), 180, dtype=np.uint8)
        outer = np.zeros((100, 120), dtype=np.uint8)
        inner = np.zeros((100, 120), dtype=np.uint8)
        cv2.ellipse(
            outer,
            (60, 50),
            (48, 36),
            0,
            0,
            360,
            255,
            -1,
            cv2.LINE_AA,
        )
        cv2.ellipse(
            inner,
            (60, 50),
            (44, 32),
            0,
            0,
            360,
            255,
            -1,
            cv2.LINE_AA,
        )
        original[outer > 0] = 20
        original[inner > 0] = 245
        texture_mask = np.zeros((100, 120), dtype=np.uint8)
        for y in range(20, 82, 8):
            for x in range(18, 104, 8):
                patch = inner[y:y + 4, x:x + 4] > 0
                texture_mask[y:y + 4, x:x + 4][patch] = 255
        original[texture_mask > 0] = 85
        source_mask = np.zeros((100, 120), dtype=np.uint8)
        source_mask[47:53, 57:63] = 255
        original[source_mask > 0] = 248
        texture_mask[source_mask > 0] = 0
        block = _block(
            xyxy=[40, 30, 78, 68],
            bubble_xyxy=[0, 0, 120, 100],
        )

        self.assertIsNotNone(
            _validated_bubble_interior_cap_mask(original, source_mask)
        )

        result = erase_text_bubble_regions(
            original,
            original.copy(),
            source_mask,
            [block],
        )

        union_mask = np.where(
            (result.edit_mask > 0) | (result.fallback_mask > 0),
            255,
            0,
        ).astype(np.uint8)
        self.assertEqual(
            np.count_nonzero(union_mask[texture_mask > 0]),
            0,
        )
        self.assertTrue(np.all(union_mask[source_mask > 0] > 0))

    def test_real_extractor_preserves_rotated_mixed_texture_at_curved_boundary(
        self,
    ) -> None:
        for angle_degrees in (0, 33, 44):
            with self.subTest(angle_degrees=angle_degrees):
                original = np.full((100, 120, 3), 180, dtype=np.uint8)
                outer = np.zeros((100, 120), dtype=np.uint8)
                inner = np.zeros((100, 120), dtype=np.uint8)
                cv2.ellipse(
                    outer,
                    (60, 50),
                    (48, 36),
                    0,
                    0,
                    360,
                    255,
                    -1,
                    cv2.LINE_AA,
                )
                cv2.ellipse(
                    inner,
                    (60, 50),
                    (44, 32),
                    0,
                    0,
                    360,
                    255,
                    -1,
                    cv2.LINE_AA,
                )
                original[outer > 0] = 20
                original[inner > 0] = 245
                texture_mask = np.zeros((100, 120), dtype=np.uint8)
                angle = math.radians(float(angle_degrees))
                cosine = math.cos(angle)
                sine = math.sin(angle)
                for row, grid_y in enumerate(range(-4, 5)):
                    for column, grid_x in enumerate(range(-4, 5)):
                        dot_size = (3, 7)[(row + column) % 2]
                        x = round(
                            60
                            + (grid_x * 12 * cosine)
                            - (grid_y * 12 * sine)
                        )
                        y = round(
                            50
                            + (grid_x * 12 * sine)
                            + (grid_y * 12 * cosine)
                        )
                        if not (0 <= x < 120 and 0 <= y < 100):
                            continue
                        patch = inner[
                            y:y + dot_size,
                            x:x + dot_size,
                        ] > 0
                        texture_mask[
                            y:y + dot_size,
                            x:x + dot_size,
                        ][patch] = 255
                original[texture_mask > 0] = 85
                source_mask = np.zeros((100, 120), dtype=np.uint8)
                source_mask[47:53, 57:63] = 255
                original[source_mask > 0] = 248
                texture_mask[source_mask > 0] = 0
                block = _block(
                    xyxy=[40, 30, 78, 68],
                    bubble_xyxy=[0, 0, 120, 100],
                )

                self.assertIsNotNone(
                    _validated_bubble_interior_cap_mask(
                        original,
                        source_mask,
                    )
                )

                result = erase_text_bubble_regions(
                    original,
                    original.copy(),
                    source_mask,
                    [block],
                )

                union_mask = np.where(
                    (result.edit_mask > 0)
                    | (result.fallback_mask > 0),
                    255,
                    0,
                ).astype(np.uint8)
                self.assertEqual(
                    np.count_nonzero(union_mask[texture_mask > 0]),
                    0,
                )
                self.assertTrue(
                    np.all(union_mask[source_mask > 0] > 0)
                )

    def test_texture_field_routing_generalizes_across_mixed_and_staggered_grids(
        self,
    ) -> None:
        cases = (
            ("staggered", (3, 3), 8, 2.5, 21.0),
            ("dense_mixed", (2, 6), 7, 0.0, 45.0),
            ("sparse_mixed", (3, 7), 10, 0.0, 44.0),
            ("cap_truncated_dense", (2, 6), 10, 0.0, 33.0),
            ("cap_truncated_sparse", (3, 7), 10, 0.0, 33.0),
            ("rotated_staggered", (4, 8), 11, 5.5, 67.0),
            ("boundary_dense", (3, 7), 8, 0.0, 45.0),
            ("boundary_staggered", (4, 8), 10, 5.0, 17.0),
            ("boundary_rotated", (4, 8), 10, 5.0, 67.0),
        )
        for name, sizes, pitch, row_offset, angle_degrees in cases:
            with self.subTest(name=name):
                original, texture_mask, source_mask, block = (
                    _round_texture_fixture(
                        sizes=sizes,
                        pitch=pitch,
                        row_offset=row_offset,
                        angle_degrees=angle_degrees,
                    )
                )
                self.assertIsNotNone(
                    _validated_bubble_interior_cap_mask(
                        original,
                        source_mask,
                    )
                )

                result = erase_text_bubble_regions(
                    original,
                    original.copy(),
                    source_mask,
                    [block],
                )

                self.assertEqual(
                    block._erase_mode,
                    ERASE_MODE_BUBBLE_LAMA_FALLBACK,
                )
                self.assertEqual(
                    block._erase_skipped_reason,
                    "microtexture_intrusion",
                )
                np.testing.assert_array_equal(
                    result.fallback_mask,
                    source_mask,
                )
                self.assertEqual(mask_pixel_count(result.edit_mask), 0)
                self.assertEqual(
                    np.count_nonzero(
                        result.fallback_mask[texture_mask > 0]
                    ),
                    0,
                )

    def test_distributed_contrast_fallback_covers_connected_texture_field(
        self,
    ) -> None:
        original, texture_mask, source_mask, block = (
            _round_texture_fixture(
                sizes=(3, 7),
                pitch=7,
                row_offset=3.5,
                angle_degrees=60.0,
                phase=(2.0, 1.0),
                texture_value=210,
            )
        )
        self.assertIsNotNone(
            _validated_bubble_interior_cap_mask(
                original,
                source_mask,
            )
        )

        result = erase_text_bubble_regions(
            original,
            original.copy(),
            source_mask,
            [block],
        )

        self.assertEqual(
            block._erase_mode,
            ERASE_MODE_BUBBLE_LAMA_FALLBACK,
        )
        self.assertEqual(
            block._erase_skipped_reason,
            "microtexture_intrusion",
        )
        np.testing.assert_array_equal(result.fallback_mask, source_mask)
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[texture_mask > 0]),
            0,
        )

    def test_texture_field_detection_does_not_depend_on_text_prior_escape(
        self,
    ) -> None:
        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        texture_mask = np.zeros((96, 96), dtype=np.uint8)
        for y in range(16, 82, 6):
            for x in range(14, 82, 6):
                texture_mask[y:y + 3, x:x + 3] = 255
        original[texture_mask > 0] = 85
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        source_mask[42:48, 42:48] = 255
        original[source_mask > 0] = 245
        texture_mask[source_mask > 0] = 0
        block = _block(
            xyxy=[8, 8, 88, 88],
            bubble_xyxy=[8, 8, 88, 88],
        )
        interior_cap = np.zeros((80, 80), dtype=np.uint8)
        interior_cap[2:78, 2:78] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(
            block._erase_mode,
            ERASE_MODE_BUBBLE_LAMA_FALLBACK,
        )
        self.assertEqual(
            block._erase_skipped_reason,
            "microtexture_intrusion",
        )
        np.testing.assert_array_equal(result.fallback_mask, source_mask)
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[texture_mask > 0]),
            0,
        )

    def test_minimum_distributed_texture_field_is_preserved_under_full_prior(
        self,
    ) -> None:
        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        texture_mask = np.zeros((96, 96), dtype=np.uint8)
        for local_y in (10, 35, 60):
            for local_x in (10, 30, 50, 70):
                y = 8 + local_y
                x = 8 + local_x
                texture_mask[y:y + 3, x:x + 3] = 255
        original[texture_mask > 0] = 85
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        source_mask[42:48, 42:48] = 255
        original[source_mask > 0] = 245
        texture_mask[source_mask > 0] = 0
        block = _block(
            xyxy=[8, 8, 88, 88],
            bubble_xyxy=[8, 8, 88, 88],
        )
        interior_cap = np.zeros((80, 80), dtype=np.uint8)
        interior_cap[2:78, 2:78] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(
            block._erase_skipped_reason,
            "microtexture_intrusion",
        )
        np.testing.assert_array_equal(result.fallback_mask, source_mask)
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[texture_mask > 0]),
            0,
        )

    def test_five_isolated_marks_do_not_trigger_texture_field_routing(
        self,
    ) -> None:
        original = np.full((80, 80, 3), 150, dtype=np.uint8)
        for x, y in (
            (18, 18),
            (38, 38),
            (58, 58),
            (18, 58),
            (58, 18),
        ):
            original[y:y + 3, x:x + 3] = 85
        source_mask = np.zeros((80, 80), dtype=np.uint8)
        source_mask[37:43, 37:43] = 255
        original[source_mask > 0] = 245
        block = _block(
            xyxy=[14, 14, 66, 66],
            bubble_xyxy=[0, 0, 80, 80],
        )
        interior_cap = np.zeros((80, 80), dtype=np.uint8)
        interior_cap[2:78, 2:78] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ):
            erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertNotEqual(
            block._erase_skipped_reason,
            "microtexture_intrusion",
        )

    def test_texture_field_span_is_normalized_to_validated_cap(self) -> None:
        original = np.full((200, 200, 3), 150, dtype=np.uint8)
        texture_mask = np.zeros((200, 200), dtype=np.uint8)
        for y in (60, 75, 100, 105):
            for x in (60, 75, 105):
                texture_mask[y:y + 3, x:x + 3] = 255
        original[texture_mask > 0] = 85
        source_mask = np.zeros((200, 200), dtype=np.uint8)
        source_mask[90:96, 90:96] = 255
        original[source_mask > 0] = 245
        texture_mask[source_mask > 0] = 0
        block = _block(
            xyxy=[0, 0, 200, 200],
            bubble_xyxy=[0, 0, 200, 200],
        )
        interior_cap = np.zeros((200, 200), dtype=np.uint8)
        interior_cap[50:150, 50:150] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(
            block._erase_skipped_reason,
            "microtexture_intrusion",
        )
        np.testing.assert_array_equal(result.fallback_mask, source_mask)
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[texture_mask > 0]),
            0,
        )

    def test_texture_field_with_dense_box_seed_fails_closed_explicitly(
        self,
    ) -> None:
        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        texture_mask = np.zeros((96, 96), dtype=np.uint8)
        for y in range(16, 82, 6):
            for x in range(14, 82, 6):
                texture_mask[y:y + 3, x:x + 3] = 255
        original[texture_mask > 0] = 85
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        source_mask[36:60, 36:60] = 255
        original[source_mask > 0] = 245
        texture_mask[source_mask > 0] = 0
        block = _block(
            xyxy=[30, 30, 66, 66],
            bubble_xyxy=[8, 8, 88, 88],
        )
        interior_cap = np.zeros((80, 80), dtype=np.uint8)
        interior_cap[2:78, 2:78] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_SKIPPED)
        self.assertEqual(
            block._erase_skipped_reason,
            "microtexture_source_seed_unavailable",
        )
        self.assertEqual(mask_pixel_count(result.edit_mask), 0)
        self.assertEqual(mask_pixel_count(result.fallback_mask), 0)
        self.assertFalse(result.stats["applied"])
        self.assertEqual(result.stats["fallback_block_count"], 0)

    def test_detector_owned_dense_seed_uses_source_only_texture_fallback(
        self,
    ) -> None:
        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        texture_mask = np.zeros((96, 96), dtype=np.uint8)
        for y in range(16, 82, 6):
            for x in range(14, 82, 6):
                texture_mask[y:y + 3, x:x + 3] = 255
        original[texture_mask > 0] = 85
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        source_mask[36:60, 36:60] = 255
        original[source_mask > 0] = 245
        texture_mask[source_mask > 0] = 0
        block = _block(
            xyxy=[30, 30, 66, 66],
            bubble_xyxy=[8, 8, 88, 88],
        )
        block.mask_actual_pixel_count = mask_pixel_count(source_mask)
        interior_cap = np.zeros((80, 80), dtype=np.uint8)
        interior_cap[2:78, 2:78] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(
            block._erase_mode,
            ERASE_MODE_BUBBLE_LAMA_FALLBACK,
        )
        self.assertEqual(
            block._erase_skipped_reason,
            "microtexture_intrusion",
        )
        np.testing.assert_array_equal(result.fallback_mask, source_mask)
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[texture_mask > 0]),
            0,
        )

    def test_texture_field_fallback_never_reopens_true_line_art(self) -> None:
        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        texture_mask = np.zeros((96, 96), dtype=np.uint8)
        for y in range(16, 82, 6):
            for x in range(14, 82, 6):
                texture_mask[y:y + 3, x:x + 3] = 255
        original[texture_mask > 0] = 85
        line_mask = np.zeros((96, 96), dtype=np.uint8)
        cv2.line(line_mask, (10, 34), (86, 34), 255, thickness=3)
        original[line_mask > 0] = 20
        texture_mask[line_mask > 0] = 0
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        source_mask[42:48, 42:48] = 255
        original[source_mask > 0] = 245
        texture_mask[source_mask > 0] = 0
        block = _block(
            xyxy=[38, 38, 52, 52],
            bubble_xyxy=[8, 8, 88, 88],
        )
        interior_cap = np.zeros((80, 80), dtype=np.uint8)
        interior_cap[2:78, 2:78] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        np.testing.assert_array_equal(result.fallback_mask, source_mask)
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[texture_mask > 0]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[line_mask > 0]),
            0,
        )

    def test_bubble_fill_excludes_priority_owned_pixels_from_background_sample(self) -> None:
        original = np.full((48, 48, 3), 100, dtype=np.uint8)
        original[10:38, 10:35] = 245
        edit_mask = np.zeros((48, 48), dtype=np.uint8)
        edit_mask[20:24, 36:39] = 255
        priority_owned = np.zeros((48, 48), dtype=np.uint8)
        priority_owned[10:38, 10:35] = 255

        result = _bubble_roi_median_fill(
            original,
            edit_mask,
            (8, 8, 40, 40),
            background_exclude_mask=priority_owned,
        )

        self.assertTrue(np.all(result[priority_owned > 0] == 245))
        self.assertEqual(int(np.mean(result[edit_mask > 0])), 100)

    def test_telea_fill_is_invariant_to_priority_owned_pixel_values(self) -> None:
        row, column = np.indices((64, 64))
        checker = np.where((row + column) % 2 == 0, 80, 160).astype(np.uint8)
        base = np.repeat(checker[:, :, None], 3, axis=2)
        protected = np.zeros((64, 64), dtype=np.uint8)
        protected[20:44, 18:32] = 255
        edit_mask = np.zeros((64, 64), dtype=np.uint8)
        edit_mask[24:40, 32:42] = 255
        first = base.copy()
        second = base.copy()
        first[protected > 0] = 245
        second[protected > 0] = 120

        first_result, first_mode = _fill_bubble_mask(
            first,
            edit_mask,
            bubble_roi=(8, 8, 56, 56),
            background_exclude_mask=protected,
        )
        second_result, second_mode = _fill_bubble_mask(
            second,
            edit_mask,
            bubble_roi=(8, 8, 56, 56),
            background_exclude_mask=protected,
        )

        self.assertEqual(first_mode, ERASE_MODE_BUBBLE_TELEA)
        self.assertEqual(second_mode, ERASE_MODE_BUBBLE_TELEA)
        np.testing.assert_array_equal(
            first_result[edit_mask > 0],
            second_result[edit_mask > 0],
        )

    def test_fill_routes_to_lama_when_all_background_samples_are_excluded(self) -> None:
        original = np.full((64, 64, 3), 150, dtype=np.uint8)
        edit_mask = np.zeros((64, 64), dtype=np.uint8)
        edit_mask[30:34, 30:34] = 255
        original[edit_mask > 0] = 245
        protected = np.full((64, 64), 255, dtype=np.uint8)
        protected[edit_mask > 0] = 0

        filled, mode = _fill_bubble_mask(
            original,
            edit_mask,
            bubble_roi=(8, 8, 56, 56),
            background_exclude_mask=protected,
        )

        self.assertEqual(mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        np.testing.assert_array_equal(filled, original)

    def test_fill_routes_to_lama_when_only_page_external_samples_remain(self) -> None:
        original = np.full((64, 64, 3), 40, dtype=np.uint8)
        original[16:48, 16:48] = 180
        edit_mask = np.zeros((64, 64), dtype=np.uint8)
        edit_mask[30:34, 30:34] = 255
        original[edit_mask > 0] = 245
        protected = np.zeros((64, 64), dtype=np.uint8)
        protected[16:48, 16:48] = 255
        protected[edit_mask > 0] = 0

        filled, mode = _fill_bubble_mask(
            original,
            edit_mask,
            bubble_roi=(16, 16, 48, 48),
            background_exclude_mask=protected,
        )

        self.assertEqual(mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        np.testing.assert_array_equal(filled, original)

    def test_fill_rejects_sparse_local_samples_before_telea(self) -> None:
        for external_value in (0, 40, 220, 255):
            with self.subTest(external_value=external_value):
                original = np.full(
                    (64, 64, 3),
                    external_value,
                    dtype=np.uint8,
                )
                original[16:48, 16:48] = 150
                edit_mask = np.zeros((64, 64), dtype=np.uint8)
                edit_mask[30:34, 30:34] = 255
                original[edit_mask > 0] = 245
                protected = np.zeros((64, 64), dtype=np.uint8)
                protected[16:48, 16:48] = 255
                protected[edit_mask > 0] = 0
                protected[24, 24] = 0
                protected[40, 40] = 0
                original[24, 24] = 40
                original[40, 40] = 220

                filled, mode = _fill_bubble_mask(
                    original,
                    edit_mask,
                    bubble_roi=(16, 16, 48, 48),
                    background_exclude_mask=protected,
                )

                self.assertEqual(
                    mode,
                    ERASE_MODE_BUBBLE_LAMA_FALLBACK,
                )
                np.testing.assert_array_equal(filled, original)

    def test_fill_rejects_sparse_local_samples_without_exclusions(
        self,
    ) -> None:
        original = np.full((64, 64, 3), 150, dtype=np.uint8)
        edit_mask = np.zeros((64, 64), dtype=np.uint8)
        edit_mask[16:48, 16:48] = 255
        edit_mask[24, 24] = 0
        edit_mask[40, 40] = 0
        original[edit_mask > 0] = 245
        original[24, 24] = 40
        original[40, 40] = 220

        filled, mode = _fill_bubble_mask(
            original,
            edit_mask,
            bubble_roi=(16, 16, 48, 48),
        )

        self.assertEqual(mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        np.testing.assert_array_equal(filled, original)

    def test_fill_requires_bilateral_samples_for_each_edit_component(
        self,
    ) -> None:
        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        edit_mask = np.zeros((96, 96), dtype=np.uint8)
        edit_mask[44:48, 28:32] = 255
        edit_mask[44:48, 64:68] = 255
        original[edit_mask > 0] = 245
        protected = np.zeros((96, 96), dtype=np.uint8)
        protected[16:80, 16:80] = 255
        protected[edit_mask > 0] = 0
        protected[36:42, 34:40] = 0
        protected[50:56, 56:62] = 0
        original[36:42, 34:40] = 40
        original[50:56, 56:62] = 220

        filled, mode = _fill_bubble_mask(
            original,
            edit_mask,
            bubble_roi=(16, 16, 80, 80),
            background_exclude_mask=protected,
        )

        self.assertEqual(mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        np.testing.assert_array_equal(filled, original)

    def test_far_protection_does_not_penalize_well_sampled_large_bubble(self) -> None:
        for roi_size in (96, 128, 192):
            with self.subTest(roi_size=roi_size):
                canvas_size = roi_size + 32
                original = np.full(
                    (canvas_size, canvas_size, 3),
                    150,
                    dtype=np.uint8,
                )
                edit_mask = np.zeros(
                    (canvas_size, canvas_size),
                    dtype=np.uint8,
                )
                center = 16 + (roi_size // 2)
                edit_mask[center - 2:center + 2, center - 2:center + 2] = 255
                original[edit_mask > 0] = 245
                protected = np.zeros_like(edit_mask)
                protected[22:26, 22:26] = 255

                filled, mode = _fill_bubble_mask(
                    original,
                    edit_mask,
                    bubble_roi=(16, 16, 16 + roi_size, 16 + roi_size),
                    background_exclude_mask=protected,
                )

                self.assertEqual(mode, ERASE_MODE_BUBBLE_FLAT_FILL)
                self.assertTrue(np.all(filled[edit_mask > 0] == 150))

    def test_telea_sampling_is_invariant_to_page_external_colors(self) -> None:
        edit_results: list[np.ndarray] = []
        for external_value in (0, 40, 120, 220, 255):
            original = np.full(
                (64, 64, 3),
                external_value,
                dtype=np.uint8,
            )
            original[16:48, 16:48] = 150
            edit_mask = np.zeros((64, 64), dtype=np.uint8)
            edit_mask[30:34, 30:34] = 255
            original[edit_mask > 0] = 245
            protected = np.zeros((64, 64), dtype=np.uint8)
            protected[16:48, 16:48] = 255
            protected[edit_mask > 0] = 0
            protected[23:25, 21:41] = 0
            original[23, 21:41] = 40
            original[24, 21:41] = 220

            filled, mode = _fill_bubble_mask(
                original,
                edit_mask,
                bubble_roi=(16, 16, 48, 48),
                background_exclude_mask=protected,
            )

            self.assertEqual(mode, ERASE_MODE_BUBBLE_TELEA)
            edit_results.append(filled[edit_mask > 0].copy())

        for edit_result in edit_results[1:]:
            np.testing.assert_array_equal(edit_result, edit_results[0])

    def test_local_flat_fill_is_invariant_to_page_external_colors(self) -> None:
        ring_six_samples = (
            (24, 30), (24, 32), (25, 27), (25, 29), (25, 36),
            (26, 37), (27, 38), (28, 39), (30, 24), (31, 24),
            (32, 24), (33, 24), (34, 39), (35, 39), (36, 38),
            (37, 37), (38, 29), (38, 35), (39, 30), (39, 33),
        )
        outer_samples = (
            (20, 30), (21, 35), (22, 38), (24, 41), (27, 22),
            (30, 21), (33, 42), (36, 41), (39, 22), (41, 25),
            (42, 28), (43, 33),
        )
        edit_results: list[np.ndarray] = []
        for external_value in (0, 40, 120, 220, 255):
            original = np.full(
                (64, 64, 3),
                external_value,
                dtype=np.uint8,
            )
            original[16:48, 16:48] = 150
            edit_mask = np.zeros((64, 64), dtype=np.uint8)
            edit_mask[30:34, 30:34] = 255
            original[edit_mask > 0] = 245
            protected = np.zeros((64, 64), dtype=np.uint8)
            protected[16:48, 16:48] = 255
            protected[edit_mask > 0] = 0
            for y, x in ring_six_samples:
                protected[y, x] = 0
            for index, (y, x) in enumerate(outer_samples):
                protected[y, x] = 0
                original[y, x] = 40 if index % 2 == 0 else 220

            filled, mode = _fill_bubble_mask(
                original,
                edit_mask,
                bubble_roi=(16, 16, 48, 48),
                background_exclude_mask=protected,
            )

            self.assertEqual(mode, ERASE_MODE_BUBBLE_FLAT_FILL)
            edit_results.append(filled[edit_mask > 0].copy())

        for edit_result in edit_results[1:]:
            np.testing.assert_array_equal(edit_result, edit_results[0])

    def test_bubble_erase_routes_unsampleable_edit_mask_to_lama(self) -> None:
        original = np.full((64, 64, 3), 150, dtype=np.uint8)
        source_mask = np.zeros((64, 64), dtype=np.uint8)
        source_mask[30:34, 30:34] = 255
        original[source_mask > 0] = 245
        protected = np.full((64, 64), 255, dtype=np.uint8)
        protected[source_mask > 0] = 0
        block = _block(
            xyxy=[26, 26, 38, 38],
            bubble_xyxy=[8, 8, 56, 56],
        )
        interior_cap = np.zeros((48, 48), dtype=np.uint8)
        interior_cap[2:46, 2:46] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
                protected_edit_mask=protected,
            )

        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        self.assertEqual(
            block._erase_skipped_reason,
            "bubble_background_sample_unavailable",
        )
        self.assertEqual(mask_pixel_count(result.edit_mask), 0)
        np.testing.assert_array_equal(result.fallback_mask, source_mask)
        np.testing.assert_array_equal(result.image, original)

    def test_gradient_partial_sampling_routes_whole_mask_to_lama(self) -> None:
        y_coords, x_coords = np.indices((128, 128))
        plane = np.clip(
            20 + (2 * x_coords) + (2 * y_coords),
            0,
            255,
        ).astype(np.uint8)
        original = np.repeat(plane[:, :, None], 3, axis=2)
        edit_mask = np.zeros((128, 128), dtype=np.uint8)
        edit_mask[10:14, 10:14] = 255
        edit_mask[100:104, 100:104] = 255
        original[edit_mask > 0] = 245
        protected = np.full((128, 128), 255, dtype=np.uint8)
        protected[20:36, 20:36] = 0
        protected[edit_mask > 0] = 0

        filled, mode = _fill_bubble_mask(
            original,
            edit_mask,
            bubble_roi=(0, 0, 128, 128),
            background_exclude_mask=protected,
        )

        self.assertEqual(mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        np.testing.assert_array_equal(filled, original)

    def test_flat_bubble_fill_changes_only_edit_mask(self) -> None:
        original = np.full((48, 48, 3), 128, dtype=np.uint8)
        original[20:24, 20:24] = 245
        current = original.copy()
        current[0, 0] = [5, 6, 7]
        source_mask = np.zeros((48, 48), dtype=np.uint8)
        source_mask[20:24, 20:24] = 255
        block = _block(xyxy=[18, 18, 28, 28], bubble_xyxy=[8, 8, 40, 40])
        interior_cap = np.zeros((32, 32), dtype=np.uint8)
        interior_cap[1:31, 1:31] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ):
            result = erase_text_bubble_regions(
                original,
                current,
                source_mask,
                [block],
            )

        self.assertTrue(result.stats["applied"])
        self.assertEqual(result.stats["changed_outside_edit_mask_pixel_count"], 0)
        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_FLAT_FILL)
        self.assertTrue(np.all(result.image[0, 0] == current[0, 0]))
        self.assertLess(int(np.mean(result.image[20:24, 20:24])), 180)

    def test_complex_bubble_uses_telea_and_preserves_outside_mask(self) -> None:
        original = np.full((56, 56, 3), 128, dtype=np.uint8)
        for x in range(8, 48, 4):
            original[8:48, x:x + 2] = 80
        original[24:28, 24:28] = 245
        current = original.copy()
        current[1, 1] = [9, 10, 11]
        source_mask = np.zeros((56, 56), dtype=np.uint8)
        source_mask[24:28, 24:28] = 255
        block = _block(xyxy=[20, 20, 34, 34], bubble_xyxy=[8, 8, 48, 48])
        interior_cap = np.zeros((40, 40), dtype=np.uint8)
        interior_cap[1:39, 1:39] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ), mock.patch(
            "modules.utils.bubble_erase._line_art_protect_mask",
            return_value=np.zeros((40, 40), dtype=np.uint8),
        ):
            result = erase_text_bubble_regions(
                original,
                current,
                source_mask,
                [block],
            )

        self.assertTrue(result.stats["applied"])
        self.assertEqual(result.stats["changed_outside_edit_mask_pixel_count"], 0)
        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_TELEA)
        self.assertTrue(np.all(result.image[1, 1] == current[1, 1]))

    def test_smooth_bubble_roi_prefers_flat_fill_for_white_text_ghosts(self) -> None:
        original = np.full((72, 72, 3), 142, dtype=np.uint8)
        original[24:42, 28:34] = 248
        original[28:46, 40:46] = 248
        current = original.copy()
        current[0, 0] = [5, 6, 7]
        source_mask = np.zeros((72, 72), dtype=np.uint8)
        source_mask[24:42, 28:34] = 255
        source_mask[28:46, 40:46] = 255
        block = _block(xyxy=[22, 20, 50, 50], bubble_xyxy=[12, 12, 60, 60])
        interior_cap = np.zeros((48, 48), dtype=np.uint8)
        interior_cap[2:46, 2:46] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ):
            result = erase_text_bubble_regions(
                original,
                current,
                source_mask,
                [block],
            )

        self.assertTrue(result.stats["applied"])
        self.assertEqual(result.stats["changed_outside_edit_mask_pixel_count"], 0)
        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_FLAT_FILL)
        self.assertTrue(np.all(result.image[0, 0] == current[0, 0]))
        self.assertLess(abs(int(np.mean(result.image[24:46, 28:46])) - 142), 4)

    def test_smooth_gradient_bubble_uses_gradient_fill_for_glyph_source_mask(self) -> None:
        gradient = np.tile(np.linspace(120, 170, 80, dtype=np.uint8), (80, 1))
        expected_background = np.repeat(gradient[:, :, None], 3, axis=2)
        original = expected_background.copy()
        original[28:48, 30:36] = 248
        original[28:48, 46:52] = 248
        original[36:40, 30:52] = 248
        current = original.copy()
        current[0, 0] = [5, 6, 7]
        source_mask = np.zeros((80, 80), dtype=np.uint8)
        source_mask[28:48, 30:36] = 255
        source_mask[28:48, 46:52] = 255
        source_mask[36:40, 30:52] = 255
        block = _block(xyxy=[22, 20, 60, 56], bubble_xyxy=[10, 10, 70, 70])
        interior_cap = np.zeros((60, 60), dtype=np.uint8)
        interior_cap[2:58, 2:58] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ):
            result = erase_text_bubble_regions(
                original,
                current,
                source_mask,
                [block],
            )

        self.assertTrue(result.stats["applied"])
        self.assertEqual(result.stats["changed_outside_edit_mask_pixel_count"], 0)
        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_GRADIENT_FILL)
        self.assertTrue(np.all(result.image[0, 0] == current[0, 0]))
        self.assertGreater(mask_pixel_count(result.edit_mask), 0)
        self.assertLess(
            abs(int(np.mean(result.image[28:48, 30:52])) - int(np.mean(expected_background[28:48, 30:52]))),
            12,
        )

    def test_dense_source_region_without_a_line_gap_keeps_gradient_fill(self) -> None:
        gradient = np.tile(np.linspace(120, 170, 80, dtype=np.uint8), (80, 1))
        expected_background = np.repeat(gradient[:, :, None], 3, axis=2)
        original = expected_background.copy()
        original[28:48, 30:36] = 248
        original[28:48, 46:52] = 248
        source_mask = np.zeros((80, 80), dtype=np.uint8)
        source_mask[24:52, 24:58] = 255
        block = _block(
            xyxy=[22, 20, 60, 56],
            bubble_xyxy=[10, 10, 70, 70],
        )
        interior_cap = np.zeros((60, 60), dtype=np.uint8)
        interior_cap[2:58, 2:58] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_GRADIENT_FILL)
        self.assertEqual(result.stats["fallback_block_count"], 0)
        self.assertLess(mask_pixel_count(result.edit_mask), mask_pixel_count(source_mask))
        self.assertLess(
            abs(
                int(np.mean(result.image[28:48, 30:52]))
                - int(np.mean(expected_background[28:48, 30:52]))
            ),
            12,
        )

    def test_round_bubble_outline_and_glyph_strokes_do_not_force_lama_fallback(
        self,
    ) -> None:
        original = np.full((120, 140, 3), 180, dtype=np.uint8)
        yy, xx = np.ogrid[:120, :140]
        oval = (((xx - 70) / 48.0) ** 2 + ((yy - 60) / 36.0) ** 2) <= 1.0
        inner = (((xx - 70) / 44.0) ** 2 + ((yy - 60) / 32.0) ** 2) <= 1.0
        outline = oval & ~inner
        original[oval] = 245
        original[outline] = 20
        original[44:78, 54:62] = 12
        original[44:52, 54:86] = 12
        original[58:66, 54:82] = 248
        source_mask = np.zeros((120, 140), dtype=np.uint8)
        source_mask[44:78, 54:62] = 255
        source_mask[44:52, 54:86] = 255
        source_mask[58:66, 54:82] = 255
        block = _block(
            xyxy=[48, 36, 94, 88],
            bubble_xyxy=[20, 20, 120, 100],
        )
        interior_cap = np.where(
            inner[20:100, 20:120],
            255,
            0,
        ).astype(np.uint8)

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ) as extract:
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(extract.call_args.kwargs["min_seed_coverage"], 0.98)
        self.assertEqual(extract.call_args.kwargs["max_area_ratio"], 0.995)
        self.assertNotEqual(block._erase_mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        self.assertEqual(result.stats["fallback_block_count"], 0)
        self.assertGreater(mask_pixel_count(result.edit_mask), 0)
        self.assertTrue(np.array_equal(result.image[outline], original[outline]))

    def test_real_silhouette_extractor_keeps_round_outline_out_of_line_routing(
        self,
    ) -> None:
        original = np.full((100, 120, 3), 180, dtype=np.uint8)
        cv2.ellipse(
            original,
            (60, 50),
            (48, 36),
            0,
            0,
            360,
            (245, 245, 245),
            -1,
            lineType=cv2.LINE_AA,
        )
        outline_before = original.copy()
        cv2.ellipse(
            original,
            (60, 50),
            (48, 36),
            0,
            0,
            360,
            (20, 20, 20),
            3,
            lineType=cv2.LINE_AA,
        )
        outline_mask = np.any(original != outline_before, axis=2)
        source_mask = np.zeros((100, 120), dtype=np.uint8)
        cv2.putText(
            source_mask,
            "H",
            (45, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            255,
            2,
            cv2.LINE_AA,
        )
        original[source_mask > 0] = 248
        block = _block(
            xyxy=[40, 30, 78, 68],
            bubble_xyxy=[0, 0, 120, 100],
        )

        result = erase_text_bubble_regions(
            original,
            original.copy(),
            source_mask,
            [block],
        )

        self.assertNotEqual(block._erase_mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        self.assertEqual(result.stats["fallback_block_count"], 0)
        self.assertTrue(np.array_equal(result.image[outline_mask], original[outline_mask]))

    def test_missing_bubble_silhouette_uses_lama_fallback(self) -> None:
        original = np.full((64, 64, 3), 142, dtype=np.uint8)
        original[30:34, 30:34] = 248
        source_mask = np.zeros((64, 64), dtype=np.uint8)
        source_mask[30:34, 30:34] = 255
        block = _block(
            xyxy=[26, 26, 38, 38],
            bubble_xyxy=[8, 8, 56, 56],
        )

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=None,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        self.assertEqual(block._erase_skipped_reason, "bubble_interior_cap_unavailable")
        self.assertEqual(result.stats["fallback_block_count"], 1)
        self.assertGreater(mask_pixel_count(result.fallback_mask), 0)
        self.assertEqual(mask_pixel_count(result.edit_mask), 0)

    def test_missing_silhouette_dense_hard_box_fails_closed_when_safe_seed_is_empty(
        self,
    ) -> None:
        original = np.full((64, 64, 3), 150, dtype=np.uint8)
        source_mask = np.zeros((64, 64), dtype=np.uint8)
        source_mask[20:36, 20:44] = 255
        original[source_mask > 0] = 167
        block = _block(
            xyxy=[20, 20, 44, 36],
            bubble_xyxy=[8, 8, 56, 56],
        )

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=None,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_SKIPPED)
        self.assertEqual(
            block._erase_skipped_reason,
            "bubble_interior_cap_source_seed_unavailable",
        )
        self.assertEqual(result.stats["fallback_block_count"], 0)
        self.assertEqual(mask_pixel_count(result.fallback_mask), 0)
        self.assertEqual(mask_pixel_count(result.edit_mask), 0)
        np.testing.assert_array_equal(result.image, original)

    def test_missing_silhouette_uses_source_only_lama_mask(self) -> None:
        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        original[42:45, 18:78] = 20
        original[28:38, 40:46] = 245
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        source_mask[28:38, 40:46] = 255
        block = _block(
            xyxy=[20, 20, 76, 60],
            bubble_xyxy=[8, 8, 88, 88],
        )

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=None,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        self.assertEqual(block._erase_skipped_reason, "bubble_interior_cap_unavailable")
        np.testing.assert_array_equal(
            result.fallback_mask,
            source_mask,
        )
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[42:45, 18:78]),
            0,
        )

    def test_missing_silhouette_excludes_attached_source_owned_rule_from_fallback(
        self,
    ) -> None:
        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        structural_bar = np.zeros((96, 96), dtype=np.uint8)
        structural_bar[46:53, 24:72] = 255
        original[structural_bar > 0] = 20
        source_mask = structural_bar.copy()
        attached_glyph = np.zeros((96, 96), dtype=np.uint8)
        attached_glyph[40:46, 44:50] = 255
        source_mask[attached_glyph > 0] = 255
        original[attached_glyph > 0] = 245
        isolated_glyph = np.zeros((96, 96), dtype=np.uint8)
        isolated_glyph[30:36, 60:66] = 255
        source_mask[isolated_glyph > 0] = 255
        original[isolated_glyph > 0] = 245
        block = _block(
            xyxy=[20, 28, 76, 62],
            bubble_xyxy=[8, 8, 88, 88],
        )

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=None,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        self.assertEqual(
            block._erase_skipped_reason,
            "bubble_interior_cap_source_seed_partially_suppressed",
        )
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[structural_bar > 0]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[attached_glyph > 0]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[isolated_glyph > 0]),
            36,
        )

    def test_capless_texture_route_uses_same_safe_source_only_mask(self) -> None:
        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        texture_mask = np.zeros((96, 96), dtype=np.uint8)
        for y in range(16, 82, 6):
            for x in range(14, 82, 6):
                texture_mask[y:y + 3, x:x + 3] = 255
        original[texture_mask > 0] = 85
        structural_bar = np.zeros((96, 96), dtype=np.uint8)
        structural_bar[46:53, 24:72] = 255
        original[structural_bar > 0] = 20
        source_mask = structural_bar.copy()
        attached_glyph = np.zeros((96, 96), dtype=np.uint8)
        attached_glyph[40:46, 44:50] = 255
        source_mask[attached_glyph > 0] = 255
        original[attached_glyph > 0] = 245
        isolated_glyph = np.zeros((96, 96), dtype=np.uint8)
        isolated_glyph[30:36, 60:66] = 255
        source_mask[isolated_glyph > 0] = 255
        original[isolated_glyph > 0] = 245
        texture_mask[source_mask > 0] = 0
        block = _block(
            xyxy=[20, 28, 76, 62],
            bubble_xyxy=[8, 8, 88, 88],
        )

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=None,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        self.assertEqual(
            block._erase_skipped_reason,
            "microtexture_source_seed_partially_suppressed",
        )
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[structural_bar > 0]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[attached_glyph > 0]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[isolated_glyph > 0]),
            36,
        )
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[texture_mask > 0]),
            0,
        )

    def test_valid_cap_zero_candidate_is_an_explicit_required_skip(self) -> None:
        original = np.full((64, 64, 3), 150, dtype=np.uint8)
        source_mask = np.zeros((64, 64), dtype=np.uint8)
        source_mask[20:36, 20:44] = 255
        original[source_mask > 0] = 167
        block = _block(
            xyxy=[20, 20, 44, 36],
            bubble_xyxy=[8, 8, 56, 56],
        )
        interior_cap = np.zeros((48, 48), dtype=np.uint8)
        interior_cap[2:46, 2:46] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_SKIPPED)
        self.assertEqual(
            block._erase_skipped_reason,
            "bubble_residual_source_seed_unavailable",
        )
        self.assertEqual(block._erase_edit_pixel_count, 0)
        self.assertEqual(mask_pixel_count(result.edit_mask), 0)
        self.assertEqual(mask_pixel_count(result.fallback_mask), 0)
        self.assertFalse(result.stats["applied"])
        np.testing.assert_array_equal(result.image, original)

    def test_capless_safe_source_seed_has_an_explicit_23_to_24_pixel_boundary(
        self,
    ) -> None:
        source_mask = np.zeros((48, 64), dtype=np.uint8)
        source_mask[6:12, 4:27] = 255
        source_mask[20:27, 4:28] = 255

        safe_mask = _capless_safe_source_seed_mask(source_mask)

        self.assertEqual(np.count_nonzero(safe_mask[6:12, 4:27]), 138)
        self.assertEqual(np.count_nonzero(safe_mask[20:27, 4:28]), 0)

    def test_capless_safe_source_seed_rejects_broad_structural_components(
        self,
    ) -> None:
        cases: dict[str, np.ndarray] = {}
        tee = np.zeros((80, 80), dtype=np.uint8)
        tee[38:45, 16:64] = 255
        tee[14:38, 36:43] = 255
        cases["tee"] = tee
        plus = np.zeros((80, 80), dtype=np.uint8)
        plus[37:44, 16:64] = 255
        plus[16:64, 37:44] = 255
        cases["plus"] = plus
        frame = np.zeros((80, 80), dtype=np.uint8)
        cv2.rectangle(frame, (18, 18), (61, 61), 255, 4)
        cases["frame"] = frame

        for name, structure in cases.items():
            with self.subTest(name=name):
                source_mask = structure.copy()
                isolated_glyph = np.zeros_like(source_mask)
                isolated_glyph[68:74, 68:74] = 255
                source_mask[isolated_glyph > 0] = 255

                safe_mask = _capless_safe_source_seed_mask(source_mask)

                self.assertEqual(
                    np.count_nonzero(safe_mask[structure > 0]),
                    0,
                )
                self.assertEqual(
                    np.count_nonzero(safe_mask[isolated_glyph > 0]),
                    36,
                )

    def test_capless_safe_source_seed_rejects_disconnected_structure_groups(
        self,
    ) -> None:
        structures: list[tuple[str, np.ndarray]] = []
        for angle_degrees in (0, 17, 33, 57, 91):
            angle = math.radians(float(angle_degrees))
            cosine = math.cos(angle)
            sine = math.sin(angle)
            for row_offset in (0.0, 5.0):
                grid = np.zeros((96, 96), dtype=np.uint8)
                for row in range(-2, 2):
                    for column in range(-3, 3):
                        shifted_x = (column * 12) + (
                            (row & 1) * row_offset
                        )
                        shifted_y = row * 12
                        x = round(
                            48
                            + (shifted_x * cosine)
                            - (shifted_y * sine)
                        )
                        y = round(
                            48
                            + (shifted_x * sine)
                            + (shifted_y * cosine)
                        )
                        grid[y:y + 3, x:x + 3] = 255
                structures.append(
                    (
                        f"grid-{angle_degrees}-{row_offset}",
                        grid,
                    )
                )

        dash_line = np.zeros((96, 96), dtype=np.uint8)
        for x in (12, 28, 44, 60):
            dash_line[42:45, x:x + 12] = 255

        diagonal_dashes = np.zeros((96, 96), dtype=np.uint8)
        direction = np.asarray([2.0, 1.0], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        for offset in (0, 18, 36, 54):
            center = np.asarray(
                [18 + offset, 22 + (offset // 2)],
                dtype=np.float64,
            )
            start = tuple(np.round(center - (direction * 5.0)).astype(int))
            end = tuple(np.round(center + (direction * 5.0)).astype(int))
            cv2.line(
                diagonal_dashes,
                start,
                end,
                255,
                3,
                cv2.LINE_8,
            )
        structures.extend(
            (
                ("dash_line", dash_line),
                ("diagonal_dashes", diagonal_dashes),
            )
        )

        for name, structure in structures:
            with self.subTest(name=name):
                safe_mask = _capless_safe_source_seed_mask(structure)
                self.assertEqual(mask_pixel_count(safe_mask), 0)

    def test_capless_safe_source_seed_keeps_small_non_repeated_glyph_groups(
        self,
    ) -> None:
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        source_mask[20:42, 20:26] = 255
        source_mask[20:42, 42:48] = 255
        for x, y in ((68, 14), (74, 34), (62, 54), (76, 70), (50, 76)):
            source_mask[y:y + 3, x:x + 3] = 255

        safe_mask = _capless_safe_source_seed_mask(source_mask)

        np.testing.assert_array_equal(safe_mask, source_mask)

    def test_capless_dash_group_requires_component_axis_alignment(self) -> None:
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        for x in (18, 42, 66):
            source_mask[30:52, x:x + 6] = 255

        safe_mask = _capless_safe_source_seed_mask(source_mask)

        np.testing.assert_array_equal(safe_mask, source_mask)

    def test_capless_dash_group_ignores_unrelated_glyph_orientations(
        self,
    ) -> None:
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        dash_mask = np.zeros_like(source_mask)
        for x in (12, 36, 60):
            dash_mask[47:50, x:x + 12] = 255
        glyph_mask = np.zeros_like(source_mask)
        glyph_mask[14:36, 80:86] = 255
        glyph_mask[60:82, 80:86] = 255
        source_mask[(dash_mask > 0) | (glyph_mask > 0)] = 255

        safe_mask = _capless_safe_source_seed_mask(source_mask)

        self.assertEqual(np.count_nonzero(safe_mask[dash_mask > 0]), 0)
        self.assertEqual(
            np.count_nonzero(safe_mask[glyph_mask > 0]),
            np.count_nonzero(glyph_mask),
        )

        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        original[source_mask > 0] = 245
        block = _block(
            xyxy=[8, 10, 90, 86],
            bubble_xyxy=[4, 4, 92, 92],
        )
        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=None,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        self.assertEqual(
            block._erase_skipped_reason,
            "bubble_interior_cap_source_seed_partially_suppressed",
        )
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[dash_mask > 0]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[glyph_mask > 0]),
            np.count_nonzero(glyph_mask),
        )

    def test_capless_dash_group_splits_parallel_collinear_bands(self) -> None:
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        for y in (28, 60):
            for x in (12, 36, 60):
                source_mask[y:y + 3, x:x + 12] = 255

        safe_mask = _capless_safe_source_seed_mask(source_mask)

        self.assertEqual(mask_pixel_count(safe_mask), 0)

        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        original[source_mask > 0] = 85
        block = _block(
            xyxy=[8, 20, 78, 70],
            bubble_xyxy=[4, 4, 92, 92],
        )
        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=None,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_SKIPPED)
        self.assertEqual(
            block._erase_skipped_reason,
            "bubble_interior_cap_source_seed_unavailable",
        )
        self.assertEqual(mask_pixel_count(result.edit_mask), 0)
        self.assertEqual(mask_pixel_count(result.fallback_mask), 0)
        np.testing.assert_array_equal(result.image, original)

    def test_capless_compact_shape_matching_preserves_aspect_and_jitter(
        self,
    ) -> None:
        varied_glyphs = np.zeros((96, 96), dtype=np.uint8)
        for (width, height), (x, y) in zip(
            ((3, 5), (4, 6), (5, 7), (6, 8), (4, 9), (5, 10)),
            ((12, 14), (36, 14), (62, 14), (12, 54), (36, 52), (62, 52)),
        ):
            varied_glyphs[y:y + height, x:x + width] = 255

        safe_glyphs = _capless_safe_source_seed_mask(varied_glyphs)

        np.testing.assert_array_equal(safe_glyphs, varied_glyphs)

        notched_dots = np.zeros((96, 96), dtype=np.uint8)
        origins = (
            (20, 24),
            (44, 24),
            (68, 24),
            (20, 60),
            (44, 60),
            (68, 60),
        )
        missing_pixels = (
            (0, 0),
            (1, 0),
            (2, 0),
            (0, 2),
            (1, 2),
            (2, 2),
        )
        for (x, y), (missing_x, missing_y) in zip(
            origins,
            missing_pixels,
        ):
            notched_dots[y:y + 3, x:x + 3] = 255
            notched_dots[y + missing_y, x + missing_x] = 0

        safe_dots = _capless_safe_source_seed_mask(notched_dots)

        self.assertEqual(mask_pixel_count(safe_dots), 0)

        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        original[notched_dots > 0] = 85
        block = _block(
            xyxy=[16, 18, 76, 68],
            bubble_xyxy=[8, 8, 88, 88],
        )
        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=None,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                notched_dots,
                [block],
            )

        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_SKIPPED)
        self.assertEqual(
            block._erase_skipped_reason,
            "bubble_interior_cap_source_seed_unavailable",
        )
        self.assertEqual(mask_pixel_count(result.edit_mask), 0)
        self.assertEqual(mask_pixel_count(result.fallback_mask), 0)
        np.testing.assert_array_equal(result.image, original)

        two_notch_dots = np.zeros((96, 96), dtype=np.uint8)
        missing_pixel_pairs = (
            ((0, 0), (1, 0)),
            ((1, 0), (2, 0)),
            ((0, 2), (1, 2)),
            ((1, 2), (2, 2)),
            ((0, 0), (0, 1)),
            ((2, 1), (2, 2)),
        )
        for (x, y), missing_pair in zip(origins, missing_pixel_pairs):
            two_notch_dots[y:y + 3, x:x + 3] = 255
            for missing_x, missing_y in missing_pair:
                two_notch_dots[y + missing_y, x + missing_x] = 0

        safe_two_notch_dots = _capless_safe_source_seed_mask(
            two_notch_dots
        )

        self.assertEqual(mask_pixel_count(safe_two_notch_dots), 0)

        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        original[two_notch_dots > 0] = 85
        block = _block(
            xyxy=[16, 18, 76, 68],
            bubble_xyxy=[8, 8, 88, 88],
        )
        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=None,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                two_notch_dots,
                [block],
            )

        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_SKIPPED)
        self.assertEqual(
            block._erase_skipped_reason,
            "bubble_interior_cap_source_seed_unavailable",
        )
        self.assertEqual(mask_pixel_count(result.edit_mask), 0)
        self.assertEqual(mask_pixel_count(result.fallback_mask), 0)
        np.testing.assert_array_equal(result.image, original)

    def test_capless_interior_field_suppresses_matching_boundary_continuation(
        self,
    ) -> None:
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        interior_mask = np.zeros_like(source_mask)
        for y in (12, 36):
            for x in (24, 48, 72):
                interior_mask[y:y + 3, x:x + 3] = 255
        boundary_mask = np.zeros_like(source_mask)
        for x in (24, 48, 72):
            boundary_mask[0:3, x:x + 3] = 255
        source_mask[(interior_mask > 0) | (boundary_mask > 0)] = 255

        safe_mask = _capless_safe_source_seed_mask(source_mask)

        self.assertEqual(
            np.count_nonzero(safe_mask[interior_mask > 0]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(safe_mask[boundary_mask > 0]),
            0,
        )

    def test_capless_compact_buckets_separate_glyph_size_and_merge_jitter(
        self,
    ) -> None:
        source_mask = np.zeros((128, 128), dtype=np.uint8)
        texture_mask = np.zeros_like(source_mask)
        sizes = (
            (3, 4),
            (4, 3),
            (3, 5),
            (5, 3),
            (4, 5),
            (5, 4),
        ) * 2
        origins = tuple(
            (18 + (column * 28), 18 + (row * 42))
            for row in range(3)
            for column in range(4)
        )
        for (width, height), (x, y) in zip(sizes, origins):
            texture_mask[y:y + height, x:x + width] = 255
        glyph_mask = np.zeros_like(source_mask)
        glyph_mask[94:102, 58:66] = 255
        source_mask[(texture_mask > 0) | (glyph_mask > 0)] = 255

        safe_mask = _capless_safe_source_seed_mask(source_mask)

        self.assertEqual(
            np.count_nonzero(safe_mask[texture_mask > 0]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(safe_mask[glyph_mask > 0]),
            np.count_nonzero(glyph_mask),
        )

        original = np.full((128, 128, 3), 150, dtype=np.uint8)
        original[texture_mask > 0] = 85
        original[glyph_mask > 0] = 245
        block = _block(
            xyxy=[12, 12, 118, 108],
            bubble_xyxy=[4, 4, 124, 124],
        )
        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=None,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        self.assertEqual(
            block._erase_skipped_reason,
            "bubble_interior_cap_source_seed_partially_suppressed",
        )
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[texture_mask > 0]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[glyph_mask > 0]),
            np.count_nonzero(glyph_mask),
        )

    def test_capless_six_component_field_accepts_minority_size_jitter(
        self,
    ) -> None:
        origins = (
            (20, 24),
            (44, 24),
            (68, 24),
            (20, 60),
            (44, 60),
            (68, 60),
        )
        cases = {
            "five_plus_one": ((3, 3),) * 5 + ((5, 4),),
            "four_plus_two": ((3, 3),) * 4 + ((5, 4),) * 2,
        }
        for case_name, sizes in cases.items():
            with self.subTest(case_name=case_name):
                source_mask = np.zeros((96, 96), dtype=np.uint8)
                for (width, height), (x, y) in zip(sizes, origins):
                    source_mask[y:y + height, x:x + width] = 255

                safe_mask = _capless_safe_source_seed_mask(source_mask)

                self.assertEqual(mask_pixel_count(safe_mask), 0)

                original = np.full((96, 96, 3), 150, dtype=np.uint8)
                original[source_mask > 0] = 85
                block = _block(
                    xyxy=[16, 18, 76, 68],
                    bubble_xyxy=[8, 8, 88, 88],
                )
                with mock.patch(
                    "modules.utils.bubble_erase."
                    "extract_bubble_interior_cap_crop",
                    return_value=None,
                ):
                    result = erase_text_bubble_regions(
                        original,
                        original.copy(),
                        source_mask,
                        [block],
                    )

                self.assertEqual(
                    block._erase_mode,
                    ERASE_MODE_BUBBLE_SKIPPED,
                )
                self.assertEqual(
                    block._erase_skipped_reason,
                    "bubble_interior_cap_source_seed_unavailable",
                )
                self.assertEqual(mask_pixel_count(result.edit_mask), 0)
                self.assertEqual(mask_pixel_count(result.fallback_mask), 0)
                np.testing.assert_array_equal(result.image, original)

    def test_capless_reverse_size_jitter_keeps_unrelated_large_glyph(
        self,
    ) -> None:
        origins = (
            (20, 24),
            (44, 24),
            (68, 24),
            (20, 60),
            (44, 60),
            (68, 60),
        )
        cases = {
            "five_plus_one": ((5, 4),) * 5 + ((3, 3),),
            "four_plus_two": ((5, 4),) * 4 + ((3, 3),) * 2,
        }
        for case_name, sizes in cases.items():
            with self.subTest(case_name=case_name):
                structure_mask = np.zeros((128, 128), dtype=np.uint8)
                for (width, height), (x, y) in zip(sizes, origins):
                    structure_mask[y:y + height, x:x + width] = 255
                glyph_mask = np.zeros_like(structure_mask)
                glyph_mask[94:102, 58:66] = 255
                source_mask = np.where(
                    (structure_mask > 0) | (glyph_mask > 0),
                    255,
                    0,
                ).astype(np.uint8)

                safe_mask = _capless_safe_source_seed_mask(source_mask)

                self.assertEqual(
                    np.count_nonzero(safe_mask[structure_mask > 0]),
                    0,
                )
                self.assertEqual(
                    np.count_nonzero(safe_mask[glyph_mask > 0]),
                    np.count_nonzero(glyph_mask),
                )

                original = np.full((128, 128, 3), 150, dtype=np.uint8)
                original[structure_mask > 0] = 85
                original[glyph_mask > 0] = 245
                block = _block(
                    xyxy=[16, 18, 76, 108],
                    bubble_xyxy=[8, 8, 120, 120],
                )
                with mock.patch(
                    "modules.utils.bubble_erase."
                    "extract_bubble_interior_cap_crop",
                    return_value=None,
                ):
                    result = erase_text_bubble_regions(
                        original,
                        original.copy(),
                        source_mask,
                        [block],
                    )

                self.assertEqual(
                    block._erase_mode,
                    ERASE_MODE_BUBBLE_LAMA_FALLBACK,
                )
                self.assertEqual(
                    block._erase_skipped_reason,
                    "bubble_interior_cap_source_seed_partially_suppressed",
                )
                self.assertEqual(
                    np.count_nonzero(
                        result.fallback_mask[structure_mask > 0]
                    ),
                    0,
                )
                self.assertEqual(
                    np.count_nonzero(result.fallback_mask[glyph_mask > 0]),
                    np.count_nonzero(glyph_mask),
                )

    def test_capless_size_tie_prefers_the_lattice_continuation(self) -> None:
        source_mask = np.zeros((128, 128), dtype=np.uint8)
        structure_mask = np.zeros_like(source_mask)
        for x, y in (
            (20, 24),
            (44, 24),
            (68, 24),
            (20, 60),
            (44, 60),
        ):
            structure_mask[y:y + 4, x:x + 5] = 255
        jitter_mask = np.zeros_like(source_mask)
        jitter_mask[60:65, 68:74] = 255
        glyph_mask = np.zeros_like(source_mask)
        glyph_mask[96:99, 96:99] = 255
        glyph_mask[96:99, 108:111] = 255
        source_mask[
            (structure_mask > 0)
            | (jitter_mask > 0)
            | (glyph_mask > 0)
        ] = 255

        safe_mask = _capless_safe_source_seed_mask(source_mask)

        self.assertEqual(
            np.count_nonzero(safe_mask[structure_mask > 0]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(safe_mask[jitter_mask > 0]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(safe_mask[glyph_mask > 0]),
            np.count_nonzero(glyph_mask),
        )

        original = np.full((128, 128, 3), 150, dtype=np.uint8)
        original[(structure_mask > 0) | (jitter_mask > 0)] = 85
        original[glyph_mask > 0] = 245
        block = _block(
            xyxy=[12, 16, 108, 108],
            bubble_xyxy=[8, 8, 120, 120],
        )
        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=None,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(
            block._erase_mode,
            ERASE_MODE_BUBBLE_LAMA_FALLBACK,
        )
        self.assertEqual(
            block._erase_skipped_reason,
            "bubble_interior_cap_source_seed_partially_suppressed",
        )
        self.assertEqual(
            np.count_nonzero(
                result.fallback_mask[
                    (structure_mask > 0) | (jitter_mask > 0)
                ]
            ),
            0,
        )
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[glyph_mask > 0]),
            np.count_nonzero(glyph_mask),
        )

    def test_capless_three_size_groups_preserve_one_repeated_field(
        self,
    ) -> None:
        source_mask = np.zeros((128, 128), dtype=np.uint8)
        structure_mask = np.zeros_like(source_mask)
        origins = (
            (16, 24),
            (40, 24),
            (64, 24),
            (88, 24),
            (16, 64),
            (40, 64),
            (64, 64),
            (88, 64),
        )
        sizes = (
            (3, 3),
            (5, 4),
            (6, 5),
            (3, 3),
            (5, 4),
            (6, 5),
            (3, 3),
            (5, 4),
        )
        for (width, height), (x, y) in zip(sizes, origins):
            structure_mask[y:y + height, x:x + width] = 255
        glyph_mask = np.zeros_like(source_mask)
        glyph_mask[116:119, 116:119] = 255
        source_mask[
            (structure_mask > 0) | (glyph_mask > 0)
        ] = 255

        safe_mask = _capless_safe_source_seed_mask(source_mask)

        self.assertEqual(
            np.count_nonzero(safe_mask[structure_mask > 0]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(safe_mask[glyph_mask > 0]),
            np.count_nonzero(glyph_mask),
        )

        original = np.full((128, 128, 3), 150, dtype=np.uint8)
        original[structure_mask > 0] = 85
        original[glyph_mask > 0] = 245
        block = _block(
            xyxy=[12, 16, 108, 108],
            bubble_xyxy=[8, 8, 120, 120],
        )
        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=None,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(
            block._erase_mode,
            ERASE_MODE_BUBBLE_LAMA_FALLBACK,
        )
        self.assertEqual(
            block._erase_skipped_reason,
            "bubble_interior_cap_source_seed_partially_suppressed",
        )
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[structure_mask > 0]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[glyph_mask > 0]),
            np.count_nonzero(glyph_mask),
        )

    def test_capless_boundary_threshold_does_not_absorb_remote_glyph(
        self,
    ) -> None:
        field_origins = (
            (20, 20),
            (44, 20),
            (68, 20),
            (20, 48),
            (44, 48),
            (68, 48),
            (20, 76),
            (44, 76),
            (68, 76),
            (92, 76),
        )
        for field_count in (8, 9, 10):
            with self.subTest(field_count=field_count):
                field_mask = np.zeros((128, 128), dtype=np.uint8)
                for x, y in field_origins[:field_count]:
                    field_mask[y:y + 3, x:x + 3] = 255
                glyph_mask = np.zeros_like(field_mask)
                glyph_mask[116:119, 116:119] = 255
                source_mask = np.where(
                    (field_mask > 0) | (glyph_mask > 0),
                    255,
                    0,
                ).astype(np.uint8)

                safe_mask = _capless_safe_source_seed_mask(source_mask)

                self.assertEqual(
                    np.count_nonzero(safe_mask[field_mask > 0]),
                    0,
                )
                self.assertEqual(
                    np.count_nonzero(safe_mask[glyph_mask > 0]),
                    np.count_nonzero(glyph_mask),
                )

        field_mask = np.zeros((128, 128), dtype=np.uint8)
        for x, y in field_origins:
            field_mask[y:y + 3, x:x + 3] = 255
        glyph_mask = np.zeros_like(field_mask)
        glyph_mask[116:119, 116:119] = 255
        source_mask = np.where(
            (field_mask > 0) | (glyph_mask > 0),
            255,
            0,
        ).astype(np.uint8)

        original = np.full((128, 128, 3), 150, dtype=np.uint8)
        original[field_mask > 0] = 85
        original[glyph_mask > 0] = 245
        block = _block(
            xyxy=[12, 12, 122, 122],
            bubble_xyxy=[4, 4, 124, 124],
        )
        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=None,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(
            block._erase_mode,
            ERASE_MODE_BUBBLE_LAMA_FALLBACK,
        )
        self.assertEqual(
            block._erase_skipped_reason,
            "bubble_interior_cap_source_seed_partially_suppressed",
        )
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[field_mask > 0]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[glyph_mask > 0]),
            np.count_nonzero(glyph_mask),
        )

    def test_capless_unrelated_interior_field_keeps_boundary_field_safe(
        self,
    ) -> None:
        source_mask = np.zeros((128, 128), dtype=np.uint8)
        interior_mask = np.zeros_like(source_mask)
        for y in (20, 50):
            for x in (28, 60, 92):
                interior_mask[y:y + 3, x:x + 3] = 255
        boundary_mask = np.zeros_like(source_mask)
        for y in (100, 122):
            for x in (8, 32, 56, 80, 104):
                boundary_mask[y:y + 6, x:x + 6] = 255
        source_mask[
            (interior_mask > 0) | (boundary_mask > 0)
        ] = 255

        safe_mask = _capless_safe_source_seed_mask(source_mask)

        self.assertEqual(
            np.count_nonzero(safe_mask[interior_mask > 0]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(safe_mask[boundary_mask > 0]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(safe_mask[122:128]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(safe_mask[100:106]),
            0,
        )

        original = np.full((128, 128, 3), 150, dtype=np.uint8)
        original[source_mask > 0] = 85
        block = _block(
            xyxy=[4, 4, 124, 124],
            bubble_xyxy=[0, 0, 128, 128],
        )
        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=None,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_SKIPPED)
        self.assertEqual(
            block._erase_skipped_reason,
            "bubble_interior_cap_source_seed_unavailable",
        )
        self.assertEqual(mask_pixel_count(result.edit_mask), 0)
        self.assertEqual(mask_pixel_count(result.fallback_mask), 0)
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[122:128]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[100:106]),
            0,
        )
        np.testing.assert_array_equal(result.image, original)

    def test_capless_boundary_spacing_uses_its_own_descriptor_group(
        self,
    ) -> None:
        source_mask = np.zeros((256, 256), dtype=np.uint8)
        interior_mask = np.zeros_like(source_mask)
        for row in range(6):
            for column in range(9):
                y = 24 + (row * 20)
                x = 28 + (column * 20)
                interior_mask[y:y + 3, x:x + 3] = 255
        boundary_mask = np.zeros_like(source_mask)
        for y in (210, 250):
            for x in (24, 72, 120, 168, 216):
                boundary_mask[y:y + 6, x:x + 6] = 255
        source_mask[
            (interior_mask > 0) | (boundary_mask > 0)
        ] = 255

        safe_mask = _capless_safe_source_seed_mask(source_mask)

        self.assertEqual(
            np.count_nonzero(safe_mask[interior_mask > 0]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(safe_mask[boundary_mask > 0]),
            0,
        )
        self.assertEqual(np.count_nonzero(safe_mask[210:216]), 0)
        self.assertEqual(np.count_nonzero(safe_mask[250:256]), 0)

        original = np.full((256, 256, 3), 150, dtype=np.uint8)
        original[source_mask > 0] = 85
        block = _block(
            xyxy=[8, 8, 248, 248],
            bubble_xyxy=[0, 0, 256, 256],
        )
        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=None,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_SKIPPED)
        self.assertEqual(mask_pixel_count(result.edit_mask), 0)
        self.assertEqual(mask_pixel_count(result.fallback_mask), 0)
        np.testing.assert_array_equal(result.image, original)

    def test_capless_boundary_lattice_does_not_absorb_remote_glyph(
        self,
    ) -> None:
        source_mask = np.zeros((256, 256), dtype=np.uint8)
        interior_mask = np.zeros_like(source_mask)
        for y in (36, 76):
            for x in (32, 72, 112):
                interior_mask[y:y + 3, x:x + 3] = 255
        boundary_mask = np.zeros_like(source_mask)
        for y in (210, 250):
            for x in (24, 72, 120, 168, 216):
                boundary_mask[y:y + 6, x:x + 6] = 255
        glyph_mask = np.zeros_like(source_mask)
        glyph_mask[160:166, 120:126] = 255
        source_mask[
            (interior_mask > 0)
            | (boundary_mask > 0)
            | (glyph_mask > 0)
        ] = 255

        safe_mask = _capless_safe_source_seed_mask(source_mask)

        self.assertEqual(
            np.count_nonzero(safe_mask[boundary_mask > 0]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(safe_mask[glyph_mask > 0]),
            np.count_nonzero(glyph_mask),
        )

        original = np.full((256, 256, 3), 150, dtype=np.uint8)
        original[(interior_mask > 0) | (boundary_mask > 0)] = 85
        original[glyph_mask > 0] = 245
        block = _block(
            xyxy=[112, 152, 134, 174],
            bubble_xyxy=[0, 0, 256, 256],
        )
        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=None,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(
            block._erase_mode,
            ERASE_MODE_BUBBLE_LAMA_FALLBACK,
        )
        self.assertEqual(
            block._erase_skipped_reason,
            "bubble_interior_cap_source_seed_partially_suppressed",
        )
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[boundary_mask > 0]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[glyph_mask > 0]),
            np.count_nonzero(glyph_mask),
        )

    def test_capless_boundary_lattice_accepts_repeated_staggered_row(
        self,
    ) -> None:
        source_mask = np.zeros((256, 256), dtype=np.uint8)
        interior_mask = np.zeros_like(source_mask)
        for y in (36, 76):
            for x in (32, 72, 112):
                interior_mask[y:y + 3, x:x + 3] = 255
        boundary_mask = np.zeros_like(source_mask)
        for x in (24, 72, 120, 168, 216):
            boundary_mask[250:256, x:x + 6] = 255
        for x in (48, 96, 144, 192, 240):
            boundary_mask[210:216, x:x + 6] = 255
        source_mask[
            (interior_mask > 0) | (boundary_mask > 0)
        ] = 255

        safe_mask = _capless_safe_source_seed_mask(source_mask)

        self.assertEqual(
            np.count_nonzero(safe_mask[boundary_mask > 0]),
            0,
        )
        self.assertEqual(np.count_nonzero(safe_mask[210:216]), 0)
        self.assertEqual(np.count_nonzero(safe_mask[250:256]), 0)

        original = np.full((256, 256, 3), 150, dtype=np.uint8)
        original[source_mask > 0] = 85
        block = _block(
            xyxy=[8, 8, 248, 248],
            bubble_xyxy=[0, 0, 256, 256],
        )
        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=None,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(
            block._erase_mode,
            ERASE_MODE_BUBBLE_LAMA_FALLBACK,
        )
        self.assertEqual(
            block._erase_skipped_reason,
            "bubble_interior_cap_source_seed_partially_suppressed",
        )
        self.assertEqual(mask_pixel_count(result.edit_mask), 0)
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[boundary_mask > 0]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[interior_mask > 0]),
            np.count_nonzero(interior_mask),
        )

    def test_capless_boundary_lattice_requires_two_same_phase_marks(
        self,
    ) -> None:
        for candidate_count in (1, 2):
            with self.subTest(candidate_count=candidate_count):
                source_mask = np.zeros((256, 256), dtype=np.uint8)
                repeated_mask = np.zeros_like(source_mask)
                for y in (210, 250):
                    for x in (24, 72, 120, 168, 216):
                        repeated_mask[y:y + 6, x:x + 6] = 255
                candidate_mask = np.zeros_like(source_mask)
                for x in (48, 96)[:candidate_count]:
                    candidate_mask[170:176, x:x + 6] = 255
                source_mask[
                    (repeated_mask > 0) | (candidate_mask > 0)
                ] = 255

                safe_mask = _capless_safe_source_seed_mask(source_mask)

                self.assertEqual(
                    np.count_nonzero(safe_mask[repeated_mask > 0]),
                    0,
                )
                expected_candidate_pixels = (
                    np.count_nonzero(candidate_mask)
                    if candidate_count == 1
                    else 0
                )
                self.assertEqual(
                    np.count_nonzero(safe_mask[candidate_mask > 0]),
                    expected_candidate_pixels,
                )

                original = np.full((256, 256, 3), 150, dtype=np.uint8)
                original[source_mask > 0] = 85
                block = _block(
                    xyxy=[8, 8, 248, 248],
                    bubble_xyxy=[0, 0, 256, 256],
                )
                with mock.patch(
                    "modules.utils.bubble_erase."
                    "extract_bubble_interior_cap_crop",
                    return_value=None,
                ):
                    result = erase_text_bubble_regions(
                        original,
                        original.copy(),
                        source_mask,
                        [block],
                    )

                self.assertEqual(
                    np.count_nonzero(
                        result.fallback_mask[candidate_mask > 0]
                    ),
                    expected_candidate_pixels,
                )
                self.assertEqual(
                    np.count_nonzero(
                        result.fallback_mask[repeated_mask > 0]
                    ),
                    0,
                )

    def test_capless_compact_group_requires_repeated_component_shapes(
        self,
    ) -> None:
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        cv2.circle(source_mask, (20, 24), 4, 255, -1, cv2.LINE_8)
        cv2.line(source_mask, (44, 24), (52, 24), 255, 3, cv2.LINE_8)
        cv2.line(source_mask, (48, 20), (48, 28), 255, 3, cv2.LINE_8)
        cv2.line(source_mask, (72, 20), (72, 28), 255, 3, cv2.LINE_8)
        cv2.line(source_mask, (72, 28), (80, 28), 255, 3, cv2.LINE_8)
        cv2.line(source_mask, (16, 58), (24, 58), 255, 3, cv2.LINE_8)
        cv2.line(source_mask, (20, 58), (20, 66), 255, 3, cv2.LINE_8)
        cv2.rectangle(source_mask, (44, 58), (52, 66), 255, 2)
        cv2.line(source_mask, (72, 58), (80, 66), 255, 2, cv2.LINE_8)
        cv2.line(source_mask, (80, 58), (72, 66), 255, 2, cv2.LINE_8)

        safe_mask = _capless_safe_source_seed_mask(source_mask)

        np.testing.assert_array_equal(safe_mask, source_mask)

        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        original[source_mask > 0] = 245
        block = _block(
            xyxy=[12, 16, 84, 70],
            bubble_xyxy=[8, 8, 88, 88],
        )
        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=None,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        self.assertEqual(
            block._erase_skipped_reason,
            "bubble_interior_cap_unavailable",
        )
        np.testing.assert_array_equal(result.fallback_mask, source_mask)

    def test_capless_compact_field_uses_six_interior_and_ten_boundary_thresholds(
        self,
    ) -> None:
        interior_six = np.zeros((96, 96), dtype=np.uint8)
        for y in (24, 60):
            for x in (24, 48, 72):
                interior_six[y:y + 3, x:x + 3] = 255

        interior_five = interior_six.copy()
        interior_five[60:63, 72:75] = 0

        boundary_ten = np.zeros((96, 96), dtype=np.uint8)
        for x in (1, 20, 40, 60, 82):
            boundary_ten[1:4, x:x + 3] = 255
            boundary_ten[82:85, x:x + 3] = 255

        self.assertEqual(
            mask_pixel_count(
                _capless_safe_source_seed_mask(interior_six)
            ),
            0,
        )
        np.testing.assert_array_equal(
            _capless_safe_source_seed_mask(interior_five),
            interior_five,
        )
        self.assertEqual(
            mask_pixel_count(
                _capless_safe_source_seed_mask(boundary_ten)
            ),
            0,
        )

    def test_capless_disconnected_structure_is_an_explicit_required_skip(
        self,
    ) -> None:
        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        for row in range(4):
            for column in range(6):
                y = 18 + (row * 12)
                x = 12 + (column * 12)
                source_mask[y:y + 3, x:x + 3] = 255
        original[source_mask > 0] = 85
        block = _block(
            xyxy=[10, 14, 78, 62],
            bubble_xyxy=[8, 8, 88, 88],
        )

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=None,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_SKIPPED)
        self.assertEqual(
            block._erase_skipped_reason,
            "bubble_interior_cap_source_seed_unavailable",
        )
        self.assertEqual(mask_pixel_count(result.edit_mask), 0)
        self.assertEqual(mask_pixel_count(result.fallback_mask), 0)
        np.testing.assert_array_equal(result.image, original)

    def test_capless_detector_owned_compact_glyphs_remain_fallback_owned(
        self,
    ) -> None:
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        for y in (24, 56):
            for x in (24, 48, 72):
                source_mask[y:y + 3, x:x + 3] = 255
        self.assertEqual(
            mask_pixel_count(_capless_safe_source_seed_mask(source_mask)),
            0,
        )
        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        original[source_mask > 0] = 85
        block = _block(
            xyxy=[16, 16, 80, 64],
            bubble_xyxy=[8, 8, 88, 88],
        )
        block.mask_actual_pixel_count = mask_pixel_count(source_mask)

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=None,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(
            block._erase_mode,
            ERASE_MODE_BUBBLE_LAMA_FALLBACK,
        )
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[source_mask > 0]),
            np.count_nonzero(source_mask),
        )

    def test_capless_detector_owned_dense_seed_uses_prior_residual_fallback(
        self,
    ) -> None:
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        source_mask[34:50, 30:54] = 255
        self.assertEqual(
            mask_pixel_count(_capless_safe_source_seed_mask(source_mask)),
            0,
        )
        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        original[source_mask > 0] = 85
        block = _block(
            xyxy=[30, 34, 54, 50],
            bubble_xyxy=[8, 8, 88, 88],
        )
        block.mask_actual_pixel_count = mask_pixel_count(source_mask)

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=None,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(
            block._erase_mode,
            ERASE_MODE_BUBBLE_LAMA_FALLBACK,
        )
        self.assertGreater(mask_pixel_count(result.fallback_mask), 0)
        self.assertEqual(mask_pixel_count(result.edit_mask), 0)

    def test_capless_detector_owned_low_contrast_seed_uses_source_fallback(
        self,
    ) -> None:
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        source_mask[34:50, 30:54] = 255
        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        original[source_mask > 0] = 167
        block = _block(
            xyxy=[30, 34, 54, 50],
            bubble_xyxy=[8, 8, 88, 88],
        )
        block.mask_actual_pixel_count = mask_pixel_count(source_mask)

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=None,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(
            block._erase_mode,
            ERASE_MODE_BUBBLE_LAMA_FALLBACK,
        )
        self.assertEqual(
            mask_pixel_count(result.fallback_mask),
            mask_pixel_count(source_mask),
        )
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[source_mask <= 0]),
            0,
        )

    def test_capless_detector_owned_seed_does_not_expand_to_orphan_text(
        self,
    ) -> None:
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        source_mask[32:38, 32:38] = 255
        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        original[source_mask > 0] = 245
        orphan = np.zeros((96, 96), dtype=np.uint8)
        orphan[46:51, 58:63] = 255
        original[orphan > 0] = 245
        block = _block(
            xyxy=[24, 24, 72, 60],
            bubble_xyxy=[8, 8, 88, 88],
        )
        block.mask_actual_pixel_count = mask_pixel_count(source_mask)

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=None,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(
            mask_pixel_count(result.fallback_mask),
            mask_pixel_count(source_mask),
        )
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[orphan > 0]),
            0,
        )

    def test_dense_source_mask_exposes_upstream_protected_line_gap(self) -> None:
        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        original[42:45, 36:60] = 20
        original[28:38, 42:48] = 245
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        source_mask[22:64, 28:68] = 255
        source_mask[42:45, 36:60] = 0
        block = _block(
            xyxy=[28, 22, 68, 64],
            bubble_xyxy=[8, 8, 88, 88],
        )
        interior_cap = np.zeros((80, 80), dtype=np.uint8)
        interior_cap[2:78, 2:78] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        self.assertEqual(block._erase_skipped_reason, "line_art_intrusion")
        self.assertGreater(mask_pixel_count(result.fallback_mask), 0)
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[42:45, 36:60]),
            0,
        )

    def test_dense_low_contrast_hard_box_with_line_fails_closed_when_safe_seed_is_empty(
        self,
    ) -> None:
        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        source_mask[22:62, 28:68] = 255
        original[source_mask > 0] = 167
        protected_line = np.zeros((96, 96), dtype=np.uint8)
        cv2.line(protected_line, (10, 47), (86, 47), 255, 3, cv2.LINE_8)
        original[protected_line > 0] = 20
        block = _block(
            xyxy=[28, 22, 68, 62],
            bubble_xyxy=[8, 8, 88, 88],
        )
        interior_cap = np.zeros((80, 80), dtype=np.uint8)
        interior_cap[2:78, 2:78] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_SKIPPED)
        self.assertEqual(
            block._erase_skipped_reason,
            "line_art_source_seed_unavailable",
        )
        self.assertEqual(result.stats["fallback_block_count"], 0)
        self.assertEqual(mask_pixel_count(result.fallback_mask), 0)
        self.assertEqual(mask_pixel_count(result.edit_mask), 0)
        np.testing.assert_array_equal(result.image, original)

    def test_dense_source_mask_protects_traversing_lines_across_orientations(
        self,
    ) -> None:
        cases = (
            ((28, 43), (67, 43)),
            ((30, 55), (66, 42)),
        )
        for start, end in cases:
            with self.subTest(start=start, end=end):
                original = np.full((96, 96, 3), 150, dtype=np.uint8)
                source_mask = np.zeros((96, 96), dtype=np.uint8)
                source_mask[22:64, 28:68] = 255
                protected_line = np.zeros((96, 96), dtype=np.uint8)
                cv2.line(protected_line, start, end, 255, 3, cv2.LINE_8)
                original[protected_line > 0] = 20
                original[28:38, 42:48] = 245
                source_mask[protected_line > 0] = 0
                block = _block(
                    xyxy=[28, 22, 68, 64],
                    bubble_xyxy=[8, 8, 88, 88],
                )
                interior_cap = np.zeros((80, 80), dtype=np.uint8)
                interior_cap[2:78, 2:78] = 255

                with mock.patch(
                    "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
                    return_value=interior_cap,
                ):
                    result = erase_text_bubble_regions(
                        original,
                        original.copy(),
                        source_mask,
                        [block],
                    )

                self.assertEqual(
                    block._erase_mode,
                    ERASE_MODE_BUBBLE_LAMA_FALLBACK,
                )
                self.assertEqual(block._erase_skipped_reason, "line_art_intrusion")
                self.assertGreater(mask_pixel_count(result.fallback_mask), 0)
                self.assertEqual(
                    np.count_nonzero(
                        result.fallback_mask[protected_line > 0]
                    ),
                    0,
                )

    def test_dense_source_mask_line_floor_is_rotation_independent(self) -> None:
        for line_length in (14, 16, 18):
            for angle_degrees in range(0, 180, 10):
                with self.subTest(
                    line_length=line_length,
                    angle_degrees=angle_degrees,
                ):
                    radians = math.radians(angle_degrees)
                    half_span = float(line_length - 1) / 2.0
                    start = (
                        round(48 - half_span * math.cos(radians)),
                        round(48 - half_span * math.sin(radians)),
                    )
                    end = (
                        round(48 + half_span * math.cos(radians)),
                        round(48 + half_span * math.sin(radians)),
                    )
                    original = np.full((96, 96, 3), 150, dtype=np.uint8)
                    source_mask = np.zeros((96, 96), dtype=np.uint8)
                    source_mask[20:76, 20:76] = 255
                    protected_line = np.zeros((96, 96), dtype=np.uint8)
                    cv2.line(
                        protected_line,
                        start,
                        end,
                        255,
                        3,
                        cv2.LINE_8,
                    )
                    original[protected_line > 0] = 20
                    original[28:38, 42:48] = 245
                    source_mask[protected_line > 0] = 0
                    block = _block(
                        xyxy=[20, 20, 76, 76],
                        bubble_xyxy=[8, 8, 88, 88],
                    )
                    interior_cap = np.zeros((80, 80), dtype=np.uint8)
                    interior_cap[2:78, 2:78] = 255

                    with mock.patch(
                        "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
                        return_value=interior_cap,
                    ):
                        result = erase_text_bubble_regions(
                            original,
                            original.copy(),
                            source_mask,
                            [block],
                        )

                    self.assertEqual(
                        block._erase_mode,
                        ERASE_MODE_BUBBLE_LAMA_FALLBACK,
                    )
                    self.assertEqual(
                        np.count_nonzero(
                            result.fallback_mask[protected_line > 0]
                        ),
                        0,
                    )

    def test_dense_glyph_background_slot_does_not_route_as_structure(self) -> None:
        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        source_mask[22:64, 28:70] = 255
        original[source_mask > 0] = 245
        source_mask[30:56, 48:51] = 0
        original[30:56, 48:51] = 150
        block = _block(
            xyxy=[28, 22, 70, 64],
            bubble_xyxy=[8, 8, 88, 88],
        )
        interior_cap = np.zeros((80, 80), dtype=np.uint8)
        interior_cap[2:78, 2:78] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertNotEqual(block._erase_mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        self.assertEqual(result.stats["fallback_block_count"], 0)
        self.assertGreater(mask_pixel_count(result.edit_mask), 0)

    def test_post_erosion_cap_must_still_cover_the_source_seed(self) -> None:
        original = np.full((64, 64, 3), 142, dtype=np.uint8)
        original[30:34, 30:34] = 248
        source_mask = np.zeros((64, 64), dtype=np.uint8)
        source_mask[30:34, 30:34] = 255
        block = _block(
            xyxy=[26, 26, 38, 38],
            bubble_xyxy=[8, 8, 56, 56],
        )
        eroded_cap = np.zeros((48, 48), dtype=np.uint8)
        eroded_cap[2:46, 2:46] = 255
        eroded_cap[22:23, 22:26] = 0

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=eroded_cap,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        self.assertEqual(block._erase_skipped_reason, "bubble_interior_cap_unavailable")
        self.assertEqual(result.stats["fallback_block_count"], 1)
        self.assertTrue(
            np.all(result.fallback_mask[source_mask > 0] > 0)
        )

    def test_empty_or_wrong_shape_post_erosion_cap_is_invalid(self) -> None:
        crop = np.full((32, 32, 3), 245, dtype=np.uint8)
        seed = np.zeros((32, 32), dtype=np.uint8)
        seed[14:18, 14:18] = 255

        for invalid_cap in (
            np.zeros((32, 32), dtype=np.uint8),
            np.full((16, 16), 255, dtype=np.uint8),
        ):
            with self.subTest(shape=invalid_cap.shape), mock.patch(
                "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
                return_value=invalid_cap,
            ):
                self.assertIsNone(
                    _validated_bubble_interior_cap_mask(crop, seed)
                )

    def test_post_erosion_cap_accepts_98_percent_and_rejects_97_percent(self) -> None:
        crop = np.full((32, 32, 3), 245, dtype=np.uint8)
        seed = np.zeros((32, 32), dtype=np.uint8)
        seed[10:20, 10:20] = 255
        accepted = np.full((32, 32), 255, dtype=np.uint8)
        accepted[10, 10:12] = 0
        rejected = accepted.copy()
        rejected[10, 12] = 0

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=accepted,
        ):
            self.assertIsNotNone(
                _validated_bubble_interior_cap_mask(crop, seed)
            )
        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=rejected,
        ):
            self.assertIsNone(
                _validated_bubble_interior_cap_mask(crop, seed)
            )

    def test_glyph_occluded_interior_line_still_uses_lama_fallback(self) -> None:
        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        original[34:37, 14:82] = 20
        original[26:40, 32:38] = 245
        original[26:30, 32:46] = 245
        original[26:40, 50:56] = 245
        original[26:30, 50:64] = 245
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        source_mask[26:40, 32:38] = 255
        source_mask[26:30, 32:46] = 255
        source_mask[26:40, 50:56] = 255
        source_mask[26:30, 50:64] = 255
        block = _block(
            xyxy=[26, 22, 62, 46],
            bubble_xyxy=[8, 8, 88, 88],
        )
        interior_cap = np.zeros((80, 80), dtype=np.uint8)
        interior_cap[6:74, 6:74] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        self.assertEqual(block._erase_skipped_reason, "line_art_intrusion")
        self.assertEqual(result.stats["fallback_block_count"], 1)
        self.assertGreater(mask_pixel_count(result.fallback_mask), 0)
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[34:37, 14:30]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[34:37, 64:82]),
            0,
        )

    def test_full_edge_proposal_recovers_line_crossing_source_glyph(self) -> None:
        original = np.full((112, 112, 3), 150, dtype=np.uint8)
        source_mask = np.zeros((112, 112), dtype=np.uint8)
        protected_line = np.zeros((112, 112), dtype=np.uint8)
        cv2.line(
            protected_line,
            (44, 56),
            (68, 56),
            255,
            2,
            cv2.LINE_8,
        )
        original[protected_line > 0] = 20
        source_mask[50:62, 53:59] = 255
        original[source_mask > 0] = 245
        orphan_branch = np.zeros((112, 112), dtype=np.uint8)
        cv2.line(
            orphan_branch,
            (49, 55),
            (49, 46),
            255,
            2,
            cv2.LINE_8,
        )
        original[orphan_branch > 0] = 245
        block = _block(
            xyxy=[49, 46, 63, 66],
            bubble_xyxy=[8, 8, 104, 104],
        )
        interior_cap = np.zeros((96, 96), dtype=np.uint8)
        interior_cap[2:94, 2:94] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        visible_line = (protected_line > 0) & (source_mask <= 0)
        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        self.assertEqual(block._erase_skipped_reason, "line_art_intrusion")
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[visible_line]),
            0,
        )
        self.assertGreater(
            np.count_nonzero(result.fallback_mask[source_mask > 0]),
            0,
        )
        branch_far_from_line = (orphan_branch > 0) & (
            np.indices(orphan_branch.shape)[0] <= 51
        )
        self.assertTrue(
            np.all(result.fallback_mask[branch_far_from_line] > 0)
        )

    def test_full_edge_proposal_does_not_protect_orphan_glyph_in_text_prior(
        self,
    ) -> None:
        for orphan_y in (44, 52):
            with self.subTest(orphan_y=orphan_y):
                original = np.full((96, 96, 3), 150, dtype=np.uint8)
                source_mask = np.zeros((96, 96), dtype=np.uint8)
                source_mask[40:46, 40:46] = 255
                original[source_mask > 0] = 245
                orphan_glyph = np.zeros((96, 96), dtype=np.uint8)
                orphan_glyph[orphan_y:orphan_y + 3, 34:52] = 255
                original[orphan_glyph > 0] = 245
                block = _block(
                    xyxy=[30, 30, 60, 60],
                    bubble_xyxy=[24, 24, 72, 72],
                )
                interior_cap = np.zeros((48, 48), dtype=np.uint8)
                interior_cap[2:46, 2:46] = 255

                with mock.patch(
                    "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
                    return_value=interior_cap,
                ):
                    result = erase_text_bubble_regions(
                        original,
                        original.copy(),
                        source_mask,
                        [block],
                    )

                self.assertNotEqual(
                    block._erase_mode,
                    ERASE_MODE_BUBBLE_LAMA_FALLBACK,
                )
                self.assertGreater(
                    np.count_nonzero(result.edit_mask[orphan_glyph > 0]),
                    0,
                )
                self.assertTrue(
                    np.any(
                        result.image[orphan_glyph > 0]
                        != original[orphan_glyph > 0]
                    )
                )

    def test_text_prior_boundary_does_not_turn_orphan_glyph_into_structure(
        self,
    ) -> None:
        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        source_mask[40:46, 40:46] = 255
        original[source_mask > 0] = 245
        orphan_glyph = np.zeros((96, 96), dtype=np.uint8)
        orphan_glyph[52:55, 34:52] = 255
        original[orphan_glyph > 0] = 245
        block = _block(
            xyxy=[30, 30, 60, 60],
            bubble_xyxy=[24, 24, 72, 72],
        )
        interior_cap = np.zeros((48, 48), dtype=np.uint8)
        interior_cap[2:46, 2:46] = 255
        partial_prior = np.zeros((48, 48), dtype=np.uint8)
        partial_prior[2:40, 2:27] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ), mock.patch(
            "modules.utils.bubble_erase.build_text_prior_mask",
            return_value=partial_prior,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertNotEqual(block._erase_mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        self.assertTrue(
            np.all(result.edit_mask[orphan_glyph > 0] > 0)
        )

    def test_cropped_line_candidate_keeps_global_text_prior_availability(
        self,
    ) -> None:
        analysis = np.full((57, 188), 255, dtype=np.uint8)
        candidate = np.zeros_like(analysis)
        cv2.line(candidate, (119, 46), (128, 46), 255, thickness=1)
        text_prior = np.zeros_like(analysis)
        text_prior[4:14, 8:76] = 255

        self.assertTrue(
            _line_candidate_has_outside_text_support(
                candidate,
                text_prior,
                analysis,
                min_support=10,
            )
        )

        cropped_candidate = candidate[44:49, 117:131]
        cropped_analysis = analysis[44:49, 117:131]
        cropped_prior = text_prior[44:49, 117:131]
        self.assertFalse(np.any(cropped_prior))
        self.assertTrue(
            _line_candidate_has_outside_text_support(
                cropped_candidate,
                cropped_prior,
                cropped_analysis,
                min_support=10,
                text_prior_available=True,
            )
        )
        self.assertFalse(
            _line_candidate_has_outside_text_support(
                cropped_candidate,
                cropped_prior,
                cropped_analysis,
                min_support=10,
                text_prior_available=False,
            )
        )

    def test_line_art_crop_uses_remote_text_prior_availability(self) -> None:
        image = np.full((57, 188, 3), 150, dtype=np.uint8)
        cv2.line(image, (166, 29), (176, 31), (20, 20, 20), thickness=1)
        cap = np.full((57, 188), 255, dtype=np.uint8)
        source_seed = np.zeros((57, 188), dtype=np.uint8)
        remote_prior = np.zeros((57, 188), dtype=np.uint8)
        remote_prior[4:14, 8:76] = 255

        cv2.setRNGSeed(1)
        protected = _line_art_protect_mask(
            image,
            interior_cap=cap,
            source_seed_mask=source_seed,
            source_glyph_mask=source_seed,
            text_prior_mask=remote_prior,
        )
        cv2.setRNGSeed(1)
        empty_prior_protected = _line_art_protect_mask(
            image,
            interior_cap=cap,
            source_seed_mask=source_seed,
            source_glyph_mask=source_seed,
            text_prior_mask=np.zeros_like(remote_prior),
        )

        self.assertEqual(np.count_nonzero(protected), 220)
        self.assertEqual(np.count_nonzero(empty_prior_protected), 0)

    def test_missing_text_prior_fails_closed_for_full_edge_proposals(self) -> None:
        original = np.full((112, 112, 3), 150, dtype=np.uint8)
        source_mask = np.zeros((112, 112), dtype=np.uint8)
        visible_line = np.zeros((112, 112), dtype=np.uint8)
        cv2.line(visible_line, (44, 56), (68, 56), 255, thickness=3)
        cv2.line(visible_line, (48, 56), (48, 46), 255, thickness=3)
        original[visible_line > 0] = 20
        original[43:46, 55:58] = 20
        source_mask[50:62, 53:59] = 255
        original[source_mask > 0] = 245
        visible_line[source_mask > 0] = 0
        block = _block(
            xyxy=[0, 0, 0, 0],
            bubble_xyxy=[8, 8, 104, 104],
        )
        interior_cap = np.zeros((96, 96), dtype=np.uint8)
        interior_cap[2:94, 2:94] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        self.assertEqual(
            block._erase_skipped_reason,
            "text_prior_unavailable_structure_ambiguous",
        )
        self.assertEqual(result.stats["fallback_block_count"], 1)
        np.testing.assert_array_equal(result.fallback_mask, source_mask)
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[visible_line > 0]),
            0,
        )
        np.testing.assert_array_equal(
            result.image[visible_line > 0],
            original[visible_line > 0],
        )

    def test_missing_text_prior_keeps_smooth_bubble_on_local_fill_path(self) -> None:
        original = np.full((112, 112, 3), 150, dtype=np.uint8)
        source_mask = np.zeros((112, 112), dtype=np.uint8)
        source_mask[50:62, 53:59] = 255
        original[source_mask > 0] = 245
        block = _block(
            xyxy=[0, 0, 0, 0],
            bubble_xyxy=[8, 8, 104, 104],
        )
        interior_cap = np.zeros((96, 96), dtype=np.uint8)
        interior_cap[2:94, 2:94] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_FLAT_FILL)
        self.assertEqual(result.stats["fallback_block_count"], 0)
        self.assertGreater(mask_pixel_count(result.edit_mask), 0)

    def test_missing_text_prior_routes_rotated_structure_across_sizes(self) -> None:
        for roi_size, line_length, angle in (
            (48, 24, 20),
            (128, 40, 30),
        ):
            with self.subTest(
                roi_size=roi_size,
                line_length=line_length,
                angle=angle,
            ):
                margin = 8
                canvas_size = roi_size + (2 * margin)
                original = np.full(
                    (canvas_size, canvas_size, 3),
                    150,
                    dtype=np.uint8,
                )
                center = np.asarray(
                    [margin + (roi_size // 2)] * 2,
                    dtype=np.float64,
                )
                direction = np.asarray(
                    [
                        math.cos(math.radians(angle)),
                        math.sin(math.radians(angle)),
                    ],
                    dtype=np.float64,
                )
                half_line = direction * (line_length / 2.0)
                start = tuple(np.round(center - half_line).astype(np.int32))
                end = tuple(np.round(center + half_line).astype(np.int32))
                line_mask = np.zeros(
                    (canvas_size, canvas_size),
                    dtype=np.uint8,
                )
                cv2.line(line_mask, start, end, 255, thickness=3)
                original[line_mask > 0] = 20
                source_mask = np.zeros_like(line_mask)
                center_x, center_y = np.round(center).astype(np.int32)
                source_mask[
                    center_y - 3:center_y + 3,
                    center_x - 3:center_x + 3,
                ] = 255
                original[source_mask > 0] = 245
                visible_line = np.where(
                    (line_mask > 0) & (source_mask <= 0),
                    255,
                    0,
                ).astype(np.uint8)
                block = _block(
                    xyxy=[0, 0, 0, 0],
                    bubble_xyxy=[
                        margin,
                        margin,
                        margin + roi_size,
                        margin + roi_size,
                    ],
                )
                interior_cap = np.zeros(
                    (roi_size, roi_size),
                    dtype=np.uint8,
                )
                interior_cap[2:-2, 2:-2] = 255

                with mock.patch(
                    "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
                    return_value=interior_cap,
                ):
                    result = erase_text_bubble_regions(
                        original,
                        original.copy(),
                        source_mask,
                        [block],
                    )

                self.assertEqual(
                    block._erase_skipped_reason,
                    "text_prior_unavailable_structure_ambiguous",
                )
                np.testing.assert_array_equal(
                    result.fallback_mask,
                    source_mask,
                )
                self.assertEqual(
                    np.count_nonzero(
                        result.fallback_mask[visible_line > 0]
                    ),
                    0,
                )

    def test_interior_lines_route_to_lama_across_bubble_sizes(self) -> None:
        for roi_size, line_length in ((24, 20), (100, 18), (100, 40)):
            with self.subTest(roi_size=roi_size, line_length=line_length):
                margin = 8
                canvas_size = roi_size + (margin * 2)
                original = np.full(
                    (canvas_size, canvas_size, 3),
                    150,
                    dtype=np.uint8,
                )
                center_y = margin + (roi_size // 2)
                line_x1 = margin + ((roi_size - line_length) // 2)
                line_x2 = line_x1 + line_length
                original[center_y - 1:center_y + 2, line_x1:line_x2] = 20

                glyph_x = max(margin + 2, line_x1 - 1)
                glyph_y = center_y - 6
                original[glyph_y:glyph_y + 4, glyph_x:glyph_x + 4] = 245
                source_mask = np.zeros(
                    (canvas_size, canvas_size),
                    dtype=np.uint8,
                )
                source_mask[glyph_y:glyph_y + 4, glyph_x:glyph_x + 4] = 255
                block = _block(
                    xyxy=[glyph_x - 2, glyph_y - 2, glyph_x + 8, glyph_y + 8],
                    bubble_xyxy=[
                        margin,
                        margin,
                        margin + roi_size,
                        margin + roi_size,
                    ],
                )
                interior_cap = np.zeros((roi_size, roi_size), dtype=np.uint8)
                interior_cap[1:-1, 1:-1] = 255

                with mock.patch(
                    "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
                    return_value=interior_cap,
                ):
                    result = erase_text_bubble_regions(
                        original,
                        original.copy(),
                        source_mask,
                        [block],
                    )

                self.assertEqual(
                    block._erase_mode,
                    ERASE_MODE_BUBBLE_LAMA_FALLBACK,
                )
                self.assertEqual(block._erase_skipped_reason, "line_art_intrusion")
                self.assertEqual(result.stats["fallback_block_count"], 1)
                self.assertGreater(mask_pixel_count(result.fallback_mask), 0)
                self.assertEqual(
                    np.count_nonzero(
                        result.fallback_mask[
                            center_y - 1:center_y + 2,
                            line_x1:line_x2,
                        ]
                    ),
                    0,
                )
                self.assertTrue(
                    np.array_equal(
                        result.image[
                            center_y - 1:center_y + 2,
                            line_x1:line_x2,
                        ],
                        original[
                            center_y - 1:center_y + 2,
                            line_x1:line_x2,
                        ],
                    )
                )

    def test_short_interior_line_routing_has_no_adjacent_glyph_length_hole(
        self,
    ) -> None:
        for line_length in (8, 9, 10):
            with self.subTest(line_length=line_length):
                original = np.full((40, 40, 3), 150, dtype=np.uint8)
                original[20:23, 10:10 + line_length] = 20
                original[14:18, 20:24] = 245
                source_mask = np.zeros((40, 40), dtype=np.uint8)
                source_mask[14:18, 20:24] = 255
                block = _block(
                    xyxy=[18, 12, 28, 22],
                    bubble_xyxy=[8, 8, 32, 32],
                )
                interior_cap = np.zeros((24, 24), dtype=np.uint8)
                interior_cap[1:23, 1:23] = 255

                with mock.patch(
                    "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
                    return_value=interior_cap,
                ):
                    result = erase_text_bubble_regions(
                        original,
                        original.copy(),
                        source_mask,
                        [block],
                    )

                self.assertEqual(
                    block._erase_mode,
                    ERASE_MODE_BUBBLE_LAMA_FALLBACK,
                )
                self.assertEqual(block._erase_skipped_reason, "line_art_intrusion")
                self.assertEqual(result.stats["fallback_block_count"], 1)
                self.assertEqual(
                    np.count_nonzero(
                        result.fallback_mask[20:23, 10:10 + line_length]
                    ),
                    0,
                )

    def test_long_source_glyph_is_not_mistaken_for_interior_line(self) -> None:
        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        original[24:66, 44:50] = 245
        source_mask[24:66, 44:50] = 255
        original[24:30, 35:59] = 245
        source_mask[24:30, 35:59] = 255
        block = _block(
            xyxy=[32, 20, 62, 70],
            bubble_xyxy=[8, 8, 88, 88],
        )
        interior_cap = np.zeros((80, 80), dtype=np.uint8)
        interior_cap[2:78, 2:78] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertNotEqual(block._erase_mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        self.assertEqual(result.stats["fallback_block_count"], 0)
        self.assertGreater(mask_pixel_count(result.edit_mask), 0)
        self.assertLess(float(np.mean(result.image[source_mask > 0])), 180.0)

    def test_sparse_halftone_dots_are_not_included_in_the_edit_mask(self) -> None:
        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        texture_mask = np.zeros((96, 96), dtype=np.uint8)
        for y in range(18, 79, 7):
            for x in range(18, 79, 7):
                texture_mask[y:y + 2, x:x + 2] = 255
        original[texture_mask > 0] = 85
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        original[42:48, 42:48] = 245
        source_mask[42:48, 42:48] = 255
        texture_mask[source_mask > 0] = 0
        block = _block(
            xyxy=[38, 38, 52, 52],
            bubble_xyxy=[8, 8, 88, 88],
        )
        interior_cap = np.zeros((80, 80), dtype=np.uint8)
        interior_cap[2:78, 2:78] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        self.assertEqual(block._erase_skipped_reason, "microtexture_intrusion")
        self.assertEqual(result.stats["fallback_block_count"], 1)
        np.testing.assert_array_equal(result.fallback_mask, source_mask)
        self.assertEqual(mask_pixel_count(result.edit_mask), 0)
        self.assertEqual(
            np.count_nonzero(result.fallback_mask[texture_mask > 0]),
            0,
        )

    def test_dense_halftone_fallback_matches_the_source_seed(self) -> None:
        for dot_size, pitch in ((3, 6), (4, 8)):
            with self.subTest(dot_size=dot_size, pitch=pitch):
                original = np.full((96, 96, 3), 150, dtype=np.uint8)
                texture_mask = np.zeros((96, 96), dtype=np.uint8)
                for y in range(16, 82, pitch):
                    for x in range(14, 82, pitch):
                        texture_mask[
                            y:y + dot_size,
                            x:x + dot_size,
                        ] = 255
                original[texture_mask > 0] = 85
                source_mask = np.zeros((96, 96), dtype=np.uint8)
                source_mask[42:48, 42:48] = 255
                original[source_mask > 0] = 245
                texture_mask[source_mask > 0] = 0
                block = _block(
                    xyxy=[38, 38, 52, 52],
                    bubble_xyxy=[8, 8, 88, 88],
                )
                interior_cap = np.zeros((80, 80), dtype=np.uint8)
                interior_cap[2:78, 2:78] = 255

                with mock.patch(
                    "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
                    return_value=interior_cap,
                ):
                    result = erase_text_bubble_regions(
                        original,
                        original.copy(),
                        source_mask,
                        [block],
                    )

                self.assertEqual(
                    block._erase_mode,
                    ERASE_MODE_BUBBLE_LAMA_FALLBACK,
                )
                self.assertEqual(
                    block._erase_skipped_reason,
                    "microtexture_intrusion",
                )
                np.testing.assert_array_equal(
                    result.fallback_mask,
                    source_mask,
                )
                self.assertEqual(mask_pixel_count(result.edit_mask), 0)
                self.assertEqual(
                    np.count_nonzero(
                        result.fallback_mask[texture_mask > 0]
                    ),
                    0,
                )

    def test_dense_texture_short_circuits_expensive_line_art_analysis(
        self,
    ) -> None:
        crop = np.full((256, 256, 3), 150, dtype=np.uint8)
        for y in range(12, 244, 6):
            for x in range(12, 244, 6):
                crop[y:y + 3, x:x + 3] = 85
        source_seed = np.zeros((256, 256), dtype=np.uint8)
        source_seed[124:132, 124:132] = 255
        crop[source_seed > 0] = 245
        interior_cap = np.zeros((256, 256), dtype=np.uint8)
        interior_cap[2:254, 2:254] = 255

        with (
            mock.patch(
                "modules.utils.bubble_erase._validated_bubble_interior_cap_mask",
                return_value=interior_cap,
            ),
            mock.patch(
                "modules.utils.bubble_erase._line_art_protect_mask",
                side_effect=AssertionError(
                    "dense texture must bypass expensive line-art analysis"
                ),
            ),
        ):
            context = _build_bubble_line_art_context(
                crop,
                source_seed,
            )

        self.assertTrue(context.texture_field_detected)
        self.assertEqual(mask_pixel_count(context.line_protect_mask), 0)

    def test_mixed_dot_size_halftone_does_not_expand_the_edit_mask(self) -> None:
        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        texture_mask = np.zeros((96, 96), dtype=np.uint8)
        for row, y in enumerate(range(15, 82, 11)):
            for column, x in enumerate(range(14, 82, 11)):
                dot_size = 3 if (row + column) % 2 == 0 else 7
                texture_mask[
                    y:y + dot_size,
                    x:x + dot_size,
                ] = 255
        original[texture_mask > 0] = 85
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        source_mask[42:48, 42:48] = 255
        original[source_mask > 0] = 245
        texture_mask[source_mask > 0] = 0
        block = _block(
            xyxy=[38, 38, 52, 52],
            bubble_xyxy=[8, 8, 88, 88],
        )
        interior_cap = np.zeros((80, 80), dtype=np.uint8)
        interior_cap[2:78, 2:78] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        self.assertEqual(
            np.count_nonzero(
                np.where(
                    (result.edit_mask > 0) | (result.fallback_mask > 0),
                    255,
                    0,
                )[texture_mask > 0]
            ),
            0,
        )

    def test_rotated_and_staggered_halftone_do_not_expand_the_edit_mask(self) -> None:
        texture_cases: list[tuple[str, np.ndarray]] = []

        rotated = np.zeros((96, 96), dtype=np.uint8)
        angle = math.radians(5.0)
        for grid_y in range(-4, 5):
            for grid_x in range(-4, 5):
                x = round(
                    48
                    + (grid_x * 7 * math.cos(angle))
                    - (grid_y * 7 * math.sin(angle))
                )
                y = round(
                    48
                    + (grid_x * 7 * math.sin(angle))
                    + (grid_y * 7 * math.cos(angle))
                )
                if 10 <= x < 83 and 10 <= y < 83:
                    rotated[y:y + 3, x:x + 3] = 255
        texture_cases.append(("rotated", rotated))

        staggered = np.zeros((96, 96), dtype=np.uint8)
        for row, y in enumerate(range(16, 82, 7)):
            offset = (row % 3) * 2
            for x in range(14 + offset, 82, 7):
                staggered[y:y + 3, x:x + 3] = 255
        texture_cases.append(("staggered", staggered))

        for name, texture_mask in texture_cases:
            with self.subTest(name=name):
                original = np.full((96, 96, 3), 150, dtype=np.uint8)
                original[texture_mask > 0] = 85
                source_mask = np.zeros((96, 96), dtype=np.uint8)
                source_mask[42:48, 42:48] = 255
                original[source_mask > 0] = 245
                texture_mask[source_mask > 0] = 0
                block = _block(
                    xyxy=[38, 38, 52, 52],
                    bubble_xyxy=[8, 8, 88, 88],
                )
                interior_cap = np.zeros((80, 80), dtype=np.uint8)
                interior_cap[2:78, 2:78] = 255

                with mock.patch(
                    "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
                    return_value=interior_cap,
                ):
                    result = erase_text_bubble_regions(
                        original,
                        original.copy(),
                        source_mask,
                        [block],
                    )

                union_mask = np.where(
                    (result.edit_mask > 0) | (result.fallback_mask > 0),
                    255,
                    0,
                ).astype(np.uint8)
                self.assertEqual(
                    np.count_nonzero(union_mask[texture_mask > 0]),
                    0,
                )

    def test_true_lattice_cell_inside_text_prior_remains_protected(self) -> None:
        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        texture_mask = np.zeros((96, 96), dtype=np.uint8)
        for y in range(16, 81, 8):
            for x in range(14, 79, 8):
                texture_mask[y:y + 3, x:x + 3] = 255
        original[texture_mask > 0] = 85
        protected_cell = np.zeros((96, 96), dtype=np.uint8)
        protected_cell[56:59, 54:57] = 255
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        source_mask[42:48, 42:48] = 255
        original[source_mask > 0] = 245
        texture_mask[source_mask > 0] = 0
        block = _block(
            xyxy=[36, 36, 65, 65],
            bubble_xyxy=[8, 8, 88, 88],
        )
        interior_cap = np.zeros((80, 80), dtype=np.uint8)
        interior_cap[2:78, 2:78] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        union_mask = np.where(
            (result.edit_mask > 0) | (result.fallback_mask > 0),
            255,
            0,
        ).astype(np.uint8)
        self.assertEqual(
            np.count_nonzero(union_mask[protected_cell > 0]),
            0,
        )
        self.assertTrue(np.all(union_mask[source_mask > 0] > 0))

    def test_dense_source_rescue_never_reopens_protected_interior_line(self) -> None:
        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        line_mask = np.zeros((96, 96), dtype=np.uint8)
        cv2.line(line_mask, (10, 43), (86, 43), 255, thickness=3)
        original[line_mask > 0] = 20
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        source_mask[22:64, 28:68] = 255
        block = _block(
            xyxy=[28, 22, 68, 64],
            bubble_xyxy=[8, 8, 88, 88],
        )
        interior_cap = np.zeros((80, 80), dtype=np.uint8)
        interior_cap[2:78, 2:78] = 255

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=interior_cap,
        ):
            result = erase_text_bubble_regions(
                original,
                original.copy(),
                source_mask,
                [block],
            )

        union_mask = np.where(
            (result.edit_mask > 0) | (result.fallback_mask > 0),
            255,
            0,
        ).astype(np.uint8)
        self.assertEqual(np.count_nonzero(union_mask[line_mask > 0]), 0)
        np.testing.assert_array_equal(result.image[line_mask > 0], original[line_mask > 0])

    def test_line_art_bubble_defers_to_lama_fallback_without_flattening(self) -> None:
        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        original[46:49, 8:88] = 20
        original[12:84, 68:71] = 30
        original[30:52, 32:38] = 245
        original[30:52, 48:54] = 245
        current = original.copy()
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        source_mask[30:52, 32:38] = 255
        source_mask[30:52, 48:54] = 255
        block = _block(xyxy=[26, 24, 60, 58], bubble_xyxy=[8, 8, 88, 88])

        result = erase_text_bubble_regions(original, current, source_mask, [block])

        self.assertTrue(result.stats["applied"])
        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        self.assertEqual(result.stats["fallback_block_count"], 1)
        self.assertEqual(mask_pixel_count(result.edit_mask), 0)
        self.assertGreater(mask_pixel_count(result.fallback_mask), 0)
        self.assertTrue(np.array_equal(result.image, current))

    def test_text_free_blocks_are_not_modified_by_bubble_erase(self) -> None:
        original = np.full((32, 32, 3), 128, dtype=np.uint8)
        current = original.copy()
        current[10:14, 10:14] = 64
        source_mask = np.zeros((32, 32), dtype=np.uint8)
        source_mask[10:14, 10:14] = 255
        block = _block(xyxy=[8, 8, 16, 16], text_class="text_free")

        result = erase_text_bubble_regions(original, current, source_mask, [block])

        self.assertFalse(result.stats["applied"])
        self.assertEqual(mask_pixel_count(result.edit_mask), 0)
        self.assertTrue(np.array_equal(result.image, current))
        self.assertEqual(block._erase_mode, "text_free_lama")


if __name__ == "__main__":
    unittest.main()
