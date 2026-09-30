from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from modules.utils.image_utils import (
    _build_block_bubble_seed_crop,
    _build_candidate_window_mask,
    apply_spotting_source_glyph_extension,
    generate_mask,
    restore_original_for_block_masks,
    release_protected_mask_for_explicit_additions,
)
from modules.utils.textblock import TextBlock


def _spotting_line_block(*, text_class: str = "text_free") -> TextBlock:
    block = TextBlock(
        text_bbox=np.asarray([100, 70, 142, 102], dtype=np.int32),
        text_class=text_class,
        text="TEST",
        block_id="spotting-line-test",
        processing_action="translate_inpaint",
        ocr_strategy="paddle_spotting_full_page",
        ocr_geometry_provenance={
            "source": "paddle_spotting_normalized_quad",
            "detector_geometry_authoritative": True,
        },
        ocr_regions=[
            {
                "source_line": 1,
                "bbox_xyxy": [100, 70, 142, 102],
                "points": [[100, 70], [142, 70], [142, 102], [100, 102]],
            }
        ],
    )
    return block


class ImageUtilsMaskingTests(unittest.TestCase):
    def test_generate_mask_preserves_full_page_positive_claim_separately(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        block = TextBlock(
            text_bbox=np.array([4, 4, 12, 12]),
            text_class="text_bubble",
            detector_origin="direct_text",
            detector_text_bbox=[4, 4, 12, 12],
        )
        base_mask = np.zeros((32, 32), dtype=np.uint8)
        base_mask[6:8, 6:8] = 255
        full_page_raw = np.zeros_like(base_mask)
        full_page_raw[20:24, 20:24] = 255

        with (
            mock.patch("modules.utils.image_utils.CTDRefiner") as refiner_cls,
            mock.patch("modules.utils.image_utils.CTDPositiveClaimProvider") as provider_cls,
            mock.patch(
                "modules.utils.image_utils.build_protect_mask",
                return_value=np.zeros_like(base_mask),
            ),
        ):
            refiner_cls.return_value.refine.return_value = SimpleNamespace(
                raw_mask=base_mask.copy(),
                refined_mask=base_mask.copy(),
                final_mask=base_mask.copy(),
                backend="torch",
                device="cuda",
                fallback_used=False,
            )
            provider_cls.return_value.infer.return_value = SimpleNamespace(
                raw_mask=full_page_raw,
                providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
                detect_size=1280,
                model_sha256="model-sha",
                model_opset=12,
            )
            details = generate_mask(
                image,
                [block],
                settings={
                    "mask_refiner": "ctd",
                    "keep_existing_lines": False,
                    "final_mask_dilate_size": 0,
                },
                return_details=True,
            )

        self.assertTrue(np.array_equal(details["raw_mask"], base_mask))
        self.assertTrue(
            np.array_equal(details["positive_claim_raw_mask"], full_page_raw)
        )
        self.assertEqual(details["positive_claim_runtime"]["status"], "completed")
        self.assertEqual(details["positive_claim_runtime"]["model_opset"], 12)
        self.assertEqual(int(np.count_nonzero(details["final_mask"])), 4)

    def test_positive_claim_provider_failure_does_not_change_base_mask(self) -> None:
        image = np.zeros((24, 24, 3), dtype=np.uint8)
        block = TextBlock(
            text_bbox=np.array([4, 4, 12, 12]),
            text_class="text_bubble",
            detector_origin="direct_text",
            detector_text_bbox=[4, 4, 12, 12],
        )
        base_mask = np.zeros((24, 24), dtype=np.uint8)
        base_mask[6:8, 6:8] = 255

        with (
            mock.patch("modules.utils.image_utils.CTDRefiner") as refiner_cls,
            mock.patch(
                "modules.utils.image_utils.CTDPositiveClaimProvider",
                side_effect=RuntimeError("provider unavailable"),
            ),
            mock.patch(
                "modules.utils.image_utils.build_protect_mask",
                return_value=np.zeros_like(base_mask),
            ),
        ):
            refiner_cls.return_value.refine.return_value = SimpleNamespace(
                raw_mask=base_mask.copy(),
                refined_mask=base_mask.copy(),
                final_mask=base_mask.copy(),
                backend="torch",
                device="cuda",
                fallback_used=False,
            )
            details = generate_mask(
                image,
                [block],
                settings={
                    "mask_refiner": "ctd",
                    "keep_existing_lines": False,
                    "final_mask_dilate_size": 0,
                },
                return_details=True,
            )

        self.assertTrue(np.array_equal(details["final_mask"], base_mask))
        self.assertEqual(
            int(np.count_nonzero(details["positive_claim_raw_mask"])),
            0,
        )
        self.assertEqual(details["positive_claim_runtime"]["status"], "failed")

    def test_explicit_added_mask_releases_automatic_corner_protection(self) -> None:
        automatic = np.zeros((12, 12), dtype=np.uint8)
        automatic[5:7, 5:7] = 255
        merged = automatic.copy()
        merged[2:4, 2:4] = 255
        protected = np.zeros((12, 12), dtype=np.uint8)
        protected[1:5, 1:5] = 255

        updated, released = release_protected_mask_for_explicit_additions(
            protected,
            automatic,
            merged,
            (12, 12, 3),
        )

        self.assertEqual(released, 4)
        self.assertEqual(int(updated[2, 2]), 0)
        self.assertEqual(int(updated[1, 1]), 255)

    def test_generate_mask_applies_default_final_d08_dilation(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        base_mask = np.zeros((32, 32), dtype=np.uint8)
        base_mask[16, 16] = 255

        with (
            mock.patch("modules.utils.image_utils.CTDRefiner") as refiner_cls,
            mock.patch("modules.utils.image_utils.build_protect_mask", return_value=np.zeros((32, 32), dtype=np.uint8)),
        ):
            refiner_cls.return_value.refine.return_value = SimpleNamespace(
                raw_mask=base_mask.copy(),
                refined_mask=base_mask.copy(),
                final_mask=base_mask.copy(),
                backend="torch",
                device="cuda",
                fallback_used=False,
            )
            details = generate_mask(
                image,
                [],
                settings={"mask_refiner": "ctd", "keep_existing_lines": False},
                return_details=True,
            )

        self.assertEqual(details["final_mask_dilate_size"], 8)
        self.assertGreater(int(np.count_nonzero(details["final_mask"])), 1)
        self.assertEqual(int(details["final_mask"][16, 16]), 255)
        self.assertEqual(int(details["final_mask"][16, 24]), 255)

    def test_generate_mask_ctd_path_honors_protect_mask(self) -> None:
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        block = TextBlock(
            text_bbox=np.array([2, 2, 10, 10]),
            bubble_bbox=np.array([1, 1, 11, 11]),
            text_class="text_bubble",
        )
        base_mask = np.zeros((16, 16), dtype=np.uint8)
        base_mask[2:10, 2:10] = 255
        protect_mask = np.zeros((16, 16), dtype=np.uint8)
        protect_mask[2:4, 2:4] = 255

        with (
            mock.patch("modules.utils.image_utils.CTDRefiner") as refiner_cls,
            mock.patch("modules.utils.image_utils.build_protect_mask", return_value=protect_mask),
        ):
            refiner_cls.return_value.refine.return_value = SimpleNamespace(
                raw_mask=base_mask.copy(),
                refined_mask=base_mask.copy(),
                final_mask=base_mask.copy(),
                backend="torch",
                device="cuda",
                fallback_used=False,
            )
            details = generate_mask(
                image,
                [block],
                settings={"mask_refiner": "ctd", "keep_existing_lines": True, "final_mask_dilate_size": 0},
                return_details=True,
            )

        self.assertEqual(details["mask_refiner"], "ctd")
        self.assertTrue(details["keep_existing_lines"])
        self.assertEqual(details["refiner_backend"], "torch")
        self.assertEqual(details["refiner_device"], "cuda")
        self.assertGreater(int(np.count_nonzero(details["protect_mask"])), 0)
        self.assertEqual(int(np.count_nonzero(details["final_mask"])), int(np.count_nonzero(base_mask)) - 4)

    def test_final_dilation_cannot_reintroduce_protected_bubble_border_pixels(self) -> None:
        image = np.zeros((24, 24, 3), dtype=np.uint8)
        block = TextBlock(
            text_bbox=np.array([8, 8, 16, 16]),
            text_class="text_bubble",
        )
        base_mask = np.zeros((24, 24), dtype=np.uint8)
        base_mask[10, 10] = 255
        protect_mask = np.zeros_like(base_mask)
        protect_mask[10, 11] = 255

        with (
            mock.patch("modules.utils.image_utils.CTDRefiner") as refiner_cls,
            mock.patch(
                "modules.utils.image_utils.build_protect_mask",
                return_value=protect_mask,
            ),
        ):
            refiner_cls.return_value.refine.return_value = SimpleNamespace(
                raw_mask=base_mask.copy(),
                refined_mask=base_mask.copy(),
                final_mask=base_mask.copy(),
                backend="torch",
                device="cuda",
                fallback_used=False,
            )
            details = generate_mask(
                image,
                [block],
                settings={
                    "mask_refiner": "ctd",
                    "keep_existing_lines": True,
                    "final_mask_dilate_size": 2,
                },
                return_details=True,
            )

        final_mask = np.asarray(details["final_mask"]) > 0
        protected = np.asarray(details["protect_mask"]) > 0
        self.assertEqual(int(np.count_nonzero(final_mask & protected)), 0)
        self.assertEqual(
            int(details["mask_policy_protected_pixel_removed_count"]),
            1,
        )

    def test_fully_protected_ctd_claim_becomes_review_instead_of_edit_permission(self) -> None:
        image = np.zeros((24, 24, 3), dtype=np.uint8)
        block = TextBlock(
            text_bbox=np.array([8, 8, 16, 16]),
            text_class="text_bubble",
        )
        base_mask = np.zeros((24, 24), dtype=np.uint8)
        base_mask[10, 10] = 255
        protect_mask = base_mask.copy()

        with (
            mock.patch("modules.utils.image_utils.CTDRefiner") as refiner_cls,
            mock.patch(
                "modules.utils.image_utils.build_protect_mask",
                return_value=protect_mask,
            ),
        ):
            refiner_cls.return_value.refine.return_value = SimpleNamespace(
                raw_mask=base_mask.copy(),
                refined_mask=base_mask.copy(),
                final_mask=base_mask.copy(),
                backend="torch",
                device="cuda",
                fallback_used=False,
            )
            details = generate_mask(
                image,
                [block],
                settings={
                    "mask_refiner": "ctd",
                    "keep_existing_lines": True,
                    "final_mask_dilate_size": 0,
                },
                return_details=True,
            )

        self.assertEqual(int(np.count_nonzero(details["final_mask"])), 0)
        self.assertEqual(details["mask_decision"], "review")
        self.assertEqual(details["mask_candidate_source"], "none")
        self.assertEqual(details["mask_reject_reason"], "ctd_claim_fully_protected")

    def test_generate_mask_ctd_path_keeps_legacy_hard_box_as_window_only(self) -> None:
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        block = TextBlock(
            text_bbox=np.array([2, 2, 10, 10]),
            bubble_bbox=np.array([1, 1, 12, 12]),
            text_class="text_bubble",
        )
        ctd_mask = np.zeros((16, 16), dtype=np.uint8)
        ctd_mask[2:4, 2:4] = 255
        protect_mask = np.zeros((16, 16), dtype=np.uint8)
        protect_mask[5:6, 5:6] = 255
        rescue_mask = np.zeros((16, 16), dtype=np.uint8)
        rescue_mask[5:8, 5:8] = 255
        legacy_details = {
            "legacy_base_mask": np.zeros((16, 16), dtype=np.uint8),
            "hard_box_rescue_mask": rescue_mask,
            "hard_box_applied_count": 1,
            "hard_box_reason_totals": {"color_core_detected": 1},
            "legacy_base_mask_pixel_count": 0,
            "hard_box_rescue_mask_pixel_count": int(np.count_nonzero(rescue_mask)),
        }

        with (
            mock.patch("modules.utils.image_utils.CTDRefiner") as refiner_cls,
            mock.patch("modules.utils.image_utils.build_protect_mask", return_value=protect_mask),
            mock.patch(
                "modules.utils.image_utils.build_legacy_bbox_mask_details",
                return_value=legacy_details,
            ),
        ):
            refiner_cls.return_value.refine.return_value = SimpleNamespace(
                raw_mask=ctd_mask.copy(),
                refined_mask=ctd_mask.copy(),
                final_mask=ctd_mask.copy(),
                backend="torch",
                device="cuda",
                fallback_used=False,
            )
            details = generate_mask(
                image,
                [block],
                settings={"mask_refiner": "ctd", "keep_existing_lines": True, "final_mask_dilate_size": 0},
                return_details=True,
            )

        final_mask = details["final_mask"]
        self.assertFalse(details["hard_box_rescue_used"])
        self.assertNotIn("hard_box_rescue", details["refiner_backend"])
        self.assertEqual(details["legacy_bbox_role"], "window_only")
        self.assertTrue(details["legacy_bbox_direct_erase_disabled"])
        self.assertEqual(details["mask_candidate_source"], "ctd_raw_refined_final_or")
        self.assertEqual(details["mask_decision"], "accepted")
        self.assertEqual(int(final_mask[5, 5]), 0)
        self.assertEqual(int(final_mask[6, 6]), 0)
        self.assertEqual(int(final_mask[2, 2]), 255)

    def test_generate_mask_ctd_empty_does_not_promote_legacy_bbox_fallback_to_erase_mask(self) -> None:
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        block = TextBlock(
            text_bbox=np.array([2, 2, 10, 10]),
            bubble_bbox=np.array([1, 1, 12, 12]),
            text_class="text_bubble",
        )
        empty_mask = np.zeros((16, 16), dtype=np.uint8)
        legacy_mask = np.zeros((16, 16), dtype=np.uint8)
        legacy_mask[2:10, 2:10] = 255
        legacy_details = {
            "raw_mask": empty_mask.copy(),
            "refined_mask": empty_mask.copy(),
            "protect_mask": empty_mask.copy(),
            "final_mask_pre_expand": legacy_mask.copy(),
            "final_mask_post_expand": legacy_mask.copy(),
            "final_mask": legacy_mask.copy(),
            "legacy_base_mask": legacy_mask.copy(),
            "hard_box_rescue_mask": legacy_mask.copy(),
            "hard_box_applied_count": 1,
            "hard_box_reason_totals": {"legacy_window": 1},
            "legacy_base_mask_pixel_count": int(np.count_nonzero(legacy_mask)),
            "hard_box_rescue_mask_pixel_count": int(np.count_nonzero(legacy_mask)),
            "final_mask_pixel_count": int(np.count_nonzero(legacy_mask)),
        }

        with (
            mock.patch("modules.utils.image_utils.CTDRefiner") as refiner_cls,
            mock.patch("modules.utils.image_utils.build_protect_mask", return_value=empty_mask.copy()),
            mock.patch(
                "modules.utils.image_utils.build_legacy_bbox_mask_details",
                return_value=legacy_details,
            ),
        ):
            refiner_cls.return_value.refine.return_value = SimpleNamespace(
                raw_mask=empty_mask.copy(),
                refined_mask=empty_mask.copy(),
                final_mask=empty_mask.copy(),
                backend="torch",
                device="cuda",
                fallback_used=False,
            )
            details = generate_mask(
                image,
                [block],
                settings={"mask_refiner": "ctd", "keep_existing_lines": True, "final_mask_dilate_size": 0},
                return_details=True,
            )

        self.assertEqual(int(np.count_nonzero(details["final_mask"])), 0)
        self.assertEqual(details["legacy_bbox_role"], "window_only")
        self.assertTrue(details["legacy_bbox_direct_erase_disabled"])
        self.assertEqual(details["mask_candidate_source"], "none")
        self.assertEqual(details["mask_decision"], "review")
        self.assertEqual(details["mask_reject_reason"], "legacy_bbox_window_only_no_ctd_mask")
        self.assertNotIn("legacy_bbox_fallback", details["refiner_backend"])

    def test_generate_mask_ctd_exception_keeps_legacy_bbox_as_window_only(self) -> None:
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        block = TextBlock(
            text_bbox=np.array([2, 2, 10, 10]),
            bubble_bbox=np.array([1, 1, 12, 12]),
            text_class="text_bubble",
        )
        legacy_mask = np.zeros((16, 16), dtype=np.uint8)
        legacy_mask[2:10, 2:10] = 255
        legacy_details = {
            "raw_mask": np.zeros((16, 16), dtype=np.uint8),
            "refined_mask": np.zeros((16, 16), dtype=np.uint8),
            "protect_mask": np.zeros((16, 16), dtype=np.uint8),
            "final_mask_pre_expand": legacy_mask.copy(),
            "final_mask_post_expand": legacy_mask.copy(),
            "final_mask": legacy_mask.copy(),
            "legacy_base_mask": legacy_mask.copy(),
            "hard_box_rescue_mask": legacy_mask.copy(),
            "hard_box_applied_count": 1,
            "hard_box_reason_totals": {"legacy_window": 1},
            "legacy_base_mask_pixel_count": int(np.count_nonzero(legacy_mask)),
            "hard_box_rescue_mask_pixel_count": int(np.count_nonzero(legacy_mask)),
            "final_mask_pixel_count": int(np.count_nonzero(legacy_mask)),
        }

        with (
            mock.patch("modules.utils.image_utils.CTDRefiner") as refiner_cls,
            mock.patch(
                "modules.utils.image_utils.build_legacy_bbox_mask_details",
                return_value=legacy_details,
            ),
        ):
            refiner_cls.side_effect = RuntimeError("ctd unavailable")
            details = generate_mask(
                image,
                [block],
                settings={"mask_refiner": "ctd", "keep_existing_lines": True, "final_mask_dilate_size": 0},
                return_details=True,
            )

        self.assertEqual(int(np.count_nonzero(details["final_mask"])), 0)
        self.assertEqual(details["legacy_bbox_role"], "window_only")
        self.assertTrue(details["legacy_bbox_direct_erase_disabled"])
        self.assertEqual(details["mask_candidate_source"], "none")
        self.assertEqual(details["mask_decision"], "review")
        self.assertEqual(details["mask_reject_reason"], "ctd_exception_legacy_bbox_window_only")
        self.assertEqual(details["refiner_backend"], "ctd+legacy_bbox_exception_window_only")

    def test_generate_mask_clamps_dilated_ctd_mask_to_candidate_windows(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        block = TextBlock(
            text_bbox=np.array([10, 10, 14, 14]),
            bubble_bbox=np.array([8, 8, 24, 24]),
            text_class="text_bubble",
        )
        ctd_mask = np.zeros((32, 32), dtype=np.uint8)
        ctd_mask[8, 8] = 255

        with (
            mock.patch("modules.utils.image_utils.CTDRefiner") as refiner_cls,
            mock.patch("modules.utils.image_utils.build_protect_mask", return_value=np.zeros((32, 32), dtype=np.uint8)),
            mock.patch("modules.utils.image_utils.build_legacy_bbox_mask_details") as legacy_builder,
        ):
            legacy_builder.return_value = {
                "legacy_base_mask": np.zeros((32, 32), dtype=np.uint8),
                "hard_box_rescue_mask": np.zeros((32, 32), dtype=np.uint8),
                "hard_box_applied_count": 0,
                "hard_box_reason_totals": {},
                "legacy_base_mask_pixel_count": 0,
                "hard_box_rescue_mask_pixel_count": 0,
            }
            refiner_cls.return_value.refine.return_value = SimpleNamespace(
                raw_mask=ctd_mask.copy(),
                refined_mask=ctd_mask.copy(),
                final_mask=ctd_mask.copy(),
                backend="torch",
                device="cuda",
                fallback_used=False,
            )
            details = generate_mask(
                image,
                [block],
                settings={"mask_refiner": "ctd", "keep_existing_lines": True, "final_mask_dilate_size": 4},
                return_details=True,
            )

        final_mask = details["final_mask"]
        self.assertEqual(int(final_mask[8, 8]), 255)
        self.assertEqual(int(final_mask[7, 8]), 0)
        self.assertEqual(int(final_mask[8, 7]), 0)
        self.assertGreater(details["mask_policy_outside_bubble_removed_pixel_count"], 0)
        self.assertEqual(details["mask_policy_bubble_clamp_applied_count"], 1)

    def test_generate_mask_clamps_rounded_bubble_corners_to_detected_silhouette(self) -> None:
        image = np.full((120, 140, 3), 40, dtype=np.uint8)
        cv2.ellipse(image, (70, 60), (50, 38), 0, 0, 360, (245, 245, 245), -1)
        cv2.ellipse(image, (70, 60), (50, 38), 0, 0, 360, (10, 10, 10), 3)
        image[52:57, 50:90] = 10
        block = TextBlock(
            text_bbox=np.array([48, 50, 92, 60]),
            bubble_bbox=np.array([18, 20, 122, 100]),
            text_class="text_bubble",
        )
        raw_mask = np.zeros((120, 140), dtype=np.uint8)
        raw_mask[50:60, 48:92] = 255
        corner_mask = raw_mask.copy()
        corner_mask[20:30, 18:30] = 255
        corner_mask[20:30, 110:122] = 255
        corner_mask[90:100, 18:30] = 255
        corner_mask[90:100, 110:122] = 255

        with (
            mock.patch("modules.utils.image_utils.CTDRefiner") as refiner_cls,
            mock.patch(
                "modules.utils.image_utils.build_protect_mask",
                return_value=np.zeros((120, 140), dtype=np.uint8),
            ),
            mock.patch("modules.utils.image_utils.build_legacy_bbox_mask_details") as legacy_builder,
        ):
            legacy_builder.return_value = {
                "legacy_base_mask": np.zeros((120, 140), dtype=np.uint8),
                "hard_box_rescue_mask": np.zeros((120, 140), dtype=np.uint8),
                "hard_box_applied_count": 0,
                "hard_box_reason_totals": {},
                "legacy_base_mask_pixel_count": 0,
                "hard_box_rescue_mask_pixel_count": 0,
            }
            refiner_cls.return_value.refine.return_value = SimpleNamespace(
                raw_mask=raw_mask,
                refined_mask=corner_mask,
                final_mask=corner_mask,
                backend="torch",
                device="cuda",
                fallback_used=False,
            )
            details = generate_mask(
                image,
                [block],
                settings={
                    "mask_refiner": "ctd",
                    "keep_existing_lines": False,
                    "final_mask_dilate_size": 0,
                },
                return_details=True,
            )

        final_mask = details["final_mask"]
        protected = details["protected_corner_mask"]
        self.assertGreater(int(np.count_nonzero(protected)), 0)
        self.assertEqual(int(np.count_nonzero(final_mask & protected)), 0)
        self.assertGreater(int(np.count_nonzero(final_mask[50:60, 48:92])), 0)
        self.assertEqual(details["mask_policy_bubble_silhouette_applied_count"], 1)
        self.assertEqual(details["mask_policy_bubble_silhouette_fallback_count"], 0)

    def test_verified_bubble_cap_reduces_dilation_but_fallback_keeps_current_width(self) -> None:
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        block = TextBlock(
            text_bbox=np.array([30, 34, 70, 46]),
            bubble_bbox=np.array([10, 10, 90, 70]),
            text_class="text_bubble",
        )
        base_mask = np.zeros((80, 100), dtype=np.uint8)
        base_mask[36:44, 34:66] = 255
        empty_mask = np.zeros_like(base_mask)
        legacy_details = {
            "legacy_base_mask": empty_mask.copy(),
            "hard_box_rescue_mask": empty_mask.copy(),
            "hard_box_applied_count": 0,
            "hard_box_reason_totals": {},
            "legacy_base_mask_pixel_count": 0,
            "hard_box_rescue_mask_pixel_count": 0,
        }
        cap_crop = np.full((60, 80), 255, dtype=np.uint8)
        with (
            mock.patch("modules.utils.image_utils.CTDRefiner") as refiner_cls,
            mock.patch(
                "modules.utils.image_utils.build_protect_mask",
                return_value=empty_mask.copy(),
            ),
            mock.patch(
                "modules.utils.image_utils.build_legacy_bbox_mask_details",
                return_value=legacy_details,
            ),
            mock.patch(
                "modules.utils.image_utils.extract_bubble_interior_cap_crop",
                return_value=cap_crop,
            ),
        ):
            refiner_cls.return_value.refine.return_value = SimpleNamespace(
                raw_mask=base_mask.copy(),
                refined_mask=base_mask.copy(),
                final_mask=base_mask.copy(),
                backend="torch",
                device="cuda",
                fallback_used=False,
            )
            capped = generate_mask(
                image,
                [block],
                settings={
                    "mask_refiner": "ctd",
                    "keep_existing_lines": False,
                    "final_mask_dilate_size": 8,
                },
                return_details=True,
            )

        radius_four = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 9),
            (4, 4),
        )
        expected_capped = cv2.dilate(base_mask, radius_four, iterations=1)
        expected_capped = np.where(
            (expected_capped > 0) & (capped["bubble_interior_cap_mask"] > 0),
            255,
            0,
        ).astype(np.uint8)
        self.assertTrue(np.array_equal(capped["final_mask"], expected_capped))
        self.assertEqual(
            capped["mask_policy_verified_bubble_dilate_reduction_applied_count"],
            1,
        )
        self.assertGreater(
            capped["mask_policy_verified_bubble_dilate_reduced_pixel_count"],
            0,
        )
        self.assertEqual(
            int(np.count_nonzero((capped["final_mask"] > 0) & (capped["bubble_interior_cap_mask"] == 0))),
            0,
        )

        with (
            mock.patch("modules.utils.image_utils.CTDRefiner") as refiner_cls,
            mock.patch(
                "modules.utils.image_utils.build_protect_mask",
                return_value=empty_mask.copy(),
            ),
            mock.patch(
                "modules.utils.image_utils.build_legacy_bbox_mask_details",
                return_value=legacy_details,
            ),
            mock.patch(
                "modules.utils.image_utils.extract_bubble_interior_cap_crop",
                return_value=None,
            ),
        ):
            refiner_cls.return_value.refine.return_value = SimpleNamespace(
                raw_mask=base_mask.copy(),
                refined_mask=base_mask.copy(),
                final_mask=base_mask.copy(),
                backend="torch",
                device="cuda",
                fallback_used=False,
            )
            fallback = generate_mask(
                image,
                [block],
                settings={
                    "mask_refiner": "ctd",
                    "keep_existing_lines": False,
                    "final_mask_dilate_size": 8,
                },
                return_details=True,
            )

        radius_eight = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (17, 17),
            (8, 8),
        )
        expected_fallback = cv2.dilate(base_mask, radius_eight, iterations=1)
        x1, y1, x2, y2 = block.bubble_xyxy
        window = np.zeros_like(base_mask)
        window[y1:y2, x1:x2] = 255
        expected_fallback = np.where(
            (expected_fallback > 0) & (window > 0),
            255,
            0,
        ).astype(np.uint8)
        self.assertTrue(np.array_equal(fallback["final_mask"], expected_fallback))
        self.assertEqual(
            fallback["mask_policy_verified_bubble_dilate_reduction_applied_count"],
            0,
        )
        self.assertEqual(fallback["mask_policy_bubble_silhouette_fallback_count"], 1)

    def test_mixed_page_uses_narrow_dilation_only_for_verified_bubble_cap(self) -> None:
        image = np.zeros((100, 240, 3), dtype=np.uint8)
        verified_block = TextBlock(
            text_bbox=np.array([42, 39, 46, 43]),
            bubble_bbox=np.array([10, 10, 100, 70]),
            text_class="text_bubble",
        )
        fallback_block = TextBlock(
            text_bbox=np.array([154, 39, 158, 43]),
            bubble_bbox=np.array([120, 10, 210, 70]),
            text_class="text_bubble",
        )
        base_mask = np.zeros((100, 240), dtype=np.uint8)
        base_mask[39:43, 42:46] = 255
        base_mask[39:43, 154:158] = 255
        empty_mask = np.zeros_like(base_mask)
        legacy_details = {
            "legacy_base_mask": empty_mask.copy(),
            "hard_box_rescue_mask": empty_mask.copy(),
            "hard_box_applied_count": 0,
            "hard_box_reason_totals": {},
            "legacy_base_mask_pixel_count": 0,
            "hard_box_rescue_mask_pixel_count": 0,
        }
        cap_crop = np.zeros((60, 90), dtype=np.uint8)
        cv2.ellipse(cap_crop, (45, 30), (38, 24), 0, 0, 360, 255, -1)

        with (
            mock.patch("modules.utils.image_utils.CTDRefiner") as refiner_cls,
            mock.patch(
                "modules.utils.image_utils.build_protect_mask",
                return_value=empty_mask.copy(),
            ),
            mock.patch(
                "modules.utils.image_utils.build_legacy_bbox_mask_details",
                return_value=legacy_details,
            ),
            mock.patch(
                "modules.utils.image_utils.extract_bubble_interior_cap_crop",
                side_effect=[cap_crop, None],
            ),
        ):
            refiner_cls.return_value.refine.return_value = SimpleNamespace(
                raw_mask=base_mask.copy(),
                refined_mask=base_mask.copy(),
                final_mask=base_mask.copy(),
                backend="torch",
                device="cuda",
                fallback_used=False,
            )
            details = generate_mask(
                image,
                [verified_block, fallback_block],
                settings={
                    "mask_refiner": "ctd",
                    "keep_existing_lines": False,
                    "final_mask_dilate_size": 8,
                },
                return_details=True,
            )

        radius_four = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 9),
            (4, 4),
        )
        radius_eight = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (17, 17),
            (8, 8),
        )
        dilated_four = cv2.dilate(base_mask, radius_four, iterations=1)
        dilated_eight = cv2.dilate(base_mask, radius_eight, iterations=1)
        cap_window = np.zeros_like(base_mask)
        cap_window[10:70, 10:100] = cap_crop
        fallback_window = np.zeros_like(base_mask)
        fallback_window[10:70, 120:210] = 255
        expected_verified = np.where(
            (dilated_four > 0) & (cap_window > 0),
            255,
            0,
        ).astype(np.uint8)
        expected_fallback = np.where(
            (dilated_eight > 0) & (fallback_window > 0),
            255,
            0,
        ).astype(np.uint8)

        self.assertTrue(
            np.array_equal(
                details["final_mask"],
                expected_verified | expected_fallback,
            )
        )
        self.assertEqual(details["mask_policy_bubble_silhouette_applied_count"], 1)
        self.assertEqual(details["mask_policy_bubble_silhouette_fallback_count"], 1)
        self.assertEqual(
            details["mask_policy_verified_bubble_dilate_reduction_applied_count"],
            1,
        )

    def test_overlapping_bubble_cap_is_not_marked_as_a_protected_corner(self) -> None:
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        first = TextBlock(
            text_bbox=np.array([4, 4, 8, 8]),
            bubble_bbox=np.array([2, 2, 12, 12]),
            text_class="text_bubble",
        )
        second = TextBlock(
            text_bbox=np.array([10, 4, 14, 8]),
            bubble_bbox=np.array([8, 2, 18, 12]),
            text_class="text_bubble",
        )
        first_cap = np.zeros((10, 10), dtype=np.uint8)
        first_cap[2:8, 2:8] = 255
        second_cap = np.zeros((10, 10), dtype=np.uint8)
        second_cap[2:8, 1:8] = 255

        with mock.patch(
            "modules.utils.image_utils.extract_bubble_interior_cap_crop",
            side_effect=[first_cap, second_cap],
        ):
            _window, _count, _caps, protected, applied, fallback = (
                _build_candidate_window_mask(
                    image,
                    [first, second],
                    bubble_seed_mask=np.ones((20, 20), dtype=np.uint8),
                )
            )

        self.assertEqual(int(protected[5, 10]), 0)
        self.assertEqual(applied, 2)
        self.assertEqual(fallback, 0)

    def test_protected_corners_exclude_other_text_and_fallback_windows(self) -> None:
        image = np.zeros((24, 24, 3), dtype=np.uint8)
        rounded = TextBlock(
            text_bbox=np.array([4, 4, 8, 8]),
            bubble_bbox=np.array([2, 2, 14, 14]),
            text_class="text_bubble",
        )
        text_free = TextBlock(
            text_bbox=np.array([10, 4, 13, 8]),
            text_class="text_free",
        )
        text_free.ctd_roi_xyxy = [9, 3, 14, 9]
        fallback = TextBlock(
            text_bbox=np.array([5, 10, 9, 13]),
            bubble_bbox=np.array([3, 9, 11, 16]),
            text_class="text_bubble",
        )
        cap = np.zeros((12, 12), dtype=np.uint8)
        cap[2:8, 2:8] = 255

        with mock.patch(
            "modules.utils.image_utils.extract_bubble_interior_cap_crop",
            side_effect=[cap, None],
        ):
            _window, _count, _caps, protected, applied, fallback_count = (
                _build_candidate_window_mask(
                    image,
                    [rounded, text_free, fallback],
                    bubble_seed_mask=np.ones((24, 24), dtype=np.uint8),
                )
            )

        self.assertEqual(int(protected[5, 11]), 0)
        self.assertEqual(int(protected[11, 5]), 0)
        self.assertEqual(applied, 1)
        self.assertEqual(fallback_count, 1)

    def test_bubble_silhouette_seed_is_isolated_to_the_bubble_roi(self) -> None:
        block = TextBlock(
            text_bbox=np.array([5, 5, 9, 9]),
            bubble_bbox=np.array([2, 2, 22, 22]),
            text_class="text_bubble",
        )
        raw_mask = np.zeros((24, 24), dtype=np.uint8)
        raw_mask[6, 6] = 255
        raw_mask[20, 20] = 255
        raw_mask[23, 23] = 255

        bubble_roi, isolated = _build_block_bubble_seed_crop(
            raw_mask,
            block,
            (24, 24, 3),
        )

        self.assertEqual(bubble_roi, (2, 2, 22, 22))
        self.assertEqual(int(isolated[4, 4]), 255)
        self.assertEqual(int(isolated[18, 18]), 255)
        self.assertEqual(int(np.count_nonzero(isolated)), 2)

    def test_generate_mask_ctd_path_does_not_union_text_free_hard_box_rescue(self) -> None:
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        block = TextBlock(
            text_bbox=np.array([2, 2, 10, 10]),
            text_class="text_free",
        )
        ctd_mask = np.zeros((16, 16), dtype=np.uint8)
        ctd_mask[2:4, 2:4] = 255
        rescue_mask = np.zeros((16, 16), dtype=np.uint8)
        rescue_mask[5:12, 5:12] = 255
        legacy_details = {
            "legacy_base_mask": np.zeros((16, 16), dtype=np.uint8),
            "hard_box_rescue_mask": rescue_mask,
            "hard_box_applied_count": 1,
            "hard_box_reason_totals": {"color_core_detected": 1},
            "legacy_base_mask_pixel_count": 0,
            "hard_box_rescue_mask_pixel_count": int(np.count_nonzero(rescue_mask)),
        }

        with (
            mock.patch("modules.utils.image_utils.CTDRefiner") as refiner_cls,
            mock.patch("modules.utils.image_utils.build_protect_mask", return_value=np.zeros((16, 16), dtype=np.uint8)),
            mock.patch(
                "modules.utils.image_utils.build_legacy_bbox_mask_details",
                return_value=legacy_details,
            ) as legacy_builder,
        ):
            refiner_cls.return_value.refine.return_value = SimpleNamespace(
                raw_mask=ctd_mask.copy(),
                refined_mask=ctd_mask.copy(),
                final_mask=ctd_mask.copy(),
                backend="torch",
                device="cuda",
                fallback_used=False,
            )
            details = generate_mask(
                image,
                [block],
                settings={"mask_refiner": "ctd", "keep_existing_lines": True, "final_mask_dilate_size": 0},
                return_details=True,
            )

        legacy_builder.assert_not_called()
        final_mask = details["final_mask"]
        self.assertFalse(details["hard_box_rescue_used"])
        self.assertNotIn("hard_box_rescue", details["refiner_backend"])
        self.assertEqual(int(np.count_nonzero(final_mask)), int(np.count_nonzero(ctd_mask)))
        self.assertEqual(int(final_mask[6, 6]), 0)
        self.assertEqual(int(final_mask[2, 2]), 255)

    def test_generate_mask_text_free_uses_thin_dilation_instead_of_global_d8(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        block = TextBlock(
            text_bbox=np.array([12, 12, 20, 20]),
            text_class="text_free",
        )
        ctd_mask = np.zeros((32, 32), dtype=np.uint8)
        ctd_mask[16, 16] = 255

        with (
            mock.patch("modules.utils.image_utils.CTDRefiner") as refiner_cls,
            mock.patch("modules.utils.image_utils.build_protect_mask", return_value=np.zeros((32, 32), dtype=np.uint8)),
        ):
            refiner_cls.return_value.refine.return_value = SimpleNamespace(
                raw_mask=ctd_mask.copy(),
                refined_mask=ctd_mask.copy(),
                final_mask=ctd_mask.copy(),
                backend="torch",
                device="cuda",
                fallback_used=False,
            )
            details = generate_mask(
                image,
                [block],
                settings={
                    "mask_refiner": "ctd",
                    "keep_existing_lines": False,
                    "final_mask_dilate_size": 8,
                    "text_free_final_mask_dilate_size": 1,
                },
                return_details=True,
            )

        final_mask = details["final_mask"]
        self.assertEqual(int(final_mask[16, 16]), 255)
        self.assertEqual(int(final_mask[16, 17]), 255)
        self.assertEqual(int(final_mask[16, 24]), 0)
        self.assertEqual(details["mask_policy_text_free_glyph_applied_count"], 1)
        self.assertGreater(block.block_final_mask_pixel_count, 0)
        self.assertEqual(block.block_mask_source, "ctd_raw_refined_final_or")
        self.assertEqual(block.block_mask_decision, "accepted")

    def test_generate_mask_ctd_or_rescues_raw_mask_when_final_is_empty(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        block = TextBlock(
            text_bbox=np.array([10, 8, 18, 24]),
            text_class="text_free",
        )
        raw_mask = np.zeros((32, 32), dtype=np.uint8)
        raw_mask[9:23, 11:17] = 255
        empty_mask = np.zeros((32, 32), dtype=np.uint8)

        with (
            mock.patch("modules.utils.image_utils.CTDRefiner") as refiner_cls,
            mock.patch("modules.utils.image_utils.build_protect_mask", return_value=empty_mask.copy()),
        ):
            refiner_cls.return_value.refine.return_value = SimpleNamespace(
                raw_mask=raw_mask.copy(),
                refined_mask=empty_mask.copy(),
                final_mask=empty_mask.copy(),
                backend="torch",
                device="cuda",
                fallback_used=False,
            )
            details = generate_mask(
                image,
                [block],
                settings={"mask_refiner": "ctd", "keep_existing_lines": False, "final_mask_dilate_size": 0},
                return_details=True,
            )

        self.assertEqual(int(np.count_nonzero(details["final_mask"])), int(np.count_nonzero(raw_mask)))
        self.assertEqual(details["mask_candidate_source"], "ctd_raw_refined_final_or")
        self.assertEqual(block.block_mask_source, "ctd_raw_refined_final_or")
        self.assertGreater(block.block_final_mask_pixel_count, 0)
        self.assertEqual(block.block_mask_decision, "accepted")

    def test_restore_original_for_block_masks_restores_skipped_mask_pixels(self) -> None:
        original = np.full((12, 12, 3), 240, dtype=np.uint8)
        original[4:8, 4:8] = [32, 64, 96]
        cleaned = original.copy()
        cleaned[4:8, 4:8] = [250, 250, 250]
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[4:8, 4:8] = 255
        block = TextBlock(
            text_bbox=np.array([4, 4, 8, 8]),
            text_class="text_free",
        )

        restored, updated_mask, stats = restore_original_for_block_masks(
            original,
            cleaned,
            mask,
            [block],
        )

        self.assertTrue(stats["applied"])
        self.assertEqual(stats["block_count"], 1)
        self.assertEqual(stats["pixel_count"], 16)
        self.assertTrue(np.array_equal(restored[4:8, 4:8], original[4:8, 4:8]))
        self.assertEqual(int(np.count_nonzero(updated_mask)), 0)
        self.assertTrue(block._render_restore_applied)

    def test_generate_mask_legacy_mode_still_uses_legacy_builder(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        legacy_details = {
            "raw_mask": np.zeros((8, 8), dtype=np.uint8),
            "refined_mask": np.zeros((8, 8), dtype=np.uint8),
            "protect_mask": np.zeros((8, 8), dtype=np.uint8),
            "final_mask_pre_expand": np.zeros((8, 8), dtype=np.uint8),
            "final_mask_post_expand": np.zeros((8, 8), dtype=np.uint8),
            "final_mask": np.zeros((8, 8), dtype=np.uint8),
            "legacy_base_mask": np.zeros((8, 8), dtype=np.uint8),
            "hard_box_rescue_mask": np.zeros((8, 8), dtype=np.uint8),
            "hard_box_applied_count": 0,
            "hard_box_reason_totals": {},
            "legacy_base_mask_pixel_count": 0,
            "hard_box_rescue_mask_pixel_count": 0,
            "final_mask_pixel_count": 0,
            "mask_refiner": "legacy_bbox",
            "keep_existing_lines": False,
            "refiner_backend": "legacy_bbox_rescue",
            "refiner_device": "cpu",
            "fallback_used": False,
            "mask_inpaint_mode": "rtdetr_legacy_bbox_source_lama",
        }
        with mock.patch(
            "modules.utils.image_utils.build_legacy_bbox_mask_details",
            return_value=legacy_details,
        ) as legacy_builder:
            details = generate_mask(
                image,
                [],
                settings={"mask_refiner": "legacy_bbox", "final_mask_dilate_size": 0},
                return_details=True,
            )

        legacy_builder.assert_called_once()
        self.assertIs(details, legacy_details)

    def test_spotting_source_glyph_extension_is_engine_gated_and_protected(self) -> None:
        image = np.zeros((140, 240, 3), dtype=np.uint8)
        cv2.putText(
            image,
            "TEST",
            (80, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        base = np.zeros(image.shape[:2], dtype=np.uint8)
        base[70:102, 100:142] = 255
        protect = np.zeros_like(base)
        protect[75:98, 100:104] = 255
        details = {
            "mask_refiner": "ctd",
            "mask_candidate_source": "ctd_refined",
            "final_mask": base.copy(),
            "protect_mask": protect,
            "protected_corner_mask": np.zeros_like(base),
            "bubble_interior_cap_mask": np.zeros_like(base),
        }
        block = _spotting_line_block()

        result = apply_spotting_source_glyph_extension(image, [block], details)

        final_mask = details["final_mask"]
        addition = (final_mask > 0) & (base == 0)
        self.assertGreater(int(np.count_nonzero(addition)), 0)
        self.assertEqual(int(np.count_nonzero(addition & (protect > 0))), 0)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(int(np.count_nonzero(final_mask)), details["final_mask_pixel_count"])

        crop_only = _spotting_line_block()
        crop_only.ocr_strategy = "paddle_crop"
        crop_details = {**details, "final_mask": base.copy()}
        crop_result = apply_spotting_source_glyph_extension(
            image, [crop_only], crop_details
        )
        self.assertTrue(np.array_equal(crop_details["final_mask"], base))
        self.assertEqual(crop_result["status"], "no_source_supported_additions")

    def test_spotting_source_glyph_extension_recovers_foreground_inside_native_line_edge(self) -> None:
        image = np.zeros((140, 200, 3), dtype=np.uint8)
        cv2.putText(
            image,
            "TEST",
            (62, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        base = np.zeros(image.shape[:2], dtype=np.uint8)
        base[52:88, 84:132] = 255
        block = _spotting_line_block()
        block.text_bbox = np.asarray([84, 52, 132, 88], dtype=np.int32)
        block.ocr_regions = [
            {
                "source_line": 1,
                "bbox_xyxy": [60, 50, 132, 90],
                "points": [[60, 50], [132, 50], [132, 90], [60, 90]],
            }
        ]
        details = {
            "mask_refiner": "ctd",
            "mask_candidate_source": "ctd_refined",
            "final_mask": base.copy(),
            "protect_mask": np.zeros_like(base),
            "protected_corner_mask": np.zeros_like(base),
            "bubble_interior_cap_mask": np.zeros_like(base),
        }

        result = apply_spotting_source_glyph_extension(image, [block], details)

        source_ink = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) >= 128
        final_mask = details["final_mask"] > 0
        line_roi = np.zeros(source_ink.shape, dtype=np.uint8)
        cv2.fillPoly(line_roi, [np.asarray(block.ocr_regions[0]["points"], dtype=np.int32)], 1)
        self.assertGreater(result["inside_line_added_pixels"], 0)
        self.assertEqual(int(np.count_nonzero(source_ink & (line_roi > 0) & ~final_mask)), 0)
        self.assertEqual(int(np.count_nonzero(final_mask & (base == 0) & (details["protect_mask"] > 0))), 0)

    def test_spotting_source_glyph_extension_stays_inside_verified_bubble_cap(self) -> None:
        image = np.zeros((140, 240, 3), dtype=np.uint8)
        cv2.putText(
            image,
            "TEST",
            (80, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        base = np.zeros(image.shape[:2], dtype=np.uint8)
        base[70:102, 100:142] = 255
        cap = np.zeros_like(base)
        cap[60:110, 70:108] = 255
        details = {
            "mask_refiner": "ctd",
            "mask_candidate_source": "ctd_refined",
            "final_mask": base.copy(),
            "protect_mask": np.zeros_like(base),
            "protected_corner_mask": np.zeros_like(base),
            "bubble_interior_cap_mask": cap,
        }
        block = _spotting_line_block(text_class="text_bubble")

        result = apply_spotting_source_glyph_extension(image, [block], details)

        addition = (details["final_mask"] > 0) & (base == 0)
        self.assertEqual(int(np.count_nonzero(addition & (cap == 0))), 0)
        self.assertEqual(result["status"], "completed")

    def test_generate_mask_applies_source_extension_only_for_native_spotting(self) -> None:
        image = np.zeros((140, 240, 3), dtype=np.uint8)
        cv2.putText(
            image,
            "TEST",
            (80, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        base_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        base_mask[70:102, 100:142] = 255
        block = _spotting_line_block()

        with (
            mock.patch("modules.utils.image_utils.CTDRefiner") as refiner_cls,
            mock.patch("modules.utils.image_utils.CTDPositiveClaimProvider") as provider_cls,
            mock.patch(
                "modules.utils.image_utils.build_protect_mask",
                return_value=np.zeros_like(base_mask),
            ),
        ):
            refiner_cls.return_value.refine.return_value = SimpleNamespace(
                raw_mask=base_mask.copy(),
                refined_mask=base_mask.copy(),
                final_mask=base_mask.copy(),
                backend="torch",
                device="cuda",
                fallback_used=False,
            )
            provider_cls.return_value.infer.return_value = SimpleNamespace(
                raw_mask=np.zeros_like(base_mask),
                providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
                detect_size=1280,
                model_sha256="model-sha",
                model_opset=12,
            )
            details = generate_mask(
                image,
                [block],
                settings={
                    "mask_refiner": "ctd",
                    "keep_existing_lines": False,
                    "final_mask_dilate_size": 0,
                },
                return_details=True,
            )

        self.assertEqual(details["source_glyph_extension"]["status"], "completed")
        self.assertGreater(details["source_glyph_extension"]["added_pixels"], 0)
        self.assertEqual(
            int(np.count_nonzero((details["final_mask"] > 0) & (details["protect_mask"] > 0))),
            0,
        )
        block.ocr_strategy = "paddle_crop"
        with (
            mock.patch("modules.utils.image_utils.CTDRefiner") as refiner_cls,
            mock.patch("modules.utils.image_utils.CTDPositiveClaimProvider") as provider_cls,
            mock.patch(
                "modules.utils.image_utils.build_protect_mask",
                return_value=np.zeros_like(base_mask),
            ),
        ):
            refiner_cls.return_value.refine.return_value = SimpleNamespace(
                raw_mask=base_mask.copy(),
                refined_mask=base_mask.copy(),
                final_mask=base_mask.copy(),
                backend="torch",
                device="cuda",
                fallback_used=False,
            )
            provider_cls.return_value.infer.return_value = SimpleNamespace(
                raw_mask=np.zeros_like(base_mask),
                providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
                detect_size=1280,
                model_sha256="model-sha",
                model_opset=12,
            )
            core_details = generate_mask(
                image,
                [block],
                settings={
                    "mask_refiner": "ctd",
                    "keep_existing_lines": False,
                    "final_mask_dilate_size": 0,
                },
                return_details=True,
            )

        self.assertEqual(
            core_details["source_glyph_extension"]["status"],
            "no_source_supported_additions",
        )
        self.assertTrue(np.array_equal(core_details["final_mask"], base_mask))


if __name__ == "__main__":
    unittest.main()
