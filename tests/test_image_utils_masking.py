from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from modules.utils.image_utils import generate_mask
from modules.utils.textblock import TextBlock


class ImageUtilsMaskingTests(unittest.TestCase):
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
                settings={"mask_refiner": "ctd", "keep_existing_lines": True},
                return_details=True,
            )

        self.assertEqual(details["mask_refiner"], "ctd")
        self.assertTrue(details["keep_existing_lines"])
        self.assertEqual(details["refiner_backend"], "torch")
        self.assertEqual(details["refiner_device"], "cuda")
        self.assertGreater(int(np.count_nonzero(details["protect_mask"])), 0)
        self.assertEqual(int(np.count_nonzero(details["final_mask"])), int(np.count_nonzero(base_mask)) - 4)

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
                settings={"mask_refiner": "legacy_bbox"},
                return_details=True,
            )

        legacy_builder.assert_called_once()
        self.assertIs(details, legacy_details)


if __name__ == "__main__":
    unittest.main()
