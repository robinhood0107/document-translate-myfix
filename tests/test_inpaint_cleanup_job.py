from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imkit as imk  # noqa: E402

from modules.utils.inpaint_cleanup import (  # noqa: E402
    apply_duplicate_bubble_inner_fill,
)
from modules.utils.inpaint_composite import (  # noqa: E402
    composite_with_edit_mask,
    count_changed_outside_edit_mask,
)
from pipeline.inpaint_cleanup_job import (  # noqa: E402
    InpaintCleanupInput,
    run_inpaint_cleanup,
)


def _scene(seed: int = 20260809, shape=(160, 220, 3)):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=shape, dtype=np.uint8)
    inpainted = image.copy()
    # 인페인팅이 바꾼 것처럼 보이도록 마스크 영역을 덮어쓴다.
    inpainted[20:60, 30:90] = 255
    mask = np.zeros(shape[:2], dtype=np.uint8)
    mask[20:60, 30:90] = 255
    return image, inpainted, mask


def _reference(image, inpainted, mask, mask_details, blocks, config, label, edit_mask):
    """스윕 안에 있던 원래 순서 그대로. 동등성의 기준이다."""

    out = imk.convert_scale_abs(inpainted)
    work_mask = mask
    if edit_mask is not None:
        work_mask = np.where((work_mask > 0) | (edit_mask > 0), 255, 0).astype(np.uint8)
    stats = {"autonomous_residue_cleanup": "disabled"}
    out, work_mask, stats = apply_duplicate_bubble_inner_fill(
        out, work_mask, mask_details, stats
    )
    before = count_changed_outside_edit_mask(image, out, work_mask)
    out = composite_with_edit_mask(image, out, work_mask)
    after = count_changed_outside_edit_mask(image, out, work_mask)
    return out, work_mask, stats, before, after


class CleanupEquivalenceTests(unittest.TestCase):
    def _run_both(self, *, edit_mask=None, seed: int = 20260809):
        image, inpainted, mask = _scene(seed)
        mask_details: dict = {}
        result = run_inpaint_cleanup(
            InpaintCleanupInput(
                image=image,
                inpaint_input_img=inpainted.copy(),
                mask=mask.copy(),
                mask_details=dict(mask_details),
                inpaint_blocks=[],
                config=None,
                page_label="1/1",
                inpaint_edit_mask=None if edit_mask is None else edit_mask.copy(),
            )
        )
        expected = _reference(
            image,
            inpainted.copy(),
            mask.copy(),
            dict(mask_details),
            [],
            None,
            "1/1",
            None if edit_mask is None else edit_mask.copy(),
        )
        return result, expected

    def test_the_extracted_job_matches_the_original_order_byte_for_byte(self) -> None:
        # 후처리는 순서가 결과를 바꾼다. 옮기면서 순서가 흔들리지 않았는지가 핵심이다.
        result, (out, work_mask, _stats, before, after) = self._run_both()

        np.testing.assert_array_equal(result.inpaint_input_img, out)
        np.testing.assert_array_equal(result.mask, work_mask)
        self.assertEqual(result.outside_before_restore, before)
        self.assertEqual(result.outside_after_restore, after)

    def test_an_edit_mask_is_unioned_the_same_way(self) -> None:
        edit = np.zeros((160, 220), dtype=np.uint8)
        edit[100:130, 20:60] = 255

        result, (out, work_mask, _stats, _b, _a) = self._run_both(edit_mask=edit)

        np.testing.assert_array_equal(result.inpaint_input_img, out)
        np.testing.assert_array_equal(result.mask, work_mask)
        # 합집합이 실제로 반영됐는지도 본다.
        self.assertTrue(bool(result.mask[110, 30]))

    def test_protected_corner_mask_is_restored_and_removed_from_final_mask(self) -> None:
        image, inpainted, mask = _scene()
        protected = np.zeros(mask.shape, dtype=np.uint8)
        protected[20:30, 30:40] = 255

        result = run_inpaint_cleanup(
            InpaintCleanupInput(
                image=image,
                inpaint_input_img=inpainted,
                mask=mask,
                mask_details={"protected_corner_mask": protected},
                inpaint_blocks=[],
                config=None,
                page_label="1/1",
            )
        )

        self.assertEqual(int(np.count_nonzero(result.mask[protected > 0])), 0)
        np.testing.assert_array_equal(
            result.inpaint_input_img[protected > 0],
            image[protected > 0],
        )
        self.assertGreater(int(np.count_nonzero(result.mask[30:60, 40:90])), 0)

    def test_several_scenes_stay_equivalent(self) -> None:
        for seed in (1, 7, 4242):
            with self.subTest(seed=seed):
                result, (out, work_mask, _s, _b, _a) = self._run_both(seed=seed)
                np.testing.assert_array_equal(result.inpaint_input_img, out)
                np.testing.assert_array_equal(result.mask, work_mask)


class CleanupContractTests(unittest.TestCase):
    def test_product_job_never_calls_autonomous_residue_cleanup(self) -> None:
        image, inpainted, mask = _scene()
        with mock.patch(
            "modules.utils.inpaint_cleanup.refine_bubble_residue_inpaint",
            side_effect=AssertionError("autonomous cleanup must stay retired"),
        ):
            result = run_inpaint_cleanup(
                InpaintCleanupInput(
                    image=image,
                    inpaint_input_img=inpainted,
                    mask=mask,
                    mask_details={},
                    inpaint_blocks=[object()],
                    config=None,
                    page_label="1/1",
                )
            )

        self.assertEqual(
            result.cleanup_stats["autonomous_residue_cleanup"],
            "disabled",
        )

    def test_the_result_is_contiguous_uint8(self) -> None:
        # 하류가 이 배열을 그대로 저장하고 해시한다.
        image, inpainted, mask = _scene()
        result = run_inpaint_cleanup(
            InpaintCleanupInput(
                image=image,
                inpaint_input_img=inpainted,
                mask=mask,
                mask_details={},
                inpaint_blocks=[],
                config=None,
                page_label="1/1",
            )
        )

        self.assertEqual(result.inpaint_input_img.dtype, np.uint8)
        self.assertEqual(result.mask.dtype, np.uint8)
        self.assertTrue(result.inpaint_input_img.flags["C_CONTIGUOUS"])
        self.assertTrue(result.mask.flags["C_CONTIGUOUS"])

    def test_compositing_keeps_pixels_outside_the_mask_untouched(self) -> None:
        # 이 값이 0 이 아니면 인페인팅이 마스크 밖으로 번진 것이고, 호출부가
        # 그 페이지를 실패시킨다. 후처리를 옮기면서 이 보증이 약해지면 안 된다.
        image, inpainted, mask = _scene()
        result = run_inpaint_cleanup(
            InpaintCleanupInput(
                image=image,
                inpaint_input_img=inpainted,
                mask=mask,
                mask_details={},
                inpaint_blocks=[],
                config=None,
                page_label="1/1",
            )
        )

        self.assertEqual(result.outside_after_restore, 0)
        outside = mask == 0
        np.testing.assert_array_equal(
            result.inpaint_input_img[outside], image[outside]
        )

    def test_the_job_reports_how_long_it_took(self) -> None:
        # 워커로 옮긴 뒤에도 이 시간이 보여야 한다.
        image, inpainted, mask = _scene()
        result = run_inpaint_cleanup(
            InpaintCleanupInput(
                image=image,
                inpaint_input_img=inpainted,
                mask=mask,
                mask_details={},
                inpaint_blocks=[],
                config=None,
                page_label="1/1",
            )
        )

        self.assertGreaterEqual(result.worker_seconds, 0.0)

    def test_the_job_does_not_touch_its_inputs(self) -> None:
        # 워커 스레드에서 돌 것이므로 입력을 건드리면 안 된다.
        image, inpainted, mask = _scene()
        image_before = image.copy()
        mask_before = mask.copy()

        run_inpaint_cleanup(
            InpaintCleanupInput(
                image=image,
                inpaint_input_img=inpainted,
                mask=mask,
                mask_details={},
                inpaint_blocks=[],
                config=None,
                page_label="1/1",
            )
        )

        np.testing.assert_array_equal(image, image_before)
        np.testing.assert_array_equal(mask, mask_before)


if __name__ == "__main__":
    unittest.main()
