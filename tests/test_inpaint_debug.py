from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from modules.utils.inpaint_debug import (
    build_inpaint_debug_metadata,
    export_inpaint_debug_artifacts,
)
from pipeline.batch_processor import BatchProcessor


@dataclass
class _Block:
    xyxy: list[int]
    bubble_xyxy: list[int] | None = None
    text_class: str = "text_bubble"
    inpaint_bboxes: list[list[int]] = field(default_factory=list)


class InpaintDebugTests(unittest.TestCase):
    def test_build_metadata_counts_masks_and_blocks(self) -> None:
        raw_mask = np.zeros((8, 8), dtype=np.uint8)
        raw_mask[1:3, 1:3] = 255
        cleanup_delta = np.zeros((8, 8), dtype=np.uint8)
        cleanup_delta[4:5, 4:6] = 255
        block = _Block(
            xyxy=[1, 1, 4, 5],
            bubble_xyxy=[0, 0, 7, 7],
            text_class="text_bubble",
            inpaint_bboxes=[[1, 1, 4, 5]],
        )

        metadata = build_inpaint_debug_metadata(
            image_path="page.png",
            run_type="batch",
            detector_key="RT-DETR-v2",
            detector_engine="RTDetrV2ONNXDetection",
            device="cpu",
            inpainter="AOT",
            hd_strategy="Resize",
            blocks=[block],
            raw_mask=raw_mask,
            cleanup_delta=cleanup_delta,
            cleanup_stats={"applied": True, "component_count": 2, "block_count": 1},
        )

        self.assertEqual(metadata["block_count"], 1)
        self.assertEqual(metadata["raw_mask_pixel_count"], 4)
        self.assertEqual(metadata["cleanup_delta_pixel_count"], 2)
        self.assertTrue(metadata["cleanup_applied"])
        self.assertEqual(metadata["blocks"][0]["text_class"], "text_bubble")
        self.assertEqual(metadata["blocks"][0]["inpaint_bboxes"], [[1, 1, 4, 5]])

    def test_export_artifacts_only_writes_selected_debug_outputs(self) -> None:
        image = np.full((10, 12, 3), 255, dtype=np.uint8)
        raw_mask = np.zeros((10, 12), dtype=np.uint8)
        raw_mask[2:6, 3:7] = 255
        cleanup_delta = np.zeros((10, 12), dtype=np.uint8)
        cleanup_delta[6:8, 4:5] = 255
        block = _Block(xyxy=[3, 2, 7, 6], bubble_xyxy=[1, 1, 10, 9])

        with tempfile.TemporaryDirectory() as tmp_dir:
            export_inpaint_debug_artifacts(
                export_root=tmp_dir,
                archive_bname="",
                page_base_name="page",
                image=image,
                blocks=[block],
                export_settings={
                    "export_detector_overlay": True,
                    "export_raw_mask": True,
                    "export_mask_overlay": False,
                    "export_cleanup_mask_delta": False,
                    "export_debug_metadata": True,
                },
                raw_mask=raw_mask,
                cleanup_delta=cleanup_delta,
                metadata={"hello": "world"},
            )

            root = Path(tmp_dir)
            self.assertTrue((root / "detector_overlays" / "page_detector_overlay.png").exists())
            self.assertTrue((root / "raw_masks" / "page_raw_mask.png").exists())
            self.assertTrue((root / "debug_metadata" / "page_debug.json").exists())
            self.assertFalse((root / "mask_overlays" / "page_mask_overlay.png").exists())
            self.assertFalse((root / "cleanup_mask_delta" / "page_cleanup_delta.png").exists())
            payload = json.loads((root / "debug_metadata" / "page_debug.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["hello"], "world")

    def test_cleanup_delta_only_contains_new_pixels(self) -> None:
        processor = BatchProcessor.__new__(BatchProcessor)
        raw_mask = np.zeros((5, 5), dtype=np.uint8)
        raw_mask[1:3, 1:3] = 255
        final_mask = raw_mask.copy()
        final_mask[3:5, 2:4] = 255

        cleanup_delta = processor._build_cleanup_delta_mask(raw_mask, final_mask)

        expected = np.zeros((5, 5), dtype=np.uint8)
        expected[3:5, 2:4] = 255
        np.testing.assert_array_equal(cleanup_delta, expected)


if __name__ == "__main__":
    unittest.main()
