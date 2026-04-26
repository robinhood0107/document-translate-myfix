from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from pipeline.batch_processor import BatchProcessor


class _PreviewMain:
    def __init__(self) -> None:
        self.events = []
        self._intermediate_preview_disabled_notices = set()

    def report_runtime_progress(self, payload):
        self.events.append(dict(payload))


class BatchDebugOutputTests(unittest.TestCase):
    def test_inpainted_debug_image_stays_out_of_final_output_folder(self) -> None:
        processor = object.__new__(BatchProcessor)
        with tempfile.TemporaryDirectory() as temp_dir:
            export_root = os.path.join(temp_dir, "comic_translate_run")
            source_path = os.path.join(temp_dir, "92.png")
            with open(source_path, "wb") as fh:
                fh.write(b"source")

            output_path = processor._write_inpainted_debug_image(
                export_root=export_root,
                archive_bname="",
                image_path=source_path,
                cleaned_image=np.zeros((8, 8, 3), dtype=np.uint8),
                export_settings={
                    "export_inpainted_image": True,
                    "resolved_automatic_output_image_format": "png",
                },
            )

            self.assertTrue(os.path.isfile(output_path))
            self.assertIn(os.path.join("comic_translate_run", "inpainted_images"), output_path)
            self.assertTrue(output_path.endswith("92_cleaned.png"))
            self.assertFalse(os.path.exists(os.path.join(temp_dir, "92_cleaned.png")))

    def test_preview_is_not_generated_when_debug_checkbox_is_off(self) -> None:
        processor = object.__new__(BatchProcessor)
        processor.main_page = _PreviewMain()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "92.png")
            preview_path = os.path.join(temp_dir, "92_raw_mask.png")
            with open(image_path, "wb") as fh:
                fh.write(b"source")
            with open(preview_path, "wb") as fh:
                fh.write(b"preview")

            processor._maybe_emit_preview_image(
                index=0,
                total=1,
                image_path=image_path,
                stage_key="raw_mask",
                stage_label="원본 마스크",
                export_settings={"export_raw_mask": False},
                preferred_path=preview_path,
            )

            self.assertEqual(len(processor.main_page.events), 1)
            event = processor.main_page.events[0]
            self.assertEqual(event["preview_disabled_reason"], "intermediate_preview_disabled")
            self.assertNotIn("preview_path", event)

    def test_preview_uses_existing_debug_export_when_checkbox_is_on(self) -> None:
        processor = object.__new__(BatchProcessor)
        processor.main_page = _PreviewMain()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "92.png")
            preview_path = os.path.join(temp_dir, "92_raw_mask.png")
            with open(image_path, "wb") as fh:
                fh.write(b"source")
            with open(preview_path, "wb") as fh:
                fh.write(b"preview")

            processor._maybe_emit_preview_image(
                index=0,
                total=1,
                image_path=image_path,
                stage_key="raw_mask",
                stage_label="원본 마스크",
                export_settings={"export_raw_mask": True},
                preferred_path=preview_path,
            )

            self.assertEqual(len(processor.main_page.events), 1)
            event = processor.main_page.events[0]
            self.assertEqual(event["preview_path"], preview_path)
            self.assertFalse(event["temporary_preview"])


if __name__ == "__main__":
    unittest.main()
