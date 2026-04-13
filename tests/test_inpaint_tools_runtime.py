from __future__ import annotations

import os
import tempfile
import unittest

import imkit as imk
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtGui, QtWidgets

from app.projects.project_state_v2 import (
    close_cached_connection,
    load_state_from_proj_file_v2,
    save_state_to_proj_file_v2,
)
from modules.utils.inpaint_strokes import (
    PATCH_KIND_RESTORE,
    STROKE_ROLE_ADD,
    STROKE_ROLE_EXCLUDE,
    STROKE_ROLE_GENERATED,
    normalize_stroke_role,
    retain_non_manual_strokes,
)
from pipeline.inpainting import InpaintingHandler


class _DummySettingsPage:
    def get_llm_settings(self) -> dict:
        return {"extra_context": ""}


class _DummyViewer:
    def __init__(self) -> None:
        self.webtoon_view_state = {}


class _DummyBatchReportCtrl:
    def export_latest_report_for_project(self):
        return None

    def import_latest_report_from_project(self, report, refresh=False) -> None:
        self.report = report
        self.refresh = refresh


class _DummyComicTranslate:
    def __init__(self, image_path: str, patch_path: str) -> None:
        self.curr_img_idx = 0
        self.image_files = [image_path]
        self.image_states = {image_path: {"viewer_state": {"brush_strokes": []}}}
        self.image_data = {}
        self.displayed_images = set()
        self.loaded_images = []
        self.in_memory_history = {image_path: [None]}
        self.image_history = {image_path: [image_path]}
        self.current_history_index = {image_path: 0}
        self.image_patches = {
            image_path: [
                {
                    "bbox": [0, 0, 6, 6],
                    "png_path": patch_path,
                    "hash": "patch-hash-1",
                    "kind": PATCH_KIND_RESTORE,
                    "order": 7,
                }
            ]
        }
        self.export_source_by_path = {}
        self.settings_page = _DummySettingsPage()
        self.webtoon_mode = False
        self.image_viewer = _DummyViewer()
        self.batch_report_ctrl = _DummyBatchReportCtrl()


class InpaintToolsRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_normalize_stroke_role_preserves_legacy_generated_brush(self) -> None:
        self.assertEqual(normalize_stroke_role(None, brush="#80ff0000"), STROKE_ROLE_GENERATED)
        self.assertEqual(normalize_stroke_role(None, brush="#00000000"), STROKE_ROLE_ADD)

    def test_retain_non_manual_strokes_keeps_generated_only(self) -> None:
        strokes = [
            {"role": STROKE_ROLE_GENERATED, "brush": "#80ff0000"},
            {"role": STROKE_ROLE_ADD, "brush": "#00000000"},
            {"role": STROKE_ROLE_EXCLUDE, "brush": "#00000000"},
        ]

        filtered = retain_non_manual_strokes(strokes)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["role"], STROKE_ROLE_GENERATED)

    def test_generate_mask_from_saved_strokes_applies_exclude_after_include(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        include_path = QtGui.QPainterPath()
        include_path.moveTo(10, 50)
        include_path.lineTo(90, 50)
        exclude_path = QtGui.QPainterPath()
        exclude_path.moveTo(35, 50)
        exclude_path.lineTo(65, 50)

        handler = InpaintingHandler.__new__(InpaintingHandler)
        mask = handler._generate_mask_from_saved_strokes(
            [
                {"path": include_path, "width": 24, "role": STROKE_ROLE_ADD},
                {"path": exclude_path, "width": 12, "role": STROKE_ROLE_EXCLUDE},
            ],
            image,
        )

        self.assertIsNotNone(mask)
        self.assertEqual(int(mask[50, 20]), 255)
        self.assertEqual(int(mask[50, 50]), 0)

    def test_project_state_v2_roundtrips_patch_kind_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "page.png")
            patch_path = os.path.join(temp_dir, "patch.png")
            project_path = os.path.join(temp_dir, "demo.ctpr")
            try:
                imk.write_image(image_path, np.full((12, 12, 3), 180, dtype=np.uint8))
                imk.write_image(patch_path, np.full((6, 6, 3), 60, dtype=np.uint8))

                source = _DummyComicTranslate(image_path, patch_path)
                save_state_to_proj_file_v2(source, project_path)

                restored = _DummyComicTranslate(image_path, patch_path)
                restored.image_files = []
                restored.image_states = {}
                restored.image_data = {}
                restored.displayed_images = set()
                restored.loaded_images = []
                restored.in_memory_history = {}
                restored.image_history = {}
                restored.current_history_index = {}
                restored.image_patches = {}
                restored.export_source_by_path = {}

                load_state_from_proj_file_v2(restored, project_path)
                restored_page = restored.image_files[0]
                restored_patch = restored.image_patches[restored_page][0]

                self.assertEqual(restored_patch["kind"], PATCH_KIND_RESTORE)
                self.assertEqual(restored_patch["order"], 7)
            finally:
                close_cached_connection(project_path)


if __name__ == "__main__":
    unittest.main()
