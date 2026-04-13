from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace

from app.projects.project_state_v2 import (
    close_cached_connection,
    load_state_from_proj_file_v2,
    save_state_to_proj_file_v2,
)


class _SettingsPage:
    def get_llm_settings(self) -> dict:
        return {"extra_context": ""}


class _BatchReportCtrl:
    def __init__(self) -> None:
        self.latest_report = None

    def export_latest_report_for_project(self):
        return None

    def import_latest_report_from_project(self, report, refresh: bool = False):
        self.latest_report = report


class _FakeProjectMain:
    def __init__(self, image_path: str | None = None, export_source: dict | None = None) -> None:
        self.image_data = {}
        self.in_memory_history = {}
        self.image_history = {}
        self.curr_img_idx = 0
        self.webtoon_mode = False
        self.image_viewer = SimpleNamespace(webtoon_view_state={})
        self.image_files = [image_path] if image_path else []
        self.image_states = {image_path: {}} if image_path else {}
        self.current_history_index = {}
        self.displayed_images = set()
        self.loaded_images = []
        self.image_patches = {}
        self.settings_page = _SettingsPage()
        self.batch_report_ctrl = _BatchReportCtrl()
        self.export_source_by_path = {image_path: export_source} if image_path and export_source else {}
        self.project_file = None
        self.project_output_preferences = {
            "output_use_global": False,
            "output_target": "single_archive",
            "output_image_format": "same_as_source",
            "output_archive_format": "cbz",
            "output_archive_image_format": "webp",
            "output_archive_compression_level": 7,
        }


class ProjectExportSourceRoundTripTests(unittest.TestCase):
    def test_v2_project_round_trip_restores_export_source_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "page.png")
            project_path = os.path.join(temp_dir, "demo.ctpr")
            archive_path = os.path.join(temp_dir, "chapter.zip")
            with open(image_path, "wb") as fh:
                fh.write(b"not-a-real-png-but-good-enough")

            source_main = _FakeProjectMain(
                image_path=image_path,
                export_source={
                    "kind": "archive",
                    "source_path": archive_path,
                },
            )
            source_main.project_file = project_path

            try:
                save_state_to_proj_file_v2(source_main, project_path)

                restored_main = _FakeProjectMain()
                saved_ctx = load_state_from_proj_file_v2(restored_main, project_path)

                self.assertEqual(saved_ctx, "")
                self.assertEqual(len(restored_main.image_files), 1)
                restored_path = restored_main.image_files[0]
                self.assertIn(restored_path, restored_main.export_source_by_path)
                self.assertEqual(
                    restored_main.export_source_by_path[restored_path],
                    {
                        "kind": "archive",
                        "source_path": os.path.abspath(archive_path),
                    },
                )
                self.assertEqual(
                    restored_main.project_output_preferences,
                    {
                        "output_use_global": False,
                        "output_target": "single_archive",
                        "output_image_format": "same_as_source",
                        "output_archive_format": "cbz",
                        "output_archive_image_format": "webp",
                        "output_archive_compression_level": 7,
                    },
                )
            finally:
                close_cached_connection(project_path)


if __name__ == "__main__":
    unittest.main()
