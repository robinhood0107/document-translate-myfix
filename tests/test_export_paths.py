from __future__ import annotations

import os
import tempfile
import unittest

from modules.utils.export_paths import resolve_export_directory


class ExportPathResolutionTests(unittest.TestCase):
    def test_resolve_export_directory_uses_saved_archive_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "unique_images", "1", "01.png")
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            open(image_path, "wb").close()
            archive_path = os.path.join(temp_dir, "chapter.zip")

            directory, archive_bname = resolve_export_directory(
                image_path,
                source_records={
                    image_path: {
                        "kind": "archive",
                        "source_path": archive_path,
                    }
                },
            )

            self.assertEqual(directory, temp_dir)
            self.assertEqual(archive_bname, "chapter")

    def test_resolve_export_directory_falls_back_to_project_dir_for_temp_project_paths(self) -> None:
        with tempfile.TemporaryDirectory() as root_dir:
            temp_dir = os.path.join(root_dir, "tmp-project")
            image_path = os.path.join(temp_dir, "unique_images", "1", "01.png")
            project_dir = os.path.join(root_dir, "projects")
            project_file = os.path.join(project_dir, "demo.ctpr")
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            os.makedirs(project_dir, exist_ok=True)
            open(image_path, "wb").close()
            open(project_file, "wb").close()

            directory, archive_bname = resolve_export_directory(
                image_path,
                project_file=project_file,
                temp_dir=temp_dir,
            )

            self.assertEqual(directory, project_dir)
            self.assertEqual(archive_bname, "")

    def test_resolve_export_directory_uses_live_archive_info_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = os.path.join(temp_dir, "series.cbz")
            image_path = os.path.join(temp_dir, "extract", "001.png")
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            open(image_path, "wb").close()

            directory, archive_bname = resolve_export_directory(
                image_path,
                archive_info=[
                    {
                        "archive_path": archive_path,
                        "extracted_images": [image_path],
                    }
                ],
            )

            self.assertEqual(directory, temp_dir)
            self.assertEqual(archive_bname, "series")


if __name__ == "__main__":
    unittest.main()
