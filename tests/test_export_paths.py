from __future__ import annotations

import os
import tempfile
import unittest

from modules.utils.export_paths import (
    export_run_root,
    reserve_export_run_token,
    resolve_export_directory,
)


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
            self.assertEqual(archive_bname, "demo")

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

    def test_resolve_export_directory_matches_windows_source_record_for_wsl_temp_path(self) -> None:
        image_path = "/mnt/c/ExampleWorkspace/project/tmpabc/001.png"
        source_image_path = r"C:\ExampleWorkspace\project\tmpabc\001.png"
        archive_path = r"C:\ExampleWorkspace\project\example_source_chapter.pdf"

        directory, archive_bname = resolve_export_directory(
            image_path,
            source_records={
                source_image_path: {
                    "kind": "archive",
                    "source_path": archive_path,
                }
            },
        )

        self.assertTrue(directory.endswith("project"))
        self.assertEqual(archive_bname, "example_source_chapter")

    def test_export_run_root_uses_source_named_log_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache: dict[str, str] = {}
            token = reserve_export_run_token(
                temp_dir,
                "May-29-2026_03-48-56AM",
                cache,
                source_name="example_source_chapter",
            )
            run_root = export_run_root(
                temp_dir,
                token,
                source_name="example_source_chapter",
            )

            self.assertEqual(
                os.path.basename(run_root),
                "log_example_source_chapter_May-29-2026_03-48-56AM",
            )
            self.assertTrue(os.path.isdir(run_root))


if __name__ == "__main__":
    unittest.main()
