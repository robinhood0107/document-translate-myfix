from __future__ import annotations

import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from PIL import Image

from modules.utils.archives import (
    close_comic_cache,
    list_archive_image_entries,
    materialize_archive_entry,
)
from modules.utils.file_handler import FileHandler, ensure_prepared_path_materialized


class ArchiveCbzIntegrationTests(unittest.TestCase):
    def tearDown(self) -> None:
        close_comic_cache()

    def _make_png(self, path: Path, color: tuple[int, int, int]) -> bytes:
        Image.new("RGB", (12, 12), color=color).save(path)
        return path.read_bytes()

    def test_cbz_entries_are_unique_and_materialize_by_page_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            src_a = root / "a.png"
            src_b = root / "b.png"
            bytes_a = self._make_png(src_a, (255, 0, 0))
            bytes_b = self._make_png(src_b, (0, 255, 0))

            archive_path = root / "sample.cbz"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.write(src_a, arcname="chapter_a/page.png")
                archive.write(src_b, arcname="chapter_b/page.png")

            entries = list_archive_image_entries(str(archive_path))
            self.assertEqual(len(entries), 2)
            self.assertEqual([entry["page_index"] for entry in entries], [0, 1])
            self.assertNotEqual(entries[0]["entry_name"], entries[1]["entry_name"])
            self.assertEqual(entries[0]["ext"], ".png")
            self.assertTrue(entries[0]["entry_name"].startswith("000001_"))
            self.assertTrue(entries[1]["entry_name"].startswith("000002_"))

            out_a = root / "out_a.png"
            out_b = root / "out_b.png"
            self.assertTrue(materialize_archive_entry(str(archive_path), entries[0], str(out_a)))
            self.assertTrue(materialize_archive_entry(str(archive_path), entries[1], str(out_b)))
            self.assertEqual(out_a.read_bytes(), bytes_a)
            self.assertEqual(out_b.read_bytes(), bytes_b)

    def test_file_handler_keeps_lazy_contract_for_cbz(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            src_a = root / "001.png"
            src_b = root / "002.png"
            self._make_png(src_a, (255, 0, 0))
            self._make_png(src_b, (0, 255, 0))

            archive_path = root / "sample.cbz"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.write(src_a, arcname="001.png")
                archive.write(src_b, arcname="002.png")

            handler = FileHandler()
            paths = handler.prepare_files([str(archive_path)])
            self.assertEqual(len(paths), 2)
            self.assertTrue(os.path.isfile(paths[0]))
            self.assertFalse(os.path.exists(paths[1]))
            self.assertTrue(ensure_prepared_path_materialized(paths[1]))
            self.assertTrue(os.path.isfile(paths[1]))

    def test_cbr_dispatch_uses_cbz_native_loader_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "sample.cbr"
            archive_path.write_bytes(b"fake-cbr")

            fake_page = mock.Mock()
            fake_page.name = "page01.png"
            fake_page.suffix = ".png"
            fake_page.content = b"page-bytes"
            fake_comic = [fake_page]

            with mock.patch("modules.utils.archives._load_comic_archive", return_value=fake_comic) as mocked_loader:
                entries = list_archive_image_entries(str(archive_path))
                self.assertEqual(len(entries), 1)
                self.assertEqual(entries[0]["page_index"], 0)

                output_path = Path(temp_dir) / "materialized.png"
                self.assertTrue(materialize_archive_entry(str(archive_path), entries[0], str(output_path)))
                self.assertEqual(output_path.read_bytes(), b"page-bytes")
                mocked_loader.assert_called_once_with(os.path.abspath(str(archive_path)))


if __name__ == "__main__":
    unittest.main()
