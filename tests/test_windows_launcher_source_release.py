from __future__ import annotations

import importlib.util
import hashlib
import re
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_windows_launcher_source_bundle.py"
SPEC = importlib.util.spec_from_file_location("launcher_source_release", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
release = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release
SPEC.loader.exec_module(release)


class WindowsLauncherSourceReleaseTests(unittest.TestCase):
    def test_allowlist_contains_runtime_and_excludes_dev_assets(self) -> None:
        entries = release.list_release_entries(ROOT, "HEAD")
        paths = {entry.path for entry in entries}

        self.assertTrue(release.REQUIRED_BUNDLE_FILES <= paths)
        self.assertIn("app/ui/main_window/window.py", paths)
        self.assertIn("modules/ocr/local_runtime.py", paths)
        self.assertIn("pipeline/main_pipeline.py", paths)
        self.assertNotIn("scripts/benchmark_cold_cache_finalization.py", paths)
        self.assertNotIn("scripts/build_windows_gpu_onefile.ps1", paths)
        self.assertFalse(any(path.startswith("tests/") for path in paths))
        self.assertFalse(any(path.startswith("benchmarks/") for path in paths))
        self.assertFalse(any(path.startswith(".github/") for path in paths))

    def test_windows_scripts_are_crlf_normalized(self) -> None:
        source = b"line1\nline2\r\nline3\r"
        normalized = release.normalize_release_bytes("example.ps1", source)
        self.assertEqual(normalized, b"line1\r\nline2\r\nline3\r\n")
        self.assertNotIn(b"\n", normalized.replace(b"\r\n", b""))

    def test_unsafe_and_model_paths_are_rejected(self) -> None:
        for path in ("../secret.txt", "/absolute.txt", r"folder\escape.txt"):
            with self.subTest(path=path):
                with self.assertRaises(RuntimeError):
                    release._validate_relative_path(path)
        for path in ("models/model.gguf", "cache/checkpoint.sqlite3", ".env"):
            with self.subTest(path=path):
                with self.assertRaises(RuntimeError):
                    release._validate_allowed_file(path)
        with self.assertRaises(RuntimeError):
            release._scan_release_bytes(
                "README.md",
                b"private path: C:\\Users\\example\\Desktop\\artifact",
            )
        with self.assertRaises(RuntimeError):
            release._scan_release_bytes(
                "README.md",
                b"-----BEGIN PRIVATE KEY-----",
            )

    def test_release_build_is_deterministic_and_self_verifying(self) -> None:
        head_version_raw = release._run_git(
            ROOT,
            ["show", "HEAD:app/version.py"],
        )
        match = re.search(rb'__version__\s*=\s*["\']([^"\']+)["\']', head_version_raw)
        self.assertIsNotNone(match)
        version = match.group(1).decode("ascii")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            first = release.build_release_bundle(
                repo_root=ROOT,
                commit="HEAD",
                version=version,
                output_dir=temp_root / "first",
            )
            second = release.build_release_bundle(
                repo_root=ROOT,
                commit="HEAD",
                version=version,
                output_dir=temp_root / "second",
            )

            first_archive = Path(first["archive"])
            second_archive = Path(second["archive"])
            self.assertEqual(first_archive.read_bytes(), second_archive.read_bytes())
            self.assertEqual(first["archive_sha256"], second["archive_sha256"])
            self.assertEqual(first["verification"]["version"], version)

            with zipfile.ZipFile(first_archive) as archive:
                names = archive.namelist()
                self.assertTrue(any(name.endswith("/RELEASE-MANIFEST.json") for name in names))
                self.assertTrue(any(name.endswith("/run_comic.bat") for name in names))
                self.assertFalse(any("/tests/" in name for name in names))
                self.assertFalse(any("/benchmarks/" in name for name in names))
                self.assertFalse(any("benchmark_" in name for name in names))
                self.assertFalse(any(name.lower().endswith(".gguf") for name in names))

    def test_launchers_offer_no_install_release_contract(self) -> None:
        for launcher in ("run_comic.bat", "run_comic_cuda13.bat"):
            text = (ROOT / launcher).read_text(encoding="utf-8")
            self.assertIn('if /I "%COMIC_VERIFY_ONLY%"=="1"', text)
            self.assertIn("scripts\\prepare_gemma_runtime.ps1", text)
            self.assertIn("resources\\translations\\compiled\\ct_ko.qm", text)

    def test_product_updater_prefers_launcher_source_zip(self) -> None:
        from app.update_checker import (
            UpdateChecker,
            checksum_url_for_release_asset,
            parse_release_checksum,
            select_release_asset_url,
            validate_download_filename,
        )

        data = {
            "assets": [
                {
                    "name": "legacy-installer.exe",
                    "browser_download_url": "https://example.invalid/legacy.exe",
                },
                {
                    "name": "comic-translate-v1.1.0-windows-launcher-source.zip",
                    "browser_download_url": "https://example.invalid/source.zip",
                },
            ]
        }
        self.assertEqual(UpdateChecker.REPO_OWNER, "robinhood0107")
        self.assertEqual(UpdateChecker.REPO_NAME, "document-translate-myfix")
        self.assertEqual(
            select_release_asset_url(data, "Windows", "1.1.0"),
            "https://example.invalid/source.zip",
        )
        filename = "comic-translate-v1.1.0-windows-launcher-source.zip"
        digest = "a" * 64
        self.assertEqual(
            parse_release_checksum(f"{digest}  {filename}\n", filename),
            digest,
        )
        self.assertEqual(
            checksum_url_for_release_asset(
                "https://github.com/example/releases/download/v1.1.0/" + filename
            ),
            "https://github.com/example/releases/download/v1.1.0/SHA256SUMS.txt",
        )
        with self.assertRaises(ValueError):
            parse_release_checksum(f"{'b' * 64}  other.zip\n", filename)
        self.assertEqual(validate_download_filename(filename), filename)
        for unsafe in ("../release.zip", r"..\release.zip", r"C:\release.zip"):
            with self.subTest(unsafe=unsafe):
                with self.assertRaises(ValueError):
                    validate_download_filename(unsafe)

    def test_source_update_download_verifies_sha_and_renames_atomically(self) -> None:
        from app.update_checker import DownloadWorker

        filename = "comic-translate-v1.1.0-windows-launcher-source.zip"
        url = f"https://example.invalid/releases/v1.1.0/{filename}"
        payload = b"verified bundle bytes"
        checksum_response = mock.Mock()
        checksum_response.text = f"{hashlib.sha256(payload).hexdigest()}  {filename}\n"
        checksum_response.raise_for_status.return_value = None
        download_response = mock.Mock()
        download_response.headers = {"content-length": str(len(payload))}
        download_response.raise_for_status.return_value = None
        download_response.iter_content.return_value = [payload[:5], payload[5:]]

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
            "app.update_checker.QStandardPaths.writableLocation",
            return_value=temp_dir,
        ), mock.patch(
            "app.update_checker.requests.get",
            side_effect=[checksum_response, download_response],
        ):
            worker = DownloadWorker(url, filename)
            finished: list[str] = []
            errors: list[str] = []
            worker.finished_path.connect(finished.append)
            worker.error.connect(errors.append)
            worker.run()

            final_path = Path(temp_dir) / filename
            self.assertEqual(errors, [])
            self.assertEqual(finished, [str(final_path)])
            self.assertEqual(final_path.read_bytes(), payload)
            self.assertFalse(Path(f"{final_path}.partial").exists())

    def test_source_update_checksum_failure_leaves_no_output(self) -> None:
        from app.update_checker import DownloadWorker

        filename = "comic-translate-v1.1.0-windows-launcher-source.zip"
        url = f"https://example.invalid/releases/v1.1.0/{filename}"
        checksum_response = mock.Mock()
        checksum_response.text = f"{'0' * 64}  {filename}\n"
        checksum_response.raise_for_status.return_value = None
        download_response = mock.Mock()
        download_response.headers = {}
        download_response.raise_for_status.return_value = None
        download_response.iter_content.return_value = [b"wrong bytes"]

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
            "app.update_checker.QStandardPaths.writableLocation",
            return_value=temp_dir,
        ), mock.patch(
            "app.update_checker.requests.get",
            side_effect=[checksum_response, download_response],
        ):
            worker = DownloadWorker(url, filename)
            finished: list[str] = []
            errors: list[str] = []
            worker.finished_path.connect(finished.append)
            worker.error.connect(errors.append)
            worker.run()

            final_path = Path(temp_dir) / filename
            self.assertEqual(finished, [])
            self.assertTrue(any("SHA-256 does not match" in item for item in errors))
            self.assertFalse(final_path.exists())
            self.assertFalse(Path(f"{final_path}.partial").exists())


if __name__ == "__main__":
    unittest.main()
