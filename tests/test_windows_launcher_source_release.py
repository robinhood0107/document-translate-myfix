from __future__ import annotations

import importlib.metadata
import importlib.util
import hashlib
import re
import sys
import tempfile
import tomllib
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_windows_launcher_source_bundle.py"
SPEC = importlib.util.spec_from_file_location("launcher_source_release", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
release = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release
SPEC.loader.exec_module(release)

RUNTIME_SCRIPT_PATH = ROOT / "scripts" / "verify_windows_runtime.py"
RUNTIME_SPEC = importlib.util.spec_from_file_location(
    "windows_runtime_verifier",
    RUNTIME_SCRIPT_PATH,
)
assert RUNTIME_SPEC is not None and RUNTIME_SPEC.loader is not None
runtime_verifier = importlib.util.module_from_spec(RUNTIME_SPEC)
sys.modules[RUNTIME_SPEC.name] = runtime_verifier
RUNTIME_SPEC.loader.exec_module(runtime_verifier)


class WindowsLauncherSourceReleaseTests(unittest.TestCase):
    def test_allowlist_contains_runtime_and_excludes_dev_assets(self) -> None:
        entries = release.list_release_entries(ROOT, "HEAD")
        paths = {entry.path for entry in entries}

        self.assertTrue(release.REQUIRED_BUNDLE_FILES <= paths)
        self.assertIn("app/ui/main_window/window.py", paths)
        self.assertIn("modules/ocr/local_runtime.py", paths)
        self.assertIn("pipeline/main_pipeline.py", paths)
        self.assertIn(
            "scripts/prepare_paddleocr_llamacpp_runtime.ps1",
            paths,
        )
        self.assertIn(
            "scripts/prepare_mangalmm_llamacpp_runtime.ps1",
            paths,
        )
        self.assertIn(
            "scripts/derive_paddleocr_spotting_mmproj.py",
            paths,
        )
        self.assertIn(
            "scripts/prepare_paddleocr_spotting_llamacpp_runtime.ps1",
            paths,
        )
        self.assertIn(
            "paddleocr_vl_spotting_docker_files/docker-compose.yaml",
            paths,
        )
        self.assertIn("scripts/verify_windows_runtime.py", paths)
        self.assertIn("scripts/bootstrap_windows.ps1", paths)
        self.assertIn("scripts/lib/WindowsBootstrap.psm1", paths)
        self.assertIn("scripts/lib/ManagedRuntimeModelSource.psm1", paths)
        self.assertIn("scripts/prepare_hunyuanocr_llamacpp_runtime.ps1", paths)
        self.assertIn("docs/setup/quickstart-ko.md", paths)
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
        for secret in (
            b"AK" + b"IA1234567890ABCDEF",
            b"github_" + b"pat_1234567890abcdefghijklmnopqrstuvwxyz_ABCD",
            b"sk-" + b"proj-1234567890abcdefghijklmnopqrstuvwxyz",
            b"xox" + b"b-1234567890-abcdefghijklmnop",
        ):
            with self.subTest(secret_prefix=secret[:12]):
                with self.assertRaises(RuntimeError):
                    release._scan_release_bytes("README.md", secret)

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
        expected_runtime = {
            "run_comic.bat": "cuda12",
            "run_comic_cuda13.bat": "cuda13",
        }
        for launcher, runtime in expected_runtime.items():
            text = (ROOT / launcher).read_text(encoding="utf-8")
            self.assertIn('if /I "%COMIC_VERIFY_ONLY%"=="1"', text)
            self.assertIn("scripts\\bootstrap_windows.ps1", text)
            self.assertIn(f"-Runtime {runtime}", text)
            self.assertNotIn("pip install", text)

        bootstrap = (ROOT / "scripts" / "bootstrap_windows.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("prepare_gemma_runtime.ps1", bootstrap)
        self.assertIn("prepare_hunyuanocr_llamacpp_runtime.ps1", bootstrap)
        self.assertIn("prepare_paddleocr_llamacpp_runtime.ps1", bootstrap)
        self.assertNotIn("prepare_mangalmm_llamacpp_runtime.ps1", bootstrap)
        self.assertNotIn("prepare_paddleocr_spotting_llamacpp_runtime.ps1", bootstrap)
        module = (ROOT / "scripts" / "lib" / "WindowsBootstrap.psm1").read_text(
            encoding="utf-8"
        )
        self.assertIn("$env:PYTHONNOUSERSITE = '1'", module)
        self.assertIn("$env:PYTHONHOME = ''", module)
        self.assertIn("$env:PYTHONPATH = ''", module)
        self.assertIn("include-system-site-packages", bootstrap)
        self.assertIn("Docker model volume is not installed yet", bootstrap)
        self.assertIn(
            "$SkipRuntimeSetup = [bool]$env:COMIC_SKIP_STARTUP_MODELS",
            bootstrap,
        )
        self.assertNotIn("$env:COMIC_SMOKE_EXIT_MS", bootstrap)
        self.assertNotIn("llama.cpp Docker image pull", bootstrap)

    def test_bootstrap_configuration_keeps_cuda_variants_separate(self) -> None:
        bootstrap = (ROOT / "scripts" / "bootstrap_windows.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("venv = '.venv-win'", bootstrap)
        self.assertIn("venv = '.venv-win-cuda13'", bootstrap)
        self.assertIn("llama.cpp:server-cuda'", bootstrap)
        self.assertIn("llama.cpp:server-cuda13'", bootstrap)
        self.assertIn("fallback_llama_image", bootstrap)
        self.assertIn("preferred llama.cpp image failed", bootstrap)
        self.assertLess(
            bootstrap.index("label = 'HunyuanOCR'"),
            bootstrap.index("label = 'PaddleOCR VL'"),
        )
        self.assertLess(
            bootstrap.index("label = 'PaddleOCR VL'"),
            bootstrap.index("label = 'Gemma IQ4_NL'"),
        )
        self.assertNotIn("windows_bootstrap.json", bootstrap)

    def test_release_dependency_closure_rejects_missing_local_targets(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "PowerShell module dependency"):
            release.validate_release_dependency_closure(
                {
                    "scripts/example.ps1": b"Import-Module (Join-Path $PSScriptRoot 'lib/Missing.psm1')",
                }
            )
        with self.assertRaisesRegex(RuntimeError, "Markdown dependency"):
            release.validate_release_dependency_closure(
                {"README.md": b"[setup](docs/setup/missing.md)"}
            )

    def test_runtime_requirements_are_exact_and_complete(self) -> None:
        for requirements_name, expected_cuda in (
            ("requirements-cuda12.txt", "2.11.0+cu128"),
            ("requirements-cuda13.txt", "2.11.0+cu130"),
        ):
            with self.subTest(requirements_name=requirements_name):
                pinned = runtime_verifier.load_pinned_requirements(
                    [ROOT / requirements_name]
                )
                self.assertEqual(len(pinned), 36)
                self.assertEqual(pinned["torch"][1], expected_cuda)
                self.assertEqual(pinned["pillow"][1], "12.3.0")
                self.assertEqual(pinned["setuptools"][1], "80.9.0")
                self.assertEqual(pinned["msgpack"][1], "1.2.1")
                self.assertEqual(pinned["py7zr"][1], "1.1.3")
                self.assertEqual(pinned["pyside6"][1], "6.11.0")
                self.assertEqual(pinned["send2trash"][1], "2.1.0")
                self.assertEqual(pinned["pikepdf"][1], "10.5.1")
                self.assertEqual(pinned["pypdfium2"][1], "5.7.0")
                self.assertNotIn("pdfplumber", pinned)

    def test_pyproject_matches_pdf_runtime_pins(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        dependencies = set(project["project"]["dependencies"])

        self.assertIn("pikepdf==10.5.1", dependencies)
        self.assertIn("pypdfium2==5.7.0", dependencies)
        self.assertFalse(any(item.startswith("pdfplumber") for item in dependencies))

    def test_runtime_requirement_parser_rejects_floating_versions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            requirements_path = Path(temp_dir) / "requirements.txt"
            requirements_path.write_text("example>=1.0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact NAME==VERSION pin"):
                runtime_verifier.load_pinned_requirements([requirements_path])

    def test_runtime_verifier_reports_missing_and_mismatched_packages(self) -> None:
        pinned = {
            "alpha": ("alpha", "1.0"),
            "beta": ("beta", "2.0"),
            "gamma": ("gamma", "3.0"),
        }

        def fake_version(name: str) -> str:
            if name == "alpha":
                return "1.0"
            if name == "beta":
                return "9.0"
            raise importlib.metadata.PackageNotFoundError(name)

        with mock.patch.object(
            runtime_verifier.importlib.metadata,
            "version",
            side_effect=fake_version,
        ):
            errors = runtime_verifier.verify_installed_requirements(pinned)

        self.assertEqual(
            errors,
            [
                "beta expected 2.0, found 9.0",
                "gamma is not installed",
            ],
        )

    def test_managed_docker_ports_are_loopback_only(self) -> None:
        compose_paths = (
            ROOT / "docker-compose.yaml",
            ROOT / "hunyuanocr_docker_files" / "docker-compose.yaml",
            ROOT / "mangalmm_docker_files" / "docker-compose.yaml",
            ROOT / "paddleocr_vl_docker_files" / "docker-compose.yaml",
            (
                ROOT
                / "paddleocr_vl_spotting_docker_files"
                / "docker-compose.yaml"
            ),
        )
        published_ports: list[str] = []
        for compose_path in compose_paths:
            payload = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
            for service in payload.get("services", {}).values():
                published_ports.extend(str(item) for item in service.get("ports", []))

        self.assertEqual(len(published_ports), 5)
        self.assertTrue(
            all(port.startswith("127.0.0.1:") for port in published_ports),
            published_ports,
        )

    def test_workflow_actions_are_pinned_to_commit_shas(self) -> None:
        for workflow_path in (ROOT / ".github" / "workflows").glob("*.yml"):
            text = workflow_path.read_text(encoding="utf-8")
            for line in text.splitlines():
                match = re.search(r"\buses:\s*[^@\s]+@([^#\s]+)", line)
                if match is None:
                    continue
                self.assertRegex(
                    match.group(1),
                    r"^[0-9a-f]{40}$",
                    f"{workflow_path.name}: {line.strip()}",
                )

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
        self.assertIsNone(
            select_release_asset_url(
                {
                    "assets": [
                        {
                            "name": "legacy-installer.exe",
                            "browser_download_url": "https://example.invalid/legacy.exe",
                        }
                    ]
                },
                "Windows",
                "1.1.0",
            )
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
