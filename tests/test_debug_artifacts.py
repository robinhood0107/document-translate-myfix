from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from modules.utils.debug_artifacts import (
    DebugArtifactError,
    active_debug_page_directory,
    active_debug_runtime_directory,
    append_active_raw_response,
    atomic_debug_json,
    finish_debug_artifact_run,
    prepare_debug_artifact_run,
    sanitize_debug_component,
    set_active_debug_run,
)


class _SettingsPage:
    def __init__(self, *, raw_response: bool = False) -> None:
        self.raw_response = raw_response

    def get_gemma_local_server_settings(self) -> dict:
        return {"raw_response_logging": self.raw_response}

    def get_hunyuan_ocr_settings(self) -> dict:
        return {"raw_response_logging": False}

    def get_mangalmm_ocr_settings(self) -> dict:
        return {"raw_response_logging": False}


class _MainPage:
    def __init__(
        self,
        *,
        project_file: str = "",
        export_settings: dict | None = None,
        raw_response: bool = False,
    ) -> None:
        self.project_file = project_file
        self.export_settings = dict(export_settings or {})
        self.export_source_by_path: dict[str, dict[str, str]] = {}
        self.file_handler = SimpleNamespace(archive_info=[])
        self.settings_page = _SettingsPage(raw_response=raw_response)
        self._memlogger = None

    def get_resolved_export_settings(self) -> dict:
        return dict(self.export_settings)


class DebugArtifactRunTests(unittest.TestCase):
    def tearDown(self) -> None:
        set_active_debug_run(None)

    def test_project_debug_run_uses_adjacent_cache_and_safe_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "chapter01.ctpr"
            project.write_text("{}", encoding="utf-8")
            image = Path(tmp) / ".. evil CON?.png"
            main = _MainPage(
                project_file=str(project),
                export_settings={"export_ocr_debug": True},
            )

            run = prepare_debug_artifact_run(
                main,
                [str(image)],
                run_type="batch",
            )

            self.assertIsNotNone(run)
            assert run is not None
            self.assertEqual(run.sidecar_root, Path(f"{project}.cache"))
            page_dir = Path(active_debug_page_directory(main, str(image)))
            self.assertTrue(page_dir.is_dir())
            self.assertEqual(page_dir.parent, run.run_root)
            self.assertNotIn("..", page_dir.name)
            self.assertNotIn("?", page_dir.name)

            finish_debug_artifact_run(main, status="completed")
            payload = json.loads(
                (run.run_root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["page_count"], 1)
            self.assertNotIn(str(project), json.dumps(payload))

    def test_folder_and_archive_sources_use_distinct_ctcache_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_folder = Path(tmp) / "source-folder"
            source_folder.mkdir()
            first = source_folder / "001.png"
            second = source_folder / "002.png"
            first.write_bytes(b"1")
            second.write_bytes(b"2")
            folder_main = _MainPage(
                export_settings={"export_detector_overlay": True}
            )
            folder_run = prepare_debug_artifact_run(
                folder_main,
                [str(first), str(second)],
                run_type="batch",
            )
            self.assertIsNotNone(folder_run)
            assert folder_run is not None
            self.assertEqual(
                folder_run.sidecar_root,
                Path(tmp) / "source-folder.ctcache",
            )
            finish_debug_artifact_run(folder_main, status="completed")

            archive = Path(tmp) / "book.zip"
            archive.write_bytes(b"archive")
            extracted = Path(tmp) / "prepared" / "001.png"
            extracted.parent.mkdir()
            extracted.write_bytes(b"prepared")
            archive_main = _MainPage(
                export_settings={"export_raw_mask": True}
            )
            archive_main.export_source_by_path[str(extracted)] = {
                "kind": "archive",
                "source_path": str(archive),
            }
            archive_run = prepare_debug_artifact_run(
                archive_main,
                [str(extracted)],
                run_type="batch",
            )
            self.assertIsNotNone(archive_run)
            assert archive_run is not None
            self.assertEqual(
                archive_run.sidecar_root,
                Path(f"{archive}.ctcache"),
            )
            finish_debug_artifact_run(archive_main, status="completed")

    def test_raw_response_is_written_only_for_an_active_opted_in_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "001.png"
            image.write_bytes(b"image")
            main = _MainPage(raw_response=True)
            run = prepare_debug_artifact_run(
                main,
                [str(image)],
                run_type="batch",
            )
            self.assertIsNotNone(run)
            assert run is not None

            self.assertTrue(
                append_active_raw_response(
                    "gemma",
                    {"choices": [{"message": {"content": "민감 응답"}}]},
                )
            )
            runtime_path = (
                run.runtime_root / "gemma-raw-responses.jsonl"
            )
            record = json.loads(
                runtime_path.read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(
                record["payload"]["choices"][0]["message"]["content"],
                "민감 응답",
            )

            finish_debug_artifact_run(main, status="cancelled")
            self.assertFalse(
                append_active_raw_response("gemma", {"late": True})
            )
            payload = json.loads(
                run.manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(payload["status"], "cancelled")
            self.assertEqual(
                payload["files"][0]["path"],
                "runtime/gemma-raw-responses.jsonl",
            )

    def test_active_run_rejects_raw_response_without_its_service_toggle(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "001.png"
            image.write_bytes(b"image")
            main = _MainPage(
                export_settings={"export_raw_mask": True},
            )
            run = prepare_debug_artifact_run(
                main,
                [str(image)],
                run_type="batch",
            )
            self.assertIsNotNone(run)
            self.assertFalse(
                append_active_raw_response("gemma", {"secret": True})
            )
            assert run is not None
            self.assertFalse(
                (run.runtime_root / "gemma-raw-responses.jsonl").exists()
            )
            finish_debug_artifact_run(main, status="completed")

    def test_replaced_running_manifest_is_marked_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "001.png"
            image.write_bytes(b"image")
            main = _MainPage(
                export_settings={"export_ocr_debug": True},
            )
            first = prepare_debug_artifact_run(
                main,
                [str(image)],
                run_type="batch",
            )
            assert first is not None
            running = json.loads(
                first.manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(running["status"], "running")

            second = prepare_debug_artifact_run(
                main,
                [str(image)],
                run_type="retry_failed",
            )
            self.assertIsNotNone(second)
            interrupted = json.loads(
                first.manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(interrupted["status"], "interrupted")
            finish_debug_artifact_run(main, status="completed")

    def test_closed_run_rejects_late_writes_and_unknown_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "001.png"
            other = Path(tmp) / "002.png"
            image.write_bytes(b"image")
            other.write_bytes(b"other")
            main = _MainPage(raw_response=True)
            run = prepare_debug_artifact_run(
                main,
                [str(image)],
                run_type="batch",
            )
            assert run is not None

            self.assertEqual(
                active_debug_page_directory(main, str(other)),
                "",
            )
            finish_debug_artifact_run(main, status="completed")

            self.assertFalse(
                run.append_runtime(
                    "gemma",
                    {"late": True},
                    kind="response_json",
                )
            )
            with self.assertRaises(DebugArtifactError):
                run.page_directory(str(image))

    def test_runtime_component_and_file_name_cannot_escape_run_root(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "001.png"
            image.write_bytes(b"image")
            main = _MainPage(raw_response=True)
            run = prepare_debug_artifact_run(
                main,
                [str(image)],
                run_type="batch",
            )
            assert run is not None
            component = Path(active_debug_runtime_directory("../AUX"))
            self.assertEqual(component.parent, run.runtime_root)
            self.assertNotIn("..", component.name)

            with self.assertRaises(DebugArtifactError):
                atomic_debug_json(
                    run.runtime_directory(),
                    "../escape.json",
                    {"secret": "blocked"},
                )
            with self.assertRaises(DebugArtifactError):
                atomic_debug_json(
                    run.runtime_directory(),
                    r"..\escape.json",
                    {"secret": "blocked"},
                )
            self.assertFalse((run.run_root / "escape.json").exists())
            finish_debug_artifact_run(main, status="completed")

    def test_no_diagnostic_toggle_does_not_create_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "pages"
            source.mkdir()
            image = source / "001.png"
            image.write_bytes(b"image")
            main = _MainPage()

            with patch.dict(
                os.environ,
                {
                    "CT_ENABLE_MEMLOG": "",
                    "CT_ENABLE_GPU_BENCH": "",
                    "CT_MANGALMM_DEBUG_ROOT": "",
                },
            ):
                self.assertIsNone(
                    prepare_debug_artifact_run(
                        main,
                        [str(image)],
                        run_type="batch",
                    )
                )
            self.assertFalse((Path(tmp) / "pages.ctcache").exists())

    def test_malformed_source_metadata_fails_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "pages"
            source.mkdir()
            image = source / "001.png"
            image.write_bytes(b"image")
            main = _MainPage(
                export_settings={"export_debug_metadata": True}
            )
            main.export_source_by_path = object()  # type: ignore[assignment]

            self.assertIsNone(
                prepare_debug_artifact_run(
                    main,
                    [str(image)],
                    run_type="batch",
                )
            )
            self.assertIsNone(main._debug_artifact_run)

    def test_symlink_sidecar_fails_open_without_writing_target(self) -> None:
        if not hasattr(os, "symlink"):
            self.skipTest("symlinks are unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "pages"
            source.mkdir()
            image = source / "001.png"
            image.write_bytes(b"image")
            outside = Path(tmp) / "outside"
            outside.mkdir()
            sidecar = Path(tmp) / "pages.ctcache"
            try:
                sidecar.symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("symlink creation is unavailable")
            main = _MainPage(
                export_settings={"export_debug_metadata": True}
            )

            self.assertIsNone(
                prepare_debug_artifact_run(
                    main,
                    [str(image)],
                    run_type="batch",
                )
            )
            self.assertEqual(list(outside.iterdir()), [])

    def test_windows_reserved_components_are_prefixed(self) -> None:
        self.assertEqual(sanitize_debug_component("CON"), "_CON")
        self.assertEqual(sanitize_debug_component("AUX.txt"), "_AUX.txt")
        self.assertEqual(sanitize_debug_component("../AUX?.png"), "AUX_.png")


if __name__ == "__main__":
    unittest.main()
