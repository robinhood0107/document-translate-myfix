from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import hashlib
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import unittest
import zipfile

import msgpack

from app.projects import checkpoint_store as checkpoint_store_module
from app.controllers.projects import ProjectController
from app.projects.checkpoint_store import (
    PROJECT_CHECKPOINT_OBJECT_ROOT,
    PROJECT_CHECKPOINT_REFERENCE_KEY,
    ProjectCheckpointError,
    ProjectCheckpointStore,
    checkpoint_reference_for_save,
    checkpoint_sidecar_path,
    expected_checkpoint_sidecar_name,
    finalize_checkpoint_sidecar_move,
    normalize_checkpoint_reference,
    prepare_checkpoint_sidecar,
    remove_checkpoint_sidecar,
    stage_downstream,
)
from app.projects.project_state_v2 import (
    close_cached_connection,
    load_state_from_proj_file_v2,
    save_state_to_proj_file_v2,
)
from app.projects.project_state import load_state_from_proj_file
from app.projects.project_types import PROJECT_KIND_SINGLE


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _SettingsPage:
    def get_llm_settings(self) -> dict:
        return {"extra_context": ""}


class _BatchReportController:
    def export_latest_report_for_project(self):
        return None

    def import_latest_report_from_project(self, _report, refresh: bool = False):
        return None


class _ProjectMain:
    def __init__(self, project_file: str | None = None) -> None:
        self.image_data = {}
        self.in_memory_history = {}
        self.image_history = {}
        self.curr_img_idx = 0
        self.webtoon_mode = False
        self.image_viewer = SimpleNamespace(webtoon_view_state={})
        self.image_files = []
        self.image_states = {}
        self.current_history_index = {}
        self.displayed_images = set()
        self.loaded_images = []
        self.image_patches = {}
        self.settings_page = _SettingsPage()
        self.batch_report_ctrl = _BatchReportController()
        self.export_source_by_path = {}
        self.project_file = project_file
        self.project_kind = PROJECT_KIND_SINGLE
        self.project_output_preferences = {}


class ProjectCheckpointReferenceTests(unittest.TestCase):
    def test_reference_is_stable_for_save_and_clone_gets_new_identity(self) -> None:
        reference = checkpoint_reference_for_save(None, "chapter.ctpr")
        same = checkpoint_reference_for_save(
            reference,
            "renamed.ctpr",
        )
        clone = checkpoint_reference_for_save(
            reference,
            "copy.ctpr",
            clone_identity=True,
        )

        self.assertEqual(same.project_uuid, reference.project_uuid)
        self.assertEqual(same.cache_id, reference.cache_id)
        self.assertEqual(same.sidecar_name, "renamed.ctpr.cache")
        self.assertNotEqual(clone.project_uuid, reference.project_uuid)
        self.assertNotEqual(clone.cache_id, reference.cache_id)
        self.assertEqual(clone.sidecar_name, "copy.ctpr.cache")

    def test_reference_rejects_path_traversal_and_absolute_sidecar(self) -> None:
        reference = checkpoint_reference_for_save(None, "chapter.ctpr").to_dict()
        for unsafe_name in (
            "../outside.ctpr.cache",
            "nested/outside.ctpr.cache",
            r"nested\outside.ctpr.cache",
            os.path.abspath("outside.ctpr.cache"),
        ):
            with self.subTest(unsafe_name=unsafe_name):
                reference["sidecar_name"] = unsafe_name
                with self.assertRaises(ProjectCheckpointError):
                    normalize_checkpoint_reference(
                        reference,
                        "chapter.ctpr",
                    )

    def test_expected_sidecar_name_keeps_full_project_filename(self) -> None:
        self.assertEqual(
            expected_checkpoint_sidecar_name("chapter01.ctpr"),
            "chapter01.ctpr.cache",
        )

    def test_stage_dag_fans_out_to_inpaint_and_translation_after_ocr(self) -> None:
        self.assertEqual(
            stage_downstream("ocr"),
            ("ocr", "translation", "inpaint", "render"),
        )
        self.assertEqual(
            stage_downstream("inpaint"),
            ("inpaint", "render"),
        )


class ProjectCheckpointStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.project_file = os.path.join(self.temp.name, "chapter.ctpr")
        Path(self.project_file).write_bytes(b"project")
        self.reference = checkpoint_reference_for_save(None, self.project_file)
        self.store = ProjectCheckpointStore(
            self.project_file,
            self.reference,
            enabled=True,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_initialization_creates_documented_sidecar_layout(self) -> None:
        self.assertTrue(self.store.ensure_initialized())

        self.assertTrue(self.store.db_path.is_file())
        self.assertTrue((self.store.sidecar_path / "README.txt").is_file())
        self.assertTrue(
            self.store.sidecar_path.joinpath(
                *PROJECT_CHECKPOINT_OBJECT_ROOT
            ).is_dir()
        )
        connection = sqlite3.connect(self.store.db_path)
        try:
            journal_mode = str(
                connection.execute("PRAGMA journal_mode").fetchone()[0]
            ).lower()
        finally:
            connection.close()
        self.assertEqual(journal_mode, "wal")

    def test_has_stage_records_is_stage_specific(self) -> None:
        self.assertFalse(self.store.has_stage_records("ocr"))
        self.assertFalse(self.store.has_stage_record("page-1", "ocr"))
        self.assertTrue(
            self.store.record_stage(
                "page-1",
                "detection",
                _sha("detection"),
            )
        )
        self.assertFalse(self.store.has_stage_records("ocr"))
        self.assertTrue(
            self.store.record_stage(
                "page-1",
                "ocr",
                _sha("ocr"),
            )
        )
        self.assertTrue(self.store.has_stage_records("ocr"))
        self.assertTrue(self.store.has_stage_record("page-1", "ocr"))
        self.assertFalse(self.store.has_stage_record("page-2", "ocr"))

    def test_record_lookup_and_object_integrity_round_trip(self) -> None:
        object_hash = self.store.put_object(b"lossless artifact")
        self.assertIsNotNone(object_hash)
        fingerprint = _sha("detection-v1")

        stored = self.store.record_stage(
            "page-1",
            "detection",
            fingerprint,
            payload={"boxes": 3, "ordered": True},
            objects={"mask": str(object_hash)},
        )
        hit = self.store.lookup_stage("page-1", "detection", fingerprint)

        self.assertTrue(stored)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.payload, {"boxes": 3, "ordered": True})
        self.assertEqual(hit.objects, {"mask": object_hash})
        self.assertEqual(
            self.store.read_object(str(object_hash)),
            b"lossless artifact",
        )

    def test_changed_upstream_fingerprint_invalidates_only_downstream(self) -> None:
        fingerprints = {
            stage: _sha(stage)
            for stage in ("detection", "ocr", "translation", "inpaint", "render")
        }
        for stage in ("detection", "ocr", "translation", "inpaint", "render"):
            self.assertTrue(
                self.store.record_stage(
                    "page-1",
                    stage,
                    fingerprints[stage],
                )
            )
        self.assertTrue(
            self.store.record_stage(
                "page-2",
                "detection",
                fingerprints["detection"],
            )
        )

        self.assertTrue(
            self.store.record_stage(
                "page-1",
                "ocr",
                _sha("ocr-v2"),
            )
        )

        self.assertIsNotNone(
            self.store.lookup_stage(
                "page-1",
                "detection",
                fingerprints["detection"],
            )
        )
        self.assertIsNone(
            self.store.lookup_stage("page-1", "inpaint", fingerprints["inpaint"])
        )
        self.assertIsNone(
            self.store.lookup_stage(
                "page-1",
                "translation",
                fingerprints["translation"],
            )
        )
        self.assertIsNone(
            self.store.lookup_stage("page-1", "render", fingerprints["render"])
        )
        self.assertIsNotNone(
            self.store.lookup_stage(
                "page-2",
                "detection",
                fingerprints["detection"],
            )
        )

    def test_changed_payload_with_same_fingerprint_invalidates_downstream(self) -> None:
        detection_fingerprint = _sha("detection")
        self.assertTrue(
            self.store.record_stage(
                "page-1",
                "detection",
                detection_fingerprint,
                payload={"boxes": [1]},
            )
        )
        self.assertTrue(
            self.store.record_stage(
                "page-1",
                "ocr",
                _sha("ocr"),
            )
        )
        self.assertTrue(
            self.store.record_stage(
                "page-1",
                "translation",
                _sha("translation"),
            )
        )

        self.assertTrue(
            self.store.record_stage(
                "page-1",
                "detection",
                detection_fingerprint,
                payload={"boxes": [2]},
            )
        )

        self.assertIsNone(
            self.store.lookup_stage("page-1", "ocr", _sha("ocr"))
        )
        self.assertIsNone(
            self.store.lookup_stage(
                "page-1",
                "translation",
                _sha("translation"),
            )
        )

    def test_manual_invalidation_is_page_scoped_and_dag_aware(self) -> None:
        for page_key in ("page-1", "page-2"):
            for stage in ("detection", "ocr", "translation"):
                self.assertTrue(
                    self.store.record_stage(
                        page_key,
                        stage,
                        _sha(f"{page_key}-{stage}"),
                    )
                )

        removed = self.store.invalidate(page_key="page-1", stage="ocr")

        self.assertEqual(removed, 2)
        self.assertIsNotNone(
            self.store.lookup_stage(
                "page-1",
                "detection",
                _sha("page-1-detection"),
            )
        )
        self.assertIsNone(
            self.store.lookup_stage("page-1", "ocr", _sha("page-1-ocr"))
        )
        self.assertIsNotNone(
            self.store.lookup_stage("page-2", "ocr", _sha("page-2-ocr"))
        )

    def test_missing_object_is_cache_miss_without_manifest_rewrite(self) -> None:
        object_hash = self.store.put_object(b"artifact")
        assert object_hash is not None
        fingerprint = _sha("stage")
        self.assertTrue(
            self.store.record_stage(
                "page-1",
                "detection",
                fingerprint,
                objects={"mask": object_hash},
            )
        )
        object_path = (
            self.store.object_root / object_hash[:2] / object_hash
        )
        object_path.unlink()
        before = self.store.db_path.read_bytes()

        self.assertIsNone(
            self.store.lookup_stage("page-1", "detection", fingerprint)
        )
        self.assertEqual(self.store.db_path.read_bytes(), before)
        self.assertFalse(self.store.disabled_reason)

    def test_corrupt_database_is_preserved_and_store_fails_open(self) -> None:
        self.store.sidecar_path.mkdir(parents=True)
        self.store.db_path.write_bytes(b"not sqlite")
        before = self.store.db_path.read_bytes()

        self.assertFalse(self.store.ensure_initialized())

        self.assertEqual(self.store.db_path.read_bytes(), before)
        self.assertTrue(self.store.disabled_reason)

    def test_schema_mismatch_is_preserved_and_store_fails_open(self) -> None:
        self.assertTrue(self.store.ensure_initialized())
        connection = sqlite3.connect(self.store.db_path)
        with connection:
            connection.execute(
                "UPDATE metadata SET value = '999' WHERE key = 'schema_version'"
            )
        connection.close()
        before = self.store.db_path.read_bytes()
        reopened = ProjectCheckpointStore(
            self.project_file,
            self.reference,
            enabled=True,
        )

        self.assertFalse(reopened.ensure_initialized())
        self.assertEqual(reopened.db_path.read_bytes(), before)

    def test_empty_preexisting_database_is_not_initialized_in_place(self) -> None:
        self.store.sidecar_path.mkdir(parents=True)
        self.store.db_path.write_bytes(b"")

        self.assertFalse(self.store.ensure_initialized())

        self.assertEqual(self.store.db_path.read_bytes(), b"")
        self.assertTrue(self.store.disabled_reason)

    def test_locked_database_disables_only_this_store_instance(self) -> None:
        self.assertTrue(self.store.ensure_initialized())
        connection = sqlite3.connect(self.store.db_path, timeout=0.1)
        connection.execute("BEGIN EXCLUSIVE")
        locked_store = ProjectCheckpointStore(
            self.project_file,
            self.reference,
            enabled=True,
            timeout_sec=0.01,
        )
        try:
            self.assertEqual(locked_store.invalidate(), 0)
            self.assertTrue(locked_store.disabled_reason)
        finally:
            connection.rollback()
            connection.close()

        reopened = ProjectCheckpointStore(
            self.project_file,
            self.reference,
            enabled=True,
        )
        self.assertTrue(reopened.ensure_initialized())

    def test_cleanup_removes_only_unreferenced_objects(self) -> None:
        used_hash = self.store.put_object(b"used")
        unused_hash = self.store.put_object(b"unused")
        assert used_hash is not None
        assert unused_hash is not None
        self.assertTrue(
            self.store.record_stage(
                "page-1",
                "detection",
                _sha("detection"),
                objects={"mask": used_hash},
            )
        )

        result = self.store.clean_unused_objects()

        self.assertEqual(result["removed_files"], 1)
        self.assertIsNotNone(self.store.read_object(used_hash))
        self.assertIsNone(self.store.read_object(unused_hash))

    def test_cleanup_never_removes_debug_run_folders(self) -> None:
        debug_run = self.store.sidecar_path / "debug" / "run-test"
        debug_run.mkdir(parents=True)
        manifest = debug_run / "manifest.json"
        manifest.write_text('{"status":"completed"}', encoding="utf-8")
        unused_hash = self.store.put_object(b"unused")
        assert unused_hash is not None

        result = self.store.clean_unused_objects()

        self.assertEqual(result["removed_files"], 1)
        self.assertTrue(manifest.is_file())
        self.assertEqual(
            manifest.read_text(encoding="utf-8"),
            '{"status":"completed"}',
        )

    def test_cleanup_cannot_race_manifest_into_dangling_object(self) -> None:
        object_hash = self.store.put_object(b"concurrent")
        assert object_hash is not None
        entered_verification = threading.Event()
        allow_record = threading.Event()
        record_result: list[bool] = []
        cleanup_result: list[dict[str, int]] = []
        original_sha256_file = checkpoint_store_module._sha256_file

        def blocking_sha256(path: Path) -> str:
            if (
                threading.current_thread().name == "checkpoint-record"
                and not entered_verification.is_set()
            ):
                entered_verification.set()
                self.assertTrue(allow_record.wait(timeout=2.0))
            return original_sha256_file(path)

        record_store = ProjectCheckpointStore(
            self.project_file,
            self.reference,
            enabled=True,
            timeout_sec=2.0,
        )
        clean_store = ProjectCheckpointStore(
            self.project_file,
            self.reference,
            enabled=True,
            timeout_sec=2.0,
        )
        with mock.patch(
            "app.projects.checkpoint_store._sha256_file",
            side_effect=blocking_sha256,
        ):
            record_thread = threading.Thread(
                name="checkpoint-record",
                target=lambda: record_result.append(
                    record_store.record_stage(
                        "page-1",
                        "detection",
                        _sha("concurrent"),
                        objects={"mask": object_hash},
                    )
                ),
            )
            cleanup_thread = threading.Thread(
                name="checkpoint-clean",
                target=lambda: cleanup_result.append(
                    clean_store.clean_unused_objects()
                ),
            )
            record_thread.start()
            self.assertTrue(entered_verification.wait(timeout=2.0))
            cleanup_thread.start()
            time.sleep(0.05)
            self.assertTrue(cleanup_thread.is_alive())
            allow_record.set()
            record_thread.join(timeout=2.0)
            cleanup_thread.join(timeout=2.0)

        self.assertEqual(record_result, [True])
        self.assertEqual(cleanup_result, [{"removed_files": 0, "removed_bytes": 0}])
        self.assertIsNotNone(
            self.store.lookup_stage(
                "page-1",
                "detection",
                _sha("concurrent"),
            )
        )
        self.assertEqual(self.store.read_object(object_hash), b"concurrent")

    def test_invalid_new_manifest_does_not_replace_existing_record(self) -> None:
        fingerprint = _sha("valid")
        self.assertTrue(
            self.store.record_stage(
                "page-1",
                "detection",
                fingerprint,
                payload={"value": "kept"},
            )
        )

        self.assertFalse(
            self.store.record_stage(
                "page-1",
                "detection",
                _sha("invalid"),
                objects={"mask": _sha("missing")},
            )
        )
        reopened = ProjectCheckpointStore(
            self.project_file,
            self.reference,
            enabled=True,
        )
        hit = reopened.lookup_stage("page-1", "detection", fingerprint)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.payload, {"value": "kept"})

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink unsupported")
    def test_symlink_sidecar_is_rejected(self) -> None:
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        sidecar = checkpoint_sidecar_path(
            self.project_file,
            self.reference,
        )
        try:
            os.symlink(outside, sidecar, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation not permitted")

        with self.assertRaises(ProjectCheckpointError):
            ProjectCheckpointStore(
                self.project_file,
                self.reference,
                enabled=True,
            )


class ProjectCheckpointSidecarTransferTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.source_project = os.path.join(self.temp.name, "source.ctpr")
        self.target_project = os.path.join(self.temp.name, "target.ctpr")
        Path(self.source_project).write_bytes(b"source")
        Path(self.target_project).write_bytes(b"target")
        self.source_reference = checkpoint_reference_for_save(
            None,
            self.source_project,
        )
        self.source_store = ProjectCheckpointStore(
            self.source_project,
            self.source_reference,
            enabled=True,
        )
        self.object_hash = self.source_store.put_object(b"artifact")
        assert self.object_hash is not None
        self.assertTrue(
            self.source_store.record_stage(
                "page-1",
                "detection",
                _sha("detection"),
                objects={"mask": self.object_hash},
            )
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_save_as_clone_preserves_data_with_new_identity(self) -> None:
        target_reference = checkpoint_reference_for_save(
            self.source_reference,
            self.target_project,
            clone_identity=True,
        )
        prepared = prepare_checkpoint_sidecar(
            self.source_project,
            self.source_reference,
            self.target_project,
            target_reference,
        )
        prepared.commit()
        target_store = ProjectCheckpointStore(
            self.target_project,
            target_reference,
            enabled=True,
        )

        self.assertNotEqual(
            target_reference.project_uuid,
            self.source_reference.project_uuid,
        )
        self.assertEqual(prepared.linked_files + prepared.copied_files, 1)
        self.assertIsNotNone(
            target_store.lookup_stage(
                "page-1",
                "detection",
                _sha("detection"),
            )
        )
        self.assertTrue(self.source_store.db_path.is_file())

    def test_clone_falls_back_to_copy_when_hardlink_is_unavailable(self) -> None:
        target_reference = checkpoint_reference_for_save(
            self.source_reference,
            self.target_project,
            clone_identity=True,
        )
        with mock.patch(
            "app.projects.checkpoint_store.os.link",
            side_effect=OSError("hardlink unavailable"),
        ):
            prepared = prepare_checkpoint_sidecar(
                self.source_project,
                self.source_reference,
                self.target_project,
                target_reference,
            )
        prepared.commit()

        self.assertEqual(prepared.linked_files, 0)
        self.assertEqual(prepared.copied_files, 1)
        target_store = ProjectCheckpointStore(
            self.target_project,
            target_reference,
            enabled=True,
        )
        self.assertEqual(
            target_store.read_object(self.object_hash),
            b"artifact",
        )

    def test_rollback_restores_preexisting_target_sidecar(self) -> None:
        old_target_reference = checkpoint_reference_for_save(
            None,
            self.target_project,
        )
        old_store = ProjectCheckpointStore(
            self.target_project,
            old_target_reference,
            enabled=True,
        )
        self.assertTrue(
            old_store.record_stage(
                "old-page",
                "detection",
                _sha("old"),
            )
        )
        new_target_reference = checkpoint_reference_for_save(
            self.source_reference,
            self.target_project,
            clone_identity=True,
        )

        prepared = prepare_checkpoint_sidecar(
            self.source_project,
            self.source_reference,
            self.target_project,
            new_target_reference,
        )
        prepared.rollback()
        restored_store = ProjectCheckpointStore(
            self.target_project,
            old_target_reference,
            enabled=True,
        )

        self.assertIsNotNone(
            restored_store.lookup_stage(
                "old-page",
                "detection",
                _sha("old"),
            )
        )

    def test_remove_requires_exact_matching_identity(self) -> None:
        wrong_reference = checkpoint_reference_for_save(
            None,
            self.source_project,
        )

        self.assertFalse(
            remove_checkpoint_sidecar(
                self.source_project,
                wrong_reference,
            )
        )
        self.assertTrue(self.source_store.sidecar_path.is_dir())
        self.assertTrue(
            remove_checkpoint_sidecar(
                self.source_project,
                self.source_reference,
            )
        )
        self.assertFalse(self.source_store.sidecar_path.exists())

    def test_move_preserves_source_until_target_identity_is_verified(self) -> None:
        target_reference = checkpoint_reference_for_save(
            self.source_reference,
            self.target_project,
        )

        self.assertFalse(
            finalize_checkpoint_sidecar_move(
                self.source_project,
                self.source_reference,
                self.target_project,
                target_reference,
            )
        )
        self.assertTrue(self.source_store.sidecar_path.is_dir())

        prepared = prepare_checkpoint_sidecar(
            self.source_project,
            self.source_reference,
            self.target_project,
            target_reference,
        )
        prepared.commit()

        self.assertTrue(
            finalize_checkpoint_sidecar_move(
                self.source_project,
                self.source_reference,
                self.target_project,
                target_reference,
            )
        )
        self.assertFalse(self.source_store.sidecar_path.exists())


class ProjectCheckpointProjectStateTests(unittest.TestCase):
    def test_old_v1_project_opens_without_modifying_project_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = os.path.join(temp_dir, "legacy.ctpr")
            state = {
                "current_image_index": 0,
                "webtoon_mode": False,
                "webtoon_view_state": {},
                "original_image_files": [],
                "image_states": {},
                "current_history_index": {},
                "displayed_images": [],
                "loaded_images": [],
                "llm_extra_context": "",
            }
            with zipfile.ZipFile(project_path, "w") as archive:
                archive.writestr(
                    "state.msgpack",
                    msgpack.packb(state, use_bin_type=True),
                )
            before = Path(project_path).read_bytes()
            restored = _ProjectMain()
            try:
                load_state_from_proj_file(restored, project_path)

                self.assertEqual(Path(project_path).read_bytes(), before)
                self.assertEqual(
                    restored.project_checkpoint_reference["sidecar_name"],
                    "legacy.ctpr.cache",
                )
                self.assertFalse(
                    checkpoint_sidecar_path(
                        project_path,
                        restored.project_checkpoint_reference,
                    ).exists()
                )
            finally:
                shutil_target = getattr(restored, "temp_dir", "")
                if shutil_target:
                    shutil.rmtree(shutil_target, ignore_errors=True)

    def test_v2_round_trip_persists_only_relative_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = os.path.join(temp_dir, "chapter.ctpr")
            source = _ProjectMain(project_path)
            restored = _ProjectMain()
            try:
                save_state_to_proj_file_v2(source, project_path)
                load_state_from_proj_file_v2(restored, project_path)

                reference = restored.project_checkpoint_reference
                self.assertEqual(
                    reference["sidecar_name"],
                    "chapter.ctpr.cache",
                )
                self.assertFalse(os.path.isabs(reference["sidecar_name"]))
                self.assertFalse(
                    checkpoint_sidecar_path(project_path, reference).exists()
                )
            finally:
                close_cached_connection(project_path)

    def test_old_v2_without_reference_opens_and_adds_reference_on_next_save(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            old_path = os.path.join(temp_dir, "old.ctpr")
            saved_path = os.path.join(temp_dir, "saved.ctpr")
            source = _ProjectMain(old_path)
            save_state_to_proj_file_v2(source, old_path)
            connection = sqlite3.connect(old_path)
            row = connection.execute(
                "SELECT manifest_blob FROM project_manifest WHERE id = 1"
            ).fetchone()
            manifest = msgpack.unpackb(row[0], strict_map_key=True)
            manifest.pop(PROJECT_CHECKPOINT_REFERENCE_KEY, None)
            with connection:
                connection.execute(
                    "UPDATE project_manifest SET manifest_blob = ? WHERE id = 1",
                    (msgpack.packb(manifest, use_bin_type=True),),
                )
            connection.close()
            close_cached_connection(old_path)

            restored = _ProjectMain()
            try:
                load_state_from_proj_file_v2(restored, old_path)
                generated = restored.project_checkpoint_reference
                self.assertEqual(generated["sidecar_name"], "old.ctpr.cache")

                save_state_to_proj_file_v2(restored, saved_path)
                saved_connection = sqlite3.connect(saved_path)
                saved_row = saved_connection.execute(
                    "SELECT manifest_blob FROM project_manifest WHERE id = 1"
                ).fetchone()
                saved_manifest = msgpack.unpackb(
                    saved_row[0],
                    strict_map_key=True,
                )
                saved_connection.close()
                self.assertIn(
                    PROJECT_CHECKPOINT_REFERENCE_KEY,
                    saved_manifest,
                )
                self.assertEqual(
                    saved_manifest[PROJECT_CHECKPOINT_REFERENCE_KEY][
                        "sidecar_name"
                    ],
                    "saved.ctpr.cache",
                )
            finally:
                close_cached_connection(old_path)
                close_cached_connection(saved_path)

    def test_controller_save_as_clones_sidecar_and_keeps_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, "source.ctpr")
            target_path = os.path.join(temp_dir, "copy.ctpr")
            main = _ProjectMain(source_path)
            source_reference = checkpoint_reference_for_save(
                None,
                source_path,
            )
            main.project_checkpoint_reference = source_reference.to_dict()
            source_store = ProjectCheckpointStore(
                source_path,
                source_reference,
                enabled=True,
            )
            self.assertTrue(
                source_store.record_stage(
                    "page-1",
                    "detection",
                    _sha("detection"),
                )
            )
            controller = ProjectController.__new__(ProjectController)
            controller.main = main

            with mock.patch.object(
                controller,
                "_project_checkpoint_enabled",
                return_value=False,
            ):
                controller.save_project(
                    target_path,
                    "clone",
                    source_path,
                )

            target_reference = normalize_checkpoint_reference(
                main.project_checkpoint_reference,
                target_path,
            )
            assert target_reference is not None
            target_store = ProjectCheckpointStore(
                target_path,
                target_reference,
                enabled=True,
            )
            self.assertNotEqual(
                source_reference.project_uuid,
                target_reference.project_uuid,
            )
            self.assertTrue(source_store.sidecar_path.is_dir())
            self.assertIsNotNone(
                target_store.lookup_stage(
                    "page-1",
                    "detection",
                    _sha("detection"),
                )
            )
            close_cached_connection(target_path)

    def test_controller_regular_save_initializes_enabled_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = os.path.join(temp_dir, "active.ctpr")
            main = _ProjectMain(project_path)
            controller = ProjectController.__new__(ProjectController)
            controller.main = main

            with mock.patch.object(
                controller,
                "_project_checkpoint_enabled",
                return_value=True,
            ):
                controller.save_project(
                    project_path,
                    "same",
                    project_path,
                )

            reference = normalize_checkpoint_reference(
                main.project_checkpoint_reference,
                project_path,
            )
            assert reference is not None
            store = ProjectCheckpointStore(
                project_path,
                reference,
                enabled=True,
            )
            self.assertTrue(store.db_path.is_file())
            self.assertTrue(
                (store.sidecar_path / "README.txt").is_file()
            )
            close_cached_connection(project_path)

    def test_checkpoint_initialization_failure_does_not_fail_project_save(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = os.path.join(temp_dir, "active.ctpr")
            main = _ProjectMain(project_path)
            controller = ProjectController.__new__(ProjectController)
            controller.main = main

            with (
                mock.patch.object(
                    controller,
                    "_project_checkpoint_enabled",
                    return_value=True,
                ),
                mock.patch(
                    "app.controllers.projects.ProjectCheckpointStore",
                    side_effect=ProjectCheckpointError("unsafe sidecar"),
                ),
            ):
                controller.save_project(
                    project_path,
                    "same",
                    project_path,
                )

            self.assertTrue(os.path.isfile(project_path))
            close_cached_connection(project_path)

    def test_controller_move_keeps_identity_until_source_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, "source.ctpr")
            target_path = os.path.join(temp_dir, "renamed.ctpr")
            main = _ProjectMain(source_path)
            source_reference = checkpoint_reference_for_save(
                None,
                source_path,
            )
            main.project_checkpoint_reference = source_reference.to_dict()
            source_store = ProjectCheckpointStore(
                source_path,
                source_reference,
                enabled=True,
            )
            self.assertTrue(
                source_store.record_stage(
                    "page-1",
                    "detection",
                    _sha("detection"),
                )
            )
            controller = ProjectController.__new__(ProjectController)
            controller.main = main

            with mock.patch.object(
                controller,
                "_project_checkpoint_enabled",
                return_value=False,
            ):
                controller.save_project(
                    target_path,
                    "move",
                    source_path,
                )

            target_reference = normalize_checkpoint_reference(
                main.project_checkpoint_reference,
                target_path,
            )
            assert target_reference is not None
            self.assertEqual(
                target_reference.project_uuid,
                source_reference.project_uuid,
            )
            self.assertEqual(
                target_reference.cache_id,
                source_reference.cache_id,
            )
            self.assertTrue(source_store.sidecar_path.is_dir())
            self.assertTrue(
                remove_checkpoint_sidecar(
                    source_path,
                    source_reference,
                )
            )
            self.assertFalse(source_store.sidecar_path.exists())
            close_cached_connection(target_path)

    def test_recovery_snapshot_does_not_change_active_project_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = os.path.join(temp_dir, "active.ctpr")
            recovery_path = os.path.join(temp_dir, "recovery.ctpr")
            main = _ProjectMain(project_path)
            reference = checkpoint_reference_for_save(
                None,
                project_path,
            )
            main.project_checkpoint_reference = reference.to_dict()
            controller = ProjectController.__new__(ProjectController)
            controller.main = main

            with mock.patch.object(
                controller,
                "_project_checkpoint_enabled",
                return_value=True,
            ):
                controller.save_project(recovery_path)

            self.assertEqual(
                main.project_checkpoint_reference,
                reference.to_dict(),
            )
            recovery_reference = checkpoint_reference_for_save(
                reference,
                recovery_path,
            )
            self.assertFalse(
                checkpoint_sidecar_path(
                    recovery_path,
                    recovery_reference,
                ).exists()
            )
            close_cached_connection(recovery_path)


if __name__ == "__main__":
    unittest.main()
