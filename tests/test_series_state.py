from __future__ import annotations

import os
import tempfile
import unittest

from app.projects.project_types import (
    PROJECT_KIND_SERIES,
    PROJECT_KIND_SINGLE,
    ensure_project_extension,
    has_project_file_extension,
    is_series_project_file,
    is_single_project_file,
    project_file_filter_for_kind,
)
from app.projects.series_state_v1 import (
    create_series_project,
    filter_series_candidate_paths,
    load_series_project,
    materialize_series_child_project,
    normalize_series_global_settings,
    normalize_series_queue_runtime,
    normalize_series_recovery_state,
    normalize_series_settings,
    relative_series_source_path,
    scan_series_source_files,
)


class SeriesStateTests(unittest.TestCase):
    def test_scan_series_source_files_finds_supported_inputs_and_skips_series_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            nested = os.path.join(temp_dir, "nested", "chapter")
            os.makedirs(nested, exist_ok=True)
            keep_paths = [
                os.path.join(temp_dir, "001.png"),
                os.path.join(nested, "002.jpg"),
                os.path.join(nested, "existing.ctpr"),
            ]
            skip_paths = [
                os.path.join(temp_dir, "ignore.txt"),
                os.path.join(nested, "series.seriesctpr"),
            ]
            for path in keep_paths + skip_paths:
                with open(path, "wb") as fh:
                    fh.write(b"x")

            result = scan_series_source_files(temp_dir)

            self.assertEqual(result, sorted(os.path.abspath(path) for path in keep_paths))
            self.assertNotIn(os.path.abspath(skip_paths[1]), result)

    def test_relative_series_source_path_uses_root_relative_path(self) -> None:
        root = os.path.join("C:/tmp", "series-root")
        target = os.path.join(root, "nested", "page.png")
        self.assertEqual(
            relative_series_source_path(root, target).replace("\\", "/"),
            "nested/page.png",
        )

    def test_normalize_series_settings_clamps_and_defaults(self) -> None:
        normalized = normalize_series_settings(
            {
                "queue_failure_policy": "unknown",
                "retry_count": -5,
                "retry_delay_sec": -3,
                "auto_open_failed_child": 0,
            }
        )
        self.assertEqual(normalized["queue_failure_policy"], "stop")
        self.assertEqual(normalized["retry_count"], 0)
        self.assertEqual(normalized["retry_delay_sec"], 0)
        self.assertFalse(normalized["auto_open_failed_child"])
        self.assertTrue(normalized["resume_from_first_incomplete"])

    def test_normalize_series_global_settings_sanitizes_fields(self) -> None:
        normalized = normalize_series_global_settings(
            {
                "source_language": " Japanese ",
                "target_language": " Korean ",
                "ocr": " paddleocr_vl ",
                "translator": "gemma_local",
                "workflow_mode": "stage_batched_pipeline",
                "use_gpu": 0,
            }
        )
        self.assertEqual(normalized["source_language"], "Japanese")
        self.assertEqual(normalized["target_language"], "Korean")
        self.assertEqual(normalized["ocr"], "paddleocr_vl")
        self.assertEqual(normalized["workflow_mode"], "stage_batched_pipeline")
        self.assertFalse(normalized["use_gpu"])

    def test_normalize_series_queue_runtime_defaults_to_idle_pause_safe_shape(self) -> None:
        runtime = normalize_series_queue_runtime(
            {
                "queue_state": "unknown",
                "pause_requested": 1,
                "pending_item_ids": ["a", "", None],
                "completed_item_ids": ["b", ""],
                "failed_item_ids": ["c", ""],
                "skipped_item_ids": ["d", ""],
                "retry_remaining_by_item": {"x": -1, "": 3},
            }
        )
        self.assertEqual(runtime["queue_state"], "idle")
        self.assertTrue(runtime["pause_requested"])
        self.assertEqual(runtime["pending_item_ids"], ["a"])
        self.assertEqual(runtime["completed_item_ids"], ["b"])
        self.assertEqual(runtime["failed_item_ids"], ["c"])
        self.assertEqual(runtime["skipped_item_ids"], ["d"])
        self.assertEqual(runtime["retry_remaining_by_item"], {"x": 0})

    def test_filter_series_candidate_paths_blocks_existing_and_duplicate_inputs(self) -> None:
        filtered = filter_series_candidate_paths(
            existing_source_paths=["/tmp/a.png"],
            candidate_paths=["/tmp/a.png", "/tmp/b.png", "/tmp/b.png", "/tmp/c.ctpr"],
        )
        self.assertEqual(filtered["accepted"], [os.path.abspath("/tmp/b.png"), os.path.abspath("/tmp/c.ctpr")])
        self.assertEqual(filtered["skipped_existing"], [os.path.abspath("/tmp/a.png")])
        self.assertEqual(filtered["skipped_duplicates"], [os.path.abspath("/tmp/b.png")])

    def test_normalize_series_recovery_state_converts_running_to_paused(self) -> None:
        manifest = {
            "series_queue_runtime": {
                "queue_state": "running",
                "active_item_id": "item-2",
                "pending_item_ids": ["item-3"],
                "completed_item_ids": ["item-1"],
            }
        }
        items = [
            {"series_item_id": "item-1", "queue_index": 1, "status": "done"},
            {"series_item_id": "item-2", "queue_index": 2, "status": "running"},
            {"series_item_id": "item-3", "queue_index": 3, "status": "pending"},
        ]
        next_manifest, next_items, changed = normalize_series_recovery_state(manifest, items)
        self.assertTrue(changed)
        self.assertEqual(next_manifest["series_queue_runtime"]["queue_state"], "paused")
        self.assertIsNone(next_manifest["series_queue_runtime"]["active_item_id"])
        self.assertEqual(next_manifest["series_queue_runtime"]["pending_item_ids"], ["item-2", "item-3"])
        self.assertEqual(
            [item["status"] for item in next_items],
            ["done", "pending", "pending"],
        )

    def test_project_type_helpers_cover_series_extension(self) -> None:
        self.assertTrue(has_project_file_extension("test.ctpr"))
        self.assertTrue(has_project_file_extension("test.seriesctpr"))
        self.assertTrue(is_single_project_file("test.ctpr"))
        self.assertTrue(is_series_project_file("test.seriesctpr"))
        self.assertEqual(
            ensure_project_extension("/tmp/series", ".seriesctpr"),
            os.path.abspath("/tmp/series.seriesctpr"),
        )
        self.assertIn(".ctpr", project_file_filter_for_kind(PROJECT_KIND_SINGLE))
        self.assertIn(".seriesctpr", project_file_filter_for_kind(PROJECT_KIND_SERIES))

    def test_create_and_load_series_project_with_embedded_child_blob(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            series_path = os.path.join(temp_dir, "queue.seriesctpr")
            payload = b"fake-ctpr-payload"
            blob_hash = "hash-1"
            item = {
                "series_item_id": "item-1",
                "queue_index": 1,
                "display_name": "child.ctpr",
                "source_kind": "ctpr_import",
                "source_origin_path": os.path.join(temp_dir, "child.ctpr"),
                "source_origin_relpath": "child.ctpr",
                "imported_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
                "status": "pending",
                "embedded_project_blob_hash": blob_hash,
                "child_page_count": None,
            }
            project = {
                "project_hash": blob_hash,
                "display_name": "child.ctpr",
                "project_size": len(payload),
                "project_blob": payload,
            }
            create_series_project(
                series_path,
                root_dir=temp_dir,
                items=[item],
                embedded_projects=[project],
            )

            loaded = load_series_project(series_path)
            self.assertEqual(loaded["manifest"]["series_project_type"], PROJECT_KIND_SERIES)
            self.assertEqual(len(loaded["items"]), 1)

            child_path = materialize_series_child_project(series_path, loaded["items"][0], temp_dir=temp_dir)
            with open(child_path, "rb") as fh:
                self.assertEqual(fh.read(), payload)


if __name__ == "__main__":
    unittest.main()
