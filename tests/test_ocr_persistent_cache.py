from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from modules.ocr.persistent_cache import (
    OCRPersistentResultCache,
    OCRResultCacheRecord,
    apply_raw_ocr_result,
    canonical_json,
    snapshot_raw_ocr_result,
)
from modules.utils.textblock import TextBlock


class OCRPersistentResultCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = Path(self.temp_dir.name) / "paddle-cache.sqlite3"

    def _store(self, **kwargs) -> OCRPersistentResultCache:
        store = OCRPersistentResultCache(self.db_path, **kwargs)
        self.addCleanup(store.close)
        return store

    def test_exact_lookup_round_trip_and_stats(self) -> None:
        store = self._store()
        record = OCRResultCacheRecord(
            cache_key="exact-key",
            identity_json=canonical_json({"jpeg": "abc", "model": "paddle"}),
            result_json=canonical_json(
                {
                    "text": "raw text",
                    "texts": [],
                    "confidence": 0.0,
                    "status": "ok",
                    "empty_reason": "",
                    "attempt_count": 1,
                    "raw_text": "raw text",
                    "sanitized_text": "raw text",
                    "reject_reason": "",
                    "ocr_regions": [],
                    "ocr_crop_bbox": None,
                    "ocr_resize_scale": 1.0,
                }
            ),
        )
        self.assertTrue(store.store_records([record]))

        lookups = store.lookup_many(["exact-key", "missing-key"])
        self.assertTrue(lookups["exact-key"].hit)
        self.assertEqual(lookups["exact-key"].result["text"], "raw text")
        self.assertFalse(lookups["missing-key"].hit)
        self.assertEqual(
            store.stats(),
            {
                "enabled": True,
                "disabled_reason": "",
                "item_count": 1,
                "lookup_hits": 1,
                "lookup_misses": 1,
            },
        )

    def test_retention_keeps_most_recent_records(self) -> None:
        store = self._store(result_cache_limit=2)
        for index in range(3):
            self.assertTrue(
                store.store_records(
                    [
                        OCRResultCacheRecord(
                            cache_key=f"key-{index}",
                            identity_json="{}",
                            result_json=canonical_json({"text": str(index)}),
                        )
                    ]
                )
            )

        lookups = store.lookup_many(["key-0", "key-1", "key-2"])
        self.assertFalse(lookups["key-0"].hit)
        self.assertTrue(lookups["key-1"].hit)
        self.assertTrue(lookups["key-2"].hit)
        self.assertEqual(store.stats()["item_count"], 2)

    def test_lowered_retention_limit_prunes_immediately(self) -> None:
        store = self._store(result_cache_limit=3)
        store.store_records(
            [
                OCRResultCacheRecord(
                    cache_key=f"key-{index}",
                    identity_json="{}",
                    result_json=canonical_json({"text": str(index)}),
                )
                for index in range(3)
            ]
        )

        store.configure_limit(1)

        self.assertEqual(store.stats()["item_count"], 1)

    def test_recent_hit_is_retained_by_lru_pruning(self) -> None:
        store = self._store(result_cache_limit=2)
        for cache_key in ("older", "newer"):
            self.assertTrue(
                store.store_records(
                    [
                        OCRResultCacheRecord(
                            cache_key=cache_key,
                            identity_json="{}",
                            result_json=canonical_json({"text": cache_key}),
                        )
                    ]
                )
            )
        self.assertTrue(store.lookup_many(["older"])["older"].hit)

        self.assertTrue(
            store.store_records(
                [
                    OCRResultCacheRecord(
                        cache_key="newest",
                        identity_json="{}",
                        result_json=canonical_json({"text": "newest"}),
                    )
                ]
            )
        )

        lookups = store.lookup_many(["older", "newer", "newest"])
        self.assertTrue(lookups["older"].hit)
        self.assertFalse(lookups["newer"].hit)
        self.assertTrue(lookups["newest"].hit)

    def test_schema_mismatch_disables_without_deleting_database(self) -> None:
        connection = sqlite3.connect(self.db_path)
        with connection:
            connection.execute(
                "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES('schema_version', '999')"
            )
        connection.close()
        before = self.db_path.read_bytes()

        store = self._store()
        lookup = store.lookup_many(["key"])["key"]

        self.assertTrue(lookup.disabled)
        self.assertFalse(store.enabled)
        self.assertEqual(self.db_path.read_bytes(), before)

    def test_corrupt_database_disables_without_replacing_file(self) -> None:
        original = b"not-a-sqlite-database"
        self.db_path.write_bytes(original)
        store = self._store()

        lookup = store.lookup_many(["key"])["key"]

        self.assertTrue(lookup.disabled)
        self.assertFalse(store.enabled)
        self.assertEqual(self.db_path.read_bytes(), original)

    def test_database_write_lock_disables_only_current_store(self) -> None:
        seed = self._store()
        self.assertTrue(
            seed.store_records(
                [
                    OCRResultCacheRecord(
                        cache_key="locked-key",
                        identity_json="{}",
                        result_json=canonical_json({"text": "raw"}),
                    )
                ]
            )
        )
        seed.close()
        blocker = sqlite3.connect(self.db_path, timeout=0.0)
        blocker.execute("BEGIN IMMEDIATE")
        self.addCleanup(blocker.close)
        locked_store = self._store(timeout_sec=0.0)

        lookup = locked_store.lookup_many(["locked-key"])["locked-key"]

        self.assertTrue(lookup.disabled)
        self.assertFalse(locked_store.enabled)
        blocker.rollback()
        with OCRPersistentResultCache(self.db_path) as recovered:
            self.assertTrue(recovered.lookup_many(["locked-key"])["locked-key"].hit)

    def test_invalid_cached_field_shape_disables_without_rewriting_row(self) -> None:
        seed = self._store()
        self.assertTrue(
            seed.store_records(
                [
                    OCRResultCacheRecord(
                        cache_key="invalid-shape",
                        identity_json="{}",
                        result_json=canonical_json(
                            {
                                "text": "raw",
                                "confidence": 0.0,
                                "attempt_count": 1,
                            }
                        ),
                    )
                ]
            )
        )
        seed.close()
        invalid_result_json = canonical_json(
            {
                "text": "raw",
                "confidence": "not-a-number",
                "attempt_count": 1,
            }
        )
        connection = sqlite3.connect(self.db_path)
        with connection:
            connection.execute(
                "UPDATE ocr_results SET result_json = ? WHERE cache_key = ?",
                (invalid_result_json, "invalid-shape"),
            )
        connection.close()
        store = self._store()

        lookup = store.lookup_many(["invalid-shape"])["invalid-shape"]

        self.assertTrue(lookup.disabled)
        self.assertFalse(store.enabled)
        verifier = sqlite3.connect(self.db_path)
        try:
            stored_json = verifier.execute(
                "SELECT result_json FROM ocr_results WHERE cache_key = ?",
                ("invalid-shape",),
            ).fetchone()[0]
        finally:
            verifier.close()
        self.assertEqual(stored_json, invalid_result_json)

    def test_invalid_new_record_is_rejected_without_poisoning_cache(self) -> None:
        store = self._store()
        self.assertTrue(
            store.store_records(
                [
                    OCRResultCacheRecord(
                        cache_key="valid",
                        identity_json="{}",
                        result_json=canonical_json(
                            {"text": "raw", "confidence": 0.0}
                        ),
                    )
                ]
            )
        )

        self.assertFalse(
            store.store_records(
                [
                    OCRResultCacheRecord(
                        cache_key="invalid",
                        identity_json="{}",
                        result_json=canonical_json(
                            {"text": "raw", "confidence": "broken"}
                        ),
                    )
                ]
            )
        )

        self.assertTrue(store.enabled)
        lookups = store.lookup_many(["valid", "invalid"])
        self.assertTrue(lookups["valid"].hit)
        self.assertFalse(lookups["invalid"].hit)

    def test_export_is_jsonl_and_clear_is_explicit(self) -> None:
        store = self._store()
        store.store_records(
            [
                OCRResultCacheRecord(
                    cache_key="key",
                    identity_json=canonical_json({"jpeg": "abc"}),
                    result_json=canonical_json({"text": "raw"}),
                )
            ]
        )
        output = Path(self.temp_dir.name) / "cache.jsonl"

        self.assertEqual(store.export_jsonl(output), 1)
        exported = json.loads(output.read_text(encoding="utf-8").strip())
        self.assertEqual(exported["cache_key"], "key")
        self.assertEqual(exported["raw_ocr_result"]["text"], "raw")

        store.clear()
        self.assertEqual(store.stats()["item_count"], 0)
        self.assertEqual(store.stats()["lookup_hits"], 0)
        self.assertEqual(store.stats()["lookup_misses"], 0)

    def test_snapshot_and_restore_preserve_raw_diagnostics(self) -> None:
        source = TextBlock(
            text_bbox=np.array([1, 2, 10, 20], dtype=np.int32),
            text="sanitized",
        )
        source.texts = ["sanitized"]
        source.ocr_confidence = 0.75
        source.ocr_status = "ok"
        source.ocr_attempt_count = 1
        source.ocr_raw_text = " raw "
        source.ocr_sanitized_text = "sanitized"
        source.ocr_reject_reason = ""
        source.ocr_regions = [{"text": "sanitized"}]
        payload = snapshot_raw_ocr_result(source)

        restored = TextBlock(
            text_bbox=np.array([1, 2, 10, 20], dtype=np.int32)
        )
        apply_raw_ocr_result(restored, payload)

        self.assertEqual(restored.text, "sanitized")
        self.assertEqual(restored.texts, ["sanitized"])
        self.assertEqual(restored.ocr_raw_text, " raw ")
        self.assertEqual(restored.ocr_regions, [{"text": "sanitized"}])


if __name__ == "__main__":
    unittest.main()
