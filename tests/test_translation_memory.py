from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from modules.translation.translation_memory import (
    ResultCacheRecord,
    TranslationMemoryStore,
    canonical_json,
    normalize_exact_tm_text,
)


class TranslationMemoryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = Path(self.temp_dir.name) / "translation-memory.sqlite3"

    def _store(self, **kwargs) -> TranslationMemoryStore:
        store = TranslationMemoryStore(self.db_path, **kwargs)
        self.addCleanup(store.close)
        return store

    def test_exact_normalization_is_conservative(self) -> None:
        self.assertEqual(
            normalize_exact_tm_text("  A\u030A\r\nB  "),
            "Å\nB",
        )
        self.assertNotEqual(
            normalize_exact_tm_text("two  spaces"),
            normalize_exact_tm_text("two spaces"),
        )

    def test_result_cache_reports_hit_and_stale_identity(self) -> None:
        store = self._store()
        record = ResultCacheRecord(
            cache_key="exact-key",
            scope_key="scope-key",
            identity_json=canonical_json({"model": "a"}),
            source_text="source",
            translation="translation",
            metadata_json=canonical_json({"guard": {"changed": True}}),
        )
        self.assertTrue(store.store_results([record]))

        hit = store.lookup_result("exact-key", "scope-key")
        self.assertTrue(hit.hit)
        self.assertEqual(hit.translation, "translation")
        self.assertEqual(hit.metadata, {"guard": {"changed": True}})

        stale = store.lookup_result("different-key", "scope-key")
        self.assertFalse(stale.hit)
        self.assertTrue(stale.stale_reject)

        miss = store.lookup_result("different-key", "different-scope")
        self.assertFalse(miss.hit)
        self.assertFalse(miss.stale_reject)

    def test_result_cache_retention_keeps_most_recent_entries(self) -> None:
        store = self._store(result_cache_limit=2)
        for index in range(3):
            self.assertTrue(
                store.store_results(
                    [
                        ResultCacheRecord(
                            cache_key=f"key-{index}",
                            scope_key=f"scope-{index}",
                            identity_json="{}",
                            source_text=f"source-{index}",
                            translation=f"translation-{index}",
                        )
                    ]
                )
            )

        self.assertFalse(store.lookup_result("key-0", "missing-scope").hit)
        self.assertTrue(store.lookup_result("key-1", "scope-1").hit)
        self.assertTrue(store.lookup_result("key-2", "scope-2").hit)
        self.assertEqual(store.stats()["result_cache_entries"], 2)

    def test_result_cache_touch_updates_lru_in_one_explicit_write(self) -> None:
        store = self._store(result_cache_limit=2)
        self.assertTrue(
            store.store_results(
                [
                    ResultCacheRecord(
                        cache_key="key-0",
                        scope_key="scope-0",
                        identity_json="{}",
                        source_text="source-0",
                        translation="translation-0",
                    ),
                    ResultCacheRecord(
                        cache_key="key-1",
                        scope_key="scope-1",
                        identity_json="{}",
                        source_text="source-1",
                        translation="translation-1",
                    ),
                ]
            )
        )
        self.assertTrue(store.lookup_result("key-0", "scope-0").hit)
        self.assertTrue(store.store_results([], touched_cache_keys=["key-0"]))
        self.assertTrue(
            store.store_results(
                [
                    ResultCacheRecord(
                        cache_key="key-2",
                        scope_key="scope-2",
                        identity_json="{}",
                        source_text="source-2",
                        translation="translation-2",
                    )
                ]
            )
        )

        self.assertTrue(store.lookup_result("key-0", "scope-0").hit)
        self.assertFalse(store.lookup_result("key-1", "missing-scope").hit)
        self.assertTrue(store.lookup_result("key-2", "scope-2").hit)

    def test_unapproved_candidate_never_bypasses_translation(self) -> None:
        store = self._store()
        self.assertTrue(
            store.record_tm_candidate(
                " source\r\ntext ",
                "translation",
                "Japanese",
                "Korean",
            )
        )
        self.assertFalse(
            store.lookup_exact_tm(
                "source\ntext",
                "Japanese",
                "Korean",
            ).hit
        )

        entry_id = store.list_tm_entries()[0]["id"]
        self.assertEqual(store.set_approved([entry_id], True), 1)
        approved_revision = store.get_tm_revision()
        self.assertEqual(store.set_approved([entry_id], True), 0)
        self.assertEqual(store.get_tm_revision(), approved_revision)
        approved = store.lookup_exact_tm(
            "source\ntext",
            "Japanese",
            "Korean",
        )
        self.assertTrue(approved.hit)
        self.assertEqual(approved.translation, "translation")
        self.assertEqual(len(approved.entry_ids), 1)
        self.assertTrue(
            store.store_results(
                [],
                touched_tm_entry_ids=approved.entry_ids,
            )
        )
        self.assertEqual(store.list_tm_entries()[0]["use_count"], 1)

    def test_ambiguous_approved_entries_do_not_bypass_translation(self) -> None:
        store = self._store()
        for translation in ("first", "second"):
            self.assertTrue(
                store.record_tm_candidate(
                    "same source",
                    translation,
                    "Japanese",
                    "Korean",
                )
            )
        ids = [entry["id"] for entry in store.list_tm_entries()]
        self.assertEqual(store.set_approved(ids, True), 2)

        lookup = store.lookup_exact_tm("same source", "Japanese", "Korean")
        self.assertFalse(lookup.hit)
        self.assertTrue(lookup.ambiguous)

    def test_exact_tm_does_not_collapse_internal_whitespace(self) -> None:
        store = self._store()
        store.record_tm_candidate(
            "two  spaces",
            "translation",
            "English",
            "Korean",
        )
        entry_id = store.list_tm_entries()[0]["id"]
        store.set_approved([entry_id], True)

        self.assertFalse(
            store.lookup_exact_tm("two spaces", "English", "Korean").hit
        )

    def test_candidate_retention_never_prunes_approved_entries(self) -> None:
        store = self._store(candidate_limit=1)
        store.record_tm_candidate("approved", "kept", "Japanese", "Korean")
        approved_id = store.list_tm_entries()[0]["id"]
        store.set_approved([approved_id], True)

        store.record_tm_candidate("old candidate", "old", "Japanese", "Korean")
        store.record_tm_candidate("new candidate", "new", "Japanese", "Korean")

        entries = store.list_tm_entries()
        self.assertEqual(sum(1 for entry in entries if entry["approved"]), 1)
        self.assertEqual(sum(1 for entry in entries if not entry["approved"]), 1)
        self.assertIn("approved", {entry["source_text"] for entry in entries})

    def test_export_import_preserves_explicit_approval(self) -> None:
        store = self._store()
        store.record_tm_candidate("source", "translation", "Japanese", "Korean")
        entry_id = store.list_tm_entries()[0]["id"]
        store.set_approved([entry_id], True)
        export_path = Path(self.temp_dir.name) / "exact-tm.json"

        self.assertEqual(store.export_tm(export_path), 1)
        exported = json.loads(export_path.read_text(encoding="utf-8"))
        self.assertTrue(exported["entries"][0]["approved"])

        imported_path = Path(self.temp_dir.name) / "imported.sqlite3"
        imported_store = TranslationMemoryStore(imported_path)
        self.addCleanup(imported_store.close)
        self.assertEqual(imported_store.import_tm(export_path), 1)
        self.assertTrue(
            imported_store.lookup_exact_tm(
                "source",
                "Japanese",
                "Korean",
            ).hit
        )

    def test_import_rejects_non_boolean_approval_values(self) -> None:
        import_path = Path(self.temp_dir.name) / "invalid-approval.json"
        import_path.write_text(
            json.dumps(
                {
                    "format": "comic-translate-exact-tm",
                    "schema_version": 1,
                    "normalization_version": 1,
                    "entries": [
                        {
                            "source_text": "source",
                            "translation": "translation",
                            "source_lang": "Japanese",
                            "target_lang": "Korean",
                            "approved": "false",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        store = self._store()

        with self.assertRaisesRegex(ValueError, "JSON booleans"):
            store.import_tm(import_path)

        self.assertEqual(store.list_tm_entries(), [])

    def test_import_rejects_non_string_text_fields(self) -> None:
        store = self._store()
        for field in (
            "source_text",
            "translation",
            "source_lang",
            "target_lang",
        ):
            with self.subTest(field=field):
                entry = {
                    "source_text": "source",
                    "translation": "translation",
                    "source_lang": "Japanese",
                    "target_lang": "Korean",
                    "approved": False,
                }
                entry[field] = {"unexpected": "object"}
                import_path = Path(self.temp_dir.name) / f"invalid-{field}.json"
                import_path.write_text(
                    json.dumps(
                        {
                            "format": "comic-translate-exact-tm",
                            "schema_version": 1,
                            "normalization_version": 1,
                            "entries": [entry],
                        }
                    ),
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(ValueError, "must be strings"):
                    store.import_tm(import_path)

        self.assertEqual(store.list_tm_entries(), [])

    def test_corrupt_database_disables_cache_without_deleting_it(self) -> None:
        original_bytes = b"not a sqlite database"
        self.db_path.write_bytes(original_bytes)
        store = self._store()

        lookup = store.lookup_result("key", "scope")

        self.assertTrue(lookup.disabled)
        self.assertFalse(store.enabled)
        self.assertEqual(self.db_path.read_bytes(), original_bytes)

        export_path = Path(self.temp_dir.name) / "must-not-exist.json"
        with self.assertRaisesRegex(RuntimeError, "left unchanged"):
            store.export_tm(export_path)
        self.assertFalse(export_path.exists())

    def test_locked_database_disables_only_the_current_store(self) -> None:
        initializer = self._store()
        self.assertEqual(initializer.stats()["result_cache_entries"], 0)
        initializer.close()

        lock_connection = sqlite3.connect(self.db_path, timeout=0)
        self.addCleanup(lock_connection.close)
        lock_connection.execute("PRAGMA journal_mode = DELETE")
        lock_connection.execute("BEGIN EXCLUSIVE")

        locked_store = self._store(timeout_sec=0)
        stored = locked_store.store_results(
            [
                ResultCacheRecord(
                    cache_key="key",
                    scope_key="scope",
                    identity_json="{}",
                    source_text="source",
                    translation="translation",
                )
            ]
        )

        self.assertFalse(stored)
        self.assertFalse(locked_store.enabled)
        lock_connection.rollback()

        fresh_store = TranslationMemoryStore(self.db_path)
        self.addCleanup(fresh_store.close)
        self.assertTrue(fresh_store.enabled)
        self.assertEqual(fresh_store.stats()["result_cache_entries"], 0)


if __name__ == "__main__":
    unittest.main()
