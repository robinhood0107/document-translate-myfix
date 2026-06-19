from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import msgpack
import numpy as np

from app.projects.parsers import ProjectEncoder
from app.projects.series_state_v1 import create_series_project
from modules.translation.llm.custom_local_gemma import (
    DEFAULT_GEMMA_CHUNK_SIZE,
    DEFAULT_GEMMA_MAX_COMPLETION_TOKENS,
    DEFAULT_GEMMA_PROMPT_PROFILE,
    DEFAULT_GEMMA_RESPONSE_FORMAT_MODE,
    DEFAULT_GEMMA_RESPONSE_SCHEMA_MODE,
    DEFAULT_GEMMA_TRANSLATION_MIN_P,
    DEFAULT_GEMMA_TRANSLATION_TEMPERATURE,
    DEFAULT_GEMMA_TRANSLATION_TOP_K,
    DEFAULT_GEMMA_TRANSLATION_TOP_P,
)
from modules.utils.textblock import TextBlock


def _load_validation_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "gemma_exact_validation.py"
    spec = importlib.util.spec_from_file_location("gemma_exact_validation", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GemmaExactValidationTests(unittest.TestCase):
    def _settings(self):
        module = _load_validation_module()
        return module.ValidationSettings(
            source_lang="English",
            target_lang="Korean",
            model="gemma-4-26B-IQ4_NL.gguf",
            chunk_size=DEFAULT_GEMMA_CHUNK_SIZE,
            max_tokens=DEFAULT_GEMMA_MAX_COMPLETION_TOKENS,
            temperature=DEFAULT_GEMMA_TRANSLATION_TEMPERATURE,
            top_k=DEFAULT_GEMMA_TRANSLATION_TOP_K,
            top_p=DEFAULT_GEMMA_TRANSLATION_TOP_P,
            min_p=DEFAULT_GEMMA_TRANSLATION_MIN_P,
            prompt_profile=DEFAULT_GEMMA_PROMPT_PROFILE,
            response_format_mode=DEFAULT_GEMMA_RESPONSE_FORMAT_MODE,
            response_schema_mode=DEFAULT_GEMMA_RESPONSE_SCHEMA_MODE,
        )

    def _project_blob(self, temp_dir: str) -> bytes:
        project_path = os.path.join(temp_dir, "child.ctpr")
        conn = sqlite3.connect(project_path)
        encoder = ProjectEncoder()
        try:
            conn.execute(
                """
                CREATE TABLE project_manifest (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    manifest_blob BLOB NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE page_state (
                    page_path TEXT PRIMARY KEY,
                    row_blob BLOB NOT NULL
                )
                """
            )
            block = TextBlock(
                text_bbox=np.array([0, 0, 100, 100]),
                text="Alpha source line.",
                translation="Saved target line.",
            )
            manifest = {
                "original_image_files": ["page-001.png"],
                "llm_extra_context": "",
            }
            row = {
                "image_state": {
                    "source_lang": "English",
                    "target_lang": "Korean",
                    "blk_list": [block],
                }
            }
            conn.execute(
                "INSERT INTO project_manifest(id, manifest_blob) VALUES(1, ?)",
                (sqlite3.Binary(msgpack.packb(manifest, default=encoder.encode, use_bin_type=True)),),
            )
            conn.execute(
                "INSERT INTO page_state(page_path, row_blob) VALUES(?, ?)",
                (
                    "page-001.png",
                    sqlite3.Binary(msgpack.packb(row, default=encoder.encode, use_bin_type=True)),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        with open(project_path, "rb") as fh:
            return fh.read()

    def _series_project(self, temp_dir: str) -> str:
        series_path = os.path.join(temp_dir, "queue.seriesctpr")
        payload = self._project_blob(temp_dir)
        create_series_project(
            series_path,
            root_dir=temp_dir,
            items=[
                {
                    "series_item_id": "item-1",
                    "queue_index": 1,
                    "display_name": "child.ctpr",
                    "source_kind": "ctpr_import",
                    "source_origin_path": os.path.join(temp_dir, "child.ctpr"),
                    "source_origin_relpath": "child.ctpr",
                    "imported_at": "2026-01-01T00:00:00",
                    "updated_at": "2026-01-01T00:00:00",
                    "status": "done",
                    "embedded_project_blob_hash": "hash-1",
                    "child_page_count": 1,
                }
            ],
            embedded_projects=[
                {
                    "project_hash": "hash-1",
                    "display_name": "child.ctpr",
                    "project_size": len(payload),
                    "project_blob": payload,
                }
            ],
        )
        return series_path

    def test_baseline_writes_hashes_without_raw_text(self) -> None:
        module = _load_validation_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            self._series_project(temp_dir)
            output_dir = Path(temp_dir) / "validation"

            summary = module.build_baseline(Path(temp_dir), output_dir, self._settings())

            self.assertEqual(summary["counts"]["series_files"], 1)
            self.assertEqual(summary["counts"]["embedded_projects"], 1)
            self.assertEqual(summary["counts"]["pages"], 1)
            self.assertEqual(summary["counts"]["blocks"], 1)
            baseline_text = (output_dir / "baseline.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("Alpha source line.", baseline_text)
            self.assertNotIn("Saved target line.", baseline_text)
            row = json.loads(baseline_text)
            self.assertEqual(row["source_length"], len("Alpha source line."))
            self.assertTrue(row["has_translation"])

    def test_compare_baselines_rejects_translation_hash_mismatch(self) -> None:
        module = _load_validation_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            self._series_project(temp_dir)
            output_dir = Path(temp_dir) / "validation"
            module.build_baseline(Path(temp_dir), output_dir, self._settings())
            baseline_path = output_dir / "baseline.jsonl"
            candidate_path = output_dir / "candidate.jsonl"
            row = json.loads(baseline_path.read_text(encoding="utf-8"))
            row["translation_hash"] = "changed"
            candidate_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            result = module.compare_baselines(baseline_path, candidate_path, output_dir / "compare")

            self.assertFalse(result["gate_passed"])
            self.assertEqual(result["counters"]["translation_hash_mismatch"], 1)

    def test_review_board_creates_local_html_and_decision_file(self) -> None:
        module = _load_validation_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            diff_path = Path(temp_dir) / "diff.json"
            diff_path.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "id": "sample-001",
                                "reason": "golden set",
                                "source": "Alpha <source>",
                                "baseline": "Saved target line.",
                                "candidate": "Candidate target line.",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result = module.make_review_board(diff_path, Path(temp_dir) / "out")

            self.assertEqual(result["sample_count"], 1)
            self.assertTrue(os.path.isfile(result["index_html"]))
            self.assertTrue(os.path.isfile(result["decision_path"]))
            html_text = Path(result["index_html"]).read_text(encoding="utf-8")
            self.assertIn("완전 동일/통과", html_text)
            self.assertIn("달라졌지만 승인", html_text)
            self.assertIn("불합격", html_text)
            self.assertIn("\\u003csource>", html_text)
            self.assertNotIn("\\n\\n<span", html_text)


if __name__ == "__main__":
    unittest.main()
