from __future__ import annotations

import importlib.util
import contextlib
import io
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

    def _project_blob_with_blocks(
        self,
        temp_dir: str,
        texts: list[str],
        translations: list[str] | None = None,
    ) -> bytes:
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
            blocks = []
            translations = translations or []
            for index, text in enumerate(texts):
                block = TextBlock(
                    text_bbox=np.array([0, index * 100, 100, (index + 1) * 100]),
                    text=text,
                    translation=translations[index] if index < len(translations) else "",
                )
                blocks.append(block)
            manifest = {
                "original_image_files": ["page-001.png"],
                "llm_extra_context": "",
            }
            row = {
                "image_state": {
                    "source_lang": "English",
                    "target_lang": "Korean",
                    "blk_list": blocks,
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

    def _project_blob(self, temp_dir: str) -> bytes:
        return self._project_blob_with_blocks(
            temp_dir,
            ["Alpha source line."],
            ["Saved target line."],
        )

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

    def _series_project_with_two_items(self, temp_dir: str) -> str:
        series_path = os.path.join(temp_dir, "queue.seriesctpr")
        payload = self._project_blob(temp_dir)
        create_series_project(
            series_path,
            root_dir=temp_dir,
            items=[
                {
                    "series_item_id": "item-1",
                    "queue_index": 1,
                    "display_name": "example_other_chapter_source",
                    "source_kind": "archive",
                    "source_origin_path": os.path.join(temp_dir, "example_other_chapter_source"),
                    "source_origin_relpath": "example_other_chapter_source",
                    "imported_at": "2026-01-01T00:00:00",
                    "updated_at": "2026-01-01T00:00:00",
                    "status": "done",
                    "embedded_project_blob_hash": "hash-1",
                    "child_page_count": 1,
                },
                {
                    "series_item_id": "item-2",
                    "queue_index": 2,
                    "display_name": "example_selected_chapter_source",
                    "source_kind": "archive",
                    "source_origin_path": os.path.join(temp_dir, "example_selected_chapter_source"),
                    "source_origin_relpath": "example_selected_chapter_source",
                    "imported_at": "2026-01-01T00:00:00",
                    "updated_at": "2026-01-01T00:00:00",
                    "status": "done",
                    "embedded_project_blob_hash": "hash-1",
                    "child_page_count": 1,
                },
            ],
            embedded_projects=[
                {
                    "project_hash": "hash-1",
                    "display_name": "example_project.ctpr",
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

    def test_review_board_keeps_comparison_scroll_inside_panel(self) -> None:
        module = _load_validation_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            diff_path = Path(temp_dir) / "diff.json"
            diff_path.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "id": "sample-001",
                                "reason": "long review sample",
                                "source": "\n".join(f"source line {index}" for index in range(80)),
                                "baseline": "\n".join(f"baseline line {index}" for index in range(80)),
                                "candidate": "\n".join(f"candidate line {index}" for index in range(80)),
                            },
                            {
                                "id": "sample-002",
                                "reason": "second sample",
                                "source": "second source",
                                "baseline": "second baseline",
                                "candidate": "second candidate",
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result = module.make_review_board(diff_path, Path(temp_dir) / "out")

            html_text = Path(result["index_html"]).read_text(encoding="utf-8")
            self.assertIn('id="previousSample"', html_text)
            self.assertIn('id="nextSample"', html_text)
            self.assertIn("function setCurrent", html_text)
            self.assertRegex(html_text, r"body\s*\{[^}]*overflow:\s*hidden")
            self.assertRegex(html_text, r"\.comparison-grid\s*\{[^}]*overflow:\s*hidden")
            self.assertRegex(html_text, r"\.box\s*\{[^}]*overflow:\s*auto")
            self.assertRegex(html_text, r"\.actions\s*\{[^}]*position:\s*sticky")

    def test_review_board_has_filter_controls_for_large_diff_sets(self) -> None:
        module = _load_validation_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            diff_path = Path(temp_dir) / "diff.json"
            diff_path.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "id": "sample-001",
                                "reason": "sensitive_terms=1",
                                "source": "Line containing a trigger term.",
                                "baseline": "baseline text",
                                "candidate": "candidate text",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result = module.make_review_board(diff_path, Path(temp_dir) / "out")

            html_text = Path(result["index_html"]).read_text(encoding="utf-8")
            self.assertIn('id="filterMode"', html_text)
            self.assertIn('value="unreviewed"', html_text)
            self.assertIn('value="rejected"', html_text)
            self.assertIn("function filteredItems", html_text)

    def test_select_canary_samples_writes_local_raw_samples_and_summary(self) -> None:
        module = _load_validation_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            self._series_project(temp_dir)
            output_dir = Path(temp_dir) / "canary"

            summary = module.select_canary_samples(
                Path(temp_dir),
                output_dir,
                self._settings(),
                project_count=1,
                max_project_blocks=10,
                max_samples_per_project=1,
            )

            self.assertEqual(summary["sample_count"], 1)
            self.assertFalse(summary["privacy"]["git_safe"])
            samples_path = Path(summary["samples_path"])
            self.assertTrue(samples_path.is_file())
            samples = json.loads(samples_path.read_text(encoding="utf-8"))["samples"]
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["blocks"], ["Alpha source line."])
            self.assertEqual(samples[0]["saved_translations"], ["Saved target line."])
            summary_text = (output_dir / "canary_summary.json").read_text(encoding="utf-8")
            self.assertNotIn("Alpha source line.", summary_text)
            self.assertNotIn("Saved target line.", summary_text)

    def test_select_canary_samples_can_filter_by_source_metadata(self) -> None:
        module = _load_validation_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            self._series_project_with_two_items(temp_dir)
            output_dir = Path(temp_dir) / "canary"

            summary = module.select_canary_samples(
                Path(temp_dir),
                output_dir,
                self._settings(),
                project_count=1,
                max_project_blocks=10,
                max_samples_per_project=1,
                source_filter="example_selected_chapter",
            )

            self.assertEqual(summary["sample_count"], 1)
            self.assertEqual(summary["selected_projects"][0]["project_index"], 1)

    def test_select_sensitive_samples_balances_trigger_and_control_without_summary_raw_text(self) -> None:
        module = _load_validation_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = self._project_blob_with_blocks(
                temp_dir,
                [
                    "This quiet control line has no trigger.",
                    "The wife marker line should be sampled.",
                ],
                [
                    "control saved translation",
                    "selected saved translation",
                ],
            )
            create_series_project(
                os.path.join(temp_dir, "queue.seriesctpr"),
                root_dir=temp_dir,
                items=[
                    {
                        "series_item_id": "item-1",
                        "queue_index": 1,
                        "display_name": "example_selected_chapter_source",
                        "source_kind": "archive",
                        "source_origin_path": os.path.join(temp_dir, "example_selected_chapter_source"),
                        "source_origin_relpath": "example_selected_chapter_source",
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
                        "display_name": "example_project.ctpr",
                        "project_size": len(payload),
                        "project_blob": payload,
                    }
                ],
            )
            output_dir = Path(temp_dir) / "sensitive"
            settings = self._settings()
            settings = module.ValidationSettings(
                source_lang=settings.source_lang,
                target_lang=settings.target_lang,
                model=settings.model,
                chunk_size=1,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                top_k=settings.top_k,
                top_p=settings.top_p,
                min_p=settings.min_p,
                prompt_profile=settings.prompt_profile,
                response_format_mode=settings.response_format_mode,
                response_schema_mode=settings.response_schema_mode,
            )

            summary = module.select_sensitive_samples(
                Path(temp_dir),
                output_dir,
                settings,
                golden_limit=1,
                control_limit=1,
            )

            self.assertEqual(summary["sample_count"], 2)
            self.assertEqual(summary["sensitive_sample_count"], 1)
            self.assertEqual(summary["control_sample_count"], 1)
            self.assertFalse(summary["privacy"]["git_safe"])
            samples = json.loads(Path(summary["samples_path"]).read_text(encoding="utf-8"))["samples"]
            self.assertEqual(len(samples), 2)
            self.assertTrue(any("wife" in " ".join(item["blocks"]) for item in samples))
            summary_text = (output_dir / "sensitive_summary.json").read_text(encoding="utf-8")
            self.assertNotIn("wife marker", summary_text)
            self.assertNotIn("selected saved translation", summary_text)

    def test_select_sensitive_samples_includes_phrase_and_saved_translation_risk(self) -> None:
        module = _load_validation_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            phrase_source = "pack such " + "a punch"
            saved_risk = "<|channel>" + "thought"
            payload = self._project_blob_with_blocks(
                temp_dir,
                [
                    f"This synthetic line can {phrase_source}.",
                    "This source line looks harmless.",
                    "A quiet control line.",
                ],
                [
                    "saved neutral translation",
                    saved_risk,
                    "control saved translation",
                ],
            )
            create_series_project(
                os.path.join(temp_dir, "queue.seriesctpr"),
                root_dir=temp_dir,
                items=[
                    {
                        "series_item_id": "item-1",
                        "queue_index": 1,
                        "display_name": "example_selected_chapter_source",
                        "source_kind": "archive",
                        "source_origin_path": os.path.join(temp_dir, "example_selected_chapter_source"),
                        "source_origin_relpath": "example_selected_chapter_source",
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
                        "display_name": "example_project.ctpr",
                        "project_size": len(payload),
                        "project_blob": payload,
                    }
                ],
            )
            settings = self._settings()
            settings = module.ValidationSettings(
                source_lang=settings.source_lang,
                target_lang=settings.target_lang,
                model=settings.model,
                chunk_size=1,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                top_k=settings.top_k,
                top_p=settings.top_p,
                min_p=settings.min_p,
                prompt_profile=settings.prompt_profile,
                response_format_mode=settings.response_format_mode,
                response_schema_mode=settings.response_schema_mode,
            )

            summary = module.select_sensitive_samples(
                Path(temp_dir),
                Path(temp_dir) / "sensitive",
                settings,
                golden_limit=10,
                control_limit=1,
            )

            self.assertEqual(summary["sensitive_sample_count"], 2)
            self.assertEqual(summary["counts"]["source_trigger_chunks"], 1)
            self.assertEqual(summary["counts"]["saved_translation_risk_chunks"], 1)
            reason_text = "\n".join(item["reason"] for item in summary["selected"])
            self.assertIn("source_phrase", reason_text)
            self.assertIn("saved_translation_risk", reason_text)
            summary_text = (Path(temp_dir) / "sensitive" / "sensitive_summary.json").read_text(encoding="utf-8")
            self.assertNotIn(phrase_source, summary_text)
            self.assertNotIn(saved_risk, summary_text)

    def test_sensitive_sampler_matrix_flags_bad_outputs_and_preserves_prompt_prefix_hash(self) -> None:
        module = _load_validation_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            samples_path = Path(temp_dir) / "samples.json"
            samples_path.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "id": "sample-001",
                                "reason": "sensitive_terms=1",
                                "source_lang": "English",
                                "target_lang": "Korean",
                                "blocks": ["The wife marker line should keep its meaning."],
                                "saved_translations": ["saved baseline"],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            calls = []

            def fake_request(engine, system_prompt, user_prompt, *, expected_keys=None):
                calls.append(
                    {
                        "temperature": engine.temperature,
                        "top_k": engine.top_k,
                        "top_p": engine.top_p,
                        "system_prompt": system_prompt,
                        "expected_keys": expected_keys,
                    }
                )
                if engine.temperature == 0.7:
                    return {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": json.dumps({"block_0": "와" + "님 typo"})},
                            }
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                    }
                return {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps({"block_0": "안정 번역"})},
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                }

            original_request = module.CustomLocalGemmaTranslation._request_translation
            try:
                module.CustomLocalGemmaTranslation._request_translation = fake_request
                summary = module.run_sensitive_sampler_matrix(
                    samples_path,
                    Path(temp_dir) / "matrix",
                    self._settings(),
                    api_base_url="http://127.0.0.1:18080/v1",
                    timeout=1.0,
                )
            finally:
                module.CustomLocalGemmaTranslation._request_translation = original_request

            self.assertEqual(summary["counters"]["failed_samples"], 0)
            self.assertEqual(summary["candidates"]["baseline"]["known_bad_output_count"], 1)
            self.assertEqual(summary["candidates"]["stable_b"]["known_bad_output_count"], 0)
            self.assertFalse(summary["prompt_prefix"]["changed"])
            self.assertEqual({call["expected_keys"][0] for call in calls}, {"block_0"})
            self.assertIn(0.3, {call["temperature"] for call in calls})
            review_items = json.loads(Path(summary["review_diff_path"]).read_text(encoding="utf-8"))["items"]
            self.assertTrue(any(item["candidate_name"] == "stable_b" for item in review_items))

    def test_sensitive_prompt_matrix_keeps_prefix_and_compares_prompt_candidates(self) -> None:
        module = _load_validation_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            phrase_source = "pack such " + "a punch"
            samples_path = Path(temp_dir) / "samples.json"
            samples_path.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "id": "sample-001",
                                "reason": "source_phrase=1",
                                "source_lang": "English",
                                "target_lang": "Korean",
                                "blocks": [f"This synthetic line can {phrase_source}."],
                                "saved_translations": [""],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            seen_system_prompts = {}

            def fake_request(engine, system_prompt, user_prompt, *, expected_keys=None):
                profile = getattr(engine, "_validation_prompt_candidate_name", "unknown")
                seen_system_prompts[profile] = system_prompt
                value = "완곡 번역" if profile == "baseline" else "직접 번역"
                return {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps({"block_0": value})},
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                }

            original_request = module.CustomLocalGemmaTranslation._request_translation
            try:
                module.CustomLocalGemmaTranslation._request_translation = fake_request
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    summary = module.run_sensitive_prompt_matrix(
                        samples_path,
                        Path(temp_dir) / "prompt-matrix",
                        self._settings(),
                        api_base_url="http://127.0.0.1:18080/v1",
                        timeout=1.0,
                    )
            finally:
                module.CustomLocalGemmaTranslation._request_translation = original_request

            self.assertFalse(summary["prompt_prefix"]["changed"])
            self.assertIn("baseline", summary["candidates"])
            self.assertIn("preserve_explicitness", summary["candidates"])
            self.assertEqual(summary["candidates"]["baseline"]["translation_hashes_changed_vs_baseline"], 0)
            self.assertEqual(summary["candidates"]["preserve_explicitness"]["translation_hashes_changed_vs_baseline"], 1)
            self.assertTrue(seen_system_prompts["baseline"].startswith("You are Gemma, a large language model."))
            self.assertIn("do not soften", seen_system_prompts["preserve_explicitness"])
            self.assertIn("prompt-matrix progress sample=1/1 candidate=baseline", stderr.getvalue())
            self.assertNotIn(phrase_source, stderr.getvalue())
            review_items = json.loads(Path(summary["review_diff_path"]).read_text(encoding="utf-8"))["items"]
            self.assertTrue(any(item["candidate_name"] == "preserve_explicitness" for item in review_items))


if __name__ == "__main__":
    unittest.main()
