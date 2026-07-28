from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark_translation_memory_fast_path as benchmark  # noqa: E402


class TranslationMemoryFastPathBenchmarkTests(unittest.TestCase):
    def test_new_engine_uses_retired_grouped_replacement_mode(self) -> None:
        class StubRuntime:
            def __init__(self) -> None:
                self.settings = benchmark.RuntimeSettings(
                    model="example.gguf",
                )

            @staticmethod
            def ensure() -> None:
                return None

            @staticmethod
            def identity() -> dict[str, str]:
                return {}

        engine = benchmark.new_engine(
            language="Japanese",
            store=None,
            runtime=StubRuntime(),
            persistent_cache_enabled=False,
            exact_tm_enabled=False,
            group_size=6,
            max_completion_tokens=512,
        )

        self.assertEqual(engine.request_mode, "contextual-single")

    def test_load_multilingual_corpora_requires_exact_language_counts(self) -> None:
        translations = []
        for language in ("japanese", "chinese", "english"):
            for index in range(18):
                translations.append(
                    {
                        "round": 1,
                        "mode": "contextual-single",
                        "case": f"{language}-{index}",
                        "source": f"source-{language}-{index}",
                        "old_log_reference": f"reference-{language}-{index}",
                    }
                )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(
                json.dumps({"translations": translations}),
                encoding="utf-8",
            )
            corpora = benchmark.load_multilingual_corpora(path)

        self.assertEqual(
            {language: len(items) for language, items in corpora.items()},
            {"Japanese": 18, "Chinese": 18, "English": 18},
        )

    def test_public_scenario_removes_private_outputs(self) -> None:
        public = benchmark.public_scenario(
            {
                "elapsed_sec": 1.2,
                "outputs": [{"source": "private", "translation": "private"}],
            }
        )

        self.assertEqual(public, {"elapsed_sec": 1.2})

    def test_prefix_selection_prefers_lowest_ram_within_three_percent(self) -> None:
        selected = benchmark.select_prefix_candidate(
            [
                {
                    "cache_ram_mib": 256,
                    "cache_prompt": True,
                    "median_elapsed_sec": 1.0,
                    "failure_count": 0,
                },
                {
                    "cache_ram_mib": 0,
                    "cache_prompt": True,
                    "median_elapsed_sec": 1.02,
                    "failure_count": 0,
                },
                {
                    "cache_ram_mib": 0,
                    "cache_prompt": False,
                    "median_elapsed_sec": 1.04,
                    "failure_count": 0,
                },
            ]
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["cache_ram_mib"], 0)
        self.assertTrue(selected["cache_prompt"])

    def test_output_comparison_uses_only_matching_language_indices(self) -> None:
        compared = benchmark.compare_outputs(
            {
                "outputs": [
                    {"language": "Japanese", "index": 0, "translation": "same"},
                    {"language": "Japanese", "index": 1, "translation": "left"},
                ]
            },
            {
                "outputs": [
                    {"language": "Japanese", "index": 0, "translation": "same"},
                    {"language": "English", "index": 1, "translation": "other"},
                ]
            },
        )

        self.assertEqual(compared["compared_count"], 1)
        self.assertEqual(compared["exact_match_count"], 1)

    def test_automated_gate_requires_server_skip_and_exact_preservation(self) -> None:
        empty_stats = {"gemma_http_attempt_count": 0}
        report = {
            "scenarios": {
                "stopped_all_result_cache_hit": {
                    "nonempty_count": 54,
                    "runtime_ensure_calls": 0,
                    "container_after": {"running": False},
                    "stats": empty_stats,
                },
                "stopped_all_approved_tm_hit": {
                    "nonempty_count": 54,
                    "runtime_ensure_calls": 0,
                    "container_after": {"running": False},
                    "stats": {
                        "gemma_http_attempt_count": 0,
                        "gemma_tm_exact_hit_count": 54,
                    },
                },
                "mixed_result_cache_hit": {
                    "nonempty_count": 54,
                    "severe_telemetry_count": 0,
                    "stats": {
                        "gemma_tm_result_cache_hit_count": 27,
                        "gemma_tm_result_cache_miss_count": 27,
                    },
                },
                "requested_blocks_no_hit": {
                    "nonempty_count": 27,
                    "severe_telemetry_count": 0,
                },
                "stale_sampler_key": {
                    "stats": {"gemma_tm_stale_reject_count": 21}
                },
                "corrupt_db_fail_open_preflight": {
                    "runtime_required": True,
                    "store_enabled": False,
                    "disabled_reason_type": "DatabaseError",
                    "database_preserved_sha256": benchmark._sha256_bytes(
                        b"not a sqlite database"
                    ),
                },
            },
            "comparisons": {
                "cold_vs_stopped_result_hit": {
                    "compared_count": 54,
                    "exact_match_count": 54,
                },
                "cold_vs_warm_result_hit": {
                    "compared_count": 54,
                    "exact_match_count": 54,
                },
                "cold_vs_approved_tm": {
                    "compared_count": 54,
                    "exact_match_count": 54,
                },
            },
            "prefix_cache_matrix": {"selected": {"cache_ram_mib": 0}},
        }

        gate = benchmark.automated_gate_report(report)

        self.assertTrue(gate["passed"])
        self.assertTrue(all(gate["checks"].values()))

        report["comparisons"]["cold_vs_stopped_result_hit"][
            "exact_match_count"
        ] = 53
        failed_gate = benchmark.automated_gate_report(report)
        self.assertFalse(failed_gate["passed"])
        self.assertFalse(
            failed_gate["checks"][
                "stopped_result_cache_exact_output_preservation"
            ]
        )


if __name__ == "__main__":
    unittest.main()
