from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_cold_cache_finalization as finalization  # noqa: E402


class ColdCacheFinalizationTests(unittest.TestCase):
    def test_protocol_locks_current_product_baseline_and_limits(self) -> None:
        protocol = finalization.load_protocol()
        baseline = finalization._read_json(
            ROOT / str(protocol["baseline_preset"])
        )

        self.assertEqual(protocol["protocol_version"], 2)
        self.assertEqual(protocol["limits"]["pipeline_max_pages"], 6)
        self.assertEqual(protocol["limits"]["pipeline_max_blocks"], 54)
        self.assertEqual(protocol["limits"]["completion_tokens"], 512)
        self.assertEqual(
            protocol["limits"]["cache_stabilization_pairs"],
            1,
        )
        self.assertEqual(
            baseline["gemma"]["model"],
            "gemma-4-26B-IQ4_NL.gguf",
        )
        self.assertEqual(baseline["gemma"]["chunk_size"], 6)
        self.assertEqual(baseline["gemma"]["batch_size"], 2048)
        self.assertEqual(baseline["gemma"]["ubatch_size"], 512)
        self.assertEqual(
            baseline["benchmark_contract"]["request_mode"],
            "contextual-single",
        )
        self.assertFalse(
            any(
                baseline["benchmark_cache_policy"][key]
                for key in (
                    "paddleocr_persistent",
                    "translation_persistent",
                    "exact_tm",
                    "project_checkpoint",
                )
            )
        )

    def test_output_must_be_outside_repository_and_new(self) -> None:
        with self.assertRaises(ValueError):
            finalization.ensure_external_output(
                ROOT / "ignored-output",
                require_new=True,
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "new-suite"
            self.assertEqual(
                finalization.ensure_external_output(
                    path,
                    require_new=True,
                ),
                path.resolve(),
            )
            with self.assertRaises(FileExistsError):
                finalization.ensure_external_output(
                    path,
                    require_new=True,
                )

    def test_formal_run_rejects_dirty_checkout(self) -> None:
        with mock.patch.object(
            finalization,
            "_git_output",
            return_value=" M scripts/example.py",
        ):
            with self.assertRaises(RuntimeError):
                finalization._require_reproducible_checkout()

    def test_balanced_orders_reverse_and_rotate(self) -> None:
        orders = finalization.balanced_orders(
            ["a", "b", "c"],
            3,
        )

        self.assertEqual(orders[0], ["a", "b", "c"])
        self.assertEqual(orders[1], ["c", "b", "a"])
        self.assertEqual(orders[2], ["b", "c", "a"])
        self.assertTrue(
            all(sorted(order) == ["a", "b", "c"] for order in orders)
        )
        self.assertEqual(
            finalization.balanced_orders(["a", "b"], 3),
            [["a", "b"], ["b", "a"], ["a", "b"]],
        )

    def test_cache_protocol_state_records_input_content_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            (input_dir / "page.png").write_bytes(b"cache-input")
            with mock.patch.object(
                finalization,
                "_protocol_state",
                return_value={"commit": "abc123"},
            ):
                state = finalization._cache_protocol_state(
                    {},
                    scenario="global-ocr",
                    input_dir=input_dir,
                    sample_count=1,
                    source_language="Japanese",
                    stabilization_orders=[
                        ["enabled_empty_cold", "disabled_cold"]
                    ],
                    cold_orders=[
                        ["disabled_cold", "enabled_empty_cold"],
                        ["enabled_empty_cold", "disabled_cold"],
                        ["disabled_cold", "enabled_empty_cold"],
                    ],
                )

            expected = finalization._input_contract(input_dir, 1)
            self.assertEqual(
                state["input_contract_sha256"],
                expected["sha256"],
            )
            self.assertEqual(state["sample_count"], 1)
            self.assertEqual(state["source_language"], "Japanese")
            self.assertEqual(state["scenario"], "global-ocr")

    def test_standalone_runner_bootstraps_repo_root_for_input_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            (input_dir / "page.png").write_bytes(b"standalone-input")
            runner = ROOT / "scripts" / "benchmark_cold_cache_finalization.py"
            probe = (
                "import runpy,sys;"
                "from pathlib import Path;"
                f"sys.path.insert(0, {str(runner.parent)!r});"
                f"namespace=runpy.run_path({str(runner)!r}, "
                "run_name='cold_cache_probe');"
                "print(namespace['_input_contract']("
                f"Path({str(input_dir)!r}), 1)['sample_count'])"
            )
            completed = subprocess.run(
                [sys.executable, "-I", "-c", probe],
                cwd=tmp,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stderr,
        )
        self.assertEqual(completed.stdout.strip(), "1")

    def test_deep_merge_preserves_unmodified_runtime_contract(self) -> None:
        merged = finalization._deep_merge(
            {
                "gemma": {
                    "model": "baseline.gguf",
                    "chunk_size": 6,
                },
                "ocr_client": {"parallel_workers": 8},
            },
            {"ocr_client": {"parallel_workers": 4}},
        )

        self.assertEqual(merged["gemma"]["model"], "baseline.gguf")
        self.assertEqual(merged["gemma"]["chunk_size"], 6)
        self.assertEqual(merged["ocr_client"]["parallel_workers"], 4)

    def test_finalization_pipeline_delegates_runtime_start_to_product(
        self,
    ) -> None:
        command = finalization._pipeline_command(
            preset_path=Path("preset.json"),
            input_dir=Path("input"),
            sample_count=1,
            run_dir=Path("run"),
            shared_corpus_dir=Path("shared"),
            stage="ocr",
            source_lang="Japanese",
        )

        mode_index = command.index("--runtime-mode")
        self.assertEqual(command[mode_index + 1], "attach-running")
        self.assertIn("--product-managed-runtime", command)

    def test_generated_runtime_axis_changes_only_one_setting(self) -> None:
        protocol = finalization.load_protocol()
        family = finalization._family(protocol, "paddle-runtime")

        candidates, baseline = finalization.pipeline_candidates(
            family,
            axis="max_num_seqs",
        )

        self.assertEqual(baseline, "max_num_seqs-32")
        self.assertEqual(
            [item["id"] for item in candidates],
            [
                "max_num_seqs-16",
                "max_num_seqs-32",
                "max_num_seqs-48",
            ],
        )
        self.assertEqual(
            candidates[0]["patch"],
            {"ocr_runtime": {"max_num_seqs": 16}},
        )

    def test_np2_translation_candidate_keeps_4096_context_per_slot(
        self,
    ) -> None:
        protocol = finalization.load_protocol()
        family = finalization._family(protocol, "gemma-parallel")

        candidates, _baseline = (
            finalization._translation_candidate_profiles(
                protocol,
                family,
                axis="",
                model_key="iq4_nl",
            )
        )
        np2 = next(
            candidate
            for candidate in candidates
            if candidate["id"] == "np2-concurrent"
        )
        self.assertEqual(np2["context_size"], 8192)
        self.assertGreaterEqual(
            np2["context_size"] // np2["n_parallel"],
            4096,
        )
        self.assertEqual(np2["batch_size"], 2048)
        self.assertEqual(np2["ubatch_size"], 512)

    def test_generated_gemma_batch_axes_change_one_runtime_setting(
        self,
    ) -> None:
        protocol = finalization.load_protocol()
        family = finalization._family(protocol, "gemma-batch")

        batch_candidates, batch_baseline = (
            finalization._translation_candidate_profiles(
                protocol,
                family,
                axis="batch_size",
                model_key="iq4_nl",
            )
        )
        self.assertEqual(batch_baseline, "batch_size-2048")
        self.assertEqual(
            [candidate["batch_size"] for candidate in batch_candidates],
            [512, 1024, 2048, 4096],
        )
        self.assertEqual(
            {candidate["ubatch_size"] for candidate in batch_candidates},
            {512},
        )

        ubatch_candidates, ubatch_baseline = (
            finalization._translation_candidate_profiles(
                protocol,
                family,
                axis="ubatch_size",
                model_key="iq4_nl",
                base_batch_size=1024,
            )
        )
        self.assertEqual(ubatch_baseline, "ubatch_size-512")
        self.assertEqual(
            {candidate["batch_size"] for candidate in ubatch_candidates},
            {1024},
        )
        self.assertEqual(
            [candidate["ubatch_size"] for candidate in ubatch_candidates],
            [128, 256, 512, 1024],
        )
        self.assertTrue(
            all(
                candidate["ubatch_size"] <= candidate["batch_size"]
                for candidate in ubatch_candidates
            )
        )
        constrained, _ = finalization._translation_candidate_profiles(
            protocol,
            family,
            axis="ubatch_size",
            model_key="iq4_nl",
            base_batch_size=512,
        )
        self.assertEqual(
            [candidate["ubatch_size"] for candidate in constrained],
            [128, 256, 512],
        )

    def test_generated_gemma_ubatch_rejects_unapproved_base_batch(
        self,
    ) -> None:
        protocol = finalization.load_protocol()
        family = finalization._family(protocol, "gemma-batch")

        with self.assertRaises(ValueError):
            finalization._translation_candidate_profiles(
                protocol,
                family,
                axis="ubatch_size",
                model_key="iq4_nl",
                base_batch_size=1536,
            )

    def test_translation_profile_command_carries_batch_contract(self) -> None:
        command = finalization._translation_profile_command(
            source_summary=Path("summary.json"),
            output_path=Path("output.json"),
            candidate={
                "id": "batch_size-1024",
                "model_name": "example.gguf",
                "model_sha256": "a" * 64,
                "chunk_size": 6,
                "context_size": 4096,
                "n_parallel": 1,
                "concurrency": 1,
                "batch_size": 1024,
                "ubatch_size": 256,
            },
            language_order=("Japanese", "Chinese", "English"),
        )

        self.assertEqual(
            command[command.index("--batch-size") + 1],
            "1024",
        )
        self.assertEqual(
            command[command.index("--ubatch-size") + 1],
            "256",
        )

    def test_candidate_base_rejects_cache_or_grouped_drift(self) -> None:
        protocol = finalization.load_protocol()
        baseline = finalization._read_json(
            ROOT / str(protocol["baseline_preset"])
        )
        finalization._validate_cold_candidate_base(baseline)

        cache_drift = finalization._deep_merge(
            baseline,
            {
                "benchmark_cache_policy": {
                    "paddleocr_persistent": True,
                }
            },
        )
        with self.assertRaises(ValueError):
            finalization._validate_cold_candidate_base(cache_drift)

        grouped_drift = finalization._deep_merge(
            baseline,
            {
                "benchmark_contract": {
                    "request_mode": "contextual-grouped",
                }
            },
        )
        with self.assertRaises(ValueError):
            finalization._validate_cold_candidate_base(grouped_drift)

        spec_drift = finalization._deep_merge(
            baseline,
            {"gemma": {"spec_type": "ngram-mod"}},
        )
        with self.assertRaises(ValueError):
            finalization._validate_cold_candidate_base(spec_drift)

    def test_page_contract_hashes_private_ocr_and_render_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page_snapshots.json"
            path.write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "page_failed": False,
                                "stage_status": {},
                                "translated_image_path": "private/output.png",
                                "translated_image_exists": True,
                                "translated_image_sha256": "a" * 64,
                                "blocks": [
                                    {
                                        "xyxy": [1, 2, 3, 4],
                                        "bubble_xyxy": None,
                                        "angle": 0,
                                        "text_class": "speech",
                                        "text": "private source",
                                        "ocr_status": "ok",
                                        "ocr_empty_reason": "",
                                        "ocr_attempt_count": 1,
                                        "ocr_raw_text": "private source",
                                        "ocr_sanitized_text": "private source",
                                        "ocr_reject_reason": "",
                                        "translation": "private translation",
                                    }
                                ],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            contract = finalization._page_contract(path)

        serialized = json.dumps(contract, ensure_ascii=False)
        self.assertEqual(contract["page_count"], 1)
        self.assertEqual(contract["block_count"], 1)
        self.assertEqual(len(contract["ocr_sha256"]), 64)
        self.assertNotIn("private source", serialized)
        self.assertNotIn("private translation", serialized)

    def test_exact_ocr_candidate_requires_stable_equal_digests(self) -> None:
        protocol = finalization.load_protocol()
        family = finalization._family(protocol, "paddle-workers")
        results = []
        for candidate in ("workers-8", "workers-4"):
            for round_index in range(1, 4):
                results.append(
                    {
                        "candidate": candidate,
                        "round": round_index,
                        "status": "passed",
                        "elapsed_sec": 10.0 if candidate == "workers-8" else 9.0,
                        "performance_stats": {
                            "stages": {
                                "ocr": {
                                    "wall_ms": (
                                        10_000
                                        if candidate == "workers-8"
                                        else 9_000
                                    )
                                }
                            }
                        },
                        "page_contract": {
                            "page_count": 6,
                            "block_count": 54,
                            # OCR-only runs intentionally stop before
                            # translation, so every translation is empty.
                            "empty_translation_count": 54,
                            "detection_sha256": "d" * 64,
                            "ocr_sha256": "o" * 64,
                        },
                    }
                )

        analysis = finalization.analyze_pipeline_results(
            protocol=protocol,
            family=family,
            baseline_id="workers-8",
            results=results,
        )
        candidate = next(
            item
            for item in analysis["candidates"]
            if item["candidate"] == "workers-4"
        )

        self.assertTrue(analysis["baseline_exact_output_stable"])
        self.assertTrue(candidate["exact_quality_passed"])
        self.assertTrue(candidate["structural_quality_passed"])
        self.assertIsNone(candidate["translation_structure_passed"])
        self.assertTrue(candidate["speed_gate_passed"])
        self.assertTrue(candidate["automated_gate_passed"])

    def test_ocr_candidate_needs_five_percent_and_stable_rounds(self) -> None:
        protocol = finalization.load_protocol()
        family = finalization._family(protocol, "paddle-workers")

        def result(candidate: str, round_index: int, wall_ms: int):
            return {
                "candidate": candidate,
                "round": round_index,
                "status": "passed",
                "performance_stats": {
                    "run_wall_ms": wall_ms,
                    "stages": {"ocr": {"wall_ms": wall_ms}},
                },
                "page_contract": {
                    "page_count": 6,
                    "block_count": 54,
                    "detection_sha256": "d" * 64,
                    "ocr_sha256": "o" * 64,
                },
            }

        results = [
            result("workers-8", round_index, 10_000)
            for round_index in range(1, 4)
        ]
        results.extend(
            result("workers-6", round_index, 9_711)
            for round_index in range(1, 4)
        )
        results.extend(
            [
                result("workers-4", 1, 8_000),
                result("workers-4", 2, 8_000),
                result("workers-4", 3, 10_000),
            ]
        )

        analysis = finalization.analyze_pipeline_results(
            protocol=protocol,
            family=family,
            baseline_id="workers-8",
            results=results,
        )
        by_id = {
            item["candidate"]: item for item in analysis["candidates"]
        }

        self.assertFalse(by_id["workers-6"]["speed_gate_passed"])
        self.assertFalse(by_id["workers-6"]["automated_gate_passed"])
        self.assertFalse(by_id["workers-4"]["variance_gate_passed"])
        self.assertFalse(by_id["workers-4"]["automated_gate_passed"])

    def test_stage_candidate_can_qualify_via_expected_full_improvement(
        self,
    ) -> None:
        protocol = finalization.load_protocol()
        family = finalization._family(protocol, "paddle-workers")

        def result(candidate: str, wall_ms: int):
            return {
                "candidate": candidate,
                "status": "passed",
                "performance_stats": {
                    "run_wall_ms": wall_ms,
                    "stages": {"ocr": {"wall_ms": wall_ms}},
                },
                "page_contract": {
                    "page_count": 6,
                    "block_count": 54,
                    "detection_sha256": "d" * 64,
                    "ocr_sha256": "o" * 64,
                },
            }

        results = [
            result(candidate, wall_ms)
            for candidate, wall_ms in (
                ("workers-8", 10_000),
                ("workers-8", 10_000),
                ("workers-8", 10_000),
                ("workers-6", 9_800),
                ("workers-6", 9_800),
                ("workers-6", 9_800),
            )
        ]
        without_reference = finalization.analyze_pipeline_results(
            protocol=protocol,
            family=family,
            baseline_id="workers-8",
            results=results,
        )
        with_reference = finalization.analyze_pipeline_results(
            protocol=protocol,
            family=family,
            baseline_id="workers-8",
            results=results,
            full_reference={
                "full_median_sec": 10.0,
                "stage_median_sec": 6.0,
                "stage_share": 0.6,
            },
        )
        without_candidate = next(
            item
            for item in without_reference["candidates"]
            if item["candidate"] == "workers-6"
        )
        with_candidate = next(
            item
            for item in with_reference["candidates"]
            if item["candidate"] == "workers-6"
        )

        self.assertFalse(without_candidate["speed_gate_passed"])
        self.assertEqual(
            with_candidate["expected_full_improvement_percent"],
            2.0,
        )
        self.assertTrue(with_candidate["speed_gate_passed"])
        self.assertTrue(with_candidate["automated_gate_passed"])

    def test_cache_analysis_requires_zero_runtime_and_http(self) -> None:
        protocol = finalization.load_protocol()
        cold = {
            "elapsed_sec": 10.0,
            "performance_stats": {
                "stages": {"ocr": {"wall_ms": 10_000}},
            },
            "page_contract": {"ocr_sha256": "a" * 64},
        }
        hit = {
            "elapsed_sec": 1.0,
            "performance_stats": {
                "stages": {"ocr": {"wall_ms": 1_000}},
                "runtime": {"paddleocr_vl": {"start_count": 0}},
                "paddleocr_vl": {"http_attempt_count": 0},
            },
            "page_contract": {"ocr_sha256": "a" * 64},
        }

        results = {
            "stabilization": [
                {**cold, "candidate": "enabled_empty_cold"},
                {**cold, "candidate": "disabled_cold"},
            ],
            "disabled_cold": [
                {**cold, "round": round_index}
                for round_index in range(1, 4)
            ],
            "enabled_empty_cold": [
                {
                    **cold,
                    "round": round_index,
                    "performance_stats": {
                        "stages": {"ocr": {"wall_ms": 10_200}},
                    },
                }
                for round_index in range(1, 4)
            ],
            "all_hit": hit,
        }
        analysis = finalization.analyze_cache_results(
            protocol=protocol,
            scenario="global-ocr",
            results=results,
        )

        self.assertTrue(analysis["passed"])
        self.assertEqual(analysis["all_hit_reduction_percent"], 90.0)
        self.assertEqual(analysis["cache_miss_overhead_percent"], 2.0)
        self.assertEqual(
            analysis["cold_plus_all_hit_net_gain_percent"],
            44.0,
        )
        without_stabilization = dict(results)
        without_stabilization.pop("stabilization")
        incomplete = finalization.analyze_cache_results(
            protocol=protocol,
            scenario="global-ocr",
            results=without_stabilization,
        )
        self.assertFalse(incomplete["passed"])
        self.assertFalse(
            incomplete["checks"]["stabilization_runs_complete"]
        )
        noisy = copy.deepcopy(results)
        noisy["disabled_cold"][0]["performance_stats"] = copy.deepcopy(
            noisy["disabled_cold"][0]["performance_stats"]
        )
        noisy["disabled_cold"][0]["performance_stats"]["stages"][
            "ocr"
        ]["wall_ms"] = 20_000
        noisy_analysis = finalization.analyze_cache_results(
            protocol=protocol,
            scenario="global-ocr",
            results=noisy,
        )
        self.assertTrue(noisy_analysis["passed"])
        self.assertFalse(
            noisy_analysis["diagnostics"][
                "disabled_cold_within_variance_reference"
            ]
        )

    def test_cache_promotion_uses_net_gain_not_miss_overhead_gate(
        self,
    ) -> None:
        protocol = finalization.load_protocol()

        def cold(seconds: float) -> dict:
            return {
                "status": "passed",
                "performance_stats": {
                    "stages": {"ocr": {"wall_ms": seconds * 1000}},
                },
                "page_contract": {"ocr_sha256": "a" * 64},
            }

        hit = {
            **cold(1.0),
            "performance_stats": {
                "stages": {"ocr": {"wall_ms": 1_000}},
                "runtime": {"paddleocr_vl": {"start_count": 0}},
                "paddleocr_vl": {"http_attempt_count": 0},
            },
        }

        def analyze(enabled_seconds: float) -> dict:
            disabled = cold(10.0)
            enabled = cold(enabled_seconds)
            return finalization.analyze_cache_results(
                protocol=protocol,
                scenario="global-ocr",
                results={
                    "stabilization": [enabled, disabled],
                    "disabled_cold": [
                        {**disabled, "round": round_index}
                        for round_index in range(1, 4)
                    ],
                    "enabled_empty_cold": [
                        {**enabled, "round": round_index}
                        for round_index in range(1, 4)
                    ],
                    "all_hit": hit,
                },
            )

        large_overhead_but_net_faster = analyze(15.0)
        self.assertTrue(large_overhead_but_net_faster["passed"])
        self.assertEqual(
            large_overhead_but_net_faster[
                "cache_miss_overhead_percent"
            ],
            50.0,
        )
        self.assertEqual(
            large_overhead_but_net_faster[
                "cold_plus_all_hit_net_gain_percent"
            ],
            20.0,
        )

        net_slower = analyze(21.0)
        self.assertFalse(net_slower["passed"])
        self.assertFalse(
            net_slower["checks"][
                "cold_plus_all_hit_net_gain_positive"
            ]
        )

    def test_project_cache_accepts_changed_invalidated_page_only(
        self,
    ) -> None:
        protocol = finalization.load_protocol()
        stages = (
            "detection",
            "ocr",
            "inpaint",
            "translation",
            "render",
        )

        def page_contract(render_digest: str, first_page_digest: str):
            return {
                "detection_sha256": "d" * 64,
                "ocr_sha256": "o" * 64,
                "render_sha256": render_digest,
                "pages": [
                    {
                        "page_index": 0,
                        "block_count": 1,
                        "output_exists": True,
                        "output_sha256": first_page_digest,
                    },
                    {
                        "page_index": 1,
                        "block_count": 1,
                        "output_exists": True,
                        "output_sha256": "b" * 64,
                    },
                ],
            }

        cold = {
            "status": "passed",
            "performance_stats": {
                "run_wall_ms": 10_000,
            },
            "page_contract": page_contract("r" * 64, "a" * 64),
        }
        all_hit_entries = [
            {
                "page_index": page_index,
                "stage": stage,
                "project_checkpoint_status": "hit",
                "skip_reason": "",
                "render_skipped": stage == "render",
                "output_materialized": False,
            }
            for page_index in range(2)
            for stage in stages
        ]
        hit = {
            "status": "passed",
            "performance_stats": {
                "run_wall_ms": 1_000,
                "runtime": {
                    "paddleocr_vl": {"start_count": 0},
                    "gemma": {"start_count": 0},
                },
                "paddleocr_vl": {"http_attempt_count": 0},
                "gemma": {"gemma_http_attempt_count": 0},
            },
            "page_contract": page_contract("r" * 64, "a" * 64),
            "checkpoint_events": {
                "page_count": 2,
                "unmatched_event_count": 0,
                "entries": all_hit_entries,
            },
        }
        missing_entries = [
            {
                **entry,
                "output_materialized": entry["stage"] == "render",
            }
            for entry in all_hit_entries
        ]
        missing = {
            **hit,
            "checkpoint_events": {
                "page_count": 2,
                "unmatched_event_count": 0,
                "entries": missing_entries,
            },
        }
        partial_entries = []
        for entry in all_hit_entries:
            updated = dict(entry)
            if (
                updated["page_index"] == 0
                and updated["stage"] != "detection"
            ):
                updated["project_checkpoint_status"] = "refreshed"
                updated["render_skipped"] = False
            partial_entries.append(updated)
        partial = {
            "status": "passed",
            "invalidated_page_index": 0,
            "performance_stats": {
                "run_wall_ms": 3_000,
                "paddleocr_vl": {"http_attempt_count": 1},
                "gemma": {"gemma_http_attempt_count": 1},
            },
            "page_contract": page_contract("s" * 64, "c" * 64),
            "checkpoint_events": {
                "page_count": 2,
                "unmatched_event_count": 0,
                "entries": partial_entries,
            },
        }
        analysis = finalization.analyze_cache_results(
            protocol=protocol,
            scenario="project",
            results={
                "stabilization": [
                    {**cold, "candidate": "enabled_empty_cold"},
                    {**cold, "candidate": "disabled_cold"},
                ],
                "disabled_cold": [
                    {**cold, "round": round_index}
                    for round_index in range(1, 4)
                ],
                "enabled_empty_cold": [
                    {**cold, "round": round_index}
                    for round_index in range(1, 4)
                ],
                "all_hit_existing_output": hit,
                "all_hit_missing_output": missing,
                "single_page_ocr_invalidated": partial,
            },
        )

        self.assertTrue(analysis["passed"])
        self.assertTrue(
            analysis["checks"][
                "unaffected_pages_exact_after_partial_recompute"
            ]
        )
        self.assertTrue(
            analysis["checks"]["cached_render_output_exact"]
        )
        self.assertEqual(
            analysis["cold_plus_all_hit_net_gain_percent"],
            45.0,
        )
        self.assertEqual(
            analysis["cold_plus_one_page_edit_net_gain_percent"],
            35.0,
        )

    def test_translation_screen_requires_key_and_runtime_contracts(
        self,
    ) -> None:
        protocol = finalization.load_protocol()
        family = finalization._family(protocol, "gemma-model")
        results = []
        for candidate, elapsed in (("iq4-nl", 10.0), ("iq4-xs", 8.0)):
            for round_index in range(1, 4):
                results.append(
                    {
                        "candidate": candidate,
                        "round": round_index,
                        "elapsed_sec": elapsed,
                        "output_count": 54,
                        "nonempty_count": 54,
                        "severe_telemetry_count": 0,
                        "output_key_sha256": "k" * 64,
                        "model_contract_valid": not (
                            candidate == "iq4-xs" and round_index == 3
                        ),
                    }
                )

        analysis = finalization._analyze_translation_results(
            protocol=protocol,
            family=family,
            baseline_id="iq4-nl",
            results=results,
        )
        candidate = next(
            item
            for item in analysis["candidates"]
            if item["candidate"] == "iq4-xs"
        )

        self.assertFalse(candidate["structural_gate_passed"])
        self.assertEqual(candidate["promotion_status"], "rejected")

    def test_private_translation_review_requires_stable_54_row_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = []
            for round_index in range(1, 4):
                path = root / f"round-{round_index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "outputs": [
                                {
                                    "language": language,
                                    "index": index,
                                    "case_id": f"{language}-{index}",
                                    "source": f"source-{language}-{index}",
                                    "translation": (
                                        f"translation-{round_index}-"
                                        f"{language}-{index}"
                                    ),
                                }
                                for language in (
                                    "Japanese",
                                    "Chinese",
                                    "English",
                                )
                                for index in range(18)
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                results.append(
                    {
                        "candidate": "baseline",
                        "round": round_index,
                        "order": 1,
                        "private_output": str(path),
                    }
                )
            output_path = root / "translation-review.json"
            artifact = finalization._write_private_translation_review(
                results,
                output_path,
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(artifact["row_count"], 54)
        self.assertEqual(payload["row_count"], 54)
        self.assertEqual(len(payload["run_labels"]), 3)
        self.assertEqual(
            len(payload["rows"][0]["outputs"]),
            3,
        )

    def test_project_event_gates_distinguish_all_hit_and_one_page_recompute(
        self,
    ) -> None:
        stages = (
            "detection",
            "ocr",
            "inpaint",
            "translation",
            "render",
        )
        all_hit_entries = [
            {
                "page_index": page_index,
                "stage": stage,
                "project_checkpoint_status": "hit",
                "cache_status": "project-checkpoint",
                "skip_reason": "",
                "render_skipped": stage == "render",
                "output_materialized": False,
            }
            for page_index in range(2)
            for stage in stages
        ]
        all_hit = {
            "checkpoint_events": {
                "page_count": 2,
                "unmatched_event_count": 0,
                "entries": all_hit_entries,
            }
        }
        self.assertTrue(
            finalization._project_checkpoint_event_gate(
                all_hit,
                require_render_materialized=False,
            )
        )

        partial_entries = []
        for entry in all_hit_entries:
            updated = dict(entry)
            if (
                updated["page_index"] == 0
                and updated["stage"] != "detection"
            ):
                updated["project_checkpoint_status"] = "refreshed"
                updated["render_skipped"] = False
            partial_entries.append(updated)
        partial = {
            "checkpoint_events": {
                "page_count": 2,
                "unmatched_event_count": 0,
                "entries": partial_entries,
            }
        }
        self.assertTrue(
            finalization._project_partial_recompute_gate(
                partial,
                page_index=0,
                invalidated_stage="ocr",
            )
        )

    def test_checkpoint_event_contract_maps_private_paths_to_page_indexes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot_path = root / "page_snapshots.json"
            metrics_path = root / "metrics.jsonl"
            snapshot_path.write_text(
                json.dumps(
                    {
                        "pages": [
                            {"image_path": "C:/private/page.png"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            metrics_path.write_text(
                json.dumps(
                    {
                        "tag": "detect_end",
                        "image_path": "C:/private/page.png",
                        "project_checkpoint_status": "hit",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            contract = finalization._checkpoint_event_contract(
                metrics_path,
                snapshot_path,
            )

        self.assertEqual(contract["unmatched_event_count"], 0)
        self.assertEqual(contract["entries"][0]["page_index"], 0)
        self.assertNotIn(
            "private",
            json.dumps(contract, ensure_ascii=False),
        )

    def test_owned_render_cleanup_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root.parent / "outside-render.png"
            outside.write_bytes(b"image")
            result = {
                "page_contract": {
                    "pages": [{"output_path": str(outside)}]
                }
            }
            try:
                with self.assertRaises(ValueError):
                    finalization._delete_owned_render_outputs(
                        result,
                        owned_root=root,
                    )
                self.assertTrue(outside.is_file())
            finally:
                outside.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
