from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_gemma_final_ab as final_ab  # noqa: E402


class GemmaFinalABBenchmarkTests(unittest.TestCase):
    def _write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _output_rows(
        self,
        *,
        candidate: str,
        round_index: int,
    ) -> list[dict[str, object]]:
        rows = []
        for row_number, (page_index, block_index) in enumerate(
            ((1, 1), (1, 2), (2, 1)),
            start=1,
        ):
            source = f"neutral source {row_number}"
            variant = {
                final_ab.CANDIDATE_BASELINE: "variant-x",
                final_ab.CANDIDATE_GROUPED_F16: "variant-y",
                final_ab.CANDIDATE_GROUPED_Q8: "variant-z",
            }[candidate]
            translation = (
                f"neutral translation {variant} "
                f"round {round_index} row {row_number}"
            )
            rows.append(
                {
                    "row_id": f"p{page_index:03d}-b{block_index:03d}",
                    "page_index": page_index,
                    "page_name": f"page-{page_index:03d}.dat",
                    "block_index": block_index,
                    "source": source,
                    "source_sha256": hashlib.sha256(
                        source.encode("utf-8")
                    ).hexdigest(),
                    "translation": translation,
                    "translation_sha256": hashlib.sha256(
                        translation.encode("utf-8")
                    ).hexdigest(),
                    "empty": False,
                    "structural_output": False,
                }
            )
        return rows

    def _source_suite(
        self,
        root: Path,
    ) -> tuple[Path, final_ab.SourceContract]:
        source_dir = root / "source-suite"
        runs_dir = source_dir / "runs"
        runs_dir.mkdir(parents=True)

        input_files = [
            {
                "name": "page-001.dat",
                "size_bytes": 11,
                "sha256": "1" * 64,
            },
            {
                "name": "page-002.dat",
                "size_bytes": 12,
                "sha256": "2" * 64,
            },
        ]
        input_manifest_sha256 = final_ab.canonical_sha256(input_files)
        page_contract = [
            {
                "page_index": 1,
                "page_name": "page-001.dat",
                "block_count": 2,
                "ordered_source_sha256": [
                    hashlib.sha256(
                        f"neutral source {index}".encode("utf-8")
                    ).hexdigest()
                    for index in (1, 2)
                ],
            },
            {
                "page_index": 2,
                "page_name": "page-002.dat",
                "block_count": 1,
                "ordered_source_sha256": [
                    hashlib.sha256(
                        b"neutral source 3"
                    ).hexdigest()
                ],
            },
        ]
        snapshot_contract_sha256 = final_ab.canonical_sha256(page_contract)
        snapshot_sha256 = "3" * 64
        model_name = "neutral-model.gguf"
        model_size = 1024
        model_sha256 = "4" * 64
        image_id = "sha256:" + ("5" * 64)
        helper_image_id = "sha256:" + ("6" * 64)
        runtime_fingerprints = {
            final_ab.CANDIDATE_BASELINE: "7" * 64,
            final_ab.CANDIDATE_GROUPED_F16: "8" * 64,
            final_ab.CANDIDATE_GROUPED_Q8: "9" * 64,
        }
        measurement_environment = {
            "docker_server": {"available": True, "version": "neutral"},
            "nvidia_driver": {
                "available": True,
                "gpus": [{"name": "neutral GPU"}],
            },
        }
        behavior_without_digest = {
            "benchmark_protocol_version": 3,
            "source_sha256": {"neutral.py": "a" * 64},
            "engine": {
                "source_language": "Japanese",
                "target_language": "Korean",
                "max_completion_tokens": 512,
                "contextual_merge_input": True,
                "persistent_cache_enabled": False,
                "exact_tm_enabled": False,
            },
            "candidate_request_contract": {
                final_ab.CANDIDATE_BASELINE: {
                    "request_mode": "contextual-single",
                    "group_size": 6,
                },
                final_ab.CANDIDATE_GROUPED_F16: {
                    "request_mode": "contextual-grouped",
                    "group_size": 7,
                },
                final_ab.CANDIDATE_GROUPED_Q8: {
                    "request_mode": "contextual-grouped",
                    "group_size": 7,
                },
            },
        }
        translation_contract_sha256 = final_ab.canonical_sha256(
            behavior_without_digest
        )
        behavior = {
            **behavior_without_digest,
            "contract_sha256": translation_contract_sha256,
        }
        suite_contract = {
            "benchmark_protocol_version": 3,
            "input_manifest_sha256": input_manifest_sha256,
            "snapshot_sha256": snapshot_sha256,
            "snapshot_contract_sha256": snapshot_contract_sha256,
            "model_sha256": model_sha256,
            "image_id": image_id,
            "hash_helper_image_id": helper_image_id,
            "runtime_config_fingerprints": runtime_fingerprints,
            "translation_behavior_contract_sha256": (
                translation_contract_sha256
            ),
            "group_size": 7,
            "max_completion_tokens": 512,
            "measurement_environment": measurement_environment,
        }
        suite_fingerprint = final_ab.canonical_sha256(suite_contract)

        runtime_contracts = []
        for candidate in final_ab.SOURCE_CANDIDATES:
            runtime_contracts.append(
                {
                    "candidate": candidate,
                    "container_name": f"neutral-{candidate}",
                    "container_id": f"container-{candidate}",
                    "image_id": image_id,
                    "config_fingerprint": runtime_fingerprints[candidate],
                    "profile_label": (
                        final_ab.EXPECTED_RUNTIME_PROFILE_LABELS[candidate]
                    ),
                    "command": final_ab._runtime_command(
                        candidate,
                        model_name,
                    ),
                    "model_volume": "neutral-volume",
                    "model_mount_read_only": True,
                    "network_ports": {},
                }
            )
        self._write_json(
            source_dir / "runtime_contracts.json",
            runtime_contracts,
        )
        self._write_json(
            source_dir / "measurement_environment.json",
            measurement_environment,
        )
        self._write_json(
            source_dir / "model_contract.json",
            {
                "model_name": model_name,
                "expected_size_bytes": model_size,
                "source_size_bytes": model_size,
                "volume_size_bytes": model_size,
                "expected_sha256": model_sha256,
                "source_sha256": model_sha256,
                "volume_sha256": model_sha256,
                "volume_name": "neutral-volume",
                "model_copy_verified": True,
                "hash_helper_image": {"image_id": helper_image_id},
            },
        )
        self._write_json(
            source_dir / "frozen_assets.json",
            {
                "expected_input_manifest_sha256": input_manifest_sha256,
                "expected_ocr_snapshot_sha256": snapshot_sha256,
                "input_manifest": {
                    "file_count": 2,
                    "files": input_files,
                    "manifest_sha256": input_manifest_sha256,
                },
                "snapshot": {
                    "snapshot_sha256": snapshot_sha256,
                    "contract_sha256": snapshot_contract_sha256,
                    "page_count": 2,
                    "block_count": 3,
                    "source_language": "Japanese",
                    "target_language": "Korean",
                    "page_contract": page_contract,
                },
            },
        )

        result_sha256: dict[tuple[int, str], str] = {}
        run_records = []
        for round_index, candidate in final_ab.SOURCE_RUN_ORDER:
            failed_q8 = (
                round_index == 3
                and candidate == final_ab.CANDIDATE_GROUPED_Q8
            )
            elapsed = {
                final_ab.CANDIDATE_BASELINE: 20.0 + round_index,
                final_ab.CANDIDATE_GROUPED_F16: 10.0 + round_index,
                final_ab.CANDIDATE_GROUPED_Q8: 11.0 + round_index,
            }[candidate]
            result = {
                "round": round_index,
                "candidate": candidate,
                "status": "failed" if failed_q8 else "passed",
                "contract_fingerprint": suite_fingerprint,
                "container_name": f"neutral-{candidate}",
                "container_id": f"container-{candidate}",
                "container_stopped": True,
                "page_count": 2,
                "block_count": 3,
                "request_mode": (
                    "contextual-single"
                    if candidate == final_ab.CANDIDATE_BASELINE
                    else "contextual-grouped"
                ),
                "configured_group_size": (
                    6
                    if candidate == final_ab.CANDIDATE_BASELINE
                    else 7
                ),
                "translation_elapsed_sec": elapsed,
                "gates": {
                    "page_count_ok": True,
                    "block_count_ok": True,
                    "order_preserved": True,
                    "empty_translation_count": 0,
                    "structural_output_count": 0,
                    "structural_telemetry_count": 0,
                    "unresolved_failure_count": 0,
                    "clean_run_telemetry_count": 3 if failed_q8 else 0,
                    "request_contract_passed": not failed_q8,
                    "hard_gate_passed": not failed_q8,
                    "clean_run_passed": not failed_q8,
                },
                "stats": {
                    "gemma_tm_result_cache_hit_count": 0,
                    "gemma_tm_exact_hit_count": 0,
                    "gemma_tm_cache_write_count": 0,
                    "gemma_tm_runtime_skipped_count": 0,
                    "gemma_partial_response_count": 1 if failed_q8 else 0,
                    "gemma_partial_fallback_block_count": (
                        1 if failed_q8 else 0
                    ),
                    "gemma_invalid_value_count": 1 if failed_q8 else 0,
                },
                "outputs": self._output_rows(
                    candidate=candidate,
                    round_index=round_index,
                ),
            }
            relative = final_ab.SOURCE_RUN_FILENAMES[
                (round_index, candidate)
            ]
            result_path = source_dir / relative
            self._write_json(result_path, result)
            digest = final_ab.sha256_file(result_path)
            result_sha256[(round_index, candidate)] = digest
            run_records.append(
                {
                    "round": round_index,
                    "candidate": candidate,
                    "status": result["status"],
                    "translation_elapsed_sec": elapsed,
                    "hard_gate_passed": not failed_q8,
                    "clean_run_passed": not failed_q8,
                    "result_file": relative,
                    "result_sha256": digest,
                }
            )
        self._write_json(
            source_dir / "suite_state.json",
            {
                "status": "failed",
                "quality_status": "pending_user_review",
                "full_pipeline_executed": False,
                "contract_fingerprint": suite_fingerprint,
                "suite_contract": suite_contract,
                "translation_behavior_contract": behavior,
                "runs": run_records,
            },
        )
        contract = final_ab.SourceContract(
            protocol_version=3,
            page_count=2,
            block_count=3,
            suite_fingerprint=suite_fingerprint,
            input_manifest_sha256=input_manifest_sha256,
            snapshot_sha256=snapshot_sha256,
            snapshot_contract_sha256=snapshot_contract_sha256,
            model_name=model_name,
            model_size_bytes=model_size,
            model_sha256=model_sha256,
            image_id=image_id,
            hash_helper_image_id=helper_image_id,
            translation_contract_sha256=translation_contract_sha256,
            runtime_fingerprints=runtime_fingerprints,
            run_order=final_ab.SOURCE_RUN_ORDER,
            run_filenames=final_ab.SOURCE_RUN_FILENAMES,
            run_sha256=result_sha256,
            round_difference_counts={
                final_ab.CANDIDATE_BASELINE: 3,
                final_ab.CANDIDATE_GROUPED_F16: 3,
            },
            q8_slowdown_percent=(
                ((12.5 - 11.5) / 11.5) * 100.0
            ),
        )
        return source_dir, contract

    def _complete_review(
        self,
        path: Path,
        *,
        flagged_column: str | None = None,
    ) -> None:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for column in final_ab.REVIEW_FLAG_COLUMNS:
                row[column] = "no"
        if flagged_column:
            rows[0][flagged_column] = "yes"
            rows[0]["regression_types"] = "action"
            rows[0]["notes"] = "행동 의미가 달라짐"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=final_ab.REVIEW_COLUMNS,
            )
            writer.writeheader()
            writer.writerows(rows)

    def test_import_creates_two_round_blind_review_without_public_leaks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir, contract = self._source_suite(root)
            output_dir = root / "blind-output"
            with patch.object(
                final_ab,
                "LOCKED_SOURCE_CONTRACT",
                contract,
            ), patch.object(
                final_ab.subprocess,
                "run",
                side_effect=AssertionError("no subprocess expected"),
            ):
                state = final_ab.import_clean_suite(
                    source_dir,
                    output_dir,
                    contract=contract,
                    mapping={
                        "A": final_ab.CANDIDATE_BASELINE,
                        "B": final_ab.CANDIDATE_GROUPED_F16,
                    },
                )

            with (output_dir / final_ab.REVIEW_FILENAME).open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as stream:
                rows = list(csv.DictReader(stream))
            public_text = "\n".join(
                (output_dir / filename).read_text(encoding="utf-8-sig")
                for filename in (
                    final_ab.REVIEW_FILENAME,
                    final_ab.REVIEW_HTML_FILENAME,
                    final_ab.REVIEW_INSTRUCTIONS_FILENAME,
                    final_ab.STATE_FILENAME,
                )
            )

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["A1_regression"], "")
        self.assertIn("A1", rows[0])
        self.assertIn("A2", rows[0])
        self.assertIn("B1", rows[0])
        self.assertIn("B2", rows[0])
        self.assertNotIn(final_ab.CANDIDATE_BASELINE, public_text)
        self.assertNotIn(final_ab.CANDIDATE_GROUPED_F16, public_text)
        self.assertNotIn(final_ab.CANDIDATE_GROUPED_Q8, public_text)
        self.assertNotIn("21.0", public_text)
        self.assertIn("window.location.pathname", public_text)
        self.assertFalse(state["docker_or_model_requests_executed"])
        self.assertFalse(state["q8_in_candidate_set"])

    def test_source_result_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_dir, contract = self._source_suite(Path(temporary))
            result_path = source_dir / final_ab.SOURCE_RUN_FILENAMES[
                (1, final_ab.CANDIDATE_GROUPED_F16)
            ]
            result_path.write_text(
                result_path.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "result file digest differs",
            ):
                final_ab.validate_source_suite(
                    source_dir,
                    contract=contract,
                )

    def test_missing_source_result_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_dir, contract = self._source_suite(Path(temporary))
            result_path = source_dir / final_ab.SOURCE_RUN_FILENAMES[
                (2, final_ab.CANDIDATE_BASELINE)
            ]
            result_path.unlink()

            with self.assertRaisesRegex(FileNotFoundError, "is missing"):
                final_ab.validate_source_suite(
                    source_dir,
                    contract=contract,
                )

    def test_source_run_order_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_dir, contract = self._source_suite(Path(temporary))
            state_path = source_dir / "suite_state.json"
            state = final_ab.read_json(state_path)
            state["runs"][0], state["runs"][1] = (
                state["runs"][1],
                state["runs"][0],
            )
            self._write_json(state_path, state)

            with self.assertRaisesRegex(
                ValueError,
                "source run record order differs",
            ):
                final_ab.validate_source_suite(
                    source_dir,
                    contract=contract,
                )

    def test_source_output_order_change_is_rejected_even_with_new_file_hash(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_dir, contract = self._source_suite(Path(temporary))
            run_key = (2, final_ab.CANDIDATE_GROUPED_F16)
            result_path = source_dir / final_ab.SOURCE_RUN_FILENAMES[run_key]
            result = final_ab.read_json(result_path)
            result["outputs"][0], result["outputs"][1] = (
                result["outputs"][1],
                result["outputs"][0],
            )
            self._write_json(result_path, result)
            new_digest = final_ab.sha256_file(result_path)
            state_path = source_dir / "suite_state.json"
            state = final_ab.read_json(state_path)
            for record in state["runs"]:
                if (
                    record["round"],
                    record["candidate"],
                ) == run_key:
                    record["result_sha256"] = new_digest
            self._write_json(state_path, state)
            updated_hashes = dict(contract.run_sha256)
            updated_hashes[run_key] = new_digest
            updated_contract = replace(
                contract,
                run_sha256=updated_hashes,
            )

            with self.assertRaisesRegex(
                ValueError,
                "output order differs",
            ):
                final_ab.validate_source_suite(
                    source_dir,
                    contract=updated_contract,
                )

    def test_q8_cannot_be_added_to_blind_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_dir, contract = self._source_suite(Path(temporary))
            audit = final_ab.validate_source_suite(
                source_dir,
                contract=contract,
            )

            with self.assertRaisesRegex(ValueError, "mapping candidates"):
                final_ab.build_blind_payload(
                    audit,
                    mapping={
                        "A": final_ab.CANDIDATE_BASELINE,
                        "B": final_ab.CANDIDATE_GROUPED_Q8,
                    },
                    contract=contract,
                )

    def test_incomplete_review_cannot_be_unblinded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir, contract = self._source_suite(root)
            output_dir = root / "blind-output"
            final_ab.import_clean_suite(
                source_dir,
                output_dir,
                contract=contract,
                mapping={
                    "A": final_ab.CANDIDATE_BASELINE,
                    "B": final_ab.CANDIDATE_GROUPED_F16,
                },
            )

            with self.assertRaises(final_ab.ReviewIncompleteError):
                final_ab.unblind_review(
                    output_dir,
                    output_dir / final_ab.REVIEW_FILENAME,
                    confirmation="3-ROWS-REVIEWED",
                    contract=contract,
                )
            self.assertFalse(
                (output_dir / final_ab.UNBLIND_SUMMARY_FILENAME).exists()
            )

    def test_existing_output_directory_is_never_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir, contract = self._source_suite(root)
            output_dir = root / "blind-output"
            output_dir.mkdir()

            with self.assertRaisesRegex(
                FileExistsError,
                "already exists",
            ):
                final_ab.import_clean_suite(
                    source_dir,
                    output_dir,
                    contract=contract,
                )

    def test_report_tool_digest_drift_blocks_review_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir, contract = self._source_suite(root)
            output_dir = root / "blind-output"
            final_ab.import_clean_suite(
                source_dir,
                output_dir,
                contract=contract,
                mapping={
                    "A": final_ab.CANDIDATE_BASELINE,
                    "B": final_ab.CANDIDATE_GROUPED_F16,
                },
            )
            state_path = output_dir / final_ab.STATE_FILENAME
            state = final_ab.read_json(state_path)
            state["report_tool_sha256"] = "f" * 64
            self._write_json(state_path, state)

            with self.assertRaisesRegex(
                ValueError,
                "report tool digest differs",
            ):
                final_ab.validate_review_file(
                    output_dir,
                    output_dir / final_ab.REVIEW_FILENAME,
                    contract=contract,
                )

    def test_complete_review_unblinds_and_approves_clean_grouped_output(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir, contract = self._source_suite(root)
            output_dir = root / "blind-output"
            final_ab.import_clean_suite(
                source_dir,
                output_dir,
                contract=contract,
                mapping={
                    "A": final_ab.CANDIDATE_BASELINE,
                    "B": final_ab.CANDIDATE_GROUPED_F16,
                },
            )
            review_path = output_dir / final_ab.REVIEW_FILENAME
            self._complete_review(review_path)

            validation = final_ab.validate_review_file(
                output_dir,
                review_path,
                contract=contract,
            )
            summary = final_ab.unblind_review(
                output_dir,
                review_path,
                confirmation="3-ROWS-REVIEWED",
                contract=contract,
            )

        self.assertEqual(validation["reviewed_row_count"], 3)
        self.assertEqual(summary["status"], "quality_approved")
        self.assertTrue(summary["product_default_implementation_allowed"])
        self.assertFalse(summary["final_product_promotion_allowed"])
        self.assertTrue(summary["full_pipeline_comparison_required"])
        self.assertEqual(
            summary["grouped_candidate_semantic_regression_count"],
            0,
        )

    def test_wrong_explicit_confirmation_cannot_unblind(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir, contract = self._source_suite(root)
            output_dir = root / "blind-output"
            final_ab.import_clean_suite(
                source_dir,
                output_dir,
                contract=contract,
                mapping={
                    "A": final_ab.CANDIDATE_BASELINE,
                    "B": final_ab.CANDIDATE_GROUPED_F16,
                },
            )
            review_path = output_dir / final_ab.REVIEW_FILENAME
            self._complete_review(review_path)

            with self.assertRaisesRegex(
                ValueError,
                "confirmation differs",
            ):
                final_ab.unblind_review(
                    output_dir,
                    review_path,
                    confirmation="WRONG",
                    contract=contract,
                )
            self.assertFalse(
                (output_dir / final_ab.UNBLIND_SUMMARY_FILENAME).exists()
            )

    def test_any_grouped_semantic_regression_stops_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir, contract = self._source_suite(root)
            output_dir = root / "blind-output"
            final_ab.import_clean_suite(
                source_dir,
                output_dir,
                contract=contract,
                mapping={
                    "A": final_ab.CANDIDATE_BASELINE,
                    "B": final_ab.CANDIDATE_GROUPED_F16,
                },
            )
            review_path = output_dir / final_ab.REVIEW_FILENAME
            self._complete_review(
                review_path,
                flagged_column="B1_regression",
            )

            summary = final_ab.unblind_review(
                output_dir,
                review_path,
                confirmation="3-ROWS-REVIEWED",
                contract=contract,
            )

        self.assertEqual(summary["status"], "quality_rejected")
        self.assertFalse(summary["product_default_implementation_allowed"])
        self.assertFalse(summary["final_product_promotion_allowed"])
        self.assertEqual(
            summary["grouped_candidate_semantic_regression_count"],
            1,
        )

    def test_review_immutable_content_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir, contract = self._source_suite(root)
            output_dir = root / "blind-output"
            final_ab.import_clean_suite(
                source_dir,
                output_dir,
                contract=contract,
                mapping={
                    "A": final_ab.CANDIDATE_BASELINE,
                    "B": final_ab.CANDIDATE_GROUPED_F16,
                },
            )
            review_path = output_dir / final_ab.REVIEW_FILENAME
            self._complete_review(review_path)
            with review_path.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["source"] = "changed source"
            with review_path.open(
                "w",
                encoding="utf-8-sig",
                newline="",
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=final_ab.REVIEW_COLUMNS,
                )
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(
                final_ab.ReviewIncompleteError,
                "incomplete or invalid",
            ):
                final_ab.validate_review_file(
                    output_dir,
                    review_path,
                    contract=contract,
                )

    def test_locked_model_contract_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_dir, contract = self._source_suite(Path(temporary))
            model_path = source_dir / "model_contract.json"
            model = final_ab.read_json(model_path)
            model["source_sha256"] = "f" * 64
            self._write_json(model_path, model)

            with self.assertRaisesRegex(
                ValueError,
                "model contract source_sha256 differs",
            ):
                final_ab.validate_source_suite(
                    source_dir,
                    contract=contract,
                )

    def test_contract_can_be_replaced_only_with_matching_locked_digests(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_dir, contract = self._source_suite(Path(temporary))
            wrong_contract = replace(
                contract,
                suite_fingerprint="f" * 64,
            )

            with self.assertRaisesRegex(
                ValueError,
                "source suite contract SHA-256 differs",
            ):
                final_ab.validate_source_suite(
                    source_dir,
                    contract=wrong_contract,
                )


if __name__ == "__main__":
    unittest.main()
