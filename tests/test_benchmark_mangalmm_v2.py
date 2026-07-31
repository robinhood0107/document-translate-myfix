from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_mangalmm_v2 import (
    ANNOTATION_SCHEMA_VERSION,
    DETECTOR_SNAPSHOT_SCHEMA_VERSION,
    EVALUATION_PROTOCOL_VERSION,
    FULL_PAGE_PROFILES,
    HISTORICAL_AUDIT,
    OFFICIAL_MANGA_OCR_PROMPT,
    PROFILE_ROUND_ORDER,
    PROFILE_RUN_PROTOCOL_VERSION,
    PROFILE_RUN_SCHEMA_VERSION,
    ROOT,
    BenchmarkContractError,
    FullPageProfile,
    _BenchmarkSettings,
    audit_history,
    build_profile_execution_plan,
    canonical_json_sha256,
    detect_repetition,
    evaluate_profile_regions,
    load_external_manifest,
    profile_escalation_reasons,
    require_external_path,
    summarize_profile_results,
    validate_detector_snapshot,
    validate_evaluation_manifest,
    validate_profile_matrix,
    validate_profile_results_file,
    validate_profile_run_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MangaLMMV2BenchmarkTests(unittest.TestCase):
    def _write_profile_result_fixture(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        run_units = []
        for unit_index, (round_index, profile_id) in enumerate(
            (
                (round_number, candidate)
                for round_number, candidates in enumerate(
                    PROFILE_ROUND_ORDER,
                    start=1,
                )
                for candidate in candidates
            )
        ):
            artifact_dir = root / f"unit-{unit_index}" / "neutral-case"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            raw_path = artifact_dir / "raw-response.txt"
            raw_path.write_text('{"bbox_2d":[1,2,3,4]}', encoding="utf-8")
            overlay_path = artifact_dir / "mapped-overlay.png"
            overlay_path.write_bytes(b"neutral-overlay")
            case_result = {
                "round": round_index,
                "profile_id": profile_id,
                "case_id": "neutral-case",
                "request_seconds": 1.0 + unit_index,
                "request": {
                    "raw_response_file": str(raw_path.relative_to(root)),
                    "raw_response_sha256": _sha256(raw_path),
                    "finish_reason": "stop",
                    "parser_error_code": "",
                },
                "evaluation": {
                    "required_region_count": 1,
                    "matched_required_region_count": 1,
                    "exact_predicted_duplicate_count": 0,
                    "matches": [
                        {
                            "region_id": "dialogue-1",
                            "processing_action": "translate_inpaint",
                            "normalized_text_exact": True,
                        },
                        {
                            "region_id": "ui-1",
                            "processing_action": "preserve",
                            "normalized_text_exact": False,
                        },
                    ],
                },
                "overlay_file": str(overlay_path.relative_to(root)),
                "overlay_sha256": _sha256(overlay_path),
            }
            case_result["result_sha256"] = canonical_json_sha256(case_result)
            run_units.append(
                {
                    "round": round_index,
                    "profile_id": profile_id,
                    "startup_seconds": 2.0,
                    "cases": [case_result],
                }
            )

        blind_rows = []
        for unit_index, unit in enumerate(run_units):
            label_by_profile = {
                PROFILE_ROUND_ORDER[0][0]: "A",
                PROFILE_ROUND_ORDER[0][1]: "B",
            }
            source_overlay = (
                root
                / unit["cases"][0]["overlay_file"]
            )
            blind_overlay = (
                root
                / "blind"
                / f"round-{unit['round']}"
                / "neutral-case"
                / f"candidate-{unit_index}.png"
            )
            blind_overlay.parent.mkdir(parents=True, exist_ok=True)
            blind_overlay.write_bytes(source_overlay.read_bytes())
            blind_rows.append(
                {
                    "round": unit["round"],
                    "case_id": "neutral-case",
                    "candidate": label_by_profile[unit["profile_id"]],
                    "overlay_file": str(blind_overlay.relative_to(root)),
                    "overlay_sha256": _sha256(blind_overlay),
                }
            )
        blind_rows.sort(
            key=lambda row: (
                int(row["round"]),
                str(row["case_id"]),
                str(row["candidate"]),
            )
        )
        blind = {
            "protocol_version": PROFILE_RUN_PROTOCOL_VERSION,
            "candidate_labels": ["A", "B"],
            "rows": blind_rows,
            "unblind_forbidden_until_review_complete": True,
        }
        blind["review_sha256"] = canonical_json_sha256(blind)
        (root / "blind-review.json").write_text(
            json.dumps(blind),
            encoding="utf-8",
        )
        key = {
            "protocol_version": PROFILE_RUN_PROTOCOL_VERSION,
            "profile_by_candidate": {
                "A": PROFILE_ROUND_ORDER[0][0],
                "B": PROFILE_ROUND_ORDER[0][1],
            },
            "review_sha256": blind["review_sha256"],
        }
        key["key_sha256"] = canonical_json_sha256(key)
        (root / "unblind-key.json").write_text(
            json.dumps(key),
            encoding="utf-8",
        )
        payload = {
            "schema_version": PROFILE_RUN_SCHEMA_VERSION,
            "protocol_version": PROFILE_RUN_PROTOCOL_VERSION,
            "manifest_contract_sha256": "3" * 64,
            "profile_contract": validate_profile_matrix(),
            "split": "development",
            "case_ids": ["neutral-case"],
            "initial_gpu": {},
            "max_idle_gpu_used_mib": 2048.0,
            "run_units": run_units,
            "blind_review_sha256": blind["review_sha256"],
            "unblind_key_sha256": key["key_sha256"],
        }
        payload["result_sha256"] = canonical_json_sha256(payload)
        result_path = root / "profile-results.json"
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        return result_path

    def _write_valid_fixture(
        self,
        root: Path,
        *,
        split: str = "development",
        frozen: bool = False,
        with_detector_snapshot: bool = False,
    ) -> tuple[Path, dict]:
        root.mkdir(parents=True, exist_ok=True)
        source = root / "source.bin"
        source.write_bytes(b"neutral-image-fixture")
        source_sha = _sha256(source)
        annotation = root / "annotation.json"
        annotation.write_text(
            json.dumps(
                {
                    "schema_version": ANNOTATION_SCHEMA_VERSION,
                    "case_id": "translucent-screen-dev",
                    "source_sha256": source_sha,
                    "regions": [
                        {
                            "region_id": "dialogue-1",
                            "bbox_xyxy": [10, 20, 100, 140],
                            "original_text": "fixture",
                            "semantic_role": "dialogue_bubble",
                            "processing_action": "translate_inpaint",
                            "bubble_type": "translucent",
                            "human_translation_expected": True,
                        },
                        {
                            "region_id": "micro-ui-1",
                            "polygon": [[2, 2], [8, 2], [8, 9], [2, 9]],
                            "original_text": "",
                            "semantic_role": "ui_or_sign",
                            "processing_action": "preserve",
                            "bubble_type": "none",
                            "human_translation_expected": False,
                        },
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        manifest = {
            "protocol_version": EVALUATION_PROTOCOL_VERSION,
            "corpus_id": "neutral-corpus",
            "cases": [
                {
                    "case_id": "translucent-screen-dev",
                    "split": split,
                    "frozen_before_candidate_run": frozen,
                    "source_image": str(source),
                    "source_sha256": source_sha,
                    "annotation": str(annotation),
                    "annotation_sha256": _sha256(annotation),
                }
            ],
        }
        if with_detector_snapshot:
            snapshot = root / "detector-snapshot.json"
            snapshot.write_text(
                json.dumps(
                    {
                        "schema_version": DETECTOR_SNAPSHOT_SCHEMA_VERSION,
                        "case_id": "translucent-screen-dev",
                        "source_sha256": source_sha,
                        "source_decoded_sha256": "1" * 64,
                        "source_shape_hw": [200, 160],
                        "detector_identity": {
                            "detector": "RT-DETR-v2",
                            "schema_version": 1,
                        },
                        "detector_fingerprint": "2" * 64,
                        "blocks": [
                            {
                                "block_id": "block-0000",
                                "text_bbox_xyxy": [10, 20, 100, 140],
                                "bubble_bbox_xyxy": [5, 10, 120, 160],
                                "text_class": "text_bubble",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest["cases"][0].update(
                {
                    "detector_snapshot": str(snapshot),
                    "detector_snapshot_sha256": _sha256(snapshot),
                }
            )
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path, manifest

    def test_historical_audit_verifies_every_locked_commit(self) -> None:
        entries_by_commit = {entry.commit: entry for entry in HISTORICAL_AUDIT}

        def fake_git(_root: Path, *args: str) -> str:
            if args[0] == "rev-parse":
                return f"{args[1]}\n"
            if args[0] == "show":
                commit = args[1].split(":", 1)[0]
                return "\n".join(entries_by_commit[commit].required_needles)
            raise AssertionError(args)

        result = audit_history(ROOT, git_reader=fake_git)

        self.assertEqual(result["audit_entry_count"], len(HISTORICAL_AUDIT))
        self.assertEqual(
            {entry["status"] for entry in result["entries"]},
            {"verified"},
        )
        self.assertEqual(len(result["audit_sha256"]), 64)

    def test_live_historical_audit_when_checkout_contains_history(self) -> None:
        available = subprocess.run(
            ["git", "cat-file", "-e", f"{HISTORICAL_AUDIT[0].commit}^{{commit}}"],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        if available.returncode != 0:
            self.skipTest("CI checkout does not contain the historical audit commits.")

        result = audit_history(ROOT)

        self.assertEqual(result["audit_entry_count"], len(HISTORICAL_AUDIT))

    def test_valid_external_manifest_reports_roles_without_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, manifest = self._write_valid_fixture(Path(directory))

            loaded = load_external_manifest(manifest_path)
            summary = validate_evaluation_manifest(loaded)

        self.assertEqual(summary["case_count"], 1)
        self.assertEqual(summary["split_counts"], {"development": 1})
        self.assertEqual(
            summary["cases"][0]["role_counts"],
            {"dialogue_bubble": 1, "ui_or_sign": 1},
        )
        self.assertNotIn("source_image", json.dumps(summary))
        self.assertEqual(loaded, manifest)

    def test_holdout_requires_pre_execution_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = self._write_valid_fixture(
                Path(directory),
                split="holdout",
                frozen=False,
            )
            with self.assertRaisesRegex(
                BenchmarkContractError,
                "must be frozen",
            ):
                validate_evaluation_manifest(manifest)

    def test_source_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = self._write_valid_fixture(Path(directory))
            manifest["cases"][0]["source_sha256"] = "0" * 64

            with self.assertRaisesRegex(
                BenchmarkContractError,
                "source_image SHA-256 mismatch",
            ):
                validate_evaluation_manifest(manifest)

    def test_duplicate_case_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = self._write_valid_fixture(Path(directory))
            manifest["cases"].append(dict(manifest["cases"][0]))

            with self.assertRaisesRegex(BenchmarkContractError, "Duplicate case_id"):
                validate_evaluation_manifest(manifest)

    def test_duplicate_source_hashes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = self._write_valid_fixture(Path(directory))
            duplicate = dict(manifest["cases"][0])
            duplicate["case_id"] = "translucent-screen-dev-copy"
            manifest["cases"].append(duplicate)

            with self.assertRaisesRegex(
                BenchmarkContractError,
                "Duplicate source_sha256",
            ):
                validate_evaluation_manifest(manifest)

    def test_duplicate_region_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = self._write_valid_fixture(Path(directory))
            annotation_path = Path(manifest["cases"][0]["annotation"])
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            annotation["regions"].append(dict(annotation["regions"][0]))
            annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
            manifest["cases"][0]["annotation_sha256"] = _sha256(annotation_path)

            with self.assertRaisesRegex(BenchmarkContractError, "Duplicate region_id"):
                validate_evaluation_manifest(manifest)

    def test_sfx_cannot_be_routed_to_translate_inpaint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = self._write_valid_fixture(Path(directory))
            annotation_path = Path(manifest["cases"][0]["annotation"])
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            region = annotation["regions"][0]
            region["semantic_role"] = "sfx"
            region["processing_action"] = "translate_inpaint"
            annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
            manifest["cases"][0]["annotation_sha256"] = _sha256(annotation_path)

            with self.assertRaisesRegex(
                BenchmarkContractError,
                "must preserve role",
            ):
                validate_evaluation_manifest(manifest)

    def test_ambiguous_region_must_route_to_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = self._write_valid_fixture(Path(directory))
            annotation_path = Path(manifest["cases"][0]["annotation"])
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            region = annotation["regions"][0]
            region["semantic_role"] = "ambiguous"
            region["processing_action"] = "translate_inpaint"
            annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
            manifest["cases"][0]["annotation_sha256"] = _sha256(annotation_path)

            with self.assertRaisesRegex(
                BenchmarkContractError,
                "must route ambiguous text to review",
            ):
                validate_evaluation_manifest(manifest)

    def test_manifest_and_outputs_must_stay_outside_git_tree(self) -> None:
        with self.assertRaisesRegex(
            BenchmarkContractError,
            "outside the Git working tree",
        ):
            require_external_path(
                ROOT / "benchmarks/mangalmm_v2/evaluation-manifest.example.json",
                "Evaluation manifest",
            )

    def test_profile_matrix_locks_capacity_and_ab_ba_order(self) -> None:
        result = validate_profile_matrix()

        self.assertEqual(
            [item["profile_id"] for item in result["profiles"]],
            list(PROFILE_ROUND_ORDER[0]),
        )
        self.assertEqual(
            result["round_order"][1],
            list(reversed(PROFILE_ROUND_ORDER[0])),
        )
        self.assertGreaterEqual(
            FULL_PAGE_PROFILES[1].capacity_key(),
            FULL_PAGE_PROFILES[0].capacity_key(),
        )
        self.assertEqual(
            result["official_prompt"],
            "Please perform OCR on this image and output the recognized "
            "Japanese text along with its position (grounding).",
        )
        self.assertEqual(OFFICIAL_MANGA_OCR_PROMPT, result["official_prompt"])

    def test_profile_matrix_rejects_dense_capacity_reduction(self) -> None:
        standard = FULL_PAGE_PROFILES[0]
        reduced = FullPageProfile(
            profile_id="high-reduced",
            short_side=standard.short_side,
            long_side=standard.long_side - 1,
            max_pixels=standard.max_pixels,
            context_size=standard.context_size,
        )

        with self.assertRaisesRegex(
            BenchmarkContractError,
            "profile|capacity|order",
        ):
            validate_profile_matrix((standard, reduced))

    def test_profile_request_timeout_uses_cli_contract(self) -> None:
        settings = _BenchmarkSettings(
            FULL_PAGE_PROFILES[0],
            request_timeout_sec=47,
        )

        self.assertEqual(
            settings.get_mangalmm_ocr_settings()["request_timeout_sec"],
            47,
        )

    def test_profile_run_requires_hashed_detector_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = self._write_valid_fixture(Path(directory))
            with self.assertRaisesRegex(
                BenchmarkContractError,
                "frozen detector snapshot",
            ):
                validate_profile_run_manifest(manifest)

            _, ready_manifest = self._write_valid_fixture(
                Path(directory) / "ready",
                with_detector_snapshot=True,
            )
            result = validate_profile_run_manifest(ready_manifest)

        self.assertTrue(result["profile_run_ready"])
        self.assertEqual(
            result["cases"][0]["detector_snapshot"]["block_count"],
            1,
        )

    def test_detector_snapshot_rejects_geometry_outside_image(self) -> None:
        snapshot = {
            "schema_version": DETECTOR_SNAPSHOT_SCHEMA_VERSION,
            "case_id": "neutral-case",
            "source_sha256": "0" * 64,
            "source_decoded_sha256": "1" * 64,
            "source_shape_hw": [100, 100],
            "detector_identity": {"detector": "RT-DETR-v2"},
            "detector_fingerprint": "2" * 64,
            "blocks": [
                {
                    "block_id": "block-0000",
                    "text_bbox_xyxy": [80, 80, 120, 120],
                    "text_class": "text_free",
                }
            ],
        }

        with self.assertRaisesRegex(
            BenchmarkContractError,
            "outside image bounds",
        ):
            validate_detector_snapshot(
                snapshot,
                case_id="neutral-case",
                source_sha256="0" * 64,
            )

    def test_profile_execution_plan_is_fixed_ab_ba(self) -> None:
        manifest = {
            "cases": [
                {"case_id": "case-b", "split": "development"},
                {"case_id": "case-a", "split": "development"},
                {"case_id": "case-z", "split": "holdout"},
            ]
        }

        plan = build_profile_execution_plan(manifest)

        self.assertEqual(
            [(item["round"], item["profile_id"]) for item in plan],
            [
                (1, PROFILE_ROUND_ORDER[0][0]),
                (1, PROFILE_ROUND_ORDER[0][1]),
                (2, PROFILE_ROUND_ORDER[1][0]),
                (2, PROFILE_ROUND_ORDER[1][1]),
            ],
        )
        self.assertEqual(plan[0]["case_ids"], ["case-a", "case-b"])

    def test_region_evaluation_flags_required_coverage_gap(self) -> None:
        annotation = {
            "regions": [
                {
                    "region_id": "dialogue-1",
                    "bbox_xyxy": [10, 10, 60, 60],
                    "original_text": "必要",
                    "processing_action": "translate_inpaint",
                    "semantic_role": "dialogue_bubble",
                },
                {
                    "region_id": "ui-1",
                    "bbox_xyxy": [70, 70, 90, 90],
                    "original_text": "",
                    "processing_action": "preserve",
                    "semantic_role": "ui_or_sign",
                },
            ]
        }

        missing = evaluate_profile_regions(
            annotation=annotation,
            predicted_regions=[],
        )
        matched = evaluate_profile_regions(
            annotation=annotation,
            predicted_regions=[
                {"bbox_xyxy": [8, 8, 62, 62], "text": "必要"}
            ],
        )

        self.assertTrue(missing["coverage_gap"])
        self.assertFalse(matched["coverage_gap"])
        self.assertTrue(matched["matches"][0]["normalized_text_exact"])
        self.assertEqual(
            matched["matches"][0]["processing_action"],
            "translate_inpaint",
        )

    def test_escalation_is_limited_to_locked_failure_reasons(self) -> None:
        repeated = "反復" * 300

        self.assertTrue(detect_repetition(repeated))
        reasons = profile_escalation_reasons(
            request_metadata={
                "finish_reason": "length",
                "parser_error_code": "invalid_json",
                "raw_response": repeated,
            },
            evaluation={"coverage_gap": True},
        )

        self.assertEqual(
            reasons,
            [
                "coverage_gap",
                "finish_reason_length",
                "parser_error",
                "repetition",
            ],
        )

    def test_profile_result_validation_covers_external_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_path = self._write_profile_result_fixture(Path(directory))

            verified = validate_profile_results_file(result_path)
            summary = summarize_profile_results(verified["payload"])

        self.assertEqual(verified["status"], "verified")
        self.assertEqual(verified["case_result_count"], 4)
        self.assertEqual(verified["blind_overlay_count"], 4)
        self.assertEqual(summary["decision"], "manual_review_required")
        for profile in summary["profiles"]:
            self.assertEqual(profile["required_total"], 2)
            self.assertEqual(profile["required_normalized_exact"], 2)
            self.assertTrue(profile["required_exact_available"])

    def test_profile_result_validation_rejects_overlay_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = self._write_profile_result_fixture(root)
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            overlay = (
                root
                / payload["run_units"][0]["cases"][0]["overlay_file"]
            )
            overlay.write_bytes(b"tampered-overlay")

            with self.assertRaisesRegex(
                BenchmarkContractError,
                "Overlay SHA-256 mismatch",
            ):
                validate_profile_results_file(result_path)

    def test_profile_result_validation_rejects_raw_response_tampering(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = self._write_profile_result_fixture(root)
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            raw_response = (
                root
                / payload["run_units"][0]["cases"][0]["request"][
                    "raw_response_file"
                ]
            )
            raw_response.write_text("tampered", encoding="utf-8")

            with self.assertRaisesRegex(
                BenchmarkContractError,
                "Raw response SHA-256 mismatch",
            ):
                validate_profile_results_file(result_path)

    def test_profile_result_validation_rejects_result_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_path = self._write_profile_result_fixture(Path(directory))
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["split"] = "holdout"
            result_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                BenchmarkContractError,
                "Profile result SHA-256 mismatch",
            ):
                validate_profile_results_file(result_path)

    def test_profile_result_validation_rejects_blind_overlay_tampering(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = self._write_profile_result_fixture(root)
            blind = json.loads(
                (root / "blind-review.json").read_text(encoding="utf-8")
            )
            blind_overlay = root / blind["rows"][0]["overlay_file"]
            blind_overlay.write_bytes(b"tampered-blind-overlay")

            with self.assertRaisesRegex(
                BenchmarkContractError,
                "Blind overlay SHA-256 mismatch",
            ):
                validate_profile_results_file(result_path)


if __name__ == "__main__":
    unittest.main()
