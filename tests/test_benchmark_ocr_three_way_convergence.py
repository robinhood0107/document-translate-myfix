from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark_ocr_three_way_convergence as convergence  # noqa: E402
import benchmark_ocr_three_way_human_truth as three_way  # noqa: E402


class OCRThreeWayConvergenceTests(unittest.TestCase):
    def _json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _replay_fixture(self, root: Path) -> dict[str, Path]:
        source = root / "source" / "page.png"
        source.parent.mkdir(parents=True)
        Image.new("RGB", (100, 80), "white").save(source)
        source_sha = three_way.sha256_file(source)
        detector = root / "detector" / "page.json"
        self._json(
            detector,
            {
                "status": "success",
                "source_sha256": source_sha,
                "blocks": [
                    {
                        "block_id": "block-0",
                        "xyxy": [5, 5, 45, 35],
                        "bubble_xyxy": [3, 3, 48, 38],
                        "text_class": "text_bubble",
                        "direction": "horizontal",
                    }
                ],
            },
        )
        manifest = root / "corpus-manifest.json"
        manifest_payload = {
            "schema_version": three_way.CORPUS_SCHEMA_VERSION,
            "protocol_version": three_way.PROTOCOL_VERSION,
            "suite_id": "manga-live-replay-fixture",
            "pages": [
                {
                    "page_id": "page",
                    "split": "development",
                    "language": "ja",
                    "source_image": {
                        "path": str(source),
                        "sha256": source_sha,
                        "width": 100,
                        "height": 80,
                    },
                    "detector_snapshot": {
                        "path": str(detector),
                        "sha256": three_way.sha256_file(detector),
                    },
                }
            ],
        }
        manifest_payload["manifest_sha256"] = three_way.canonical_sha256(
            manifest_payload
        )
        self._json(manifest, manifest_payload)

        runtime = root / "runtime.json"
        runtime_payload = {
            "schema_version": 1,
            "route_id": "mangalmm_full_page",
            "backend": "llama.cpp",
            "model_sha256": "a" * 64,
            "mmproj_sha256": "b" * 64,
            "command_sha256": "c" * 64,
            "image_digest": "sha256:" + "d" * 64,
            "prompt_mode": "mangalmm_official_full_page",
            "image_max_pixels": three_way.ROUTE_IMAGE_MAX_PIXELS[
                "mangalmm_full_page"
            ],
            "special_tokens": False,
        }
        runtime_payload["fingerprint_sha256"] = three_way.canonical_sha256(
            runtime_payload
        )
        self._json(runtime, runtime_payload)

        raw_response = (
            '[{"bbox_2d":[10,10,40,30],'
            '"text_content":"こんにちは"}]'
        )
        debug = root / "debug" / "page"
        debug.mkdir(parents=True)
        (debug / "raw_response.txt").write_text(
            raw_response, encoding="utf-8"
        )
        self._json(
            debug / "page_summary.json",
            {
                "image": "page.png",
                "source_path": str(source),
                "failure": "",
                "request": {
                    "raw_response": raw_response,
                    "original_shape": [80, 100],
                    "request_shape": [80, 100],
                    "base_scale": 1.0,
                    "scale_x": 1.0,
                    "scale_y": 1.0,
                    "resize_profile": "standard",
                    "max_completion_tokens": 4096,
                    "block_count": 1,
                    "small_block_ratio": 0.0,
                    "text_cover_ratio": 0.1,
                },
                "attempts": [
                    {
                        "finish_reason": "stop",
                        "parser_error_code": "",
                    }
                ],
                "blocks": [{"xyxy": [5, 5, 45, 35]}],
            },
        )
        return {
            "manifest": manifest,
            "runtime": runtime,
            "debug": root / "debug",
        }

    def test_replay_current_product_reconciliation_into_locked_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._replay_fixture(root)
            output = root / "run"
            result = convergence.replay_manga_debug_run(
                corpus_manifest=fixture["manifest"],
                debug_root=fixture["debug"],
                runtime_contract=fixture["runtime"],
                output_dir=output,
            )
            page = result["pages"][0]
            self.assertEqual(page["status"], "success")
            self.assertEqual(page["canonical_units"][0]["text"], "こんにちは")
            self.assertEqual(
                page["canonical_units"][0]["detector_block_ids"],
                ["block-0"],
            )
            self.assertEqual(page["diagnostics"]["shadow_region_count"], 0)
            three_way.validate_run(output / three_way.RUN_FILENAME)

    def test_replay_rejects_changed_detector_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._replay_fixture(root)
            summary_path = fixture["debug"] / "page" / "page_summary.json"
            summary = three_way.read_json(summary_path)
            summary["blocks"][0]["xyxy"] = [6, 5, 45, 35]
            self._json(summary_path, summary)
            with self.assertRaisesRegex(
                convergence.ReplayContractError, "geometry changed"
            ):
                convergence.replay_manga_debug_run(
                    corpus_manifest=fixture["manifest"],
                    debug_root=fixture["debug"],
                    runtime_contract=fixture["runtime"],
                    output_dir=root / "run",
                )

    def _review_row(self, labels: dict[str, str], *, complete: bool) -> dict[str, str]:
        row = {column: "" for column in three_way._review_headers()}
        row.update(
            {
                "row_number": "1",
                "row_id": "row-000001",
                "row_kind": "truth",
                "page_id": "page",
                "truth_region_id": "page-det-0000",
                "truth_region_source": "detector",
                "language": "ja",
                "truth_transcription": "こんにちは",
                "truth_semantic_role": "dialogue_bubble",
                "truth_processing_action": "translate_inpaint",
                "truth_confidence": "high",
                "truth_bbox_xyxy": "[5,5,45,35]",
                "source_page": "source/page.png",
                "source_crop": "source/crop.png",
            }
        )
        for label, route in labels.items():
            row[f"{label}_text"] = f"text-{route}"
            row[f"{label}_bbox_xyxy"] = "[5,5,45,35]"
            row[f"{label}_raw_region_ids"] = "[\"raw-0\"]"
            row[f"{label}_semantic_role"] = "dialogue_bubble"
            row[f"{label}_processing_action"] = "translate_inpaint"
            row[f"{label}_geometry_status"] = "one_to_one"
            row[f"{label}_assets_json"] = "{}"
            if complete:
                row[f"{label}_transcription_correct"] = "yes"
                row[f"{label}_semantic_correct"] = "yes"
                row[f"{label}_role_action_correct"] = "yes"
                row[f"{label}_merge_split_error"] = "no"
                row[f"{label}_destructive_edit"] = "not_applicable"
                row[f"{label}_notes"] = f"reviewed-{route}"
        return row

    def test_transfer_decisions_follows_route_not_blind_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source-review"
            target = root / "target-review"
            source.mkdir()
            target.mkdir()
            source_labels = {
                "A": "mangalmm_full_page",
                "B": "paddle_crop",
                "C": "paddle_spotting_full_page",
            }
            target_labels = {
                "A": "paddle_crop",
                "B": "paddle_spotting_full_page",
                "C": "mangalmm_full_page",
            }
            for review, labels in (
                (source, source_labels),
                (target, target_labels),
            ):
                self._json(
                    review / "private" / three_way.BLIND_KEY_FILENAME,
                    {"label_to_route": labels},
                )
            three_way._write_review_csv(
                source / three_way.REVIEW_CSV_FILENAME,
                [self._review_row(source_labels, complete=True)],
            )
            three_way._write_review_csv(
                target / three_way.REVIEW_CSV_FILENAME,
                [self._review_row(target_labels, complete=False)],
            )
            self._json(
                source / three_way.FINAL_METRICS_FILENAME,
                {
                    "completed_review_csv_sha256": three_way.sha256_file(
                        source / three_way.REVIEW_CSV_FILENAME
                    )
                },
            )
            result = convergence.transfer_review_decisions(
                completed_review=source,
                target_review=target,
            )
            self.assertEqual(result["pending_error_count"], 0)
            rows = three_way._read_review_rows(
                target / three_way.REVIEW_CSV_FILENAME
            )
            self.assertEqual(rows[0]["A_notes"], "reviewed-paddle_crop")
            self.assertEqual(rows[0]["C_notes"], "reviewed-mangalmm_full_page")

    def test_transfer_candidate_extra_ignores_diagnostic_status_rename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source-review"
            target = root / "target-review"
            labels = {
                "A": "mangalmm_full_page",
                "B": "paddle_crop",
                "C": "paddle_spotting_full_page",
            }
            for review in (source, target):
                review.mkdir()
                self._json(
                    review / "private" / three_way.BLIND_KEY_FILENAME,
                    {"label_to_route": labels},
                )

            source_row = {
                column: "" for column in three_way._review_headers()
            }
            source_row.update(
                {
                    "row_number": "1",
                    "row_id": "extra-page-0001",
                    "row_kind": "candidate_extra",
                    "page_id": "page",
                    "language": "ja",
                    "source_page": "source/page.png",
                    "source_crop": "source/crop.png",
                    "A_text": "ビクッ",
                    "A_bbox_xyxy": "[10,20,30,40]",
                    "A_raw_region_ids": '["raw-0"]',
                    "A_geometry_status": "unmatched_extra",
                    "A_assets_json": "{}",
                    "A_role_action_correct": "no",
                    "A_merge_split_error": "no",
                    "A_destructive_edit": "not_applicable",
                    "A_false_positive": "no",
                    "A_notes": "source pixels contain a sound effect",
                }
            )
            target_row = dict(source_row)
            target_row["A_geometry_status"] = "other"
            target_row["A_role_action_correct"] = ""
            target_row["A_merge_split_error"] = ""
            target_row["A_destructive_edit"] = ""
            target_row["A_false_positive"] = ""
            target_row["A_notes"] = ""

            three_way._write_review_csv(
                source / three_way.REVIEW_CSV_FILENAME, [source_row]
            )
            three_way._write_review_csv(
                target / three_way.REVIEW_CSV_FILENAME, [target_row]
            )
            self._json(
                source / three_way.FINAL_METRICS_FILENAME,
                {
                    "completed_review_csv_sha256": three_way.sha256_file(
                        source / three_way.REVIEW_CSV_FILENAME
                    )
                },
            )

            result = convergence.transfer_review_decisions(
                completed_review=source,
                target_review=target,
            )

            self.assertEqual(result["pending_error_count"], 0)
            self.assertEqual(
                result["matched_candidate_extra_rows_by_route"]
                ["mangalmm_full_page"],
                1,
            )
            rows = three_way._read_review_rows(
                target / three_way.REVIEW_CSV_FILENAME
            )
            self.assertEqual(rows[0]["A_false_positive"], "no")
            self.assertEqual(
                rows[0]["A_notes"], "source pixels contain a sound effect"
            )


if __name__ == "__main__":
    unittest.main()
