from __future__ import annotations

import csv
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

import benchmark_ocr_three_way_human_truth as benchmark  # noqa: E402


class OCRThreeWayHumanTruthTests(unittest.TestCase):
    def _json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _fixture(self, root: Path) -> dict[str, Path]:
        evidence = root / "evidence"
        source_dir = evidence / "source"
        source_dir.mkdir(parents=True)
        image_path = source_dir / "neutral-page.png"
        Image.new("RGB", (100, 80), "white").save(image_path)
        source_sha = benchmark.sha256_file(image_path)

        detector_path = evidence / "detector" / "neutral-page.json"
        detector = {
            "status": "success",
            "source_sha256": source_sha,
            "blocks": [
                {
                    "block_id": "block-0",
                    "xyxy": [5, 5, 45, 30],
                    "bubble_xyxy": [3, 3, 48, 33],
                    "text_class": "text_bubble",
                    "direction": "horizontal",
                },
                {
                    "block_id": "block-1",
                    "xyxy": [55, 40, 95, 70],
                    "bubble_xyxy": [52, 37, 98, 73],
                    "text_class": "text_bubble",
                    "direction": "vertical",
                },
            ],
        }
        self._json(detector_path, detector)

        manifest_path = evidence / "corpus-manifest.json"
        manifest = {
            "schema_version": benchmark.CORPUS_SCHEMA_VERSION,
            "protocol_version": benchmark.PROTOCOL_VERSION,
            "suite_id": "neutral-suite",
            "pages": [
                {
                    "page_id": "neutral-page",
                    "split": "development",
                    "language": "ja",
                    "source_image": {
                        "path": str(image_path),
                        "sha256": source_sha,
                        "width": 100,
                        "height": 80,
                    },
                    "detector_snapshot": {
                        "path": str(detector_path),
                        "sha256": benchmark.sha256_file(detector_path),
                    },
                }
            ],
        }
        manifest["manifest_sha256"] = benchmark.canonical_sha256(manifest)
        self._json(manifest_path, manifest)

        crop_root = evidence / "paddle-crop"
        crop_asset = crop_root / "neutral-page" / "raw-mask.png"
        crop_asset.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (40, 25), "white").save(crop_asset)
        crop_result = {
            **detector,
            "error": "",
            "request_seconds": 1.2,
            "blocks": [
                {
                    **detector["blocks"][0],
                    "text": "こんにちは。",
                    "semantic_role": "dialogue_bubble",
                    "processing_action": "translate_inpaint",
                    "assets": {"raw-mask": "neutral-page/raw-mask.png"},
                },
                {
                    **detector["blocks"][1],
                    "text": "またね",
                    "semantic_role": "dialogue_bubble",
                    "processing_action": "translate_inpaint",
                },
            ],
            "page_profile": {
                "performance": {
                    "http_attempt_count": 2,
                    "http_retry_count": 0,
                }
            },
        }
        self._json(crop_root / "neutral-page" / "result.json", crop_result)

        spotting_root = evidence / "paddle-spotting"
        self._json(
            spotting_root / "geometry-audit" / "neutral-page.json",
            {
                "shape_hw": [80, 100],
                "spotting": [
                    {"text": "こんに", "bbox_xyxy": [7, 7, 25, 28]},
                    {"text": "ちは", "bbox_xyxy": [26, 7, 43, 28]},
                    {"text": "またね", "bbox_xyxy": [58, 42, 93, 68]},
                    {"text": "ドン", "bbox_xyxy": [45, 2, 65, 15]},
                ],
            },
        )
        self._json(
            spotting_root
            / "detector-fused-comparison-v1"
            / "neutral-page.json",
            {
                "detector_block_count": 2,
                "mapped_detector_count": 2,
                "blocks": [
                    {
                        **detector["blocks"][0],
                        "page": "neutral-page",
                        "spotting_text": "こんにちは",
                        "spot_indices": [0, 1],
                        "semantic_role": "dialogue_bubble",
                        "processing_action": "translate_inpaint",
                    },
                    {
                        **detector["blocks"][1],
                        "page": "neutral-page",
                        "spotting_text": "またね",
                        "spot_indices": [2],
                        "semantic_role": "dialogue_bubble",
                        "processing_action": "translate_inpaint",
                    },
                ],
            },
        )
        self._json(
            spotting_root / "neutral-page_spotting.json",
            {
                "input": str(image_path),
                "elapsed_seconds": 0.8,
                "finish_reason": "stop",
                "image_width": 100,
                "image_height": 80,
            },
        )

        manga_root = evidence / "manga"
        self._json(
            manga_root / "neutral-page" / "result.json",
            {
                "image": "neutral-page.png",
                "detector_block_count": 2,
                "elapsed_seconds": 1.1,
                "failure": None,
                "regions": [
                    {"text": "こんにちは", "bbox_xyxy": [8, 8, 42, 28]},
                    {"text": "またね", "bbox_xyxy": [57, 42, 94, 69]},
                    {"text": "ドン", "bbox_xyxy": [46, 2, 66, 15]},
                ],
                "attempts": [
                    {
                        "finish_reason": "stop",
                        "parser_error_code": "",
                    }
                ],
                "merge_split_diagnostics": [],
            },
        )

        runtime_paths: dict[str, Path] = {}
        for index, route in enumerate(benchmark.ROUTES, start=1):
            runtime_path = evidence / "runtime" / f"{route}.json"
            runtime_payload = {
                "schema_version": 1,
                "route_id": route,
                "backend": "llama.cpp",
                "model_sha256": "a" * 64,
                "mmproj_sha256": "b" * 64,
                "command_sha256": "c" * 64,
                "image_digest": "sha256:" + "d" * 64,
                "prompt_mode": {
                    "paddle_crop": "OCR:",
                    "paddle_spotting_full_page": "Spotting:",
                    "mangalmm_full_page": "mangalmm_official_full_page",
                }[route],
                "image_max_pixels": benchmark.ROUTE_IMAGE_MAX_PIXELS[route],
                "special_tokens": route == "paddle_spotting_full_page",
                "fixture_identity": index,
            }
            runtime_payload["fingerprint_sha256"] = benchmark.canonical_sha256(
                runtime_payload
            )
            self._json(
                runtime_path,
                runtime_payload,
            )
            runtime_paths[route] = runtime_path
        for route, source_root in {
            "paddle_crop": crop_root,
            "paddle_spotting_full_page": spotting_root,
            "mangalmm_full_page": manga_root,
        }.items():
            benchmark.create_source_bindings(
                route=route,
                corpus_manifest=manifest_path,
                source_results=source_root,
            )
        return {
            "manifest": manifest_path,
            "crop": crop_root,
            "spotting": spotting_root,
            "manga": manga_root,
            **{f"runtime-{route}": path for route, path in runtime_paths.items()},
        }

    def _complete_truth(self, truth_dir: Path) -> None:
        page_path = truth_dir / "pages" / "neutral-page.json"
        page = benchmark.read_json(page_path)
        for index, region in enumerate(page["regions"]):
            region.update(
                {
                    "transcription": ("こんにちは" if index == 0 else "またね"),
                    "semantic_role": "dialogue_bubble",
                    "processing_action": "translate_inpaint",
                    "confidence": "high",
                }
            )
        benchmark.write_json(page_path, page)
        benchmark.export_truth_csv(truth_dir)

    def _import_runs(
        self,
        root: Path,
        fixture: dict[str, Path],
    ) -> list[Path]:
        source_roots = {
            "paddle_crop": fixture["crop"],
            "paddle_spotting_full_page": fixture["spotting"],
            "mangalmm_full_page": fixture["manga"],
        }
        results: list[Path] = []
        for route in benchmark.ROUTES:
            output = root / "runs" / route
            benchmark.import_existing_run(
                route=route,
                corpus_manifest=fixture["manifest"],
                source_results=source_roots[route],
                runtime_contract=fixture[f"runtime-{route}"],
                output_dir=output,
            )
            results.append(output / benchmark.RUN_FILENAME)
        return results

    def _rebind_source_results(
        self,
        fixture: dict[str, Path],
        route: str,
    ) -> None:
        source_root = {
            "paddle_crop": fixture["crop"],
            "paddle_spotting_full_page": fixture["spotting"],
            "mangalmm_full_page": fixture["manga"],
        }[route]
        (source_root / benchmark.SOURCE_BINDING_FILENAME).unlink()
        benchmark.create_source_bindings(
            route=route,
            corpus_manifest=fixture["manifest"],
            source_results=source_root,
        )

    def _complete_review(self, review_dir: Path) -> None:
        path = review_dir / benchmark.REVIEW_CSV_FILENAME
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
        for row in rows:
            for label in benchmark.BLIND_LABELS:
                has_text = bool(row[f"{label}_text"].strip())
                if row["row_kind"] == "truth":
                    row[f"{label}_transcription_correct"] = "yes"
                    row[f"{label}_semantic_correct"] = "yes"
                    row[f"{label}_role_action_correct"] = "yes"
                if has_text:
                    row[f"{label}_merge_split_error"] = "no"
                    row[f"{label}_destructive_edit"] = "not_applicable"
                if row["row_kind"] == "candidate_extra" and has_text:
                    row[f"{label}_false_positive"] = "no"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_text_normalization_ignores_spacing_and_punctuation(self) -> None:
        self.assertEqual(benchmark.normalize_text("こ ん\nにちは。"), "こんにちは")
        self.assertEqual(
            benchmark.normalized_character_accuracy("こんにちは。", "こ んにちは"),
            1.0,
        )

    def test_truth_must_be_complete_before_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            truth = root / "truth"
            benchmark.init_truth(fixture["manifest"], truth)
            with self.assertRaises(benchmark.IncompleteReviewError):
                benchmark.lock_truth(truth)
            self._complete_truth(truth)
            lock = benchmark.lock_truth(truth)
            self.assertRegex(lock["truth_contract_sha256"], r"^[0-9a-f]{64}$")

            crop = next((truth / "assets" / "crops").rglob("*.png"))
            crop.write_bytes(b"tampered")
            with self.assertRaisesRegex(benchmark.ContractError, "asset changed"):
                benchmark.validate_locked_truth(truth)

    def test_truth_lock_rejects_removed_or_changed_detector_regions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            truth = root / "truth"
            benchmark.init_truth(fixture["manifest"], truth)
            self._complete_truth(truth)
            page_path = truth / "pages" / "neutral-page.json"
            page = benchmark.read_json(page_path)
            page["regions"] = page["regions"][:1]
            benchmark.write_json(page_path, page)
            benchmark.export_truth_csv(truth)
            with self.assertRaises(benchmark.IncompleteReviewError) as caught:
                benchmark.lock_truth(truth)
            self.assertTrue(
                any("detector truth set changed" in error for error in caught.exception.errors)
            )

    def test_build_manifest_hashes_source_and_detector(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            source_dir = fixture["manifest"].parent / "source"
            detector_dir = fixture["manifest"].parent / "detector"
            spec_path = root / "build-spec.json"
            self._json(
                spec_path,
                {
                    "schema_version": 1,
                    "suite_id": "built-suite",
                    "groups": [
                        {
                            "source_dir": str(source_dir),
                            "detector_results": str(detector_dir),
                            "language": "ja",
                            "split": "development",
                            "page_ids": ["neutral-page"],
                        }
                    ],
                },
            )
            output = root / "built-manifest.json"
            result = benchmark.build_corpus_manifest(spec_path, output)
            self.assertEqual(result["pages"][0]["page_id"], "neutral-page")
            self.assertEqual(
                benchmark.validate_corpus_manifest(output)["suite_id"],
                "built-suite",
            )

    def test_full_import_blind_review_and_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            truth = root / "truth"
            benchmark.init_truth(fixture["manifest"], truth)
            self._complete_truth(truth)
            benchmark.lock_truth(truth)
            runs = self._import_runs(root, fixture)
            review = root / "review"
            payload = benchmark.make_review(
                truth_dir=truth,
                runs=runs,
                output_dir=review,
            )
            self.assertEqual(payload["truth_row_count"], 2)
            self.assertGreaterEqual(payload["candidate_extra_row_count"], 1)
            public_html = (review / benchmark.REVIEW_HTML_FILENAME).read_text(
                encoding="utf-8"
            )
            public_csv = (review / benchmark.REVIEW_CSV_FILENAME).read_text(
                encoding="utf-8-sig"
            )
            for route in benchmark.ROUTES:
                self.assertNotIn(route, public_html)
                self.assertNotIn(route, public_csv)
            self.assertEqual(
                len(list((review / "assets" / "candidates").rglob("raw-mask.png"))),
                1,
            )
            self.assertIn("<figure>", public_html)
            with self.assertRaises(benchmark.IncompleteReviewError):
                benchmark.finalize_review(review)
            self._complete_review(review)
            final = benchmark.finalize_review(review)
            self.assertEqual(set(final["route_metrics"]), set(benchmark.ROUTES))
            self.assertTrue(final["automatic_similarity_is_not_semantic_truth"])
            for metrics in final["route_metrics"].values():
                self.assertEqual(metrics["run_statistics"]["page_count"], 1)
                self.assertEqual(metrics["page_complete_count"], 1)

    def test_truth_csv_round_trip_and_immutable_review_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            truth = root / "truth"
            benchmark.init_truth(fixture["manifest"], truth)
            csv_path = truth / benchmark.TRUTH_CSV_FILENAME
            with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fieldnames = list(reader.fieldnames or [])
                rows = [dict(row) for row in reader]
            for index, row in enumerate(rows):
                row["transcription"] = "こんにちは" if index == 0 else "またね"
                row["semantic_role"] = "dialogue_bubble"
                row["processing_action"] = "translate_inpaint"
                row["confidence"] = "high"
            with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            imported = benchmark.import_truth_csv(truth, csv_path)
            self.assertEqual(imported["region_count"], 2)
            benchmark.lock_truth(truth)

            runs = self._import_runs(root, fixture)
            review = root / "review"
            benchmark.make_review(truth_dir=truth, runs=runs, output_dir=review)
            review_csv = review / benchmark.REVIEW_CSV_FILENAME
            with review_csv.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                review_fields = list(reader.fieldnames or [])
                review_rows = [dict(row) for row in reader]
            review_rows[0]["A_text"] = "tampered"
            with review_csv.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=review_fields)
                writer.writeheader()
                writer.writerows(review_rows)
            with self.assertRaisesRegex(benchmark.ContractError, "evidence columns"):
                benchmark.finalize_review(review)

    def test_tampered_source_result_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            run_path = self._import_runs(root, fixture)[0]
            source = fixture["crop"] / "neutral-page" / "result.json"
            source.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(benchmark.ContractError):
                benchmark.validate_run(run_path)

    def test_review_assets_are_unique_per_row_and_hash_locked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            crop_result_path = fixture["crop"] / "neutral-page" / "result.json"
            crop_result = benchmark.read_json(crop_result_path)
            second_asset = fixture["crop"] / "neutral-page" / "second-mask.png"
            Image.new("L", (40, 25), "black").save(second_asset)
            crop_result["blocks"][1]["assets"] = {
                "raw-mask": "neutral-page/second-mask.png"
            }
            self._json(crop_result_path, crop_result)
            self._rebind_source_results(fixture, "paddle_crop")

            truth = root / "truth"
            benchmark.init_truth(fixture["manifest"], truth)
            self._complete_truth(truth)
            benchmark.lock_truth(truth)
            review = root / "review"
            benchmark.make_review(
                truth_dir=truth,
                runs=self._import_runs(root, fixture),
                output_dir=review,
            )

            key = benchmark.read_json(
                review / "private" / benchmark.BLIND_KEY_FILENAME
            )
            crop_label = next(
                label
                for label, route in key["label_to_route"].items()
                if route == "paddle_crop"
            )
            with (review / benchmark.REVIEW_CSV_FILENAME).open(
                "r", encoding="utf-8-sig", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            paths = []
            for row in rows:
                assets = json.loads(row[f"{crop_label}_assets_json"] or "{}")
                paths.extend(assets.values())
            raw_masks = sorted(path for path in paths if path.endswith("raw-mask.png"))
            self.assertEqual(len(raw_masks), 2)
            self.assertEqual(len(set(raw_masks)), 2)
            self.assertNotEqual(
                benchmark.sha256_file(review / raw_masks[0]),
                benchmark.sha256_file(review / raw_masks[1]),
            )

            candidate_asset = review / raw_masks[0]
            candidate_asset.write_bytes(b"tampered")
            self._complete_review(review)
            with self.assertRaisesRegex(
                benchmark.ContractError, "visual evidence assets changed"
            ):
                benchmark.finalize_review(review)

    def test_ambiguous_manga_region_is_exposed_for_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            manga_path = fixture["manga"] / "neutral-page" / "result.json"
            manga = benchmark.read_json(manga_path)
            manga["regions"] = [
                {"text": "MERGED_TEXT", "bbox_xyxy": [5, 5, 95, 70]}
            ]
            self._json(manga_path, manga)
            self._rebind_source_results(fixture, "mangalmm_full_page")
            run_path = self._import_runs(root, fixture)[2]
            run = benchmark.read_json(run_path)
            units = run["pages"][0]["canonical_units"]
            self.assertEqual(len(units), 1)
            self.assertEqual(units[0]["text"], "MERGED_TEXT")
            self.assertEqual(units[0]["geometry_status"], "ambiguous_multi_detector")
            self.assertEqual(units[0]["semantic_role"], "ambiguous")
            self.assertEqual(units[0]["processing_action"], "review")
            self.assertEqual(units[0]["detector_block_ids"], [])

    def test_cross_truth_candidate_is_not_duplicated(self) -> None:
        truth = {
            "bbox_xyxy": [5, 5, 45, 30],
            "detector_block_ids": ["block-0"],
        }
        merged_unit = {
            "canonical_unit_id": "merged",
            "bbox_xyxy": [5, 5, 95, 70],
            "detector_block_ids": ["block-0", "block-1"],
        }
        matched, status = benchmark._match_truth_to_units(truth, [merged_unit])
        self.assertEqual(matched, [])
        self.assertEqual(status, "ambiguous_cross_truth_unit")

        human_extra = {
            "bbox_xyxy": [5, 5, 45, 30],
            "detector_block_ids": [],
        }
        matched, status = benchmark._match_truth_to_units(
            human_extra, [merged_unit]
        )
        self.assertEqual(matched, [])
        self.assertEqual(status, "missing_full_page_region")

    def test_spotting_region_cannot_be_assigned_twice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            fused_path = (
                fixture["spotting"]
                / "detector-fused-comparison-v1"
                / "neutral-page.json"
            )
            fused = benchmark.read_json(fused_path)
            fused["blocks"][1]["spot_indices"] = [0, 2]
            self._json(fused_path, fused)
            self._rebind_source_results(fixture, "paddle_spotting_full_page")
            with self.assertRaisesRegex(
                benchmark.ContractError, "assigned to multiple blocks"
            ):
                self._import_runs(root, fixture)

    def test_missing_crop_block_is_explicit_and_fails_the_page(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            crop_path = fixture["crop"] / "neutral-page" / "result.json"
            crop = benchmark.read_json(crop_path)
            crop["blocks"] = crop["blocks"][:1]
            self._json(crop_path, crop)
            self._rebind_source_results(fixture, "paddle_crop")
            output = root / "missing-crop-run"
            result = benchmark.import_existing_run(
                route="paddle_crop",
                corpus_manifest=fixture["manifest"],
                source_results=fixture["crop"],
                runtime_contract=fixture["runtime-paddle_crop"],
                output_dir=output,
            )
            page = result["pages"][0]
            self.assertEqual(page["status"], "failure")
            self.assertEqual(len(page["canonical_units"]), 2)
            self.assertEqual(
                page["diagnostics"]["missing_detector_block_ids"], ["block-1"]
            )
            benchmark.validate_run(output / benchmark.RUN_FILENAME)

    def test_spotting_requires_the_locked_detector_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            fused_path = (
                fixture["spotting"]
                / "detector-fused-comparison-v1"
                / "neutral-page.json"
            )
            fused = benchmark.read_json(fused_path)
            fused["blocks"][0]["xyxy"] = [6, 5, 45, 30]
            self._json(fused_path, fused)
            self._rebind_source_results(fixture, "paddle_spotting_full_page")
            with self.assertRaisesRegex(
                benchmark.ContractError, "detector geometry changed"
            ):
                benchmark.import_existing_run(
                    route="paddle_spotting_full_page",
                    corpus_manifest=fixture["manifest"],
                    source_results=fixture["spotting"],
                    runtime_contract=fixture[
                        "runtime-paddle_spotting_full_page"
                    ],
                    output_dir=root / "invalid-spotting",
                )

    def test_full_page_results_are_bound_to_the_locked_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            raw_path = fixture["spotting"] / "neutral-page_spotting.json"
            raw = benchmark.read_json(raw_path)
            raw["input"] = r"C:\unrelated\different-page.png"
            self._json(raw_path, raw)
            self._rebind_source_results(fixture, "paddle_spotting_full_page")
            with self.assertRaisesRegex(benchmark.ContractError, "locked source image"):
                benchmark.import_existing_run(
                    route="paddle_spotting_full_page",
                    corpus_manifest=fixture["manifest"],
                    source_results=fixture["spotting"],
                    runtime_contract=fixture[
                        "runtime-paddle_spotting_full_page"
                    ],
                    output_dir=root / "wrong-spotting-source",
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            result_path = fixture["manga"] / "neutral-page" / "result.json"
            result = benchmark.read_json(result_path)
            result["image"] = "different-page.png"
            self._json(result_path, result)
            self._rebind_source_results(fixture, "mangalmm_full_page")
            with self.assertRaisesRegex(benchmark.ContractError, "locked source image"):
                benchmark.import_existing_run(
                    route="mangalmm_full_page",
                    corpus_manifest=fixture["manifest"],
                    source_results=fixture["manga"],
                    runtime_contract=fixture["runtime-mangalmm_full_page"],
                    output_dir=root / "wrong-manga-source",
                )

    def test_non_finite_json_numbers_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "non-finite.json"
            path.write_text('{"elapsed_seconds": NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(benchmark.ContractError, "Non-finite"):
                benchmark.read_json(path)

    def test_source_binding_must_match_the_locked_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            binding_path = fixture["spotting"] / benchmark.SOURCE_BINDING_FILENAME
            binding = benchmark.read_json(binding_path)
            binding["pages"][0]["source_sha256"] = "f" * 64
            binding["binding_contract_sha256"] = benchmark.canonical_sha256(
                {
                    key: value
                    for key, value in binding.items()
                    if key != "binding_contract_sha256"
                }
            )
            self._json(binding_path, binding)
            with self.assertRaisesRegex(benchmark.ContractError, "locked corpus"):
                benchmark.import_existing_run(
                    route="paddle_spotting_full_page",
                    corpus_manifest=fixture["manifest"],
                    source_results=fixture["spotting"],
                    runtime_contract=fixture[
                        "runtime-paddle_spotting_full_page"
                    ],
                    output_dir=root / "wrong-binding",
                )

    def test_mangalmm_attempts_require_an_explicit_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            result_path = fixture["manga"] / "neutral-page" / "result.json"
            result = benchmark.read_json(result_path)
            result["attempts"] = [{}]
            self._json(result_path, result)
            self._rebind_source_results(fixture, "mangalmm_full_page")
            with self.assertRaisesRegex(benchmark.ContractError, "lacks an outcome"):
                benchmark.import_existing_run(
                    route="mangalmm_full_page",
                    corpus_manifest=fixture["manifest"],
                    source_results=fixture["manga"],
                    runtime_contract=fixture["runtime-mangalmm_full_page"],
                    output_dir=root / "invalid-manga-attempt",
                )

    def test_runtime_contract_requires_complete_official_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            runtime_path = fixture["runtime-paddle_crop"]
            payload = benchmark.read_json(runtime_path)
            del payload["model_sha256"]
            payload["fingerprint_sha256"] = benchmark.canonical_sha256(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "fingerprint_sha256"
                }
            )
            self._json(runtime_path, payload)
            with self.assertRaisesRegex(benchmark.ContractError, "model_sha256"):
                benchmark.import_existing_run(
                    route="paddle_crop",
                    corpus_manifest=fixture["manifest"],
                    source_results=fixture["crop"],
                    runtime_contract=runtime_path,
                    output_dir=root / "invalid-run",
                )

            payload = benchmark.read_json(runtime_path)
            payload["model_sha256"] = "0" * 64
            payload["fingerprint_sha256"] = benchmark.canonical_sha256(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "fingerprint_sha256"
                }
            )
            self._json(runtime_path, payload)
            with self.assertRaisesRegex(benchmark.ContractError, "placeholder"):
                benchmark.import_existing_run(
                    route="paddle_crop",
                    corpus_manifest=fixture["manifest"],
                    source_results=fixture["crop"],
                    runtime_contract=runtime_path,
                    output_dir=root / "placeholder-run",
                )

    def test_missing_meaning_text_cannot_receive_quality_credit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            crop_path = fixture["crop"] / "neutral-page" / "result.json"
            crop = benchmark.read_json(crop_path)
            crop["blocks"][0]["text"] = ""
            self._json(crop_path, crop)
            self._rebind_source_results(fixture, "paddle_crop")
            truth = root / "truth"
            benchmark.init_truth(fixture["manifest"], truth)
            self._complete_truth(truth)
            benchmark.lock_truth(truth)
            review = root / "review"
            benchmark.make_review(
                truth_dir=truth,
                runs=self._import_runs(root, fixture),
                output_dir=review,
            )
            self._complete_review(review)
            with self.assertRaises(benchmark.IncompleteReviewError) as caught:
                benchmark.finalize_review(review)
            self.assertTrue(
                any("missing meaning text" in error for error in caught.exception.errors)
            )

    def test_blind_payload_self_hash_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            truth = root / "truth"
            benchmark.init_truth(fixture["manifest"], truth)
            self._complete_truth(truth)
            benchmark.lock_truth(truth)
            review = root / "review"
            benchmark.make_review(
                truth_dir=truth,
                runs=self._import_runs(root, fixture),
                output_dir=review,
            )
            payload_path = (
                review / "private" / benchmark.BLIND_PAYLOAD_FILENAME
            )
            payload = benchmark.read_json(payload_path)
            payload["row_count"] += 1
            self._json(payload_path, payload)
            with self.assertRaisesRegex(
                benchmark.ContractError, "payload changed"
            ):
                benchmark.finalize_review(review)

    def test_uncertain_candidate_extra_prevents_page_complete_credit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            truth = root / "truth"
            benchmark.init_truth(fixture["manifest"], truth)
            self._complete_truth(truth)
            benchmark.lock_truth(truth)
            review = root / "review"
            benchmark.make_review(
                truth_dir=truth,
                runs=self._import_runs(root, fixture),
                output_dir=review,
            )
            self._complete_review(review)
            key = benchmark.read_json(
                review / "private" / benchmark.BLIND_KEY_FILENAME
            )
            review_csv = review / benchmark.REVIEW_CSV_FILENAME
            with review_csv.open(
                "r", encoding="utf-8-sig", newline=""
            ) as stream:
                reader = csv.DictReader(stream)
                fieldnames = list(reader.fieldnames or [])
                rows = [dict(row) for row in reader]
            selected_label = ""
            for row in rows:
                if row["row_kind"] != "candidate_extra":
                    continue
                for label in benchmark.BLIND_LABELS:
                    if row[f"{label}_text"].strip():
                        row[f"{label}_merge_split_error"] = "uncertain"
                        selected_label = label
                        break
                if selected_label:
                    break
            self.assertTrue(selected_label)
            with review_csv.open(
                "w", encoding="utf-8-sig", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            final = benchmark.finalize_review(review)
            selected_route = key["label_to_route"][selected_label]
            self.assertEqual(
                final["route_metrics"][selected_route]["page_complete_count"], 0
            )

    def test_vllm_runtime_contract_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            runtime_path = fixture["runtime-paddle_crop"]
            payload = benchmark.read_json(runtime_path)
            payload["backend"] = "vllm"
            self._json(runtime_path, payload)
            with self.assertRaisesRegex(benchmark.ContractError, "llama.cpp"):
                benchmark.import_existing_run(
                    route="paddle_crop",
                    corpus_manifest=fixture["manifest"],
                    source_results=fixture["crop"],
                    runtime_contract=runtime_path,
                    output_dir=root / "invalid-run",
                )

    def test_manifest_requires_its_canonical_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            manifest = benchmark.read_json(fixture["manifest"])
            manifest.pop("manifest_sha256")
            self._json(fixture["manifest"], manifest)
            with self.assertRaisesRegex(benchmark.ContractError, "hash is required"):
                benchmark.validate_corpus_manifest(fixture["manifest"])

    def test_csv_formula_escape_is_reversible(self) -> None:
        for value in ("=1+1", " +cmd", "-danger", "@SUM(A1)", "'=literal"):
            encoded = benchmark._csv_encode_cell(value)
            self.assertTrue(encoded.startswith("'"))
            self.assertEqual(benchmark._csv_decode_cell(encoded), value)

    def test_bound_primary_result_cannot_change_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            source = fixture["crop"] / "neutral-page" / "result.json"
            source.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                benchmark.ContractError, "Bound source result changed"
            ):
                benchmark.import_existing_run(
                    route="paddle_crop",
                    corpus_manifest=fixture["manifest"],
                    source_results=fixture["crop"],
                    runtime_contract=fixture["runtime-paddle_crop"],
                    output_dir=root / "invalid-run",
                )

    def test_detectorless_unit_is_assigned_to_only_one_human_truth(self) -> None:
        truth_regions = [
            {
                "truth_region_id": "human-a",
                "region_source": "human_extra",
                "detector_block_ids": [],
                "bbox_xyxy": [0, 0, 60, 40],
            },
            {
                "truth_region_id": "human-b",
                "region_source": "human_extra",
                "detector_block_ids": [],
                "bbox_xyxy": [40, 0, 100, 40],
            },
        ]
        unit = {
            "canonical_unit_id": "extra-0",
            "detector_block_ids": [],
            "bbox_xyxy": [10, 0, 55, 40],
        }
        assigned = benchmark._assign_detectorless_units_to_human_truth(
            truth_regions, [unit]
        )
        self.assertEqual(assigned["human-a"], [unit])
        self.assertEqual(assigned["human-b"], [])

    def test_spotting_fused_text_must_preserve_raw_segments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            fused_path = (
                fixture["spotting"]
                / "detector-fused-comparison-v1"
                / "neutral-page.json"
            )
            fused = benchmark.read_json(fused_path)
            fused["blocks"][0]["spotting_text"] = "こんちはに"
            self._json(fused_path, fused)
            self._rebind_source_results(fixture, "paddle_spotting_full_page")
            with self.assertRaisesRegex(
                benchmark.ContractError, "differs from its raw regions"
            ):
                self._import_runs(root, fixture)

    def test_blind_review_uses_route_neutral_geometry_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            truth = root / "truth"
            benchmark.init_truth(fixture["manifest"], truth)
            self._complete_truth(truth)
            benchmark.lock_truth(truth)
            review = root / "review"
            benchmark.make_review(
                truth_dir=truth,
                runs=self._import_runs(root, fixture),
                output_dir=review,
            )
            rows = benchmark._read_review_rows(
                review / benchmark.REVIEW_CSV_FILENAME
            )
            allowed = {
                "matched",
                "compound",
                "ambiguous",
                "missing_or_partial",
                "unmatched_extra",
                "other",
                "absent",
            }
            for row in rows:
                for label in benchmark.BLIND_LABELS:
                    self.assertIn(row[f"{label}_geometry_status"], allowed)

    def test_whitespace_around_decisions_does_not_corrupt_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            truth = root / "truth"
            benchmark.init_truth(fixture["manifest"], truth)
            self._complete_truth(truth)
            benchmark.lock_truth(truth)
            review = root / "review"
            benchmark.make_review(
                truth_dir=truth,
                runs=self._import_runs(root, fixture),
                output_dir=review,
            )
            self._complete_review(review)
            rows = benchmark._read_review_rows(
                review / benchmark.REVIEW_CSV_FILENAME
            )
            key = benchmark.read_json(
                review / "private" / benchmark.BLIND_KEY_FILENAME
            )
            label = "A"
            for row in rows:
                if row["row_kind"] == "truth":
                    row[f"{label}_semantic_correct"] = " yes "
                    row[f"{label}_transcription_correct"] = " yes "
                    row[f"{label}_role_action_correct"] = " yes "
                if row[f"{label}_text"].strip():
                    row[f"{label}_merge_split_error"] = " no "
                    row[f"{label}_destructive_edit"] = " no "
                if row["row_kind"] == "candidate_extra" and row[
                    f"{label}_text"
                ].strip():
                    row[f"{label}_false_positive"] = " no "
            benchmark._write_review_csv(
                review / benchmark.REVIEW_CSV_FILENAME, rows
            )
            final = benchmark.finalize_review(review)
            route = key["label_to_route"][label]
            self.assertEqual(
                final["route_metrics"][route]["page_complete_count"], 1
            )

    def test_historical_spotting_attempts_are_not_invented(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            run_path = self._import_runs(root, fixture)[1]
            page = benchmark.read_json(run_path)["pages"][0]
            self.assertEqual(page["attempt_count"], 0)
            self.assertFalse(page["attempt_telemetry_complete"])

    def test_repository_paths_are_rejected(self) -> None:
        with self.assertRaises(benchmark.ContractError):
            benchmark.require_external_path(
                ROOT / "docs" / "benchmark" / "not-evidence.json",
                label="evidence",
            )


if __name__ == "__main__":
    unittest.main()
