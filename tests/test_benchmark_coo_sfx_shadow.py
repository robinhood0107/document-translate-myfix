from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark_coo_sfx_shadow as benchmark  # noqa: E402
import benchmark_ocr_three_way_human_truth as three_way  # noqa: E402


class COOSFXShadowBenchmarkTests(unittest.TestCase):
    def _json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _fixture(self, root: Path) -> dict[str, Path]:
        source = root / "source" / "neutral-page.png"
        source.parent.mkdir(parents=True)
        Image.new("RGB", (120, 100), "white").save(source)
        source_sha = three_way.sha256_file(source)
        detector = root / "detector" / "neutral-page.json"
        self._json(
            detector,
            {
                "status": "success",
                "source_sha256": source_sha,
                "blocks": [
                    {
                        "block_id": "block-sfx",
                        "xyxy": [5, 5, 45, 35],
                        "bubble_xyxy": None,
                        "text_class": "text_free",
                        "direction": "vertical",
                    },
                    {
                        "block_id": "block-dialogue",
                        "xyxy": [60, 10, 110, 50],
                        "bubble_xyxy": [57, 7, 113, 53],
                        "text_class": "text_bubble",
                        "direction": "horizontal",
                    },
                ],
            },
        )
        manifest_path = root / "corpus-manifest.json"
        manifest = {
            "schema_version": three_way.CORPUS_SCHEMA_VERSION,
            "protocol_version": three_way.PROTOCOL_VERSION,
            "suite_id": "neutral-coo-suite",
            "pages": [
                {
                    "page_id": "neutral-page",
                    "split": "development",
                    "language": "ja",
                    "source_image": {
                        "path": str(source),
                        "sha256": source_sha,
                        "width": 120,
                        "height": 100,
                    },
                    "detector_snapshot": {
                        "path": str(detector),
                        "sha256": three_way.sha256_file(detector),
                    },
                }
            ],
        }
        manifest["manifest_sha256"] = three_way.canonical_sha256(manifest)
        self._json(manifest_path, manifest)
        return {
            "source": source,
            "manifest": manifest_path,
        }

    def _predictions(
        self,
        root: Path,
        fixture: dict[str, Path],
        *,
        device: str,
        coordinate_delta: float = 0.0,
    ) -> Path:
        source_sha = three_way.sha256_file(fixture["source"])
        path = root / f"{device}-predictions.json"
        regions = [
            {
                "region_id": "coo-0000",
                "score": 0.8,
                "bbox_xyxy": [
                    8.0 + coordinate_delta,
                    8.0,
                    42.0 + coordinate_delta,
                    32.0,
                ],
                "polygon_xy": [
                    [8.0 + coordinate_delta, 8.0],
                    [42.0 + coordinate_delta, 8.0],
                    [42.0 + coordinate_delta, 32.0],
                    [8.0 + coordinate_delta, 32.0],
                ],
            },
            {
                "region_id": "coo-0001",
                "score": 0.7,
                "bbox_xyxy": [
                    65.0 + coordinate_delta,
                    15.0,
                    105.0 + coordinate_delta,
                    45.0,
                ],
                "polygon_xy": [
                    [65.0 + coordinate_delta, 15.0],
                    [105.0 + coordinate_delta, 15.0],
                    [105.0 + coordinate_delta, 45.0],
                    [65.0 + coordinate_delta, 45.0],
                ],
            },
        ]
        payload = {
            "schema_version": benchmark.PREDICTION_SCHEMA_VERSION,
            "model": benchmark.EXPECTED_MODEL,
            "model_sha256": benchmark.EXPECTED_MODEL_SHA256,
            "source_commit": benchmark.EXPECTED_SOURCE_COMMIT,
            "runtime_image_digest": "sha256:" + "a" * 64,
            "threshold": 0.3,
            "device": device,
            "normalization": benchmark.ALLOWED_NORMALIZATION[device],
            "torch_version": "1.9.0",
            "cuda_runtime_version": "11.1",
            "cuda_device_name": "Neutral CUDA Device" if device == "cuda" else None,
            "cuda_peak_allocated_bytes": 512 * 1024 * 1024 if device == "cuda" else None,
            "cuda_peak_reserved_bytes": 640 * 1024 * 1024 if device == "cuda" else None,
            "model_load_seconds": 4.0 if device == "cuda" else 0.5,
            "process_elapsed_seconds": 6.0 if device == "cuda" else 8.0,
            "pages": [
                {
                    "source_basename": "neutral-page.png",
                    "source_sha256": source_sha,
                    "width": 120,
                    "height": 100,
                    "elapsed_seconds": 1.0 if device == "cuda" else 5.0,
                    "region_count": len(regions),
                    "regions": regions,
                }
            ],
        }
        self._json(path, payload)
        return path

    def _locked_truth(self, root: Path, fixture: dict[str, Path]) -> Path:
        truth = root / "truth"
        three_way.init_truth(fixture["manifest"], truth)
        page_path = truth / "pages" / "neutral-page.json"
        page = three_way.read_json(page_path)
        for region in page["regions"]:
            if region["detector_block_ids"] == ["block-sfx"]:
                region.update(
                    {
                        "transcription": "ドン",
                        "semantic_role": "sfx",
                        "processing_action": "preserve",
                        "confidence": "high",
                    }
                )
            else:
                region.update(
                    {
                        "transcription": "こんにちは",
                        "semantic_role": "dialogue_bubble",
                        "processing_action": "translate_inpaint",
                        "confidence": "high",
                    }
                )
        three_way.write_json(page_path, page)
        three_way.export_truth_csv(truth)
        three_way.lock_truth(truth)
        return truth

    def test_validates_official_model_and_source_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            predictions = self._predictions(root, fixture, device="cuda")
            result = benchmark.validate_predictions(
                predictions, fixture["manifest"]
            )
            self.assertEqual(result["device"], "cuda")
            self.assertEqual(result["pages"][0]["page_id"], "neutral-page")
            self.assertRegex(result["prediction_contract_sha256"], r"^[0-9a-f]{64}$")

            payload = three_way.read_json(predictions)
            payload["model_sha256"] = "b" * 64
            self._json(predictions, payload)
            with self.assertRaisesRegex(benchmark.ContractError, "model SHA"):
                benchmark.validate_predictions(predictions, fixture["manifest"])

    def test_rejects_duplicate_region_and_source_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            predictions = self._predictions(root, fixture, device="cuda")
            payload = three_way.read_json(predictions)
            payload["pages"][0]["regions"][1]["region_id"] = "coo-0000"
            self._json(predictions, payload)
            with self.assertRaisesRegex(benchmark.ContractError, "Duplicate COO region"):
                benchmark.validate_predictions(predictions, fixture["manifest"])

            predictions = self._predictions(root, fixture, device="cuda")
            payload = three_way.read_json(predictions)
            payload["pages"][0]["source_sha256"] = "c" * 64
            self._json(predictions, payload)
            with self.assertRaisesRegex(benchmark.ContractError, "corpus manifest"):
                benchmark.validate_predictions(predictions, fixture["manifest"])

    def test_clips_small_model_overflow_and_rejects_implausible_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            predictions = self._predictions(root, fixture, device="cuda")
            payload = three_way.read_json(predictions)
            region = payload["pages"][0]["regions"][1]
            region["bbox_xyxy"] = [100.0, 10.0, 125.0, 40.0]
            region["polygon_xy"] = [
                [100.0, 10.0],
                [125.0, 10.0],
                [125.0, 40.0],
                [100.0, 40.0],
            ]
            self._json(predictions, payload)
            result = benchmark.validate_predictions(
                predictions, fixture["manifest"]
            )
            self.assertEqual(result["clipped_region_count"], 1)
            self.assertEqual(
                result["pages"][0]["regions"][1]["bbox_xyxy"][2], 120.0
            )

            payload = three_way.read_json(predictions)
            payload["pages"][0]["regions"][1]["bbox_xyxy"][2] = 200.0
            self._json(predictions, payload)
            with self.assertRaisesRegex(benchmark.ContractError, "implausibly outside"):
                benchmark.validate_predictions(predictions, fixture["manifest"])

    def test_cpu_cuda_comparison_accepts_subpixel_equivalence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            cpu = self._predictions(root, fixture, device="cpu")
            cuda = self._predictions(
                root, fixture, device="cuda", coordinate_delta=0.01
            )
            result = benchmark.compare_devices(
                cpu_predictions=cpu,
                cuda_predictions=cuda,
                corpus_manifest=fixture["manifest"],
                output_dir=root / "comparison",
            )
            self.assertTrue(result["geometry_equivalent"])
            self.assertEqual(result["region_counts_equal"], True)
            self.assertGreater(result["inference_speedup_percent"], 0)
            self.assertEqual(result["promotion_allowed"], False)
            self.assertTrue((root / "comparison" / "device-comparison-ko.md").is_file())

    def test_shadow_is_review_only_and_never_auto_hides_dialogue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            predictions = self._predictions(root, fixture, device="cuda")
            truth = self._locked_truth(root, fixture)
            result = benchmark.evaluate_shadow(
                predictions_path=predictions,
                corpus_manifest=fixture["manifest"],
                truth_dir=truth,
                output_dir=root / "evaluation",
            )
            self.assertEqual(result["sfx_or_decorative_count"], 1)
            self.assertEqual(result["sfx_or_decorative_signal_count"], 1)
            self.assertEqual(result["meaningful_text_signal_count"], 1)
            self.assertEqual(result["review_only_nonbubble_signal_count"], 1)
            self.assertEqual(result["automatic_preserve_count"], 0)
            self.assertEqual(result["meaningful_text_auto_hidden_count"], 0)
            self.assertEqual(result["promotion_allowed"], False)

    def test_shadow_scores_real_normalized_run_manifest_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            predictions = self._predictions(root, fixture, device="cuda")
            truth = self._locked_truth(root, fixture)
            run_path = root / "normalized-run.json"
            run_path.touch()
            normalized_run = {
                "route_id": "paddle_crop",
                "corpus_manifest_path": str(fixture["manifest"]),
                "pages": [
                    {
                        "page_id": "neutral-page",
                        "canonical_units": [
                            {
                                "detector_block_ids": ["block-sfx"],
                                "processing_action": "translate_inpaint",
                            },
                            {
                                "detector_block_ids": ["block-dialogue"],
                                "processing_action": "translate_inpaint",
                            },
                        ],
                    }
                ],
            }

            with mock.patch.object(
                three_way,
                "validate_run",
                return_value=normalized_run,
            ):
                result = benchmark.evaluate_shadow(
                    predictions_path=predictions,
                    corpus_manifest=fixture["manifest"],
                    truth_dir=truth,
                    output_dir=root / "evaluation",
                    normalized_runs=[run_path],
                )

            route = result["route_metrics"]["paddle_crop"]
            self.assertEqual(route["baseline_sfx_or_decorative_auto_edit_count"], 1)
            self.assertEqual(
                route["caught_by_review_only_nonbubble_signal_count"], 1
            )
            self.assertEqual(route["meaningful_text_sent_to_review_count"], 0)
            self.assertEqual(route["meaningful_text_auto_hidden_count"], 0)

    def test_unlocked_truth_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            predictions = self._predictions(root, fixture, device="cuda")
            truth = root / "truth"
            three_way.init_truth(fixture["manifest"], truth)
            with self.assertRaises(three_way.ContractError):
                benchmark.evaluate_shadow(
                    predictions_path=predictions,
                    corpus_manifest=fixture["manifest"],
                    truth_dir=truth,
                    output_dir=root / "evaluation",
                )


if __name__ == "__main__":
    unittest.main()
