from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_mangalmm_v2 import (
    ANNOTATION_SCHEMA_VERSION,
    EVALUATION_PROTOCOL_VERSION,
    HISTORICAL_AUDIT,
    ROOT,
    BenchmarkContractError,
    audit_history,
    load_external_manifest,
    require_external_path,
    validate_evaluation_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MangaLMMV2BenchmarkTests(unittest.TestCase):
    def _write_valid_fixture(
        self,
        root: Path,
        *,
        split: str = "development",
        frozen: bool = False,
    ) -> tuple[Path, dict]:
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


if __name__ == "__main__":
    unittest.main()
