from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from scripts import validation_artifact_harness as harness


class ValidationArtifactHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.archive_root = Path(self._temporary_directory.name) / "banchmark_result_log"
        self.archive_root.mkdir()
        self.default_patch = mock.patch.object(
            harness,
            "default_archive_root",
            return_value=self.archive_root,
        )
        self.ignore_patch = mock.patch.object(harness, "_is_ignored_by_git", return_value=True)
        self.default_patch.start()
        self.ignore_patch.start()

    def tearDown(self) -> None:
        self.ignore_patch.stop()
        self.default_patch.stop()
        self._temporary_directory.cleanup()

    def create_run(
        self,
        *,
        family: str = "harness-test",
        category: str = "90-cross-cutting",
        run_id: str | None = None,
        hash_limit_bytes: int = harness.DEFAULT_HASH_LIMIT_BYTES,
    ) -> harness.ManagedArtifactRun:
        return harness.ManagedArtifactRun.create(
            family=family,
            category=category,
            run_id=run_id,
            hash_limit_bytes=hash_limit_bytes,
        )

    def test_category_is_required_and_root_must_be_canonical(self) -> None:
        with self.assertRaises(harness.ArtifactHarnessError):
            harness.ManagedArtifactRun.create(family="missing-category", category="")

        with self.assertRaises(harness.ArtifactHarnessError):
            harness.ManagedArtifactRun.create(
                family="wrong-root",
                category="90-cross-cutting",
                archive_root=Path(self._temporary_directory.name) / "elsewhere" / "banchmark_result_log",
            )

    def test_fresh_ignored_archive_root_is_created_before_the_run(self) -> None:
        self.archive_root.rmdir()

        run = self.create_run()

        self.assertTrue(self.archive_root.is_dir())
        self.assertTrue(run.artifact_root.is_dir())

    def test_completed_run_records_document_hash_and_verifies(self) -> None:
        run = self.create_run(hash_limit_bytes=0)
        document = run.artifact_root / "result.json"
        document.write_text('{"status": "ok"}\n', encoding="utf-8")
        run.complete(metadata={"purpose": "unit-test"})

        manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
        record = next(entry for entry in manifest["entries"] if entry["path"] == "artifacts/result.json")
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(record["hash_state"], "recorded-document")
        self.assertTrue(record["sha256"])
        self.assertEqual(run.verify(), [])

        report = json.loads((run.artifact_root / harness.RUN_REPORT_JSON_NAME).read_text(encoding="utf-8"))
        self.assertEqual([event["event"] for event in report["events"]], ["run_started", "run_ended"])
        self.assertEqual(report["events"][-1]["status"], "completed")
        self.assertIsNone(report["metrics"]["process_rss_bytes"]["value"])
        self.assertEqual(report["metrics"]["process_rss_bytes"]["status"], "not_enabled")
        self.assertEqual(report["metrics"]["process_rss_bytes"]["reason_code"], "collection_not_enabled")

    def test_public_finalizers_record_metrics_and_keep_terminal_status_separate(self) -> None:
        measured = {
            "source_units": {
                "value": 17,
                "unit": "source_units",
                "source": "test.counter",
                "status": "measured",
                "reason_code": None,
            }
        }
        completed = self.create_run(run_id="metrics-completed")
        completed.complete(metrics=measured)
        completed_report = json.loads(
            (completed.artifact_root / harness.RUN_REPORT_JSON_NAME).read_text(encoding="utf-8")
        )
        self.assertEqual(completed_report["status"], "completed")
        self.assertEqual(completed_report["metrics"]["source_units"]["value"], 17)
        self.assertEqual(completed.verify(), [])

        failed = self.create_run(run_id="metrics-failed")
        failed.fail(RuntimeError("expected failure"), metrics=measured)
        failed_report = json.loads(
            (failed.artifact_root / harness.RUN_REPORT_JSON_NAME).read_text(encoding="utf-8")
        )
        self.assertEqual(failed_report["status"], "failed")
        self.assertEqual(failed.verify(), [])

        cancelled = self.create_run(run_id="metrics-cancelled")
        cancelled.cancel()
        cancelled_report = json.loads(
            (cancelled.artifact_root / harness.RUN_REPORT_JSON_NAME).read_text(encoding="utf-8")
        )
        self.assertEqual(cancelled_report["status"], "cancelled")
        self.assertEqual(cancelled.verify(), [])

    def test_metrics_reject_non_numeric_or_inconsistent_values(self) -> None:
        invalid_measured_values = (True, float("nan"), float("inf"), "17", None)
        for index, value in enumerate(invalid_measured_values):
            with self.subTest(value=value):
                run = self.create_run(run_id=f"invalid-measured-{index}")
                metric = {
                    "value": value,
                    "unit": "source_units",
                    "source": "test.counter",
                    "status": "measured",
                    "reason_code": None,
                }
                with self.assertRaises(harness.ArtifactHarnessError):
                    run.complete(metrics={"source_units": metric})

        inconsistent = {
            "value": 17,
            "unit": "source_units",
            "source": "test.counter",
            "status": "unavailable",
            "reason_code": "collection_not_enabled",
        }
        run = self.create_run(run_id="invalid-non-measured-value")
        with self.assertRaises(harness.ArtifactHarnessError):
            run.complete(metrics={"source_units": inconsistent})

        invalid_contracts = (
            {"value": 17, "unit": 1},
            {"value": 17, "source": ["test.counter"]},
            {"value": 17, "status": []},
            {"value": 17, "reason_code": "unexpected"},
            {"value": None, "status": "unavailable", "reason_code": 1},
        )
        for index, overrides in enumerate(invalid_contracts):
            with self.subTest(contract=overrides):
                run = self.create_run(run_id=f"invalid-contract-{index}")
                metric = {
                    "value": 17,
                    "unit": "source_units",
                    "source": "test.counter",
                    "status": "measured",
                    "reason_code": None,
                }
                metric.update(overrides)
                with self.assertRaises(harness.ArtifactHarnessError):
                    run.complete(metrics={"source_units": metric})

    def test_running_run_is_not_a_verified_completion(self) -> None:
        run = self.create_run()

        self.assertTrue(any("Run is not finalized" in error for error in run.verify()))

        manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
        manifest["schema_version"] = 1
        run.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertTrue(any("Run is not finalized" in error for error in run.verify()))

    def test_schema_one_completed_manifest_remains_compatible(self) -> None:
        run = self.create_run()
        run.complete()

        manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
        manifest["schema_version"] = 1
        (run.artifact_root / harness.RUN_REPORT_JSON_NAME).unlink()
        (run.artifact_root / harness.RUN_REPORT_MARKDOWN_NAME).unlink()
        run.events_path.unlink()
        manifest["entries"] = harness._collect_entries(
            run.run_root,
            hash_limit_bytes=manifest["hash_limit_bytes"],
        )
        run.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        self.assertEqual(run.verify(), [])

    def test_report_failure_cannot_complete_a_run(self) -> None:
        run = self.create_run()
        with mock.patch.object(run, "_render_report", side_effect=OSError("disk unavailable")):
            with self.assertRaises(harness.ArtifactHarnessError):
                run.complete()

        manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "report_failed")
        self.assertTrue(run.verify())

    def test_cancelled_interrupted_and_timed_out_are_terminal_non_successes(self) -> None:
        for run_id, method_name, expected in (
            ("cancelled-case", "cancel", "cancelled"),
            ("interrupted-case", "interrupt", "interrupted"),
            ("timeout-case", "timeout", "timed_out"),
        ):
            with self.subTest(status=expected):
                run = self.create_run(run_id=run_id)
                getattr(run, method_name)()
                manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"], expected)
                self.assertEqual(run.verify(), [])

        publish_run = self.create_run(run_id="publish-failed-case")
        publish_run.publish_failed(RuntimeError("publish denied"))
        publish_manifest = json.loads(publish_run.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(publish_manifest["status"], "publish_failed")
        self.assertEqual(publish_run.verify(), [])

    def test_fail_classifies_cancellation_timeout_and_interruption(self) -> None:
        for run_id, error, expected in (
            ("keyboard-case", KeyboardInterrupt(), "cancelled"),
            ("interrupted-error-case", InterruptedError(), "interrupted"),
            ("timeout-error-case", TimeoutError(), "timed_out"),
            ("subprocess-timeout-case", harness.subprocess.TimeoutExpired(["command"], 1), "timed_out"),
        ):
            with self.subTest(status=expected):
                run = self.create_run(run_id=run_id)
                run.fail(error)
                manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"], expected)

    def test_verification_ignores_access_time_but_detects_content_tampering(self) -> None:
        run = self.create_run()
        artifact = run.artifact_root / "artifact.txt"
        artifact.write_text("original", encoding="utf-8")
        run.complete()

        manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
        for entry in manifest["entries"]:
            if entry["path"] == "artifacts/artifact.txt":
                entry["access_time_ns"] = 1
        run.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(run.verify(), [])

        artifact.write_text("tampered", encoding="utf-8")
        self.assertTrue(any("Metadata mismatch" in error for error in run.verify()))

    def test_inherited_output_must_belong_to_managed_run(self) -> None:
        run = self.create_run(family="child-script")
        inherited = {
            "CT_VALIDATION_OUTPUT_DIR": str(run.artifact_root),
            "CT_VALIDATION_RUN_ROOT": str(run.run_root),
            "CT_VALIDATION_CATEGORY": run.category,
            "CT_VALIDATION_FAMILY": "child-script",
        }
        with mock.patch.dict(os.environ, inherited, clear=False):
            output_root, owner = harness.select_managed_output_directory(
                family="child-script",
                category="90-cross-cutting",
            )
        self.assertEqual(output_root, run.artifact_root)
        self.assertIsNone(owner)

        with mock.patch.dict(
            os.environ,
            {"CT_VALIDATION_OUTPUT_DIR": str(run.artifact_root)},
            clear=False,
        ):
            os.environ.pop("CT_VALIDATION_RUN_ROOT", None)
            with self.assertRaises(harness.ArtifactHarnessError):
                harness.select_managed_output_directory(
                    family="child-script",
                    category="90-cross-cutting",
                )

        mismatched = {
            "CT_VALIDATION_OUTPUT_DIR": str(run.artifact_root),
            "CT_VALIDATION_RUN_ROOT": str(run.run_root),
            "CT_VALIDATION_CATEGORY": run.category,
            "CT_VALIDATION_FAMILY": "different-family",
        }
        with mock.patch.dict(os.environ, mismatched, clear=False):
            with self.assertRaises(harness.ArtifactHarnessError):
                harness.select_managed_output_directory(
                    family="child-script",
                    category="90-cross-cutting",
                )

        wrong_output = dict(inherited)
        wrong_output["CT_VALIDATION_OUTPUT_DIR"] = str(run.run_root / "wrong-output")
        with mock.patch.dict(os.environ, wrong_output, clear=False):
            with self.assertRaises(harness.ArtifactHarnessError):
                harness.select_managed_output_directory(
                    family="child-script",
                    category="90-cross-cutting",
                )

    def test_refuses_symbolic_linked_managed_runs_directory(self) -> None:
        external_directory = Path(self._temporary_directory.name) / "outside"
        external_directory.mkdir()
        try:
            (self.archive_root / harness.MANAGED_RUNS_DIRECTORY_NAME).symlink_to(
                external_directory,
                target_is_directory=True,
            )
        except OSError:
            self.skipTest("symbolic links are unavailable in this environment")

        with self.assertRaises(harness.ArtifactHarnessError):
            self.create_run()

    def test_shell_runner_captures_child_output_and_completes_manifest(self) -> None:
        code = (
            "import os; from pathlib import Path; "
            "Path(os.environ['CT_VALIDATION_OUTPUT_DIR'], 'child.txt').write_text('ok', encoding='utf-8')"
        )
        arguments = argparse.Namespace(
            family="runner-test",
            category="90-cross-cutting",
            run_id="runner-case",
            cwd=str(self._temporary_directory.name),
            command=[sys.executable, "-c", code],
        )

        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(harness._run_command(arguments), 0)
        run_root = self.archive_root / "managed-runs" / "90-cross-cutting" / "runner-test" / "runner-case"
        manifest = json.loads((run_root / harness.MANIFEST_FILE_NAME).read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "completed")
        self.assertTrue((run_root / "artifacts" / "child.txt").is_file())
        self.assertTrue((run_root / "logs" / "stdout.log").is_file())
        self.assertEqual(harness.verify_run(run_root), [])

    def test_cli_separator_is_not_executed_as_the_child_command(self) -> None:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            result = harness.main(
                [
                    "run",
                    "--family",
                    "cli-separator",
                    "--category",
                    "90-cross-cutting",
                    "--run-id",
                    "separator-case",
                    "--",
                    sys.executable,
                    "-c",
                    "pass",
                ]
            )

        self.assertEqual(result, 0)
        run_root = self.archive_root / "managed-runs" / "90-cross-cutting" / "cli-separator" / "separator-case"
        self.assertEqual(harness.verify_run(run_root), [])


if __name__ == "__main__":
    unittest.main()
