from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from scripts import workflow_evidence as evidence


def make_needs(overrides: dict[str, str] | None = None) -> str:
    needs = {
        job_id: {
            "result": "success",
            "outputs": {
                "argv": "SECRET argv",
                "environment": "SECRET environment",
                "raw": "SECRET response",
                "path": "private directory",
                "stdout": "SECRET log",
            },
        }
        for job_id in evidence.JOB_IDS
    }
    needs["validate-pr-flow"]["result"] = "skipped"
    needs["validate-instruction-sync"]["result"] = "skipped"
    needs["validate-main-docs"]["result"] = "skipped"
    for job_id, result in (overrides or {}).items():
        needs[job_id]["result"] = result
    return json.dumps(needs)


class WorkflowEvidenceTests(unittest.TestCase):
    def build_report(self, *, event: str = "push", base_ref: str = "", needs: str | None = None):
        if needs is None:
            overrides = {}
            if event == "pull_request":
                overrides.update(
                    {
                        "validate-pr-flow": "success",
                        "validate-instruction-sync": "success",
                    }
                )
                if base_ref == "main":
                    overrides["validate-main-docs"] = "success"
            needs = make_needs(overrides)
        return evidence.build_report(
            event=event,
            git_sha="a" * 40,
            needs_json=needs,
            base_ref=base_ref,
            generated_at="2026-11-13T08:00:00Z",
        )

    def test_report_keeps_only_job_results_and_safe_metrics(self) -> None:
        report = self.build_report()
        serialized = json.dumps(report)

        self.assertNotIn("SECRET", serialized)
        self.assertNotIn("private directory", serialized)
        self.assertEqual(report["status"], "completed")
        metric = report["metrics"]["ci_duration_seconds"]
        self.assertIsNone(metric["value"])
        self.assertEqual(metric["status"], "unavailable")
        self.assertEqual(metric["source"], "not_provided_by_github.needs")
        self.assertEqual(metric["reason_code"], "job_timestamps_not_supplied")

    def test_required_job_failure_and_cancellation_are_not_completed(self) -> None:
        failure = self.build_report(needs=make_needs({"validate": "failure"}))
        cancelled = self.build_report(needs=make_needs({"validate": "cancelled"}))

        self.assertEqual(failure["status"], "failed")

        optional_failure = self.build_report(needs=make_needs({"validate-main-docs": "failure"}))
        self.assertEqual(optional_failure["status"], "failed")
        self.assertEqual(cancelled["status"], "cancelled")
        with self.assertRaises(evidence.WorkflowEvidenceError):
            evidence.validate_report(
                {**failure, "status": "completed"},
                event="push",
            )
        with self.assertRaises(evidence.WorkflowEvidenceError):
            evidence.validate_report(failure, event="push", require_completed=True)
        self.assertEqual(failure["status"], "failed")

    def test_pr_required_jobs_include_main_docs_only_for_main(self) -> None:
        main = self.build_report(event="pull_request", base_ref="main")
        lab = self.build_report(event="pull_request", base_ref="benchmarking/lab")
        self.assertIn("validate-main-docs", evidence.required_job_ids("pull_request", "main"))
        self.assertNotIn("validate-main-docs", evidence.required_job_ids("pull_request", "benchmarking/lab"))
        self.assertEqual(main["status"], "completed")
        self.assertEqual(lab["status"], "completed")

    def test_collect_binds_source_sha_and_preserves_runner_sha(self) -> None:
        class FakeManagedRun:
            def __init__(self, artifact_root: Path) -> None:
                self.artifact_root = artifact_root
                self.metadata = None

            def complete(self, *, metadata):
                self.metadata = metadata

            def report_failed(self, error):
                raise AssertionError("Collection unexpectedly failed.") from error

        with tempfile.TemporaryDirectory() as temporary_directory:
            managed_run = FakeManagedRun(Path(temporary_directory))
            environment = {
                "GITHUB_EVENT_NAME": "push",
                "GITHUB_SHA": "c" * 40,
                "WORKFLOW_SOURCE_SHA": "d" * 40,
                "GITHUB_BASE_REF": "",
                "GITHUB_RUN_ID": "123",
                "GITHUB_RUN_ATTEMPT": "1",
                "WORKFLOW_NEEDS_JSON": make_needs(),
            }
            with (
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch.object(
                    evidence.artifact_harness.ManagedArtifactRun,
                    "create",
                    return_value=managed_run,
                ),
            ):
                evidence.collect_from_environment()

            report = json.loads(
                (managed_run.artifact_root / "workflow-report.json").read_text(encoding="utf-8")
            )
            markdown = (managed_run.artifact_root / "workflow-report.md").read_text(encoding="utf-8")

        self.assertEqual(report["git_sha"], "d" * 40)
        self.assertEqual(managed_run.metadata["runner_git_sha"], "c" * 40)
        self.assertIn("Source Git SHA", markdown)

    def test_unknown_fields_and_job_states_are_rejected(self) -> None:
        report = self.build_report()
        with self.assertRaises(evidence.WorkflowEvidenceError):
            evidence.validate_report({**report, "private": "secret"}, event="push")

        invalid_jobs = copy.deepcopy(report)
        invalid_jobs["jobs"]["quality"] = "queued"
        with self.assertRaises(evidence.WorkflowEvidenceError):
            evidence.validate_report(invalid_jobs, event="push")

        invalid_metric = copy.deepcopy(report)
        invalid_metric["metrics"]["ci_jobs_successful"]["source"] = "untrusted input"
        with self.assertRaises(evidence.WorkflowEvidenceError):
            evidence.validate_report(invalid_metric, event="push")

    def test_metric_values_must_match_their_status_and_numeric_contract(self) -> None:
        report = self.build_report()
        invalid_values = (True, float("nan"), float("inf"), "12", "unmeasured")
        for value in invalid_values:
            with self.subTest(value=value):
                invalid = copy.deepcopy(report)
                invalid["metrics"]["ci_jobs_successful"]["value"] = value
                with self.assertRaises(evidence.WorkflowEvidenceError):
                    evidence.validate_report(invalid, event="push")

        invalid = copy.deepcopy(report)
        invalid["metrics"]["ci_duration_seconds"].update(
            {"value": 0, "status": "unavailable"}
        )
        with self.assertRaises(evidence.WorkflowEvidenceError):
            evidence.validate_report(invalid, event="push")

    def test_direct_cli_roundtrip_without_pythonpath(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts" / "workflow_evidence.py"
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("GITHUB_BASE_REF", None)

        help_result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)

        isolated_help = subprocess.run(
            [sys.executable, "-I", str(script), "--help"],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(isolated_help.returncode, 0, isolated_help.stderr)

        sha = "a" * 40
        needs = json.dumps({job_id: {"result": "success"} for job_id in evidence.JOB_IDS})
        run_id = str(time.time_ns())
        environment.update(
            {
                "GITHUB_EVENT_NAME": "push",
                "GITHUB_SHA": sha,
                "GITHUB_RUN_ID": run_id,
                "GITHUB_RUN_ATTEMPT": "1",
                "WORKFLOW_NEEDS_JSON": needs,
            }
        )
        collect_result = subprocess.run(
            [sys.executable, str(script), "collect"],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(collect_result.returncode, 0, collect_result.stderr)

        report_path = (
            root
            / "banchmark_result_log"
            / "managed-runs"
            / "80-pr-governance"
            / "ci-workflow-report"
            / f"ci-{run_id}-1"
            / "artifacts"
            / "workflow-report.json"
        )
        run_root = report_path.parents[1]
        manifest = json.loads(
            (run_root / "artifact-manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(evidence.artifact_harness.verify_run(run_root), [])
        with tempfile.TemporaryDirectory() as temporary_directory:
            summary_path = Path(temporary_directory) / "summary.md"
            summary_result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "summary",
                    "--report",
                    str(report_path),
                    "--event",
                    "push",
                    "--summary",
                    str(summary_path),
                ],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(summary_result.returncode, 0, summary_result.stderr)
            self.assertIn(evidence.COMMENT_MARKER, summary_path.read_text(encoding="utf-8"))

        validate_result = subprocess.run(
            [
                sys.executable,
                str(script),
                "validate",
                "--report",
                str(report_path),
                "--event",
                "push",
                "--expected-sha",
                sha,
            ],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(validate_result.returncode, 0, validate_result.stderr)

    def test_comment_marker_creates_then_updates_one_comment(self) -> None:
        comments: list[dict[str, object]] = []
        calls: list[tuple[str, str, object]] = []

        def request(_token: str, method: str, path: str, payload=None):
            calls.append((method, path, payload))
            if method == "GET":
                return comments, {}
            if method == "POST":
                comments.append({"id": 7, "body": payload["body"], "user": {"login": "github-actions[bot]"}})
                return comments[-1], {}
            if method == "PATCH":
                comments[0]["body"] = payload["body"]
                return comments[0], {}
            raise AssertionError(method)

        report = self.build_report()
        body = evidence.render_markdown(report)
        created = evidence.upsert_comment(
            repository="owner/repository",
            pull_request_number=12,
            token="test-token",
            body=body,
            request=request,
        )
        updated = evidence.upsert_comment(
            repository="owner/repository",
            pull_request_number=12,
            token="test-token",
            body=body,
            request=request,
        )

        self.assertEqual((created, updated), ("created", "updated"))
        self.assertEqual(len(comments), 1)
        self.assertEqual([method for method, _path, _payload in calls].count("POST"), 1)
        self.assertEqual([method for method, _path, _payload in calls].count("PATCH"), 1)

    def test_publisher_skips_a_report_when_the_live_pr_head_changed(self) -> None:
        report = self.build_report(event="pull_request", base_ref="main")
        calls: list[tuple[str, str]] = []

        def request(_token: str, method: str, path: str, _payload=None):
            calls.append((method, path))
            if path == "/repos/owner/repository/pulls/12":
                return {
                    "state": "open",
                    "head": {"sha": "b" * 40},
                    "base": {"ref": "main"},
                }, {}
            raise AssertionError("A stale report must not read or change PR comments.")

        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "workflow-report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")

            result = evidence.publish_report(
                report_path=report_path,
                event="pull_request",
                base_ref="main",
                expected_sha="a" * 40,
                repository="owner/repository",
                pull_request_number=12,
                token="test-token",
                request=request,
            )

        self.assertEqual(result, "skipped_stale")
        self.assertEqual(calls, [("GET", "/repos/owner/repository/pulls/12")])

    def test_publisher_rejects_completed_report_from_failed_workflow_run(self) -> None:
        report = self.build_report(event="pull_request", base_ref="main")

        def unexpected(*_args, **_kwargs):
            raise AssertionError("A failed workflow conclusion must be checked before PR access.")

        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "workflow-report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaises(evidence.WorkflowEvidenceError):
                evidence.publish_report(
                    report_path=report_path,
                    event="pull_request",
                    base_ref="main",
                    expected_sha="a" * 40,
                    repository="owner/repository",
                    pull_request_number=12,
                    token="test-token",
                    workflow_conclusion="failure",
                    request=unexpected,
                )

    def test_publisher_checks_live_pr_before_upserting_a_current_report(self) -> None:
        report = self.build_report(event="pull_request", base_ref="main")
        comments: list[dict[str, object]] = []

        def request(_token: str, method: str, path: str, payload=None):
            if path == "/repos/owner/repository/pulls/12":
                return {
                    "state": "open",
                    "head": {"sha": "a" * 40},
                    "base": {"ref": "main"},
                }, {}
            if path == "/repos/owner/repository/issues/12/comments?per_page=100&page=1":
                return comments, {}
            if method == "POST":
                comments.append(
                    {"id": 7, "body": payload["body"], "user": {"login": "github-actions[bot]"}}
                )
                return comments[-1], {}
            raise AssertionError((method, path))

        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "workflow-report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")

            result = evidence.publish_report(
                report_path=report_path,
                event="pull_request",
                base_ref="main",
                expected_sha="a" * 40,
                repository="owner/repository",
                pull_request_number=12,
                token="test-token",
                request=request,
            )

        self.assertEqual(result, "created")
        self.assertEqual(len(comments), 1)

    def test_comment_publish_failure_and_duplicate_marker_fail_closed(self) -> None:
        def denied(*_args, **_kwargs):
            raise evidence.WorkflowEvidenceError("denied")

        with self.assertRaises(evidence.WorkflowEvidenceError):
            evidence.upsert_comment(
                repository="owner/repository",
                pull_request_number=12,
                token="test-token",
                body=evidence.COMMENT_MARKER,
                request=denied,
            )

        def duplicate(_token: str, method: str, _path: str, _payload=None):
            if method == "GET":
                return [
                    {"id": 1, "body": evidence.COMMENT_MARKER, "user": {"login": "github-actions[bot]"}},
                    {"id": 2, "body": evidence.COMMENT_MARKER, "user": {"login": "github-actions[bot]"}},
                ], {}
            raise AssertionError("A duplicate marker must not be modified.")

        with self.assertRaises(evidence.WorkflowEvidenceError):
            evidence.upsert_comment(
                repository="owner/repository",
                pull_request_number=12,
                token="test-token",
                body=evidence.COMMENT_MARKER,
                request=duplicate,
            )

    def test_summary_only_contains_the_validated_report(self) -> None:
        report = self.build_report()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            report_path = root / "workflow-report.json"
            summary_path = root / "summary.md"
            report_path.write_text(json.dumps(report), encoding="utf-8")

            evidence.write_step_summary(
                report_path,
                event="push",
                base_ref="",
                summary_path=summary_path,
            )

            summary = summary_path.read_text(encoding="utf-8")
            self.assertIn(evidence.COMMENT_MARKER, summary)
            self.assertNotIn("SECRET", summary)


if __name__ == "__main__":
    unittest.main()
