from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_REQUIRED = {
    "branch-name",
    "validate-pr-flow",
    "validate-pr-metadata",
    "validate-commit-attribution",
    "validate",
    "report-guard",
}


class WorkflowRequiredCheckTests(unittest.TestCase):
    def test_protected_branch_contexts_keep_existing_checks_and_require_report_guard(self) -> None:
        for ruleset_name in ("01-protect-main.json", "02-protect-develop.json"):
            with self.subTest(ruleset=ruleset_name):
                payload = json.loads((ROOT / ".github" / "rulesets" / ruleset_name).read_text(encoding="utf-8"))
                rules = next(rule for rule in payload["rules"] if rule["type"] == "required_status_checks")
                contexts = {
                    item["context"]
                    for item in rules["parameters"]["required_status_checks"]
                }

                self.assertTrue(EXPECTED_REQUIRED <= contexts)

    def test_main_keeps_its_additional_policy_checks(self) -> None:
        payload = json.loads((ROOT / ".github" / "rulesets" / "01-protect-main.json").read_text(encoding="utf-8"))
        rules = next(rule for rule in payload["rules"] if rule["type"] == "required_status_checks")
        contexts = {item["context"] for item in rules["parameters"]["required_status_checks"]}

        self.assertTrue({"validate-main-docs"} <= contexts)

    def test_report_guard_is_an_always_run_gate_for_the_full_job_set(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        report_guard = workflow.split("  report-guard:\n", 1)[1]

        self.assertIn("    if: always()", report_guard)
        self.assertIn("        id: collect", report_guard)
        self.assertIn('if [[ "$COLLECT_OUTCOME" != "success" ]]', report_guard)
        self.assertIn('test -s "$REPORT_PATH"', report_guard)
        self.assertIn('test -s "$REPORT_MARKDOWN_PATH"', report_guard)
        self.assertIn("python scripts/workflow_evidence.py validate", report_guard)
        self.assertIn("--expected-sha", report_guard)
        self.assertIn("The sanitized workflow report could not be finalized.", report_guard)
        for job in (
            "branch-name",
            "windows-launcher-contract",
            "validate-pr-flow",
            "validate-instruction-sync",
            "validate-commit-attribution",
            "validate-main-docs",
            "validate",
            "quality",
        ):
            self.assertIn(f"      - {job}", report_guard)

        upload = workflow.split("      - name: Upload sanitized workflow report\n", 1)[1].split(
            "      - name: Require completed and valid workflow report", 1
        )[0]
        self.assertIn("workflow-report.json", upload)
        self.assertIn("workflow-report.md", upload)
        self.assertNotIn("artifact-manifest.json", upload)
        self.assertNotIn("/logs/", upload)

    def test_pr_metadata_check_revalidates_edits_without_persisted_credentials(self) -> None:
        metadata = (ROOT / ".github" / "workflows" / "pr-metadata.yml").read_text(
            encoding="utf-8"
        )
        for event_type in ("opened", "reopened", "synchronize", "edited", "ready_for_review"):
            self.assertIn(f"      - {event_type}", metadata)
        self.assertIn("persist-credentials: false", metadata)
        self.assertIn("permissions:\n  contents: read", metadata)
        self.assertIn("python scripts/validate_pr_report_contract.py", metadata)

        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        flow = ci.split("  validate-pr-flow:\n", 1)[1].split(
            "\n  validate-instruction-sync:", 1
        )[0]
        self.assertIn("python scripts/validate_pr_flow.py", flow)
        self.assertNotIn("validate_pr_report_contract.py", flow)

        publisher = (ROOT / ".github" / "workflows" / "publish-ci-report.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("  pull-requests: read", publisher)
        self.assertIn("WORKFLOW_CONCLUSION: ${{ github.event.workflow_run.conclusion }}", publisher)
        self.assertIn("--workflow-conclusion \"$WORKFLOW_CONCLUSION\"", publisher)
        self.assertIn("ref: ${{ github.event.repository.default_branch }}", publisher)

    def test_quality_lane_collects_all_contract_tests_with_pinned_pytest_once(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        quality = workflow.split("  quality:\n", 1)[1].split("\n  report-guard:", 1)[0]
        self.assertIn("python -m pytest -q", quality)
        for path in (
            "tests/test_validation_artifact_harness.py",
            "tests/test_workflow_evidence.py",
            "tests/test_validate_pr_report_contract.py",
            "tests/test_validate_changed_ruff.py",
            "tests/test_validate_pr_flow.py",
            "tests/test_validate_main_docs_policy.py",
            "tests/test_workflow_required_checks.py",
        ):
            self.assertIn(path, quality)

        validate = workflow.split("  validate:\n", 1)[1].split("\n  quality:", 1)[0]
        self.assertNotIn("test_validation_artifact_harness", validate)
        requirements = (ROOT / "requirements-quality.txt").read_text(encoding="utf-8")
        self.assertIn("pytest==9.0.3", requirements)


if __name__ == "__main__":
    unittest.main()
