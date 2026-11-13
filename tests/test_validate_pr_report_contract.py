from __future__ import annotations

import unittest

from scripts.validate_pr_report_contract import validate_pr_body


VALID_BODY = """## Issue references
Refs #315 — workflow reporting baseline

## Measurement denominator
Count source units that were confirmed as required before processing; N/A is allowed only with a reason.

## Validation commands and results
`python -m unittest tests.test_workflow_evidence` — passed.

## Failed / unjudged / unmeasured
- failed: 0
- unjudged: 0
- unmeasured: 1

## Evidence status
status: partial — one environment did not provide the optional measurement.

## Impact
This changes the CI evidence summary and PR metadata validation.

## Rollback
Revert the reporting workflow and restore the previous required check list.

## Not run
No model-backed runs were needed for this change.
"""


class PullRequestReportContractTests(unittest.TestCase):
    def test_complete_report_contract_passes(self) -> None:
        self.assertEqual(validate_pr_body(VALID_BODY), [])

    def test_missing_issue_reference_and_count_rows_fail(self) -> None:
        body = VALID_BODY.replace("Refs #315 — workflow reporting baseline", "No issue linked")
        body = body.replace("- unjudged: 0\n", "")

        errors = validate_pr_body(body)

        self.assertTrue(any("Issue references" in error for error in errors))
        self.assertTrue(any("unjudged" in error for error in errors))

    def test_placeholder_or_missing_validation_command_fails(self) -> None:
        body = VALID_BODY.replace("`python -m unittest tests.test_workflow_evidence` — passed.", "TBD")

        errors = validate_pr_body(body)

        self.assertTrue(any("Validation section" in error for error in errors))

    def test_empty_required_section_fails(self) -> None:
        body = VALID_BODY.replace("## Rollback\nRevert the reporting workflow and restore the previous required check list.", "## Rollback\n")

        errors = validate_pr_body(body)

        self.assertTrue(any("Required section is empty: Rollback" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
