#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys

FEATURE_BRANCH_RE = re.compile(r"^feature/[a-z0-9][a-z0-9._-]*$")
FIX_BRANCH_RE = re.compile(r"^fix/[a-z0-9][a-z0-9._-]*$")
CHORE_BRANCH_RE = re.compile(r"^chore/[a-z0-9][a-z0-9._-]*$")
HOTFIX_BRANCH_RE = re.compile(r"^hotfix/[a-z0-9][a-z0-9._-]*$")
BENCHMARK_BRANCH_RE = re.compile(r"^benchmarking/lab$")


def validate_pr_flow(head: str, base: str) -> list[str]:
    errors: list[str] = []
    if not head or not base:
        errors.append("Both --head and --base are required.")
        return errors

    if BENCHMARK_BRANCH_RE.match(head):
        errors.append(
            "benchmarking/lab must not be merged into product branches. Promote product changes on a separate feature|fix|chore branch."
        )
        return errors

    if head == "main":
        if base != "develop":
            errors.append("Back-merge from main is only allowed into develop.")
        return errors

    if head == "develop":
        errors.append(
            "Do not open PRs from develop directly. Promote selected work branches into develop, and only back-merge main into develop."
        )
        return errors

    if FEATURE_BRANCH_RE.match(head) or FIX_BRANCH_RE.match(head) or CHORE_BRANCH_RE.match(head):
        if base != "develop":
            errors.append(f"{head} must target develop, not {base}.")
        return errors

    if HOTFIX_BRANCH_RE.match(head):
        if base != "main":
            errors.append(f"{head} must target main, not {base}.")
        return errors

    errors.append(
        f"Unsupported PR head branch '{head}'. Allowed heads are feature|fix|chore|hotfix/* or main for a back-merge to develop."
    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate pull request source/target branch flow.")
    parser.add_argument("--head", required=True, help="Pull request head branch name.")
    parser.add_argument("--base", required=True, help="Pull request base branch name.")
    args = parser.parse_args()

    errors = validate_pr_flow(args.head, args.base)
    if errors:
        for error in errors:
            print(f"[PR-FLOW] {error}", file=sys.stderr)
        return 1

    print(f"Pull request flow checks passed for {args.head} -> {args.base}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
