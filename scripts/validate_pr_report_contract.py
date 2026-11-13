#!/usr/bin/env python3
"""PR 본문에 issue 연결과 검증 상태가 빠지지 않았는지 확인한다."""

from __future__ import annotations

import os
import re
import sys


REQUIRED_SECTIONS = (
    "Issue references",
    "Measurement denominator",
    "Validation commands and results",
    "Failed / unjudged / unmeasured",
    "Evidence status",
    "Impact",
    "Rollback",
    "Not run",
)
ISSUE_REFERENCE_RE = re.compile(r"\b(?:Refs|Closes|Fixes|Resolves)\s+#[1-9]\d*\b", re.IGNORECASE)
PLACEHOLDER_RE = re.compile(r"\b(?:todo|tbd|fill\s+in|replace\s+me)\b|<[^>]+>", re.IGNORECASE)
COMMAND_RE = re.compile(r"\b(?:python(?:3)?|pytest|ruff|git|pwsh|cmd(?:\.exe)?)\s+[^\n`]+", re.IGNORECASE)
COUNT_LABELS = ("failed", "unjudged", "unmeasured")


def _sections(body: str) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    current = ""
    for line in body.splitlines():
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            current = heading.group(1).strip()
            found.setdefault(current, [])
        elif current:
            found[current].append(line)
    return found


def _visible_text(lines: list[str]) -> str:
    text = "\n".join(lines)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    return text.strip()


def _has_reason(text: str) -> bool:
    normalized = re.sub(r"[`*_]", "", text).strip().lower()
    return len(re.sub(r"[^a-z0-9가-힣]", "", normalized)) >= 3 and not PLACEHOLDER_RE.search(normalized)


def validate_pr_body(body: str) -> list[str]:
    sections = _sections(body)
    errors: list[str] = []
    for section in REQUIRED_SECTIONS:
        if section not in sections:
            errors.append(f"Missing required section: {section}.")
            continue
        if not _visible_text(sections[section]):
            errors.append(f"Required section is empty: {section}.")

    issue_text = _visible_text(sections.get("Issue references", []))
    if issue_text and not ISSUE_REFERENCE_RE.search(issue_text):
        errors.append("Issue references must contain Refs #N or Closes #N.")

    denominator = _visible_text(sections.get("Measurement denominator", []))
    if denominator and not _has_reason(denominator):
        errors.append("Measurement denominator must name a countable unit or explain why it does not apply.")

    validation = _visible_text(sections.get("Validation commands and results", []))
    if validation and (PLACEHOLDER_RE.search(validation) or not COMMAND_RE.search(validation)):
        errors.append("Validation section must include a command and its result or reason it was not run.")

    counts = _visible_text(sections.get("Failed / unjudged / unmeasured", []))
    for label in COUNT_LABELS:
        match = re.search(rf"(?im)^\s*[-*]?\s*{label}\s*:\s*(.+?)\s*$", counts)
        if not match:
            errors.append(f"Failed / unjudged / unmeasured section must include {label}:.")
            continue
        value = match.group(1).strip().strip("` ")
        if not (re.fullmatch(r"\d+", value) or re.match(r"(?i)^n/?a\s*[:—-]\s*\S", value)):
            errors.append(f"{label}: must be a count or N/A with a reason.")

    evidence = _visible_text(sections.get("Evidence status", []))
    if evidence and not re.search(r"(?im)^\s*status\s*:\s*(?:complete|partial|unavailable)\b.+", evidence):
        errors.append("Evidence status must be complete, partial, or unavailable with a reason.")
    for section in ("Impact", "Rollback", "Not run"):
        content = _visible_text(sections.get(section, []))
        if content and not _has_reason(content):
            errors.append(f"{section} must contain a concrete value or a reasoned N/A.")
    return errors


def main() -> int:
    errors = validate_pr_body(os.environ.get("PR_BODY", ""))
    if errors:
        for error in errors:
            print(f"[PR-REPORT-CONTRACT] {error}", file=sys.stderr)
        return 1
    print("Pull request issue and evidence contract passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
