#!/usr/bin/env python3
"""GitHub Actions 상태를 허용 목록으로 정제해 보고한다."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import validation_artifact_harness as artifact_harness


COMMENT_MARKER = "<!-- comic-translate-ci-evidence-v1 -->"
JOB_IDS = (
    "branch-name",
    "windows-launcher-contract",
    "validate-pr-flow",
    "validate-instruction-sync",
    "validate-commit-attribution",
    "validate-main-docs",
    "validate",
    "quality",
)
JOB_RESULTS = frozenset({"success", "failure", "cancelled", "skipped"})
REPORT_EVENTS = frozenset({"push", "pull_request"})
REPORT_STATUSES = frozenset({"completed", "failed", "cancelled"})
WORKFLOW_CONCLUSIONS = frozenset(
    {
        "success",
        "failure",
        "cancelled",
        "timed_out",
        "action_required",
        "neutral",
        "stale",
        "startup_failure",
        "skipped",
    }
)
METRIC_KEYS = (
    "ci_jobs_successful",
    "ci_jobs_failed",
    "ci_jobs_cancelled",
    "ci_jobs_skipped",
    "ci_duration_seconds",
)
SHA_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class WorkflowEvidenceError(RuntimeError):
    """보고 입력이나 게시 경로가 계약을 지키지 않을 때 발생한다."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def required_job_ids(event: str, base_ref: str = "") -> set[str]:
    required = {
        "branch-name",
        "windows-launcher-contract",
        "validate-commit-attribution",
        "validate",
        "quality",
    }
    if event == "pull_request":
        required.update({"validate-pr-flow", "validate-instruction-sync"})
        if base_ref == "main":
            required.add("validate-main-docs")
    return required


def _metric(value: int | None, *, unit: str, source: str, status: str, reason: str | None) -> dict[str, Any]:
    return {
        "value": value,
        "unit": unit,
        "source": source,
        "status": status,
        "reason_code": reason,
    }


def build_report(
    *,
    event: str,
    git_sha: str,
    needs_json: str,
    base_ref: str = "",
    generated_at: str | None = None,
) -> dict[str, Any]:
    if event not in REPORT_EVENTS or not SHA_RE.fullmatch(git_sha):
        raise WorkflowEvidenceError("Workflow event or Git SHA is invalid.")
    if len(needs_json) > 262144:
        raise WorkflowEvidenceError("Workflow job result input is too large.")
    try:
        needs = json.loads(needs_json, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, ValueError) as exc:
        raise WorkflowEvidenceError("Workflow job result input is invalid.") from exc
    if not isinstance(needs, dict) or set(needs) != set(JOB_IDS):
        raise WorkflowEvidenceError("Workflow job result set does not match the contract.")

    jobs: dict[str, str] = {}
    for job_id in JOB_IDS:
        value = needs[job_id]
        result = value.get("result") if isinstance(value, dict) else None
        if not isinstance(result, str) or result not in JOB_RESULTS:
            raise WorkflowEvidenceError("Workflow job result is invalid.")
        jobs[job_id] = result

    required = required_job_ids(event, base_ref)
    required_results = {jobs[job_id] for job_id in required}
    all_results = set(jobs.values())
    if "cancelled" in all_results:
        status = "cancelled"
    elif "failure" in all_results or required_results != {"success"}:
        status = "failed"
    else:
        status = "completed"

    counts = {result: sum(value == result for value in jobs.values()) for result in JOB_RESULTS}
    metrics = {
        "ci_jobs_successful": _metric(
            counts["success"], unit="jobs", source="github.needs.results", status="measured", reason=None
        ),
        "ci_jobs_failed": _metric(
            counts["failure"], unit="jobs", source="github.needs.results", status="measured", reason=None
        ),
        "ci_jobs_cancelled": _metric(
            counts["cancelled"], unit="jobs", source="github.needs.results", status="measured", reason=None
        ),
        "ci_jobs_skipped": _metric(
            counts["skipped"], unit="jobs", source="github.needs.results", status="measured", reason=None
        ),
        "ci_duration_seconds": _metric(
            None,
            unit="seconds",
            source="not_provided_by_github.needs",
            status="unavailable",
            reason="job_timestamps_not_supplied",
        ),
    }
    report = {
        "schema_version": 1,
        "event": event,
        "git_sha": git_sha,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": status,
        "jobs": jobs,
        "metrics": metrics,
    }
    validate_report(report, event=event, base_ref=base_ref)
    return report


def validate_report(
    report: Mapping[str, Any],
    *,
    event: str,
    base_ref: str = "",
    require_completed: bool = False,
) -> None:
    expected_keys = {"schema_version", "event", "git_sha", "generated_at", "status", "jobs", "metrics"}
    if (
        set(report) != expected_keys
        or type(report.get("schema_version")) is not int
        or report.get("schema_version") != 1
    ):
        raise WorkflowEvidenceError("Sanitized workflow report has an invalid schema.")
    if report.get("event") != event or event not in REPORT_EVENTS:
        raise WorkflowEvidenceError("Sanitized workflow report event does not match.")
    git_sha = report.get("git_sha")
    generated_at = report.get("generated_at")
    status = report.get("status")
    if not isinstance(git_sha, str) or not SHA_RE.fullmatch(git_sha):
        raise WorkflowEvidenceError("Sanitized workflow report Git SHA is invalid.")
    if not isinstance(generated_at, str) or not UTC_RE.fullmatch(generated_at):
        raise WorkflowEvidenceError("Sanitized workflow report timestamp is invalid.")
    try:
        datetime.strptime(generated_at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise WorkflowEvidenceError("Sanitized workflow report timestamp is invalid.") from exc
    if not isinstance(status, str) or status not in REPORT_STATUSES:
        raise WorkflowEvidenceError("Sanitized workflow report status is invalid.")

    jobs = report.get("jobs")
    if not isinstance(jobs, dict) or set(jobs) != set(JOB_IDS):
        raise WorkflowEvidenceError("Sanitized workflow report job allowlist is invalid.")
    if any(not isinstance(result, str) or result not in JOB_RESULTS for result in jobs.values()):
        raise WorkflowEvidenceError("Sanitized workflow report job state is invalid.")
    expected_metrics = set(METRIC_KEYS)
    metrics = report.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != expected_metrics:
        raise WorkflowEvidenceError("Sanitized workflow report metric allowlist is invalid.")
    for name in METRIC_KEYS[:-1]:
        metric = metrics[name]
        if (
            not isinstance(metric, dict)
            or set(metric) != {"value", "unit", "source", "status", "reason_code"}
            or type(metric["value"]) is not int
            or metric["value"] < 0
            or metric["unit"] != "jobs"
            or metric["source"] != "github.needs.results"
            or metric["status"] != "measured"
            or metric["reason_code"] is not None
        ):
            raise WorkflowEvidenceError("Sanitized workflow report metric is invalid.")
    expected_counts = {
        "ci_jobs_successful": sum(result == "success" for result in jobs.values()),
        "ci_jobs_failed": sum(result == "failure" for result in jobs.values()),
        "ci_jobs_cancelled": sum(result == "cancelled" for result in jobs.values()),
        "ci_jobs_skipped": sum(result == "skipped" for result in jobs.values()),
    }
    if any(metrics[name]["value"] != value for name, value in expected_counts.items()):
        raise WorkflowEvidenceError("Sanitized workflow report metric does not match job results.")
    duration = metrics["ci_duration_seconds"]
    if (
        not isinstance(duration, dict)
        or set(duration) != {"value", "unit", "source", "status", "reason_code"}
        or duration["value"] is not None
        or duration["unit"] != "seconds"
        or duration["source"] != "not_provided_by_github.needs"
        or duration["status"] != "unavailable"
        or duration["reason_code"] != "job_timestamps_not_supplied"
    ):
        raise WorkflowEvidenceError("Sanitized workflow report unavailable metric is invalid.")

    required_results = {jobs[job_id] for job_id in required_job_ids(event, base_ref)}
    if "cancelled" in jobs.values():
        expected_status = "cancelled"
    elif "failure" in jobs.values() or required_results != {"success"}:
        expected_status = "failed"
    else:
        expected_status = "completed"
    if status != expected_status:
        raise WorkflowEvidenceError("Sanitized workflow report completion state is inconsistent.")
    if require_completed and status != "completed":
        raise WorkflowEvidenceError("Required CI jobs did not complete successfully.")


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        COMMENT_MARKER,
        "## CI evidence",
        "",
        f"- Status: `{report['status']}`",
        f"- Event: `{report['event']}`",
        f"- Git SHA: `{report['git_sha']}`",
        f"- Generated: `{report['generated_at']}`",
        "",
        "### Jobs",
        "",
        "| Job | Result |",
        "| --- | --- |",
    ]
    lines.extend(f"| `{job_id}` | `{report['jobs'][job_id]}` |" for job_id in JOB_IDS)
    lines.extend(["", "### Metrics", "", "| Metric | Value | State | Source |", "| --- | --- | --- | --- |"])
    for name in METRIC_KEYS:
        metric = report["metrics"][name]
        value = "null" if metric["value"] is None else str(metric["value"])
        reason = metric["reason_code"] or "none"
        lines.append(
            f"| `{name}` | `{value} {metric['unit']}` | `{metric['status']}` "
            f"(reason: `{reason}`) | `{metric['source']}` |"
        )
    return "\n".join(lines) + "\n"


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    artifact_harness._atomic_write_json(path / "workflow-report.json", report)
    artifact_harness._atomic_write_text(path / "workflow-report.md", render_markdown(report))


def collect_from_environment() -> Path:
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    sha = os.environ.get("GITHUB_SHA", "")
    needs_json = os.environ.get("WORKFLOW_NEEDS_JSON", "")
    base_ref = os.environ.get("GITHUB_BASE_REF", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "")
    if not run_id.isdigit() or not attempt.isdigit():
        raise WorkflowEvidenceError("GitHub workflow run identity is invalid.")

    managed_run = artifact_harness.ManagedArtifactRun.create(
        family="ci-workflow-report",
        category="80-pr-governance",
        run_id=f"ci-{run_id}-{attempt}",
    )
    try:
        report = build_report(event=event, git_sha=sha, needs_json=needs_json, base_ref=base_ref)
        _write_report(managed_run.artifact_root, report)
        managed_run.complete(metadata={"event": event, "report_schema_version": 1})
    except Exception as exc:
        managed_run.report_failed(exc)
        raise WorkflowEvidenceError("Unable to create the sanitized workflow report.") from exc
    return managed_run.artifact_root


def read_validated_report(
    path: Path,
    *,
    event: str,
    base_ref: str,
    expected_sha: str = "",
    require_completed: bool = False,
) -> dict[str, Any]:
    try:
        if path.stat().st_size > 262144:
            raise WorkflowEvidenceError("Sanitized workflow report is too large.")
        report = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except WorkflowEvidenceError:
        raise
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise WorkflowEvidenceError("Sanitized workflow report is missing or unreadable.") from exc
    if not isinstance(report, dict):
        raise WorkflowEvidenceError("Sanitized workflow report is invalid.")
    validate_report(report, event=event, base_ref=base_ref, require_completed=require_completed)
    if expected_sha and (not SHA_RE.fullmatch(expected_sha) or report["git_sha"] != expected_sha):
        raise WorkflowEvidenceError("Sanitized workflow report Git SHA does not match.")
    return report


def write_step_summary(report_path: Path, *, event: str, base_ref: str, summary_path: Path) -> None:
    try:
        report = read_validated_report(report_path, event=event, base_ref=base_ref)
        content = render_markdown(report)
    except WorkflowEvidenceError:
        content = "## CI evidence unavailable\n\nThe required sanitized report could not be validated.\n"
        try:
            existing = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
            artifact_harness._atomic_write_text(summary_path, existing + content)
        except OSError as exc:
            raise WorkflowEvidenceError("Workflow step summary could not be written.") from exc
        raise
    try:
        existing = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
        artifact_harness._atomic_write_text(summary_path, existing + content)
    except OSError as exc:
        raise WorkflowEvidenceError("Workflow step summary could not be written.") from exc


def _summary_command(args: argparse.Namespace) -> None:
    summary_path = args.summary
    if summary_path is None:
        selected = os.environ.get("GITHUB_STEP_SUMMARY", "")
        if not selected:
            raise WorkflowEvidenceError("Workflow step summary path is unavailable.")
        summary_path = Path(selected)
    write_step_summary(
        args.report,
        event=args.event,
        base_ref=args.base_ref,
        summary_path=summary_path,
    )


def _github_request(
    token: str,
    method: str,
    path: str,
    payload: Mapping[str, Any] | None = None,
) -> tuple[Any, Mapping[str, str]]:
    if not token:
        raise WorkflowEvidenceError("GitHub report publishing token is unavailable.")
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        f"https://api.github.com{path}",
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=20) as response:
            body = response.read()
            result = json.loads(body, object_pairs_hook=_unique_object) if body else None
            return result, response.headers
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError) as exc:
        raise WorkflowEvidenceError("GitHub report publishing request failed.") from exc


def upsert_comment(
    *,
    repository: str,
    pull_request_number: int,
    token: str,
    body: str,
    request: Callable[..., tuple[Any, Mapping[str, str]]] = _github_request,
) -> str:
    if (
        not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)
        or type(pull_request_number) is not int
        or pull_request_number < 1
    ):
        raise WorkflowEvidenceError("GitHub pull request target is invalid.")
    matches: list[Mapping[str, Any]] = []
    page = 1
    while True:
        comments, headers = request(
            token,
            "GET",
            f"/repos/{repository}/issues/{pull_request_number}/comments?per_page=100&page={page}",
        )
        if not isinstance(comments, list):
            raise WorkflowEvidenceError("GitHub comments response is invalid.")
        matches.extend(
            comment
            for comment in comments
            if isinstance(comment, dict) and COMMENT_MARKER in str(comment.get("body", ""))
        )
        link_header = headers.get("Link", "")
        if not isinstance(link_header, str) or 'rel="next"' not in link_header:
            break
        page += 1
    if len(matches) > 1:
        raise WorkflowEvidenceError("Duplicate CI evidence markers require manual review.")
    if matches:
        comment = matches[0]
        user = comment.get("user")
        if not isinstance(user, dict) or user.get("login") != "github-actions[bot]":
            raise WorkflowEvidenceError("CI evidence marker is owned by another account.")
        comment_id = comment.get("id")
        if type(comment_id) is not int:
            raise WorkflowEvidenceError("Existing CI evidence comment has no valid ID.")
        request(token, "PATCH", f"/repos/{repository}/issues/comments/{comment_id}", {"body": body})
        return "updated"
    request(token, "POST", f"/repos/{repository}/issues/{pull_request_number}/comments", {"body": body})
    return "created"


def publish_report(
    *,
    report_path: Path,
    event: str,
    base_ref: str,
    expected_sha: str,
    repository: str,
    pull_request_number: int,
    token: str,
    workflow_conclusion: str = "success",
    request: Callable[..., tuple[Any, Mapping[str, str]]] = _github_request,
) -> str:
    if (
        not isinstance(repository, str)
        or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)
        or type(pull_request_number) is not int
        or pull_request_number < 1
        or not isinstance(expected_sha, str)
        or not SHA_RE.fullmatch(expected_sha)
        or not isinstance(workflow_conclusion, str)
        or workflow_conclusion not in WORKFLOW_CONCLUSIONS
    ):
        raise WorkflowEvidenceError("GitHub pull request target is invalid.")
    report = read_validated_report(
        report_path,
        event=event,
        base_ref=base_ref,
        expected_sha=expected_sha,
    )
    if (report["status"] == "completed") != (workflow_conclusion == "success"):
        raise WorkflowEvidenceError("Workflow conclusion does not match the sanitized report.")
    pull_request, _headers = request(
        token,
        "GET",
        f"/repos/{repository}/pulls/{pull_request_number}",
    )
    if not isinstance(pull_request, dict):
        raise WorkflowEvidenceError("GitHub pull request response is invalid.")
    head = pull_request.get("head")
    base = pull_request.get("base")
    head_sha = head.get("sha") if isinstance(head, dict) else None
    live_base_ref = base.get("ref") if isinstance(base, dict) else None
    state = pull_request.get("state")
    if (
        not isinstance(head_sha, str)
        or not SHA_RE.fullmatch(head_sha)
        or not isinstance(live_base_ref, str)
        or not isinstance(state, str)
        or state not in {"open", "closed"}
    ):
        raise WorkflowEvidenceError("GitHub pull request response is invalid.")
    if state != "open" or head_sha != expected_sha or live_base_ref != base_ref:
        return "skipped_stale"
    return upsert_comment(
        repository=repository,
        pull_request_number=pull_request_number,
        token=token,
        body=render_markdown(report),
        request=request,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="정제된 GitHub Actions 보고서를 생성하거나 검증한다.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect")
    collect.set_defaults(handler=lambda _args: collect_from_environment())
    validate = subparsers.add_parser("validate")
    validate.add_argument("--report", type=Path, required=True)
    validate.add_argument("--event", choices=sorted(REPORT_EVENTS), required=True)
    validate.add_argument("--base-ref", default="")
    validate.add_argument("--expected-sha", default="")
    validate.set_defaults(
        handler=lambda args: read_validated_report(
            args.report,
            event=args.event,
            base_ref=args.base_ref,
            expected_sha=args.expected_sha,
            require_completed=True,
        )
    )
    summary = subparsers.add_parser("summary")
    summary.add_argument("--report", type=Path, required=True)
    summary.add_argument("--event", choices=sorted(REPORT_EVENTS), required=True)
    summary.add_argument("--base-ref", default="")
    summary.add_argument("--summary", type=Path)
    summary.set_defaults(handler=_summary_command)
    publish = subparsers.add_parser("publish")
    publish.add_argument("--report", type=Path, required=True)
    publish.add_argument("--event", choices=sorted(REPORT_EVENTS), required=True)
    publish.add_argument("--base-ref", default="")
    publish.add_argument("--expected-sha", required=True)
    publish.add_argument("--repository", required=True)
    publish.add_argument("--pull-request-number", type=int, required=True)
    publish.add_argument("--workflow-conclusion", required=True)
    publish.set_defaults(
        handler=lambda args: publish_report(
            report_path=args.report,
            event=args.event,
            base_ref=args.base_ref,
            expected_sha=args.expected_sha,
            repository=args.repository,
            pull_request_number=args.pull_request_number,
            token=os.environ.get("GITHUB_TOKEN", ""),
            workflow_conclusion=args.workflow_conclusion,
        )
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
        if args.command == "collect":
            return 0
        if args.command == "validate":
            print("Sanitized workflow report validation passed.")
        elif args.command == "summary":
            print("Sanitized workflow summary recorded.")
        elif args.command == "publish":
            if result == "skipped_stale":
                print("Sanitized workflow report comment skipped because the PR changed.")
            else:
                print(f"Sanitized workflow report comment {result}.")
        return 0
    except (WorkflowEvidenceError, artifact_harness.ArtifactHarnessError, OSError, ValueError):
        print("[WORKFLOW-EVIDENCE] Report generation, validation, or publication failed.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
