#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Iterable
from email.utils import parseaddr
from pathlib import Path
import re
import subprocess
import sys
import unicodedata

ZERO_SHA_RE = re.compile(r"^0+$")

AI_NAME_PATTERNS = (
    re.compile(
        r"(?:openai\s+)?codex(?:codex)?",
        re.IGNORECASE,
    ),
    re.compile(r"(?:openai\s+)?chatgpt(?:\s+(?:assistant|bot))?", re.I),
    re.compile(r"(?:github\s+)?copilot(?:\s+(?:assistant|bot))?", re.I),
    re.compile(
        r"(?:anthropic\s+)?claude(?:\s+(?:ai|code|assistant|bot))?",
        re.I,
    ),
    re.compile(
        r"(?:google\s+)?gemini(?:\s+(?:ai|cli|code|assistant|bot))?",
        re.I,
    ),
)
EXACT_AI_EMAIL_LOCAL_PARTS = {
    "anthropic-claude",
    "chatgpt",
    "chatgpt-assistant",
    "chatgpt-bot",
    "claude",
    "claude-ai",
    "claude-assistant",
    "claude-bot",
    "claude-code",
    "copilot",
    "copilot-assistant",
    "copilot-bot",
    "gemini",
    "gemini-ai",
    "gemini-assistant",
    "gemini-bot",
    "gemini-cli",
    "gemini-code",
    "github-copilot",
    "google-gemini",
}


def run_git(*args: str, input_text: str | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        input=input_text,
    )
    return completed.stdout


def normalize_identity_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    without_format_chars = "".join(
        char for char in normalized if unicodedata.category(char) != "Cf"
    )
    return " ".join(without_format_chars.strip().casefold().split())


def parsed_identity(value: str) -> tuple[str, str]:
    compact = " ".join(value.strip().split())
    parsed_name, parsed_email = parseaddr(compact)
    if parsed_email and "@" in parsed_email:
        return (
            normalize_identity_text(parsed_name),
            normalize_identity_text(parsed_email),
        )
    return normalize_identity_text(value.split("<", 1)[0]), ""


def forbidden_ai_identity(value: str) -> bool:
    name, email = parsed_identity(value)
    if any(pattern.fullmatch(name) for pattern in AI_NAME_PATTERNS):
        return True

    if not email:
        return False
    local_part = email.split("@", 1)[0]
    local_tokens = set(filter(None, re.split(r"[^a-z0-9]+", local_part)))
    return bool(
        {"codex", "codexcodex", "chatgpt"} & local_tokens
        or local_part in EXACT_AI_EMAIL_LOCAL_PARTS
    )


def parsed_trailers(message: str) -> list[tuple[str, str]]:
    output = run_git("interpret-trailers", "--parse", input_text=message)
    trailers: list[tuple[str, str]] = []
    for line in output.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            trailers.append((key.strip(), value.strip()))
    return trailers


def validate_message_trailers(
    message: str,
    *,
    source: str,
) -> list[str]:
    errors: list[str] = []
    for key, value in parsed_trailers(message):
        if forbidden_ai_identity(value):
            errors.append(
                f"{source}: forbidden AI identity in "
                f"{key} trailer: {value}"
            )
    return errors


def validate_identity(
    value: str,
    *,
    field: str,
    source: str,
) -> list[str]:
    if not forbidden_ai_identity(value):
        return []
    return [f"{source}: forbidden AI identity in {field}: {value}"]


def strip_git_ident_timestamp(value: str) -> str:
    return re.sub(r"\s+\d+\s+[+-]\d{4}\s*$", "", value.strip())


def prospective_commit_errors(message_path: Path) -> list[str]:
    message = message_path.read_text(encoding="utf-8", errors="replace")
    author = strip_git_ident_timestamp(run_git("var", "GIT_AUTHOR_IDENT"))
    committer = strip_git_ident_timestamp(run_git("var", "GIT_COMMITTER_IDENT"))
    errors = validate_identity(
        author,
        field="author",
        source="prospective commit",
    )
    errors.extend(
        validate_identity(
            committer,
            field="committer",
            source="prospective commit",
        )
    )
    errors.extend(
        validate_message_trailers(
            message,
            source="prospective commit",
        )
    )
    return errors


def commit_errors(commit_sha: str) -> list[str]:
    output = run_git(
        "show",
        "-s",
        "--format=%an%x00%ae%x00%cn%x00%ce%x00%B",
        commit_sha,
    )
    fields = output.split("\0", 4)
    if len(fields) != 5:
        return [f"{commit_sha}: unable to parse commit attribution"]
    author_name, author_email, committer_name, committer_email, message = fields
    short_sha = commit_sha[:12]
    errors = validate_identity(
        f"{author_name} <{author_email}>",
        field="author",
        source=short_sha,
    )
    errors.extend(
        validate_identity(
            f"{committer_name} <{committer_email}>",
            field="committer",
            source=short_sha,
        )
    )
    errors.extend(
        validate_message_trailers(
            message,
            source=short_sha,
        )
    )
    return errors


def commits_in_range(revision_range: str) -> list[str]:
    return [
        line
        for line in run_git("rev-list", "--reverse", revision_range).splitlines()
        if line
    ]


def new_ref_commits(
    after_sha: str,
    *,
    branch: str,
    remote: str,
) -> list[str]:
    args = ["rev-list", "--reverse", after_sha]
    remote_refs = run_git(
        "for-each-ref",
        "--format=%(refname)",
        f"refs/remotes/{remote}",
    ).splitlines()
    current_remote_ref = f"refs/remotes/{remote}/{branch}"
    excluded = [ref for ref in remote_refs if ref != current_remote_ref]
    if excluded:
        args.extend(["--not", *excluded])
    return [line for line in run_git(*args).splitlines() if line]


def push_commits(
    before_sha: str,
    after_sha: str,
    *,
    branch: str,
    remote: str,
) -> list[str]:
    if ZERO_SHA_RE.fullmatch(after_sha):
        return []
    if ZERO_SHA_RE.fullmatch(before_sha):
        return new_ref_commits(after_sha, branch=branch, remote=remote)
    return commits_in_range(f"{before_sha}..{after_sha}")


def commits_from_push_updates(
    updates_path: Path,
    *,
    remote: str,
) -> list[str]:
    commits: list[str] = []
    seen: set[str] = set()
    for line in updates_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        fields = line.split()
        if len(fields) != 4:
            continue
        local_ref, local_sha, _remote_ref, remote_sha = fields
        branch = local_ref.removeprefix("refs/heads/")
        for commit_sha in push_commits(
            remote_sha,
            local_sha,
            branch=branch,
            remote=remote,
        ):
            if commit_sha not in seen:
                seen.add(commit_sha)
                commits.append(commit_sha)
    return commits


def validate_commits(commits: Iterable[str]) -> tuple[list[str], int]:
    errors: list[str] = []
    count = 0
    for commit_sha in commits:
        count += 1
        errors.extend(commit_errors(commit_sha))
    return errors, count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reject AI assistants in commit authors, committers, and "
            "contributor trailers."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--message-file", type=Path)
    mode.add_argument("--range", dest="revision_range")
    mode.add_argument("--push-updates-file", type=Path)
    mode.add_argument("--push-before")
    parser.add_argument("--push-after")
    parser.add_argument("--branch")
    parser.add_argument("--remote", default="origin")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.message_file is not None:
        errors = prospective_commit_errors(args.message_file)
        count = 1
    elif args.revision_range is not None:
        errors, count = validate_commits(
            commits_in_range(args.revision_range)
        )
    elif args.push_updates_file is not None:
        errors, count = validate_commits(
            commits_from_push_updates(
                args.push_updates_file,
                remote=args.remote,
            )
        )
    else:
        if not args.push_after or not args.branch:
            parser.error(
                "--push-before requires --push-after and --branch"
            )
        errors, count = validate_commits(
            push_commits(
                args.push_before,
                args.push_after,
                branch=args.branch,
                remote=args.remote,
            )
        )

    if errors:
        for error in errors:
            print(f"[COMMIT-ATTRIBUTION] {error}", file=sys.stderr)
        return 1

    print(f"Commit attribution check passed ({count} commit(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
