from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from scripts.validate_commit_attribution import (
    commit_errors,
    commits_in_range,
    forbidden_ai_identity,
    prospective_commit_errors,
    push_commits,
    validate_message_trailers,
)


def git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env=process_env,
    )
    return completed.stdout.strip()


def init_repo(repo: Path) -> None:
    git(repo, "init")
    git(repo, "config", "user.name", "Example Human")
    git(repo, "config", "user.email", "human@example.com")


def commit_file(
    repo: Path,
    *,
    message: str,
    content: str,
) -> str:
    (repo / "tracked.txt").write_text(content, encoding="utf-8")
    git(repo, "add", "tracked.txt")
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD")


@pytest.mark.parametrize(
    "identity",
    [
        "Codex <codex@example.com>",
        "OpenAI Codex <assistant@example.com>",
        "codexCodex <assistant@example.com>",
        "ChatGPT <chatgpt@example.com>",
        "GitHub Copilot <copilot@example.com>",
        "Anthropic Claude <claude@example.com>",
        "Google Gemini <gemini@example.com>",
    ],
)
def test_forbidden_ai_identity_variants(identity: str) -> None:
    assert forbidden_ai_identity(identity)


def test_human_name_containing_claude_is_allowed() -> None:
    assert not forbidden_ai_identity(
        "Claude Monet <claude.monet@example.com>"
    )
    assert not forbidden_ai_identity(
        "Cody Codexample <cody@codexample.com>"
    )


def test_plain_commit_body_can_mention_codex() -> None:
    errors = validate_message_trailers(
        "docs(repo): explain attribution policy\n\n"
        "The policy documents the Codex restriction.\n",
        source="test",
    )
    assert errors == []


def test_non_contributor_trailer_can_document_codex() -> None:
    errors = validate_message_trailers(
        "docs(repo): explain attribution policy\n\n"
        "Policy: Codex identities are forbidden.\n",
        source="test",
    )
    assert errors == []


def test_ai_contributor_trailer_is_rejected() -> None:
    errors = validate_message_trailers(
        "fix(repo): enforce attribution\n\n"
        "Co-authored-by: codexCodex <assistant@example.com>\n",
        source="test",
    )
    assert len(errors) == 1
    assert "Co-authored-by" in errors[0]


def test_custom_ai_contributor_trailer_is_rejected() -> None:
    errors = validate_message_trailers(
        "fix(repo): enforce attribution\n\n"
        "AI-Assisted-By: Google Gemini <assistant@example.com>\n",
        source="test",
    )
    assert len(errors) == 1
    assert "AI-Assisted-By" in errors[0]


def test_prospective_commit_rejects_ai_author(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message_path = tmp_path / "COMMIT_EDITMSG"
    message_path.write_text(
        "fix(repo): reject AI author\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Codex")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "assistant@example.com")

    errors = prospective_commit_errors(message_path)

    assert any("author" in error for error in errors)


def test_prospective_commit_rejects_ai_committer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message_path = tmp_path / "COMMIT_EDITMSG"
    message_path.write_text(
        "fix(repo): reject AI committer\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Claude Code")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "assistant@example.com")

    errors = prospective_commit_errors(message_path)

    assert any("committer" in error for error in errors)


def test_commit_range_does_not_rescan_historical_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_repo(tmp_path)
    historical = commit_file(
        tmp_path,
        message=(
            "chore(repo): historical commit\n\n"
            "Co-authored-by: Codex <codex@example.com>"
        ),
        content="historical",
    )
    current = commit_file(
        tmp_path,
        message="fix(repo): current human commit",
        content="current",
    )
    monkeypatch.chdir(tmp_path)

    commits = commits_in_range(f"{historical}..{current}")

    assert commits == [current]
    assert commit_errors(current) == []


def test_new_branch_push_uses_develop_as_history_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_repo(tmp_path)
    historical = commit_file(
        tmp_path,
        message=(
            "chore(repo): historical commit\n\n"
            "Co-authored-by: Codex <codex@example.com>"
        ),
        content="historical",
    )
    git(tmp_path, "update-ref", "refs/remotes/origin/develop", historical)
    current = commit_file(
        tmp_path,
        message="fix(repo): current human commit",
        content="current",
    )
    monkeypatch.chdir(tmp_path)

    commits = push_commits(
        "0" * 40,
        current,
        branch="fix/example",
        remote="origin",
    )

    assert commits == [current]


def test_existing_branch_push_checks_only_new_commits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_repo(tmp_path)
    before = commit_file(
        tmp_path,
        message="chore(repo): base commit",
        content="base",
    )
    after = commit_file(
        tmp_path,
        message="fix(repo): current human commit",
        content="current",
    )
    monkeypatch.chdir(tmp_path)

    commits = push_commits(
        before,
        after,
        branch="fix/example",
        remote="origin",
    )

    assert commits == [after]


def test_deleted_ref_has_no_commits() -> None:
    commits = push_commits(
        "a" * 40,
        "0" * 40,
        branch="fix/example",
        remote="origin",
    )

    assert commits == []


def test_existing_commit_with_ai_trailer_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_repo(tmp_path)
    commit_sha = commit_file(
        tmp_path,
        message=(
            "fix(repo): reject contributor\n\n"
            "Signed-off-by: OpenAI Codex <assistant@example.com>"
        ),
        content="bad",
    )
    monkeypatch.chdir(tmp_path)

    errors = commit_errors(commit_sha)

    assert len(errors) == 1
    assert "Signed-off-by" in errors[0]


def test_existing_commit_with_ai_author_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_repo(tmp_path)
    (tmp_path / "tracked.txt").write_text("bad", encoding="utf-8")
    git(tmp_path, "add", "tracked.txt")
    git(
        tmp_path,
        "commit",
        "-m",
        "fix(repo): reject author",
        env={
            "GIT_AUTHOR_NAME": "GitHub Copilot",
            "GIT_AUTHOR_EMAIL": "assistant@example.com",
        },
    )
    commit_sha = git(tmp_path, "rev-parse", "HEAD")
    monkeypatch.chdir(tmp_path)

    errors = commit_errors(commit_sha)

    assert len(errors) == 1
    assert "author" in errors[0]
