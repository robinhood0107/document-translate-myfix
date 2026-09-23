from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

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


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class CommitAttributionValidationTests(unittest.TestCase):
    def test_forbidden_ai_identity_variants(self) -> None:
        identities = [
            "Codex <codex@example.com>",
            "OpenAI Codex <assistant@example.com>",
            "codexCodex <assistant@example.com>",
            "ChatGPT <chatgpt@example.com>",
            "GitHub Copilot <copilot@example.com>",
            "Anthropic Claude <claude@example.com>",
            "Google Gemini <gemini@example.com>",
            "Gemini CLI <assistant@example.com>",
            "Claude Bot <assistant@example.com>",
            "Co\u200bdex <assistant@example.com>",
        ]
        for identity in identities:
            with self.subTest(identity=identity):
                self.assertTrue(forbidden_ai_identity(identity))

    def test_human_name_containing_claude_is_allowed(self) -> None:
        self.assertFalse(
            forbidden_ai_identity(
                "Claude Monet <claude.monet@example.com>"
            )
        )
        self.assertFalse(
            forbidden_ai_identity(
                "Cody Codexample <cody@codexample.com>"
            )
        )
        self.assertFalse(
            forbidden_ai_identity(
                "Example Human <human@codex.example>"
            )
        )

    def test_plain_commit_body_can_mention_codex(self) -> None:
        errors = validate_message_trailers(
            "docs(repo): explain attribution policy\n\n"
            "The policy documents the Codex restriction.\n",
            source="test",
        )
        self.assertEqual(errors, [])

    def test_non_contributor_trailer_can_document_codex(self) -> None:
        errors = validate_message_trailers(
            "docs(repo): explain attribution policy\n\n"
            "Policy: Codex identities are forbidden.\n",
            source="test",
        )
        self.assertEqual(errors, [])

    def test_ai_contributor_trailer_is_rejected(self) -> None:
        errors = validate_message_trailers(
            "fix(repo): enforce attribution\n\n"
            "Co-authored-by: codexCodex <assistant@example.com>\n",
            source="test",
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("Co-authored-by", errors[0])

    def test_custom_ai_contributor_trailer_is_rejected(self) -> None:
        errors = validate_message_trailers(
            "fix(repo): enforce attribution\n\n"
            "AI-Assisted-By: Google Gemini <assistant@example.com>\n",
            source="test",
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("AI-Assisted-By", errors[0])

    def test_every_trailer_is_inspected_for_ai_identity(self) -> None:
        errors = validate_message_trailers(
            "fix(repo): enforce attribution\n\n"
            "Pair-programmed-by: Gemini CLI <assistant@example.com>\n",
            source="test",
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("Pair-programmed-by", errors[0])

    def test_prospective_commit_rejects_ai_author(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            message_path = Path(temporary) / "COMMIT_EDITMSG"
            message_path.write_text(
                "fix(repo): reject AI author\n",
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {
                    "GIT_AUTHOR_NAME": "Codex",
                    "GIT_AUTHOR_EMAIL": "assistant@example.com",
                },
            ):
                errors = prospective_commit_errors(message_path)
        self.assertTrue(any("author" in error for error in errors))

    def test_prospective_commit_rejects_ai_committer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            message_path = Path(temporary) / "COMMIT_EDITMSG"
            message_path.write_text(
                "fix(repo): reject AI committer\n",
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {
                    "GIT_COMMITTER_NAME": "Claude Code",
                    "GIT_COMMITTER_EMAIL": "assistant@example.com",
                },
            ):
                errors = prospective_commit_errors(message_path)
        self.assertTrue(any("committer" in error for error in errors))

    def test_commit_range_does_not_rescan_historical_base(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            init_repo(repo)
            historical = commit_file(
                repo,
                message=(
                    "chore(repo): historical commit\n\n"
                    "Co-authored-by: Codex <codex@example.com>"
                ),
                content="historical",
            )
            current = commit_file(
                repo,
                message="fix(repo): current human commit",
                content="current",
            )
            with working_directory(repo):
                commits = commits_in_range(
                    f"{historical}..{current}"
                )
                errors = commit_errors(current)
        self.assertEqual(commits, [current])
        self.assertEqual(errors, [])

    def test_new_branch_push_excludes_existing_remote_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            init_repo(repo)
            historical = commit_file(
                repo,
                message=(
                    "chore(repo): historical commit\n\n"
                    "Co-authored-by: Codex <codex@example.com>"
                ),
                content="historical",
            )
            git(
                repo,
                "update-ref",
                "refs/remotes/origin/develop",
                historical,
            )
            current = commit_file(
                repo,
                message="fix(repo): current human commit",
                content="current",
            )
            with working_directory(repo):
                commits = push_commits(
                    "0" * 40,
                    current,
                    branch="fix/example",
                    remote="origin",
                )
        self.assertEqual(commits, [current])

    def test_lab_based_work_branch_excludes_existing_lab_history(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            init_repo(repo)
            product_base = commit_file(
                repo,
                message="chore(repo): product base",
                content="product",
            )
            git(
                repo,
                "update-ref",
                "refs/remotes/origin/develop",
                product_base,
            )
            lab_base = commit_file(
                repo,
                message="chore(benchmark): existing lab history",
                content="lab",
            )
            git(
                repo,
                "update-ref",
                "refs/remotes/origin/benchmarking/lab",
                lab_base,
            )
            current = commit_file(
                repo,
                message="chore(benchmark): sync current policy",
                content="current",
            )
            with working_directory(repo):
                commits = push_commits(
                    "0" * 40,
                    current,
                    branch="chore/benchmark-lab-sync-policy",
                    remote="origin",
                )
        self.assertEqual(commits, [current])

    def test_existing_branch_push_checks_only_new_commits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            init_repo(repo)
            before = commit_file(
                repo,
                message="chore(repo): base commit",
                content="base",
            )
            after = commit_file(
                repo,
                message="fix(repo): current human commit",
                content="current",
            )
            with working_directory(repo):
                commits = push_commits(
                    before,
                    after,
                    branch="fix/example",
                    remote="origin",
                )
        self.assertEqual(commits, [after])

    def test_force_pushed_before_commit_falls_back_to_new_ref_walk(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            init_repo(repo)
            historical = commit_file(
                repo,
                message=(
                    "chore(repo): historical commit\n\n"
                    "Co-authored-by: Codex <codex@example.com>"
                ),
                content="historical",
            )
            git(
                repo,
                "update-ref",
                "refs/remotes/origin/develop",
                historical,
            )
            current = commit_file(
                repo,
                message="fix(repo): current human commit",
                content="current",
            )
            with working_directory(repo):
                commits = push_commits(
                    "b" * 40,
                    current,
                    branch="fix/example",
                    remote="origin",
                )
        self.assertEqual(commits, [current])

    def test_deleted_ref_has_no_commits(self) -> None:
        commits = push_commits(
            "a" * 40,
            "0" * 40,
            branch="fix/example",
            remote="origin",
        )
        self.assertEqual(commits, [])

    def test_existing_commit_with_ai_trailer_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            init_repo(repo)
            commit_sha = commit_file(
                repo,
                message=(
                    "fix(repo): reject contributor\n\n"
                    "Signed-off-by: OpenAI Codex "
                    "<assistant@example.com>"
                ),
                content="bad",
            )
            with working_directory(repo):
                errors = commit_errors(commit_sha)
        self.assertEqual(len(errors), 1)
        self.assertIn("Signed-off-by", errors[0])

    def test_existing_commit_with_ai_author_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            init_repo(repo)
            (repo / "tracked.txt").write_text(
                "bad",
                encoding="utf-8",
            )
            git(repo, "add", "tracked.txt")
            git(
                repo,
                "commit",
                "-m",
                "fix(repo): reject author",
                env={
                    "GIT_AUTHOR_NAME": "GitHub Copilot",
                    "GIT_AUTHOR_EMAIL": "assistant@example.com",
                },
            )
            commit_sha = git(repo, "rev-parse", "HEAD")
            with working_directory(repo):
                errors = commit_errors(commit_sha)
        self.assertEqual(len(errors), 1)
        self.assertIn("author", errors[0])


if __name__ == "__main__":
    unittest.main()
