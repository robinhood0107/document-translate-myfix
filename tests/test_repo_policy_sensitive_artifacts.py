from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from scripts import validate_repo_policy as repo_policy
from scripts.validate_repo_policy import (
    scan_sensitive_content,
    validate_tracked_path_name,
)


def test_repo_policy_rejects_private_artifact_paths() -> None:
    assert validate_tracked_path_name("Sample/japan/001.png")
    assert validate_tracked_path_name("testmodel/model.gguf")
    assert validate_tracked_path_name("banchmark_result_log/run/report.json")
    assert validate_tracked_path_name("benchmark_result_log/run/report.json")
    assert validate_tracked_path_name(".gstack/security-reports/latest.json")
    assert validate_tracked_path_name("docs/assets/benchmarking/run/chart.png")
    assert validate_tracked_path_name("result_private_title/001.png")
    assert validate_tracked_path_name("log_private_title/output.log")


def test_repo_policy_allows_static_app_media() -> None:
    assert not validate_tracked_path_name("resources/static/icon-loading.gif")
    assert not validate_tracked_path_name("resources/icons/splash.png")


def test_repo_policy_rejects_local_credentials_but_allows_sanitized_example() -> None:
    assert validate_tracked_path_name(".env.local")
    assert validate_tracked_path_name("runtime/private.pem")
    assert not validate_tracked_path_name(".env.example")


def test_repo_policy_rejects_private_content_strings() -> None:
    errors = scan_sensitive_content(
        "tests/example.py",
        "\n".join(
            [
                "C:" + r"\Users" + r"\pjjpj" + r"\Desktop\project",
                "False_" + "Honour_8_Part_3_English",
                "我的" + "妈妈被" + "损" + "友穿上了",
                "banchmark_result_log/family/" + "20260415_010203_run/report.json",
            ]
        ),
    )

    assert len(errors) == 3


def test_repo_policy_allows_neutral_fixture_content() -> None:
    assert not scan_sensitive_content(
        "tests/example.py",
        "\n".join(
            [
                r"C:\ExampleWorkspace\project",
                "example_source_chapter",
                "<benchmark-log-root>/family/<run-id>/report.json",
            ]
        ),
    )


class RepoPolicyUnicodePathTests(unittest.TestCase):
    def test_tracked_paths_preserve_unicode_and_embedded_newlines(self) -> None:
        expected = [
            "docs/한글 경로.md",
            "benchmarks-fonts/Korean/글꼴.ttf",
            "docs/line\nbreak.md",
        ]
        completed = subprocess.CompletedProcess(
            args=["git", "ls-files", "-z"],
            returncode=0,
            stdout=b"\0".join(path.encode("utf-8") for path in expected) + b"\0",
            stderr=b"",
        )

        with mock.patch.object(
            repo_policy.subprocess,
            "run",
            return_value=completed,
        ):
            self.assertEqual(repo_policy.tracked_paths(), expected)

    def test_unicode_paths_cannot_bypass_path_or_content_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            document = Path(temp_dir) / "검사 문서.md"
            document.write_text(
                "banchmark_result_log/family/"
                + "20260729_010203_run/report.json",
                encoding="utf-8",
            )
            tracked = [
                str(document),
                "benchmarks-fonts/Korean/글꼴.ttf",
            ]

            with mock.patch.object(
                repo_policy,
                "tracked_paths",
                return_value=tracked,
            ):
                path_errors = repo_policy.validate_tracked_paths()
                content_errors = repo_policy.validate_sensitive_content()

        self.assertTrue(
            any("글꼴.ttf" in error for error in path_errors),
            path_errors,
        )
        self.assertTrue(
            any("concrete benchmark output path" in error for error in content_errors),
            content_errors,
        )


class RepoPolicyBenchmarkBranchTests(unittest.TestCase):
    def test_benchmark_assets_require_lab_branch_or_lab_pr_base(self) -> None:
        with mock.patch.object(
            repo_policy,
            "tracked_paths",
            return_value=["benchmarks/example/protocol.json"],
        ):
            self.assertTrue(
                repo_policy.validate_benchmark_asset_placement("feature/runtime")
            )
            self.assertFalse(
                repo_policy.validate_benchmark_asset_placement("benchmarking/lab")
            )
            self.assertFalse(
                repo_policy.validate_benchmark_asset_placement(
                    "chore/benchmark-example",
                )
            )
            self.assertFalse(
                repo_policy.validate_benchmark_asset_placement(
                    "feature/runtime",
                    "benchmarking/lab",
                )
            )

    def test_main_forwards_base_branch_to_benchmark_policy(self) -> None:
        argv = [
            "validate_repo_policy.py",
            "--mode",
            "ci",
            "--branch",
            "feature/runtime",
            "--base-branch",
            "benchmarking/lab",
        ]
        with (
            mock.patch.object(repo_policy.sys, "argv", argv),
            mock.patch.object(
                repo_policy,
                "tracked_paths",
                return_value=["benchmarks/example/protocol.json"],
            ),
            mock.patch.object(
                repo_policy,
                "validate_sensitive_content",
                return_value=[],
            ),
        ):
            self.assertEqual(repo_policy.main(), 0)
