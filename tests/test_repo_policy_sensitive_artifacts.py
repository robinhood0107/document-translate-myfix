from __future__ import annotations

from scripts.validate_repo_policy import (
    scan_sensitive_content,
    validate_tracked_path_name,
)


def test_repo_policy_rejects_private_artifact_paths() -> None:
    assert validate_tracked_path_name("Sample/japan/001.png")
    assert validate_tracked_path_name("testmodel/model.gguf")
    assert validate_tracked_path_name("banchmark_result_log/run/report.json")
    assert validate_tracked_path_name("docs/assets/benchmarking/run/chart.png")
    assert validate_tracked_path_name("result_private_title/001.png")
    assert validate_tracked_path_name("log_private_title/output.log")


def test_repo_policy_allows_static_app_media() -> None:
    assert not validate_tracked_path_name("resources/static/icon-loading.gif")
    assert not validate_tracked_path_name("resources/icons/splash.png")


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
