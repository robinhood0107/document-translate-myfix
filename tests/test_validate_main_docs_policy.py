from scripts.validate_main_docs_policy import is_allowed


def test_main_docs_policy_allows_agent_instructions() -> None:
    assert is_allowed("AGENTS.md")
    assert is_allowed("CLAUDE.md")


def test_main_docs_policy_allows_active_runtime_operations_docs() -> None:
    expected_paths = (
        "docs/architecture/codebase-map-ko.md",
        "docs/runtime/managed-llamacpp-only-ko.md",
        "docs/runtime/obsolete-vllm-runtime-manifest.json",
        "mangalmm_docker_files/README.md",
        "paddleocr_vl_spotting_docker_files/README.md",
    )

    for path in expected_paths:
        assert is_allowed(path)


def test_main_docs_policy_still_rejects_unlisted_root_markdown() -> None:
    assert not is_allowed("internal-dev-notes.md")


def test_main_docs_policy_still_rejects_unlisted_runtime_docs() -> None:
    assert not is_allowed("docs/runtime/internal-experiment-ko.md")
