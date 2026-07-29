from scripts.validate_main_docs_policy import is_allowed


def test_main_docs_policy_allows_agent_instructions() -> None:
    assert is_allowed("AGENTS.md")


def test_main_docs_policy_still_rejects_unlisted_root_markdown() -> None:
    assert not is_allowed("internal-dev-notes.md")
