from scripts.validate_instruction_sync import validate_changed_paths


def test_instruction_harness_accepts_all_three_files() -> None:
    assert validate_changed_paths(["AGENTS.md", "CLAUDE.md", "rules.md"]) == []


def test_instruction_harness_rejects_partial_update() -> None:
    errors = validate_changed_paths(["AGENTS.md", "rules.md"])

    assert errors == [
        "Instruction-harness updates must change AGENTS.md, CLAUDE.md, and rules.md together. "
        "Changed: AGENTS.md, rules.md. Missing: CLAUDE.md."
    ]


def test_instruction_harness_ignores_unrelated_change() -> None:
    assert validate_changed_paths(["modules/ocr/runtime.py"]) == []
