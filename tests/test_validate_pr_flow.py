from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_pr_flow", ROOT / "scripts" / "validate_pr_flow.py"
)
assert SPEC is not None
validate_pr_flow_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(validate_pr_flow_module)


def validate_pr_flow(head: str, base: str) -> list[str]:
    return validate_pr_flow_module.validate_pr_flow(head, base)


def test_main_promotion_branch_may_target_main() -> None:
    assert validate_pr_flow("chore/main-promotion-2026-06-02", "main") == []


def test_main_promotion_branch_must_not_target_develop() -> None:
    errors = validate_pr_flow("chore/main-promotion-2026-06-02", "develop")

    assert errors == ["chore/main-promotion-2026-06-02 must target main, not develop."]


def test_regular_chore_branch_still_targets_develop() -> None:
    assert validate_pr_flow("chore/nuitka-release-policy", "develop") == []
    assert validate_pr_flow("chore/nuitka-release-policy", "main") == [
        "chore/nuitka-release-policy must target develop, not main."
    ]
