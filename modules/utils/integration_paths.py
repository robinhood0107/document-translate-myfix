from __future__ import annotations

from pathlib import Path

from .paths import get_repo_root


def get_integration_root() -> Path:
    return Path(get_repo_root()) / "이식"


def get_integration_docs_root() -> Path:
    return get_integration_root() / "docs"


def get_integration_source_reference_root() -> Path:
    return get_integration_root() / "source_reference"
