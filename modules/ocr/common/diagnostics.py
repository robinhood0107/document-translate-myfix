"""Versioned strategy-neutral OCR diagnostics containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


OCR_STRATEGY_DIAGNOSTICS_SCHEMA_VERSION = 1


@dataclass(slots=True)
class OCRStrategyDiagnostics:
    strategy: str
    parser_errors: int = 0
    length_failures: int = 0
    retry_count: int = 0
    merge_split_reviews: int = 0
    coverage: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": OCR_STRATEGY_DIAGNOSTICS_SCHEMA_VERSION,
            "strategy": str(self.strategy),
            "parser_errors": int(self.parser_errors),
            "length_failures": int(self.length_failures),
            "retry_count": int(self.retry_count),
            "merge_split_reviews": int(self.merge_split_reviews),
            "coverage": dict(self.coverage),
        }


__all__ = [
    "OCR_STRATEGY_DIAGNOSTICS_SCHEMA_VERSION",
    "OCRStrategyDiagnostics",
]
