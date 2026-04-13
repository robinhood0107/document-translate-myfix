from __future__ import annotations

import logging
import re
from typing import Iterable

logger = logging.getLogger(__name__)


def normalize_substitution_rules(rules: Iterable[dict] | None) -> list[dict]:
    normalized: list[dict] = []
    for rule in rules or []:
        if not isinstance(rule, dict):
            continue
        normalized.append(
            {
                "keyword": str(rule.get("keyword", "") or ""),
                "sub": str(rule.get("sub", "") or ""),
                "use_reg": bool(rule.get("use_reg", False)),
                "case_sens": bool(rule.get("case_sens", True)),
            }
        )
    return normalized


def apply_substitution_rules(text: str | None, rules: Iterable[dict] | None) -> str:
    value = "" if text is None else str(text)
    for index, rule in enumerate(normalize_substitution_rules(rules)):
        keyword = rule["keyword"]
        if not keyword:
            continue

        pattern = keyword if rule["use_reg"] else re.escape(keyword)
        flags = re.DOTALL
        if not rule["case_sens"]:
            flags |= re.IGNORECASE

        try:
            value = re.sub(pattern, rule["sub"], value, flags=flags)
        except Exception:
            logger.exception(
                "Invalid correction dictionary rule at index %d: %s",
                index,
                pattern,
            )
            continue
    return value


def apply_ocr_result_dictionary(blocks, rules: Iterable[dict] | None) -> None:
    for block in blocks or []:
        block.text = apply_substitution_rules(getattr(block, "text", ""), rules)


def apply_translation_result_dictionary(blocks, rules: Iterable[dict] | None) -> None:
    for block in blocks or []:
        block.translation = apply_substitution_rules(
            getattr(block, "translation", ""),
            rules,
        )
