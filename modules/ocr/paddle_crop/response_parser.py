"""Response parsing and text normalization for Paddle crop OCR."""

from __future__ import annotations

import re
from typing import Any, Callable

from modules.utils.text_normalization import (
    DECORATIVE_NOISE_GLYPHS,
    normalize_decorative_ocr_text,
)


def normalize_output_text(text: str) -> str:
    if not text:
        return ""
    normalized = normalize_decorative_ocr_text(
        text,
        glyphs=DECORATIVE_NOISE_GLYPHS,
    )
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def markdown_to_text(
    text: str,
    *,
    text_normalizer: Callable[[str], str] = normalize_output_text,
) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", text)
    cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"(\*\*|__)(.*?)\1", r"\2", cleaned)
    cleaned = re.sub(r"(\*|_)(.*?)\1", r"\2", cleaned)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    return text_normalizer(cleaned)


def extract_texts_from_pruned(node: Any) -> list[str]:
    texts: list[str] = []

    def walk(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            dict_text = value.get("text")
            dict_texts = value.get("texts")
            if isinstance(dict_text, str):
                cleaned = dict_text.strip()
                if cleaned:
                    texts.append(cleaned)
            if isinstance(dict_texts, list):
                combined = "".join(str(part) for part in dict_texts).strip()
                if combined:
                    texts.append(combined)
            elif isinstance(dict_texts, str):
                cleaned = dict_texts.strip()
                if cleaned:
                    texts.append(cleaned)
            for child in value.values():
                walk(child)
            return
        if isinstance(value, list):
            for child in value:
                walk(child)
            return
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                texts.append(cleaned)

    walk(node)
    return texts


def extract_text_from_layout_item(
    item: Any,
    *,
    markdown_converter: Callable[[str], str] = markdown_to_text,
    pruned_extractor: Callable[[Any], list[str]] = extract_texts_from_pruned,
    text_normalizer: Callable[[str], str] = normalize_output_text,
) -> str:
    if not isinstance(item, dict):
        return ""
    markdown = item.get("markdown")
    if isinstance(markdown, dict):
        markdown_text = markdown_converter(markdown.get("text", ""))
        if markdown_text:
            return markdown_text
    pruned = item.get("prunedResult")
    if pruned is not None:
        extracted = pruned_extractor(pruned)
        if extracted:
            return text_normalizer("\n".join(extracted))
    raw_text = item.get("text")
    if isinstance(raw_text, str):
        return text_normalizer(raw_text)
    return ""


def extract_text_from_response(
    data: dict[str, Any],
    *,
    item_extractor: Callable[[Any], str] = extract_text_from_layout_item,
) -> str:
    result = data.get("result", data)
    if not isinstance(result, dict):
        return ""
    layout_results = result.get("layoutParsingResults")
    if isinstance(layout_results, list):
        for item in layout_results:
            text = item_extractor(item)
            if text:
                return text
    return item_extractor(result)


__all__ = [
    "extract_text_from_layout_item",
    "extract_text_from_response",
    "extract_texts_from_pruned",
    "markdown_to_text",
    "normalize_output_text",
]
