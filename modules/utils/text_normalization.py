from __future__ import annotations

import re
from typing import Iterable


_ELLIPSIS_RUN_RE = re.compile(r"(?:[…⋯]+|[.．・･]{3,})")
_INVISIBLE_CHAR_TRANSLATION = str.maketrans({
    "\u200b": "",
    "\u200c": "",
    "\u200d": "",
    "\ufeff": "",
})

OCR_DECORATIVE_NOISE_GLYPHS = frozenset({"⌒", "✺", "︸"})
RENDER_NORMALIZABLE_GLYPHS = frozenset({"「", "」", "『", "』", "♥", "♡", "❤"})

# Backward-compatible aliases for existing callers.
DECORATIVE_NOISE_GLYPHS = OCR_DECORATIVE_NOISE_GLYPHS
# Backward-compatible alias for existing callers.
PADDLE_DECORATIVE_NOISE_GLYPHS = OCR_DECORATIVE_NOISE_GLYPHS


def remove_invisible_format_chars(text: str) -> str:
    if not text:
        return ""
    return text.translate(_INVISIBLE_CHAR_TRANSLATION)


def canonicalize_ellipsis_runs(text: str) -> str:
    if not text:
        return ""
    return _ELLIPSIS_RUN_RE.sub("...", text)


def strip_selected_glyphs(text: str, glyphs: Iterable[str]) -> str:
    if not text:
        return ""
    drop = set(glyphs)
    if not drop:
        return text
    return "".join(ch for ch in text if ch not in drop)


def normalize_decorative_ocr_text(
    text: str,
    *,
    glyphs: Iterable[str] | None = None,
) -> str:
    if not text:
        return ""
    normalized = remove_invisible_format_chars(str(text or ""))
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized = strip_selected_glyphs(
        normalized,
        OCR_DECORATIVE_NOISE_GLYPHS if glyphs is None else glyphs,
    )
    normalized = canonicalize_ellipsis_runs(normalized)
    return normalized.strip()
