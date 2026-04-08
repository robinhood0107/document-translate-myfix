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

PADDLE_DECORATIVE_NOISE_GLYPHS = frozenset({"⌒", "✺", "︸"})


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
