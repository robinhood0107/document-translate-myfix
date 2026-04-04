from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .window import ComicTranslateUI

__all__ = ["ComicTranslateUI"]


def __getattr__(name: str):
    if name == "ComicTranslateUI":
        from .window import ComicTranslateUI

        return ComicTranslateUI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
