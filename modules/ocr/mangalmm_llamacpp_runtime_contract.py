"""Compatibility alias for the MangaLMM llama.cpp runtime contract."""

import sys

from .mangalmm_full_page import runtime as _runtime

sys.modules[__name__] = _runtime
