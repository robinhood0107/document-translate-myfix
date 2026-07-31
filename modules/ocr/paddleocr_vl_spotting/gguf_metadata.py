"""Compatibility alias for Paddle Spotting GGUF metadata helpers."""

import sys

from ..paddle_spotting import gguf_metadata as _gguf_metadata

sys.modules[__name__] = _gguf_metadata
