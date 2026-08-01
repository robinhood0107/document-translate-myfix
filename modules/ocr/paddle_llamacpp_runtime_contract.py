"""Compatibility alias for the Paddle crop llama.cpp runtime contract."""

import sys

from .paddle_crop import runtime as _runtime

sys.modules[__name__] = _runtime
