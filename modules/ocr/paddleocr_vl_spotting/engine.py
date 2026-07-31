"""Compatibility alias for Paddle full-page Spotting."""

import sys

from ..paddle_spotting import engine as _engine

sys.modules[__name__] = _engine
