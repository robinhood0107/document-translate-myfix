"""Compatibility alias for the Paddle Spotting runtime contract."""

import sys

from ..paddle_spotting import runtime as _runtime

sys.modules[__name__] = _runtime
