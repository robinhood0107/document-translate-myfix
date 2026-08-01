"""Compatibility alias for Paddle Spotting reconciliation."""

import sys

from ..paddle_spotting import reconciliation as _reconciliation

sys.modules[__name__] = _reconciliation
