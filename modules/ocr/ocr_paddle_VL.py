"""Compatibility alias for :mod:`modules.ocr.paddle_crop.engine`."""

import sys

from .paddle_crop import engine as _engine

sys.modules[__name__] = _engine
