"""Compatibility alias for :mod:`modules.ocr.mangalmm_full_page.engine`."""

import sys

from .mangalmm_full_page import engine as _engine

sys.modules[__name__] = _engine
