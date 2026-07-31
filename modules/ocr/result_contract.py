"""Compatibility alias for the shared OCR result contract."""

import sys

from .common import result_contract as _result_contract

sys.modules[__name__] = _result_contract
