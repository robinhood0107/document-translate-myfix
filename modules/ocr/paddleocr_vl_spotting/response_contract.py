"""Compatibility alias for the Paddle Spotting response parser."""

import sys

from ..paddle_spotting import response_parser as _response_parser

sys.modules[__name__] = _response_parser
