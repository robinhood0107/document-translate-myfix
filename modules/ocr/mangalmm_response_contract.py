"""Compatibility alias for the MangaLMM full-page response parser."""

import sys

from .mangalmm_full_page import response_parser as _response_parser

sys.modules[__name__] = _response_parser
