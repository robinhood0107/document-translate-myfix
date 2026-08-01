"""Shared contracts used by independent OCR strategies."""

from .result_contract import (
    OCR_STRATEGY_MANGALMM_FULL_PAGE,
    OCR_STRATEGY_PADDLE_CROP,
    OCR_STRATEGY_PADDLE_SPOTTING,
)

__all__ = [
    "OCR_STRATEGY_MANGALMM_FULL_PAGE",
    "OCR_STRATEGY_PADDLE_CROP",
    "OCR_STRATEGY_PADDLE_SPOTTING",
]
