"""Compatibility package for the renamed Paddle Spotting strategy.

New code should import from :mod:`modules.ocr.paddle_spotting`.
"""

from ..paddle_spotting.engine import PaddleOCRVLSpottingEngine

__all__ = ["PaddleOCRVLSpottingEngine"]
