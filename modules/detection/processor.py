import logging

import numpy as np

from ..utils.device import resolve_device
from ..utils.mask_inpaint_mode import uses_source_parity_detector
from ..utils.textblock import TextBlock
from ..source_compat import SourceCTDDetector
from .factory import DetectionEngineFactory

logger = logging.getLogger(__name__)


class TextBlockDetector:
    """Detector for finding text blocks in images."""

    def __init__(self, settings_page):
        self.settings = settings_page
        self.detector = 'RT-DETR-v2'
        self.last_engine_name = None
        self.last_device = None
        self.last_mask_details = None
        self.source_parity_detector = None

    def _detect_source_parity(self, img: np.ndarray) -> list[TextBlock]:
        if self.source_parity_detector is None:
            self.source_parity_detector = SourceCTDDetector()
        cfg = self.settings.get_mask_refiner_settings()
        result = self.source_parity_detector.detect(img, cfg)
        self.last_engine_name = 'SourceCTDDetector'
        self.last_device = result.device
        self.last_mask_details = dict(result.mask_details or {})
        self.detector = 'Source Parity CTD'
        logger.info(
            'detection self-check: selected=%s resolved=%s device=%s image_shape=%s blocks=%d',
            self.detector,
            self.last_engine_name,
            self.last_device,
            getattr(img, 'shape', None),
            len(result.blocks or []),
        )
        return result.blocks

    def detect(self, img: np.ndarray) -> list[TextBlock]:
        cfg = self.settings.get_mask_refiner_settings()
        if uses_source_parity_detector(cfg.get('mask_inpaint_mode')):
            return self._detect_source_parity(img)

        self.detector = self.settings.get_tool_selection('detector') or self.detector
        engine = DetectionEngineFactory.create_engine(self.settings, self.detector)
        self.last_engine_name = engine.__class__.__name__
        self.last_device = resolve_device(self.settings.is_gpu_enabled(), 'onnx')
        self.last_mask_details = None
        logger.info(
            'detection self-check: selected=%s resolved=%s device=%s image_shape=%s',
            self.detector,
            self.last_engine_name,
            self.last_device,
            getattr(img, 'shape', None),
        )
        return engine.detect(img)
