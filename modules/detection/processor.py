import logging

import numpy as np

from ..utils.device import resolve_device
from ..utils.textblock import TextBlock
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

    def detect(self, img: np.ndarray) -> list[TextBlock]:
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
