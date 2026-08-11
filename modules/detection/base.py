from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from ..utils.textblock import TextBlock
from .utils.geometry import does_rectangle_fit, do_rectangles_overlap, \
    merge_overlapping_boxes
from .font.engine import FontEngineFactory
from .utils.bubble_text_rescue import detect_bubble_text_rescue_boxes
from .utils.content import filter_and_fix_bboxes


class DetectionEngine(ABC):
    """
    Abstract base class for all detection engines.
    Each model implementation should inherit from this class.
    """
    
    def __init__(self, settings=None):
        self.settings = settings
    
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """
        Initialize the detection model with necessary parameters.
        
        Args:
            **kwargs: Engine-specific initialization parameters
        """
        pass
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> list[TextBlock]:
        """
        Detect text blocks in an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of TextBlock objects with detected regions
        """
        pass
        
    def create_text_blocks(
        self, 
        image: np.ndarray, 
        text_boxes: np.ndarray,
        bubble_boxes: Optional[np.ndarray] = None
    ) -> list[TextBlock]:
        
        text_boxes = filter_and_fix_bboxes(text_boxes, image.shape)
        bubble_boxes = filter_and_fix_bboxes(bubble_boxes, image.shape)
        text_boxes = merge_overlapping_boxes(text_boxes)

        text_blocks = []
        text_matched = [False] * len(text_boxes)  # Track matched text boxes
        matched_bubble_indices = set()
        
        # Set bubble_boxes to empty array if None
        if bubble_boxes is None:
            bubble_boxes = np.array([])
        
        # Process text boxes
        if len(text_boxes) > 0:
            for txt_idx, txt_box in enumerate(text_boxes):
                font_attrs = {}
                # Calculate font attributes using FontEngine
                try:
                    x1, y1, x2, y2 = map(int, txt_box)
                    # Ensure coordinates are within image bounds
                    h, w = image.shape[:2]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        crop = image[y1:y2, x1:x2]
                        font_engine = FontEngineFactory.create_engine(self.settings, backend='onnx')
                        font_attrs = font_engine.process(crop)
                except Exception as e:
                    print(f"Failed to detect font attributes for text block {txt_idx}: {e}")

                direction = font_attrs.get('direction', '')
                text_color = tuple(font_attrs.get('text_color', ()))

                # If no bubble boxes, all text is free text
                if len(bubble_boxes) == 0:
                    text_blocks.append(
                        TextBlock(
                            text_bbox=txt_box,
                            text_class='text_free',
                            detector_origin='direct_text',
                            detector_text_bbox=np.asarray(txt_box, dtype=np.int32),
                            detector_provider=type(self).__name__,
                            direction=direction,
                            font_color=text_color,
                        )
                    )
                    continue
                
                for bubble_idx, bble_box in enumerate(bubble_boxes):
                    if bble_box is None:
                        continue
                    if does_rectangle_fit(bble_box, txt_box):
                        # Text is inside a bubble
                        text_blocks.append(
                            TextBlock(
                                text_bbox=txt_box,
                                bubble_bbox=bble_box,
                                text_class='text_bubble',
                                detector_origin='direct_text',
                                detector_text_bbox=np.asarray(txt_box, dtype=np.int32),
                                detector_provider=type(self).__name__,
                                direction=direction,
                                font_color=text_color,
                            )
                        )
                        text_matched[txt_idx] = True  
                        matched_bubble_indices.add(bubble_idx)
                        break
                    elif do_rectangles_overlap(bble_box, txt_box):
                        # Text overlaps with a bubble
                        text_blocks.append(
                            TextBlock(
                                text_bbox=txt_box,
                                bubble_bbox=bble_box,
                                text_class='text_bubble',
                                detector_origin='direct_text',
                                detector_text_bbox=np.asarray(txt_box, dtype=np.int32),
                                detector_provider=type(self).__name__,
                                direction=direction,
                                font_color=text_color,
                            )
                        )
                        text_matched[txt_idx] = True  
                        matched_bubble_indices.add(bubble_idx)
                        break
                
                if not text_matched[txt_idx]:
                    text_blocks.append(
                        TextBlock(
                            text_bbox=txt_box,
                            text_class='text_free',
                            detector_origin='direct_text',
                            detector_text_bbox=np.asarray(txt_box, dtype=np.int32),
                            detector_provider=type(self).__name__,
                            direction=direction,
                            font_color=text_color,
                        )
                    )

        unmatched_bubbles = [
            bubble_box
            for bubble_idx, bubble_box in enumerate(bubble_boxes)
            if bubble_idx not in matched_bubble_indices
        ]
        for text_box, bubble_box in detect_bubble_text_rescue_boxes(image, unmatched_bubbles, text_boxes):
            x1, y1, x2, y2 = [int(v) for v in text_box]
            direction = 'vertical' if (y2 - y1) >= (x2 - x1) * 1.15 else 'horizontal'
            text_blocks.append(
                TextBlock(
                    text_bbox=np.asarray(text_box, dtype=np.int32),
                    bubble_bbox=np.asarray(bubble_box, dtype=np.int32),
                    text_class='text_bubble',
                    detector_origin='bubble_text_rescue',
                    detector_text_bbox=None,
                    detector_provider=type(self).__name__,
                    direction=direction,
                )
            )
        
        return text_blocks
    
