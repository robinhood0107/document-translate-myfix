from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable

import cv2
import numpy as np

from modules.detection.utils.slicer import ImageSlicer

from .contracts import CandidateMaskResult, DetectorBox, binary_mask


@dataclass(frozen=True, slots=True)
class CTBDSettings:
    confidence_threshold: float = 0.3
    inpaint_mask_dilate: int = 4
    detect_bubbles: bool = True
    detect_text: bool = True
    text_region_filter: str = "all"


def preprocess_ctbd(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Port BallonsTranslator's CTBD preprocessing without changing semantics."""

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("CTBD expects a BGR image with three channels")
    height, width = image_bgr.shape[:2]
    resized = cv2.resize(image_bgr, (640, 640), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1)
    tensor = np.expand_dims(chw, axis=0).astype(np.float32) / 255.0
    orig_target_sizes = np.array([[width, height]], dtype=np.int64)
    return np.ascontiguousarray(tensor), orig_target_sizes


def _boxes_from_outputs(
    outputs: Iterable[np.ndarray],
    *,
    threshold: float,
    provider: str,
) -> tuple[list[DetectorBox], list[DetectorBox]]:
    labels, boxes, scores = list(outputs)[:3]
    labels = np.asarray(labels)
    boxes = np.asarray(boxes)
    scores = np.asarray(scores)
    if labels.ndim == 2 and labels.shape[0] == 1:
        labels = labels[0]
    if boxes.ndim == 3 and boxes.shape[0] == 1:
        boxes = boxes[0]
    if scores.ndim == 2 and scores.shape[0] == 1:
        scores = scores[0]

    bubbles: list[DetectorBox] = []
    texts: list[DetectorBox] = []
    label_names = {0: "bubble", 1: "text_bubble", 2: "text_free"}
    for label, box, score in zip(labels, boxes, scores):
        confidence = float(score)
        if confidence < float(threshold):
            continue
        label_id = int(label)
        if label_id not in label_names:
            continue
        xyxy = tuple(map(int, box[:4]))
        record = DetectorBox(xyxy, label_names[label_id], confidence, provider)
        if label_id == 0:
            bubbles.append(record)
        else:
            texts.append(record)
    return bubbles, texts


def _detect_content_in_bbox(image_crop: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Exact Ballons CTBD adaptive-threshold content proposal."""

    if image_crop is None or image_crop.size == 0:
        return []
    gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    proposals: list[tuple[int, int, int, int]] = []
    for threshold_type in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            threshold_type,
            11,
            2,
        )
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
        for index in range(1, count):
            if int(stats[index, cv2.CC_STAT_AREA]) <= 10:
                continue
            x = int(stats[index, cv2.CC_STAT_LEFT])
            y = int(stats[index, cv2.CC_STAT_TOP])
            component_width = int(stats[index, cv2.CC_STAT_WIDTH])
            component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
            if (
                x > 0
                and y > 0
                and x + component_width < width
                and y + component_height < height
            ):
                proposals.append((x, y, x + component_width, y + component_height))
    return proposals


def _fits(container: tuple[int, int, int, int], item: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = container
    ix1, iy1, ix2, iy2 = item
    return x1 <= ix1 and y1 <= iy1 and x2 >= ix2 and y2 >= iy2


def _intersection_over_union(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> float:
    x1, y1, x2, y2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(x1, rx1), max(y1, ry1)
    ix2, iy2 = min(x2, rx2), min(y2, ry2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (x2 - x1) * (y2 - y1) + (rx2 - rx1) * (ry2 - ry1) - intersection
    return float(intersection) / float(union) if union > 0 else 0.0


def _build_content_mask(
    image_bgr: np.ndarray,
    text_boxes: Iterable[DetectorBox],
    bubble_boxes: Iterable[DetectorBox],
    settings: CTBDSettings,
) -> tuple[np.ndarray, tuple[DetectorBox, ...]]:
    height, width = image_bgr.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    bubbles = [box.clipped((height, width)) for box in bubble_boxes]
    bubbles = [box for box in bubbles if box is not None]
    accepted: list[DetectorBox] = []

    for original_box in text_boxes:
        box = original_box.clipped((height, width))
        if box is None:
            continue
        x1, y1, x2, y2 = box.xyxy
        if x2 - x1 <= 5 or y2 - y1 <= 5:
            continue
        best_bubble: DetectorBox | None = None
        for bubble in bubbles:
            if _fits(bubble.xyxy, box.xyxy):
                best_bubble = bubble
                break
            if _intersection_over_union(bubble.xyxy, box.xyxy) >= 0.2 and best_bubble is None:
                best_bubble = bubble
        text_class = "text_bubble" if best_bubble is not None else "text_free"
        if settings.text_region_filter not in {"all", text_class}:
            continue

        local = np.zeros_like(mask)
        crop_y1 = max(0, y1 - 10)
        crop_y2 = min(height, y2 + 10)
        crop = image_bgr[crop_y1:crop_y2, x1:x2]
        for px1, py1, px2, py2 in _detect_content_in_bbox(crop):
            cv2.rectangle(
                local,
                (x1 + px1, crop_y1 + py1),
                (x1 + px2, crop_y1 + py2),
                255,
                -1,
            )
        if settings.inpaint_mask_dilate > 0:
            kernel = np.ones(
                (settings.inpaint_mask_dilate, settings.inpaint_mask_dilate),
                dtype=np.uint8,
            )
            local = cv2.dilate(local, kernel, iterations=1)
        mask = cv2.bitwise_or(mask, local)
        accepted.append(DetectorBox(box.xyxy, text_class, box.score, box.provider))
    return binary_mask(mask), tuple(accepted)


class BallonsCTBDReference:
    """Exact Python CTBD preprocessing and content-mask reference adapter."""

    def __init__(self, model_path: str, providers: list[str], settings: CTBDSettings | None = None):
        import onnxruntime as ort

        self.settings = settings or CTBDSettings()
        self.session = ort.InferenceSession(model_path, providers=list(providers))
        self.slicer = ImageSlicer(
            height_to_width_ratio_threshold=3.5,
            target_slice_ratio=3.0,
            overlap_height_ratio=0.2,
            min_slice_height_ratio=0.7,
            merge_iou_threshold=0.2,
            duplicate_iou_threshold=0.5,
            merge_y_distance_threshold=0.1,
            containment_threshold=0.85,
        )

    @property
    def providers(self) -> list[str]:
        return list(self.session.get_providers())

    def _detect_single(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor, original_size = preprocess_ctbd(image_bgr)
        outputs = self.session.run(
            None,
            {"images": tensor, "orig_target_sizes": original_size},
        )
        bubbles, texts = _boxes_from_outputs(
            outputs,
            threshold=self.settings.confidence_threshold,
            provider="ballons_ctbd",
        )
        bubble_array = np.asarray([box.xyxy for box in bubbles], dtype=np.int32)
        text_array = np.asarray([box.xyxy for box in texts], dtype=np.int32)
        if bubble_array.size == 0:
            bubble_array = np.empty((0, 4), dtype=np.int32)
        if text_array.size == 0:
            text_array = np.empty((0, 4), dtype=np.int32)
        return bubble_array, text_array

    def infer(self, image_bgr: np.ndarray) -> CandidateMaskResult:
        start = time.perf_counter()
        bubble_array, text_array = self.slicer.process_slices_for_detection(
            image_bgr,
            self._detect_single,
        )
        bubbles = tuple(
            DetectorBox(tuple(map(int, box)), "bubble", 1.0, "ballons_ctbd")
            for box in np.asarray(bubble_array).reshape(-1, 4)
        )
        texts = tuple(
            DetectorBox(tuple(map(int, box)), "text", 1.0, "ballons_ctbd")
            for box in np.asarray(text_array).reshape(-1, 4)
        )
        content_mask, accepted = _build_content_mask(
            image_bgr,
            texts,
            bubbles,
            self.settings,
        )
        elapsed = time.perf_counter() - start
        return CandidateMaskResult(
            candidate_id="ballons_ctbd",
            raw_mask=content_mask,
            refined_mask=content_mask,
            dilated_mask=content_mask,
            boxes=tuple(bubbles) + tuple(accepted),
            stage_tensors={
                "bubble_boxes": np.asarray([box.xyxy for box in bubbles], dtype=np.int32),
                "text_boxes": np.asarray([box.xyxy for box in accepted], dtype=np.int32),
            },
            runtime={
                "seconds": elapsed,
                "providers": self.providers,
                "reference": "BallonsTranslator CTBD",
            },
        )
