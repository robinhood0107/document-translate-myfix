from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .contracts import CandidateMaskResult, DetectorBox


TEXT_CLASS_ID = 1
BALLOON_CLASS_ID = 2


def _as_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    current = value
    for method in ("detach", "cpu"):
        callback = getattr(current, method, None)
        if callable(callback):
            current = callback()
    numpy_callback = getattr(current, "numpy", None)
    if callable(numpy_callback):
        current = numpy_callback()
    return np.asarray(current)


def class_mask_from_result(
    result: Any,
    shape: tuple[int, int],
    *,
    class_id: int,
    label: str,
    provider: str,
) -> tuple[np.ndarray, tuple[DetectorBox, ...]]:
    """Union one YOLO instance class without turning boxes into mask pixels."""

    ownership = np.zeros(shape, dtype=np.uint8)
    boxes_object = getattr(result, "boxes", None)
    masks_object = getattr(result, "masks", None)
    if boxes_object is None or masks_object is None:
        return ownership, ()

    classes = _as_numpy(getattr(boxes_object, "cls", None)).reshape(-1)
    confidences = _as_numpy(getattr(boxes_object, "conf", None)).reshape(-1)
    coordinates = _as_numpy(getattr(boxes_object, "xyxy", None)).reshape(-1, 4)
    masks = _as_numpy(getattr(masks_object, "data", None))
    if masks.ndim != 3:
        return ownership, ()
    count = min(len(classes), len(confidences), len(coordinates), len(masks))
    records: list[DetectorBox] = []
    for index in range(count):
        if int(classes[index]) != int(class_id):
            continue
        local = masks[index]
        if tuple(local.shape) != tuple(shape):
            local = cv2.resize(
                local.astype(np.float32),
                (shape[1], shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        ownership[local > 0.5] = 255
        x1, y1, x2, y2 = coordinates[index]
        record = DetectorBox(
            (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
            str(label),
            float(confidences[index]),
            provider,
        ).clipped(shape)
        if record is not None:
            records.append(record)
    return np.ascontiguousarray(ownership), tuple(records)


def text_ownership_from_result(
    result: Any,
    shape: tuple[int, int],
    *,
    provider: str,
) -> tuple[np.ndarray, tuple[DetectorBox, ...]]:
    """Union only YOLO text instances without turning a box into claim pixels."""

    return class_mask_from_result(
        result,
        shape,
        class_id=TEXT_CLASS_ID,
        label="text",
        provider=provider,
    )


def balloon_silhouette_from_result(
    result: Any,
    shape: tuple[int, int],
    *,
    provider: str,
) -> tuple[np.ndarray, tuple[DetectorBox, ...]]:
    """Union only the model's class-2 balloon instance pixels."""

    return class_mask_from_result(
        result,
        shape,
        class_id=BALLOON_CLASS_ID,
        label="balloon",
        provider=provider,
    )


@dataclass(frozen=True, slots=True)
class Manga109YOLO26Settings:
    image_size: int = 1280
    confidence: float = 0.25
    iou: float = 0.7
    device: str = "cpu"


class Manga109YOLO26OwnershipReference:
    """Pinned Python reference for the Manga109 YOLO26 segmentation model."""

    def __init__(
        self,
        model_path: str | Path,
        settings: Manga109YOLO26Settings | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        self.settings = settings or Manga109YOLO26Settings()
        self.model = YOLO(str(self.model_path))

    def infer(self, image_bgr: np.ndarray) -> CandidateMaskResult:
        text, _balloon = self.infer_evidence(image_bgr)
        return text

    def infer_evidence(
        self,
        image_bgr: np.ndarray,
    ) -> tuple[CandidateMaskResult, CandidateMaskResult]:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Manga109 YOLO26 expects a BGR image")
        result = self.model.predict(
            source=image_bgr,
            imgsz=int(self.settings.image_size),
            conf=float(self.settings.confidence),
            iou=float(self.settings.iou),
            retina_masks=True,
            device=self.settings.device,
            verbose=False,
        )[0]
        provider = f"ultralytics-{self.settings.device}"
        ownership, boxes = text_ownership_from_result(
            result,
            image_bgr.shape[:2],
            provider=provider,
        )
        balloon, balloon_boxes = balloon_silhouette_from_result(
            result,
            image_bgr.shape[:2],
            provider=provider,
        )
        common_runtime = {
            "provider": provider,
            "image_size": int(self.settings.image_size),
            "confidence": float(self.settings.confidence),
            "iou": float(self.settings.iou),
            "retina_masks": True,
        }
        text_result = CandidateMaskResult(
            "manga109-yolo26-text-ownership",
            ownership,
            ownership,
            ownership,
            boxes=boxes,
            runtime={
                **common_runtime,
                "text_instance_count": len(boxes),
            },
        )
        balloon_result = CandidateMaskResult(
            "manga109-yolo26-balloon-silhouette",
            balloon,
            balloon,
            balloon,
            boxes=balloon_boxes,
            runtime={
                **common_runtime,
                "balloon_instance_count": len(balloon_boxes),
            },
        )
        return text_result, balloon_result
