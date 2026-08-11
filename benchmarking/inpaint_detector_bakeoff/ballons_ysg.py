from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np

from .contracts import CandidateMaskResult, DetectorBox, binary_mask


DEFAULT_LABELS = frozenset(
    {"balloon", "qipao", "shuqing", "changfangtiao", "hengxie"}
)


@dataclass(frozen=True, slots=True)
class YSGSettings:
    confidence_threshold: float = 0.3
    iou_threshold: float = 0.5
    mask_dilate_size: int = 2
    valid_labels: frozenset[str] = DEFAULT_LABELS


def mask_and_boxes_from_result(
    result,
    shape: tuple[int, int],
    settings: YSGSettings,
) -> tuple[np.ndarray, tuple[DetectorBox, ...]]:
    """Port Ballons YSG box/OBB mask construction without changing semantics."""

    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    names = dict(result.names)
    valid_ids = {index for index, name in names.items() if name in settings.valid_labels}
    records: list[DetectorBox] = []

    detections = getattr(result, "boxes", None)
    if detections is not None and len(detections.cls) > 0:
        for index in range(len(detections.cls)):
            class_index = int(detections.cls[index])
            if class_index not in valid_ids:
                continue
            x1, y1, x2, y2 = detections.xyxy[index].cpu().numpy().astype(int)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            records.append(
                DetectorBox(
                    (int(x1), int(y1), int(x2), int(y2)),
                    str(names[class_index]),
                    1.0,
                    "ballons_ysg",
                )
            )

    oriented = getattr(result, "obb", None)
    if oriented is not None and len(oriented.cls) > 0:
        for index in range(len(oriented.cls)):
            class_index = int(oriented.cls[index])
            if class_index not in valid_ids:
                continue
            points = oriented.xyxyxyxy[index].cpu().numpy().astype(int)
            cv2.fillPoly(mask, [points], 255)
            x1, y1 = points.min(axis=0)
            x2, y2 = points.max(axis=0)
            records.append(
                DetectorBox(
                    (int(x1), int(y1), int(x2), int(y2)),
                    str(names[class_index]),
                    1.0,
                    "ballons_ysg_obb",
                )
            )

    size = max(0, int(settings.mask_dilate_size))
    if size > 0 and np.any(mask):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * size + 1, 2 * size + 1),
            (size, size),
        )
        mask = cv2.dilate(mask, kernel)
    return binary_mask(mask), tuple(records)


class BallonsYSGReference:
    """Original Ultralytics runtime plus Ballons' exact YSG postprocessing."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        settings: YSGSettings | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path).to(device=device)
        self.device = str(device)
        self.settings = settings or YSGSettings()

    def infer(self, image_bgr: np.ndarray) -> CandidateMaskResult:
        start = time.perf_counter()
        result = self.model.predict(
            source=image_bgr,
            save=False,
            show=False,
            verbose=False,
            conf=float(self.settings.confidence_threshold),
            iou=float(self.settings.iou_threshold),
            agnostic_nms=True,
        )[0]
        mask, boxes = mask_and_boxes_from_result(
            result,
            image_bgr.shape[:2],
            self.settings,
        )
        return CandidateMaskResult(
            candidate_id="ballons_ysg",
            raw_mask=mask,
            refined_mask=mask,
            dilated_mask=mask,
            boxes=boxes,
            runtime={
                "seconds": time.perf_counter() - start,
                "device": self.device,
                "reference": "BallonsTranslator YSG original Ultralytics runtime",
            },
        )
