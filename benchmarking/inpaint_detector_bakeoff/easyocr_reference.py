from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path
import sys
import time
from types import ModuleType

import cv2
import numpy as np
import torch

from .contracts import CandidateMaskResult, DetectorBox, binary_mask


@dataclass(frozen=True, slots=True)
class CRAFTSettings:
    canvas_size: int = 2560
    mag_ratio: float = 1.0
    text_threshold: float = 0.7
    link_threshold: float = 0.4
    low_text: float = 0.4
    dilate_size: int = 3


@dataclass(frozen=True, slots=True)
class DBNetSettings:
    detection_size: int = 1280
    text_threshold: float = 0.2
    bbox_min_score: float = 0.2
    bbox_min_size: int = 3
    dilate_size: int = 3


def _reference_package(reference_root: Path) -> str:
    package_root = reference_root.resolve() / "easyocr"
    if not (package_root / "detection.py").is_file():
        raise FileNotFoundError(
            f"EasyOCR reference package is incomplete: {package_root}"
        )
    package_name = "_inpaint_easyocr_reference"
    try:
        importlib.import_module("scipy.ndimage")
    except ModuleNotFoundError:
        # CRAFT imports scipy's character-count helper unconditionally, while
        # the detector path below always requests estimate_num_chars=False.
        # Provide the equivalent label API so the pinned reference can load
        # without adding a product-wide SciPy dependency.
        scipy_module = ModuleType("scipy")
        ndimage_module = ModuleType("scipy.ndimage")

        def connected_component_label(value):
            count, labels = cv2.connectedComponents(
                np.asarray(value, dtype=np.uint8),
                connectivity=8,
            )
            return labels, int(count - 1)

        ndimage_module.label = connected_component_label  # type: ignore[attr-defined]
        scipy_module.ndimage = ndimage_module  # type: ignore[attr-defined]
        sys.modules["scipy"] = scipy_module
        sys.modules["scipy.ndimage"] = ndimage_module
    try:
        importlib.import_module("skimage.io")
    except ModuleNotFoundError:
        # imgproc imports skimage only for its path-based convenience loader;
        # this adapter supplies decoded arrays directly.
        skimage_module = ModuleType("skimage")
        io_module = ModuleType("skimage.io")

        def imread(path):
            value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if value is None:
                raise FileNotFoundError(path)
            if value.ndim == 3:
                value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
            return value

        io_module.imread = imread  # type: ignore[attr-defined]
        skimage_module.io = io_module  # type: ignore[attr-defined]
        sys.modules["skimage"] = skimage_module
        sys.modules["skimage.io"] = io_module
    existing = sys.modules.get(package_name)
    if existing is not None:
        existing_paths = tuple(str(value) for value in getattr(existing, "__path__", ()))
        if existing_paths != (str(package_root),):
            raise RuntimeError("EasyOCR reference root changed within one process")
        return package_name
    package = ModuleType(package_name)
    package.__path__ = [str(package_root)]  # type: ignore[attr-defined]
    package.__package__ = package_name
    package.__spec__ = importlib.util.spec_from_loader(
        package_name,
        loader=None,
        is_package=True,
    )
    sys.modules[package_name] = package
    public_alias = sys.modules.get("easyocr")
    if public_alias is not None and tuple(
        str(value) for value in getattr(public_alias, "__path__", ())
    ) != (str(package_root),):
        raise RuntimeError("an unrelated EasyOCR package is already loaded")
    sys.modules["easyocr"] = package
    return package_name


def _dilate(mask: np.ndarray, size: int) -> np.ndarray:
    if size <= 0:
        return binary_mask(mask)
    kernel = np.ones((size, size), dtype=np.uint8)
    return binary_mask(cv2.dilate(binary_mask(mask), kernel, iterations=1))


def _polygon_mask(
    polygons: list[np.ndarray | None] | tuple[np.ndarray | None, ...],
    shape: tuple[int, int],
) -> tuple[np.ndarray, tuple[DetectorBox, ...]]:
    mask = np.zeros(shape, dtype=np.uint8)
    boxes: list[DetectorBox] = []
    for polygon in polygons:
        if polygon is None:
            continue
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if points.shape[0] < 3 or not np.isfinite(points).all():
            continue
        rounded = np.rint(points).astype(np.int32)
        rounded[:, 0] = np.clip(rounded[:, 0], 0, shape[1] - 1)
        rounded[:, 1] = np.clip(rounded[:, 1], 0, shape[0] - 1)
        if cv2.contourArea(rounded) <= 0:
            continue
        cv2.fillPoly(mask, [rounded], 255)
        x, y, width, height = cv2.boundingRect(rounded)
        boxes.append(
            DetectorBox(
                (x, y, x + width, y + height),
                "text",
                1.0,
                "easyocr_reference",
            )
        )
    return binary_mask(mask), tuple(boxes)


def _resize_probability_to_source(
    probability: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    resized = cv2.resize(
        np.asarray(probability, dtype=np.float32),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.ascontiguousarray(resized)


class EasyOCRCRAFTReference:
    """Official EasyOCR CRAFT runtime exposed as pixel-mask variants."""

    def __init__(
        self,
        reference_root: Path,
        model_path: Path,
        *,
        device: str = "cuda",
        settings: CRAFTSettings | None = None,
    ) -> None:
        package = _reference_package(reference_root)
        self._detection = importlib.import_module(f"{package}.detection")
        self._craft_utils = importlib.import_module(f"{package}.craft_utils")
        self._imgproc = importlib.import_module(f"{package}.imgproc")
        self.device = device
        self.settings = settings or CRAFTSettings()
        self.model = self._detection.get_detector(
            str(model_path.resolve()),
            device=device,
            quantize=False,
            cudnn_benchmark=False,
        )

    def infer(self, image_bgr: np.ndarray) -> CandidateMaskResult:
        start = time.perf_counter()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized, target_ratio, _heatmap_size = self._imgproc.resize_aspect_ratio(
            image_rgb,
            self.settings.canvas_size,
            interpolation=cv2.INTER_LINEAR,
            mag_ratio=self.settings.mag_ratio,
        )
        normalized = self._imgproc.normalizeMeanVariance(resized)
        tensor = torch.from_numpy(
            np.transpose(normalized, (2, 0, 1))[None, ...]
        ).to(self.device)
        with torch.no_grad():
            output, _features = self.model(tensor)
        text_score = output[0, :, :, 0].detach().float().cpu().numpy()
        link_score = output[0, :, :, 1].detach().float().cpu().numpy()
        raw_probability = _resize_probability_to_source(
            text_score,
            image_bgr.shape[:2],
        )
        raw = binary_mask(raw_probability >= self.settings.low_text)
        boxes, polygons, _mapper = self._craft_utils.getDetBoxes(
            text_score,
            link_score,
            self.settings.text_threshold,
            self.settings.link_threshold,
            self.settings.low_text,
            False,
            False,
        )
        ratio = 1.0 / float(target_ratio)
        boxes = self._craft_utils.adjustResultCoordinates(boxes, ratio, ratio)
        polygons = self._craft_utils.adjustResultCoordinates(polygons, ratio, ratio)
        resolved = [polygon if polygon is not None else box for box, polygon in zip(boxes, polygons)]
        refined, detector_boxes = _polygon_mask(resolved, image_bgr.shape[:2])
        return CandidateMaskResult(
            candidate_id="easyocr_craft",
            raw_mask=raw,
            refined_mask=refined,
            dilated_mask=_dilate(refined, self.settings.dilate_size),
            boxes=detector_boxes,
            stage_tensors={
                "text_score": text_score,
                "link_score": link_score,
            },
            runtime={
                "seconds": time.perf_counter() - start,
                "backend": "torch",
                "device": self.device,
                "reference": "EasyOCR official CRAFT",
                "target_ratio": float(target_ratio),
            },
        )


class EasyOCRDBNetReference:
    """Official EasyOCR DBNet18 runtime exposed as pixel-mask variants."""

    def __init__(
        self,
        reference_root: Path,
        model_path: Path,
        *,
        device: str = "cuda",
        settings: DBNetSettings | None = None,
    ) -> None:
        package = _reference_package(reference_root)
        detection_db = importlib.import_module(f"{package}.detection_db")
        self.device = device
        self.settings = settings or DBNetSettings()
        self.model = detection_db.get_detector(
            str(model_path.resolve()),
            backbone="resnet18",
            device=device,
            quantize=False,
            cudnn_benchmark=False,
        )

    def infer(self, image_bgr: np.ndarray) -> CandidateMaskResult:
        start = time.perf_counter()
        resized, original_shape = self.model.resize_image(
            image_bgr,
            self.settings.detection_size,
        )
        normalized = self.model.normalize_image(resized)
        tensor = torch.from_numpy(
            np.transpose(normalized, (2, 0, 1))[None, ...]
        ).to(self.device)
        with torch.no_grad():
            heatmap = self.model.image2hmap(tensor)
        probability = heatmap[0, 0].detach().float().cpu().numpy()
        raw_probability = _resize_probability_to_source(
            probability,
            image_bgr.shape[:2],
        )
        raw = binary_mask(raw_probability >= self.settings.text_threshold)
        polygon_batches, _scores = self.model.hmap2bbox(
            tensor,
            (original_shape,),
            heatmap,
            text_threshold=self.settings.text_threshold,
            bbox_min_score=self.settings.bbox_min_score,
            bbox_min_size=self.settings.bbox_min_size,
            max_candidates=0,
            as_polygon=True,
        )
        polygons = list(polygon_batches[0]) if polygon_batches else []
        refined, detector_boxes = _polygon_mask(polygons, image_bgr.shape[:2])
        return CandidateMaskResult(
            candidate_id="easyocr_dbnet18",
            raw_mask=raw,
            refined_mask=refined,
            dilated_mask=_dilate(refined, self.settings.dilate_size),
            boxes=detector_boxes,
            stage_tensors={"probability": probability},
            runtime={
                "seconds": time.perf_counter() - start,
                "backend": "torch",
                "device": self.device,
                "reference": "EasyOCR official DBNet18",
            },
        )
