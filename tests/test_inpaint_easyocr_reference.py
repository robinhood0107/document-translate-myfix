from __future__ import annotations

import numpy as np
import torch

from benchmarking.inpaint_detector_bakeoff.easyocr_reference import (
    CRAFTSettings,
    DBNetSettings,
    EasyOCRCRAFTReference,
    EasyOCRDBNetReference,
)
from scripts.benchmark_inpaint_detector_bakeoff import build_parser


class _CraftImageProc:
    @staticmethod
    def resize_aspect_ratio(image, *_args, **_kwargs):
        return image, 1.0, image.shape[:2]

    @staticmethod
    def normalizeMeanVariance(image):
        return image.astype(np.float32) / 255.0


class _CraftUtils:
    @staticmethod
    def getDetBoxes(*_args):
        polygon = np.array([[1, 1], [5, 1], [5, 5], [1, 5]], dtype=np.float32)
        return [polygon], [polygon], []

    @staticmethod
    def adjustResultCoordinates(values, *_args):
        return values


class _CraftModel:
    def __call__(self, tensor):
        height, width = tensor.shape[-2:]
        output = torch.zeros((1, height, width, 2), dtype=torch.float32)
        output[:, 2:4, 2:4, 0] = 0.9
        return output, torch.zeros((1, 1), dtype=torch.float32)


def test_craft_reference_exposes_raw_native_and_dilated_masks() -> None:
    adapter = EasyOCRCRAFTReference.__new__(EasyOCRCRAFTReference)
    adapter._imgproc = _CraftImageProc()
    adapter._craft_utils = _CraftUtils()
    adapter.model = _CraftModel()
    adapter.device = "cpu"
    adapter.settings = CRAFTSettings(canvas_size=32, dilate_size=3)

    result = adapter.infer(np.full((8, 8, 3), 255, dtype=np.uint8))

    assert result.candidate_id == "easyocr_craft"
    assert int(np.count_nonzero(result.raw_mask)) == 4
    assert int(np.count_nonzero(result.refined_mask)) == 25
    assert int(np.count_nonzero(result.dilated_mask)) > 25
    assert len(result.boxes) == 1


class _DBNetModel:
    @staticmethod
    def get_mini_boxes(contour):
        assert np.asarray(contour).dtype == np.float32
        return np.asarray(contour).reshape(-1, 2), 4.0

    @staticmethod
    def resize_image(image, _size):
        return image, image.shape[:2]

    @staticmethod
    def normalize_image(image):
        return image.astype(np.float32) / 255.0

    @staticmethod
    def image2hmap(tensor):
        output = torch.zeros((1, 1, tensor.shape[-2], tensor.shape[-1]))
        output[:, :, 3:5, 3:5] = 0.8
        return output

    @staticmethod
    def boxes_from_bitmap(*_args, **_kwargs):
        polygon = np.array([[2, 2], [6, 2], [6, 6], [2, 6]], dtype=np.float32)
        return (polygon,), (0.9,)


def test_dbnet_reference_exposes_raw_native_and_dilated_masks() -> None:
    adapter = EasyOCRDBNetReference.__new__(EasyOCRDBNetReference)
    adapter.model = _DBNetModel()
    adapter.device = "cpu"
    adapter.settings = DBNetSettings(detection_size=32, dilate_size=3)

    result = adapter.infer(np.full((10, 10, 3), 255, dtype=np.uint8))

    assert result.candidate_id == "easyocr_dbnet18"
    assert int(np.count_nonzero(result.raw_mask)) == 4
    assert int(np.count_nonzero(result.refined_mask)) == 25
    assert int(np.count_nonzero(result.dilated_mask)) > 25
    assert len(result.boxes) == 1


def test_detector_bakeoff_parser_accepts_official_easyocr_references() -> None:
    parser = build_parser()
    for candidate in ("easyocr-craft", "easyocr-dbnet18"):
        args = parser.parse_args(
            [
                "--manifest",
                "manifest.json",
                "--candidate",
                candidate,
                "--model",
                "model.pt",
                "--reference-root",
                "EasyOCR",
            ]
        )
        assert args.candidate == candidate
