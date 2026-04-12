from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from modules.source_parity_vendor import ExactSourceParityRuntime, ExactSourceParityResult
from modules.source_parity_vendor.textdetector.ctd import CTDModel
from modules.source_parity_vendor.textdetector.ctd.inference import (
    det_rearrange_forward,
    postprocess_mask,
    postprocess_yolo,
    preprocess_img,
)
from modules.source_parity_vendor.textdetector.ctd.textmask import (
    REFINEMASK_INPAINT,
    refine_mask,
    refine_undetected_mask,
)
from modules.source_parity_vendor.utils.textblock import TextBlock as SourceTextBlock
from modules.utils.download import ModelDownloader, ModelID
from modules.utils.textblock import TextBlock

SourceParityDetectionResult = ExactSourceParityResult


class SourceCTDDetector:
    def __init__(self) -> None:
        self._runtime = ExactSourceParityRuntime()

    def detect(self, image_rgb: np.ndarray, cfg: dict[str, Any]) -> SourceParityDetectionResult:
        return self._runtime.detect(image_rgb, cfg)


class RTDetrSourceMaskRuntime:
    def __init__(self) -> None:
        self._model: CTDModel | None = None
        self._model_key: tuple[str, str, int, int] | None = None

    def _choose_model_path(self, device: str) -> str:
        if str(device).lower() != "cpu":
            return ModelDownloader.primary_path(ModelID.CTD_TORCH)
        return ModelDownloader.primary_path(ModelID.CTD_ONNX)

    def _ensure_model(self, cfg: dict[str, Any]) -> CTDModel:
        device = str(cfg.get("ctd_device", "cuda") or "cuda")
        detect_size = int(cfg.get("ctd_detect_size", 1280) or 1280)
        det_rearrange_max_batches = int(cfg.get("ctd_det_rearrange_max_batches", 4) or 4)
        model_path = self._choose_model_path(device)
        model_key = (model_path, device, detect_size, det_rearrange_max_batches)
        if self._model is None or self._model_key != model_key:
            self._model = CTDModel(
                model_path,
                detect_size=detect_size,
                device=device,
                det_rearrange_max_batches=det_rearrange_max_batches,
            )
            self._model_key = model_key
        else:
            self._model.detect_size = detect_size
            self._model.det_rearrange_max_batches = det_rearrange_max_batches
            self._model.device = device
        return self._model

    def _infer_raw_mask(self, image_rgb: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, str, str]:
        model = self._ensure_model(cfg)
        detect_size = model.detect_size if model.backend != "opencv" else 1024
        im_h, im_w = image_rgb.shape[:2]
        lines_map, mask = det_rearrange_forward(
            image_rgb,
            model.det_batch_forward_ctd,
            detect_size,
            model.det_rearrange_max_batches,
            model.device,
        )
        if lines_map is None:
            img_in, _ratio, dw, dh = preprocess_img(
                image_rgb,
                bgr2rgb=False,
                detect_size=detect_size,
                device=model.device,
                half=model.half,
                to_tensor=model.backend == "torch",
            )
            blks, mask, lines_map = model.net(img_in)
            if model.backend == "opencv" and mask.shape[1] == 2:
                tmp = mask
                mask = lines_map
                lines_map = tmp
            mask = mask.squeeze()
            resize_ratio = (im_w / (detect_size - dw), im_h / (detect_size - dh))
            postprocess_yolo(blks, model.conf_thresh, model.nms_thresh, resize_ratio)
            mask = mask[..., : mask.shape[0] - dh, : mask.shape[1] - dw]
            lines_map = lines_map[..., : lines_map.shape[2] - dh, : lines_map.shape[3] - dw]
        raw_mask = postprocess_mask(mask)
        raw_mask = cv2.resize(raw_mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
        raw_mask = np.where(np.asarray(raw_mask) > 0, 255, 0).astype(np.uint8)
        return raw_mask, str(model.backend or "torch"), str(model.device or cfg.get("ctd_device", "cuda"))


_RTDETR_SOURCE_MASK_RUNTIME = RTDetrSourceMaskRuntime()


def _serialize_source_block(block: SourceTextBlock) -> dict[str, Any]:
    return {
        "xyxy": [int(v) for v in list(getattr(block, "xyxy", [0, 0, 0, 0]))],
        "language": str(getattr(block, "language", "unknown") or "unknown"),
        "vertical": bool(getattr(block, "vertical", False)),
        "src_is_vertical": bool(getattr(block, "src_is_vertical", False)),
        "angle": int(getattr(block, "angle", 0) or 0),
        "font_size": float(getattr(block, "font_size", -1) or -1),
        "detected_font_size": float(getattr(block, "_detected_font_size", -1) or -1),
        "text": block.get_text() if hasattr(block, "get_text") else str(getattr(block, "text", "") or ""),
        "lines": [
            np.asarray(line, dtype=np.int32).tolist()
            for line in getattr(block, "lines", []) or []
        ],
    }


def _adapt_rtdetr_block(block: TextBlock) -> SourceTextBlock:
    xyxy = [int(v) for v in list(np.asarray(getattr(block, "xyxy", [0, 0, 0, 0]), dtype=np.int32))]
    lines = []
    for line in list(getattr(block, "lines", []) or []):
        arr = np.asarray(line, dtype=np.int32)
        if arr.ndim == 2 and arr.shape[0] >= 1:
            lines.append(arr.tolist())
    source_block = SourceTextBlock(
        xyxy=xyxy,
        lines=lines,
        angle=int(getattr(block, "angle", 0) or 0),
        text=[str(getattr(block, "text", "") or "")],
    )
    direction = str(getattr(block, "direction", "") or "")
    if direction == "vertical":
        source_block.vertical = True
        source_block.src_is_vertical = True
    return source_block


def _attach_source_blocks(blocks: list[TextBlock], source_blocks: list[SourceTextBlock]) -> None:
    for index, (block, source_block) in enumerate(zip(list(blocks or []), list(source_blocks or []))):
        setattr(block, "_source_parity_source_block", source_block)
        setattr(block, "_source_parity_source_index", int(index))


def build_rtdetr_source_mask(image_rgb: np.ndarray, blocks: list[TextBlock], cfg: dict[str, Any]) -> dict[str, Any]:
    raw_mask, backend, device = _RTDETR_SOURCE_MASK_RUNTIME._infer_raw_mask(image_rgb, cfg)
    source_blocks = [_adapt_rtdetr_block(block) for block in list(blocks or []) if getattr(block, "xyxy", None) is not None]
    _attach_source_blocks(list(blocks or []), source_blocks)

    refined_mask = refine_mask(
        image_rgb,
        raw_mask,
        source_blocks,
        refine_mode=REFINEMASK_INPAINT,
    )
    if bool(cfg.get("source_keep_undetected_mask", False)):
        refined_mask = refine_undetected_mask(
            image_rgb,
            raw_mask.copy(),
            refined_mask,
            source_blocks,
            refine_mode=REFINEMASK_INPAINT,
        )

    final_mask = np.where(np.asarray(refined_mask) > 0, 255, 0).astype(np.uint8)
    ksize = int(cfg.get("ctd_mask_dilate_size", 2) or 2)
    if ksize > 0:
        element = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * ksize + 1, 2 * ksize + 1),
            (ksize, ksize),
        )
        final_mask = cv2.dilate(final_mask, element)

    refined_mask = np.where(np.asarray(refined_mask) > 0, 255, 0).astype(np.uint8)
    return {
        "raw_mask": raw_mask,
        "refined_mask": refined_mask,
        "protect_mask": np.zeros_like(raw_mask, dtype=np.uint8),
        "final_mask_pre_expand": refined_mask.copy(),
        "final_mask_post_expand": final_mask.copy(),
        "final_mask": np.where(np.asarray(final_mask) > 0, 255, 0).astype(np.uint8),
        "mask_refiner": "ctd",
        "keep_existing_lines": False,
        "refiner_backend": f"source_{backend}",
        "refiner_device": device,
        "fallback_used": False,
        "mask_inpaint_mode": "rtdetr_source_ctd_lama",
        "source_blocks": list(source_blocks),
        "source_blocks_serialized": [_serialize_source_block(block) for block in source_blocks],
    }
