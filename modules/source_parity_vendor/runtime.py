from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import imkit as imk
import numpy as np

from modules.source_parity_vendor.textdetector.ctd import CTDModel
from modules.source_parity_vendor.textdetector.ctd.textmask import REFINEMASK_INPAINT
from modules.source_parity_vendor.inpainter import get_source_parity_lama_large
from modules.source_parity_vendor.utils.textblock import TextBlock as SourceParityTextBlock
from modules.utils.download import ModelDownloader, ModelID
from modules.utils.textblock import TextBlock


@dataclass(slots=True)
class ExactSourceParityResult:
    blocks: list[TextBlock]
    mask_details: dict[str, Any]
    backend: str
    device: str


def _normalize_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=np.uint8)
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return np.where(arr > 0, 255, 0).astype(np.uint8)


def _choose_model_path(device: str) -> str:
    if str(device).lower() != "cpu":
        return ModelDownloader.primary_path(ModelID.CTD_TORCH)
    return ModelDownloader.primary_path(ModelID.CTD_ONNX)


def _serialize_source_block(block) -> dict[str, Any]:
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


def _apply_source_detector_postprocess(source_blocks, cfg: dict[str, Any]) -> None:
    fnt_rsz = float(cfg.get("ctd_font_size_multiplier", 1.0) or 1.0)
    fnt_max = int(cfg.get("ctd_font_size_max", -1) or -1)
    fnt_min = int(cfg.get("ctd_font_size_min", -1) or -1)
    for block in source_blocks:
        detected_size = float(getattr(block, "_detected_font_size", -1) or -1)
        if detected_size <= 0:
            detected_size = float(getattr(block, "font_size", -1) or -1)
        if detected_size <= 0:
            continue
        size = detected_size * fnt_rsz
        if fnt_max > 0:
            size = min(size, fnt_max)
        if fnt_min > 0:
            size = max(size, fnt_min)
        block.font_size = size
        block._detected_font_size = size


def _adapt_source_block(source_block, index: int) -> TextBlock:
    lines = [
        np.asarray(line, dtype=np.int32).tolist()
        for line in getattr(source_block, "lines", []) or []
    ]
    block = TextBlock(
        text_bbox=np.asarray(getattr(source_block, "xyxy", [0, 0, 0, 0]), dtype=np.int32),
        lines=lines,
        text_class="source_parity_text",
        angle=int(getattr(source_block, "angle", 0) or 0),
        text=source_block.get_text() if hasattr(source_block, "get_text") else str(getattr(source_block, "text", "") or ""),
        direction="vertical" if bool(getattr(source_block, "vertical", False)) else "horizontal",
    )
    detected_size = float(getattr(source_block, "_detected_font_size", -1) or -1)
    if detected_size > 0:
        block.min_font_size = int(round(detected_size))
        block.max_font_size = int(round(detected_size))
    setattr(block, "_source_parity_source_block", source_block)
    setattr(block, "_source_parity_source_index", int(index))
    return block


class ExactSourceParityRuntime:
    def __init__(self) -> None:
        self._model: CTDModel | None = None
        self._model_key: tuple[str, str, int, int] | None = None

    def _ensure_model(self, cfg: dict[str, Any]) -> CTDModel:
        device = str(cfg.get("ctd_device", "cuda") or "cuda")
        detect_size = int(cfg.get("ctd_detect_size", 1280) or 1280)
        det_rearrange_max_batches = int(cfg.get("ctd_det_rearrange_max_batches", 4) or 4)
        model_path = _choose_model_path(device)
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

    def detect(self, image_rgb: np.ndarray, cfg: dict[str, Any]) -> ExactSourceParityResult:
        device = str(cfg.get("ctd_device", "cuda") or "cuda")
        model = self._ensure_model(cfg)
        raw_mask, refined_mask, source_blocks = model(
            image_rgb,
            refine_mode=REFINEMASK_INPAINT,
            keep_undetected_mask=False,
        )
        _apply_source_detector_postprocess(source_blocks, cfg)

        raw_mask = _normalize_mask(raw_mask, image_rgb.shape[:2])
        refined_mask = _normalize_mask(refined_mask, image_rgb.shape[:2])
        final_mask = refined_mask.copy()

        ksize = int(cfg.get("ctd_mask_dilate_size", 2) or 2)
        if ksize > 0:
            element = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * ksize + 1, 2 * ksize + 1),
                (ksize, ksize),
            )
            final_mask = cv2.dilate(final_mask, element)

        adapted_blocks = [_adapt_source_block(block, idx) for idx, block in enumerate(source_blocks)]
        backend = str(getattr(model, "backend", "torch") or "torch")
        mask_details = {
            "raw_mask": raw_mask,
            "refined_mask": refined_mask,
            "protect_mask": np.zeros_like(final_mask, dtype=np.uint8),
            "final_mask_pre_expand": refined_mask.copy(),
            "final_mask_post_expand": final_mask.copy(),
            "final_mask": final_mask.copy(),
            "mask_refiner": "ctd",
            "mask_inpaint_mode": "source_parity_ctd_lama",
            "keep_existing_lines": False,
            "refiner_backend": f"source_{backend}",
            "refiner_device": device,
            "fallback_used": False,
            "source_blocks": list(source_blocks),
            "source_blocks_serialized": [_serialize_source_block(block) for block in source_blocks],
        }
        return ExactSourceParityResult(
            blocks=adapted_blocks,
            mask_details=mask_details,
            backend=backend,
            device=device,
        )


def _adapt_generic_block_to_source_block(block: TextBlock) -> SourceParityTextBlock | None:
    xyxy = getattr(block, "xyxy", None)
    if xyxy is None:
        return None
    source_block = SourceParityTextBlock(
        xyxy=[int(v) for v in list(np.asarray(xyxy, dtype=np.int32))],
        lines=[np.asarray(line, dtype=np.int32).tolist() for line in list(getattr(block, "lines", []) or [])],
        angle=int(getattr(block, "angle", 0) or 0),
        text=[str(getattr(block, "text", "") or "")],
    )
    direction = str(getattr(block, "direction", "") or "")
    if direction == "vertical":
        source_block.vertical = True
        source_block.src_is_vertical = True
    return source_block


def _resolve_source_blocks(blocks: list[TextBlock]) -> list[Any]:
    resolved = []
    for index, block in enumerate(list(blocks or [])):
        source_block = getattr(block, "_source_parity_source_block", None)
        if source_block is None:
            source_block = _adapt_generic_block_to_source_block(block)
        if source_block is None:
            continue
        source_index = int(getattr(block, "_source_parity_source_index", index) or index)
        resolved.append((source_index, source_block))
    resolved.sort(key=lambda item: item[0])
    return [item[1] for item in resolved]


def source_parity_blockwise_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    blocks: list[TextBlock],
    inpainter,
    config,
    *,
    check_need_inpaint: bool = True,
) -> np.ndarray:
    if image is None or mask is None or not np.any(mask) or not blocks:
        result = inpainter(image, mask, config)
        return imk.convert_scale_abs(result)

    source_blocks = _resolve_source_blocks(blocks)
    if not source_blocks:
        result = inpainter(image, mask, config)
        return imk.convert_scale_abs(result)

    device = str(getattr(inpainter, "device", "cuda") or "cuda")
    precision = str(getattr(inpainter, "precision", "bf16") or "bf16")
    inpaint_size = int(getattr(inpainter, "inpaint_size", 1536) or 1536)
    parity_inpainter = get_source_parity_lama_large(device=device, precision=precision, inpaint_size=inpaint_size)
    result = parity_inpainter.inpaint(image, np.where(mask > 0, 255, 0).astype(np.uint8), source_blocks, check_need_inpaint=check_need_inpaint)
    return imk.convert_scale_abs(result)
