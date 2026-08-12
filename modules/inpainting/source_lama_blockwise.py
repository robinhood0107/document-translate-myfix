from __future__ import annotations

from contextlib import nullcontext
from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter

import cv2
import imkit as imk
import numpy as np
import torch

from modules.inpainting.lama_torch_network import load_lama_mpe
from modules.source_parity_vendor.utils.imgproc_utils import enlarge_window, resize_keepasp
from modules.source_parity_vendor.utils.textblock import TextBlock as SourceLaMaTextBlock
from modules.source_parity_vendor.utils.textblock_mask import extract_ballon_mask
from modules.utils.bubble_erase import (
    BubbleEraseBlockStats,
    ERASE_MODE_BUBBLE_LAMA_FALLBACK,
    ERASE_MODE_TEXT_FREE_LAMA,
    erase_text_bubble_regions,
    set_block_erase_metadata,
)
from modules.utils.download import ModelDownloader, ModelID
from modules.utils.gpu_handoff import estimate_torch_cuda_storage_mb
from modules.utils.inpaint_composite import composite_with_edit_mask, normalize_edit_mask
from modules.utils.inpaint_evidence import (
    BlockInpaintEvidence,
    SourceLamaBlockwiseResult,
    mask_patch_from_page_mask,
)
from modules.utils.inpaint_positive_evidence import (
    build_detector_positive_text_evidence,
)
from modules.utils.inpainting_runtime import is_lama_family_inpainter
from modules.utils.mask_roi import normalize_xyxy, resolve_inpaint_text_xyxy
from modules.utils.textblock import TextBlock
from modules.inpainting.runtime_contract import (
    INPAINT_RETRY_POLICY_VERSION,
    InpaintingCudaOOMError,
    bounded_retry_roi,
    inpaint_cuda_oom_message,
    inspect_learned_inpainter_runtime,
    is_cuda_device,
    is_cuda_oom_error,
    runtime_mask_diagnostics,
    validate_learned_inpaint_runtime,
)


def _clip_half_open_bbox(xyxy, im_w: int, im_h: int) -> list[int] | None:
    try:
        x1, y1, x2, y2 = [int(v) for v in xyxy]
    except Exception:
        return None
    x1 = max(0, min(int(im_w), x1))
    x2 = max(0, min(int(im_w), x2))
    y1 = max(0, min(int(im_h), y1))
    y2 = max(0, min(int(im_h), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _inpaint_handle_alpha_channel(original_alpha: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result_alpha = original_alpha.copy()
    mask_dilated = cv2.dilate((mask > 127).astype(np.uint8), np.ones((15, 15), np.uint8), iterations=1)
    surrounding_mask = mask_dilated - (mask > 127).astype(np.uint8)
    if np.any(surrounding_mask > 0):
        surrounding_alpha = original_alpha[surrounding_mask > 0]
        if len(surrounding_alpha) > 0:
            median_surrounding_alpha = np.median(surrounding_alpha)
            if median_surrounding_alpha < 128:
                result_alpha[mask > 127] = median_surrounding_alpha
    return result_alpha


@dataclass(frozen=True)
class SourceLaMaKey:
    device: str
    precision: str
    inpaint_size: int


class SourceLaMaLarge:
    inpaint_by_block = True
    check_need_inpaint = True

    def __init__(self, device: str = "cuda", precision: str = "bf16", inpaint_size: int = 1536) -> None:
        self.device = str(device or "cuda")
        self.precision = str(precision or "bf16")
        self.inpaint_size = int(inpaint_size or 1536)
        self.model = None
        self.run_diagnostics: list[dict] = []
        validate_learned_inpaint_runtime(
            inpainter_key="lama_large_512px",
            device=self.device,
            precision=self.precision,
        )

    @property
    def key(self) -> SourceLaMaKey:
        return SourceLaMaKey(self.device, self.precision, self.inpaint_size)

    def ensure_loaded(self) -> None:
        if self.model is None:
            self.model = load_lama_mpe(
                str(ModelDownloader.primary_path(ModelID.LAMA_LARGE_512PX)),
                device="cpu",
                use_mpe=False,
                large_arch=True,
            )
            self.moveToDevice(self.device, precision=self.precision)

    def moveToDevice(self, device: str, precision: str | None = None) -> None:
        self.ensure_loaded()
        resolved_precision = (
            str(precision)
            if precision is not None
            else str(self.precision)
        )
        validate_learned_inpaint_runtime(
            inpainter_key="lama_large_512px",
            device=str(device),
            precision=resolved_precision,
        )
        self.model.to(device)
        self.device = str(device)
        self.precision = resolved_precision

    def memory_safe_inpaint(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        textblock_list=None,
        *,
        diagnostic_context: dict | None = None,
    ) -> np.ndarray:
        self.ensure_loaded()
        context = dict(diagnostic_context or {})
        block_index = context.get("block_index")
        try:
            normalized_block_index = (
                int(block_index) if block_index is not None else None
            )
        except (TypeError, ValueError):
            normalized_block_index = None
        diagnostics = {
            **inspect_learned_inpainter_runtime(
                self,
                inpainter_key="lama_large_512px",
                requested_device=self.device,
                requested_precision=self.precision,
            ),
            **runtime_mask_diagnostics(mask, img.shape),
            "retry_policy": INPAINT_RETRY_POLICY_VERSION,
            "oom_retry_count": 0,
            "oom_retry_roi": None,
            "status": "running",
            "phase": str(context.get("phase", "full") or "full"),
            "block_index": normalized_block_index,
            "is_inference": True,
        }
        started = perf_counter()

        def append_diagnostics() -> None:
            diagnostics["elapsed_seconds"] = float(perf_counter() - started)
            try:
                if is_cuda_device(self.device):
                    if not torch.cuda.is_available():
                        diagnostics["cuda_memory_diagnostics_unavailable"] = True
                    else:
                        device = torch.device(self.device)
                        diagnostics["cuda_memory_allocated_mb"] = float(
                            torch.cuda.memory_allocated(device) / (1024 * 1024)
                        )
                        diagnostics["cuda_memory_reserved_mb"] = float(
                            torch.cuda.memory_reserved(device) / (1024 * 1024)
                        )
                        diagnostics["page_peak_vram_allocated_mb"] = float(
                            torch.cuda.max_memory_allocated(device)
                            / (1024 * 1024)
                        )
                        diagnostics["page_peak_vram_reserved_mb"] = float(
                            torch.cuda.max_memory_reserved(device)
                            / (1024 * 1024)
                        )
                        diagnostics["cuda_memory_diagnostics_available"] = True
            except Exception:
                diagnostics["cuda_memory_diagnostics_unavailable"] = True
            self.run_diagnostics.append(diagnostics)

        try:
            result = self._inpaint(img, mask, textblock_list)
        except Exception as exc:
            if is_cuda_device(self.device) and is_cuda_oom_error(exc):
                retry_roi = bounded_retry_roi(mask, img.shape)
                diagnostics["oom_retry_count"] = 1
                diagnostics["oom_retry_roi"] = (
                    retry_roi.as_list() if retry_roi is not None else None
                )
                diagnostics["first_error"] = type(exc).__name__
                if retry_roi is None:
                    diagnostics["status"] = "failed_no_smaller_roi"
                    append_diagnostics()
                    raise InpaintingCudaOOMError(
                        inpaint_cuda_oom_message(),
                        diagnostics=diagnostics,
                    ) from exc
                torch.cuda.empty_cache()
                x1, y1, x2, y2 = retry_roi.as_list()
                retry_img = np.ascontiguousarray(img[y1:y2, x1:x2])
                retry_mask = np.ascontiguousarray(mask[y1:y2, x1:x2])
                try:
                    retry_result = self._inpaint(
                        retry_img,
                        retry_mask,
                        None,
                    )
                except Exception as retry_exc:
                    if not is_cuda_oom_error(retry_exc):
                        diagnostics["status"] = "failed_during_roi_retry"
                        diagnostics["retry_error"] = type(
                            retry_exc
                        ).__name__
                        append_diagnostics()
                        raise
                    diagnostics["status"] = "failed_after_roi_retry"
                    diagnostics["retry_error"] = type(retry_exc).__name__
                    append_diagnostics()
                    raise InpaintingCudaOOMError(
                        inpaint_cuda_oom_message(),
                        diagnostics=diagnostics,
                    ) from retry_exc
                result = np.asarray(img).copy()
                result[y1:y2, x1:x2] = composite_with_edit_mask(
                    retry_img,
                    retry_result,
                    retry_mask,
                )
                diagnostics["status"] = "completed_after_roi_retry"
            else:
                diagnostics["status"] = "failed"
                diagnostics["first_error"] = type(exc).__name__
                append_diagnostics()
                raise
        else:
            diagnostics["status"] = "completed"
        append_diagnostics()
        return result

    def inpaint_preprocess(self, img: np.ndarray, mask: np.ndarray):
        img_original = np.copy(img)
        mask_original = np.copy(mask)
        mask_original[mask_original < 127] = 0
        mask_original[mask_original >= 127] = 1
        mask_original = mask_original[:, :, None]

        new_shape = self.inpaint_size if max(img.shape[0:2]) > self.inpaint_size else None
        img = resize_keepasp(img, new_shape, stride=64)
        mask = resize_keepasp(mask, new_shape, stride=64)

        im_h, im_w = img.shape[:2]
        longer = max(im_h, im_w)
        pad_bottom = longer - im_h if im_h < longer else 0
        pad_right = longer - im_w if im_w < longer else 0
        mask = cv2.copyMakeBorder(mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
        img = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)

        img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze_(0).float() / 255.0
        mask_torch = torch.from_numpy(mask).unsqueeze_(0).unsqueeze_(0).float() / 255.0
        mask_torch[mask_torch < 0.5] = 0
        mask_torch[mask_torch >= 0.5] = 1
        if hasattr(self.model, "load_masked_position_encoding"):
            rel_pos, _abs_pos, direct = self.model.load_masked_position_encoding(mask_torch[0][0].numpy())
            rel_pos = torch.LongTensor(rel_pos).unsqueeze_(0)
            direct = torch.LongTensor(direct).unsqueeze_(0)
        else:
            rel_pos, direct = None, None

        if self.device != "cpu":
            img_torch = img_torch.to(self.device)
            mask_torch = mask_torch.to(self.device)
            if rel_pos is not None:
                rel_pos = rel_pos.to(self.device)
            if direct is not None:
                direct = direct.to(self.device)
        img_torch *= 1 - mask_torch
        return img_torch, mask_torch, rel_pos, direct, img_original, mask_original, pad_bottom, pad_right

    @torch.no_grad()
    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list=None) -> np.ndarray:
        self.ensure_loaded()
        im_h, im_w = img.shape[:2]
        img_torch, mask_torch, rel_pos, direct, img_original, mask_original, pad_bottom, pad_right = self.inpaint_preprocess(img, mask)

        precision_name = str(self.precision).lower()
        if precision_name == "bf16":
            precision_context = torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            )
        elif precision_name in {"fp16", "float16"}:
            precision_context = torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            )
        else:
            precision_context = nullcontext()
        with precision_context:
            img_inpainted_torch = self.model(
                img_torch,
                mask_torch,
                rel_pos,
                direct,
            )

        img_inpainted = (
            img_inpainted_torch.to(device="cpu", dtype=torch.float32)
            .squeeze_(0)
            .permute(1, 2, 0)
            .numpy()
            * 255
        )
        img_inpainted = np.clip(np.round(img_inpainted), 0, 255).astype(np.uint8)
        if pad_bottom > 0:
            img_inpainted = img_inpainted[:-pad_bottom]
        if pad_right > 0:
            img_inpainted = img_inpainted[:, :-pad_right]
        new_shape = img_inpainted.shape[:2]
        if new_shape[0] != im_h or new_shape[1] != im_w:
            img_inpainted = cv2.resize(img_inpainted, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
        img_inpainted = img_inpainted * mask_original + img_original * (1 - mask_original)
        return img_inpainted

    def inpaint(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        textblock_list=None,
        check_need_inpaint: bool = False,
        *,
        diagnostic_block_indices: list[int | None] | None = None,
    ) -> np.ndarray:
        self.ensure_loaded()
        original_alpha = None
        if len(img.shape) == 3 and img.shape[2] == 4:
            original_alpha = img[:, :, 3:4]
            img_rgb = img[:, :, :3]
        else:
            img_rgb = img

        if not self.inpaint_by_block or textblock_list is None:
            if check_need_inpaint:
                ballon_msk, non_text_msk = extract_ballon_mask(img_rgb, mask)
                if ballon_msk is not None:
                    non_text_region = np.where(non_text_msk > 0)
                    non_text_px = img_rgb[non_text_region]
                    if non_text_px.size > 0:
                        average_bg_color = np.median(non_text_px, axis=0)
                        std_rgb = np.std(non_text_px - average_bg_color, axis=0)
                        std_max = np.max(std_rgb)
                        inpaint_thresh = 7 if np.std(std_rgb) > 1 else 10
                        if std_max < inpaint_thresh:
                            result_rgb = img_rgb.copy()
                            result_rgb[np.where(ballon_msk > 0)] = average_bg_color
                            if original_alpha is not None:
                                return np.concatenate([result_rgb, original_alpha], axis=2)
                            return result_rgb
            result_rgb = self.memory_safe_inpaint(img_rgb, mask, textblock_list)
            result_rgb = composite_with_edit_mask(img_rgb, result_rgb, mask)
            if original_alpha is not None:
                result_alpha = _inpaint_handle_alpha_channel(original_alpha, mask)
                return np.concatenate([result_rgb, result_alpha], axis=2)
            return result_rgb

        im_h, im_w = img_rgb.shape[:2]
        inpainted = np.copy(img_rgb)
        original_mask = mask.copy()
        work_mask = mask.copy()
        normalized_diagnostic_indices = (
            None
            if diagnostic_block_indices is None
            else list(diagnostic_block_indices)
        )
        for block_index, blk in enumerate(textblock_list):
            xyxy = _clip_half_open_bbox(getattr(blk, "xyxy", [0, 0, 0, 0]), im_w, im_h)
            if xyxy is None:
                continue
            xyxy_e = enlarge_window(xyxy, im_w, im_h, ratio=1.7)
            image_crop = inpainted[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]]
            mask_crop = work_mask[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]]
            if image_crop.size == 0 or mask_crop.size == 0:
                continue
            need_inpaint = True
            if self.check_need_inpaint or check_need_inpaint:
                ballon_msk, non_text_msk = extract_ballon_mask(image_crop, mask_crop)
                if ballon_msk is not None:
                    non_text_region = np.where(non_text_msk > 0)
                    non_text_px = image_crop[non_text_region]
                    average_bg_color = np.median(non_text_px, axis=0)
                    std_rgb = np.std(non_text_px - average_bg_color, axis=0)
                    std_max = np.max(std_rgb)
                    inpaint_thresh = 7 if np.std(std_rgb) > 1 else 10
                    if std_max < inpaint_thresh:
                        need_inpaint = False
                        image_crop[np.where(ballon_msk > 0)] = average_bg_color
            if need_inpaint:
                inpainted[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]] = self.memory_safe_inpaint(
                    image_crop,
                    mask_crop,
                    diagnostic_context={
                        "phase": "block",
                        "block_index": (
                            block_index
                            if normalized_diagnostic_indices is None
                            else (
                                normalized_diagnostic_indices[block_index]
                                if block_index
                                < len(normalized_diagnostic_indices)
                                else None
                            )
                        ),
                    },
                )
            else:
                inpainted[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]] = image_crop
            work_mask[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = 0

        if original_alpha is not None:
            inpainted = composite_with_edit_mask(img_rgb, inpainted, original_mask)
            result_alpha = _inpaint_handle_alpha_channel(original_alpha, original_mask)
            return np.concatenate([inpainted, result_alpha], axis=2)
        return composite_with_edit_mask(img_rgb, inpainted, original_mask)


_INPAINTER_CACHE: dict[SourceLaMaKey, SourceLaMaLarge] = {}


def release_source_lama_cache() -> dict[str, int | float | bool]:
    """Release only Source LaMa model objects retained by this module."""

    cached_inpainters = list(_INPAINTER_CACHE.values())
    _INPAINTER_CACHE.clear()
    loaded_model_count = 0
    gpu_loaded_model_count = 0
    expected_process_reclaim_mb = 0.0
    untracked_gpu_resource_count = 0
    for inpainter in cached_inpainters:
        model = getattr(inpainter, "model", None)
        if model is not None:
            loaded_model_count += 1
            if str(getattr(inpainter, "device", "") or "").lower().startswith("cuda"):
                gpu_loaded_model_count += 1
                estimate = estimate_torch_cuda_storage_mb(model)
                tracked_mb = float(estimate.get("total_mb", 0.0) or 0.0)
                if tracked_mb > 0.0:
                    expected_process_reclaim_mb += tracked_mb
                else:
                    untracked_gpu_resource_count += 1
        inpainter.model = None
    return {
        "cache_entry_count": len(cached_inpainters),
        "loaded_model_count": loaded_model_count,
        "gpu_loaded_model_count": gpu_loaded_model_count,
        "expected_process_reclaim_mb": expected_process_reclaim_mb,
        "untracked_gpu_resource_count": untracked_gpu_resource_count,
        "gpu_release_expected": gpu_loaded_model_count > 0,
    }


def get_source_lama_large(device: str = "cuda", precision: str = "bf16", inpaint_size: int = 1536) -> SourceLaMaLarge:
    key = SourceLaMaKey(device=str(device or "cuda"), precision=str(precision or "bf16"), inpaint_size=int(inpaint_size or 1536))
    cached = _INPAINTER_CACHE.get(key)
    if cached is None:
        cached = SourceLaMaLarge(device=key.device, precision=key.precision, inpaint_size=key.inpaint_size)
        _INPAINTER_CACHE[key] = cached
    return cached


def _adapt_generic_block_to_source_block(block: TextBlock) -> SourceLaMaTextBlock | None:
    xyxy = getattr(block, "xyxy", None)
    if xyxy is None:
        return None
    source_block = SourceLaMaTextBlock(
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


def _resolve_source_blocks(
    blocks: Iterable[TextBlock],
    diagnostic_block_indices: list[int | None] | None = None,
) -> tuple[list[SourceLaMaTextBlock], list[int | None]]:
    resolved: list[SourceLaMaTextBlock] = []
    resolved_indices: list[int | None] = []
    block_list = list(blocks or [])
    requested_indices = (
        None
        if diagnostic_block_indices is None
        else list(diagnostic_block_indices)
    )
    for local_index, block in enumerate(block_list):
        source_block = _adapt_generic_block_to_source_block(block)
        if source_block is not None:
            resolved.append(source_block)
            resolved_indices.append(
                local_index
                if requested_indices is None
                else (
                    requested_indices[local_index]
                    if local_index < len(requested_indices)
                    else None
                )
            )
    return resolved, resolved_indices


def _split_bubble_source_mask(
    mask: np.ndarray,
    blocks: list[TextBlock],
    image_shape: tuple[int, ...],
) -> tuple[
    np.ndarray,
    list[TextBlock],
    list[TextBlock],
    np.ndarray,
    np.ndarray,
    list[TextBlock],
    np.ndarray,
]:
    source_mask = normalize_edit_mask(mask, image_shape)
    bubble_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    lama_priority_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    bubble_protected_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    missing_bubble_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    bubble_blocks: list[TextBlock] = []
    lama_blocks: list[TextBlock] = []
    missing_bubble_blocks: list[TextBlock] = []
    for block in list(blocks or []):
        if getattr(block, "text_class", "") == "text_bubble":
            continue
        lama_blocks.append(block)
        roi = resolve_inpaint_text_xyxy(block, image_shape)
        if roi is None:
            continue
        x1, y1, x2, y2 = roi
        owned = source_mask[y1:y2, x1:x2]
        lama_priority_mask[y1:y2, x1:x2] = np.where(
            (lama_priority_mask[y1:y2, x1:x2] > 0)
            | (owned > 0),
            255,
            0,
        ).astype(np.uint8)
        bubble_protection = owned
        if (
            getattr(block, "text_class", "") == "text_free"
            and np.any(owned)
        ):
            bubble_protection = cv2.dilate(
                owned,
                cv2.getStructuringElement(
                    cv2.MORPH_RECT,
                    (5, 5),
                ),
                iterations=1,
            )
        bubble_protected_mask[y1:y2, x1:x2] = np.where(
            (bubble_protected_mask[y1:y2, x1:x2] > 0)
            | (bubble_protection > 0),
            255,
            0,
        ).astype(np.uint8)
    for block in list(blocks or []):
        if getattr(block, "text_class", "") != "text_bubble":
            continue
        roi = normalize_xyxy(getattr(block, "bubble_xyxy", None), image_shape)
        if roi is None:
            missing_bubble_blocks.append(block)
            priority_roi = resolve_inpaint_text_xyxy(block, image_shape)
            owned_pixel_count = 0
            if priority_roi is not None:
                x1, y1, x2, y2 = priority_roi
                owned = np.where(
                    (source_mask[y1:y2, x1:x2] > 0)
                    & (lama_priority_mask[y1:y2, x1:x2] <= 0),
                    255,
                    0,
                ).astype(np.uint8)
                lama_priority_mask[y1:y2, x1:x2] = np.where(
                    (lama_priority_mask[y1:y2, x1:x2] > 0)
                    | (owned > 0),
                    255,
                    0,
                ).astype(np.uint8)
                missing_bubble_mask[y1:y2, x1:x2] = np.where(
                    (missing_bubble_mask[y1:y2, x1:x2] > 0)
                    | (owned > 0),
                    255,
                    0,
                ).astype(np.uint8)
                bubble_protected_mask[y1:y2, x1:x2] = np.where(
                    (bubble_protected_mask[y1:y2, x1:x2] > 0)
                    | (owned > 0),
                    255,
                    0,
                ).astype(np.uint8)
                owned_pixel_count = int(np.count_nonzero(owned))
            set_block_erase_metadata(
                block,
                BubbleEraseBlockStats(
                    mode=ERASE_MODE_BUBBLE_LAMA_FALLBACK,
                    edit_pixel_count=owned_pixel_count,
                    skipped_reason="missing_bubble_roi",
                ),
            )
            continue
        bubble_blocks.append(block)
        x1, y1, x2, y2 = roi
        bubble_mask[y1:y2, x1:x2] = np.where(source_mask[y1:y2, x1:x2] > 0, 255, bubble_mask[y1:y2, x1:x2])
    bubble_mask = np.where(
        (bubble_mask > 0) & (lama_priority_mask <= 0),
        255,
        0,
    ).astype(np.uint8)
    active_bubble_blocks: list[TextBlock] = []
    for block in bubble_blocks:
        bubble_roi = normalize_xyxy(
            getattr(block, "bubble_xyxy", None),
            image_shape,
        )
        if bubble_roi is None:
            active_bubble_blocks.append(block)
            continue
        x1, y1, x2, y2 = bubble_roi
        source_owned = source_mask[y1:y2, x1:x2] > 0
        owned_pixel_count = int(np.count_nonzero(source_owned))
        priority_owned_pixel_count = int(
            np.count_nonzero(
                source_owned
                & (lama_priority_mask[y1:y2, x1:x2] > 0)
            )
        )
        if (
            owned_pixel_count <= 0
            or priority_owned_pixel_count != owned_pixel_count
        ):
            active_bubble_blocks.append(block)
            continue
        set_block_erase_metadata(
            block,
            BubbleEraseBlockStats(
                mode=ERASE_MODE_BUBBLE_LAMA_FALLBACK,
                edit_pixel_count=owned_pixel_count,
                skipped_reason="lama_priority_owned",
            ),
        )
    bubble_blocks = active_bubble_blocks
    return (
        bubble_mask,
        bubble_blocks,
        lama_blocks,
        lama_priority_mask,
        missing_bubble_mask,
        missing_bubble_blocks,
        bubble_protected_mask,
    )


def _run_lama_or_fallback(
    image: np.ndarray,
    mask: np.ndarray,
    blocks: list[TextBlock],
    inpainter,
    config,
    *,
    check_need_inpaint: bool,
    diagnostic_block_indices: list[int | None] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    if not np.any(mask):
        return np.asarray(image).copy(), []

    source_blocks, source_block_indices = _resolve_source_blocks(
        blocks,
        diagnostic_block_indices,
    )
    if not source_blocks:
        inpainter_key = str(getattr(inpainter, "name", "") or "")
        requested_device = str(
            getattr(
                inpainter,
                "runtime_device",
                getattr(inpainter, "device", ""),
            )
            or ""
        )
        requested_precision = str(
            getattr(inpainter, "precision", "fp32") or "fp32"
        )
        diagnostics = {
            **inspect_learned_inpainter_runtime(
                inpainter,
                inpainter_key=inpainter_key,
                requested_device=requested_device,
                requested_precision=requested_precision,
            ),
            **runtime_mask_diagnostics(mask, image.shape),
            "retry_policy": INPAINT_RETRY_POLICY_VERSION,
            "oom_retry_count": 0,
            "oom_retry_roi": None,
            "status": "running",
            "phase": "generic",
            "block_index": None,
            "is_inference": True,
        }
        started = perf_counter()

        def append_memory_diagnostics() -> None:
            diagnostics["elapsed_seconds"] = float(
                perf_counter() - started
            )
            try:
                if is_cuda_device(requested_device):
                    if not torch.cuda.is_available():
                        diagnostics[
                            "cuda_memory_diagnostics_unavailable"
                        ] = True
                    else:
                        device = torch.device(requested_device)
                        diagnostics["cuda_memory_allocated_mb"] = float(
                            torch.cuda.memory_allocated(device)
                            / (1024 * 1024)
                        )
                        diagnostics["cuda_memory_reserved_mb"] = float(
                            torch.cuda.memory_reserved(device)
                            / (1024 * 1024)
                        )
                        diagnostics[
                            "page_peak_vram_allocated_mb"
                        ] = float(
                            torch.cuda.max_memory_allocated(device)
                            / (1024 * 1024)
                        )
                        diagnostics[
                            "page_peak_vram_reserved_mb"
                        ] = float(
                            torch.cuda.max_memory_reserved(device)
                            / (1024 * 1024)
                        )
                        diagnostics[
                            "cuda_memory_diagnostics_available"
                        ] = True
            except Exception:
                diagnostics["cuda_memory_diagnostics_unavailable"] = True

        try:
            result = inpainter(image, mask, config)
        except Exception as exc:
            if is_cuda_device(requested_device) and is_cuda_oom_error(exc):
                retry_roi = bounded_retry_roi(mask, image.shape)
                diagnostics["oom_retry_count"] = 1
                diagnostics["oom_retry_roi"] = (
                    retry_roi.as_list() if retry_roi is not None else None
                )
                diagnostics["first_error"] = type(exc).__name__
                if retry_roi is None:
                    diagnostics["status"] = "failed_no_smaller_roi"
                    append_memory_diagnostics()
                    raise InpaintingCudaOOMError(
                        inpaint_cuda_oom_message(),
                        diagnostics=diagnostics,
                    ) from exc
                torch.cuda.empty_cache()
                x1, y1, x2, y2 = retry_roi.as_list()
                retry_image = np.ascontiguousarray(image[y1:y2, x1:x2])
                retry_mask = np.ascontiguousarray(mask[y1:y2, x1:x2])
                try:
                    retry_result = inpainter(
                        retry_image,
                        retry_mask,
                        config,
                    )
                except Exception as retry_exc:
                    diagnostics["retry_error"] = type(retry_exc).__name__
                    if not is_cuda_oom_error(retry_exc):
                        diagnostics["status"] = "failed_during_roi_retry"
                        append_memory_diagnostics()
                        raise InpaintingCudaOOMError(
                            inpaint_cuda_oom_message(),
                            diagnostics=diagnostics,
                        ) from retry_exc
                    diagnostics["status"] = "failed_after_roi_retry"
                    append_memory_diagnostics()
                    raise InpaintingCudaOOMError(
                        inpaint_cuda_oom_message(),
                        diagnostics=diagnostics,
                    ) from retry_exc
                result = np.asarray(image).copy()
                result[y1:y2, x1:x2] = composite_with_edit_mask(
                    retry_image,
                    imk.convert_scale_abs(retry_result),
                    retry_mask,
                )
                diagnostics["status"] = "completed_after_roi_retry"
            else:
                diagnostics["status"] = "failed"
                diagnostics["first_error"] = type(exc).__name__
                append_memory_diagnostics()
                raise
        else:
            diagnostics["status"] = "completed"
        append_memory_diagnostics()
        converted = imk.convert_scale_abs(result)
        return composite_with_edit_mask(image, converted, mask), [diagnostics]

    device = str(getattr(inpainter, "runtime_device", getattr(inpainter, "device", "cuda")) or "cuda")
    precision = str(getattr(inpainter, "precision", "bf16") or "bf16")
    inpaint_size = int(getattr(inpainter, "inpaint_size", 1536) or 1536)
    source_inpainter = get_source_lama_large(device=device, precision=precision, inpaint_size=inpaint_size)
    if hasattr(source_inpainter, "run_diagnostics"):
        source_inpainter.run_diagnostics.clear()
    result = source_inpainter.inpaint(
        image,
        np.where(mask > 0, 255, 0).astype(np.uint8),
        source_blocks,
        check_need_inpaint=check_need_inpaint,
        diagnostic_block_indices=source_block_indices,
    )
    converted = imk.convert_scale_abs(result)
    diagnostics = list(
        getattr(source_inpainter, "run_diagnostics", []) or []
    )
    if hasattr(source_inpainter, "run_diagnostics"):
        source_inpainter.run_diagnostics.clear()
    return (
        composite_with_edit_mask(image, converted, mask),
        diagnostics,
    )


def _maybe_return_edit_mask(
    result: np.ndarray,
    edit_mask: np.ndarray | None,
    return_edit_mask: bool,
    *,
    diagnostics: list[dict],
    return_diagnostics: bool,
) -> (
    np.ndarray
    | tuple[np.ndarray, np.ndarray | None]
    | tuple[np.ndarray, list[dict]]
    | tuple[np.ndarray, np.ndarray | None, list[dict]]
):
    if return_edit_mask and return_diagnostics:
        return result, edit_mask, diagnostics
    if return_edit_mask:
        return result, edit_mask
    if return_diagnostics:
        return result, diagnostics
    return result


def _apply_protected_corner_guard(
    original_image: np.ndarray | None,
    result_image: np.ndarray,
    edit_mask: np.ndarray | None,
    protected_corner_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if original_image is None:
        return result_image, edit_mask
    protected = normalize_edit_mask(protected_corner_mask, original_image.shape)
    if not np.any(protected):
        return result_image, edit_mask
    if edit_mask is None:
        if np.asarray(result_image).shape != np.asarray(original_image).shape:
            return result_image, edit_mask
        guarded = np.asarray(result_image).copy()
        guarded[protected > 0] = np.asarray(original_image)[protected > 0]
        return guarded, None
    normalized_edit = normalize_edit_mask(edit_mask, original_image.shape)
    normalized_edit = np.where(
        (normalized_edit > 0) & (protected <= 0),
        255,
        0,
    ).astype(np.uint8)
    guarded = composite_with_edit_mask(
        original_image,
        result_image,
        normalized_edit,
    )
    return guarded, normalized_edit


def _build_owned_block_evidence(
    blocks: list[TextBlock],
    raw_source_mask: np.ndarray | None,
    source_mask: np.ndarray,
    ownership_mask: np.ndarray,
    image_shape: tuple[int, ...],
    original_block_indices: dict[int, int],
) -> list[BlockInpaintEvidence]:
    evidence: list[BlockInpaintEvidence] = []
    for block in blocks:
        roi = resolve_inpaint_text_xyxy(block, image_shape)
        if roi is None:
            roi = normalize_xyxy(getattr(block, "bubble_xyxy", None), image_shape)
        if roi is None:
            continue
        evidence.append(
            BlockInpaintEvidence(
                block_id=str(getattr(block, "block_id", "") or ""),
                block_index=original_block_indices.get(id(block)),
                erase_mode=str(getattr(block, "_erase_mode", "") or ""),
                skipped_reason=str(
                    getattr(block, "_erase_skipped_reason", "") or ""
                ),
                source_raw_owned=mask_patch_from_page_mask(
                    raw_source_mask,
                    roi,
                    image_shape,
                ),
                source_owned=mask_patch_from_page_mask(
                    source_mask,
                    roi,
                    image_shape,
                ),
                ownership_protect=mask_patch_from_page_mask(
                    ownership_mask,
                    roi,
                    image_shape,
                ),
            )
        )
    return evidence


def _apply_detector_positive_text_evidence(
    original_image: np.ndarray,
    current_image: np.ndarray,
    combined_mask: np.ndarray,
    blocks: list[TextBlock],
    evidence: list[BlockInpaintEvidence],
    diagnostics: list[dict],
    inpainter,
    config,
    positive_claim_raw_mask: np.ndarray | None,
    protected_corner_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, list[BlockInpaintEvidence]]:
    positive = build_detector_positive_text_evidence(
        blocks,
        positive_claim_raw_mask,
        evidence,
        image_shape=original_image.shape,
        existing_edit_mask=combined_mask,
        protected_corner_mask=protected_corner_mask,
        source_image=original_image,
    )
    evidence_by_index = {
        item.block_index: item
        for item in evidence
        if item.block_index is not None
    }
    positive_backend_supported = is_lama_family_inpainter(
        getattr(inpainter, "name", "")
    )
    for block_index, claim_patch in positive.block_claim_patches.items():
        block = blocks[block_index]
        item = evidence_by_index.get(block_index)
        if item is None:
            item = BlockInpaintEvidence(
                block_id=str(
                    getattr(block, "canonical_block_id", "")
                    or getattr(block, "block_id", "")
                    or ""
                ),
                block_index=block_index,
            )
            evidence.append(item)
            evidence_by_index[block_index] = item
        item.positive_claim = claim_patch
        item.positive_edit = (
            positive.block_edit_patches.get(block_index)
            if positive_backend_supported
            else None
        )
        item.claim_providers = positive.block_claim_providers.get(
            block_index,
            (),
        )
        item.route_decision = positive.block_route_decisions.get(
            block_index,
            "narrow",
        )
        item.route_reasons = positive.block_route_reasons.get(block_index, ())

    if not np.any(positive.positive_edit):
        return current_image, combined_mask, evidence
    if not positive_backend_supported:
        return current_image, combined_mask, evidence

    broad_applied = np.zeros(original_image.shape[:2], dtype=np.uint8)
    for block_index, broad_patch in positive.block_broad_edit_patches.items():
        item = evidence_by_index.get(block_index)
        interior_patch = positive.block_bubble_interior_patches.get(block_index)
        if item is None or interior_patch is None:
            continue
        bx1, by1, bx2, by2 = broad_patch.xyxy
        ix1, iy1, ix2, iy2 = interior_patch.xyxy
        broad_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        interior_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        broad_mask[by1:by2, bx1:bx2] = broad_patch.mask
        interior_mask[iy1:iy2, ix1:ix2] = interior_patch.mask
        sample_mask = (interior_mask > 0) & (broad_mask <= 0)
        samples = np.asarray(original_image)[sample_mask, :3]
        if samples.shape[0] < 32:
            item.route_decision = "narrow"
            item.route_reasons = tuple(
                dict.fromkeys((*item.route_reasons, "insufficient_roi_background_samples"))
            )
            continue
        fill_color = np.clip(
            np.rint(np.median(samples.astype(np.float32), axis=0)),
            0,
            255,
        ).astype(np.uint8)
        generated_broad = np.asarray(original_image).copy()
        generated_broad[broad_mask > 0, :3] = fill_color
        current_image = composite_with_edit_mask(
            current_image,
            generated_broad,
            broad_mask,
        )
        broad_applied[broad_mask > 0] = 255

    narrow_edit = np.where(
        (positive.narrow_edit > 0) & (broad_applied <= 0),
        255,
        0,
    ).astype(np.uint8)
    positive_diagnostics: list[dict] = []
    if np.any(narrow_edit):
        generated, positive_diagnostics = _run_lama_or_fallback(
            original_image,
            narrow_edit,
            [],
            inpainter,
            config,
            check_need_inpaint=False,
        )
        current_image = composite_with_edit_mask(
            current_image,
            generated,
            narrow_edit,
        )
    positive_indices = sorted(positive.block_edit_patches)
    for diagnostic in positive_diagnostics:
        diagnostic["phase"] = "positive_evidence"
        diagnostic["positive_text_evidence"] = True
        diagnostic["positive_block_indices"] = positive_indices
        diagnostic["positive_claim_pixel_count"] = int(
            np.count_nonzero(positive.positive_claim)
        )
        diagnostic["positive_edit_pixel_count"] = int(
            np.count_nonzero(positive.positive_edit)
        )
    diagnostics.extend(positive_diagnostics)
    if np.any(broad_applied):
        diagnostics.append(
            {
                "phase": "positive_evidence",
                "is_inference": False,
                "status": "completed",
                "backend": "robust_flat_median",
                "positive_text_evidence": True,
                "positive_block_indices": sorted(
                    positive.block_broad_edit_patches
                ),
                "positive_claim_pixel_count": int(
                    np.count_nonzero(positive.positive_claim)
                ),
                "positive_edit_pixel_count": int(
                    np.count_nonzero(broad_applied)
                ),
            }
        )
    applied_positive = np.where(
        (narrow_edit > 0) | (broad_applied > 0),
        255,
        0,
    ).astype(np.uint8)
    combined_mask = np.where(
        (combined_mask > 0) | (applied_positive > 0),
        255,
        0,
    ).astype(np.uint8)
    for block_index, patch in positive.block_edit_patches.items():
        block = blocks[block_index]
        mode = (
            ERASE_MODE_TEXT_FREE_LAMA
            if str(getattr(block, "text_class", "") or "") == "text_free"
            else ERASE_MODE_BUBBLE_LAMA_FALLBACK
        )
        set_block_erase_metadata(
            block,
            BubbleEraseBlockStats(
                mode=mode,
                edit_pixel_count=patch.pixel_count,
                skipped_reason="positive_text_evidence_recovered",
            ),
        )
        item = evidence_by_index[block_index]
        item.erase_mode = mode
        item.skipped_reason = "positive_text_evidence_recovered"
    return current_image, combined_mask, evidence


def source_lama_blockwise_inpaint_result(
    image: np.ndarray,
    mask: np.ndarray,
    blocks: Iterable[TextBlock],
    inpainter,
    config,
    *,
    raw_source_mask: np.ndarray | None = None,
    positive_claim_raw_mask: np.ndarray | None = None,
    check_need_inpaint: bool = True,
    protected_corner_mask: np.ndarray | None = None,
) -> SourceLamaBlockwiseResult:
    block_list = list(blocks or [])
    if image is None or mask is None or not block_list:
        result = inpainter(image, mask, config)
        converted = imk.convert_scale_abs(result)
        cleaned = composite_with_edit_mask(image, converted, mask)
        edit_mask = normalize_edit_mask(mask, image.shape) if image is not None and mask is not None else mask
        cleaned, edit_mask = _apply_protected_corner_guard(
            image,
            cleaned,
            edit_mask,
            protected_corner_mask,
        )
        return SourceLamaBlockwiseResult(
            image=cleaned,
            edit_mask=edit_mask,
            diagnostics=[],
        )

    source_mask = normalize_edit_mask(mask, image.shape)
    normalized_raw_source = (
        normalize_edit_mask(raw_source_mask, image.shape)
        if raw_source_mask is not None
        else None
    )
    normalized_positive_claim = (
        normalize_edit_mask(positive_claim_raw_mask, image.shape)
        if positive_claim_raw_mask is not None
        else None
    )
    has_bubble_candidates = any(
        getattr(block, "text_class", "") == "text_bubble"
        for block in block_list
    )
    original_block_indices = {
        id(block): index
        for index, block in enumerate(block_list)
    }
    (
        bubble_mask,
        bubble_blocks,
        lama_blocks,
        lama_priority_mask,
        missing_bubble_mask,
        missing_bubble_blocks,
        bubble_protected_mask,
    ) = _split_bubble_source_mask(source_mask, block_list, image.shape)
    if not has_bubble_candidates:
        cleaned, diagnostics = _run_lama_or_fallback(
            image,
            source_mask,
            block_list,
            inpainter,
            config,
            check_need_inpaint=check_need_inpaint,
            diagnostic_block_indices=list(range(len(block_list))),
        )
        evidence = _build_owned_block_evidence(
            block_list,
            normalized_raw_source,
            source_mask,
            source_mask,
            image.shape,
            original_block_indices,
        )
        cleaned, source_mask, evidence = _apply_detector_positive_text_evidence(
            image,
            cleaned,
            source_mask,
            block_list,
            evidence,
            diagnostics,
            inpainter,
            config,
            normalized_positive_claim,
            protected_corner_mask,
        )
        cleaned, guarded_mask = _apply_protected_corner_guard(
            image,
            cleaned,
            source_mask,
            protected_corner_mask,
        )
        return SourceLamaBlockwiseResult(
            image=cleaned,
            edit_mask=guarded_mask,
            diagnostics=diagnostics,
            evidence=tuple(evidence),
        )

    lama_mask = np.where(
        (source_mask > 0)
        & (bubble_mask <= 0)
        & (missing_bubble_mask <= 0),
        255,
        0,
    ).astype(np.uint8)
    lama_owned_mask = np.where(
        (lama_mask > 0) & (lama_priority_mask > 0),
        255,
        0,
    ).astype(np.uint8)
    lama_unowned_mask = np.where(
        (lama_mask > 0) & (lama_priority_mask <= 0),
        255,
        0,
    ).astype(np.uint8)
    cleaned, diagnostics = _run_lama_or_fallback(
        image,
        lama_owned_mask,
        lama_blocks,
        inpainter,
        config,
        check_need_inpaint=check_need_inpaint,
        diagnostic_block_indices=[
            original_block_indices.get(id(block))
            for block in lama_blocks
        ],
    )
    if np.any(lama_unowned_mask):
        unowned_result, unowned_diagnostics = _run_lama_or_fallback(
            cleaned,
            lama_unowned_mask,
            [],
            inpainter,
            config,
            check_need_inpaint=False,
        )
        cleaned = composite_with_edit_mask(
            cleaned,
            unowned_result,
            lama_unowned_mask,
        )
        diagnostics.extend(unowned_diagnostics)
    if np.any(missing_bubble_mask):
        cleaned, missing_diagnostics = _run_lama_or_fallback(
            cleaned,
            missing_bubble_mask,
            missing_bubble_blocks,
            inpainter,
            config,
            check_need_inpaint=False,
            diagnostic_block_indices=[
                original_block_indices.get(id(block))
                for block in missing_bubble_blocks
            ],
        )
        diagnostics.extend(missing_diagnostics)
    bubble_result = erase_text_bubble_regions(
        image,
        cleaned,
        bubble_mask,
        bubble_blocks,
        config,
        protected_edit_mask=bubble_protected_mask,
    )
    evidence = _build_owned_block_evidence(
        lama_blocks,
        normalized_raw_source,
        source_mask,
        bubble_protected_mask,
        image.shape,
        original_block_indices,
    )
    for item in tuple(getattr(bubble_result, "evidence", ()) or ()):
        bubble_block = (
            bubble_blocks[item.block_index]
            if item.block_index is not None
            and 0 <= item.block_index < len(bubble_blocks)
            else None
        )
        evidence.append(
            BlockInpaintEvidence(
                block_id=item.block_id,
                block_index=original_block_indices.get(id(bubble_block)),
                erase_mode=item.erase_mode,
                skipped_reason=item.skipped_reason,
                source_raw_owned=mask_patch_from_page_mask(
                    normalized_raw_source,
                    item.source_owned.xyxy if item.source_owned else None,
                    image.shape,
                ),
                source_owned=item.source_owned,
                structure_protect=item.structure_protect,
                ownership_protect=item.ownership_protect,
                bubble_interior=item.bubble_interior,
                positive_claim=item.positive_claim,
                positive_edit=item.positive_edit,
                claim_providers=item.claim_providers,
                route_decision=item.route_decision,
                route_reasons=item.route_reasons,
            )
        )
    for item in list(bubble_result.stats.get("blocks", []) or []):
        try:
            bubble_index = int(item.get("index", -1))
        except (TypeError, ValueError):
            bubble_index = -1
        bubble_block = (
            bubble_blocks[bubble_index]
            if 0 <= bubble_index < len(bubble_blocks)
            else None
        )
        diagnostics.append(
            {
                "phase": "bubble_erase",
                "block_index": original_block_indices.get(id(bubble_block)),
                "is_inference": False,
                "status": "completed",
                "erase_mode": str(item.get("mode", "") or ""),
                "elapsed_seconds": float(item.get("elapsed_seconds", 0.0) or 0.0),
            }
        )
    fallback_mask = normalize_edit_mask(getattr(bubble_result, "fallback_mask", None), image.shape)
    result_image = bubble_result.image
    if np.any(fallback_mask):
        fallback_blocks = [
            block
            for block in bubble_blocks
            if getattr(block, "_erase_mode", "") == ERASE_MODE_BUBBLE_LAMA_FALLBACK
        ]
        fallback_result, fallback_diagnostics = _run_lama_or_fallback(
            result_image,
            fallback_mask,
            fallback_blocks,
            inpainter,
            config,
            check_need_inpaint=False,
            diagnostic_block_indices=[
                original_block_indices.get(id(block))
                for block in fallback_blocks
            ],
        )
        diagnostics.extend(fallback_diagnostics)
        result_image = composite_with_edit_mask(result_image, fallback_result, fallback_mask)
    combined_mask = np.where(
        (lama_mask > 0)
        | (missing_bubble_mask > 0)
        | (bubble_result.edit_mask > 0)
        | (fallback_mask > 0),
        255,
        0,
    ).astype(np.uint8)
    result_image, combined_mask, evidence = _apply_detector_positive_text_evidence(
        image,
        result_image,
        combined_mask,
        block_list,
        evidence,
        diagnostics,
        inpainter,
        config,
        normalized_positive_claim,
        protected_corner_mask,
    )
    result = composite_with_edit_mask(image, result_image, combined_mask)
    result, combined_mask = _apply_protected_corner_guard(
        image,
        result,
        combined_mask,
        protected_corner_mask,
    )
    return SourceLamaBlockwiseResult(
        image=result,
        edit_mask=combined_mask,
        diagnostics=diagnostics,
        evidence=tuple(evidence),
    )


def source_lama_blockwise_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    blocks: Iterable[TextBlock],
    inpainter,
    config,
    *,
    raw_source_mask: np.ndarray | None = None,
    positive_claim_raw_mask: np.ndarray | None = None,
    check_need_inpaint: bool = True,
    return_edit_mask: bool = False,
    return_diagnostics: bool = False,
    protected_corner_mask: np.ndarray | None = None,
) -> (
    np.ndarray
    | tuple[np.ndarray, np.ndarray | None]
    | tuple[np.ndarray, list[dict]]
    | tuple[np.ndarray, np.ndarray | None, list[dict]]
):
    result = source_lama_blockwise_inpaint_result(
        image,
        mask,
        blocks,
        inpainter,
        config,
        raw_source_mask=raw_source_mask,
        positive_claim_raw_mask=positive_claim_raw_mask,
        check_need_inpaint=check_need_inpaint,
        protected_corner_mask=protected_corner_mask,
    )
    return _maybe_return_edit_mask(
        result.image,
        result.edit_mask,
        return_edit_mask,
        diagnostics=result.diagnostics,
        return_diagnostics=return_diagnostics,
    )
