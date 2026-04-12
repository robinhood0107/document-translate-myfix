from __future__ import annotations

from dataclasses import dataclass

import cv2
import imkit as imk
import numpy as np
import torch

from modules.inpainting.lama_torch_network import load_lama_mpe
from modules.source_parity_vendor.utils.imgproc_utils import enlarge_window, resize_keepasp
from modules.source_parity_vendor.utils.textblock import TextBlock as SourceLaMaTextBlock
from modules.source_parity_vendor.utils.textblock_mask import extract_ballon_mask
from modules.utils.carrier import CARRIER_KIND_CAPTION_PLATE
from modules.utils.download import ModelDownloader, ModelID
from modules.utils.textblock import TextBlock


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


def _clip_roi_to_crop(
    roi_xyxy,
    crop_xyxy: list[int] | tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    if roi_xyxy is None or len(roi_xyxy) < 4:
        return None
    cx1, cy1, cx2, cy2 = [int(v) for v in crop_xyxy[:4]]
    rx1, ry1, rx2, ry2 = [int(float(v)) for v in roi_xyxy[:4]]
    ix1 = max(cx1, rx1)
    iy1 = max(cy1, ry1)
    ix2 = min(cx2, rx2)
    iy2 = min(cy2, ry2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1 - cx1, iy1 - cy1, ix2 - cx1, iy2 - cy1


def _try_caption_plate_local_inpaint(
    image_crop: np.ndarray,
    mask_crop: np.ndarray,
    block,
    crop_xyxy: list[int] | tuple[int, int, int, int],
) -> np.ndarray | None:
    if (getattr(block, "carrier_kind", "") or "") != CARRIER_KIND_CAPTION_PLATE:
        return None
    carrier_roi = getattr(block, "carrier_mask_roi_xyxy", None)
    local_roi = _clip_roi_to_crop(carrier_roi, crop_xyxy)
    if local_roi is None:
        return None

    lx1, ly1, lx2, ly2 = local_roi
    plate_crop = image_crop[ly1:ly2, lx1:lx2]
    plate_mask = np.where(mask_crop[ly1:ly2, lx1:lx2] > 0, 255, 0).astype(np.uint8)
    if plate_crop.size == 0 or plate_mask.size == 0 or not np.any(plate_mask):
        return None

    sample_ring = cv2.dilate((plate_mask > 0).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
    sample_mask = np.where((sample_ring > 0) & (plate_mask == 0), 255, 0).astype(np.uint8)
    sample_pixels = plate_crop[sample_mask > 0]
    if sample_pixels.size <= 0:
        return None

    sample_median = np.median(sample_pixels.astype(np.float32), axis=0)
    sample_std = np.max(np.std(sample_pixels.astype(np.float32) - sample_median, axis=0))
    radius = 3 if sample_std < 20.0 else 5
    inpainted_plate = cv2.inpaint(plate_crop, plate_mask, float(radius), cv2.INPAINT_TELEA)
    if inpainted_plate is None or inpainted_plate.shape != plate_crop.shape:
        return None

    result = image_crop.copy()
    result[ly1:ly2, lx1:lx2] = inpainted_plate
    return result


def _resolve_caption_plate_erase_mask(
    image_crop: np.ndarray,
    mask_crop: np.ndarray,
    block,
) -> np.ndarray | None:
    if (getattr(block, "carrier_kind", "") or "") != CARRIER_KIND_CAPTION_PLATE:
        return None

    text_mask = np.where(mask_crop > 0, 255, 0).astype(np.uint8)
    text_area = int(np.count_nonzero(text_mask))
    if text_area <= 0:
        return None

    ballon_mask, _non_text_mask = extract_ballon_mask(image_crop, text_mask)
    if ballon_mask is None:
        return None

    ballon_mask = np.where(ballon_mask > 0, 255, 0).astype(np.uint8)
    ballon_area = int(np.count_nonzero(ballon_mask))
    crop_area = int(image_crop.shape[0] * image_crop.shape[1])
    if (
        ballon_area <= int(round(text_area * 1.15))
        or ballon_area >= int(round(crop_area * 0.92))
    ):
        return None

    expand = max(4, min(12, int(round(np.sqrt(float(max(1, ballon_area))) * 0.015))))
    ballon_mask = cv2.morphologyEx(
        ballon_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    ballon_mask = cv2.dilate(
        ballon_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand * 2 + 1, expand * 2 + 1)),
        iterations=1,
    )
    return np.where(ballon_mask > 0, 255, 0).astype(np.uint8)


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
        self.model.to(device)
        self.device = str(device)
        if precision is not None:
            self.precision = str(precision)

    def memory_safe_inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list=None) -> np.ndarray:
        self.ensure_loaded()
        try:
            return self._inpaint(img, mask, textblock_list)
        except Exception as exc:
            if self.device == "cuda" and isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
                try:
                    return self._inpaint(img, mask, textblock_list)
                except Exception as retry_exc:
                    if isinstance(retry_exc, torch.cuda.OutOfMemoryError):
                        previous_device = self.device
                        previous_precision = self.precision
                        self.moveToDevice("cpu", precision="fp32")
                        inpainted = self._inpaint(img, mask, textblock_list)
                        self.moveToDevice(previous_device, precision=previous_precision)
                        return inpainted
            raise

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

        precision_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
        }
        precision = precision_map.get(str(self.precision).lower(), torch.float32)
        if self.device == "cuda":
            try:
                with torch.autocast(device_type=self.device, dtype=precision):
                    img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)
            except Exception:
                img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)
        else:
            img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)

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

    def inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list=None, check_need_inpaint: bool = False) -> np.ndarray:
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
            if original_alpha is not None:
                result_alpha = _inpaint_handle_alpha_channel(original_alpha, mask)
                return np.concatenate([result_rgb, result_alpha], axis=2)
            return result_rgb

        im_h, im_w = img_rgb.shape[:2]
        inpainted = np.copy(img_rgb)
        original_mask = mask.copy()
        work_mask = mask.copy()
        for blk in textblock_list:
            xyxy = [int(v) for v in getattr(blk, "xyxy", [0, 0, 0, 0])]
            crop_anchor = xyxy
            if (getattr(blk, "carrier_kind", "") or "") == CARRIER_KIND_CAPTION_PLATE:
                crop_anchor = [
                    int(v) for v in getattr(blk, "bubble_xyxy", getattr(blk, "xyxy", xyxy))[:4]
                ]
            xyxy_e = enlarge_window(crop_anchor, im_w, im_h, ratio=1.7)
            image_crop = inpainted[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]]
            mask_crop = work_mask[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]]
            caption_plate_erase_mask = _resolve_caption_plate_erase_mask(image_crop, mask_crop, blk)
            if caption_plate_erase_mask is not None:
                inpainted[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]] = self.memory_safe_inpaint(
                    image_crop,
                    caption_plate_erase_mask,
                )
                work_mask[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = 0
                continue
            caption_plate_crop = _try_caption_plate_local_inpaint(image_crop, mask_crop, blk, xyxy_e)
            if caption_plate_crop is not None:
                inpainted[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]] = caption_plate_crop
                work_mask[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = 0
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
                inpainted[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]] = self.memory_safe_inpaint(image_crop, mask_crop)
            else:
                inpainted[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]] = image_crop
            work_mask[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = 0

        if original_alpha is not None:
            result_alpha = _inpaint_handle_alpha_channel(original_alpha, original_mask)
            return np.concatenate([inpainted, result_alpha], axis=2)
        return inpainted


_INPAINTER_CACHE: dict[SourceLaMaKey, SourceLaMaLarge] = {}


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
    source_block.carrier_kind = str(getattr(block, "carrier_kind", "") or "")
    bubble_xyxy = getattr(block, "bubble_xyxy", None)
    source_block.bubble_xyxy = (
        [int(v) for v in bubble_xyxy[:4]]
        if bubble_xyxy is not None
        else None
    )
    carrier_mask_roi_xyxy = getattr(block, "carrier_mask_roi_xyxy", None)
    source_block.carrier_mask_roi_xyxy = (
        [int(v) for v in carrier_mask_roi_xyxy[:4]]
        if carrier_mask_roi_xyxy is not None
        else None
    )
    return source_block


def _resolve_source_blocks(blocks: list[TextBlock]) -> list[SourceLaMaTextBlock]:
    resolved: list[SourceLaMaTextBlock] = []
    for block in list(blocks or []):
        source_block = _adapt_generic_block_to_source_block(block)
        if source_block is not None:
            resolved.append(source_block)
    return resolved


def source_lama_blockwise_inpaint(
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

    device = str(getattr(inpainter, "runtime_device", getattr(inpainter, "device", "cuda")) or "cuda")
    precision = str(getattr(inpainter, "precision", "bf16") or "bf16")
    inpaint_size = int(getattr(inpainter, "inpaint_size", 1536) or 1536)
    source_inpainter = get_source_lama_large(device=device, precision=precision, inpaint_size=inpaint_size)
    result = source_inpainter.inpaint(
        image,
        np.where(mask > 0, 255, 0).astype(np.uint8),
        source_blocks,
        check_need_inpaint=check_need_inpaint,
    )
    return imk.convert_scale_abs(result)
