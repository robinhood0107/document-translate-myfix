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
            xyxy_e = enlarge_window(xyxy, im_w, im_h, ratio=1.7)
            image_crop = inpainted[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]]
            mask_crop = work_mask[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]]
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
