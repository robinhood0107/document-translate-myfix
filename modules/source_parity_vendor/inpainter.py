from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from modules.inpainting.lama_torch_network import load_lama_mpe
from modules.source_parity_vendor.utils.imgproc_utils import enlarge_window, resize_keepasp
from modules.source_parity_vendor.utils.textblock_mask import extract_ballon_mask
from modules.utils.download import ModelDownloader, ModelID


def inpaint_handle_alpha_channel(original_alpha, mask):
    result_alpha = original_alpha.copy()
    mask_dilated = cv2.dilate((mask > 127).astype(np.uint8), np.ones((15, 15), np.uint8), iterations=1)
    surrounding_mask = mask_dilated - (mask > 127).astype(np.uint8)
    if np.any(surrounding_mask > 0):
        surrounding_alpha = original_alpha[surrounding_mask > 0]
        if len(surrounding_alpha) > 0:
            median_surrounding_alpha = np.median(surrounding_alpha)
            if median_surrounding_alpha < 128:
                inpainted_mask = mask > 127
                result_alpha[inpainted_mask] = median_surrounding_alpha
    return result_alpha


@dataclass(frozen=True)
class SourceParityInpainterKey:
    device: str
    precision: str
    inpaint_size: int


class SourceParityLaMaLarge:
    inpaint_by_block = True
    check_need_inpaint = True

    def __init__(self, device: str = "cuda", precision: str = "bf16", inpaint_size: int = 1536) -> None:
        self.device = str(device or "cuda")
        self.precision = str(precision or "bf16")
        self.inpaint_size = int(inpaint_size or 1536)
        self.model = None

    @property
    def key(self) -> SourceParityInpainterKey:
        return SourceParityInpainterKey(self.device, self.precision, self.inpaint_size)

    def ensure_loaded(self) -> None:
        if self.model is None:
            self.model = load_lama_mpe(
                str(ModelDownloader.primary_path(ModelID.LAMA_LARGE_512PX)),
                device="cpu",
                use_mpe=False,
                large_arch=True,
            )
            self.moveToDevice(self.device, precision=self.precision)

    def moveToDevice(self, device: str, precision: str | None = None):
        self.ensure_loaded()
        self.model.to(device)
        self.device = str(device)
        if precision is not None:
            self.precision = str(precision)

    def memory_safe_inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list=None) -> np.ndarray:
        self.ensure_loaded()
        try:
            return self._inpaint(img, mask, textblock_list)
        except Exception as e:
            if self.device == "cuda" and isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
                try:
                    return self._inpaint(img, mask, textblock_list)
                except Exception as ee:
                    if isinstance(ee, torch.cuda.OutOfMemoryError):
                        self.moveToDevice("cpu", precision="fp32")
                        inpainted = self._inpaint(img, mask, textblock_list)
                        self.moveToDevice("cuda", precision=self.precision)
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
        if self.device in {"cuda"}:
            try:
                with torch.autocast(device_type=self.device, dtype=precision):
                    img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)
            except Exception:
                img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)
        else:
            img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)

        img_inpainted = (img_inpainted_torch.to(device="cpu", dtype=torch.float32).squeeze_(0).permute(1, 2, 0).numpy() * 255)
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
                result_alpha = inpaint_handle_alpha_channel(original_alpha, mask)
                return np.concatenate([result_rgb, result_alpha], axis=2)
            return result_rgb

        im_h, im_w = img_rgb.shape[:2]
        inpainted = np.copy(img_rgb)
        original_mask = mask.copy()
        for blk in textblock_list:
            xyxy = [int(v) for v in getattr(blk, "xyxy", [0, 0, 0, 0])]
            xyxy_e = enlarge_window(xyxy, im_w, im_h, ratio=1.7)
            im = inpainted[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]]
            msk = mask[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]]
            need_inpaint = True
            if self.check_need_inpaint or check_need_inpaint:
                ballon_msk, non_text_msk = extract_ballon_mask(im, msk)
                if ballon_msk is not None:
                    non_text_region = np.where(non_text_msk > 0)
                    non_text_px = im[non_text_region]
                    average_bg_color = np.median(non_text_px, axis=0)
                    std_rgb = np.std(non_text_px - average_bg_color, axis=0)
                    std_max = np.max(std_rgb)
                    inpaint_thresh = 7 if np.std(std_rgb) > 1 else 10
                    if std_max < inpaint_thresh:
                        need_inpaint = False
                        im[np.where(ballon_msk > 0)] = average_bg_color
            if need_inpaint:
                inpainted[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]] = self.memory_safe_inpaint(im, msk)
            else:
                inpainted[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]] = im
            mask[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = 0

        if original_alpha is not None:
            result_alpha = inpaint_handle_alpha_channel(original_alpha, original_mask)
            return np.concatenate([inpainted, result_alpha], axis=2)
        return inpainted


_INPAINTER_CACHE: dict[SourceParityInpainterKey, SourceParityLaMaLarge] = {}


def get_source_parity_lama_large(device: str = "cuda", precision: str = "bf16", inpaint_size: int = 1536) -> SourceParityLaMaLarge:
    key = SourceParityInpainterKey(str(device or "cuda"), str(precision or "bf16"), int(inpaint_size or 1536))
    cached = _INPAINTER_CACHE.get(key)
    if cached is None:
        cached = SourceParityLaMaLarge(device=key.device, precision=key.precision, inpaint_size=key.inpaint_size)
        _INPAINTER_CACHE[key] = cached
    return cached
