from __future__ import annotations

import numpy as np
import cv2
import imkit as imk
from PIL import Image

from .base import InpaintModel
from .schema import Config
from modules.utils.download import ModelDownloader, ModelID


def _resize_keep_aspect(img, target_size: int | None, *, stride: int | None = None):
    if target_size is None:
        out = img.copy()
    else:
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest <= target_size:
            out = img.copy()
        else:
            scale = float(target_size) / float(longest)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if stride and stride > 1:
        h, w = out.shape[:2]
        target_h = int(np.ceil(h / stride) * stride)
        target_w = int(np.ceil(w / stride) * stride)
        pad_bottom = max(0, target_h - h)
        pad_right = max(0, target_w - w)
        if pad_bottom or pad_right:
            out = cv2.copyMakeBorder(out, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
    return out


class _TorchLamaBase(InpaintModel):
    pad_mod = 8

    def init_model(self, device, **kwargs):
        self.backend = str(kwargs.get("backend", "torch") or "torch")
        self.runtime_device = str(kwargs.get("runtime_device", device) or device)
        self.inpaint_size = int(kwargs.get("inpaint_size", self.default_inpaint_size()))
        self.precision = str(kwargs.get("precision", self.default_precision()) or self.default_precision())
        if self.backend != "torch":
            self.backend = "torch"
        self.device = self.runtime_device
        self.model = None
        self._load_torch_model()

    @classmethod
    def default_precision(cls) -> str:
        return "fp32"

    @classmethod
    def default_inpaint_size(cls) -> int:
        return 1536

    @classmethod
    def model_path(cls):
        raise NotImplementedError

    @classmethod
    def load_model_impl(cls, model_path: str, device: str):
        raise NotImplementedError

    def _load_torch_model(self) -> None:
        import torch  # noqa: F401

        model_path = str(self.model_path())
        self.model = self.load_model_impl(model_path, device="cpu")
        self.move_to_device(self.device, precision=self.precision)

    def move_to_device(self, device: str, precision: str | None = None) -> None:
        self.device = device
        if precision is not None:
            self.precision = precision
        self.model.to(device)

    @staticmethod
    def is_downloaded() -> bool:
        return True

    def _prepare(self, img: np.ndarray, mask: np.ndarray, *, stride: int = 64):
        import torch

        img_original = np.copy(img)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask_original = np.where(mask > 127, 1, 0).astype(np.uint8)[:, :, None]

        new_shape = self.inpaint_size if max(img.shape[:2]) > self.inpaint_size else None
        img = _resize_keep_aspect(img, new_shape, stride=stride)
        mask = _resize_keep_aspect(mask, new_shape, stride=stride)

        im_h, im_w = img.shape[:2]
        longer = max(im_h, im_w)
        pad_bottom = longer - im_h if im_h < longer else 0
        pad_right = longer - im_w if im_w < longer else 0
        if pad_bottom or pad_right:
            mask = cv2.copyMakeBorder(mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
            img = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)

        img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        mask_torch = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() / 255.0
        mask_torch = torch.where(mask_torch >= 0.5, 1.0, 0.0)
        return img_torch, mask_torch, img_original, mask_original, pad_bottom, pad_right, im_h, im_w

    def _autocast_context(self):
        import torch
        from contextlib import nullcontext

        precision = str(self.precision or "fp32").lower()
        if self.device != "cuda":
            return nullcontext()
        if precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if precision in {"fp16", "float16"}:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _postprocess(self, img_inpainted, *, img_original, mask_original, pad_bottom, pad_right, im_h, im_w):
        if pad_bottom > 0:
            img_inpainted = img_inpainted[:-pad_bottom]
        if pad_right > 0:
            img_inpainted = img_inpainted[:, :-pad_right]
        target_h, target_w = img_original.shape[:2]
        new_shape = img_inpainted.shape[:2]
        if new_shape[0] != target_h or new_shape[1] != target_w:
            img_inpainted = cv2.resize(img_inpainted, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        mask_comp = mask_original
        if mask_comp.ndim == 2:
            mask_comp = mask_comp[:, :, None]
        img_inpainted = img_inpainted * mask_comp + img_original * (1 - mask_comp)
        return img_inpainted.astype(np.uint8)


class LaMaMPE(_TorchLamaBase):
    name = "lama_mpe"

    @classmethod
    def default_precision(cls) -> str:
        return "fp32"

    @classmethod
    def default_inpaint_size(cls) -> int:
        return 2048

    @classmethod
    def model_path(cls):
        return ModelDownloader.primary_path(ModelID.LAMA_MPE)

    @classmethod
    def load_model_impl(cls, model_path: str, device: str):
        from .lama_torch_network import load_lama_mpe
        return load_lama_mpe(model_path, device=device, use_mpe=True, large_arch=False)

    def forward(self, image, mask, config: Config):
        import torch

        img_torch, mask_torch, img_original, mask_original, pad_bottom, pad_right, im_h, im_w = self._prepare(image, mask, stride=64)
        rel_pos, _abs_pos, direct = self.model.load_masked_position_encoding(mask_torch[0][0].numpy())
        rel_pos = torch.LongTensor(rel_pos).unsqueeze(0)
        direct = torch.LongTensor(direct).unsqueeze(0)

        if self.device != "cpu":
            img_torch = img_torch.to(self.device)
            mask_torch = mask_torch.to(self.device)
            rel_pos = rel_pos.to(self.device)
            direct = direct.to(self.device)
        img_torch = img_torch * (1 - mask_torch)

        with self._autocast_context():
            img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)

        img_inpainted = (
            img_inpainted_torch.to(device="cpu", dtype=torch.float32)
            .squeeze(0)
            .permute(1, 2, 0)
            .numpy()
            * 255.0
        )
        img_inpainted = np.clip(np.round(img_inpainted), 0, 255).astype(np.uint8)
        return self._postprocess(
            img_inpainted,
            img_original=img_original,
            mask_original=mask_original,
            pad_bottom=pad_bottom,
            pad_right=pad_right,
            im_h=im_h,
            im_w=im_w,
        )


class LaMaLarge512px(LaMaMPE):
    name = "lama_large_512px"

    @classmethod
    def default_precision(cls) -> str:
        return "bf16"

    @classmethod
    def default_inpaint_size(cls) -> int:
        return 1536

    @classmethod
    def model_path(cls):
        return ModelDownloader.primary_path(ModelID.LAMA_LARGE_512PX)

    @classmethod
    def load_model_impl(cls, model_path: str, device: str):
        from .lama_torch_network import load_lama_mpe
        return load_lama_mpe(model_path, device=device, use_mpe=False, large_arch=True)
